"""
scopeID 클러스터링 서비스 - embedding 기반 안정화 버전 v2

흐름:
  1. ES news_economy에서 scopeID가 없는 기사 조회
  2. news_economy.embedding 벡터 사용
  3. FAISS로 기존 scope centroid(news_scopes)와 코사인 유사도 비교
  4. 유사도 >= THRESHOLD → 기존 scopeID 배정
     유사도 <  THRESHOLD → 신규 scopeID 발급
  5. news_economy.scopeID bulk upsert
  6. news_scopes centroid_embedding / news_count bulk upsert
  7. scopeTitle 생성 트리거 호출 (배치 단위, scope당 1회)

개선 사항 (v1 → v2):
  [1] centroid 누적 평균 오류 수정
      - 정규화된 단위벡터에 old_count를 곱해 누적합 복원 시 오차가 누적되는 문제 해결
      - scope_raw_sum_map 으로 정규화 전 누적합 벡터를 별도 관리
      - 누적합에 새 벡터를 더한 뒤 정규화 → 수학적으로 정확한 평균 방향 벡터 유지

  [2] trigger_scope_title 과호출 방지
      - 루프 내 매 기사마다 호출하던 방식 → 배치 완료 후 scope당 1회만 호출
      - Gemini API 과호출 및 중복 큐 등록 방지

  [3] ES 개별 update → bulk API 전환
      - 기사당 2회(NEWS + SCOPES) × 최대 10,000건 = 최대 20,000 ES 요청을 bulk로 축소
      - news_actions / scope_actions 리스트에 누적 후 루프 종료 시 일괄 전송
      - 신규 scope index는 즉시성이 필요하므로 루프 내 개별 호출 유지

  [4] embedding 없는 기사의 영구 재처리 루프 방지
      - embedding이 없어 skip된 기사에 scopeID = "__no_embedding__" 마킹
      - 다음 배치에서 동일 기사가 반복 조회되는 비효율 제거

  [5] FAISS 인덱스와 scope_ids 리스트 정합성 명시화
      - _append_scope_to_faiss가 새 IndexFlatIP를 반환할 수 있는 케이스를
        호출부에서 명시적으로 재할당하도록 구조 유지 (기존과 동일, 주석 보강)
      - size 상한 10000 하드코딩을 MAX_SCOPE_SIZE 상수로 통합
"""

import logging
from datetime import datetime, timezone

import faiss
import numpy as np
from elasticsearch.helpers import bulk

from model.database import get_es
from model.services.error_logger import log_pipeline_error
from model.services.scope_title import trigger_scope_title

logger = logging.getLogger(__name__)

# =========================================================
# 설정값
# =========================================================

# embedding cosine similarity 기준
# 너무 섞이면 0.80~0.82로 올리고,
# 너무 잘게 쪼개지면 0.75 정도로 낮추면 됨.
SIMILARITY_THRESHOLD = 0.78

BATCH_SIZE          = 10000
MAX_SCOPE_SIZE      = 10000   # _build_faiss_index / _initialize_scope_sequence 공용 상한
SCOPE_ID_PREFIX_FMT = "%Y%m%d"

INDEX_NEWS   = "news_economy"
INDEX_SCOPES = "news_scopes"

EXPECTED_EMBEDDING_DIM = 768

# embedding이 없어 분류 불가한 기사에 부여하는 마커값.
# 다음 배치에서 반복 조회되는 것을 방지한다.
SKIP_SCOPE_MARKER = "__no_embedding__"


# =========================================================
# 벡터 처리
# =========================================================

def _normalize_vector(raw_embedding) -> np.ndarray | None:
    """
    raw embedding을 float32 2차원 벡터(1, dim)로 변환하고 L2 정규화.
    문제가 있으면 None 반환.
    """
    if not raw_embedding:
        return None

    try:
        vec = np.array(raw_embedding, dtype=np.float32).reshape(1, -1)
    except Exception:
        return None

    if vec.shape[1] != EXPECTED_EMBEDDING_DIM:
        logger.warning(
            f"embedding 차원 불일치: expected={EXPECTED_EMBEDDING_DIM}, actual={vec.shape[1]}"
        )
        return None

    norm = np.linalg.norm(vec)
    if norm == 0:
        return None

    return vec / norm


def _update_centroid(raw_sum: np.ndarray, new_vec: np.ndarray, old_count: int) -> tuple:
    """
    정규화 전 누적합 벡터(raw_sum)에 새 기사 벡터를 더해
    새로운 누적합과 정규화된 centroid를 반환한다.

    [v1 문제]
    기존 코드는 정규화된 centroid * old_count 로 누적합을 역산했는데,
    centroid가 매 단계마다 정규화되어 있으므로 old_count를 곱해도
    원래 합 벡터가 복원되지 않는다.
    → 누적될수록 초기 기사 방향이 희석되어 centroid가 drift하는 문제.

    [v2 수정]
    raw_sum(누적합)을 별도로 유지하고, 여기에 새 벡터를 그냥 더한 뒤
    정규화한다. 수학적으로 정확한 평균 방향 벡터를 보장한다.

    Returns:
        new_raw_sum  : 갱신된 누적합 벡터 (float32, shape=(dim,))
        new_centroid : 정규화된 centroid 벡터 (float32, shape=(dim,))
    """
    new_raw_sum = raw_sum.flatten() + new_vec.flatten()

    norm = np.linalg.norm(new_raw_sum)
    if norm > 0:
        new_centroid = (new_raw_sum / norm).astype(np.float32)
    else:
        new_centroid = new_raw_sum.astype(np.float32)

    return new_raw_sum.astype(np.float32), new_centroid


# =========================================================
# scope 상태 로드
# =========================================================

def _build_faiss_index(es) -> tuple:
    """
    news_scopes의 centroid_embedding을 읽어 FAISS inner product index 생성.
    벡터는 정규화되어 있으므로 inner product = cosine similarity.

    Returns:
        faiss_index       : IndexFlatIP 또는 None (scope 없을 때)
        scope_ids         : FAISS index 순서와 1:1 대응하는 scopeID 리스트
        scope_centroid_map: {scopeID: centroid 단위벡터 (float32, shape=(dim,))}
        scope_count_map   : {scopeID: news_count (int)}
        scope_raw_sum_map : {scopeID: 정규화 전 누적합 벡터 (float32, shape=(dim,))}
    """
    res = es.search(
        index=INDEX_SCOPES,
        body={
            "query": {"exists": {"field": "centroid_embedding"}},
            "_source": ["scopeID", "centroid_embedding", "news_count"],
            "size": MAX_SCOPE_SIZE,
        },
    )

    hits = res["hits"]["hits"]
    if not hits:
        return None, [], {}, {}, {}

    scope_ids         = []
    centroids         = []
    scope_centroid_map = {}
    scope_count_map   = {}
    scope_raw_sum_map  = {}

    for hit in hits:
        src      = hit["_source"]
        scope_id = src.get("scopeID") or hit["_id"]

        centroid_vec = _normalize_vector(src.get("centroid_embedding"))
        if centroid_vec is None:
            logger.warning(f"centroid_embedding 무시: scopeID={scope_id}")
            continue

        centroid_flat = centroid_vec.flatten()
        count         = int(src.get("news_count", 0) or 0)

        scope_ids.append(scope_id)
        centroids.append(centroid_flat)

        scope_centroid_map[scope_id] = centroid_flat
        scope_count_map[scope_id]    = count

        # 누적합 역산: centroid(단위벡터) × news_count
        # 초기 로드 시점에 한해 이 근사값을 사용한다.
        # 이후 갱신은 정확한 raw_sum 누적으로 진행된다.
        scope_raw_sum_map[scope_id] = (centroid_flat * count).astype(np.float32)

    if not centroids:
        return None, [], {}, {}, {}

    matrix = np.vstack(centroids).astype(np.float32)
    index  = faiss.IndexFlatIP(EXPECTED_EMBEDDING_DIM)
    index.add(matrix)

    return index, scope_ids, scope_centroid_map, scope_count_map, scope_raw_sum_map


def _append_scope_to_faiss(
    faiss_index,
    scope_ids: list,
    scope_id: str,
    centroid_vec: np.ndarray,
) -> faiss.IndexFlatIP:
    """
    새로 생성된 scope를 현재 배치의 FAISS 인덱스에 즉시 추가.
    같은 배치에서 들어오는 후속 유사 기사들이 신규 scope를 바로 찾아 붙을 수 있도록 한다.

    faiss_index가 None(첫 번째 신규 scope)이면 새 인덱스를 생성해 반환한다.
    호출부에서 반환값으로 반드시 재할당해야 한다.
    scope_ids는 mutable 리스트이므로 in-place 추가된다.
    """
    if faiss_index is None:
        faiss_index = faiss.IndexFlatIP(EXPECTED_EMBEDDING_DIM)

    faiss_index.add(centroid_vec.reshape(1, -1).astype(np.float32))
    scope_ids.append(scope_id)

    return faiss_index


# =========================================================
# scopeID 생성
# =========================================================

def _initialize_scope_sequence(es) -> dict:
    """
    오늘 날짜 prefix 기준으로 기존 최대 scope 번호를 읽어
    배치 내부 시퀀스 상태를 초기화한다.

    배치 내부에서 메모리 시퀀스로 증가시키므로
    ES refresh 지연으로 인한 중복 ID 생성을 방지한다.
    """
    prefix = datetime.now().strftime(SCOPE_ID_PREFIX_FMT)

    res = es.search(
        index=INDEX_SCOPES,
        body={
            "query": {"prefix": {"scopeID": prefix}},
            "_source": ["scopeID"],
            "size": MAX_SCOPE_SIZE,
        },
    )

    max_seq = 0
    for hit in res["hits"]["hits"]:
        scope_id = hit["_source"].get("scopeID") or hit["_id"]
        try:
            seq     = int(str(scope_id).split("-")[-1])
            max_seq = max(max_seq, seq)
        except Exception:
            continue

    return {"prefix": prefix, "next_seq": max_seq + 1}


def _generate_scope_id(sequence_state: dict) -> str:
    """
    YYYYMMDD-XXXX 형식 scopeID 생성.
    호출마다 sequence_state["next_seq"]를 1씩 증가시킨다.
    """
    scope_id = f"{sequence_state['prefix']}-{sequence_state['next_seq']:04d}"
    sequence_state["next_seq"] += 1
    return scope_id


# =========================================================
# 메인 분류 파이프라인
# =========================================================

def run_classification_pipeline() -> dict:
    """
    scopeID가 없는 뉴스를 embedding 기준으로 scope에 배정한다.
    """
    try:
        es = get_es()

        # -------------------------------------------------
        # 1. scopeID가 없는 기사 조회
        #    embedding 필드가 있는 기사만 대상으로 한다.
        #    SKIP_SCOPE_MARKER가 이미 붙은 기사는 must_not으로 제외된다.
        # -------------------------------------------------
        res = es.search(
            index=INDEX_NEWS,
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"exists": {"field": "embedding"}}
                        ],
                        "must_not": [
                            {"exists": {"field": "scopeID"}}
                        ],
                    }
                },
                "_source": ["article_id", "title", "embedding"],
                "sort":    [{"collected_at": {"order": "asc"}}],
                "size":    BATCH_SIZE,
            },
        )
        hits = res["hits"]["hits"]

        if not hits:
            logger.info("분류할 뉴스 없음")
            es.close()
            return {
                "success":         True,
                "message":         "분류할 뉴스 없음",
                "processed":       0,
                "new_scopes":      0,
                "merged_articles": 0,
                "skipped":         0,
            }

        logger.info(f"스콥 분류 시작: {len(hits)}건")

        # -------------------------------------------------
        # 2. 기존 scope centroid FAISS 인덱스 로드
        # -------------------------------------------------
        (
            faiss_index,
            scope_ids,
            scope_centroid_map,
            scope_count_map,
            scope_raw_sum_map,
        ) = _build_faiss_index(es)

        # -------------------------------------------------
        # 3. scopeID 발급용 시퀀스 초기화
        # -------------------------------------------------
        sequence_state = _initialize_scope_sequence(es)

        now_utc = datetime.now(timezone.utc).isoformat()

        # bulk 전송용 액션 버퍼
        news_bulk_actions  = []
        scope_bulk_actions = []

        # 배치 완료 후 scope당 1회만 trigger_scope_title 호출하기 위한 집합
        triggered_scopes: set[str] = set()

        processed_count      = 0
        new_scope_count      = 0
        merged_article_count = 0
        skipped_count        = 0

        # -------------------------------------------------
        # 4. 기사별 scope 배정
        # -------------------------------------------------
        for hit in hits:
            src        = hit["_source"]
            doc_id     = hit["_id"]
            article_id = src.get("article_id") or doc_id
            title      = src.get("title", "")

            vec = _normalize_vector(src.get("embedding"))

            # --- 4-0. embedding 없는 기사: 마커 부여 후 skip ---
            # SKIP_SCOPE_MARKER를 scopeID에 기록해 두면
            # 다음 배치에서 exists(scopeID) 조건에 걸려 반복 조회되지 않는다.
            if vec is None:
                logger.warning(
                    f"embedding 없음 → 마커 부여 후 스킵: "
                    f"article_id={article_id}, title={title[:40]}"
                )
                news_bulk_actions.append({
                    "_op_type": "update",
                    "_index":   INDEX_NEWS,
                    "_id":      doc_id,
                    "doc":      {"scopeID": SKIP_SCOPE_MARKER},
                })
                skipped_count += 1
                continue

            assigned_scope_id = None
            best_score        = None

            # --- 4-1. 기존 scope와 유사도 비교 ---
            if faiss_index is not None and len(scope_ids) > 0:
                scores, indices = faiss_index.search(vec, k=1)
                best_score = float(scores[0][0])
                best_idx   = int(indices[0][0])

                if best_idx >= 0 and best_score >= SIMILARITY_THRESHOLD:
                    assigned_scope_id = scope_ids[best_idx]

            # --- 4-2. 기존 scope에 병합 ---
            if assigned_scope_id is not None:
                old_count   = int(scope_count_map.get(assigned_scope_id, 0) or 0)
                old_raw_sum = scope_raw_sum_map.get(assigned_scope_id)

                if old_raw_sum is None:
                    # 방어: raw_sum이 없으면 centroid를 초기값으로 사용
                    old_raw_sum = scope_centroid_map.get(
                        assigned_scope_id, vec.flatten()
                    ).copy()

                new_raw_sum, new_centroid = _update_centroid(
                    raw_sum=old_raw_sum,
                    new_vec=vec,
                    old_count=old_count,
                )
                new_count = old_count + 1

                # 메모리 상태 갱신
                scope_raw_sum_map[assigned_scope_id]  = new_raw_sum
                scope_centroid_map[assigned_scope_id] = new_centroid
                scope_count_map[assigned_scope_id]    = new_count

                scope_bulk_actions.append({
                    "_op_type": "update",
                    "_index":   INDEX_SCOPES,
                    "_id":      assigned_scope_id,
                    "doc": {
                        "centroid_embedding": new_centroid.tolist(),
                        "news_count":         new_count,
                        "updated_at":         now_utc,
                    },
                })

                merged_article_count += 1
                logger.debug(
                    f"기존 scope 배정: article_id={article_id}, "
                    f"scopeID={assigned_scope_id}, score={best_score:.4f}"
                )

            # --- 4-3. 신규 scope 생성 ---
            else:
                assigned_scope_id = _generate_scope_id(sequence_state)
                centroid_flat     = vec.flatten().astype(np.float32)

                # 신규 scope는 즉시성이 필요하므로 bulk 대신 개별 index 호출
                es.index(
                    index=INDEX_SCOPES,
                    id=assigned_scope_id,
                    body={
                        "scopeID":            assigned_scope_id,
                        "centroid_embedding": centroid_flat.tolist(),
                        "news_count":         1,
                        "created_at":         now_utc,
                        "updated_at":         now_utc,
                    },
                )

                # 메모리 상태 등록
                scope_centroid_map[assigned_scope_id] = centroid_flat
                scope_count_map[assigned_scope_id]    = 1
                scope_raw_sum_map[assigned_scope_id]  = centroid_flat.copy()

                # 같은 배치 내 후속 기사들이 신규 scope를 바로 인식할 수 있도록
                # FAISS 인덱스에 즉시 추가한다.
                # faiss_index가 None이면 새 객체가 반환되므로 반드시 재할당.
                faiss_index = _append_scope_to_faiss(
                    faiss_index=faiss_index,
                    scope_ids=scope_ids,
                    scope_id=assigned_scope_id,
                    centroid_vec=centroid_flat,
                )

                new_scope_count += 1
                logger.debug(
                    f"신규 scope 생성: article_id={article_id}, "
                    f"scopeID={assigned_scope_id}"
                )

            # --- 4-4. 기사에 scopeID 저장 (bulk 버퍼에 추가) ---
            news_bulk_actions.append({
                "_op_type": "update",
                "_index":   INDEX_NEWS,
                "_id":      doc_id,
                "doc":      {"scopeID": assigned_scope_id},
            })

            # --- 4-5. trigger 대상 scope 등록 (루프 종료 후 1회 호출) ---
            triggered_scopes.add(assigned_scope_id)

            processed_count += 1

        # -------------------------------------------------
        # 5. bulk 전송: news scopeID 업데이트
        # -------------------------------------------------
        if news_bulk_actions:
            success, errors = bulk(
                es,
                news_bulk_actions,
                raise_on_error=False,
                stats_only=False,
            )
            if errors:
                logger.warning(f"news bulk update 일부 실패: {len(errors)}건")

        # -------------------------------------------------
        # 6. bulk 전송: scope centroid / news_count 업데이트
        # -------------------------------------------------
        if scope_bulk_actions:
            success, errors = bulk(
                es,
                scope_bulk_actions,
                raise_on_error=False,
                stats_only=False,
            )
            if errors:
                logger.warning(f"scope bulk update 일부 실패: {len(errors)}건")

        # -------------------------------------------------
        # 7. scope title 트리거: scope당 1회만 호출
        # -------------------------------------------------
        for scope_id in triggered_scopes:
            trigger_scope_title(
                es=es,
                scope_id=scope_id,
                news_count=scope_count_map.get(scope_id, 1),
            )

        # -------------------------------------------------
        # 8. 인덱스 refresh
        # -------------------------------------------------
        es.indices.refresh(index=INDEX_NEWS)
        es.indices.refresh(index=INDEX_SCOPES)

        es.close()

        logger.info(
            f"스콥 분류 완료 | "
            f"processed={processed_count}, "
            f"new_scopes={new_scope_count}, "
            f"merged={merged_article_count}, "
            f"skipped={skipped_count}"
        )

        return {
            "success":         True,
            "processed":       processed_count,
            "new_scopes":      new_scope_count,
            "merged_articles": merged_article_count,
            "skipped":         skipped_count,
        }

    except Exception as e:
        log_pipeline_error(pipeline="classifier", error=e)
        raise
