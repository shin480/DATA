"""
scopeTitle 생성 서비스 (Gemini + 폴백)

트리거 규칙:
  - news_count < 10  : 최신 기사 제목 그대로 사용 (Gemini 호출 없음)
  - news_count >= 10 : Gemini로 대표 제목 생성
                       → 키 전부 소진 시 최신 기사 제목으로 폴백

[수정 이력]
- 모델 교체: gemini-2.0-flash → gemini-2.5-flash-lite
- trigger_scope_title: 즉시 생성 실패 시 queue 자동 등록으로 변경 (유실 방지)
- _enqueue: 중복 방지 queue 등록 함수 분리
- enqueue_missing_scope_titles: scopeTitle 없는 scope 일괄 queue 등록 복구 함수 추가
- API 키 로테이션: 7개 키 순환 사용 (429/일일 한도 초과 시 자동 교체)
- SDK 교체: google.generativeai(deprecated) → google.genai
- 예외 처리 보강: 403 포함 모든 API 오류 시 키 교체 후 재시도
- 생성 전략 변경:
    · news_count < 10  → 최신 제목 그대로 (Gemini 호출 없음)
    · news_count >= 10 → Gemini 시도 → 실패 시 최신 제목 폴백
    · 고유명사 절단 문제로 키워드 조합 방식 미사용
- 배치 무한 반복 방지:
    · retry_count 필드 추가 → MAX_RETRY 초과 시 failed 마킹 후 스킵
    · generate_scope_title 내 예외를 폴백으로 흡수 → 배치에서 예외 미전파
- _enqueue 중복 방지 강화:
    · pending만 체크 → failed 제외 전체 상태 체크로 변경
    · 동일 scope_id 중복 큐 등록 원천 차단
- API 키 상태 모듈 레벨 영속 관리:
    · 403 PERMISSION_DENIED → 해당 키 영구 차단 (프로세스 수명 동안)
    · 429 RESOURCE_EXHAUSTED → 해당 키 1시간 쿨다운 후 자동 복구
    · 스콥 간 키 상태 유지로 이미 실패한 키 반복 시도 제거
    · 배치 시작 전 사용 가능한 키 없으면 즉시 전체 폴백
"""

import logging
import os
import time
from datetime import datetime, timedelta, timezone

from google import genai
from google.genai import types

from model.database import get_es
from model.services.error_logger import log_pipeline_error

logger = logging.getLogger(__name__)

BATCH_SIZE        = 10000
NEWS_COUNT_GEMINI = 10      # 이 이상일 때만 Gemini 호출
MAX_RETRY         = 3       # 큐 재시도 최대 횟수 초과 시 failed 처리
INDEX_NEWS        = "news_economy"
INDEX_SCOPES      = "news_scopes"
INDEX_QUEUE       = "scope_refresh_queue"
MODEL_NAME        = "gemini-2.5-flash-lite"

# API 키 로테이션 (7개)
_API_KEYS = [
    os.getenv("GOOGLE_API_KEY_1"),
    os.getenv("GOOGLE_API_KEY_2"),
    os.getenv("GOOGLE_API_KEY_3"),
    os.getenv("GOOGLE_API_KEY_4"),
    os.getenv("GOOGLE_API_KEY_5"),
    os.getenv("GOOGLE_API_KEY_6"),
    os.getenv("GOOGLE_API_KEY_7"),
]
_key_index = 0
_client    = None

# 키별 차단 상태 (모듈 레벨 영속)
# None      → 사용 가능
# datetime  → 해당 시각까지 차단 (429 쿨다운)
# "permanent" → 영구 차단 (403)
_key_blocked_until: dict[int, datetime | str] = {}


# ── 키 상태 관리 헬퍼 ────────────────────────────────────────────

def _is_key_available(idx: int) -> bool:
    """해당 인덱스 키가 현재 사용 가능한지 반환"""
    state = _key_blocked_until.get(idx)
    if state is None:
        return True
    if state == "permanent":
        return False
    # datetime: 쿨다운 만료 여부 확인
    if datetime.now() > state:
        del _key_blocked_until[idx]  # 만료된 차단 해제
        return True
    return False


def _mark_key_failed(idx: int, err_str: str):
    """오류 종류에 따라 키 차단 상태 기록"""
    if "403" in err_str:
        _key_blocked_until[idx] = "permanent"
        logger.warning(f"키 {idx} 영구 차단 (403 PERMISSION_DENIED)")
    else:
        # 429, 500, 503 등 → 1시간 쿨다운
        _key_blocked_until[idx] = datetime.now() + timedelta(hours=1)
        logger.warning(f"키 {idx} 1시간 차단 (일시 오류: {err_str[:40]})")


def _get_next_available_key() -> int | None:
    """현재 인덱스 이후부터 순환하며 사용 가능한 키 인덱스 반환. 없으면 None."""
    for i in range(len(_API_KEYS)):
        idx = (_key_index + i) % len(_API_KEYS)
        if _is_key_available(idx):
            return idx
    return None


# ── 클라이언트 관리 ──────────────────────────────────────────────

def _get_client():
    """현재 키 인덱스로 google.genai 클라이언트 반환 (캐시)"""
    global _client, _key_index
    if _client is None:
        key = _API_KEYS[_key_index]
        if not key:
            raise RuntimeError(f"GOOGLE_API_KEY_{_key_index + 1} 환경변수 없음")
        _client = genai.Client(api_key=key)
        logger.info(f"Gemini 클라이언트 초기화 (key index: {_key_index})")
    return _client


def _rotate_key(failed_index: int, err_str: str):
    """실패한 키를 상태 기록 후 다음 사용 가능한 키로 이동"""
    global _key_index, _client
    _mark_key_failed(failed_index, err_str)
    next_idx = _get_next_available_key()
    if next_idx is not None:
        _key_index = next_idx
        _client    = None
        logger.warning(f"API 키 교체 → index {_key_index}")
    else:
        logger.error("사용 가능한 API 키 없음 (전체 차단 상태)")


def _is_retryable(err_str: str) -> bool:
    """키 교체 후 재시도할 오류인지 판단"""
    err_lower = err_str.lower()
    return any(code in err_str for code in ("429", "403", "500", "503")) \
        or "quota" in err_lower \
        or "rate" in err_lower \
        or "limit" in err_lower


# ── 핵심 로직 ────────────────────────────────────────────────────

def _fetch_titles(es, scope_id: str) -> list[str]:
    """scope에 속한 기사 제목 목록 반환 (최신순)"""
    res = es.search(
        index=INDEX_NEWS,
        body={
            "query":   {"term": {"scopeID": scope_id}},
            "_source": ["title"],
            "sort":    [{"published_at": "desc"}],
            "size":    20,
        },
    )
    return [h["_source"]["title"] for h in res["hits"]["hits"] if h["_source"].get("title")]


def _gemini_generate(scope_id: str, titles: list[str]) -> str | None:
    """Gemini로 대표 제목 생성. 키 전부 소진 시 None 반환.

    - 호출 전 사용 가능한 키가 없으면 즉시 None 반환 (불필요한 루프 생략)
    - 실패한 키는 모듈 레벨 상태에 기록 → 다음 스콥에서도 해당 키 스킵
    - 403: 영구 차단 / 429·기타: 1시간 쿨다운
    """
    global _key_index, _client

    prompt = (
        "다음은 동일한 사건/이슈를 다룬 한국 경제 뉴스 기사 제목들입니다.\n"
        "이 제목들을 대표하는 하나의 핵심 제목을 만들어주세요.\n\n"
        "조건:\n- 30자 이내\n- 핵심 사실만 담을 것\n- 중립적인 톤\n- 제목만 출력\n\n"
        "뉴스 제목들:\n" + "\n".join(f"- {t}" for t in titles)
    )

    for attempt in range(len(_API_KEYS)):
        # 루프마다 사용 가능한 키 먼저 확인
        available = _get_next_available_key()
        if available is None:
            logger.error(f"모든 키 차단 상태 → 즉시 폴백 (scope_id={scope_id})")
            return None

        # 사용 가능한 키로 전환
        if _key_index != available:
            _key_index = available
            _client    = None

        failed_index = _key_index  # 실패 시 마킹할 인덱스 고정
        try:
            response = _get_client().models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    max_output_tokens=60,
                    temperature=0.3,
                ),
            )
            return response.text.strip()

        except Exception as e:
            err_str = str(e)
            if _is_retryable(err_str):
                logger.warning(
                    f"Gemini API 오류 (scope_id={scope_id}, attempt={attempt + 1}, "
                    f"key_index={failed_index}): {err_str[:80]} → 키 교체"
                )
                _rotate_key(failed_index, err_str)
                time.sleep(2)
            else:
                logger.error(f"Gemini 재시도 불가 오류 (scope_id={scope_id}): {e}")
                return None

    logger.error(f"모든 API 키 소진 (scope_id={scope_id}) → 최신 제목으로 폴백")
    return None


def _upsert_scope_title(es, scope_id: str, scope_title: str):
    """news_scopes에 scopeTitle upsert"""
    es.update(
        index=INDEX_SCOPES,
        id=scope_id,
        body={"doc": {
            "scopeTitle": scope_title,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }},
        retry_on_conflict=3,
    )
    logger.info(f"scopeTitle 저장: {scope_id} → {scope_title}")


def generate_scope_title(es, scope_id: str, news_count: int | None = None) -> str | None:
    """
    scopeTitle 생성 메인 함수. 내부 예외는 폴백으로 흡수.

    news_count < 10  : 최신 제목 그대로 (플랜 A)
    news_count >= 10 : Gemini 시도 → 실패 시 최신 제목 폴백 (플랜 B → 플랜 C)
    news_count=None  : titles 길이로 판단 (배치 처리 경로)
    """
    try:
        titles = _fetch_titles(es, scope_id)
        if not titles:
            logger.warning(f"기사 제목 없음 (scope_id={scope_id})")
            return None

        count = news_count if news_count is not None else len(titles)

        if count >= NEWS_COUNT_GEMINI:
            scope_title = _gemini_generate(scope_id, titles)
            if scope_title is None:
                # 플랜 C: 최신 제목 폴백
                scope_title = titles[0]
                logger.warning(f"Gemini 실패, 최신 제목 폴백: {scope_id} → {scope_title}")
        else:
            # 플랜 A: 최신 제목 그대로
            scope_title = titles[0]
            logger.info(f"최신 제목 사용 (news_count={count}): {scope_id} → {scope_title}")

        _upsert_scope_title(es, scope_id, scope_title)
        return scope_title

    except Exception as e:
        logger.error(f"generate_scope_title 예외 (scope_id={scope_id}): {e}")
        return None


def _enqueue(es, scope_id: str):
    """scope_refresh_queue에 pending 항목 등록 (중복 방지)

    failed 제외한 모든 상태(pending/processing/done) 체크하여
    같은 scope_id 중복 등록을 원천 차단.
    """
    existing = es.search(
        index=INDEX_QUEUE,
        body={
            "query": {
                "bool": {
                    "must":     {"term": {"scopeID": scope_id}},
                    "must_not": {"term": {"status":  "failed"}},
                }
            },
            "size": 1,
        },
    )
    if existing["hits"]["total"]["value"] == 0:
        now_utc = datetime.now(timezone.utc).isoformat()
        es.index(
            index=INDEX_QUEUE,
            body={
                "scopeID":     scope_id,
                "queued_at":   now_utc,
                "status":      "pending",
                "retry_count": 0,
            },
        )
        logger.info(f"scope_refresh_queue 등록: {scope_id}")


def trigger_scope_title(es, scope_id: str, news_count: int):
    """
    분류 파이프라인에서 scopeID 배정 직후 호출.

    news_count < 10  : 즉시 최신 제목 배정 (Gemini 없음)
    news_count >= 10 : scope_refresh_queue 등록 → 배치에서 Gemini 처리
    """
    if news_count < NEWS_COUNT_GEMINI:
        generate_scope_title(es, scope_id, news_count)
    else:
        _enqueue(es, scope_id)


def enqueue_missing_scope_titles():
    """
    scopeTitle이 없는 scope를 전체 조회하여 처리.
    news_count < 10  → 즉시 최신 제목 배정
    news_count >= 10 → queue 등록 후 배치 처리
    """
    try:
        es = get_es()
        res = es.search(
            index=INDEX_SCOPES,
            body={
                "query": {
                    "bool": {"must_not": {"exists": {"field": "scopeTitle"}}}
                },
                "_source": ["scopeID", "news_count"],
                "size": 1000,
            },
        )
        hits = res["hits"]["hits"]
        if not hits:
            logger.info("scopeTitle 누락 scope 없음")
            es.close()
            return 0

        count = 0
        for hit in hits:
            src      = hit["_source"]
            scope_id = src["scopeID"]
            nc       = src.get("news_count", 0)
            if nc >= NEWS_COUNT_GEMINI:
                _enqueue(es, scope_id)
            else:
                generate_scope_title(es, scope_id, nc)
            count += 1

        es.close()
        logger.info(f"scopeTitle 누락 scope {count}건 처리 완료")
        return count

    except Exception as e:
        log_pipeline_error(pipeline="scope_title_recovery", error=e)
        raise


def run_scope_title_batch():
    """
    scope_refresh_queue의 pending 항목 전체를 배치 루프로 처리합니다 — news_count >= 10 전용.

    - generate_scope_title 성공(str 반환) → done
    - generate_scope_title 실패(None 반환) → retry_count 증가
      · retry_count < MAX_RETRY : pending 유지 (다음 배치에서 재시도)
      · retry_count >= MAX_RETRY: failed 마킹 후 스킵 (무한 반복 방지)
    """
    try:
        es = get_es()

        total_processed = 0
        batch_num       = 0

        while True:
            batch_num += 1

            res = es.search(
                index=INDEX_QUEUE,
                body={
                    "query": {"term": {"status": "pending"}},
                    "sort":  [{"queued_at": "asc"}],
                    "size":  BATCH_SIZE,
                },
            )
            hits = res["hits"]["hits"]

            if not hits:
                if batch_num == 1:
                    logger.info("scope_refresh_queue 처리할 항목 없음")
                else:
                    logger.info(f"scopeTitle 배치 전체 완료 | total={total_processed}")
                break

            logger.info(f"[배치 {batch_num}] scopeTitle 처리 시작: {len(hits)}건")

            for hit in hits:
                doc_id      = hit["_id"]
                scope_id    = hit["_source"]["scopeID"]
                retry_count = hit["_source"].get("retry_count", 0)
                now_utc     = datetime.now(timezone.utc).isoformat()

                # MAX_RETRY 초과 시 failed 처리 후 스킵
                if retry_count >= MAX_RETRY:
                    logger.error(
                        f"최대 재시도 초과, failed 처리: scopeID={scope_id} "
                        f"(retry_count={retry_count})"
                    )
                    es.update(index=INDEX_QUEUE, id=doc_id,
                              body={"doc": {"status": "failed", "processed_at": now_utc}})
                    continue

                # 사용 가능한 키가 아예 없으면 배치 중단 (남은 항목 모두 pending 유지)
                if _get_next_available_key() is None:
                    logger.error("모든 Gemini 키 차단 상태 → 배치 중단, 다음 실행 시 재시도")
                    es.close()
                    return

                es.update(index=INDEX_QUEUE, id=doc_id,
                          body={"doc": {"status": "processing"}})

                result = generate_scope_title(es, scope_id, news_count=None)

                if result is not None:
                    es.update(index=INDEX_QUEUE, id=doc_id,
                              body={"doc": {"status": "done", "processed_at": now_utc}})
                    total_processed += 1
                else:
                    new_retry = retry_count + 1
                    if new_retry >= MAX_RETRY:
                        logger.error(f"재시도 한도 도달, failed 처리: scopeID={scope_id}")
                        es.update(index=INDEX_QUEUE, id=doc_id,
                                  body={"doc": {
                                      "status":       "failed",
                                      "retry_count":  new_retry,
                                      "processed_at": now_utc,
                                  }})
                    else:
                        logger.warning(
                            f"scopeTitle 생성 실패, 재시도 예정: scopeID={scope_id} "
                            f"(retry_count={new_retry}/{MAX_RETRY})"
                        )
                        es.update(index=INDEX_QUEUE, id=doc_id,
                                  body={"doc": {
                                      "status":      "pending",
                                      "retry_count": new_retry,
                                  }})

                time.sleep(10)  # gemini-2.5-flash-lite RPM 대응

            es.indices.refresh(index=INDEX_QUEUE)
            logger.info(f"[배치 {batch_num}] 완료 | 누적 processed={total_processed}")

        es.close()
        logger.info(f"scopeTitle 배치 처리 완료 | total={total_processed}")

    except Exception as e:
        log_pipeline_error(pipeline="scope_title", error=e)
        raise
