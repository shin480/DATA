"""
scope 대표 요약 생성 서비스 — 기존 embedding 재활용 기반

흐름:
  1. news_scopes에서 scope_summary=NULL scope 조회
  2. news_economy에서 content + embedding 수집 (동일 언론사 최대 3건, 최신순)
  3. 본문 클렌징 (언론사 헤더/기자명/저작권/AI분석 문구 등 제거)
  4. 문장 분리 → 중복 제거
  5. 기사 embedding 평균 → 스콥 중심 벡터
     각 문장이 속한 기사의 embedding을 문장 벡터로 사용
     → 중심 벡터와 코사인 유사도 가장 높은 문장 선택
  6. 70자 자연 절단
  7. news_scopes.scope_summary upsert

[수정 이력]
- 속도 개선: KR-FinBERT 문장별 실시간 임베딩 → 기존 news_economy.embedding 재활용
  · KR-FinBERT CPU 추론 → 시간당 1000건 수준으로 너무 느림
  · embedding 필드는 이미 계산된 768차원 벡터 → 추가 연산 없음
- 클렌징 강화: "관련 기사 N건을 분석한 결과입니다" 등 AI 생성 문구 패턴 추가
- 기사 없는 스콥: 빈 문자열 저장 → return None으로 변경 (카운트 오염 방지)
"""

import logging
import re
from datetime import datetime, timezone

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from model.database import get_es
from model.services.error_logger import log_pipeline_error

logger = logging.getLogger(__name__)

BATCH_SIZE          = 10000
MAX_NEWS_PER_PRESS  = 3
DUPLICATE_THRESHOLD = 0.85
MAX_SENTENCE_LENGTH = 70
MIN_SENTENCE_LENGTH = 15
INDEX_NEWS          = "news_economy"
INDEX_SCOPES        = "news_scopes"

_TRIM_CHARS = set('다며서고은는이가을를에의')


# ── 본문 클렌징 ────────────────────────────────────────────────────

_CLEAN_PATTERNS = [
    # 방송사 태그: [KBS 춘천], [MBC] 등
    (re.compile(r"^\s*\[[^\]]{1,20}\]\s*"), ""),
    (re.compile(r"\[[^\]]{1,20}\]"), " "),
    # 언론사 발신 헤더: (서울=뉴스1) 홍길동 기자 =
    (re.compile(r"\([^)]{1,20}=[^)]{1,20}\)\s*[가-힣\s]{2,10}기자\s*=?\s*"), ""),
    # 기자 byline
    (re.compile(r"[가-힣]{2,4}[·]?[가-힣]{0,4}\s*기자\s*=?\s*"), ""),
    # ⓒ 저작권 표시
    (re.compile(r"ⓒ\s*[^\s,]{1,20}"), ""),
    # 날짜 패턴
    (re.compile(r"\d{4}[.\-]\d{1,2}[.\-]\d{1,2}"), ""),
    # 저작권 문구
    (re.compile(r"무단\s*전재[^.]*재배포\s*금지[^.]*\.?"), ""),
    (re.compile(r"저작권자[^.]*\.?"), ""),
    (re.compile(r"All rights reserved[^.]*\.?", re.IGNORECASE), ""),
    # 사진/그래픽 캡션
    (re.compile(r"\([^)]{1,30}=[^)]{1,30}\)"), ""),
    # AI 분석 문구: "관련 기사 N건을 분석한 결과입니다" 등
    (re.compile(r"관련\s*기사\s*\d+건[^.]*\.?"), ""),
    (re.compile(r"\d+건을?\s*분석한\s*결과[^.]*\.?"), ""),
    (re.compile(r"이\s*기사[는은]\s*AI[^.]*\.?"), ""),
    (re.compile(r"AI\s*(가|이|로)\s*[^.]{1,30}(작성|생성|요약)[^.]*\.?"), ""),
    # 중복 공백
    (re.compile(r"\s{2,}"), " "),
]


def _clean_content(text: str) -> str:
    for pattern, repl in _CLEAN_PATTERNS:
        text = pattern.sub(repl, text)
    return text.strip()


# ── 문장 처리 ──────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) >= MIN_SENTENCE_LENGTH]


def _remove_duplicates(sentences: list[str]) -> list[str]:
    if len(sentences) < 2:
        return sentences
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 3), max_features=2000
    )
    try:
        matrix = vectorizer.fit_transform(sentences).toarray()
    except ValueError:
        return sentences
    sim           = cosine_similarity(matrix)
    keep, removed = [], set()
    for i in range(len(sentences)):
        if i in removed:
            continue
        keep.append(sentences[i])
        for j in range(i + 1, len(sentences)):
            if sim[i][j] >= DUPLICATE_THRESHOLD:
                removed.add(j)
    return keep


def _natural_trim(sentence: str, limit: int = MAX_SENTENCE_LENGTH) -> str:
    if len(sentence) <= limit:
        return sentence
    cut = sentence[:limit]
    for ch in range(len(cut) - 1, -1, -1):
        if cut[ch] in _TRIM_CHARS:
            return cut[:ch + 1] + "…"
    return cut[:limit] + "…"


# ── 임베딩 기반 중심 문장 선택 ────────────────────────────────────

def _select_central_sentence(
    sentences: list[str],
    sent_embeddings: list[np.ndarray],
) -> int:
    """
    기사 embedding을 문장 벡터로 사용하여 스콥 중심에 가장 가까운 문장 선택.

    sent_embeddings: 각 문장이 속한 기사의 embedding (문장과 1:1 대응)
    중심 벡터 = 전체 embedding 평균
    → 코사인 유사도 가장 높은 문장 인덱스 반환
    """
    matrix   = np.vstack(sent_embeddings).astype(np.float32)  # (N, 768)
    centroid = matrix.mean(axis=0, keepdims=True)              # (1, 768)
    # L2 정규화
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    sims = cosine_similarity(centroid, matrix)[0]              # (N,)
    return int(np.argmax(sims))


# ── 메인 생성 함수 ─────────────────────────────────────────────────

def generate_scope_summary(es, scope_id: str):
    # 뉴스 content + embedding 수집
    res = es.search(
        index=INDEX_NEWS,
        body={
            "query": {
                "bool": {
                    "must": [
                        {"term":   {"scopeID": scope_id}},
                        {"exists": {"field":   "content"}},
                        {"term":   {"has_embedding": True}},
                    ]
                }
            },
            "_source": ["content", "press", "embedding"],
            "sort":    [{"published_at": "desc"}],
            "size":    100,
        },
    )
    rows = [h["_source"] for h in res["hits"]["hits"]]
    if not rows:
        # 빈 값 저장 안 함 → 다음 배치에서 재시도 가능하도록
        logger.warning(f"content/embedding 없음, 스킵: scope_id={scope_id}")
        return None

    # 동일 언론사 최대 MAX_NEWS_PER_PRESS건
    press_count = {}
    filtered    = []  # (cleaned_content, embedding_vec)
    for r in rows:
        press = r.get("press") or "unknown"
        press_count[press] = press_count.get(press, 0) + 1
        if press_count[press] <= MAX_NEWS_PER_PRESS and r.get("content") and r.get("embedding"):
            cleaned = _clean_content(r["content"])
            emb     = np.array(r["embedding"], dtype=np.float32)
            if emb.shape[0] == 768:
                filtered.append((cleaned, emb))

    if not filtered:
        logger.warning(f"유효한 content/embedding 없음, 스킵: scope_id={scope_id}")
        return None

    # 문장 분리 → 문장별로 소속 기사 embedding 매핑
    all_sentences  = []
    all_embeddings = []
    for cleaned, emb in filtered:
        sents = _split_sentences(cleaned)
        for s in sents:
            all_sentences.append(s)
            all_embeddings.append(emb)

    if not all_sentences:
        # 문장 분리 실패 → 첫 번째 기사 첫 단락 절단
        summary = _natural_trim(filtered[0][0])
        _save(es, scope_id, summary)
        return summary

    # 중복 제거 (embedding도 같이 필터링)
    if len(all_sentences) >= 2:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 3), max_features=2000
        )
        try:
            matrix = vectorizer.fit_transform(all_sentences).toarray()
            sim    = cosine_similarity(matrix)
            keep_idx = []
            removed  = set()
            for i in range(len(all_sentences)):
                if i in removed:
                    continue
                keep_idx.append(i)
                for j in range(i + 1, len(all_sentences)):
                    if sim[i][j] >= DUPLICATE_THRESHOLD:
                        removed.add(j)
            all_sentences  = [all_sentences[i]  for i in keep_idx]
            all_embeddings = [all_embeddings[i] for i in keep_idx]
        except ValueError:
            pass

    # 문장 1개면 바로 사용
    if len(all_sentences) == 1:
        summary = _natural_trim(all_sentences[0])
        _save(es, scope_id, summary)
        return summary

    # embedding 기반 중심 문장 선택
    best_idx = _select_central_sentence(all_sentences, all_embeddings)
    summary  = _natural_trim(all_sentences[best_idx])
    _save(es, scope_id, summary)
    return summary


def _save(es, scope_id: str, summary: str):
    es.update(
        index=INDEX_SCOPES,
        id=scope_id,
        body={"doc": {
            "scope_summary": summary,
            "updated_at":    datetime.now(timezone.utc).isoformat(),
        }},
    )
    logger.info(f"scope_summary 저장: {scope_id} ({len(summary)}자)")


# ── 배치 처리 ─────────────────────────────────────────────────────

def run_scope_summary_batch():
    """scope_summary 없는 scope 전체를 배치 루프로 처리합니다."""
    try:
        es = get_es()

        total_processed = 0
        batch_num       = 0

        while True:
            batch_num += 1

            res = es.search(
                index=INDEX_SCOPES,
                body={
                    "query": {
                        "bool": {
                            "must_not": {"exists": {"field": "scope_summary"}}
                        }
                    },
                    "_source": ["scopeID"],
                    "sort":    [{"created_at": "asc"}],
                    "size":    BATCH_SIZE,
                },
            )
            hits = res["hits"]["hits"]

            if not hits:
                if batch_num == 1:
                    logger.info("scope_summary 처리할 scope 없음")
                else:
                    logger.info(f"scope_summary 전체 완료 | total={total_processed}")
                break

            logger.info(f"[배치 {batch_num}] scope_summary 처리 시작: {len(hits)}건")

            processed_in_batch = 0
            for hit in hits:
                scope_id = hit["_source"]["scopeID"]
                try:
                    result = generate_scope_summary(es, scope_id)
                    if result is not None:
                        total_processed    += 1
                        processed_in_batch += 1
                except Exception as e:
                    logger.error(f"scope_summary 생성 실패 scopeID={scope_id}: {e}")
                    continue

            es.indices.refresh(index=INDEX_SCOPES)
            logger.info(f"[배치 {batch_num}] 완료 | 누적 processed={total_processed}")

            if processed_in_batch == 0:
                logger.warning("배치에서 처리된 건 없음, 루프 강제 탈출")
                break

        es.close()

    except Exception as e:
        log_pipeline_error(pipeline="scope_summarizer", error=e)
        raise
