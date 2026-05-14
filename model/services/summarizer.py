"""
뉴스 본문 추출 요약 서비스 — TextRank + 첫 문장 유사도 방식

흐름:
  1. ES news_economy에서 summary=NULL / scopeID 존재 / keywords 존재 레코드 polling
  2. 캡션 제거 → 문장 분리
  3. TextRank 상위 3개 중 첫 번째 문장과 가장 유사한 1문장 선택
  4. 70자 초과 시 자연스러운 위치에서 절단 + … 처리
  5. news_economy.summary upsert

[수정 이력]
- 요약 방식 변경: TextRank 다중 문장 → 1문장 (70자 이내)
- 선택 로직: TextRank 상위 3개 중 첫 번째 문장과 코사인 유사도 가장 높은 문장 선택
- Gemini 트리밍 제거: 규칙 기반 자연 절단으로 대체 (API 한도 절약)
- 캡션 제거: [출처명] 패턴 전처리 추가
- [2026-05] summarize_single_article 추가: 단건 재처리용 래퍼
"""

import logging
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from model.database import get_es
from model.services.error_logger import log_pipeline_error

logger = logging.getLogger(__name__)

BATCH_SIZE          = 10000
TOP_N               = 3
MIN_SENTENCES       = 2
FALLBACK_LENGTH     = 70
DAMPING             = 0.85
MAX_SENTENCE_LENGTH = 70
INDEX_NEWS          = "news_economy"

# 자연스러운 절단 기준 어미/조사
_TRIM_CHARS = set('다며서고은는이가을를에의')


def _remove_captions(text: str) -> str:
    """[출처명] 패턴 등 이미지 캡션 제거"""
    text = re.sub(r"\[[^\]]{1,30}\]", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) >= 15]


def _textrank_scores(sentences: list[str]) -> np.ndarray:
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), max_features=2000)
    try:
        tfidf = vectorizer.fit_transform(sentences).toarray()
    except ValueError:
        return np.ones(len(sentences)) / len(sentences), None
    sim = cosine_similarity(tfidf)
    np.fill_diagonal(sim, 0)
    n      = len(sentences)
    scores = np.ones(n) / n
    for _ in range(50):
        prev = scores.copy()
        for i in range(n):
            s = sum(sim[i][j] * prev[j] / (sim[j].sum() or 1)
                    for j in range(n) if i != j)
            scores[i] = (1 - DAMPING) / n + DAMPING * s
        if np.abs(scores - prev).sum() < 1e-5:
            break
    return scores, tfidf


def _natural_trim(sentence: str, limit: int = MAX_SENTENCE_LENGTH) -> str:
    """70자 초과 시 조사/어미 단위로 자연스럽게 절단"""
    if len(sentence) <= limit:
        return sentence
    cut = sentence[:limit]
    for ch in range(len(cut) - 1, -1, -1):
        if cut[ch] in _TRIM_CHARS:
            return cut[:ch + 1] + "…"
    return cut[:limit] + "…"


def summarize(content: str) -> str:
    if not content or not content.strip():
        return "요약 품질이 충분치 않아 원문을 통해 내용 확인"

    content   = _remove_captions(content)
    sentences = _split_sentences(content)

    if len(sentences) < MIN_SENTENCES:
        return _natural_trim(content.strip())

    scores, tfidf = _textrank_scores(sentences)
    if tfidf is None:
        return _natural_trim(sentences[0])

    # TextRank 상위 3개 중 첫 번째 문장과 가장 유사한 1문장 선택
    top3_idx   = np.argsort(scores)[-TOP_N:].tolist()
    first_vec  = tfidf[0].reshape(1, -1)
    best_idx   = max(
        top3_idx,
        key=lambda i: cosine_similarity(first_vec, tfidf[i].reshape(1, -1))[0][0]
    )
    return _natural_trim(sentences[best_idx])


# ── 단건 재처리 래퍼 ───────────────────────────────────────────

def summarize_single_article(article_id: str, src: dict) -> None:
    """
    단건 아티클 요약 생성 후 ES 업데이트.
    classify.py의 reprocess_article 엔드포인트에서 호출됩니다.

    배치와 달리 scopeID / keywords 존재 여부를 사전에 체크하지 않습니다.
    (관리자가 명시적으로 재처리를 요청한 단건이므로 조건 무시)

    Args:
        article_id: news_economy 문서 ID
        src: ES _source dict (content 필드 포함)

    Raises:
        Exception: 요약 생성 실패 또는 ES 업데이트 실패 시
    """
    summary = summarize(src.get("content", ""))
    es = get_es()
    try:
        es.update(
            index=INDEX_NEWS,
            id=article_id,
            body={"doc": {"summary": summary}},
        )
        logger.info(f"[단건] 요약 완료 article_id={article_id} → {summary[:30]}…")
    finally:
        es.close()


# ── 배치 파이프라인 진입점 ─────────────────────────────────────

def run_summary_pipeline():
    """summary=NULL / scopeID·keywords 존재 뉴스 전체를 배치 루프로 요약합니다."""
    try:
        es = get_es()

        total_processed = 0
        batch_num       = 0

        while True:
            batch_num += 1

            res = es.search(
                index=INDEX_NEWS,
                body={
                    "query": {
                        "bool": {
                            "must": [
                                {"exists": {"field": "scopeID"}},
                                {"exists": {"field": "keywords"}},
                            ],
                            "must_not": {"exists": {"field": "summary"}},
                        }
                    },
                    "_source": ["article_id", "content"],
                    "sort":    [{"published_at": "asc"}],
                    "size":    BATCH_SIZE,
                },
            )
            hits = res["hits"]["hits"]

            if not hits:
                if batch_num == 1:
                    logger.info("요약할 뉴스 없음")
                else:
                    logger.info(f"요약 전체 완료 | total={total_processed}")
                break

            logger.info(f"[배치 {batch_num}] 요약 시작: {len(hits)}건")

            for hit in hits:
                src        = hit["_source"]
                article_id = src["article_id"]
                try:
                    summary = summarize(src.get("content", ""))
                    es.update(
                        index=INDEX_NEWS,
                        id=article_id,
                        body={"doc": {"summary": summary}},
                    )
                    total_processed += 1
                except Exception as e:
                    logger.error(f"요약 실패 article_id={article_id}: {e}")
                    continue

            es.indices.refresh(index=INDEX_NEWS)
            logger.info(f"[배치 {batch_num}] 완료 | 누적 processed={total_processed}")

        es.close()

    except Exception as e:
        log_pipeline_error(pipeline="summarizer", error=e)
        raise
