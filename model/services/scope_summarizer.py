"""
scope 대표 요약 생성 서비스

흐름:
  1. news_scopes에서 scope_summary=NULL scope 조회
  2. news_economy.summary 수집 (동일 언론사 최대 3건)
  3. 문장 분리 → TextRank 상위 3개 중 scope_keywords 포함 빈도 가장 높은 1문장 선택
  4. 70자 자연 절단
  5. news_scopes.scope_summary upsert

[수정 이력]
- Gemini 완전 제거: _trim_sentence Gemini 트리밍 → 규칙 기반 자연 절단으로 대체
- 요약 방식 변경: TextRank 다중 문장 → 1문장 (70자 이내)
- 선택 로직: TextRank 상위 3개 중 scope_keywords 포함 빈도 가장 높은 문장 선택
- MIN_NEWS_COUNT 제거: 뉴스 1건이어도 요약 생성
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

BATCH_SIZE          = 500
MAX_NEWS_PER_PRESS  = 3
TOP_N               = 3
DAMPING             = 0.85
DUPLICATE_THRESHOLD = 0.85
MAX_SENTENCE_LENGTH = 70
INDEX_NEWS          = "news_economy"
INDEX_SCOPES        = "news_scopes"

_TRIM_CHARS = set('다며서고은는이가을를에의')


def _remove_captions(text: str) -> str:
    text = re.sub(r"\[[^\]]{1,30}\]", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _split_sentences(text: str) -> list:
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) >= 15]


def _remove_duplicates(sentences: list) -> list:
    if len(sentences) < 2:
        return sentences
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), max_features=2000)
    try:
        matrix = vectorizer.fit_transform(sentences).toarray()
    except ValueError:
        return sentences
    sim = cosine_similarity(matrix)
    keep, removed = [], set()
    for i in range(len(sentences)):
        if i in removed:
            continue
        keep.append(sentences[i])
        for j in range(i + 1, len(sentences)):
            if sim[i][j] >= DUPLICATE_THRESHOLD:
                removed.add(j)
    return keep


def _textrank_scores(sentences: list):
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
            s = sum(sim[i][j] * prev[j] / (sim[j].sum() or 1) for j in range(n) if i != j)
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


def _keyword_score(sentence: str, keywords: list) -> int:
    """문장에 scope_keywords가 몇 개 포함되는지 카운트"""
    return sum(1 for kw in keywords if kw in sentence)


def generate_scope_summary(es, scope_id: str):
    # scope_keywords 조회
    scope_res = es.get(index=INDEX_SCOPES, id=scope_id, ignore=404)
    scope_keywords = []
    if scope_res.get("found"):
        kw_str = scope_res["_source"].get("scope_keywords", "")
        scope_keywords = [k.strip() for k in kw_str.split(",") if k.strip()]

    # 뉴스 summary 수집
    res = es.search(
        index=INDEX_NEWS,
        body={
            "query": {
                "bool": {
                    "must": [
                        {"term":   {"scopeID": scope_id}},
                        {"exists": {"field": "summary"}},
                    ]
                }
            },
            "_source": ["summary", "press"],
            "sort":    [{"published_at": "desc"}],
            "size":    100,
        },
    )
    rows = [h["_source"] for h in res["hits"]["hits"]]
    if not rows:
        return None

    # 동일 언론사 최대 MAX_NEWS_PER_PRESS건
    press_count, filtered = {}, []
    for r in rows:
        press = r.get("press") or "unknown"
        press_count[press] = press_count.get(press, 0) + 1
        if press_count[press] <= MAX_NEWS_PER_PRESS and r.get("summary"):
            filtered.append(r["summary"])

    if not filtered:
        return None

    # 뉴스 1건이면 그대로 절단
    if len(filtered) == 1:
        summary = _natural_trim(_remove_captions(filtered[0]))
        _save(es, scope_id, summary)
        return summary

    # 문장 분리 → 중복 제거
    combined  = " ".join([_remove_captions(s) for s in filtered])
    sentences = _split_sentences(combined)
    if not sentences:
        summary = _natural_trim(combined)
        _save(es, scope_id, summary)
        return summary

    sentences = _remove_duplicates(sentences)

    if len(sentences) == 1:
        summary = _natural_trim(sentences[0])
        _save(es, scope_id, summary)
        return summary

    # TextRank 상위 TOP_N개 중 scope_keywords 포함 빈도 가장 높은 1문장 선택
    scores, _ = _textrank_scores(sentences)
    top_n_idx = np.argsort(scores)[-TOP_N:].tolist()

    if scope_keywords:
        best_idx = max(top_n_idx, key=lambda i: _keyword_score(sentences[i], scope_keywords))
    else:
        best_idx = int(np.argmax(scores))

    summary = _natural_trim(sentences[best_idx])
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


def run_scope_summary_batch():
    """scope_summary 배치 처리 (1일 1회)"""
    try:
        es = get_es()

        res = es.search(
            index=INDEX_SCOPES,
            body={
                "query": {
                    "bool": {
                        "should": [
                            {"bool": {"must_not": {"exists": {"field": "scope_summary"}}}},
                            {"range": {"news_count": {"gt": 0}}},
                        ],
                        "minimum_should_match": 1,
                    }
                },
                "_source": ["scopeID"],
                "sort":    [{"created_at": "asc"}],
                "size":    BATCH_SIZE,
            },
        )
        hits = res["hits"]["hits"]

        if not hits:
            logger.info("scope_summary 처리할 scope 없음")
            es.close()
            return

        logger.info(f"scope_summary 배치 처리 시작: {len(hits)}건")

        for hit in hits:
            scope_id = hit["_source"]["scopeID"]
            try:
                generate_scope_summary(es, scope_id)
            except Exception as e:
                logger.error(f"scope_summary 생성 실패 scopeID={scope_id}: {e}")
                continue

        es.close()
        logger.info(f"scope_summary 배치 처리 완료: {len(hits)}건")

    except Exception as e:
        log_pipeline_error(pipeline="scope_summarizer", error=e)
        raise
