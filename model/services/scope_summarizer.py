"""
scope 대표 요약 생성 서비스

흐름:
  1. news_scopes에서 scope_summary=NULL 이거나 갱신 필요 scope 조회
  2. news_economy.summary 수집 (동일 언론사 최대 3건)
  3. TextRank + Gemini 길이 조정
  4. news_scopes.scope_summary upsert
"""

import logging
import os
import re
from datetime import datetime, timezone

import numpy as np
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from model.database import get_es
from model.services.error_logger import log_pipeline_error

logger = logging.getLogger(__name__)

BATCH_SIZE             = 50
MIN_NEWS_COUNT         = 3
MAX_NEWS_PER_PRESS     = 3
TOP_N                  = 5
FINAL_N                = 3
MIN_SENTENCES          = 2
FALLBACK_LENGTH        = 200
MAX_SENTENCE_LENGTH    = 70
MAX_SENTENCE_TOLERANCE = 5
DAMPING                = 0.85
DUPLICATE_THRESHOLD    = 0.85
INDEX_NEWS             = "news_economy"
INDEX_SCOPES           = "news_scopes"

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
_key_index   = 0
_gemini_model = None


def _rotate_key():
    global _key_index, _gemini_model
    _key_index    = (_key_index + 1) % len(_API_KEYS)
    _gemini_model = None
    import logging
    logging.getLogger(__name__).warning(f"API 키 교체 → index {_key_index}")


def _get_gemini_model():
    global _gemini_model, _key_index
    if _gemini_model is None:
        genai.configure(api_key=_API_KEYS[_key_index])
        _gemini_model = genai.GenerativeModel("gemini-2.5-flash-lite")
    return _gemini_model


def _split_sentences(text):
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) >= 15]


def _remove_duplicates(sentences):
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


def _textrank(sentences, top_n):
    if len(sentences) <= top_n:
        return sentences
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), max_features=2000)
    try:
        tfidf = vectorizer.fit_transform(sentences).toarray()
    except ValueError:
        return sentences[:top_n]
    sim = cosine_similarity(tfidf)
    np.fill_diagonal(sim, 0)
    n, scores = len(sentences), np.ones(len(sentences)) / len(sentences)
    for _ in range(50):
        prev = scores.copy()
        for i in range(n):
            s = sum(sim[i][j] * prev[j] / (sim[j].sum() or 1) for j in range(n) if i != j)
            scores[i] = (1 - DAMPING) / n + DAMPING * s
        if np.abs(scores - prev).sum() < 1e-5:
            break
    top_idx = sorted(np.argsort(scores)[-top_n:].tolist())
    return [sentences[i] for i in top_idx]


def _trim_sentence(sentence):
    if len(sentence) <= MAX_SENTENCE_LENGTH + MAX_SENTENCE_TOLERANCE:
        return sentence
    try:
        prompt = (f"다음 문장을 {MAX_SENTENCE_LENGTH}자 이내로 줄여주세요.\n"
                  f"핵심 내용은 유지하고, 문장만 출력하세요.\n\n문장: {sentence}")
        return _get_model().generate_content(prompt).text.strip()
    except Exception as e:
        logger.warning(f"문장 재요약 실패: {e}")
        return sentence[:MAX_SENTENCE_LENGTH]


def generate_scope_summary(es, scope_id: str):
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

    press_count, filtered = {}, []
    for r in rows:
        press = r.get("press") or "unknown"
        press_count[press] = press_count.get(press, 0) + 1
        if press_count[press] <= MAX_NEWS_PER_PRESS and r.get("summary"):
            filtered.append(r["summary"])

    if len(filtered) < MIN_NEWS_COUNT:
        return None

    combined  = " ".join(filtered)
    sentences = _split_sentences(combined)
    if len(sentences) < MIN_SENTENCES:
        summary = combined[:FALLBACK_LENGTH]
    else:
        sentences = _remove_duplicates(sentences)
        selected  = _textrank(sentences, TOP_N)
        if not selected:
            summary = "요약 품질이 충분치 않아 원문을 통해 내용 확인"
        else:
            trimmed = [_trim_sentence(s) for s in selected]
            summary = " ".join(trimmed[:FINAL_N])

    es.update(
        index=INDEX_SCOPES,
        id=scope_id,
        body={"doc": {
            "scope_summary": summary,
            "updated_at":    datetime.now(timezone.utc).isoformat(),
        }},
    )
    return summary


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
