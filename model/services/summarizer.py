"""
뉴스 본문 추출 요약 서비스 — TextRank 방식

흐름:
  1. ES news_economy에서 summary=NULL / scopeID 존재 / keywords 존재 레코드 polling
  2. TextRank로 본문 핵심 문장 추출
  3. news_economy.summary upsert
"""

import logging
import os
import re

import numpy as np
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from model.database import get_es
from model.services.error_logger import log_pipeline_error

logger = logging.getLogger(__name__)

BATCH_SIZE             = 100
TOP_N                  = 3
FINAL_N                = 2
MIN_SENTENCES          = 2
FALLBACK_LENGTH        = 200
DAMPING                = 0.85
MAX_SENTENCE_LENGTH    = 70
MAX_SENTENCE_TOLERANCE = 5
INDEX_NEWS             = "news_economy"

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


def _split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) >= 15]


def _textrank(sentences: list[str], top_n: int) -> list[str]:
    if len(sentences) <= top_n:
        return sentences
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), max_features=2000)
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences).toarray()
    except ValueError:
        return sentences[:top_n]
    sim_matrix = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(sim_matrix, 0)
    n      = len(sentences)
    scores = np.ones(n) / n
    for _ in range(50):
        prev = scores.copy()
        for i in range(n):
            s = sum(sim_matrix[i][j] * prev[j] / (sim_matrix[j].sum() or 1)
                    for j in range(n) if i != j)
            scores[i] = (1 - DAMPING) / n + DAMPING * s
        if np.abs(scores - prev).sum() < 1e-5:
            break
    top_idx = sorted(np.argsort(scores)[-top_n:].tolist())
    return [sentences[i] for i in top_idx]


def _trim_sentence(sentence: str) -> str:
    if len(sentence) <= MAX_SENTENCE_LENGTH + MAX_SENTENCE_TOLERANCE:
        return sentence
    try:
        prompt = (f"다음 문장을 {MAX_SENTENCE_LENGTH}자 이내로 줄여주세요.\n"
                  f"핵심 내용은 유지하고, 문장만 출력하세요.\n\n문장: {sentence}")
        return _get_gemini_model().generate_content(prompt).text.strip()
    except Exception as e:
        logger.warning(f"문장 재요약 실패: {e}")
        return sentence[:MAX_SENTENCE_LENGTH]


def summarize(content: str) -> str:
    if not content or not content.strip():
        return "요약 품질이 충분치 않아 원문을 통해 내용 확인"
    sentences = _split_sentences(content)
    if len(sentences) < MIN_SENTENCES:
        return content.strip()[:FALLBACK_LENGTH]
    selected = _textrank(sentences, TOP_N)
    if not selected:
        return "요약 품질이 충분치 않아 원문을 통해 내용 확인"
    trimmed = [_trim_sentence(s) for s in selected]
    return " ".join(trimmed[:FINAL_N])


def run_summary_pipeline():
    """summary=NULL / scopeID·keywords 존재 뉴스를 배치 요약합니다."""
    try:
        es = get_es()

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
            logger.info("요약할 뉴스 없음")
            es.close()
            return

        logger.info(f"요약 시작: {len(hits)}건")

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
            except Exception as e:
                logger.error(f"요약 실패 article_id={article_id}: {e}")
                continue

        es.close()
        logger.info(f"요약 완료: {len(hits)}건")

    except Exception as e:
        log_pipeline_error(pipeline="summarizer", error=e)
        raise
