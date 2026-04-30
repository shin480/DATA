"""
뉴스 키워드 추출 서비스 — TF-IDF 방식

흐름:
  1. ES news_economy에서 keywords=NULL 레코드 polling
  2. 제목 + 본문으로 TF-IDF 키워드 추출
  3. news_economy.keywords upsert
"""

import logging
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from model.database import get_es
from model.services.error_logger import log_pipeline_error

logger = logging.getLogger(__name__)

BATCH_SIZE   = 100
MAX_KEYWORDS = 5
TITLE_WEIGHT = 2.0
MIN_WORD_LEN = 2
INDEX_NEWS   = "news_economy"


def _preprocess(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_valid_keyword(word: str) -> bool:
    if len(word) < MIN_WORD_LEN:
        return False
    if re.fullmatch(r"\d+", word):
        return False
    return True


def extract_keywords(title: str, content: str, max_keywords: int = MAX_KEYWORDS) -> list[str]:
    title   = _preprocess(title)
    content = _preprocess(content)
    title_repeated = " ".join([title] * int(TITLE_WEIGHT))
    full_text = f"{title_repeated} {content}"

    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[가-힣a-zA-Z0-9]+",
        ngram_range=(1, 1),
        max_features=1000,
        sublinear_tf=True,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform([full_text])
    except ValueError:
        return []

    feature_names = vectorizer.get_feature_names_out()
    scores        = tfidf_matrix.toarray()[0]
    word_scores   = [
        (word, score)
        for word, score in zip(feature_names, scores)
        if score > 0 and _is_valid_keyword(word)
    ]
    word_scores.sort(key=lambda x: x[1], reverse=True)
    return [word for word, _ in word_scores[:max_keywords]]


def run_keyword_pipeline():
    """keywords=NULL 뉴스를 배치로 가져와 키워드를 추출합니다."""
    try:
        es = get_es()

        res = es.search(
            index=INDEX_NEWS,
            body={
                "query": {"bool": {"must_not": {"exists": {"field": "keywords"}}}},
                "_source": ["article_id", "title", "content"],
                "sort":    [{"published_at": "asc"}],
                "size":    BATCH_SIZE,
            },
        )
        hits = res["hits"]["hits"]

        if not hits:
            logger.info("키워드 추출할 뉴스 없음")
            es.close()
            return

        logger.info(f"키워드 추출 시작: {len(hits)}건")

        for hit in hits:
            src        = hit["_source"]
            article_id = src["article_id"]
            try:
                keywords     = extract_keywords(src.get("title", ""), src.get("content", ""))
                keywords_str = ",".join(keywords)
                es.update(
                    index=INDEX_NEWS,
                    id=article_id,
                    body={"doc": {"keywords": keywords_str}},
                )
            except Exception as e:
                logger.error(f"키워드 추출 실패 article_id={article_id}: {e}")
                continue

        es.close()
        logger.info(f"키워드 추출 완료: {len(hits)}건")

    except Exception as e:
        log_pipeline_error(pipeline="keyword", error=e)
        raise
