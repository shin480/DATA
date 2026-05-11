"""
scope 단위 감성 집계 서비스

흐름:
  1. news_scopes에서 sentiment=NULL 이거나 갱신 필요한 scope 조회
  2. news_economy에서 해당 scope 뉴스의 sentiment/sentiment_score 집계
  3. news_scopes.sentiment / sentiment_score / sentiment_dist upsert
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone

from model.database import get_es
from model.services.error_logger import log_pipeline_error

logger = logging.getLogger(__name__)

BATCH_SIZE     = 500
MIN_NEWS_COUNT = 3
INDEX_NEWS     = "news_economy"
INDEX_SCOPES   = "news_scopes"


def aggregate_scope_sentiment(es, scope_id: str):
    res = es.search(
        index=INDEX_NEWS,
        body={
            "query": {
                "bool": {
                    "must": [
                        {"term":   {"scopeID": scope_id}},
                        {"exists": {"field": "sentiment"}},
                        {"exists": {"field": "sentiment_score"}},
                    ]
                }
            },
            "_source": ["sentiment", "sentiment_score"],
            "size": 10000,
        },
    )
    rows = [h["_source"] for h in res["hits"]["hits"]]

    if len(rows) < MIN_NEWS_COUNT:
        return None

    score_sum   = defaultdict(float)
    score_count = defaultdict(int)
    for r in rows:
        s = r["sentiment"]
        score_sum[s]   += r["sentiment_score"]
        score_count[s] += 1

    sentiment       = max(score_sum, key=score_sum.get)
    sentiment_score = round(score_sum[sentiment] / score_count[sentiment], 4)
    total           = len(rows)
    sentiment_dist  = {
        label: round(score_count[label] / total, 4)
        for label in ("positive", "negative", "neutral")
    }

    es.update(
        index=INDEX_SCOPES,
        id=scope_id,
        body={"doc": {
            "sentiment":       sentiment,
            "sentiment_score": sentiment_score,
            "sentiment_dist":  sentiment_dist,
            "updated_at":      datetime.now(timezone.utc).isoformat(),
        }},
    )
    return sentiment, sentiment_score, sentiment_dist


def run_scope_sentiment_batch():
    """scope 감성 집계 배치 처리 (30분 주기)"""
    try:
        es = get_es()

        res = es.search(
            index=INDEX_SCOPES,
            body={
                "query": {
                    "bool": {
                        "should": [
                            {"bool": {"must_not": {"exists": {"field": "sentiment"}}}},
                            {"range": {"news_count": {"gt": 0}}},
                        ],
                        "minimum_should_match": 1,
                    }
                },
                "_source": ["scopeID"],
                "size": BATCH_SIZE,
            },
        )
        hits = res["hits"]["hits"]

        if not hits:
            logger.info("scope 감성 집계할 대상 없음")
            es.close()
            return

        logger.info(f"scope 감성 집계 시작: {len(hits)}건")

        for hit in hits:
            scope_id = hit["_source"]["scopeID"]
            try:
                aggregate_scope_sentiment(es, scope_id)
            except Exception as e:
                logger.error(f"scope 감성 집계 실패 scopeID={scope_id}: {e}")
                continue

        es.close()
        logger.info(f"scope 감성 집계 완료: {len(hits)}건")

    except Exception as e:
        log_pipeline_error(pipeline="scope_sentiment", error=e)
        raise
