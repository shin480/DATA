"""
scope 단위 감성 집계 서비스

흐름:
  1. news_scopes에서 sentiment_dist=NULL인 scope 조회
  2. news_economy에서 해당 scope 뉴스의 sentiment/sentiment_score 집계
  3. news_scopes.sentiment_score / sentiment_dist upsert

[수정 이력]
- sentiment(대표 레이블) 제거: scope 자체는 단일 감성 레이블을 갖지 않음
  · sentiment_score: 전체 기사 sentiment_score 평균
  · sentiment_dist:  전체 기사의 레이블 분포 비율 (positive/negative/neutral)
- 배치 조건 변경: must_not exists sentiment → must_not exists sentiment_dist
"""

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

    score_count = defaultdict(int)
    for r in rows:
        score_count[r["sentiment"]] += 1

    total          = len(rows)
    sentiment_dist = {
        label: round(score_count[label] / total, 4)
        for label in ("positive", "negative", "neutral")
    }

    # 전체 기사 sentiment_score 평균
    all_scores      = [r["sentiment_score"] for r in rows]
    sentiment_score = round(sum(all_scores) / len(all_scores), 4)

    es.update(
        index=INDEX_SCOPES,
        id=scope_id,
        body={"doc": {
            "sentiment_score": sentiment_score,
            "sentiment_dist":  sentiment_dist,
            "updated_at":      datetime.now(timezone.utc).isoformat(),
        }},
    )
    return sentiment_score, sentiment_dist


def run_scope_sentiment_batch():
    """scope 감성 집계 전체를 배치 루프로 처리합니다."""
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
                            "must": [
                                {"range": {"news_count": {"gte": MIN_NEWS_COUNT}}}
                            ],
                            "must_not": {"exists": {"field": "sentiment_dist"}},
                        }
                    },
                    "_source": ["scopeID"],
                    "size": BATCH_SIZE,
                },
            )
            hits = res["hits"]["hits"]

            if not hits:
                if batch_num == 1:
                    logger.info("scope 감성 집계할 대상 없음")
                else:
                    logger.info(f"scope 감성 집계 전체 완료 | total={total_processed}")
                break

            logger.info(f"[배치 {batch_num}] scope 감성 집계 시작: {len(hits)}건")

            for hit in hits:
                scope_id = hit["_source"]["scopeID"]
                try:
                    aggregate_scope_sentiment(es, scope_id)
                    total_processed += 1
                except Exception as e:
                    logger.error(f"scope 감성 집계 실패 scopeID={scope_id}: {e}")
                    continue

            es.indices.refresh(index=INDEX_SCOPES)
            logger.info(f"[배치 {batch_num}] 완료 | 누적 processed={total_processed}")

        es.close()

    except Exception as e:
        log_pipeline_error(pipeline="scope_sentiment", error=e)
        raise
