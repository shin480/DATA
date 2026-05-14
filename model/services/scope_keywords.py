"""
scope 단위 대표 키워드 집계 서비스

흐름:
  1. news_scopes에서 scope_keywords=NULL 이거나 updated_at 이후 갱신 필요한 scope 조회
  2. 해당 scope의 news_economy.keywords 집계
  3. news_scopes.scope_keywords upsert
"""

import logging
from collections import Counter
from datetime import datetime, timezone

from model.database import get_es
from model.services.error_logger import log_pipeline_error

logger = logging.getLogger(__name__)

MAX_SCOPE_KEYWORDS = 5
BATCH_SIZE         = 500
INDEX_NEWS         = "news_economy"
INDEX_SCOPES       = "news_scopes"
MIN_NEWS_COUNT = 3

def aggregate_scope_keywords(es, scope_id: str) -> str:
    # 해당 scope 뉴스의 keywords 수집
    res = es.search(
        index=INDEX_NEWS,
        body={
            "query": {
                "bool": {
                    "must": [
                        {"term":   {"scopeID": scope_id}},
                        {"exists": {"field": "keywords"}},
                    ]
                }
            },
            "_source": ["keywords"],
            "size": 1000,
        },
    )
    hits = res["hits"]["hits"]
    if not hits:
        return ""

    # 기존 scope_keywords 2배 가중
    scope_res  = es.get(index=INDEX_SCOPES, id=scope_id, ignore=404)
    all_keywords = []
    if scope_res.get("found"):
        existing_kw = scope_res["_source"].get("scope_keywords")
        if existing_kw:
            existing = [k.strip() for k in existing_kw.split(",") if k.strip()]
            all_keywords.extend(existing * 2)

    for hit in hits:
        kw = hit["_source"].get("keywords", "")
        all_keywords.extend([k.strip() for k in kw.split(",") if k.strip()])

    if not all_keywords:
        return ""

    counter     = Counter(all_keywords)
    top_keywords = [w for w, _ in counter.most_common(MAX_SCOPE_KEYWORDS)]
    result       = ",".join(top_keywords)

    es.update(
        index=INDEX_SCOPES,
        id=scope_id,
        body={"doc": {
            "scope_keywords": result,
            "updated_at":     datetime.now(timezone.utc).isoformat(),
        }},
    )
    return result


def run_scope_keywords_batch():
    """scope_keywords 없는 scope 전체를 배치 루프로 집계합니다."""
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
                            "must_not": {"exists": {"field": "scope_keywords"}},
                        }
                    },
                    "_source": ["scopeID"],
                    "size": BATCH_SIZE,
                },
            )
            hits = res["hits"]["hits"]

            if not hits:
                if batch_num == 1:
                    logger.info("scope 키워드 집계할 대상 없음")
                else:
                    logger.info(f"scope 키워드 집계 전체 완료 | total={total_processed}")
                break

            logger.info(f"[배치 {batch_num}] scope 키워드 집계 시작: {len(hits)}건")

            for hit in hits:
                scope_id = hit["_source"]["scopeID"]
                try:
                    aggregate_scope_keywords(es, scope_id)
                    total_processed += 1
                except Exception as e:
                    logger.error(f"scope 키워드 집계 실패 scopeID={scope_id}: {e}")
                    continue

            es.indices.refresh(index=INDEX_SCOPES)
            logger.info(f"[배치 {batch_num}] 완료 | 누적 processed={total_processed}")

        es.close()

    except Exception as e:
        log_pipeline_error(pipeline="scope_keywords", error=e)
        raise
