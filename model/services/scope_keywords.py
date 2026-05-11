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
    """scope_keywords 집계 배치 처리 (30분 주기)"""
    try:
        es = get_es()

        # scope_keywords가 없거나 news_count가 있는 scope 조회
        res = es.search(
            index=INDEX_SCOPES,
            body={
                "query": {
                    "bool": {
                        "should": [
                            {"bool": {"must_not": {"exists": {"field": "scope_keywords"}}}},
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
            logger.info("scope 키워드 집계할 대상 없음")
            es.close()
            return

        logger.info(f"scope 키워드 집계 시작: {len(hits)}건")

        for hit in hits:
            scope_id = hit["_source"]["scopeID"]
            try:
                aggregate_scope_keywords(es, scope_id)
            except Exception as e:
                logger.error(f"scope 키워드 집계 실패 scopeID={scope_id}: {e}")
                continue

        es.close()
        logger.info(f"scope 키워드 집계 완료: {len(hits)}건")

    except Exception as e:
        log_pipeline_error(pipeline="scope_keywords", error=e)
        raise
