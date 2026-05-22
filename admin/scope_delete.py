from util.es import get_es, NEWS_ECONOMY_INDEX
from elasticsearch.helpers import scan

es = get_es()

# 1. news_economy에서 실제 살아있는 scopeID 수집
scope_count_map = {}

result = es.search(
    index=NEWS_ECONOMY_INDEX,
    body={
        "size": 0,
        "query": {
            "bool": {
                "must": [
                    {"exists": {"field": "scopeID"}}
                ],
                "must_not": [
                    {"term": {"is_disabled": True}}
                ]
            }
        },
        "aggs": {
            "scope_counts": {
                "terms": {
                    "field": "scopeID",
                    "size": 10000
                }
            }
        }
    }
)

for bucket in result["aggregations"]["scope_counts"]["buckets"]:
    scope_count_map[bucket["key"]] = bucket["doc_count"]


# 2. news_scopes 전체 돌면서 실제 기사 0건인 스콥 찾기
delete_ids = []

for doc in scan(
    es,
    index="news_scopes",
    query={
        "_source": ["scopeID", "scopeTitle", "news_count"],
        "query": {"match_all": {}}
    }
):
    source = doc["_source"]
    scope_id = source.get("scopeID") or doc["_id"]

    real_count = scope_count_map.get(scope_id, 0)

    if real_count == 0:
        delete_ids.append(doc["_id"])


print("실제 기사 0건 스콥 수:", len(delete_ids))
print(delete_ids[:20])

print("실제 기사 0건 스콥 수:", len(delete_ids))
print(delete_ids[:20])

DRY_RUN = False

if DRY_RUN:
    print("DRY_RUN 상태라 삭제 안 함")
else:
    from elasticsearch.helpers import bulk

    actions = []

    for scope_doc_id in delete_ids:
        actions.append({
            "_op_type": "delete",
            "_index": "news_scopes",
            "_id": scope_doc_id
        })

    if actions:
        bulk(es, actions)

    print("삭제 완료:", len(actions))