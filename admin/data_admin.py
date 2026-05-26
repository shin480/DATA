from util.es import get_es
from datetime import date
from util.db import get_engine
from sqlalchemy import text

def get_search_summary(start_date: str, end_date: str):
    es = get_es()

    try:
        # =========================
        # 공통 날짜 범위 쿼리
        # =========================
        base_query = {
            "range": {
                "collected_at": {
                    "gte": f"{start_date}T00:00:00",
                    "lte": f"{end_date}T23:59:59"
                }
            }
        }

        # =========================
        # 1. article_raw 원본 수집 수
        # =========================
        raw_count_res = es.count(
            index="article_raw",
            body={"query": base_query}
        )

        collected_count = raw_count_res.get("count", 0)

        # =========================
        # 2. news_economy 최종 등록 수
        # ========================
        economy_count_res = es.count(
            index="news_economy",
            body={"query": base_query}
        )

        processed_count = economy_count_res.get("count", 0)

        removed_count = collected_count - processed_count

        # =========================
        # 3. 언론사별 기사 수
        # =========================
        press_query = {
            "size": 0,
            "query": base_query,
            "aggs": {
                "press_stats": {
                    "terms": {
                        "field": "press",
                        "size": 100
                    }
                }
            }
        }

        press_res = es.search(index="news_economy", body=press_query)

        press_stats = [
            {
                "name": bucket["key"],
                "count": bucket["doc_count"]
            }
            for bucket in press_res["aggregations"]["press_stats"]["buckets"]
        ]

        # =========================
        # 4. 스코프별 기사 수
        # =========================
        scope_query = {
            "size": 0,
            "query": base_query,
            "aggs": {
                "scope_stats": {
                    "terms": {
                        "field": "scopeID",
                        "size": 100
                    }
                }
            }
        }

        scope_res = es.search(index="news_economy", body=scope_query)

        scope_stats = [
            {
                "title": bucket["key"],
                "count": bucket["doc_count"]
            }
            for bucket in scope_res["aggregations"]["scope_stats"]["buckets"]
        ]

        # =========================
        # 5. 긍정/중립/부정 수집 비율
        # 1차 분류가 완료된 news_economy 기준
        # =========================
        sentiment_query = {
            "size": 0,
            "query": {
                "bool": {
                    "must": [
                        base_query,
                        {
                            "exists": {
                                "field": "sentiment"
                            }
                        }
                    ]
                }
            },
            "aggs": {
                "sentiment_stats": {
                    "terms": {
                        "field": "sentiment",
                        "size": 10
                    }
                }
            }
        }

        sentiment_res = es.search(
            index="news_economy",
            body=sentiment_query
        )

        sentiment_ratio = {
            "positive": 0,
            "neutral": 0,
            "negative": 0
        }

        for bucket in sentiment_res["aggregations"]["sentiment_stats"]["buckets"]:
            key = bucket["key"]

            if key in sentiment_ratio:
                sentiment_ratio[key] = bucket["doc_count"]

        # =========================
        # 반환
        # =========================
        return {
            "success": True,
            "summary": {
                "collected": collected_count,
                "processed": processed_count,
                "removed": removed_count if removed_count > 0 else 0,
                "press_stats": press_stats,
                "scope_stats": scope_stats,
                "sentiment_ratio": sentiment_ratio
            }
        }

    except Exception as e:
        print(f"ES 통계 조회 실패: {e}")

        return {
            "success": False,
            "message": str(e)
        }