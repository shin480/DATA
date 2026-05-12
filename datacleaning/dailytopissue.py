from util.es import get_es, bulk


# ==============================
# keywords 문자열("A,B,C") → 개별 키워드 카운트
# ==============================
def extract_keyword_list(keyword_string: str):
    if not keyword_string:
        return []

    return [
        kw.strip()
        for kw in keyword_string.split(",")
        if kw.strip()
    ]


# ==============================
# 특정 날짜 기사 전체에서 키워드 빈도 계산
# ==============================
def get_top_keyword_for_day(es, date_str: str):
    query = {
        "_source": ["keywords"],
        "size": 10000,
        "query": {
            "range": {
                "published_at": {
                    "gte": f"{date_str}T00:00:00",
                    "lte": f"{date_str}T23:59:59",
                    "time_zone": "+09:00"
                }
            }
        }
    }

    res = es.search(index="news_economy", body=query)

    keyword_counter = {}

    for hit in res["hits"]["hits"]:
        raw_keywords = hit["_source"].get("keywords", "")

        for keyword in extract_keyword_list(raw_keywords):
            keyword_counter[keyword] = keyword_counter.get(keyword, 0) + 1

    if not keyword_counter:
        return None, 0

    top_keyword = max(keyword_counter, key=keyword_counter.get)
    print(f"[DEBUG] {date_str} | keyword_counter={keyword_counter}")

    return top_keyword, keyword_counter[top_keyword]


# ==============================
# TOP 키워드 포함 기사 감성분포
# ==============================
def get_sentiment_distribution_for_keyword(es, date_str: str, keyword: str):
    query = {
        "_source": ["sentiment", "keywords"],
        "size": 10000,
        "query": {
            "range": {
                "published_at": {
                    "gte": f"{date_str}T00:00:00",
                    "lte": f"{date_str}T23:59:59",
                    "time_zone": "+09:00"
                }
            }
        }
    }

    res = es.search(index="news_economy", body=query)

    positive = 0
    negative = 0
    neutral = 0

    for hit in res["hits"]["hits"]:
        source = hit["_source"]

        keyword_list = extract_keyword_list(source.get("keywords", ""))

        if keyword not in keyword_list:
            continue

        sentiment = source.get("sentiment", "").lower()

        if sentiment == "positive":
            positive += 1
        elif sentiment == "negative":
            negative += 1
        else:
            neutral += 1

    total = positive + negative + neutral


    print(
        f"[감성집계] keyword={keyword} | "
        f"positive={positive}, negative={negative}, neutral={neutral}, total={total}"
    )

    return {
        "positive_ratio": positive / total if total else 0.0,
        "negative_ratio": negative / total if total else 0.0,
        "neutral_ratio": neutral / total if total else 0.0
    }


# ==============================
# 날짜 범위 전체 일간 TOP 이슈 생성
# ==============================
def get_daily_top_keywords(start_date: str, end_date: str):
    es = get_es()

    count_res = es.count(
        index="news_economy",
        body={
            "query": {
                "range": {
                    "published_at": {
                        "gte": f"{start_date}T00:00:00",
                        "lte": f"{end_date}T23:59:59",
                        "time_zone": "+09:00"
                    }
                }
            }
        }
    )

    print(f"[DEBUG] 날짜 범위 기사 수: {count_res['count']}")

    date_query = {
        "size": 0,
        "query": {
            "range": {
                "published_at": {
                    "gte": f"{start_date}T00:00:00",
                    "lte": f"{end_date}T23:59:59",
                    "time_zone": "+09:00"
                }
            }
        },
        "aggs": {
            "daily_trends": {
                "date_histogram": {
                    "field": "published_at",
                    "calendar_interval": "day",
                    "format": "yyyy-MM-dd",
                    "time_zone": "+09:00",
                    "min_doc_count": 1
                }
            }
        }
    }

    res = es.search(index="news_economy", body=date_query)

    report_data = []

    for bucket in res["aggregations"]["daily_trends"]["buckets"]:
        date_str = bucket["key_as_string"]

        top_keyword, total_mentions = get_top_keyword_for_day(es, date_str)

        if not top_keyword:
            continue

        sentiment_distribution = get_sentiment_distribution_for_keyword(
            es,
            date_str,
            top_keyword
        )

        report_data.append({
            "date": date_str,
            "top_keyword": top_keyword,
            "total_mentions": total_mentions,
            "sentiment_distribution": sentiment_distribution
        })

        print(f"[완료] {date_str} | TOP={top_keyword} | 언급량={total_mentions}")

    return report_data


# ==============================
# ES 저장
# ==============================
def save_daily_top_issue_report(start_date: str, end_date: str):
    es = get_es()

    reports = get_daily_top_keywords(start_date, end_date)

    if not reports:
        print("저장할 일간 TOP 이슈 데이터가 없습니다.")
        return {
            "success": False,
            "count": 0
        }

    actions = []

    for report in reports:
        actions.append({
            "_op_type": "index",
            "_index": "daily_top_issue_report",
            "_id": report["date"],
            "_source": report
        })

    success, failed = bulk(es, actions, raise_on_error=False)

    es.indices.refresh(index="daily_top_issue_report")

    print(f">>> daily_top_issue_report 저장 완료: {success}건")

    if failed:
        print(f">>> 저장 실패: {len(failed)}건")
        print(failed[0])

    return {
        "success": True,
        "count": success,
        "failed": len(failed) if failed else 0
    }


if __name__ == "__main__":
    save_daily_top_issue_report("2026-05-01", "2026-05-12")