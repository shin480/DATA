from util.es import get_es, bulk

# ==============================
# 불용어(Stopwords) 사전 정의
# ==============================
# 분석에 방해되는 무의미한 단어들을 여기에 계속 추가하시면 됩니다.
STOPWORDS = {
     # 기사/보도 일반어
    "관계자", "관계인", "기자", "뉴스", "기사", "기사문", "보도자료",
    "뉴시스", "연합뉴스", "뉴스1", "tv", "그래픽", "게시판",
    "헤드라인", "리포트", "브리핑",

    # 시간/흐름 일반어
    "오늘", "내일", "어제", "올해", "내년", "지난해", "지난달",
    "최근", "하루", "시간", "이날", "이번", "지난", "그동안",
    "당시", "현재", "향후", "다음", "먼저",

    # 의미 약한 일반어
    "경우", "때문", "정도", "우리", "이들", "통해", "대해",
    "위해", "관련", "가운데", "상황", "모두", "일부",
    "수준", "부분", "가능성", "전망", "분위기",

    # 경제/뉴스 범용어
    "정부", "국회", "기업", "업계", "시장", "경제", "금융",
    "경기", "경쟁력", "글로벌", "성장", "투자", "지원",
    "서비스", "브랜드", "데이터", "시스템", "사업", "산업",
    "소비", "고객", "대상자", "구성원", "임직원",

    # 행정/조직 일반어
    "공무원", "대통령", "위원장", "위원회", "감독원", "금감원",
    "거래소", "국토부", "산업부", "중기부", "공정위",

    # 흔한 동사형/상태형 키워드
    "확대", "강화", "추진", "진행", "개최", "발표", "확인",
    "검토", "제기", "밝히", "나오", "따르", "전하", "이어지",
    "올리", "내리", "들어가", "가져오", "기다리"
}
# ==============================
# keywords 문자열("A,B,C") → 개별 키워드 카운트
# ==============================
def extract_keyword_list(keyword_string: str):
    if not keyword_string:
        return []

    cleaned_keywords = []

    for kw in keyword_string.split(","):
        word = kw.strip().lower()

        if not word:
            continue

        if len(word) <= 1:
            continue

        if word in STOPWORDS:
            continue

        # 관계자들, 정부관계자, 기업들 같은 파생형 제거
        if (
                "관계자" in word
                or "기자" in word
                or "정부" in word
                or "업계" in word
                or "기업" in word
                or "위원" in word
                or "감독원" in word
        ):
            continue

        # 숫자만 있거나 숫자 비중이 큰 키워드 제거
        digit_count = sum(ch.isdigit() for ch in word)
        if digit_count >= len(word) / 2:
            continue

        cleaned_keywords.append(word)

    return cleaned_keywords


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
    save_daily_top_issue_report("2026-04-01", "2026-05-12")