from util.es import get_es

def get_search_summary(start_date: str, end_date: str):
    es = get_es()

    try:
        # 공유해주신 매핑에 맞춰 날짜 필드를 'collected_at'으로 지정
        # ES 날짜 인식 오류를 방지하기 위해 공백 대신 'T'를 삽입 (ISO 8601 형식)
        query = {
            "query": {
                "range": {
                    "collected_at": {
                        "gte": f"{start_date}T00:00:00",
                        "lte": f"{end_date}T23:59:59"
                    }
                }
            }
        }

        # 1. 수집된 전체 원본 기사 수 (article_raw)
        raw_count_res = es.count(index="article_raw", body=query)
        collected_count = raw_count_res.get("count", 0)

        # 2. 전처리 및 중복 제거 후 최종 등록된 기사 수 (news_economy)
        economy_count_res = es.count(index="news_economy", body=query)
        processed_count = economy_count_res.get("count", 0)

        # 3. 제거된 기사 수 계산 (원본 수 - 최종 등록 수)
        removed_count = collected_count - processed_count

        return {
            "success": True,
            "summary": {
                "collected": collected_count,
                "processed": processed_count, # 최종 등록 건수도 대시보드에 보여주면 좋습니다.
                "removed": removed_count if removed_count > 0 else 0
            }
        }

    except Exception as e:
        print(f"ES 통계 조회 실패: {e}")
        return {"success": False, "message": str(e)}