import json
import time
import requests
from bs4 import BeautifulSoup

from util.es import get_es, NEWS_ECONOMY_INDEX


# =========================================================
# 설정
# =========================================================
OUTPUT_FILE = "view/press_logo_map.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/148.0.0.0 Safari/537.36"
    )
}


# =========================================================
# 1. ES에서 언론사별 대표 기사 URL 1개씩 가져오기
# =========================================================
def get_press_sample_urls():
    es = get_es()

    body = {
        "size": 0,
        "aggs": {
            "press_groups": {
                "terms": {
                    "field": "press",
                    "size": 300
                },
                "aggs": {
                    "sample_article": {
                        "top_hits": {
                            "size": 1,
                            "_source": ["press", "url", "title"],
                            "sort": [
                                {
                                    "published_at": {
                                        "order": "desc"
                                    }
                                }
                            ]
                        }
                    }
                }
            }
        }
    }

    result = es.search(
        index=NEWS_ECONOMY_INDEX,
        body=body
    )

    press_samples = []

    buckets = result["aggregations"]["press_groups"]["buckets"]

    for bucket in buckets:
        hits = bucket["sample_article"]["hits"]["hits"]

        if not hits:
            continue

        source = hits[0]["_source"]

        press = source.get("press")
        url = source.get("url")
        title = source.get("title", "")

        if press and url:
            press_samples.append({
                "press": press,
                "url": url,
                "title": title
            })

    return press_samples


# =========================================================
# 2. 기사 페이지에서 언론사 로고 URL 추출
# =========================================================
def extract_press_logo_url(article_url: str):
    try:
        response = requests.get(
            article_url,
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # 우리가 크롤링하는 네이버 기사 페이지의 언론사 로고
        logo_img = soup.select_one("img.media_end_head_top_img")

        if not logo_img:
            return None

        logo_url = (
            logo_img.get("src")
            or logo_img.get("data-src")
        )

        if not logo_url:
            return None

        return logo_url

    except Exception as e:
        print(f"[로고 추출 실패] {article_url}")
        print(f"  └ 사유: {e}")
        return None


# =========================================================
# 3. press_logo_map.json 생성
# =========================================================
def make_press_logo_map():
    print("1. ES에서 언론사별 대표 기사 가져오는 중...")
    press_samples = get_press_sample_urls()

    print(f"   → 총 {len(press_samples)}개 언론사 확인")
    print()

    press_logo_map = {}
    failed_list = []

    print("2. 각 언론사 로고 URL 추출 중...")

    for idx, item in enumerate(press_samples, start=1):
        press = item["press"]
        url = item["url"]

        print(f"[{idx}/{len(press_samples)}] {press}")

        logo_url = extract_press_logo_url(url)

        if logo_url:
            press_logo_map[press] = logo_url
            print(f"  └ 성공: {logo_url}")
        else:
            failed_list.append({
                "press": press,
                "url": url
            })
            print("  └ 실패")

        # 너무 빠르게 요청하지 않도록 살짝 대기
        time.sleep(0.3)

    print()
    print("3. JSON 파일 저장 중...")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            press_logo_map,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"   → 저장 완료: {OUTPUT_FILE}")
    print(f"   → 성공: {len(press_logo_map)}개")
    print(f"   → 실패: {len(failed_list)}개")

    if failed_list:
        print()
        print("===== 로고 추출 실패 언론사 =====")
        for item in failed_list:
            print(f"- {item['press']}")
            print(f"  {item['url']}")


# =========================================================
# 실행
# =========================================================
if __name__ == "__main__":
    make_press_logo_map()