import re
from email.utils import parsedate_to_datetime
from datetime import datetime, timedelta

import httpx
import asyncio
import feedparser
import pymysql
from bs4 import BeautifulSoup
from elasticsearch import Elasticsearch, helpers

from util.logger import Logger

logger = Logger().get_logger(__name__)
es = Elasticsearch("http://192.168.0.23:9200/")

INDEX_NAME = "article_raw"
# True = 테스트 모드 / False = 운영 모드
DEBUG_MODE = True


def now_iso():
    return datetime.now().isoformat()


def to_iso(dt):
    if dt is None:
        return None

    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)

    return dt.isoformat()


def get_db():
    conn = pymysql.connect(
        host="192.168.0.23",
        port=3306,
        user="web_user",
        password="pass",
        database="data_platform",
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor
    )
    return conn


def save_log(code_id):
    conn = get_db()
    cursor = conn.cursor()

    sql = """
    INSERT INTO batch_jobs (code_id, start_at)
    VALUES (%s, NOW())
    """

    cursor.execute(sql, (code_id,))
    conn.commit()

    job_id = cursor.lastrowid
    conn.close()

    return job_id


def finish_log(job_id, total_count, fail_count):
    conn = get_db()
    cursor = conn.cursor()

    sql = """
    UPDATE batch_jobs
    SET end_at = NOW(),
        total_count = %s,
        fail_count = %s
    WHERE job_id = %s
    """

    cursor.execute(sql, (total_count, fail_count, job_id))
    conn.commit()
    conn.close()


def save_error(job_id, error_code, message, url):
    conn = get_db()
    cursor = conn.cursor()

    error_message = f"{message} | url={url}" if url else message

    sql = """
    INSERT INTO article_error_logs (error_code, error_message)
    VALUES (%s, %s)
    """

    cursor.execute(sql, (error_code, error_message))
    conn.commit()
    conn.close()


def get_error_code(e):
    if isinstance(e, httpx.TimeoutException):
        return "E001"
    elif isinstance(e, httpx.RequestError):
        return "E002"
    else:
        return "E999"


def make_naver_id(url):
    match = re.search(r"/article/(\d+)/(\d+)", url)
    if match:
        return f"NAVER_{match.group(1)}_{match.group(2)}"
    return None


def make_yonhap_id(guid):
    match = re.search(r"/view/([A-Z0-9]+)", guid)
    if match:
        return f"YNA_{match.group(1)}"
    return None


def make_hankyung_id(url):
    match = re.search(r"hankyung\.com/article/([^/?#\s]+)", url)
    if match:
        return f"HK_{match.group(1)}"
    return None


def get_rss_author(entry):
    if entry.get("dc_creator"):
        return entry.get("dc_creator", "").strip()

    if entry.get("author"):
        return entry.get("author", "").strip()

    if entry.get("authors"):
        authors = entry.get("authors")
        if isinstance(authors, list) and len(authors) > 0:
            return authors[0].get("name", "").strip()

    return ""


def init_es_index():
    if not es.indices.exists(index=INDEX_NAME):
        es.indices.create(
            index=INDEX_NAME,
            body={
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 1
                },
                "mappings": {
                    "properties": {
                        "raw_id": {"type": "keyword"},
                        "article_id": {"type": "keyword"},
                        "source": {"type": "keyword"},
                        "press": {"type": "keyword"},
                        "author": {"type": "keyword"},
                        "url": {"type": "keyword"},
                        "title": {"type": "text"},
                        "raw_text": {
                            "type": "text",
                            "index": False
                        },
                        "published_at": {"type": "date"},
                        "collected_at": {"type": "date"},
                        "status": {"type": "keyword"},
                        "error_message": {
                            "type": "text",
                            "index": False
                        }
                    }
                }
            }
        )


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/rss+xml, application/xml, text/xml, text/html;q=0.9,*/*;q=0.8"
}


async def get_article_details(url: str):
    try:
        async with httpx.AsyncClient(
            headers=HEADERS,
            timeout=10,
            follow_redirects=True
        ) as client:
            res = await client.get(url)

        raw_html = res.text
        soup = BeautifulSoup(raw_html, "html.parser")

        content_area = (
            soup.select_one("#dic_area")
            or soup.select_one("#articleBodyContents")
            or soup.select_one(".story-news.article")
            or soup.select_one("#articletxt")
            or soup.select_one(".article-body")
        )
        content = content_area.get_text(" ", strip=True) if content_area else ""

        media = "Unknown"

        el = soup.select_one("span.media_end_head_top_press")
        if el:
            media = el.get_text(strip=True)

        if media == "Unknown":
            img = soup.select_one(".media_end_head_top_logo img")
            if img and img.has_attr("alt"):
                media = img["alt"]

        author = ""
        author_el = (
            soup.select_one(".media_end_head_journalist_name")
            or soup.select_one(".byline_s")
        )

        if author_el:
            author = author_el.get_text(strip=True)
            author = re.sub(r"\(.*?\)", "", author).strip()

        date_el = soup.select_one("span.media_end_head_info_datestamp_time")
        published_at = None

        if date_el and date_el.has_attr("data-date-time"):
            published_at = datetime.strptime(
                date_el["data-date-time"],
                "%Y-%m-%d %H:%M:%S"
            )

        return {
            "content": content,
            "media": media,
            "author": author,
            "published_at": published_at
        }

    except Exception:
        return {
            "content": "",
            "media": "error",
            "author": "",
            "published_at": None
        }


def save_bulk_to_es(news_list):
    actions = []

    for news in news_list:
        actions.append({
            "_index": INDEX_NAME,
            "_id": news["raw_id"],
            "_source": {
                "raw_id": news["raw_id"],
                "article_id": news["article_id"],
                "source": news["source"],
                "press": news["press"],
                "author": news["author"],
                "url": news["url"],
                "title": news["title"],
                "raw_text": news["raw_text"],
                "published_at": news["published_at"],
                "collected_at": news["collected_at"],
                "status": news["status"],
                "error_message": news["error_message"]
            }
        })

    success, errors = helpers.bulk(
        es,
        actions,
        raise_on_error=False
    )

    logger.info(f"ES 저장 완료: {success}건")

    if errors:
        logger.error(f"ES 저장 실패 예시: {errors[:3]}")


def is_valid_date(pub_date):
    if pub_date.tzinfo is not None:
        pub_date = pub_date.replace(tzinfo=None)

    now = datetime.now()

    start = (now - timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    end = now.replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    return start <= pub_date < end


async def parse_rss_feed(rss_url: str, media_name: str, job_id):
    results = []

    async with httpx.AsyncClient(
        headers=HEADERS,
        timeout=10,
        follow_redirects=True
    ) as client:
        res = await client.get(rss_url)

    if res.status_code != 200:
        logger.error(f"{media_name} RSS 요청 실패: status={res.status_code}")
        return results

    feed = feedparser.parse(res.text)

    for entry in feed.entries:
        link = ""

        try:
            link = entry.link.strip()
            title = entry.title.strip()

            pub_date = parsedate_to_datetime(entry.published)

            if not DEBUG_MODE:
                if not is_valid_date(pub_date):
                    continue

            detail = await get_article_details(link)

            if media_name == "연합뉴스":
                article_id = make_yonhap_id(entry.id)
                source = "yonhap"
            else:
                article_id = make_hankyung_id(link)
                source = "hankyung"

            if not article_id:
                logger.warning(f"{media_name} article_id 생성 실패: {link}")
                continue

            author = get_rss_author(entry)

            results.append({
                "raw_id": article_id,
                "article_id": article_id,
                "source": source,
                "press": detail["media"] if detail["media"] not in ["error", "Unknown"] else media_name,
                "author": author,
                "url": link,
                "title": title,
                "raw_text": detail["content"],
                "published_at": to_iso(pub_date),
                "collected_at": now_iso(),
                "status": "collected",
                "error_message": ""
            })

        except Exception as e:
            save_error(job_id, get_error_code(e), str(e), link)
            continue

    logger.info(f"{media_name} RSS 수집 완료: {len(results)}건")
    return results


async def crawl_yonhap(job_id):
    return await parse_rss_feed(
        "https://www.yna.co.kr/rss/economy.xml",
        "연합뉴스",
        job_id
    )


async def crawl_hankyung(job_id):
    return await parse_rss_feed(
        "https://www.hankyung.com/feed/economy",
        "한국경제",
        job_id
    )


async def crawl_naver(job_id, pages=50):
    if DEBUG_MODE:
        pages = 50

    results = []
    empty_page_count = 0

    async with httpx.AsyncClient(
        headers=HEADERS,
        follow_redirects=True
    ) as client:
        for page in range(1, pages + 1):
            has_valid_article = False

            url = (
                "https://news.naver.com/main/list.naver"
                f"?mode=LSD&mid=sec&sid1=101&page={page}"
            )

            res = await client.get(url)
            soup = BeautifulSoup(res.text, "html.parser")

            articles = soup.select(".type06_headline li, .type06 li")

            for ar in articles:
                link = ""

                try:
                    anchor = ar.select("dt a")[-1]
                    title = anchor.get_text(strip=True)
                    link = anchor["href"].split("?")[0]

                    detail = await get_article_details(link)

                    if not DEBUG_MODE:
                        if not detail["published_at"]:
                            continue

                        if not is_valid_date(detail["published_at"]):
                            continue

                        has_valid_article = True
                    else:
                        has_valid_article = True

                    article_id = make_naver_id(link)

                    if not article_id:
                        continue

                    results.append({
                        "raw_id": article_id,
                        "article_id": article_id,
                        "source": "naver",
                        "press": detail["media"],
                        "author": detail["author"],
                        "url": link,
                        "title": title,
                        "raw_text": detail["content"],
                        "published_at": to_iso(detail["published_at"]),
                        "collected_at": now_iso(),
                        "status": "collected",
                        "error_message": ""
                    })

                except Exception as e:
                    save_error(job_id, get_error_code(e), str(e), link)
                    continue

            if not DEBUG_MODE:
                if has_valid_article:
                    empty_page_count = 0
                else:
                    empty_page_count += 1
                    logger.info(f"네이버 {page}페이지에 유효 기사 없음 ({empty_page_count}/3)")

                if empty_page_count >= 3:
                    logger.info(f"네이버 유효 기사 없는 페이지 3회 연속 → 종료")
                    break

    logger.info(f"네이버 수집 완료: {len(results)}건")
    return results


async def run_crawling_job():
    fail_count = 0

    job_id = save_log("C101")

    logger.info("크롤링 시작")
    logger.info(f"DEBUG_MODE = {DEBUG_MODE}")

    results = await asyncio.gather(
        crawl_naver(job_id),
        crawl_yonhap(job_id),
        crawl_hankyung(job_id)
    )

    all_data = []

    for r in results:
        all_data.extend(r)

    logger.info(f"총 수집: {len(all_data)}건")

    save_bulk_to_es(all_data)

    finish_log(job_id, len(all_data), fail_count)

    return {
        "status": "success",
        "job_id": job_id,
        "total": len(all_data),
        "data": all_data[:10]
    }