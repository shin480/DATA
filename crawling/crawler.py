import re
import asyncio
from email.utils import parsedate_to_datetime
from datetime import datetime, timedelta

import httpx
import feedparser
from bs4 import BeautifulSoup
from sqlalchemy import text

from util.logger import Logger
from util.db import get_engine
from util.es import get_es, bulk

logger = Logger().get_logger(__name__)
es = get_es()

INDEX_NAME = "test_article_raw"
# INDEX_NAME = "article_raw"
CODE_ID = "C101"

# 운영 모드 : False 테스트 모드 : True
DEBUG_MODE = True

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
    "Accept": "application/rss+xml, application/xml, text/xml, text/html;q=0.9,*/*;q=0.8"
}


def now_iso():
    return datetime.now().isoformat()


def to_iso(dt):
    if dt is None:
        return None

    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)

    return dt.isoformat()


def is_valid_date(pub_date):
    if pub_date is None:
        return False

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


def save_batch_start():
    db = get_engine()

    try:
        sql = text("""
            INSERT INTO batch_jobs (code_id, start_at)
            VALUES (:code_id, NOW())
        """)

        result = db.execute(sql, {"code_id": CODE_ID})
        db.commit()

        return result.lastrowid

    finally:
        db.close()


def finish_batch(job_id, total_count, fail_count):
    db = get_engine()

    try:
        sql = text("""
            UPDATE batch_jobs
            SET end_at = NOW(),
                total_count = :total_count,
                fail_count = :fail_count
            WHERE job_id = :job_id
        """)

        db.execute(sql, {
            "job_id": job_id,
            "total_count": total_count,
            "fail_count": fail_count
        })

        db.commit()

    finally:
        db.close()


def save_process_log(job_id, status, article_id=None):
    """
    article_process_logs.article_id는 article_meta FK라서
    원시 수집 단계에서는 None으로 저장하는 게 안전함.
    """
    db = get_engine()

    try:
        sql = text("""
            INSERT INTO article_process_logs
                (job_id, code_id, article_id, status)
            VALUES
                (:job_id, :code_id, :article_id, :status)
        """)

        result = db.execute(sql, {
            "job_id": job_id,
            "code_id": CODE_ID,
            "article_id": article_id,
            "status": status
        })

        db.commit()

        return result.lastrowid

    finally:
        db.close()


def save_error(history_id, error_code, message):
    db = get_engine()

    try:
        sql = text("""
            INSERT INTO article_error_logs
                (history_id, error_code, error_message)
            VALUES
                (:history_id, :error_code, :error_message)
        """)

        db.execute(sql, {
            "history_id": history_id,
            "error_code": error_code,
            "error_message": message
        })

        db.commit()

    finally:
        db.close()


def save_fail_log(job_id, error_code, message, url=None):
    error_message = f"{message} | url={url}" if url else message

    history_id = save_process_log(
        job_id=job_id,
        status="fail",
        article_id=None
    )

    save_error(
        history_id=history_id,
        error_code=error_code,
        message=error_message
    )


def get_error_code(e):
    if isinstance(e, httpx.TimeoutException):
        return "E001"

    if isinstance(e, httpx.RequestError):
        return "E002"

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


def is_missing_article(news):
    required_fields = [
        "raw_id",
        "article_id",
        "source",
        "press",
        "url",
        "title",
        "raw_text",
        "published_at"
    ]

    for field in required_fields:
        if not news.get(field):
            return True

    return False


def is_duplicate(raw_id):
    try:
        return es.exists(index=INDEX_NAME, id=raw_id)
    except Exception:
        return False


async def get_article_details(client, url: str):
    res = await client.get(url)
    res.raise_for_status()

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
        "raw_html": raw_html,
        "media": media,
        "author": author,
        "published_at": published_at
    }


def save_bulk_to_es(news_list):
    if not news_list:
        logger.warning("ES 저장할 데이터 없음")
        return 0, 0

    actions = []

    for news in news_list:
        actions.append({
            "_index": INDEX_NAME,
            "_id": news["raw_id"],
            "_source": news
        })

    success, errors = bulk(
        es,
        actions,
        raise_on_error=False
    )

    fail_count = len(errors) if errors else 0

    logger.info(f"ES 저장 성공: {success}건")

    if errors:
        logger.error(f"ES 저장 실패 예시: {errors[:3]}")

    return success, fail_count


async def parse_rss_feed(rss_url: str, media_name: str, job_id):
    results = []
    fail_count = 0

    async with httpx.AsyncClient(
        headers=HEADERS,
        timeout=10,
        follow_redirects=True
    ) as client:
        try:
            res = await client.get(rss_url)
            res.raise_for_status()
        except Exception as e:
            save_fail_log(job_id, get_error_code(e), str(e), rss_url)
            return results, 1

        feed = feedparser.parse(res.text)

        for entry in feed.entries:
            link = ""

            try:
                link = entry.link.strip()
                title = entry.title.strip()

                pub_date = parsedate_to_datetime(entry.published)

                # if not is_valid_date(pub_date):
                #     continue

                detail = await get_article_details(client, link)

                if media_name == "연합뉴스":
                    article_id = make_yonhap_id(entry.id)
                    source = "yonhap"
                else:
                    article_id = make_hankyung_id(link)
                    source = "hankyung"

                if not article_id:
                    raise ValueError("article_id 생성 실패")

                if is_duplicate(article_id):
                    continue

                author = get_rss_author(entry)

                news = {
                    "raw_id": article_id,
                    "article_id": article_id,
                    "source": source,
                    "press": detail["media"] if detail["media"] != "Unknown" else media_name,
                    "author": author,
                    "url": link,
                    "title": title,
                    "raw_text": detail["content"],
                    "raw_html": detail["raw_html"],
                    "published_at": to_iso(pub_date),
                    "collected_at": now_iso(),
                    "status": "collected",
                    "error_message": ""
                }

                if is_missing_article(news):
                    raise ValueError("필수값 누락 기사")

                results.append(news)

                save_process_log(
                    job_id=job_id,
                    status="success",
                    article_id=None
                )

            except Exception as e:
                fail_count += 1
                save_fail_log(job_id, get_error_code(e), str(e), link)
                continue

    logger.info(f"{media_name} RSS 수집 완료: {len(results)}건 / 실패 {fail_count}건")
    return results, fail_count


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
    results = []
    fail_count = 0
    empty_page_count = 0

    async with httpx.AsyncClient(
        headers=HEADERS,
        timeout=10,
        follow_redirects=True
    ) as client:

        for page in range(1, pages + 1):
            has_valid_article = False

            url = (
                "https://news.naver.com/main/list.naver"
                f"?mode=LSD&mid=sec&sid1=101&page={page}"
            )

            try:
                res = await client.get(url)
                res.raise_for_status()
            except Exception as e:
                fail_count += 1
                save_fail_log(job_id, get_error_code(e), str(e), url)
                continue

            soup = BeautifulSoup(res.text, "html.parser")
            articles = soup.select(".type06_headline li, .type06 li")

            for ar in articles:
                link = ""

                try:
                    anchor_list = ar.select("dt a")

                    if not anchor_list:
                        continue

                    anchor = anchor_list[-1]
                    title = anchor.get_text(strip=True)
                    link = anchor["href"].split("?")[0]

                    article_id = make_naver_id(link)

                    if not article_id:
                        raise ValueError("article_id 생성 실패")

                    if is_duplicate(article_id):
                        continue

                    detail = await get_article_details(client, link)

                    if not detail["published_at"]:
                        continue

                    # if not is_valid_date(detail["published_at"]):
                    #     continue

                    has_valid_article = True

                    news = {
                        "raw_id": article_id,
                        "article_id": article_id,
                        "source": "naver",
                        "press": detail["media"],
                        "author": detail["author"],
                        "url": link,
                        "title": title,
                        "raw_text": detail["content"],
                        "raw_html": detail["raw_html"],
                        "published_at": to_iso(detail["published_at"]),
                        "collected_at": now_iso(),
                        "status": "collected",
                        "error_message": ""
                    }

                    if is_missing_article(news):
                        raise ValueError("필수값 누락 기사")

                    results.append(news)

                    save_process_log(
                        job_id=job_id,
                        status="success",
                        article_id=None
                    )

                except Exception as e:
                    fail_count += 1
                    save_fail_log(job_id, get_error_code(e), str(e), link)
                    continue

            if has_valid_article:
                empty_page_count = 0
            else:
                empty_page_count += 1
                logger.info(f"네이버 {page}페이지 유효 기사 없음 ({empty_page_count}/3)")

            if empty_page_count >= 3:
                logger.info("네이버 유효 기사 없는 페이지 3회 연속 → 종료")
                break

    logger.info(f"네이버 수집 완료: {len(results)}건 / 실패 {fail_count}건")
    return results, fail_count


async def run_crawling_job():
    job_id = save_batch_start()

    logger.info("운영 크롤링 시작")
    logger.info(f"job_id={job_id}")

    total_fail_count = 0

    results = await asyncio.gather(
        crawl_naver(job_id),
        crawl_yonhap(job_id),
        crawl_hankyung(job_id)
    )

    all_data = []

    for data, fail_count in results:
        all_data.extend(data)
        total_fail_count += fail_count

    logger.info(f"총 수집 대상: {len(all_data)}건")

    success_count, es_fail_count = save_bulk_to_es(all_data)

    total_fail_count += es_fail_count

    finish_batch(
        job_id=job_id,
        total_count=len(all_data),
        fail_count=total_fail_count
    )

    logger.info(
        f"운영 크롤링 종료: 수집 {len(all_data)}건 / ES 성공 {success_count}건 / 실패 {total_fail_count}건"
    )

    return {
        "status": "success",
        "job_id": job_id,
        "total": len(all_data),
        "es_success": success_count,
        "fail_count": total_fail_count
    }


if __name__ == "__main__":
    asyncio.run(run_crawling_job())