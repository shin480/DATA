from apscheduler.schedulers.asyncio import AsyncIOScheduler
from crawler import run_crawling_job


def get_scheduler():
    sch = AsyncIOScheduler(timezone="Asia/Seoul")

    # 매일 00:00 자동 크롤링
    sch.add_job(
        run_crawling_job,
        "cron",
        hour=0,
        minute=0,
        id="daily_news_crawling",
        replace_existing=True,
        max_instances=1
    )

    return sch