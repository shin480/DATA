import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

from apscheduler.schedulers.blocking import BlockingScheduler

from cleaning import get_preprocessed_data


def setup_logger():
    Path("logs").mkdir(exist_ok=True)

    logger = logging.getLogger("NewsPipeline")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # setup_logger가 여러 번 호출될 때 핸들러 중복 추가 방지
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    file_handler = RotatingFileHandler(
        "logs/pipeline.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


logger = setup_logger()


def run_preprocess_job():
    logger.info("전처리 작업을 시작합니다.")

    try:
        get_preprocessed_data()
        logger.info("전처리 작업이 완료되었습니다.")
    except Exception:
        logger.exception("전처리 작업 중 예외가 발생했습니다.")
        raise


def create_preprocess_scheduler():
    job_defaults = {
        "max_instances": 1,
        "coalesce": True,
        "misfire_grace_time": 60 * 60,
    }

    scheduler = BlockingScheduler(
        timezone="Asia/Seoul",
        job_defaults=job_defaults,
    )

    scheduler.add_job(
        run_preprocess_job,
        "cron",
        hour=0,
        minute=30,
        id="daily_preprocess",
        replace_existing=True,
    )

    return scheduler


def run_preprocess_scheduler():
    scheduler = create_preprocess_scheduler()

    logger.info("스케줄러가 시작되었습니다.")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("스케줄러 종료 신호를 받았습니다.")
    finally:
        if scheduler.running:
            scheduler.shutdown(wait=False)

        logger.info("스케줄러가 종료되었습니다.")


if __name__ == "__main__":
    run_preprocess_scheduler()