from cleaning import get_preprocessed_data
import logging
import os
from logging.handlers import RotatingFileHandler
from apscheduler.schedulers.blocking import BlockingScheduler


def setup_logger():
    # log 폴더 생성
    if not os.path.exists('logs'):
        os.makedirs('logs')

    logger = logging.getLogger("NewsPipeline")
    logger.setLevel(logging.INFO)

    # 1. 콘솔 출력 핸들러
    stream_handler = logging.StreamHandler()

    # 2. 파일 출력 핸들러 (파일당 10MB, 최대 5개까지 유지 후 덮어쓰기)
    file_handler = RotatingFileHandler(
        'logs/pipeline.log', maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8'
    )

    # 로그 포맷 설정
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


logger = setup_logger()


def get_preprocess_scheduler():
    # 스케줄러 생성
    # max_instances=1: 동일한 작업이 이미 실행 중이면 다음 작업을 건너뜁니다.
    # coalesce=True: 시스템이 일시적으로 중단되었다가 복구되었을 때,
    #                밀린 작업들을 한 번만 실행합니다.

    job_defaults = {
        'max_instances': 1,
        'coalesce': True
    }

    sch = BlockingScheduler(timezone='Asia/Seoul', job_defaults=job_defaults)

    # --- 스케줄 설정 예시 ---

    # 매일 새벽 12시 30분에 전처리
    sch.add_job(
        get_preprocessed_data,
        'cron',
        hour=0,
        minute=30,
        id='daily_preprocess'
    )

    logger.info("🚀 스케줄러가 시작되었습니다. (로그 관리 및 중복 방지 활성화)")

    try:
        sch.start()
        return sch
    except (KeyboardInterrupt, SystemExit):
        logger.info("👋 스케줄러가 정상적으로 종료되었습니다.")

if __name__ == "__main__":
    get_preprocess_scheduler()








