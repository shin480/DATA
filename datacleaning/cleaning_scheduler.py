from cleaning import get_preprocessed_data, sync_es_to_db, retokenize_news_economy
import logging
import os
from logging.handlers import RotatingFileHandler
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime


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

def run_main_pipeline():
    logger.info(">>> [작업 시작] 신규 데이터 전처리 파이프라인")
    try:
        # 기존 로직 실행
        result = get_preprocessed_data()
        logger.info(f"전처리 결과: {result.get('message', '처리 완료')}")

        sync_es_to_db()
        logger.info("DB 동기화 완료")

    except Exception as e:
        logger.error(f"🚨 파이프라인 실행 중 오류 발생: {str(e)}", exc_info=True)
    finally:
        logger.info(">>> [작업 종료]")


if __name__ == "__main__":
    # 스케줄러 생성
    # max_instances=1: 동일한 작업이 이미 실행 중이면 다음 작업을 건너뜁니다.
    # coalesce=True: 시스템이 일시적으로 중단되었다가 복구되었을 때,
    #                밀린 작업들을 한 번만 실행합니다.

    job_defaults = {
        'max_instances': 1,
        'coalesce': True
    }

    scheduler = BlockingScheduler(timezone='Asia/Seoul', job_defaults=job_defaults)

    # --- 스케줄 설정 예시 ---

    # 1. 10분마다 전처리 (중복 방지 적용)
    scheduler.add_job(
        run_main_pipeline,
        'interval',
        minutes=10,
        id='news_batch_process',
        misfire_grace_time=60  # 실행 시간이 60초 이상 지연되면 해당 회차는 포기
    )

    # 2. 매일 새벽 3시에 재토큰화
    scheduler.add_job(
        retokenize_news_economy,
        'cron',
        hour=3,
        minute=0,
        id='daily_retokenize'
    )

    logger.info("🚀 스케줄러가 시작되었습니다. (로그 관리 및 중복 방지 활성화)")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("👋 스케줄러가 정상적으로 종료되었습니다.")








