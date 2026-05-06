"""
뉴스 분류 파이프라인 FastAPI 앱

스케줄:
  - 1일 1회 (새벽 2시): scopeID 클러스터링 (분류 파이프라인)
  - 1일 1회 (새벽 2시 10분): 감성 분류
  - 1일 1회 (새벽 2시 20분): 키워드 추출
  - 1일 1회 (새벽 2시 30분): 뉴스 요약
  - 1일 1회 (새벽 3시): scope 감성 집계
  - 1일 1회 (새벽 3시 10분): scope 대표 요약 생성/갱신
  - 30분 간격: scope 키워드 집계
  - 30분 간격: scopeTitle 큐 배치 처리
"""

import logging
import os

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MISSED
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI

from model.models import init_db
from model.routers.admin import router as admin_router
from model.routers.classify import router as classify_router
from model.routers.sentiment_eval import router as eval_router
from model.routers.correction import router as correction_router
from model.services.classifier import run_classification_pipeline
from model.services.scope_title import run_scope_title_batch
from model.services.scope_summarizer import run_scope_summary_batch
from model.services.scope_sentiment import run_scope_sentiment_batch
from model.services.sentiment import run_sentiment_pipeline
from model.services.summarizer import run_summary_pipeline
from model.services.keyword_extractor import run_keyword_pipeline
from model.services.scope_keywords import run_scope_keywords_batch
from model.services.finetuner import check_and_trigger_finetune

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── FastAPI 앱 ─────────────────────────────────────────
app = FastAPI(
    title="뉴스 분류 파이프라인 API",
    description="경제 뉴스 scopeID 클러스터링 + KoELECTRA 감성 분류",
    version="1.0.0",
)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(classify_router)
app.include_router(admin_router)
app.include_router(eval_router)
app.include_router(correction_router)


# ── APScheduler ────────────────────────────────────────
scheduler = BackgroundScheduler(timezone="Asia/Seoul")


def _on_job_error(event):
    """
    APScheduler 잡 실행 중 예외 발생 시 호출됩니다.
    services/error_logger.py를 통해 pipeline_error_log 테이블에 기록됩니다.
    """
    from model.services.error_logger import log_pipeline_error
    job_id = event.job_id
    error  = event.exception
    logger.critical(f"[APScheduler] 잡 실패: job_id={job_id} | error={error}")
    log_pipeline_error(pipeline=job_id, error=error)


def _on_job_missed(event):
    """
    APScheduler 잡 실행 시간을 놓쳤을 때 호출됩니다.
    (서버 과부하, 슬립 등으로 실행 시각을 지나친 경우)
    """
    logger.warning(f"[APScheduler] 잡 실행 누락: job_id={event.job_id} | scheduled={event.scheduled_run_time}")


@app.on_event("startup")
def startup():
    # DB/ES 초기화 (없으면 생성)
    # MySQL pipeline_error_log 테이블 + ES 인덱스 생성
    # 최초 실행 시에만 필요하므로 오류가 나도 서버는 계속 실행
    try:
        init_db()
    except Exception as e:
        logger.warning(f"init_db 실패 (이미 존재하거나 연결 불가): {e}")

    # ── 1일 1회 배치 (순차 실행, 앞 작업 결과를 뒤 작업이 소비) ──
    scheduler.add_job(
        run_classification_pipeline,
        trigger="cron", hour=2, minute=0,
        id="classification",
        replace_existing=True,
    )
    scheduler.add_job(
        run_sentiment_pipeline,
        trigger="cron", hour=2, minute=10,
        id="sentiment",
        replace_existing=True,
    )
    scheduler.add_job(
        run_keyword_pipeline,
        trigger="cron", hour=2, minute=20,
        id="keywords",
        replace_existing=True,
    )
    scheduler.add_job(
        run_summary_pipeline,
        trigger="cron", hour=2, minute=30,
        id="summary",
        replace_existing=True,
    )
    scheduler.add_job(
        run_scope_sentiment_batch,
        trigger="cron", hour=3, minute=0,
        id="scope_sentiment",
        replace_existing=True,
    )
    scheduler.add_job(
        run_scope_summary_batch,
        trigger="cron", hour=3, minute=10,
        id="scope_summary_batch",
        replace_existing=True,
    )

    # ── 30분 간격 배치 ─────────────────────────────────
    scheduler.add_job(
        run_scope_keywords_batch,
        trigger="interval", minutes=30,
        id="scope_keywords",
        replace_existing=True,
    )
    scheduler.add_job(
        run_scope_title_batch,
        trigger="interval", minutes=30,
        id="scope_title_batch",
        replace_existing=True,
    )

    # ── 1시간 간격: fine-tuning 자동 트리거 확인 ───────
    scheduler.add_job(
        check_and_trigger_finetune,
        trigger="interval", hours=1,
        id="finetune_checker",
        replace_existing=True,
    )

    scheduler.add_listener(_on_job_error,  EVENT_JOB_ERROR)
    scheduler.add_listener(_on_job_missed, EVENT_JOB_MISSED)
    scheduler.start()
    logger.info("✅ 스케줄러 시작 완료")


@app.on_event("shutdown")
def shutdown():
    scheduler.shutdown()
    logger.info("스케줄러 종료")


# ── 헬스 체크 ──────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/pipeline")
def root():
    return {
        "name":    "뉴스 분류 파이프라인",
        "version": "1.0.0",
        "docs":    "/docs",
    }
