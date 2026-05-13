"""
뉴스 분류 파이프라인 FastAPI 앱

스케줄 (체이닝 방식 — 선행 작업 완료 후 다음 작업 즉시 실행):
  [일 1회 체인, 새벽 2시 시작]
    classification → sentiment → keywords → summary
      → scope_title → scope_sentiment → scope_summary → scope_keywords

  [독립 배치]
  - 없음 (모든 배치는 일배치 체인으로 실행, 수동 트리거 가능)

※ 각 작업은 선행 작업이 정상 완료된 경우에만 다음 작업을 예약합니다.
   실패 또는 타임아웃 시 체인이 중단되고 pipeline_error_log에 기록됩니다.

타임아웃 기본값: 각 job당 30분 (JOB_TIMEOUT_SEC)
  → 무한루프 등으로 초과 시 TimeoutError 발생, 체인 중단
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

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
from model.routers.stopwords import router as stopwords_router
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

# ── 타임아웃 설정 ──────────────────────────────────────
# 각 job의 최대 허용 실행 시간 (초). 초과 시 TimeoutError → 체인 중단.
# 운영: 1800 (30분) / 테스트: 300 (5분) 등으로 조정
JOB_TIMEOUT_SEC = 1800


def _run_with_timeout(fn, job_id: str, timeout_sec: int = JOB_TIMEOUT_SEC):
    """
    fn을 별도 스레드에서 실행하고 timeout_sec 내에 완료되지 않으면
    FuturesTimeoutError를 raise합니다.
    - TimeoutError: 무한루프 등 초과 시 → 체인 중단, _on_job_error 기록
    - 일반 Exception: 서비스 내부 오류 → 동일하게 체인 중단
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        try:
            future.result(timeout=timeout_sec)
        except FuturesTimeoutError:
            logger.critical(
                f"[타임아웃] {job_id} — {timeout_sec}초 초과. 체인 중단."
            )
            raise


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
app.include_router(stopwords_router)


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


# ── 체이닝 래퍼 함수 ───────────────────────────────────
# 각 함수는 _run_with_timeout으로 실행 → 정상 완료 시 다음 job 예약.
# 타임아웃 또는 예외 발생 시 체인 중단, _on_job_error + pipeline_error_log 기록.

def classification_job():
    """[1/8] 분류 파이프라인 → 완료 시 sentiment_job 예약"""
    logger.info("[체인 1/8] classification 시작")
    _run_with_timeout(run_classification_pipeline, "classification")
    logger.info("[체인 1/8] classification 완료 → sentiment 예약")
    scheduler.add_job(
        sentiment_job,
        trigger="date",
        id="sentiment",
        replace_existing=True,
    )


def sentiment_job():
    """[2/8] 감성 분류 → 완료 시 keyword_job 예약"""
    logger.info("[체인 2/8] sentiment 시작")
    _run_with_timeout(run_sentiment_pipeline, "sentiment")
    logger.info("[체인 2/8] sentiment 완료 → keywords 예약")
    scheduler.add_job(
        keyword_job,
        trigger="date",
        id="keywords",
        replace_existing=True,
    )


def keyword_job():
    """[3/8] 키워드 추출 → 완료 시 summary_job 예약"""
    logger.info("[체인 3/8] keywords 시작")
    _run_with_timeout(run_keyword_pipeline, "keywords")
    logger.info("[체인 3/8] keywords 완료 → summary 예약")
    scheduler.add_job(
        summary_job,
        trigger="date",
        id="summary",
        replace_existing=True,
    )


def summary_job():
    """[4/8] 뉴스 요약 → 완료 시 scope_title_job 예약"""
    logger.info("[체인 4/8] summary 시작")
    _run_with_timeout(run_summary_pipeline, "summary")
    logger.info("[체인 4/8] summary 완료 → scope_title 예약")
    scheduler.add_job(
        scope_title_job,
        trigger="date",
        id="scope_title_chain",
        replace_existing=True,
    )


def scope_title_job():
    """[5/8] scope 대표 제목 생성 → 완료 시 scope_sentiment_job 예약"""
    logger.info("[체인 5/8] scope_title 시작")
    _run_with_timeout(run_scope_title_batch, "scope_title_chain")
    logger.info("[체인 5/8] scope_title 완료 → scope_sentiment 예약")
    scheduler.add_job(
        scope_sentiment_job,
        trigger="date",
        id="scope_sentiment",
        replace_existing=True,
    )


def scope_sentiment_job():
    """[6/8] scope 감성 집계 → 완료 시 scope_summary_job 예약"""
    logger.info("[체인 6/8] scope_sentiment 시작")
    _run_with_timeout(run_scope_sentiment_batch, "scope_sentiment")
    logger.info("[체인 6/8] scope_sentiment 완료 → scope_summary 예약")
    scheduler.add_job(
        scope_summary_job,
        trigger="date",
        id="scope_summary_batch",
        replace_existing=True,
    )


def scope_summary_job():
    """[7/8] scope 대표 요약 생성/갱신 → 완료 시 scope_keywords_job 예약"""
    logger.info("[체인 7/8] scope_summary 시작")
    _run_with_timeout(run_scope_summary_batch, "scope_summary_batch")
    logger.info("[체인 7/8] scope_summary 완료 → scope_keywords 예약")
    scheduler.add_job(
        scope_keywords_job,
        trigger="date",
        id="scope_keywords_chain",
        replace_existing=True,
    )


def scope_keywords_job():
    """[8/8] scope 키워드 집계 (체인 종료)"""
    logger.info("[체인 8/8] scope_keywords 시작")
    _run_with_timeout(run_scope_keywords_batch, "scope_keywords_chain")
    logger.info("[체인 8/8] scope_keywords 완료 — 일배치 체인 종료")


# ── 앱 생명주기 ────────────────────────────────────────
@app.on_event("startup")
def startup():
    # DB/ES 초기화 (없으면 생성)
    # MySQL pipeline_error_log 테이블 + ES 인덱스 생성
    # 최초 실행 시에만 필요하므로 오류가 나도 서버는 계속 실행
    try:
        init_db()
    except Exception as e:
        logger.warning(f"init_db 실패 (이미 존재하거나 연결 불가): {e}")

    # ── 일배치 체인 시작점 ─────────────────────────────
    # classification_job 완료 → sentiment_job → keyword_job
    # → summary_job → scope_sentiment_job → scope_summary_job
    #
    # TODO [테스트 완료 후 교체]
    #   운영: trigger="cron", hour=2, minute=0  (day_of_week 제거)
    scheduler.add_job(
        classification_job,
        trigger="cron", hour=2, minute=0,  # 테스트 완료: 수요일 11:30
        id="classification",
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
