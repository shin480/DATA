"""
분류 파이프라인 라우터 — ES 기반으로 status 조회 수정
"""

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from model.database import get_es
from model.services.classifier import run_classification_pipeline
from model.services.scope_title import run_scope_title_batch, enqueue_missing_scope_titles
from model.services.scope_summarizer import run_scope_summary_batch
from model.services.sentiment import run_sentiment_pipeline
from model.services.summarizer import run_summary_pipeline
from model.services.keyword_extractor import run_keyword_pipeline
from model.services.scope_keywords import run_scope_keywords_batch
from model.services.scope_sentiment import run_scope_sentiment_batch

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/classify", tags=["classify"])


# ── Response Models ────────────────────────────────────
class SentimentDist(BaseModel):
    positive: float = 0.0
    negative: float = 0.0
    neutral:  float = 0.0

class ScopeOut(BaseModel):
    scopeID:         str
    scopeTitle:      str | None = None
    sentiment:       str | None = None
    sentiment_score: float | None = None
    sentiment_dist:  SentimentDist | None = None
    news_count:      int | None = None
    scope_keywords:  list[str] | None = None
    scope_summary:   str | None = None
    created_at:      str | None = None
    updated_at:      str | None = None

INDEX_NEWS   = "news_economy"
INDEX_SCOPES = "news_scopes"


@router.post("/run")
def trigger_classification(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_classification_pipeline)
    return {"message": "분류 파이프라인 시작됨 (백그라운드)"}


@router.post("/sentiment")
def trigger_sentiment(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_sentiment_pipeline)
    return {"message": "감성 분류 시작됨 (백그라운드)"}


@router.post("/scope-titles")
def trigger_scope_titles(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_scope_title_batch)
    return {"message": "scopeTitle 배치 처리 시작됨 (백그라운드)"}


@router.post("/scope-titles/recover")
def recover_scope_titles(background_tasks: BackgroundTasks):
    """scopeTitle 누락 scope를 queue에 일괄 등록 후 배치 처리합니다."""
    def _recover():
        import time
        enqueue_missing_scope_titles()
        time.sleep(2)  # ES 인덱싱 반영 대기
        run_scope_title_batch()
    background_tasks.add_task(_recover)
    return {"message": "scopeTitle 복구 시작됨 (백그라운드)"}


@router.post("/summarize")
def trigger_summarize(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_summary_pipeline)
    return {"message": "뉴스 요약 시작됨 (백그라운드)"}


@router.post("/keywords")
def trigger_keywords(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_keyword_pipeline)
    return {"message": "키워드 추출 시작됨 (백그라운드)"}


@router.post("/scope-keywords")
def trigger_scope_keywords(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_scope_keywords_batch)
    return {"message": "scope 키워드 집계 시작됨 (백그라운드)"}


@router.post("/scope-sentiment")
def trigger_scope_sentiment(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_scope_sentiment_batch)
    return {"message": "scope 감성 집계 시작됨 (백그라운드)"}


@router.post("/scope-summary")
def trigger_scope_summary(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_scope_summary_batch)
    return {"message": "scope 요약 생성 시작됨 (백그라운드)"}


@router.get("/scopes/{scope_id}", response_model=ScopeOut, summary="scope 상세 조회")
def get_scope(scope_id: str):
    """
    scopeID로 단일 scope를 조회합니다.

    - **sentiment_dist**: scope에 속한 전체 뉴스의 감성 비율 (positive/negative/neutral 합산 = 1.0)
    - **sentiment**: 가장 높은 비율의 감성 레이블
    - **sentiment_score**: 해당 감성 레이블의 평균 score
    """
    try:
        es  = get_es()
        res = es.get(index=INDEX_SCOPES, id=scope_id, ignore=[404])
        es.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not res.get("found"):
        raise HTTPException(status_code=404, detail=f"scopeID '{scope_id}' 를 찾을 수 없습니다.")

    src  = res["_source"]
    raw  = src.get("sentiment_dist") or {}
    dist = SentimentDist(
        positive=raw.get("positive", 0.0),
        negative=raw.get("negative", 0.0),
        neutral =raw.get("neutral",  0.0),
    )

    return ScopeOut(
        scopeID         = src.get("scopeID", scope_id),
        scopeTitle      = src.get("scopeTitle"),
        sentiment       = src.get("sentiment"),
        sentiment_score = src.get("sentiment_score"),
        sentiment_dist  = dist,
        news_count      = src.get("news_count"),
        scope_keywords  = src.get("scope_keywords"),
        scope_summary   = src.get("scope_summary"),
        created_at      = src.get("created_at"),
        updated_at      = src.get("updated_at"),
    )


@router.get("/status")
def get_classification_status():
    """미처리 뉴스 현황과 scope 통계를 ES에서 조회합니다."""
    try:
        es = get_es()

        def _count(index, query):
            return es.count(index=index, body={"query": query})["count"]

        unclassified = _count(INDEX_NEWS, {
            "bool": {"must_not": {"exists": {"field": "scopeID"}}}
        })
        no_sentiment = _count(INDEX_NEWS, {
            "bool": {
                "must":     {"exists": {"field": "scopeID"}},
                "must_not": {"exists": {"field": "sentiment"}},
            }
        })
        no_summary = _count(INDEX_NEWS, {
            "bool": {"must_not": {"exists": {"field": "summary"}}}
        })
        no_keywords = _count(INDEX_NEWS, {
            "bool": {"must_not": {"exists": {"field": "keywords"}}}
        })
        total_scopes = _count(INDEX_SCOPES, {"match_all": {}})
        no_scope_keywords = _count(INDEX_SCOPES, {
            "bool": {"must_not": {"exists": {"field": "scope_keywords"}}}
        })
        no_scope_summary = _count(INDEX_SCOPES, {
            "bool": {"must_not": {"exists": {"field": "scope_summary"}}}
        })
        no_scope_title = _count(INDEX_SCOPES, {
            "bool": {"must_not": {"exists": {"field": "scopeTitle"}}}
        })
        has_scope_title = _count(INDEX_SCOPES, {
            "bool": {"must": {"exists": {"field": "scopeTitle"}}}
        })

        # 감성 분포
        agg_res = es.search(
            index=INDEX_NEWS,
            body={
                "size": 0,
                "query": {"exists": {"field": "sentiment"}},
                "aggs": {
                    "by_sentiment": {
                        "terms": {"field": "sentiment", "size": 10}
                    }
                },
            },
        )
        sentiment_dist = {
            b["key"]: b["doc_count"]
            for b in agg_res["aggregations"]["by_sentiment"]["buckets"]
        }

        es.close()

        return {
            "unclassified_news":      unclassified,
            "no_sentiment_news":      no_sentiment,
            "no_scope_title":         no_scope_title,
            "has_scope_title":        has_scope_title,
            "no_summary_news":        no_summary,
            "no_keywords_news":       no_keywords,
            "no_scope_keywords":      no_scope_keywords,
            "sentiment_distribution": sentiment_dist,
            "no_scope_summary":       no_scope_summary,
            "total_scopes":           total_scopes,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
