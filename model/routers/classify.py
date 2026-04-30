"""
분류 파이프라인 라우터 — ES 기반으로 status 조회 수정
"""

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException

from model.database import get_es
from model.services.classifier import run_classification_pipeline
from model.services.scope_title import run_scope_title_batch
from model.services.scope_summarizer import run_scope_summary_batch
from model.services.sentiment import run_sentiment_pipeline
from model.services.summarizer import run_summary_pipeline
from model.services.keyword_extractor import run_keyword_pipeline
from model.services.scope_keywords import run_scope_keywords_batch
from model.services.scope_sentiment import run_scope_sentiment_batch

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/classify", tags=["classify"])

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
            "no_summary_news":        no_summary,
            "no_keywords_news":       no_keywords,
            "no_scope_keywords":      no_scope_keywords,
            "no_scope_summary":       no_scope_summary,
            "total_scopes":           total_scopes,
            "sentiment_distribution": sentiment_dist,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
