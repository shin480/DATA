"""
분류 파이프라인 라우터 — ES 기반으로 status 조회 수정
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from model.database import get_es
from model.services.classifier import run_classification_pipeline
from model.services.scope_title import run_scope_title_batch, enqueue_missing_scope_titles
from model.services.scope_summarizer import run_scope_summary_batch
from model.services.sentiment import run_sentiment_pipeline, classify_single_article
from model.services.summarizer import run_summary_pipeline, summarize_single_article
from model.services.keyword_extractor import run_keyword_pipeline, extract_keywords_single
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

# ── 단건 재처리 Models ─────────────────────────────────
class ArticleOut(BaseModel):
    article_id:      str
    title:           str | None = None
    sentiment:       str | None = None
    sentiment_score: float | None = None
    keywords:        list[str] | None = None
    summary:         str | None = None
    scopeID:         str | None = None
    processed_status: str | None = None
    published_at:    str | None = None
    press:           str | None = None

class ReprocessRequest(BaseModel):
    targets: list[str]  # 예: ["sentiment", "keywords", "summary"] 또는 부분 선택

class ReprocessResult(BaseModel):
    article_id: str
    targets:    list[str]
    results:    dict        # 각 target별 성공/실패 여부
    after:      ArticleOut  # 재처리 후 갱신된 doc


INDEX_NEWS   = "news_economy"
INDEX_SCOPES = "news_scopes"

VALID_TARGETS = {"sentiment", "keywords", "summary"}


# ── 기존 배치 트리거 엔드포인트 ────────────────────────

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


# ── 단건 아티클 조회 ───────────────────────────────────

@router.get("/article/search", response_model=ArticleOut, summary="아티클 단건 조회")
def search_article(article_id: str):
    """
    article_id로 뉴스 단건을 조회합니다.
    재처리 전 현재 상태 확인용입니다.
    """
    try:
        es  = get_es()
        res = es.get(index=INDEX_NEWS, id=article_id, ignore=[404])
        es.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not res.get("found"):
        raise HTTPException(status_code=404, detail=f"article_id '{article_id}' 를 찾을 수 없습니다.")

    src = res["_source"]
    return _to_article_out(article_id, src)


# ── 단건 재처리 ────────────────────────────────────────

@router.post("/article/{article_id}/reprocess", response_model=ReprocessResult, summary="아티클 단건 재처리")
def reprocess_article(article_id: str, body: ReprocessRequest):
    """
    지정한 article_id의 선택 필드(sentiment / keywords / summary)를
    초기화 후 즉시 재처리합니다.

    - targets에 포함된 항목만 처리됩니다.
    - 재처리는 스케줄러 체인과 무관하게 즉시 동기 실행됩니다.
    - 완료 후 갱신된 doc을 반환합니다.
    """
    targets = [t for t in body.targets if t in VALID_TARGETS]
    if not targets:
        raise HTTPException(
            status_code=400,
            detail=f"유효한 target이 없습니다. 가능한 값: {sorted(VALID_TARGETS)}"
        )

    # ── 1. 아티클 존재 확인 ────────────────────────────
    try:
        es  = get_es()
        res = es.get(index=INDEX_NEWS, id=article_id, ignore=[404])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if not res.get("found"):
        es.close()
        raise HTTPException(status_code=404, detail=f"article_id '{article_id}' 를 찾을 수 없습니다.")

    src = res["_source"]

    # ── 2. 대상 필드 null 초기화 ──────────────────────
    clear_doc: dict = {}
    if "sentiment" in targets:
        clear_doc["sentiment"]       = None
        clear_doc["sentiment_score"] = None
    if "keywords" in targets:
        clear_doc["keywords"] = None
    if "summary" in targets:
        clear_doc["summary"] = None

    try:
        es.update(
            index=INDEX_NEWS,
            id=article_id,
            body={"doc": clear_doc},
        )
    except Exception as e:
        es.close()
        raise HTTPException(status_code=500, detail=f"필드 초기화 실패: {e}")

    # ── 3. 단건 재처리 실행 ───────────────────────────
    results: dict = {}

    if "sentiment" in targets:
        try:
            classify_single_article(article_id=article_id, src=src)
            results["sentiment"] = "success"
        except Exception as e:
            logger.error(f"[단건재처리] sentiment 실패 article_id={article_id}: {e}")
            results["sentiment"] = f"error: {e}"

    if "keywords" in targets:
        try:
            extract_keywords_single(article_id=article_id, src=src)
            results["keywords"] = "success"
        except Exception as e:
            logger.error(f"[단건재처리] keywords 실패 article_id={article_id}: {e}")
            results["keywords"] = f"error: {e}"

    if "summary" in targets:
        try:
            summarize_single_article(article_id=article_id, src=src)
            results["summary"] = "success"
        except Exception as e:
            logger.error(f"[단건재처리] summary 실패 article_id={article_id}: {e}")
            results["summary"] = f"error: {e}"

    # ── 4. 재처리 후 최신 doc 조회 ────────────────────
    try:
        updated = es.get(index=INDEX_NEWS, id=article_id)
        after_src = updated["_source"]
    except Exception as e:
        after_src = src  # 조회 실패 시 이전 값으로 대체
        logger.warning(f"[단건재처리] 갱신 doc 조회 실패: {e}")
    finally:
        es.close()

    return ReprocessResult(
        article_id = article_id,
        targets    = targets,
        results    = results,
        after      = _to_article_out(article_id, after_src),
    )


# ── 내부 헬퍼 ──────────────────────────────────────────

def _to_article_out(article_id: str, src: dict) -> ArticleOut:
    return ArticleOut(
        article_id       = src.get("article_id", article_id),
        title            = src.get("title"),
        sentiment        = src.get("sentiment"),
        sentiment_score  = src.get("sentiment_score"),
        keywords         = src.get("keywords"),
        summary          = src.get("summary"),
        scopeID          = src.get("scopeID"),
        processed_status = src.get("processed_status"),
        published_at     = src.get("published_at"),
        press            = src.get("press"),
    )


# ── scope 조회 ─────────────────────────────────────────

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
