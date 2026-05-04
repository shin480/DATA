"""
scopeTitle 생성 서비스 (Gemini)

트리거 규칙:
  - news_count <= 3 : 즉시 생성 / 실패 시 queue 자동 등록
  - news_count >  3 : scope_refresh_queue에 등록 → 30분 배치 처리

[수정 이력]
- 모델 교체: gemini-2.5-flash-lite → gemini-2.5-flash-lite (free tier 할당량 확보)
- trigger_scope_title: 즉시 생성 실패 시 queue 자동 등록으로 변경 (유실 방지)
- _enqueue: 중복 방지 queue 등록 함수 분리
- enqueue_missing_scope_titles: scopeTitle 없는 scope 일괄 queue 등록 복구 함수 추가
"""

import logging
import os
from datetime import datetime, timezone

import google.generativeai as genai

from model.database import get_es
from model.services.error_logger import log_pipeline_error

logger = logging.getLogger(__name__)

BATCH_SIZE        = 50
NEWS_COUNT_DIRECT = 3
INDEX_NEWS        = "news_economy"
INDEX_SCOPES      = "news_scopes"
INDEX_QUEUE       = "scope_refresh_queue"

_model = None


def get_model():
    global _model
    if _model is None:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        _model = genai.GenerativeModel("gemini-2.5-flash-lite")
    return _model


def generate_scope_title(es, scope_id: str):
    res = es.search(
        index=INDEX_NEWS,
        body={
            "query":   {"term": {"scopeID": scope_id}},
            "_source": ["title"],
            "sort":    [{"published_at": "desc"}],
            "size":    20,
        },
    )
    titles = [h["_source"]["title"] for h in res["hits"]["hits"] if h["_source"].get("title")]
    if not titles:
        return None

    if len(titles) == 1:
        scope_title = titles[0]
    else:
        prompt = (
            "다음은 동일한 사건/이슈를 다룬 한국 경제 뉴스 기사 제목들입니다.\n"
            "이 제목들을 대표하는 하나의 핵심 제목을 만들어주세요.\n\n"
            "조건:\n- 30자 이내\n- 핵심 사실만 담을 것\n- 중립적인 톤\n- 제목만 출력\n\n"
            "뉴스 제목들:\n" + "\n".join(f"- {t}" for t in titles)
        )
        try:
            scope_title = get_model().generate_content(prompt).text.strip()
        except Exception as e:
            logger.error(f"Gemini API 오류 (scope_id={scope_id}): {e}")
            return None

    es.update(
        index=INDEX_SCOPES,
        id=scope_id,
        body={"doc": {
            "scopeTitle": scope_title,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }},
        retry_on_conflict=3,
    )
    logger.info(f"scopeTitle 생성: {scope_id} → {scope_title}")
    return scope_title


def _enqueue(es, scope_id: str):
    """scope_refresh_queue에 pending 항목 등록 (중복 방지)"""
    existing = es.search(
        index=INDEX_QUEUE,
        body={
            "query": {
                "bool": {
                    "must": [
                        {"term": {"scopeID": scope_id}},
                        {"term": {"status":  "pending"}},
                    ]
                }
            },
            "size": 1,
        },
    )
    if existing["hits"]["total"]["value"] == 0:
        now_utc = datetime.now(timezone.utc).isoformat()
        es.index(
            index=INDEX_QUEUE,
            body={
                "scopeID":   scope_id,
                "queued_at": now_utc,
                "status":    "pending",
            },
        )
        logger.info(f"scope_refresh_queue 등록: {scope_id}")


def trigger_scope_title(es, scope_id: str, news_count: int):
    """분류 파이프라인에서 scopeID 배정 직후 호출"""
    if news_count <= NEWS_COUNT_DIRECT:
        result = generate_scope_title(es, scope_id)
        if result is None:
            # 즉시 생성 실패 시 queue에 등록하여 배치 재처리
            logger.warning(f"즉시 생성 실패, queue 등록: {scope_id}")
            _enqueue(es, scope_id)
    else:
        _enqueue(es, scope_id)


def enqueue_missing_scope_titles():
    """
    scopeTitle이 없는 scope를 전체 조회하여 queue에 일괄 등록.
    Gemini API 장애 등으로 유실된 scopeTitle 복구 시 사용.
    """
    try:
        es = get_es()
        res = es.search(
            index=INDEX_SCOPES,
            body={
                "query": {
                    "bool": {"must_not": {"exists": {"field": "scopeTitle"}}}
                },
                "_source": ["scopeID"],
                "size": 1000,
            },
        )
        hits = res["hits"]["hits"]
        if not hits:
            logger.info("scopeTitle 누락 scope 없음")
            es.close()
            return 0

        count = 0
        for hit in hits:
            scope_id = hit["_source"]["scopeID"]
            _enqueue(es, scope_id)
            count += 1

        es.close()
        logger.info(f"scopeTitle 누락 scope {count}건 queue 등록 완료")
        return count

    except Exception as e:
        log_pipeline_error(pipeline="scope_title_recovery", error=e)
        raise


def run_scope_title_batch():
    """scope_refresh_queue의 pending 항목 배치 처리 (30분 주기)"""
    try:
        es = get_es()

        res = es.search(
            index=INDEX_QUEUE,
            body={
                "query": {"term": {"status": "pending"}},
                "sort":  [{"queued_at": "asc"}],
                "size":  BATCH_SIZE,
            },
        )
        hits = res["hits"]["hits"]

        if not hits:
            logger.info("scope_refresh_queue 처리할 항목 없음")
            es.close()
            return

        logger.info(f"scopeTitle 배치 처리 시작: {len(hits)}건")

        for hit in hits:
            doc_id   = hit["_id"]
            scope_id = hit["_source"]["scopeID"]
            now_utc  = datetime.now(timezone.utc).isoformat()

            # processing 상태로 변경
            es.update(index=INDEX_QUEUE, id=doc_id,
                      body={"doc": {"status": "processing"}})
            try:
                generate_scope_title(es, scope_id)
                es.update(index=INDEX_QUEUE, id=doc_id,
                          body={"doc": {"status": "done", "processed_at": now_utc}})
            except Exception as e:
                logger.error(f"scopeTitle 생성 실패 scopeID={scope_id}: {e}")
                es.update(index=INDEX_QUEUE, id=doc_id,
                          body={"doc": {"status": "pending"}})

        es.close()
        logger.info(f"scopeTitle 배치 처리 완료: {len(hits)}건")

    except Exception as e:
        log_pipeline_error(pipeline="scope_title", error=e)
        raise
