"""
scopeTitle 생성 서비스 (Gemini)

트리거 규칙:
  - news_count <= 3 : 즉시 생성
  - news_count >  3 : scope_refresh_queue에 등록 → 30분 배치 처리
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
        _model = genai.GenerativeModel("gemini-2.0-flash")
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


def trigger_scope_title(es, scope_id: str, news_count: int):
    """분류 파이프라인에서 scopeID 배정 직후 호출"""
    if news_count <= NEWS_COUNT_DIRECT:
        generate_scope_title(es, scope_id)
    else:
        # 이미 pending 상태인 항목이 있으면 스킵
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
