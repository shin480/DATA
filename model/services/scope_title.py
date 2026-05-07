"""
scopeTitle 생성 서비스 (Gemini)

트리거 규칙:
  - news_count <= 3 : 즉시 생성 / 실패 시 queue 자동 등록
  - news_count >  3 : scope_refresh_queue에 등록 → 30분 배치 처리

[수정 이력]
- 모델 교체: gemini-2.0-flash → gemini-2.5-flash-lite
- trigger_scope_title: 즉시 생성 실패 시 queue 자동 등록으로 변경 (유실 방지)
- _enqueue: 중복 방지 queue 등록 함수 분리
- enqueue_missing_scope_titles: scopeTitle 없는 scope 일괄 queue 등록 복구 함수 추가
- API 키 로테이션: 7개 키 순환 사용 (429/일일 한도 초과 시 자동 교체)
- SDK 교체: google.generativeai(deprecated) → google.genai
- 예외 처리 보강: 403 포함 모든 API 오류 시 키 교체 후 재시도
"""

import logging
import os
import time
from datetime import datetime, timezone

from google import genai
from google.genai import types

from model.database import get_es
from model.services.error_logger import log_pipeline_error

logger = logging.getLogger(__name__)

BATCH_SIZE        = 50
NEWS_COUNT_DIRECT = 3
INDEX_NEWS        = "news_economy"
INDEX_SCOPES      = "news_scopes"
INDEX_QUEUE       = "scope_refresh_queue"
MODEL_NAME        = "gemini-2.5-flash-lite-preview-06-17"

# API 키 로테이션 (7개)
_API_KEYS = [
    os.getenv("GOOGLE_API_KEY_1"),
    os.getenv("GOOGLE_API_KEY_2"),
    os.getenv("GOOGLE_API_KEY_3"),
    os.getenv("GOOGLE_API_KEY_4"),
    os.getenv("GOOGLE_API_KEY_5"),
    os.getenv("GOOGLE_API_KEY_6"),
    os.getenv("GOOGLE_API_KEY_7"),
]
_key_index = 0
_client    = None


def _get_client():
    """현재 키 인덱스로 google.genai 클라이언트 반환 (캐시)"""
    global _client, _key_index
    if _client is None:
        key = _API_KEYS[_key_index]
        if not key:
            raise RuntimeError(f"GOOGLE_API_KEY_{_key_index + 1} 환경변수 없음")
        _client = genai.Client(api_key=key)
        logger.info(f"Gemini 클라이언트 초기화 (key index: {_key_index})")
    return _client


def _rotate_key():
    """API 오류(429/403 등) 시 다음 키로 교체"""
    global _key_index, _client
    _key_index = (_key_index + 1) % len(_API_KEYS)
    _client    = None
    logger.warning(f"API 키 교체 → index {_key_index}")


def _is_retryable(err_str: str) -> bool:
    """키 교체 후 재시도할 오류인지 판단"""
    err_lower = err_str.lower()
    return any(code in err_str for code in ("429", "403", "500", "503")) \
        or "quota" in err_lower \
        or "rate" in err_lower \
        or "limit" in err_lower


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
        # 키 로테이션 포함 재시도 (최대 7회)
        for attempt in range(len(_API_KEYS)):
            try:
                response = _get_client().models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=60,
                        temperature=0.3,
                    ),
                )
                scope_title = response.text.strip()
                break
            except Exception as e:
                err_str = str(e)
                if _is_retryable(err_str):
                    logger.warning(
                        f"Gemini API 오류 (scope_id={scope_id}, attempt={attempt + 1}, "
                        f"key_index={_key_index}): {err_str[:80]} → 키 교체"
                    )
                    _rotate_key()
                    time.sleep(2)
                else:
                    logger.error(f"Gemini 재시도 불가 오류 (scope_id={scope_id}): {e}")
                    return None
        else:
            logger.error(f"모든 API 키 소진 (scope_id={scope_id})")
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

            time.sleep(10)  # gemini-2.5-flash-lite RPM 대응

        es.close()
        logger.info(f"scopeTitle 배치 처리 완료: {len(hits)}건")

    except Exception as e:
        log_pipeline_error(pipeline="scope_title", error=e)
        raise
