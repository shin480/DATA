"""
scopeTitle 생성 서비스 — 동사/형용사 포함 서술형 타이틀 (Gemini 완전 제거)

생성 전략:
  1. 스콥 내 뉴스 제목들 수집
  2. kiwipiepy로 명사(NNG/NNP/SL) + 동사 어간(VV/VA/XSV) 분리 추출
  3. 명사 TF-IDF 상위 3개 (주제어) + 동사 TF-IDF 상위 2개 → 명사형 서술어 변환
  4. 조합 패턴으로 서술형 타이틀 생성
     예) "삼성전자 반도체, 수출 급감"
         "한국은행 금리, 추가 인상"
         "비트코인 ETF·가상자산 급등"

[수정 이력]
- Gemini 의존도 완전 제거
- 타이틀 엔진 v1: 명사 TF-IDF + 역할 분류 사전 → "관련 동향" 남발 문제
- 타이틀 엔진 v2: 동사 어간 추출 추가 + 명사형 서술어 변환 테이블 + 쉼표 패턴
- scope_refresh_queue / 배치 구조 유지 (운영 로직 변경 없음)
- NEWS_COUNT_GEMINI 임계값 제거 → 기사 수 무관하게 동일 로직 적용
- kiwipiepy Kiwi 인스턴스 모듈 레벨 싱글턴으로 관리
"""

import logging
import math
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Optional

from kiwipiepy import Kiwi

from model.database import get_es
from model.services.error_logger import log_pipeline_error

logger = logging.getLogger(__name__)

BATCH_SIZE   = 10000
MAX_RETRY    = 3
INDEX_NEWS   = "news_economy"
INDEX_SCOPES = "news_scopes"
INDEX_QUEUE  = "scope_refresh_queue"

# ── kiwipiepy 싱글턴 ───────────────────────────────────────────────
_kiwi: Optional[Kiwi] = None

def _get_kiwi() -> Kiwi:
    global _kiwi
    if _kiwi is None:
        _kiwi = Kiwi()
        logger.info("Kiwi 형태소 분석기 초기화 완료")
    return _kiwi


# ── 추출 대상 품사 ─────────────────────────────────────────────────
# 명사류: NNG(일반명사) NNP(고유명사) SL(외래어)
_NOUN_POS = {"NNG", "NNP", "SL"}

# 명사 불용어
_NOUN_STOPWORDS = {
    "관련", "문제", "상황", "내용", "부분", "경우", "방안", "계획",
    "이후", "현재", "최근", "지난", "올해", "이번", "해당",
    "오늘", "어제", "지금", "당시", "발표", "보도",
    "뉴스", "기자", "기사", "취재", "종합", "단독", "속보", "업데이트",
    "기록", "수준", "규모", "이상", "이하", "대비", "대상", "기준",
    "분석", "이슈", "스퀘어", "넘버스", "특집", "기획",
    "동향", "전망", "현황", "정리", "요약", "심층", "집중",
    "미디어", "채널", "방송", "라이브", "온라인",
    "결과", "영향", "효과", "변화", "추이", "흐름", "움직임",
    "가운데", "속에서", "상황에서", "따라서",
}


# ── 핵심 함수: 타이틀 생성 ─────────────────────────────────────────

def _extract_nouns(titles: list[str]) -> list[list[str]]:
    """각 제목에서 명사 토큰 추출 → 문서별 리스트 반환"""
    kiwi      = _get_kiwi()
    noun_docs = []
    for title in titles:
        nouns = [
            token.form
            for token in kiwi.analyze(title)[0][0]
            if token.tag in _NOUN_POS
            and len(token.form) >= 2
            and token.form not in _NOUN_STOPWORDS
        ]
        noun_docs.append(nouns)
    return noun_docs


def _tfidf_top(doc_tokens: list[list[str]], top_n: int) -> list[str]:
    """TF-IDF 점수 상위 top_n 토큰 반환"""
    if not doc_tokens:
        return []

    total_counter: Counter = Counter()
    for tokens in doc_tokens:
        total_counter.update(tokens)

    doc_count = len(doc_tokens)
    df: Counter = Counter()
    for tokens in doc_tokens:
        for word in set(tokens):
            df[word] += 1

    scores: dict[str, float] = {}
    for word, tf in total_counter.items():
        idf = math.log((doc_count + 1) / (df[word] + 1)) + 1.0
        scores[word] = tf * idf

    return [w for w, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_n]]


def _make_scope_title(titles: list[str]) -> str:
    """
    뉴스 제목 리스트 → 명사 TF-IDF 기반 스콥 타이틀 생성

    단계:
      1. kiwipiepy로 명사(NNG/NNP/SL) 추출
      2. TF-IDF 상위 5개 선별
      3. 중요도 순 공백 조합
         예) "트럼프 관세 수출 무역수지 충격"
             "한국은행 기준금리 인상 물가 대응"
             "삼성전자 반도체 실적 영업이익 하락"
    """
    noun_docs = _extract_nouns(titles)
    top_nouns = _tfidf_top(noun_docs, top_n=5)

    if not top_nouns:
        # 형태소 추출 전체 실패 — 제목 앞 단어로 최소 보장
        words = []
        for t in titles[:3]:
            parts = t.split()
            if parts:
                words.append(parts[0])
        return " ".join(dict.fromkeys(words)) or "주요 경제 이슈"

    title = " ".join(top_nouns)

    # 50자 초과 시 공백 기준으로 자름
    if len(title) > 50:
        title = title[:50].rsplit(" ", 1)[0] if " " in title[:50] else title[:50]

    return title


# ── ES 헬퍼 ───────────────────────────────────────────────────────

# 제목 전처리 패턴 (섹션 태그, 언론사 브랜드 등 제거)
_TITLE_CLEAN_PATTERNS = [
    # 대괄호 섹션 태그: [ETF 스퀘어], [단독], [넘버스] 등
    re.compile(r"^\s*\[[^\]]{1,20}\]\s*"),
    re.compile(r"\[[^\]]{1,20}\]"),
    # 소괄호 분류 태그: (종합), (상보), (1보) 등
    re.compile(r"\((?:종합\d*|상보|속보|\d+보)\)"),
    # 언론사 섹션 구분자: "AI 뉴스 분석 |", "ETF 스퀘어 |" 등
    re.compile(r"^[가-힣a-zA-Z\s·]{2,15}\s*[|｜]\s*"),
    # 말줄임표/특수문자 정리
    re.compile(r"\s{2,}"),
]

def _clean_title(title: str) -> str:
    """기사 제목에서 섹션 태그 및 언론사 브랜드 제거"""
    for pattern in _TITLE_CLEAN_PATTERNS:
        title = pattern.sub(" ", title)
    return title.strip()


def _fetch_titles(es, scope_id: str) -> list[str]:
    """scope에 속한 기사 제목 목록 반환 (최신순 최대 20개, 섹션 태그 제거)"""
    res = es.search(
        index=INDEX_NEWS,
        body={
            "query":   {"term": {"scopeID": scope_id}},
            "_source": ["title"],
            "sort":    [{"published_at": "desc"}],
            "size":    20,
        },
    )
    titles = []
    for h in res["hits"]["hits"]:
        raw = h["_source"].get("title", "")
        if raw:
            cleaned = _clean_title(raw)
            if cleaned:
                titles.append(cleaned)
    return titles


def _upsert_scope_title(es, scope_id: str, scope_title: str):
    """news_scopes에 scopeTitle upsert"""
    es.update(
        index=INDEX_SCOPES,
        id=scope_id,
        body={"doc": {
            "scopeTitle": scope_title,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }},
        retry_on_conflict=3,
    )
    logger.info(f"scopeTitle 저장: {scope_id} → {scope_title}")


# ── 메인 생성 함수 ─────────────────────────────────────────────────

def generate_scope_title(es, scope_id: str, news_count: int | None = None) -> str | None:
    """
    scopeTitle 생성 메인 함수.
    기사 수와 무관하게 동일한 TF-IDF 기반 로직 적용.
    내부 예외는 흡수하여 None 반환 (배치 중단 방지).
    """
    try:
        titles = _fetch_titles(es, scope_id)
        if not titles:
            logger.warning(f"기사 제목 없음 (scope_id={scope_id})")
            return None

        scope_title = _make_scope_title(titles)
        _upsert_scope_title(es, scope_id, scope_title)
        return scope_title

    except Exception as e:
        logger.error(f"generate_scope_title 예외 (scope_id={scope_id}): {e}")
        return None


# ── 큐 관리 ───────────────────────────────────────────────────────

def _enqueue(es, scope_id: str):
    """scope_refresh_queue에 pending 항목 등록 (중복 방지)"""
    existing = es.search(
        index=INDEX_QUEUE,
        body={
            "query": {
                "bool": {
                    "must":     {"term": {"scopeID": scope_id}},
                    "must_not": {"term": {"status":  "failed"}},
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
                "scopeID":     scope_id,
                "queued_at":   now_utc,
                "status":      "pending",
                "retry_count": 0,
            },
        )
        logger.info(f"scope_refresh_queue 등록: {scope_id}")


def trigger_scope_title(es, scope_id: str, news_count: int):
    """
    분류 파이프라인에서 scopeID 배정 직후 호출.
    기사 수 무관하게 즉시 생성 시도 → 실패 시 queue 등록.
    """
    result = generate_scope_title(es, scope_id, news_count)
    if result is None:
        _enqueue(es, scope_id)


def enqueue_missing_scope_titles():
    """
    scopeTitle이 없는 scope 중 news_economy에 실제 기사가 있는 것만 queue 등록.

    news_scopes 기준 전체 조회 → 유령 스콥(기사 없는 스콥) 포함 문제 해결.
    news_economy에서 scopeID 집계 → 실제 기사 있는 스콥만 등록.
    즉시 생성 시도 없이 queue 등록만 수행 → API 타임아웃 방지.
    실제 생성은 run_scope_title_batch()(/classify/scope-titles)에서 처리.
    """
    try:
        es    = get_es()
        count = 0

        # 1. scopeTitle 없는 스콥 ID 목록
        missing_ids: set[str] = set()
        from_idx  = 0
        page_size = 1000
        while True:
            res = es.search(
                index=INDEX_SCOPES,
                body={
                    "query": {
                        "bool": {"must_not": {"exists": {"field": "scopeTitle"}}}
                    },
                    "_source": ["scopeID"],
                    "size":   page_size,
                    "from":   from_idx,
                    "sort":   [{"created_at": "asc"}],
                },
            )
            hits = res["hits"]["hits"]
            if not hits:
                break
            for hit in hits:
                missing_ids.add(hit["_source"]["scopeID"])
            from_idx += page_size
            if len(hits) < page_size:
                break

        if not missing_ids:
            logger.info("scopeTitle 누락 scope 없음")
            es.close()
            return 0

        logger.info(f"scopeTitle 누락 scope 총 {len(missing_ids)}건 확인")

        # 2. news_economy에서 실제 기사 있는 scopeID 집계
        agg_res = es.search(
            index=INDEX_NEWS,
            body={
                "query": {
                    "terms": {"scopeID": list(missing_ids)}
                },
                "aggs": {
                    "real_scopes": {
                        "terms": {
                            "field": "scopeID",
                            "size":  len(missing_ids),
                        }
                    }
                },
                "size": 0,
            },
        )
        real_scope_ids = {
            b["key"]
            for b in agg_res["aggregations"]["real_scopes"]["buckets"]
        }
        logger.info(f"실제 기사 있는 scope: {len(real_scope_ids)}건 / 누락 {len(missing_ids)}건")

        # 3. 실제 기사 있는 스콥만 queue 등록
        for scope_id in real_scope_ids:
            _enqueue(es, scope_id)
            count += 1

        if count == 0:
            logger.info("등록할 scope 없음 (전부 유령 스콥)")
        else:
            # 등록 후 refresh → 배치가 즉시 검색 가능하도록
            es.indices.refresh(index=INDEX_QUEUE)
            logger.info(f"scopeTitle queue 등록 완료: {count}건")

        es.close()
        return count

    except Exception as e:
        log_pipeline_error(pipeline="scope_title_recovery", error=e)
        raise


# ── 배치 처리 ─────────────────────────────────────────────────────

def run_scope_title_batch():
    """
    scope_refresh_queue의 pending 항목을 배치 처리.

    - generate_scope_title 성공(str 반환) → done
    - 실패(None 반환) → retry_count 증가
      · retry_count < MAX_RETRY  : pending 유지 (다음 배치 재시도)
      · retry_count >= MAX_RETRY : failed 마킹 후 스킵
    """
    try:
        es = get_es()

        total_processed = 0
        batch_num       = 0

        while True:
            batch_num += 1

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
                if batch_num == 1:
                    logger.info("scope_refresh_queue 처리할 항목 없음")
                else:
                    logger.info(f"scopeTitle 배치 전체 완료 | total={total_processed}")
                break

            logger.info(f"[배치 {batch_num}] scopeTitle 처리 시작: {len(hits)}건")

            for hit in hits:
                doc_id      = hit["_id"]
                scope_id    = hit["_source"]["scopeID"]
                retry_count = hit["_source"].get("retry_count", 0)
                now_utc     = datetime.now(timezone.utc).isoformat()

                # MAX_RETRY 초과 → failed 처리
                if retry_count >= MAX_RETRY:
                    logger.error(
                        f"최대 재시도 초과, failed 처리: scopeID={scope_id} "
                        f"(retry_count={retry_count})"
                    )
                    es.update(index=INDEX_QUEUE, id=doc_id,
                              body={"doc": {"status": "failed", "processed_at": now_utc}})
                    continue

                es.update(index=INDEX_QUEUE, id=doc_id,
                          body={"doc": {"status": "processing"}})

                result = generate_scope_title(es, scope_id, news_count=None)

                if result is not None:
                    es.update(index=INDEX_QUEUE, id=doc_id,
                              body={"doc": {"status": "done", "processed_at": now_utc}})
                    total_processed += 1
                else:
                    new_retry = retry_count + 1
                    if new_retry >= MAX_RETRY:
                        logger.error(f"재시도 한도 도달, failed 처리: scopeID={scope_id}")
                        es.update(index=INDEX_QUEUE, id=doc_id,
                                  body={"doc": {
                                      "status":       "failed",
                                      "retry_count":  new_retry,
                                      "processed_at": now_utc,
                                  }})
                    else:
                        logger.warning(
                            f"scopeTitle 생성 실패, 재시도 예정: scopeID={scope_id} "
                            f"(retry_count={new_retry}/{MAX_RETRY})"
                        )
                        es.update(index=INDEX_QUEUE, id=doc_id,
                                  body={"doc": {
                                      "status":      "pending",
                                      "retry_count": new_retry,
                                  }})

            es.indices.refresh(index=INDEX_QUEUE)
            logger.info(f"[배치 {batch_num}] 완료 | 누적 processed={total_processed}")

        es.close()
        logger.info(f"scopeTitle 배치 처리 완료 | total={total_processed}")

    except Exception as e:
        log_pipeline_error(pipeline="scope_title", error=e)
        raise
