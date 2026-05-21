"""
scopeTitle 생성 서비스 — 키워드 기반 문장형 타이틀 (Gemini 완전 제거)

생성 전략:
  1. 스콥 내 뉴스 제목들 수집
  2. kiwipiepy 형태소 분석으로 명사/고유명사 추출
  3. TF-IDF로 스콥 대표 키워드 3~5개 선별
  4. 키워드 역할(주체/사건/상태) 분류 후 패턴 템플릿으로 자연스러운 문장 조합

[수정 이력]
- Gemini 의존도 완전 제거 (google.genai 삭제)
- 타이틀 생성 엔진 교체: 최신 기사 제목 복사 → TF-IDF + 문장 패턴 조합
- scope_refresh_queue / 배치 구조 유지 (운영 로직 변경 없음)
- NEWS_COUNT_GEMINI 임계값 제거 → 기사 수 무관하게 동일 로직 적용
- kiwipiepy Kiwi 인스턴스 모듈 레벨 싱글턴으로 관리
"""

import logging
import math
import re
from collections import Counter, defaultdict
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
# NNG: 일반명사  NNP: 고유명사  SL: 외래어
_TARGET_POS = {"NNG", "NNP", "SL"}

# 타이틀에 불필요한 단일 일반 단어 필터 (너무 광범위한 단어)
_STOPWORDS = {
    "관련", "문제", "상황", "내용", "부분", "경우", "방안", "계획",
    "결과", "이후", "현재", "최근", "지난", "올해", "이번", "해당",
    "오늘", "어제", "지금", "당시", "전망", "분석", "발표", "보도",
    "뉴스", "기자", "기사", "취재", "종합", "단독", "속보", "업데이트",
}


# ── 경제 도메인 키워드 역할 사전 ───────────────────────────────────
# 주체(subject): 문장 앞에 오는 행위자
# 사건(event):   핵심 동작/이슈
# 상태(state):   결과/방향성을 나타내는 명사

_SUBJECT_HINTS = {
    "한국은행", "기재부", "금융위", "금감원", "정부", "국회", "청와대",
    "삼성", "삼성전자", "현대", "SK", "LG", "롯데", "포스코", "카카오",
    "네이버", "쿠팡", "하이닉스", "현대차", "기아", "셀트리온",
    "연준", "Fed", "ECB", "IMF", "세계은행",
    "미국", "중국", "일본", "유럽", "러시아", "중동",
}

_EVENT_HINTS = {
    "금리", "기준금리", "금리인상", "금리인하",
    "수출", "수입", "무역", "무역수지", "경상수지",
    "물가", "인플레이션", "디플레이션", "CPI",
    "환율", "달러", "원달러", "엔화", "위안화",
    "주가", "코스피", "코스닥", "증시", "주식",
    "부동산", "집값", "전세", "매매", "아파트",
    "고용", "실업", "취업", "일자리",
    "성장률", "GDP", "경기", "경제성장",
    "반도체", "배터리", "전기차", "AI", "인공지능",
    "유가", "원유", "에너지",
    "회사채", "국채", "채권",
    "공급망", "공급", "수요",
}

_STATE_HINTS = {
    "상승", "하락", "급등", "급락", "하향", "상향",
    "위기", "리스크", "불안", "불확실",
    "호조", "부진", "침체", "회복", "반등",
    "확대", "축소", "감소", "증가", "둔화", "가속",
    "흑자", "적자", "손실", "이익",
    "전망", "우려", "기대",
}


# ── 문장 패턴 템플릿 ───────────────────────────────────────────────
# {S}: 주체  {E}: 사건  {T}: 상태/방향
# 키워드 역할 조합에 따라 가장 자연스러운 패턴 선택

def _build_title(subject: str, event: str, state: str) -> str:
    """주체 / 사건 / 상태 키워드 조합으로 자연스러운 문장형 타이틀 반환"""
    has_s = bool(subject)
    has_e = bool(event)
    has_t = bool(state)

    if has_s and has_e and has_t:
        return f"{subject} {event} {state} 동향"
    if has_s and has_e:
        return f"{subject} {event} 이슈 분석"
    if has_s and has_t:
        return f"{subject} 관련 {state} 흐름"
    if has_e and has_t:
        return f"{event} {state}과 시장 영향"
    if has_e:
        return f"{event} 관련 동향"
    if has_s:
        return f"{subject} 주요 경제 이슈"
    if has_t:
        return f"경제 {state} 흐름과 전망"
    return "주요 경제 이슈 동향"


# ── 핵심 함수: 타이틀 생성 ─────────────────────────────────────────

def _extract_nouns(titles: list[str]) -> list[list[str]]:
    """각 제목에서 명사/고유명사 추출 → 문서별 토큰 리스트 반환"""
    kiwi   = _get_kiwi()
    result = []
    for title in titles:
        tokens = [
            token.form
            for token in kiwi.analyze(title)[0][0]
            if token.tag in _TARGET_POS
               and len(token.form) >= 2
               and token.form not in _STOPWORDS
        ]
        result.append(tokens)
    return result


def _tfidf_keywords(doc_tokens: list[list[str]], top_n: int = 5) -> list[str]:
    """
    TF-IDF 방식으로 스콥 대표 키워드 추출.

    - TF  : 전체 스콥 제목을 하나의 문서로 보고 단어 빈도 계산
    - IDF : 단어가 몇 개의 제목에 등장하는지로 역빈도 계산
            → 모든 제목에 공통으로 나오는 단어 억제 (너무 일반적인 단어 제거)
    """
    if not doc_tokens:
        return []

    # TF: 전체 단어 빈도
    total_counter: Counter = Counter()
    for tokens in doc_tokens:
        total_counter.update(tokens)

    # IDF: 각 단어가 등장한 문서(제목) 수
    doc_count = len(doc_tokens)
    df: Counter = Counter()
    for tokens in doc_tokens:
        for word in set(tokens):
            df[word] += 1

    # TF-IDF 점수 계산
    scores: dict[str, float] = {}
    for word, tf in total_counter.items():
        idf = math.log((doc_count + 1) / (df[word] + 1)) + 1.0
        scores[word] = tf * idf

    # 점수 상위 top_n 반환
    return [w for w, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_n]]


def _classify_keywords(keywords: list[str]) -> tuple[str, str, str]:
    """
    키워드를 주체 / 사건 / 상태로 분류.
    사전 미등재 키워드는 등장 순서로 사건(event) 슬롯에 우선 배치.
    """
    subject = ""
    event   = ""
    state   = ""

    for kw in keywords:
        if not subject and kw in _SUBJECT_HINTS:
            subject = kw
        elif not state and kw in _STATE_HINTS:
            state = kw
        elif not event and kw in _EVENT_HINTS:
            event = kw
        elif not event:
            # 사전 미등재 → 사건 슬롯에 배치 (도메인 고유명사일 가능성 높음)
            event = kw

    return subject, event, state


def _make_scope_title(titles: list[str]) -> str:
    """
    뉴스 제목 리스트 → 문장형 스콥 타이틀 생성

    단계:
      1. kiwipiepy 명사 추출
      2. TF-IDF 핵심 키워드 5개 선별
      3. 역할 분류(주체/사건/상태)
      4. 패턴 템플릿 조합
    """
    doc_tokens = _extract_nouns(titles)
    keywords   = _tfidf_keywords(doc_tokens, top_n=5)

    if not keywords:
        # 형태소 추출 실패 시 — 제목 첫 단어 조합으로 최소 보장
        fallback_words = []
        for t in titles[:3]:
            parts = t.split()
            if parts:
                fallback_words.append(parts[0])
        return " ".join(dict.fromkeys(fallback_words))[:30] or "주요 경제 이슈"

    subject, event, state = _classify_keywords(keywords)
    title = _build_title(subject, event, state)

    # 30자 초과 시 상태 키워드 제거 후 재생성
    if len(title) > 30:
        title = _build_title(subject, event, "")
    if len(title) > 30:
        title = _build_title("", event, state)
    if len(title) > 30:
        title = _build_title("", event, "")

    return title[:30]


# ── ES 헬퍼 ───────────────────────────────────────────────────────

def _fetch_titles(es, scope_id: str) -> list[str]:
    """scope에 속한 기사 제목 목록 반환 (최신순 최대 20개)"""
    res = es.search(
        index=INDEX_NEWS,
        body={
            "query":   {"term": {"scopeID": scope_id}},
            "_source": ["title"],
            "sort":    [{"published_at": "desc"}],
            "size":    20,
        },
    )
    return [h["_source"]["title"] for h in res["hits"]["hits"] if h["_source"].get("title")]


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
    scopeTitle이 없는 scope를 전체 조회하여 즉시 생성 처리.
    실패한 경우에만 queue 등록.
    """
    try:
        es = get_es()
        res = es.search(
            index=INDEX_SCOPES,
            body={
                "query": {
                    "bool": {"must_not": {"exists": {"field": "scopeTitle"}}}
                },
                "_source": ["scopeID", "news_count"],
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
            nc       = hit["_source"].get("news_count", 0)
            result   = generate_scope_title(es, scope_id, nc)
            if result is None:
                _enqueue(es, scope_id)
            count += 1

        es.close()
        logger.info(f"scopeTitle 누락 scope {count}건 처리 완료")
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
