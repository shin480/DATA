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
# 용언류: VV(동사) VA(형용사) XSV(동사파생접사) → 서술어 재구성용
_NOUN_POS = {"NNG", "NNP", "SL"}
_VERB_POS = {"VV", "VA", "XSV"}

# 명사 불용어
_NOUN_STOPWORDS = {
    "관련", "문제", "상황", "내용", "부분", "경우", "방안", "계획",
    "이후", "현재", "최근", "지난", "올해", "이번", "해당",
    "오늘", "어제", "지금", "당시", "발표", "보도",
    "뉴스", "기자", "기사", "취재", "종합", "단독", "속보", "업데이트",
    "기록", "수준", "규모", "이상", "이하", "대비", "대상", "기준",
}

# 동사 불용어: 너무 일반적이어서 서술어로 쓰기 애매한 것
_VERB_STOPWORDS = {
    "하다", "되다", "있다", "없다", "이다", "아니다",
    "말하다", "밝히다", "전하다", "나타나다", "보이다",
    "받다", "주다", "가다", "오다", "만들다",
}

# 동사 어간 → 자연스러운 명사형 서술어 변환 테이블
_VERB_TO_NOUN: dict[str, str] = {
    # 상승/하락 계열
    "오르": "상승", "내리": "하락", "떨어지": "하락",
    "급등하": "급등", "급락하": "급락", "폭등하": "폭등", "폭락하": "폭락",
    "반등하": "반등", "반락하": "반락",
    # 확대/축소 계열
    "늘": "증가", "줄": "감소", "확대되": "확대", "축소되": "축소",
    "증가하": "증가", "감소하": "감소", "둔화되": "둔화",
    # 위기/압박 계열
    "흔들리": "불안", "위협받": "위기", "압박받": "압박",
    "불안하": "불안", "악화되": "악화", "위축되": "위축",
    # 회복/개선 계열
    "회복되": "회복", "개선되": "개선", "호전되": "호전",
    # 결정/발표 계열
    "인상하": "인상", "인하하": "인하", "동결하": "동결",
    "결정하": "결정", "승인하": "승인", "거부하": "거부",
    "도입하": "도입", "폐지하": "폐지", "규제하": "규제",
    # 시장 계열
    "돌파하": "돌파", "하회하": "하회", "상회하": "상회",
    "초과하": "초과", "급증하": "급증", "급감하": "급감",
    # 기타
    "촉구하": "촉구", "반발하": "반발", "우려하": "우려",
    "기대하": "기대", "전망하": "전망",
}

_VERB_ENDINGS = ["하", "되", "시키", "받", "당하"]


# ── 핵심 함수: 타이틀 생성 ─────────────────────────────────────────

def _extract_tokens(titles: list[str]) -> tuple[list[list[str]], list[list[str]]]:
    """
    각 제목에서 명사 토큰과 동사 어간 토큰을 분리 추출.
    반환: (noun_docs, verb_docs) — 문서별 리스트
    """
    kiwi      = _get_kiwi()
    noun_docs = []
    verb_docs = []

    for title in titles:
        nouns = []
        verbs = []
        for token in kiwi.analyze(title)[0][0]:
            form = token.form
            tag  = token.tag
            if tag in _NOUN_POS and len(form) >= 2 and form not in _NOUN_STOPWORDS:
                nouns.append(form)
            elif tag in _VERB_POS and len(form) >= 2 and form not in _VERB_STOPWORDS:
                verbs.append(form)
        noun_docs.append(nouns)
        verb_docs.append(verbs)

    return noun_docs, verb_docs


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


def _verb_to_predicate(stem: str) -> str:
    """
    동사 어간 → 타이틀에 쓸 수 있는 명사형 서술어 변환.
    테이블 히트 → 매핑값 반환
    테이블 미스 → 어미 제거 후 stem 반환 (예: "인상되" → "인상")
    """
    if stem in _VERB_TO_NOUN:
        return _VERB_TO_NOUN[stem]
    for ending in _VERB_ENDINGS:
        if stem.endswith(ending):
            return stem[: -len(ending)]
    return stem


def _make_scope_title(titles: list[str]) -> str:
    """
    뉴스 제목 리스트 → 서술형 스콥 타이틀 생성

    단계:
      1. kiwipiepy로 명사 / 동사 어간 분리 추출
      2. 명사: TF-IDF 상위 3개 (주제어)
         동사: TF-IDF 상위 2개 → 명사형 서술어로 변환
      3. 조합 패턴:
         주제어 2개 + 서술어 → "{noun1} {noun2}, {predicate}"
         주제어 1~2개 + 서술어 → "{nouns}, {predicate}"
         서술어 없을 경우 → 명사 나열 (가운뎃점)
    """
    noun_docs, verb_docs = _extract_tokens(titles)

    top_nouns = _tfidf_top(noun_docs, top_n=3)
    top_verbs = _tfidf_top(verb_docs, top_n=2)

    # 동사 어간 → 명사형 서술어 (명사 목록과 중복 제거)
    predicates = []
    for stem in top_verbs:
        pred = _verb_to_predicate(stem)
        if pred not in top_nouns:
            predicates.append(pred)

    # 타이틀 조합
    if len(top_nouns) >= 2 and predicates:
        # "삼성전자 반도체, 수출 급감"
        title = f"{top_nouns[0]} {top_nouns[1]}, {predicates[0]}"

    elif len(top_nouns) >= 1 and predicates:
        # "한국은행 금리, 인상"
        noun_part = " ".join(top_nouns[:2])
        title = f"{noun_part}, {predicates[0]}"

    elif len(top_nouns) >= 3:
        # 서술어 없음 → 가운뎃점 나열
        title = f"{top_nouns[0]} {top_nouns[1]}·{top_nouns[2]}"

    elif len(top_nouns) >= 2:
        title = f"{top_nouns[0]}·{top_nouns[1]} 동향"

    elif top_nouns:
        title = f"{top_nouns[0]} 주요 이슈"

    else:
        # 형태소 추출 전체 실패 — 제목 앞 단어로 최소 보장
        words = []
        for t in titles[:3]:
            parts = t.split()
            if parts:
                words.append(parts[0])
        title = " ".join(dict.fromkeys(words)) or "주요 경제 이슈"

    # 30자 초과 시 공백 기준으로 자름
    if len(title) > 30:
        title = title[:30].rsplit(" ", 1)[0] if " " in title[:30] else title[:30]

    return title


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
    scopeTitle이 없는 scope를 전체 조회하여 scope_refresh_queue에 등록.

    즉시 생성 시도 없이 queue 등록만 수행 → API 타임아웃 방지.
    실제 생성은 run_scope_title_batch()(/classify/scope-titles)에서 처리.
    """
    try:
        es       = get_es()
        count    = 0
        from_idx = 0
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
                scope_id = hit["_source"]["scopeID"]
                _enqueue(es, scope_id)
                count += 1

            from_idx += page_size
            if len(hits) < page_size:
                break

        es.close()

        if count == 0:
            logger.info("scopeTitle 누락 scope 없음")
        else:
            logger.info(f"scopeTitle 누락 scope {count}건 queue 등록 완료")

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
