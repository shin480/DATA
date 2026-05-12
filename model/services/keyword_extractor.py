"""
뉴스 키워드 추출 서비스 — TF-IDF 방식

흐름:
  1. ES news_economy에서 keywords=NULL 레코드 polling
  2. tokens(형태소 분석 완료) + 제목으로 TF-IDF 키워드 추출
  3. news_economy.keywords upsert

[수정 이력]
- 입력 변경: content 원문 → tokens 필드 사용 (조사/어미 제거된 형태소 단위)
- 불용어 처리 강화: 고빈도 동사/조사/어미 등 의미 없는 단어 필터링
- _is_valid_keyword: 한글 단어 최소 2자, 단순 어간 제거 강화
- [2026-05] 불용어 확장: 뉴스 어미 활용형 / URL 도메인 조각 / 언론 메타어 추가
- [2026-05] 전처리 강화: URL/이메일/도메인/숫자/특수문자 패턴 제거 (title + tokens 양쪽 적용)
- [2026-05] 동적 불용어: ES keyword_stopwords 인덱스에서 배치 시작 시 로드해 STOPWORDS에 병합
- [2026-05] 명사 필터: 동사 어간 패턴으로 끝나는 토큰 제거 (_is_noun_like)
"""

import logging
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from model.database import get_es
from model.services.error_logger import log_pipeline_error

logger = logging.getLogger(__name__)

BATCH_SIZE   = 10000
MAX_KEYWORDS = 5
TITLE_WEIGHT = 3.0
MIN_WORD_LEN = 2
INDEX_NEWS       = "news_economy"
INDEX_STOPWORDS  = "keyword_stopwords"

# ─────────────────────────────────────────────────────────────
# 불용어 정의
# ─────────────────────────────────────────────────────────────

# 동사/형용사 어간 (형태소 분석 후 단독 토큰으로 나오는 경우)
_STOPWORDS_STEM = {
    "있", "하", "되", "이", "없", "않", "같", "위", "말", "보",
    "오", "가", "나", "받", "통", "되다", "하다", "이다",
}

# 문장 어미 / 서술어 활용형 (tokens에 활용형이 그대로 남는 경우)
_STOPWORDS_ENDINGS = {
    "있다", "없다", "한다", "된다", "됩니다", "합니다", "입니다",
    "됐다", "했다", "였다", "있습니다", "없습니다", "않다", "같다",
    "크다", "작다", "많다", "적다", "높다", "낮다", "좋다", "나쁘다",
    "는데", "는데요", "지만", "으며", "이며", "거나", "든지",
    "이고", "하고", "이며", "으로서", "로서",
    "밝혔다", "말했다", "전했다", "설명했다", "강조했다", "덧붙였다",
    "지적했다", "밝혀졌다", "알려졌다", "나타났다", "드러났다",
}

# 접속사 / 부사
_STOPWORDS_CONJUNCTIONS = {
    "그리고", "그러나", "그런데", "그래서", "또한", "하지만",
    "따라서", "왜냐하면", "그러므로", "결국", "즉", "또", "및",
    "더욱", "더욱이", "오히려", "특히", "다만", "단지",
}

# 뉴스 상투어 (시간/맥락)
_STOPWORDS_TIME = {
    "이번", "지난", "최근", "오늘", "내일", "어제", "현재",
    "올해", "지금", "당시", "이후", "이전", "앞서", "이날",
    "오는", "올초", "연초", "연말", "상반기", "하반기",
}

# 뉴스 상투어 (관계/방향 — 형태소 분리 후 단독 토큰)
_STOPWORDS_RELATIONAL = {
    "대한", "관련", "통해", "위해", "따른", "위한", "관한",
    "의한", "바탕", "가운데", "경우", "것으로", "것이다",
    "때문", "통한", "위하여", "대해", "따라", "해당",
    "일부", "모두", "전체", "각각", "일정", "다양",
}

# 언론/저작권 메타어
_STOPWORDS_MEDIA = {
    "기자", "뉴스", "기사", "보도", "사진", "제공", "출처",
    "무단", "전재", "재배포", "금지", "저작권", "연합뉴스",
    "취재", "인터뷰", "copyright", "reserved", "rights",
    "ⓒ", "all",
}

# URL / 도메인 조각
_STOPWORDS_URL = {
    "www", "com", "co", "kr", "net", "org", "http", "https",
    "html", "php", "asp", "jsp",
}

# 단위 (숫자와 결합해서 토큰이 분리되는 경우)
_STOPWORDS_UNITS = {
    "억원", "만원", "조원", "달러", "유로", "퍼센트", "위안",
    "엔화", "파운드",
}

# 최종 통합 불용어 SET
STOPWORDS: set[str] = (
    _STOPWORDS_STEM
    | _STOPWORDS_ENDINGS
    | _STOPWORDS_CONJUNCTIONS
    | _STOPWORDS_TIME
    | _STOPWORDS_RELATIONAL
    | _STOPWORDS_MEDIA
    | _STOPWORDS_URL
    | _STOPWORDS_UNITS
)

# ─────────────────────────────────────────────────────────────
# 2자 단어 허용 예외 (경제 도메인 핵심어)
# 2자 이하 단어는 기본 차단 후 여기에 있는 단어만 허용
# ─────────────────────────────────────────────────────────────
_ALLOWED_SHORT_WORDS = {
    # 금융/시장 지표
    "금리", "주가", "환율", "채권", "주식", "물가", "환율",
    "코스피", "코스닥", "나스닥", "다우",
    # 경제 주체
    "기업", "은행", "정부", "국회", "법원", "청와대",
    # 업종/분야
    "반도체", "배터리", "바이오", "부동산", "원자재",
    # 기타 자주 쓰이는 2자 경제어
    "수출", "수입", "투자", "융자", "금융", "재정", "예산",
    "세금", "세율", "적자", "흑자", "성장", "침체", "경기",
    "물가", "임금", "고용", "실업", "소비", "생산", "무역",
}


# ─────────────────────────────────────────────────────────────
# 동적 불용어 로드
# ─────────────────────────────────────────────────────────────

def _load_dynamic_stopwords(es) -> set[str]:
    """
    ES keyword_stopwords 인덱스에서 불용어 목록을 로드.
    배치 시작 시 1회만 호출해 메모리에 올린다.
    인덱스가 없거나 오류 시 빈 set 반환 (파이프라인 중단 없음).
    """
    try:
        res = es.search(
            index=INDEX_STOPWORDS,
            body={"query": {"match_all": {}}, "size": 10000},
        )
        words = {hit["_source"]["word"] for hit in res["hits"]["hits"]}
        if words:
            logger.info(f"동적 불용어 로드: {len(words)}개")
        return words
    except Exception as e:
        logger.warning(f"동적 불용어 로드 실패 (기본 불용어로 진행): {e}")
        return set()


# ─────────────────────────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """
    URL / 이메일 / 도메인 / 숫자 / 특수문자를 제거한 뒤 공백 정규화.
    title 전처리와 tokens 개별 정제 양쪽에서 사용.
    """
    # 1) URL (http/https/ftp 스킴 포함)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"ftp://\S+", " ", text)
    # 2) www로 시작하는 도메인
    text = re.sub(r"www\.\S+", " ", text)
    # 3) 이메일
    text = re.sub(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", " ", text)
    # 4) 도메인처럼 보이는 패턴 (영문.영문 조합 — news.com, naver.kr 등)
    text = re.sub(r"\b[a-zA-Z0-9-]+\.[a-zA-Z]{2,6}\b", " ", text)
    # 5) 숫자 (단독 숫자 토큰, 콤마/점 포함 수치 모두)
    text = re.sub(r"\b[\d,.\-]+\b", " ", text)
    # 6) 특수문자 (한글·영문·숫자·공백 외 전부)
    text = re.sub(r"[^\w\s가-힣]", " ", text)
    # 7) 공백 정규화
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# 하위 호환용 alias (title 전처리에서 기존 이름으로 호출)
_preprocess = _clean_text


def _clean_token(token: str) -> str | None:
    """ES tokens 배열의 개별 토큰에 _clean_text 적용. 정제 후 빈 문자열이면 None 반환."""
    cleaned = _clean_text(token).strip()
    return cleaned if cleaned else None


# 동사/형용사 어간 말음 패턴
# Nori 분석 후 단독 토큰으로 남는 동사 어간은 대부분 이 글자로 끝남
_VERB_ENDING_RE = re.compile(
    r"[하되이아어고며서지않받줄쓸올갈올낼쓸줄살볼]$"
)

def _is_noun_like(word: str) -> bool:
    """
    명사성 토큰 근사 판별.
    - 영문은 무조건 허용 (AI, ESG, GDP 등)
    - 한글은 동사 어간 말음 패턴으로 끝나면 제외
    """
    # 영문 단독 토큰은 허용
    if re.fullmatch(r"[a-zA-Z]+", word):
        return True
    # 동사 어간 패턴으로 끝나는 한글 토큰 제외
    if _VERB_ENDING_RE.search(word):
        return False
    return True


def _is_valid_keyword(word: str, stopwords: set | None = None) -> bool:
    """키워드 유효성 검사. stopwords 미전달 시 모듈 기본 STOPWORDS 사용."""
    sw = stopwords if stopwords is not None else STOPWORDS
    # 길이 체크
    if len(word) < MIN_WORD_LEN:
        return False
    # 숫자·특수문자 혼합 토큰 제거 (ex: 3분기, 2024년 — 숫자 포함 토큰)
    if re.search(r"\d", word):
        return False
    # 알파벳/한글이 하나도 없는 토큰 제거
    if not any(c.isalpha() for c in word):
        return False
    # 불용어 체크 (소문자 변환 후 비교 — 영문 대소문자 대응)
    if word in sw or word.lower() in sw:
        return False
    # 2자 한글 단어: 허용 예외가 아니면 차단
    if re.fullmatch(r"[가-힣]{1,2}", word) and word not in _ALLOWED_SHORT_WORDS:
        return False
    # 명사성 토큰 필터
    if not _is_noun_like(word):
        return False
    return True


def _remove_josa(word: str) -> str:
    """단어 끝의 조사 제거"""
    josa_pattern = r"(을|를|이|가|은|는|에서|에게|에|의|로부터|으로|로|와|과|도|만|까지|부터|한테|께서)$"
    return re.sub(josa_pattern, "", word)


# ─────────────────────────────────────────────────────────────
# 키워드 추출
# ─────────────────────────────────────────────────────────────

def extract_keywords(title: str, tokens: list, max_keywords: int = MAX_KEYWORDS, stopwords: set | None = None) -> list[str]:
    """
    tokens: ES에 저장된 형태소 분석 결과 리스트
    title: 뉴스 제목 (가중치 적용)
    stopwords: 배치 단위 동적 불용어 합산 set (미전달 시 기본 STOPWORDS)
    """
    # tokens 필터링: URL/이메일/숫자/특수문자 정제 후 유효 단어만 추출
    valid_tokens = []
    for t in tokens:
        if not isinstance(t, str):
            continue
        cleaned = _clean_token(t)
        if cleaned and re.fullmatch(r"[가-힣a-zA-Z]+", cleaned) and _is_valid_keyword(cleaned, stopwords):
            valid_tokens.append(cleaned)

    if not valid_tokens:
        return []

    # 제목 가중치: 제목 토큰 조사 제거 후 TITLE_WEIGHT배 반복
    title_tokens = [
        _remove_josa(t)
        for t in _preprocess(title).split()
        if re.fullmatch(r"[가-힣a-zA-Z0-9]+", t)
    ]
    title_tokens = [t for t in title_tokens if _is_valid_keyword(t, stopwords)]
    title_repeated = title_tokens * int(TITLE_WEIGHT)

    full_tokens = title_repeated + valid_tokens
    full_text   = " ".join(full_tokens)

    if not full_text.strip():
        return []

    vectorizer = TfidfVectorizer(
        analyzer="word",
        token_pattern=r"[가-힣a-zA-Z0-9]{2,}",
        ngram_range=(1, 1),
        max_features=1000,
        sublinear_tf=True,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform([full_text])
    except ValueError:
        return []

    feature_names = vectorizer.get_feature_names_out()
    scores        = tfidf_matrix.toarray()[0]
    word_scores   = [
        (word, score)
        for word, score in zip(feature_names, scores)
        if score > 0 and _is_valid_keyword(word, stopwords)
    ]
    word_scores.sort(key=lambda x: x[1], reverse=True)
    return [word for word, _ in word_scores[:max_keywords]]


# ─────────────────────────────────────────────────────────────
# 단건 재처리 래퍼
# ─────────────────────────────────────────────────────────────

def extract_keywords_single(article_id: str, src: dict) -> None:
    """
    단건 아티클 키워드 추출 후 ES 업데이트.
    classify.py의 reprocess_article 엔드포인트에서 호출됩니다.
    동적 불용어는 단건 실행 시에도 ES에서 로드합니다.
    """
    es = get_es()
    try:
        dynamic_sw      = _load_dynamic_stopwords(es)
        batch_stopwords = STOPWORDS | dynamic_sw
        keywords     = extract_keywords(
            src.get("title", ""),
            src.get("tokens", []),
            stopwords=batch_stopwords,
        )
        keywords_str = ",".join(keywords)
        es.update(
            index=INDEX_NEWS,
            id=article_id,
            body={"doc": {"keywords": keywords_str}},
        )
        logger.info(f"[단건] 키워드 추출 완료 article_id={article_id} → {keywords}")
    finally:
        es.close()


# ─────────────────────────────────────────────────────────────
# 파이프라인 진입점
# ─────────────────────────────────────────────────────────────

def run_keyword_pipeline():
    """keywords=NULL 뉴스를 배치로 가져와 키워드를 추출합니다."""
    try:
        es = get_es()

        res = es.search(
            index=INDEX_NEWS,
            body={
                "query": {"bool": {"must_not": {"exists": {"field": "keywords"}}}},
                "_source": ["article_id", "title", "tokens"],
                "sort":    [{"published_at": "asc"}],
                "size":    BATCH_SIZE,
            },
        )
        hits = res["hits"]["hits"]

        if not hits:
            logger.info("키워드 추출할 뉴스 없음")
            es.close()
            return

        logger.info(f"키워드 추출 시작: {len(hits)}건")

        # 동적 불용어 로드 — 배치 실행마다 새로 로드해 기본 STOPWORDS와 합산
        # 전역 STOPWORDS를 직접 수정하지 않고 배치 범위 내 local set 사용
        dynamic_sw      = _load_dynamic_stopwords(es)
        batch_stopwords = STOPWORDS | dynamic_sw

        for hit in hits:
            src        = hit["_source"]
            article_id = src["article_id"]
            try:
                keywords     = extract_keywords(
                    src.get("title", ""),
                    src.get("tokens", []),
                    stopwords=batch_stopwords,
                )
                keywords_str = ",".join(keywords)
                es.update(
                    index=INDEX_NEWS,
                    id=article_id,
                    body={"doc": {"keywords": keywords_str}},
                )
            except Exception as e:
                logger.error(f"키워드 추출 실패 article_id={article_id}: {e}")
                continue

        es.close()
        logger.info(f"키워드 추출 완료: {len(hits)}건")

    except Exception as e:
        log_pipeline_error(pipeline="keyword", error=e)
        raise
