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
"""

import logging
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from model.database import get_es
from model.services.error_logger import log_pipeline_error

logger = logging.getLogger(__name__)

BATCH_SIZE   = 1000
MAX_KEYWORDS = 5
TITLE_WEIGHT = 2.0
MIN_WORD_LEN = 2
INDEX_NEWS   = "news_economy"

# 한국어 불용어: 고빈도 동사/형용사 어간, 조사, 의존명사 등
STOPWORDS = {
    # 동사/형용사 어간
    "있", "하", "되", "이", "없", "않", "같", "위", "말", "보",
    "오", "가", "나", "받", "통", "위해", "대해", "따라", "통해",
    "위한", "대한", "관련", "경우", "등", "및", "또", "더", "수",
    # 완성형 불용어
    "있다", "하다", "됩니다", "합니다", "있습니다", "했다", "된다",
    "한다", "이다", "않다", "없다", "같다", "크다", "작다", "많다",
    "이번", "지난", "올해", "올해는", "현재", "이후", "이전", "최근",
    "오는", "지금", "당시", "이날", "해당", "일부", "모두", "전체",
    "것으로", "것이다", "때문", "가운데", "바탕", "통한", "위하여",
    # 기자/언론 관련
    "기자", "기사", "보도", "취재", "인터뷰", "제공", "연합뉴스",
    # 단위/숫자 관련
    "억원", "만원", "조원", "달러", "유로", "퍼센트",
}


def _preprocess(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _is_valid_keyword(word: str) -> bool:
    if len(word) < MIN_WORD_LEN:
        return False
    if re.fullmatch(r"\d+", word):
        return False
    if word in STOPWORDS:
        return False
    # 단순 어간 패턴 제거 (1~2자 한글 단독)
    if re.fullmatch(r"[가-힣]{1,2}", word) and word not in {"금리", "주가", "환율", "채권", "주식", "부동산"}:
        return False
    return True


def _remove_josa(word: str) -> str:
    """단어 끝의 조사 제거"""
    josa_pattern = r"(을|를|이|가|은|는|에서|에게|에|의|로부터|으로|로|와|과|도|만|까지|부터|한테|께서)$"
    return re.sub(josa_pattern, "", word)


def extract_keywords(title: str, tokens: list, max_keywords: int = MAX_KEYWORDS) -> list[str]:
    """
    tokens: ES에 저장된 형태소 분석 결과 리스트
    title: 뉴스 제목 (가중치 적용)
    """
    # tokens 필터링: 의미 있는 단어만 추출
    valid_tokens = [
        t for t in tokens
        if isinstance(t, str)
        and re.fullmatch(r"[가-힣a-zA-Z0-9]+", t)
        and _is_valid_keyword(t)
    ]

    if not valid_tokens:
        return []

    # 제목 가중치: 제목 토큰 조사 제거 후 TITLE_WEIGHT배 반복
    title_tokens = [
        _remove_josa(t)
        for t in _preprocess(title).split()
        if re.fullmatch(r"[가-힣a-zA-Z0-9]+", t)
    ]
    title_tokens = [t for t in title_tokens if _is_valid_keyword(t)]
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
        if score > 0 and _is_valid_keyword(word)
    ]
    word_scores.sort(key=lambda x: x[1], reverse=True)
    return [word for word, _ in word_scores[:max_keywords]]


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

        for hit in hits:
            src        = hit["_source"]
            article_id = src["article_id"]
            try:
                keywords     = extract_keywords(
                    src.get("title", ""),
                    src.get("tokens", []),
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
