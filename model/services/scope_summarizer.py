"""
scope 대표 요약 생성 서비스 — KR-FinBERT 문장 임베딩 기반

흐름:
  1. news_scopes에서 scope_summary=NULL scope 조회
  2. news_economy.content 수집 (동일 언론사 최대 3건, 최신순)
  3. 본문 클렌징 (언론사 헤더/기자명/저작권 문구 제거)
  4. 문장 분리 → 중복 제거
  5. KR-FinBERT [CLS] 임베딩으로 각 문장 벡터화
     → 전체 문장 임베딩 평균 = "스콥 중심 벡터"
     → 중심 벡터와 코사인 유사도 가장 높은 문장 선택
  6. 70자 자연 절단
  7. news_scopes.scope_summary upsert

[수정 이력]
- 요약 엔진 교체: TextRank(그래프 기반) → KR-FinBERT 임베딩 코사인 유사도
  · TextRank는 표면적 단어 빈도 기반 → 배경 설명 문장이 높은 점수 받는 문제
  · KR-FinBERT는 의미론적 유사도 기반 → 스콥 전체를 가장 잘 대표하는 문장 선택
- 입력 소스 유지: news_economy.content (summary 미사용)
- 본문 클렌징 유지: 언론사 헤더/기자명/날짜/저작권 문구 제거
- KR-FinBERT 싱글턴: embedding_generator.py와 동일한 모델 재활용
- 문장 수 부족(1~2개) 시 폴백: 임베딩 없이 첫 번째 유효 문장 사용
"""

import logging
import re
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer

from model.database import get_es
from model.services.error_logger import log_pipeline_error

logger = logging.getLogger(__name__)

BATCH_SIZE          = 10000
MAX_NEWS_PER_PRESS  = 3
DUPLICATE_THRESHOLD = 0.85
MAX_SENTENCE_LENGTH = 70
MIN_SENTENCE_LENGTH = 15
INDEX_NEWS          = "news_economy"
INDEX_SCOPES        = "news_scopes"

_TRIM_CHARS = set('다며서고은는이가을를에의')

# KR-FinBERT 모델명 (embedding_generator.py와 동일)
FINBERT_MODEL = "snunlp/KR-FinBert-SC"


# ── KR-FinBERT 싱글턴 ─────────────────────────────────────────────

_tokenizer = None
_model     = None
_device    = None

def _get_finbert():
    global _tokenizer, _model, _device
    if _model is None:
        logger.info("KR-FinBERT 로드 중...")
        _tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        _model     = AutoModel.from_pretrained(FINBERT_MODEL)
        _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model.to(_device)
        _model.eval()
        logger.info(f"KR-FinBERT 로드 완료 (device={_device})")
    return _tokenizer, _model, _device


def _embed_sentences(sentences: list[str]) -> np.ndarray:
    """
    문장 리스트 → KR-FinBERT [CLS] 임베딩 행렬 반환 (shape: N x 768)
    배치 단위로 처리하여 메모리 효율 확보
    """
    tokenizer, model, device = _get_finbert()
    embeddings = []

    EMBED_BATCH = 16  # 문장 임베딩 배치 크기
    for i in range(0, len(sentences), EMBED_BATCH):
        batch = sentences[i: i + EMBED_BATCH]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output = model(**encoded)
        # [CLS] 토큰 임베딩 (첫 번째 토큰)
        cls_vecs = output.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_vecs)

    return np.vstack(embeddings)  # (N, 768)


def _most_central_sentence(sentences: list[str]) -> int:
    """
    KR-FinBERT 임베딩으로 스콥 중심 벡터와 가장 유사한 문장 인덱스 반환.

    중심 벡터 = 전체 문장 임베딩의 평균
    → 스콥 전체 맥락을 가장 잘 대표하는 문장 선택
    """
    matrix  = _embed_sentences(sentences)          # (N, 768)
    centroid = matrix.mean(axis=0, keepdims=True)  # (1, 768)
    sims    = cosine_similarity(centroid, matrix)[0]  # (N,)
    return int(np.argmax(sims))


# ── 본문 클렌징 ────────────────────────────────────────────────────

_CLEAN_PATTERNS = [
    # 방송사 태그: [KBS 춘천], [MBC] 등
    (re.compile(r"^\s*\[[^\]]{1,20}\]\s*"), ""),
    (re.compile(r"\[[^\]]{1,20}\]"), " "),
    # 언론사 발신 헤더: (서울=뉴스1) 홍길동 기자 =
    (re.compile(r"\([^)]{1,20}=[^)]{1,20}\)\s*[가-힣\s]{2,10}기자\s*=?\s*"), ""),
    # 기자 byline
    (re.compile(r"[가-힣]{2,4}[·]?[가-힣]{0,4}\s*기자\s*=?\s*"), ""),
    # ⓒ 저작권 표시
    (re.compile(r"ⓒ\s*[^\s,]{1,20}"), ""),
    # 날짜 패턴
    (re.compile(r"\d{4}[.\-]\d{1,2}[.\-]\d{1,2}"), ""),
    # 저작권 문구
    (re.compile(r"무단\s*전재[^.]*재배포\s*금지[^.]*\.?"), ""),
    (re.compile(r"저작권자[^.]*\.?"), ""),
    (re.compile(r"All rights reserved[^.]*\.?", re.IGNORECASE), ""),
    # 사진/그래픽 캡션
    (re.compile(r"\([^)]{1,30}=[^)]{1,30}\)"), ""),
    # 중복 공백
    (re.compile(r"\s{2,}"), " "),
]


def _clean_content(text: str) -> str:
    for pattern, repl in _CLEAN_PATTERNS:
        text = pattern.sub(repl, text)
    return text.strip()


# ── 문장 처리 ──────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) >= MIN_SENTENCE_LENGTH]


def _remove_duplicates(sentences: list[str]) -> list[str]:
    if len(sentences) < 2:
        return sentences
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 3), max_features=2000
    )
    try:
        matrix = vectorizer.fit_transform(sentences).toarray()
    except ValueError:
        return sentences
    sim            = cosine_similarity(matrix)
    keep, removed  = [], set()
    for i in range(len(sentences)):
        if i in removed:
            continue
        keep.append(sentences[i])
        for j in range(i + 1, len(sentences)):
            if sim[i][j] >= DUPLICATE_THRESHOLD:
                removed.add(j)
    return keep


def _natural_trim(sentence: str, limit: int = MAX_SENTENCE_LENGTH) -> str:
    if len(sentence) <= limit:
        return sentence
    cut = sentence[:limit]
    for ch in range(len(cut) - 1, -1, -1):
        if cut[ch] in _TRIM_CHARS:
            return cut[:ch + 1] + "…"
    return cut[:limit] + "…"


# ── 메인 생성 함수 ─────────────────────────────────────────────────

def generate_scope_summary(es, scope_id: str):
    # 뉴스 content 수집
    res = es.search(
        index=INDEX_NEWS,
        body={
            "query": {
                "bool": {
                    "must": [
                        {"term":   {"scopeID": scope_id}},
                        {"exists": {"field":   "content"}},
                    ]
                }
            },
            "_source": ["content", "press"],
            "sort":    [{"published_at": "desc"}],
            "size":    100,
        },
    )
    rows = [h["_source"] for h in res["hits"]["hits"]]
    if not rows:
        _save(es, scope_id, "")
        return None

    # 동일 언론사 최대 MAX_NEWS_PER_PRESS건
    press_count, filtered = {}, []
    for r in rows:
        press = r.get("press") or "unknown"
        press_count[press] = press_count.get(press, 0) + 1
        if press_count[press] <= MAX_NEWS_PER_PRESS and r.get("content"):
            filtered.append(_clean_content(r["content"]))

    if not filtered:
        _save(es, scope_id, "")
        return None

    # 문장 분리 → 중복 제거
    combined  = " ".join(filtered)
    sentences = _split_sentences(combined)

    if not sentences:
        summary = _natural_trim(combined)
        _save(es, scope_id, summary)
        return summary

    sentences = _remove_duplicates(sentences)

    # 문장 3개 미만이면 임베딩 없이 첫 번째 문장 사용
    if len(sentences) < 3:
        summary = _natural_trim(sentences[0])
        _save(es, scope_id, summary)
        return summary

    # KR-FinBERT 임베딩 기반 중심 문장 선택
    try:
        best_idx = _most_central_sentence(sentences)
    except Exception as e:
        logger.warning(f"임베딩 실패, 첫 문장 폴백 (scope_id={scope_id}): {e}")
        best_idx = 0

    summary = _natural_trim(sentences[best_idx])
    _save(es, scope_id, summary)
    return summary


def _save(es, scope_id: str, summary: str):
    es.update(
        index=INDEX_SCOPES,
        id=scope_id,
        body={"doc": {
            "scope_summary": summary,
            "updated_at":    datetime.now(timezone.utc).isoformat(),
        }},
    )
    logger.info(f"scope_summary 저장: {scope_id} ({len(summary)}자)")


# ── 배치 처리 ─────────────────────────────────────────────────────

def run_scope_summary_batch():
    """scope_summary 없는 scope 전체를 배치 루프로 처리합니다."""
    try:
        es = get_es()

        # 배치 시작 전 모델 미리 로드 (첫 scope에서 지연 방지)
        _get_finbert()

        total_processed = 0
        batch_num       = 0

        while True:
            batch_num += 1

            res = es.search(
                index=INDEX_SCOPES,
                body={
                    "query": {
                        "bool": {
                            "must_not": {"exists": {"field": "scope_summary"}}
                        }
                    },
                    "_source": ["scopeID"],
                    "sort":    [{"created_at": "asc"}],
                    "size":    BATCH_SIZE,
                },
            )
            hits = res["hits"]["hits"]

            if not hits:
                if batch_num == 1:
                    logger.info("scope_summary 처리할 scope 없음")
                else:
                    logger.info(f"scope_summary 전체 완료 | total={total_processed}")
                break

            logger.info(f"[배치 {batch_num}] scope_summary 처리 시작: {len(hits)}건")

            processed_in_batch = 0
            for hit in hits:
                scope_id = hit["_source"]["scopeID"]
                try:
                    generate_scope_summary(es, scope_id)
                    total_processed    += 1
                    processed_in_batch += 1
                except Exception as e:
                    logger.error(f"scope_summary 생성 실패 scopeID={scope_id}: {e}")
                    try:
                        _save(es, scope_id, "")
                    except Exception:
                        pass
                    continue

            es.indices.refresh(index=INDEX_SCOPES)
            logger.info(f"[배치 {batch_num}] 완료 | 누적 processed={total_processed}")

            if processed_in_batch == 0:
                logger.warning("배치에서 처리된 건 없음, 루프 강제 탈출")
                break

        es.close()

    except Exception as e:
        log_pipeline_error(pipeline="scope_summarizer", error=e)
        raise
