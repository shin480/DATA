"""
KR-FinBert-SC 감성 분류 서비스

흐름:
  1. ES news_economy에서 sentiment=NULL / scopeID 존재 레코드 polling
  2. KR-FinBert-SC 추론 (한국 경제 뉴스 특화 모델)
  3. sentiment_labels(MySQL 관리) 키워드로 점수 보정
  4. news_economy.sentiment, sentiment_score upsert

[수정 이력]
- 모델 교체: monologg/koelectra-base-finetuned-sentiment → snunlp/KR-FinBert-SC
  · 기존 모델: 네이버 영화 리뷰 기반 2-class → 뉴스 도메인 부적합
  · 변경 모델: 한국 경제 뉴스 72개 매체 + 증권사 애널리스트 리포트 학습, 2-class
- IDX_TO_LABEL: 모델 config.id2label 기준 동적 로딩으로 변경 (하드코딩 제거)
- neutral 처리: gap 기반 후처리 유지 (NEUTRAL_GAP_THRESHOLD 기본값 0.2)
- 토크나이저: ElectraTokenizer → BertTokenizer 변경
- [2026-05] classify_single_article 추가: 단건 재처리용 래퍼
- [2026-05] 예외 시 sentiment 폴백 저장: 추론 실패 아티클이 배치마다 재조회되는 무한루프 방지
"""

import logging
from typing import Optional

import torch
from transformers import BertForSequenceClassification, BertTokenizer

from model.database import get_es
from model.services.error_logger import log_pipeline_error

from util.logger import save_article_process_log,save_article_error_log

logger = logging.getLogger(__name__)

MODEL_NAME   = "snunlp/KR-FinBert-SC"
BATCH_SIZE   = 10000
MAX_LENGTH   = 512

# neutral 판정 임계값: negative/positive 확률 차이 < 이 값이면 neutral
# 0.2 → 0.15 → 0.1로 조정 (neutral 과다 보정)
NEUTRAL_GAP_THRESHOLD = 0.1

KEYWORD_BOOST = 0.15

INDEX_NEWS = "news_economy"

_tokenizer: Optional[BertTokenizer]                  = None
_model:     Optional[BertForSequenceClassification]  = None
_device:    Optional[torch.device]                   = None
_id2label:  Optional[dict]                           = None  # config에서 동적 로딩


def _load_model():
    global _tokenizer, _model, _device, _id2label
    if _model is not None:
        return
    logger.info(f"KR-FinBert-SC 모델 로딩: {MODEL_NAME}")
    _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    _model     = BertForSequenceClassification.from_pretrained(MODEL_NAME)
    _model.to(_device)
    _model.eval()

    # 라벨 매핑을 모델 config에서 동적으로 로딩 (하드코딩 오류 방지)
    _id2label = {int(k): v.lower() for k, v in _model.config.id2label.items()}
    logger.info(f"모델 로딩 완료 (device: {_device}, labels: {_id2label})")


def _load_keyword_dict() -> dict[str, tuple[str, float]]:
    """
    sentiment_labels 키워드는 Admin CSV 업로드로 관리합니다.
    현재는 routers/admin.py 에서 MySQL sentiment_labels 테이블로 관리하므로
    MySQL에서 읽어옵니다.
    테이블이 없거나 연결 실패 시 빈 딕셔너리 반환 (키워드 보정 없이 진행).
    """
    try:
        from model.database import get_db
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT keyword, sentiment, weight FROM sentiment_labels")
                rows = cur.fetchall()
        return {r["keyword"]: (r["sentiment"], r["weight"]) for r in rows}
    except Exception as e:
        logger.warning(f"sentiment_labels 로드 실패, 키워드 보정 없이 진행: {e}")
        return {}


def _apply_keyword_boost(probs: dict, text: str, keyword_dict: dict) -> dict:
    """
    키워드 보정: negative/positive 확률에만 적용.
    neutral은 모델 출력이 아닌 후처리로 결정되므로 보정 대상에서 제외.
    """
    boosted = dict(probs)
    for keyword, (sentiment, weight) in keyword_dict.items():
        if keyword in text and sentiment in boosted:
            boost = KEYWORD_BOOST * weight
            boosted[sentiment] = min(1.0, boosted[sentiment] + boost)
    total = sum(boosted.values())
    if total > 0:
        boosted = {k: v / total for k, v in boosted.items()}
    return boosted


def _apply_gap_neutral(probs: dict) -> tuple[str, float]:
    """
    gap 기반 neutral 후처리.
    - negative/positive 확률 차이 < NEUTRAL_GAP_THRESHOLD → neutral
    - sentiment_score: neutral이면 1 - gap (모델 불확실성 반영)
    """
    p_neg = probs.get("negative", 0.0)
    p_pos = probs.get("positive", 0.0)
    gap   = abs(p_pos - p_neg)

    if gap < NEUTRAL_GAP_THRESHOLD:
        return "neutral", round(1.0 - gap, 6)
    elif p_pos > p_neg:
        return "positive", round(p_pos, 6)
    else:
        return "negative", round(p_neg, 6)


def predict_single(title: str, content: str, keyword_dict: dict) -> tuple[str, float]:
    _load_model()
    text   = f"{title} {content[:300]}"
    inputs = _tokenizer(text, return_tensors="pt", max_length=MAX_LENGTH,
                        truncation=True, padding=True)
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs_t = torch.softmax(logits, dim=1)[0].cpu().numpy()
    probs   = {_id2label[i]: float(p) for i, p in enumerate(probs_t)}
    if keyword_dict:
        probs = _apply_keyword_boost(probs, title + content[:300], keyword_dict)
    return _apply_gap_neutral(probs)


# ── 단건 재처리 래퍼 ───────────────────────────────────────────

def classify_single_article(article_id: str, src: dict) -> None:
    """
    단건 아티클 감성 분류 후 ES 업데이트.
    classify.py의 reprocess_article 엔드포인트에서 호출됩니다.

    Args:
        article_id: news_economy 문서 ID
        src: ES _source dict (title, content 필드 포함)

    Raises:
        Exception: 모델 추론 실패 또는 ES 업데이트 실패 시
    """
    keyword_dict = _load_keyword_dict()
    sentiment, score = predict_single(
        src.get("title", ""),
        src.get("content", ""),
        keyword_dict,
    )
    es = get_es()
    try:
        es.update(
            index=INDEX_NEWS,
            id=article_id,
            body={"doc": {
                "sentiment":       sentiment,
                "sentiment_score": round(score, 6),
            }},
        )
        logger.info(f"[단건] 감성 분류 완료 article_id={article_id} → {sentiment} ({score:.4f})")
    finally:
        es.close()


# ── 배치 파이프라인 진입점 ─────────────────────────────────────

def run_sentiment_pipeline():
    """sentiment=NULL / scopeID 존재 뉴스 전체를 배치 루프로 감성 분류합니다."""
    try:
        es = get_es()

        keyword_dict = _load_keyword_dict()
        logger.info(f"Admin 키워드 {len(keyword_dict)}개 로드")

        total_processed = 0
        batch_num       = 0

        while True:
            batch_num += 1

            res = es.search(
                index=INDEX_NEWS,
                body={
                    "query": {
                        "bool": {
                            "must":     {"exists": {"field": "scopeID"}},
                            "must_not": {"exists": {"field": "sentiment"}},
                        }
                    },
                    "_source": ["article_id", "title", "content"],
                    "sort":    [{"published_at": "asc"}],
                    "size":    BATCH_SIZE,
                },
            )
            hits = res["hits"]["hits"]

            if not hits:
                if batch_num == 1:
                    logger.info("감성 분류할 뉴스 없음")
                else:
                    logger.info(f"감성 분류 전체 완료 | total={total_processed}")
                break

            logger.info(f"[배치 {batch_num}] 감성 분류 시작: {len(hits)}건")

            for hit in hits:
                src        = hit["_source"]
                article_id = src["article_id"]
                try:
                    sentiment, score = predict_single(
                        src.get("title", ""),
                        src.get("content", ""),
                        keyword_dict,
                    )
                    es.update(
                        index=INDEX_NEWS,
                        id=article_id,
                        body={"doc": {
                            "sentiment":       sentiment,
                            "sentiment_score": round(score, 6),
                        }},
                    )
                    save_article_process_log("A201", article_id, "success")
                    total_processed += 1
                except Exception as e:
                    logger.error(f"감성 분류 실패 article_id={article_id}: {e}")
                    history_id = save_article_process_log("A201", article_id, "fail")
                    save_article_error_log(history_id, "E004", f"감성 분류 실패 article_id={article_id}: {e}")
                    try:
                        es.update(
                            index=INDEX_NEWS,
                            id=article_id,
                            body={"doc": {
                                "sentiment":       "neutral",
                                "sentiment_score": 0.0,
                            }},
                        )
                    except Exception:
                        pass
                    continue

            es.indices.refresh(index=INDEX_NEWS)
            logger.info(f"[배치 {batch_num}] 완료 | 누적 processed={total_processed}")

        es.close()

    except Exception as e:
        log_pipeline_error(pipeline="sentiment", error=e)
        raise
