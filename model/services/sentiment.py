"""
KoELECTRA 감성 분류 서비스

흐름:
  1. ES news_economy에서 sentiment=NULL / scopeID 존재 레코드 polling
  2. KoELECTRA 추론
  3. sentiment_labels(ES 관리) 키워드로 점수 보정
  4. news_economy.sentiment, sentiment_score upsert
"""

import logging
from typing import Optional

import torch
from transformers import ElectraForSequenceClassification, ElectraTokenizer

from model.database import get_es
from model.services.error_logger import log_pipeline_error

logger = logging.getLogger(__name__)

MODEL_NAME   = "monologg/koelectra-base-finetuned-sentiment"
BATCH_SIZE   = 1000
MAX_LENGTH   = 512
IDX_TO_LABEL = {0: "negative", 1: "neutral", 2: "positive"}
KEYWORD_BOOST = 0.15

INDEX_NEWS   = "news_economy"

_tokenizer: Optional[ElectraTokenizer] = None
_model:     Optional[ElectraForSequenceClassification] = None
_device:    Optional[torch.device] = None


def _load_model():
    global _tokenizer, _model, _device
    if _model is not None:
        return
    logger.info(f"KoELECTRA 모델 로딩: {MODEL_NAME}")
    _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)
    _model     = ElectraForSequenceClassification.from_pretrained(MODEL_NAME)
    _model.to(_device)
    _model.eval()
    logger.info(f"모델 로딩 완료 (device: {_device})")


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


def _apply_keyword_boost(probs, text, keyword_dict):
    boosted = dict(probs)
    for keyword, (sentiment, weight) in keyword_dict.items():
        if keyword in text:
            boost = KEYWORD_BOOST * weight
            boosted[sentiment] = min(1.0, boosted[sentiment] + boost)
    total = sum(boosted.values())
    if total > 0:
        boosted = {k: v / total for k, v in boosted.items()}
    return boosted


def predict_single(title: str, content: str, keyword_dict: dict) -> tuple[str, float]:
    _load_model()
    text   = f"{title} {content[:300]}"
    inputs = _tokenizer(text, return_tensors="pt", max_length=MAX_LENGTH,
                        truncation=True, padding=True)
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs_t = torch.softmax(logits, dim=1)[0].cpu().numpy()
    probs   = {IDX_TO_LABEL[i]: float(p) for i, p in enumerate(probs_t)}
    if keyword_dict:
        probs = _apply_keyword_boost(probs, title + content[:300], keyword_dict)
    sentiment = max(probs, key=probs.get)
    return sentiment, probs[sentiment]


def run_sentiment_pipeline():
    """sentiment=NULL / scopeID 존재 뉴스를 배치 감성 분류합니다."""
    try:
        es = get_es()

        keyword_dict = _load_keyword_dict()
        logger.info(f"Admin 키워드 {len(keyword_dict)}개 로드")

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
            logger.info("감성 분류할 뉴스 없음")
            es.close()
            return

        logger.info(f"감성 분류 시작: {len(hits)}건")

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
            except Exception as e:
                logger.error(f"감성 분류 실패 article_id={article_id}: {e}")
                continue

        es.close()
        logger.info(f"감성 분류 완료: {len(hits)}건")

    except Exception as e:
        log_pipeline_error(pipeline="sentiment", error=e)
        raise
