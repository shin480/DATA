"""
embedding 생성 서비스

흐름:
  1. ES news_economy에서 embedding=NULL / clean_text 존재 레코드 polling
  2. KR-FinBert-SC의 [CLS] 토큰 벡터를 embedding으로 추출 (768차원)
  3. L2 정규화 후 news_economy.embedding 필드에 bulk upsert
  4. embedding 없는 기사가 없을 때까지 BATCH_SIZE 단위로 반복 처리

모델 선택 근거:
  - 이미 감성 분류에서 로딩된 KR-FinBert-SC 재사용 -> 추가 모델 불필요
  - [CLS] 토큰: BERT 계열 모델에서 문장 전체 의미를 압축한 벡터
  - 768차원: news_economy 매핑의 dense_vector dims=768 과 일치
"""

import logging
from typing import Optional

import numpy as np
import torch
from elasticsearch.helpers import bulk
from transformers import BertForSequenceClassification, BertTokenizer

from model.database import get_es
from model.services.error_logger import log_pipeline_error

logger = logging.getLogger(__name__)

MODEL_NAME = "snunlp/KR-FinBert-SC"
BATCH_SIZE = 10000
MAX_LENGTH = 512
EMBED_DIM  = 768
INDEX_NEWS = "news_economy"

_tokenizer: Optional[BertTokenizer]                 = None
_model:     Optional[BertForSequenceClassification] = None
_device:    Optional[torch.device]                  = None


def _load_model():
    global _tokenizer, _model, _device
    if _model is not None:
        return
    logger.info(f"KR-FinBert-SC 모델 로딩 (embedding용): {MODEL_NAME}")
    _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    _model     = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, output_hidden_states=True
    )
    _model.to(_device)
    _model.eval()
    logger.info(f"모델 로딩 완료 (device: {_device})")


def _extract_cls_vector(text: str) -> Optional[np.ndarray]:
    """
    텍스트를 KR-FinBert-SC에 입력하고 [CLS] 토큰의 마지막 hidden state를
    L2 정규화하여 반환한다.

    Returns:
        np.ndarray: shape=(768,), dtype=float32 또는 None (오류 시)
    """
    try:
        inputs = _tokenizer(
            text,
            return_tensors="pt",
            max_length=MAX_LENGTH,
            truncation=True,
            padding=True,
        )
        inputs = {k: v.to(_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _model(**inputs)

        # 마지막 hidden state의 [CLS] 토큰 (index 0)
        cls_vec = outputs.hidden_states[-1][0][0].cpu().numpy().astype(np.float32)

        norm = np.linalg.norm(cls_vec)
        if norm == 0:
            return None

        return (cls_vec / norm).astype(np.float32)

    except Exception as e:
        logger.error(f"[CLS] 벡터 추출 실패: {e}")
        return None


def run_embedding_pipeline() -> dict:
    """
    embedding이 없는 뉴스 전체를 BATCH_SIZE 단위로 반복 처리한다.
    embedding이 없는 기사가 없을 때까지 루프를 돌며 전량 처리한다.
    """
    try:
        _load_model()
        es = get_es()

        total_processed = 0
        total_skipped   = 0
        batch_num       = 0

        while True:
            batch_num += 1

            # 1. embedding 없는 기사 조회 (배치 단위)
            res = es.search(
                index=INDEX_NEWS,
                body={
                    "query": {
                        "bool": {
                            "must":     [{"exists": {"field": "clean_text"}}],
                            "must_not": [{"exists": {"field": "embedding"}}],
                        }
                    },
                    "_source": ["article_id", "title", "clean_text"],
                    "sort":    [{"collected_at": {"order": "asc"}}],
                    "size":    BATCH_SIZE,
                },
            )
            hits = res["hits"]["hits"]

            if not hits:
                if batch_num == 1:
                    logger.info("embedding 생성할 뉴스 없음")
                else:
                    logger.info(
                        f"전체 처리 완료 | "
                        f"total_processed={total_processed}, total_skipped={total_skipped}"
                    )
                break

            logger.info(f"[배치 {batch_num}] embedding 생성 시작: {len(hits)}건")

            bulk_actions    = []
            batch_processed = 0
            batch_skipped   = 0

            # 2. 기사별 [CLS] 벡터 추출
            for hit in hits:
                src        = hit["_source"]
                doc_id     = hit["_id"]
                article_id = src.get("article_id") or doc_id
                clean_text = src.get("clean_text", "").strip()
                title      = src.get("title", "")

                # clean_text 없으면 title로 fallback
                text = clean_text if clean_text else title
                if not text:
                    logger.warning(f"텍스트 없음 -> 스킵: article_id={article_id}")
                    batch_skipped += 1
                    continue

                vec = _extract_cls_vector(text)

                if vec is None:
                    logger.warning(f"embedding 추출 실패 -> 스킵: article_id={article_id}")
                    batch_skipped += 1
                    continue

                bulk_actions.append({
                    "_op_type": "update",
                    "_index":   INDEX_NEWS,
                    "_id":      doc_id,
                    "doc":      {"embedding": vec.tolist()},
                })
                batch_processed += 1

                if (total_processed + batch_processed) % 500 == 0:
                    logger.info(
                        f"embedding 진행 중: "
                        f"총 {total_processed + batch_processed}건 처리됨"
                    )

            # 3. bulk upsert
            if bulk_actions:
                success, errors = bulk(
                    es,
                    bulk_actions,
                    raise_on_error=False,
                    stats_only=False,
                )
                if errors:
                    logger.warning(
                        f"[배치 {batch_num}] bulk update 일부 실패: {len(errors)}건"
                    )

            # 4. refresh 후 다음 배치 조회에 반영
            es.indices.refresh(index=INDEX_NEWS)

            total_processed += batch_processed
            total_skipped   += batch_skipped

            logger.info(
                f"[배치 {batch_num}] 완료 | "
                f"processed={batch_processed}, skipped={batch_skipped} | "
                f"누적 processed={total_processed}"
            )

        es.close()

        return {
            "success":   True,
            "processed": total_processed,
            "skipped":   total_skipped,
        }

    except Exception as e:
        log_pipeline_error(pipeline="embedding", error=e)
        raise
