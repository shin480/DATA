"""
KoELECTRA 감성 분류 Fine-tuning 파이프라인

변경사항:
  - correction_log / finetune_history / news(title+content 조회) → MySQL 유지
    (정정 워크플로우는 MySQL 기반 admin 페이지에서 관리)
  - 단, fine-tuning 학습에 필요한 원문(title, content)은 ES news_economy에서 조회
"""

import logging
import os
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from model.database import get_db, get_es
from model.services.error_logger import log_pipeline_error

logger = logging.getLogger(__name__)

FINETUNE_THRESHOLD = int(os.getenv("FINETUNE_THRESHOLD", 500))
BASE_MODEL_NAME    = "snunlp/KR-FinBert-SC"
MODEL_SAVE_DIR     = Path(os.getenv("MODEL_SAVE_DIR", "models"))
EPOCHS             = 3
LEARNING_RATE      = 2e-5
MAX_LENGTH         = 128
BATCH_SIZE         = 16

LABEL_TO_IDX = {"negative": 0, "neutral": 1, "positive": 2}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}


class CorrectionDataset(Dataset):
    """
    Lazy 토크나이징 방식 — __getitem__ 호출 시 건별 처리.
    기존 방식(전체 일괄 토크나이징)은 2000건 이상에서 메모리 초과로
    프로세스가 죽는 문제가 있어 수정.
    """
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts      = texts
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc.get("token_type_ids",
                              torch.zeros(self.max_length, dtype=torch.long)).squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# 균형 보완용 고확신 샘플 수 (sentiment별 최대 건수)
BALANCE_SAMPLE_SIZE      = 100
BALANCE_SCORE_THRESHOLD  = 0.8  # sentiment_score >= 이 값인 것만 사용


def _fetch_balance_samples(sentiment: str, size: int) -> tuple[list[str], list[int]]:
    """
    ES에서 고확신도 긍정/부정 기사를 샘플링하여 균형 데이터로 활용.
    사람 검토 없이 모델 확신도가 높은 것만 레이블로 신뢰.
    """
    es = get_es()
    try:
        res = es.search(
            index="news_economy",
            body={
                "query": {
                    "bool": {
                        "must": [
                            {"term":  {"sentiment": sentiment}},
                            {"range": {"sentiment_score": {"gte": BALANCE_SCORE_THRESHOLD}}},
                            {"exists": {"field": "content"}},
                        ]
                    }
                },
                "_source": ["title", "content"],
                "sort":    [{"sentiment_score": "desc"}],
                "size":    size,
            },
        )
    finally:
        es.close()

    texts, labels = [], []
    for h in res["hits"]["hits"]:
        src  = h["_source"]
        text = f"{src.get('title', '')} {src.get('content', '')[:200]}"
        texts.append(text)
        labels.append(LABEL_TO_IDX[sentiment])

    logger.info(f"균형 샘플 추가: {sentiment} {len(texts)}건 (score >= {BALANCE_SCORE_THRESHOLD})")
    return texts, labels


def _fetch_correction_data(conn) -> tuple[list[str], list[int], list[int]]:
    """
    1. correction_log(MySQL)에서 미사용 정정 데이터 조회
    2. ES에서 고확신도 긍정/부정 샘플 추가 (균형 보완)
    3. 전체를 합쳐서 반환

    log_ids는 correction_log 항목만 추적 (균형 샘플은 used_in_finetune 업데이트 불필요)
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, newsID, corrected_sentiment
            FROM correction_log
            WHERE used_in_finetune = 0
            ORDER BY created_at ASC
        """)
        rows = cur.fetchall()

    if not rows:
        return [], [], []

    # ES에서 정정 원문 일괄 조회
    es = get_es()
    article_ids = [r["newsID"] for r in rows]
    mget_res    = es.mget(index="news_economy",
                          body={"ids": article_ids},
                          _source=["article_id", "title", "content"])
    es.close()

    docs = {d["_id"]: d["_source"] for d in mget_res["docs"] if d.get("found")}

    texts, labels, log_ids = [], [], []
    for r in rows:
        doc = docs.get(r["newsID"])
        if not doc:
            continue
        text = f"{doc.get('title', '')} {doc.get('content', '')[:200]}"
        texts.append(text)
        labels.append(LABEL_TO_IDX[r["corrected_sentiment"]])
        log_ids.append(r["id"])

    # 균형 보완: ES 고확신도 긍정/부정 샘플 추가
    for sentiment in ("positive", "negative"):
        b_texts, b_labels = _fetch_balance_samples(sentiment, BALANCE_SAMPLE_SIZE)
        texts  += b_texts
        labels += b_labels

    logger.info(f"파인튜닝 데이터: 정정 {len(log_ids)}건 + 균형 보완 {len(texts) - len(log_ids)}건 = 총 {len(texts)}건")
    return texts, labels, log_ids


def _get_current_model_path() -> str:
    """
    가장 최근 파인튜닝 체크포인트를 반환.
    필수 파일(config.json, model 가중치)이 없거나 용량이 비정상이면
    BASE_MODEL_NAME으로 폴백 — 불완전 저장된 디렉토리로 인한
    'header too small' 오류 방지.
    """
    if not MODEL_SAVE_DIR.exists():
        return BASE_MODEL_NAME

    # 필수 파일: config.json + 가중치 파일(safetensors 우선, 없으면 bin)
    REQUIRED_CONFIG   = "config.json"
    WEIGHT_CANDIDATES = ["model.safetensors", "pytorch_model.bin"]
    MIN_WEIGHT_BYTES  = 10 * 1024 * 1024  # 10 MB 미만이면 불완전 저장으로 판단

    for subdir in sorted(MODEL_SAVE_DIR.glob("krfinbert-finetuned-*"), reverse=True):
        config_ok  = (subdir / REQUIRED_CONFIG).exists()
        weight_file = next(
            (subdir / w for w in WEIGHT_CANDIDATES if (subdir / w).exists()), None
        )
        weight_ok  = weight_file is not None and weight_file.stat().st_size >= MIN_WEIGHT_BYTES

        if config_ok and weight_ok:
            logger.info(f"체크포인트 로드: {subdir}")
            return str(subdir)
        else:
            logger.warning(
                f"불완전 체크포인트 건너뜀: {subdir} "
                f"(config={config_ok}, weight={weight_file}, "
                f"size={weight_file.stat().st_size if weight_file else 0})"
            )

    logger.info("유효한 체크포인트 없음 → BASE_MODEL 사용")
    return BASE_MODEL_NAME


def run_finetune(trigger_type: str = "manual") -> dict:
    history_id = None
    try:
        with get_db() as conn:
            texts, labels, log_ids = _fetch_correction_data(conn)
            if not texts:
                return {"status": "skipped", "correction_count": 0}

            logger.info(f"fine-tuning 시작: {len(texts)}건 (trigger={trigger_type})")
            base_model = _get_current_model_path()
            output_dir = str(MODEL_SAVE_DIR /
                             f"krfinbert-finetuned-{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO finetune_history
                        (trigger_type, status, correction_count, base_model, started_at)
                    VALUES (%s, 'running', %s, %s, NOW())
                """, (trigger_type, len(texts), base_model))
                history_id = cur.lastrowid

        device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # tokenizer는 항상 허깅페이스 원본에서 로드 (체크포인트 tokenizer 파일 불완전 대비)
        tokenizer = BertTokenizer.from_pretrained(BASE_MODEL_NAME)
        model     = BertForSequenceClassification.from_pretrained(
            base_model, num_labels=3).to(device)

        dataset    = CorrectionDataset(texts, labels, tokenizer, MAX_LENGTH)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        optimizer  = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
        total_steps = len(dataloader) * EPOCHS
        scheduler  = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=max(1, total_steps // 10),
            num_training_steps=total_steps,
        )

        model.train()
        total_correct = total_samples = 0
        for epoch in range(EPOCHS):
            epoch_correct = 0
            for batch in dataloader:
                optimizer.zero_grad()
                outputs = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    token_type_ids=batch["token_type_ids"].to(device),
                    labels=batch["labels"].to(device),
                )
                outputs.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                epoch_correct += (outputs.logits.argmax(dim=-1) ==
                                  batch["labels"].to(device)).sum().item()
            total_correct += epoch_correct
            total_samples += len(dataset)
            logger.info(f"epoch {epoch+1}/{EPOCHS} acc={epoch_correct/len(dataset):.4f}")

        train_accuracy = round(total_correct / total_samples, 4)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        _reload_sentiment_model(output_dir, device)

        with get_db() as conn:
            with conn.cursor() as cur:
                if log_ids:
                    placeholders = ",".join(["%s"] * len(log_ids))
                    cur.execute(f"UPDATE correction_log SET used_in_finetune=1 "
                                f"WHERE id IN ({placeholders})", log_ids)
                cur.execute("""
                    UPDATE finetune_history
                    SET status='done', output_model=%s, train_accuracy=%s, finished_at=NOW()
                    WHERE id=%s
                """, (output_dir, train_accuracy, history_id))

        return {"history_id": history_id, "correction_count": len(texts),
                "output_model": output_dir, "train_accuracy": train_accuracy, "status": "done"}

    except Exception as e:
        logger.error(f"fine-tuning 실패: {e}")
        log_pipeline_error(pipeline="finetuner", error=e)
        if history_id:
            try:
                with get_db() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            UPDATE finetune_history
                            SET status='failed', error_message=%s, finished_at=NOW()
                            WHERE id=%s
                        """, (str(e), history_id))
            except Exception:
                pass
        return {"status": "failed", "error": str(e)}


def _reload_sentiment_model(model_path: str, device: torch.device):
    import model.services.sentiment as svc
    svc._tokenizer = BertTokenizer.from_pretrained(BASE_MODEL_NAME)  # tokenizer는 원본 사용
    svc._model     = BertForSequenceClassification.from_pretrained(model_path).to(device)
    svc._model.eval()
    svc._device    = device
    logger.info(f"sentiment 모델 교체 완료: {model_path}")


def check_and_trigger_finetune():
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS cnt FROM finetune_history WHERE status='running'")
                if cur.fetchone()["cnt"] > 0:
                    return
                cur.execute("SELECT COUNT(*) AS cnt FROM correction_log WHERE used_in_finetune=0")
                pending = cur.fetchone()["cnt"]
        if pending >= FINETUNE_THRESHOLD:
            run_finetune(trigger_type="auto")
    except Exception as e:
        log_pipeline_error(pipeline="finetune_checker", error=e)
