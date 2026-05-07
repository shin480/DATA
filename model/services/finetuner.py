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
    def __init__(self, texts, labels, tokenizer, max_length):
        self.encodings = tokenizer(texts, truncation=True, padding=True,
                                   max_length=max_length, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "token_type_ids": self.encodings.get("token_type_ids",
                              torch.zeros_like(self.encodings["input_ids"]))[idx],
            "labels":         self.labels[idx],
        }


def _fetch_correction_data(conn) -> tuple[list[str], list[int], list[int]]:
    """
    correction_log(MySQL)에서 미사용 정정 데이터 조회 후
    ES news_economy에서 원문(title, content) 보완
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

    # ES에서 원문 일괄 조회
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

    return texts, labels, log_ids


def _get_current_model_path() -> str:
    if not MODEL_SAVE_DIR.exists():
        return BASE_MODEL_NAME
    subdirs = sorted(MODEL_SAVE_DIR.glob("krfinbert-finetuned-*"), reverse=True)
    return str(subdirs[0]) if subdirs else BASE_MODEL_NAME


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
        tokenizer = BertTokenizer.from_pretrained(base_model)
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
    svc._tokenizer = BertTokenizer.from_pretrained(model_path)
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
