"""Train and serve the government-responsibility viewpoint model."""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_DATA_PATH = (
    BASE_DIR / "labels" / "government_responsibility_binary_review_pool.xlsx"
)
DEFAULT_MODEL_PATH = BASE_DIR / "government_responsibility_model.joblib"
DEFAULT_METRICS_PATH = BASE_DIR / "government_responsibility_model_metrics.json"
DEFAULT_THRESHOLD = 0.6

# Backward-compatible name used by early draft code.
MODEL_PATH = str(DEFAULT_MODEL_PATH)

_government_model: Optional[Pipeline] = None


def clean_text(text: Any) -> str:
    """Normalize article text without discarding useful Korean tokens."""
    if text is None or pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[[^\]]*]|\([^)]*\)|#\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_input_text(title: Any, clean_text_body: Any, body_limit: int = 2000) -> str:
    """Give the title extra weight while bounding very long article bodies."""
    title_text = clean_text(title)
    body_text = clean_text(clean_text_body)[:body_limit]
    return f"{title_text} {title_text} {body_text}".strip()


def normalize_label(label: Any) -> Optional[int]:
    """Return a binary label or None for blank/ambiguous values."""
    if label is None or pd.isna(label):
        return None

    if isinstance(label, bool):
        return int(label)

    if isinstance(label, (int, float)) and label in (0, 1):
        return int(label)

    value = str(label).strip().lower()
    if not value:
        return None

    positive_values = {
        "1",
        "true",
        "o",
        "yes",
        "positive",
        "government",
        "government_responsibility",
        "government responsibility",
        "\uc815\ubd80 \ucc45\uc784",
        "\uc815\ubd80\ucc45\uc784",
    }
    negative_values = {
        "0",
        "false",
        "x",
        "no",
        "negative",
        "non_government",
        "not_government_responsibility",
    }

    if value in positive_values:
        return 1
    if value in negative_values:
        return 0
    return None


def read_training_frame(data_path: str | Path, sheet_name: str = "review_pool") -> pd.DataFrame:
    """Read either the review-pool workbook or a CSV training file."""
    path = Path(data_path)
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported training file type: {path.suffix}")


def resolve_label_column(df: pd.DataFrame, requested: Optional[str]) -> str:
    """Choose a usable label column for current and future review files."""
    if requested:
        if requested not in df.columns:
            raise ValueError(f"Label column not found: {requested}")
        return requested

    # Keep the combined pool stable by default. A reviewed label column can be
    # selected explicitly after the external review workbook is complete.
    candidates = [
        "final_binary_label",
        "codex_binary_label",
        "label_binary",
        "label",
        "review_ai_binary_label",
    ]
    for candidate in candidates:
        if candidate in df.columns and df[candidate].notna().any():
            return candidate

    raise ValueError(f"No label column found. Tried: {candidates}")


def prepare_training_frame(
    df: pd.DataFrame,
    title_column: str,
    text_column: str,
    label_column: str,
) -> pd.DataFrame:
    """Validate rows, build model text, and drop unlabeled samples."""
    required = {title_column, text_column, label_column}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing training columns: {sorted(missing)}")

    frame = df.copy()
    frame["y"] = frame[label_column].apply(normalize_label)
    frame = frame.dropna(subset=[title_column, text_column, "y"]).copy()
    frame["y"] = frame["y"].astype(int)
    frame["input_text"] = frame.apply(
        lambda row: build_input_text(row[title_column], row[text_column]),
        axis=1,
    )
    frame = frame[frame["input_text"].str.len() > 0].copy()

    if "article_id" in frame.columns:
        frame = frame.drop_duplicates(subset=["article_id"], keep="first")

    positive_count = int(frame["y"].sum())
    negative_count = int((frame["y"] == 0).sum())
    if positive_count < 10:
        raise ValueError(f"Need at least 10 positive rows, got {positive_count}")
    if negative_count < 10:
        raise ValueError(f"Need at least 10 negative rows, got {negative_count}")

    return frame


def build_model() -> Pipeline:
    """Build a compact baseline suitable for the current labeled sample size."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=40000,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.98,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate_model(
    model: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
    threshold: float,
) -> Dict[str, Any]:
    """Evaluate the viewpoint threshold used by prediction code."""
    probability = model.predict_proba(X_test)[:, 1]
    prediction = (probability >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        prediction,
        average="binary",
        zero_division=0,
    )

    metrics: Dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_test, prediction)), 4),
        "positive_precision": round(float(precision), 4),
        "positive_recall": round(float(recall), 4),
        "positive_f1": round(float(f1), 4),
        "confusion_matrix": confusion_matrix(y_test, prediction, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_test,
            prediction,
            labels=[0, 1],
            target_names=[
                "not_government_responsibility",
                "government_responsibility",
            ],
            zero_division=0,
            output_dict=True,
        ),
    }
    if y_test.nunique() == 2:
        metrics["roc_auc"] = round(float(roc_auc_score(y_test, probability)), 4)

    return metrics


def write_metrics(metrics: Dict[str, Any], metrics_path: str | Path) -> None:
    path = Path(metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def train_government_responsibility_model(
    data_path: str | Path = DEFAULT_TRAIN_DATA_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    text_column: str = "clean_text",
    title_column: str = "title",
    label_column: Optional[str] = None,
    sheet_name: str = "review_pool",
    test_size: float = 0.2,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    """Train, evaluate, then refit on all valid labeled rows."""
    raw = read_training_frame(data_path, sheet_name=sheet_name)
    chosen_label_column = resolve_label_column(raw, label_column)
    frame = prepare_training_frame(
        raw,
        title_column=title_column,
        text_column=text_column,
        label_column=chosen_label_column,
    )

    X = frame["input_text"]
    y = frame["y"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    evaluation_model = build_model()
    evaluation_model.fit(X_train, y_train)
    holdout_metrics = evaluate_model(
        evaluation_model,
        X_test=X_test,
        y_test=y_test,
        threshold=threshold,
    )

    final_model = build_model()
    final_model.fit(X, y)

    model_output = Path(model_path)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_output)

    result = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(Path(data_path).resolve()),
        "model_path": str(model_output.resolve()),
        "metrics_path": str(Path(metrics_path).resolve()),
        "label_column": chosen_label_column,
        "threshold": threshold,
        "total": int(len(frame)),
        "positive": int(y.sum()),
        "negative": int((y == 0).sum()),
        "train_rows": int(len(X_train)),
        "holdout_rows": int(len(X_test)),
        "holdout": holdout_metrics,
    }
    write_metrics(result, metrics_path)

    global _government_model
    _government_model = final_model
    return result


def load_government_responsibility_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> Optional[Pipeline]:
    global _government_model

    if _government_model is not None:
        return _government_model

    path = Path(model_path)
    if not path.exists():
        return None

    _government_model = joblib.load(path)
    return _government_model


def predict_government_responsibility(
    title: str,
    clean_text_body: str,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    threshold: float = DEFAULT_THRESHOLD,
) -> Tuple[bool, float]:
    """Return the binary decision and positive probability."""
    model = load_government_responsibility_model(model_path)
    if model is None:
        return False, 0.0

    probability = float(model.predict_proba([build_input_text(title, clean_text_body)])[0][1])
    return probability >= threshold, round(probability, 6)


def government_responsibility_score(
    title: str,
    clean_text_body: str,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> float:
    """Convert model probability into a small bonus for the rule classifier."""
    _, probability = predict_government_responsibility(
        title,
        clean_text_body,
        model_path=model_path,
    )
    if probability >= 0.8:
        return 4.0
    if probability >= 0.7:
        return 3.0
    if probability >= 0.6:
        return 2.0
    if probability >= 0.5:
        return 0.8
    return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the government-responsibility viewpoint model."
    )
    parser.add_argument(
        "--train-data",
        default=str(DEFAULT_TRAIN_DATA_PATH),
        help="CSV or XLSX file containing title, clean_text, and binary labels.",
    )
    parser.add_argument("--sheet-name", default="review_pool")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--metrics-path", default=str(DEFAULT_METRICS_PATH))
    parser.add_argument(
        "--label-column",
        default=None,
        help="Override auto label selection, e.g. codex_binary_label.",
    )
    parser.add_argument("--title-column", default="title")
    parser.add_argument("--text-column", default="clean_text")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = train_government_responsibility_model(
        data_path=args.train_data,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        text_column=args.text_column,
        title_column=args.title_column,
        label_column=args.label_column,
        sheet_name=args.sheet_name,
        test_size=args.test_size,
        threshold=args.threshold,
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))
