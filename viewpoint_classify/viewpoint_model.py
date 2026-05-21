# viewpoint_model.py

import os
import re
import joblib
import pandas as pd

from typing import Optional, Dict, Any, Tuple

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


MODEL_PATH = "government_responsibility_model.joblib"

_government_model: Optional[Pipeline] = None


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = str(text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\[.*?\]|\(.*?\)|=연합뉴스|기자|뉴스|#.*", "", text)
    return text.strip()


def build_input_text(title: str, clean_text_body: str) -> str:
    title = clean_text(title)
    clean_text_body = clean_text(clean_text_body)

    # 제목을 더 중요하게 보기 위해 2번 반복
    return f"{title} {title} {clean_text_body[:1500]}"


def normalize_label(label) -> int:
    """
    정부 책임이면 1, 아니면 0.
    CSV label 컬럼에 아래 형태 모두 허용:
    - 1 / 0
    - 정부 책임 / 기타
    - government / non_government
    - true / false
    - O / X
    """
    if label is None:
        return 0

    value = str(label).strip().lower()

    positive_values = {
        "1",
        "true",
        "o",
        "yes",
        "정부 책임",
        "정부책임",
        "government",
        "government_responsibility",
    }

    return 1 if value in positive_values else 0


def train_government_responsibility_model(
    csv_path: str,
    model_path: str = MODEL_PATH,
    text_column: str = "clean_text",
    title_column: str = "title",
    label_column: str = "label",
) -> Dict[str, Any]:
    """
    정부 책임 이진 분류 모델 학습.

    CSV 필수 컬럼:
    - title
    - clean_text
    - label

    label 예:
    정부 책임 = 1 또는 '정부 책임' 또는 O
    나머지 = 0 또는 '기타' 또는 X
    """

    df = pd.read_csv(csv_path)

    required = {title_column, text_column, label_column}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {missing}")

    df = df.dropna(subset=[title_column, text_column, label_column]).copy()

    df["input_text"] = df.apply(
        lambda row: build_input_text(
            row.get(title_column, ""),
            row.get(text_column, "")
        ),
        axis=1
    )

    df["y"] = df[label_column].apply(normalize_label)

    positive_count = int(df["y"].sum())
    negative_count = int((df["y"] == 0).sum())

    if positive_count < 10:
        raise ValueError(f"정부 책임 positive 샘플이 너무 적습니다: {positive_count}개")

    if negative_count < 10:
        raise ValueError(f"negative 샘플이 너무 적습니다: {negative_count}개")

    X = df["input_text"]
    y = df["y"]

    stratify = y if len(set(y)) == 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=30000,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                sublinear_tf=True
            )
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42
            )
        )
    ])

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    report = classification_report(
        y_test,
        pred,
        target_names=["not_government_responsibility", "government_responsibility"],
        zero_division=0,
        output_dict=True
    )

    accuracy = accuracy_score(y_test, pred)

    joblib.dump(model, model_path)

    global _government_model
    _government_model = model

    return {
        "model_path": model_path,
        "total": len(df),
        "positive": positive_count,
        "negative": negative_count,
        "accuracy": round(float(accuracy), 4),
        "report": report,
    }


def load_government_responsibility_model(
    model_path: str = MODEL_PATH
) -> Optional[Pipeline]:
    global _government_model

    if _government_model is not None:
        return _government_model

    if not os.path.exists(model_path):
        return None

    _government_model = joblib.load(model_path)
    return _government_model


def predict_government_responsibility(
    title: str,
    clean_text_body: str,
    model_path: str = MODEL_PATH
) -> Tuple[bool, float]:
    """
    정부 책임 여부 예측.

    Returns:
        (is_government_responsibility, probability)
    """

    model = load_government_responsibility_model(model_path)

    if model is None:
        return False, 0.0

    input_text = build_input_text(title, clean_text_body)

    prob = float(model.predict_proba([input_text])[0][1])
    is_positive = prob >= 0.6

    return is_positive, round(prob, 6)


def government_responsibility_score(
    title: str,
    clean_text_body: str,
    model_path: str = MODEL_PATH
) -> float:
    """
    룰 기반 점수에 더할 보조 점수.
    """
    _, prob = predict_government_responsibility(
        title,
        clean_text_body,
        model_path
    )

    if prob >= 0.8:
        return 4.0
    elif prob >= 0.7:
        return 3.0
    elif prob >= 0.6:
        return 2.0
    elif prob >= 0.5:
        return 0.8

    return 0.0


if __name__ == "__main__":
    result = train_government_responsibility_model(
        csv_path="government_responsibility_train.csv"
    )

    print(result)