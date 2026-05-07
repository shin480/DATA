"""
감성 분류 품질 평가 라우터 — ES 기반

Endpoints:
  GET  /eval/sentiment/dashboard       - 대시보드 HTML
  GET  /eval/sentiment/confidence      - confidence 분포 통계
  GET  /eval/sentiment/low-confidence  - 낮은 확신도 기사 샘플
  POST /eval/sentiment/evaluate        - CSV 업로드 → 정확도 평가
  GET  /eval/sentiment/summary         - 종합 리포트
"""

import io
import logging
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse

from model.database import get_es
from model.services.sentiment import predict_single

TEMPLATE_PATH = Path(__file__).parent.parent / "templates" / "sentiment_dashboard_dark.html"

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/eval/sentiment", tags=["sentiment-eval"])

VALID_SENTIMENTS   = {"positive", "negative", "neutral"}
LOW_CONF_THRESHOLD = 0.55
INDEX_NEWS         = "news_economy"


@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
def dashboard():
    if not TEMPLATE_PATH.exists():
        raise HTTPException(status_code=404,
                            detail=f"템플릿 파일을 찾을 수 없습니다: {TEMPLATE_PATH}")
    return HTMLResponse(content=TEMPLATE_PATH.read_text(encoding="utf-8"))


@router.get("/confidence")
def get_confidence_stats(
    date_from: str | None = Query(default=None, description="조회 시작일 (YYYY-MM-DD)"),
    date_to:   str | None = Query(default=None, description="조회 종료일 (YYYY-MM-DD)"),
):
    import re as _re
    DATE_RE = _re.compile(r"^\d{4}-\d{2}-\d{2}$")
    if date_from and not DATE_RE.match(date_from):
        raise HTTPException(status_code=400, detail="date_from 형식은 YYYY-MM-DD")
    if date_to and not DATE_RE.match(date_to):
        raise HTTPException(status_code=400, detail="date_to 형식은 YYYY-MM-DD")

    # 날짜 필터 구성
    must = [
        {"exists": {"field": "sentiment"}},
        {"exists": {"field": "sentiment_score"}},
    ]
    if date_from or date_to:
        range_filter = {}
        if date_from:
            range_filter["gte"] = date_from
        if date_to:
            range_filter["lte"] = date_to
        must.append({"range": {"published_at": range_filter}})

    query = {"bool": {"must": must}}

    try:
        es = get_es()

        # 전체 통계 (aggregation)
        agg_res = es.search(
            index=INDEX_NEWS,
            body={
                "size": 0,
                "query": query,
                "aggs": {
                    "total":    {"value_count": {"field": "sentiment_score"}},
                    "avg_conf": {"avg": {"field": "sentiment_score"}},
                    "low_cnt":  {"filter": {"range": {"sentiment_score": {"lt": LOW_CONF_THRESHOLD}}}},
                    "by_sentiment": {
                        "terms": {"field": "sentiment", "size": 10},
                        "aggs": {
                            "avg_score": {"avg":    {"field": "sentiment_score"}},
                            "high":      {"filter": {"range": {"sentiment_score": {"gte": 0.8}}}},
                            "mid":       {"filter": {"range": {"sentiment_score": {"gte": 0.55, "lt": 0.8}}}},
                            "low":       {"filter": {"range": {"sentiment_score": {"lt": 0.55}}}},
                        },
                    },
                    "histogram": {
                        "histogram": {"field": "sentiment_score", "interval": 0.05, "min_doc_count": 1},
                    },
                },
            },
        )
        aggs  = agg_res["aggregations"]
        total = aggs["total"]["value"] or 0
        avg_c = round(aggs["avg_conf"]["value"] or 0.0, 4)
        low_c = aggs["low_cnt"]["doc_count"]

        by_sentiment = [
            {
                "sentiment": b["key"],
                "count":     b["doc_count"],
                "avg_score": round(b["avg_score"]["value"] or 0.0, 4),
                "high":      b["high"]["doc_count"],
                "mid":       b["mid"]["doc_count"],
                "low":       b["low"]["doc_count"],
            }
            for b in aggs["by_sentiment"]["buckets"]
        ]

        histogram = [
            {
                "bucket": f"{round(b['key'], 2)}~{round(b['key'] + 0.05, 2)}",
                "count":  b["doc_count"],
            }
            for b in aggs["histogram"]["buckets"]
        ]

        es.close()

        return {
            "total_classified":    total,
            "avg_confidence":      avg_c,
            "low_confidence_rate": round(low_c / total, 4) if total else 0,
            "low_conf_threshold":  LOW_CONF_THRESHOLD,
            "date_from":           date_from,
            "date_to":             date_to,
            "by_sentiment":        by_sentiment,
            "histogram":           histogram,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _fetch_low_confidence(threshold: float, limit: int, sentiment: str | None,
                          date_from: str | None = None, date_to: str | None = None) -> dict:
    must = [
        {"range":  {"sentiment_score": {"lt": threshold}}},
        {"exists": {"field": "sentiment"}},
    ]
    if sentiment:
        if sentiment not in VALID_SENTIMENTS:
            raise HTTPException(status_code=400, detail="유효하지 않은 sentiment 값")
        must.append({"term": {"sentiment": sentiment}})
    if date_from or date_to:
        range_filter = {}
        if date_from:
            range_filter["gte"] = date_from
        if date_to:
            range_filter["lte"] = date_to
        must.append({"range": {"published_at": range_filter}})

    query = {"bool": {"must": must}}
    es    = get_es()

    total_res = es.count(index=INDEX_NEWS, body={"query": query})
    total     = total_res["count"]

    res = es.search(
        index=INDEX_NEWS,
        body={
            "query":   query,
            "_source": ["article_id", "title", "sentiment", "sentiment_score",
                        "published_at", "scopeID"],
            "sort":    [{"sentiment_score": "asc"}],
            "size":    limit,
        },
    )
    samples = [
        {
            "newsID":          h["_source"].get("article_id"),
            "title":           h["_source"].get("title"),
            "sentiment":       h["_source"].get("sentiment"),
            "sentiment_score": h["_source"].get("sentiment_score"),
            "published_at":    str(h["_source"].get("published_at", ""))[:10],
            "scopeID":         h["_source"].get("scopeID"),
        }
        for h in res["hits"]["hits"]
    ]
    es.close()
    return {"threshold": threshold, "total_low_confidence": total, "samples": samples}


@router.get("/low-confidence")
def get_low_confidence_samples(
    threshold: float      = Query(default=LOW_CONF_THRESHOLD, ge=0.0, le=1.0),
    limit:     int        = Query(default=20, ge=1, le=100),
    sentiment: str | None = Query(default=None),
):
    try:
        return _fetch_low_confidence(threshold, limit, sentiment)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate")
async def evaluate_with_labels(
    file: UploadFile = File(...),
    use_keyword_boost: bool = Query(default=True),
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV 파일만 업로드 가능합니다.")
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV 파싱 오류: {e}")

    # newsID도 article_id와 동일하게 처리
    if "newsID" in df.columns and "article_id" not in df.columns:
        df = df.rename(columns={"newsID": "article_id"})

    has_article_id = "article_id" in df.columns
    has_text       = "title" in df.columns and "content" in df.columns
    has_true_label = "true_sentiment" in df.columns

    if not has_true_label:
        raise HTTPException(status_code=400, detail="'true_sentiment' 컬럼이 필요합니다.")
    if not has_article_id and not has_text:
        raise HTTPException(status_code=400,
                            detail="'newsID' 또는 'article_id' 또는 'title'+'content' 컬럼이 필요합니다.")

    df["true_sentiment"] = df["true_sentiment"].str.strip().str.lower()
    invalid = df[~df["true_sentiment"].isin(VALID_SENTIMENTS)]
    if not invalid.empty:
        raise HTTPException(status_code=400,
                            detail=f"유효하지 않은 true_sentiment: {invalid['true_sentiment'].unique().tolist()}")

    keyword_dict = {}
    if use_keyword_boost:
        from model.database import get_db
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT keyword, sentiment, weight FROM sentiment_labels")
                keyword_dict = {r["keyword"]: (r["sentiment"], r["weight"])
                                for r in cur.fetchall()}

    es = get_es()
    records = []

    if has_article_id:
        ids  = df["article_id"].astype(str).tolist()
        mget = es.mget(index=INDEX_NEWS, body={"ids": ids},
                       _source=["article_id", "title", "content",
                                 "sentiment", "sentiment_score"])
        db_rows = {d["_id"]: d["_source"] for d in mget["docs"] if d.get("found")}

        for _, row in df.iterrows():
            aid = str(row["article_id"])
            if aid not in db_rows:
                continue
            doc = db_rows[aid]
            records.append({
                "article_id": aid,
                "true":       row["true_sentiment"],
                "db_pred":    doc.get("sentiment"),
                "db_score":   doc.get("sentiment_score"),
            })
    else:
        for _, row in df.iterrows():
            pred, score = predict_single(str(row["title"]), str(row["content"]), keyword_dict)
            records.append({"true": row["true_sentiment"], "db_pred": pred, "db_score": score})

    es.close()

    if not records:
        raise HTTPException(status_code=404, detail="일치하는 article_id를 찾을 수 없습니다.")

    labels = list(VALID_SENTIMENTS)
    y_true = [r["true"] for r in records]
    y_pred = [r["db_pred"] for r in records]
    n      = len(records)

    cm = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    per_class = {}
    for label in labels:
        tp = cm[label][label]
        fp = sum(cm[t][label] for t in labels if t != label)
        fn = sum(cm[label][p] for p in labels if p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall    = tp / (tp + fn) if (tp + fn) else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[label] = {"precision": round(precision, 4), "recall": round(recall, 4),
                             "f1": round(f1, 4), "support": sum(cm[label].values())}

    accuracy    = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    macro_p     = sum(per_class[l]["precision"] for l in labels) / len(labels)
    macro_r     = sum(per_class[l]["recall"]    for l in labels) / len(labels)
    macro_f1    = sum(per_class[l]["f1"]        for l in labels) / len(labels)
    total_sup   = sum(per_class[l]["support"]   for l in labels)
    w_p  = sum(per_class[l]["precision"] * per_class[l]["support"] for l in labels) / total_sup
    w_r  = sum(per_class[l]["recall"]    * per_class[l]["support"] for l in labels) / total_sup
    w_f1 = sum(per_class[l]["f1"]        * per_class[l]["support"] for l in labels) / total_sup

    misclassified = [r for r in records if r["true"] != r["db_pred"]][:30]

    return {
        "n_samples":    n,
        "accuracy":     round(accuracy, 4),
        "macro_avg":    {"precision": round(macro_p, 4), "recall": round(macro_r, 4),
                         "f1": round(macro_f1, 4)},
        "weighted_avg": {"precision": round(w_p, 4), "recall": round(w_r, 4),
                         "f1": round(w_f1, 4)},
        "per_class":    per_class,
        "confusion_matrix": cm,
        "misclassified_samples": misclassified,
    }


@router.get("/summary")
def get_summary(low_conf_limit: int = Query(default=10, ge=1, le=50)):
    confidence  = get_confidence_stats()
    low_samples = _fetch_low_confidence(LOW_CONF_THRESHOLD, low_conf_limit, None)
    return {"confidence_stats": confidence, "low_confidence_samples": low_samples}
