"""
Admin 라우터 — 감성 레이블 CSV 업로드

CSV 형식 (헤더 필수):
  keyword,sentiment,weight
  비리,negative,1.2
  협력,positive,1.0
  발표,neutral,0.8

Endpoints:
  POST /admin/labels/upload   - CSV 업로드 → sentiment_labels 테이블 갱신
  GET  /admin/labels          - 현재 등록된 레이블 목록 조회
  DELETE /admin/labels/{keyword} - 특정 키워드 삭제
"""

import io
import logging

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from model.database import get_db

logger = APIRouter.__module__
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])

VALID_SENTIMENTS = {"positive", "negative", "neutral"}


# ── CSV 업로드 ─────────────────────────────────────────
@router.post("/labels/upload")
async def upload_labels(
    file: UploadFile = File(...),
    replace_all: bool = Query(
        default=False,
        description="True면 기존 레이블 전체 삭제 후 재등록. False면 upsert(키워드 기준 덮어쓰기)."
    ),
):
    """
    감성 키워드 CSV를 업로드합니다.

    - replace_all=False (기본): 기존 키워드는 덮어쓰고 신규는 추가합니다.
    - replace_all=True: 기존 레이블을 모두 지우고 CSV 내용으로 교체합니다.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="CSV 파일만 업로드 가능합니다.")

    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"CSV 파싱 오류: {e}")

    # ── 컬럼 검증
    required_cols = {"keyword", "sentiment"}
    missing = required_cols - set(df.columns)
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"필수 컬럼 누락: {missing}. 필요한 컬럼: keyword, sentiment, weight(선택)"
        )

    # weight 컬럼 없으면 1.0으로 채움
    if "weight" not in df.columns:
        df["weight"] = 1.0

    # 공백 제거
    df["keyword"]   = df["keyword"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].astype(str).str.strip().str.lower()
    df["weight"]    = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0)

    # ── 유효성 검사
    invalid_sentiments = df[~df["sentiment"].isin(VALID_SENTIMENTS)]
    if not invalid_sentiments.empty:
        bad_vals = invalid_sentiments["sentiment"].unique().tolist()
        raise HTTPException(
            status_code=400,
            detail=f"sentiment 값은 positive/negative/neutral만 허용됩니다. 잘못된 값: {bad_vals}"
        )

    empty_keywords = df[df["keyword"] == ""]
    if not empty_keywords.empty:
        raise HTTPException(status_code=400, detail="빈 keyword가 포함되어 있습니다.")

    # 중복 keyword 처리 (마지막 값 사용)
    df = df.drop_duplicates(subset=["keyword"], keep="last")

    records = df[["keyword", "sentiment", "weight"]].to_dict("records")

    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                if replace_all:
                    cur.execute("DELETE FROM sentiment_labels")
                    logger.info("기존 sentiment_labels 전체 삭제")

                inserted = 0
                updated  = 0
                for row in records:
                    # upsert: 키워드 존재 시 sentiment/weight 덮어쓰기
                    result = cur.execute("""
                        INSERT INTO sentiment_labels (keyword, sentiment, weight, source, uploaded_at)
                        VALUES (%s, %s, %s, 'csv_upload', NOW())
                        ON DUPLICATE KEY UPDATE
                            sentiment   = VALUES(sentiment),
                            weight      = VALUES(weight),
                            source      = 'csv_upload',
                            uploaded_at = NOW()
                    """, (row["keyword"], row["sentiment"], float(row["weight"])))

                    if result == 1:
                        inserted += 1
                    else:
                        updated += 1

        return {
            "message":  "업로드 완료",
            "inserted": inserted,
            "updated":  updated,
            "total":    len(records),
            "replace_all": replace_all,
        }

    except Exception as e:
        logger.error(f"CSV 업로드 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── 레이블 목록 조회 ───────────────────────────────────
@router.get("/labels")
def get_labels(
    sentiment: str | None = Query(default=None, description="positive | negative | neutral"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=50, ge=1, le=200),
):
    """등록된 감성 키워드 목록을 페이지네이션으로 반환합니다."""
    offset = (page - 1) * page_size

    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                if sentiment:
                    if sentiment not in VALID_SENTIMENTS:
                        raise HTTPException(status_code=400, detail="sentiment 값이 유효하지 않습니다.")
                    cur.execute("""
                        SELECT keyword, sentiment, weight, uploaded_at
                        FROM sentiment_labels
                        WHERE sentiment = %s
                        ORDER BY uploaded_at DESC
                        LIMIT %s OFFSET %s
                    """, (sentiment, page_size, offset))
                    cur.execute("SELECT COUNT(*) as cnt FROM sentiment_labels WHERE sentiment = %s", (sentiment,))
                else:
                    cur.execute("""
                        SELECT keyword, sentiment, weight, uploaded_at
                        FROM sentiment_labels
                        ORDER BY uploaded_at DESC
                        LIMIT %s OFFSET %s
                    """, (page_size, offset))

                rows = cur.fetchall()

                cur.execute("SELECT COUNT(*) as cnt FROM sentiment_labels" +
                            (" WHERE sentiment = %s" if sentiment else ""),
                            *([sentiment] if sentiment else []))
                total = cur.fetchone()["cnt"]

        return {
            "total":    total,
            "page":     page,
            "page_size": page_size,
            "items":    rows,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 단일 키워드 삭제 ───────────────────────────────────
@router.delete("/labels/{keyword}")
def delete_label(keyword: str):
    """특정 키워드를 삭제합니다."""
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                affected = cur.execute(
                    "DELETE FROM sentiment_labels WHERE keyword = %s", (keyword,)
                )

        if affected == 0:
            raise HTTPException(status_code=404, detail=f"키워드 '{keyword}' 를 찾을 수 없습니다.")

        return {"message": f"키워드 '{keyword}' 삭제 완료"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
