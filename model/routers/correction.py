"""
휴먼 감성 정정 라우터

정정 워크플로우(correction_log, finetune_history)는 MySQL 유지.
뉴스 원문 조회(목록/상세)는 ES news_economy에서 조회.
정정 후 news_economy.sentiment를 ES에서 즉시 업데이트.
"""

import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel

from model.database import get_db, get_es

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin/corrections", tags=["corrections"])

VALID_SENTIMENTS = {"positive", "negative", "neutral"}
INDEX_NEWS       = "news_economy"


class CorrectionCreate(BaseModel):
    newsID:              str
    corrected_sentiment: str
    corrected_by:        str           = "admin"
    memo:                Optional[str] = None


# ── 1. 정정 대상 뉴스 목록 ──────────────────────────────
@router.get("/news")
def list_news_for_correction(
    mode:      str        = Query(default="low_confidence"),
    query:     str | None = Query(default=None),
    sentiment: str | None = Query(default=None),
    threshold: float      = Query(default=0.55, ge=0.0, le=1.0),
    page:      int        = Query(default=1,    ge=1),
    page_size: int        = Query(default=20,   ge=1, le=100),
):
    if sentiment and sentiment not in VALID_SENTIMENTS:
        raise HTTPException(status_code=400, detail="유효하지 않은 sentiment 값")

    must      = [{"exists": {"field": "sentiment"}}]
    must_not  = []
    sort_field = "published_at"

    if mode == "low_confidence":
        must.append({"range": {"sentiment_score": {"lt": threshold}}})
        sort_field = "sentiment_score"
    elif mode == "search":
        if not query:
            raise HTTPException(status_code=400, detail="검색 모드에서는 query 파라미터가 필요합니다.")
        must.append({"multi_match": {"query": query, "fields": ["title", "keywords"]}})

    if sentiment:
        must.append({"term": {"sentiment": sentiment}})

    es_query = {"bool": {"must": must}}
    sort_asc  = (mode == "low_confidence")

    try:
        es = get_es()

        total = es.count(index=INDEX_NEWS, body={"query": es_query})["count"]

        res = es.search(
            index=INDEX_NEWS,
            body={
                "query":   es_query,
                "_source": ["article_id", "title", "sentiment", "sentiment_score",
                            "keywords", "published_at", "scopeID"],
                "sort":    [{sort_field: "asc" if sort_asc else "desc"}],
                "from":    (page - 1) * page_size,
                "size":    page_size,
            },
        )

        # correction_log에서 정정 이력 조회
        article_ids = [h["_source"]["article_id"] for h in res["hits"]["hits"]]
        corrections = {}
        if article_ids:
            with get_db() as conn:
                with conn.cursor() as cur:
                    placeholders = ",".join(["%s"] * len(article_ids))
                    cur.execute(f"""
                        SELECT c1.newsID, c1.corrected_sentiment, c1.corrected_by,
                               DATE_FORMAT(c1.created_at, '%%Y-%%m-%%d %%H:%%i') AS corrected_at
                        FROM correction_log c1
                        WHERE c1.id = (
                            SELECT MAX(id) FROM correction_log c2 WHERE c2.newsID = c1.newsID
                        )
                        AND c1.newsID IN ({placeholders})
                    """, article_ids)
                    corrections = {r["newsID"]: r for r in cur.fetchall()}

        items = []
        for h in res["hits"]["hits"]:
            src = h["_source"]
            aid = src.get("article_id")
            cor = corrections.get(aid, {})
            items.append({
                "newsID":              aid,
                "title":               src.get("title"),
                "model_sentiment":     src.get("sentiment"),
                "model_score":         src.get("sentiment_score"),
                "keywords":            src.get("keywords"),
                "published_at":        str(src.get("published_at", ""))[:10],
                "scopeID":             src.get("scopeID"),
                "corrected_sentiment": cor.get("corrected_sentiment"),
                "corrected_by":        cor.get("corrected_by"),
                "corrected_at":        cor.get("corrected_at"),
            })

        es.close()
        return {"total": total, "page": page, "page_size": page_size, "items": items}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 2. 뉴스 상세 조회 ──────────────────────────────────
@router.get("/news/{newsID}")
def get_news_detail(newsID: str):
    try:
        es  = get_es()
        res = es.get(index=INDEX_NEWS, id=newsID, ignore=404)
        es.close()

        if not res.get("found"):
            raise HTTPException(status_code=404, detail="뉴스를 찾을 수 없습니다.")

        src  = res["_source"]
        news = {
            "newsID":          src.get("article_id"),
            "title":           src.get("title"),
            "content":         src.get("content"),
            "sentiment":       src.get("sentiment"),
            "sentiment_score": src.get("sentiment_score"),
            "keywords":        src.get("keywords"),
            "summary":         src.get("summary"),
            "published_at":    str(src.get("published_at", ""))[:10],
            "scopeID":         src.get("scopeID"),
        }

        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, original_sentiment, original_score,
                           corrected_sentiment, corrected_by, memo, used_in_finetune,
                           DATE_FORMAT(created_at, '%%Y-%%m-%%d %%H:%%i') AS created_at
                    FROM correction_log
                    WHERE newsID = %s ORDER BY created_at DESC
                """, (newsID,))
                corrections = cur.fetchall()

        return {**news, "correction_history": corrections}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 3. 정정 등록 ───────────────────────────────────────
@router.post("")
def create_correction(body: CorrectionCreate):
    if body.corrected_sentiment not in VALID_SENTIMENTS:
        raise HTTPException(status_code=400, detail="유효하지 않은 corrected_sentiment 값")

    try:
        # ES에서 현재 감성 확인
        es  = get_es()
        res = es.get(index=INDEX_NEWS, id=body.newsID, ignore=404)
        if not res.get("found"):
            es.close()
            raise HTTPException(status_code=404, detail="뉴스를 찾을 수 없습니다.")

        src = res["_source"]
        current_sentiment = src.get("sentiment")
        current_score     = src.get("sentiment_score")

        if current_sentiment is None:
            es.close()
            raise HTTPException(status_code=400, detail="아직 감성 분류가 완료되지 않은 뉴스입니다.")
        if current_sentiment == body.corrected_sentiment:
            es.close()
            raise HTTPException(status_code=400,
                                detail=f"현재 분류({current_sentiment})와 동일한 값으로 정정할 수 없습니다.")

        # correction_log MySQL 저장
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO correction_log
                        (newsID, original_sentiment, original_score,
                         corrected_sentiment, corrected_by, memo)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (body.newsID, current_sentiment, current_score,
                      body.corrected_sentiment, body.corrected_by, body.memo))
                correction_id = cur.lastrowid

        # ES 즉시 업데이트
        es.update(index=INDEX_NEWS, id=body.newsID,
                  body={"doc": {"sentiment": body.corrected_sentiment}})
        es.close()

        return {"correction_id": correction_id, "newsID": body.newsID,
                "original": current_sentiment, "corrected": body.corrected_sentiment}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 4. 정정 이력 목록 ──────────────────────────────────
@router.get("")
def list_corrections(
    used_in_finetune: int | None = Query(default=None),
    page:      int = Query(default=1,  ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
):
    offset = (page - 1) * page_size
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                where_parts, params = [], []
                if used_in_finetune is not None:
                    where_parts.append("c.used_in_finetune = %s")
                    params.append(used_in_finetune)
                where_sql = "WHERE " + " AND ".join(where_parts) if where_parts else ""

                cur.execute(f"SELECT COUNT(*) AS cnt FROM correction_log c {where_sql}", params)
                total = cur.fetchone()["cnt"]

                # 제목은 ES에서 가져오기 어려우므로 newsID만 표시 (필요시 별도 조회)
                cur.execute(f"""
                    SELECT c.id, c.newsID,
                           c.original_sentiment, c.original_score,
                           c.corrected_sentiment, c.corrected_by, c.memo,
                           c.used_in_finetune,
                           DATE_FORMAT(c.created_at, '%%Y-%%m-%%d %%H:%%i') AS created_at
                    FROM correction_log c
                    {where_sql}
                    ORDER BY c.created_at DESC
                    LIMIT %s OFFSET %s
                """, params + [page_size, offset])
                rows = cur.fetchall()

        # 제목 ES에서 일괄 보완
        if rows:
            es   = get_es()
            ids  = [r["newsID"] for r in rows]
            mget = es.mget(index=INDEX_NEWS, body={"ids": ids}, _source=["title"])
            es.close()
            titles = {d["_id"]: d["_source"].get("title") for d in mget["docs"] if d.get("found")}
            for r in rows:
                r["title"] = titles.get(r["newsID"])

        return {"total": total, "page": page, "page_size": page_size, "items": rows}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 5. 정정 취소 ───────────────────────────────────────
@router.delete("/{correction_id}")
def delete_correction(correction_id: int):
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT newsID, original_sentiment, used_in_finetune
                    FROM correction_log WHERE id = %s
                """, (correction_id,))
                row = cur.fetchone()

                if not row:
                    raise HTTPException(status_code=404, detail="정정 이력을 찾을 수 없습니다.")
                if row["used_in_finetune"] == 1:
                    raise HTTPException(status_code=400,
                                        detail="이미 fine-tuning에 사용된 정정은 취소할 수 없습니다.")

                cur.execute("DELETE FROM correction_log WHERE id = %s", (correction_id,))

        # ES 롤백
        es = get_es()
        es.update(index=INDEX_NEWS, id=row["newsID"],
                  body={"doc": {"sentiment": row["original_sentiment"]}})
        es.close()

        return {"message": f"정정 취소 완료 (id={correction_id})"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 6. Fine-tuning 현황 ────────────────────────────────
@router.get("/finetune/status")
def get_finetune_status():
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS cnt FROM correction_log WHERE used_in_finetune=0")
                pending = cur.fetchone()["cnt"]
                cur.execute("""
                    SELECT id, trigger_type, status, correction_count,
                           base_model, output_model, train_accuracy, error_message,
                           DATE_FORMAT(started_at,  '%%Y-%%m-%%d %%H:%%i') AS started_at,
                           DATE_FORMAT(finished_at, '%%Y-%%m-%%d %%H:%%i') AS finished_at
                    FROM finetune_history ORDER BY created_at DESC LIMIT 5
                """)
                history = cur.fetchall()

        from model.services.finetuner import FINETUNE_THRESHOLD
        return {
            "pending_corrections": pending,
            "finetune_threshold":  FINETUNE_THRESHOLD,
            "remaining":           max(0, FINETUNE_THRESHOLD - pending),
            "history":             history,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── 7. Fine-tuning 수동 트리거 ─────────────────────────
@router.post("/finetune")
def trigger_finetune(background_tasks: BackgroundTasks):
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS cnt FROM finetune_history WHERE status='running'")
                if cur.fetchone()["cnt"] > 0:
                    raise HTTPException(status_code=409, detail="이미 fine-tuning이 실행 중입니다.")
                cur.execute("SELECT COUNT(*) AS cnt FROM correction_log WHERE used_in_finetune=0")
                pending = cur.fetchone()["cnt"]
                if pending == 0:
                    raise HTTPException(status_code=400, detail="fine-tuning할 정정 데이터가 없습니다.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    from model.services.finetuner import run_finetune
    background_tasks.add_task(run_finetune, trigger_type="manual")
    return {"message": f"fine-tuning 시작됨 (백그라운드) — 정정 데이터 {pending}건", "pending": pending}
