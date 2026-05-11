"""
불용어 관리 라우터

엔드포인트:
  GET    /admin/stopwords          — 불용어 목록 조회
  POST   /admin/stopwords          — 불용어 등록
  DELETE /admin/stopwords/{word}   — 불용어 삭제

ES 인덱스: keyword_stopwords
  fields: word (keyword), reason (text), added_at (date)
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from model.database import get_es

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/stopwords", tags=["stopwords"])

INDEX = "keyword_stopwords"


# ── 스키마 ─────────────────────────────────────────────

class StopwordIn(BaseModel):
    word:   str
    reason: Optional[str] = None


# ── 엔드포인트 ─────────────────────────────────────────

@router.get("")
def list_stopwords():
    """등록된 불용어 전체 조회"""
    es = get_es()
    try:
        res = es.search(
            index=INDEX,
            body={
                "query": {"match_all": {}},
                "sort":  [{"added_at": {"order": "desc"}}],
                "size":  10000,
            },
        )
        stopwords = [
            {
                "word":     hit["_source"]["word"],
                "reason":   hit["_source"].get("reason"),
                "added_at": hit["_source"].get("added_at"),
            }
            for hit in res["hits"]["hits"]
        ]
        return {"stopwords": stopwords, "total": len(stopwords)}
    except Exception as e:
        logger.error(f"불용어 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        es.close()


@router.post("", status_code=201)
def add_stopword(body: StopwordIn):
    """불용어 등록 (중복 시 reason/added_at 갱신)"""
    word = body.word.strip()
    if not word:
        raise HTTPException(status_code=400, detail="word는 필수입니다.")

    es = get_es()
    try:
        es.index(
            index=INDEX,
            id=word,          # word를 _id로 사용 → 자동 중복 방지
            body={
                "word":     word,
                "reason":   body.reason,
                "added_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        logger.info(f"불용어 등록: {word}")
        return {"message": f'"{word}" 등록 완료', "word": word}
    except Exception as e:
        logger.error(f"불용어 등록 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        es.close()


@router.delete("/{word}")
def delete_stopword(word: str):
    """불용어 삭제"""
    es = get_es()
    try:
        res = es.delete(index=INDEX, id=word, ignore=[404])
        if res.get("result") == "not_found":
            raise HTTPException(status_code=404, detail=f'"{word}" 를 찾을 수 없습니다.')
        logger.info(f"불용어 삭제: {word}")
        return {"message": f'"{word}" 삭제 완료', "word": word}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"불용어 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        es.close()
