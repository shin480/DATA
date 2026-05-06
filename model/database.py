"""
데이터베이스 연결 관리

- Elasticsearch : 뉴스/scope 데이터 전체 (읽기/쓰기)
- MySQL         : pipeline_error_log 전용 (SQLAlchemy + pymysql)
"""

import logging
import os
from contextlib import contextmanager

from pathlib import Path

from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# database.py 기준으로 .env 절대경로 지정
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

logger = logging.getLogger(__name__)

# ── Elasticsearch ──────────────────────────────────────
def get_es() -> Elasticsearch:
    host     = os.getenv("ES_HOST", "localhost")
    port     = int(os.getenv("ES_PORT", 9200))
    user     = os.getenv("ES_USER", "elastic")
    password = os.getenv("ES_PASSWORD", "")

    auth = (user, password) if password else None
    return Elasticsearch(
        f"http://{host}:{port}",
        basic_auth=auth,
        request_timeout=30,
    )


# ── MySQL (에러 로그 전용, SQLAlchemy + pymysql) ───────
def _build_engine():
    user     = os.getenv("DB_USER",     "root")
    password = os.getenv("DB_PASSWORD", "")
    host     = os.getenv("DB_HOST",     "localhost")
    port     = os.getenv("DB_PORT",     "3306")
    db       = os.getenv("DB_NAME",     "news_db")
    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url=url, echo=True, pool_size=1)


_engine       = None
_SessionLocal = None


def _get_session():
    """엔진/세션을 첫 호출 시점에 생성 (lazy init — .env 로드 이후 보장)"""
    global _engine, _SessionLocal
    if _SessionLocal is None:
        _engine       = _build_engine()
        _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
    return _SessionLocal()


@contextmanager
def get_db():
    """MySQL 세션 컨텍스트 매니저. 에러 로그 저장 전용."""
    import pymysql.cursors

    session  = _get_session()
    raw_conn = session.connection().connection

    class DictConnWrapper:
        def cursor(self):
            return raw_conn.cursor(pymysql.cursors.DictCursor)
        def commit(self):
            return raw_conn.commit()
        def rollback(self):
            return raw_conn.rollback()
        def __getattr__(self, name):
            return getattr(raw_conn, name)

    try:
        yield DictConnWrapper()
        raw_conn.commit()
    except Exception:
        raw_conn.rollback()
        raise
    finally:
        session.close()
