"""
파이프라인 오류 로깅 유틸리티

개별 뉴스 skip 오류가 아닌 파이프라인 자체 크래시를 DB에 기록합니다.
모든 서비스 파일에서 공통으로 import하여 사용합니다.

pipeline_error_log 테이블:
  - error_code   : 에러 코드 VARCHAR(50) (현재 고정값: 'ML_ERROR')
  - error_message: '[파이프라인명] 예외 메시지' 형식으로 기록
  - article_id   : 개별 뉴스 오류 시 article_id (nullable)
  - scope_id     : 개별 scope 오류 시 scope_id (nullable)
  - occurred_at  : 오류 발생 시각
"""

import logging
import traceback
from typing import Optional

from model.database import get_db

logger = logging.getLogger(__name__)


def log_pipeline_error(
    pipeline: str,
    error: Exception,
    article_id: Optional[str] = None,
    scope_id: Optional[str] = None,
):
    """
    파이프라인 오류를 pipeline_error_log 테이블에 저장합니다.

    Args:
        pipeline  : 파이프라인 식별자 (error_message 앞에 [pipeline] 형식으로 포함됨)
                    'classifier' | 'sentiment' | 'summarizer' | 'keyword' | 'scope_title'
        error     : 발생한 예외 객체
        article_id: 개별 뉴스 오류 시 article_id (파이프라인 전체 크래시면 None)
        scope_id  : 개별 scope 오류 시 scope_id
    """
    error_code    = "ML_ERROR"
    error_message = f"[{pipeline}] {str(error)}\n\n{traceback.format_exc()}"

    # 콘솔 로그는 항상 남김
    logger.error(
        f"[{pipeline}] 파이프라인 오류 | "
        f"code={error_code} | article_id={article_id} | scope_id={scope_id} | "
        f"msg={str(error)}"
    )

    # DB 저장 (DB 자체 오류면 콘솔 로그로만 처리)
    try:
        with get_db() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO pipeline_error_log
                        (error_code, error_message, article_id, scope_id)
                    VALUES (%s, %s, %s, %s)
                """, (error_code, error_message, article_id, scope_id))
    except Exception as db_err:
        logger.critical(
            f"[{pipeline}] 오류 로그 DB 저장 실패 (DB 연결 불가): {db_err}"
        )
