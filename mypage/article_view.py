from typing import Dict, Any

from starlette.requests import Request
from sqlalchemy import text
from util.db import get_engine


def view_log(info: Dict[str, Any], req: Request):
    conn = None

    try:
        article_id = info.get("article_id")
        is_valid_view = info.get("is_valid_view", False)

        if not article_id:
            return {
                "success": False,
                "message": "article_id가 없습니다."
            }

        user = req.session.get("user")

        if not user:
            return {
                "success": False,
                "message": "로그인 사용자만 조회 기록이 저장됩니다."
            }

        user_id = user.get("user_id")

        if not user_id:
            return {
                "success": False,
                "message": "세션 사용자 정보가 올바르지 않습니다."
            }

        conn = get_engine()

        # ==============================
        # 유효 조회(True)면 UPDATE
        # ==============================
        if int(is_valid_view) == 1:

            sql = text("""
                UPDATE article_views
                SET is_valid_view = TRUE
                WHERE user_id = :user_id
                  AND article_id = :article_id
            """)

            conn.execute(sql, {
                "user_id": user_id,
                "article_id": article_id
            })

            conn.commit()

            return {
                "success": True,
                "message": "유효 조회로 업데이트되었습니다."
            }

        # ==============================
        # 최초 조회(False)면 INSERT
        # ==============================
        else:
            sql = text("""
                INSERT IGNORE INTO article_views (
                    user_id,
                    article_id,
                    is_valid_view
                )
                VALUES (
                    :user_id,
                    :article_id,
                    :is_valid_view
                )
            """)

            conn.execute(sql, {
                "user_id": user_id,
                "article_id": article_id,
                "is_valid_view": False
            })

            conn.commit()

            return {
                "success": True,
                "message": "기사 조회 기록이 저장되었습니다."
            }

    except Exception as e:
        print(f"🚨 기사 조회 기록 저장 에러: {e}")

        return {
            "success": False,
            "message": "기사 조회 기록 저장 중 서버 오류가 발생했습니다."
        }

    finally:
        if conn:
            conn.close()
