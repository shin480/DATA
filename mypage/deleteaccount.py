from sqlalchemy import text
from typing import Dict, Any
from starlette.requests import Request
from util.db import get_engine


def delete_user_account(info: Dict[str, Any], req: Request):
    # 1. 세션 체크
    session_user = req.session.get("user")
    if not session_user:
        return {"success": False, "message": "로그인이 만료되었습니다."}

    user_id = info.get("user_id")

    # 2. 본인 확인 (세션의 이메일과 요청온 이메일 대조)
    if session_user.get('user_id') != user_id:
        return {"success": False, "message": "권한이 없습니다."}

    conn = None
    try:
        conn = get_engine()

        # 3. DB에서 유저 삭제 SQL
        sql = text("""
            DELETE FROM users 
            WHERE user_id = :user_id
        """)

        conn.execute(sql, {"user_id": user_id})
        conn.commit()

        # 4. 핵심: 서버 세션 파괴 (로그아웃 처리)
        req.session.clear()

        return {"success": True, "message": "회원 탈퇴가 완료되었습니다."}

    except Exception as e:
        print(f"🚨 회원 탈퇴 에러: {e}")
        return {"success": False, "message": "데이터베이스 오류가 발생했습니다."}
    finally:
        if conn:
            conn.close()