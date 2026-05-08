from sqlalchemy import text
from typing import Dict, Any
from starlette.requests import Request

from util.db import get_engine
from util.logger import log_user_activity
from regist.hashing import hash_password

def update_user_password(info: Dict[str, Any], req: Request):
    # 1. 세션 체크
    session_user = req.session.get("user")
    if not session_user:
        return {"success": False, "message": "로그인이 만료되었습니다."}

    email = info.get("email")
    new_pw = info.get("new_pw")

    # 2. 보안 검증 (세션 이메일 vs 요청 이메일)
    if session_user.get('user_id') != email:
        return {"success": False, "message": "권한이 없습니다."}

    conn = None
    try:
        conn = get_engine()
        # 3. 비밀번호 해싱
        hashed_pw = hash_password(new_pw)

        # 4. DB 업데이트 실행
        sql = text("""
            UPDATE users 
            SET password = :hashed_pw 
            WHERE user_id = :email
        """)

        conn.execute(sql, {"hashed_pw": hashed_pw, "email": email})
        conn.commit()

        log_user_activity(email, "mpac103", req)

        return {"success": True, "message": "비밀번호가 성공적으로 변경되었습니다."}

    except Exception as e:
        print(f"🚨 비밀번호 변경 에러: {e}")
        return {"success": False, "message": "데이터베이스 오류가 발생했습니다."}
    finally:
        if conn:
            conn.close()