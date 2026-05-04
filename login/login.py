from fastapi import Request
from typing import Any, Dict
from sqlalchemy import text
from util.db import get_engine


def login_user(info: Dict[str, Any], req: Request):
    conn = None
    try:
        # 프론트에서 넘어온 값 (params = {"email": ..., "pw": ...})
        email = info.get("email")
        password = info.get("pw")

        if not email or not password:
            return {"success": False, "message": "이메일과 비밀번호를 모두 입력해주세요."}

        conn = get_engine()

        # 1. 컬럼명 아다리 수정
        # 실제 DB 컬럼: user_id, password, name (이미지 확인 결과)
        # birth와 gender는 테이블에 없으므로 SELECT에서 제외하거나 기본값 처리해야 합니다.
        sql = text("""
            SELECT name, user_id
            FROM users 
            WHERE user_id = :email AND password = :password
        """)

        # 2. 바인딩 파라미터 매칭
        # SQL의 :email -> 변수 email, :password -> 변수 password
        result = conn.execute(sql, {"email": email, "password": password}).fetchone()

        if result:
            # 3. 세션 저장 (결과 객체 필드명 주의)
            user_info = {
                "name": result.name,
                "email": result.user_id,
                # 테이블에 없는 정보는 일단 빈값이나 세션 유지용으로만 둠
                "birth": "",
                "gender": ""
            }
            req.session["user"] = user_info

            return {"success": True, "message": f"{result.name}님, 환영합니다!", "user": user_info}
        else:
            return {"success": False, "message": "이메일 또는 비밀번호가 일치하지 않습니다."}

    except Exception as e:
        print(f"🚨 로그인 로직 에러: {e}")
        return {"success": False, "message": "서버 오류가 발생했습니다."}
    finally:
        if conn:
            conn.close()