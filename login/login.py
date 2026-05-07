from fastapi import Request
from typing import Any, Dict
from sqlalchemy import text
from util.db import get_engine
from util.logger import log_user_activity, log_login
from regist.hashing import verify_password

def login_user(info: Dict[str, Any], req: Request):
    conn = None
    try:
        # 프론트에서 넘어온 값 (params = {"email": ..., "pw": ...})
        email = info.get("email")
        password = info.get("pw")

        if not email or not password:
            return {"success": False, "message": "이메일과 비밀번호를 모두 입력해주세요."}

        conn = get_engine()

        # 1. 이메일로 사용자 정보 추출
        sql = text("""
                    SELECT name, user_id, password, birth_date, gender, role
                    FROM users 
                    WHERE user_id = :email
                """)

        # 2. 바인딩 파라미터 매칭
        # SQL의 :email -> 변수 email, :password -> 변수 password
        result = conn.execute(sql, {"email": email}).fetchone()

        if result:
            # 로그인 실패 횟수 확인
            login_log_sql = text("""
                                SELECT fail_count_at_time 
                                FROM login_logs 
                                WHERE user_id = :user_id 
                                ORDER BY created_at DESC 
                                LIMIT 1
                            """)
            # 1. DB에서 값을 빼옵니다.
            fetched_fail_count = conn.execute(login_log_sql, {"user_id": result.user_id}).scalar()

            # 2. 기록이 없어서 None이면 0으로, 아니면 원래 숫자로 세팅합니다.
            fail_count = fetched_fail_count if fetched_fail_count is not None else 0

            if fail_count >= 5:
                return {"success": False, "message": "로그인 시도 가능 횟수(5회)를 초가했습니다."}


            db_hashed_password = result.password  # DB에 저장된 암호화된 비번

            # [핵심] 입력받은 평문 비번과 DB의 해시 비번을 비교
            if verify_password(password, db_hashed_password):
                # 3. 세션 저장 (결과 객체 필드명 주의)
                user_info = {
                    "name": result.name,
                    "user_id": result.user_id,
                    # 테이블에 없는 정보는 일단 빈값이나 세션 유지용으로만 둠
                    "role": result.role,  # DB에서 가져온 권한 정보 (예: 'USER', 'ADMIN')
                    "birth": str(result.birth_date) if result.birth_date else "", # 날짜 객체일 경우 문자열 변환
                    "gender": result.gender if result.gender else ""
                }
                req.session["user"] = user_info

                log_login(result.user_id, "login", "success")
                log_user_activity(result.user_id, "lgn103", req)

                return {"success": True, "message": f"{result.name}님, 환영합니다!",
                        "user": {
                        "access_token": "session-active",
                        "user_name": result.name,
                        "user_id": result.user_id,
                        "user_role": result.role,  # 이 값이 프론트엔드의 sessionStorage('user_role')로 들어갑니다.
                        "user_birth": str(result.birth_date) if result.birth_date else "",  # 추가
                        "user_gender": result.gender if result.gender else ""  # 추가
                }}
            else:
                log_login(result.user_id, "login", "fail")
                return {"success": False, "message": "아이디 또는 비밀번호가 일치하지 않습니다."}
        else:
            return {"success": False, "message": "아이디 또는 비밀번호가 일치하지 않습니다."}


    except Exception as e:
        print(f"🚨 로그인 로직 에러: {e}")
        return {"success": False, "message": "서버 오류가 발생했습니다."}
    finally:
        if conn:
            conn.close()