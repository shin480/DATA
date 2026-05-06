from sqlalchemy import text
from typing import Dict, Any
from starlette.requests import Request
import time

from util.db import get_engine
from regist.hashing import verify_password


def check_user_password(info: Dict[str, Any], req: Request):
    # 1. 세션 체크 (로그인 여부 확인)
    session_user = req.session.get("user")
    if not session_user:
        return {"success": False, "message": "로그인이 만료되었습니다. 다시 로그인해주세요."}

    # 2. 프론트엔드에서 넘어온 데이터 추출
    input_pw = info.get("password")
    user_id = info.get("user_id")

    # 3. 보안 검증: 세션의 user_id와 요청의 user_id가 일치하는지 확인
    if session_user.get('user_id') != user_id:
        return {"success": False, "message": "잘못된 접근입니다."}

    conn = None
    try:
        # 4. DB 연결 및 해당 유저의 해시된 비밀번호 조회
        conn = get_engine()
        sql = text("""
            SELECT password 
            FROM users 
            WHERE user_id = :user_id
        """)

        result = conn.execute(sql, {"user_id": user_id}).fetchone()

        if result:
            db_hashed_pw = result.password  # DB에 저장된 암호화된 비밀번호

            # 5. 입력받은 비번(input_pw)과 DB 비번 비교 검증
            if verify_password(input_pw, db_hashed_pw):
                req.session["pw_auth_time"] = time.time()
                return {"success": True, "message": "본인 확인에 성공하였습니다."}
            else:
                return {"success": False, "message": "비밀번호가 일치하지 않습니다."}
        else:
            return {"success": False, "message": "사용자 정보를 찾을 수 없습니다."}

    except Exception as e:
        print(f"🚨 비밀번호 확인 중 에러: {e}")
        return {"success": False, "message": "데이터베이스 오류가 발생했습니다."}
    finally:
        if conn:
            conn.close()


def check_auth_status(req: Request):
    auth_time = req.session.get("pw_auth_time")

    if not auth_time:
        return {"is_authenticated": False}

    # 10분(600초)이 지났는지 계산
    if time.time() - auth_time > 600:
        del req.session["pw_auth_time"]  # 만료됐으면 삭제
        return {"is_authenticated": False}

    return {"is_authenticated": True}