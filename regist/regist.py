from fastapi import APIRouter, Request
from typing import Any, Dict
from sqlalchemy import text
from util.db import get_engine


def regist_user(info: Dict[str, Any], req: Request):
    conn = None
    try:
        # 1. 세션 데이터 확인
        required_agreed = req.session.get("requiredChecks")
        optional_list = req.session.get("optionalChecksList")  # [1] 또는 ['1'] 등으로 들어올 수 있음

        if not required_agreed:
            return {"success": False, "message": "필수 약관 동의가 필요합니다."}

        # 2. 기본 정보 추출
        email = info.get("email")
        # ... (name, birth, gender, pw 등 추출)

        conn = get_engine()

        # 3. users 테이블 가입
        user_sql = text("""
            INSERT INTO users (name, birth_date, gender, user_id, password)
            VALUES (:name, :birth, :gender, :email, :pw)
        """)
        conn.execute(user_sql, info)

        user_id = email  # user_id가 이메일인 경우

        # 4. user_terms_agreements 테이블 저장
        terms_sql = text("""
            INSERT INTO user_terms_agreements (user_id, term_id, is_agreed, agreed_at)
            VALUES (:user_id, :term_id, 1, NOW())
        """)

        # (A) 필수 약관 저장 (1~4번)
        for t_id in [1, 2, 3, 4]:
            conn.execute(terms_sql, {"user_id": user_id, "term_id": t_id})

        # (B) 선택 약관 저장 (화면상의 1번을 DB의 5번으로 매핑)
        if optional_list:
            # 리스트 안에 무엇이 들어있든(예: '1'),
            # 현재 선택 약관이 하나뿐이므로 체크가 되어 있다면 ID 5번으로 저장합니다.
            conn.execute(terms_sql, {"user_id": user_id, "term_id": 5})

        conn.commit()

        # 5. 세션 정리
        for key in ["requiredChecks", "optionalChecksList", "cert_code"]:
            req.session.pop(key, None)

        return {"success": True, "message": "회원가입이 완료되었습니다!"}

    except Exception as e:
        if conn: conn.rollback()
        print(f"🚨 가입 에러: {e}")
        return {"success": False, "message": "서버 오류가 발생했습니다."}
    finally:
        if conn: conn.close()