import smtplib
from email.message import EmailMessage
import random
from sqlalchemy import text
from util.db import get_engine
import os
from dotenv import load_dotenv

# env 파일에 있는 변수들을 파이썬 환경으로 불러옵니다.
load_dotenv()

# os 모듈을 사용해 환경변수에서 값을 쏙쏙 뽑아옵니다.
SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")

# 1. DB에서 이메일 중복 확인 함수
def check_email_duplicate(email: str):
    # 1. 엔진으로부터 연결을 시도합니다.
    conn = get_engine()
    try:
        # 2. 쿼리를 실행합니다.
        sql = text("SELECT COUNT(*) FROM users WHERE user_id = :email")
        result = conn.execute(sql, {"email": email}).fetchone()

        # 0보다 크면 중복(True), 아니면 False
        return result[0] > 0
    except Exception as e:
        print(e)
    finally:
        # 4. 에러 발생 여부와 상관없이 연결은 반드시 닫아줍니다.
        if conn is not None:
            conn.close()


def check_daily_limit(email: str):
    """
    해당 이메일의 오늘 하루 발송 횟수가 5회를 초과했는지 확인합니다.
    """
    conn = get_engine()
    try:
        # DATE(created_at)을 사용하여 오늘 날짜의 발송 기록만 카운트합니다.
        query = text("""
            SELECT COUNT(*) 
            FROM email_verifications 
            WHERE email = :email 
              AND DATE(created_at) = CURDATE()
        """)
        count = conn.execute(query, {"email": email}).scalar()

        # 5번 이상이면 발송 차단
        if count >= 5:
            return False
        return True
    except Exception as e:
        print(f"❌ 횟수 제한 확인 중 오류: {e}")
        return False
    finally:
        conn.close()

# 2. 메일 발송 함수
def send_cert_email(email: str):
    # 1. 오늘 하루 발송 횟수 제한 확인 (5회)
    if not check_daily_limit(email):
        print(f"⚠️ 발송 제한 초과: {email}")
        return {"success": False, "message": "오늘 발송 가능 횟수(5회)를 초과했습니다."}

    # 함수 내부에서 4자리 인증번호 직접 생성
    cert_code = str(random.randint(1000, 9999))

    conn = get_engine()
    try:
        # is_verified는 기본값 0(False)으로, created_at은 DB 설정에 따라 자동 생성되거나 현재시간 입력
        sql = text("""
                INSERT INTO email_verifications (email, verification_code, is_verified) 
                VALUES (:email, :code, 0)
            """)
        conn.execute(sql, {"email": email, "code": cert_code})
        conn.commit()  # INSERT 문이므로 커밋 필수
    except Exception as e:
        print(f"❌ DB 로그 저장 실패: {e}")
        return {"success": False, "message": "인증 정보 저장 중 오류가 발생했습니다."}
    finally:
        if conn is not None:
            conn.close()

    msg = EmailMessage()
    msg['Subject'] = "[DATA] 회원가입 인증번호입니다."
    msg['From'] = SENDER_EMAIL
    msg['To'] = email

    # 사용자에게 보여줄 HTML 내용
    html_content = f"""
        <div style="text-align: center; padding: 40px; border: 1px solid #ddd; border-radius: 10px;">
            <h2 style="color: #2c3e50;">회원가입 인증번호</h2>
            <p style="font-size: 16px; color: #555;">아래의 인증번호를 입력창에 입력해 주세요.</p>
            <div style="background-color: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 5px;">
                <span style="font-size: 32px; font-weight: bold; color: #e74c3c; letter-spacing: 5px;">
                    {cert_code}
                </span>
            </div>
            <p style="font-size: 13px; color: #999;">인증번호는 3분 동안 유효합니다.</p>
        </div>
        """
    msg.add_alternative(html_content, subtype='html')

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        return {"success": True, "code": cert_code}
    except Exception as e:
        print(f"❌ 메일 발송 실패: {e}")
        return {"success": False, "message": "메일 서버 통신 실패"}

# 3. 인증 확인 함수
def verify_certification_number(email: str, user_code: str):
    conn = get_engine()

    try:
        # 가장 최근의 미인증 인증 요청 조회
        query = text("""
            SELECT 
                verification_id,
                verification_code,
                created_at
            FROM email_verifications 
            WHERE email = :email 
              AND is_verified = 0
            ORDER BY created_at DESC 
            LIMIT 1
        """)

        result = conn.execute(query, {"email": email}).fetchone()

        if not result:
            return {
                "success": False,
                "message": "유효한 인증 요청이 없습니다. 다시 발송해주세요."
            }

        v_id = result[0]
        db_code = result[1]

        # DB 시간 기준으로 3분 만료 검사
        expire_query = text("""
            SELECT 
                CASE 
                    WHEN created_at < DATE_SUB(NOW(), INTERVAL 3 MINUTE)
                    THEN 1
                    ELSE 0
                END
            FROM email_verifications
            WHERE verification_id = :id
        """)

        is_expired = conn.execute(
            expire_query,
            {"id": v_id}
        ).scalar()

        if is_expired == 1:
            update_sql = text("""
                UPDATE email_verifications 
                SET is_verified = 2 
                WHERE verification_id = :id
            """)

            conn.execute(update_sql, {"id": v_id})
            conn.commit()

            return {
                "success": False,
                "message": "인증번호가 만료되었습니다. 다시 발송해주세요."
            }

        if db_code == user_code:
            update_sql = text("""
                UPDATE email_verifications 
                SET is_verified = 1 
                WHERE verification_id = :id
            """)

            conn.execute(update_sql, {"id": v_id})
            conn.commit()

            return {
                "success": True
            }

        else:
            update_sql = text("""
                UPDATE email_verifications 
                SET is_verified = 2 
                WHERE verification_id = :id
            """)

            conn.execute(update_sql, {"id": v_id})
            conn.commit()

            return {
                "success": False,
                "message": "인증번호가 틀렸습니다. 다시 받아주세요."
            }

    except Exception as e:
        print(f"❌ 인증 처리 중 오류: {e}")

        return {
            "success": False,
            "message": "서버 오류가 발생했습니다."
        }

    finally:
        conn.close()