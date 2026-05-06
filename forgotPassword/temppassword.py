import string
import secrets  # 보안에 더 적합한 랜덤 모듈
from regist.emailvalidation import check_email_duplicate
from util.db import get_engine
from sqlalchemy import text
from regist.hashing import hash_password
from email.message import EmailMessage
import smtplib

SENDER_EMAIL = "data260417@gmail.com"
SENDER_PASSWORD = "zvalcyxiydtnzewi"

# --- [추가] 임시 비밀번호 생성 함수 ---
def generate_temp_password(length=8):
    """영문 대소문자와 숫자를 조합한 8자리 임시 비밀번호 생성"""
    characters = string.ascii_letters + string.digits
    return ''.join(secrets.choice(characters) for _ in range(length))


# --- [메인] 비밀번호 찾기 및 임시 비번 발송 함수 ---
def send_temporary_password(email: str):
    """
    1. 아이디(이메일) 존재 여부 확인
    2. 임시 비밀번호 생성
    3. DB 내 해당 유저의 비밀번호 업데이트
    4. 메일 발송
    """

    # 1. 아이디 존재 여부 확인
    if not check_email_duplicate(email):
        return {"success": False, "message": "등록되지 않은 아이디(이메일)입니다."}

    # 2. 임시 비밀번호 생성 (8자리)
    temp_pw = generate_temp_password(8)
    hashed_temp_pw = hash_password(temp_pw)  # DB 저장용 해싱

    # 3. DB 업데이트 (users 테이블의 password 컬럼 업데이트)
    conn = get_engine()
    try:
        # 실제 서비스 시에는 temp_pw를 해싱(암호화)해서 저장하는 것이 원칙입니다.
        sql = text("UPDATE users SET password = :pw WHERE user_id = :email")
        conn.execute(sql, {"pw": hashed_temp_pw, "email": email})
        conn.commit()
    except Exception as e:
        print(f"❌ 비밀번호 업데이트 실패: {e}")
        return {"success": False, "message": "비밀번호 갱신 중 서버 오류가 발생했습니다."}
    finally:
        conn.close()

    # 4. 메일 발송 설정
    msg = EmailMessage()
    msg['Subject'] = "[DATA] 요청하신 임시 비밀번호입니다."
    msg['From'] = SENDER_EMAIL
    msg['To'] = email

    html_content = f"""
        <div style="text-align: center; padding: 40px; border: 1px solid #ddd; border-radius: 10px; font-family: sans-serif;">
            <h2 style="color: #2c3e50;">임시 비밀번호 발송</h2>
            <p style="font-size: 16px; color: #555;">로그인 후 마이페이지에서 반드시 비밀번호를 변경해 주세요.</p>
            <div style="background-color: #f1f2f6; padding: 20px; margin: 20px 0; border-radius: 5px; border: 1px dashed #7f8c8d;">
                <span style="font-size: 28px; font-weight: bold; color: #2980b9; letter-spacing: 2px;">
                    {temp_pw}
                </span>
            </div>
            <p style="font-size: 13px; color: #e74c3c;">주의: 본 비밀번호는 임시로 발급된 것이므로 노출되지 않도록 주의하십시오.</p>
        </div>
    """
    msg.add_alternative(html_content, subtype='html')

    # 5. 실제 발송
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        return {"success": True, "message": "임시 비밀번호가 메일로 발송되었습니다."}
    except Exception as e:
        print(f"❌ 메일 발송 실패: {e}")
        return {"success": False, "message": "메일 발송 중 오류가 발생했습니다."}