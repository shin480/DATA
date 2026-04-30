import smtplib
import uvicorn
from email.message import EmailMessage
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

# --- [설정] 발신 정보 ---
SENDER_EMAIL = "data260417@gmail.com"
SENDER_PASSWORD = "zvalcyxiydtnzewi"
RECEIVER_EMAIL = "suai8402@gmail.com"  # 내 계정


# 1. 메일 발송 함수
def send_button_email():
    # 내 로컬 서버의 /log 주소를 링크로 설정
    # (주의: 본인 PC에서만 작동하는 주소입니다)
    verify_url = f"http://127.0.0.1:8000/click-log?email={RECEIVER_EMAIL}"

    msg = EmailMessage()
    msg['Subject'] = "이 버튼을 누르면 서버에 로그가 찍힙니다!"
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL

    html_content = f"""
    <div style="text-align: center; padding: 20px; border: 2px solid #000;">
        <h2>로그 테스트 메일</h2>
        <p>아래 버튼을 누르면 실행 중인 파이썬 터미널에 로그가 찍힙니다.</p>
        <a href="{verify_url}" 
           style="background-color: #e74c3c; color: white; padding: 15px 25px; 
                  text-decoration: none; border-radius: 5px; font-weight: bold;">
           서버 로그 남기기
        </a>
    </div>
    """
    msg.add_alternative(html_content, subtype='html')

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        print(f"✅ {RECEIVER_EMAIL}로 테스트 메일을 보냈습니다!")
    except Exception as e:
        print(f"❌ 메일 발송 실패: {e}")


# 2. 버튼 클릭 시 호출될 서버 엔드포인트
@app.get("/click-log", response_class=HTMLResponse)
async def click_log(email: str):
    # --- 여기가 로그가 찍히는 부분입니다 ---
    print("\n" + "=" * 50)
    print(f"🔔 [로그 발생] 인증 버튼 클릭됨!")
    print(f"📩 클릭한 유저: {email}")
    print("=" * 50 + "\n")

    return "<h1>로그가 성공적으로 기록되었습니다! 터미널을 확인하세요.</h1>"


# 3. 실행 로직
if __name__ == "__main__":
    # 서버를 띄우기 전에 메일을 먼저 한 통 보냅니다.
    send_button_email()

    # 서버 실행
    print("🚀 서버가 대기 중입니다... 메일함을 확인하고 버튼을 눌러보세요!")
    uvicorn.run(app, host="127.0.0.1", port=8000)