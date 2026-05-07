from typing import Dict, List, Any

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import RedirectResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

import pandas as pd
from util.db import get_engine
from model.model_main import app as pipeline_app
from datacleaning.cleaning import get_preprocessed_data
from pydantic import BaseModel

from regist.emailvalidation import check_email_duplicate, send_cert_email, verify_certification_number
from regist.regist import regist_user

from login.login import login_user

from forgotPassword.temppassword import send_temporary_password

from mypage.updatepassword import update_user_password
from mypage.deleteaccount import delete_user_account
from mypage.passwordcheck import check_user_password, check_auth_status

app = FastAPI()
app.mount("/view", StaticFiles(directory="view"))
# 파이프라인 앱을 통째로 "/pipeline" 주소에 마운트
app.mount("/pipeline", pipeline_app)

app.add_middleware(SessionMiddleware, secret_key="motmachugetjyo")

@app.get("/")
def main():
    return RedirectResponse("/view/main.html")

# 테이블 내용 가져오기 테스트
@app.get('/read_table')
def read_table():
    df = pd.read_sql_table(table_name='users', con=get_engine())
    print(df.head())
    return df.to_dict() # http://127.0.0.1:8000/read_table

# 전처리 테스트
@app.get('/cleaning')
def data_cleaning():
    result = get_preprocessed_data()  # 아까 만든 전처리 함수 호출
    return result  # 처리 건수나 성공 메시지를 화면에 출력

# 이용약관 구조 정의
class TermsRequest(BaseModel):
    requiredChecks: bool  # 또는 str, 실제 데이터에 맞게 설정
    optionalChecksList: List[str] # 리스트 형태임을 명시

@app.post('/terms')
def terms(info: TermsRequest, req: Request):
    print(f"필수 동의 여부 : {info.requiredChecks} / 선택 동의 리스트 {info.optionalChecksList}")
    req.session["requiredChecks"] = info.requiredChecks
    req.session["optionalChecksList"] = info.optionalChecksList
    return info

@app.post('/send_cert')
def send_cert(info:Dict[str,str],req: Request):
    dup = check_email_duplicate(info["email"])
    if not dup:
        result = send_cert_email(info["email"])
        if result["success"]:
            # 반환받은 번호를 세션에 저장
            req.session["cert_code"] = result["code"]
            req.session["cert_email"] = info["email"]
            # 성공 응답 전송
            result["duplicate"] = False
            return result
        else:
            # 메일 서버 문제 등으로 발송 실패 시
            result["duplicate"] = False
            return result
    # 중복된 이메일인 경우
    return {"duplicate": True, "success": False}

@app.post('/verify_cert')
def verify_cert(info:Dict[str,str],req: Request):
    result = verify_certification_number(info["email"], info["code"])
    return result

@app.post("/regist")
def regist(info: Dict[str,Any], req: Request):
    print(info)
    result = regist_user(info, req)
    print(result)
    return result

@app.post("/login")
def login(info: Dict[str,Any], req: Request):
    print(f"로그인 시도: {info.get('email')}")
    result = login_user(info, req)
    return result

@app.post("/find_password")
def find_password(info: Dict[str,Any], req: Request):
    result = send_temporary_password(info["email"])
    return result

@app.post("/update_password")
def api_update_password(info: Dict[str, Any], req: Request):
    # 분리된 파일의 함수 호출
    return update_user_password(info, req)

@app.post("/delete_account")
def api_delete_account(info: Dict[str, Any], req: Request):
    return delete_user_account(info, req)

@app.post("/pw_check")
def api_pw_check(info: Dict[str, Any], req: Request):
    return check_user_password(info, req)

@app.post("/pw_check_time")
def pw_check_time(req: Request):
    return check_auth_status(req)