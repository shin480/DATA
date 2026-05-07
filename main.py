from typing import Dict, List, Any

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import RedirectResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

import pandas as pd
from util.db import get_engine
from util.es import get_es, NEWS_ECONOMY_INDEX

from model.model_main import app as pipeline_app
from datacleaning.cleaning import get_preprocessed_data
from pydantic import BaseModel
from viewpoint_classify.viewpoint_classify import update_perspective_to_es

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

@app.get("/viewpoint_classify")
def viewpoint_classify():
    update_perspective_to_es()
    return {
        "success": True,
        "message": "관점 분류 Top-3 저장 완료"
    }

# =========================================
# 키워드 상세 조회 API
# =========================================
@app.get("/api/keywords/{keyword}")
def get_keyword_detail(keyword: str):

    es = get_es()

    body = {
        "size": 50,
        "query": {
            "bool": {
                "should": [
                    {
                        "match_phrase": {
                            "title": {
                                "query": keyword,
                                "boost": 5
                            }
                        }
                    },
                    {
                        "match_phrase": {
                            "summary": {
                                "query": keyword,
                                "boost": 3
                            }
                        }
                    },
                    {
                        "match_phrase": {
                            "keywords": {
                                "query": keyword,
                                "boost": 4
                            }
                        }
                    },
                    {
                        "match_phrase": {
                            "clean_text": {
                                "query": keyword,
                                "boost": 1
                            }
                        }
                    }
                ],
                "minimum_should_match": 1
            }
        },
        "sort": [
            {
                "_score": {
                    "order": "desc"
                }
            },
            {
                "published_at": {
                    "order": "desc"
                }
            }
        ]
    }

    result = es.search(
        index=NEWS_ECONOMY_INDEX,
        body=body
    )

    hits = result["hits"]["hits"]

    total_count = len(hits)

    # =========================
    # 검색 결과 없을 때
    # =========================
    if total_count == 0:

        return {
            "category": "ECONOMY",
            "keyword": keyword,

            "pressCount": 0,

            "summary": f"{keyword} 관련 뉴스 데이터가 없습니다.",

            "aiCount": 0,

            "newsCount": 0,

            "flow": "데이터 없음",

            "sentiment": {
                "positive": 0,
                "neutral": 0,
                "negative": 0
            },

            "articles": []
        }

    # =========================
    # 감성 계산
    # =========================
    positive_count = 0
    neutral_count = 0
    negative_count = 0

    press_set = set()

    articles = []

    # =========================
    # 기사 리스트 생성
    # =========================
    for hit in hits:

        source = hit["_source"]

        press = source.get("press", "언론사 없음")

        sentiment = source.get("sentiment", "neutral")

        press_set.add(press)

        # 감성 카운트
        if sentiment == "positive":
            positive_count += 1

        elif sentiment == "negative":
            negative_count += 1

        else:
            neutral_count += 1

        # 기사 리스트
        articles.append({

            "article_id": source.get("article_id", ""),

            "source": press,

            "title": source.get("title", "제목 없음"),

            "desc":
                source.get("summary")
                or source.get("content", "")[:120],

            "link": source.get("url", "#"),

            "published_at":
                source.get("published_at", ""),

            "sentiment": sentiment,

            "highlight": False
        })

    # =========================
    # 감성 퍼센트 계산
    # =========================
    positive_percent = round(
        (positive_count / total_count) * 100
    )

    neutral_percent = round(
        (neutral_count / total_count) * 100
    )

    negative_percent = (
        100
        - positive_percent
        - neutral_percent
    )

    first_article = hits[0]["_source"]

    # =========================
    # 최종 반환
    # =========================
    return {

        "category": "ECONOMY",

        "keyword": keyword,

        "pressCount": len(press_set),

        "summary":
            first_article.get("summary")
            or f"{keyword} 관련 뉴스 {total_count}건 분석 결과입니다.",

        "aiCount": 1,

        "newsCount": total_count,

        "flow": "감성 분석",

        "sentiment": {

            "positive": positive_percent,

            "neutral": neutral_percent,

            "negative": negative_percent
        },

        "articles": articles
    }

@app.get("/api/search/suggest")
def search_suggest(q: str):
    es = get_es()

    if not q.strip():
        return []

    body = {
        "size": 20,

        "_source": [
            "keywords",
            "title"
        ],

        "query": {
            "bool": {
                "should": [

                    {
                        "wildcard": {
                            "keywords": {
                                "value": f"*{q}*",
                                "boost": 5
                            }
                        }
                    },

                    {
                        "match_phrase_prefix": {
                            "title": {
                                "query": q,
                                "boost": 3
                            }
                        }
                    }

                ],

                "minimum_should_match": 1
            }
        }
    }

    result = es.search(
        index=NEWS_ECONOMY_INDEX,
        body=body
    )

    suggestions = []

    for hit in result["hits"]["hits"]:
        source = hit["_source"]

        keyword_text = source.get("keywords", "")

        if keyword_text:
            for item in keyword_text.split(","):
                word = item.strip()

                if q in word and word not in suggestions:
                    suggestions.append(word)

        title = source.get("title", "")

        if q in title:
            for word in title.replace("…", " ").replace("·", " ").split():
                clean_word = word.strip("[]()\"'‘’“”,.")

                if q in clean_word and clean_word not in suggestions:
                    suggestions.append(clean_word)

        if len(suggestions) >= 6:
            break

    return suggestions[:6]