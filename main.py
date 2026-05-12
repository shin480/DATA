from typing import Dict, List, Any
import os
import uuid

from fastapi import FastAPI, Query, UploadFile, File
from starlette.requests import Request
from starlette.responses import RedirectResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

import pandas as pd
from util.db import get_engine
from util.es import get_es, NEWS_ECONOMY_INDEX
from crawling.crawler import run_crawling_job
from sqlalchemy import text

from model.model_main import app as pipeline_app
from datacleaning.cleaning import get_preprocessed_data
from schemas import TermsRequest, VoteRequest
from viewpoint_classify.viewpoint_classify import update_perspective_to_es

from regist.emailvalidation import check_email_duplicate, send_cert_email, verify_certification_number
from regist.regist import regist_user

from login.login import login_user,logout_user

from forgotPassword.temppassword import send_temporary_password

from mypage.updatepassword import update_user_password
from mypage.deleteaccount import delete_user_account
from mypage.passwordcheck import check_user_password, check_auth_status
from mypage.article_view import view_log

from admin.data_admin import get_search_summary
from admin.user_admin import get_user_search

from collections import Counter
from model.model_main import startup as pipeline_startup


app = FastAPI()
app.mount("/view", StaticFiles(directory="view"))
# 파이프라인 앱을 통째로 "/pipeline" 주소에 마운트
app.mount("/pipeline", pipeline_app)

@app.on_event("startup")
async def startup_event():
    pipeline_startup()

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

@app.post("/logout")
def logout(req: Request):
    user = req.session.get("user")
    user_id = user.get("user_id") if user else "비회원/이미로그아웃"
    print(f"로그아웃: {user_id}")
    result = logout_user(req)
    return result

@app.post("/find_password")
def find_password(info: Dict[str,Any], req: Request):
    result = send_temporary_password(info["email"], req)
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

@app.get("/api/main/top5-keywords")
def get_main_top5_keywords():
    es = get_es()

    # 1. daily_keyword_metrics에서 가장 최신 날짜 찾기
    latest_result = es.search(
        index="daily_keyword_metrics",
        body={
            "size": 1,
            "_source": ["date"],
            "query": {
                "match_all": {}
            },
            "sort": [
                {
                    "date": {
                        "order": "desc"
                    }
                }
            ]
        }
    )

    latest_hits = latest_result["hits"]["hits"]

    if not latest_hits:
        return {
            "date": "",
            "keywords": []
        }

    latest_date = latest_hits[0]["_source"]["date"]

    # 2. 최신 날짜 기준 기사 수 Top5 키워드 조회
    result = es.search(
        index="daily_keyword_metrics",
        body={
            "size": 5,
            "_source": [
                "date",
                "keyword",
                "article_count"
            ],
            "query": {
                "term": {
                    "date": latest_date
                }
            },
            "sort": [
                {
                    "article_count": {
                        "order": "desc"
                    }
                }
            ]
        }
    )

    keywords = []

    for hit in result["hits"]["hits"]:
        source = hit["_source"]

        keywords.append({
            "keyword": source.get("keyword", ""),
            "article_count": source.get("article_count", 0)
        })

    return {
        "date": latest_date,
        "keywords": keywords
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

    total_count = result["hits"]["total"]["value"]
    # =========================
    # scope 개수 계산
    # =========================
    scope_set = set()

    for hit in hits:

        source = hit["_source"]

        scope_id = source.get("scopeID")

        if scope_id:
            scope_set.add(scope_id)

    ai_count = len(scope_set)

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
            "press": press,

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
    # 여론 흐름 계산
    # =========================
    perspective_count = {}

    for hit in hits:

        source = hit["_source"]

        perspectives = source.get("perspective", [])

        if not perspectives:
            continue

        top_perspective = perspectives[0].get("category")

        if not top_perspective:
            continue

        if top_perspective not in perspective_count:
            perspective_count[top_perspective] = 0

        perspective_count[top_perspective] += 1

    if perspective_count:
        flow = max(
            perspective_count,
            key=perspective_count.get
        )
    else:
        flow = "관점 분석 없음"

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

        "aiCount": ai_count,

        "newsCount": total_count,

        "flow": flow,

        "sentiment": {

            "positive": positive_percent,

            "neutral": neutral_percent,

            "negative": negative_percent
        },

        "articles": articles
    }
# =========================
# 랜덤 키워드 조회 API
# 최신 날짜 기준 기사 수 Top100 중 랜덤
# =========================
import random
@app.get("/api/main/random-keyword")
def get_random_keyword():

    es = get_es()

    latest_date_result = es.search(
        index="daily_keyword_metrics",
        body={
            "size": 1,
            "sort": [
                {
                    "date": {
                        "order": "desc"
                    }
                }
            ],
            "_source": ["date"]
        }
    )

    hits = latest_date_result["hits"]["hits"]

    if not hits:
        return {
            "keyword": None
        }

    latest_date = hits[0]["_source"]["date"]

    result = es.search(
        index="daily_keyword_metrics",
        body={
            "size": 100,
            "query": {
                "term": {
                    "date": latest_date
                }
            },
            "sort": [
                {
                    "article_count": {
                        "order": "desc"
                    }
                }
            ],
            "_source": [
                "keyword",
                "article_count",
                "date"
            ]
        }
    )

    keyword_hits = result["hits"]["hits"]

    if not keyword_hits:
        return {
            "keyword": None
        }

    random_hit = random.choice(keyword_hits)
    source = random_hit["_source"]

    return {
        "keyword": source.get("keyword"),
        "article_count": source.get("article_count", 0),
        "date": source.get("date")
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

def denormalize_sentiment_for_es(sentiment):
    if sentiment == "positive":
        return ["positive", "긍정"]
    if sentiment == "neutral":
        return ["neutral", "중립"]
    if sentiment == "negative":
        return ["negative", "부정"]
    return [sentiment]
# 기사 목록
@app.get("/api/articles")
def get_article_list(
    page: int = 1,
    size: int = 50,
    sentiment: str = "",
    viewpoint: str = ""
):
    es = get_es()

    from_value = (page - 1) * size

    must_conditions = []

    if sentiment:
        must_conditions.append({
            "terms": {
                "sentiment": denormalize_sentiment_for_es(sentiment)
            }
        })

    if viewpoint:
        must_conditions.append({
            "nested": {
                "path": "perspective",
                "query": {
                    "bool": {
                        "must": [
                            {
                                "term": {
                                    "perspective.category": viewpoint
                                }
                            },
                            {
                                "term": {
                                    "perspective.rank": 1
                                }
                            }
                        ]
                    }
                }
            }
        })

    query = {
        "match_all": {}
    }

    if must_conditions:
        query = {
            "bool": {
                "must": must_conditions
            }
        }

    body = {
        "from": from_value,
        "size": size,
        "query": query,
        "sort": [
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
    total = result["hits"]["total"]["value"]

    articles = []

    for hit in hits:
        source = hit["_source"]

        articles.append({
            "article_id": source.get("article_id") or hit["_id"],
            "source": source.get("press", "언론사 없음"),
            "press": source.get("press", "언론사 없음"),
            "title": source.get("title", "제목 없음"),
            "summary": source.get("summary") or source.get("content", "")[:120],
            "published_at": source.get("published_at", ""),
            "url": source.get("url", "#"),
            "sentiment": source.get("sentiment", ""),
        })

    return {
        "page": page,
        "size": size,
        "total": total,
        "articles": articles,
        "hasMore": from_value + len(articles) < total
    }

# 좋아요/싫어요
@app.post("/api/articles/vote")
def vote_article(info: VoteRequest, req: Request):
    es = get_es()
    db = get_engine()

    login_user = req.session.get("user")
    user_id = login_user.get("user_id") if login_user else None

    print("투표 세션:", req.session)
    print("투표 user_id:", user_id)
    print("투표 요청:", info)

    if not user_id:
        db.close()
        return {
            "success": False,
            "message": "로그인이 필요합니다."
        }

    article_id = info.article_id
    vote_type = info.vote_type

    if vote_type not in ["like", "hate"]:
        db.close()
        return {
            "success": False,
            "message": "잘못된 투표 타입입니다."
        }

    db.execute(
        text("""
            INSERT IGNORE INTO article_meta (article_id)
            VALUES(:article_id)
        """),
        {
            "article_id": article_id,
        }
    )

    # 프론트는 hate, DB는 dislike로 저장
    db_vote_type = "dislike" if vote_type == "hate" else "like"

    row = db.execute(
        text("""
            SELECT reaction_type
            FROM article_reactions
            WHERE user_id = :user_id
              AND article_id = :article_id
        """),
        {
            "user_id": user_id,
            "article_id": article_id
        }
    ).fetchone()

    old_vote = row[0] if row else None

    # DB에 저장된 dislike를 프론트 기준 hate로 변환
    old_vote_front = "hate" if old_vote == "dislike" else old_vote

    # 1. 처음 누름
    if old_vote is None:
        db.execute(
            text("""
                INSERT INTO article_reactions
                    (user_id, article_id, reaction_type)
                VALUES
                    (:user_id, :article_id, :reaction_type)
            """),
            {
                "user_id": user_id,
                "article_id": article_id,
                "reaction_type": db_vote_type
            }
        )

        if vote_type == "like":
            script = """
                if (ctx._source.like_count == null) ctx._source.like_count = 0;
                ctx._source.like_count += 1;
            """
        else:
            script = """
                if (ctx._source.hate_count == null) ctx._source.hate_count = 0;
                ctx._source.hate_count += 1;
            """

        current_vote = vote_type

    # 2. 같은 버튼 다시 누름 → 취소
    elif old_vote_front == vote_type:
        db.execute(
            text("""
                DELETE FROM article_reactions
                WHERE user_id = :user_id
                  AND article_id = :article_id
            """),
            {
                "user_id": user_id,
                "article_id": article_id
            }
        )

        if vote_type == "like":
            script = """
                if (ctx._source.like_count == null) ctx._source.like_count = 0;
                if (ctx._source.like_count > 0) ctx._source.like_count -= 1;
            """
        else:
            script = """
                if (ctx._source.hate_count == null) ctx._source.hate_count = 0;
                if (ctx._source.hate_count > 0) ctx._source.hate_count -= 1;
            """

        current_vote = ""

    # 3. 다른 버튼 누름 → 변경
    else:
        db.execute(
            text("""
                UPDATE article_reactions
                SET reaction_type = :reaction_type
                WHERE user_id = :user_id
                  AND article_id = :article_id
            """),
            {
                "reaction_type": db_vote_type,
                "user_id": user_id,
                "article_id": article_id
            }
        )

        if vote_type == "like":
            script = """
                if (ctx._source.like_count == null) ctx._source.like_count = 0;
                if (ctx._source.hate_count == null) ctx._source.hate_count = 0;

                ctx._source.like_count += 1;
                if (ctx._source.hate_count > 0) ctx._source.hate_count -= 1;
            """
        else:
            script = """
                if (ctx._source.like_count == null) ctx._source.like_count = 0;
                if (ctx._source.hate_count == null) ctx._source.hate_count = 0;

                ctx._source.hate_count += 1;
                if (ctx._source.like_count > 0) ctx._source.like_count -= 1;
            """

        current_vote = vote_type

    db.commit()
    db.close()

    es.update(
        index=NEWS_ECONOMY_INDEX,
        id=article_id,
        body={
            "script": {
                "source": script,
                "lang": "painless"
            }
        },
        refresh=True
    )

    result = es.get(index=NEWS_ECONOMY_INDEX, id=article_id)
    source = result["_source"]

    return {
        "success": True,
        "like_count": source.get("like_count", 0),
        "hate_count": source.get("hate_count", 0),
        "current_vote": current_vote
    }

@app.post("/view")
def view(info:Dict[str,Any], req: Request):
    view_log(info, req)

# 기사 상세페이지
@app.get("/api/articles/{article_id}")
def get_article_detail(article_id: str, req: Request):
    es = get_es()

    result = es.get(
        index=NEWS_ECONOMY_INDEX,
        id=article_id
    )

    source = result["_source"]

    current_scope_id = source.get("scopeID", "")
    current_sentiment = source.get("sentiment", "neutral")

    deep_news = []

    if current_scope_id:
        related_result = es.search(
            index=NEWS_ECONOMY_INDEX,
            body={
                "size": 20,
                "query": {
                    "bool": {
                        "must": [
                            {
                                "term": {
                                    "scopeID": current_scope_id
                                }
                            }
                        ],
                        "must_not": [
                            {
                                "term": {
                                    "article_id": article_id
                                }
                            }
                        ]
                    }
                },
                "sort": [
                    {
                        "published_at": {
                            "order": "desc"
                        }
                    }
                ]
            }
        )

        same_news = None
        diff_news = None

        for hit in related_result["hits"]["hits"]:
            item = hit["_source"]

            sentiment = item.get("sentiment", "neutral")

            news_item = {
                "article_id": item.get("article_id", ""),
                "title": item.get("title", ""),
                "press": item.get("press", ""),
                "sentiment": sentiment,
                "type": sentiment,
                "tag": "관련 뉴스" if sentiment == current_sentiment else "다른 시각추천"
            }

            if sentiment == current_sentiment and same_news is None:
                same_news = news_item

            if sentiment != current_sentiment and diff_news is None:
                diff_news = news_item

            if same_news and diff_news:
                break

        deep_news = []

        if same_news:
            deep_news.append(same_news)

        if diff_news:
            deep_news.append(diff_news)

    login_user = req.session.get("user")
    user_id = login_user.get("user_id") if login_user else None

    current_vote = ""

    if user_id:
        db = get_engine()

        row = db.execute(
            text("""
                SELECT reaction_type
                FROM article_reactions
                WHERE user_id = :user_id
                  AND article_id = :article_id
            """),
            {
                "user_id": user_id,
                "article_id": article_id
            }
        ).fetchone()

        db.close()

        if row:
            current_vote = "hate" if row[0] == "dislike" else row[0]

    return {
        "article_id": source.get("article_id", ""),
        "title": source.get("title", ""),
        "date": source.get("published_at", ""),
        "press": source.get("press", ""),
        "reporter": source.get("author", "기자 정보 없음"),
        "content": source.get("content") or source.get("clean_text", ""),
        "summary": source.get("summary", ""),
        "sourceUrl": source.get("url", ""),
        "sentiment": source.get("sentiment", ""),
        "likeCount": source.get("like_count", 0),
        "dislikeCount": source.get("hate_count", 0),
        "currentVote": current_vote,
        "deepNews": deep_news
    }

@app.get("/api/mypage/viewed-news")
def get_viewed_news(
    req: Request,
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=30)
):
    login_user = req.session.get("user")
    user_id = login_user.get("user_id") if login_user else None

    if not user_id:
        return {
            "total_count": 0,
            "articles": []
        }

    db = get_engine()

    try:
        offset = (page - 1) * size

        rows = db.execute(
            text("""
                SELECT article_id, created_at
                FROM article_views
                WHERE is_valid_view = true
                  AND user_id = :user_id
                ORDER BY created_at DESC
                LIMIT :size OFFSET :offset
            """),
            {
                "user_id": user_id,
                "size": size,
                "offset": offset
            }
        ).fetchall()

        total = db.execute(
            text("""
                SELECT COUNT(*)
                FROM article_views
                WHERE is_valid_view = true
                  AND user_id = :user_id
            """),
            {
                "user_id": user_id
            }
        ).scalar()

        articles = []

        for row in rows:
            row = dict(row._mapping)
            article_id = row["article_id"]

            try:
                es_doc = get_es().get(
                    index=NEWS_ECONOMY_INDEX,
                    id=str(article_id).strip()
                )

                source = es_doc["_source"]

                articles.append({
                    "article_id": article_id,
                    "title": source.get("title") or "제목 없음",
                    "description": (
                        source.get("content")
                        or source.get("summary")
                        or ""
                    )[:120],
                    "category": "경제",
                    "created_at": row["created_at"]
                })

            except Exception as e:
                print("ES 기사 조회 실패:", article_id, e)

        return {
            "total_count": total,
            "articles": articles
        }

    finally:
        db.close()

@app.get("/api/mypage/reactions")
def get_my_reactions(
    req: Request,
    type: str = Query(...),
    page: int = Query(1, ge=1),
    size: int = Query(10, ge=1, le=30)
):
    login_user = req.session.get("user")
    user_id = login_user.get("user_id") if login_user else None

    if not user_id:
        return {
            "total_count": 0,
            "articles": []
        }

    db = get_engine()

    try:
        offset = (page - 1) * size

        rows = db.execute(
            text("""
                SELECT article_id, created_at
                FROM article_reactions
                WHERE reaction_type = :reaction_type
                  AND user_id = :user_id
                ORDER BY created_at DESC
                LIMIT :size OFFSET :offset
            """),
            {
                "reaction_type": type,
                "user_id": user_id,
                "size": size,
                "offset": offset
            }
        ).fetchall()

        total = db.execute(
            text("""
                SELECT COUNT(*)
                FROM article_reactions
                WHERE reaction_type = :reaction_type
                  AND user_id = :user_id
            """),
            {
                "reaction_type": type,
                "user_id": user_id
            }
        ).scalar()

        articles = []

        for row in rows:
            row = dict(row._mapping)
            article_id = row["article_id"]

            try:
                es_doc = get_es().get(
                    index=NEWS_ECONOMY_INDEX,
                    id=str(article_id).strip()
                )

                source = es_doc["_source"]

                articles.append({
                    "article_id": article_id,
                    "title": source.get("title") or "제목 없음",
                    "description": (
                        source.get("content")
                        or source.get("summary")
                        or ""
                    )[:120],
                    "category": "경제",
                    "created_at": row["created_at"]
                })

            except Exception as e:
                print("ES 기사 조회 실패:", article_id, e)

        return {
            "total_count": total,
            "articles": articles
        }

    finally:
        db.close()

# =========================
# 메인 1위 키워드
# =========================
@app.get("/api/main/top-keyword")
def get_top_keyword():

    es = get_es()

    result = es.search(
        index="daily_top_issue_report",
        body={
            "size": 1,
            "sort": [
                {
                    "date": {
                        "order": "desc"
                    }
                }
            ]
        }
    )

    hits = result["hits"]["hits"]

    if not hits:
        return {
            "keyword": "데이터 없음",
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "status": "분석 데이터 없음",
            "chips": []
        }

    source = hits[0]["_source"]

    sentiment = source.get("sentiment_distribution", {})

    positive = sentiment.get("positive", 0)
    neutral = sentiment.get("neutral", 0)
    negative = sentiment.get("negative", 0)

    # 여론 흐름 계산
    if positive >= neutral and positive >= negative:
        status = "긍정 여론 우세"

    elif neutral >= positive and neutral >= negative:
        status = "중립 여론 우세"

    else:
        status = "부정 여론 우세"

    return {
        "keyword": source.get("top_keyword", "키워드 없음"),

        "positive": positive,
        "neutral": neutral,
        "negative": negative,

        "status": status,

        "chips": source.get("top_keywords", [])
    }
# =========================
# 지금 뜨는 뉴스 - 최근 1시간 반응순
# =========================
@app.get("/api/main/hot-news")
def get_hot_news():
    es = get_es()
    db = get_engine()

    rows = db.execute(
        # text("""
        #     SELECT
        #         article_id,
        #         SUM(CASE WHEN reaction_type = 'like' THEN 1 ELSE 0 END) AS like_count,
        #         SUM(CASE WHEN reaction_type IN ('hate', 'dislike') THEN 1 ELSE 0 END) AS hate_count,
        #         COUNT(*) AS reaction_count
        #     FROM article_reactions
        #     WHERE created_at >= DATE_SUB(NOW(), INTERVAL 1 HOUR)
        #     GROUP BY article_id
        #     ORDER BY reaction_count DESC
        #     LIMIT 5
        # """)
        text("""
                SELECT
                    article_id,
                    SUM(CASE WHEN reaction_type = 'like' THEN 1 ELSE 0 END) AS like_count,
                    SUM(CASE WHEN reaction_type IN ('hate', 'dislike') THEN 1 ELSE 0 END) AS hate_count,
                    COUNT(*) AS reaction_count
                FROM article_reactions
                GROUP BY article_id
                ORDER BY reaction_count DESC
                LIMIT 5
            """)
    ).fetchall()

    db.close()

    articles = []

    for row in rows:
        article_id = row.article_id

        try:
            result = es.get(
                index=NEWS_ECONOMY_INDEX,
                id=article_id
            )

            source = result["_source"]

            articles.append({
                "article_id": article_id,
                "title": source.get("title", "제목 없음"),
                "press": source.get("press", "언론사 없음"),
                "summary": source.get("summary") or source.get("content", "")[:120],
                "sentiment": source.get("sentiment", "neutral"),
                "like_count": int(row.like_count or 0),
                "hate_count": int(row.hate_count or 0),
                "reaction_count": int(row.reaction_count or 0)
            })

        except Exception as e:
            print("지금 뜨는 뉴스 ES 조회 실패:", article_id, e)
            continue

    return {
        "articles": articles
    }

# =========================
# 오늘의 지배적 여론
# =========================
@app.get("/api/main/dominant-opinions")
def get_dominant_opinions():
    es = get_es()

    result = es.search(
        index="news_scopes",
        body={
            "size": 5,
            "_source": [
                "scope_summary",
                "scope_keywords",
                "sentiment"
            ],
            "query": {
                "match_all": {}
            },
            "sort": [
                {
                    "news_count": {
                        "order": "desc"
                    }
                },
                {
                    "updated_at": {
                        "order": "desc"
                    }
                }
            ]
        }
    )

    opinions = []

    for hit in result["hits"]["hits"]:
        source = hit["_source"]

        keyword_text = source.get("scope_keywords") or ""

        keyword_list = [
            keyword.strip()
            for keyword in str(keyword_text).split(",")
            if keyword.strip()
        ]

        opinions.append({
            "title": source.get("scope_summary") or "요약 없음",
            "keyword": keyword_list[0] if keyword_list else "키워드 없음",
            "sentiment": source.get("sentiment") or "neutral"
        })

    return {
        "opinions": opinions
    }

# =========================
# 같은 이슈, 다른 해석
# =========================
@app.get("/api/main/sentiment-compare")
def get_sentiment_compare(keyword: str):
    es = get_es()

    sentiments = ["positive", "negative", "neutral"]
    articles = []

    for sentiment in sentiments:
        result = es.search(
            index="news_economy",
            body={
                "size": 1,
                "_source": [
                    "article_id",
                    "title",
                    "summary",
                    "press",
                    "sentiment",
                    "published_at"
                ],
                "query": {
                    "bool": {
                        "must": [
                            {
                                "match": {
                                    "title": keyword
                                }
                            },
                            {
                                "term": {
                                    "sentiment": sentiment
                                }
                            }
                        ]
                    }
                },
                "sort": [
                    {
                        "published_at": {
                            "order": "desc"
                        }
                    }
                ]
            }
        )

        hits = result["hits"]["hits"]

        if not hits:
            continue

        source = hits[0]["_source"]

        articles.append({
            "article_id": source.get("article_id") or hits[0]["_id"],
            "title": source.get("title") or "기사 제목 없음",
            "summary": source.get("summary") or "",
            "press": source.get("press") or "언론사 없음",
            "sentiment": source.get("sentiment") or sentiment
        })

    return {
        "articles": articles
    }

# =========================================
# 배너 관리 API
# =========================================

@app.get("/api/admin/banners")
def get_admin_banners():
    db = get_engine()

    try:
        rows = db.execute(text("""
            SELECT
                banner_id,
                banner_type,
                title,
                landing_url,
                image_url,
                content,
                is_pinned,
                DATE_FORMAT(start_at, '%Y-%m-%d') AS start_at,
                DATE_FORMAT(end_at, '%Y-%m-%d') AS end_at,
                is_active
            FROM banners
            ORDER BY is_pinned DESC, created_at DESC
        """)).mappings().all()

        return {
            "banners": [dict(row) for row in rows]
        }

    finally:
        db.close()


@app.post("/api/admin/banners")
def create_admin_banner(info: Dict[str, Any]):
    db = get_engine()

    try:
        db.execute(text("""
            INSERT INTO banners (
                banner_type,
                title,
                landing_url,
                image_url,
                content,
                is_pinned,
                start_at,
                end_at,
                is_active
            )
            VALUES (
                :banner_type,
                :title,
                :landing_url,
                :image_url,
                :content,
                :is_pinned,
                :start_at,
                :end_at,
                :is_active
            )
        """), {
            "title": info.get("title"),
            "banner_type": info.get("banner_type", "notice"),
            "landing_url": info.get("landing_url"),
            "image_url": info.get("image_url"),
            "content": info.get("content"),
            "is_pinned": info.get("is_pinned", False),
            "start_at": info.get("start_at"),
            "end_at": info.get("end_at"),
            "is_active": info.get("is_active", True)
        })

        db.commit()

        return {
            "success": True,
            "message": "배너가 등록되었습니다."
        }

    except Exception as e:
        db.rollback()
        print("배너 등록 실패:", e)

        return {
            "success": False,
            "message": "배너 등록 중 오류가 발생했습니다."
        }

    finally:
        db.close()


# =========================================
# 배너 이미지 업로드 API
# =========================================

@app.post("/api/admin/banners/upload-image")
async def upload_banner_image(file: UploadFile = File(...)):
    upload_dir = "view/img/banners"

    os.makedirs(upload_dir, exist_ok=True)

    original_name = file.filename
    ext = os.path.splitext(original_name)[1].lower()

    if ext not in [".png", ".jpg", ".jpeg", ".webp"]:
        return {
            "success": False,
            "message": "PNG, JPG, JPEG, WEBP 이미지만 업로드할 수 있습니다."
        }

    saved_name = f"{uuid.uuid4().hex}{ext}"
    saved_path = os.path.join(upload_dir, saved_name)

    content = await file.read()

    with open(saved_path, "wb") as f:
        f.write(content)

    return {
        "success": True,
        "image_url": f"/view/img/banners/{saved_name}"
    }

@app.put("/api/admin/banners/{banner_id}")
def update_admin_banner(banner_id: int, info: Dict[str, Any]):
    db = get_engine()

    try:
        db.execute(text("""
            UPDATE banners
            SET
                banner_type = :banner_type,
                title = :title,
                landing_url = :landing_url,
                image_url = :image_url,
                content = :content,
                is_pinned = :is_pinned,
                start_at = :start_at,
                end_at = :end_at,
                is_active = :is_active
            WHERE banner_id = :banner_id
        """), {
            "banner_id": banner_id,
            "banner_type": info.get("banner_type", "notice"),
            "title": info.get("title"),
            "landing_url": info.get("landing_url"),
            "image_url": info.get("image_url"),
            "content": info.get("content"),
            "is_pinned": info.get("is_pinned", False),
            "start_at": info.get("start_at"),
            "end_at": info.get("end_at"),
            "is_active": info.get("is_active", True)
        })

        db.commit()

        return {
            "success": True,
            "message": "배너가 수정되었습니다."
        }

    except Exception as e:
        db.rollback()
        print("배너 수정 실패:", e)

        return {
            "success": False,
            "message": "배너 수정 중 오류가 발생했습니다."
        }

    finally:
        db.close()


@app.delete("/api/admin/banners/{banner_id}")
def delete_admin_banner(banner_id: int):
    db = get_engine()

    try:
        db.execute(text("""
            DELETE FROM banners
            WHERE banner_id = :banner_id
        """), {
            "banner_id": banner_id
        })

        db.commit()

        return {
            "success": True,
            "message": "배너가 삭제되었습니다."
        }

    except Exception as e:
        db.rollback()
        print("배너 삭제 실패:", e)

        return {
            "success": False,
            "message": "배너 삭제 중 오류가 발생했습니다."
        }

    finally:
        db.close()


@app.get("/api/banner/active-list")
def get_active_banner_list():
    db = get_engine()

    try:
        rows = db.execute(text("""
            SELECT
                banner_id,
                banner_type,
                title,
                landing_url,
                image_url,
                content,
                is_pinned,
                start_at,
                end_at,
                created_at,
                updated_at
            FROM banners
            WHERE is_active = TRUE
              AND DATE(start_at) <= CURDATE()
              AND DATE(end_at) >= CURDATE()
            ORDER BY is_pinned DESC, created_at DESC
        """)).mappings().all()

        return {
            "banners": [dict(row) for row in rows]
        }

    finally:
        db.close()

@app.get("/crawl")
async def crawl():
    return await run_crawling_job()

# =========================
# 메인 2차 관점 분석
# =========================
@app.get("/api/main/viewpoint-analysis")
def get_main_viewpoint_analysis():

    es = get_es()

    # =========================
    # 1위 키워드 조회
    # =========================
    top_result = es.search(
        index="daily_top_issue_report",
        body={
            "size": 1,
            "sort": [
                {
                    "date": {
                        "order": "desc"
                    }
                }
            ]
        }
    )

    top_hits = top_result["hits"]["hits"]

    if not top_hits:
        return {
            "keyword": "데이터 없음",
            "groups": []
        }

    top_keyword = (
        top_hits[0]["_source"]
        .get("top_keyword", "")
    )

    if not top_keyword:
        return {
            "keyword": "키워드 없음",
            "groups": []
        }

    # =========================
    # 1위 키워드 기사 조회
    # =========================
    result = es.search(
        index=NEWS_ECONOMY_INDEX,
        body={
            "size": 1000,
            "_source": [
                "perspective"
            ],
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": top_keyword,
                                "fields": [
                                    "title^3",
                                    "summary",
                                    "keywords",
                                    "clean_text"
                                ]
                            }
                        }
                    ]
                }
            }
        }
    )

    hits = result["hits"]["hits"]

    # =========================
    # 그룹 정의
    # =========================
    group_map = {
        "책임 소재": [
            "정부 책임",
            "기업 책임",
            "개인 책임",
            "외부 책임",
            "복합 책임"
        ],

        "태도 및 감성": [
            "비판적 태도",
            "우려",
            "기대",
            "성과 예찬"
        ],

        "정보 전달 및 분석": [
            "단순 전달",
            "원인 분석",
            "결과 분석",
            "대응 분석",
            "전망 분석"
        ],

        "정책 개입": [
            "정부 개입 강조",
            "시장 자율 강조"
        ],

        "환경 요인": [
            "외부 요인(글로벌)",
            "정책 요인(국내)"
        ]
    }

    # =========================
    # 관점 집계
    # =========================
    perspective_count = {}

    total = 0

    for hit in hits:

        perspectives = (
            hit["_source"]
            .get("perspective", [])
        )

        for item in perspectives:

            if item.get("rank") != 1:
                continue

            category = item.get("category")

            if not category:
                continue

            perspective_count[category] = (
                perspective_count.get(category, 0) + 1
            )

            total += 1

    # =========================
    # 그룹별 반환
    # =========================
    groups = []

    for group_name, categories in group_map.items():

        items = []

        for category in categories:

            count = perspective_count.get(category, 0)

            percent = (
                round((count / total) * 100)
                if total else 0
            )

            items.append({
                "title": category,
                "percent": percent,
                "count": count,
                "reason": f"{category} 관점으로 분류된 기사 {count}건."
            })

        items.sort(
            key=lambda x: x["percent"],
            reverse=True
        )

        groups.append({
            "group": group_name,
            "items": items
        })

    return {
        "keyword": top_keyword,
        "groups": groups
    }

# =========================================
# 관점 상세 조회 API
# viewpoint_master 기반
# =========================================

def normalize_sentiment(sentiment):
    if sentiment in ["positive", "긍정"]:
        return "positive"

    if sentiment in ["negative", "부정"]:
        return "negative"

    return "neutral"


def parse_keywords(raw_keywords):
    if not raw_keywords:
        return []

    if isinstance(raw_keywords, list):
        return [
            str(keyword).strip()
            for keyword in raw_keywords
            if str(keyword).strip()
        ]

    if isinstance(raw_keywords, str):
        return [
            keyword.strip()
            for keyword in raw_keywords.split(",")
            if keyword.strip()
        ]

    return []


def format_base_date(date_text):
    if not date_text:
        return "기준일"

    date_text = str(date_text)

    if len(date_text) >= 10:
        return "기준일 " + date_text[:10].replace("-", ".")

    return "기준일 " + date_text


def calc_sentiment_percent(sentiment_counter, total_count):
    positive_count = sentiment_counter.get("positive", 0)
    neutral_count = sentiment_counter.get("neutral", 0)
    negative_count = sentiment_counter.get("negative", 0)

    if total_count > 0:
        positive_percent = round((positive_count / total_count) * 100)
        neutral_percent = round((neutral_count / total_count) * 100)
        negative_percent = 100 - positive_percent - neutral_percent
    else:
        positive_percent = 0
        neutral_percent = 0
        negative_percent = 0

    return {
        "positive": {
            "percent": positive_percent,
            "count": positive_count
        },
        "neutral": {
            "percent": neutral_percent,
            "count": neutral_count
        },
        "negative": {
            "percent": negative_percent,
            "count": negative_count
        }
    }


def get_viewpoint_master(es):
    result = es.search(
        index="viewpoint_master",
        body={
            "size": 100,
            "query": {
                "match_all": {}
            },
            "sort": [
                {
                    "sort_order": {
                        "order": "asc"
                    }
                }
            ]
        }
    )

    items = []

    for hit in result["hits"]["hits"]:
        source = hit["_source"]

        items.append({
            "group_name": source.get("group_name", ""),
            "display_title": source.get("display_title", ""),
            "es_category": source.get("es_category", ""),
            "icon": source.get("icon", "📌"),
            "color_key": source.get("color_key", "teal"),
            "sort_order": source.get("sort_order", 999),
            "analysis_title": source.get("analysis_title", ""),
            "analysis_desc": source.get("analysis_desc", "")
        })

    return items


def get_rank1_count_map(es):
    result = es.search(
        index=NEWS_ECONOMY_INDEX,
        body={
            "size": 0,
            "aggs": {
                "perspective_nested": {
                    "nested": {
                        "path": "perspective"
                    },
                    "aggs": {
                        "rank_1_only": {
                            "filter": {
                                "term": {
                                    "perspective.rank": 1
                                }
                            },
                            "aggs": {
                                "categories": {
                                    "terms": {
                                        "field": "perspective.category",
                                        "size": 100
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    )

    buckets = (
        result
        .get("aggregations", {})
        .get("perspective_nested", {})
        .get("rank_1_only", {})
        .get("categories", {})
        .get("buckets", [])
    )

    return {
        bucket["key"]: bucket["doc_count"]
        for bucket in buckets
    }


def get_articles_by_perspective(es, es_category, size=500, recent_7days=False):
    filter_query = [
        {
            "nested": {
                "path": "perspective",
                "query": {
                    "bool": {
                        "must": [
                            {
                                "term": {
                                    "perspective.rank": 1
                                }
                            },
                            {
                                "term": {
                                    "perspective.category": es_category
                                }
                            }
                        ]
                    }
                }
            }
        }
    ]

    if recent_7days:
        filter_query.append({
            "range": {
                "published_at": {
                    "gte": "now-7d/d",
                    "lte": "now"
                }
            }
        })

    result = es.search(
        index=NEWS_ECONOMY_INDEX,
        body={
            "size": size,
            "_source": [
                "article_id",
                "title",
                "content",
                "summary",
                "press",
                "url",
                "published_at",
                "sentiment",
                "keywords",
                "like_count",
                "hate_count",
                "perspective"
            ],
            "query": {
                "bool": {
                    "filter": filter_query
                }
            },
            "sort": [
                {
                    "published_at": {
                        "order": "desc"
                    }
                }
            ]
        }
    )

    articles = []

    for hit in result["hits"]["hits"]:
        source = hit["_source"]

        articles.append({
            "article_id": source.get("article_id", hit.get("_id", "")),
            "title": source.get("title", "제목 없음"),
            "press": source.get("press", "언론사 없음"),
            "summary": source.get("summary") or source.get("content", "")[:120],
            "url": source.get("url", "#"),
            "published_at": source.get("published_at", ""),
            "sentiment": normalize_sentiment(source.get("sentiment", "neutral")),
            "keywords": parse_keywords(source.get("keywords")),
            "like_count": source.get("like_count", 0) or 0,
            "hate_count": source.get("hate_count", 0) or 0
        })

    return articles


@app.get("/api/viewpoints/detail")
def get_viewpoint_detail(viewpoint: str = ""):
    es = get_es()

    master_items = get_viewpoint_master(es)

    if not master_items:
        return {
            "title": "관점 데이터 없음",
            "icon": "📌",
            "es_category": "",
            "color_key": "teal",
            "percent": 0,
            "count": 0,
            "total_count": 0,
            "date": "기준일",
            "analysis_title": "관점 기준 데이터가 없습니다.",
            "analysis_desc": "viewpoint_master 인덱스에 관점 기준 데이터를 먼저 등록해야 합니다.",
            "sentiment": {
                "positive": {"percent": 0, "count": 0},
                "neutral": {"percent": 0, "count": 0},
                "negative": {"percent": 0, "count": 0}
            },
            "keywords": [],
            "articles": [],
            "compare_groups": []
        }

    title_to_master = {
        item["display_title"]: item
        for item in master_items
    }

    es_to_master = {
        item["es_category"]: item
        for item in master_items
    }

    # rank 1 관점별 전체 기사 수
    es_count_map = get_rank1_count_map(es)
    total_count = sum(es_count_map.values())

    # 선택 관점 결정
    selected_master = None

    if viewpoint:
        selected_master = title_to_master.get(viewpoint)

    if selected_master is None and es_count_map:
        top_es_category = max(es_count_map, key=es_count_map.get)
        selected_master = es_to_master.get(top_es_category)

    if selected_master is None:
        selected_master = master_items[0]

    selected_title = selected_master["display_title"]
    selected_es_category = selected_master["es_category"]
    selected_group_name = selected_master["group_name"]

    selected_count = es_count_map.get(selected_es_category, 0)
    selected_percent = round((selected_count / total_count) * 100) if total_count else 0

    # 선택 관점 기사 전체 조회
    selected_articles_all = get_articles_by_perspective(
        es=es,
        es_category=selected_es_category,
        size=10000,
        recent_7days=False
    )

    latest_date = ""

    for article in selected_articles_all:
        if article["published_at"] and not latest_date:
            latest_date = article["published_at"]

    # 감성 분포
    sentiment_counter = Counter(
        article["sentiment"]
        for article in selected_articles_all
    )

    selected_sentiment = calc_sentiment_percent(
        sentiment_counter,
        len(selected_articles_all)
    )

    # 주요 키워드 TOP 8
    keyword_counter = Counter()

    for article in selected_articles_all:
        for keyword in article["keywords"]:
            keyword_counter[keyword] += 1

    keywords = [
        keyword
        for keyword, count in keyword_counter.most_common(8)
    ]

    # 관련 기사 TOP 5
    # 기준: 최근 7일 + 선택 관점 + 대표 키워드 포함 + 최신순
    recent_articles = get_articles_by_perspective(
        es=es,
        es_category=selected_es_category,
        size=500,
        recent_7days=True
    )

    representative_keywords = set(keywords[:5])

    if representative_keywords:
        filtered_articles = [
            article
            for article in recent_articles
            if representative_keywords.intersection(set(article["keywords"]))
        ]
    else:
        filtered_articles = recent_articles

    filtered_articles.sort(
        key=lambda article: article["published_at"] or "",
        reverse=True
    )

    top_articles = []

    for article in filtered_articles[:5]:
        top_articles.append({
            "article_id": article["article_id"],
            "press": article["press"],
            "title": article["title"],
            "summary": article["summary"],
            "sentiment": article["sentiment"],
            "published_at": article["published_at"],
            "url": article["url"]
        })

    # 같은 그룹 관점만 비교 카드 생성
    same_group_items = []

    for item in master_items:
        if item["group_name"] != selected_group_name:
            continue

        item_es_category = item["es_category"]
        item_count = es_count_map.get(item_es_category, 0)
        item_percent = round((item_count / total_count) * 100) if total_count else 0

        item_articles = get_articles_by_perspective(
            es=es,
            es_category=item_es_category,
            size=500,
            recent_7days=False
        )

        item_sentiment_counter = Counter(
            article["sentiment"]
            for article in item_articles
        )

        item_sentiment = calc_sentiment_percent(
            item_sentiment_counter,
            len(item_articles)
        )

        same_group_items.append({
            "title": item["display_title"],
            "es_category": item_es_category,
            "percent": item_percent,
            "count": item_count,
            "color_key": item.get("color_key", "teal"),
            "analysis_title": item.get("analysis_title") or f"{item['display_title']} 분석",
            "analysis_desc": item.get("analysis_desc") or f"{item['display_title']}에 해당하는 기사 흐름입니다.",
            "core": item.get("analysis_desc") or f"{item['display_title']} 관점의 기사 흐름을 확인합니다.",
            "sentiment": item_sentiment
        })

    compare_groups = [
        {
            "group": selected_group_name,
            "items": same_group_items
        }
    ]

    return {
        "title": selected_title,
        "icon": selected_master.get("icon", "📌"),
        "es_category": selected_es_category,
        "color_key": selected_master.get("color_key", "teal"),
        "percent": selected_percent,
        "count": selected_count,
        "total_count": total_count,
        "date": format_base_date(latest_date),
        "analysis_title": selected_master.get("analysis_title") or f"{selected_title} 분석",
        "analysis_desc": selected_master.get("analysis_desc") or f"{selected_title}에 해당하는 기사 흐름을 분석한 결과입니다.",
        "sentiment": selected_sentiment,
        "keywords": keywords,
        "articles": top_articles,
        "compare_groups": compare_groups
    }

# 스콥 상세
@app.get("/api/scopes/random")
def get_random_scope_detail():
    es = get_es()

    result = es.search(
        index="news_scopes",
        body={
            "size": 1,
            "query": {
                "function_score": {
                    "query": {
                        "match_all": {}
                    },
                    "random_score": {}
                }
            }
        }
    )

    hits = result["hits"]["hits"]

    if not hits:
        return {
            "title": "분석 데이터 없음",
            "summary": "표시할 AI 뉴스 분석 데이터가 없습니다.",
            "keywords": [],
            "sentimentDist": {
                "positive": 0,
                "neutral": 0,
                "negative": 0
            },
            "viewpoints": [],
            "articleCount": 0,
            "lastUpdated": "-",
            "articles": []
        }

    scope_id = hits[0]["_source"].get("scopeID") or hits[0]["_id"]

    return get_scope_detail(scope_id)

@app.get("/api/scopes/{scope_id}")
def get_scope_detail(scope_id: str):
    es = get_es()

    # 1. news_scopes에서 스콥 기본 정보 조회
    try:
        scope_result = es.get(
            index="news_scopes",
            id=scope_id
        )

        scope_source = scope_result["_source"]

    except Exception as e:
        print("스콥 조회 실패:", e)

        scope_source = {
            "scopeID": scope_id,
            "scopeTitle": "분석 데이터 없음",
            "scope_keywords": "",
            "updated_at": ""
        }

    # 2. news_economy에서 같은 scopeID 기사 조회
    result = es.search(
        index=NEWS_ECONOMY_INDEX,
        body={
            "size": 100,
            "query": {
                "term": {
                    "scopeID": scope_id
                }
            },
            "sort": [
                {
                    "published_at": {
                        "order": "desc"
                    }
                }
            ]
        }
    )

    hits = result["hits"]["hits"]

    articles = []

    sentiment_count = {
        "positive": 0,
        "neutral": 0,
        "negative": 0
    }

    viewpoint_count = {}

    for hit in hits:
        source = hit["_source"]

        sentiment = source.get("sentiment", "neutral")

        if sentiment not in sentiment_count:
            sentiment = "neutral"

        sentiment_count[sentiment] += 1

        keyword_text = source.get("keywords") or ""

        keyword_list = [
            keyword.strip()
            for keyword in keyword_text.split(",")
            if keyword.strip()
        ]

        perspectives = source.get("perspective", [])

        for item in perspectives:
            category = item.get("category")

            if category:
                viewpoint_count[category] = viewpoint_count.get(category, 0) + 1

        article_id = source.get("article_id") or hit["_id"]

        articles.append({
            "article_id": article_id,
            "sentiment": sentiment,
            "press": source.get("press", "언론사 없음"),
            "time": source.get("published_at", "")[11:16],
            "title": source.get("title", "제목 없음"),
            "summary": source.get("summary") or source.get("content", "")[:120],
            "keywords": keyword_list[:3],
            "url": f"/view/article_detail.html?article_id={article_id}"
        })

    total = len(articles)

    if total > 0:
        positive_percent = round((sentiment_count["positive"] / total) * 100)
        neutral_percent = round((sentiment_count["neutral"] / total) * 100)
        negative_percent = 100 - positive_percent - neutral_percent
    else:
        positive_percent = 0
        neutral_percent = 0
        negative_percent = 0

    top_viewpoints = sorted(
        viewpoint_count.items(),
        key=lambda x: x[1],
        reverse=True
    )[:4]

    scope_keyword_text = scope_source.get("scope_keywords") or ""

    scope_keywords = [
        keyword.strip()
        for keyword in scope_keyword_text.split(",")
        if keyword.strip()
    ]

    updated_at = str(scope_source.get("updated_at") or "")

    return {
        "scopeID": scope_source.get("scopeID", scope_id),
        "title": scope_source.get("scopeTitle", "AI 뉴스 분석"),
        "summary": scope_source.get("scope_summary")
           or f"{scope_source.get('scopeTitle', '해당 이슈')} 관련 기사 {total}건을 분석한 결과입니다.",
        "keywords": scope_keywords[:5],
        "scopeSentiment": scope_source.get("sentiment", ""),
        "scopeSentimentScore": scope_source.get("sentiment_score", 0),
        "sentimentDist": {
            "positive": positive_percent,
            "neutral": neutral_percent,
            "negative": negative_percent
        },
        "viewpoints": [
            {
                "name": item[0],
                "percent": round((item[1] / total) * 100) if total else 0
            }
            for item in top_viewpoints
        ],
        "articleCount": scope_source.get("news_count") or total,
        "lastUpdated": updated_at[11:16] if len(updated_at) >= 16 else "-",
        "articles": articles
    }

@app.post("/api/batch/daily-keyword-metrics")
def create_daily_keyword_metrics(target_date: str = None):
    es = get_es()

    # target_date를 안 넣으면 keywords가 있는 최신 기사 날짜 기준으로 집계
    if target_date is None:
        latest_result = es.search(
            index=NEWS_ECONOMY_INDEX,
            body={
                "size": 1,
                "_source": ["published_at"],
                "query": {
                    "bool": {
                        "must": [
                            {
                                "exists": {
                                    "field": "keywords"
                                }
                            }
                        ],
                        "must_not": [
                            {
                                "term": {
                                    "keywords": ""
                                }
                            }
                        ]
                    }
                },
                "sort": [
                    {
                        "published_at": {
                            "order": "desc"
                        }
                    }
                ]
            }
        )

        latest_hits = latest_result["hits"]["hits"]

        if not latest_hits:
            return {
                "success": False,
                "message": "keywords가 있는 기사가 없습니다."
            }

        target_date = latest_hits[0]["_source"]["published_at"][:10]

    start_at = f"{target_date}T00:00:00"
    end_at = f"{target_date}T23:59:59"

    result = es.search(
        index=NEWS_ECONOMY_INDEX,
        body={
            "size": 10000,
            "_source": [
                "article_id",
                "published_at",
                "keywords"
            ],
            "query": {
                "bool": {
                    "filter": [
                        {
                            "range": {
                                "published_at": {
                                    "gte": start_at,
                                    "lte": end_at
                                }
                            }
                        },
                        {
                            "exists": {
                                "field": "keywords"
                            }
                        }
                    ],
                    "must_not": [
                        {
                            "term": {
                                "keywords": ""
                            }
                        }
                    ]
                }
            }
        }
    )

    hits = result["hits"]["hits"]

    keyword_counter = Counter()

    stopwords = {
        "com", "co", "kr", "www", "http", "https",
        "db", "photo", "newsis", "yna", "yonhap",
        "graphics", "graphic",
        "그래픽", "사진", "기자", "제공", "금지",
        "관련", "종합", "단독", "속보", "뉴스",
        "보도", "기사", "무단", "전재", "재배포",
        "습니다", "합니다", "했습니다", "있습니다", "없습니다",
        "됩니다", "됐습니다"
    }

    bad_endings = (
        "습니다",
        "합니다",
        "했습니다",
        "됩니다",
        "됐습니다",
        "있습니다",
        "없습니다",
        "한다고",
        "했다고",
        "된다",
        "됐다",
        "한다",
        "했다",
        "있다",
        "없다",
        "하다",
        "되다"
    )

    for hit in hits:
        source = hit["_source"]

        raw_keywords = source.get("keywords", "")

        if isinstance(raw_keywords, list):
            keyword_list = raw_keywords
        else:
            keyword_list = str(raw_keywords).split(",")

        # 동일 기사 안의 동일 키워드는 1번만 카운트
        unique_keywords = set()

        for keyword in keyword_list:
            keyword = str(keyword).strip().lower()

            keyword = (
                keyword.replace("#", "")
                       .replace('"', "")
                       .replace("'", "")
                       .replace("“", "")
                       .replace("”", "")
                       .replace("‘", "")
                       .replace("’", "")
                       .replace("…", "")
                       .replace("·", "")
                       .replace(".", "")
                       .replace(",", "")
                       .replace("?", "")
                       .replace("!", "")
                       .strip()
            )

            if not keyword:
                continue

            if len(keyword) < 2:
                continue

            if keyword.isdigit():
                continue

            if keyword in stopwords:
                continue

            if keyword.endswith(bad_endings):
                continue

            if "@" in keyword:
                continue

            unique_keywords.add(keyword)

        for keyword in unique_keywords:
            keyword_counter[keyword] += 1

    if not keyword_counter:
        return {
            "success": False,
            "date": target_date,
            "source_article_count": len(hits),
            "message": "집계할 키워드가 없습니다."
        }

    # 같은 날짜 데이터가 이미 있으면 지우고 다시 저장
    es.delete_by_query(
        index="daily_keyword_metrics",
        body={
            "query": {
                "term": {
                    "date": target_date
                }
            }
        },
        conflicts="proceed",
        refresh=True
    )

    saved_count = 0

    for keyword, count in keyword_counter.items():
        doc_id = f"{target_date}_{keyword}"

        es.index(
            index="daily_keyword_metrics",
            id=doc_id,
            body={
                "date": target_date,
                "keyword": keyword,
                "article_count": count
            },
            refresh=False
        )

        saved_count += 1

    es.indices.refresh(index="daily_keyword_metrics")

    top5 = [
        {
            "keyword": keyword,
            "article_count": count
        }
        for keyword, count in keyword_counter.most_common(5)
    ]

    return {
        "success": True,
        "date": target_date,
        "source_article_count": len(hits),
        "saved_keyword_count": saved_count,
        "top5": top5
    }

@app.get("/search-summary")
def search_summary(start_date: str, end_date: str):
    return get_search_summary(start_date, end_date)

@app.get("/search-users")
def search_users(user_id:str, role:str):
    return get_user_search(user_id, role)