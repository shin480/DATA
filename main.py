from typing import Dict, List, Any

from fastapi import FastAPI
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

        "aiCount": total_count,

        "newsCount": total_count,

        "flow": flow,

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

# 기사 목록
@app.get("/api/articles")
def get_article_list(size: int = 50):
    es = get_es()

    body = {
        "size": size,
        "query": {
            "match_all": {}
        },
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

    articles = []

    for hit in result["hits"]["hits"]:
        source = hit["_source"]

        articles.append({
            "article_id": source.get("article_id", ""),
            "source": source.get("press", "언론사 없음"),
            "title": source.get("title", "제목 없음"),
            "summary": source.get("summary") or source.get("content", "")[:120],
            "published_at": source.get("published_at", ""),
            "url": source.get("url", "#")
        })

    return {
        "count": len(articles),
        "articles": articles
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
def view(req: Request):
    view_log(req)
# 기사 상세페이지
@app.get("/api/articles/{article_id}")
def get_article_detail(article_id: str, req: Request):
    es = get_es()

    result = es.get(
        index=NEWS_ECONOMY_INDEX,
        id=article_id
    )

    source = result["_source"]

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
        "currentVote": current_vote
    }

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


@app.get("/crawl")
async def crawl():
    return await run_crawling_job()
