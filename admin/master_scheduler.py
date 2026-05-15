from apscheduler.schedulers.asyncio import AsyncIOScheduler
import asyncio

from util.es import get_es, NEWS_ECONOMY_INDEX
from util.logger import log_admin_activity

from starlette.requests import Request
from collections import Counter
from datetime import date

from crawling.crawler import run_crawling_job
from datacleaning.cleaning import get_preprocessed_data
from model.model_main import classification_job
from viewpoint_classify.viewpoint_classify import update_perspective_to_es
from datacleaning.dailytopissue import save_daily_top_issue_report
from model.services.embedding_generator import run_embedding_pipeline
from model.model_main import (
    _run_with_timeout,
    run_classification_pipeline,
    run_sentiment_pipeline,
    run_keyword_pipeline,
    run_summary_pipeline,
    run_scope_title_batch,
    run_scope_sentiment_batch,
    run_scope_summary_batch,
    run_scope_keywords_batch,
)

def create_daily_keyword_metrics(target_date: str = None):
    es = get_es()

    # =========================================================
    # 0. 공통 필터 기준
    # =========================================================
    stopwords = {
        "com", "co", "kr", "www", "http", "https",
        "db", "photo", "newsis", "yna", "yonhap",
        "graphics", "graphic",
        "그래픽", "사진", "기자", "제공", "금지",
        "관련", "종합", "단독", "속보", "뉴스",
        "보도", "기사", "무단", "전재", "재배포",
        "습니다", "합니다", "했습니다", "있습니다", "없습니다",
        "됩니다", "됐습니다",

        # 과거 집계에서 튀어나온 잡키워드 직접 차단
        "일제히", "본격적인", "굳어지", "손본다"
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

    def clean_keyword(raw_keyword: str):
        """
        키워드 1개를 정리하고,
        저장 가능한 키워드면 문자열 반환,
        버릴 키워드면 None 반환
        """
        keyword = str(raw_keyword).strip().lower()

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
            return None

        if len(keyword) < 2:
            return None

        if keyword.isdigit():
            return None

        if keyword in stopwords:
            return None

        if keyword.endswith(bad_endings):
            return None

        if "@" in keyword:
            return None

        return keyword

    # =========================================================
    # 1. target_date가 없으면 keywords가 있는 최신 기사 날짜 기준
    # =========================================================
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

    # =========================================================
    # 2. 1차 시도: 원본 news_economy의 keywords로 정상 재집계
    # =========================================================
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
            cleaned_keyword = clean_keyword(keyword)

            if cleaned_keyword is None:
                continue

            unique_keywords.add(cleaned_keyword)

        for keyword in unique_keywords:
            keyword_counter[keyword] += 1

    # =========================================================
    # 3. 정상 재집계가 가능하면 그대로 저장
    # =========================================================
    if keyword_counter:
        rebuild_mode = "source_keywords_rebuild"

        # 같은 날짜 기존 집계 삭제
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
            "mode": rebuild_mode,
            "date": target_date,
            "source_article_count": len(hits),
            "saved_keyword_count": saved_count,
            "top5": top5,
            "message": "news_economy의 keywords 기준으로 재집계 완료"
        }

    # =========================================================
    # 4. 원본 keywords가 없으면:
    #    기존 daily_keyword_metrics 데이터를 청소해서 다시 저장
    # =========================================================
    existing_metrics_result = es.search(
        index="daily_keyword_metrics",
        body={
            "size": 10000,
            "_source": [
                "date",
                "keyword",
                "article_count"
            ],
            "query": {
                "term": {
                    "date": target_date
                }
            }
        }
    )

    existing_metric_hits = existing_metrics_result["hits"]["hits"]

    if not existing_metric_hits:
        return {
            "success": False,
            "date": target_date,
            "source_article_count": len(hits),
            "message": "원본 keywords도 없고, 정리할 기존 daily_keyword_metrics 데이터도 없습니다."
        }

    cleaned_metric_counter = Counter()
    removed_keywords = []

    for hit in existing_metric_hits:
        source = hit["_source"]

        original_keyword = source.get("keyword", "")
        article_count = source.get("article_count", 0)

        cleaned_keyword = clean_keyword(original_keyword)

        if cleaned_keyword is None:
            removed_keywords.append(original_keyword)
            continue

        cleaned_metric_counter[cleaned_keyword] += article_count

    if not cleaned_metric_counter:
        return {
            "success": False,
            "date": target_date,
            "message": "기존 집계 데이터를 정리했지만 남는 키워드가 없습니다.",
            "removed_keywords": removed_keywords
        }

    # 기존 해당 날짜 집계 전체 삭제
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

    # 정리된 집계 다시 저장
    for keyword, count in cleaned_metric_counter.items():
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
        for keyword, count in cleaned_metric_counter.most_common(5)
    ]

    return {
        "success": True,
        "mode": "existing_metrics_cleanup",
        "date": target_date,
        "source_article_count": len(hits),
        "saved_keyword_count": saved_count,
        "removed_keyword_count": len(removed_keywords),
        "removed_keywords": removed_keywords[:50],
        "top5": top5,
        "message": "원본 keywords가 없어 기존 daily_keyword_metrics 집계값을 정리해서 다시 저장했습니다."
    }

def run_model_pipeline_sync():
    print("[model] 1 임베딩 시작")
    _run_with_timeout(run_embedding_pipeline, "embedding")
    print("[model] 1 임베딩 완료")
    print("[model] 2 분류 시작")
    _run_with_timeout(run_classification_pipeline, "classification")
    print("[model] 2 분류 완료")
    print("[model] 3 감성 분류 시작")
    _run_with_timeout(run_sentiment_pipeline, "sentiment")
    print("[model] 3 감성 분류 완료")
    print("[model] 4 키워드 추출 시작")
    _run_with_timeout(run_keyword_pipeline, "keywords")
    print("[model] 4 키워드 추출 완료")
    print("[model] 5 요약 시작")
    _run_with_timeout(run_summary_pipeline, "summary")
    print("[model] 5 요약 완료")
    print("[model] 6 스콥 타이틀 추출 시작")
    _run_with_timeout(run_scope_title_batch, "scope_title_chain")
    print("[model] 6 스콥 타이틀 추출 완료")
    print("[model] 7 스콥 감성 집계 시작")
    _run_with_timeout(run_scope_sentiment_batch, "scope_sentiment")
    print("[model] 7 스콥 감성 집계 완료")
    print("[model] 8 스콥 요약 시작")
    _run_with_timeout(run_scope_summary_batch, "scope_summary")
    print("[model] 8 스콥 요약 완료")
    print("[model] 9 스콥 키워드 추출 시작")
    _run_with_timeout(run_scope_keywords_batch, "scope_keywords")
    print("[model] 9 스콥 키워드 추출 완료")

def get_master_scheduler():
    sch = AsyncIOScheduler(timezone="Asia/Seoul")

    # 1. 크롤링
    sch.add_job(
        run_crawling_job,
        "cron",
        hour=0,
        minute=0,
        id="daily_news_crawling",
        replace_existing=True,
        max_instances=1
    )

    # 2. 전처리
    sch.add_job(
        get_preprocessed_data,
        "cron",
        hour=0,
        minute=30,
        id="daily_preprocessing",
        replace_existing=True,
        max_instances=1
    )

    # 3. AI 파이프라인 시작
    sch.add_job(
        classification_job,
        "cron",
        hour=1,
        minute=0,
        id="daily_ai_pipeline",
        replace_existing=True,
        max_instances=1
    )

    # 4. 관점 분석
    sch.add_job(
        update_perspective_to_es,
        "cron",
        hour=2,
        minute=0,
        id="daily_viewpoint",
        replace_existing=True,
        max_instances=1
    )

    # 5. TOP 이슈
    sch.add_job(
        lambda: save_daily_top_issue_report(None, None),
        "cron",
        hour=2,
        minute=30,
        id="daily_top_issue",
        replace_existing=True,
        max_instances=1
    )

    return sch

async def run_full_pipeline(req: Request):
    user = req.session.get("user")
    log_admin_activity(user.get("user_id"), "BATCH_RUN", f"{user.get('user_id')}({user.get('name')}) 배치 수동 실행")

    results = {}

    try:
        today = date.today().strftime("%Y-%m-%d")
        print(f"[FULL_PIPELINE] 시작 | target_date={today}")

        print("[FULL_PIPELINE] 1/6 크롤링 시작")
        await run_crawling_job(mode="manual")
        print("[FULL_PIPELINE] 1/6 크롤링 완료")
        results["crawling"] = "success"

        print("[FULL_PIPELINE] 2/6 전처리 시작")
        preprocess_result = get_preprocessed_data()
        print(f"[FULL_PIPELINE] 2/6 전처리 완료 | result={preprocess_result}")
        results["preprocessing"] = "success"

        print("[FULL_PIPELINE] 3/6 AI 파이프라인 시작")
        run_model_pipeline_sync()
        print("[FULL_PIPELINE] 3/6 AI 파이프라인 완료")
        results["classification"] = "success"

        print("[FULL_PIPELINE] 4/6 관점 분석 시작")
        viewpoint_result = update_perspective_to_es()
        print(f"[FULL_PIPELINE] 4/6 관점 분석 완료 | result={viewpoint_result}")
        results["viewpoint"] = "success"

        print(f"[FULL_PIPELINE] 5/6 데일리 키워드 집계 시작 | date={today}")
        keyword_result = create_daily_keyword_metrics(today)
        print(f"[FULL_PIPELINE] 5/6 데일리 키워드 집계 완료 | result={keyword_result}")
        results["daily_keyword_metrics"] = "success"

        print(f"[FULL_PIPELINE] 6/6 TOP 이슈 리포트 시작 | date={today}")
        top_issue_result = save_daily_top_issue_report(today, today)
        print(f"[FULL_PIPELINE] 6/6 TOP 이슈 리포트 완료 | result={top_issue_result}")
        results["daily_top_issue"] = "success"

        print("[FULL_PIPELINE] 전체 완료")

        return {
            "success": True,
            "message": "전체 수집 파이프라인 실행 완료",
            "results": results
        }

    except Exception as e:
        print(f"[FULL_PIPELINE_ERROR] {e}")
        return {
            "success": False,
            "message": str(e),
            "results": results
        }

async def run_full_pipeline_for_schedule():

    results = {}

    try:
        today = date.today().strftime("%Y-%m-%d")
        print(f"[FULL_PIPELINE] 시작 | target_date={today}")

        print("[FULL_PIPELINE] 1/6 크롤링 시작")
        await run_crawling_job()
        print("[FULL_PIPELINE] 1/6 크롤링 완료")
        results["crawling"] = "success"

        print("[FULL_PIPELINE] 2/6 전처리 시작")
        preprocess_result = get_preprocessed_data()
        print(f"[FULL_PIPELINE] 2/6 전처리 완료 | result={preprocess_result}")
        results["preprocessing"] = "success"

        print("[FULL_PIPELINE] 3/6 AI 파이프라인 시작")
        run_model_pipeline_sync()
        print("[FULL_PIPELINE] 3/6 AI 파이프라인 완료")
        results["classification"] = "success"

        print("[FULL_PIPELINE] 4/6 관점 분석 시작")
        viewpoint_result = update_perspective_to_es()
        print(f"[FULL_PIPELINE] 4/6 관점 분석 완료 | result={viewpoint_result}")
        results["viewpoint"] = "success"

        print(f"[FULL_PIPELINE] 5/6 데일리 키워드 집계 시작 | date={today}")
        keyword_result = create_daily_keyword_metrics(today)
        print(f"[FULL_PIPELINE] 5/6 데일리 키워드 집계 완료 | result={keyword_result}")
        results["daily_keyword_metrics"] = "success"

        print(f"[FULL_PIPELINE] 6/6 TOP 이슈 리포트 시작 | date={today}")
        top_issue_result = save_daily_top_issue_report(today, today)
        print(f"[FULL_PIPELINE] 6/6 TOP 이슈 리포트 완료 | result={top_issue_result}")
        results["daily_top_issue"] = "success"

        print("[FULL_PIPELINE] 전체 완료")

        return {
            "success": True,
            "message": "전체 수집 파이프라인 실행 완료",
            "results": results
        }

    except Exception as e:
        print(f"[FULL_PIPELINE_ERROR] {e}")
        return {
            "success": False,
            "message": str(e),
            "results": results
        }

def get_scheduler_00():
    sch = AsyncIOScheduler(timezone="Asia/Seoul")

    sch.add_job(
        run_full_pipeline_for_schedule,  # 람다 래핑 없이 함수 자체를 전달
        "cron",
        hour=0,
        minute=0,
        id="daily_full_pipeline",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=600
    )

    return sch