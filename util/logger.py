import logging
from sqlalchemy import text
from util.db import get_engine  # 기존에 사용하시던 DB 연결 함수
from util.es import get_es, NEWS_ECONOMY_INDEX

from fastapi import Request
from datetime import datetime

class Logger:
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - [%(name)s] - %(message)s - %(asctime)s',
        datefmt = '%Y-%m-%d %H:%M:%S'
    )
    def get_logger(self, name):
        return logging.getLogger(name)

# 이 파일 안에서 사용할 전용 로거 객체 생성
logger = logging.getLogger("log_update")

def log_user_activity(user_id: str, event_type: str, req: Request = None):
    # req 객체에서 자동으로 IP 추출
    user_ip = "Unknown"
    if req:
        forwarded_for = req.headers.get("X-Forwarded-For")
        if forwarded_for:
            user_ip = forwarded_for.split(",")[0].strip()
        elif req.client:
            user_ip = req.client.host

    logger.info(f"User: {user_id} | Event: {event_type} | IP: {user_ip}")

    # DB 저장
    conn = None
    try:
        conn = get_engine()  # 연결 생성
        query = text("""
                INSERT INTO user_activity_logs (user_id, event_type, user_ip)
                VALUES (:user_id, :event_type, :user_ip)
            """)
        conn.execute(query, {
            "user_id": user_id,
            "event_type": event_type,
            "user_ip": user_ip
        })
        conn.commit()  # 변경사항 반영
    except Exception as e:
        # DB 저장 실패 시 에러 내용만 출력하고 함수를 종료 (메인 로직 보호)
        logger.error(f"🚨 DB 로깅 실패: {e}")
    finally:
        if conn:
            conn.close()  # 연결 반환

def log_login(user_id: str, event_type: str, result: str):
    logger.info(f"User: {user_id} | Event: {event_type} | result: {result}")

    # DB 저장
    conn = None
    try:
        conn = get_engine()  # 연결 생성
        query = text("""
                            INSERT INTO login_logs (user_id, event_type, result, fail_count_at_time)
                            VALUES (:user_id, :event_type, :result, :fail_count_at_time)
                        """)
        # result에 따라 사용자 테이블의 실패 횟수 업데이트
        fail_count_at_time = 0
        if result == "success":
            fail_count_at_time = 0
        else:  # result == "fail"
            select_query = text("""
                        SELECT fail_count_at_time 
                        FROM login_logs 
                        WHERE user_id = :user_id AND DATE(created_at) = CURDATE()
                        ORDER BY created_at DESC 
                        LIMIT 1
                    """)
            query_result = conn.execute(select_query, {"user_id": user_id}).scalar()
            if query_result is None:
                fail_count_at_time = 1
            else:
                fail_count_at_time = query_result + 1


        conn.execute(query, {
            "user_id": user_id,
            "event_type": event_type,
            "result": result,
            "fail_count_at_time": fail_count_at_time
        })
        conn.commit()  # 변경사항 반영
    except Exception as e:
        # DB 저장 실패 시 에러 내용만 출력하고 함수를 종료 (메인 로직 보호)
        logger.error(f"🚨 DB 로깅 실패: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()  # 연결 반환


def log_admin_activity(admin_id: str,action_code: str,action_detail: str = ""):
    """
    admin_logs 기록
    저장 컬럼:
    - admin_id
    - action_code
    - action_detail
    """

    logger.info(
        f"[ADMIN LOG] Admin: {admin_id} | Action: {action_code} | Detail: {action_detail}"
    )

    conn = None

    try:
        conn = get_engine()

        query = text("""
            INSERT INTO admin_logs (
                admin_id,
                action_code,
                action_detail
            )
            VALUES (
                :admin_id,
                :action_code,
                :action_detail
            )
        """)

        conn.execute(
            query,
            {
                "admin_id": admin_id,
                "action_code": action_code,
                "action_detail": action_detail
            }
        )

        conn.commit()

    except Exception as e:
        logger.error(f"🚨 ADMIN LOG 저장 실패: {e}")

    finally:
        if conn:
            conn.close()

def get_fail_count_from_es(start_at, end_at):
    es = get_es()
    query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "range": {
                            "collected_at": {
                                "gte": start_at.isoformat(),
                                "lte": end_at.isoformat()
                            }
                        }
                    }
                ],
                "should": [
                    {"bool": {"must_not": {"exists": {"field": "sentiment"}}}},
                    {"bool": {"must_not": {"exists": {"field": "sentiment_score"}}}},
                    {"bool": {"must_not": {"exists": {"field": "keywords"}}}},
                    {"bool": {"must_not": {"exists": {"field": "scopeID"}}}},
                    {"bool": {"must_not": {"exists": {"field": "summary"}}}},

                    # perspective nested 값이 없거나 불완전한 경우
                    {
                        "bool": {
                            "must_not": {
                                "nested": {
                                    "path": "perspective",
                                    "query": {
                                        "bool": {
                                            "must": [
                                                {"exists": {"field": "perspective.category"}},
                                                {"exists": {"field": "perspective.rank"}},
                                                {"exists": {"field": "perspective.score"}}
                                            ]
                                        }
                                    }
                                }
                            }
                        }
                    }
                ],
                "minimum_should_match": 1
            }
        }
    }

    result = es.count(
        index="news_economy",
        body=query
    )

    return result.get("count", 0)

def create_batch_job(code_id: str):
    """
    배치 시작 기록 생성
    - start_at만 먼저 저장
    - job_id 반환
    """

    conn = None

    try:
        conn = get_engine()

        query = text("""
            INSERT INTO batch_jobs (
                code_id,
                start_at,
                total_count,
                fail_count
            )
            VALUES (
                :code_id,
                :start_at,
                0,
                0
            )
        """)

        result = conn.execute(
            query,
            {
                "code_id": code_id,
                "start_at": datetime.now()
            }
        )

        conn.commit()

        return result.lastrowid

    except Exception as e:
        logger.error(f"🚨 BATCH JOB 생성 실패: {e}")
        return None

    finally:
        if conn:
            conn.close()

def update_batch_job(job_id: int):
    db = get_engine()

    try:
        # 1. 배치 시작 시간 조회
        batch_sql = text("""
            SELECT start_at
            FROM batch_jobs
            WHERE job_id = :job_id
        """)

        batch = db.execute(batch_sql, {"job_id": job_id}).fetchone()

        if not batch:
            logger.error(f"🚨 batch_jobs에 job_id={job_id} 없음")
            return

        start_at = batch.start_at
        end_at = datetime.now()

        # 2. total_count 계산
        total_sql = text("""
            SELECT COUNT(DISTINCT article_id)
            FROM article_meta
            WHERE created_at BETWEEN :start_at AND :end_at
        """)

        total_count = db.execute(
            total_sql,
            {
                "start_at": start_at,
                "end_at": end_at
            }
        ).scalar() or 0

        # 3. ES에서 fail_count 계산
        fail_count = get_fail_count_from_es(start_at, end_at)

        # 4. batch_jobs 업데이트
        update_sql = text("""
            UPDATE batch_jobs
            SET
                end_at = :end_at,
                total_count = :total_count,
                fail_count = :fail_count
            WHERE job_id = :job_id
        """)

        db.execute(
            update_sql,
            {
                "job_id": job_id,
                "end_at": end_at,
                "total_count": total_count,
                "fail_count": fail_count
            }
        )

        db.commit()

    finally:
        db.close()

# 기사 처리 로그
def save_article_process_log(code_id: str, article_id: str = None, status: str = "success"):
    """
    article_process_logs 기록
    - 가장 최근 batch_jobs.job_id 자동 조회
    """

    conn = None

    try:
        conn = get_engine()
        # 1. 가장 최근 job_id 조회
        job_sql = text("""
            SELECT job_id
            FROM batch_jobs
            ORDER BY job_id DESC
            LIMIT 1
        """)

        latest_job = conn.execute(job_sql).fetchone()

        if not latest_job:
            logger.error("🚨 batch_jobs에 job_id 없음")
            return

        job_id = latest_job.job_id

        # 2. 로그 저장
        log_sql = text("""
            INSERT INTO article_process_logs (
                job_id,
                code_id,
                article_id,
                status,
                occurred_at
            )
            VALUES (
                :job_id,
                :code_id,
                :article_id,
                :status,
                NOW()
            )
        """)

        conn.execute(
            log_sql,
            {
                "job_id": job_id,
                "code_id": code_id,
                "article_id": article_id,
                "status": status
            }
        )

        conn.commit()

    except Exception as e:
        logger.error(
            f"🚨 ARTICLE PROCESS LOG 저장 실패 | "
            f"code_id={code_id}, article_id={article_id}, status={status} | error={e}"
        )
        conn.rollback()

    finally:
        conn.close()