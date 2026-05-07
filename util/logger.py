import logging
from sqlalchemy import text
from util.db import get_engine  # 기존에 사용하시던 DB 연결 함수
from fastapi import Request

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

