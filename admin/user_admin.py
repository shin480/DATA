from datetime import datetime
from util.db import get_engine
from sqlalchemy import text
from zoneinfo import ZoneInfo

def get_user_search(user_id: str = "", role: str = ""):
    conn = None

    try:
        conn = get_engine()

        # 실제 DB role 값 기준
        role_map = {
            "일반": "general",
            "관리자": "admin",
            "최고관리자": "superadmin",
            "general": "general",
            "admin": "admin",
            "superadmin": "superadmin"
        }

        sql = """
            SELECT
                u.user_id,
                u.name,
                u.created_at,
                u.role,

                COALESCE(r.like_count, 0) AS like_count,
                COALESCE(r.dislike_count, 0) AS dislike_count,

                l.last_login

            FROM users u

            LEFT JOIN (
                SELECT
                    user_id,
                    SUM(CASE WHEN reaction_type = 'like' THEN 1 ELSE 0 END) AS like_count,
                    SUM(CASE WHEN reaction_type = 'dislike' THEN 1 ELSE 0 END) AS dislike_count
                FROM article_reactions
                GROUP BY user_id
            ) r
                ON u.user_id = r.user_id

            LEFT JOIN (
                SELECT
                    user_id,
                    MAX(created_at) AS last_login
                FROM login_logs
                GROUP BY user_id
            ) l
                ON u.user_id = l.user_id

            WHERE 1=1
        """

        params = {}

        # 사용자 검색
        if user_id:
            sql += """
                AND (
                    u.user_id LIKE :keyword
                    OR u.name LIKE :keyword
                )
            """
            params["keyword"] = f"%{user_id}%"

        # 권한 필터
        if role and role in role_map:
            sql += " AND u.role = :role "
            params["role"] = role_map[role]

        sql += """
            ORDER BY u.created_at DESC
            LIMIT 300
        """

        result = conn.execute(text(sql), params)

        users = []

        for row in result:
            # 프론트 표시용 한글 변환
            display_role = {
                "general": "일반",
                "admin": "관리자",
                "superadmin": "최고관리자"
            }.get(str(row.role).lower(), row.role)

            users.append({
                "user_id": row.user_id,
                "user_name": row.name if row.name else "-",
                "created_at": str(row.created_at) if row.created_at else "-",
                "role": display_role,
                "like_count": int(row.like_count or 0),
                "dislike_count": int(row.dislike_count or 0),
                "last_login": str(row.last_login) if row.last_login else "-"
            })

        return {
            "success": True,
            "count": len(users),
            "users": users
        }

    except Exception as e:
        print(f"[USER_SEARCH_ERROR] {e}")

        return {
            "success": False,
            "count": 0,
            "users": [],
            "message": str(e)
        }

    finally:
        if conn:
            conn.close()

def get_user_usage_stats():
    target_date = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")

    db = get_engine()

    try:
        # =========================
        # 1. 로그인/로그아웃 로그 조회
        # =========================
        login_query = text("""
            SELECT
                user_id,
                event_type,
                created_at
            FROM login_logs
            WHERE DATE(created_at) = :target_date
              AND user_id IS NOT NULL
              AND result = 'success'
            ORDER BY user_id, created_at ASC
        """)

        rows = db.execute(
            login_query,
            {"target_date": target_date}
        ).fetchall()

        unique_users = set()
        total_stay_seconds = 0
        active_logins = {}

        for row in rows:
            user_id = row.user_id
            event_type = row.event_type
            event_time = row.created_at

            if event_type == "login":
                unique_users.add(user_id)

                # logout 없이 다시 login한 경우 이전 세션은 5분 처리
                if user_id in active_logins:
                    total_stay_seconds += 300

                active_logins[user_id] = event_time

            elif event_type == "logout":
                if user_id in active_logins:
                    login_time = active_logins[user_id]

                    stay_seconds = max(
                        int((event_time - login_time).total_seconds()),
                        0
                    )

                    total_stay_seconds += stay_seconds
                    del active_logins[user_id]

        # logout 없는 로그인 세션은 5분 처리
        for user_id in active_logins:
            total_stay_seconds += 300

        daily_users = len(unique_users)

        avg_stay_seconds = (
            total_stay_seconds / daily_users
            if daily_users > 0 else 0
        )

        # =========================
        # 2. 일일 누적 기사 조회수
        # 로그인 사용자 기준
        # =========================
        view_query = text("""
            SELECT COUNT(*) AS total_views
            FROM article_views
            WHERE DATE(created_at) = :target_date
              AND user_id IS NOT NULL
              AND is_valid_view = 1
        """)

        view_result = db.execute(
            view_query,
            {"target_date": target_date}
        ).fetchone()

        total_article_views = (
            view_result.total_views
            if view_result and view_result.total_views is not None
            else 0
        )

        # =========================
        # 3. 시간 포맷
        # =========================
        def format_seconds(seconds):
            seconds = int(seconds)
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60

            return f"{hours}시간 {minutes}분"

        return {
            "success": True,
            "date": target_date,

            "avg_stay_seconds": round(avg_stay_seconds, 2),
            "avg_stay_text": format_seconds(avg_stay_seconds),

            "daily_users": daily_users,

            "total_stay_seconds": total_stay_seconds,
            "total_stay_text": format_seconds(total_stay_seconds),

            "daily_article_views": total_article_views
        }

    except Exception as e:
        print(f"[USER_USAGE_STATS_ERROR] {e}")

        return {
            "success": False,
            "message": str(e)
        }

    finally:
        db.close()