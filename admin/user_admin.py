from util.db import get_engine
from sqlalchemy import text

from sqlalchemy import text

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