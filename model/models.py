"""
DB/ES 초기화

MySQL  : pipeline_error_log 테이블만 생성
ES     : 인덱스 매핑 정의 (없으면 생성, 있으면 스킵)
"""

from model.database import get_db, get_es

# ── MySQL: 에러 로그 전용 ──────────────────────────────
CREATE_PIPELINE_ERROR_LOG = """
CREATE TABLE IF NOT EXISTS pipeline_error_log (
    history_id    INT          NOT NULL AUTO_INCREMENT PRIMARY KEY,
    error_code    VARCHAR(50)  NOT NULL,
    error_message TEXT         NOT NULL,
    article_id    VARCHAR(50)  DEFAULT NULL,
    scope_id      VARCHAR(20)  DEFAULT NULL,
    occurred_at   DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_error_code (error_code),
    INDEX idx_occurred   (occurred_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
"""

# ── ES 인덱스 매핑 ─────────────────────────────────────
ES_INDICES = {
    "news_economy": {
        "mappings": {
            "properties": {
                "article_id":      {"type": "keyword"},
                "title":           {"type": "text",    "analyzer": "nori"},
                "content":         {"type": "text",    "analyzer": "nori"},
                "clean_text":      {"type": "text",    "analyzer": "nori"},
                "published_at":    {"type": "date"},
                "press":           {"type": "keyword"},
                "url":             {"type": "keyword", "index": False},
                "keywords":        {"type": "keyword"},
                "sentiment":       {"type": "keyword"},
                "sentiment_score": {"type": "float"},
                "perspective":     {"type": "text"},
                "summary":         {"type": "text"},
                "scopeID":         {"type": "keyword"},
            }
        }
    },
    "news_scopes": {
        "mappings": {
            "properties": {
                "scopeID":             {"type": "keyword"},
                "centroid_embedding":  {"type": "dense_vector", "dims": 4096, "index": False},
                "news_count":          {"type": "integer"},
                "scope_keywords":      {"type": "keyword"},
                "scopeTitle":          {"type": "text"},
                "sentiment":           {"type": "keyword"},
                "sentiment_score":     {"type": "float"},
                "sentiment_dist":      {"type": "object", "enabled": False},
                "scope_summary":       {"type": "text"},
                "created_at":          {"type": "date"},
                "updated_at":          {"type": "date"},
            }
        }
    },
    "scope_refresh_queue": {
        "mappings": {
            "properties": {
                "scopeID":      {"type": "keyword"},
                "queued_at":    {"type": "date"},
                "status":       {"type": "keyword"},
                "processed_at": {"type": "date"},
                "reason":       {"type": "keyword"},
            }
        }
    },
    "daily_keyword_metrics": {
        "mappings": {
            "properties": {
                "date":          {"type": "date"},
                "keyword":       {"type": "keyword"},
                "article_count": {"type": "integer"},
            }
        }
    },
    "daily_top_issue_report": {
        "mappings": {
            "properties": {
                "date":                 {"type": "date"},
                "top_keyword":          {"type": "keyword"},
                "total_mentions":       {"type": "integer"},
                "sentiment_distribution": {"type": "object", "enabled": False},
            }
        }
    },
}


def init_db():
    # MySQL 테이블
    with get_db() as conn:
        with conn.cursor() as cur:
            cur.execute(CREATE_PIPELINE_ERROR_LOG)
    print("✅ MySQL pipeline_error_log 테이블 생성 완료")

    # ES 인덱스
    es = get_es()
    for index_name, body in ES_INDICES.items():
        if not es.indices.exists(index=index_name):
            es.indices.create(index=index_name, body=body)
            print(f"✅ ES 인덱스 생성: {index_name}")
        else:
            print(f"⏭  ES 인덱스 이미 존재: {index_name}")
    es.close()


if __name__ == "__main__":
    init_db()
