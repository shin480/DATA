"""
ES 인덱스 생성 스크립트 (MySQL 없이 독립 실행 가능)
실행: python create_es_indices.py
"""
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

from model.database import get_es

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
                "scopeID":            {"type": "keyword"},
                "centroid_embedding": {"type": "dense_vector", "dims": 4096, "index": False},
                "news_count":         {"type": "integer"},
                "scope_keywords":     {"type": "keyword"},
                "scopeTitle":         {"type": "text"},
                "sentiment":          {"type": "keyword"},
                "sentiment_score":    {"type": "float"},
                "sentiment_dist":     {"type": "object", "enabled": False},
                "scope_summary":      {"type": "text"},
                "created_at":         {"type": "date"},
                "updated_at":         {"type": "date"},
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
                "date":                   {"type": "date"},
                "top_keyword":            {"type": "keyword"},
                "total_mentions":         {"type": "integer"},
                "sentiment_distribution": {"type": "object", "enabled": False},
            }
        }
    },
}

if __name__ == "__main__":
    es = get_es()
    for index_name, body in ES_INDICES.items():
        if not es.indices.exists(index=index_name):
            es.indices.create(index=index_name, body=body)
            print(f"✅ ES 인덱스 생성: {index_name}")
        else:
            print(f"⏭  ES 인덱스 이미 존재: {index_name}")
    es.close()
    print("완료")
