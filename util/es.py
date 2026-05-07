# utils/elastic_utils.py
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk as es_bulk

# 접속 정보 설정
ES_HOST = "http://100.99.254.68:9200"

# =========================
# 인덱스 관리
# =========================
NEWS_ECONOMY_INDEX = "news_economy"

# 전역 객체로 관리 (SQLAlchemy의 engine 역할)
es_client = Elasticsearch(ES_HOST)

def get_es():
    """Elasticsearch 클라이언트 객체 반환"""
    return es_client

def close_es():
    """연결 종료 (필요 시)"""
    es_client.transport.close()

# 앞서 만든 tokenizer도 여기에 포함하면 관리가 편합니다.
def tokenizer(es, text):
    if not text: return []
    body = {
        "tokenizer": {
            "type": "nori_tokenizer",
            "decompound_mode": "discard"
        },
        "filter": ["nori_readingform"],
        "text": text
    }
    res = es.indices.analyze(body=body)
    return [token['token'] for token in res['tokens']]

def bulk(es, actions, *args, **kwargs):
    """
    원본 bulk 함수가 받는 모든 인자(*args, **kwargs)를
    그대로 전달할 수 있도록 수정합니다.
    """
    return es_bulk(es, actions, *args, **kwargs)