"""
scopeID 클러스터링 서비스

흐름:
  1. ES news_economy에서 scopeID=NULL 레코드 polling
  2. clean_text → TF-IDF 벡터화
  3. FAISS로 기존 scope centroid(news_scopes)와 코사인 유사도 비교
  4. 유사도 >= THRESHOLD → 기존 scopeID 배정
     유사도 <  THRESHOLD → 신규 scopeID 발급
  5. news_economy.scopeID upsert + news_scopes centroid/news_count upsert
"""

import json
import logging
import re
from datetime import datetime, timezone

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from model.database import get_es
from model.services.error_logger import log_pipeline_error
from model.services.scope_title import trigger_scope_title

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.65
BATCH_SIZE           = 100
SCOPE_ID_PREFIX_FMT  = "%Y%m%d"
TFIDF_MAX_FEATURES   = 4096

INDEX_NEWS   = "news_economy"
INDEX_SCOPES = "news_scopes"


def preprocess(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _generate_scope_id(es) -> str:
    """YYYYMMDD-XXXX 형식 scopeID 생성"""
    prefix = datetime.now().strftime(SCOPE_ID_PREFIX_FMT)
    res = es.count(
        index=INDEX_SCOPES,
        body={"query": {"prefix": {"scopeID": prefix}}},
    )
    count = res["count"]
    return f"{prefix}-{count + 1:04d}"


def _build_faiss_index(es) -> tuple:
    """ES news_scopes에서 centroid를 읽어 FAISS 인덱스 구축"""
    res = es.search(
        index=INDEX_SCOPES,
        body={
            "query": {"exists": {"field": "centroid_embedding"}},
            "_source": ["scopeID", "centroid_embedding"],
            "size": 10000,
        },
    )
    hits = res["hits"]["hits"]
    if not hits:
        return None, [], 0

    scope_ids, centroids = [], []
    for h in hits:
        src = h["_source"]
        centroid = src.get("centroid_embedding")
        if centroid:
            scope_ids.append(src["scopeID"])
            centroids.append(np.array(centroid, dtype=np.float32))

    if not centroids:
        return None, [], 0

    dim    = len(centroids[0])
    matrix = np.vstack(centroids).astype(np.float32)
    index  = faiss.IndexFlatIP(dim)
    index.add(matrix)
    return index, scope_ids, dim


def _update_centroid(old_centroid, new_vec, old_count):
    old     = np.array(old_centroid, dtype=np.float32)
    updated = (old * old_count + new_vec.flatten()) / (old_count + 1)
    norm    = np.linalg.norm(updated)
    if norm > 0:
        updated /= norm
    return updated.tolist()


def run_classification_pipeline():
    """scopeID=NULL 뉴스를 배치로 가져와 scopeID를 배정합니다."""
    try:
        es = get_es()

        # 1. 미분류 뉴스 조회
        res = es.search(
            index=INDEX_NEWS,
            body={
                "query": {
                    "bool": {
                        "must_not": {"exists": {"field": "scopeID"}},
                        "must":     {"exists": {"field": "clean_text"}},
                    }
                },
                "_source": ["article_id", "title", "clean_text"],
                "sort":    [{"collected_at": "asc"}],
                "size":    BATCH_SIZE,
            },
        )
        hits = res["hits"]["hits"]

        if not hits:
            logger.info("분류할 뉴스 없음")
            es.close()
            return

        logger.info(f"분류 시작: {len(hits)}건")

        # 2. TF-IDF 벡터화 (clean_text 사용)
        texts = [
            preprocess(h["_source"].get("clean_text") or h["_source"].get("title", ""))
            for h in hits
        ]
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            analyzer="char_wb",
            ngram_range=(2, 3),
        )
        matrix  = vectorizer.fit_transform(texts).toarray().astype(np.float32)
        norms   = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = matrix / norms

        # 3. FAISS 인덱스 구축
        faiss_index, scope_ids, centroid_dim = _build_faiss_index(es)

        now_utc = datetime.now(timezone.utc).isoformat()

        for i, hit in enumerate(hits):
            article_id = hit["_source"]["article_id"]
            vec        = vectors[i].reshape(1, -1)
            assigned_scope_id = None

            # 4. 기존 scope와 유사도 비교
            if faiss_index is not None and vec.shape[1] == centroid_dim:
                scores, indices = faiss_index.search(vec, k=1)
                best_score = float(scores[0][0])
                best_idx   = int(indices[0][0])
                if best_score >= SIMILARITY_THRESHOLD:
                    assigned_scope_id = scope_ids[best_idx]

            # 5. 신규 scope 발급
            if assigned_scope_id is None:
                assigned_scope_id = _generate_scope_id(es)
                es.index(
                    index=INDEX_SCOPES,
                    id=assigned_scope_id,
                    body={
                        "scopeID":            assigned_scope_id,
                        "centroid_embedding": vec.flatten().tolist(),
                        "news_count":         0,
                        "created_at":         now_utc,
                        "updated_at":         now_utc,
                    },
                )

            # 6. news_economy.scopeID upsert
            es.update(
                index=INDEX_NEWS,
                id=article_id,
                body={"doc": {"scopeID": assigned_scope_id}},
            )

            # 7. scope centroid + news_count 갱신
            scope_res  = es.get(index=INDEX_SCOPES, id=assigned_scope_id)
            scope_src  = scope_res["_source"]
            old_count  = scope_src.get("news_count", 0)
            old_centroid = scope_src.get("centroid_embedding", vec.flatten().tolist())
            new_centroid = _update_centroid(old_centroid, vec, old_count)

            es.update(
                index=INDEX_SCOPES,
                id=assigned_scope_id,
                body={"doc": {
                    "centroid_embedding": new_centroid,
                    "news_count":         old_count + 1,
                    "updated_at":         now_utc,
                }},
            )

            # 8. scopeTitle 큐 등록 or 즉시 생성
            trigger_scope_title(es, assigned_scope_id, old_count + 1)

        es.close()
        logger.info(f"분류 완료: {len(hits)}건 처리")

    except Exception as e:
        log_pipeline_error(pipeline="classifier", error=e)
        raise
