import pandas as pd
import re

from util.es import get_es, tokenizer, bulk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

es = get_es()

# 불용어 리스트 정의
STOPWORDS = ["은", "는", "이", "가", "을", "를", "의", "에서", "에", "와", "과", "도", "만", "기자", "뉴스"]

DEBUG_MODE = False
target_index = "news_economy"

def advanced_clean_text(text: str) -> str:
    """[1-03] 요구사항: 태그/이모지 제거 및 불용어 처리"""
    if not text: return ""
    text = re.sub(r'<[^>]*>', ' ', text)  # HTML 태그 제거
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)  # 특수문자/이모지 제거
    text = re.sub(r'\s+', ' ', text).strip()  # 공백 정규화

    # 불용어 제거
    words = text.split()
    return " ".join([w for w in words if w not in STOPWORDS])

def get_preprocessed_data():
    # 1. 전처리 안 된 데이터만 가져오기
    query = {
        "query": {
            "bool": {
                "must_not": [{"term": {"status": "preprocessed"}}]
            }
        },
        "size": 1000
    }
    res = es.search(index="article_raw", body=query)
    hits = res["hits"]["hits"]
    initial_count = len(hits)

    if initial_count == 0:
        return {"message": "새로 처리할 데이터가 없습니다."}

    # 2. DataFrame 생성 시 ES 내부 ID(_id)를 반드시 포함
    raw_df = pd.DataFrame([
        {**hit["_source"], "_id": hit["_id"]} for hit in hits
    ])

    logs = {"missing_fields": [], "url_duplicate": [], "meta_duplicate": [],
            "content_duplicate": [], "high_similarity": []}

    required_fields = ['collected_at', 'title', 'raw_text', 'url', 'press', 'published_at']

    # 결측치 확인 로직
    def check_valid(row):
        for field in required_fields:
            val = row.get(field)
            if pd.isna(val) or str(val).strip() == "" or str(val).lower() == "nan":
                logs["missing_fields"].append(row.get('title', '제목없음'))
                return False
        return True

    df_step1 = raw_df[raw_df.apply(check_valid, axis=1)].copy()

    # 3. 중복 및 유사도 제거
    final_data = []
    seen_urls = set()
    passed_texts = []

    for _, row in df_step1.iterrows():
        title, url, content = str(row['title']), str(row['url']), str(row['raw_text'])

        # url 중복 체크
        if url in seen_urls:
            logs["url_duplicate"].append(title);
            continue

        # 메타데이터 중복 체크
        if any(p['title'] == title and p['press'] == row['press'] for p in final_data):
            logs["meta_duplicate"].append(title);
            continue

        if passed_texts:
            try:
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform([content] + passed_texts)
                cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
                if (cosine_sim >= 0.9).any():
                    logs["high_similarity"].append(title)
                    continue
            except:
                pass

        # [1-03] 통합 텍스트 생성 및 정제
        combined_text = f"{title} {content}"
        row['clean_text'] = advanced_clean_text(combined_text)

        final_data.append(row.to_dict())
        seen_urls.add(url)
        passed_texts.append(content)

    # 4. news_economy 인덱스에 Bulk 저장
    actions = []
    for row in final_data:
        actions.append({
            "_index": target_index,
            "_id": row['_id'],  # 원본 ID 유지
            "_source": {
                "article_id": row.get('article_id'),
                "title": row['title'],
                "content": row['raw_text'],
                "clean_text": row['clean_text'],  # 정제된 필드
                "press": row['press'],
                "author": row.get('author', ''),
                "url": row['url'],
                "published_at": row['published_at'],
                "collected_at": row['collected_at']
            }
        })

    if actions:
        bulk(es, actions)
        print(f"[저장완료] news_economy에 {len(actions)}건 저장.")

    # 원문 데이터에 업뎃
    if not DEBUG_MODE:
        update_actions = [
            {
                "_op_type": "update",
                "_index": "article_raw",
                "_id": hit["_id"],
                "doc": {"status": "preprocessed"}  # 쿼리 조건과 일치하도록 설정
            } for hit in hits
        ]

        if update_actions:
            # 성공 건수와 실패 상세 내역을 받아 에러 모니터링 강화
            success, failed = bulk(es, update_actions, raise_on_error=False)

            # 즉시 반영을 위해 리프레시 실행
            es.indices.refresh(index="article_raw")

            print(f"[업데이트완료] article_raw 총 {len(hits)}건 중 {success}건 'preprocessed' 상태로 변경.")

            if failed:
                print(f"[경고] 업데이트 실패 건수: {len(failed)}건")
                print(f"[에러샘플] {failed[0]}")
    else:
        # 디버그 모드일 때는 출력만 수행
        print(f"[디버그모드] 실제 article_raw 업데이트는 건너뜁니다. (대상: {len(hits)}건)")

    es.indices.refresh(index=target_index)

    token_update_actions = []
    token_model_inputs = []

    print(">>> Nori 토큰화 및 tokens 필드 업데이트 시작...")
    for row in final_data:
        doc_id = row['_id']
        try:
            # 사용자 정의 tokenizer 함수 호출
            tokenized_str = tokenizer(es, row['clean_text'])

            token_model_inputs.append([doc_id, tokenized_str])

            token_update_actions.append({
                "_op_type": "update",
                "_index": target_index,
                "_id": doc_id,
                "doc": {
                    "tokens": tokenized_str,
                    "tokens_status": "success",
                    "processed_status": "preprocessed"  # 여기서 최종 상태 부여
                }
            })
        except Exception as e:
            token_update_actions.append({
                "_op_type": "update",
                "_index": target_index,
                "_id": doc_id,
                "doc": {"tokens_status": "failed"}
            })

    # 6. 최종 토큰 데이터 업데이트
    if token_update_actions:
        bulk(es, token_update_actions)
        es.indices.refresh(index=target_index)
        print(f"[토큰화완료] tokens 필드 업데이트 및 리프레시 완료.")

    return {
        "message": "전처리 및 Nori 토큰화 통합 프로세스 완료",
        "count": len(token_model_inputs),
        "data": token_model_inputs  # [id, token] 리스트 반환
    }
