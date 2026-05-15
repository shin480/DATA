import pandas as pd
import re

from util.es import get_es, bulk
from elasticsearch.helpers import scan
from sqlalchemy import text
from util.db import get_engine

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from kiwipiepy import Kiwi

es = get_es()

try:
    # 기존 컴퓨터에서는 원래 방식으로 정상 실행
    kiwi = Kiwi()

except Exception as e:
    # 노트북처럼 OneDrive/한글 경로 문제로 Kiwi 모델을 못 읽는 경우만 대체 경로 사용
    if "Cannot open extract.mdl" in str(e):
        kiwi = Kiwi(model_path=r"C:\kiwi_model")
    else:
        raise

# 불용어 리스트 정의
STOPWORDS = ["은", "는", "이", "가", "을", "를", "의", "에서", "에", "와", "과", "도", "만", "기자", "뉴스", "영상", "편집", "앵커", "연합뉴스", "뉴시스", "한경", "한경닷컴",
 "게티이미지뱅크", "사진", "제공",
 "기자", "특파원", "앵커", "포토", "사진", "이미지", "캡처", "닷컴",
"헤럴드", "경제", "뉴스", "기자",
"이상섭", "최혁", "정호원", "김익환"]

DEBUG_MODE = False
target_index = "news_economy"

def kiwi_noun_tokenizer(text: str, author: str = "") -> list[str]:
    if not text:
        return []

    banned_names = set()

    # =========================
    # author 기반 기자명 제거
    # =========================
    if author:
        author = str(author).strip()

        if author:
            banned_names.add(author)
            banned_names.add(author.replace(" ", ""))
            banned_names.add(author.replace("기자", "").strip())
            banned_names.add(
                author.replace("기자", "").replace(" ", "").strip()
            )

    # =========================
    # 본문 내 기자명 패턴 제거
    # =========================
    text = re.sub(
        r'[가-힣]{2,4}\s?(기자|앵커|특파원|리포터|캐스터)',
        ' ',
        text
    )

    text = re.sub(
        r'(사진|제공|연합뉴스|뉴시스|한경DB|게티이미지뱅크)[=:\s]*[가-힣A-Za-z0-9· ]{0,20}',
        ' ',
        text
    )

    text = re.sub(
        r'(기자|앵커|특파원|리포터|캐스터)\s?[가-힣]{2,4}',
        ' ',
        text
    )

    text = re.sub(
        r'\[[가-힣]{2,4}의 [^\]]+\]',
        ' ',
        text
    )

    text = re.sub(
        r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
        ' ',
        text
    )

    # =========================
    # Kiwi 토큰화 (author 여부와 무관하게 항상 실행)
    # =========================
    tokens = kiwi.tokenize(
        text,
        split_complex=False
    )

    results = []

    ALLOWED_NOUN_TAGS = {"NNG", "NNP"}

    EXCLUDED_SUFFIXES = (
        "세요", "시오", "합니다",
        "했다", "된다", "됐다", "니다"
    )

    for token in tokens:
        word = token.form.strip()

        if len(word) <= 1 or word in STOPWORDS:
            continue

        # 기자명 제거
        if word in banned_names:
            continue

        if any(name and name in word for name in banned_names):
            continue

        if "기자" in word or "특파원" in word:
            continue

        if token.tag in ALLOWED_NOUN_TAGS:

            if word.endswith(EXCLUDED_SUFFIXES):
                continue

            results.append(word)

    return list(dict.fromkeys(results))

def advanced_clean_text(text: str) -> str:
    if not text:
        return ""

    # 기자명 / 앵커명 제거
    text = re.sub(
        r'([가-힣]{2,4})\s?(기자|앵커|특파원|리포터|캐스터)',
        ' ',
        text
    )

    # 영상편집/그래픽 제거
    text = re.sub(
        r'(영상편집|그래픽|촬영|편집)\s*([가-힣]{2,4})',
        ' ',
        text
    )

    # 지역=이름 기자
    text = re.sub(
        r'[가-힣]+=\s*([가-힣]{2,4})\s?(기자|특파원)',
        ' ',
        text
    )

    # 기자 이름 역순
    text = re.sub(
        r'(기자|앵커|특파원)\s*([가-힣]{2,4})',
        ' ',
        text
    )

    # 사진/출처 제거
    text = re.sub(
        r'(사진|제공|연합뉴스|뉴시스|한경DB|게티이미지뱅크)[=:\s]*[가-힣A-Za-z0-9· ]{0,20}',
        ' ',
        text
    )

    # 코너명 제거
    text = re.sub(
        r'\[[가-힣]{2,4}의 [^\]]+\]',
        ' ',
        text
    )

    # [김익환의 ...] title 앞뒤 포함 제거 강화
    text = re.sub(
        r'\[[^\]]+\]',
        ' ',
        text
    )

    # 포즈 / 사진설명 제거
    text = re.sub(
        r'(포즈|캡처|자료사진)',
        ' ',
        text
    )

    text = re.sub(r'<[^>]*>', ' ', text)
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()

    return " ".join([
        w for w in words
        if w not in STOPWORDS
    ])

# news_economy에 저장된 데이터 중복 검사
def exists_in_news_economy(article_id, url, title, press):
    print(f"[ES중복검사] article_id={article_id} | title={title[:40]}")
    should = []

    if article_id:
        should.append({"term": {"article_id": article_id}})

    if url:
        should.append({"term": {"url": url}})

    # 제목 + 언론사
    if title and press:
        should.append({
            "bool": {
                "must": [
                    {"match_phrase": {"title": title}},
                    {"term": {"press": press}}
                ]
            }
        })

    if not should:
        return False

    res = es.search(
        index=target_index,
        body={
            "query": {
                "bool": {
                    "should": should,
                    "minimum_should_match": 1
                }
            },
            "size": 1
        }
    )

    return res["hits"]["total"]["value"] > 0

def get_preprocessed_data():
    print("전처리 프로세스 시작")
    # 1. 전처리 안 된 데이터만 가져오기
    query = {
        "query": {
            "bool": {
                "must": [{"term": {"status": "collected"}}]
            }
        },
        "size": 7000
    }
    res = es.search(index="article_raw", body=query)
    hits = res["hits"]["hits"]
    initial_count = len(hits)

    if initial_count == 0:
        print("새로 처리할 데이터가 없습니다.")
        return {"message": "새로 처리할 데이터가 없습니다."}

    # 2. DataFrame 생성 시 ES 내부 ID(_id)를 반드시 포함
    raw_df = pd.DataFrame([
        {**hit["_source"], "_id": hit["_id"]} for hit in hits
    ])

    logs = {"missing_fields": [], "url_duplicate": [], "meta_duplicate": [],
            "content_duplicate": [], "high_similarity": [], "es_duplicate": []}
    skipped_updates = []

    required_fields = ['collected_at', 'title', 'raw_text', 'url', 'press', 'published_at']

    # 결측치 확인 로직
    def check_valid(row):
        for field in required_fields:
            val = row.get(field)
            if pd.isna(val) or str(val).strip() == "" or str(val).lower() == "nan":
                logs["missing_fields"].append(row.get('title', '제목없음'))
                skipped_updates.append({
                    "_op_type": "update",
                    "_index": "article_raw",
                    "_id": row["_id"],
                    "doc": {"status": "skipped_missing"}
                })
                return False
        return True

    df_step1 = raw_df[raw_df.apply(check_valid, axis=1)].copy()

    # 3. 중복 및 유사도 제거
    final_data = []
    seen_urls = set()
    seen_title_press = set()
    passed_texts = []

    # =========================
    # 기존 news_economy 전체 중복 데이터 선로드
    # =========================
    print(">>> 기존 news_economy 중복검사용 데이터 로딩 시작...")

    existing_article_ids = set()
    existing_urls = set()
    existing_title_press = set()

    for doc in scan(
            es,
            index=target_index,
            query={
                "query": {"match_all": {}},
                "_source": ["article_id", "url", "title", "press"]
            },
            size=5000,
            scroll="10m"
    ):
        src = doc["_source"]

        if src.get("article_id"):
            existing_article_ids.add(src["article_id"])

        if src.get("url"):
            existing_urls.add(src["url"])

        if src.get("title") and src.get("press"):
            existing_title_press.add((src["title"], src["press"]))

    print(
        f">>> 중복검사 데이터 로딩 완료 | "
        f"article_id: {len(existing_article_ids)}건 | "
        f"url: {len(existing_urls)}건 | "
        f"title+press: {len(existing_title_press)}건"
    )

    # =========================
    # 신규 데이터 검사 시작
    # =========================
    for idx, (_, row) in enumerate(df_step1.iterrows(), start=1):
        title = str(row['title'])
        url = str(row['url'])
        content = str(row['raw_text'])
        press = str(row['press'])
        article_id = row.get("article_id", "")

        # 진행률 출력
        print(f"[진행률] {idx}/{len(df_step1)} | article_id={article_id} | title={title[:50]}")

        # 1. 현재 배치 내 URL 중복
        if url in seen_urls:
            logs["url_duplicate"].append(title)
            skipped_updates.append({
                "_op_type": "update",
                "_index": "article_raw",
                "_id": row["_id"],
                "doc": {"status": "skipped_duplicate_url"}
            })
            continue

        # 2. 현재 배치 내 title + press 중복
        if (title, press) in seen_title_press:
            logs["meta_duplicate"].append(title)
            skipped_updates.append({
                "_op_type": "update",
                "_index": "article_raw",
                "_id": row["_id"],
                "doc": {"status": "skipped_duplicate_meta"}
            })

            continue

        # 3. 기존 news_economy 중복 체크 (메모리 set 방식)
        if (
                (article_id and article_id in existing_article_ids)
                or (url and url in existing_urls)
                or ((title, press) in existing_title_press)
        ):
            logs["es_duplicate"].append(title)
            skipped_updates.append({
                "_op_type": "update",
                "_index": "article_raw",
                "_id": row["_id"],
                "doc": {"status": "skipped_es_duplicate"}
            })
            continue

        # 4. 본문 유사도 검사
        if passed_texts:
            try:
                vectorizer = TfidfVectorizer()
                tfidf_matrix = vectorizer.fit_transform([content] + passed_texts)
                cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

                if (cosine_sim >= 0.9).any():
                    logs["high_similarity"].append(title)
                    skipped_updates.append({
                        "_op_type": "update",
                        "_index": "article_raw",
                        "_id": row["_id"],
                        "doc": {"status": "skipped_similarity"}
                    })
                    continue

            except Exception as e:
                print(f"[유사도검사오류] {title[:30]} | {e}")

        # [1-03] 통합 텍스트 생성 및 정제
        combined_text = f"{title} {content}"
        row['clean_text'] = advanced_clean_text(combined_text)

        final_data.append(row.to_dict())

        # 현재 배치 중복 방지
        seen_urls.add(url)
        seen_title_press.add((title, press))
        passed_texts.append(content)

        # 기존 ES 기준 set에도 즉시 추가
        if article_id:
            existing_article_ids.add(article_id)

        if url:
            existing_urls.add(url)

        existing_title_press.add((title, press))

    if skipped_updates and not DEBUG_MODE:
        batch_size = 500

        for i in range(0, len(skipped_updates), batch_size):
            batch = skipped_updates[i:i + batch_size]

            bulk(es, batch, raise_on_error=False)

            print(f"[스킵상태업데이트] article_raw {i + len(batch)}/{len(skipped_updates)}건 skipped 처리 완료")
        es.indices.refresh(index="article_raw")

    if len(final_data) == 0:
        es.indices.refresh(index="article_raw")

        return {
            "message": "저장할 신규 전처리 데이터가 없습니다. 중복/결측치 데이터는 skipped 처리했습니다.",
            "count": 0,
            "skip_count": len(skipped_updates),
            "logs": logs
        }

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
        batch_size = 500

        # ES에 500건씩 중간 저장
        for i in range(0, len(actions), batch_size):
            batch = actions[i:i + batch_size]

            bulk(es, batch)

            print(f"[ES중간저장] news_economy {i + len(batch)}/{len(actions)}건 저장 완료")

        print(f"[ES저장완료] news_economy에 총 {len(actions)}건 저장.")

        # DB에도 500건씩 중간 저장
        db_insert_data = [
            {"article_id": row.get('article_id')}
            for row in final_data if row.get('article_id')
        ]

        if db_insert_data:
            conn = None
            try:
                conn = get_engine()

                sql = text("""
                    INSERT IGNORE INTO article_meta (article_id, created_at)
                    VALUES (:article_id, NOW())
                """)

                for i in range(0, len(db_insert_data), batch_size):
                    batch = db_insert_data[i:i + batch_size]

                    conn.execute(sql, batch)
                    conn.commit()

                    print(f"[DB중간저장] article_meta {i + len(batch)}/{len(db_insert_data)}건 반영 완료")

                print(f"[DB저장완료] article_meta 테이블에 총 {len(db_insert_data)}건 반영.")

            except Exception as e:
                print(f"[DB에러] article_meta 저장 중 오류 발생: {e}")

            finally:
                if conn:
                    conn.close()
    # 원문 데이터에 업뎃
    if not DEBUG_MODE:
        update_actions = [
            {
                "_op_type": "update",
                "_index": "article_raw",
                "_id": row["_id"],
                "doc": {"status": "preprocessed"}  # 쿼리 조건과 일치하도록 설정
            } for _, row in df_step1.iterrows()
        ]

        if update_actions:
            batch_size = 500
            total_success = 0
            total_failed = []

            for i in range(0, len(update_actions), batch_size):
                batch = update_actions[i:i + batch_size]

                success, failed = bulk(es, batch, raise_on_error=False)
                total_success += success

                if failed:
                    total_failed.extend(failed)

                print(f"[상태중간업데이트] article_raw {i + len(batch)}/{len(update_actions)}건 preprocessed 변경 완료")

            print(f"[업데이트완료] article_raw 총 {len(final_data)}건 중 {total_success}건 'preprocessed' 상태로 변경.")

            if total_failed:
                print(f"[경고] 업데이트 실패 건수: {len(total_failed)}건")
                print(f"[에러샘플] {total_failed[0]}")
    else:
        # 디버그 모드일 때는 출력만 수행
        print(f"[디버그모드] 실제 article_raw 업데이트는 건너뜁니다. (대상: {len(final_data)}건)")

    token_update_actions = []
    token_model_inputs = []

    print(">>> 토큰화 및 tokens 필드 업데이트 시작...")
    for row in final_data:
        doc_id = row['_id']
        try:
            # 사용자 정의 tokenizer 함수 호출
            tokenized_str = kiwi_noun_tokenizer(row.get("clean_text", ""),row.get("author", ""))

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
        batch_size = 500

        for i in range(0, len(token_update_actions), batch_size):
            batch = token_update_actions[i:i + batch_size]

            bulk(es, batch)

            print(f"[토큰중간저장] tokens {i + len(batch)}/{len(token_update_actions)}건 업데이트 완료")

        print(f"[토큰화완료] tokens 필드 업데이트")

    es.indices.refresh(index=target_index)
    es.indices.refresh(index="article_raw")

    return {
        "message": "전처리 및 토큰화 통합 프로세스 완료",
        "count": len(token_model_inputs),
        "data": token_model_inputs  # [id, token] 리스트 반환
    }


#db에 article_id 넣기
def sync_es_to_db():
    print(">>> ES 데이터 DB 동기화 작업을 시작합니다.")

    # 1. Elasticsearch 연결 및 데이터 추출
    es = get_es()
    target_index = "news_economy"

    # scan을 사용하면 1만 건 이상의 대량 데이터도 끊김 없이 가져올 수 있습니다.
    es_data_generator = scan(
        es,
        index=target_index,
        query={"query": {"match_all": {}}, "_source": ["article_id"]}
    )

    # ES에서 article_id만 리스트로 추출
    es_ids = [doc['_source'].get('article_id') for doc in es_data_generator if doc['_source'].get('article_id')]

    if not es_ids:
        print("ES에 동기화할 데이터가 존재하지 않습니다.")
        return

    print(f"ES에서 총 {len(es_ids)}건의 article_id를 발견했습니다.")

    # 2. DB 저장 (사용자님의 Session 반환 규칙 적용)
    # 대량 데이터를 한 번에 넣으면 DB 부하가 생길 수 있으므로 1,000건씩 나누어 처리합니다.
    batch_size = 1000
    session = None

    try:
        session = get_engine()  # 사용자님 규칙: Session 객체 획득

        sql = text("""
            INSERT IGNORE INTO article_meta (article_id, created_at)
            VALUES (:article_id, NOW())
        """)

        for i in range(0, len(es_ids), batch_size):
            batch = [{"article_id": aid} for aid in es_ids[i:i + batch_size]]
            session.execute(sql, batch)
            session.commit()  # 배치 단위로 커밋
            print(f"[{i + len(batch)}/{len(es_ids)}] 동기화 진행 중...")

        print(">>> [완료] 모든 데이터가 성공적으로 DB에 동기화되었습니다.")

    except Exception as e:
        if session:
            session.rollback()  # 에러 발생 시 롤백
        print(f"🚨 [오류발생] 동기화 중 에러가 발생했습니다: {e}")

    finally:
        if session:
            session.close()  # 세션 종료 (연결 반환)

def retokenize_news_economy(): # news_economy 전체 재토큰화
    print(">>> news_economy 전체 Kiwi 재토큰화 시작")

    docs = scan(
        es,
        index=target_index,
        query={
            "query": {"match_all": {}},
            "_source": ["clean_text", "author"]
        },
        size=5000,
        scroll="10m"
    )

    actions = []
    total = 0
    failed = 0
    batch_size = 1000

    for doc in docs:
        total += 1
        doc_id = doc["_id"]

        try:
            source = doc.get("_source", {})
            clean_text = source.get("clean_text", "")
            author = source.get("author", "")
            tokens = kiwi_noun_tokenizer(clean_text, author)

            actions.append({
                "_op_type": "update",
                "_index": target_index,
                "_id": doc_id,
                "doc": {
                    "tokens": tokens,
                    "tokens_status": "success",
                    "tokenizer_type": "kiwi"
                }
            })

        except Exception as e:
            failed += 1
            actions.append({
                "_op_type": "update",
                "_index": target_index,
                "_id": doc_id,
                "doc": {
                    "tokens_status": "failed",
                    "tokenizer_type": "kiwi",
                    "tokenizer_error": str(e)
                }
            })

        if len(actions) >= batch_size:
            bulk(es, actions, raise_on_error=False)
            print(f"[재토큰화진행] {total}건 처리 완료 / 실패 {failed}건")
            actions.clear()

    if actions:
        bulk(es, actions, raise_on_error=False)

    es.indices.refresh(index=target_index)

    print(f">>> 완료: 총 {total}건 재토큰화 / 실패 {failed}건")

    return {
        "success": True,
        "total": total,
        "failed": failed
    }


if __name__ == "__main__":
    retokenize_news_economy()