import re
from util.es import get_es, NEWS_ECONOMY_INDEX

# 1. 유의어 사전 (기존과 동일하게 유지하되 '강화' 관련 억지 매핑 제거)
synonym_map = {
    "불황": "침체", "경기악화": "침체", "마이너스 성장": "침체",
    "상승세": "상승", "랠리": "상승", "급등": "상승", "호조": "상승",
    "해외": "글로벌", "외국": "글로벌", "미국": "글로벌", "중동": "글로벌",
    "예측": "전망", "관측": "전망", "내다봤": "전망", "추산": "전망"
}


def clean_and_normalize(text):
    if not text: return ""
    text = re.sub(r'\[.*?\]|\(.*?\)|=연합뉴스|기자|뉴스|#.*', '', text)
    for syn, rep in synonym_map.items():
        text = text.replace(syn, rep)
    return text


# 2. [스마트 딕셔너리] 중의적 단어 제거 + 핵심 단어/짧은 N-gram 혼합
category_keywords = {
    # [책임 소재 그룹]
    "정부 책임": ["당국 책임", "정책 실패", "정부 무능", "실책", "늑장 대응", "관리 부실"],
    "기업 책임": ["경영진 책임", "도덕적 해이", "불법", "방만 경영", "기업 과실"],
    "개인 책임": ["개인 과실", "투자자 책임", "본인 부주의", "개인 탓"],
    "외부 책임": ["외생 변수", "불가항력", "외부 충격", "어쩔 수 없는"],
    "복합 책임": ["공동 책임", "복합적", "상호 작용", "맞물려"],

    # [태도 및 감성 그룹]
    "비판적 태도": ["비판", "논란", "지적", "규탄", "부당", "소송", "반발", "의혹", "갈등"],
    "우려": ["우려", "위기", "타격", "침체", "불확실성", "경고", "악화", "부담", "리스크"],
    "기대": ["기대", "호재", "상승", "회복", "활기", "수혜", "낙관", "모멘텀"],
    "성과 예찬": ["성과", "달성", "최고등급", "최우수", "입증", "우수성", "기록", "쾌거", "성공적"],

    # [정보 전달 및 분석 관점 그룹] (팩트는 이쪽으로 모입니다)
    "단순 전달": [
        # 1. 기존 기본 전달 용어
        "개최", "열렸다", "전했다", "밝혔다", "참석", "알려졌다", "체결", "방문",

        # 2. [추가] 행사/전시 관련 (형태소 한계 극복용 어간 및 유의어)
        "열린", "열린다", "개막", "선보이", "진행", "참여",

        # 3. [추가] 기업 비즈니스 팩트 (수주, 계약, 실적 등)
        "수주", "계약", "M&A", "인수", "합병", "출시", "발표", "공급", "실적", "MOU"
    ],
    "원인 분석": ["원인", "배경", "이유", "기인", "때문에"],
    "결과 분석": ["결과", "집계", "나타났다", "분석됨", "수치", "통계", "증가", "감소"],
    "대응 분석": ["대응책", "해법", "대안", "방안", "강구", "준비", "대책", "TF"],
    "전망 분석": ["전망", "예상", "할 것으로", "될 듯", "향후"],

    # [정책 개입/자율 주장 그룹] (팩트가 아닌 '주장'만 필터링)
    "정부 개입 강조": ["개입해야", "정부가 나서야", "대책 마련 시급", "제도적 지원 절실", "적극적인 역할", "법제화 필요", "촉구", "가이드라인 마련해야"],
    "시장 자율 강조": ["자율에 맡겨야", "규제 완화 시급", "시장 논리에", "민간 주도로 풀어야", "불필요한 규제 철폐", "관치 벗어나야", "자율성 보장", "과도한 개입"],

    # [팩트성 환경 요인 그룹] (기존의 팩트성 정책/통제 단어는 '정책 요인'으로 흡수)
    "외부 요인(글로벌)": ["글로벌", "유가", "전쟁", "환율", "이란", "공급망"],
    "정책 요인(국내)": ["국내법", "금통위", "한국은행", "국회", "입법", "불승인", "엄격한 심사", "과징금", "규제 완화 발표", "인센티브 제공"]
}


# 3. 가중치 기반 Top-3 분류 엔진
def smart_classify(title, content):
    c_title = clean_and_normalize(title)
    c_content = clean_and_normalize(content)

    scores = {cat: 0 for cat in category_keywords.keys()}

    TITLE_W = 5.0
    CONTENT_W = 1.0

    for cat, keywords in category_keywords.items():
        for kw in keywords:
            scores[cat] += c_title.count(kw) * TITLE_W
            scores[cat] += c_content.count(kw) * CONTENT_W

    # [핵심 해결책] '단순 전달' 점수 후려치기 (Penalty)
    # 다른 분석/감성 카테고리가 우선권을 갖도록 단순 전달 점수에 0.4(60% 감점)를 곱합니다.
    scores["단순 전달"] = scores["단순 전달"] * 0.4

    # 정렬 및 Top 3 추출
    sorted_res = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    top3 = []
    for i in range(3):
        cat, score = sorted_res[i]
        top3.append((cat, round(score, 1)) if score > 0 else ("미분류", 0.0))

    return top3

def update_perspective_to_es():
    es = get_es()

    query = {
        "_source": ["article_id", "title", "clean_text"],
        "query": {
            "bool": {
                "must": [
                    {"exists": {"field": "title"}},
                    {"exists": {"field": "clean_text"}}
                ]
            }
        },
        "size": 500
    }

    resp = es.search(
        index=NEWS_ECONOMY_INDEX,
        body=query,
        scroll="2m"
    )

    scroll_id = resp.get("_scroll_id")

    while True:
        hits = resp["hits"]["hits"]
        if not hits:
            break

        for hit in hits:
            doc_id = hit["_id"]
            src = hit["_source"]

            title = src.get("title", "")
            clean_text = src.get("clean_text", "")

            top3 = smart_classify(title, clean_text)

            perspective = [
                {
                    "rank": i + 1,
                    "category": cat,
                    "score": score
                }
                for i, (cat, score) in enumerate(top3)
            ]

            es.update(
                index=NEWS_ECONOMY_INDEX,
                id=doc_id,
                body={
                    "doc": {
                        "perspective": perspective
                    }
                }
            )

        resp = es.scroll(scroll_id=scroll_id, scroll="2m")

    if scroll_id:
        es.clear_scroll(scroll_id=scroll_id)

    print("관점 분류 Top-3 ES 저장 완료")