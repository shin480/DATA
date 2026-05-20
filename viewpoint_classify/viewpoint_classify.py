import re
from util.es import get_es, NEWS_ECONOMY_INDEX

# 1. 유의어 사전 (기존과 동일하게 유지하되 '강화' 관련 억지 매핑 제거)
synonym_map = {
    "불황": "침체", "경기악화": "침체", "마이너스 성장": "침체",
    "호조": "상승", "해외": "글로벌", "외국": "글로벌", "미국": "글로벌", "중동": "글로벌",
    "예측": "전망", "관측": "전망", "내다봤": "전망", "추산": "전망"
}


def clean_and_normalize(text):
    if not text: return ""
    text = re.sub(r'\[.*?\]|\(.*?\)|=연합뉴스|기자|뉴스|#.*', '', text)
    text = re.sub(r'성과급\w*', '', text)

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
    "기대": ["기대", "호재", "회복", "활기", "수혜", "낙관", "모멘텀", "흥행", "인기", "품절", "운집", "몰린", "열풍","품절대란", "완판", "흥행몰이", "매진"],
    "성과 예찬": ["성과", "달성", "최고등급", "최우수", "입증", "우수성", "쾌거", "성공적", "완판", "호실적", "최대", "돌파"],

    # [정보 전달 및 분석 관점 그룹] (팩트는 이쪽으로 모입니다)
    "단순 전달": [
        # 1. 기존 기본 전달 용어
        "개최", "열렸다", "전했다", "밝혔다", "참석", "알려졌다", "체결", "방문",

        # 2. [추가] 행사/전시 관련 (형태소 한계 극복용 어간 및 유의어)
        "열린", "열린다", "개막", "선보이", "진행", "참여",

        # 3. [추가] 기업 비즈니스 팩트 (수주, 계약, 실적 등)
        "수주", "계약", "M&A", "인수", "합병", "출시", "발표", "공급", "실적", "MOU",
        "모집", "접수", "지급", "시작", "확정", "오픈", "공개",
        "선정", "판매", "할인", "개편", "도입"
    ],
    "원인 분석": ["원인", "배경", "기인", "때문에"],
    "결과 분석": ["집계", "나타났다", "통계", "변동률",
                "기록했다", "상승률", "하락률",
                "흑자", "적자", "급등", "급락", "반등"],
    "대응 분석": ["대응책", "해법", "대안", "방안", "강구", "준비", "대책", "TF"],
    "전망 분석": ["전망", "예상", "할 것으로", "될 듯", "향후"],

    # [정책 개입/자율 주장 그룹] (팩트가 아닌 '주장'만 필터링)
    "정부 개입 강조": ["개입해야", "정부가 나서야", "대책 마련 시급", "제도적 지원 절실", "적극적인 역할", "법제화 필요", "촉구", "가이드라인 마련해야"],
    "시장 자율 강조": ["자율에 맡겨야", "규제 완화 시급", "시장 논리에", "민간 주도로 풀어야", "불필요한 규제 철폐", "관치 벗어나야", "자율성 보장", "과도한 개입"],

    # [팩트성 환경 요인 그룹] (기존의 팩트성 정책/통제 단어는 '정책 요인'으로 흡수)
    "외부 요인(글로벌)": ["국제 유가", "유가 상승", "고환율", "환율 급등", "환율", "공급망", "공급망 차질", "전쟁 여파", "관세", "글로벌 공급망", "유가 상승", "지정학적 리스크"],
    "정책 요인(국내)": ["국내법", "금통위", "한국은행", "국회", "입법", "불승인",
                  "엄격한 심사", "과징금", "규제 완화 발표", "인센티브 제공",
                  "지원금", "정책", "공약", "국정과제", "법정 공휴일",
                  "부정평가", "평가 기준", "제도", "규정", "시행"
                  ]
}


# 3. 가중치 기반 Top-3 분류 엔진
def smart_classify(title, content, sentiment=None, sentiment_score=0.0):
    c_title = clean_and_normalize(title)
    c_content = clean_and_normalize(content)

    scores = {cat: 0 for cat in category_keywords.keys()}

    TITLE_W = 3.0
    CONTENT_W = 1.0

    for cat, keywords in category_keywords.items():
        for kw in keywords:
            scores[cat] += c_title.count(kw) * TITLE_W
            scores[cat] += c_content.count(kw) * CONTENT_W

    # =========================
    # 결과 분석 약한 키워드 보정
    # =========================
    weak_result_keywords = ["상승", "하락", "증가", "감소", "인상", "인하"]

    for kw in weak_result_keywords:
        scores["결과 분석"] += c_title.count(kw) * 1.0
        scores["결과 분석"] += c_content.count(kw) * 0.2

    result_context_words = ["전년", "전월", "대비", "%", "집계", "통계", "기록", "수치"]

    if any(w in c_content for w in result_context_words):
        for kw in weak_result_keywords:
            scores["결과 분석"] += c_title.count(kw) * 1.5
            scores["결과 분석"] += c_content.count(kw) * 0.4


    # =========================
    # 우려 키워드 문맥 보정
    # =========================
    if "부담 완화" in c_content or "부담을 줄" in c_content or "비용 부담을 줄" in c_content:
        scores["우려"] -= 2

    # =========================
    # 홍보성 기사 키워드 보정
    # =========================
    promo_words = ["출시", "선보", "공개", "도입", "확대", "개선", "혜택", "지원"]

    if any(w in c_title for w in promo_words):
        scores["기대"] += 0.8
        scores["성과 예찬"] += 0.3

    # =========================
    # 노사/조정/협상 기사 보정
    # =========================
    labor_policy_words = [
        "노사", "노조", "사측", "중노위", "중앙노동위원회",
        "사후조정", "교섭", "협상", "대화 제안", "재협상"
    ]

    if any(w in c_title or w in c_content for w in labor_policy_words):
        scores["정책 요인(국내)"] += 1.5
        scores["대응 분석"] += 1.2

        # 단순 갈등 기사라고 무조건 비판으로 쏠리는 것 방지
        if "비판" not in c_title and "의혹" not in c_title and "불법" not in c_title:
            scores["비판적 태도"] *= 0.75

    # 단순 전달 penalty
    scores["단순 전달"] *= 0.25
    if scores["단순 전달"] < 2:
        scores["단순 전달"] = 0

    # 감성 confidence 과신 방지
    conf = min(float(sentiment_score or 0), 0.85)
    bias = conf * 2

    if sentiment == "negative":
        scores["비판적 태도"] += bias
        scores["우려"] += bias
        scores["정부 책임"] += bias * 0.2
        scores["기업 책임"] += bias * 0.2
        scores["기대"] -= bias * 0.5
        scores["성과 예찬"] -= bias
    elif sentiment == "positive":
        scores["기대"] += bias
        scores["성과 예찬"] += bias
        scores["비판적 태도"] -= bias * 0.5
        scores["우려"] -= bias * 0.5


    # 음수 방지
    for cat in scores:
        scores[cat] = max(0, scores[cat])

    sorted_res = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # 너무 낮은 점수면 미분류 처리
    if sorted_res[0][1] < 1.0:
        return [("단순 전달", round(sorted_res[0][1], 1)), ("미분류", 0.0), ("미분류", 0.0)]

    top3 = []
    for i in range(3):
        cat, score = sorted_res[i]
        top3.append((cat, round(score, 1)) if score > 0 else ("미분류", 0.0))
    print(top3)
    return top3

def update_perspective_to_es(all:bool=False):
    es = get_es()
    query = {}

    if all:
        query = {
            "_source": ["article_id", "title", "clean_text", "sentiment", "sentiment_score"],
            "query": {
                "bool": {
                    "must": [
                        {"exists": {"field": "title"}},
                        {"exists": {"field": "clean_text"}}
                    ]
                }
            },
        }
    else:
        query = {
            "_source": ["article_id", "title", "clean_text", "sentiment", "sentiment_score"],
            "query": {
                "bool": {
                    "must": [
                        {"exists": {"field": "title"}},
                        {"exists": {"field": "clean_text"}}
                    ],
                    "must_not": [
                        {
                            "nested": {
                                "path": "perspective",
                                "query": {"exists": {"field": "perspective"}}
                            }
                        }
                    ]
                }
            },
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

            sentiment = src.get("sentiment")
            sentiment_score = src.get("sentiment_score", 0.0)

            top3 = smart_classify(title, clean_text, sentiment, sentiment_score)

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

if __name__ == "__main__":
    update_perspective_to_es(True)