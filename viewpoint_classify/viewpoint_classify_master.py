import re
from util.es import get_es, NEWS_ECONOMY_INDEX
from util.logger import save_article_process_log,save_article_error_log

from viewpoint_classify.viewpoint_model import predict_government_responsibility

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


def has_nearby_terms(text, anchors, context_words, window=24):
    """Return True when an anchor and its context occur in the same short span."""
    for anchor in anchors:
        start = 0
        while True:
            idx = text.find(anchor, start)
            if idx < 0:
                break

            left = max(0, idx - window)
            right = min(len(text), idx + len(anchor) + window)
            if any(word in text[left:right] for word in context_words):
                return True

            start = idx + len(anchor)

    return False


# 2. [스마트 딕셔너리] 중의적 단어 제거 + 핵심 단어/짧은 N-gram 혼합
category_keywords = {
    # [책임 소재 그룹]
    "정부 책임": [
        "정부 책임", "정부 책임론", "당국 책임", "정책 실패", "정부 무능",
        "행정 실패", "감독 부실", "관리 감독 부실", "관리 실패",
        "늑장 대응", "정책 혼선", "정책 부작용", "규제 실패",
        "제도 미비", "허술한 관리", "당국의 관리", "당국의 감독",
        "감독당국 책임", "당국 책임 피", "정부 책임 피", "행정 책임",
        "방치 책임", "책임 떠넘긴 정부",
        "정부 탓", "당국 탓", "정부가 방치", "당국이 방치"
    ],
    "기업 책임": ["경영진 책임","경영 실패","의사 결정 실패","책임 전가",
        "현장을 외면","대응 실패","실력 부족","총체적 난국","관리 부실",
        "부실 경영","기업 과실","방만 경영","도덕적 해이","불법 영업",
        "불법 대출", "불법 사채", "불법 유통", "불법 반입", "불법 조업",
        "불법 도박장 출입", "불법 촬영", "불법 행위"],
    "개인 책임": [
        "개인 과실","투자자 책임","본인 부주의","개인 탓",
        "무리한 투자","과도한 투자","묻지마 투자","투자 실패","잘못된 투자 판단","자기 책임","본인 책임",
        "영끌","빚투","투기 수요","투기 심리","과도한 레버리지","고위험 투자","위험을 감수","리스크를 감수",
        "충분히 고지","충분히 설명","설명을 들었음에도","도박성 투자","무분별한 투자","과도한 대출", "스스로 선택",
        "자발적 선택","투자자 스스로","개인 선택의 결과","판단 착오","판단 미스","본인이 감수","책임은 투자자에게","투자자의 몫"
    ],
    "외부 책임": [
        "외생 변수", "불가항력", "불가항력적", "외부 충격",
        "천재지변", "전쟁 탓", "글로벌 변수"
    ],
    "복합 책임": [
        "공동 책임", "공동 과실", "책임 분담", "책임이 엇갈",
        "책임을 나눠", "책임을 함께", "여러 주체의 책임", "책임 소재가 복합"
    ],

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
    "정부 개입 강조": [
        "개입해야", "정부가 나서야", "당국이 나서야", "대책 마련 시급",
        "제도적 지원 절실", "적극적인 역할", "법제화 필요",
        "가이드라인 마련해야", "정부 지원 필요", "공적 지원 필요",
        "보호 장치 필요", "규제 필요"
    ],
    "시장 자율 강조": [
        "자율에 맡겨야", "시장에 맡겨야", "규제 완화 시급", "시장 논리에",
        "민간 주도로 풀어야", "불필요한 규제 철폐", "관치 벗어나야",
        "자율성 보장", "정부 개입 최소화", "규제보다 혁신",
        "규제 풀어야", "규제 걷어", "규제 철폐해야", "민간에 맡겨야",
        "시장 자율", "민간 활력"
    ],

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
    # 4. 단순 전달 (정책/행사 발표)
    # =========================
    info_words = ["개최", "신설", "추진", "토론회", "설명회", "체결", "수주", "용역"]
    # 안내성 단어가 2개 이상 중복해서 나오면 단순 기사일 확률이 높음
    info_hit_count = sum(f"{c_title} {c_content}".count(w) for w in info_words)

    # 기사 내에 명백한 비판/부정적 단어가 있는지 확인
    has_critique = any(w in c_title + c_content for w in ["비판", "논란", "지적", "규탄", "부당", "의혹", "갈등", "실패", "부실", "공백"])

    if info_hit_count >= 2:
        scores["단순 전달"] += 2.0
        # 비판/부정적 문맥이 전혀 없을 때만(진짜 단순 전달/홍보일 때만) 정부 책임 점수 삭감
        if not has_critique:
            scores["정부 책임"] -= 1.0


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

    if any(w in (c_title + c_content) for w in labor_policy_words):
        scores["정책 요인(국내)"] += 1.5
        scores["대응 분석"] += 1.2

        # 단순 갈등 기사라고 무조건 비판으로 쏠리는 것 방지
        if "비판" not in c_title and "의혹" not in c_title and "불법" not in c_title:
            scores["비판적 태도"] *= 0.75

    # =========================
    # 기업/경영진
    # =========================
    company_blame_words = [
        "경영 실패",
        "의사 결정 실패",
        "책임 전가",
        "관리 부실",
        "대응 실패"
    ]

    if any(w in (c_title + c_content) for w in company_blame_words):
        scores["기업 책임"] += 3.0

    company_failure_words = [
        "내부통제 부실",
        "개인정보 유출",
        "불완전판매",
        "담합",
        "갑질",
        "횡령",
        "배임",
        "허위 공시",
        "가격 조작",
        "시세 조작",
        "회계 조작",
        "불법 조작",
        "리콜 은폐",
        "불법 영업",
        "부당거래",
        "분식회계",
        "전산 장애",
        "오지급 사고",
        "부실 대응",
        "노동권 침해",
        "하도급 갑질",
        "임금 체불",
        "불법 대출",
        "불법 사채",
        "토큰 갈취",
        "사기 행위"
    ]

    company_subject_words = [
        "기업",
        "회사",
        "거래소",
        "은행",
        "보험사",
        "카드사",
        "플랫폼",
        "쿠팡",
        "빗썸",
        "업체",
        "사업자",
        "프랜차이즈"
    ]

    if (
            any(s in c_title + c_content for s in company_subject_words)
            and
            any(w in c_title + c_content for w in company_failure_words)
    ):
        scores["기업 책임"] += 3.5

    company_exclude_words = [
        "캠페인",
        "동참",
        "예방",
        "근절",
        "사회공헌",
        "후원",
        "공익",
        "강화",
        "대응 촉구",
        "단속",
        "규제",
        "정부",
        "경찰청",
        "문체부",
        "농식품부",
        "해수부"
    ]

    if any(x in c_title + c_content for x in company_exclude_words):
        scores["기업 책임"] *= 0.3

    # =========================
    # 정부 책임 강조
    # =========================
    government_blamed_patterns = [
        "정부 책임론",
        "당국 책임론",
        "금융당국 책임론",
        "정부 책임",
        "당국 책임",
        "정부를 비판",
        "당국을 비판",
        "정부에 책임",
        "당국에 책임",
        "정부 책임을 묻",
        "당국 책임을 묻",
        "정부가 책임져야",
        "정부가 함께 책임져야",
        "책임을 방기",
        "정책 실패를 인정",
        "정책 실패를 자백"
    ]

    if any(w in c_title + c_content for w in government_blamed_patterns):
        scores["정부 책임"] += 4.0

    government_actor_patterns = [
        "정부가 지원",
        "정부가 추진",
        "정부가 발표",
        "정부가 도입",
        "정부가 확대",
        "정부가 공급",
        "정부가 출범",
        "정부가 질타",
        "국토부가 질타",
        "금감원이 질타",
        "금융당국이 점검",
        "금감원이 검사",
        "노동부는 조사",
        "국세청은 징수",
        "공정위 고발",
        "금감원 제재",
        "금융당국 긴급 대응",
        "노동부가",
        "고용노동부가",
        "금융감독원이",
        "금융위원회",
        "FIU",
        "국세청은",
        "금감원은",
        "공정위는",
        "입건",
        "조사에 착수",
        "검사에 착수",
        "영업정지",
        "과태료",
        "제재",
        "처분",
        "점검",
        "시정 조치",
        "고발 조치"
    ]

    if any(w in c_title + c_content for w in government_actor_patterns):
        scores["정부 책임"] -= 1.0
        scores["정책 요인(국내)"] += 1.5
        scores["대응 분석"] += 1.5

    political_debate_words = [
        "여야 공방",
        "정치권 공방",
        "정치적 공세",
        "정치 이벤트",
        "후보",
        "선거",
        "국민의힘",
        "더불어민주당",
        "민주당",
        "대통령실",
        "논평",
        "원내대변인"
    ]

    if any(w in c_title + c_content for w in political_debate_words):
        scores["정부 책임"] -= 2.0
        scores["비판적 태도"] += 1.5
        scores["정책 요인(국내)"] += 1.0

    responsibility_anchors = [
        "정부 책임",
        "당국 책임",
        "금융당국 책임",
        "기업 책임",
        "회사 책임",
        "투자자 책임",
        "개인 책임",
        "본인 책임"
    ]

    responsibility_denial_words = [
        "아니다",
        "없다",
        "없다고",
        "반박",
        "부인",
        "일축",
        "선 긋",
        "해명",
        "사실과 다르"
    ]

    responsibility_claim_words = [
        "주장",
        "공방",
        "논쟁",
        "책임 떠넘기",
        "책임을 둘러싼"
    ]

    responsibility_fact_words = [
        "감사 결과",
        "조사 결과",
        "판결",
        "확인됐다",
        "드러났다",
        "적발"
    ]

    responsibility_text = c_title + c_content

    has_responsibility_denial = has_nearby_terms(
            responsibility_text,
            responsibility_anchors,
            responsibility_denial_words
    )

    has_unverified_responsibility_claim = (
            has_nearby_terms(
                responsibility_text,
                responsibility_anchors,
                responsibility_claim_words
            )
            and not any(w in responsibility_text for w in responsibility_fact_words)
    )

    # 책임 반박/공방 문맥은 책임 점수 가산이 끝난 뒤 최종 감점한다.
    if has_responsibility_denial or has_unverified_responsibility_claim:
        scores["비판적 태도"] += 1.0


    strict_government_subject_words = [
        "정부", "당국", "국토부", "금융당국", "공정위", "금감원",
    "기재부", "복지부", "노동부", "환경부", "대통령실", "지자체", "당정"
    ]

    loose_government_subject_words = [
        "공정위","금융당국","금감원","국토부"
    ]

    government_failure_words = [
        "정책 실패", "행정 실패", "감독 부실", "관리 실패",
        "늑장 대응", "부실 대응", "정책 혼선",
        "규제 실패", "규제 공백", "제도 허점", "허술한 관리",
        "감독 실패", "행정 공백", "정책 공백", "규제 사각지대",
        "제도 방치", "방치 책임", "책임 회피",
        "관리 소홀", "감독 소홀", "대응 미흡", "대책 부재",
        "예산 삭감", "전액 삭감", "사업 무산", "감사 적발",

        "성과 부실", "규제 논란", "규제 잣대", "법적 허점",
        "관리 체계 문제", "구조조정 대상", "폐지 등급",

        "책임론",
        "금융당국 책임론",
        "정부 책임론",
        "감독 책임",
        "관리 책임",
        "책임 공방",
        "늑장 대응 책임",
        "관리 감독 부실",
        "관리 공백",
        "단속 관리는 공백",
        "강하게 질타",
        "부실했던 업무체계",
        "정책 실패를 자백",
        "정책 실패라는 평가",

        "책임을 방기",
        "정책 실패를 인정",
        "정부 책임론",
        "관리 감독 부실",
        "방치한 결과",
        "감독 책임",
        "부담을 떠안고",
        "현실을 반영하지 못",

        "관리 감독은 허술",
        "전력망 관리 실패",
        "관리 감독 기능이 약",
        "책임을 방기",
        "공백이 도마",
        "공백이 드러",
        "문책"
    ]

    risky_government_failure_words = [
        "허점",
        "구조적 문제",
        "부정수급",
        "실질적 조치",
        "근본 대책"
    ]

    government_loose_failure_words = [
        "실패","허점","혼선","방치"
    ]

    policy_promo_words = [
        "신설", "출범", "확대", "추진", "공급",
        "도입", "개최", "토론회", "설명회"
    ]

    government_victim_patterns = [
        "정부 지원",
        "정부 추진",
        "정부 발표",
        "정부 확대",
        "정부 도입",
        "정부 출범"
    ]

    government_guidance_patterns = [
        "유의사항 안내",
        "소비자 안내",
        "민원 사례",
        "안내했다",
        "설명했다",
        "가입자 유의",
        "계약 전환",
        "보험료 납입",
        "보험료 납부"
    ]

    negative_context_words = [
        "실패", "부실", "허술", "질타",
        "책임 공방", "문제", "책임", "공백",
        "방치", "미흡", "논란", "한계",
        "불가능", "지적", "비판"
    ]

    official_blame_words = [
        "질타",
        "문책",
        "감사 적발",
        "책임론",
        "책임 공방",
        "도덕적 해이",
        "관리 부실",
        "감독 부실",
        "운영 부실",
        "부실 대응",
        "늑장 대응"
    ]

    if (
            any(w in c_title for w in policy_promo_words)
            and
            not any(w in c_title + c_content for w in negative_context_words)
    ):
        scores["정부 책임"] *= 0.6
        scores["정책 요인(국내)"] += 1.5

    gov_loose_title_hit = (
            any(s in c_title for s in loose_government_subject_words)
            and
            any(w in c_title for w in government_loose_failure_words)
    )

    gov_title_hit = (
            any(s in c_title for s in strict_government_subject_words)
            and
            any(f in c_title for f in government_failure_words)
    )

    gov_content_hit = (
            any(s in c_content for s in strict_government_subject_words)
            and
            any(f in c_content for f in government_failure_words)
    )

    gov_risky_context_hit = (
            any(s in c_title + c_content for s in strict_government_subject_words)
            and
            any(w in c_title + c_content for w in risky_government_failure_words)
    )

    if (
            any(s in c_title + c_content for s in strict_government_subject_words)
            and
            any(w in c_title + c_content for w in official_blame_words)
    ):
        scores["정부 책임"] += 2.5

    if gov_risky_context_hit:
        scores["정부 책임"] += 1.2

    if gov_title_hit:
        scores["정부 책임"] += 3.5
    elif gov_loose_title_hit:
        scores["정부 책임"] += 2.8
    elif gov_content_hit:
        scores["정부 책임"] += 2.2

    policy_announcement_patterns = [
        "신설 본격화",
        "출범",
        "공급 확대",
        "도입",
        "추진한다",
        "추진 중",
        "발표했다",
        "검토하고 있다",
        "토론회",
        "협의체",
        "공모사업",
        "지원 확대",
        "전용 펀드",
        "기금 가입",
        "후속 조치"
    ]

    strong_blame_context = [
        "책임론",
        "정책 실패",
        "감독 부실",
        "관리 감독 부실",
        "늑장 대응",
        "책임 공방",
        "문책",
        "감사 적발",
        "방치",
        "질타를 받",
        "비판을 받"
    ]

    non_government_blame_context = [
        "정부와 갈등",
        "정부 상대",
        "대관 업무",
        "국정감사에서",
        "의원들의 질타"
    ]

    if any(w in c_title + c_content for w in non_government_blame_context):
        scores["정부 책임"] -= 2.0
        scores["기업 책임"] += 1.0

    regulator_enforcement_patterns = [
        "점검에 나섰",
        "검사에 착수",
        "조사에 착수",
        "제재에 나섰",
        "시정 명령",
        "과징금 부과",
        "고발 조치",
        "영업정지 처분"
    ]

    if (
            any(s in c_title + c_content for s in strict_government_subject_words)
            and any(w in c_title + c_content for w in regulator_enforcement_patterns)
    ):
        scores["정부 책임"] -= 2.0
        scores["정책 요인(국내)"] += 1.5
        scores["대응 분석"] += 1.0

        if any(w in c_title + c_content for w in company_failure_words):
            scores["기업 책임"] += 1.5

    if (
            any(w in c_title + c_content for w in policy_announcement_patterns)
            and not any(w in c_title + c_content for w in strong_blame_context)
    ):
        scores["정부 책임"] -= 2.0
        scores["정책 요인(국내)"] += 2.0
        scores["대응 분석"] += 1.0

    if any(w in c_title + c_content for w in government_victim_patterns):
        scores["정부 책임"] -= 1.5
        scores["정책 요인(국내)"] += 1.0

    market_autonomy_claim_words = [
        "시장에 맡",
        "자율에 맡",
        "민간 주도",
        "민간 자율",
        "기업 자율",
        "자율 규제",
        "시장 기능 회복",
        "시장 논리",
        "정부 개입 최소화",
        "개입 줄여",
        "규제보다 혁신",
        "과도한 규제",
        "관치 논란",
        "관치 벗어나",
        "규제 풀어",
        "규제 걷어",
        "민간에 맡",
        "시장 자율",
        "민간 활력"
    ]

    market_autonomy_direct_claim_words = [
        "시장에 맡겨야",
        "자율에 맡겨야",
        "민간 주도로 풀어야",
        "민간에 맡겨야",
        "정부 개입 최소화",
        "개입 줄여야",
        "관치 벗어나야",
        "규제 풀어야",
        "규제 걷어야",
        "규제 철폐해야",
        "규제보다 혁신"
    ]

    market_autonomy_normative_words = [
        "필요",
        "해야",
        "시급",
        "주장",
        "촉구",
        "요구",
        "건의",
        "제안",
        "권고",
        "요청",
        "풀어야",
        "걷어야",
        "폐지해야"
    ]

    market_autonomy_announcement_words = [
        "정부가 발표",
        "당국이 발표",
        "지원책 발표",
        "대책 발표",
        "시행한다",
        "도입한다",
        "확대한다"
    ]

    market_autonomy_regulation_words = [
        "규제 완화",
        "규제 철폐",
        "규제 개선",
        "규제 혁신",
        "진입 규제",
        "인허가 규제"
    ]

    market_autonomy_execution_words = [
        "공공주도",
        "공공 주도",
        "감시 강화",
        "감시 고삐",
        "보호 체계",
        "보호 강화",
        "점검에 나섰",
        "검사에 착수",
        "규제 강화",
        "관리 강화"
    ]

    has_market_autonomy_anchor = any(
        w in c_title + c_content for w in market_autonomy_claim_words
    )

    has_direct_market_autonomy_claim = any(
        w in c_title + c_content for w in market_autonomy_direct_claim_words
    )

    has_normative_regulation_claim = has_nearby_terms(
        c_title + c_content,
        market_autonomy_regulation_words,
        market_autonomy_normative_words,
        window=18
    )

    has_normative_market_autonomy_claim = (
            has_normative_regulation_claim
            or has_nearby_terms(
                c_title + c_content,
                market_autonomy_claim_words,
                market_autonomy_normative_words,
                window=14
            )
    )

    has_market_autonomy_claim = (
            has_direct_market_autonomy_claim
            or has_normative_market_autonomy_claim
    )

    if has_market_autonomy_claim:
        scores["시장 자율 강조"] += 2.0

        if has_normative_market_autonomy_claim:
            scores["시장 자율 강조"] += 2.0

        if any(w in c_title + c_content for w in market_autonomy_announcement_words):
            scores["시장 자율 강조"] *= 0.5

    if (
            has_market_autonomy_anchor
            and any(w in c_title + c_content for w in market_autonomy_execution_words)
            and not has_market_autonomy_claim
    ):
        scores["시장 자율 강조"] *= 0.25

    intervention_direct_claim_words = [
        "정부가 나서야",
        "당국이 나서야",
        "정부 개입 필요",
        "당국 개입 필요",
        "공적 지원 필요",
        "정부 지원 필요",
        "제도적 지원 절실",
        "보호 장치 필요",
        "규제 필요",
        "법제화 필요"
    ]

    intervention_context_words = [
        "정부",
        "당국",
        "공공",
        "법제",
        "규제",
        "보호"
    ]

    intervention_action_words = [
        "나서야",
        "개입해야",
        "지원해야",
        "필요",
        "절실",
        "마련해야",
        "강화해야",
        "입법해야"
    ]

    has_intervention_claim = (
            any(w in c_title + c_content for w in intervention_direct_claim_words)
            or has_nearby_terms(
                c_title + c_content,
                intervention_context_words,
                intervention_action_words,
                window=16
            )
    )

    if has_intervention_claim:
        scores["정부 개입 강조"] += 3.0

    anti_intervention_words = [
        "정부 개입을 줄",
        "정부 개입 최소화",
        "개입 줄여",
        "과도한 개입",
        "관치 벗어나"
    ]

    if (
            has_market_autonomy_claim
            and any(w in c_title + c_content for w in anti_intervention_words)
    ):
        scores["정부 개입 강조"] *= 0.25

    if any(w in c_title for w in ["정부 책임론", "금융당국 책임론", "당국 책임론"]):
        scores["정부 책임"] += 6

    # =========================
    # 5. 예외 규칙 (오분류 방지)
    # =========================
    # 정부가 사기를 당했거나 주체가 국민연금 등 제3자인 경우 '정부 책임' 점수 차감
    gov_victim_words = ["정부가 사기", "정부 기관이 사기", "정부를 믿고"]
    if any(w in (c_title + c_content) for w in gov_victim_words):
        scores["정부 책임"] -= 1.0
        scores["기업 책임"] += 2.0  # 보통 이런 류는 기업의 비위인 경우가 많음

    if "국민연금" in c_title + c_content and any(w in c_title + c_content for w in ["개인정보 유출", "책임 투자", "ESG", "쿠팡"]):
        scores["기업 책임"] += 1.5

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

    # =========================
    # 개인 책임
    # =========================

    personal_words = [
        "영끌",
        "빚투",
        "묻지마 투자",
        "고위험 투자",
        "레버리지 투자",
        "신용융자",
        "신용거래",
        "투기 심리",
        "무리한 투자",
        "과도한 대출",
        "투자 실패",
        "자기 책임",
        "본인 책임"
    ]

    personal_subject_words = [
        "개인",
        "본인",
        "투자자",
        "차주",
        "영끌",
        "빚투"
    ]

    has_personal_subject = any(
        w in c_title + c_content for w in personal_subject_words
    )

    if (
            has_personal_subject
            and any(w in (c_title + c_content) for w in personal_words)
    ):
        scores["개인 책임"] += 3.0

    personal_research_context_words = [
        "연구 결과",
        "보고서",
        "결혼 확률",
        "출산",
        "임대주택",
        "연구진",
        "통계 분석"
    ]

    personal_strong_blame_words = [
        "영끌",
        "빚투",
        "묻지마 투자",
        "투자자 책임",
        "자기 책임",
        "본인 책임",
        "스스로 선택",
        "투기 심리"
    ]

    investor_victim_words = [
        "투자자 피해",
        "피해 투자자",
        "피해자",
        "피해 구제",
        "구제 방안",
        "배상",
        "보상",
        "설명의무 위반",
        "불완전판매",
        "기망",
        "속여"
    ]

    has_investor_victim_context = (
            "투자자" in c_title + c_content
            and any(w in c_title + c_content for w in investor_victim_words)
    )

    if (
            "투자자" in c_title + c_content
            and
            any(w in c_title + c_content for w in [
                "손실",
                "피해",
                "판단",
                "선택",
                "위험"
            ])
            and not has_investor_victim_context
    ):
        scores["개인 책임"] += 2.0

    if has_investor_victim_context:
        scores["개인 책임"] *= 0.25
        scores["대응 분석"] += 1.0

    if any(w in (c_title + c_content) for w in [
        "사기",
        "횡령",
        "배임",
        "불완전판매",
        "개인정보 유출",
        "내부통제 부실",
        "담합"
    ]):
        scores["개인 책임"] *= 0.3
        scores["기업 책임"] += 2.0

    if (
            any(w in c_title + c_content for w in personal_research_context_words)
            and not any(w in c_title + c_content for w in personal_strong_blame_words)
    ):
        scores["개인 책임"] *= 0.35

    external_shock_words = [
        "전쟁 여파",
        "기상이변",
        "자연재해",
        "공급망 차질",
        "공급망 교란",
        "국제 유가",
        "환율 충격",
        "글로벌 수요 둔화"
    ]

    external_responsibility_context_words = [
        "불가항력",
        "외부 충격",
        "외생 변수",
        "통제하기 어려",
        "피할 수 없",
        "어쩔 수 없"
    ]

    external_impact_words = [
        "타격",
        "악화",
        "급등",
        "급락",
        "차질",
        "침체",
        "부담",
        "리스크"
    ]

    if (
            any(w in c_title + c_content for w in external_shock_words)
            and any(w in c_title + c_content for w in external_responsibility_context_words)
    ):
        scores["외부 책임"] += 2.5

    if (
            any(w in c_title + c_content for w in [
                "전쟁 여파",
                "기상이변",
                "자연재해",
                "공급망 차질",
                "공급망 교란",
                "환율 충격",
                "글로벌 수요 둔화"
            ])
            and any(w in c_title + c_content for w in external_impact_words)
    ):
        scores["외부 책임"] += 0.8

    mixed_responsibility_words = [
        "공동 책임",
        "책임이 엇갈",
        "책임 소재가 복합",
        "여러 주체의 책임",
        "정부와 업계 모두 책임",
        "감독당국과 금융사 책임",
        "공동 과실",
        "책임 분담",
        "책임을 나눠",
        "책임을 함께"
    ]

    if any(w in c_title + c_content for w in mixed_responsibility_words):
        scores["복합 책임"] += 2.5

    if has_nearby_terms(
            c_title + c_content,
            ["책임", "과실", "책임 소재"],
            ["공동", "복합", "맞물", "여러 주체", "엇갈"]
    ):
        scores["복합 책임"] += 2.0

    if has_responsibility_denial:
        scores["정부 책임"] *= 0.03
        scores["기업 책임"] *= 0.2
        scores["개인 책임"] *= 0.2
    elif has_unverified_responsibility_claim:
        scores["정부 책임"] *= 0.55
        scores["기업 책임"] *= 0.75
        scores["개인 책임"] *= 0.75

    if (
            any(w in c_title + c_content for w in government_guidance_patterns)
            and any(w in c_title + c_content for w in government_actor_patterns)
            and not any(w in c_title + c_content for w in strong_blame_context)
    ):
        scores["정부 책임"] *= 0.2
        scores["정책 요인(국내)"] += 0.5

    # 넓은 기사 형식/감성 카테고리가 책임·정책 주장 카테고리를 덮지 않도록
    # 단일 Top-3 경쟁에서만 점수 균형을 보정한다.
    dominant_category_scales = {
        "정책 요인(국내)": 0.72,
        "결과 분석": 0.78,
        "단순 전달": 0.72,
        "성과 예찬": 0.84,
        "기대": 0.86,
        "우려": 0.86,
        "비판적 태도": 0.9,
        "외부 요인(글로벌)": 0.9
    }

    for category, scale in dominant_category_scales.items():
        scores[category] *= scale

    responsibility_categories = [
        "정부 책임",
        "기업 책임",
        "개인 책임",
        "외부 책임",
        "복합 책임"
    ]

    responsibility_boost_floors = {
        "정부 책임": 1.0,
        "기업 책임": 2.5,
        "개인 책임": 3.0,
        "외부 책임": 2.5,
        "복합 책임": 2.5
    }

    for category in responsibility_categories:
        if scores[category] >= responsibility_boost_floors[category]:
            scores[category] = scores[category] * 1.2 + 0.8

    if scores["정부 책임"] >= 2.0:
        scores["정부 책임"] += 1.2

    for category in ["정부 개입 강조", "시장 자율 강조"]:
        if scores[category] >= 2.0:
            scores[category] = scores[category] * 1.25 + 1.0


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

            try:
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
                # save_article_process_log("A202", doc_id, "success")
            except Exception as e:
                print("관점 분석 실패:", doc_id, e)

                # =========================
                # 실패 process log
                # =========================
                history_id = save_article_process_log(
                    "A202",
                    doc_id,
                    "fail"
                )

                # =========================
                # 상세 에러 로그
                # =========================
                save_article_error_log(
                    history_id=history_id,
                    error_code="E005",
                    error_message=str(e)
                )

        resp = es.scroll(scroll_id=scroll_id, scroll="2m")

    if scroll_id:
        es.clear_scroll(scroll_id=scroll_id)

    print("관점 분류 Top-3 ES 저장 완료")

if __name__ == "__main__":
    update_perspective_to_es(True)
