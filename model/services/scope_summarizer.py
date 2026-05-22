"""
scope 대표 요약 생성 서비스 — 기존 embedding 재활용 기반

흐름:
  1. news_scopes에서 scope_summary=NULL scope 조회
  2. news_economy에서 content + embedding 수집 (동일 언론사 최대 3건, 최신순)
  3. 본문 클렌징 (언론사 헤더/기자명/저작권/AI분석 문구 등 제거)
  4. 문장 분리 → 중복 제거
  5. 기사 embedding 평균 → 스콥 중심 벡터
     각 문장이 속한 기사의 embedding을 문장 벡터로 사용
     → 중심 벡터와 코사인 유사도 가장 높은 문장 선택
  6. 70자 자연 절단
  7. news_scopes.scope_summary upsert

[수정 이력]
- 속도 개선: KR-FinBERT 문장별 실시간 임베딩 → 기존 news_economy.embedding 재활용
  · KR-FinBERT CPU 추론 → 시간당 1000건 수준으로 너무 느림
  · embedding 필드는 이미 계산된 768차원 벡터 → 추가 연산 없음
- 클렌징 강화: "관련 기사 N건을 분석한 결과입니다" 등 AI 생성 문구 패턴 추가
- 기사 없는 스콥: 빈 문자열 저장 → return None으로 변경 (카운트 오염 방지)
"""

import logging
import re
from datetime import datetime, timezone

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from model.database import get_es
from model.services.error_logger import log_pipeline_error

logger = logging.getLogger(__name__)

BATCH_SIZE          = 10000
MAX_NEWS_PER_PRESS  = 3
DUPLICATE_THRESHOLD = 0.85
MAX_SENTENCE_LENGTH = 100
MIN_SENTENCE_LENGTH = 15
INDEX_NEWS          = "news_economy"
INDEX_SCOPES        = "news_scopes"

_TRIM_CHARS = set('다며서고은는이가을를에의')


# ── 본문 클렌징 ────────────────────────────────────────────────────

_CLEAN_PATTERNS = [
    # 방송사 태그: [KBS 춘천], [MBC] 등
    (re.compile(r"^\s*\[[^\]]{1,20}\]\s*"), ""),
    (re.compile(r"\[[^\]]{1,20}\]"), " "),
    # 방송 스크립트 앵커/리포트 블록 제거
    # "[앵커] ...문장들... [리포트]" 구간 전체 제거
    (re.compile(r"\[앵커\].+?(?=\[리포트\])", re.DOTALL), ""),
    (re.compile(r"\[앵커\].+", re.DOTALL), ""),
    (re.compile(r"\[리포트\]\s*"), ""),
    # 【 앵커멘트 】 패턴 (KBS 등 방송 스크립트)
    (re.compile(r"【\s*앵커멘트\s*】.+?(?=【)", re.DOTALL), ""),
    (re.compile(r"【\s*앵커멘트\s*】.+", re.DOTALL), ""),
    (re.compile(r"【[^】]{1,20}】\s*"), ""),
    # MBC 방송 스크립트: ◀앵커▶, ◀리포트▶ 패턴
    (re.compile(r"◀\s*앵커\s*▶.+?(?=◀\s*(?:리포트|기자)\s*▶)", re.DOTALL), ""),
    (re.compile(r"◀\s*앵커\s*▶.+", re.DOTALL), ""),
    (re.compile(r"◀\s*(?:리포트|기자|END)\s*▶\s*"), ""),
    # 인터뷰 인용 태그: [홍길동/직책 : "..."] → 태그만 제거, 내용 유지
    (re.compile(r"\[[가-힣\s/·]{2,30}\s*:\s*"), ""),
    # 언론사 발신 헤더: (서울=뉴스1) 홍길동 기자 =
    (re.compile(r"\([^)]{1,20}=[^)]{1,20}\)\s*[가-힣\s]{2,10}기자\s*=?\s*"), ""),
    # 기자 byline
    (re.compile(r"[가-힣]{2,4}[·]?[가-힣]{0,4}\s*기자\s*=?\s*"), ""),
    # ⓒ 저작권 표시
    (re.compile(r"ⓒ\s*[^\s,]{1,20}"), ""),
    # 날짜 패턴
    (re.compile(r"\d{4}[.\-]\d{1,2}[.\-]\d{1,2}"), ""),
    # 저작권 문구
    (re.compile(r"무단\s*전재[^.]*재배포\s*금지[^.]*\.?"), ""),
    (re.compile(r"저작권자[^.]*\.?"), ""),
    (re.compile(r"All rights reserved[^.]*\.?", re.IGNORECASE), ""),
    # 사진/그래픽 캡션
    (re.compile(r"\([^)]{1,30}=[^)]{1,30}\)"), ""),
    # 외신 사진 크레딧: (REUTERS/Mohammed Aty), (AP/...), (AFP/...)
    (re.compile(r"\((?:REUTERS|AP|AFP|EPA|Getty)[^)]*\)"), ""),
    # 이미지 확대 텍스트
    (re.compile(r"이미지\s*확대\s*"), ""),
    # 사진 면책 문구: "사진은 기사와 관련 없음", "사진=게티이미지뱅크"
    (re.compile(r"사진은\s*기사와\s*관련\s*없음"), ""),
    (re.compile(r"/?\s*사진=게티이미지[^\s/]*"), ""),
    (re.compile(r"/?\s*사진=[^\s/]{1,30}"), ""),
    # 유료/선공개 안내 문구
    (re.compile(r"이\s*기사는?\s*\d{4}년.+?(?:선공개|공개)\s*되었습니다[.。]?"), ""),
    (re.compile(r"이\s*기사는?\s*.{1,30}(?:프리미엄|유료)\s*콘텐츠.+?[.。]"), ""),
    (re.compile(r"(?:프리미엄|유료)\s*콘텐츠로?\s*선공개.+?[.。]"), ""),
    (re.compile(r"구독자\s*전용.+?[.。]"), ""),
    (re.compile(r"유료\s*기사입니다.+?[.。]"), ""),
    # AI 분석 문구: "관련 기사 N건을 분석한 결과입니다" 등
    (re.compile(r"관련\s*기사\s*\d+건[^.]*\.?"), ""),
    (re.compile(r"\d+건을?\s*분석한\s*결과[^.]*\.?"), ""),
    (re.compile(r"이\s*기사[는은]\s*AI[^.]*\.?"), ""),
    (re.compile(r"AI\s*(가|이|로)\s*[^.]{1,30}(작성|생성|요약)[^.]*\.?"), ""),
    # 언론사 기사 분류 태그: (종합), (종합2보), (상보), (1보), (속보) 등
    (re.compile(r"^\s*\((?:종합\d*|상보|속보|\d+보)\)\s*"), ""),
    (re.compile(r"\((?:종합\d*|상보|속보|\d+보)\)"), ""),
    # 기자 이메일 포함 byline: [김혜인 haileykim0516@gmail.com]
    (re.compile(r"\[[가-힣a-zA-Z\s]+\s+[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\]"), ""),
    # 이메일 주소 단독
    (re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"), ""),
    # 연합뉴스 모바일 UI 네비게이션 텍스트
    # "최찬흥 기자 구독 구독중 이전 다음" 패턴
    (re.compile(r"구독중?\s*"), ""),
    (re.compile(r"이전\s+다음"), ""),
    # "관련 뉴스" 이하 블록 제거 (연합뉴스 관련기사 목록)
    (re.compile(r"관련\s*뉴스.+", re.DOTALL), ""),
    # URL 제거 (http/https 링크 전체)
    (re.compile(r"https?://\S+"), ""),
    # SNS 계정: @노컷뉴스, @연합뉴스 등
    (re.compile(r"@[가-힣a-zA-Z0-9_]{2,20}"), ""),
    # 언론사 SNS/연락처 블록: "이메일 : ... 카카오톡 : ... 사이트 : ..."
    (re.compile(r"이메일\s*:\s*.+", re.DOTALL), ""),
    (re.compile(r"카카오톡\s*:\s*\S+"), ""),
    (re.compile(r"사이트\s*:\s*\S+"), ""),
    (re.compile(r"유튜브\s*:\s*\S+"), ""),
    (re.compile(r"텔레그램\s*:\s*\S+"), ""),
    (re.compile(r"인스타그램\s*:\s*\S+"), ""),
    # 전화번호 패턴
    (re.compile(r"\d{2,4}[-\s]?\d{3,4}[-\s]?\d{4}"), ""),
    # 언론사 구독 유도 문구
    (re.compile(r"구독하기.+", re.DOTALL), ""),
    (re.compile(r"뉴스레터.+", re.DOTALL), ""),
    (re.compile(r"더\s*많은\s*뉴스.+", re.DOTALL), ""),
    (re.compile(r"자세한\s*내용은.+", re.DOTALL), ""),
    # 기사 말미 광고/홍보 문구 패턴
    (re.compile(r"▶.+", re.DOTALL), ""),
    (re.compile(r"☞.+", re.DOTALL), ""),
    (re.compile(r"※.+", re.DOTALL), ""),
    # 사진 캡션 패턴: "~하는 모습.", "~장면.", "~모습이다." 등
    (re.compile(r"[^.!?]{5,60}(?:하는|된|찍은|촬영한|모인|참석한)\s*모습[.。]?"), ""),
    (re.compile(r"[^.!?]{5,60}(?:장면|광경)[.。]?"), ""),
    (re.compile(r"[^.!?]{5,60}모습이다[.。]?"), ""),
    # 날짜 + 인물 + 행위 + "모습" 패턴: "27일 홍길동이 ~하는 모습"
    (re.compile(r"\d{1,2}일\s+[가-힣]{2,10}(?:이|가|은|는)\s+[^.]{5,50}모습[.。]?"), ""),
    # 제작진 크레딧 블록
    # "그래픽 홍길동 / 영상취재 김철수 / 영상편집 이영희" 등 패턴
    (re.compile(
        r"(?:그래픽|디자인|영상취재|영상촬영|영상기자|촬영기자|영상편집|작가)"
        r"[^.!?\n]{0,50}"
    ), ""),
    # 중복 공백
    (re.compile(r"\s{2,}"), " "),
]


def _clean_content(text: str) -> str:
    for pattern, repl in _CLEAN_PATTERNS:
        text = pattern.sub(repl, text)
    return text.strip()


# ── 문장 처리 ──────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in raw if len(s.strip()) >= MIN_SENTENCE_LENGTH]


def _remove_duplicates(sentences: list[str]) -> list[str]:
    if len(sentences) < 2:
        return sentences
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 3), max_features=2000
    )
    try:
        matrix = vectorizer.fit_transform(sentences).toarray()
    except ValueError:
        return sentences
    sim           = cosine_similarity(matrix)
    keep, removed = [], set()
    for i in range(len(sentences)):
        if i in removed:
            continue
        keep.append(sentences[i])
        for j in range(i + 1, len(sentences)):
            if sim[i][j] >= DUPLICATE_THRESHOLD:
                removed.add(j)
    return keep


# ── 인용구 문장 필터 ──────────────────────────────────────────────

# 이 패턴이 포함된 문장은 요약 후보에서 제외
# 사진 캡션 문장 감지 패턴
_CAPTION_PATTERNS = [
    re.compile(r"(?:하는|된|찍은|모인|참석한)\s*모습[.。]?\s*$"),
    re.compile(r"(?:장면|광경)[.。]?\s*$"),
    re.compile(r"모습이다[.。]?\s*$"),
    re.compile(r"\d{1,2}일\s+[가-힣]{2,10}(?:이|가)\s+[^.]{5,40}모습"),
]

def _is_caption_sentence(sentence: str) -> bool:
    """사진 캡션 문장이면 True 반환"""
    return any(p.search(sentence) for p in _CAPTION_PATTERNS)


# 메타/안내 문장 감지 패턴 (유료 안내, 날짜 고지 등)
_META_PATTERNS = [
    re.compile(r"이\s*기사는?\s*\d{4}년"),
    re.compile(r"프리미엄\s*콘텐츠"),
    re.compile(r"선공개\s*되었"),
    re.compile(r"구독자\s*전용"),
    re.compile(r"유료\s*기사"),
    re.compile(r"본\s*기사는?\s*.{1,30}제공"),
]

def _is_meta_sentence(sentence: str) -> bool:
    """유료안내/메타 문장이면 True 반환"""
    return any(p.search(sentence) for p in _META_PATTERNS)


# ── 문장 품질 필터 ─────────────────────────────────────────────────

# 이미지 출처 패턴
_IMAGE_SOURCE_PATTERN = re.compile(
    r"(?:\[|\()?[가-힣a-zA-Z\s]{2,20}(?:홈페이지|캡처|제공|촬영|AFP|AP|EPA|로이터)[.\])]?"
)

def _is_valid_sentence(sentence: str) -> bool:
    """
    최소 품질 기준을 통과한 문장만 요약 후보로 허용.

    기준:
      1. 20자 이상
      2. 한글 비율 40% 이상 (이미지 캡션, 영문 URL 등 제거)
      3. 경제 키워드 1개 이상 포함
      4. 캡션/인용/메타/이미지출처 패턴 없음
    """
    # 1. 길이 기준
    if len(sentence) < 20:
        return False

    # 2. 한글 비율
    korean_chars = sum(1 for c in sentence if '가' <= c <= '힣')
    if korean_chars / len(sentence) < 0.4:
        return False

    # 3. 경제 키워드 최소 1개
    if not any(kw in sentence for kw in _ECON_KEYWORDS):
        return False

    # 4. 노이즈 패턴 없음
    if _is_caption_sentence(sentence):
        return False
    if _is_meta_sentence(sentence):
        return False
    if _IMAGE_SOURCE_PATTERN.search(sentence):
        return False

    return True


_QUOTE_PATTERNS = [
    re.compile(r"익명"),
    re.compile(r"한\s*전문가"),
    re.compile(r"관계자[는은]\s*"),
    re.compile(r"당국자[는은]\s*"),
    re.compile(r"[가-힣]{2,4}\s*(?:장관|의원|대표|대통령|총리|장|회장|사장|이사)[는은이가]\s*[""']"),
    re.compile(r"^[""']"),           # 따옴표로 시작
    re.compile(r"라고\s*(?:말했|밝혔|전했|했)"),
    re.compile(r"라며\s*"),
    re.compile(r"고\s*(?:말했|밝혔|전했|했)"),
]

def _is_quote_sentence(sentence: str) -> bool:
    """인용구/익명 발언 문장이면 True 반환"""
    return any(p.search(sentence) for p in _QUOTE_PATTERNS)


def _natural_trim(sentence: str, limit: int = MAX_SENTENCE_LENGTH) -> str:
    if len(sentence) <= limit:
        return sentence
    cut = sentence[:limit]
    for ch in range(len(cut) - 1, -1, -1):
        if cut[ch] in _TRIM_CHARS:
            return cut[:ch + 1] + "…"
    return cut[:limit] + "…"


# ── 문장 품질 점수 ────────────────────────────────────────────────

# 경제 관련성 키워드
_ECON_KEYWORDS = {
    "은행", "증권", "금융", "보험", "투자", "펀드", "자산",
    "기업", "회사", "법인", "계열사", "자회사", "지주",
    "인수", "합병", "매각", "계약", "협약", "협력", "제휴",
    "상장", "공모", "IPO", "유상증자",
    "금리", "환율", "물가", "GDP", "성장률", "실업률",
    "수출", "수입", "무역", "관세", "경상수지",
    "주가", "코스피", "코스닥", "증시", "시가총액",
    "매출", "영업이익", "순이익", "실적", "분기", "연간",
    "부동산", "집값", "전세", "아파트", "분양",
    "세금", "세율", "과세", "감세", "세제", "절세",
    "규제", "완화", "정책", "예산", "재정", "국채",
    "금통위", "한국은행", "연준", "기재부", "금감원",
}

# 수치/변화 키워드: 구체적 팩트 문장 우선
_FACT_PATTERNS = [
    re.compile(r"\d+[\.,]?\d*\s*(?:%|퍼센트|배|조|억|만|천|달러|원|위안|엔)"),
    re.compile(r"역대|최초|처음|사상|최대|최소|최고|최저|신기록"),
    re.compile(r"전년\s*(?:대비|동기|동월)"),
    re.compile(r"전월\s*(?:대비|대비)"),
    re.compile(r"(?:증가|감소|상승|하락|급등|급락|돌파|하회|상회)\s*(?:했|했다|했으며|했고)"),
]

def _fact_score(sentence: str) -> float:
    """수치/구체적 변화 포함 문장에 가중치 반환 (0.0 ~ 0.15)"""
    count = sum(1 for p in _FACT_PATTERNS if p.search(sentence))
    return min(count * 0.05, 0.15)

def _econ_score(sentence: str) -> float:
    """경제 키워드 포함 가중치 반환 (0.0 ~ 0.10)"""
    count = sum(1 for kw in _ECON_KEYWORDS if kw in sentence)
    return min(count * 0.05, 0.10)

def _keyword_overlap_penalty(sentence: str, scope_keywords: list[str]) -> float:
    """scope_keywords와 겹치는 단어가 많을수록 패널티 (중복 정보 방지)"""
    if not scope_keywords:
        return 0.0
    overlap = sum(1 for kw in scope_keywords if kw in sentence)
    # 키워드 3개 이상 겹치면 -0.10 (타이틀과 차별화)
    return -min(overlap * 0.03, 0.10)


# ── 임베딩 기반 중심 문장 선택 ────────────────────────────────────

def _select_central_sentence(
    sentences: list[str],
    sent_embeddings: list[np.ndarray],
    scope_keywords: list[str] | None = None,
) -> int:
    """
    최적 요약 문장 선택:
      코사인유사도 + 경제관련성 + 수치/팩트 가중치 - 인용구 패널티 - 키워드 중복 패널티

    타이틀/키워드와 중복되지 않는 구체적 팩트 문장을 우선 선택.
    """
    matrix   = np.vstack(sent_embeddings).astype(np.float32)
    centroid = matrix.mean(axis=0, keepdims=True)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    sims = cosine_similarity(centroid, matrix)[0]

    kws = scope_keywords or []
    final_scores = np.array([
        sim
        + _econ_score(sent)
        + _fact_score(sent)
        + _keyword_overlap_penalty(sent, kws)
        + (-0.5 if _is_quote_sentence(sent) else 0.0)
        + (-0.5 if _is_caption_sentence(sent) else 0.0)
        + (-0.5 if _is_meta_sentence(sent) else 0.0)
        for sim, sent in zip(sims, sentences)
    ])
    return int(np.argmax(final_scores))


# ── 메인 생성 함수 ─────────────────────────────────────────────────

def generate_scope_summary(es, scope_id: str):
    # scope_keywords 조회 (키워드 중복 패널티용)
    scope_keywords: list[str] = []
    try:
        scope_res = es.get(index=INDEX_SCOPES, id=scope_id, ignore=404)
        if scope_res.get("found"):
            kw_str = scope_res["_source"].get("scope_keywords", "")
            scope_keywords = [k.strip() for k in kw_str.split(",") if k.strip()]
    except Exception:
        pass

    # 뉴스 content + embedding 수집
    res = es.search(
        index=INDEX_NEWS,
        body={
            "query": {
                "bool": {
                    "must": [
                        {"term":   {"scopeID": scope_id}},
                        {"exists": {"field":   "content"}},
                        {"exists": {"field":   "embedding"}},
                    ]
                }
            },
            "_source": ["content", "press", "embedding"],
            "sort":    [{"published_at": "desc"}],
            "size":    100,
        },
    )
    rows = [h["_source"] for h in res["hits"]["hits"]]
    if not rows:
        # 빈 값 저장 안 함 → 다음 배치에서 재시도 가능하도록
        logger.warning(f"content/embedding 없음, 스킵: scope_id={scope_id}")
        return None

    # 동일 언론사 최대 MAX_NEWS_PER_PRESS건
    press_count = {}
    filtered    = []  # (cleaned_content, embedding_vec)
    for r in rows:
        press = r.get("press") or "unknown"
        press_count[press] = press_count.get(press, 0) + 1
        if press_count[press] <= MAX_NEWS_PER_PRESS and r.get("content") and r.get("embedding"):
            cleaned = _clean_content(r["content"])
            emb     = np.array(r["embedding"], dtype=np.float32)
            if emb.shape[0] == 768:
                filtered.append((cleaned, emb))

    if not filtered:
        logger.warning(f"유효한 content/embedding 없음, 스킵: scope_id={scope_id}")
        return None

    # 문장 분리 → 문장별로 소속 기사 embedding 매핑
    all_sentences  = []
    all_embeddings = []
    for cleaned, emb in filtered:
        sents = _split_sentences(cleaned)
        for s in sents:
            all_sentences.append(s)
            all_embeddings.append(emb)

    if not all_sentences:
        # 문장 분리 실패 → 첫 번째 기사 첫 단락 절단
        summary = _natural_trim(filtered[0][0])
        _save(es, scope_id, summary)
        return summary

    # 품질 필터: 최소 기준 통과한 문장만 후보로
    valid_sentences  = []
    valid_embeddings = []
    for s, e in zip(all_sentences, all_embeddings):
        if _is_valid_sentence(s):
            valid_sentences.append(s)
            valid_embeddings.append(e)

    # 유효 문장이 없으면 저장 안 함 (유료기사/노이즈만 있는 스콥)
    if not valid_sentences:
        logger.warning(f"유효 문장 없음, 스킵: scope_id={scope_id}")
        return None

    all_sentences  = valid_sentences
    all_embeddings = valid_embeddings

    # 중복 제거 (embedding도 같이 필터링)
    if len(all_sentences) >= 2:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 3), max_features=2000
        )
        try:
            matrix = vectorizer.fit_transform(all_sentences).toarray()
            sim    = cosine_similarity(matrix)
            keep_idx = []
            removed  = set()
            for i in range(len(all_sentences)):
                if i in removed:
                    continue
                keep_idx.append(i)
                for j in range(i + 1, len(all_sentences)):
                    if sim[i][j] >= DUPLICATE_THRESHOLD:
                        removed.add(j)
            all_sentences  = [all_sentences[i]  for i in keep_idx]
            all_embeddings = [all_embeddings[i] for i in keep_idx]
        except ValueError:
            pass

    # 문장 1개면 바로 사용
    if len(all_sentences) == 1:
        summary = _natural_trim(all_sentences[0])
        _save(es, scope_id, summary)
        return summary

    # embedding 기반 중심 문장 선택
    best_idx = _select_central_sentence(all_sentences, all_embeddings, scope_keywords)
    summary  = _natural_trim(all_sentences[best_idx])
    _save(es, scope_id, summary)
    return summary


def _save(es, scope_id: str, summary: str):
    es.update(
        index=INDEX_SCOPES,
        id=scope_id,
        body={"doc": {
            "scope_summary": summary,
            "updated_at":    datetime.now(timezone.utc).isoformat(),
        }},
    )
    logger.info(f"scope_summary 저장: {scope_id} ({len(summary)}자)")


# ── 배치 처리 ─────────────────────────────────────────────────────

def run_scope_summary_batch():
    """scope_summary 없는 scope 전체를 배치 루프로 처리합니다."""
    try:
        es = get_es()

        total_processed = 0
        batch_num       = 0

        while True:
            batch_num += 1

            res = es.search(
                index=INDEX_SCOPES,
                body={
                    "query": {
                        "bool": {
                            "must_not": {"exists": {"field": "scope_summary"}}
                        }
                    },
                    "_source": ["scopeID"],
                    "sort":    [{"created_at": "asc"}],
                    "size":    BATCH_SIZE,
                },
            )
            hits = res["hits"]["hits"]

            if not hits:
                if batch_num == 1:
                    logger.info("scope_summary 처리할 scope 없음")
                else:
                    logger.info(f"scope_summary 전체 완료 | total={total_processed}")
                break

            logger.info(f"[배치 {batch_num}] scope_summary 처리 시작: {len(hits)}건")

            processed_in_batch = 0
            for hit in hits:
                scope_id = hit["_source"]["scopeID"]
                try:
                    result = generate_scope_summary(es, scope_id)
                    if result is not None:
                        total_processed    += 1
                        processed_in_batch += 1
                except Exception as e:
                    logger.error(f"scope_summary 생성 실패 scopeID={scope_id}: {e}")
                    continue

            es.indices.refresh(index=INDEX_SCOPES)
            logger.info(f"[배치 {batch_num}] 완료 | 누적 processed={total_processed}")

            if processed_in_batch == 0:
                logger.warning("배치에서 처리된 건 없음, 루프 강제 탈출")
                break

        es.close()

    except Exception as e:
        log_pipeline_error(pipeline="scope_summarizer", error=e)
        raise
