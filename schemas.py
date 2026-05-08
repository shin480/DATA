from typing import List
from pydantic import BaseModel


# 이용약관 구조 정의
class TermsRequest(BaseModel):
    requiredChecks: bool  # 또는 str, 실제 데이터에 맞게 설정
    optionalChecksList: List[str] # 리스트 형태임을 명시


class VoteRequest(BaseModel):
    article_id: str
    vote_type: str