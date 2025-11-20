"""
Query Transformer 모델 정의 (Pydantic)

쿼리 리라이팅 및 디컴포지션을 위한 데이터 모델
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class QueryTransformResult(BaseModel):
    """
    쿼리 변환 결과 모델

    LLM이 반환하는 리라이팅 + 디컴포지션 결과
    """
    rewritten_query: str = Field(
        ...,
        description="무역 문서 검색에 최적화된 개선된 쿼리"
    )
    sub_queries: Optional[List[str]] = Field(
        default=None,
        description="복합 질문인 경우 분해된 서브쿼리 리스트 (단순 질문이면 None)"
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="변환 과정에 대한 설명 (디버깅용, 선택사항)"
    )
