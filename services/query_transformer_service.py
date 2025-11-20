"""
Query Transformer 서비스

사용자 쿼리를 무역 문서 검색에 최적화된 형태로 변환
- Rewriting: 검색에 적합한 용어로 개선
- Decomposition: 복합 질문을 개별 서브쿼리로 분해
"""

import json
from typing import Dict, Any

from config import openai_client
from models.query_transformer import QueryTransformResult


# LLM 프롬프트 상수
QUERY_TRANSFORM_PROMPT = """당신은 무역 문서 검색 시스템의 쿼리 최적화 전문가입니다.

사용자의 질문을 분석하여 다음 두 가지 작업을 수행하세요:

1. **Query Rewriting (쿼리 개선)**
   - 무역 전문 용어로 정확하게 변환
   - 벡터 검색에 적합한 형태로 구체화
   - 불필요한 조사나 구어체 제거
   - 핵심 키워드 보존 및 강화

2. **Query Decomposition (쿼리 분해)**
   - 복합 질문인지 판단
   - 복합 질문이면 개별 서브쿼리로 분해
   - 단순 질문이면 분해하지 않음 (sub_queries를 null로 설정)

**복합 질문 판단 기준:**
- "A와 B의 차이는?" → 복합 (A, B 각각 검색 필요)
- "수출과 수입 절차" → 복합 (수출, 수입 개별 검색 필요)
- "FOB와 CIF 비교" → 복합 (FOB, CIF 각각 검색 필요)
- "수출 절차는?" → 단순 (분해 불필요)
- "CISG란?" → 단순 (분해 불필요)

**응답 형식 (JSON):**
{
    "rewritten_query": "개선된 쿼리",
    "sub_queries": ["서브쿼리1", "서브쿼리2"] 또는 null,
    "reasoning": "변환 근거 설명 (선택사항)"
}

**예시 1 - 복합 질문:**
입력: "수출과 수입의 차이점을 알려줘"
출력:
{
    "rewritten_query": "수출과 수입의 절차 및 규정 차이점",
    "sub_queries": [
        "수출 절차 및 규정 요건",
        "수입 절차 및 규정 요건"
    ],
    "reasoning": "수출과 수입을 비교하는 복합 질문이므로 각각 개별 검색 후 통합"
}

**예시 2 - 단순 질문:**
입력: "무역 사기 방지 방법 알려줘"
출력:
{
    "rewritten_query": "무역 사기 예방 및 대응 방법",
    "sub_queries": null,
    "reasoning": "단일 주제에 대한 질문이므로 분해 불필요"
}

**예시 3 - 복합 질문 (3개 이상):**
입력: "FOB, CIF, EXW 인코텀즈 비교해줘"
출력:
{
    "rewritten_query": "FOB, CIF, EXW 인코텀즈 조건 비교",
    "sub_queries": [
        "FOB 인코텀즈 조건 및 책임범위",
        "CIF 인코텀즈 조건 및 책임범위",
        "EXW 인코텀즈 조건 및 책임범위"
    ],
    "reasoning": "3개 인코텀즈 조건을 비교하는 복합 질문이므로 각각 개별 검색"
}

이제 다음 사용자 질문을 변환하세요:
"""


async def rewrite_and_decompose_query(
    query: str,
    model: str = "gpt-4o-mini"
) -> QueryTransformResult:
    """
    사용자 쿼리를 리라이팅하고 필요시 디컴포지션 수행

    Args:
        query: 원본 사용자 질문
        model: 사용할 LLM 모델 (기본값: gpt-4o-mini, 빠르고 저렴)

    Returns:
        QueryTransformResult: 변환된 쿼리 결과
            - rewritten_query: 개선된 단일 쿼리
            - sub_queries: 서브쿼리 리스트 (복합 질문인 경우) 또는 None
            - reasoning: 변환 근거 (선택사항)

    Raises:
        Exception: LLM 호출 실패 또는 JSON 파싱 실패 시
    """
    print(f"\n🔄 쿼리 변환 중: '{query}'")

    try:
        # LLM 호출 (JSON 모드 사용)
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": QUERY_TRANSFORM_PROMPT
                },
                {
                    "role": "user",
                    "content": query
                }
            ],
            response_format={"type": "json_object"},  # JSON 응답 강제
            temperature=0.3  # 일관성 있는 결과를 위해 낮은 temperature
        )

        # JSON 파싱
        result_json = json.loads(response.choices[0].message.content)

        # Pydantic 모델로 변환
        result = QueryTransformResult(**result_json)

        # 로그 출력
        print(f"✓ 개선된 쿼리: '{result.rewritten_query}'")
        if result.sub_queries and len(result.sub_queries) > 0:
            print(f"✓ 복합 질문 감지 → {len(result.sub_queries)}개 서브쿼리로 분해:")
            for i, sq in enumerate(result.sub_queries, 1):
                print(f"   {i}. {sq}")
        else:
            print("✓ 단순 질문 → 분해 없이 단일 검색 수행")

        if result.reasoning:
            print(f"  (근거: {result.reasoning})")

        print()  # 빈 줄

        return result

    except json.JSONDecodeError as e:
        print(f"⚠️ JSON 파싱 실패: {e}")
        print(f"⚠️ 원본 쿼리를 그대로 사용합니다.\n")
        # Fallback: 원본 쿼리 그대로 반환
        return QueryTransformResult(
            rewritten_query=query,
            sub_queries=None,
            reasoning="JSON 파싱 실패로 원본 쿼리 사용"
        )

    except Exception as e:
        print(f"⚠️ 쿼리 변환 실패: {e}")
        print(f"⚠️ 원본 쿼리를 그대로 사용합니다.\n")
        # Fallback: 원본 쿼리 그대로 반환
        return QueryTransformResult(
            rewritten_query=query,
            sub_queries=None,
            reasoning=f"변환 실패: {str(e)}"
        )
