"""
Query Transformation 기능 테스트 스크립트

실제 Qdrant 검색 없이 Query Rewriting + Decomposition만 테스트
"""

import asyncio
from services.query_transformer_service import rewrite_and_decompose_query


async def test_query_transformation():
    """다양한 유형의 쿼리로 변환 테스트"""

    test_queries = [
        "무역 사기 방지 방법 알려줘",  # 단순 질문
        "수출과 수입의 차이점을 알려줘",  # 복합 질문
        "FOB, CIF, EXW 인코텀즈 비교해줘",  # 3개 이상 비교
        "CISG란?",  # 단순 질문
        "수출 절차와 수입 절차 비교",  # 복합 질문
    ]

    print("="*60)
    print("Query Transformation 테스트")
    print("="*60)

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"테스트 {i}/{len(test_queries)}")
        print(f"{'='*60}")
        print(f"원본 쿼리: {query}")

        try:
            result = await rewrite_and_decompose_query(query)

            print("\n결과:")
            print(f"  Rewritten Query: {result.rewritten_query}")
            print(f"  Sub Queries: {result.sub_queries}")
            if result.reasoning:
                print(f"  Reasoning: {result.reasoning}")

            # 판단 확인
            if result.sub_queries:
                print(f"\n✅ 복합 질문으로 판단 → {len(result.sub_queries)}개 서브쿼리 생성")
            else:
                print(f"\n✅ 단순 질문으로 판단 → 단일 검색 수행")

        except Exception as e:
            print(f"\n❌ 오류 발생: {e}")

        print()

    print("="*60)
    print("테스트 완료!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(test_query_transformation())
