"""
Step 3 리팩토링 테스트

단순 질문과 복합 질문 모두 동일한 경로로 처리되는지 확인
"""

import asyncio
from agents import Runner
from my_agents.trade_agent import trade_agent
import config

# Reranker 설정
config.USE_RERANKER = True
config.USE_PER_QUERY_RERANK = True


async def test_query(question: str):
    """단일 쿼리 테스트"""
    print(f"\n{'='*80}")
    print(f"테스트 질문: {question}")
    print(f"{'='*80}\n")

    result = await Runner.run(trade_agent, input=question)

    print(f"\n{'='*80}")
    print("최종 답변:")
    print(f"{'-'*80}")
    print(result.final_output[:500] + "..." if len(result.final_output) > 500 else result.final_output)
    print(f"{'='*80}\n")


async def main():
    """테스트 실행"""

    # 테스트 1: 단순 질문 (분해 안 됨)
    await test_query("인코텀즈 선택 시 고려해야 할 사항")

    # 테스트 2: 복합 질문 (분해됨)
    await test_query("FOB 조건으로 미국과 중국에 수출할 때 계약서 주의사항")


if __name__ == "__main__":
    asyncio.run(main())
