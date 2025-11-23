"""
RAG 시스템 메인 실행 파일

Reranker API를 활용한 고도화된 RAG 시스템
- 초기 검색: Qdrant Vector DB에서 limit개 문서 검색
- Reranking: RunPod 서버의 Reranker 모델로 재정렬
- 최종 전달: 상위 top_k개 문서만 Agent에게 전달
"""

import asyncio
from agents import Runner  # OpenAI Agents SDK
from my_agents.trade_agent import trade_agent
from utils import dedup_consecutive_lines
import config


async def main():
    """
    RAG Agent 실행 함수

    사용자 입력을 받아 무역 전문가 Agent를 실행하고 결과를 출력합니다.
    """
    # Reranker 사용 여부 선택
    print("=" * 60)
    print("RAG 시스템 설정")
    print("=" * 60)
    reranker_choice = input("Reranker를 사용하시겠습니까? (y/n, 기본값: y): ").strip().lower()

    if reranker_choice in ['n', 'no']:
        config.USE_RERANKER = False
        print("✓ Reranker 미사용 모드로 실행합니다.\n")
    else:
        config.USE_RERANKER = True
        print("✓ Reranker 사용 모드로 실행합니다.")

        # Reranker 사용 시 개별 Rerank 방식 선택
        per_query_choice = input("복합 질문 시 개별 Rerank를 사용하시겠습니까? (y/n, 기본값: y): ").strip().lower()

        if per_query_choice in ['n', 'no']:
            config.USE_PER_QUERY_RERANK = False
            print("✓ 통합 Rerank 방식으로 실행합니다.\n")
        else:
            config.USE_PER_QUERY_RERANK = True
            print("✓ 개별 Rerank 방식으로 실행합니다. (모든 토픽 균형 보장)\n")

    # 사용자 질문 입력 (기본값: "무역 사기를 방지하는 방법은?")
    question = input("질문: ").strip() or "무역 사기를 방지하는 방법은?"

    print(f"\n{'='*60}\n")

    # Agent 실행
    print("🤖 Agent 실행 중...\n")
    result = await Runner.run(trade_agent, input=question)

    # 연속 중복 라인 제거
    cleaned = dedup_consecutive_lines(result.final_output)

    # 최종 답변 출력
    print("="*60)
    print("\n최종 답변:")
    print("-" * 60)
    print(cleaned)
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())