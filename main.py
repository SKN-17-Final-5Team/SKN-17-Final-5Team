"""
RAG 시스템 메인 실행 파일

Reranker API를 활용한 고도화된 RAG 시스템
- 초기 검색: Qdrant Vector DB에서 limit개 문서 검색
- Reranking: RunPod 서버의 Reranker 모델로 재정렬
- 최종 전달: 상위 top_k개 문서만 Agent에게 전달
"""

import asyncio
from agents import Runner  # OpenAI Agents SDK
from utils import dedup_consecutive_lines
import config


async def main():
    """
    RAG Agent 실행 함수 (대화형 루프)

    사용자 입력을 받아 무역 전문가 Agent를 실행하고 결과를 출력합니다.
    메모리 컨텍스트를 유지하여 연속적인 대화가 가능합니다.
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

    # 대화 히스토리 저장용 리스트
    history = []

    print(f"\n{'='*60}")
    print("🤖 무역 전문가 Agent와 대화를 시작합니다. (종료하려면 'exit' 또는 'quit' 입력)")
    print(f"{'='*60}\n")

    while True:
        # 사용자 질문 입력
        question = input("\n질문 (종료: exit): ").strip()
        
        if not question:
            continue
            
        if question.lower() in ['exit', 'quit', 'q']:
            print("\n대화를 종료합니다. 감사합니다!")
            break

        # 대화 히스토리를 기반으로 컨텍스트 생성
        memory_context = ""
        if history:
            memory_context = "\n".join([f"User: {q}\nAgent: {a}" for q, a in history[-5:]]) # 최근 5턴만 유지

        # Agent 생성 (컨텍스트 주입)
        # my_agents.trade_agent.create_trade_agent 함수 사용
        from my_agents.trade_agent import create_trade_agent
        current_agent = create_trade_agent(memory_context=memory_context)

        print(f"\n{'='*60}")
        print("🤖 Agent 실행 중...")
        print(f"{'='*60}\n")

        try:
            result = await Runner.run(current_agent, input=question)
            
            # 연속 중복 라인 제거
            cleaned = dedup_consecutive_lines(result.final_output)

            # 최종 답변 출력
            print("="*60)
            print("최종 답변:")
            print("-" * 60)
            print(cleaned)
            print("="*60)

            # 히스토리에 추가
            history.append((question, cleaned))

        except Exception as e:
            print(f"\n⚠️ 에러 발생: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())