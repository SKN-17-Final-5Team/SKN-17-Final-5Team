"""
RAG 시스템 메인 실행 파일

Reranker API를 활용한 고도화된 RAG 시스템
- 초기 검색: Qdrant Vector DB에서 limit개 문서 검색
- Reranking: RunPod 서버의 Reranker 모델로 재정렬
- 최종 전달: 상위 top_k개 문서만 Agent에게 전달
"""

import asyncio
from agents import Runner  # OpenAI Agents SDK
from my_agents.trade_agent import create_trade_agent
from utils import dedup_consecutive_lines
import config
from memory_modules.repository import EDARepository


async def main():
    """
    RAG Agent 실행 함수 (메모리 시스템 통합)

    사용자 입력을 받아 무역 전문가 Agent를 실행하고 결과를 출력합니다.
    이전 대화 내용을 기억하여 연속적인 대화가 가능합니다.
    """
    # 메모리 시스템 초기화
    memory = EDARepository()

    # 세션 ID 고정 (CLI 버전에서는 단일 세션 사용)
    session_id = "cli_chat_001"

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

    # 이전 대화 컨텍스트 로드
    memory_context = memory.get_gen_context(session_id, limit=10)

    # 메모리 통계 출력
    stats = memory.get_stats(session_id)
    if stats["total_messages"] > 0:
        print(f"이전 대화: {stats['total_messages']}개 메시지 (요약: {stats['summary_count']}개)\n")

    # 사용자 질문 입력 (기본값: "무역 사기를 방지하는 방법은?")
    question = input("질문: ").strip() or "무역 사기를 방지하는 방법은?"

    print(f"\n{'='*60}\n")

    # 메모리 컨텍스트와 함께 Agent 생성
    trade_agent = create_trade_agent(memory_context=memory_context)

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

    # 대화 턴 저장
    memory.save_gen_turn(session_id, question, cleaned)

    # 요약 생성을 위해 잠시 대기 (백그라운드 스레드가 완료될 시간 부여)
    import time
    time.sleep(3)  # 요약 생성에 보통 1-2초 소요


if __name__ == "__main__":
    asyncio.run(main())