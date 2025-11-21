"""
LangGraph 기반 RAG 시스템 메인 실행 파일
"""

import asyncio
import os
import uuid
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_core.messages import HumanMessage

from services.graph_state import create_initial_state
from services.graph_workflow import TradeAgentWorkflow
from utils import dedup_consecutive_lines
import config


async def main():
    """LangGraph 기반 Agent 실행"""

    # Reranker 설정
    reranker_choice = input("Reranker 사용? (y/n, 기본값: y): ").strip().lower()
    config.USE_RERANKER = reranker_choice not in ['n', 'no']

    # SQL 경로 설정
    db_path = os.path.join(config.MEMORY_DIR, "checkpoints.db")
    os.makedirs(config.MEMORY_DIR, exist_ok=True)

    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        # Workflow 생성
        workflow = TradeAgentWorkflow(checkpointer)

        # 세션 ID 생성
        session_id = input("세션 ID (Enter: 자동 생성): ").strip() or f"session_{uuid.uuid4().hex[:8]}"
        print(f"세션 ID: {session_id}\n")

        # 장기 메모리 로드
        recall_memories = workflow.memory_service.load_recall_memories(session_id)
        if recall_memories:
            print(f"장기 메모리 {len(recall_memories)}개 로드됨\n")

        # Thread 설정
        thread_config = {"configurable": {"thread_id": session_id}}

        print("대화 시작 (종료: exit/quit)\n" + "=" * 60)

        # 대화 루프
        while True:
            question = input("\n질문: ").strip()

            if not question:
                continue

            if question.lower() in ['exit', 'quit', '종료']:
                break

            try:
                # 현재 State 가져오기
                current_state = await workflow.graph.aget_state(thread_config)

                # State 초기화 또는 재사용
                if not current_state or not current_state.values:
                    initial_state = create_initial_state(
                        flow_type="GEN_CHAT",
                        gen_chat_id=session_id,
                        trade_id=session_id
                    )
                    initial_state["recall_memories"] = recall_memories
                else:
                    initial_state = current_state.values

                # 사용자 메시지 추가
                initial_state["messages"].append(HumanMessage(content=question))

                # Workflow 실행
                result = await workflow.graph.ainvoke(initial_state, config=thread_config)

                # 응답 출력
                if result and result.get("messages"):
                    last_message = result["messages"][-1]
                    answer = last_message.content if hasattr(last_message, 'content') else str(last_message)
                    cleaned_answer = dedup_consecutive_lines(answer)

                    print("\n" + "=" * 60)
                    print("답변:")
                    print("-" * 60)
                    print(cleaned_answer)
                    print("=" * 60)

                    # recall_memories 업데이트
                    recall_memories = result.get("recall_memories", [])

            except Exception as e:
                print(f"\n오류: {e}")
                import traceback
                traceback.print_exc()

        # 종료
        print(f"\n대화 저장됨 (세션: {session_id}, 메모리: {len(recall_memories)}개)")


if __name__ == "__main__":
    asyncio.run(main())
