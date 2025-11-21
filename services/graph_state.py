"""
LangGraph State 정의

단기 메모리(messages)와 장기 메모리(recall_memories)를 관리하는 State 스키마
"""

from typing import Annotated, List, Dict, Optional, Literal
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Agent 상태 정의

    단기 메모리:
        - messages: 현재 세션의 대화 내용 (LangGraph의 add_messages로 자동 관리)

    장기 메모리:
        - recall_memories: 요약된 중요 정보 (수동 관리)

    플로우 관리:
        - flow_type: GEN_CHAT(일반 대화) 또는 TRADE_FLOW(무역 문서 플로우)
        - session_id: 세션 ID (thread_id로 사용)
        - last_summary_index: 마지막 요약 시점 (메시지 인덱스)
    """

    # ===== 단기 메모리 (messages) =====
    # LangGraph의 add_messages로 자동 관리됨
    messages: Annotated[List[Dict], add_messages]

    # ===== 장기 메모리 (recall_memories) =====
    # 요약된 중요 정보를 저장 (10턴마다 자동 요약)
    recall_memories: List[Dict[str, str]]  # [{"timestamp": "...", "summary": "..."}]

    # ===== 플로우 관리 =====
    flow_type: Literal["GEN_CHAT", "TRADE_FLOW"]  # 일반 대화 vs 문서 플로우
    gen_chat_id: str  # 일반 대화용 세션 ID
    trade_id: str  # 무역 문서 플로우용 세션 ID
    last_summary_index: int  # 마지막 요약한 메시지 인덱스

    # ===== RAG 관련 =====
    use_documents: bool  # 문서 검색 사용 여부
    retrieved_docs: Optional[List[Dict]]  # 검색된 문서들


def create_initial_state(
    flow_type: Literal["GEN_CHAT", "TRADE_FLOW"],
    gen_chat_id: str,
    trade_id: str,
    user_profile: Optional[Dict[str, str]] = None
) -> AgentState:
    """
    초기 State 생성

    Args:
        flow_type: 플로우 타입 (GEN_CHAT 또는 TRADE_FLOW)
        gen_chat_id: 일반 대화용 세션 ID
        trade_id: 무역 문서 플로우용 세션 ID

    Returns:
        초기화된 AgentState
    """
    return AgentState(
        messages=[],
        recall_memories=[],
        flow_type=flow_type,
        gen_chat_id=gen_chat_id,
        trade_id=trade_id,
        last_summary_index=0,
        use_documents=False,
        retrieved_docs=None
    )
