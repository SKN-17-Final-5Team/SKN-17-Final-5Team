"""
LangGraph Workflow 정의

GEN_CHAT(일반 대화) vs TRADE_FLOW(문서 플로우) 분기 처리
"""

from typing import Literal, Optional
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver
import os

from services.graph_state import AgentState
from services.memory_service import MemoryService
from tools.search_tool import search_trade_documents
import config


class TradeAgentWorkflow:
    """
    무역 Agent의 LangGraph Workflow
    """

    def __init__(self, checkpointer: Optional[BaseCheckpointSaver] = None):
        """
        Args:
            checkpointer: LangGraph checkpointer (MemorySaver, SqliteSaver 등)
        """
        self.checkpointer = checkpointer
        self.memory_service = MemoryService(checkpointer)

        # LLM (도구는 tools_node에서 직접 호출)
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7
        )

        # 무역 에이전트 프롬프트 로드
        self.trade_instructions = self._load_trade_instructions()

        # Workflow 생성
        self.graph = self._create_graph()

    def _load_trade_instructions(self) -> str:
        """
        무역 에이전트 프롬프트 파일 로드

        Returns:
            프롬프트 내용
        """
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "my_agents", "prompts", "trade_instructions.txt"
        )

        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            print(f"⚠️ 프롬프트 파일을 찾을 수 없습니다: {prompt_path}")
            return "당신은 무역 전문가 AI 어시스턴트입니다."

    def _create_graph(self) -> StateGraph:
        """
        LangGraph workflow 생성

        Returns:
            Compiled StateGraph
        """
        workflow = StateGraph(AgentState)

        # 노드 추가
        workflow.add_node("agent", self.agent_node)
        workflow.add_node("tools", self.tools_node)
        workflow.add_node("check_summary", self.check_summary_node)

        # 엣지 정의
        workflow.set_entry_point("agent")

        # agent에서 분기 (tool 호출 여부 판단)
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "tools": "tools",
                "end": "check_summary"
            }
        )

        # tools → agent (tool 결과를 agent에게 전달)
        workflow.add_edge("tools", "agent")

        # check_summary → END
        workflow.add_edge("check_summary", END)

        # Compile with checkpointer
        return workflow.compile(checkpointer=self.checkpointer)

    # =====================================================================
    # 노드 함수들
    # =====================================================================

    async def agent_node(self, state: AgentState) -> AgentState:
        """
        Agent 노드: 사용자 질문 분석 및 문서 검색 필요 여부 판단

        Args:
            state: 현재 상태

        Returns:
            업데이트된 상태
        """
        # 컨텍스트 구성
        context_parts = []

        # 장기 메모리 (recall_memories)
        recall_context = self.memory_service.get_recall_context(
            state["recall_memories"]
        )
        if recall_context:
            context_parts.append(recall_context)

        # System message 구성 (trade_instructions.txt 사용)
        system_content = self.trade_instructions

        if context_parts:
            system_content += "\n\n" + "\n".join(context_parts)

        # 검색 결과가 있으면 추가
        if state.get("retrieved_docs"):
            docs_content = "\n\n".join([doc.get("content", "") for doc in state["retrieved_docs"]])
            system_content += f"\n\n=== 검색된 문서 ===\n{docs_content}"

        # LLM 호출 (SystemMessage 사용, 비동기)
        messages = [SystemMessage(content=system_content)]
        messages.extend(state["messages"])

        response = await self.llm.ainvoke(messages)

        # 응답을 메시지에 추가
        state["messages"].append(response)

        return state

    def should_continue(self, state: AgentState) -> Literal["tools", "end"]:
        """
        다음 노드 결정: 문서 검색이 필요하면 tools로, 아니면 종료

        Args:
            state: 현재 상태

        Returns:
            다음 노드 이름
        """
        # 마지막 사용자 메시지 확인
        last_user_message = None
        for msg in reversed(state["messages"]):
            if hasattr(msg, "type") and msg.type == "human":
                last_user_message = msg
                break

        # 이미 검색을 했다면 종료
        if state.get("retrieved_docs"):
            return "end"

        # 간단한 키워드 기반 판단 (실제로는 더 정교한 로직 가능)
        if last_user_message:
            content = last_user_message.content.lower()
            # 무역 관련 키워드가 있으면 문서 검색
            keywords = ["무역", "수출", "수입", "관세", "통관", "인증", "클레임", "사기", "cisg", "incoterms"]
            if any(keyword in content for keyword in keywords):
                return "tools"

        return "end"

    async def tools_node(self, state: AgentState) -> AgentState:
        """
        Tools 노드: 무역 문서 검색 실행

        Args:
            state: 현재 상태

        Returns:
            업데이트된 상태
        """
        # 마지막 사용자 메시지 가져오기
        last_user_message = None
        for msg in reversed(state["messages"]):
            if hasattr(msg, "type") and msg.type == "human":
                last_user_message = msg
                break

        if last_user_message:
            query = last_user_message.content

            # search_trade_documents 호출 (비동기)
            result = await search_trade_documents(query)

            # 검색 결과를 state에 저장
            state["retrieved_docs"] = [{"content": result}]

        return state

    async def check_summary_node(self, state: AgentState) -> AgentState:
        """
        요약 체크 노드: 10턴 이상 쌓이면 요약

        Args:
            state: 현재 상태

        Returns:
            업데이트된 상태
        """
        # 요약 필요 여부 체크
        if self.memory_service.should_summarize(
            state["messages"],
            state["last_summary_index"]
        ):
            print("\n대화 요약 중...\n")

            # 요약 생성
            summary = await self.memory_service.summarize_conversations(
                state["messages"],
                state["last_summary_index"]
            )

            # recall_memories에 추가
            state["recall_memories"] = self.memory_service.add_to_recall_memories(
                state["recall_memories"],
                summary
            )

            # 마지막 요약 시점 업데이트
            state["last_summary_index"] = len(state["messages"])

            print(f"요약 완료: {summary[:100]}...\n")

        return state
