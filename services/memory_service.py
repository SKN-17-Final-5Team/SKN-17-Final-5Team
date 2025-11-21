"""
LangGraph 기반 메모리 서비스

단기 메모리: MySQL Checkpointer
장기 메모리: Qdrant (요약 임베딩 저장)
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
import config


class MemoryService:
    """
    메모리 서비스

    - 단기 메모리: MySQL Checkpointer로 자동 관리
    - 장기 메모리: 10턴마다 자동 요약 → Qdrant에 임베딩 저장
    """

    def __init__(
        self,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        qdrant_service=None,
        embedding_service=None
    ):
        """
        Args:
            checkpointer: LangGraph checkpointer (MySQL)
            qdrant_service: Qdrant 서비스 (장기 메모리용)
            embedding_service: 임베딩 서비스
        """
        self.checkpointer = checkpointer
        self.qdrant = qdrant_service
        self.embedder = embedding_service

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",  # 요약용 저비용 모델
            temperature=0.3
        )

        # 백업용 JSON 경로 (Qdrant 장애 시)
        self.long_term_path = os.path.join(
            config.MEMORY_DIR,
            config.LONG_TERM_MEMORY_FILE
        )

    def should_summarize(self, messages: List[Dict], last_summary_index: int) -> bool:
        """
        요약이 필요한지 판단 (10턴 이상 쌓였는지)

        Args:
            messages: 현재 메시지 리스트
            last_summary_index: 마지막 요약 시점

        Returns:
            True if 요약 필요
        """
        current_count = len(messages)
        unsummarized_count = current_count - last_summary_index

        # 10턴(user + assistant = 2메시지) = 20개 메시지
        return unsummarized_count >= 20

    async def summarize_conversations(
        self,
        messages: List[Dict],
        last_summary_index: int
    ) -> str:
        """
        최근 대화 요약

        Args:
            messages: 전체 메시지
            last_summary_index: 마지막 요약 시점

        Returns:
            요약 텍스트
        """
        # 요약할 메시지 추출
        to_summarize = messages[last_summary_index:]

        # 메시지를 텍스트로 변환
        conversation_text = ""
        for msg in to_summarize:
            # LangChain 메시지 객체 처리
            if hasattr(msg, 'type'):
                role = msg.type
            elif hasattr(msg, '__class__'):
                role = msg.__class__.__name__.replace('Message', '').lower()
            else:
                role = "unknown"

            if hasattr(msg, 'content'):
                content = msg.content
            else:
                content = str(msg)

            conversation_text += f"[{role.upper()}]: {content}\n"

        # LLM으로 요약
        summary_prompt = f"""다음 대화 내용을 간결하게 요약해주세요.
중요한 정보(사용자 선호도, 핵심 질문, 주요 답변)만 포함하세요.

대화 내용:
{conversation_text}

요약:"""

        response = await self.llm.ainvoke([HumanMessage(content=summary_prompt)])
        summary = response.content

        return summary

    def add_to_recall_memories(
        self,
        recall_memories: List[Dict[str, str]],
        summary: str
    ) -> List[Dict[str, str]]:
        """
        장기 메모리(recall_memories)에 요약 추가

        Args:
            recall_memories: 기존 recall_memories
            summary: 요약 텍스트

        Returns:
            업데이트된 recall_memories
        """
        memory_item = {
            "timestamp": datetime.now().isoformat(),
            "summary": summary
        }

        recall_memories.append(memory_item)

        # 파일에도 저장 (영속성)
        self._save_recall_memories(recall_memories)

        return recall_memories

    def load_recall_memories(self, session_id: str) -> List[Dict[str, str]]:
        """
        장기 메모리 로드 (Qdrant에서 검색)

        Args:
            session_id: 세션 ID

        Returns:
            recall_memories 리스트
        """
        # Qdrant 사용 가능 시
        if self.qdrant:
            try:
                # 세션 ID로 필터링하여 모든 요약 조회
                results = self.qdrant.scroll_documents(
                    limit=100,
                    filter_conditions={"session_id": session_id, "type": "long_term"}
                )

                # Qdrant 결과를 recall_memories 형식으로 변환
                recall_memories = []
                for doc in results:
                    recall_memories.append({
                        "summary": doc["payload"]["text"],
                        "timestamp": doc["payload"].get("timestamp", "")
                    })

                return recall_memories

            except Exception as e:
                print(f"⚠️ Qdrant 장기 메모리 로드 실패 (JSON 백업 사용): {e}")

        # Qdrant 실패 시 또는 미사용 시 JSON 백업
        session_file = os.path.join(
            config.MEMORY_DIR,
            f"recall_{session_id}.json"
        )

        if os.path.exists(session_file):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ JSON 장기 메모리 로드 실패: {e}")

        return []

    def _save_recall_memories(self, recall_memories: List[Dict[str, str]]):
        """
        장기 메모리를 파일에 저장

        Args:
            recall_memories: 저장할 메모리 리스트
        """
        try:
            os.makedirs(config.MEMORY_DIR, exist_ok=True)

            with open(self.long_term_path, 'w', encoding='utf-8') as f:
                json.dump(recall_memories, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"⚠️ 장기 메모리 저장 실패: {e}")

    def get_recall_context(self, recall_memories: List[Dict[str, str]]) -> str:
        """
        recall_memories를 컨텍스트 문자열로 변환

        Args:
            recall_memories: 메모리 리스트

        Returns:
            컨텍스트 문자열
        """
        if not recall_memories:
            return ""

        context = "=== 이전 대화 요약 ===\n"
        for mem in recall_memories:
            summary = mem.get("summary", "")
            context += f"- {summary}\n"

        return context
