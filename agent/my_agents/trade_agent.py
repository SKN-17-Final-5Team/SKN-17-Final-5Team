"""
무역 전문가 Agent

무역사기, CISG, Incoterms, 무역 클레임, 해외인증 정보를 다루는 전문 Agent
"""

import os
from agents import Agent
from tools.search_tool import search_trade_documents
from tools.web_search_tool import search_web
from tools.document_generation_tool import generate_trade_document


def load_instructions(filename: str = "trade_instructions.txt") -> str:
    """
    프롬프트 파일을 읽어서 instructions 반환

    Args:
        filename: 프롬프트 파일명 (agents/prompts/ 디렉토리 내)

    Returns:
        파일 내용 (프롬프트 문자열)
    """
    current_dir = os.path.dirname(__file__)
    prompts_dir = os.path.join(current_dir, "prompts")
    file_path = os.path.join(prompts_dir, filename)

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def create_trade_agent(memory_context: str = "", previous_messages: list = None) -> Agent:
    """
    무역 전문가 Agent 생성 (메모리 컨텍스트 및 이전 대화 포함 가능)

    Args:
        memory_context: 대화 히스토리 컨텍스트 (요약, 문서 정보)
        previous_messages: 최근 대화 히스토리 [{"role": "user", "content": "..."}, ...]

    Returns:
        Agent 인스턴스
    """
    base_instructions = load_instructions()

    # 대화 히스토리 구성
    history_parts = []

    if memory_context:
        history_parts.append(f"[참고 정보]\n{memory_context}\n")

    if previous_messages:
        history_parts.append("[최근 대화]")
        for msg in previous_messages:
            role_label = "사용자" if msg["role"] == "user" else "어시스턴트"
            history_parts.append(f"{role_label}: {msg['content']}")
        history_parts.append("")

    if history_parts:
        history_text = "\n".join(history_parts)
        instructions = f"{history_text}\n{base_instructions}"
    else:
        instructions = base_instructions

    return Agent(
        name="Trade Compliance Analyst",
        model="gpt-4o",
        instructions=instructions,
        tools=[search_trade_documents, search_web, generate_trade_document],
    )


# =====================================================================
# 기본 Agent 인스턴스 (메모리 없는 버전)
# =====================================================================

trade_agent = Agent(
    name="Trade Compliance Analyst",
    model="gpt-4o",
    instructions=load_instructions(),  # 외부 파일에서 로드
    tools=[search_trade_documents, search_web, generate_trade_document],
)

