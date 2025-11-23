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


def create_trade_agent(memory_context: str = "") -> Agent:
    """
    무역 전문가 Agent 생성 (메모리 컨텍스트 포함 가능)

    Args:
        memory_context: 대화 히스토리 컨텍스트 (선택)

    Returns:
        Agent 인스턴스
    """
    base_instructions = load_instructions()

    # 메모리 컨텍스트가 있으면 추가
    if memory_context:
        instructions = f"""
        [대화 히스토리]
        {memory_context}

        {base_instructions}
        """
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

