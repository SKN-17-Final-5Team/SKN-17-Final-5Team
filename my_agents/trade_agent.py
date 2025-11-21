"""
무역 전문가 Agent

무역사기, CISG, Incoterms, 무역 클레임, 해외인증 정보를 다루는 전문 Agent
"""

import os
from agents import Agent
from tools.search_tool import search_trade_documents


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


# =====================================================================
# Agent 정의 (무역 전문가 Agent)
# =====================================================================

trade_agent = Agent(
    name="Trade Compliance Analyst",
    model="gpt-4o",
    instructions=load_instructions(),  # 외부 파일에서 로드
    tools=[search_trade_documents],
)