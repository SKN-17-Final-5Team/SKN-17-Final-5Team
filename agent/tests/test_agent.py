"""
Agent 테스트

my_agents/trade_agent.py의 Agent 로딩 및 설정 검증
"""

import pytest
import os


class TestTradeAgent:
    """무역 전문가 Agent 테스트"""

    def test_instructions_파일_존재(self):
        """프롬프트 파일이 존재하는지 확인"""
        instructions_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "my_agents",
            "prompts",
            "trade_instructions.txt"
        )
        assert os.path.exists(instructions_path), \
            "trade_instructions.txt 파일이 존재하지 않습니다"

    def test_instructions_로드(self):
        """instructions가 파일에서 올바르게 로드되는지 확인"""
        from my_agents.trade_agent import load_instructions

        instructions = load_instructions()

        # 프롬프트 내용 검증
        assert "무역 전문가" in instructions
        assert "데이터 기반 원칙" in instructions
        assert "search_trade_documents" in instructions

    def test_agent_초기화(self):
        """Agent가 올바르게 초기화되는지 확인"""
        from my_agents.trade_agent import trade_agent

        assert trade_agent is not None
        assert trade_agent.name == "Trade Compliance Analyst"
        assert trade_agent.model == "gpt-4o"
        assert len(trade_agent.tools) > 0

    def test_agent_tools(self):
        """Agent에 올바른 tool이 등록되었는지 확인"""
        from my_agents.trade_agent import trade_agent

        # search_trade_documents tool이 포함되어야 함
        tool_names = [tool.__name__ for tool in trade_agent.tools]
        assert "search_trade_documents" in tool_names


class TestLoadInstructions:
    """load_instructions 함수 테스트"""

    def test_파일_읽기(self):
        """파일을 읽어서 문자열 반환"""
        from my_agents.trade_agent import load_instructions

        result = load_instructions()

        assert isinstance(result, str)
        assert len(result) > 0

    def test_존재하지_않는_파일(self):
        """존재하지 않는 파일을 읽으면 에러 발생"""
        from my_agents.trade_agent import load_instructions

        with pytest.raises(FileNotFoundError):
            load_instructions("non_existent_file.txt")

    def test_UTF8_인코딩(self):
        """한국어가 올바르게 읽히는지 확인"""
        from my_agents.trade_agent import load_instructions

        instructions = load_instructions()

        # 한국어가 깨지지 않고 읽혔는지 확인
        assert "무역" in instructions
        assert "데이터" in instructions
        assert "도구" in instructions
