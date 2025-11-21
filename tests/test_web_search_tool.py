"""
웹 검색 Tool 테스트

Tavily API를 사용한 웹 검색 도구의 기본 동작을 테스트합니다.
"""

import pytest
import os
from dotenv import load_dotenv


class TestWebSearchTool:
    """Tavily 웹 검색 도구 테스트"""

    @classmethod
    def setup_class(cls):
        """테스트 클래스 초기화 - .env 로드"""
        load_dotenv()

    def test_tavily_api_key_존재(self):
        """TAVILY_API_KEY가 환경변수에 설정되어 있는지 확인"""
        api_key = os.getenv("TAVILY_API_KEY")
        assert api_key is not None, "TAVILY_API_KEY가 .env에 설정되지 않았습니다"
        assert api_key.startswith("tvly-"), "올바른 Tavily API 키 형식이 아닙니다 (tvly-로 시작해야 함)"

    def test_search_web_import(self):
        """search_web 도구가 정상적으로 import되는지 확인"""
        try:
            from tools.web_search_tool import search_web
            assert search_web is not None
        except ImportError as e:
            pytest.fail(f"search_web import 실패: {e}")

    def test_search_web_도구_속성(self):
        """search_web 도구가 올바른 속성을 가지는지 확인"""
        from tools.web_search_tool import search_web

        # FunctionTool 객체 확인
        assert hasattr(search_web, 'name'), "도구에 name 속성이 없습니다"
        assert hasattr(search_web, 'description'), "도구에 description 속성이 없습니다"

        # 도구 이름 확인
        assert search_web.name == "search_web", f"도구 이름이 'search_web'이 아닙니다: {search_web.name}"

        # 설명 확인 (Tavily 관련 내용이 포함되어야 함)
        assert "Tavily" in search_web.description or "웹 검색" in search_web.description, \
            "도구 설명에 'Tavily' 또는 '웹 검색'이 포함되지 않았습니다"

    def test_tavily_client_초기화(self):
        """Tavily 클라이언트가 정상적으로 초기화되는지 확인"""
        from tools.web_search_tool import tavily_client
        assert tavily_client is not None, "Tavily 클라이언트가 초기화되지 않았습니다"

    def test_search_web_agent_등록(self):
        """search_web 도구가 trade_agent에 등록되었는지 확인"""
        from my_agents.trade_agent import trade_agent

        # trade_agent의 도구 목록 확인
        tool_names = [tool.name if hasattr(tool, 'name') else tool.__name__ for tool in trade_agent.tools]

        assert "search_web" in tool_names, \
            f"search_web이 trade_agent에 등록되지 않았습니다. 등록된 도구: {tool_names}"

    def test_instructions_업데이트(self):
        """instructions에 웹 검색 도구 가이드가 추가되었는지 확인"""
        from my_agents.trade_agent import load_instructions

        instructions = load_instructions()

        # 웹 검색 관련 내용이 포함되었는지 확인
        assert "search_web" in instructions, "instructions에 search_web이 포함되지 않았습니다"
        assert "웹 검색" in instructions or "최신 정보" in instructions, \
            "instructions에 웹 검색 관련 설명이 포함되지 않았습니다"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
