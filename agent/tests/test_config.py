"""
설정 및 클라이언트 테스트

config.py의 설정과 클라이언트 초기화 검증
"""

import pytest
import os


class TestConfig:
    """설정 상수 테스트"""

    def test_환경변수_로드(self):
        """환경 변수가 올바르게 로드되는지 확인"""
        # config.py를 import하면 load_dotenv()가 실행됨
        import config

        # 주요 설정이 존재하는지 확인
        assert hasattr(config, 'COLLECTION_NAME')
        assert hasattr(config, 'EMBEDDING_MODEL')
        assert hasattr(config, 'RERANKER_API_URL')
        assert hasattr(config, 'USE_RERANKER')

    def test_기본_설정_값(self):
        """기본 설정 값이 올바른지 확인"""
        import config

        assert config.COLLECTION_NAME == "trade_collection"
        assert config.EMBEDDING_MODEL == "text-embedding-3-large"
        assert config.USE_RERANKER is True  # 기본값

    def test_클라이언트_초기화(self):
        """클라이언트가 초기화되는지 확인"""
        import config

        assert config.qdrant_client is not None
        assert config.openai_client is not None


class TestEnvironmentVariables:
    """환경 변수 처리 테스트"""

    def test_RERANKER_API_URL_기본값(self, monkeypatch):
        """RERANKER_API_URL 환경변수가 없으면 기본값 사용"""
        # 환경변수 제거
        monkeypatch.delenv("RERANKER_API_URL", raising=False)

        # config 모듈 재로드
        import importlib
        import config
        importlib.reload(config)

        # 기본값이 설정되었는지 확인
        assert "runpod-server" in config.RERANKER_API_URL.lower() or \
               "your-runpod-server" in config.RERANKER_API_URL.lower()
