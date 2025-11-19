"""
Pytest 설정 및 공통 Fixture

모든 테스트에서 사용하는 공통 fixture와 설정
"""

import pytest
from unittest.mock import Mock, MagicMock


@pytest.fixture
def sample_qdrant_points():
    """Qdrant 검색 결과 샘플 데이터"""
    mock_points = []

    for i in range(5):
        point = Mock()
        point.score = 0.9 - (i * 0.1)
        point.payload = {
            "text": f"샘플 문서 내용 {i+1}. 무역 관련 정보입니다.",
            "data_source": "fraud",
            "document_name": f"문서_{i+1}.pdf",
            "article": f"제{i+1}조"
        }
        mock_points.append(point)

    return mock_points


@pytest.fixture
def sample_rerank_response():
    """Reranker API 응답 샘플"""
    from models.reranker import RerankResponse, RerankResult

    results = [
        RerankResult(index=2, score=0.95, document="샘플 문서 3"),
        RerankResult(index=0, score=0.92, document="샘플 문서 1"),
        RerankResult(index=4, score=0.88, document="샘플 문서 5"),
    ]

    return RerankResponse(
        results=results,
        query="무역 사기 예방",
        total_documents=5
    )


@pytest.fixture
def mock_openai_client():
    """OpenAI 클라이언트 Mock"""
    client = MagicMock()

    # Embedding 응답 mock
    embedding_response = Mock()
    embedding_response.data = [Mock(embedding=[0.1] * 1536)]
    client.embeddings.create.return_value = embedding_response

    return client


@pytest.fixture
def mock_qdrant_client():
    """Qdrant 클라이언트 Mock"""
    client = MagicMock()
    return client
