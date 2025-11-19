"""
Reranker Pydantic 모델 테스트

models/reranker.py의 Pydantic 모델 검증
"""

import pytest
from pydantic import ValidationError
from models.reranker import RerankRequest, RerankResult, RerankResponse


class TestRerankRequest:
    """RerankRequest 모델 테스트"""

    def test_유효한_요청_생성(self):
        """정상적인 요청 객체 생성"""
        request = RerankRequest(
            query="무역 사기 예방",
            documents=["문서 1", "문서 2", "문서 3"],
            top_k=3,
            return_documents=True
        )

        assert request.query == "무역 사기 예방"
        assert len(request.documents) == 3
        assert request.top_k == 3
        assert request.return_documents is True

    def test_기본값_설정(self):
        """기본값이 올바르게 설정되는지 확인"""
        request = RerankRequest(
            query="테스트",
            documents=["문서 1"]
        )

        assert request.top_k == 5  # 기본값
        assert request.return_documents is True  # 기본값

    def test_top_k_범위_검증(self):
        """top_k가 범위를 벗어나면 에러 발생"""
        with pytest.raises(ValidationError):
            RerankRequest(
                query="테스트",
                documents=["문서 1"],
                top_k=0  # 최소값 1 미만
            )

        with pytest.raises(ValidationError):
            RerankRequest(
                query="테스트",
                documents=["문서 1"],
                top_k=101  # 최대값 100 초과
            )

    def test_필수_필드_누락(self):
        """필수 필드가 없으면 에러 발생"""
        with pytest.raises(ValidationError):
            RerankRequest(documents=["문서 1"])  # query 누락

        with pytest.raises(ValidationError):
            RerankRequest(query="테스트")  # documents 누락


class TestRerankResult:
    """RerankResult 모델 테스트"""

    def test_유효한_결과_생성(self):
        """정상적인 결과 객체 생성"""
        result = RerankResult(
            index=0,
            score=0.95,
            document="샘플 문서"
        )

        assert result.index == 0
        assert result.score == 0.95
        assert result.document == "샘플 문서"

    def test_문서_내용_선택적(self):
        """document 필드는 선택적"""
        result = RerankResult(
            index=1,
            score=0.88
        )

        assert result.index == 1
        assert result.score == 0.88
        assert result.document is None


class TestRerankResponse:
    """RerankResponse 모델 테스트"""

    def test_유효한_응답_생성(self):
        """정상적인 응답 객체 생성"""
        results = [
            RerankResult(index=0, score=0.95),
            RerankResult(index=2, score=0.90),
        ]

        response = RerankResponse(
            results=results,
            query="무역 사기",
            total_documents=10
        )

        assert len(response.results) == 2
        assert response.query == "무역 사기"
        assert response.total_documents == 10

    def test_JSON_직렬화(self):
        """JSON으로 직렬화 가능"""
        result = RerankResult(index=0, score=0.95, document="테스트")
        response = RerankResponse(
            results=[result],
            query="테스트",
            total_documents=1
        )

        json_data = response.model_dump()

        assert json_data["query"] == "테스트"
        assert json_data["total_documents"] == 1
        assert len(json_data["results"]) == 1

    def test_JSON_역직렬화(self):
        """JSON에서 객체로 역직렬화 가능"""
        json_data = {
            "results": [
                {"index": 0, "score": 0.95, "document": "문서 1"}
            ],
            "query": "테스트",
            "total_documents": 5
        }

        response = RerankResponse(**json_data)

        assert response.query == "테스트"
        assert response.total_documents == 5
        assert len(response.results) == 1
        assert response.results[0].score == 0.95
