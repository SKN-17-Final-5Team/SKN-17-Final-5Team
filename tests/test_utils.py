"""
유틸리티 함수 테스트

utils.py의 함수들에 대한 단위 테스트
"""

import pytest
from utils import dedup_consecutive_lines, print_retrieved_documents


class TestDedupConsecutiveLines:
    """dedup_consecutive_lines 함수 테스트"""

    def test_중복_라인_제거(self):
        """연속된 중복 라인이 제거되는지 확인"""
        text = "첫 번째 라인\n두 번째 라인\n두 번째 라인\n세 번째 라인"
        result = dedup_consecutive_lines(text)
        expected = "첫 번째 라인\n두 번째 라인\n세 번째 라인"
        assert result == expected

    def test_중복_없는_텍스트(self):
        """중복이 없는 텍스트는 그대로 반환"""
        text = "첫 번째\n두 번째\n세 번째"
        result = dedup_consecutive_lines(text)
        assert result == text

    def test_빈_문자열(self):
        """빈 문자열 처리"""
        result = dedup_consecutive_lines("")
        assert result == ""

    def test_공백만_있는_라인(self):
        """공백만 있는 라인 처리"""
        text = "첫 번째\n   \n   \n두 번째"
        result = dedup_consecutive_lines(text)
        expected = "첫 번째\n   \n두 번째"
        assert result == expected

    def test_비연속_중복_유지(self):
        """비연속 중복은 유지되어야 함"""
        text = "A\nB\nA"
        result = dedup_consecutive_lines(text)
        assert result == text


class TestPrintRetrievedDocuments:
    """print_retrieved_documents 함수 테스트"""

    def test_문서_출력(self, sample_qdrant_points, capsys):
        """문서가 올바르게 출력되는지 확인"""
        print_retrieved_documents(sample_qdrant_points, n=3)

        captured = capsys.readouterr()
        assert "📄 검색된 문서" in captured.out
        assert "총 5개 중 3개 표시" in captured.out
        assert "문서 1:" in captured.out
        assert "문서 2:" in captured.out
        assert "문서 3:" in captured.out

    def test_빈_리스트_처리(self, capsys):
        """빈 리스트일 때 경고 메시지 출력"""
        print_retrieved_documents([])

        captured = capsys.readouterr()
        assert "검색 결과가 없습니다" in captured.out

    def test_전체_문서_출력(self, sample_qdrant_points, capsys):
        """n=None일 때 전체 문서 출력"""
        print_retrieved_documents(sample_qdrant_points, n=None)

        captured = capsys.readouterr()
        assert "총 5개 중 5개 표시" in captured.out
        assert "문서 5:" in captured.out
