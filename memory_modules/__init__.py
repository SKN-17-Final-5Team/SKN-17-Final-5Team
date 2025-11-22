"""
메모리 모듈 v3.0 (단일 클래스 구조)

EDARepository - 통합 메모리 시스템
- Connection Pool: DB 연결 재사용
- LRU Cache: 최근 1,000개 세션 메모리 유지
- 비동기 요약: 백그라운드 스레드 (사용자 대기 없음)
"""

from .repository import EDARepository

__all__ = ["EDARepository"]
