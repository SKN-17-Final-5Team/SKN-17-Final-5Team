"""
웹 검색 Tool (Tavily API)

실시간 웹 검색으로 내부 문서에 없는 최신 정보를 찾습니다.
- 최신 뉴스 및 업데이트
- 시장 동향 및 규제 변경사항
- 정보 검증 및 크로스체크
"""

import os
from typing import Optional
from agents import function_tool
from tavily import TavilyClient
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# Tavily 클라이언트 초기화
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@function_tool
def search_web(
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_answer: bool = True
) -> str:
    """
    Tavily API를 사용한 실시간 웹 검색

    **사용 조건 (trade_instructions.txt 시나리오 기준):**

    시나리오 A) 최신 뉴스/동향만 요구하는 질문 → 이 툴만 사용
    - 예: "최근 미중 무역 갈등 상황은?", "2025년 미국 관세 정책 변화는?"
    - 순수하게 최신 뉴스, 동향, 시황만 물어보는 경우

    시나리오 B) 무역 실무 지식 질문 → 이 툴 사용 안 함
    - 문서 검색만으로 충분 (search_trade_documents만 사용)

    시나리오 C) 최신 정보 + 문서 내용 통합 질문 → search_trade_documents와 함께 사용
    - 예: "최근 미국 수출 규제 변경사항과 우리 문서의 대응 방안은?"
    - 내부 문서 검색 후 이 툴로 최신 정보 보완

    **검색 결과 제공 시 필수사항:**
    1. 출처 URL과 발행 날짜를 **반드시** 명시
    2. "최신" 정보 요청 시 2025년 → 2024년 → 2023년 순서로 최근 정보 우선 제공
    3. 내부 문서 기반 답변과 웹 검색 기반 답변을 **명확히 구분**

    Args:
        query: 검색할 질문 또는 키워드
        max_results: 반환할 검색 결과 수 (1-10, 기본값: 5)
        search_depth: 검색 깊이 "basic" 또는 "advanced" (기본값: "basic")
        include_answer: AI 요약 답변 포함 여부 (기본값: True)

    Returns:
        포맷팅된 검색 결과 텍스트 (출처 URL 및 발행 날짜 포함)
    """
    print(f"\n🌐 웹 검색 시작: '{query}' (최대 {max_results}개 결과, {search_depth} 모드)")

    try:
        # Tavily API로 웹 검색 수행
        # 최신 정보 우선: days 파라미터로 최근 데이터 우선 검색
        response = tavily_client.search(
            query=query,
            search_depth=search_depth,
            max_results=max_results,
            include_answer=include_answer,
            topic="general",
            days=730  # 최근 2년 이내 정보 우선 (2025년, 2024년 우선)
        )

        # 검색 결과 포맷팅
        formatted_results = []

        # AI 생성 요약 답변 추가 (있는 경우)
        if include_answer and response.get('answer'):
            formatted_results.append("=" * 60)
            formatted_results.append("📌 AI 요약 답변:")
            formatted_results.append("=" * 60)
            formatted_results.append(response['answer'])
            formatted_results.append("")

        # 검색 결과 추가
        if response.get('results'):
            # 결과를 발행일 기준으로 정렬 (최신순)
            results = response['results']
            # published_date가 있는 것을 우선, 그 다음 날짜순 정렬
            sorted_results = sorted(
                results,
                key=lambda x: x.get('published_date', '0000-01-01'),
                reverse=True  # 최신순
            )

            num_results = len(sorted_results)
            formatted_results.append("=" * 60)
            formatted_results.append(f"🔍 웹 검색 결과 ({num_results}개, 최신순):")
            formatted_results.append("=" * 60)

            for i, result in enumerate(sorted_results, 1):
                title = result.get('title', 'No title')
                url = result.get('url', '')
                content = result.get('content', '')
                score = result.get('score', 0)
                published_date = result.get('published_date', '날짜 정보 없음')

                # 내용이 너무 길면 잘라냄 (500자로 제한)
                content_preview = content[:500] + "..." if len(content) > 500 else content

                formatted_results.append(f"\n[{i}] {title}")
                formatted_results.append(f"   📅 발행일: {published_date}")
                formatted_results.append(f"   🔗 URL: {url}")
                formatted_results.append(f"   📊 관련도: {score:.2f}")
                formatted_results.append(f"   📰 내용: {content_preview}")

            print(f"✓ 웹 검색 완료: {num_results}개 결과 반환\n")
            return "\n".join(formatted_results)

        else:
            # 검색 결과가 없는 경우
            no_result_msg = f"'{query}'에 대한 웹 검색 결과가 없습니다."
            print(f"⚠️  {no_result_msg}\n")
            return no_result_msg

    except Exception as e:
        # 에러 발생 시
        error_msg = f"⚠️ 웹 검색 실패: {str(e)}"
        print(f"{error_msg}\n")
        return f"웹 검색 중 오류가 발생했습니다: {str(e)}"
