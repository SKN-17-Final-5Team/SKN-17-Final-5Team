"""
문서 검색 Tool (Reranker 통합)

Qdrant Vector Search와 Reranker API를 활용한 고도화된 문서 검색
"""

import asyncio
from agents import function_tool

from config import (
    qdrant_client,
    openai_client,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    USE_RERANKER
)
from utils import print_retrieved_documents
from services.reranker_service import call_reranker_api


@function_tool
async def search_trade_documents(query: str, limit: int = 25, top_k: int = 5) -> str:
    """
    무역 문서 검색 및 Reranking 수행

    프로세스:
    1. 쿼리를 Embedding으로 변환 (OpenAI text-embedding-3-large)
    2. Qdrant에서 유사도 기반 초기 검색 (limit개)
    3. RunPod Reranker API로 재정렬
    4. 상위 top_k개만 Agent에게 전달

    Args:
        query: 검색 쿼리
        limit: 초기 검색에서 가져올 문서 개수 (기본값: 25)
        top_k: Reranker 후 최종적으로 Agent에게 전달할 문서 개수 (기본값: 5)

    Returns:
        Agent가 사용할 포맷된 문서 문자열
    """

    print(f"\n🔍 검색 중: '{query}' (초기 검색: {limit}개, 최종 선정: {top_k}개)")

    # ─────────────────────────────────────────────────────────────────
    # 1단계: 쿼리 Embedding 생성
    # ─────────────────────────────────────────────────────────────────
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    )
    query_vector = response.data[0].embedding

    # ─────────────────────────────────────────────────────────────────
    # 2단계: Qdrant Vector Search
    # ─────────────────────────────────────────────────────────────────
    search_result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True
    )

    # 검색 결과 포인트 추출
    points = search_result.points if hasattr(search_result, 'points') else []

    print(f"✓ {len(points)}개 문서 발견\n")

    if not points:
        print("⚠️  검색 결과가 없습니다.\n")
        return "검색 결과가 없습니다."

    # ─────────────────────────────────────────────────────────────────
    # 3단계: 초기 검색 결과 출력 (디버깅용 - 콘솔에만 출력)
    # ─────────────────────────────────────────────────────────────────
    print_retrieved_documents(points)

    # ─────────────────────────────────────────────────────────────────
    # 4단계: Reranking을 위한 문서 텍스트 준비
    # ─────────────────────────────────────────────────────────────────
    documents_for_rerank = [
        point.payload.get("text") or point.payload.get("content") or ""
        for point in points
    ]

    # ─────────────────────────────────────────────────────────────────
    # 5단계: Reranker API 호출 (사용자 설정에 따라)
    # ─────────────────────────────────────────────────────────────────
    rerank_response = None

    if USE_RERANKER:
        # Reranker 사용 모드
        try:
            rerank_response = await call_reranker_api(query, documents_for_rerank, top_k=top_k)
        except Exception as e:
            print(f"⚠️  Reranker 실패: {e}")
            print(f"⚠️  기본 검색 결과의 상위 {top_k}개를 사용합니다.\n")
            # Fallback: Reranker 실패 시 기본 검색 결과 사용
            rerank_response = None
    else:
        # Reranker 미사용 모드
        print(f"ℹ️  Reranker 미사용 - 기본 검색 결과 상위 {top_k}개 사용\n")

    # ─────────────────────────────────────────────────────────────────
    # 6단계: 최종 결과 포맷팅 (Agent에게 전달할 문서)
    # ─────────────────────────────────────────────────────────────────
    if rerank_response:
        # Reranker 결과를 사용하는 경우
        print("="*60)
        print(f"🎯 Reranker로 선정된 최종 {len(rerank_response.results)}개 문서 (모델에게 전달)")
        print("="*60)

        formatted = []
        for rank, result in enumerate(rerank_response.results, 1):
            # 원본 문서 포인트 가져오기
            original_point = points[result.index]
            # text 또는 content 필드에서 내용 가져오기
            content = original_point.payload.get("text") or original_point.payload.get("content") or ""
            if content:
                content = content[:500]
            source_tag = original_point.payload.get("data_source", "unknown")
            rerank_score = result.score

            # Agent에게 전달할 텍스트 (출처는 data_source 태그만)
            doc_text = f"[{rank}] {content}\n   출처: {source_tag}, Rerank 점수: {rerank_score:.3f}"
            formatted.append(doc_text)

            # 콘솔 로그 (디버깅용 - Agent에게는 전달되지 않음)
            debug_doc_name = original_point.payload.get("document_name") or original_point.payload.get("file_name")
            debug_article = original_point.payload.get("article")

            print(f"\n문서 {rank}:")
            print(f"  출처: {source_tag}")
            if debug_doc_name:
                print(f"  파일명: {debug_doc_name}")
            if debug_article:
                print(f"  조문: {debug_article}")
            print(f"  원본 인덱스: {result.index + 1}")
            print(f"  Rerank 점수: {rerank_score:.3f}")
            print(f"  내용: {content[:200]}{'...' if len(content) > 200 else ''}")

    else:
        # Fallback: 기본 검색 결과를 사용하는 경우
        print("="*60)
        print(f"📄 기본 검색 결과 상위 {top_k}개 (모델에게 전달)")
        print("="*60)

        formatted = []
        for i, point in enumerate(points[:top_k], 1):
            # text 또는 content 필드에서 내용 가져오기
            content = point.payload.get("text") or point.payload.get("content") or ""
            if content:
                content = content[:500]
            score = point.score
            source_tag = point.payload.get("data_source", "unknown")

            # Agent에게 전달할 텍스트
            doc_text = f"[{i}] {content}\n   출처: {source_tag}, 점수: {score:.3f}"
            formatted.append(doc_text)

    print("\n" + "=" * 60)
    print("🤖 모델이 위 문서를 기반으로 답변 생성 중...")
    print("=" * 60 + "\n")

    # Agent에게는 data_source 태그 기반 출처만 포함된 텍스트 전달
    # (파일명, 문서명, 조문 정보는 콘솔에만 출력됨)
    return "\n\n".join(formatted)