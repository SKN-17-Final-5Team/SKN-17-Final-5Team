"""
문서 검색 Tool (Query Rewriting/Decomposition + Reranker 통합)

Qdrant Vector Search와 Reranker API를 활용한 고도화된 문서 검색
- Query Transformation: 쿼리 리라이팅 및 복합 질문 분해
- Multi-Query Search: 병렬 검색 및 중복 제거
- Reranking: 최종 문서 재정렬
"""

import asyncio
from typing import List
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
from services.query_transformer_service import rewrite_and_decompose_query


@function_tool
async def search_trade_documents(query: str, limit: int = 25, top_k: int = 5) -> str:
    """
    무역 문서 검색 (Query Transformation + Reranking)

    프로세스:
    0. Query Transformation: 쿼리 리라이팅 및 복합 질문 분해
    1. Vector Search: 단일 또는 멀티 쿼리로 Qdrant 검색
    2. 중복 제거 및 병합 (멀티 쿼리인 경우)
    3. Reranking: RunPod Reranker API로 재정렬
    4. 상위 top_k개만 Agent에게 전달

    Args:
        query: 검색 쿼리
        limit: 초기 검색에서 가져올 문서 개수 (기본값: 25)
        top_k: Reranker 후 최종적으로 Agent에게 전달할 문서 개수 (기본값: 5)

    Returns:
        Agent가 사용할 포맷된 문서 문자열
    """

    print(f"\n🔍 검색 시작: '{query}' (초기 검색: {limit}개, 최종 선정: {top_k}개)")

    # ─────────────────────────────────────────────────────────────────
    # 0단계: Query Transformation (Rewriting + Decomposition)
    # ─────────────────────────────────────────────────────────────────
    transform = await rewrite_and_decompose_query(query)
    rewritten_query = transform.rewritten_query
    sub_queries = transform.sub_queries

    # ─────────────────────────────────────────────────────────────────
    # 1단계: Vector Search (단일 vs 멀티)
    # ─────────────────────────────────────────────────────────────────
    if not sub_queries or len(sub_queries) == 0:
        # 단순 질문 → 단일 검색
        points = await _single_search(rewritten_query, limit)
    else:
        # 복합 질문 → 멀티 검색 (병렬 처리 + 중복 제거)
        points = await _multi_search(sub_queries, limit)

    print(f"✓ 최종 {len(points)}개 문서 수집\n")

    if not points:
        print("⚠️  검색 결과가 없습니다.\n")
        return "검색 결과가 없습니다."

    # ─────────────────────────────────────────────────────────────────
    # 2단계: 초기 검색 결과 출력 (디버깅용 - 콘솔에만 출력)
    # ─────────────────────────────────────────────────────────────────
    print_retrieved_documents(points, n=25)  # 최대 10개만 출력

    # ─────────────────────────────────────────────────────────────────
    # 3단계: Reranking을 위한 문서 텍스트 준비
    # ─────────────────────────────────────────────────────────────────
    documents_for_rerank = [
        point.payload.get("text") or point.payload.get("content") or ""
        for point in points
    ]

    # ─────────────────────────────────────────────────────────────────
    # 4단계: Reranker API 호출 (사용자 설정에 따라)
    # ─────────────────────────────────────────────────────────────────
    rerank_response = None

    if USE_RERANKER:
        # Reranker 사용 모드
        try:
            # Reranker는 원본 query 또는 rewritten_query를 사용 가능
            # 여기서는 rewritten_query 사용 (더 정확한 검색을 위해)
            rerank_response = await call_reranker_api(rewritten_query, documents_for_rerank, top_k=top_k)
        except Exception as e:
            print(f"⚠️  Reranker 실패: {e}")
            print(f"⚠️  기본 검색 결과의 상위 {top_k}개를 사용합니다.\n")
            # Fallback: Reranker 실패 시 기본 검색 결과 사용
            rerank_response = None
    else:
        # Reranker 미사용 모드
        print(f"ℹ️  Reranker 미사용 - 기본 검색 결과 상위 {top_k}개 사용\n")

    # ─────────────────────────────────────────────────────────────────
    # 5단계: 최종 결과 포맷팅 (Agent에게 전달할 문서)
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


# =====================================================================
# 헬퍼 함수들 (내부 사용)
# =====================================================================

async def _single_search(query: str, limit: int) -> List:
    """
    단일 쿼리로 Qdrant 검색

    Args:
        query: 검색 쿼리
        limit: 가져올 문서 개수

    Returns:
        검색된 포인트 리스트
    """
    print(f"📌 단일 검색 수행: '{query}'")

    # Embedding 생성
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    )
    query_vector = response.data[0].embedding

    # Qdrant 검색
    search_result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True
    )

    points = search_result.points if hasattr(search_result, 'points') else []
    print(f"   → {len(points)}개 문서 발견")

    return points


async def _multi_search(sub_queries: List[str], limit: int) -> List:
    """
    멀티 쿼리로 병렬 검색 + 중복 제거

    Args:
        sub_queries: 서브쿼리 리스트
        limit: 각 쿼리당 가져올 문서 개수

    Returns:
        중복 제거 및 병합된 포인트 리스트
    """
    print(f"📌 멀티 검색 수행 ({len(sub_queries)}개 서브쿼리)")

    # Step 1: 병렬로 Embedding 생성
    print("   Step 1: Embedding 생성 중...")
    embedding_tasks = [
        openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=sq
        )
        for sq in sub_queries
    ]
    embeddings = await asyncio.gather(*embedding_tasks)

    # Step 2: 병렬로 Qdrant 검색
    print("   Step 2: Qdrant 검색 중...")
    search_tasks = [
        asyncio.to_thread(
            qdrant_client.query_points,
            collection_name=COLLECTION_NAME,
            query=emb.data[0].embedding,
            limit=limit,
            with_payload=True
        )
        for emb in embeddings
    ]
    search_results = await asyncio.gather(*search_tasks)

    # Step 3: 결과 수집 및 통계 출력
    for i, (sq, result) in enumerate(zip(sub_queries, search_results), 1):
        points_count = len(result.points) if hasattr(result, 'points') else 0
        print(f"   서브쿼리 {i}: '{sq}' → {points_count}개")

    # Step 4: 중복 제거 (ID 기반, 최고 점수 보존)
    print("   Step 3: 중복 제거 및 병합 중...")
    seen_ids = {}

    for result in search_results:
        points = result.points if hasattr(result, 'points') else []
        for point in points:
            point_id = point.id
            # 처음 보는 ID이거나, 더 높은 점수를 가진 경우 업데이트
            if point_id not in seen_ids or point.score > seen_ids[point_id].score:
                seen_ids[point_id] = point

    # Step 5: 점수 순으로 정렬
    merged_points = sorted(seen_ids.values(), key=lambda p: p.score, reverse=True)

    total_before = sum(
        len(result.points) if hasattr(result, 'points') else 0
        for result in search_results
    )
    print(f"   → 중복 제거 전: {total_before}개, 후: {len(merged_points)}개")

    # Reranker에 넉넉히 전달하기 위해 limit * 2 제한
    return merged_points[:limit * 2]
