"""
무역 문서 검색 Tool

복합 질문도 처리할 수 있도록 쿼리 변환 + 병렬 검색 기능 추가
- 쿼리 개선: "무역 사기 방지 어떻게 해?" → "무역 사기 예방 및 대응 방법"
- 복합 질문 분해: "수출과 수입 차이" → ["수출 절차", "수입 절차"] 2개로 나눠서 검색
- 병렬 검색: 여러 서브쿼리를 동시에 검색해서 속도 향상
- Reranking: 최종적으로 관련도 높은 문서만 Agent에게 전달
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
    무역 문서 검색 메인 함수

    단순 질문("수출 절차는?")도, 복합 질문("수출과 수입 차이는?")도 모두 처리 가능

    Args:
        query: 사용자 질문
        limit: Qdrant에서 가져올 문서 수 (기본 25개)
        top_k: 최종적으로 Agent에게 전달할 문서 수 (기본 5개)

    Returns:
        Agent가 읽을 수 있게 포맷된 문서 텍스트
    """
    print(f"\n🔍 검색 시작: '{query}' (초기 검색: {limit}개, 최종 선정: {top_k}개)")

    # 쿼리 개선 + 필요하면 복합 질문 분해
    # 예: "수출 수입 차이" → rewritten_query + sub_queries 2개
    transform = await rewrite_and_decompose_query(query)
    rewritten_query = transform.rewritten_query
    sub_queries = transform.sub_queries

    # 단순 질문이면 그냥 검색, 복합 질문이면 병렬로 여러 개 검색
    if not sub_queries or len(sub_queries) == 0:
        points = await _single_search(rewritten_query, limit)
    else:
        # 여러 서브쿼리를 동시에 검색 → 중복 제거 → 병합
        points = await _multi_search(sub_queries, limit)

    print(f"✓ 최종 {len(points)}개 문서 수집\n")

    if not points:
        print("⚠️  검색 결과가 없습니다.\n")
        return "검색 결과가 없습니다."

    # 디버깅용: 검색된 문서 출력 (콘솔에만 표시, Agent에게는 안 보냄)
    print_retrieved_documents(points, n=25)

    # Reranker에 전달할 텍스트 추출
    documents_for_rerank = [
        point.payload.get("text") or point.payload.get("content") or ""
        for point in points
    ]

    # Reranker로 재정렬 (설정에서 켜놨으면)
    rerank_response = None

    if USE_RERANKER:
        try:
            # rewritten_query로 rerank (원본 query보다 더 정확함)
            rerank_response = await call_reranker_api(rewritten_query, documents_for_rerank, top_k=top_k)
        except Exception as e:
            print(f"⚠️  Reranker 실패: {e}")
            print(f"⚠️  기본 검색 결과의 상위 {top_k}개를 사용합니다.\n")
            rerank_response = None
    else:
        print(f"ℹ️  Reranker 미사용 - 기본 검색 결과 상위 {top_k}개 사용\n")

    # Agent에게 전달할 최종 문서 포맷팅
    if rerank_response:
        print("="*60)
        print(f"🎯 Reranker로 선정된 최종 {len(rerank_response.results)}개 문서 (모델에게 전달)")
        print("="*60)

        formatted = []
        for rank, result in enumerate(rerank_response.results, 1):
            original_point = points[result.index]
            content = original_point.payload.get("text") or original_point.payload.get("content") or ""
            if content:
                content = content[:500]  # 너무 길면 잘라냄
            source_tag = original_point.payload.get("data_source", "unknown")
            rerank_score = result.score

            # Agent에게 전달할 텍스트 (간결하게)
            doc_text = f"[{rank}] {content}\n   출처: {source_tag}, Rerank 점수: {rerank_score:.3f}"
            formatted.append(doc_text)

            # 콘솔에만 추가 정보 출력 (개발자 디버깅용)
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
        # Reranker 실패했거나 꺼져있으면 기본 검색 결과 사용
        print("="*60)
        print(f"📄 기본 검색 결과 상위 {top_k}개 (모델에게 전달)")
        print("="*60)

        formatted = []
        for i, point in enumerate(points[:top_k], 1):
            content = point.payload.get("text") or point.payload.get("content") or ""
            if content:
                content = content[:500]
            score = point.score
            source_tag = point.payload.get("data_source", "unknown")

            doc_text = f"[{i}] {content}\n   출처: {source_tag}, 점수: {score:.3f}"
            formatted.append(doc_text)

    print("\n" + "=" * 60)
    print("🤖 모델이 위 문서를 기반으로 답변 생성 중...")
    print("=" * 60 + "\n")

    return "\n\n".join(formatted)


# ===== 내부 헬퍼 함수 =====

async def _single_search(query: str, limit: int) -> List:
    """
    일반적인 단일 쿼리 검색 (복합 질문 아닐 때)
    쿼리 → Embedding → Qdrant 검색 → 결과 반환
    """
    print(f"📌 단일 검색 수행: '{query}'")

    # 쿼리를 벡터로 변환
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    )
    query_vector = response.data[0].embedding

    # Qdrant에서 유사 문서 검색
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
    복합 질문 처리용 병렬 검색

    예: ["수출 절차", "수입 절차"] 2개를 동시에 검색 → 중복 제거 → 병합
    순차 검색보다 2~3배 빠름 (asyncio.gather 덕분)
    """
    print(f"📌 멀티 검색 수행 ({len(sub_queries)}개 서브쿼리)")

    # 1) 모든 서브쿼리를 동시에 벡터로 변환 (병렬 처리)
    print("   Step 1: Embedding 생성 중...")
    embedding_tasks = [
        asyncio.to_thread(  # 동기 함수를 비동기로 감싸기
            openai_client.embeddings.create,
            model=EMBEDDING_MODEL,
            input=sq
        )
        for sq in sub_queries
    ]
    embeddings = await asyncio.gather(*embedding_tasks)  # 모두 완료될 때까지 대기

    # 2) 모든 벡터로 동시에 Qdrant 검색 (병렬 처리)
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

    # 3) 각 서브쿼리별 검색 결과 확인
    for i, (sq, result) in enumerate(zip(sub_queries, search_results), 1):
        points_count = len(result.points) if hasattr(result, 'points') else 0
        print(f"   서브쿼리 {i}: '{sq}' → {points_count}개")

    # 4) 중복 문서 제거 (같은 문서가 여러 서브쿼리에서 나올 수 있음)
    print("   Step 4: 중복 제거 및 병합 중...")
    seen_ids = {}

    for result in search_results:
        points = result.points if hasattr(result, 'points') else []
        for point in points:
            point_id = point.id
            # 같은 문서면 점수가 더 높은 쪽으로 보존
            if point_id not in seen_ids or point.score > seen_ids[point_id].score:
                seen_ids[point_id] = point

    # 5) 점수 높은 순으로 정렬
    merged_points = sorted(seen_ids.values(), key=lambda p: p.score, reverse=True)

    total_before = sum(
        len(result.points) if hasattr(result, 'points') else 0
        for result in search_results
    )
    print(f"   → 중복 제거 전: {total_before}개, 후: {len(merged_points)}개")

    # Reranker가 다시 골라낼거니까 넉넉히 전달 (limit의 2배)
    return merged_points[:limit * 2]
