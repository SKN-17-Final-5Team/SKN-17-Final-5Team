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
    USE_RERANKER,
    USE_PER_QUERY_RERANK
)
from utils import print_retrieved_documents
from services.reranker_service import call_reranker_api
from services.query_transformer_service import rewrite_and_decompose_query


@function_tool
async def search_trade_documents(query: str, limit: int = 25, top_k: int = 10) -> str:
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
    transform = await rewrite_and_decompose_query(query)
    rewritten_query = transform.rewritten_query
    sub_queries = transform.sub_queries or [rewritten_query]  # None이면 단일 쿼리로 변환

    # ===== 통합 검색 (단순/복합 질문 모두 동일한 경로 사용) =====
    grouped_points = await _multi_search(sub_queries, limit)
    total_docs = sum(len(pts) for pts in grouped_points.values())
    print(f"✓ 최종 {total_docs}개 문서 수집 ({len(sub_queries)}개 그룹)\n")

    if not grouped_points or all(len(pts) == 0 for pts in grouped_points.values()):
        print("⚠️  검색 결과가 없습니다.\n")
        return "검색 결과가 없습니다."

    # 디버깅용 출력
    all_points_for_debug = []
    for pts in grouped_points.values():
        all_points_for_debug.extend(pts)
    print_retrieved_documents(all_points_for_debug, n=25)

    # ----- 개별 Rerank vs 통합 Rerank 선택 -----
    if USE_RERANKER and USE_PER_QUERY_RERANK:
        # 개별 Rerank: 각 서브 쿼리별로 rerank
        reranked_results = await _rerank_per_query(grouped_points, sub_queries, top_k)

        if not reranked_results:
            print("⚠️  Rerank 결과가 없습니다.\n")
            return "검색 결과가 없습니다."

        # 결과 포맷팅 (개별 rerank 결과)
        print("="*60)
        print(f"🎯 개별 Rerank로 선정된 최종 {len(reranked_results)}개 문서 (모델에게 전달)")
        print("="*60)

        formatted = []
        for rank, (point, rerank_score, sub_query) in enumerate(reranked_results, 1):
            content = point.payload.get("text") or point.payload.get("content") or ""
            if content:
                content = content[:500]
            source_tag = point.payload.get("doc_id", "unknown")

            # Agent에게 전달할 텍스트
            doc_text = f"[{rank}] {content}\n   출처: {source_tag}, Rerank 점수: {rerank_score:.3f}, 서브쿼리: '{sub_query}'"
            formatted.append(doc_text)

            # 콘솔 디버깅 출력
            debug_doc_name = point.payload.get("document_name") or point.payload.get("file_name")
            debug_article = point.payload.get("article")

            print(f"\n문서 {rank}:")
            print(f"  서브쿼리: '{sub_query}'")
            print(f"  출처: {source_tag}")
            if debug_doc_name:
                print(f"  파일명: {debug_doc_name}")
            if debug_article:
                print(f"  조문: {debug_article}")
            print(f"  Rerank 점수: {rerank_score:.3f}")
            print(f"  내용: {content[:200]}{'...' if len(content) > 200 else ''}")

    else:
        # 통합 Rerank 또는 Reranker 미사용

        # 모든 그룹의 문서를 병합
        seen_ids = {}
        for pts in grouped_points.values():
            for point in pts:
                if point.id not in seen_ids or point.score > seen_ids[point.id].score:
                    seen_ids[point.id] = point

        all_points = sorted(seen_ids.values(), key=lambda p: p.score, reverse=True)

        rerank_response = None
        if USE_RERANKER:
            # 통합 Rerank 방식
            num_queries = len(sub_queries)
            rerank_msg = f"ℹ️  통합 Rerank 방식 사용 ({num_queries}개 쿼리 병합)\n"
            print(rerank_msg)

            documents_for_rerank = [
                point.payload.get("text") or point.payload.get("content") or ""
                for point in all_points
            ]

            try:
                rerank_response = await call_reranker_api(rewritten_query, documents_for_rerank, top_k=top_k)
            except Exception as e:
                print(f"⚠️  Reranker 실패: {e}")
                print(f"⚠️  기본 검색 결과의 상위 {top_k}개를 사용합니다.\n")
        else:
            # Reranker 미사용
            print(f"ℹ️  Reranker 미사용 - 기본 검색 결과 상위 {top_k}개 사용\n")

        # 결과 포맷팅
        formatted = _format_rerank_results(all_points, rerank_response, top_k)

    print("\n" + "=" * 60)
    print("🤖 모델이 위 문서를 기반으로 답변 생성 중...")
    print("=" * 60 + "\n")

    return "\n\n".join(formatted)


# ===== 내부 헬퍼 함수 =====

async def _multi_search(sub_queries: List[str], limit: int) -> dict:
    """
    병렬 검색 (단일/복합 질문 모두 처리)

    예1 (단일): ["수출 절차"] 1개 검색
    예2 (복합): ["수출 절차", "수입 절차"] 2개를 동시에 검색 → 서브 쿼리별 그룹화

    순차 검색보다 2~3배 빠름 (asyncio.gather 덕분)

    Returns:
        Dict[str, List]: {서브쿼리: 검색결과Points} 형태의 딕셔너리
    """
    num_queries = len(sub_queries)
    query_type = "단일 쿼리" if num_queries == 1 else f"{num_queries}개 서브쿼리"
    print(f"📌 검색 수행 ({query_type})")

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

    # 3) 서브 쿼리별로 그룹화
    print("   Step 3: 서브 쿼리별 그룹화 중...")
    grouped_points = {}

    for sq, result in zip(sub_queries, search_results):
        points = result.points if hasattr(result, 'points') else []

        # 각 그룹 내 중복 제거 (같은 서브쿼리 내에서만)
        seen_ids = {}
        for point in points:
            point_id = point.id
            if point_id not in seen_ids or point.score > seen_ids[point_id].score:
                seen_ids[point_id] = point

        # 점수 높은 순으로 정렬
        unique_points = sorted(seen_ids.values(), key=lambda p: p.score, reverse=True)
        grouped_points[sq] = unique_points

        print(f"   서브쿼리: '{sq}' → {len(unique_points)}개")

    return grouped_points


def _format_rerank_results(points: List, rerank_response, top_k: int) -> List[str]:
    """
    Rerank 결과를 Agent에게 전달할 형식으로 포맷팅

    Args:
        points: 검색된 문서 Points 리스트
        rerank_response: Reranker API 응답 (None이면 기본 검색 결과 사용)
        top_k: 반환할 문서 개수

    Returns:
        List[str]: 포맷팅된 문서 텍스트 리스트
    """
    formatted = []

    if rerank_response:
        # Reranker 결과 사용
        print("="*60)
        print(f"🎯 Reranker로 선정된 최종 {len(rerank_response.results)}개 문서 (모델에게 전달)")
        print("="*60)

        for rank, result in enumerate(rerank_response.results, 1):
            original_point = points[result.index]
            content = original_point.payload.get("text") or original_point.payload.get("content") or ""
            if content:
                content = content[:500]  # 너무 길면 잘라냄
            source_tag = original_point.payload.get("data_source", "unknown")
            rerank_score = result.score

            # Agent에게 전달할 텍스트
            doc_text = f"[{rank}] {content}\n   출처: {source_tag}, Rerank 점수: {rerank_score:.3f}"
            formatted.append(doc_text)

            # 콘솔 디버깅 출력
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
        # 기본 검색 결과 사용
        print("="*60)
        print(f"📄 기본 검색 결과 상위 {top_k}개 (모델에게 전달)")
        print("="*60)

        for i, point in enumerate(points[:top_k], 1):
            content = point.payload.get("text") or point.payload.get("content") or ""
            if content:
                content = content[:500]
            score = point.score
            source_tag = point.payload.get("data_source", "unknown")

            doc_text = f"[{i}] {content}\n   출처: {source_tag}, 점수: {score:.3f}"
            formatted.append(doc_text)

    return formatted


async def _rerank_per_query(grouped_points: dict, sub_queries: List[str], total_topk: int) -> List:
    """
    각 서브 쿼리별로 개별 reranking 수행

    Args:
        grouped_points: 서브 쿼리별로 그룹화된 검색 결과 {sub_query: [Points]}
        sub_queries: 서브 쿼리 리스트
        total_topk: 최종 반환할 총 문서 개수

    Returns:
        List[tuple]: [(Point, rerank_score, sub_query), ...] 형태의 리스트
    """
    # Top-k를 서브 쿼리 개수로 균등 배분 (최소 1개)
    per_query_k = max(1, total_topk // len(sub_queries))

    print(f"\n🎯 개별 Rerank 수행: {len(sub_queries)}개 서브 쿼리")
    print(f"   각 서브 쿼리당 {per_query_k}개 선정 (총 약 {per_query_k * len(sub_queries)}개)")

    all_reranked = []

    for i, sq in enumerate(sub_queries, 1):
        points = grouped_points.get(sq, [])
        if not points:
            print(f"\n   [{i}/{len(sub_queries)}] '{sq}' → 검색 결과 없음, 건너뜀")
            continue

        print(f"\n   [{i}/{len(sub_queries)}] '{sq}'")
        print(f"      검색 결과: {len(points)}개 → Rerank → top {per_query_k}")

        # 문서 텍스트 추출
        documents = [
            point.payload.get("text") or point.payload.get("content") or ""
            for point in points
        ]

        # 개별 rerank 수행
        try:
            rerank_response = await call_reranker_api(sq, documents, top_k=per_query_k)

            # 결과 저장 (원본 Point, rerank 점수, 서브 쿼리)
            for result in rerank_response.results:
                original_point = points[result.index]
                all_reranked.append((original_point, result.score, sq))

            print(f"      ✓ Rerank 완료: {len(rerank_response.results)}개 선정")

        except Exception as e:
            print(f"      ⚠️ Rerank 실패: {e}")
            print(f"      → 기본 검색 점수 기준 상위 {per_query_k}개 사용")
            # 실패 시 검색 점수 기준 상위 per_query_k개 사용
            for point in points[:per_query_k]:
                all_reranked.append((point, point.score, sq))

    print(f"\n✓ 개별 Rerank 완료: 총 {len(all_reranked)}개 문서 선정\n")

    return all_reranked
