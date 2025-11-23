"""
병렬 Embedding 생성 테스트

asyncio.to_thread를 사용한 병렬 처리가 제대로 작동하는지 검증
"""

import asyncio
import time
from config import openai_client, EMBEDDING_MODEL


async def test_sequential():
    """순차 실행 테스트"""
    queries = ["무역 사기", "수출 절차", "수입 인증"]

    print("\n" + "="*60)
    print("순차 실행 테스트")
    print("="*60)

    start = time.time()
    results = []
    for q in queries:
        result = openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=q
        )
        results.append(result)

    elapsed = time.time() - start
    print(f"✓ 완료: {len(results)}개 임베딩 생성")
    print(f"⏱️  소요 시간: {elapsed:.2f}초")

    return elapsed


async def test_parallel_wrong():
    """잘못된 병렬 실행 (기존 코드 방식)"""
    queries = ["무역 사기", "수출 절차", "수입 인증"]

    print("\n" + "="*60)
    print("잘못된 병렬 실행 시도 (기존 방식)")
    print("="*60)

    start = time.time()

    # ❌ 이렇게 하면 List comprehension에서 즉시 순차 실행됨
    # asyncio.gather()에 전달할 때쯤엔 이미 모든 호출이 끝나있음
    results = [
        openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=q
        )
        for q in queries
    ]

    elapsed = time.time() - start
    print(f"✓ 완료: {len(results)}개 임베딩 생성")
    print(f"⏱️  소요 시간: {elapsed:.2f}초")
    print("⚠️  주의: List comprehension 단계에서 이미 순차 실행 완료!")
    print("         asyncio.gather()를 쓰려 해도 이미 늦음")

    return elapsed


async def test_parallel_correct():
    """올바른 병렬 실행 (수정된 코드 방식)"""
    queries = ["무역 사기", "수출 절차", "수입 인증"]

    print("\n" + "="*60)
    print("올바른 병렬 실행 테스트 (asyncio.to_thread)")
    print("="*60)

    start = time.time()

    # ✅ asyncio.to_thread로 감싸서 진짜 병렬 실행
    embedding_tasks = [
        asyncio.to_thread(
            openai_client.embeddings.create,
            model=EMBEDDING_MODEL,
            input=q
        )
        for q in queries
    ]
    results = await asyncio.gather(*embedding_tasks)

    elapsed = time.time() - start
    print(f"✓ 완료: {len(results)}개 임베딩 생성")
    print(f"⏱️  소요 시간: {elapsed:.2f}초")

    return elapsed


async def main():
    print("\n" + "="*60)
    print("병렬 처리 성능 비교 테스트")
    print("="*60)

    # Test 1: 순차 실행
    sequential_time = await test_sequential()

    # Test 2: 잘못된 병렬 (기존 방식)
    wrong_parallel_time = await test_parallel_wrong()

    # Test 3: 올바른 병렬 (수정된 방식)
    correct_parallel_time = await test_parallel_correct()

    # 결과 비교
    print("\n" + "="*60)
    print("📊 성능 비교 결과")
    print("="*60)
    print(f"순차 실행:           {sequential_time:.2f}초")
    print(f"잘못된 병렬 (기존):   {wrong_parallel_time:.2f}초")
    print(f"올바른 병렬 (수정):   {correct_parallel_time:.2f}초")
    print()

    if correct_parallel_time < sequential_time:
        speedup = sequential_time / correct_parallel_time
        print(f"✅ 병렬 처리 성공! 약 {speedup:.1f}배 빠름")
    else:
        print(f"⚠️  병렬 처리 효과가 미미함 (네트워크 속도 영향)")

    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
