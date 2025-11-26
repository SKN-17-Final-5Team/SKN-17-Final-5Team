"""
RAG 시스템 메인 실행 파일

Reranker API를 활용한 고도화된 RAG 시스템
- 초기 검색: Qdrant Vector DB에서 limit개 문서 검색
- Reranking: RunPod 서버의 Reranker 모델로 재정렬
- 최종 전달: 상위 top_k개 문서만 Agent에게 전달
- 메모리 시스템: MySQL 기반 대화 히스토리 관리
"""

import asyncio
import uuid
from datetime import datetime
from agents import Runner  # OpenAI Agents SDK
from my_agents.trade_agent import create_trade_agent
from utils import dedup_consecutive_lines
from memory_modules.repository import EDARepository
import config


def select_or_create_chat(repo: EDARepository, user_id: int):
    """
    기존 채팅방 선택 또는 새 채팅방 생성

    Returns:
        (gen_chat_id, is_new_chat)
    """
    print("=" * 60)
    print("채팅방 선택")
    print("=" * 60)

    # 기존 채팅방 조회
    chats = repo.get_user_gen_chats(user_id)

    if chats:
        print("\n[기존 채팅방 목록]")
        for i, chat in enumerate(chats[:10], 1):  # 최근 10개만
            created = chat['created_at'].strftime("%Y-%m-%d %H:%M") if isinstance(chat['created_at'], datetime) else chat['created_at']
            print(f"{i}. {chat['title']} (생성: {created})")
        print("0. 새 채팅방 만들기")

        choice = input("\n선택 (번호 입력): ").strip()

        if choice.isdigit():
            choice_num = int(choice)
            if 1 <= choice_num <= len(chats[:10]):
                selected_chat = chats[choice_num - 1]
                print(f"✓ '{selected_chat['title']}' 채팅방을 선택했습니다.\n")
                return selected_chat['gen_chat_id'], False

    # 새 채팅방 생성 (ID는 None, 첫 메시지 입력 시 생성)
    print("\n[새 채팅방 생성]")
    return None, True


async def main():
    """
    RAG Agent 실행 함수 (메모리 시스템 통합)

    사용자 입력을 받아 무역 전문가 Agent를 실행하고,
    대화 히스토리를 MySQL에 저장하여 다음 실행 시 이어서 대화할 수 있습니다.
    """
    # 메모리 시스템 초기화
    repo = EDARepository()

    # 사용자 ID (숫자로 변환)
    user_id_str = input("사용자 ID (숫자, Enter = 1): ").strip() or "1"
    try:
        user_id = int(user_id_str)
    except ValueError:
        print("잘못된 사용자 ID입니다. 기본값 1을 사용합니다.")
        user_id = 1

    # 채팅방 선택 또는 생성
    gen_chat_id, is_new_chat = select_or_create_chat(repo, user_id)

    # Reranker 사용 여부 선택
    print("=" * 60)
    print("RAG 시스템 설정")
    print("=" * 60)
    reranker_choice = input("Reranker를 사용하시겠습니까? (y/n, 기본값: y): ").strip().lower()

    if reranker_choice in ['n', 'no']:
        config.USE_RERANKER = False
        print("✓ Reranker 미사용 모드로 실행합니다.\n")
    else:
        config.USE_RERANKER = True
        print("✓ Reranker 사용 모드로 실행합니다.")

        # Reranker 사용 시 개별 Rerank 방식 선택
        per_query_choice = input("복합 질문 시 개별 Rerank를 사용하시겠습니까? (y/n, 기본값: y): ").strip().lower()

        if per_query_choice in ['n', 'no']:
            config.USE_PER_QUERY_RERANK = False
            print("✓ 통합 Rerank 방식으로 실행합니다.\n")
        else:
            config.USE_PER_QUERY_RERANK = True
            print("✓ 개별 Rerank 방식으로 실행합니다. (모든 토픽 균형 보장)\n")

    # 대화 루프
    while True:
        # 사용자 질문 입력
        question = input("질문 (종료: 'exit'): ").strip()

        if question.lower() in ['exit', 'quit', '종료']:
            print("\n대화를 종료합니다.")
            break

        if not question:
            continue

        # 새 채팅방인 경우 첫 메시지로 제목 생성 및 ID 할당
        if is_new_chat:
            try:
                gen_chat_id = repo.create_gen_chat(
                    first_message=question,
                    user_id=user_id
                )
                print(f"✓ 새 채팅방 (ID: {gen_chat_id})이(가) 생성되었습니다.\n")
                is_new_chat = False
            except Exception as e:
                print(f"Warning: 채팅방 생성 실패: {e}")
                continue

        print(f"\n{'='*60}\n")

        # 메모리 컨텍스트 로드 (요약 및 문서 정보)
        memory_context = repo.get_gen_context(gen_chat_id, limit=10)

        # 최근 대화 히스토리 로드 (OpenAI messages 형식)
        previous_messages = repo.get_recent_turns_for_model(gen_chat_id, chat_type="gen", limit=10)

        # Agent 생성 (메모리 컨텍스트 + 이전 대화 포함)
        agent = create_trade_agent(memory_context=memory_context, previous_messages=previous_messages)

        # Agent 실행
        print("🤖 Agent 실행 중...\n")
        result = await Runner.run(agent, input=question)

        # 연속 중복 라인 제거
        cleaned = dedup_consecutive_lines(result.final_output)

        # 최종 답변 출력
        print("="*60)
        print("\n최종 답변:")
        print("-" * 60)
        print(cleaned)
        print("\n" + "="*60 + "\n")

        # 대화 저장
        try:
            repo.save_gen_turn(gen_chat_id, question, cleaned)
            print("✓ 대화가 저장되었습니다.\n")
        except Exception as e:
            print(f"Warning: 대화 저장 실패: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())