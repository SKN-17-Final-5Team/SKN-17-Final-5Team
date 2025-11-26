"""
RAG 시스템 통합 진입점

사용자 ID 입력 후 기존 채팅방 또는 새 작업을 선택할 수 있습니다.
"""

import asyncio
import sys
import os
import uuid
from datetime import datetime

# 현재 디렉토리를 Python path에 추가
sys.path.insert(0, os.path.dirname(__file__))

from memory_modules.repository import EDARepository
from my_agents.trade_agent import create_trade_agent
from utils import dedup_consecutive_lines
from agents import Runner
import config


def select_chat_or_create(repo: EDARepository, user_id: int):
    """
    기존 채팅방 선택 또는 새 작업 생성

    Returns:
        (chat_type, chat_id, is_new)
        - chat_type: "gen" | "workflow"
        - chat_id: gen_chat_id | trade_id
        - is_new: True | False
    """
    print("\n" + "=" * 60)
    print("채팅방 선택")
    print("=" * 60)

    # 기존 채팅방 목록 (일반 + 워크플로우)
    gen_chats = repo.get_user_gen_chats(user_id)
    workflows = repo.get_user_trade_flows(user_id)

    # 통합 리스트
    all_chats = []
    for chat in gen_chats:
        all_chats.append({
            "type": "gen",
            "id": chat["gen_chat_id"],
            "title": chat["title"],
            "created_at": chat["created_at"]
        })
    for wf in workflows:
        all_chats.append({
            "type": "workflow",
            "id": wf["trade_id"],
            "title": wf["title"],
            "created_at": wf["created_at"]
        })

    # 최신순 정렬
    all_chats.sort(key=lambda x: x["created_at"], reverse=True)

    if all_chats:
        print("\n[기존 채팅방 목록]")
        for i, chat in enumerate(all_chats[:10], 1):
            chat_type_label = "일반질의" if chat["type"] == "gen" else "문서생성"
            created = chat['created_at'].strftime("%Y-%m-%d %H:%M") if isinstance(chat['created_at'], datetime) else chat['created_at']
            print(f"{i}. {chat_type_label} | {chat['title']} (생성: {created})")

    print("\n[새 작업 시작]")
    next_num = len(all_chats[:10]) + 1 if all_chats else 1
    print(f"{next_num}. 문서 생성하기")
    print(f"{next_num + 1}. 일반 질의하기")
    print("0. 종료")

    choice = input(f"\n선택 (번호 입력): ").strip()

    if not choice.isdigit():
        return None, None, None

    choice_num = int(choice)

    # 기존 채팅방 선택
    if 1 <= choice_num <= len(all_chats[:10]):
        selected = all_chats[choice_num - 1]
        type_label = "일반 질의" if selected["type"] == "gen" else "문서 생성"
        print(f"✓ '{selected['title']}' ({type_label})을(를) 선택했습니다.\n")
        return selected["type"], selected["id"], False

    # 새 문서 생성
    elif choice_num == (len(all_chats[:10]) + 1 if all_chats else 1):
        print("\n📄 새 문서 생성 워크플로우를 시작합니다.")
        title = input("워크플로우 제목 (Enter = 'untitle'): ").strip() or None

        try:
            trade_id = repo.create_trade_flow(title=title, user_id=user_id)
            print(f"✓ 새 워크플로우 (ID: {trade_id})이(가) 생성되었습니다.\n")
            return "workflow", trade_id, True
        except Exception as e:
            print(f"Warning: 워크플로우 생성 실패: {e}")
            return None, None, None

    # 새 일반 질의
    elif choice_num == (len(all_chats[:10]) + 2 if all_chats else 2):
        print("\n💬 새 일반 질의 채팅을 시작합니다.\n")
        # gen_chat_id는 첫 메시지 입력 시 자동 생성되므로 None으로 시작
        return "gen", None, True

    # 종료
    elif choice_num == 0:
        return "exit", None, None

    return None, None, None


async def run_chat_session(repo: EDARepository, chat_type: str, chat_id: int, is_new: bool, user_id: int):
    """채팅 세션 실행"""

    # Reranker 설정
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

        per_query_choice = input("복합 질문 시 개별 Rerank를 사용하시겠습니까? (y/n, 기본값: y): ").strip().lower()
        if per_query_choice in ['n', 'no']:
            config.USE_PER_QUERY_RERANK = False
            print("✓ 통합 Rerank 방식으로 실행합니다.\n")
        else:
            config.USE_PER_QUERY_RERANK = True
            print("✓ 개별 Rerank 방식으로 실행합니다. (모든 토픽 균형 보장)\n")

    # 대화 루프
    while True:
        question = input("질문 (종료: 'exit'): ").strip()

        if question.lower() in ['exit', 'quit', '종료']:
            print("\n대화를 종료합니다.")
            break

        if not question:
            continue

        # 새 일반 채팅방인 경우 첫 메시지로 제목 생성 및 ID 할당
        if is_new and chat_type == "gen":
            try:
                chat_id = repo.create_gen_chat(
                    first_message=question,
                    user_id=user_id
                )
                print(f"✓ 새 채팅방 (ID: {chat_id})이(가) 생성되었습니다.\n")
                is_new = False
            except Exception as e:
                print(f"Warning: 채팅방 생성 실패: {e}")
                continue

        print(f"\n{'='*60}\n")

        # 메모리 컨텍스트 로드 (요약 및 문서 정보)
        if chat_type == "gen":
            memory_context = repo.get_gen_context(chat_id, limit=10)
        else:  # workflow
            memory_context = repo.get_trade_context(chat_id, limit=10, include_documents=True)

        # 최근 대화 히스토리 로드 (OpenAI messages 형식)
        previous_messages = repo.get_recent_turns_for_model(chat_id, chat_type=chat_type, limit=10)

        # Agent 생성 (메모리 컨텍스트 + 이전 대화 포함)
        agent = create_trade_agent(memory_context=memory_context, previous_messages=previous_messages)
        print("🤖 Agent 실행 중...\n")

        # Agent 실행
        result = await Runner.run(agent, input=question)

        # 결과 출력
        cleaned = dedup_consecutive_lines(result.final_output)
        print("="*60)
        print("\n최종 답변:")
        print("-" * 60)
        print(cleaned)
        print("\n" + "="*60 + "\n")

        # 대화 저장
        try:
            if chat_type == "gen":
                repo.save_gen_turn(chat_id, question, cleaned)
            else:  # workflow
                repo.save_trade_turn(chat_id, question, cleaned)
            print("✓ 대화가 저장되었습니다.\n")
        except Exception as e:
            print(f"Warning: 대화 저장 실패: {e}\n")


async def main():
    """메인 함수"""
    print("\n" + "=" * 60)
    print("RAG 시스템 - 무역 전문가 AI")
    print("=" * 60)

    # 메모리 시스템 초기화
    repo = EDARepository()

    # 사용자 ID 입력 (숫자로 변환)
    user_id_str = input("\n사용자 ID (숫자, Enter = 1): ").strip() or "1"
    try:
        user_id = int(user_id_str)
    except ValueError:
        print("잘못된 사용자 ID입니다. 기본값 1을 사용합니다.")
        user_id = 1

    while True:
        # 채팅방 선택 또는 새 작업 생성
        chat_type, chat_id, is_new = select_chat_or_create(repo, user_id)

        if chat_type == "exit" or chat_type is None:
            print("\n시스템을 종료합니다. 감사합니다!")
            break

        # 채팅 세션 실행
        await run_chat_session(repo, chat_type, chat_id, is_new, user_id)

        # 세션 종료 후 다시 메뉴로


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n시스템을 종료합니다.")
    except Exception as e:
        print(f"\n오류가 발생했습니다: {e}")
        import traceback
        traceback.print_exc()
