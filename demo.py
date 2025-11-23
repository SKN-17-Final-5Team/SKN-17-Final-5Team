"""
고급 통합: Chainlit UI + Trade RAG Agent
- 실시간 스트리밍 응답
- 도구 호출 시각화
- 메모리 시스템 통합
- 파일 업로드 지원
"""

import chainlit as cl
import os
import sys
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# agent 모듈을 import하기 위해 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
agent_dir = os.path.join(current_dir, "agent")
sys.path.insert(0, agent_dir)

# RAG Agent 및 메모리 시스템 import
try:
    from agents import Runner, RunHooks
    from my_agents.trade_agent import create_trade_agent
    from memory_modules.repository import EDARepository
    from utils import dedup_consecutive_lines
except ImportError as e:
    print(f"Import 오류: {e}")
    print(f"agent 디렉토리 경로: {agent_dir}")
    print("agent 디렉토리가 존재하는지 확인해주세요.")
    raise

# 메모리 시스템 초기화 (전역 인스턴스)
memory = EDARepository()


# =====================================================================
# RunHooks: 도구 호출 추적 (메인 메시지 내에서 상태 업데이트)
# =====================================================================
class ChainlitRunHooks(RunHooks):
    """Chainlit UI 업데이트를 위한 RunHooks"""

    def __init__(self, main_message):
        self.main_message = main_message

    async def on_tool_start(self, context, agent, tool):
        """도구 호출 시작 시 메인 메시지 업데이트"""
        tool_name = tool.name if hasattr(tool, 'name') else str(tool)

        # 도구별 아이콘 및 메시지
        if 'search_trade_documents' in tool_name:
            icon = "📚"
            message = "무역 문서 검색 중..."
        elif 'search_web' in tool_name:
            icon = "🌐"
            message = "웹 검색 중 (최신 정보 조회)..."
        else:
            icon = "🔧"
            message = f"{tool_name} 실행 중..."

        # 메인 메시지 내용 업데이트
        self.main_message.content = f"{icon} {message}"
        await self.main_message.update()

    async def on_tool_end(self, context, agent, tool, result):
        """도구 호출 완료 시 메인 메시지 비우기"""
        # 메인 메시지를 빈 상태로 리셋 (스트리밍이 시작되면 채워짐)
        self.main_message.content = ""
        await self.main_message.update()


@cl.on_chat_start
async def start():
    """새로운 채팅 세션 시작"""
    # 고유 세션 ID 생성 (Chainlit 세션 ID 기반)
    session_id = f"web_{cl.user_session.get('id')}"
    cl.user_session.set("session_id", session_id)

    # 환영 메시지 없음 (사용자가 바로 질문할 수 있도록)


@cl.on_message
async def main(message: cl.Message):
    """
    메시지 처리 함수 (고급 통합 버전)
    - 스트리밍 응답
    - 도구 호출 시각화
    - 메모리 통합
    """
    session_id = cl.user_session.get("session_id")

    # 파일 업로드 처리
    file_context = ""
    if message.elements:
        file_info_parts = []

        for element in message.elements:
            file_name = element.name
            file_path = element.path
            file_type = element.mime if hasattr(element, 'mime') else 'Unknown'

            # 텍스트 파일 읽기
            if file_type and 'text' in file_type:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        file_context += f"\n\n[첨부파일: {file_name}]\n{content}\n"
                        file_info_parts.append(f"✅ **{file_name}** ({len(content)} 글자)")
                except Exception as e:
                    file_info_parts.append(f"❌ **{file_name}** (읽기 실패: {str(e)})")
            else:
                file_info_parts.append(f"📎 **{file_name}** ({file_type})")

        # 파일 업로드 알림
        if file_info_parts:
            await cl.Message(
                content="업로드된 파일:\n" + "\n".join(file_info_parts)
            ).send()

    # 쿼리 준비 (파일 컨텍스트 포함)
    user_query = message.content
    if file_context:
        user_query = f"{message.content}\n{file_context}"

    # 대화 히스토리 로드 (최근 10개 메시지)
    memory_context = memory.get_gen_context(session_id, limit=10)

    # 메모리 컨텍스트를 포함한 에이전트 생성
    agent = create_trade_agent(memory_context=memory_context)

    # 응답 메시지 준비
    msg = cl.Message(content="")
    await msg.send()

    try:
        # ================================================================
        # 고급 통합: 스트리밍 + RunHooks로 도구 호출 실시간 추적
        # ================================================================

        # RunHooks 설정 (도구 호출 추적 - 메인 메시지 전달)
        hooks = ChainlitRunHooks(msg)

        # Agent 스트리밍 실행
        result = Runner.run_streamed(agent, input=user_query, hooks=hooks)

        # 스트리밍 이벤트 처리
        full_response = ""

        async for event in result.stream_events():
            # 이벤트 타입에 따라 처리
            event_type = getattr(event, 'type', None)

            # raw_response_event 처리 (OpenAI Responses API)
            if event_type == 'raw_response_event':
                # data 속성 확인
                if hasattr(event, 'data'):
                    data = event.data
                    data_type = getattr(data, 'type', None)

                    # 텍스트 델타 이벤트
                    if data_type == 'response.output_text.delta':
                        if hasattr(data, 'delta') and data.delta:
                            full_response += data.delta
                            await msg.stream_token(data.delta)

                    # 출력 아이템 델타 (다른 델타 타입)
                    elif hasattr(data, 'output'):
                        for output_item in data.output:
                            if hasattr(output_item, 'content'):
                                for content in output_item.content:
                                    if hasattr(content, 'text') and content.text:
                                        text = content.text
                                        if text not in full_response:
                                            full_response += text
                                            await msg.stream_token(text)

        # 스트리밍 완료
        await msg.update()

        # 중복 라인 제거
        if full_response:
            cleaned_output = dedup_consecutive_lines(full_response)
            if cleaned_output != full_response:
                msg.content = cleaned_output
                await msg.update()
        else:
            # fallback: result.final_output 사용
            cleaned_output = dedup_consecutive_lines(result.final_output)
            msg.content = cleaned_output
            await msg.update()

        # 메모리에 대화 저장
        memory.save_gen_turn(
            gen_chat_id=session_id,
            user_message=message.content,
            assistant_message=cleaned_output
        )

    except Exception as e:
        # 에러 처리
        error_msg = f"❌ **오류 발생**\n\n```\n{str(e)}\n```\n\n문제가 지속되면 관리자에게 문의해주세요."
        msg.content = error_msg
        await msg.update()

        # 에러 로깅
        print(f"Error in chat session {session_id}: {str(e)}")
        import traceback
        traceback.print_exc()
