"""
EDARepository - 메모리 시스템 데이터 액세스 계층

In-Memory LRU Cache + MySQL 2-Tier 아키텍처
"""

from typing import Dict
from datetime import datetime
import threading
import pymysql
from pymysql.cursors import DictCursor
from contextlib import contextmanager
from cachetools import LRUCache
from dbutils.pooled_db import PooledDB
from openai import OpenAI

import config


class EDARepository:
    """메모리 데이터 저장 및 조회"""

    # 글자수 제한 상수
    MAX_GEN_CHAT_TITLE_LENGTH = 10  # gen_chat title 최대 길이
    MAX_TRADE_FLOW_TITLE_LENGTH = 100  # trade_flow title 최대 길이
    DEFAULT_WORKFLOW_TITLE = "untitle"  # 기본 워크플로우 제목

    def __init__(self):
        # In-Memory 캐시 (LRU)
        self.memory_store = LRUCache(maxsize=1000)

        # DB Connection Pool
        self._pool = None
        self._db_config = {
            'host': config.MYSQL_HOST,
            'port': config.MYSQL_PORT,
            'user': config.MYSQL_USER,
            'password': config.MYSQL_PASSWORD,
            'database': config.MYSQL_DATABASE,
            'charset': 'utf8mb4',
            'cursorclass': DictCursor
        }

        # OpenAI 클라이언트
        self._openai_client = None

    @property
    def pool(self):
        """DB Connection Pool (Lazy)"""
        if self._pool is None:
            self._pool = PooledDB(
                creator=pymysql,
                maxconnections=10,
                mincached=2,
                maxcached=5,
                blocking=True,
                **self._db_config
            )
        return self._pool

    @property
    def openai_client(self):
        """OpenAI 클라이언트 (Lazy)"""
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=config.openai_client.api_key)
        return self._openai_client

    @contextmanager
    def _get_connection(self):
        """DB 연결 컨텍스트"""
        conn = self.pool.connection()
        try:
            yield conn
        finally:
            conn.close()

    # =====================================================================
    # 유틸리티
    # =====================================================================

    def _generate_chat_title(self, first_message: str) -> str:
        """
        첫 질의를 요약하여 채팅방 제목 생성 (10글자 이내)
        Args:
            first_message: 사용자의 첫 질의
        Returns:
            요약된 제목 (10글자 이내)
        """
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "사용자의 질문을 10글자 이내로 간결하게 요약하세요. 핵심 키워드만 추출하세요."},
                    {"role": "user", "content": f"다음 질문을 10글자 이내로 요약: {first_message}"}
                ],
                temperature=0.3,
                max_tokens=50
            )
            title = response.choices[0].message.content.strip()

            # 10글자 초과 시 자르기
            if len(title) > self.MAX_GEN_CHAT_TITLE_LENGTH:
                title = title[:self.MAX_GEN_CHAT_TITLE_LENGTH]

            return title
        except Exception as e:
            print(f"Warning: Failed to generate title: {e}")
            # 실패 시 첫 메시지의 앞 10글자 사용
            return first_message[:self.MAX_GEN_CHAT_TITLE_LENGTH]

    def _ensure_unique_gen_chat_title(self, title: str, user_id: int = None) -> str:
        """
        일반 채팅방 제목 중복 체크 및 유니크한 제목 반환
        Args:
            title: 원본 제목
            user_id: 사용자 ID
        Returns:
            유니크한 제목
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # FOR UPDATE로 동시성 문제 방지 (옵션)
                    cursor.execute("""
                        SELECT title FROM gen_chat
                        WHERE user_id = %s
                        ORDER BY created_at DESC
                        FOR UPDATE
                    """, (user_id,))
                    existing_titles = {row['title'] for row in cursor.fetchall()}

                    if title not in existing_titles:
                        return title

                    # 중복 시 숫자 붙이기
                    counter = 2
                    while True:
                        suffix = f"-{counter}"
                        max_base_length = self.MAX_GEN_CHAT_TITLE_LENGTH - len(suffix)
                        new_title = title[:max_base_length] + suffix

                        if new_title not in existing_titles:
                            return new_title
                        counter += 1

        except Exception as e:
            print(f"Warning: Failed to check title uniqueness: {e}")
            return title

    def _ensure_unique_trade_flow_title(self, title: str, user_id: int = None) -> str:
        """
        워크플로우 제목 중복 체크 및 유니크한 제목 반환 (untitle, untitle-2 방식)
        Args:
            title: 원본 제목 (없으면 "untitle" 사용)
            user_id: 사용자 ID
        Returns:
            유니크한 제목
        """
        if not title:
            title = self.DEFAULT_WORKFLOW_TITLE

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # FOR UPDATE로 동시성 문제 방지
                    cursor.execute("""
                        SELECT title FROM trade_flow
                        WHERE user_id = %s
                        ORDER BY created_at DESC
                        FOR UPDATE
                    """, (user_id,))
                    existing_titles = {row['title'] for row in cursor.fetchall()}

                    if title not in existing_titles:
                        return title

                    # 중복 시 숫자 붙이기
                    counter = 2
                    while True:
                        new_title = f"{title}-{counter}"

                        if new_title not in existing_titles:
                            return new_title
                        counter += 1

        except Exception as e:
            print(f"Warning: Failed to check title uniqueness: {e}")
            return title
    
    def clear_memory(self, chat_id: int):
        """In-Memory 캐시 제거"""
        if chat_id in self.memory_store:
            del self.memory_store[chat_id]

    # =====================================================================
    # 메시지 저장
    # =====================================================================

    def save_gen_message(self, gen_chat_id: int, sender_type: str, content: str):
        """일반 채팅 메시지 저장 (sender_type: 'U' 또는 'A')"""
        self._save_message(gen_chat_id, sender_type, content, "gen")

    def save_doc_message(self, trade_id: int, sender_type: str, content: str):
        """무역 플로우 메시지 저장 (sender_type: 'U' 또는 'A')"""
        self._save_message(trade_id, sender_type, content, "trade")

    def _save_message(self, chat_id: int, sender_type: str, content: str, chat_type: str):
        """메시지 저장 (공통 로직)"""
        # In-Memory 저장
        if chat_id not in self.memory_store:
            self.memory_store[chat_id] = {"type": chat_type, "messages": [], "summaries": []}

        self.memory_store[chat_id]["messages"].append({
            "sender_type": sender_type,
            "content": content,
            "created_at": datetime.now()
        })

        # MySQL 저장
        table = "gen_message" if chat_type == "gen" else "doc_message"
        id_col = "gen_chat_id" if chat_type == "gen" else "trade_id"

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                        INSERT INTO {table} ({id_col}, sender_type, content, created_at)
                        VALUES (%s, %s, %s, NOW())
                    """, (chat_id, sender_type, content))
                    conn.commit()
        except Exception as e:
            print(f"Warning: Failed to save message: {e}")

    # =====================================================================
    # 컨텍스트 생성
    # =====================================================================

    def get_gen_context(self, gen_chat_id: int, limit: int = 10) -> str:
        """일반 채팅 컨텍스트 생성"""
        if gen_chat_id not in self.memory_store:
            self._load_from_db(gen_chat_id, "gen")

        return self._build_context(gen_chat_id, limit)

    def get_trade_context(self, trade_id: int, limit: int = 10, include_documents: bool = True) -> str:
        """무역 플로우 컨텍스트 생성"""
        if trade_id not in self.memory_store:
            self._load_from_db(trade_id, "trade", include_documents)

        return self._build_context(trade_id, limit, include_documents)

    def _build_context(self, chat_id: int, limit: int = None, include_documents: bool = False) -> str:
        """
        컨텍스트 문자열 생성 (instructions용)

        Args:
            chat_id: 채팅방 ID
            limit: (미사용) 최근 대화는 messages로 전달되므로 여기선 요약만 포함
            include_documents: 문서 정보 포함 여부

        Returns:
            요약 및 문서 정보 문자열
        """
        if chat_id not in self.memory_store:
            return ""

        data = self.memory_store[chat_id]
        lines = []

        # 무역 플로우 정보
        if data["type"] == "trade" and "flow_info" in data and data["flow_info"]:
            lines.append("=== 무역 플로우 정보 ===")
            lines.append(f"제목: {data['flow_info']['title']}")
            lines.append(f"시작일: {data['flow_info']['created_at']}\n")

        # 문서 정보
        if include_documents and "documents" in data:
            lines.append("=== 작성된 문서 ===")
            for doc in data["documents"]:
                lines.append(f"\n[{doc['template_name']}]")
                if doc.get('version_title'):
                    lines.append(f"  버전: {doc['version_title']}")
                if doc.get('content'):
                    preview = doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                    lines.append(f"  내용: {preview}")
            lines.append("")

        # 요약 (요약된 부분만 표시)
        if data["summaries"]:
            lines.append("=== 이전 대화 요약 ===")
            for i, summary in enumerate(data["summaries"], 1):
                lines.append(f"[요약 {i}] (메시지 {summary['message_count']}개) {summary['content']}")
            lines.append("")
            lines.append("※ 최근 대화는 messages로 전달됨")

        lines.append("===================")
        return "\n".join(lines)

    # =====================================================================
    # DB 로드
    # =====================================================================

    def _load_from_db(self, chat_id: int, chat_type: str, include_documents: bool = False):
        """DB에서 데이터 로드"""
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    # 메시지 로드
                    msg_table = "gen_message" if chat_type == "gen" else "doc_message"
                    id_col = "gen_chat_id" if chat_type == "gen" else "trade_id"

                    cursor.execute(f"""
                        SELECT sender_type, content, created_at
                        FROM {msg_table}
                        WHERE {id_col} = %s
                        ORDER BY created_at ASC
                    """, (chat_id,))
                    messages = cursor.fetchall()

                    # 요약 로드
                    summary_table = "gen_chat_summary" if chat_type == "gen" else "trade_flow_summary"
                    cursor.execute(f"""
                        SELECT summary, message_count, created_at
                        FROM {summary_table}
                        WHERE {id_col} = %s
                        ORDER BY created_at ASC
                    """, (chat_id,))
                    summaries = cursor.fetchall()

                    # 무역 플로우 추가 정보
                    flow_info = None
                    documents = []
                    if chat_type == "trade":
                        cursor.execute("SELECT title, created_at FROM trade_flow WHERE trade_id = %s", (chat_id,))
                        flow_info = cursor.fetchone()

                        if include_documents:
                            cursor.execute("""
                                SELECT d.doc_id, dt.template_name, dv.title as version_title, dv.content, dv.created_at
                                FROM document d
                                LEFT JOIN doc_template dt ON d.template_id = dt.template_id
                                LEFT JOIN doc_version dv ON d.doc_id = dv.doc_id
                                WHERE d.trade_id = %s
                                ORDER BY d.created_at, dv.created_at DESC
                            """, (chat_id,))
                            documents = cursor.fetchall()

            # In-Memory 저장
            if messages or summaries or flow_info:
                self.memory_store[chat_id] = {
                    "type": chat_type,
                    "messages": [{"sender_type": m['sender_type'], "content": m['content'], "created_at": m['created_at']} for m in messages],
                    "summaries": [{"content": s['summary'], "message_count": s['message_count'], "created_at": s['created_at']} for s in summaries],
                }
                if chat_type == "trade":
                    self.memory_store[chat_id]["flow_info"] = flow_info
                    self.memory_store[chat_id]["documents"] = documents

        except Exception as e:
            print(f"Warning: Failed to load from DB: {e}")

    # =====================================================================
    # 요약 서비스
    # =====================================================================

    def trigger_gen_summary(self, gen_chat_id: int):
        """일반 채팅 요약 트리거"""
        self._trigger_summary(gen_chat_id, "gen")

    def trigger_trade_summary(self, trade_id: int):
        """무역 플로우 요약 트리거"""
        self._trigger_summary(trade_id, "trade")

    def _trigger_summary(self, chat_id: int, chat_type: str):
        """요약 트리거 (공통)"""
        if not self._should_summarize(chat_id):
            return

        threading.Thread(
            target=self._generate_summary,
            args=(chat_id, chat_type),
            daemon=True
        ).start()

    def _should_summarize(self, chat_id: int) -> bool:
        """요약 필요 여부"""
        if chat_id not in self.memory_store:
            return False

        data = self.memory_store[chat_id]
        total = len(data["messages"])
        summarized = sum(s["message_count"] for s in data["summaries"])
        return (total - summarized) >= 20

    def _generate_summary(self, chat_id: int, chat_type: str):
        """요약 생성 (백그라운드)"""
        try:
            if chat_id not in self.memory_store:
                return

            data = self.memory_store[chat_id]
            summarized_count = sum(s["message_count"] for s in data["summaries"])
            messages = data["messages"][summarized_count:summarized_count + 20]

            if not messages:
                return

            # GPT 요약
            conversation_text = "\n\n".join([
                f"{'사용자' if m['sender_type'] == 'U' else '어시스턴트'}: {m['content']}"
                for m in messages
            ])

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "당신은 대화를 간결하게 요약하는 전문가입니다."},
                    {"role": "user", "content": f"다음 대화 내용을 간결하게 요약해주세요.\n\n{conversation_text}\n\n요약:"}
                ],
                temperature=0.3,
                max_tokens=500
            )
            summary_text = response.choices[0].message.content.strip()

            # 저장
            summary = {"content": summary_text, "message_count": len(messages), "created_at": datetime.now()}
            data["summaries"].append(summary)

            # MySQL 저장
            table = "gen_chat_summary" if chat_type == "gen" else "trade_flow_summary"
            id_col = "gen_chat_id" if chat_type == "gen" else "trade_id"

            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"""
                        INSERT INTO {table} ({id_col}, summary, message_count, created_at)
                        VALUES (%s, %s, %s, NOW())
                    """, (chat_id, summary_text, len(messages)))
                    conn.commit()

            print(f"✅ 요약 생성 완료: {chat_id}")

        except Exception as e:
            print(f"Warning: Failed to generate summary: {e}")

    # =====================================================================
    # 채팅방 / 무역 플로우 조회
    # =====================================================================

    def get_user_gen_chats(self, user_id: int) -> list:
        """
        사용자의 일반 채팅방 목록 조회

        Args:
            user_id: 사용자 ID

        Returns:
            [{"gen_chat_id": "...", "title": "...", "created_at": "..."}, ...]
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT gen_chat_id, title, created_at
                        FROM gen_chat
                        WHERE user_id = %s
                        ORDER BY created_at DESC
                    """, (user_id,))
                    return cursor.fetchall()
        except Exception as e:
            print(f"Warning: Failed to get user chats: {e}")
            return []

    def get_user_trade_flows(self, user_id: int) -> list:
        """
        사용자의 워크플로우 목록 조회

        Args:
            user_id: 사용자 ID

        Returns:
            [{"trade_id": "...", "title": "...", "created_at": "..."}, ...]
        """
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT trade_id, title, created_at
                        FROM trade_flow
                        WHERE user_id = %s
                        ORDER BY created_at DESC
                    """, (user_id,))
                    return cursor.fetchall()
        except Exception as e:
            print(f"Warning: Failed to get user workflows: {e}")
            return []

    # =====================================================================
    # 채팅방 / 무역 플로우 생성
    # =====================================================================

    def create_gen_chat(self, first_message: str = None, title: str = None, user_id: int = None) -> int:
        """
        일반 채팅방 생성 (AUTO_INCREMENT로 ID 자동 생성)

        Args:
            first_message: 사용자의 첫 질의 (제목 자동 생성용)
            title: 채팅방 제목 (지정 시 우선 사용, 최대 10자)
            user_id: 사용자 ID (선택)

        Returns:
            생성된 gen_chat_id (BIGINT)

        Raises:
            ValueError: title과 first_message 모두 없는 경우
        """
        if not title and not first_message:
            raise ValueError("title 또는 first_message 중 하나는 필수입니다")

        # 제목 생성 또는 사용
        if title:
            final_title = title[:self.MAX_GEN_CHAT_TITLE_LENGTH]
        else:
            final_title = self._generate_chat_title(first_message)

        # 중복 체크 및 유니크한 제목 생성
        unique_title = self._ensure_unique_gen_chat_title(final_title, user_id)

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO gen_chat (title, user_id, created_at)
                        VALUES (%s, %s, NOW())
                    """, (unique_title, user_id))
                    gen_chat_id = cursor.lastrowid
                    conn.commit()

            return gen_chat_id
        except Exception as e:
            print(f"Warning: Failed to create gen_chat: {e}")
            raise

    def create_trade_flow(self, title: str = None, user_id: int = None) -> int:
        """
        무역 플로우 생성 (AUTO_INCREMENT로 ID 자동 생성, 중복 시 untitle, untitle-2 형식으로 처리)

        Args:
            title: 무역 플로우 제목 (없으면 "untitle" 사용, 최대 60자)
            user_id: 사용자 ID (선택)

        Returns:
            생성된 trade_id (BIGINT)
        """
        # 제목이 없으면 기본값 사용
        if not title:
            title = self.DEFAULT_WORKFLOW_TITLE

        # 최대 길이 제한 (SQL에서 VARCHAR(60))
        if len(title) > self.MAX_TRADE_FLOW_TITLE_LENGTH:
            title = title[:self.MAX_TRADE_FLOW_TITLE_LENGTH]

        # 중복 체크 및 유니크한 제목 생성
        unique_title = self._ensure_unique_trade_flow_title(title, user_id)

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        INSERT INTO trade_flow (title, user_id, created_at)
                        VALUES (%s, %s, NOW())
                    """, (unique_title, user_id))
                    trade_id = cursor.lastrowid
                    conn.commit()

            return trade_id
        except Exception as e:
            print(f"Warning: Failed to create trade_flow: {e}")
            raise    

    # =====================================================================
    # 최근 10턴 조회 (모델 저장용)
    # =====================================================================

    def get_recent_turns_for_model(self, chat_id: int, chat_type: str = "gen", limit: int = 10) -> list:
        """
        최근 N개 턴을 모델에 전달할 형식으로 반환 (메모리에서 우선, 없으면 DB 로드)

        Args:
            chat_id: 채팅방 ID (gen_chat_id 또는 trade_id)
            chat_type: "gen" 또는 "trade"
            limit: 최대 턴 수 (기본 10턴 = 20개 메시지)

        Returns:
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        """
        # In-Memory 캐시에 없으면 DB에서 로드
        if chat_id not in self.memory_store:
            self._load_from_db(chat_id, chat_type, include_documents=False)

        # 캐시에도 없으면 빈 리스트 반환
        if chat_id not in self.memory_store:
            return []

        data = self.memory_store[chat_id]
        messages = data["messages"]

        # 최근 N턴 = 최근 2*N개 메시지
        recent_messages = messages[-(limit * 2):]

        # OpenAI 형식으로 변환
        result = []
        for msg in recent_messages:
            role = "user" if msg["sender_type"] == "U" else "assistant"
            result.append({
                "role": role,
                "content": msg["content"]
            })

        return result

    # =====================================================================
    # 편의 메서드 (save_turn)
    # =====================================================================

    def save_gen_turn(self, gen_chat_id: int, user_message: str, assistant_message: str):
        """일반 채팅 턴 저장 (U + A)"""
        self.save_gen_message(gen_chat_id, "U", user_message)
        self.save_gen_message(gen_chat_id, "A", assistant_message)
        self.trigger_gen_summary(gen_chat_id)

    def save_trade_turn(self, trade_id: int, user_message: str, assistant_message: str):
        """무역 플로우 턴 저장 (U + A)"""
        self.save_doc_message(trade_id, "U", user_message)
        self.save_doc_message(trade_id, "A", assistant_message)
        self.trigger_trade_summary(trade_id)
