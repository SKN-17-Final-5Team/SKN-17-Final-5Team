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
    # 메시지 저장
    # =====================================================================

    def save_gen_message(self, gen_chat_id: str, sender_type: str, content: str):
        """일반 채팅 메시지 저장"""
        self._save_message(gen_chat_id, sender_type, content, "gen")

    def save_doc_message(self, trade_id: str, sender_type: str, content: str):
        """무역 플로우 메시지 저장"""
        self._save_message(trade_id, sender_type, content, "trade")

    def _save_message(self, chat_id: str, sender_type: str, content: str, chat_type: str):
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

    def get_gen_context(self, gen_chat_id: str, limit: int = 10) -> str:
        """일반 채팅 컨텍스트 생성"""
        if gen_chat_id not in self.memory_store:
            self._load_from_db(gen_chat_id, "gen")

        return self._build_context(gen_chat_id, limit)

    def get_trade_context(self, trade_id: str, limit: int = 10, include_documents: bool = True) -> str:
        """무역 플로우 컨텍스트 생성"""
        if trade_id not in self.memory_store:
            self._load_from_db(trade_id, "trade", include_documents)

        return self._build_context(trade_id, limit, include_documents)

    def _build_context(self, chat_id: str, limit: int, include_documents: bool = False) -> str:
        """컨텍스트 문자열 생성"""
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

        # 요약
        if data["summaries"]:
            lines.append("=== 이전 대화 요약 ===")
            for i, summary in enumerate(data["summaries"], 1):
                lines.append(f"[요약 {i}] {summary['content']}")
            lines.append("")

        # 최근 메시지
        recent_messages = data["messages"][-limit:]
        if recent_messages:
            lines.append("=== 최근 대화 기록 ===")
            for msg in recent_messages:
                role = msg['sender_type'].upper()
                lines.append(f"[{role}]: {msg['content']}")

        lines.append("===================")
        return "\n".join(lines)

    # =====================================================================
    # DB 로드
    # =====================================================================

    def _load_from_db(self, chat_id: str, chat_type: str, include_documents: bool = False):
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

    def trigger_gen_summary(self, gen_chat_id: str):
        """일반 채팅 요약 트리거"""
        self._trigger_summary(gen_chat_id, "gen")

    def trigger_trade_summary(self, trade_id: str):
        """무역 플로우 요약 트리거"""
        self._trigger_summary(trade_id, "trade")

    def _trigger_summary(self, chat_id: str, chat_type: str):
        """요약 트리거 (공통)"""
        if not self._should_summarize(chat_id):
            return

        threading.Thread(
            target=self._generate_summary,
            args=(chat_id, chat_type),
            daemon=True
        ).start()

    def _should_summarize(self, chat_id: str) -> bool:
        """요약 필요 여부"""
        if chat_id not in self.memory_store:
            return False

        data = self.memory_store[chat_id]
        total = len(data["messages"])
        summarized = sum(s["message_count"] for s in data["summaries"])
        return (total - summarized) >= 20

    def _generate_summary(self, chat_id: str, chat_type: str):
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
                f"{'사용자' if m['sender_type'] == 'user' else '어시스턴트'}: {m['content']}"
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
    # 유틸리티
    # =====================================================================

    def get_stats(self, chat_id: str) -> Dict:
        """통계 정보 (디버깅용)"""
        if chat_id not in self.memory_store:
            # DB에서 로드 시도
            try:
                with self._get_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT COUNT(*) as cnt FROM gen_message WHERE gen_chat_id = %s", (chat_id,))
                        if cursor.fetchone()['cnt'] > 0:
                            self._load_from_db(chat_id, "gen")
                        else:
                            cursor.execute("SELECT COUNT(*) as cnt FROM doc_message WHERE trade_id = %s", (chat_id,))
                            if cursor.fetchone()['cnt'] > 0:
                                self._load_from_db(chat_id, "trade")
            except:
                pass

        if chat_id not in self.memory_store:
            return {"total_messages": 0, "summary_count": 0, "summarized_messages": 0}

        data = self.memory_store[chat_id]
        return {
            "total_messages": len(data["messages"]),
            "summary_count": len(data["summaries"]),
            "summarized_messages": sum(s["message_count"] for s in data["summaries"])
        }

    def clear_memory(self, chat_id: str):
        """In-Memory 캐시 제거"""
        if chat_id in self.memory_store:
            del self.memory_store[chat_id]

    # =====================================================================
    # 편의 메서드 (save_turn)
    # =====================================================================

    def save_gen_turn(self, gen_chat_id: str, user_message: str, assistant_message: str):
        """일반 채팅 턴 저장 (user + assistant)"""
        self.save_gen_message(gen_chat_id, "user", user_message)
        self.save_gen_message(gen_chat_id, "assistant", assistant_message)
        self.trigger_gen_summary(gen_chat_id)

    def save_trade_turn(self, trade_id: str, user_message: str, assistant_message: str):
        """무역 플로우 턴 저장 (user + assistant)"""
        self.save_doc_message(trade_id, "user", user_message)
        self.save_doc_message(trade_id, "assistant", assistant_message)
        self.trigger_trade_summary(trade_id)
