# 🧠 메모리 모듈 가이드

> **작성자를 위한 메모**: 이 모듈은 RAG 팀과 독립적으로 작동하며, 대화 이력 관리를 담당합니다.

---

## 📋 목차

1. [메모리 모듈이란?](#메모리-모듈이란)
2. [전체 아키텍처](#전체-아키텍처)
3. [핵심 개념](#핵심-개념)
4. [서비스별 상세 설명](#서비스별-상세-설명)
5. [사용 방법](#사용-방법)
6. [테스트 방법](#테스트-방법)
7. [RAG 팀과의 통합](#rag-팀과의-통합)

---

## 메모리 모듈이란?

**메모리 모듈**은 에이전트의 **대화 이력**을 관리하는 독립적인 시스템입니다.

### 🎯 핵심 목적

1. **단기 메모리**: 최근 N턴의 대화를 **전체 내용**으로 저장
2. **장기 메모리**: 오래된 대화를 **요약**하여 저장 (메모리 효율)
3. **세션 관리**: 일반 대화(`gen_chat_id`)와 문서 플로우(`trade_id`)를 분리 관리

### 🚫 RAG와의 차이점

| 구분 | 메모리 모듈 | RAG 모듈 |
|------|------------|---------|
| **목적** | 대화 이력 관리 | 문서 검색 |
| **저장 대상** | 사용자-AI 대화 | 무역 문서 |
| **검색 방식** | 최근 N턴 조회 | 벡터 유사도 검색 |
| **저장소** | JSON/SQLite | Qdrant/Elasticsearch |

---

## 전체 아키텍처

```
📦 메모리 모듈
│
├── 🔵 단기 메모리 (Short-term Memory)
│   ├── 저장소: SQL (LangGraph) / 메모리 (OpenAI SDK)
│   ├── 내용: 최근 10턴의 전체 대화 내용
│   └── 용도: 현재 대화의 컨텍스트 유지
│
├── 🟢 장기 메모리 (Long-term Memory)
│   ├── 저장소: JSON 파일
│   ├── 내용: 요약된 과거 대화
│   └── 용도: 메모리 효율적 장기 보존
│
├── 🟡 벡터 검색 (선택 사항)
│   ├── 저장소: Qdrant
│   ├── 내용: 대화 임베딩
│   └── 용도: 의미 기반 과거 대화 검색
│
└── 🔴 문서 저장 (RAG 팀 담당)
    ├── 저장소: S3
    ├── 내용: 무역 문서 원본
    └── 용도: RAG 문서 검색
```

---

## 핵심 개념

### 1️⃣ 단기 메모리 (Short-term Memory)

**최근 대화를 전부 기억**합니다.

```python
# 예시: 최근 10턴 대화
[
    {"role": "user", "content": "HS CODE가 뭐야?"},
    {"role": "assistant", "content": "HS CODE는 국제 무역에서..."},
    {"role": "user", "content": "어떻게 조회해?"},
    {"role": "assistant", "content": "관세청 사이트에서..."},
    # ... (최근 10턴까지)
]
```

**특징:**
- ✅ 전체 대화 내용 보존
- ✅ 빠른 접근
- ❌ 메모리 사용량 증가 (턴이 많아지면)

**저장 위치:**
- LangGraph: `memory/checkpoints.db` (SQLite)
- OpenAI SDK: 메모리 (프로그램 종료 시 삭제)

---

### 2️⃣ 장기 메모리 (Long-term Memory)

**오래된 대화를 요약**하여 저장합니다.

```python
# 예시: 요약된 장기 메모리
[
    {
        "summary": "사용자는 HS CODE 조회 방법을 학습했으며, 관세청 사이트 사용법을 익혔다.",
        "timestamp": "2025-01-15T10:30:00"
    },
    {
        "summary": "사용자는 수출입 신고서 작성 절차에 대해 질문했으며, 필수 서류 목록을 받았다.",
        "timestamp": "2025-01-15T11:00:00"
    }
]
```

**특징:**
- ✅ 메모리 효율적 (요약본만 저장)
- ✅ 영구 보존
- ❌ 원본 대화 내용은 사라짐

**저장 위치:**
- `memory/recall_memories_{session_id}.json`

**자동 요약 트리거:**
- 단기 메모리가 10턴(20개 메시지)을 초과하면 자동 실행
- GPT-4o-mini로 요약 생성

---

### 3️⃣ 세션 분리 (gen_chat_id vs trade_id)

**일반 대화**와 **문서 플로우**를 분리합니다.

```
🔹 일반 대화 (GEN_CHAT)
├── gen_chat_id: "gen_chat_abc123"
├── 단기 메모리: SQLite (thread_id: gen_chat_abc123)
└── 장기 메모리: recall_memories_gen_chat_abc123.json

🔸 문서 플로우 (TRADE_FLOW)
├── trade_id: "trade_xyz789"
├── 단기 메모리: SQLite (thread_id: trade_xyz789)
└── 장기 메모리: recall_memories_trade_xyz789.json
```

**왜 분리하나요?**
- 일반 대화: 사용자 프로필, 선호도 저장
- 문서 플로우: 특정 무역 업무(예: 인보이스 작성)에 집중

---

## 서비스별 상세 설명

### 📄 `memory_manager.py` (OpenAI Agents SDK)

**역할:** 간단한 메모리 관리 (main.py용)

**주요 메서드:**
```python
# 단기 메모리에 추가 (10턴 초과 시 자동 요약)
memory_manager.add_to_short_term(role="user", content="안녕하세요")

# 단기 메모리 조회
context = memory_manager.get_short_term_context(last_n=5)  # 최근 5턴

# 장기 메모리 조회
long_term_context = memory_manager.get_long_term_context()
```

**파일 구조:**
```python
class MemoryManager:
    def __init__(self, long_term_memory_path, short_term_limit=10):
        self.short_term_memory = []  # 리스트
        self.long_term_memory = []   # JSON 파일에서 로드
        self.llm = ChatOpenAI(model="gpt-4o-mini")  # 요약용

    def _summarize_and_archive(self):
        # 단기 메모리의 절반을 요약 → 장기 메모리로 이동
```

---

### 📄 `memory_service.py` (LangGraph)

**역할:** LangGraph 기반 메모리 관리 (main_langgraph.py용)

**주요 메서드:**
```python
# 장기 메모리 로드
recall_memories = memory_service.load_recall_memories(session_id)

# 자동 요약 확인 및 실행
updated_state = memory_service.check_and_summarize(state, session_id)

# 장기 메모리 저장
memory_service.save_recall_memories(session_id, recall_memories)
```

**LangGraph State와 통합:**
```python
class AgentState(TypedDict):
    messages: List[Dict]  # 단기 메모리 (LangGraph 자동 관리)
    recall_memories: List[Dict]  # 장기 메모리 (수동 관리)
```

---

### 📄 ~~`embedding_service.py`~~ (비활성화됨)

**상태:** `embedding_service.py.disabled`로 변경됨

**역할:** 텍스트를 벡터로 변환 (Qdrant 검색용)

**활성화 방법:**
```bash
# 나중에 필요하면 이렇게 활성화
mv services/embedding_service.py.disabled services/embedding_service.py
```

**주의:** 현재는 사용하지 않습니다. 기본 메모리 모듈과 무관합니다.

---

### 📄 ~~`qdrant_service.py`~~ (비활성화됨)

**상태:** `qdrant_service.py.disabled`로 변경됨

**역할:** 벡터 DB 관리 (의미 기반 대화 검색)

**활성화 방법:**
```bash
# 나중에 필요하면 이렇게 활성화
mv services/qdrant_service.py.disabled services/qdrant_service.py
```

**주의:** 현재는 사용하지 않습니다. 기본 메모리 모듈과 무관합니다.

---

### 📄 ~~`s3_service.py`~~ (비활성화됨)

**상태:** `s3_service.py.disabled`로 변경됨

**역할:** 문서 원본을 S3에 저장/로드 (RAG 팀 전용)

**활성화 방법:**
```bash
# RAG 팀에서 필요하면 활성화
mv services/s3_service.py.disabled services/s3_service.py
```

**주의:** 이건 **RAG 팀**이 나중에 사용할 부분입니다. 메모리 모듈과는 독립적입니다.

---

## 사용 방법

### 🔹 방법 1: OpenAI Agents SDK (main.py)

```python
from services.memory_manager import MemoryManager

# 메모리 매니저 생성
memory = MemoryManager(
    long_term_memory_path="memory/long_term_memory.json",
    short_term_limit=10
)

# 대화 추가
memory.add_to_short_term(role="user", content="HS CODE가 뭐야?")
memory.add_to_short_term(role="assistant", content="HS CODE는...")

# 컨텍스트 가져오기
context = memory.get_short_term_context()  # 최근 전체
context = memory.get_short_term_context(last_n=5)  # 최근 5턴
```

**특징:**
- 간단한 구조
- 프로그램 종료 시 단기 메모리 삭제
- 장기 메모리만 JSON으로 보존

---

### 🔹 방법 2: LangGraph (main_langgraph.py)

```python
from services.graph_workflow import TradeAgentWorkflow
from services.graph_state import create_initial_state
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# SQLite 체크포인터 생성
async with AsyncSqliteSaver.from_conn_string("memory/checkpoints.db") as checkpointer:
    # Workflow 생성
    workflow = TradeAgentWorkflow(checkpointer)

    # 초기 State 생성
    initial_state = create_initial_state(
        flow_type="GEN_CHAT",
        gen_chat_id="gen_chat_123",
        trade_id="trade_456"
    )

    # Thread 설정
    thread_config = {"configurable": {"thread_id": "gen_chat_123"}}

    # Workflow 실행
    result = await workflow.graph.ainvoke(initial_state, config=thread_config)
```

**특징:**
- 복잡한 구조
- 모든 대화가 SQLite에 영구 보존
- 세션 복원 가능

---

## 테스트 방법

### ✅ 테스트 1: 단기/장기 메모리 동작 확인

```bash
# main.py 실행 (간단한 테스트)
py main.py
```

**테스트 시나리오:**
1. 10턴 이상 대화하기
2. 자동 요약 메시지 확인: `✓ 대화 요약 완료: N개 메시지 → 장기 메모리`
3. 프로그램 종료 후 재실행
4. 장기 메모리가 로드되는지 확인

---

### ✅ 테스트 2: 세션 분리 확인

```bash
# main_langgraph.py 실행
py main_langgraph.py
```

**테스트 시나리오:**
1. 일반 대화 세션 ID 입력: `gen_chat_test`
2. 무역 문서 세션 ID 입력: `trade_test`
3. 일반 대화 몇 턴 진행
4. 종료 후 재실행, 같은 세션 ID 입력
5. 이전 대화가 복원되는지 확인

**확인 사항:**
- `memory/checkpoints.db`에 두 개의 thread_id 존재
- `memory/recall_memories_gen_chat_test.json` 생성
- `memory/recall_memories_trade_test.json` 생성 (문서 플로우 사용 시)

---

### ✅ 테스트 3: 임베딩 서비스 (선택 사항)

```python
# test_embedding.py
from services.embedding_service import get_embedding_service

embedder = get_embedding_service()

# 단일 텍스트
embedding = embedder.encode("HS CODE가 뭐야?")
print(f"차원: {embedding.shape}")  # (768,)

# 배치 텍스트
texts = ["안녕하세요", "무역 문서 검색"]
embeddings = embedder.encode(texts)
print(f"배치 차원: {embeddings.shape}")  # (2, 768)
```

**실행:**
```bash
py test_embedding.py
```

---

## RAG 팀과의 통합

### 🔗 통합 포인트

```
┌─────────────────┐         ┌─────────────────┐
│  메모리 모듈     │         │   RAG 모듈      │
│  (대화 관리)     │         │  (문서 검색)     │
├─────────────────┤         ├─────────────────┤
│ - 대화 이력     │         │ - 무역 문서     │
│ - 요약 생성     │         │ - 벡터 검색     │
│ - 세션 관리     │         │ - 리랭킹        │
└────────┬────────┘         └────────┬────────┘
         │                           │
         └──────────┬────────────────┘
                    │
              ┌─────▼──────┐
              │  Workflow   │
              │ (graph_     │
              │  workflow)  │
              └─────────────┘
```

### 📌 통합 시나리오

**1. 일반 대화 플로우 (GEN_CHAT)**
```python
# 메모리 모듈만 사용
State["messages"]  # 단기 메모리 (최근 대화)
State["recall_memories"]  # 장기 메모리 (요약)
```

**2. 문서 플로우 (TRADE_FLOW)**
```python
# 메모리 + RAG 연동
State["messages"]  # 단기 메모리
State["recall_memories"]  # 장기 메모리
State["retrieved_docs"]  # RAG 검색 결과 ← RAG 팀이 제공
```

**3. RAG 팀이 해야 할 일**
- `tools/search_tool.py`의 `search_trade_documents()` 함수 개선
- Qdrant/Elasticsearch에서 문서 검색
- 검색 결과를 `State["retrieved_docs"]`에 저장

**4. 메모리 팀이 해야 할 일**
- 대화 이력 관리 (이미 완료 ✅)
- 자동 요약 (이미 완료 ✅)
- 세션 분리 (이미 완료 ✅)

---

## 📊 디렉토리 구조

```
project/
├── services/
│   ├── memory_manager.py                # OpenAI SDK용 메모리 ✅
│   ├── memory_service.py                # LangGraph용 메모리 ✅
│   ├── graph_state.py                   # LangGraph State 정의 ✅
│   ├── graph_workflow.py                # LangGraph Workflow ✅
│   ├── reranker_service.py              # 리랭킹 (RAG 팀) ✅
│   ├── embedding_service.py.disabled    # 임베딩 (비활성화)
│   ├── qdrant_service.py.disabled       # 벡터 DB (비활성화)
│   └── s3_service.py.disabled           # S3 저장 (비활성화)
│
├── memory/
│   ├── checkpoints.db             # SQLite (단기 메모리)
│   ├── recall_memories_{id}.json  # 장기 메모리 (요약)
│   └── long_term_memory.json      # OpenAI SDK 장기 메모리
│
├── main.py                        # OpenAI SDK 실행
├── main_langgraph.py              # LangGraph 실행
└── MEMORY_MODULE_GUIDE.md         # 이 문서

📝 비활성화된 파일 (.disabled)
   - 나중에 필요하면 .disabled 확장자만 제거하면 됨
   - 현재는 메모리 모듈 핵심 기능만 사용
```

---

## 🎯 핵심 정리

### ✅ 메모리 모듈의 역할

1. **단기 메모리**: 최근 10턴 전체 보존
2. **장기 메모리**: 오래된 대화 요약 저장
3. **자동 요약**: 10턴 초과 시 GPT-4o-mini로 자동 요약
4. **세션 분리**: `gen_chat_id`와 `trade_id`로 독립 관리

### ❌ 메모리 모듈이 하지 않는 일

1. **문서 검색**: RAG 팀 담당
2. **리랭킹**: RAG 팀 담당
3. **S3 업로드**: RAG 팀 담당

### 🔧 테스트 체크리스트

- [ ] 10턴 이상 대화 → 자동 요약 확인
- [ ] 프로그램 재실행 → 장기 메모리 로드 확인
- [ ] 세션 분리 → `gen_chat_id`와 `trade_id` 독립 확인
- [ ] SQLite 저장 → `memory/checkpoints.db` 생성 확인
- [ ] JSON 저장 → `memory/recall_memories_{id}.json` 생성 확인

---

## 📞 문의

- 메모리 모듈 관련 질문: (작성자 이름)
- RAG 통합 관련 질문: RAG 팀에 문의

---

**마지막 업데이트:** 2025-01-20
