# 🚀 무역 AI 코파일럿 - 프로덕션 메모리 설정 가이드

> **MySQL + Qdrant + S3 구조로 업그레이드**

---

## 📊 최종 아키텍처

```
┌──────────────────────────────────────────────┐
│       무역 AI 코파일럿 서비스                 │
├──────────────────────────────────────────────┤
│                                              │
│  [일반 질의]            [문서 플로우]         │
│  gen_chat_id           trade_id              │
│      ↓                     ↓                  │
│  ┌────────────┐      ┌─────────────┐        │
│  │ 일반 채팅   │      │ 문서 작성    │        │
│  │ 문서 업로드 │      │ + RAG 검색  │        │
│  └────────────┘      └─────────────┘        │
│       ↓                     ↓                 │
├──────────────────────────────────────────────┤
│              메모리 레이어                     │
│                                              │
│  📌 단기 메모리 (최근 대화)                   │
│     └─ MySQL (thread_id별 State 저장)       │
│                                              │
│  📚 장기 메모리 (요약된 대화)                 │
│     └─ Qdrant (요약 임베딩, 의미 검색)       │
│                                              │
│  📄 문서 저장                                 │
│     ├─ S3 (원본 파일)                        │
│     └─ Qdrant (문서 임베딩, RAG 검색)        │
└──────────────────────────────────────────────┘
```

---

## 🗄️ 1. MySQL 설정

### Docker로 MySQL 실행
```bash
docker run -d \
  --name trade-mysql \
  -p 3306:3306 \
  -e MYSQL_ROOT_PASSWORD=yourpassword \
  -e MYSQL_DATABASE=memory_db \
  mysql:8.0
```

### 데이터베이스 생성 (수동 설정 시)
```sql
CREATE DATABASE IF NOT EXISTS memory_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE memory_db;

-- Checkpoints 테이블은 mysql_checkpointer.py가 자동 생성함
```

### 환경 변수 설정 (.env)
```env
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=yourpassword
MYSQL_DATABASE=memory_db
```

---

## 🔍 2. Qdrant 설정

### Docker로 Qdrant 실행
```bash
docker run -d \
  --name trade-qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest
```

### 컬렉션 생성 (자동 or 수동)

**자동 생성** (코드에서):
```python
from services.qdrant_service import QdrantService

qdrant = QdrantService(host="localhost", port=6333)
qdrant.create_collection(vector_size=768)  # KoSimCSE 차원
```

**수동 생성** (Qdrant UI):
1. 브라우저에서 `http://localhost:6333/dashboard` 접속
2. Collections → Create Collection
3. 이름: `trade_documents` (또는 원하는 이름)
4. Vector 차원: `768`
5. Distance: `Cosine`

### 환경 변수 설정 (.env)
```env
QDRANT_HOST=localhost
QDRANT_PORT=6333
# QDRANT_API_KEY=  # Qdrant Cloud 사용 시
```

---

## ☁️ 3. S3 설정

### AWS S3 버킷 생성
```bash
# AWS CLI로 버킷 생성
aws s3 mb s3://trade-ai-documents --region ap-northeast-2
```

### IAM 사용자 생성 및 권한 설정
1. AWS Console → IAM → Users → Add User
2. Programmatic access 선택
3. 권한: `S3FullAccess` (또는 특정 버킷만)
4. Access Key ID / Secret Access Key 복사

### 환경 변수 설정 (.env)
```env
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_S3_BUCKET=trade-ai-documents
AWS_REGION=ap-northeast-2
```

---

## 📦 4. 의존성 설치

```bash
py -m pip install -r requirements.txt
```

**설치되는 주요 패키지:**
- `mysql-connector-python` - MySQL 연결
- `qdrant-client` - Qdrant 벡터 DB
- `transformers`, `torch` - KoSimCSE 임베딩
- `boto3` - AWS S3
- `langgraph` - LangGraph 워크플로우

---

## 🚀 5. 코드 사용 예시

### main_langgraph.py 수정 예시

```python
import asyncio
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# MySQL Checkpointer
from services.mysql_checkpointer import create_mysql_checkpointer

# Qdrant + Embedding
from services.qdrant_service import QdrantService
from services.embedding_service import get_embedding_service

# Memory Service
from services.memory_service import MemoryService
from services.graph_workflow import TradeAgentWorkflow

async def main():
    # ===== MySQL Checkpointer =====
    mysql_checkpointer = create_mysql_checkpointer(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", 3306)),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DATABASE", "memory_db")
    )

    # ===== Qdrant 초기화 =====
    qdrant = QdrantService(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333))
    )

    # 컬렉션 생성 (이미 존재하면 skip)
    qdrant.create_collection(vector_size=768)

    # ===== 임베딩 서비스 =====
    embedder = get_embedding_service()

    # ===== Memory Service (Qdrant 연동) =====
    memory_service = MemoryService(
        checkpointer=mysql_checkpointer,
        qdrant_service=qdrant,
        embedding_service=embedder
    )

    # ===== Workflow 생성 =====
    workflow = TradeAgentWorkflow(mysql_checkpointer)

    # ===== 세션 ID =====
    gen_chat_id = input("일반 대화 세션 ID: ") or "gen_chat_test"
    trade_id = input("무역 문서 세션 ID: ") or "trade_test"

    # ===== 대화 시작 =====
    thread_config = {"configurable": {"thread_id": gen_chat_id}}

    while True:
        question = input("\n질문: ")
        if question.lower() in ["exit", "quit"]:
            break

        # Workflow 실행
        result = await workflow.graph.ainvoke(
            {"messages": [{"role": "user", "content": question}]},
            config=thread_config
        )

        print(f"답변: {result['messages'][-1]['content']}")

    print("\n프로그램 종료")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 📋 6. 환경 변수 전체 예시 (.env)

```env
# OpenAI
OPENAI_API_KEY=sk-...

# MySQL
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=yourpassword
MYSQL_DATABASE=memory_db

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# AWS S3
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_S3_BUCKET=trade-ai-documents
AWS_REGION=ap-northeast-2
```

---

## 🧪 7. 테스트 방법

### 1단계: 서비스 실행 확인
```bash
# MySQL 확인
docker ps | grep trade-mysql

# Qdrant 확인
curl http://localhost:6333/collections

# S3 확인
aws s3 ls s3://trade-ai-documents
```

### 2단계: 임베딩 테스트
```python
from services.embedding_service import get_embedding_service

embedder = get_embedding_service()
embedding = embedder.encode("테스트 문장")
print(f"임베딩 차원: {embedding.shape}")  # (768,)
```

### 3단계: Qdrant 저장/검색 테스트
```python
from services.qdrant_service import QdrantService
from services.embedding_service import get_embedding_service

qdrant = QdrantService()
embedder = get_embedding_service()

# 저장
embedding = embedder.encode("HS CODE 조회 방법")
qdrant.add_document(
    embedding=embedding,
    text="HS CODE 조회 방법",
    metadata={"session_id": "test", "type": "long_term"}
)

# 검색
query_embedding = embedder.encode("HS CODE")
results = qdrant.search_similar(query_embedding, limit=5)
print(results)
```

### 4단계: 전체 시스템 테스트
```bash
py main_langgraph.py
```

---

## 📊 8. 데이터 흐름

### 일반 질의 (gen_chat_id)
```
사용자 질문
    ↓
[LLM] 답변 생성
    ↓
[MySQL] State 저장 (thread_id: gen_chat_id)
    ↓
10턴 초과?
    ↓
[LLM] 요약 생성
    ↓
[Embedder] 요약 → 임베딩 변환
    ↓
[Qdrant] 임베딩 저장 (filter: session_id=gen_chat_id)
```

### 문서 플로우 (trade_id)
```
사용자가 문서 업로드
    ↓
[S3] 원본 저장
    ↓
[LLM] 문서 청크 분할
    ↓
[Embedder] 청크 → 임베딩
    ↓
[Qdrant] 문서 임베딩 저장 (filter: doc_type=invoice)
    ↓
사용자 질문 (예: "계약서 작성해줘")
    ↓
[Embedder] 질문 → 임베딩
    ↓
[Qdrant] 유사 문서 검색 (RAG)
    ↓
[LLM] 문서 기반 답변 생성
    ↓
[MySQL] State 저장 (thread_id: trade_id)
```

---

## ⚠️ 9. 주의 사항

### MySQL
- **연결 수 제한**: MySQL max_connections 확인 (`SHOW VARIABLES LIKE 'max_connections';`)
- **백업**: 정기적인 mysqldump 설정
- **인덱싱**: thread_id에 인덱스 (자동 생성됨)

### Qdrant
- **용량**: 벡터 데이터는 메모리 많이 사용 (모니터링 필요)
- **백업**: Qdrant snapshot 기능 활용
- **스케일링**: 필요 시 Qdrant Cloud 고려

### S3
- **비용**: 저장 용량 및 API 요청 수 모니터링
- **라이프사이클**: 오래된 파일 자동 삭제 정책 설정
- **보안**: Bucket policy로 접근 제한

---

## 🔧 10. 문제 해결

### MySQL 연결 실패
```
Error: Can't connect to MySQL server
```
**해결:**
```bash
# MySQL 상태 확인
docker logs trade-mysql

# 재시작
docker restart trade-mysql
```

### Qdrant 컬렉션 생성 실패
```
Error: Collection already exists
```
**해결:** 정상 (이미 생성됨), 무시하고 진행

### 임베딩 모델 로드 실패
```
Error: Cannot load model BM-K/KoSimCSE-roberta
```
**해결:**
```bash
# transformers 재설치
py -m pip install --upgrade transformers torch
```

### S3 업로드 실패
```
Error: Access Denied
```
**해결:** AWS 자격 증명 확인, IAM 권한 확인

---

## 📞 11. 추가 정보

- **메모리 모듈 가이드**: [MEMORY_MODULE_GUIDE.md](MEMORY_MODULE_GUIDE.md)
- **빠른 시작**: [MEMORY_SETUP.md](MEMORY_SETUP.md)
- **LangGraph 아키텍처**: [LANGGRAPH_ARCHITECTURE.md](LANGGRAPH_ARCHITECTURE.md)

---

**마지막 업데이트:** 2025-01-20
**프로덕션 준비 완료** ✅
