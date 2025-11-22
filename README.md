# Memory Module 가이드

EDARepository 기반 메모리 시스템을 RAG 에이전트에 붙이는 방법을 설명합니다. In-Memory LRU 캐시와 MySQL 2-Tier 저장소를 사용해 대화와 무역 문서 흐름을 영속화하고, GPT 요약으로 컨텍스트 길이를 관리합니다.

## 주요 특징
- In-Memory LRU(기본 1000개) + MySQL 영속화로 빠른 조회와 안전한 저장을 동시 지원
- 일반 대화(`gen`)와 무역 플로우(`trade`)를 동일한 API로 처리
- 20개 메시지마다 백그라운드 스레드로 자동 요약(`gpt-4o-mini`) 생성 및 저장
- 컨텍스트 빌더가 요약, 최근 메시지, 무역 문서/플로우 메타데이터를 합쳐 에이전트 프롬프트에 바로 사용
- Qdrant 기반 검색·Reranker 옵션과 독립적으로 동작하므로 필요 시 단독 사용 가능

## 수정된 부분
- memory_modules 디렉토리 추가
- trade_agent.py 메모리 파트 추가 (create_trade_agent)
- config.py 메모리 파트 추가
- pormpts/trade_instructions.txt 메모리 내용 추가
- memory_main.py 파일 추가 -- 실행 파일



## 환경 설정
`.env` 또는 환경 변수로 다음 값을 지정합니다.
- `OPENAI_API_KEY`: 요약 생성을 위한 OpenAI 키
- `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DATABASE`: 메모리 저장용 MySQL 연결 정보
- `QDRANT_URL`, `QDRANT_API_KEY`: RAG 검색에 필요 (메모리 모듈 자체는 직접 사용하지 않음)
- `RERANKER_API_URL`, `USE_RERANKER`, `USE_PER_QUERY_RERANK`: Reranker 사용 여부


## 빠른 실행
1) 패키지 설치: `pip install -r requirements.txt`  
2) 환경 변수 설정: `.env` 작성 후 `config.py`가 로드되도록 유지  
3) 메모리 저장소 준비: 위 스키마로 MySQL 생성  
4) CLI 실행: `python memory_main.py`  
   - Reranker 사용 여부 및 방식 선택 후 질문을 입력하면, 이전 대화 컨텍스트를 포함한 RAG 응답을 반환합니다.

## EDARepository 사용법 요약 (`memory_modules/repository.py`)
- `get_gen_context(gen_chat_id, limit=10)`: 일반 대화 컨텍스트 문자열 생성 (요약 + 최근 메시지)
- `get_trade_context(trade_id, limit=10, include_documents=True)`: 무역 플로우 컨텍스트 생성 (문서/플로우 메타 포함)
- `save_gen_turn(gen_chat_id, user_msg, assistant_msg)` / `save_trade_turn(...)`: 한 턴(user+assistant) 저장 후 요약 트리거
- `clear_memory(chat_id)`: In-Memory 캐시 제거 (DB는 유지)
- 요약 로직: 메시지 누적 20개 단위로 `_generate_summary`를 백그라운드 스레드에서 실행, DB와 캐시에 동시 반영

## 운영 팁
- 요약 단위(10개 턴)를 바꾸고 싶다면 `_should_summarize` 조건을 수정하면 됩니다.
