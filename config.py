"""
설정 및 클라이언트 초기화

전역 설정 상수와 외부 API 클라이언트를 초기화합니다.
"""

import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from openai import OpenAI

# 환경 변수 로드
load_dotenv()

# =====================================================================
# 환경 변수 검증
# =====================================================================

def _validate_env_vars():
    """필수 환경 변수 검증"""
    required_vars = {
        "QDRANT_URL": "Qdrant 클라우드 URL",
        "QDRANT_API_KEY": "Qdrant API 키",
        "OPENAI_API_KEY": "OpenAI API 키",
        "RERANKER_API_URL": "Reranker API 엔드포인트 URL"
    }

    missing_vars = []
    for var_name, description in required_vars.items():
        if not os.getenv(var_name):
            missing_vars.append(f"  - {var_name}: {description}")

    if missing_vars:
        print("❌ 필수 환경 변수가 설정되지 않았습니다:")
        print("\n".join(missing_vars))
        print("\n.env 파일을 확인해주세요.")
        sys.exit(1)

# 환경 변수 검증 실행
_validate_env_vars()

# =====================================================================
# 클라이언트 초기화
# =====================================================================

# Qdrant Vector DB 클라이언트
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=60
)

# OpenAI 클라이언트 (Embedding 및 Agent용)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =====================================================================
# 설정 상수
# =====================================================================

COLLECTION_NAME = "trade_collection"  # Qdrant 컬렉션 이름
EMBEDDING_MODEL = "text-embedding-3-large"  # OpenAI Embedding 모델
RERANKER_API_URL = os.getenv("RERANKER_API_URL")  # Reranker API 엔드포인트

# Reranker 사용 여부 (실행 시 설정됨)
USE_RERANKER = True  # 기본값

# =====================================================================
# 메모리 설정 (LangGraph용)
# =====================================================================

# 메모리 저장 경로
MEMORY_DIR = "memory"  # LangGraph checkpointer 및 장기 메모리 저장 디렉토리
LONG_TERM_MEMORY_FILE = "long_term_memory.json"  # 장기 메모리 백업 파일명