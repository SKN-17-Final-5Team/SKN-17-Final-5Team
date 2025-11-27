"""
설정 및 클라이언트 초기화

전역 설정 상수와 외부 API 클라이언트를 초기화합니다.
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from openai import OpenAI

# 환경 변수 로드
load_dotenv()

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
RERANKER_API_URL = os.getenv("RERANKER_API_URL", "http://your-runpod-server/rerank")  # Reranker API 엔드포인트

# Reranker 사용 여부 (실행 시 설정됨)
USE_RERANKER = True  # 기본값
# 복합 질문 시 서브 쿼리별 개별 Reranking 사용 여부
# True: 각 서브 쿼리마다 개별 rerank → 모든 토픽 균형 보장
# False: 통합 rerank → 전체 품질 우선 (일부 토픽 누락 가능)
USE_PER_QUERY_RERANK = True  # 기본값
