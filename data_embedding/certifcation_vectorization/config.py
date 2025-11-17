"""
Qdrant RAG 시스템 설정
"""

# 기본 설정
DEFAULT_CONFIG = {
    # Qdrant 설정
    "collection_name": "trade_collection",
    "use_cloud": True,  # 로컬 저장소 사용 시 False로 설정

    # Embedding 설정
    "embedding_provider": "openai",  # "openai" 또는 "huggingface"(사용 X)
    "embedding_model": None,  # None = provider 기본값 사용
    # 기본값:
    #   - openai: "text-embedding-3-large"
    #   - huggingface: "jhgan/ko-sroberta-multitask" (사용X)

    # 청킹 설정 (청킹 비활성화하려면 None으로 설정)
    "chunk_size": 1000,  # 옵션: 500, 1000, 2000, None
    "chunk_overlap": 100,  # 청크 간 겹침

    # 인덱싱 설정
    "text_field": "full",  # 옵션: "auto", "summary", "full", "combined"
    "batch_size": 32,  # 임베딩 생성 배치 크기
    "jsonl_path": "output/certifications.jsonl",  # 데이터 파일 경로
}