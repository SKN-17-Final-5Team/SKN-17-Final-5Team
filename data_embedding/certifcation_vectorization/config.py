"""
Configuration settings for Qdrant RAG system.
"""

# Default configuration
DEFAULT_CONFIG = {
    # Qdrant settings
    "collection_name": "trade_collection",
    "use_cloud": True,  # Set to False for local storage

    # Embedding settings
    "embedding_provider": "openai",  # "openai" or "huggingface"(사용 X) 
    "embedding_model": None,  # None = use provider default
    # Defaults:
    #   - openai: "text-embedding-3-large"
    #   - huggingface: "jhgan/ko-sroberta-multitask" (사용X)

    # Chunking settings (set to None to disable chunking)
    "chunk_size": None,  # Options: 500, 1000, 2000, None
    "chunk_overlap": 100,  # Overlap between chunks

    # Indexing settings
    "text_field": "full",  # Options: "auto", "summary", "full", "combined"
    "batch_size": 32,  # Batch size for embedding generation
    "jsonl_path": "./output/certifications.jsonl",  # Path to data file
}