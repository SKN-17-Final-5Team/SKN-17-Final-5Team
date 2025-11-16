"""
Interactive CLI (run this)
: To store certification data into Qdrat 'trade_collection' 
"""

import json
from typing import List, Dict
from qdrant_certification_core import CertificationQdrant
from config import DEFAULT_CONFIG



def main():
    """Main CLI entry point."""
    print("=" * 80)
    print("QDRANT CERTIFICATION RAG SYSTEM")
    print("=" * 80)

    # Display configuration
    print("\nConfiguration:")
    print(f"  Collection: {DEFAULT_CONFIG['collection_name']}")
    print(f"  Storage: {'Qdrant Cloud' if DEFAULT_CONFIG['use_cloud'] else 'Local'}")
    print(f"  Embedding: {DEFAULT_CONFIG['embedding_provider']}")
    print(f"  Chunking: {'Enabled' if DEFAULT_CONFIG['chunk_size'] else 'Disabled'}")
    if DEFAULT_CONFIG['chunk_size']:
        print(f"    - Chunk size: {DEFAULT_CONFIG['chunk_size']}")
        print(f"    - Overlap: {DEFAULT_CONFIG['chunk_overlap']}")
    print()

    # Initialize the Qdrant embeding class
    rag = CertificationQdrant(
        collection_name=DEFAULT_CONFIG['collection_name'],
        embedding_provider=DEFAULT_CONFIG['embedding_provider'],
        embedding_model=DEFAULT_CONFIG['embedding_model'],
        chunk_size=DEFAULT_CONFIG['chunk_size'],
        chunk_overlap=DEFAULT_CONFIG['chunk_overlap'],
        use_cloud=DEFAULT_CONFIG['use_cloud']
    )

    # Create collection
    rag.create_collection(recreate=False)

    # Check if indexing is needed
    info = rag.get_collection_info()
    if info.get('points_count', 0) == 0:
        print("\n" + "=" * 80)
        print("INDEXING DOCUMENTS")
        print("=" * 80)

        num_chunks = rag.load_and_index_documents(
            DEFAULT_CONFIG['jsonl_path'],
            batch_size=DEFAULT_CONFIG['batch_size'],
            text_field=DEFAULT_CONFIG['text_field']
        )
    else:
        print(f"\n✓ 컬렉션에 이미 {info['points_count']}개의 청크가 인덱싱되어 있습니다")

    # Display collection info
    print("\n" + "=" * 80)
    print("COLLECTION INFO")
    print("=" * 80)
    info = rag.get_collection_info()
    print(f"Collection: {info.get('name')}")
    print(f"Points indexed: {info.get('points_count', 'N/A')}")
    print(f"Status: {info.get('status', 'N/A')}")
    print(f"Embedding model: {info.get('embedding_model')}")
    print(f"Vector dimension: {info.get('vector_size')}")
    print("\n")

 


if __name__ == "__main__":
    main()
