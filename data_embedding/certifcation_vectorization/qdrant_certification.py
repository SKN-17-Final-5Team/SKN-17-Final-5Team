"""
Interactive CLI (실행 파일)
: Qdrant 'trade_collection'에 인증 데이터를 저장
"""

import json
from typing import List, Dict
from qdrant_certification_core import CertificationQdrant
from config import DEFAULT_CONFIG



def main(update_existing: bool = False):
    """
    메인 CLI 진입점

    Args:
        update_existing: True면 기존 'certification' 데이터를 삭제하고 새로 업로드 (업데이트 모드)
                        False면 기존 데이터 유지
    """
    print("=" * 80)
    print("QDRANT CERTIFICATION RAG 시스템")
    print("=" * 80)

    # 설정 표시
    print("\n설정:")
    print(f"  컬렉션: {DEFAULT_CONFIG['collection_name']}")
    print(f"  저장소: {'Qdrant Cloud' if DEFAULT_CONFIG['use_cloud'] else '로컬'}")
    print(f"  Embedding: {DEFAULT_CONFIG['embedding_provider']}")
    print(f"  청킹: {'활성화' if DEFAULT_CONFIG['chunk_size'] else '비활성화'}")
    if DEFAULT_CONFIG['chunk_size']:
        print(f"    - 청크 크기: {DEFAULT_CONFIG['chunk_size']}")
        print(f"    - 겹침: {DEFAULT_CONFIG['chunk_overlap']}")
    print()

    # Qdrant embedding 클래스 초기화
    rag = CertificationQdrant(
        collection_name=DEFAULT_CONFIG['collection_name'],
        embedding_provider=DEFAULT_CONFIG['embedding_provider'],
        embedding_model=DEFAULT_CONFIG['embedding_model'],
        chunk_size=DEFAULT_CONFIG['chunk_size'],
        chunk_overlap=DEFAULT_CONFIG['chunk_overlap'],
        use_cloud=DEFAULT_CONFIG['use_cloud']
    )

    # 컬렉션 생성
    rag.create_collection(recreate=False)

    # 인덱싱 필요 여부 확인
    info = rag.get_collection_info()
    if info.get('points_count', 0) == 0 or update_existing:
        print("\n" + "=" * 80)
        print("문서 인덱싱")
        print("=" * 80)

        num_chunks = rag.load_and_index_documents(
            DEFAULT_CONFIG['jsonl_path'],
            batch_size=DEFAULT_CONFIG['batch_size'],
            text_field=DEFAULT_CONFIG['text_field'],
            update_existing=update_existing
        )
    else:
        print(f"\n✓ 컬렉션에 이미 {info['points_count']}개의 청크가 인덱싱되어 있습니다")

    # 컬렉션 정보 표시
    print("\n" + "=" * 80)
    print("컬렉션 정보")
    print("=" * 80)
    info = rag.get_collection_info()
    print(f"컬렉션: {info.get('name')}")
    print(f"인덱싱된 포인트 수: {info.get('points_count', 'N/A')}")
    print(f"상태: {info.get('status', 'N/A')}")
    print(f"Embedding 모델: {info.get('embedding_model')}")
    print(f"벡터 차원: {info.get('vector_size')}")
    print("\n")

 


if __name__ == "__main__":
    # update_existing=True: 기존 certification 데이터를 삭제하고 새로 인덱싱 (다른 소스는 유지)
    # update_existing=False: 기존 데이터에 추가 (중복 가능)
    main(update_existing=True)
