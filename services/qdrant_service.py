"""
Qdrant 벡터 DB 연동 서비스

문서 임베딩을 Qdrant에 저장하고 의미 기반 검색을 수행하는 서비스
"""

from typing import List, Dict, Optional, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)
import uuid
from datetime import datetime


class QdrantService:
    """Qdrant 벡터 DB 관리 서비스"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "trade_documents"
    ):
        """
        Args:
            host: Qdrant 호스트 (로컬)
            port: Qdrant 포트
            url: Qdrant Cloud URL (None이면 로컬 사용)
            api_key: Qdrant Cloud API Key
            collection_name: 컬렉션 이름
        """
        self.collection_name = collection_name

        # Qdrant 클라이언트 초기화
        if url and api_key:
            # Qdrant Cloud
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            # 로컬 Qdrant
            self.client = QdrantClient(host=host, port=port)

    def create_collection(
        self,
        vector_size: int = 768,
        distance: Distance = Distance.COSINE
    ):
        """
        컬렉션 생성

        Args:
            vector_size: 임베딩 벡터 차원 (KoSimCSE: 768)
            distance: 거리 메트릭 (COSINE, EUCLID, DOT)
        """
        try:
            # 컬렉션 존재 여부 확인
            collections = self.client.get_collections().collections
            if any(c.name == self.collection_name for c in collections):
                print(f"✓ 컬렉션 '{self.collection_name}' 이미 존재")
                return

            # 컬렉션 생성
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance
                )
            )
            print(f"컬렉션 '{self.collection_name}' 생성 완료")

        except Exception as e:
            print(f"❌ 컬렉션 생성 실패: {e}")
            raise

    def add_document(
        self,
        embedding: List[float],
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> str:
        """
        문서 임베딩 추가

        Args:
            embedding: 임베딩 벡터
            text: 원본 텍스트
            metadata: 메타데이터 (s3_key, doc_type, timestamp 등)
            doc_id: 문서 ID (None이면 자동 생성)

        Returns:
            문서 ID
        """
        # 문서 ID 생성
        if not doc_id:
            doc_id = str(uuid.uuid4())

        # 페이로드 구성
        payload = {
            "text": text,
            "timestamp": datetime.now().isoformat()
        }
        if metadata:
            payload.update(metadata)

        try:
            # Qdrant에 포인트 추가
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=doc_id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )
            print(f"문서 임베딩 추가 완료: {doc_id}")
            return doc_id

        except Exception as e:
            print(f"❌ 문서 임베딩 추가 실패: {e}")
            raise

    def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        유사 문서 검색

        Args:
            query_embedding: 쿼리 임베딩 벡터
            limit: 반환할 문서 수
            score_threshold: 최소 유사도 점수 (0~1)
            filter_conditions: 필터 조건 (예: {"doc_type": "invoice"})

        Returns:
            검색 결과 리스트 [{"id": ..., "score": ..., "payload": {...}}]
        """
        # 필터 설정
        query_filter = None
        if filter_conditions:
            conditions = []
            for key, value in filter_conditions.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            query_filter = Filter(must=conditions)

        try:
            # 벡터 검색
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                query_filter=query_filter,
                score_threshold=score_threshold
            )

            # 결과 포맷팅
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload
                })

            return formatted_results

        except Exception as e:
            print(f"❌ 유사 문서 검색 실패: {e}")
            return []

    def delete_document(self, doc_id: str) -> bool:
        """
        문서 삭제

        Args:
            doc_id: 문서 ID

        Returns:
            삭제 성공 여부
        """
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=[doc_id]
            )
            print(f"✓ 문서 삭제 완료: {doc_id}")
            return True

        except Exception as e:
            print(f"❌ 문서 삭제 실패: {e}")
            return False

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        문서 조회

        Args:
            doc_id: 문서 ID

        Returns:
            문서 데이터 (없으면 None)
        """
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[doc_id]
            )

            if not result:
                return None

            point = result[0]
            return {
                "id": point.id,
                "payload": point.payload
            }

        except Exception as e:
            print(f"❌ 문서 조회 실패: {e}")
            return None

    def scroll_documents(
        self,
        limit: int = 100,
        filter_conditions: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        전체 문서 스크롤 (페이지네이션)

        Args:
            limit: 가져올 문서 수
            filter_conditions: 필터 조건

        Returns:
            문서 리스트
        """
        # 필터 설정
        query_filter = None
        if filter_conditions:
            conditions = []
            for key, value in filter_conditions.items():
                conditions.append(
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value)
                    )
                )
            query_filter = Filter(must=conditions)

        try:
            result = self.client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                scroll_filter=query_filter
            )

            documents = []
            for point in result[0]:  # result는 (points, next_offset) 튜플
                documents.append({
                    "id": point.id,
                    "payload": point.payload
                })

            return documents

        except Exception as e:
            print(f"❌ 문서 스크롤 실패: {e}")
            return []

    def get_collection_info(self) -> Optional[Dict[str, Any]]:
        """
        컬렉션 정보 조회

        Returns:
            컬렉션 정보 (문서 수, 벡터 차원 등)
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": info.config.params.vectors.size,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status
            }

        except Exception as e:
            print(f"❌ 컬렉션 정보 조회 실패: {e}")
            return None

    def delete_collection(self):
        """컬렉션 삭제 (주의: 모든 데이터 삭제됨)"""
        try:
            self.client.delete_collection(self.collection_name)
            print(f"✓ 컬렉션 '{self.collection_name}' 삭제 완료")

        except Exception as e:
            print(f"❌ 컬렉션 삭제 실패: {e}")
            raise
