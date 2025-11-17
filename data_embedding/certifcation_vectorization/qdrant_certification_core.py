import json
import os
from typing import List, Dict, Optional, Literal
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
import numpy as np
import uuid


load_dotenv()


class CertificationQdrant:
    """인증 데이터를 위한 Qdrant vector database builder 클래스"""

    def __init__(
        self,
        collection_name: str = "trade_collection",
        embedding_provider: Literal["openai"] = "openai",
        embedding_model: Optional[str] = None,
        chunk_size: Optional[int] = 1000,
        chunk_overlap: Optional[int] = 100,
        use_cloud: bool = True
    ):
        """RAG 시스템 초기화

        Args:
            collection_name: Qdrant 컬렉션 이름
            embedding_provider: "openai" (지원되는 유일한 provider)
            embedding_model: 모델 이름 (None = provider 기본값)
            chunk_size: 텍스트 청크 크기 (None = 청킹 안함)
            chunk_overlap: 청크 간 겹침
            use_cloud: Qdrant Cloud 사용 (True) 또는 로컬 (지원 안함)
        """
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Embedding 모델 초기화
        if embedding_provider == "openai":
            self._init_openai_embeddings(embedding_model)
        else:
            raise ValueError(f"지원하지 않는 embedding provider: {embedding_provider}")

        # Qdrant 클라이언트 초기화
        if use_cloud:
            self._init_qdrant_cloud()
        else:
            raise ValueError("로컬 Qdrant는 지원하지 않습니다. use_cloud=True로 설정하세요.")


    def _init_openai_embeddings(self, model: Optional[str] = None):
        """OpenAI embedding 초기화"""
        try:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.embedding_model_name = model or "text-embedding-3-large"
            self.embedding_dimension = 3072 if "large" in self.embedding_model_name else 1536
            print(f"✓ OpenAI embedding 초기화 완료: {self.embedding_model_name}")
        except ImportError:
            raise ImportError("openai 설치 필요: pip install openai")


    def _init_qdrant_cloud(self):
        """Qdrant Cloud 클라이언트 초기화"""
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")

        if not url or not api_key:
            raise ValueError(".env에 QDRANT_URL과 QDRANT_API_KEY를 설정하세요")

        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=300  # 5분 타임아웃 (대용량 업로드 대비)
        )
        print(f"✓ Qdrant Cloud 연결 완료")


    def _ensure_payload_index(self) -> None:
        """data_source 필드에 payload index가 있는지 확인하고 없으면 생성"""
        try:
            # data_source 필드에 keyword index 생성
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="data_source",
                field_schema="keyword"
            )
            print(f"✓ 'data_source' 필드 인덱스 생성 완료")
        except Exception as e:
            # 이미 인덱스가 존재하면 무시
            if "already exists" in str(e).lower():
                pass
            else:
                # 다른 에러는 무시하고 계속 진행 (인덱스가 이미 있을 수 있음)
                pass

    def delete_by_data_source(self, data_source: str) -> None:
        """
        특정 data_source의 포인트만 삭제

        Args:
            data_source: 삭제할 데이터 소스 (예: 'certification', 'fraud', etc.)
        """
        try:
            # payload index 확인 및 생성
            self._ensure_payload_index()

            # 기존 포인트 수 확인
            collection_info = self.client.get_collection(self.collection_name)
            before_count = collection_info.points_count

            print(f"삭제 전 총 포인트 수: {before_count}")
            print(f"'{data_source}' 데이터 소스 삭제 중...")

            # data_source 필터로 삭제
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="data_source",
                            match=MatchValue(value=data_source)
                        )
                    ]
                )
            )

            # 삭제 후 포인트 수 확인
            collection_info = self.client.get_collection(self.collection_name)
            after_count = collection_info.points_count
            deleted_count = before_count - after_count

            print(f"✓ '{data_source}' 데이터 {deleted_count}개 삭제 완료")
            print(f"✓ 삭제 후 총 포인트 수: {after_count}")

        except Exception as e:
            print(f"삭제 중 오류 발생: {e}")
            raise

    def create_collection(self, recreate: bool = False) -> None:
        """컬렉션이 없으면 생성"""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            exists = False
            print(f"✓ 기존 컬렉션 삭제 완료: {self.collection_name}")

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE
                )
            )
            print(f"✓ 컬렉션 생성 완료: {self.collection_name}")
        else:
            print(f"✓ 컬렉션 존재: {self.collection_name}")

    def embed_text(self, text: str) -> List[float]:
        """OpenAI를 사용하여 텍스트의 임베딩 생성"""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model_name,
            input=text
        )
        return response.data[0].embedding

    def chunk_text(self, text: str) -> List[str]:
        """텍스트를 청크로 분할"""
        if not self.chunk_size:
            return [text]

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - (self.chunk_overlap or 0)

        return chunks if chunks else [text]

    def load_and_index_documents(
        self,
        jsonl_path: str,
        text_field: Literal["auto", "summary", "full", "combined"] = "full",
        batch_size: int = 32,
        update_existing: bool = False
    ) -> int:
        """JSONL 파일에서 문서를 로드하고 인덱싱

        Args:
            jsonl_path: JSONL 파일 경로
            text_field: 임베딩할 필드 ("auto", "summary", "full", "combined")
            batch_size: 임베딩 배치 크기
            update_existing: True면 기존 certification 데이터를 삭제하고 새로 업로드

        Returns:
            인덱싱된 청크 수
        """
        # 업데이트 모드: 기존 certification 데이터 삭제
        if update_existing:
            self.delete_by_data_source('certification')
        print(f"\n문서 로드 중: {jsonl_path}")

        # 문서 로드
        documents = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                documents.append(json.loads(line))

        print(f"✓ {len(documents)}개 문서 로드 완료")

        # 메타데이터 및 텍스트 준비
        doc_metadata = []
        texts_to_embed = []

        for doc in documents:
            # 텍스트 필드 선택
            if text_field == "summary":
                text = doc.get('auto_summary', '') or doc.get('cert_subject', '')
            elif text_field == "full":
                text = doc.get('cert_subject', '')
            elif text_field == "combined":
                text = f"{doc.get('auto_summary', '')}\n\n{doc.get('cert_subject', '')}"
            else:  # auto
                text = doc.get('auto_summary', '') or doc.get('cert_subject', '')

            # 텍스트 청킹
            chunks = self.chunk_text(text)

            for chunk_idx, chunk in enumerate(chunks):
                doc_metadata.append({
                    'doc': doc,
                    'chunk_idx': chunk_idx,
                    'total_chunks': len(chunks),
                    'chunk_text': chunk
                })
                texts_to_embed.append(chunk)

        print(f"✓ {len(documents)}개 문서에서 {len(texts_to_embed)}개 청크 생성 완료")

        # 배치 단위로 임베딩 생성
        print(f"\n임베딩 생성 중...")
        all_embeddings = []

        for i in range(0, len(texts_to_embed), batch_size):
            batch_texts = texts_to_embed[i:i + batch_size]

            response = self.openai_client.embeddings.create(
                model=self.embedding_model_name,
                input=batch_texts
            )
            batch_embeddings = [item.embedding for item in response.data]

            all_embeddings.extend(batch_embeddings)

            if (i + batch_size) % (batch_size * 10) == 0:
                print(f"  진행: {min(i + batch_size, len(texts_to_embed))}/{len(texts_to_embed)}")

        print(f"✓ {len(all_embeddings)}개 임베딩 생성 완료")

        # Point 생성
        print(f"\nQdrant point 생성 중...")
        points = []

        for metadata, embedding in zip(doc_metadata, all_embeddings):
            doc = metadata['doc']

            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "data_source": "certification",
                    "doc_id": f"cert_{doc['id']}",
                    "source_doc_id": doc["id"],
                    "title": doc['cert_name'],
                    "content": doc['cert_subject'][:2000],

                    "certification_meta": {
                        "country": doc.get('country', ''),
                        "category": doc.get('category', ''),
                        "cert_type": doc.get('cert_type', ''),
                        "main_cert": doc.get('main_cert', ''),
                        "url": doc.get('url', ''),
                        "summary": doc.get('auto_summary', ''),
                    },

                    "chunk_info": {
                        "chunk_idx": metadata['chunk_idx'],
                        "total_chunks": metadata['total_chunks'],
                        "chunk_text": metadata['chunk_text'][:500]
                    },

                    "embedding_info": {
                        "model": self.embedding_model_name,
                        "provider": self.embedding_provider
                    }
                }
            )
            points.append(point)

        # Qdrant에 업로드
        upload_batch_size = 20  # 타임아웃 방지를 위해 50에서 20으로 축소
        print(f"Qdrant 업로드 중 (batch_size={upload_batch_size})...")

        for i in range(0, len(points), upload_batch_size):
            batch = points[i:i + upload_batch_size]
            batch_num = i // upload_batch_size + 1
            total_batches = (len(points) + upload_batch_size - 1) // upload_batch_size

            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True
            )

            print(f"  - 배치 {batch_num}/{total_batches} 업로드 완료 ({len(batch)}개)")

        print(f"✓ {self.collection_name}에 {len(points)}개 point 업로드 완료")
        return len(points)

    def get_collection_info(self) -> Dict:
        """컬렉션 정보 조회"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                'name': self.collection_name,
                'points_count': info.points_count,
                'vector_size': info.config.params.vectors.size,
                'status': info.status,
                'embedding_model': self.embedding_model_name
            }
        except Exception as e:
            return {'error': str(e)}
