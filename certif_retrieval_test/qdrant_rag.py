"""
인증 데이터를 위한 Qdrant 기반 RAG 시스템.
키워드 매칭 대신 임베딩을 활용한 의미 기반 검색(semantic search)을 수행합니다.

지원 기능:
- 환경변수를 통한 Qdrant Cloud 연결
- 다양한 임베딩 모델 (HuggingFace, OpenAI)
- 최적 검색을 위한 청킹(chunking) 설정
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Optional, Literal
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    Modifier
)
import numpy as np
from rank_bm25 import BM25Okapi
import re


load_dotenv()


class QdrantCertificationRAG:
    """Qdrant 벡터 데이터베이스를 활용한 의미 기반 검색 RAG 시스템."""

    def __init__(
        self,
        collection_name: str = "certifications",
        embedding_provider: Literal["huggingface", "openai"] = "huggingface",
        embedding_model: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        use_cloud: bool = True
    ):
        """
        Qdrant RAG 시스템 초기화.

        Args:
            collection_name: Qdrant 컬렉션 이름
            embedding_provider: "huggingface" 또는 "openai"
            embedding_model: 모델 이름 (None = 프로바이더 기본값 사용)
            chunk_size: 텍스트 청크 크기 (None = 청킹 미사용, 전체 텍스트 사용)
            chunk_overlap: 청크 간 겹침 크기 (chunk_size 설정 시에만 사용)
            use_cloud: True면 .env의 Qdrant Cloud 사용, False면 로컬 저장소 사용

        기본 모델:
            - HuggingFace: "jhgan/ko-sroberta-multitask" (한국어 최적화)
            - OpenAI: "text-embedding-3-large" (전체 성능 최고)
        """
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap or 0

        # 하이브리드 검색을 위한 BM25 지원
        self.bm25_index = None
        self.bm25_documents = []  # BM25용 토큰화된 문서 저장
        self.bm25_doc_ids = []    # BM25 인덱스를 Qdrant point ID에 매핑

        # 임베딩 모델 초기화
        if embedding_provider == "openai":
            self._init_openai_embeddings(embedding_model)
        else:
            self._init_huggingface_embeddings(embedding_model)

        # Qdrant 클라이언트 초기화
        if use_cloud:
            self._init_qdrant_cloud()
        else:
            self._init_qdrant_local()

    def _init_openai_embeddings(self, model: Optional[str] = None):
        """OpenAI 임베딩 초기화."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Install openai: pip install openai")

        self.embedding_model_name = model or "text-embedding-3-large"
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env file")

        self.openai_client = OpenAI(api_key=api_key)

        # 모델에 따른 벡터 차원 설정
        dimensions = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536
        }
        self.vector_size = dimensions.get(self.embedding_model_name, 1536)

        print(f"OpenAI 임베딩 사용: {self.embedding_model_name}")
        print(f"임베딩 차원: {self.vector_size}")

    def _init_huggingface_embeddings(self, model: Optional[str] = None):
        """HuggingFace 임베딩 초기화."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install sentence-transformers")

        self.embedding_model_name = model or "jhgan/ko-sroberta-multitask"

        # HF 토큰 확인 (선택사항, gated 모델에만 필요)
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            print(".env에서 HuggingFace 토큰 사용")

        print(f"임베딩 모델 로딩 중: {self.embedding_model_name}...")
        self.embedding_model = SentenceTransformer(
            self.embedding_model_name,
            use_auth_token=hf_token if hf_token else None
        )
        self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
        print(f"임베딩 차원: {self.vector_size}")

    def _init_qdrant_cloud(self):
        """환경변수를 사용하여 Qdrant Cloud 클라이언트 초기화."""
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")

        if not qdrant_url or not qdrant_api_key:
            raise ValueError(
                ".env 파일에 QDRANT_URL과 QDRANT_API_KEY가 설정되어야 합니다\n"
            )

        print(f"Qdrant Cloud 연결 중: {qdrant_url}")
        self.client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=300  # 대용량 업로드를 위한 5분 타임아웃
        )
        print("✓ Qdrant Cloud 연결 완료")

    def _init_qdrant_local(self):
        """로컬 Qdrant 클라이언트 초기화."""
        storage_path = "./qdrant_storage"
        print(f"로컬 Qdrant 저장소 사용: {storage_path}")
        self.client = QdrantClient(path=storage_path)

    def create_collection(self, recreate: bool = False) -> None:
        """
        Qdrant 컬렉션 생성.

        Args:
            recreate: True면 기존 컬렉션 삭제 후 새로 생성
        """
        # 컬렉션 존재 여부 확인
        collections = self.client.get_collections().collections
        collection_exists = any(c.name == self.collection_name for c in collections)

        if collection_exists:
            if recreate:
                print(f"기존 컬렉션 삭제 중: {self.collection_name}")
                self.client.delete_collection(self.collection_name)
            else:
                print(f"컬렉션 '{self.collection_name}'이(가) 이미 존재합니다")
                return

        # 컬렉션 생성
        print(f"컬렉션 생성 중: {self.collection_name}")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE  # 임베딩을 위한 코사인 유사도
            )
        )
        print(f"✓ 컬렉션 '{self.collection_name}' 생성 완료")

    def embed_text(self, text: str) -> List[float]:
        """
        텍스트에 대한 임베딩 벡터 생성.

        Args:
            text: 입력 텍스트

        Returns:
            임베딩 벡터 (float 리스트)
        """
        if self.embedding_provider == "openai":
            # OpenAI 임베딩
            response = self.openai_client.embeddings.create(
                model=self.embedding_model_name,
                input=text
            )
            return response.data[0].embedding
        else:
            # HuggingFace 임베딩
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()

    def chunk_text(self, text: str) -> List[str]:
        """
        텍스트를 겹침을 포함하여 청크로 분할.

        Args:
            text: 입력 텍스트

        Returns:
            텍스트 청크 리스트
        """
        if not self.chunk_size:
            # 청킹 없음, 전체 텍스트 반환
            return [text]

        # 단순 문자 기반 청킹
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)

            # (chunk_size - overlap)만큼 앞으로 이동
            start += (self.chunk_size - self.chunk_overlap)

        return chunks

    def load_and_index_documents(
        self,
        jsonl_path: str,
        batch_size: int = 32,
        text_field: str = "auto",
        show_progress: bool = True
    ) -> int:
        """
        JSONL에서 문서를 로드하여 Qdrant에 인덱싱.

        Args:
            jsonl_path: certifications.jsonl 파일 경로
            batch_size: 임베딩 생성 배치 크기
            text_field: 임베딩할 필드 선택:
                - "auto": auto_summary가 있으면 사용, 없으면 전체 텍스트
                - "summary": auto_summary만 사용
                - "full": 전체 cert_subject 사용
                - "combined": 사전 구성된 'text' 필드 사용 (cert_name + metadata + subject)
            show_progress: 진행률 표시 (HuggingFace만 해당)

        Returns:
            인덱싱된 문서 수 (청킹 활성화 시 청크 수)
        """
        print(f"\n{jsonl_path}에서 문서 로드 중...")

        # 문서 로드
        documents = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                documents.append(doc)

        print(f"{len(documents)}개 문서 로드 완료")

        # 임베딩할 텍스트 준비
        print(f"임베딩용 텍스트 준비 중 (필드: {text_field})...")
        texts_to_embed = []
        doc_metadata = []  # 각 청크의 메타데이터 저장

        for doc_idx, doc in enumerate(documents):
            # 전략에 따라 텍스트 필드 선택
            if text_field == "summary":
                text = doc.get('auto_summary', '')
            elif text_field == "full":
                text = doc.get('cert_subject', '')
            elif text_field == "combined":
                text = doc.get('text', '')
            else:  # "auto"
                # 요약이 있으면 사용, 없으면 결합 텍스트 사용
                text = doc.get('auto_summary', '') or doc.get('text', '')

            # 청킹이 활성화되어 있으면 텍스트 분할
            chunks = self.chunk_text(text)

            for chunk_idx, chunk in enumerate(chunks):
                texts_to_embed.append(chunk)
                doc_metadata.append({
                    'doc_idx': doc_idx,
                    'chunk_idx': chunk_idx,
                    'total_chunks': len(chunks),
                    'doc': doc,
                    'chunk_text': chunk
                })

        print(f"임베딩할 총 청크 수: {len(texts_to_embed)}")
        if self.chunk_size:
            print(f"청킹 설정: size={self.chunk_size}, overlap={self.chunk_overlap}")

        # 임베딩 생성
        print("임베딩 생성 중...")
        all_embeddings = []

        if self.embedding_provider == "openai":
            # OpenAI: 배치 API 호출 (호출당 최대 2048개 텍스트)
            max_batch = 2048
            for i in range(0, len(texts_to_embed), max_batch):
                batch_texts = texts_to_embed[i:i + max_batch]
                print(f"  배치 처리 중 {i//max_batch + 1}/{(len(texts_to_embed)-1)//max_batch + 1}...")

                response = self.openai_client.embeddings.create(
                    model=self.embedding_model_name,
                    input=batch_texts
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

        else:
            # HuggingFace: 배치 인코딩 사용
            for i in range(0, len(texts_to_embed), batch_size):
                batch_texts = texts_to_embed[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode(
                    batch_texts,
                    convert_to_numpy=True,
                    show_progress_bar=show_progress and (i == 0)
                )
                all_embeddings.extend(batch_embeddings)
                if show_progress:
                    print(f"  처리 완료 {min(i + batch_size, len(texts_to_embed))}/{len(texts_to_embed)} 청크")

        # Qdrant용 포인트 생성
        print(f"Qdrant에 {len(all_embeddings)}개 청크 인덱싱 중...")
        points = []

        for idx, (metadata, embedding) in enumerate(zip(doc_metadata, all_embeddings)):
            doc = metadata['doc']

            # 필요 시 numpy 배열을 리스트로 변환
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            point = PointStruct(
                id=idx,
                vector=embedding,
                payload={
                    # 문서 메타데이터
                    "cert_id": doc['id'],
                    "country": doc['country'],
                    "category": doc['category'],
                    "cert_type": doc['cert_type'],
                    "main_cert": doc['main_cert'],
                    "cert_name": doc['cert_name'],
                    "cert_subject": doc['cert_subject'][:1000],  # 저장을 위해 잘라냄
                    "auto_summary": doc.get('auto_summary', ''),
                    "url": doc['url'],

                    # 청크 메타데이터 (추적용)
                    "chunk_idx": metadata['chunk_idx'],
                    "total_chunks": metadata['total_chunks'],
                    "chunk_text": metadata['chunk_text'][:500],  # 임베딩된 내용 저장

                    # 임베딩 메타데이터
                    "embedding_model": self.embedding_model_name,
                    "embedding_provider": self.embedding_provider
                }
            )
            points.append(point)

        # 배치 단위로 Qdrant에 업로드
        # OpenAI는 더 작은 배치 크기 사용 (3072차원 벡터가 HF 768차원보다 4배 큼)
        upload_batch_size = 50 if self.embedding_provider == "openai" else 100

        for i in range(0, len(points), upload_batch_size):
            batch = points[i:i + upload_batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
            print(f"  인덱싱 완료 {min(i + upload_batch_size, len(points))}/{len(points)} 청크")

        print(f"✓ {len(documents)}개 문서에서 {len(points)}개 청크 인덱싱 완료")
        return len(points)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        임베딩을 사용한 의미 기반 검색(semantic search).

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 개수
            filters: 선택적 필터 (예: {"country": "미국"})
            score_threshold: 최소 유사도 점수 (0.0 ~ 1.0)

        Returns:
            점수와 함께 매칭된 문서 리스트
        """
        # 쿼리 임베딩 생성
        query_vector = self.embed_text(query)

        # 필터 조건 구성
        search_filter = None
        if filters:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            conditions = []
            for field, value in filters.items():
                conditions.append(
                    FieldCondition(
                        key=field,
                        match=MatchValue(value=value)
                    )
                )
            search_filter = Filter(must=conditions)

        # Qdrant에서 검색
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=search_filter,
            score_threshold=score_threshold
        )

        # 결과 포맷팅
        results = []
        for hit in search_results:
            results.append({
                'score': hit.score,
                'doc': hit.payload
            })

        return results

    def _tokenize(self, text: str) -> List[str]:
        """
        BM25를 위한 단순 토큰화.
        공백 기준 분리, 구두점 제거.

        Args:
            text: 입력 텍스트

        Returns:
            토큰 리스트
        """
        # 소문자 변환
        text = text.lower()
        # 한글 문자를 제외한 구두점 제거
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        # 공백 기준 분리
        tokens = text.split()
        return tokens

    def build_bm25_index(self, documents: List[Dict], text_field: str = "auto") -> None:
        """
        키워드 검색을 위한 BM25 인덱스 구축.

        Args:
            documents: JSONL의 문서 리스트
            text_field: 인덱싱할 필드 (임베딩에 사용한 것과 동일)
        """
        print("하이브리드 검색용 BM25 인덱스 구축 중...")
        self.bm25_documents = []
        self.bm25_doc_ids = []

        for doc in documents:
            # 텍스트 필드 선택 (load_and_index_documents와 동일한 로직)
            if text_field == "summary":
                text = doc.get('auto_summary', '')
            elif text_field == "full":
                text = doc.get('cert_subject', '')
            elif text_field == "combined":
                text = doc.get('text', '')
            else:  # "auto"
                text = doc.get('auto_summary', '') or doc.get('text', '')

            # 토큰화 및 저장
            tokens = self._tokenize(text)
            self.bm25_documents.append(tokens)
            self.bm25_doc_ids.append(doc['id'])

        # BM25 인덱스 생성
        self.bm25_index = BM25Okapi(self.bm25_documents)
        print(f"✓ {len(self.bm25_documents)}개 문서로 BM25 인덱스 구축 완료")

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
        filters: Optional[Dict] = None,
        score_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        의미 기반(벡터) 검색과 키워드(BM25) 검색을 결합한 하이브리드 검색.
        Reciprocal Rank Fusion(RRF)을 사용하여 결과를 병합합니다.

        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 개수
            semantic_weight: 의미 검색 가중치 (0.0 ~ 1.0)
            bm25_weight: BM25 키워드 검색 가중치 (0.0 ~ 1.0)
            filters: 선택적 필터
            score_threshold: 최소 유사도 점수

        Returns:
            융합된 점수와 함께 매칭된 문서 리스트
        """
        if self.bm25_index is None:
            print("경고: BM25 인덱스가 구축되지 않았습니다. 의미 검색만 사용합니다.")
            return self.search(query, top_k, filters, score_threshold)

        # 의미 검색 결과 획득
        semantic_results = self.search(
            query,
            top_k=top_k * 3,  # 융합을 위해 더 많은 결과 가져오기
            filters=filters,
            score_threshold=score_threshold
        )

        # BM25 키워드 검색 결과 획득
        query_tokens = self._tokenize(query)
        bm25_scores = self.bm25_index.get_scores(query_tokens)

        # 점수와 함께 BM25 결과 생성
        bm25_results = []
        for idx, score in enumerate(bm25_scores):
            if score > 0:  # 0이 아닌 점수를 가진 문서만 포함
                # ID로 Qdrant에서 문서 가져오기
                try:
                    doc_id = self.bm25_doc_ids[idx]
                    # 의미 검색 결과에서 매칭 문서 찾기 또는 Qdrant에서 가져오기
                    doc_found = None
                    for sem_result in semantic_results:
                        if sem_result['doc'].get('cert_id') == doc_id:
                            doc_found = sem_result['doc']
                            break

                    if doc_found:
                        bm25_results.append({
                            'score': float(score),
                            'doc': doc_found,
                            'cert_id': doc_id
                        })
                except:
                    continue

        # 점수로 BM25 결과 정렬
        bm25_results = sorted(bm25_results, key=lambda x: x['score'], reverse=True)[:top_k * 3]

        # Reciprocal Rank Fusion(RRF)을 사용하여 결과 융합
        fused_scores = {}
        k = 60  # RRF 상수

        # 의미 검색 점수 추가
        for rank, result in enumerate(semantic_results, start=1):
            cert_id = result['doc'].get('cert_id', result['doc'].get('id'))
            rrf_score = semantic_weight / (k + rank)
            if cert_id not in fused_scores:
                fused_scores[cert_id] = {
                    'score': 0.0,
                    'doc': result['doc']
                }
            fused_scores[cert_id]['score'] += rrf_score

        # BM25 점수 추가
        for rank, result in enumerate(bm25_results, start=1):
            cert_id = result.get('cert_id', result['doc'].get('cert_id', result['doc'].get('id')))
            rrf_score = bm25_weight / (k + rank)
            if cert_id not in fused_scores:
                fused_scores[cert_id] = {
                    'score': 0.0,
                    'doc': result['doc']
                }
            fused_scores[cert_id]['score'] += rrf_score

        # 융합된 점수로 정렬 후 top_k 반환
        fused_results = sorted(
            fused_scores.values(),
            key=lambda x: x['score'],
            reverse=True
        )[:top_k]

        return fused_results

    def print_results(self, results: List[Dict], query: str = "", show_full: bool = False) -> None:
        """Pretty print search results.

        Args:
            results: Search results to display
            query: Optional query string to show
            show_full: If True, show full cert_subject instead of auto_summary
        """
        if not results:
            print("No results found.")
            return

        if query:
            print(f"\nQuery: '{query}'")
        print(f"Found {len(results)} results:\n")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            doc = result['doc']
            score = result['score']

            print(f"\n[Result {i}] (Similarity: {score:.4f})")
            print(f"인증명: {doc['cert_name']}")
            print(f"국가: {doc['country']}")
            print(f"카테고리: {doc['category']}")
            print(f"인증구분: {doc['cert_type']}")
            print(f"대표인증: {doc['main_cert']}")

            # Show chunk info if chunking was used
            if doc.get('total_chunks', 1) > 1:
                print(f"청크: {doc['chunk_idx'] + 1}/{doc['total_chunks']}")

            # Show full text or summary based on parameter
            if show_full and doc.get('cert_subject'):
                print(f"\n내용:\n{doc['cert_subject']}")
            elif doc.get('auto_summary'):
                print(f"\n요약: {doc['auto_summary'][:200]}")

            print(f"\n출처: {doc['url']}")
            print("-" * 80)

    def get_collection_info(self) -> Dict:
        """Get information about the collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
                "vector_size": self.vector_size,
                "embedding_model": self.embedding_model_name,
                "embedding_provider": self.embedding_provider
            }
        except Exception as e:
            return {"error": str(e)}


def main():
    """Main execution with examples."""

    print("=" * 80)
    print("QDRANT CERTIFICATION RAG SYSTEM")
    print("=" * 80)

    # Configuration - EASILY CHANGEABLE
    CONFIG = {
        # Qdrant settings
        "collection_name": "certifications",
        "use_cloud": True,  # Set to False for local storage

        # Embedding settings
        "embedding_provider": "huggingface",  # "huggingface" or "openai"
        "embedding_model": None,  # None = use default for provider
        # Defaults:
        #   - huggingface: "jhgan/ko-sroberta-multitask" (best for Korean)
        #   - openai: "text-embedding-3-large"

        # Chunking settings (set to None to disable chunking)
        "chunk_size": None,  # e.g., 500, 1000, 2000 (None = no chunking)
        "chunk_overlap": 100,  # Overlap between chunks

        # Indexing settings
        "text_field": "auto",  # "auto", "summary", "full", "combined"
        "batch_size": 32,

        # Search settings
        "top_k": 5,
        "score_threshold": None  # e.g., 0.7 to filter low-quality matches
    }

    print("\nConfiguration:")
    print(f"  Collection: {CONFIG['collection_name']}")
    print(f"  Storage: {'Qdrant Cloud' if CONFIG['use_cloud'] else 'Local'}")
    print(f"  Embedding: {CONFIG['embedding_provider']}")
    print(f"  Chunking: {'Enabled' if CONFIG['chunk_size'] else 'Disabled'}")
    if CONFIG['chunk_size']:
        print(f"    - Chunk size: {CONFIG['chunk_size']}")
        print(f"    - Overlap: {CONFIG['chunk_overlap']}")
    print()

    # Initialize RAG system
    rag = QdrantCertificationRAG(
        collection_name=CONFIG['collection_name'],
        embedding_provider=CONFIG['embedding_provider'],
        embedding_model=CONFIG['embedding_model'],
        chunk_size=CONFIG['chunk_size'],
        chunk_overlap=CONFIG['chunk_overlap'],
        use_cloud=CONFIG['use_cloud']
    )

    # Create collection (will skip if exists)
    rag.create_collection(recreate=False)

    # Check if collection is empty
    info = rag.get_collection_info()
    if info.get('points_count', 0) == 0:
        print("\n" + "=" * 80)
        print("INDEXING DOCUMENTS")
        print("=" * 80)

        # Load and index documents
        jsonl_path = "/Users/hoon/Desktop/SKN-17-Final-5Team/retrieval_test/output/certifications.jsonl"
        num_chunks = rag.load_and_index_documents(
            jsonl_path,
            batch_size=CONFIG['batch_size'],
            text_field=CONFIG['text_field']
        )
    else:
        print(f"\n✓ Collection already has {info['points_count']} indexed chunks")

    # Get collection info
    print("\n" + "=" * 80)
    print("COLLECTION INFO")
    print("=" * 80)
    info = rag.get_collection_info()
    print(f"Collection: {info.get('name')}")
    print(f"Points indexed: {info.get('points_count', 'N/A')}")
    print(f"Status: {info.get('status', 'N/A')}")
    print(f"Embedding model: {info.get('embedding_model')}")
    print(f"Vector dimension: {info.get('vector_size')}")

    # Interactive mode
    print("\n" + "=" * 80)
    print("INTERACTIVE SEARCH MODE")
    print("=" * 80)
    print("Enter search queries (or 'quit' to exit):")
    print("Special commands:")
    print("  - country:<name> - Filter by country")
    print("  - category:<name> - Filter by category")
    print()

    while True:
        try:
            user_query = input("Query: ").strip()

            if user_query.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break

            if not user_query:
                continue

            # Parse filters
            filters = {}
            if user_query.startswith("country:"):
                country = user_query.replace("country:", "").strip()
                results = rag.search("", top_k=10, filters={"country": country})
                print(f"\nAll certifications from '{country}':")
                for r in results:
                    print(f"  - {r['doc']['cert_name']}")
                continue

            elif user_query.startswith("category:"):
                category = user_query.replace("category:", "").strip()
                results = rag.search("", top_k=10, filters={"category": category})
                print(f"\nAll certifications in category '{category}':")
                for r in results:
                    print(f"  - {r['doc']['cert_name']} ({r['doc']['country']})")
                continue

            # Regular semantic search
            results = rag.search(user_query, top_k=CONFIG['top_k'])
            rag.print_results(results, user_query)
            print()

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
