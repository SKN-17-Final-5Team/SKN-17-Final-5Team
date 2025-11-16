import json
import os
from typing import List, Dict, Optional, Literal
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np
import uuid


load_dotenv()


class CertificationQdrant:
    """Qdrant vector database builder class for certification data"""

    def __init__(
        self,
        collection_name: str = "trade_collection",
        embedding_provider: Literal["openai"] = "openai",
        embedding_model: Optional[str] = None,
        chunk_size: Optional[int] = 1000,
        chunk_overlap: Optional[int] = 100,
        use_cloud: bool = True
    ):
        """Initialize RAG system.

        Args:
            collection_name: Qdrant collection name
            embedding_provider: "openai" (only supported provider)
            embedding_model: Model name (None = provider default)
            chunk_size: Text chunk size (None = no chunking)
            chunk_overlap: Overlap between chunks
            use_cloud: Use Qdrant Cloud (True) or local (No support)
        """
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize embedding model
        if embedding_provider == "openai":
            self._init_openai_embeddings(embedding_model)
        else:
            raise ValueError(f"Unsupported embedding provider: {embedding_provider}")

        # Initialize Qdrant client
        if use_cloud:
            self._init_qdrant_cloud()
        else:
            raise ValueError("Local Qdrant is not supported. Set use_cloud=True.")


    def _init_openai_embeddings(self, model: Optional[str] = None):
        """Initialize OpenAI embeddings."""
        try:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.embedding_model_name = model or "text-embedding-3-large"
            self.embedding_dimension = 3072 if "large" in self.embedding_model_name else 1536
            print(f"✓ OpenAI embeddings initialized: {self.embedding_model_name}")
        except ImportError:
            raise ImportError("Install openai: pip install openai")


    def _init_qdrant_cloud(self):
        """Initialize Qdrant Cloud client."""
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")

        if not url or not api_key:
            raise ValueError("Set QDRANT_URL and QDRANT_API_KEY in .env")

        self.client = QdrantClient(url=url, api_key=api_key)
        print(f"✓ Connected to Qdrant Cloud")


    def create_collection(self, recreate: bool = False) -> None:
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            exists = False
            print(f"✓ Deleted existing collection: {self.collection_name}")

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimension,
                    distance=Distance.COSINE
                )
            )
            print(f"✓ Created collection: {self.collection_name}")
        else:
            print(f"✓ Collection exists: {self.collection_name}")

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI."""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model_name,
            input=text
        )
        return response.data[0].embedding

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks."""
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
        batch_size: int = 32
    ) -> int:
        """Load and index documents from JSONL file.

        Args:
            jsonl_path: Path to JSONL file
            text_field: Which field to embed ("auto", "summary", "full", "combined")
            batch_size: Batch size for embedding

        Returns:
            Number of chunks indexed
        """
        print(f"\nLoading documents from: {jsonl_path}")

        # Load documents
        documents = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                documents.append(json.loads(line))

        print(f"✓ Loaded {len(documents)} documents")

        # Prepare metadata and texts
        doc_metadata = []
        texts_to_embed = []

        for doc in documents:
            # Select text field
            if text_field == "summary":
                text = doc.get('auto_summary', '') or doc.get('cert_subject', '')
            elif text_field == "full":
                text = doc.get('cert_subject', '')
            elif text_field == "combined":
                text = f"{doc.get('auto_summary', '')}\n\n{doc.get('cert_subject', '')}"
            else:  # auto
                text = doc.get('auto_summary', '') or doc.get('cert_subject', '')

            # Chunk text
            chunks = self.chunk_text(text)

            for chunk_idx, chunk in enumerate(chunks):
                doc_metadata.append({
                    'doc': doc,
                    'chunk_idx': chunk_idx,
                    'total_chunks': len(chunks),
                    'chunk_text': chunk
                })
                texts_to_embed.append(chunk)

        print(f"✓ Created {len(texts_to_embed)} chunks from {len(documents)} documents")

        # Generate embeddings in batches
        print(f"\nGenerating embeddings...")
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
                print(f"  Progress: {min(i + batch_size, len(texts_to_embed))}/{len(texts_to_embed)}")

        print(f"✓ Generated {len(all_embeddings)} embeddings")

        # Create points
        print(f"\nCreating Qdrant points...")
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

        # Upload to Qdrant
        upload_batch_size = 50
        print(f"Uploading to Qdrant (batch_size={upload_batch_size})...")

        for i in range(0, len(points), upload_batch_size):
            batch = points[i:i + upload_batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )

            if (i + upload_batch_size) % (upload_batch_size * 5) == 0:
                print(f"  Progress: {min(i + upload_batch_size, len(points))}/{len(points)}")

        print(f"✓ Uploaded {len(points)} points to {self.collection_name}")
        return len(points)

    def get_collection_info(self) -> Dict:
        """Get collection information."""
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
