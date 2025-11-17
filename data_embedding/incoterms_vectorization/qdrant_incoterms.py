import json
import numpy as np
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse

from openai import OpenAI
from dotenv import load_dotenv

import tiktoken
from typing import List

# =========================
# 0. 전역 설정 (OpenAI, 토크나이저)
# =========================

load_dotenv()

client_oa = OpenAI()
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072  # text-embedding-3-large의 output dim

tokenizer = tiktoken.get_encoding("o200k_base")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# =========================
# 1. 데이터 로드 함수
# =========================

def load_document(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    print(f"[LOAD] 문서 로드 완료: {path}, 길이={len(text)} chars")
    return text


# =========================
# 2. 토큰 기반 청킹
# =========================

def chunk_by_tokens(text: str, max_tokens: int):
    tokens = tokenizer.encode(text)
    chunks = []

    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i : i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens)

        chunks.append({
            "id": f"tok_{max_tokens}_{i}",  
            "text": chunk_text,             
        })

    print(f"토큰 청킹 완료: max_tokens={max_tokens}, chunks={len(chunks)}")
    return chunks


# =========================
# 3. OpenAI 임베딩 함수
# =========================

def get_embeddings(texts: List[str]) -> np.ndarray:
    if isinstance(texts, str):
        texts = [texts]

    resp = client_oa.embeddings.create(
        model=EMBED_MODEL,
        input=texts,
    )
    vectors = [item.embedding for item in resp.data]
    return np.array(vectors, dtype=np.float32)


# =========================
# 4. Qdrant 관련 함수
# =========================
def create_collection_for_chunks(client: QdrantClient, collection_name: str, vector_size: int):
    # 1) 존재 여부 확인
    try:
        info = client.get_collection(collection_name)
        print(f"이미 존재하는 컬렉션 사용: {collection_name}")
        return
    except Exception:
        print(f"컬렉션 없음 → 새로 생성: {collection_name}")

    # 2) 새 컬렉션 생성
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )
    print(f"컬렉션 생성 완료: {collection_name}")


def upload_chunks_to_qdrant(client: QdrantClient, collection_name: str, chunks):
    texts = [c["text"] for c in chunks]
    print(f"임베딩 계산 대상 청크 수: {len(texts)}")

    embeddings = get_embeddings(texts)
    dim = embeddings.shape[1]

    points = []
    for idx, (vec, ch) in enumerate(zip(embeddings, chunks)):
        points.append(
            PointStruct(
                id=idx,              
                vector=vec.tolist(), 
                payload=ch,         
            )
        )

    client.upsert(
        collection_name=collection_name,
        points=points,
    )
    print(f"[QDRANT] 업서트 완료: {len(points)}개 포인트")


# =========================
# 5. main
# =========================

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DOCUMENT_PATH = os.path.join(BASE_DIR, "used_data", "Incoterms_preprocessed(1).md")
    COLLECTION_NAME = "trade_collection"
    MAX_TOKENS = 2048

    # (1) 문서 로드
    text = load_document(DOCUMENT_PATH)

    # (2) 청킹
    chunks_tok = chunk_by_tokens(text, MAX_TOKENS)

    # (3) Qdrant 연결
    print("Qdrant 연결 시도")
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )
    print("Qdrant 연결 완료.")

    # (4) 컬렉션 생성
    create_collection_for_chunks(client, COLLECTION_NAME, EMBED_DIM)

    # (5) 청크 업로드
    upload_chunks_to_qdrant(client, COLLECTION_NAME, chunks_tok)


if __name__ == "__main__":
    main()
