import numpy as np
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http.exceptions import UnexpectedResponse

from openai import OpenAI
from dotenv import load_dotenv

import tiktoken
from typing import List

# =========================
# 0. 전역 설정 (OpenAI, Tokenizer)
# =========================

load_dotenv()

client_oa = OpenAI()
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072  # text-embedding-3-large의 출력 차원

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

def chunk_by_tokens(text: str, max_tokens: int, overlap_ratio: float = 0.15):
    tokens = tokenizer.encode(text)
    n = len(tokens)

    # 1) 토큰 → 원문 char offset 계산
    offsets = [0]
    cur_text = ""
    for tok in tokens:
        cur_text += tokenizer.decode([tok])
        offsets.append(len(cur_text))

    # 2) Overlap 계산
    overlap = int(max_tokens * overlap_ratio)
    step = max_tokens - overlap

    chunks = []
    chunk_idx = 0
    i = 0

    while i < n:
        j = min(i + max_tokens, n)

        start_char = offsets[i]
        end_char = offsets[j]
        chunk_text = text[start_char:end_char]

        chunks.append({
            "id": f"tok_{max_tokens}_{chunk_idx}",
            "text": chunk_text,
            "start": start_char,
            "end": end_char,
        })

        # 3) 다음 청크 시작 위치 (Overlapping)
        i += step
        chunk_idx += 1

    print(f"토큰 청킹 완료: max_tokens={max_tokens}, overlap={overlap}, chunks={len(chunks)}\n")
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
def ensure_payload_index(client: QdrantClient, collection_name: str):
    """data_source 필드에 payload index가 있는지 확인하고 없으면 생성"""
    try:
        # data_source 필드에 keyword index 생성
        client.create_payload_index(
            collection_name=collection_name,
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

def delete_by_data_source(client: QdrantClient, collection_name: str, data_source: str):
    """
    특정 data_source의 포인트만 삭제

    Args:
        client: Qdrant 클라이언트
        collection_name: 컬렉션 이름
        data_source: 삭제할 데이터 소스 (예: 'Incoterms', 'fraud', etc.)
    """
    try:
        # payload index 확인 및 생성
        ensure_payload_index(client, collection_name)

        # 기존 포인트 수 확인
        collection_info = client.get_collection(collection_name)
        before_count = collection_info.points_count

        print(f"삭제 전 총 포인트 수: {before_count}")
        print(f"'{data_source}' 데이터 소스 삭제 중...")

        # data_source 필터로 삭제
        client.delete(
            collection_name=collection_name,
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
        collection_info = client.get_collection(collection_name)
        after_count = collection_info.points_count
        deleted_count = before_count - after_count

        print(f"✓ '{data_source}' 데이터 {deleted_count}개 삭제 완료")
        print(f"삭제 후 총 포인트 수: {after_count}")

    except Exception as e:
        print(f"삭제 중 오류 발생: {e}")
        raise


def create_collection_for_chunks(client: QdrantClient, collection_name: str, vector_size: int):
    # 존재 여부 확인
    try:
        client.get_collection(collection_name)
        print(f"이미 존재하는 컬렉션 사용: {collection_name}")
        return
    except Exception:
        print(f"컬렉션 없음 → 새로 생성: {collection_name}")

    # 컬렉션 생성
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )
    print(f"컬렉션 생성 완료: {collection_name}")


def upload_chunks_to_qdrant(client: QdrantClient, collection_name: str, chunks, batch_size: int = 20):
    texts = [c["text"] for c in chunks]
    print(f"임베딩 계산 대상 청크 수: {len(texts)}")

    embeddings = get_embeddings(texts)
    dim = embeddings.shape[1]

    points = []
    for idx, (vec, ch) in enumerate(zip(embeddings, chunks)):
        payload = {
            "id": ch["id"],
            "text": ch["text"],
            "data_source": 'Incoterms'
        }

        points.append(
            PointStruct(
                id=idx,
                vector=vec.tolist(),
                payload=payload,
            )
        )

    # 배치 업로드
    total_points = len(points)
    total_batches = (total_points + batch_size - 1) // batch_size
    print(f"배치 업로드 시작 (총 {total_points}개, 배치 크기: {batch_size})...")

    for i in range(0, total_points, batch_size):
        batch = points[i:i + batch_size]
        batch_num = i // batch_size + 1
        client.upsert(
            collection_name=collection_name,
            points=batch,
            wait=True
        )
        print(f"  - 배치 {batch_num}/{total_batches} 업로드 완료 ({len(batch)}개)")

    print(f"[QDRANT] 업서트 완료: {total_points}개 포인트")



# =========================
# 5. main
# =========================

def main(update_existing: bool = False):
    """
    메인 실행 함수

    Args:
        update_existing: True면 기존 'Incoterms' 데이터를 삭제하고 새로 업로드 (업데이트 모드)
                        False면 기존 데이터에 추가 (중복 가능)
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DOCUMENT_PATH = os.path.join(BASE_DIR, "used_data", "Incoterms_preprocessed2.md")
    COLLECTION_NAME = "trade_collection"
    MAX_TOKENS = 128

    # 1) 문서 로드
    text = load_document(DOCUMENT_PATH)

    # 2) 청킹
    chunks_tok = chunk_by_tokens(text, MAX_TOKENS, 0.15)

    # 3) Qdrant 연결
    print("Qdrant 연결 시도")
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=300  # 5분 타임아웃 (대용량 업로드 대비)
    )
    print("Qdrant 연결 완료")

    # 4) 컬렉션 생성
    create_collection_for_chunks(client, COLLECTION_NAME, EMBED_DIM)

    # 5) 업데이트 모드: 기존 Incoterms 데이터 삭제
    if update_existing:
        delete_by_data_source(client, COLLECTION_NAME, 'Incoterms')

    # 6) 청크 업로드
    upload_chunks_to_qdrant(client, COLLECTION_NAME, chunks_tok)

    # 최종 상태 확인
    collection_info = client.get_collection(COLLECTION_NAME)
    print(f"✓ 컬렉션 '{COLLECTION_NAME}' 총 포인트 수: {collection_info.points_count}")


if __name__ == "__main__":
    # update_existing=True: 기존 Incoterms 데이터를 삭제하고 새로 인덱싱 (다른 소스는 유지)
    # update_existing=False: 기존 데이터에 추가 (중복 가능)
    main(update_existing=True)
