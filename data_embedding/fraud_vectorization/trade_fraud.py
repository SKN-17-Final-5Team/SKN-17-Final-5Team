import os
import uuid
import time
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
# ================================================================
load_dotenv()

EMBED_MODEL = 'text-embedding-3-large'
MAX_TOKENS = 2048     # 청크 하나당 최대 토큰 수
OVERLAP = 100         # 청크 간 토큰 겹침
BATCH_SIZE = 16

# 폴더 대신 단일 파일 경로 사용
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_FILE = os.path.join(BASE_DIR, "used_data", "2025무역사기대응매뉴얼.md")
COLLECTION_NAME = "trade_collection"

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

encoding = tiktoken.encoding_for_model(EMBED_MODEL)


def chunk_text(text, max_tokens=MAX_TOKENS, overlap=OVERLAP):
    """
    긴 텍스트를 토큰 기준으로 잘라서 리스트로 반환.
    - max_tokens: 청크 하나당 최대 토큰 수
    - overlap: 이전 청크와 겹치게 할 토큰 수
    """
    tokens = encoding.encode(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = start + max_tokens
        chunk = encoding.decode(tokens[start:end])
        chunks.append(chunk)
        # 다음 청크 시작점 = 현재 시작점 + (max_tokens - overlap)
        start += max_tokens - overlap

    return chunks


def load_chunks_from_file(file_path: str = CHUNKS_FILE):
    """
    단일 .md(또는 .txt) 파일을 읽어서
    토큰 기준으로 청킹한 결과를 리스트로 반환.
    각 원소는 {id, text, file_name, chunk_index, chunk_id} 딕셔너리.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    records = []

    # 파일명만 분리 (메타데이터용)
    filename = os.path.basename(file_path)

    # 전체 텍스트 로드
    with open(file_path, "r", encoding="utf-8") as f:
        full_text = f.read().strip()

    # 토큰 기준 청킹
    token_chunks = chunk_text(full_text)

    # 이 문서 전체를 대표하는 chunk_id
    # (eval jsonl의 gold_chunk_ids랑 맞추려면, 여기 문자열을 그 포맷에 맞게 설정해야 함)
    doc_chunk_id = filename   # 예: "2025무역사기대응매뉴얼.md"

    for idx, chunk in enumerate(token_chunks):
        records.append(
            {
                "id": str(uuid.uuid4()),   # Qdrant point id
                "text": chunk,             # 실제 청크 텍스트
                "file_name": filename,     # 원본 파일명
                "chunk_index": idx,        # 같은 파일 내 몇 번째 청크인지
                "chunk_id": doc_chunk_id,  # 문서 단위 ID
            }
        )

    print(f"총 청크 개수: {len(records)}")
    return records

def embed_batch(text_list, max_retries: int = 5):
    """텍스트 리스트 한 배치를 임베딩. RateLimit 걸리면 지수 백오프로 재시도."""
    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(
                model=EMBED_MODEL,
                input=text_list,
            )
            return [item.embedding for item in resp.data]
        except RateLimitError as e:
            wait = 2 ** attempt
            print(f"Rate limit 발생, {wait}초 후 재시도... ({e})")
            time.sleep(wait)
        except APIError as e:
            print("OpenAI APIError 발생:", e)
            raise
    raise RuntimeError("임베딩 재시도 최대 횟수 초과")

def embed_all(records):
    """records 리스트 전체에 대해 배치 임베딩을 수행하고 벡터 리스트를 반환"""
    texts = [r["text"] for r in records]
    all_vectors = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]
        print(f"임베딩 배치: {start} ~ {start + len(batch) - 1}")
        vectors = embed_batch(batch)
        all_vectors.extend(vectors)

    print(f"임베딩 완료: {len(all_vectors)}개")
    return all_vectors

# ================== 4. Qdrant 컬렉션 생성 ==================
def setup_qdrant_collection(vector_dim: int):
    try:
        client.get_collection(COLLECTION_NAME)
        print(f"이미 존재하는 컬렉션 사용: {COLLECTION_NAME}")
        return
    except Exception:
        print(f"컬렉션 없음 → 새로 생성: {COLLECTION_NAME}")

    # 2) 새 컬렉션 생성
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_dim,
            distance=Distance.COSINE,
        ),
    )


def upload_to_qdrant(records, vectors):
    """records + vectors를 Qdrant에 배치 업서트"""
    assert len(records) == len(vectors), "records와 vectors 길이가 다릅니다."

    points = []

    for rec, vec in zip(records, vectors):
        point = PointStruct(
            id=rec["id"],
            vector=vec,
            payload={
                "text": rec["text"],
                "chunk_id": rec["chunk_id"],        
                "file_name": rec["file_name"],       
                "chunk_index": rec["chunk_index"],   
            },
        )
        points.append(point)

    for start in range(0, len(points), BATCH_SIZE):
        batch_points = points[start : start + BATCH_SIZE]
        print(f"Qdrant 업서트: {start} ~ {start + len(batch_points) - 1}")
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=batch_points,
        )

    print("Qdrant 업서트 완료!")


# ================== 6. 전체 실행 ==================
def main():
    # 1) 폴더에서 텍스트 로드 + 토큰 청킹
    records = load_chunks_from_file() 
    if not records:
        print("청크가 없습니다. 폴더/파일을 확인하세요.")
        return

    # 2) 임베딩
    vectors = embed_all(records)

    # 3) Qdrant 컬렉션 생성 (임베딩 차원에 맞게)
    vector_dim = len(vectors[0])
    setup_qdrant_collection(vector_dim)

    # 4) Qdrant에 포인트 업로드
    upload_to_qdrant(records, vectors)

    print("모든 작업 완료")

if __name__ == "__main__":
    main()