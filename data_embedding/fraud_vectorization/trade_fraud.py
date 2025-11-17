import os
import uuid
import time
import tiktoken
from docx import Document
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
CHUNKS_FILE = "2025무역사기대응매뉴얼.md"   # ← 여기만 네 파일명에 맞게 수정하면 됨
COLLECTION_NAME = "trade_fraud"

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
    """
    Qdrant 컬렉션을 새로 만들거나 덮어씀.
    """
    print(f"Qdrant 컬렉션 재생성: {COLLECTION_NAME}, dim={vector_dim}")
    qdrant.recreate_collection(
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

def search_trade_fraud(
    query_text: str,
    top_k: int = 1,
    include_vectors: bool = False,
):
    """질문문(query_text)을 trade_fraud 컬렉션에서 검색해 상위 top_k 결과를 미리보기."""
    if not query_text.strip():
        raise ValueError("query_text를 입력하세요.")

    query_vec = client.embeddings.create(
        model=EMBED_MODEL,
        input=query_text,
    ).data[0].embedding

    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=top_k,
        with_payload=True,
        with_vectors=include_vectors,
    )

    if not hits:
        print("검색 결과가 없습니다.")
        return []

    print(f"=== trade_fraud 검색 결과 (top {top_k}) ===")
    for idx, hit in enumerate(hits, start=1):
        payload = hit.payload or {}
        text_preview = (payload.get("text") or "").replace("\n", " ")
        if len(text_preview) > 150:
            text_preview = text_preview[:147] + "..."
        print(f"[{idx}] score={hit.score:.4f} | id={hit.id} | text={text_preview}")

    return hits


import re
import math
import json

def normalize_text(s: str) -> str:
    """공백/줄바꿈 정리해서 비교하기 쉽게 만드는 함수"""
    if not s:
        return ""
    # 양쪽 공백 제거 + 연속 공백은 한 칸으로
    return re.sub(r"\s+", " ", s.strip())


def is_relevant_chunk(chunk_text: str, gold_answer: str) -> bool:
    """
    청크 텍스트가 정답 텍스트와 '같은 내용'이라고 볼 만한지 판단.
    1순위: gold_answer 전체가 chunk 안에 그대로 포함되면 True
    2순위: 키워드 겹치는 개수로 조금 느슨하게 판단
    """
    chunk_norm = normalize_text(chunk_text)
    gold_norm = normalize_text(gold_answer)

    if not gold_norm:
        return False

    # 1) 전체 문자열 포함
    if gold_norm in chunk_norm:
        return True

    # 2) 백업: 키워드 겹침
    gold_tokens = re.split(r"[^\w가-힣]+", gold_norm)
    gold_keywords = [t for t in gold_tokens if len(t) >= 2][:10]  # 너무 많으면 상위 10개

    if not gold_keywords:
        return False

    overlap = 0
    for kw in gold_keywords:
        if kw and kw in chunk_norm:
            overlap += 1

    # 키워드 3개 이상 겹치면 관련 있다고 간주 (필요하면 숫자 조절 가능)
    return overlap >= 3

def compute_metrics_from_hits(hits, gold_answer: str, k: int = None):
    """
    hits: search_trade_fraud 결과 리스트
    gold_answer: 정답 텍스트 (eval_queries(gold).jsonl에서 읽은 것)
    k: top-k (None이면 hits 전체 사용)

    반환: {"recall": float, "mrr": float}
    """
    if k is not None:
        hits = hits[:k]

    if not hits:
        return {"recall": 0.0, "mrr": 0.0}

    best_rank = None

    for rank, h in enumerate(hits, start=1):
        payload = h.payload or {}
        chunk_text = payload.get("text", "")  # ← Qdrant에 텍스트를 "text"로 넣었다는 전제
        if not chunk_text:
            continue

        if is_relevant_chunk(chunk_text, gold_answer):
            best_rank = rank
            break

    if best_rank is None:
        return {"recall": 0.0, "mrr": 0.0}

    # 질문 하나당 정답 청크는 1개라는 가정 → recall은 0 또는 1
    recall = 1.0
    mrr = 1.0 / best_rank

    return {"recall": recall, "mrr": mrr}

def load_eval_queries_with_gold(path: str = "eval_queries(gold).jsonl"):
    """eval_queries(gold).jsonl 로드해서 리스트로 반환"""
    eval_data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            eval_data.append(obj)
    print(f"로드한 평가용 질문 개수: {len(eval_data)}")
    return eval_data


def evaluate_trade_fraud_retriever_text(
    path: str = "eval_queries(gold).jsonl",
    top_k: int = 5,
):
    """
    gold_ids / retrieved_ids 없이,
    Qdrant에서 가져온 청크 텍스트와 gold_answer 텍스트만으로
    recall / MRR 계산하는 버전.
    """
    eval_data = load_eval_queries_with_gold(path)

    sum_recall = 0.0
    sum_mrr = 0.0
    n_evaluated = 0

    for i, item in enumerate(eval_data, start=1):
        query = item["query"]
        gold_answer = item.get("gold_answer")

        if not gold_answer:
            print(f"[WARN] gold_answer 없음 → 쿼리 {i} 스킵")
            continue

        print("\n----------------------------------------")
        print(f"[{i}] 질문")
        print("  ", query)
        print("  정답 텍스트 (앞 120자):", normalize_text(gold_answer)[:120],
              "..." if len(gold_answer) > 120 else "")

        # 1) Qdrant 검색
        hits = search_trade_fraud(
            query_text=query,
            top_k=top_k,
            include_vectors=False,
        )

        print(f"  검색 결과 개수: {len(hits)}")

        # 2) metrics 계산 (텍스트 기반)
        metrics = compute_metrics_from_hits(
            hits=hits,
            gold_answer=gold_answer,
            k=top_k,
        )

        print(f"  recall@{top_k}={metrics['recall']:.3f}, "
              f"mrr@{top_k}={metrics['mrr']:.3f}")

        sum_recall += metrics["recall"]
        sum_mrr += metrics["mrr"]
        n_evaluated += 1

    if n_evaluated == 0:
        print("평가 가능한 쿼리가 없습니다.")
        return

    avg_recall = sum_recall / n_evaluated
    avg_mrr = sum_mrr / n_evaluated
    avg_of_two = (avg_recall + avg_mrr) / 2.0

    print("\n========== 최종 결과 ==========")
    print(f"평가 쿼리 수 : {n_evaluated}")
    print(f"TOP_K: {top_k}")
    print(f"Recall@{top_k} 평균 : {avg_recall:.4f}")
    print(f"MRR@{top_k} 평균 : {avg_mrr:.4f}")
    print(f"지표 평균: {avg_of_two:.4f}")
    print("================================\n")

    return {
        "n_evaluated": n_evaluated,
        "avg_recall": avg_recall,
        "avg_mrr": avg_mrr,
        "avg_of_two": avg_of_two,
    }

def main():
    QA_PATH = "eval_queries(gold).jsonl"

    _ = evaluate_trade_fraud_retriever_text(
        path=QA_PATH,
        top_k=5,   # top_k 바꾸면서 실험 가능
    )

if __name__ == "__main__":
    main()
