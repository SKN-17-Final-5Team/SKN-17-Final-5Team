import json
import os
import re
import hashlib
from pathlib import Path
from itertools import groupby
from typing import List, Dict, Tuple

import numpy as np
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse

from openai import OpenAI
import tiktoken

from collections import Counter
from qdrant_client.models import Filter

# ============================================================
# 0. 전역 설정 (환경, OpenAI, 토크나이저, 컬렉션/파일 설정)
# ============================================================

load_dotenv()

print("[INIT] 토크나이저 로드 중...")
tokenizer = tiktoken.get_encoding("o200k_base")
print("[INIT] 토크나이저 로드 완료.\n")

print("[INIT] OpenAI 클라이언트 초기화 중...")
client_oa = OpenAI()  # OPENAI_API_KEY는 .env에서 자동 로드
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 3072
print(f"[INIT] OpenAI 임베딩 모델: {EMBED_MODEL}, dim={EMBED_DIM}\n")

# ---- Qdrant 설정 ----
QDRANT_URL = os.getenv("QDRANT_URL", None)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

COLLECTION_NAME = "trade_collection_test"

BASE_PATH = Path(__file__).parent 

# 토큰 기반 청킹 대상 문서들
DOCS_TOKEN = [
    {
        "doc_id": "incoterms",
        "path": "data/Incoterms_preprocessed.md",
        "max_tokens": 128,
        "overlap": 0.15,
    },
    {
        "doc_id": "fraud",
        "path": "data/2025무역사기예방및대응매뉴얼.md",
        "max_tokens": 128,
        "overlap": 0.15,
    },
    {
        "doc_id": "claim",
        "path": "data/무역클레임중재 50문50답(전처리).txt",
        "max_tokens": 128,
        "overlap": 0.15,
    },
    {
        "doc_id": "certification",
        "path": "data/certifications.jsonl",
        "max_tokens": 128,
        "overlap": 0.15,
    },
]

# CISG 전용 설정
CISG_DOC_ID = "CISG"
CISG_DOCUMENT_PATH = "Target_data/다자조약상세.txt"
CISG_BASE_CHUNKS_PATH = "Target_data/cisg_chunks.json"
CISG_CHUNK_STRATEGY = "Article"  # "Ho_Segmented", "Paragraph", "Article" 중 택1

# ============================================================
# A. 문서 로드
# ============================================================

def load_document(path: str | Path) -> str:
    path = Path(path)
    print(f"[LOAD] 문서 로드: {path}")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    print(f"[LOAD] 문서 길이: {len(text)} chars\n")
    return text

# ============================================================
# B-1. 토큰 기반 청킹 (incoterms / fraud / claim / certification)
# ============================================================

def chunk_by_tokens(
    text: str,
    doc_id: str,
    max_tokens: int,
    overlap_ratio: float | None = 0.15,   # 비율로 줄 때
    overlap_tokens: int | None = None     # 고정 토큰 수로 줄 때
) -> List[Dict]:
    """
    하나의 문서(text)에 대해 토큰 단위 청킹.
    - doc_id: 이 문서의 id (문서 구분용)
    - max_tokens: chunk 당 최대 토큰 수
    - overlap_ratio: 인접 chunk 간 중복 비율
    - overlap_tokens: 인접 chunk 간 고정 중복 토큰 수 (예: 50)
      -> overlap_tokens 가 주어지면 overlap_ratio 보다 우선 적용
    """

    tokens = tokenizer.encode(text)
    n = len(tokens)

    # --------- overlap 계산 ----------
    if overlap_tokens is not None:
        overlap = int(overlap_tokens)
    else:
        if overlap_ratio is None:
            overlap_ratio = 0.0
        overlap = int(max_tokens * overlap_ratio)

    if overlap >= max_tokens:
        overlap = max_tokens - 1
    step = max_tokens - overlap
    # ---------------------------------

    chunks: List[Dict] = []
    chunk_idx = 0
    i = 0

    while i < n:
        j = min(i + max_tokens, n)

        # 토큰 범위 디코딩
        chunk_text = tokenizer.decode(tokens[i:j])

        chunks.append({
            "doc_id": doc_id,
            "text": chunk_text,
        })

        i += step
        chunk_idx += 1

    print(
        f"[CHUNK] doc_id={doc_id}, max_tokens={max_tokens}, "
        f"overlap={overlap} tokens, chunks={len(chunks)}\n"
    )

    return chunks

# ============================================================
# B-2. CISG 전용: base 청크 JSON + 병합
# ============================================================

def load_base_chunks(path: str | Path) -> list:
    """기반 청크 JSON을 로드합니다."""
    path = Path(path)
    print(f"[LOAD] 기반(Base) 청크 JSON 로드: {path}")
    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks


def attach_chunk_spans(chunks: list) -> list:
    """
    cisg_chunks.json에서 불러온 raw chunk 리스트를
    내부 병합용 구조로 변환.
    최종 payload는 아니고, merge_chunks에서 Article 단위로 합칠 때만 쓸 메타 정보 포함.
    """
    print("  [CHUNK] CISG 기반 청크 전처리 중...")

    new_chunks = []
    skipped = 0

    for idx, chunk in enumerate(chunks):
        content = chunk.get("content")
        if not content:
            skipped += 1
            continue

        new_chunks.append({
            "doc_id": "CISG",
            "text": content,
            # Article 병합용 메타데이터 (내부용)
            "article": chunk.get("article"),
            "paragraph_no": chunk.get("paragraph_no"),
            "ho_no": chunk.get("ho_no"),
            "_idx": idx,  # 정렬 안정성을 위한 원래 인덱스
        })

    print(f"  [CHUNK] 사용 가능한 CISG 기본 청크: {len(new_chunks)}개 (스킵: {skipped}개)\n")
    return new_chunks

def merge_chunks(base_chunks: list, strategy_name: str, doc_id: str) -> list:
    """
    CISG 청크들을 Article 또는 Paragraph 단위로 병합.
    내부적으로는 article/paragraph 메타데이터를 이용해 묶지만,
    최종으로 반환되는 chunk에는 doc_id와 text만 남긴다.
    """
    print(f"  [Chunking] CISG 전략 '{strategy_name}' 실행 중...")

    # Ho_Segmented 그대로 사용하는 경우: 메타데이터 제거 후 그대로 반환
    if strategy_name == "Ho_Segmented":
        final_chunks = [
            {
                "doc_id": doc_id,
                "text": ch["text"],
            }
            for ch in base_chunks
            if ch.get("text")
        ]
        print(f"    [Chunking] '{strategy_name}' 완료. {len(final_chunks)}개 CISG 청크 사용.")
        return final_chunks

    # 병합 기준 key
    if strategy_name == "Paragraph":
        get_key = lambda x: (x.get("article"), x.get("paragraph_no"))
    elif strategy_name == "Article":
        get_key = lambda x: x.get("article")
    else:
        raise ValueError(f"알 수 없는 병합 전략: {strategy_name}")

    # ⚠ 여기 수정: None 이 섞여도 정렬 가능하도록 문자열로 정규화
    sorted_chunks = sorted(
        base_chunks,
        key=lambda x: (
            str(x.get("article") or ""),         # None → ""
            str(x.get("paragraph_no") or ""),    # None → ""
            str(x.get("ho_no") or ""),           # None → ""
            x.get("_idx", 0),                    # 원래 인덱스 (int)
        ),
    )

    merged_chunks: List[Dict] = []

    for key, group in groupby(sorted_chunks, key=get_key):
        # article/paragraph 정보가 없는 경우 스킵
        if key is None or (isinstance(key, tuple) and any(k is None for k in key)):
            continue

        group_list = [g for g in group if g.get("text")]
        if not group_list:
            continue

        merged_text = "\n\n".join(g["text"] for g in group_list)

        merged_chunks.append({
            "doc_id": doc_id,
            "text": merged_text,
        })

    print(f"    [Chunking] '{strategy_name}' 완료. {len(merged_chunks)}개 CISG 청크 생성.")
    return merged_chunks


# ============================================================
# B-3. 전체 청크 빌드 (토큰 문서 + CISG 병합)
# ============================================================

def build_all_chunks() -> Tuple[Dict[str, str], Dict[str, List[Dict]], List[Dict]]:
    """
    - DOCS_TOKEN 문서들: 토큰 기반 청킹
    - CISG: base_chunks + span + Article 병합 전략으로 청킹
    반환:
      - doc_texts: {doc_id: text}
      - chunks_by_doc: {doc_id: [chunks...]}
      - flat_chunks: 전체 문서 청크 리스트
    """
    doc_texts: Dict[str, str] = {}
    chunks_by_doc: Dict[str, List[Dict]] = {}
    flat_chunks: List[Dict] = []

    # 1) 토큰 기반 문서들
    for doc in DOCS_TOKEN:
        doc_id = doc["doc_id"]
        path = doc["path"]

        text = load_document(path)
        doc_texts[doc_id] = text

        max_tokens = doc.get("max_tokens", 128)
        overlap_cfg = doc.get("overlap", 0.15)

        # overlap 해석: <1 이면 ratio, >=1 이면 "토큰 수"
        if isinstance(overlap_cfg, (int, float)) and overlap_cfg >= 1:
            overlap_ratio = None
            overlap_tokens = int(overlap_cfg)
        else:
            overlap_ratio = float(overlap_cfg)
            overlap_tokens = None

        chunks = chunk_by_tokens(
            text=text,
            doc_id=doc_id,
            max_tokens=max_tokens,
            overlap_ratio=overlap_ratio,
            overlap_tokens=overlap_tokens,
        )

        chunks_by_doc[doc_id] = chunks
        flat_chunks.extend(chunks)

        # 2) CISG 전용 문서
    cisg_text = load_document(CISG_DOCUMENT_PATH)
    doc_texts[CISG_DOC_ID] = cisg_text

    base_chunks_raw = load_base_chunks(CISG_BASE_CHUNKS_PATH)
    base_chunks_ready = attach_chunk_spans(base_chunks_raw)
    cisg_chunks = merge_chunks(base_chunks_ready, CISG_CHUNK_STRATEGY, doc_id=CISG_DOC_ID)
    chunks_by_doc[CISG_DOC_ID] = cisg_chunks
    flat_chunks.extend(cisg_chunks)

    # doc_texts : 문서 원문을 그대로 저장한 딕셔너리.
    # chunks_by_docs : 문서별로 청크 리스트를 따로 저장한 구조.
    # flat_chunks : 모든 문서의 청크를 하나의 리스트로 모아둔 것.
    return doc_texts, chunks_by_doc, flat_chunks

# ============================================================
# C. 임베딩, 컬렉션 생성, 업서트
# ============================================================

def get_embeddings(texts: List[str]) -> np.ndarray:
    """
    OpenAI text-embedding-3-large 모델로부터 임베딩을 얻는 함수.
    texts: 문자열 리스트 또는 문자열 1개
    return: (len(texts), EMBED_DIM) numpy 배열
    """
    if isinstance(texts, str):
        texts = [texts]

    cleaned_texts = []
    for i, t in enumerate(texts):
        if t is None:
            print(f"[WARN] get_embeddings: index {i} 에 None 이 들어와서 스킵합니다.")
            continue
        if not isinstance(t, str):
            print(f"[WARN] get_embeddings: index {i} 타입이 {type(t)} 여서 str()로 변환합니다.")
            t = str(t)
        t = t.strip()
        if not t:
            print(f"[WARN] get_embeddings: index {i} 가 빈 문자열이라 스킵합니다.")
            continue
        cleaned_texts.append(t)

    if not cleaned_texts:
        print("[WARN] get_embeddings: 유효한 텍스트가 없어 빈 배열을 리턴합니다.")
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    resp = client_oa.embeddings.create(
        model=EMBED_MODEL,
        input=cleaned_texts,
    )
    vectors = [item.embedding for item in resp.data]
    return np.array(vectors, dtype=np.float32)


def create_collection_for_chunks(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    print(f"[QDRANT] 컬렉션 생성/재생성: {collection_name}")
    try:
        client.delete_collection(collection_name)
        print(f"[QDRANT] 기존 컬렉션 삭제 완료: {collection_name}")
    except UnexpectedResponse:
        print(f"[QDRANT] 기존 컬렉션 없음 또는 삭제 실패 무시: {collection_name}")

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )
    print(f"[QDRANT] 컬렉션 준비 완료 (dim={vector_size}, metric=COSINE)\n")


def upload_chunks_to_qdrant(
    client: QdrantClient,
    collection_name: str,
    chunks: List[Dict],
    batch_size: int = 50,
) -> None:
    print(f"[QDRANT] 청크 임베딩 계산 및 업로드 시작 (collection={collection_name})")

    texts = [c["text"] for c in chunks]
    print(f"[QDRANT] 임베딩 계산 대상 chunk 수: {len(texts)}")

    total = len(texts)
    dim = None
    global_point_id = 0   # Qdrant용 숫자 ID 카운터

    print(f"[QDRANT] 총 {total}개 포인트를 배치({batch_size})로 업로드 시작...")

    for i in range(0, total, batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_chunks = chunks[i:i + batch_size]

        batch_idx = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        print(f"    [EMBED] 배치 {batch_idx}/{total_batches} ({len(batch_texts)}개) 임베딩 계산 중...")

        # 1) 임베딩 계산
        batch_embeddings = get_embeddings(batch_texts)

        if dim is None and batch_embeddings.shape[0] > 0:
            dim = batch_embeddings.shape[1]
            print(f"[QDRANT] 임베딩 차원: {dim}")

        # 2) PointStruct 생성 (id는 정수 사용)
        points = []
        for vec, ch in zip(batch_embeddings, batch_chunks):
            points.append(
                PointStruct(
                    id=global_point_id,
                    vector=vec.tolist(),
                    payload=ch,
                )
            )
            global_point_id += 1

        # 3) Qdrant upsert
        print(f"    [UPSERT] 배치 {batch_idx}/{total_batches} (points={len(points)}) 업로드 중...")
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )

    print(f"[QDRANT] 모든 배치 업로드 완료. 총 포인트 수: {global_point_id}\n")

def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    # return QdrantClient(
    #         host="127.0.0.1",
    #         port=6333,
    #         timeout=300,
    # )

def get_doc_ids_from_collection(client: QdrantClient, collection_name: str):
    all_doc_ids = []
    next_page = None

    while True:
        points, next_page = client.scroll(
            collection_name=collection_name,
            limit=1000,
            with_payload=True,
            offset=next_page,
        )

        if not points:
            break

        for p in points:
            payload = p.payload or {}
            doc_id = payload.get("doc_id")
            if doc_id is not None:
                all_doc_ids.append(doc_id)

        if next_page is None:
            break

    # 종류만 보고 싶으면 set, 개수까지 보고 싶으면 Counter
    unique_doc_ids = sorted(set(all_doc_ids))
    counts = Counter(all_doc_ids)

    print("[DOC_ID 종류]")
    for d in unique_doc_ids:
        print(f"  - {d}: {counts[d]} points")

    return unique_doc_ids, counts

def main():
    client = get_qdrant_client()

    doc_texts, chunks_by_doc, flat_chunks = build_all_chunks()

    create_collection_for_chunks(client, COLLECTION_NAME, EMBED_DIM)
    upload_chunks_to_qdrant(client, COLLECTION_NAME, flat_chunks)
    get_doc_ids_from_collection(client, COLLECTION_NAME)

if __name__ == "__main__":
    main()