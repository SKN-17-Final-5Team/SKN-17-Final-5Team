import os
import json
from typing import List, Dict

import numpy as np
import tiktoken

import qdrant_client
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse

from dotenv import load_dotenv
from openai import OpenAI, BadRequestError


# =========================
# 0. 전역 설정 (OpenAI, 토크나이저)
# =========================

load_dotenv()

print("[INIT] 토크나이저 로드 중...")
tokenizer = tiktoken.get_encoding("o200k_base")
print("[INIT] 토크나이저 로드 완료.\n")

print("[INIT] OpenAI 클라이언트 초기화 중...")
client_oa = OpenAI()

EMBED_MODEL = "text-embedding-3-large"   # 필요하면 text-embedding-3-small 로 바꿔도 됨
EMBED_DIM = 3072                         # small 쓰면 1536
print(f"[INIT] OpenAI 임베딩 모델: {EMBED_MODEL}, dim={EMBED_DIM}\n")


# =========================
# 0.5 OpenAI 임베딩 함수
# =========================

def get_embeddings(texts: List[str]) -> np.ndarray:
    """
    OpenAI text-embedding-3-large 모델로부터 임베딩을 얻는 함수.
    texts: 문자열 리스트 또는 단일 문자열
    return: (len(texts), EMBED_DIM) numpy 배열
    """
    # 1) 단일 문자열이면 리스트로 변환
    if isinstance(texts, str):
        texts = [texts]

    # 2) numpy array 등도 리스트로 변환
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()

    # 3) 제너레이터/iterable 방지용으로 한 번 리스트화
    texts = list(texts)

    # 4) None / 비문자열 정리 및 문자열로 강제 캐스팅
    cleaned: List[str] = []
    for t in texts:
        if t is None:
            continue
        if not isinstance(t, str):
            t = str(t)
        t = t.strip()
        if t == "":
            continue
        cleaned.append(t)

    if not cleaned:
        raise ValueError("get_embeddings: 유효한 입력 텍스트가 없습니다.")

    print(f"[EMBED] 전체 입력 개수: {len(cleaned)}")

    # ---- 배치로 나눠서 호출 ----
    all_vectors: List[List[float]] = []
    BATCH_SIZE = 128   # 너무 크면 400/429 날 수 있으니 적당히

    for i in range(0, len(cleaned), BATCH_SIZE):
        batch = cleaned[i:i + BATCH_SIZE]
        print(f"[EMBED] 배치 {i // BATCH_SIZE + 1}: {len(batch)}개")

        try:
            resp = client_oa.embeddings.create(
                model=EMBED_MODEL,
                input=batch,
            )
        except BadRequestError as e:
            print("[EMBED][ERROR] BadRequestError 발생")
            print("  message:", e)
            print("  배치 예시 텍스트 (앞 1개):", batch[0][:200])
            raise

        for item in resp.data:
            all_vectors.append(item.embedding)

    vectors_np = np.array(all_vectors, dtype=np.float32)
    print(f"[EMBED] 최종 벡터 shape: {vectors_np.shape}")

    return vectors_np


# =========================
# 1. 데이터 로드 함수
# =========================

def load_document(path: str) -> str:
    print(f"[LOAD] 문서 로드: {path}")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    print(f"[LOAD] 문서 길이: {len(text)} chars\n")
    return text


def load_qa(path: str) -> List[Dict]:
    """
    QA 파일 로더
    - .json  : JSON 배열  [ {question:..., answer_text:...}, ... ]
    - .jsonl : JSON Lines 형식
        {"question": "...", "answer_text": "..."}
        {"question": "...", "answer_text": "..."}
    - 확장자가 애매해도, 내용 보고 json / jsonl 자동 판별
    """
    print(f"[LOAD] QA 세트 로드: {path}")
    qa_list: List[Dict] = []

    ext = os.path.splitext(path)[1].lower()

    # 1) jsonl 확장자면 무조건 JSON Lines 로 처리
    if ext == ".jsonl":
        with open(path, encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    qa_list.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  ❌ JSONDecodeError at line {line_no}: {repr(line[:80])}")
                    raise e

    # 2) 그 외(.json, .txt 등) 는 내용 보고 json / jsonl 판별
    else:
        with open(path, encoding="utf-8-sig") as f:
            first_non_empty = None
            lines = f.readlines()

        for line in lines:
            s = line.strip()
            if s:
                first_non_empty = s
                break

        # 배열로 시작하면 → 일반 JSON 배열이라고 가정
        if first_non_empty and first_non_empty.startswith("["):
            try:
                qa_list = json.loads("".join(lines))
            except json.JSONDecodeError as e:
                print("  ❌ JSON 배열 파싱 실패, 파일 형식을 확인하세요.")
                raise e
        else:
            # 그 외는 JSON Lines 로 가정
            for line_no, line in enumerate(lines, start=1):
                s = line.strip()
                if not s:
                    continue
                try:
                    qa_list.append(json.loads(s))
                except json.JSONDecodeError as e:
                    print(f"  ❌ JSONDecodeError at line {line_no}: {repr(s[:80])}")
                    raise e

    print(f"[LOAD] QA 개수: {len(qa_list)}\n")
    return qa_list



# =========================
# 1.5 QA answer span 자동 생성
# =========================

def attach_answer_spans(text: str, qa_list: List[Dict]) -> List[Dict]:
    """
    qa_list의 각 항목에 대해
    - answer_text를 문서(text) 안에서 찾아서
    - answer_start, answer_end를 추가해준다.
    """
    print("[SPAN] QA answer span 자동 계산 시작...")
    new_list: List[Dict] = []
    not_found = 0

    for i, qa in enumerate(qa_list):
        ans = qa.get("answer_text") or qa.get("answer")
        if not ans:
            print(f"  ⚠ #{i}번 QA: answer_text/answer 필드가 없음. 스킵.")
            not_found += 1
            continue

        start = text.find(ans)
        if start == -1:
            print(f"  ⚠ #{i}번 QA: 문서에서 answer_text를 찾지 못함. 스킵.")
            print(f"     answer_text: {ans[:80]}...")
            not_found += 1
            continue

        end = start + len(ans)
        qa["answer_start"] = start
        qa["answer_end"] = end

        print(f"  - #{i}번 QA span 설정: [{start} ~ {end})")
        new_list.append(qa)

    print(f"[SPAN] span 설정 완료: 사용 가능한 QA 수 = {len(new_list)}, 찾지 못한 QA 수 = {not_found}\n")
    return new_list


# =========================
# 2-1. 토큰 기반 청킹
# =========================

def chunk_by_tokens(text: str, max_tokens: int, overlap_ratio: float = 0.15):
    tokens = tokenizer.encode(text)
    n = len(tokens)

    # 토큰 → 원문 char offset 계산
    offsets = [0]
    cur_text = ""
    for tok in tokens:
        cur_text += tokenizer.decode([tok])
        offsets.append(len(cur_text))

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

        i += step
        chunk_idx += 1

    print(f"[CHUNK] 토큰 청킹 완료: max_tokens={max_tokens}, overlap={overlap}, chunks={len(chunks)}\n")
    return chunks

# =========================
# 3. 정답 span 겹침 확인
# =========================

def spans_overlap(a_start, a_end, b_start, b_end):
    return max(a_start, b_start) < min(a_end, b_end)


# =========================
# 4. Qdrant 관련 함수
# =========================

def create_collection_for_chunks(client: QdrantClient, collection_name: str, vector_size: int):
    print(f"[QDRANT] 컬렉션 재생성: {collection_name}")
    # recreate_collection 자체가 있으면 삭제 후 생성
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )
    print(f"[QDRANT] 컬렉션 준비 완료 (dim={vector_size}, metric=COSINE)\n")


def upload_chunks_to_qdrant(client: QdrantClient, collection_name: str, chunks):
    print(f"[QDRANT] 청크 임베딩 계산 및 업로드 시작 (collection={collection_name})")

    texts = [c["text"] for c in chunks]
    print(f"[QDRANT] 임베딩 계산 대상 chunk 수: {len(texts)}")

    embeddings = get_embeddings(texts)
    dim = embeddings.shape[1]
    print(f"[QDRANT] 임베딩 차원: {dim}")

    UPSERT_BATCH_SIZE = 64

    total_points = 0
    for start in range(0, len(chunks), UPSERT_BATCH_SIZE):
        end = min(start + UPSERT_BATCH_SIZE, len(chunks))
        batch_vecs = embeddings[start:end]
        batch_chunks = chunks[start:end]

        points = []
        for local_idx, (vec, ch) in enumerate(zip(batch_vecs, batch_chunks)):
            points.append(
                PointStruct(
                    id=start + local_idx,
                    vector=vec.tolist(),
                    payload=ch,
                )
            )

        print(f"[QDRANT] upsert 배치: {start} ~ {end-1} (points={len(points)})")
        client.upsert(
            collection_name=collection_name,
            points=points,
        )
        total_points += len(points)

    print(f"[QDRANT] 업로드 완료. 총 points={total_points}\n")


def qdrant_search(client: QdrantClient, collection_name: str, query: str, k: int):
    print(f"    [SEARCH] Qdrant 검색: query='{query}', k={k}")
    q_vec = get_embeddings([query])[0].tolist()

    # qdrant-client 1.16.0에서는 search() 대신 query_points() 사용
    result = client.query_points(
        collection_name=collection_name,
        query=q_vec,
        limit=k,
        with_payload=True,
    )

    # query_points()는 QueryResponse 객체를 반환 → result.points 안에 리스트가 있음
    hits = result.points

    print(f"    [SEARCH] 검색 결과 개수: {len(hits)}")
    retrieved_chunks = [hit.payload for hit in hits]
    return retrieved_chunks



def safe_drop_collection(client: QdrantClient, name: str):
    try:
        client.delete_collection(name)
        print(f"[QDRANT] 기존 컬렉션 삭제: {name}")
    except UnexpectedResponse:
        print(f"[QDRANT] 컬렉션 삭제 스킵 (존재하지 않을 수 있음): {name}")


# =========================
# 5. 평가 함수
# =========================

def evaluate_with_qdrant(
    text,
    qa_list,
    chunks,
    client: QdrantClient,
    collection_name: str,
    k: int = 5,
):
    print(f"\n==============================")
    print(f"=== Evaluating strategy={collection_name} ===")
    print(f"==============================\n")

    vector_size = EMBED_DIM
    create_collection_for_chunks(client, collection_name, vector_size)
    upload_chunks_to_qdrant(client, collection_name, chunks)

    recalls, rrs = [], []
    no_relevant = 0

    for idx, qa in enumerate(qa_list):
        print("\n----------------------------------------")
        print(f"[QA {idx+1}/{len(qa_list)}] 평가 시작")
        query = qa["question"]
        a_start = qa["answer_start"]
        a_end = qa["answer_end"]
        print(f"  - Query        : {query}")
        print(f"  - Answer span  : [{a_start} ~ {a_end})")

        relevant_ids = [
            c["id"] for c in chunks
            if spans_overlap(a_start, a_end, c["start"], c["end"])
        ]

        if not relevant_ids:
            print("  ⚠ 이 청킹 전략에서는 정답이 어떤 chunk에도 포함되지 않음 (손실)")
            no_relevant += 1
            recalls.append(0.0)
            rrs.append(0.0)
            continue

        print(f"  - 정답 chunk id 목록: {relevant_ids}")

        retrieved = qdrant_search(client, collection_name, query, k)
        retrieved_ids = [c["id"] for c in retrieved]

        print(f"  - Top-{k} 검색 결과:")
        for rank, ch in enumerate(retrieved, start=1):
            is_correct = " (★정답)" if ch["id"] in relevant_ids else ""
            preview = ch["text"][:70].replace("\n", " ")
            if len(ch["text"]) > 70:
                preview += "..."
            print(f"    {rank}. {ch['id']} [{ch['start']} ~ {ch['end']}] {is_correct}")
            print(f"       → \"{preview}\"")

        hit = any(rid in relevant_ids for rid in retrieved_ids)
        recall = 1.0 if hit else 0.0
        recalls.append(recall)
        print(f"  - Recall@{k} (이 QA): {recall:.1f}")

        rr = 0.0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in relevant_ids:
                rr = 1.0 / rank
                break
        rrs.append(rr)
        print(f"  - RR (이 QA)      : {rr:.4f}")

    avg_recall = float(np.mean(recalls)) if recalls else 0.0
    avg_mrr = float(np.mean(rrs)) if rrs else 0.0

    print(f"\n>>> strategy={collection_name} 최종 결과")
    print(f"    - 전체 Recall@{k}: {avg_recall:.4f}")
    print(f"    - 전체 MRR@{k}   : {avg_mrr:.4f}")
    print(f"    - 정답 chunk 자체가 없었던 QA 수: {no_relevant}\n")

    return {
        "strategy": collection_name,
        "recall": avg_recall,
        "mrr": avg_mrr,
        "no_relevant": no_relevant,
    }


# =========================
# 6. main
# =========================

def main():
    DOCUMENT_PATH = "/Users/woojin/Desktop/무역_AI_Copliot/woojin/incoterms/Incoterms_preprocessed2.md"
    QA_PATH = "/Users/woojin/Desktop/무역_AI_Copliot/woojin/incoterms/incoterms_test.json"
    K = 5

    # 진짜 qdrant_client 패키지가 import 되었는지 경로 확인 (디버깅용)
    print("qdrant_client 모듈 경로:", getattr(qdrant_client, "__file__", "알 수 없음"))

    text = load_document(DOCUMENT_PATH)
    qa_list_raw = load_qa(QA_PATH)
    qa_list = attach_answer_spans(text, qa_list_raw)

    if not qa_list:
        print("[ERROR] span이 설정된 QA가 하나도 없습니다. answer_text가 문서와 일치하는지 확인하세요.")
        return

    print("[INIT] Qdrant 연결 시도 (localhost:6333)...")
    client = QdrantClient(
        host="127.0.0.1",   # 또는 url="http://127.0.0.1:6333"
        port=6333,
        timeout=300.0,
    )
    print("[INIT] Qdrant 연결 완료.\n")

    print("QdrantClient has search? :", hasattr(client, "search"))

    results = []

    token_sizes = [128, 256, 512, 1024, 2048]
    for tok_size in token_sizes:
        chunks_tok = chunk_by_tokens(text, tok_size, overlap_ratio=0.15)
        res = evaluate_with_qdrant(
            text=text,
            qa_list=qa_list,
            chunks=chunks_tok,
            client=client,
            collection_name=f"tok_{tok_size}",
            k=K,
        )
        results.append(res)

    print("\n============== FINAL SUMMARY ==============")
    print(f"{'strategy':20s} | {'Recall@'+str(K):>10} | {'MRR@'+str(K):>10} | {'no_rel':>7}")
    print("-------------------------------------------")
    for r in results:
        print(
            f"{r['strategy']:20s} | "
            f"{r['recall']:10.4f} | "
            f"{r['mrr']:10.4f} | "
            f"{r['no_relevant']:7d}"
        )
    print("===========================================\n")
    # info = client.get_collection("tok_256")
    # print(info.points_count)


if __name__ == "__main__":
    main()
    
