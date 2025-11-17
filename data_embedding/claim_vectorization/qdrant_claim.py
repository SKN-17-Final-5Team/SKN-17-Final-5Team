from pathlib import Path
from types import SimpleNamespace
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter
import os
from dotenv import load_dotenv
import json
import time
import uuid

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    check_compatibility=False,
    timeout=300  # 5분 타임아웃 (대용량 업로드 대비)
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.getenv("OPENAI_API_KEY")
)

def docs_to_lists(docs):
    texts = [d.page_content for d in docs]
    metadatas = [getattr(d, "metadata", {}) for d in docs]
    ids = [str(i) for i in range(len(docs))]
    return texts, metadatas, ids

def upsert_collection(collection_name, docs, batch_size=20):
    texts, metadatas, ids = docs_to_lists(docs)

    # OpenAI 임베딩 생성
    print(f"  임베딩 생성 중... ({len(texts)}개)")
    vectors = embeddings.embed_documents(texts)
    vector_size = len(vectors[0])

    # 컬렉션이 존재하지 않을 경우에만 생성 (기존 데이터 보존)
    try:
        qdrant_client.get_collection(collection_name)
        print(f"✓ 컬렉션 '{collection_name}' 이미 존재함 (기존 데이터 유지)")
    except:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"✓ 새 컬렉션 '{collection_name}' 생성 완료")

    # UUID를 사용하여 다른 데이터 소스와 ID 충돌 방지
    points = []
    for idx, (text, metadata, vector) in enumerate(zip(texts, metadatas, vectors)):
        payload = {
            **(metadata or {}),
            "text": text,
            "data_source": "claim"  # 데이터 출처 식별용
        }
        points.append(PointStruct(
            id=str(uuid.uuid4()),  # 정수 ID 대신 UUID 사용
            vector=vector,
            payload=payload
        ))

    # 페이로드 크기 제한을 피하기 위한 배치 업로드
    total_points = len(points)
    total_batches = (total_points + batch_size - 1) // batch_size
    print(f"  배치 업로드 시작 (총 {total_points}개, 배치 크기: {batch_size})...")

    for i in range(0, total_points, batch_size):
        batch = points[i:i + batch_size]
        batch_num = i // batch_size + 1
        qdrant_client.upsert(collection_name=collection_name, points=batch, wait=True)
        print(f"    - 배치 {batch_num}/{total_batches} 업로드 완료 ({len(batch)}개)")

    print(f"✓ [{collection_name}] {len(docs)}개 문서 업로드 완료")
    return collection_name


# JSON 텍스트 데이터 로드
json_path = Path('./used_data/사례_응답_근거조항.json')

with json_path.open('r', encoding='utf-8') as f:
    json_records = json.load(f)

text_docs = []
for idx, record in enumerate(json_records):
    text = str(record.get('text', '')).strip()
    if not text:
        continue
    metadata = record.get('metadata', {}).copy()
    metadata.update({
        'row_index': record.get('metadata', {}).get('row_index', idx),
        'document_name': metadata.get('document_name', '무역클레임중재QA'),
        'source': 'json_text'
    })
    text_docs.append(SimpleNamespace(
        page_content=text,
        metadata=metadata
    ))

print(f"JSON 텍스트 문서: {len(text_docs)}개")


# JSON 텍스트 청크 생성 및 업로드
chunk_configs = [
    #{"size": 128, "overlap": 20, "collection": "qna_chunk_128"},
    #{"size": 256, "overlap": 39, "collection": "qna_chunk_256"},    
    #{"size": 512, "overlap": 77, "collection": "qna_chunk_512"},
    #{"size": 1024, "overlap": 154, "collection": "qna_chunk_1024"},
    # 제일 검색 성능이 좋았던 조합 (채워넣기)
    {"size": 512, "overlap": 77, "collection": "trade_collection"}
]

chunk_collections = {}
for cfg in chunk_configs:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg['size'],
        chunk_overlap=cfg['overlap'],
    )
    docs = []
    for doc in text_docs:
        chunks = splitter.split_text(doc.page_content)
        for cid, chunk in enumerate(chunks):
            docs.append(SimpleNamespace(
                page_content=chunk,
                metadata={
                    **doc.metadata,
                    'chunk_size': cfg['size'],
                    'chunk_id': f"{doc.metadata.get('row_index')}_{cid}",
                    'source': f"json_chunk_{cfg['size']}"
                }
            ))
    chunk_collections[cfg['collection']] = docs
    print(f"청크 {cfg['size']}자: {len(docs)}개 생성")

# Qdrant 업로드
for cfg in chunk_configs:
    docs = chunk_collections.get(cfg['collection'], [])
    if not docs:
        print(f"  ⚠ {cfg['collection']} 업로드할 문서가 없습니다.")
        continue
    start_time = time.time()
    upsert_collection(cfg['collection'], docs)
    elapsed = time.time() - start_time
    print(f"  ✓ {cfg['collection']} 업로드 완료 ({elapsed:.2f}s)")