import json
import re
import os
import hashlib
from pathlib import Path
from itertools import groupby
from dotenv import load_dotenv

# RAG 관련
import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Deprecation 경고 무시
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# ----------------------------------------------------
# 업로드 설정: 여기만 수정하세요
# ----------------------------------------------------
class CONFIG_UPLOAD:
    # --- 1. 파일 경로 설정 ---
    BASE_PATH = Path(__file__).parent.parent.parent  # ex) /Users/hoon/Desktop/final_project
    DOCUMENT_PATH = BASE_PATH / "data/extracted_data/다자조약상세(CISG).txt"
    BASE_CHUNKS_PATH = BASE_PATH / "data_embedding/cisg_vectorization/used_data/cisg_chunks.json"

    # --- 2. Qdrant 컬렉션 이름 설정 ---
    # 데이터를 저장할 Qdrant Cloud의 컬렉션(테이블) 이름입니다.
    # 예: "cisg_production_v1"
    COLLECTION_NAME = "trade_collection"

    # --- 3. 사용할 임베딩 모델 선택 ---
    # OpenAI 임베딩 모델 (하나만 선택)
    MODEL_NAME = "openai_text-embedding-3-large"
    # MODEL_NAME = "openai_text-embedding-ada-002"

    # --- 4. 사용할 청킹 전략 선택 (하나만 선택) ---
    # "Ho_Segmented": 가장 세분화된 단위 (기본)
    # "Paragraph": '항' 단위로 병합
    # "Article": '조' 단위로 병합
    CHUNK_STRATEGY = "Article"

    # --- 5. 환경 변수 키 이름 (.env 파일에서 로드) ---
    QDRANT_URL_KEY = "QDRANT_URL"
    QDRANT_API_KEY = "QDRANT_API_KEY"
    OPENAI_API_KEY = "OPENAI_API_KEY"



# ----------------------------------------------------
# 모델 핸들러 정의 (OpenAI)
# ----------------------------------------------------
def get_model_handler(model_name: str, keys: dict) -> dict:
    """OpenAI 임베딩 모델에 따라 임베딩 함수와 차원을 준비합니다."""

    print(f"  [모델 로더] '{model_name}' 로드 중...")

    if not model_name.startswith("openai_"):
        raise ValueError(f"OpenAI 모델만 지원합니다. 'openai_'로 시작하는 모델명을 사용하세요: {model_name}")

    model_id = model_name.split('_', 1)[1]
    client = openai.OpenAI(api_key=keys['openai'])

    # 모델별 차원 설정
    if model_id == "text-embedding-3-large":
        dim = 3072
    elif model_id == "text-embedding-ada-002":
        dim = 1536
    else:
        raise ValueError(f"지원되지 않는 OpenAI 모델입니다: {model_id}")

    # 임베딩 함수 정의
    def embed_texts(texts: list) -> list:
        response = client.embeddings.create(input=texts, model=model_id)
        return [item.embedding for item in response.data]

    print(f"  [모델 로더] OpenAI '{model_id}' 핸들러 생성 완료 (차원: {dim})")
    return {
        "name": model_name,
        "embed_texts": embed_texts,
        "dim": dim
    }



# ----------------------------------------------------
# 데이터 로드 및 정제 함수
# ----------------------------------------------------
def load_document(path) -> str:
    """원본 문서를 로드합니다. Path 객체 또는 문자열을 받습니다."""
    path = Path(path)  # Path 객체로 변환
    print(f"[LOAD] 원본 문서 로드: {path}")
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return text

def load_base_chunks(path) -> list:
    """기반 청크 JSON을 로드합니다. Path 객체 또는 문자열을 받습니다."""
    path = Path(path)  # Path 객체로 변환
    print(f"[LOAD] 기반(Base) 청크 JSON 로드: {path}")
    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)
    return chunks

def attach_chunk_spans(text: str, chunks: list) -> list:
    """청크의 content를 원본 text와 비교하여 'start', 'end' 위치를 찾습니다."""
    print(f"  [SPAN] 청크 위치(span) 계산 중...")
    current_search_idx, not_found = 0, 0
    new_chunks = []
    skipped_chunks = []  # 누락된 청크 추적

    for i, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            not_found += 1
            skipped_chunks.append({"index": i, "reason": "chunk_id 없음"})
            continue
        chunk["id"] = chunk_id

        json_content = chunk.get("content")
        if not json_content:
            not_found += 1
            skipped_chunks.append({"index": i, "chunk_id": chunk_id, "reason": "content 없음"})
            continue

        start = text.find(json_content, current_search_idx)
        if start == -1:
            temp_content = re.sub(r'\s+', ' ', json_content).strip()
            start = text.find(temp_content, current_search_idx)
        if start == -1:
            print(f"    ⚠ 청크 #{i} ('{chunk_id}') span 찾기 실패. 스킵.")
            not_found += 1
            skipped_chunks.append({
                "index": i,
                "chunk_id": chunk_id,
                "reason": "원본에서 찾을 수 없음",
                "content_preview": json_content[:100] + "..." if len(json_content) > 100 else json_content
            })
            continue

        end = start + len(json_content)
        chunk["text"] = text[start:end]  # 'text' 필드 표준화
        chunk["start"] = start
        chunk["end"] = end
        new_chunks.append(chunk)
        current_search_idx = end

    # 누락된 청크가 있으면 경고 출력
    if skipped_chunks:
        print(f"    ⚠️  누락된 청크 상세:")
        for skip in skipped_chunks[:5]:  # 처음 5개만 출력
            print(f"       - 인덱스 {skip['index']}: {skip['reason']}")
        if len(skipped_chunks) > 5:
            print(f"       ... 외 {len(skipped_chunks) - 5}개 더")

    print(f"    [SPAN] 사용 가능한 기반 청크: {len(new_chunks)}개 (누락: {not_found}개)")
    return new_chunks



# ----------------------------------------------------
# 청크 병합 함수
# ----------------------------------------------------
def merge_chunks(base_chunks: list, strategy_name: str, raw_text: str) -> list:
    """'Ho_Segmented' 단위의 청크를 더 큰 단위('Paragraph', 'Article')로 병합합니다."""
    print(f"  [Chunking] 전략 '{strategy_name}' 실행 중...")
    
    if strategy_name == "Ho_Segmented":
        for chunk in base_chunks:
            chunk['strategy_name'] = strategy_name
            # 데이터소스  구분용 메타데이터
            chunk['data_source'] = 'cisg'
            # 동일하게 생성되는 ID 생성
            if 'id' not in chunk or len(chunk.get('id', '')) > 20:
                id_string = f"CISG_{chunk.get('article', 'unknown')}_{chunk.get('paragraph_no', 'unknown')}_{chunk.get('ho_no', 'unknown')}"
                chunk['id'] = hashlib.md5(id_string.encode()).hexdigest()
        print(f"    [Chunking] '{strategy_name}' 완료. {len(base_chunks)}개 청크 사용.")
        return base_chunks

    if strategy_name == "Paragraph":
        get_key = lambda x: (x.get('article'), x.get('paragraph_no'))
    elif strategy_name == "Article":
        get_key = lambda x: x.get('article')
    else:
        raise ValueError(f"알 수 없는 병합 전략: {strategy_name}")

    merged_chunks = []
    base_chunks.sort(key=lambda x: x['start'])
    
    for key, group in groupby(base_chunks, key=get_key):
        if key is None or (isinstance(key, tuple) and None in key):
            continue
            
        group_list = list(group)
        first_chunk = group_list[0]
        last_chunk = group_list[-1]
        min_start, max_end = first_chunk['start'], last_chunk['end']
        merged_text = raw_text[min_start:max_end]

        if strategy_name == "Paragraph":
            # Paragraph: CISG_Article_{article}_Paragraph_{paragraph_no}
            id_string = f"CISG_Article_{first_chunk.get('article')}_Paragraph_{first_chunk.get('paragraph_no')}"
        else:  # Article
            # Article: CISG_Article_{article}
            id_string = f"CISG_Article_{first_chunk.get('article')}"

        deterministic_id = hashlib.md5(id_string.encode()).hexdigest()

        new_chunk = {
            "part": first_chunk.get("part"),
            "chapter": first_chunk.get("chapter"),
            "article": first_chunk.get("article"),
            "paragraph_no": first_chunk.get("paragraph_no") if strategy_name == "Paragraph" else None,
            "text": merged_text,
            "start": min_start,
            "end": max_end,
            "id": deterministic_id,  # Deterministic ID based on content
            "strategy_name": strategy_name,
            "data_source": "cisg",  # Add source metadata
        }
        merged_chunks.append(new_chunk)
        
    print(f"    [Chunking] '{strategy_name}' 완료. {len(merged_chunks)}개 청크 생성.")
    return merged_chunks



# ----------------------------------------------------
# Qdrant 업로드 함수 (핵심 로직)
# ----------------------------------------------------
def create_collection_if_not_exists(client: QdrantClient, collection_name: str, vector_size: int):
    """
    [중요] 'recreate_collection'(삭제 후 생성) 대신, 
    컬렉션이 없을 때만 새로 생성합니다. (데이터 추가/append 보장)
    """
    if client.collection_exists(collection_name):
        print(f"  [QDRANT] 컬렉션 '{collection_name}'이(가) 이미 존재합니다. 데이터를 추가(upsert)합니다.")
    else:
        print(f"  [QDRANT] 새 컬렉션 '{collection_name}'을(를) 생성합니다. (dim={vector_size})")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

def upload_to_qdrant(client: QdrantClient, collection_name: str, model_handler: dict, chunks: list, batch_size: int = 20):
    """
    청크 리스트를 임베딩하여 Qdrant에 'upsert' (추가 또는 덮어쓰기)합니다.
    대용량 데이터를 처리하기 위해 배치 단위로 업로드합니다.
    """
    texts = [c["text"] for c in chunks]
    print(f"    [QDRANT] 임베딩 계산 중 ({model_handler['name']}, {len(texts)}개)...")

    # 모델 핸들러를 사용해 텍스트를 벡터로 변환
    embeddings = model_handler['embed_texts'](texts)

    # Qdrant에 저장할 'Point' 객체 생성
    points = [
        PointStruct(id=ch["id"], vector=vec, payload=ch)
        for vec, ch in zip(embeddings, chunks)
    ]

    # 배치 단위로 업로드
    total_points = len(points)
    print(f"    [QDRANT] 배치 업로드 시작 (총 {total_points}개, 배치 크기: {batch_size})...")

    for i in range(0, total_points, batch_size):
        batch = points[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_points + batch_size - 1) // batch_size

        print(f"      - 배치 {batch_num}/{total_batches} 업로드 중 ({len(batch)}개)...")
        # 'upsert'는 ID가 없으면 새로 추가하고, ID가 이미 있으면 덮어쓰는 '안전한' 명령어입니다.
        client.upsert(collection_name=collection_name, points=batch, wait=True)

    print(f"    [QDRANT] {total_points}개 벡터 업로드/업데이트 완료.")



# ----------------------------------------------------
# 메인 실행기
# ----------------------------------------------------
def main_upload():
    print("--- (1/3) 업로드 시작: 설정 및 키 로드 ---")

    try:
        # .env 파일에서 필요한 키를 모두 로드합니다.
        load_dotenv()  # .env 파일 로드
        keys = {
            'qdrant_url': os.getenv(CONFIG_UPLOAD.QDRANT_URL_KEY),
            'qdrant_api': os.getenv(CONFIG_UPLOAD.QDRANT_API_KEY),
            'openai': os.getenv(CONFIG_UPLOAD.OPENAI_API_KEY)
        }

        # 필수 키 확인
        missing_keys = [k for k, v in keys.items() if not v]
        if missing_keys:
            raise ValueError(f"누락된 환경 변수: {missing_keys}")

        # Qdrant DB에 연결합니다. (timeout 증가: 대용량 업로드 대비)
        qdrant_client = QdrantClient(
            url=keys['qdrant_url'],
            api_key=keys['qdrant_api'],
            timeout=300  # 5분 타임아웃 (기본값: 60초)
        )
        print("  [메인] Qdrant 및 API 키 로드 완료.")
    except Exception as e:
        print(f"🚨 [오류] 환경 변수 로드 실패. .env 파일에 필요한 키 3개를 모두 설정했는지 확인하세요.")
        print(f"    필요한 키: {CONFIG_UPLOAD.QDRANT_URL_KEY}, {CONFIG_UPLOAD.QDRANT_API_KEY}, {CONFIG_UPLOAD.OPENAI_API_KEY}")
        print(f"    오류 상세: {e}")
        return

    print("\n--- (2/3) 데이터 준비: 로드, 정제, 청킹 ---")

    # 1. 원본 텍스트와 기반 청크(cisg_chunks.json)를 로드합니다.
    raw_text = load_document(CONFIG_UPLOAD.DOCUMENT_PATH)
    base_chunks_raw = load_base_chunks(CONFIG_UPLOAD.BASE_CHUNKS_PATH)
    
    # 2. 청크의 'start'/'end' 위치를 계산합니다. (매우 중요)
    base_chunks_ready = attach_chunk_spans(raw_text, base_chunks_raw)
    
    if not base_chunks_ready:
        print("🚨 [오류] 유효한 기반 청크가 없습니다. BASE_CHUNKS_PATH 파일과 DOCUMENT_PATH 파일의 내용이 일치하는지 확인하세요.")
        return
        
    # 3. (2)번 설정에서 선택한 '청킹 전략'을 실행합니다.
    # 예: 'Paragraph'를 선택했다면, 'Ho_Segmented' 청크들을 'Paragraph' 단위로 병합합니다.
    chunks_to_upload = merge_chunks(base_chunks_ready, CONFIG_UPLOAD.CHUNK_STRATEGY, raw_text)

    if not chunks_to_upload:
        print(f"🚨 [오류] '{CONFIG_UPLOAD.CHUNK_STRATEGY}' 전략으로 청크를 생성하지 못했습니다.")
        return

    print("\n--- (3/3) 모델 로드 및 업로드 실행 ---")

    try:
        # 4. (2)번 설정에서 선택한 '임베딩 모델'을 로드합니다.
        model_handler = get_model_handler(CONFIG_UPLOAD.MODEL_NAME, keys)
        
        # 5. Qdrant 컬렉션이 존재하는지 확인하고, 없으면 새로 생성합니다.
        # (주의: 기존 평가 스크립트와 달리 'recreate'(삭제)를 하지 않습니다.)
        create_collection_if_not_exists(
            qdrant_client, 
            CONFIG_UPLOAD.COLLECTION_NAME, 
            model_handler['dim']
        )
        
        # 6. 최종 청크를 임베딩하여 Qdrant에 업로드(Upsert)합니다.
        upload_to_qdrant(
            qdrant_client,
            CONFIG_UPLOAD.COLLECTION_NAME,
            model_handler,
            chunks_to_upload
        )
        
        print(f"\n🎉 === 업로드 성공! ===")
        print(f"  - 컬렉션: {CONFIG_UPLOAD.COLLECTION_NAME}")
        print(f"  - 청크 수: {len(chunks_to_upload)}개")
        print(f"  - 모델: {CONFIG_UPLOAD.MODEL_NAME}")

    except Exception as e:
        print(f"🚨 [오류] 업로드 작업 중 심각한 오류 발생: {e}")
        import traceback
        traceback.print_exc() # 상세 오류 내역 출력




# --- 스크립트 실행 ---
if __name__ == "__main__":
    print("="*60)
    print("🚀 CISG 문서 Qdrant 업로드 스크립트 시작")
    print("="*60)
    print(f"\n📋 설정 정보:")
    print(f"  - 컬렉션: {CONFIG_UPLOAD.COLLECTION_NAME}")
    print(f"  - 임베딩 모델: {CONFIG_UPLOAD.MODEL_NAME}")
    print(f"  - 청킹 전략: {CONFIG_UPLOAD.CHUNK_STRATEGY}")
    print(f"  - 문서 경로: {CONFIG_UPLOAD.DOCUMENT_PATH}")
    print(f"  - 청크 경로: {CONFIG_UPLOAD.BASE_CHUNKS_PATH}")
    print()

    main_upload()