# 데이터 임베딩 및 벡터 검색 시스템

## 개요
이 디렉토리는 다양한 데이터 소스를 벡터화하고 Qdrant 벡터 데이터베이스에 저장하여 의미 기반 검색(Semantic Search)을 수행하는 시스템입니다.

## 디렉토리 구조

```
data_embedding/
├── certifcation_vectorization/    # 인증 데이터 벡터화
│   ├── certif_doc_convert.py      # CSV → JSONL 변환
│   ├── qdrant_certification.py    # Qdrant 업로드 및 검색
│   └── evaluate_retrieval.py      # 검색 성능 평가
├── METADATA_STRUCTURE.md          # 메타데이터 구조 가이드
└── README.md                       # 이 파일
```

## 주요 기능

### 1. 데이터 변환 (certif_doc_convert.py)
CSV 데이터를 RAG에 최적화된 JSONL 형식으로 변환합니다.

```bash
python certif_doc_convert.py
```

**지원 형식:**
- JSONL (벡터 DB에 권장)
- JSON (구조화된 저장소용)
- TXT (단일 텍스트 파일)
- 개별 파일 (문서별 파일)

### 2. 벡터 검색 (qdrant_certification.py)
Qdrant에 데이터를 업로드하고 검색을 수행합니다.

```bash
python qdrant_certification.py
```

**검색 모드:**
- **Semantic Search**: 의미 기반 검색 (임베딩 활용)
- **Hybrid Search**: 의미 기반 + 키워드 검색 (BM25 + Vector)

**지원 임베딩 모델:**
- HuggingFace: `jhgan/ko-sroberta-multitask` (한국어 최적)
- OpenAI: `text-embedding-3-large` (고성능)

### 3. 성능 평가 (evaluate_retrieval.py)
다양한 청킹 설정과 임베딩 모델의 검색 성능을 평가합니다.

```bash
python evaluate_retrieval.py
```

**평가 메트릭:**
- Recall@K (재현율)
- MRR (Mean Reciprocal Rank)

## 사용 방법

### 1단계: 환경 설정

`.env` 파일 생성:
```env
# Qdrant Cloud
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key

# OpenAI (선택사항)
OPENAI_API_KEY=your-openai-key

# HuggingFace (선택사항, gated 모델 사용 시)
HF_TOKEN=your-hf-token
```

### 2단계: 데이터 변환

```python
from certif_doc_convert import CertificationRAGConverter

converter = CertificationRAGConverter("your_data.csv")
converter.load_data()
converter.convert_to_jsonl("output/certifications.jsonl")
```

### 3단계: 벡터화 및 업로드

```python
from qdrant_certification import QdrantCertificationRAG

rag = QdrantCertificationRAG(
    collection_name="trade_collection",
    embedding_provider="huggingface",  # 또는 "openai"
    use_cloud=True
)

rag.create_collection()
rag.load_and_index_documents("output/certifications.jsonl")
```

### 4단계: 검색

```python
# 의미 기반 검색
results = rag.search("전자제품 안전 인증", top_k=5)

# 하이브리드 검색
results = rag.search_hybrid(
    "전자제품 안전 인증",
    top_k=5,
    semantic_weight=0.7,
    bm25_weight=0.3
)

# 필터링 검색
results = rag.search(
    "안전 인증",
    filters={"country": "미국", "data_source": "certification"}
)
```

## 메타데이터 구조

모든 문서는 다음과 같은 계층적 메타데이터 구조를 사용합니다:

```python
{
    "data_source": "certification",     # 데이터 소스
    "data_type": "certification_info",  # 데이터 타입
    "doc_id": 123,                      # 문서 ID
    "title": "제품 안전 인증",           # 제목
    "country": "미국",                   # 국가
    "category": "전자전기",              # 카테고리

    "certification_data": {             # 인증 전용 필드
        "cert_type": "강제",
        "main_cert": "UL"
    },

    "chunk_info": {                     # 청크 정보
        "chunk_idx": 0,
        "total_chunks": 3
    },

    "embedding_info": {                 # 임베딩 정보
        "model": "jhgan/ko-sroberta-multitask",
        "provider": "huggingface"
    }
}
```

자세한 내용은 [METADATA_STRUCTURE.md](METADATA_STRUCTURE.md)를 참조하세요.

## 새로운 데이터 소스 추가

### 1. 새 디렉토리 생성
```bash
mkdir data_embedding/trade_vectorization
```

### 2. 변환 스크립트 작성
`trade_doc_convert.py` 생성 (certif_doc_convert.py 참고)

### 3. 벡터화 스크립트 작성
`qdrant_trade.py` 생성 (qdrant_certification.py 참고)

**중요:** 메타데이터 구조를 일관되게 유지하세요:
```python
payload = {
    "data_source": "trade",              # 새로운 소스 이름
    "data_type": "trade_agreement",      # 데이터 타입
    # ... 공통 필드들 ...
    "trade_data": {                      # 소스별 전용 필드
        "agreement_type": "FTA",
        "effective_date": "2024-01-01"
    }
}
```

### 4. 동일한 컬렉션에 업로드
```python
# trade_collection에 무역 데이터 추가
trade_rag = QdrantTradeRAG(collection_name="trade_collection")
trade_rag.load_and_index_documents("trade_data.jsonl")
```

## 청킹 전략

텍스트 청킹은 긴 문서를 검색에 최적화된 크기로 분할합니다:

- **청킹 없음**: 짧은 문서에 적합
- **500자**: 매우 정밀한 검색
- **1000자**: 균형잡힌 선택 (권장)
- **2000자**: 긴 문맥 유지

```python
rag = QdrantCertificationRAG(
    chunk_size=1000,      # 청크 크기
    chunk_overlap=100     # 청크 간 겹침
)
```

## 성능 최적화

### 임베딩 선택
- **한국어 우선**: HuggingFace `jhgan/ko-sroberta-multitask`
- **다국어/고성능**: OpenAI `text-embedding-3-large`

### 검색 방법
- **의미 중심**: Semantic Search
- **키워드 중요**: Hybrid Search (0.7 semantic + 0.3 BM25)

### 배치 크기
- HuggingFace: 32-64 (로컬 GPU 따라)
- OpenAI: 최대 2048 (API 제한)

## 문제 해결

### Qdrant 연결 오류
```bash
# .env 파일 확인
cat .env

# Qdrant 클러스터 상태 확인
curl https://your-cluster.qdrant.io:6333/collections
```

### 메모리 부족
```python
# 배치 크기 감소
rag.load_and_index_documents(
    "data.jsonl",
    batch_size=16  # 기본값 32에서 감소
)
```

### 검색 결과 품질 저하
```python
# 1. 임베딩 모델 변경
# 2. 청킹 크기 조정
# 3. 하이브리드 검색 시도
# 4. evaluate_retrieval.py로 성능 측정
```

## 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다.

## 기여
새로운 데이터 소스나 기능 추가는 Pull Request를 통해 제출해주세요.
