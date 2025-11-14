# 인증 데이터 RAG 검색 시스템 🔍

**Qdrant 벡터 데이터베이스를 활용한 한국어 인증 정보 의미 기반 검색 시스템**

---

## 🎯 빠른 시작 (3단계)

### 1. 환경 변수 설정

`.env` 파일에 다음 정보를 추가하세요:

```bash
# Qdrant Cloud (필수)
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key

# OpenAI (필수 - 최고 성능)
OPENAI_API_KEY=sk-...

# HuggingFace (선택 - 무료이지만 성능 낮음)
HF_TOKEN=hf_...
```

### 2. 의존성 설치

```bash
cd certif_retrieval_test
pip install -r requirements.txt
```

### 3. 실행

```bash
# 기본 RAG 시스템 실행 (대화형 검색)
python qdrant_rag.py

# 성능 평가 실행 (20개 질문으로 테스트)
python evaluate_retrieval.py

# 간단한 검색 테스트
python test_retrieval.py
```

---

## 📊 검증된 최고 성능 설정

평가 결과 **Recall@3 85%, Recall@5 90%** 달성한 최적 설정:

```python
{
    "embedding_provider": "openai",
    "embedding_model": "text-embedding-3-large",
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "text_field": "full"  # 전체 cert_subject 사용
}
```

---

## 📁 주요 파일

| 파일 | 설명 |
|------|------|
| `qdrant_rag.py` | **핵심 RAG 시스템** - 임베딩, 검색, 하이브리드 검색 |
| `evaluate_retrieval.py` | **성능 평가** - Recall@K, MRR 측정 |
| `certif_doc_convert.py` | CSV → JSONL 변환 |
| `qa_dataset.json` | 평가용 20개 Q&A 세트 |
| `프로젝트_진행_요약.md` | **전체 프로젝트 요약** (단계별 진행, 결과 분석) |

---

## 🔧 설정 변경 방법

`qdrant_rag.py`의 `CONFIG` 딕셔너리를 수정하세요:

```python
CONFIG = {
    # 컬렉션 설정
    "collection_name": "certifications",
    "use_cloud": True,  # False = 로컬 저장소

    # 임베딩 설정 (권장: OpenAI)
    "embedding_provider": "openai",  # "huggingface" 또는 "openai"
    "embedding_model": None,  # None = 기본값 사용

    # 청킹 설정 (권장: 1000자)
    "chunk_size": 1000,  # None, 500, 1000, 2000
    "chunk_overlap": 100,

    # 텍스트 필드 (권장: "full")
    "text_field": "full",  # "auto", "summary", "full", "combined"

    # 검색 설정
    "top_k": 5,
    "score_threshold": None  # 예: 0.7
}
```

---

## 📈 성능 비교 결과

| 설정 | Recall@1 | Recall@3 | Recall@5 |
|------|----------|----------|----------|
| **OpenAI + 전체 텍스트 + 청크 1000 (권장)** | **50%** | **85%** | **90%** |
| OpenAI + 전체 텍스트 + 청킹 없음 | 55% | 80% | 90% |
| OpenAI + 요약 텍스트 + 청크 1000 | 30% | 65% | 75% |
| HuggingFace (모든 설정) | 15% | 20% | 30% |

### 핵심 발견
- ✅ **전체 텍스트 >> 요약**: +31% Recall 향상
- ✅ **OpenAI >> HuggingFace**: 4배 성능 차이
- ✅ **하이브리드 검색**: 예상과 달리 성능 하락 (-10%p)

---

## 💡 주요 기능

### 1. 의미 기반 검색 (Semantic Search)
```python
from qdrant_rag import QdrantCertificationRAG

rag = QdrantCertificationRAG(
    collection_name="certifications",
    embedding_provider="openai",
    chunk_size=1000,
    use_cloud=True
)

# 검색
results = rag.search("미국 의료기기 인증", top_k=5)
rag.print_results(results)
```

### 2. 하이브리드 검색 (의미 + 키워드)
```python
# BM25 인덱스 구축
documents = [...]  # JSONL에서 로드
rag.build_bm25_index(documents, text_field="full")

# 하이브리드 검색
results = rag.search_hybrid(
    "510(k) 승인",
    top_k=5,
    semantic_weight=0.7,
    bm25_weight=0.3
)
```

### 3. 성능 평가
```python
from evaluate_retrieval import RetrievalEvaluator

evaluator = RetrievalEvaluator("qa_dataset.json")
results = evaluator.compare_configurations(configs, top_k=10)
```

---

## 🗂️ 데이터 형식

### JSONL 입력 (`output/certifications.jsonl`)
```json
{
  "id": 1,
  "country": "미국",
  "category": "의료기기",
  "cert_type": "제품인증",
  "main_cert": "FDA",
  "cert_name": "FDA(의료기기)",
  "cert_subject": "...(전체 설명)...",
  "auto_summary": "...(150자 요약)...",
  "url": "https://..."
}
```

### QA 데이터셋 (`qa_dataset.json`)
```json
[
  {
    "id": 1,
    "question": "미국에서 의료기기를 판매하려면 어떤 인증이 필요한가요?",
    "expected_certs": ["FDA(의료기기)"],
    "category": "의료기기",
    "difficulty": "easy"
  }
]
```

---

## 📊 평가 지표 설명

- **Recall@K**: 상위 K개 결과에 정답이 포함되는 비율
  - Recall@1 = 50%: 20개 중 10개가 1등으로 검색됨
  - Recall@3 = 85%: 20개 중 17개가 상위 3개 안에 포함
  - Recall@5 = 90%: 20개 중 18개가 상위 5개 안에 포함

- **MRR (Mean Reciprocal Rank)**: 정답 순위의 역수 평균
  - MRR = 0.668: 평균적으로 1.5등에 정답 등장

---

## 💰 비용 안내

### OpenAI 임베딩 (`text-embedding-3-large`)
- **가격**: $0.13 per 1M tokens
- **92개 문서 인덱싱**: ~$0.007 (약 10원)
- **쿼리당**: ~$0.000001 (무시 가능)

### Qdrant Cloud
- **Free tier**: 1GB 저장소 (충분)
- **사용량**: ~2.4MB (200 벡터 × 3072차원)

**→ 매우 저렴한 비용으로 고품질 검색 시스템 구축 가능!**

---

## 📚 상세 문서

- **[프로젝트_진행_요약.md](프로젝트_진행_요약.md)**: 전체 프로젝트 진행 과정, 실험 결과, 학습 내용
- **[qa_dataset.json](qa_dataset.json)**: 20개 평가 질문 (난이도별)
- **[evaluation_results_*.json](.)**: 성능 평가 결과

---

## 🔐 보안 주의사항

- `.env` 파일은 절대 Git에 커밋하지 마세요
- API 키는 환경 변수로만 관리하세요
- Qdrant Cloud API 키는 읽기 전용으로 설정 권장

---

## 🐛 문제 해결

### 타임아웃 오류
```
httpx.WriteTimeout: The write operation timed out
```
→ `qdrant_rag.py`의 `timeout=300` 설정 확인 (5분)

### BM25 인덱스 미구축 경고
```
Warning: BM25 index not built
```
→ `rag.build_bm25_index(documents)` 먼저 호출 필요

### 임베딩 모델 다운로드 느림
→ HuggingFace 모델 첫 실행 시 ~500MB 다운로드 (한 번만)

---

**작성일**: 2025년 11월 14일
**최종 성능**: Recall@3 85%, Recall@5 90%
**권장**: OpenAI + 전체 텍스트 + 청크 1000자
