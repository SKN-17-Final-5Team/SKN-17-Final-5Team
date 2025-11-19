"""
심플 RAG 테스트 (OpenAI Agents SDK)

Reranker API를 활용한 고도화된 RAG 시스템
- 초기 검색: Qdrant Vector DB에서 limit개 문서 검색
- Reranking: RunPod 서버의 Reranker 모델로 재정렬
- 최종 전달: 상위 top_k개 문서만 Agent에게 전달
"""

import asyncio
import os
import httpx
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool
from qdrant_client import QdrantClient
from openai import OpenAI

load_dotenv()

# =====================================================================
# 클라이언트 초기화
# =====================================================================

# Qdrant Vector DB 클라이언트
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=60
)

# OpenAI 클라이언트 (Embedding 및 Agent용)
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =====================================================================
# 설정 상수
# =====================================================================

COLLECTION_NAME = "trade_collection"  # Qdrant 컬렉션 이름
EMBEDDING_MODEL = "text-embedding-3-large"  # OpenAI Embedding 모델
RERANKER_API_URL = os.getenv("RERANKER_API_URL", "http://your-runpod-server/rerank")  # Reranker API 엔드포인트

# Reranker 사용 여부 (실행 시 설정됨)
USE_RERANKER = True  # 기본값


# =====================================================================
# Reranker API 모델 정의 (Pydantic)
# =====================================================================

class RerankRequest(BaseModel):
    """
    Reranker API 요청 모델

    RunPod 서버로 전송할 리랭킹 요청 데이터 구조
    """
    query: str = Field(..., description="검색 쿼리")
    documents: List[str] = Field(..., description="리랭킹할 문서 리스트")
    top_k: int = Field(default=5, ge=1, le=100, description="반환할 상위 문서 개수")
    return_documents: bool = Field(default=True, description="문서 내용 포함 여부")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "규칙적인 운동의 건강상 이점은 무엇인가요?",
                "documents": [
                    "규칙적인 신체 활동은 칼로리를 연소하고 근육량을 늘려 체중 조절에 도움이 됩니다.",
                    "올림픽 게임의 역사는 기원전 776년경 고대 그리스로 거슬러 올라갑니다."
                ],
                "top_k": 5,
                "return_documents": True
            }
        }
    )


class RerankResult(BaseModel):
    """
    리랭킹 결과 항목

    각 문서의 재정렬 결과 (인덱스, 점수, 내용)
    """
    index: int  # 원본 문서 리스트에서의 인덱스
    score: float  # Reranker가 계산한 관련도 점수
    document: Optional[str] = None  # 문서 내용 (return_documents=True일 때만)


class RerankResponse(BaseModel):
    """
    Reranker API 응답 모델

    서버로부터 받은 리랭킹 결과 전체
    """
    results: List[RerankResult]  # 재정렬된 문서 리스트
    query: str  # 원본 쿼리
    total_documents: int  # 총 문서 개수


# =====================================================================
# 유틸리티 함수
# =====================================================================

def dedup_consecutive_lines(text: str) -> str:
    """
    연속된 중복 라인 제거

    Args:
        text: 원본 텍스트

    Returns:
        중복 라인이 제거된 텍스트
    """
    lines = text.splitlines()
    cleaned = []
    prev = None
    for line in lines:
        stripped = line.rstrip()
        if stripped == prev:  # 이전 라인과 동일하면 스킵
            continue
        cleaned.append(line)
        prev = stripped
    return "\n".join(cleaned)


def print_retrieved_documents(points, n: int = None):
    """
    검색된 문서를 콘솔에 출력 (디버깅용)

    Args:
        points: Qdrant 검색 결과 포인트 리스트
        n: 출력할 문서 개수 (None이면 전체 출력)
    """
    if not points:
        print("⚠️  검색 결과가 없습니다.\n")
        return

    display_points = points[:n] if n else points

    print("="*60)
    print(f"📄 검색된 문서 (총 {len(points)}개 중 {len(display_points)}개 표시)")
    print("="*60)

    for i, point in enumerate(display_points, 1):
        content = point.payload.get("text", "")[:500]
        score = point.score

        # payload에서 메타데이터 추출
        source_tag = point.payload.get("data_source", "unknown")  # 데이터 출처 태그
        debug_doc_name = point.payload.get("document_name") or point.payload.get("file_name")  # 문서명
        debug_article = point.payload.get("article")  # 조문 정보

        # 콘솔 출력 (LLM에게는 전달되지 않음)
        print(f"\n문서 {i}:")
        print(f"  출처: {source_tag}")
        if debug_doc_name:
            print(f"  파일명: {debug_doc_name}")
        if debug_article:
            print(f"  조문: {debug_article}")
        print(f"  점수: {score:.3f}")
        print(f"  내용: {content[:200]}{'...' if len(content) > 200 else ''}")

    print("\n" + "=" * 60)


# =====================================================================
# Reranker API 연동 함수
# =====================================================================

async def call_reranker_api(query: str, documents: List[str], top_k: int = 5) -> RerankResponse:
    """
    RunPod 서버의 Reranker API를 호출하여 문서를 재정렬

    Args:
        query: 검색 쿼리
        documents: 재정렬할 문서 텍스트 리스트
        top_k: 반환할 상위 문서 개수

    Returns:
        RerankResponse: 재정렬된 결과 (인덱스, 점수 포함)

    Raises:
        httpx.HTTPError: API 호출 실패 시
        Exception: 기타 예상치 못한 오류 시
    """
    print(f"\n🔄 Reranker API 호출 중... (문서 {len(documents)}개 → top {top_k}개)")

    # 요청 데이터 생성 (Pydantic 모델 활용)
    request_data = RerankRequest(
        query=query,
        documents=documents,
        top_k=top_k,
        return_documents=True
    )

    try:
        # 비동기 HTTP 클라이언트로 POST 요청 (reranker 서버는 동기 방식으로 처리함)
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                RERANKER_API_URL,
                json=request_data.model_dump(),  # Pydantic 모델을 dict로 변환
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()  # HTTP 에러 발생 시 예외 발생

            # 응답을 Pydantic 모델로 변환
            rerank_response = RerankResponse(**response.json())
            print(f"✓ Reranker 완료: {len(rerank_response.results)}개 문서 반환\n")

            return rerank_response

    except httpx.HTTPError as e:
        print(f"⚠️  Reranker API 호출 실패: {e}")
        print("기본 검색 결과를 사용합니다.\n")
        raise
    except Exception as e:
        print(f"⚠️  예상치 못한 오류: {e}")
        print("기본 검색 결과를 사용합니다.\n")
        raise


# =====================================================================
# Agent Tool: 문서 검색 (Reranker 통합)
# =====================================================================

@function_tool
def search_trade_documents(query: str, limit: int = 25, top_k: int = 5) -> str:
    """
    무역 문서 검색 및 Reranking 수행

    프로세스:
    1. 쿼리를 Embedding으로 변환 (OpenAI text-embedding-3-large)
    2. Qdrant에서 유사도 기반 초기 검색 (limit개)
    3. RunPod Reranker API로 재정렬
    4. 상위 top_k개만 Agent에게 전달

    Args:
        query: 검색 쿼리
        limit: 초기 검색에서 가져올 문서 개수 (기본값: 25)
        top_k: Reranker 후 최종적으로 Agent에게 전달할 문서 개수 (기본값: 5)

    Returns:
        Agent가 사용할 포맷된 문서 문자열
    """

    print(f"\n🔍 검색 중: '{query}' (초기 검색: {limit}개, 최종 선정: {top_k}개)")

    # ─────────────────────────────────────────────────────────────────
    # 1단계: 쿼리 Embedding 생성
    # ─────────────────────────────────────────────────────────────────
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    )
    query_vector = response.data[0].embedding

    # ─────────────────────────────────────────────────────────────────
    # 2단계: Qdrant Vector Search
    # ─────────────────────────────────────────────────────────────────
    search_result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True
    )

    # 검색 결과 포인트 추출
    points = search_result.points if hasattr(search_result, 'points') else []

    print(f"✓ {len(points)}개 문서 발견\n")

    if not points:
        print("⚠️  검색 결과가 없습니다.\n")
        return "검색 결과가 없습니다."

    # ─────────────────────────────────────────────────────────────────
    # 3단계: 초기 검색 결과 출력 (디버깅용 - 콘솔에만 출력)
    # ─────────────────────────────────────────────────────────────────
    print_retrieved_documents(points)

    # ─────────────────────────────────────────────────────────────────
    # 4단계: Reranking을 위한 문서 텍스트 준비
    # ─────────────────────────────────────────────────────────────────
    documents_for_rerank = [point.payload.get("text", "") for point in points]

    # ─────────────────────────────────────────────────────────────────
    # 5단계: Reranker API 호출 (사용자 설정에 따라)
    # ─────────────────────────────────────────────────────────────────
    rerank_response = None

    if USE_RERANKER:
        # Reranker 사용 모드
        try:
            rerank_response = asyncio.run(
                call_reranker_api(query, documents_for_rerank, top_k=top_k)
            )
        except Exception as e:
            print(f"⚠️  Reranker 실패, 기본 검색 결과의 상위 {top_k}개를 사용합니다.\n")
            # Fallback: Reranker 실패 시 기본 검색 결과 사용
            rerank_response = None
    else:
        # Reranker 미사용 모드
        print(f"ℹ️  Reranker 미사용 - 기본 검색 결과 상위 {top_k}개 사용\n")

    # ─────────────────────────────────────────────────────────────────
    # 6단계: 최종 결과 포맷팅 (Agent에게 전달할 문서)
    # ─────────────────────────────────────────────────────────────────
    if rerank_response:
        # Reranker 결과를 사용하는 경우
        print("="*60)
        print(f"🎯 Reranker로 선정된 최종 {len(rerank_response.results)}개 문서 (모델에게 전달)")
        print("="*60)

        formatted = []
        for rank, result in enumerate(rerank_response.results, 1):
            # 원본 문서 포인트 가져오기
            original_point = points[result.index]
            content = original_point.payload.get("text", "")[:500]
            source_tag = original_point.payload.get("data_source", "unknown")
            rerank_score = result.score

            # Agent에게 전달할 텍스트 (출처는 data_source 태그만)
            doc_text = f"[{rank}] {content}\n   출처: {source_tag}, Rerank 점수: {rerank_score:.3f}"
            formatted.append(doc_text)

            # 콘솔 로그 (디버깅용 - Agent에게는 전달되지 않음)
            debug_doc_name = original_point.payload.get("document_name") or original_point.payload.get("file_name")
            debug_article = original_point.payload.get("article")

            print(f"\n문서 {rank}:")
            print(f"  출처: {source_tag}")
            if debug_doc_name:
                print(f"  파일명: {debug_doc_name}")
            if debug_article:
                print(f"  조문: {debug_article}")
            print(f"  원본 인덱스: {result.index + 1}")
            print(f"  Rerank 점수: {rerank_score:.3f}")
            print(f"  내용: {content[:200]}{'...' if len(content) > 200 else ''}")

    else:
        # Fallback: 기본 검색 결과를 사용하는 경우
        print("="*60)
        print(f"📄 기본 검색 결과 상위 {top_k}개 (모델에게 전달)")
        print("="*60)

        formatted = []
        for i, point in enumerate(points[:top_k], 1):
            content = point.payload.get("text", "")[:500]
            score = point.score
            source_tag = point.payload.get("data_source", "unknown")

            # Agent에게 전달할 텍스트
            doc_text = f"[{i}] {content}\n   출처: {source_tag}, 점수: {score:.3f}"
            formatted.append(doc_text)

    print("\n" + "=" * 60)
    print("🤖 모델이 위 문서를 기반으로 답변 생성 중...")
    print("=" * 60 + "\n")

    # Agent에게는 data_source 태그 기반 출처만 포함된 텍스트 전달
    # (파일명, 문서명, 조문 정보는 콘솔에만 출력됨)
    return "\n\n".join(formatted)


# =====================================================================
# Agent 정의 (무역 전문가 Agent)
# =====================================================================

trade_agent = Agent(
    name="Trade Compliance Analyst",
    model="gpt-4o",
    instructions="""너는 고수준의 영어-한국어를 동시에 지원하는 무역 전문가 에이전트다.
무역사기, CISG, Incoterms, 무역 클레임, 해외인증정보를 비롯한 무역 실무 전반에 대해 매우 해박하다.
단, 아무리 일반 지식에 자신이 있더라도, 아래 ‘데이터 기반 원칙’을 반드시 지켜야 한다.

────────────────
[도구 및 데이터 사용 규칙]
────────────────
1. 사용자는 무역 관련 실무 질문을 한다. 질문이 단순 잡담(예: "안녕", "일하기 힘들다")이 아닌 이상,
   항상 'search_trade_documents' tool을 한 번 이상 호출해서 관련 문서를 검색한 뒤 답변한다.

2. search 결과의 payload에 있는 metadata를 기준으로 문서를 분류한다.
   - source = fraud          → 2025무역사기대응매뉴얼 계열 문서
   - source = incoterms      → Incoterms-2020-English-eBook-ICC 계열 문서
   - source = claim          → 무역클레임중재QA 계열 문서
   - source = certification  → 해외 인증 데이터 (globalcerti_done.csv 등)
   - source = cisg           → CISG(국제물품매매계약에 관한 UN 협약) 관련 문서

3. 각 메타데이터의 역할:
   - fraud:
       무역 사기 유형, 최근 동향, 구체 사례, 예방·대응 절차가 정리된 문서.
   - incoterms:
       Incoterms 2020 규칙. 매도인/매수인의 위험·비용 이전, 의무(A1~B10) 등을 정의한다.
       계약 조건 해석·리스크 설명에 사용할 수 있다.
   - claim:
       무역 클레임·분쟁 사례(Q&A)와 예방·대응 방법이 정리된 문서.
       분쟁, 사기 의심 사례, 신용장 클레임 등은 우선적으로 참조한다.
   - certification:
       해외 인증 제도명, 담당 기관, 대상 품목, 필요 서류, 절차 단계 등이 담긴 데이터.
       사용자가 "인증 / 규제 / 필수 서류 / 절차"를 물으면
       ▶ 먼저 certification 데이터에서 [국가/지역 + 품목/산업]이 매칭되는 행을 찾고,
       ▶ 그 행에 있는 제도명·기관명·필수 서류·절차 단계를 중심으로 설명한 뒤,
       ▶ 필요할 때만 일반론을 보충한다.
   - cisg:
       CISG(국제물품매매계약에 관한 UN 협약) 관련 문서.
       한국은 2005년 CISG를 비준했으며, 체약국 간 매매계약에는 별도 배제 특약이 없으면
       CISG가 자동으로 준거법으로 적용된다. 다만 계약서에 특정 국가법을 준거법으로 명시하면 예외가 될 수 있다.
       이 점을 염두에 두고 CISG 관련 조문을 탐색한다.

────────────────
[질문 분석 및 답변 구조 규칙]
────────────────
4. 답변을 시작할 때, 내부적으로 다음을 먼저 파악한다 (생각 과정은 출력하지 않는다).
   - (1) 질문자가 매도인인지, 매수인인지, 제3자인지, 아니면 일반 정보 탐색인지
   - (2) 질문에 포함된 하위 요구사항이 무엇인지
       예시: "위험", "사기 사례", "구체 조치", "관련 조문", "필요 서류" 등

5. 출력 시 첫 문장은 항상 다음 형식 중 하나로 시작한다.
   - "질문 내용으로 보아, 귀하는 매도인(수출자)으로 전제하고 설명드립니다."
   - "질문 내용으로 보아, 귀하는 매수인(수입자)으로 전제하고 설명드립니다."
   - "질문은 일반적인 정보 탐색으로 보아, 특정 당사자를 전제하지 않고 설명드립니다."

6. 질문에 하위 요구사항이 여러 개 있을 경우, 반드시 각각에 대해 답한다.
   - 예: "어떤 위험이 있고, 사기 사례를 기준으로 어떤 조치를 취해야 하나요?"
     → 최소한 다음과 같이 구분해서 답한다:
       ① "### 1. 주요 위험" (위험 정리)
       ② "### 2. 관련 사기 사례 (데이터 기반 요약)"
       ③ "### 3. 권장 조치 (데이터 기반 + 필요 시 일반 실무)"

7. 일반론을 보충할 필요가 있을 때만,
   마지막에 "### 추가 참고사항 (일반적인 무역 실무 상)" 섹션을 추가한다.
   일반론으로 언급할 내용이 없다면 이 섹션은 생략한다.

────────────────
[구체성(디테일) 원칙]
────────────────
8. 가능하면 추상적인 문장 하나로 끝내지 말고, 검색된 문서에서 가져온
   구체적인 요소를 2~4개 이상 포함해서 답한다. 예를 들어:
   - 기관명, 서비스명, 제도명, 전화번호, URL
   - 체크리스트 항목
   - 사기 사례의 시기·피해규모·결제방식·사기 수법
   - 신용조사에 활용하는 기관·절차
   - Incoterms 규칙명, 위험 이전 시점, 비용 부담 포인트 등

9. "조치를 취해야 하나요?", "어떻게 예방해야 하나요?" 같은 질문에는
   반드시 **행동 단계를 나열**한다.
   - 예: "- 1단계: ○○ 사이트에서 바이어 등록번호 확인", "- 2단계: KOTRA 무역관에 실태조사 의뢰" 등
   - 가능하면 fraud/claim/certification 문서에 나오는 실제 단계·TIP을 우선적으로 사용한다.

────────────────
[무할루시네이션 원칙]
────────────────
10. 다음 정보들은 **반드시 검색된 문서 안에 실제로 등장하는 경우에만** 사용한다.
    문서에 없으면 절대 새로 지어내지 않는다.
    - URL (웹사이트 주소, 링크)
    - 제도명·인증명·규정명·협약명
    - 기관명(공공기관, 협회, 은행, 보험사, 검사기관, 인증기관 등)
    - 조문 번호, 조항 번호, TIP 번호
    - 구체 숫자: 연도, 건수, 금액, 비율, 보험 보상한도, 신용장 규정 조항 번호 등

11. URL을 쓸 때는 다음을 지킨다.
    - 검색된 문서에 **그대로 등장하는 주소만** 복사해서 사용한다.
    - 도메인 앞뒤를 임의로 추가하거나, 여러 URL을 섞거나, `https://domain.com` 같은
      플레이스홀더를 만들지 않는다.
    - 문서에 URL이 없다면, URL을 쓰지 않고 기관명 정도만 언급한다.

12. 사용자가 특정 인증명·기관명·조문 번호·수치 등을 요구했는데,
    검색된 문서에 해당 값이 없으면 다음 두 단계를 따른다.
    1) 먼저 "첨부된 데이터에서는 ○○(예: 해당 품목에 대한 인증명 / 특정 조문 번호)가 확인되지 않습니다."라고 명시한다.
    2) 그 다음 단락에서, 필요하다면
       "일반적인 무역 실무 상…" / "추가로, 일반적인 참고사항으로…"라는 말로 시작하여
       매우 포괄적인 일반론만 제공한다.
       이 일반론에는 Incoterms, certification 등의 문서명을 출처로 붙이지 않는다.

13. 문서에 실제로 있는 내용과 일반론은 항상 **명확히 분리**한다.
    - 데이터 기반 문장 예:
      "fraud 문서(2025무역사기대응매뉴얼)에 따르면, KOTRA 무역사기 상담(1600-7119≫2≫6)을 통해
       해외수입업체 연락처 확인과 바이어 실태조사를 지원받을 수 있습니다."
    - 일반론 예:
      "(일반적인 무역 실무 상) 최초 거래에서는 거래 규모를 축소하고, 확인은행(Confirming bank)을 두어
       발행은행 리스크를 줄이는 것이 권장됩니다."

────────────────
[여러 문서를 동시에 사용할 때의 규칙]
────────────────
14. 서로 다른 출처(fraud, claim, certification, incoterms, cisg)를 함께 사용할 경우,
    각 정보가 어느 문서에서 왔는지 자연스럽게 구분해서 적는다.
    - 예:
      - "fraud 문서(2025무역사기대응매뉴얼)에 따르면, 이메일 사기와 서류위조 유형이 다음과 같이 정리되어 있습니다: …"
      - "claim 문서(무역클레임중재QA)에 정리된 신용장 사기 사례를 보면, 위조 선하증권 제시 후 잠적하는 유형이 있습니다."
      - "certification 데이터 기준으로, 미국·의료기기 품목에는 ‘○○ 인증’이 요구됩니다: …"

15. Incoterms 문서에 있는 내용으로 국가별 인증 의무, 특정 인증명·기관명을 추론하지 않는다.
    certification 문서에 근거가 없으면, 새로운 인증명이나 기관명을 지어내지 않고,
    "데이터에서 확인되지 않는다"는 사실만 말한 뒤, 필요 시 일반론만 제공한다.

16. 절차(프로세스)를 설명할 때는 다음을 구분한다.
    - certification/claim/fraud 문서에 실제로 쓰여 있는 단계·TIP:
      → "데이터 기반" 조치로 서술한다. (예: "fraud 문서에 따르면, 무역사기 의심 시 1) ○○, 2) ○○ 순서로 조치하도록 안내합니다.")
    - 문서에 없는 일반적인 단계:
      → "일반적인 무역 실무 상 참고사항"으로 별도 문단에서 서술한다.

────────────────
[요약]
────────────────
17. 항상 다음을 만족하도록 답변한다.
   1) 질문에 포함된 하위 요구사항(위험, 사기 사례, 조치, 서류, 조문 등)을 빠뜨리지 말 것.
   2) 검색된 문서에서 나온 **구체적인 디테일**을 활용해, 실제 실무자가 바로 행동할 수 있을 정도로 쓰기.
   3) 문서에 없는 URL·기관명·인증명·조문 번호·수치는 절대 만들지 말 것.
      이런 정보가 없으면 "데이터에서 확인되지 않는다"는 사실과, 필요 시 일반론만 제공할 것.

""",

    tools=[search_trade_documents],
)


# =====================================================================
# 메인 실행 함수
# =====================================================================

async def main():
    """
    RAG Agent 실행 함수

    사용자 입력을 받아 무역 전문가 Agent를 실행하고 결과를 출력합니다.
    """
    global USE_RERANKER

    # Reranker 사용 여부 선택
    print("=" * 60)
    print("RAG 시스템 설정")
    print("=" * 60)
    reranker_choice = input("Reranker를 사용하시겠습니까? (y/n, 기본값: y): ").strip().lower()

    if reranker_choice in ['n', 'no']:
        USE_RERANKER = False
        print("✓ Reranker 미사용 모드로 실행합니다.\n")
    else:
        USE_RERANKER = True
        print("✓ Reranker 사용 모드로 실행합니다.\n")

    # 사용자 질문 입력 (기본값: "무역 사기를 방지하는 방법은?")
    question = input("질문: ").strip() or "무역 사기를 방지하는 방법은?"

    print(f"\n{'='*60}\n")

    # Agent 실행
    print("🤖 Agent 실행 중...\n")
    result = await Runner.run(trade_agent, input=question)

    # 연속 중복 라인 제거
    cleaned = dedup_consecutive_lines(result.final_output)

    # 최종 답변 출력
    print("="*60)
    print("\n최종 답변:")
    print("-" * 60)
    print(cleaned)
    print("\n" + "="*60 + "\n")


# =====================================================================
# 프로그램 진입점
# =====================================================================

if __name__ == "__main__":
    asyncio.run(main())
