"""심플 RAG 테스트 (OpenAI Agents SDK)"""

import asyncio
import os
from dotenv import load_dotenv
from agents import Agent, Runner, function_tool
from qdrant_client import QdrantClient
from openai import OpenAI

load_dotenv()

# Initialize clients
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=60
)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

COLLECTION_NAME = "trade_collection"
EMBEDDING_MODEL = "text-embedding-3-large"


@function_tool
def search_trade_documents(query: str, limit: int = 5) -> str:
    """Search the trade compliance knowledge base."""
    print(f"\n🔍 검색 중: '{query}' (limit: {limit})")

    # Generate query embedding
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    )
    query_vector = response.data[0].embedding

    # Search Qdrant using the new query_points API
    search_result = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True
    )

    # Access points from the response
    points = search_result.points if hasattr(search_result, 'points') else []

    print(f"✓ {len(points)}개 문서 발견\n")

    # Format results for the agent
    if not points:
        print("⚠️  검색 결과가 없습니다.\n")
        return "검색 결과가 없습니다."

    # Print retrieved documents BEFORE sending to model
    print("="*60)
    print("📄 검색된 문서 (모델에게 전달되기 전)")
    print("="*60)

    formatted = []
    for i, point in enumerate(points, 1):
        content = point.payload.get("text", "")[:500]
        score = point.score
        source = point.payload.get("data_source", "unknown")

        # Try to get more specific source info
        if "article" in point.payload:
            source = f"CISG Article {point.payload.get('article')}"
        elif "document_name" in point.payload:
            source = point.payload.get("document_name")
        elif "file_name" in point.payload:
            source = point.payload.get("file_name")

        doc_text = f"[{i}] {content}\n   출처: {source}, 점수: {score:.3f}"
        formatted.append(doc_text)

        # Print to console
        print(f"\n문서 {i}:")
        print(f"  출처: {source}")
        print(f"  점수: {score:.3f}")
        print(f"  내용: {content[:200]}{'...' if len(content) > 200 else ''}")

    print("\n" + "="*60)
    print("🤖 모델이 위 문서를 기반으로 답변 생성 중...")
    print("="*60 + "\n")

    return "\n\n".join(formatted)


# Define the RAG agent (프롬프)
trade_agent = Agent(
    name="Trade Compliance Analyst",
    model="gpt-4o",
    instructions="""You are a bilingual trade compliance analyst specializing in international commerce,
fraud mitigation, CISG, incoterms, and trade claims, and certrifications.

When answering questions:
1. Use the search_trade_documents tool to find relevant information
2. Answer in Korean always based on search results
3. Always cite sources with the meta data of the search results

Be concise and professional.""",
    tools=[search_trade_documents],
)


async def main():
    """Run the RAG agent."""
    question = input("질문: ").strip() or "무역 사기를 방지하는 방법은?"

    print(f"\n{'='*60}\n")

    # Run the agent
    print("🤖 Agent 실행 중...\n")
    result = await Runner.run(trade_agent, input=question)

    # Display final output
    print("="*60)
    print("\n최종 답변:")
    print("-" * 60)
    print(result.final_output)
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
