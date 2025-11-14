"""
Simple test script for RAG document retrieval.
Demonstrates basic keyword search without requiring external dependencies.
"""

import json
from pathlib import Path
from typing import List, Dict
import re


class SimpleRetriever:
    """Simple keyword-based retriever for testing RAG documents."""

    def __init__(self, jsonl_path: str):
        """
        Initialize retriever with JSONL documents.

        Args:
            jsonl_path: Path to certifications.jsonl file
        """
        self.documents = []
        self.load_documents(jsonl_path)

    def load_documents(self, jsonl_path: str) -> None:
        """Load documents from JSONL file."""
        print(f"Loading documents from {jsonl_path}...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                self.documents.append(doc)
        print(f"Loaded {len(self.documents)} documents")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Simple keyword-based search.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of matching documents with scores
        """
        results = []

        # Normalize query
        query_lower = query.lower()
        query_terms = set(query_lower.split())

        for doc in self.documents:
            # Calculate simple relevance score
            score = 0

            # Check in searchable text field
            text = doc.get('text', '').lower()

            # Exact phrase match (highest score)
            if query_lower in text:
                score += 10

            # Individual term matches
            for term in query_terms:
                score += text.count(term)

            # Boost for matches in important fields
            if query_lower in doc.get('cert_name', '').lower():
                score += 5
            if query_lower in doc.get('country', '').lower():
                score += 3
            if query_lower in doc.get('category', '').lower():
                score += 3

            if score > 0:
                results.append({
                    'doc': doc,
                    'score': score
                })

        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)

        return results[:top_k]

    def print_results(self, results: List[Dict]) -> None:
        """Pretty print search results."""
        if not results:
            print("No results found.")
            return

        print(f"\nFound {len(results)} results:\n")
        print("=" * 80)

        for i, result in enumerate(results, 1):
            doc = result['doc']
            score = result['score']

            print(f"\n[Result {i}] (Score: {score})")
            print(f"ID: {doc['id']}")
            print(f"인증명: {doc['cert_name']}")
            print(f"국가: {doc['country']}")
            print(f"카테고리: {doc['category']}")
            print(f"인증구분: {doc['cert_type']}")
            print(f"대표인증: {doc['main_cert']}")

            if doc.get('auto_summary'):
                print(f"\n요약: {doc['auto_summary'][:200]}...")

            print(f"\n출처: {doc['url']}")
            print("-" * 80)

    def filter_by_country(self, country: str) -> List[Dict]:
        """Filter documents by country."""
        return [doc for doc in self.documents if country.lower() in doc['country'].lower()]

    def filter_by_category(self, category: str) -> List[Dict]:
        """Filter documents by category."""
        return [doc for doc in self.documents if category.lower() in doc['category'].lower()]

    def filter_by_cert_type(self, cert_type: str) -> List[Dict]:
        """Filter documents by certification type."""
        return [doc for doc in self.documents if cert_type.lower() in doc['cert_type'].lower()]

    def get_statistics(self) -> Dict:
        """Get statistics about loaded documents."""
        countries = {}
        categories = {}
        cert_types = {}

        for doc in self.documents:
            # Count countries
            country = doc.get('country', 'Unknown')
            countries[country] = countries.get(country, 0) + 1

            # Count categories
            category = doc.get('category', 'Unknown')
            categories[category] = categories.get(category, 0) + 1

            # Count cert types
            cert_type = doc.get('cert_type', 'Unknown')
            cert_types[cert_type] = cert_types.get(cert_type, 0) + 1

        return {
            'total': len(self.documents),
            'countries': len(countries),
            'categories': len(categories),
            'top_countries': sorted(countries.items(), key=lambda x: x[1], reverse=True)[:5],
            'top_categories': sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5],
            'cert_types': sorted(cert_types.items(), key=lambda x: x[1], reverse=True)
        }


def main():
    """Main test function with example queries."""

    # Initialize retriever
    jsonl_path = "/Users/hoon/Desktop/SKN-17-Final-5Team/retrieval_test/output/certifications.jsonl"
    retriever = SimpleRetriever(jsonl_path)

    # Print statistics
    print("\n" + "=" * 80)
    print("DOCUMENT STATISTICS")
    print("=" * 80)
    stats = retriever.get_statistics()
    print(f"Total documents: {stats['total']}")
    print(f"Unique countries: {stats['countries']}")
    print(f"Unique categories: {stats['categories']}")
    print("\nTop 5 Countries:")
    for country, count in stats['top_countries']:
        print(f"  - {country}: {count}")
    print("\nTop 5 Categories:")
    for category, count in stats['top_categories']:
        print(f"  - {category}: {count}")
    print("\nCertification Types:")
    for cert_type, count in stats['cert_types']:
        print(f"  - {cert_type}: {count}")

    # Example searches
    test_queries = [
        "미국 의료기기",
        "CE 인증",
        "식품",
        "전기전자",
        "완구 안전"
    ]

    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"QUERY: '{query}'")
        print("=" * 80)

        results = retriever.search(query, top_k=3)
        retriever.print_results(results)

    # Example filtering
    print("\n" + "=" * 80)
    print("FILTER EXAMPLE: 유럽연합 certifications")
    print("=" * 80)
    eu_docs = retriever.filter_by_country("유럽연합")
    print(f"\nFound {len(eu_docs)} certifications from 유럽연합")
    if eu_docs:
        print("\nSample certifications:")
        for doc in eu_docs[:5]:
            print(f"  - {doc['cert_name']} ({doc['category']})")

    # Interactive mode
    print("\n" + "=" * 80)
    print("INTERACTIVE SEARCH MODE")
    print("=" * 80)
    print("Enter search queries (or 'quit' to exit):\n")

    while True:
        try:
            user_query = input("Query: ").strip()

            if user_query.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break

            if not user_query:
                continue

            # Check for filter commands
            if user_query.startswith("country:"):
                country = user_query.replace("country:", "").strip()
                docs = retriever.filter_by_country(country)
                print(f"\nFound {len(docs)} certifications from '{country}'")
                for doc in docs[:10]:
                    print(f"  - {doc['cert_name']} ({doc['category']})")

            elif user_query.startswith("category:"):
                category = user_query.replace("category:", "").strip()
                docs = retriever.filter_by_category(category)
                print(f"\nFound {len(docs)} certifications in category '{category}'")
                for doc in docs[:10]:
                    print(f"  - {doc['cert_name']} ({doc['country']})")

            else:
                # Regular search
                results = retriever.search(user_query, top_k=5)
                retriever.print_results(results)

            print()

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
