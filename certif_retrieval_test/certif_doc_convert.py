"""
Certification Data RAG Document Converter

This script converts certification CSV data into RAG-optimized documents.
It supports multiple output formats and optional auto-summarization.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict
import re


class CertificationRAGConverter:
    """Convert certification data to RAG-optimized documents."""

    def __init__(self, csv_path: str):
        """
        Initialize the converter.

        Args:
            csv_path: Path to the globalcerti_done.csv file
        """
        self.csv_path = Path(csv_path)
        self.df = None
        self.documents = []

    def load_data(self) -> pd.DataFrame:
        """Load CSV data with proper encoding."""
        print(f"Loading data from {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
        print(f"Loaded {len(self.df)} certification records")
        return self.df

    def generate_auto_summary(self, cert_subject: str, max_length: int = 150) -> str:
        """
        Generate a concise summary from cert_subject.
        Uses simple extraction logic (first N chars or first sentence).
        For LLM-based summarization, integrate with OpenAI/Anthropic API.

        Args:
            cert_subject: The full certification description
            max_length: Maximum length for summary

        Returns:
            Auto-generated summary
        """
        if pd.isna(cert_subject) or not cert_subject:
            return "요약 정보 없음"

        # Clean the text
        text = cert_subject.strip()

        # Try to get first meaningful sentence
        sentences = re.split(r'[.!?]\s+', text)
        if sentences and len(sentences[0]) > 20:
            first_sentence = sentences[0]
            if len(first_sentence) <= max_length:
                return first_sentence
            else:
                return first_sentence[:max_length] + "..."

        # Fallback: truncate to max_length
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text

    def create_document_text(self, row: pd.Series, idx: int, include_summary: bool = True) -> str:
        """
        Create formatted document text for a single certification.

        Args:
            row: DataFrame row containing certification data
            idx: Row index (used as ID)
            include_summary: Whether to generate and include auto-summary

        Returns:
            Formatted document string
        """
        # Generate auto-summary if requested
        auto_summary = ""
        if include_summary:
            auto_summary = self.generate_auto_summary(row.get('cert_subject', ''))

        # Create structured document
        separator = '=' * 80
        doc = f"""{separator}
[ID: {idx}]

국가: {row.get('country', 'N/A')}
카테고리: {row.get('category', 'N/A')}
인증 구분: {row.get('cert_type', 'N/A')}
대표 인증: {row.get('main_cert', 'N/A')}
인증명: {row.get('cert_name', 'N/A')}

설명:
{row.get('cert_subject', 'N/A')}
"""

        if include_summary and auto_summary:
            doc += f"""
요약:
{auto_summary}
"""

        doc += f"""
출처:
{row.get('url', 'N/A')}
{separator}
"""
        return doc

    def create_document_dict(self, row: pd.Series, idx: int, include_summary: bool = True) -> Dict:
        """
        Create structured dictionary for a single certification.
        Optimized for vector databases and JSON storage.

        Args:
            row: DataFrame row containing certification data
            idx: Row index (used as ID)
            include_summary: Whether to generate and include auto-summary

        Returns:
            Dictionary with certification data
        """
        doc = {
            "id": idx,
            "country": row.get('country', ''),
            "category": row.get('category', ''),
            "cert_type": row.get('cert_type', ''),
            "main_cert": row.get('main_cert', ''),
            "cert_name": row.get('cert_name', ''),
            "cert_subject": row.get('cert_subject', ''),
            "url": row.get('url', ''),
            "metadata": {
                "source": "globalcerti.kr",
                "type": "certification"
            }
        }

        if include_summary:
            doc["auto_summary"] = self.generate_auto_summary(row.get('cert_subject', ''))

        # Create searchable text field (combines all relevant fields)
        doc["text"] = f"""인증명: {doc['cert_name']}
국가: {doc['country']} | 카테고리: {doc['category']} | 인증구분: {doc['cert_type']}
대표인증: {doc['main_cert']}

{doc['cert_subject']}"""

        if include_summary:
            doc["text"] += f"\n\n요약: {doc['auto_summary']}"

        return doc

    def convert_to_text_format(self, output_path: str, include_summary: bool = True) -> None:
        """
        Convert all certifications to a single text file.

        Args:
            output_path: Path for output text file
            include_summary: Whether to include auto-summaries
        """
        if self.df is None:
            self.load_data()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"Converting to text format: {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, row in self.df.iterrows():
                doc_text = self.create_document_text(row, idx, include_summary)
                f.write(doc_text + "\n")

        print(f"✓ Saved {len(self.df)} documents to {output_file}")

    def convert_to_jsonl(self, output_path: str, include_summary: bool = True) -> None:
        """
        Convert to JSON Lines format (one JSON object per line).
        Recommended for vector databases like Pinecone, Weaviate, Qdrant.

        Args:
            output_path: Path for output JSONL file
            include_summary: Whether to include auto-summaries
        """
        if self.df is None:
            self.load_data()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"Converting to JSONL format: {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, row in self.df.iterrows():
                doc_dict = self.create_document_dict(row, idx, include_summary)
                f.write(json.dumps(doc_dict, ensure_ascii=False) + "\n")

        print(f"✓ Saved {len(self.df)} documents to {output_file}")

    def convert_to_json(self, output_path: str, include_summary: bool = True) -> None:
        """
        Convert to structured JSON array.

        Args:
            output_path: Path for output JSON file
            include_summary: Whether to include auto-summaries
        """
        if self.df is None:
            self.load_data()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"Converting to JSON format: {output_file}")

        documents = []
        for idx, row in self.df.iterrows():
            doc_dict = self.create_document_dict(row, idx, include_summary)
            documents.append(doc_dict)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)

        print(f"✓ Saved {len(documents)} documents to {output_file}")

    def convert_to_individual_files(self, output_dir: str, include_summary: bool = True) -> None:
        """
        Convert each certification to a separate text file.
        Useful for document-based RAG systems.

        Args:
            output_dir: Directory for output files
            include_summary: Whether to include auto-summaries
        """
        if self.df is None:
            self.load_data()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Converting to individual files in: {output_path}")

        for idx, row in self.df.iterrows():
            # Create filename from cert_name (sanitized)
            cert_name = row.get('cert_name', f'cert_{idx}')
            filename = re.sub(r'[^\w\s-]', '', cert_name)
            filename = re.sub(r'[-\s]+', '_', filename)
            filename = f"{idx:04d}_{filename[:50]}.txt"

            doc_text = self.create_document_text(row, idx, include_summary)

            file_path = output_path / filename
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(doc_text)

        print(f"✓ Saved {len(self.df)} documents to {output_path}")

    def get_statistics(self) -> Dict:
        """Get statistics about the certification data."""
        if self.df is None:
            self.load_data()

        stats = {
            "total_certifications": len(self.df),
            "countries": self.df['country'].nunique(),
            "categories": self.df['category'].nunique(),
            "cert_types": self.df['cert_type'].value_counts().to_dict(),
            "top_countries": self.df['country'].value_counts().head(10).to_dict(),
            "top_categories": self.df['category'].value_counts().head(10).to_dict()
        }

        return stats


def main():
    """Main execution function with examples."""

    # Initialize converter
    csv_path = "/Users/hoon/Desktop/SKN-17-Final-5Team/data/OCR 데이터/globalcerti_done.csv"
    converter = CertificationRAGConverter(csv_path)

    # Load data
    converter.load_data()

    # Print statistics
    print("\n" + "="*80)
    print("CERTIFICATION DATA STATISTICS")
    print("="*80)
    stats = converter.get_statistics()
    print(f"Total certifications: {stats['total_certifications']}")
    print(f"Unique countries: {stats['countries']}")
    print(f"Unique categories: {stats['categories']}")
    print("\nTop 5 Countries:")
    for country, count in list(stats['top_countries'].items())[:5]:
        print(f"  - {country}: {count}")
    print("\nTop 5 Categories:")
    for category, count in list(stats['top_categories'].items())[:5]:
        print(f"  - {category}: {count}")
    print("="*80 + "\n")

    # Create output directory
    output_dir = Path("/Users/hoon/Desktop/SKN-17-Final-5Team/retrieval_test/output")
    output_dir.mkdir(exist_ok=True)

    # Convert to different formats
    print("Converting to multiple RAG-optimized formats...\n")

    # 1. JSON Lines format (RECOMMENDED for vector DBs)
    converter.convert_to_jsonl(
        output_dir / "certifications.jsonl",
        include_summary=True
    )

    # 2. Single text file
    converter.convert_to_text_format(
        output_dir / "certifications.txt",
        include_summary=True
    )

    # 3. Structured JSON
    converter.convert_to_json(
        output_dir / "certifications.json",
        include_summary=True
    )

    # 4. Individual text files (optional - commented out by default)
    # converter.convert_to_individual_files(
    #     output_dir / "individual_docs",
    #     include_summary=True
    # )

    print("\n" + "="*80)
    print("✓ CONVERSION COMPLETE")
    print("="*80)
    print(f"\nOutput files saved to: {output_dir}")
    print("\nRecommended formats for RAG:")
    print("  1. certifications.jsonl - Best for vector databases (Pinecone, Weaviate, Qdrant)")
    print("  2. certifications.json - Good for structured storage and APIs")
    print("  3. certifications.txt - Simple text format for basic RAG systems")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
