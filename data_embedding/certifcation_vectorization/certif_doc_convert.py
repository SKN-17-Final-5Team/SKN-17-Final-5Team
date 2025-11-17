"""
인증 데이터 RAG 문서 변환기

인증 CSV 데이터를 RAG에 최적화된 문서로 변환합니다.
다양한 출력 형식과 선택적 자동 요약 기능을 지원합니다.
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict
import re


class CertificationRAGConverter:
    """인증 데이터를 RAG에 최적화된 문서로 변환합니다."""

    def __init__(self, csv_path: str):
        """
        변환기를 초기화합니다.

        Args:
            csv_path: globalcerti_done.csv 파일 경로
        """
        self.csv_path = Path(csv_path)
        self.df = None
        self.documents = []

    def load_data(self) -> pd.DataFrame:
        """CSV 데이터를 적절한 인코딩으로 로드합니다."""
        print(f"Loading data from {self.csv_path}...")
        self.df = pd.read_csv(self.csv_path, encoding='utf-8-sig')
        print(f"Loaded {len(self.df)} certification records")
        return self.df

    def generate_auto_summary(self, cert_subject: str, max_length: int = 150) -> str:
        """
        cert_subject에서 간결한 요약을 생성합니다.
        간단한 추출 로직(첫 N개 문자 또는 첫 문장)을 사용합니다.

        Args:
            cert_subject: 전체 인증 설명
            max_length: 요약의 최대 길이

        Returns:
            자동 생성된 요약
        """
        if pd.isna(cert_subject) or not cert_subject:
            return "요약 정보 없음"

        # 텍스트 정제
        text = cert_subject.strip()

        # 첫 번째 의미 있는 문장 추출 시도
        sentences = re.split(r'[.!?]\s+', text)
        if sentences and len(sentences[0]) > 20:
            first_sentence = sentences[0]
            if len(first_sentence) <= max_length:
                return first_sentence
            else:
                return first_sentence[:max_length] + "..."

        # 대체 방법: 최대 길이로 자르기
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text


    def create_document_text(self, row: pd.Series, idx: int, include_summary: bool = True) -> str:
        """
        단일 인증에 대한 형식화된 문서 텍스트를 생성합니다.

        Args:
            row: 인증 데이터를 포함하는 DataFrame 행
            idx: 행 인덱스 (ID로 사용됨)
            include_summary: 자동 요약을 생성하고 포함할지 여부

        Returns:
            형식화된 문서 문자열
        """
        # 요청 시 자동 요약 생성
        auto_summary = ""
        if include_summary:
            auto_summary = self.generate_auto_summary(row.get('cert_subject', ''))

        # 구조화된 문서 생성
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
        단일 인증에 대한 구조화된 딕셔너리를 생성합니다.
        벡터 데이터베이스 및 JSON 저장에 최적화되어 있습니다.

        Args:
            row: 인증 데이터를 포함하는 DataFrame 행
            idx: 행 인덱스 (ID로 사용됨)
            include_summary: 자동 요약을 생성하고 포함할지 여부

        Returns:
            인증 데이터를 포함하는 딕셔너리
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

        # 검색 가능한 텍스트 필드 생성 (모든 관련 필드를 결합)
        doc["text"] = f"""인증명: {doc['cert_name']}
국가: {doc['country']} | 카테고리: {doc['category']} | 인증구분: {doc['cert_type']}
대표인증: {doc['main_cert']}

{doc['cert_subject']}"""

        if include_summary:
            doc["text"] += f"\n\n요약: {doc['auto_summary']}"

        return doc

    def convert_to_text_format(self, output_path: str, include_summary: bool = True) -> None:
        """
        모든 인증을 단일 텍스트 파일로 변환합니다.

        Args:
            output_path: 출력 텍스트 파일 경로
            include_summary: 자동 요약 포함 여부
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
        JSON Lines 형식으로 변환합니다 (한 줄에 하나의 JSON 객체).
        Pinecone, Weaviate, Qdrant 같은 벡터 데이터베이스에 권장됩니다.

        Args:
            output_path: 출력 JSONL 파일 경로
            include_summary: 자동 요약 포함 여부
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
        구조화된 JSON 배열로 변환합니다.

        Args:
            output_path: 출력 JSON 파일 경로
            include_summary: 자동 요약 포함 여부
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
        각 인증을 별도의 텍스트 파일로 변환합니다.
        문서 기반 RAG 시스템에 유용합니다.

        Args:
            output_dir: 출력 파일 디렉토리
            include_summary: 자동 요약 포함 여부
        """
        if self.df is None:
            self.load_data()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Converting to individual files in: {output_path}")

        for idx, row in self.df.iterrows():
            # cert_name에서 파일명 생성 (정제)
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
        """인증 데이터에 대한 통계를 가져옵니다."""
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
    """예제를 포함한 메인 실행 함수입니다."""

    # 변환기 초기화
    csv_path = "/Users/hoon/Desktop/SKN-17-Final-5Team/data/OCR 데이터/globalcerti_done.csv"
    converter = CertificationRAGConverter(csv_path)

    # 데이터 로드
    converter.load_data()

    # 통계 출력
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

    # 출력 디렉토리 생성
    output_dir = Path("/Users/hoon/Desktop/SKN-17-Final-5Team/retrieval_test/output")
    output_dir.mkdir(exist_ok=True)

    # 다양한 형식으로 변환
    print("Converting to multiple RAG-optimized formats...\n")

    # 1. JSON Lines 형식 (벡터 DB에 권장)
    converter.convert_to_jsonl(
        output_dir / "certifications.jsonl",
        include_summary=True
    )

    # 2. 단일 텍스트 파일
    converter.convert_to_text_format(
        output_dir / "certifications.txt",
        include_summary=True
    )

    # 3. 구조화된 JSON
    converter.convert_to_json(
        output_dir / "certifications.json",
        include_summary=True
    )

    # 4. 개별 텍스트 파일 (선택 사항 - 기본적으로 주석 처리됨)
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
