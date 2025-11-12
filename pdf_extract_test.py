#!/usr/bin/env python3
"""Extract structured Q&A data from trade_claim.pdf using PyMuPDF."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Generator, Iterable, List, Sequence

import fitz  # PyMuPDF

fitz.TOOLS.mupdf_display_errors(False)


FOOTER_TOKENS = {
    "|무역클레임 및 중재 50문 50답",
    "무역클레임 및 중재 50문 50답",
    "수출입계약 관련 클레임",
    "I   수출입계약 관련 클레임",
    "I I   대금결제 관련 클레임",
    "I I I  무역운송 관련 클레임",
    "IV  전자무역 관련 클레임",
    "VI 중재와 클레임 제기절차",
    "대금결제 관련 클레임",
    "무역운송 관련 클레임",
    "전자무역 관련 클레임",
    "중재와 클레임 제기절차",
    "무역사기 관련 클레임",
    "Ⅰ",
    "Ⅱ",
    "Ⅲ",
    "Ⅳ",
    "Ⅴ",
    "Ⅵ",
}

SECTION_ALIASES = {
    "사례": "body",
    "TIP": "tips",
    "참고": "references",
    "용어": "glossary",
    "자료": "references",
    "관련법령": "references",
    "관련규정": "references",
}


@dataclass
class QuestionEntry:
    number: int
    title: str
    question: str
    body: str
    tips: List[str]
    references: List[str]
    glossary: List[str]
    pages: List[int]


def iter_clean_lines(doc: fitz.Document) -> Generator[tuple[int, str], None, None]:
    """Yield (page_no, text_line) pairs stripped of boilerplate."""
    for idx, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if not text:
            continue
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("MuPDF error:"):
                continue
            if line in FOOTER_TOKENS:
                continue
            if line.isdigit():
                continue
            yield idx, line


def split_question_blocks(
    lines: Iterable[tuple[int, str]],
) -> List[dict[str, object]]:
    """Group consecutive lines into question-sized blocks."""
    blocks: List[dict[str, object]] = []
    current: dict[str, object] | None = None
    for page_no, line in lines:
        if line == "Q":
            if current:
                blocks.append(current)
            current = {"lines": [], "pages": set()}
            current["pages"].add(page_no)
            continue
        if current is None:
            continue
        current["pages"].add(page_no)
        current["lines"].append((line, page_no))
    if current:
        blocks.append(current)
    return blocks


def structure_question(block: dict[str, object], number: int) -> QuestionEntry:
    """Convert a raw block into a QuestionEntry."""
    raw_lines: Sequence[tuple[str, int]] = block["lines"]  # type: ignore[index]
    if not raw_lines:
        raise ValueError("Encountered an empty question block.")
    title = raw_lines[0][0]
    sections: dict[str, list[str]] = {
        "question": [],
        "body": [],
        "tips": [],
        "references": [],
        "glossary": [],
    }
    current_key = "question"
    for text, _page in raw_lines[1:]:
        if text == "A":
            continue
        normalized = SECTION_ALIASES.get(text)
        if normalized:
            current_key = normalized
            continue
        sections.setdefault(current_key, []).append(text)
    question_text = " ".join(sections["question"]).strip()
    body_lines = sections["body"] or sections["question"]
    pages = sorted(block["pages"])  # type: ignore[arg-type]
    return QuestionEntry(
        number=number,
        title=title,
        question=question_text,
        body=" ".join(body_lines).strip(),
        tips=sections["tips"],
        references=sections["references"],
        glossary=sections["glossary"],
        pages=pages,
    )


def extract_questions(pdf_path: Path) -> list[QuestionEntry]:
    """High-level helper that returns structured questions for a PDF."""
    doc = fitz.open(pdf_path)
    try:
        raw_blocks = split_question_blocks(iter_clean_lines(doc))
        questions = [
            structure_question(block, idx + 1) for idx, block in enumerate(raw_blocks)
        ]
        return questions
    finally:
        doc.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Q&A data from trade_claim.pdf using PyMuPDF."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("data/raw_data/trade_claim.pdf"),
        help="Path to the source PDF file.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw_data/trade_claim_extracted.json"),
        help="Destination JSON path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Only keep the first N questions in the exported JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.pdf.exists():
        raise FileNotFoundError(f"PDF not found: {args.pdf}")
    questions = extract_questions(args.pdf)
    if args.limit is not None:
        questions = questions[: args.limit]
    payload = {
        "source": str(args.pdf),
        "question_count": len(questions),
        "questions": [asdict(q) for q in questions],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {payload['question_count']} questions to {args.out}")


if __name__ == "__main__":
    main()
