import pandas as pd 
from types import SimpleNamespace
import json
from pathlib import Path
from typing import Dict, List
import csv


column_aliases = {
    "사례내용": "사례",
    "답변": "답변",
    "근거조항": "근거조항",
}

def row_to_nl_context(row):
    """행을 자연어 문맥으로 변환"""
    lines = []
    for col, label in column_aliases.items():
        value = str(row.get(col, "")).strip()
        if value and value.lower() != "nan":
            lines.append(f"{label}: {value}")
    return "".join(lines)




def main():

    # parse csv file 
    csv_file = "./used_data/무역클레임중재_RAG데이터.csv"
    csv_df = pd.read_csv(csv_file).fillna("").reset_index(drop=True)

    print(f"CSV 로드: {csv_df.shape[0]}행 x {csv_df.shape[1]}열")
    print(f"   컬럼: {list(csv_df.columns)}")

    csv_row_docs = []
    csv_field_docs = {col: [] for col in column_aliases}
    csv_cell_docs = []

    for idx, row in csv_df.iterrows():
        metadata = {"row_index": int(idx), "source": "csv_row"}
        for col in column_aliases:
            metadata[col] = str(row.get(col, ""))

        csv_row_docs.append(SimpleNamespace(
            page_content=row_to_nl_context(row),
            metadata=metadata
        ))

        for col, label in column_aliases.items():
            value = str(row.get(col, "")).strip()
            if not value or value.lower() == "nan":
                continue
            text = f"{label}: {value}"
            csv_field_docs[col].append(SimpleNamespace(
                page_content=text,
                metadata={"row_index": int(idx), "field": col, "label": label, "source": "csv_field"}
            ))
            csv_cell_docs.append(SimpleNamespace(
                page_content=text,
                metadata={"row_index": int(idx), "field": col, "label": label, "source": "csv_cell"}
            ))

    print(f"CSV-Row: {len(csv_row_docs)}개 문서")
    for col, docs in csv_field_docs.items():
        print(f"CSV-Field[{col}]: {len(docs)}개 문서")
    print(f"CSV-Cell: {len(csv_cell_docs)}개 문서")


    # convert csv into json
    BASE_DIR = Path(__file__).resolve().parent
    INPUT_CSV = BASE_DIR / "used_data"/ "무역클레임중재_RAG데이터.csv"
    OUTPUT_JSON = BASE_DIR / "used_data" / "사례_응답_근거조항.json"

    COLUMN_MAP = {
        "사례내용": "case",
        "답변": "answer",
        "근거조항": "clause",
    }

    def normalize_value(value: str) -> str:
        return (value or "").strip()


    def convert_csv_to_json(input_path: Path, output_path: Path) -> None:
        records: List[Dict[str, str]] = []

        with input_path.open("r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                record = {}
                for csv_col, key in COLUMN_MAP.items():
                    record[key] = normalize_value(row.get(csv_col, ""))
                record["text"] = "\n".join(
                    [
                        f"사례: {record['case']}" if record["case"] else "",
                        f"응답: {record['answer']}" if record["answer"] else "",
                        f"근거조항: {record['clause']}" if record["clause"] else "",
                    ]
                ).strip()
                record["metadata"] = {"document_name": "무역클레임중재QA", "row_index": idx}
                records.append(record)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        print(f"변환 완료: {output_path} (총 {len(records)}건)")


    convert_csv_to_json(INPUT_CSV, OUTPUT_JSON)




if __name__ == "__main__":
        main()
