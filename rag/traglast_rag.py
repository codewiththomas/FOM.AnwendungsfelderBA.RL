from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Chunk:
    text: str
    metadata: dict[str, Any]


def _normalize_cell(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _is_number(text: str) -> bool:
    return bool(re.match(r"^\d+(?:[.,]\d+)?$", text.strip()))


def _is_degree(text: str) -> bool:
    return bool(re.match(r"^\d+\s*°$", text.strip()))


def _to_float(text: str) -> float | None:
    cleaned = text.strip().replace(".", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _rectangularize(table: list[list[str]]) -> list[list[str]]:
    width = max((len(r) for r in table), default=0)
    return [r + [""] * (width - len(r)) for r in table]


def _find_data_start(table: list[list[str]]) -> int:
    for row_idx, row in enumerate(table):
        first = _normalize_cell(row[0]) if row else ""
        last = _normalize_cell(row[-1]) if row else ""
        if _is_number(first) or _is_number(last):
            return row_idx
    return len(table)


def _find_angle_and_length_rows(header_rows: list[list[str]]) -> tuple[int | None, int | None]:
    angle_idx = None
    length_idx = None

    max_angles = -1
    max_lengths = -1

    for idx, row in enumerate(header_rows):
        angle_count = sum(1 for c in row if _is_degree(_normalize_cell(c)))
        length_count = sum(1 for c in row if _is_number(_normalize_cell(c)) and "," in _normalize_cell(c))

        if angle_count > max_angles:
            max_angles = angle_count
            angle_idx = idx

        if length_count > max_lengths:
            max_lengths = length_count
            length_idx = idx

    if max_angles <= 0:
        angle_idx = None
    if max_lengths <= 0:
        length_idx = None

    return angle_idx, length_idx


def _extract_long_records_from_matrix(
    matrix: list[list[str]],
    pdf_name: str,
    page_idx: int,
    table_idx: int,
) -> list[dict[str, Any]]:
    table = _rectangularize([[_normalize_cell(c) for c in row] for row in matrix if any(_normalize_cell(c) for c in row)])
    if not table:
        return []

    data_start = _find_data_start(table)
    if data_start <= 0 or data_start >= len(table):
        return []

    header_rows = table[:data_start]
    data_rows = table[data_start:]

    angle_row_idx, length_row_idx = _find_angle_and_length_rows(header_rows)

    n_cols = len(table[0])
    left_m_col = 0
    right_m_col = n_cols - 1

    current_length: float | None = None
    col_meta: dict[int, dict[str, Any]] = {}

    for col in range(n_cols):
        if col in (left_m_col, right_m_col):
            continue

        length = None
        angle = None

        if length_row_idx is not None:
            length_text = _normalize_cell(header_rows[length_row_idx][col])
            length_val = _to_float(length_text) if _is_number(length_text) else None
            if length_val is not None:
                current_length = length_val

        if current_length is not None:
            length = current_length

        if angle_row_idx is not None:
            angle_text = _normalize_cell(header_rows[angle_row_idx][col]).replace(" ", "")
            if _is_degree(angle_text):
                angle = float(angle_text.replace("°", ""))

        if length is None and angle is None:
            continue

        col_meta[col] = {"boom_length_m": length, "angle_deg": angle}

    records: list[dict[str, Any]] = []
    for row in data_rows:
        radius = None
        left_radius = _to_float(_normalize_cell(row[left_m_col])) if left_m_col < len(row) else None
        right_radius = _to_float(_normalize_cell(row[right_m_col])) if right_m_col < len(row) else None

        if left_radius is not None:
            radius = left_radius
        elif right_radius is not None:
            radius = right_radius

        if radius is None:
            continue

        for col, meta in col_meta.items():
            value_text = _normalize_cell(row[col]) if col < len(row) else ""
            load_val = _to_float(value_text) if _is_number(value_text) else None
            if load_val is None:
                continue

            records.append(
                {
                    "pdf_name": pdf_name,
                    "page": page_idx,
                    "table": table_idx,
                    "radius_m": radius,
                    "boom_length_m": meta.get("boom_length_m"),
                    "angle_deg": meta.get("angle_deg"),
                    "load_t": load_val,
                }
            )

    return records


def _chunks_from_records(records: list[dict[str, Any]], source: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for idx, r in enumerate(records, start=1):
        text = (
            f"Dokument: {r['pdf_name']}; Seite: {r['page']}; Tabelle: {r['table']}; Datensatz: {idx}; "
            f"Ausladung_m: {r['radius_m']}; Ruestlaenge_m: {r.get('boom_length_m')}; "
            f"Winkel_grad: {r.get('angle_deg')}; Traglast_t: {r['load_t']}"
        )
        chunks.append(
            Chunk(
                text=text,
                metadata={
                    "source": source,
                    "pdf_name": r["pdf_name"],
                    "page": r["page"],
                    "table": r["table"],
                    "record": idx,
                    "radius_m": r["radius_m"],
                    "boom_length_m": r.get("boom_length_m"),
                    "angle_deg": r.get("angle_deg"),
                },
            )
        )
    return chunks


def extract_table_chunks(pdf_path: Path) -> tuple[list[Chunk], list[Any]]:
    import pandas as pd
    import pdfplumber

    chunks: list[Chunk] = []
    dataframes: list[Any] = []

    table_settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "intersection_tolerance": 5,
        "snap_tolerance": 3,
        "join_tolerance": 3,
    }

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables(table_settings=table_settings)

            if not tables:
                continue

            for table_idx, table in enumerate(tables, start=1):
                if not table:
                    continue

                records = _extract_long_records_from_matrix(table, pdf_path.name, page_idx, table_idx)
                if not records:
                    continue

                df = pd.DataFrame(records)
                dataframes.append(df)
                chunks.extend(_chunks_from_records(records, str(pdf_path)))

    return chunks, dataframes


def build_index(chunks: list[Chunk], output_dir: Path) -> None:
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer

    output_dir.mkdir(parents=True, exist_ok=True)

    if not chunks:
        raise ValueError("Keine Traglast-Datensätze gefunden. Prüfe das PDF-Layout.")

    corpus = [chunk.text for chunk in chunks]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(corpus)

    model_path = output_dir / "vectorizer.joblib"
    matrix_path = output_dir / "matrix.npz"
    metadata_path = output_dir / "chunks.json"

    joblib.dump(vectorizer, model_path)
    np.savez_compressed(
        matrix_path,
        data=matrix.data,
        indices=matrix.indices,
        indptr=matrix.indptr,
        shape=matrix.shape,
    )

    metadata = [{"text": c.text, "metadata": c.metadata} for c in chunks]
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_sparse_matrix(path: Path):
    from scipy.sparse import csr_matrix

    loaded = np.load(path)
    return csr_matrix((loaded["data"], loaded["indices"], loaded["indptr"]), shape=loaded["shape"])


def query_index(index_dir: Path, question: str, top_k: int = 5) -> list[dict[str, Any]]:
    import joblib

    vectorizer = joblib.load(index_dir / "vectorizer.joblib")
    matrix = _load_sparse_matrix(index_dir / "matrix.npz")
    chunks = json.loads((index_dir / "chunks.json").read_text(encoding="utf-8"))

    q_vec = vectorizer.transform([question])
    scores = (matrix @ q_vec.T).toarray().ravel()

    best_idx = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in best_idx:
        if scores[idx] <= 0:
            continue
        item = chunks[idx]
        results.append(
            {
                "score": float(scores[idx]),
                "text": item["text"],
                "metadata": item["metadata"],
            }
        )
    return results


def export_tables(dataframes: list[Any], output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, df in enumerate(dataframes, start=1):
        path = output_dir / f"{stem}_table_{i}_long.csv"
        df.to_csv(path, index=False)


def cmd_ingest(args: argparse.Namespace) -> None:
    output_dir = Path(args.index_dir)

    pdf_paths: list[Path] = []
    if args.pdf:
        pdf_paths.append(Path(args.pdf))
    if args.pdf_dir:
        pdf_paths.extend(sorted(Path(args.pdf_dir).glob("*.pdf")))

    if not pdf_paths:
        raise ValueError("Bitte --pdf oder --pdf-dir angeben.")

    all_chunks: list[Chunk] = []
    total_tables = 0
    for pdf_path in pdf_paths:
        chunks, dataframes = extract_table_chunks(pdf_path)
        all_chunks.extend(chunks)
        total_tables += len(dataframes)
        export_tables(dataframes, output_dir / "tables", pdf_path.stem)

    build_index(all_chunks, output_dir)

    print(f"✅ Index erstellt: {output_dir}")
    print(f"✅ Verarbeitete PDFs: {len(pdf_paths)}")
    print(f"✅ Extrahierte Traglast-Tabellen: {total_tables}")
    print(f"✅ Datensätze (Chunks): {len(all_chunks)}")


def cmd_query(args: argparse.Namespace) -> None:
    results = query_index(Path(args.index_dir), args.question, top_k=args.top_k)

    if not results:
        print("Keine passenden Treffer gefunden.")
        return

    print(f"Top-{len(results)} Treffer:")
    for i, result in enumerate(results, start=1):
        meta = result["metadata"]
        print(
            f"\n[{i}] score={result['score']:.4f} "
            f"pdf={meta.get('pdf_name')} seite={meta.get('page')} tabelle={meta.get('table')} "
            f"ausladung={meta.get('radius_m')}m ruestlaenge={meta.get('boom_length_m')}m winkel={meta.get('angle_deg')}°"
        )
        print(result["text"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG für Traglasttabellen (PDF, BKL/Liebherr-Layout)")
    sub = parser.add_subparsers(required=True)

    ingest = sub.add_parser("ingest", help="PDF einlesen, Traglastdaten extrahieren und Index bauen")
    ingest.add_argument("--pdf", help="Pfad zu einer PDF mit Traglasttabellen")
    ingest.add_argument("--pdf-dir", help="Ordner mit mehreren PDFs (alle gleiches Layout)")
    ingest.add_argument("--index-dir", default="rag/index", help="Ausgabeordner für den Index")
    ingest.set_defaults(func=cmd_ingest)

    query = sub.add_parser("query", help="Frage gegen den gebauten Index stellen")
    query.add_argument("--index-dir", default="rag/index", help="Ordner mit dem Index")
    query.add_argument("--question", required=True, help="Frage, z. B. Traglast bei 33,9m und 20°")
    query.add_argument("--top-k", type=int, default=5, help="Anzahl Treffer")
    query.set_defaults(func=cmd_query)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
