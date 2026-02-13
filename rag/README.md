# RAG für Traglasttabellen (PDF)

Dieses Modul ist auf das gezeigte BKL/Liebherr-Layout ausgelegt (gleichbleibende Seitenstruktur). Es extrahiert Traglastwerte in ein **strukturiertes Long-Format**:

- `radius_m` (Ausladung)
- `boom_length_m` (Rüstlänge)
- `angle_deg` (Winkel)
- `load_t` (Traglast)

## Installation

```bash
pip install -r rag/requirements-rag.txt
```

## 1) Ingest einer PDF

```bash
python rag/traglast_rag.py ingest --pdf /pfad/zur/traglast.pdf --index-dir rag/index
```

## 1b) Ingest eines ganzen PDF-Ordners

```bash
python rag/traglast_rag.py ingest --pdf-dir /pfad/zu/traglast-pdfs --index-dir rag/index
```

Ergebnis:
- `rag/index/chunks.json` (strukturierte Datensätze + Metadaten je Punkt)
- `rag/index/vectorizer.joblib` und `rag/index/matrix.npz` (Retrieval-Index)
- `rag/index/tables/*_long.csv` (extrahierte Traglastdaten)

## 2) Fragen stellen

```bash
python rag/traglast_rag.py query --index-dir rag/index --question "Traglast bei 33,9 m Rüstlänge, 20° und 30 m Ausladung" --top-k 5
```

## Hinweise

- Parser ist für gleich aufgebaute PDFs gedacht (wie von dir gezeigt).
- Falls ein Herstellerlayout abweicht, kann ich dir eine zweite Layout-Regel ergänzen.
