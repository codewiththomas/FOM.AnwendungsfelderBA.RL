# Quick Start Guid

1. Repo klonen
2. Virtuelle Umgebung erstellen

```bash
py -3.12 -m venv .venv
```

3. Aktivieren der virtuellen Umgebung

```bash
.\.venv\Scripts\activate
```

4. Abhängigkeiten installieren

```bash
pip install -r requirements.txt
```

5. Training starten

```bash
python src/train.py
```

## RAG für Traglasttabellen aus PDF

Zusätzlich gibt es nun ein separates RAG-Modul, um Traglasttabellen aus PDF-Dateien zu extrahieren und abfragbar zu machen:

```bash
pip install -r rag/requirements-rag.txt
python rag/traglast_rag.py ingest --pdf /pfad/zur/traglast.pdf --index-dir rag/index
# oder mehrere PDFs mit identischem Layout:
python rag/traglast_rag.py ingest --pdf-dir /pfad/zu/pdfs --index-dir rag/index
python rag/traglast_rag.py query --index-dir rag/index --question "Traglast bei 33,9 m Rüstlänge, 20° und 30 m Ausladung"
```

Mehr Details: `rag/README.md`
