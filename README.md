# Sicherheitslotse

German occupational-safety Q&A with citations and measured quality. Work in progress.

```bash
uv sync
uv run python scripts/download.py
uv run python scripts/ingest.py
uv run python scripts/build_index.py
uv run python eval/run_retrieval.py
uv run pytest
```

`build_index.py` takes about 10 minutes on a laptop CPU. `run_retrieval.py --rerankers bge small`
also scores the reranked setups, which takes about 15 minutes on a CPU.

## PDFs

BAuA's site blocks scripts, so the ASR PDFs are saved by hand into `data/raw/pdf`. `download.py` names
any that are missing and records each file's hash in the manifest. Use the full texts: a "Textauszug"
change sheet prints deleted wording as if it were current, and `ingest.py` refuses one, as it does a
file whose version differs from `lotse.config`.

Docling reads each PDF once and `data/parsed` keeps the result, which takes about 20 minutes for five
PDFs on a CPU. After parsing a new PDF, run:

```bash
uv run python scripts/check_pdfs.py
```

It compares the PDF's own text layer with the final sections. Figure captions, drawing labels and
Literaturhinweise are meant to be missing. Anything else is a parsing bug.
