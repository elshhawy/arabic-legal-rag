# Arabic Legal RAG — Egyptian Civil Code Q&A

A retrieval-augmented question-answering service over the **Egyptian Civil Code**
(bilingual Arabic/English PDF). Ask a question in Arabic; get a grounded answer
plus the Civil Code articles it was retrieved from.

> Status: **Session 1 (foundation)**. Experiment tracking, data versioning, CI/CD,
> production serving, optimization and monitoring are planned for later sessions.

## Architecture

```
PDF  ->  extract.py  ->  civil_code.json  ->  validate.py
                                                  |
                         build_index.py  <--------+
                              |
                  Chroma vector index (1 article = 1 chunk)
                              |
   /ask  ->  embed query  ->  top-3 retrieval  ->  Gemini (grounded prompt)  ->  answer + sources
```

- **Extraction:** `pdfplumber`, two-column layout (English left, Arabic right),
  Arabic visual-order fix, Arabic-Indic digits to `int`, repealed articles flagged
  (`is_repealed: true`) instead of deleted.
- **Embeddings:** `intfloat/multilingual-e5-small` (with the required `query:` /
  `passage:` prefixes). The model name is stored in the index metadata and checked
  at API startup, so a mismatched index fails loudly.
- **Vector store:** ChromaDB (persistent, local).
- **LLM:** Google Gemini via `google-genai`; the model name comes from `.env`.
- **API:** FastAPI, JSON logs with a per-request id (`X-Request-ID` header).

## Repository structure

```
src/legalrag/   arabic.py, extract.py, validate.py, embeddings.py, llm.py, api.py, logging_config.py
scripts/        build_index.py, ask_demo.py, spot_check.py, embed_demo.py
tests/          unit tests (API, LLM client, Arabic helpers, validation rules, retrieval)
reports/        extraction_report.json
Dockerfile, docker-compose.yml, pyproject.toml
```

## Setup

Requirements: Python 3.10+, Docker (for the container), a Gemini API key.

```bash
pip install -e ".[dev]"
```

Create a `.env` file in the project root (never commit it):

```
GEMINI_API_KEY=your-key
GEMINI_MODEL=your-model-name
# optional: LOG_LEVEL=INFO
```

### Build the data (first time only)

`data/` is git-ignored (data versioning with DVC is planned for Session 2), so the
corpus and index are built locally. Place the Civil Code PDF at
`data/egyptian_civil_code.pdf`, then:

```bash
python src/legalrag/extract.py        # PDF -> data/processed/civil_code.json
python -m legalrag.validate           # must pass before indexing
python scripts/build_index.py         # embeddings -> data/chroma_index
```

### Run with Docker

```bash
docker compose build
docker compose up -d
python scripts/ask_demo.py
```

The image bundles the corpus, the index and the embedding model; the API key is
injected at run time from `.env` and is never copied into the image.

### Run without Docker

```bash
uvicorn legalrag.api:app --port 8000
```

## API

### `GET /health`
```json
{"status": "healthy", "documents_indexed": 1093}
```
`documents_indexed` is the number of non-repealed articles in the vector index
(1149 articles - 56 repealed = 1093, as measured from the running container).

### `POST /ask`
Request:
```json
{"question": "هل يجوز فسخ العقد؟"}
```
Response:
```json
{
  "answer": "...",
  "sources": [
    "Egyptian Civil Code, Article 157",
    "Egyptian Civil Code, Article 650"
  ]
}
```
Errors: `422` for an invalid body (missing or non-string `question`, or an empty /
whitespace-only question); `503` if answer generation fails (no internal
details are returned).

## Tests

```bash
pytest
```

API tests are skipped automatically if `data/chroma_index` does not exist.

## Known limitations (Session 1)

- **No "I don't know" enforcement in code.** The LLM is instructed (system prompt)
  to refuse when the retrieved articles are insufficient, but retrieval always
  returns the top-3 articles, and `sources` lists everything retrieved, not only
  what the answer used. A distance threshold needs measurement first.
- **Repealed articles are not indexed.** They are flagged in the corpus but excluded
  from retrieval, so the system cannot yet explain that an article was repealed.
- **Source PDF typos remain.** Scattered Arabic typos in the PDF are left as-is
  (only one verified typo was corrected); see `reports/extraction_report.json`.
- **Not reproducible from a fresh clone yet:** the corpus JSON is committed, but the source PDF and the Chroma index are not in
  Git (DVC planned for Session 2).
- **No retrieval-quality or latency measurements yet.** RAGAS: not measured yet.
  Latency/throughput: not measured yet.
- Docker rebuilds reinstall all dependencies whenever `src/` changes (the dependency
  layer is not cached separately); to be addressed with the CI image build.
- Importing `legalrag.embeddings` loads `sentence-transformers` eagerly, which makes
  some tests slow (measured: ~14 s for 2 tests).

## Changelog

### Session 1 — Foundation
- **Built:** PDF extraction to structured JSON (1149 articles, 56 flagged repealed),
  corpus validation (7 checks), article-level embeddings in Chroma, `/ask` and
  `/health`, structured logging, Dockerfile and compose file, unit tests.
- **Decisions:** one article = one chunk; e5-small multilingual embeddings; Gemini
  as the generator; index stores the embedding model name and the API verifies it.
- **Results:** container builds and serves; `/health` returns ok; `/ask` returns
  an Arabic answer with cited articles for a sample question. Quality: not measured yet.
- **Limitations:** see above.