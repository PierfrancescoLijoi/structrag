# structrag

Local-first agentic RAG. Parses PDF/DOCX/PPTX/XLSX/MD, infers document structure with heuristics
(LLM logprob fallback only for ambiguous cases), chunks per section, and answers with citations.
Everything runs on your machine.

## Run

```bash
pip install -e .[dev]          # add ,pii for PII masking
structrag doctor               # checks the model server (Ollama by default)
structrag serve                # dashboard on http://127.0.0.1:8000
```

No model server? `STRUCTRAG_EMBEDDER=hash` gives offline lexical embeddings.

CLI: `ingest`, `scan`, `watch`, `ask`, `serve`, `docs`, `review`, `doctor`.

## Dashboard

Chat with citations, drag-and-drop upload, per-document scope, human review queue for uncertain
structure decisions, and an **Overview** panel (documents, chunks, status/format breakdown, masked PII).

## PII masking

[rizzo-pii](https://huggingface.co/rizzoaiacademy/rizzo-pii-0.3B) (MIT, CPU, Italian-first, 22 entity types):

```bash
pip install -e .[pii]
STRUCTRAG_PII_MODE=mask structrag serve
```

Identifiers (names, email, phone, address, IBAN, card, CF, P.IVA, catasto, ID docs, plate) are replaced
with placeholders like `[FULLNAME_1]` before embedding and storage. Dates, amounts, organisations and
cities stay readable. Masking is one-way. Existing documents are not re-masked: re-ingest them.
The human review queue keeps the exact heading/table-header text it asks about (overrides are keyed on it),
and the source file stays untouched in the inbox: treat both as unmasked.

## Settings (env, `STRUCTRAG_*`)

`BASE_URL`, `CHAT_MODEL`, `EMBED_MODEL`, `EMBEDDER` (`server`|`hash`), `DATA_DIR`, `PROFILES_DIR`,
`SUMMARY_MODE` (`lead`|`llm`), `PII_MODE` (`off`|`mask`), `PII_MODEL`.

## Test

```bash
pytest
```
