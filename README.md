# structrag

Local-first RAG that understands document structure. Parses PDF / DOCX / PPTX / XLSX / MD, finds headings
and tables (heuristics first, a small LLM only for ambiguous cases), chunks per section with breadcrumbs,
retrieves with hybrid search + a cross-encoder reranker, and answers with citations.
Everything runs on your machine: embeddings and reranking are open-source ONNX models on CPU
(no GPU, no server, no API key); the chat model is any OpenAI-compatible local server.

## Run

```bash
pip install -e .[local]          # add ,pii for PII masking
structrag serve                  # dashboard on http://127.0.0.1:8000, watches data/inbox
```

First run downloads the models once (~120 MB embeddings, ~120 MB reranker). Chat needs a local
OpenAI-compatible server (Ollama, llama.cpp `llama-server`, LM Studio): `STRUCTRAG_BASE_URL`,
`STRUCTRAG_CHAT_MODEL`. Without one, search still works; answers need it.

CLI: `ingest`, `scan` (= sync), `watch`, `ask`, `serve`, `docs`, `review`, `doctor`.

## Documents stay in sync

Drop files in `data/inbox` (or upload in the dashboard). The watcher (and **Sync inbox** / `structrag scan`)
detects:

| change | what happens |
|---|---|
| new file | parsed, chunked, embedded |
| edited file | rebuilt; only chunks whose text changed are re-embedded; the old version is removed only after the new one is stored (a broken edit never wipes the index) |
| deleted file | document, chunks and search index entries removed |
| renamed / moved file | detected by content hash, nothing re-processed |
| chunking / embedding / PII settings changed | affected documents rebuilt automatically |
| file still being copied | skipped until it settles |

## Models (all open source, CPU)

| role | default | notes |
|---|---|---|
| embeddings | `intfloat/multilingual-e5-small` (MIT, 384-d) | ~20 chunks/s on 6 threads; Italian + English |
| reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` int8 (Apache-2.0) | reranks the top 30 hybrid candidates, ~2 s per query |

Threads default to half the cores (`STRUCTRAG_LOCAL_THREADS`). Switch off the reranker with
`STRUCTRAG_RERANK=off`, or use `STRUCTRAG_EMBEDDER=hash` for a model-free lexical fallback.

## PII masking

[rizzo-pii](https://huggingface.co/rizzoaiacademy/rizzo-pii-0.3B) (MIT, CPU, Italian-first):
`pip install -e .[pii]` then `STRUCTRAG_PII_MODE=mask`. Identifiers (names, email, phone, address, IBAN, card,
CF, P.IVA, ID docs, plate) become `[FULLNAME_1]`-style placeholders before embedding and storage. Masking is
one-way; the review queue and the source file in the inbox stay unmasked.

## Measured quality

`eval/bench` runs real public documents (arXiv papers, Wikipedia in EN/IT, an 872-page DOCX handbook, a 369-slide
PPTX, financial XLSX). A retrieval *hit* means one retrieved chunk contains **all** evidence strings; an answer is
*correct* when it contains an accepted value (unanswerable questions must be refused). Chat model: Qwen3-4B Q5 on a
6 GB laptop GPU.

| set | hit@1 | hit@3 | hit@5 | hit@8 | doc hit | answer accuracy | retrieval median |
|---|---|---|---|---|---|---|---|
| dev (80 q, tuned on) | 86% | 96% | 97% | 99% | 100% | 96-98% | 2.0 s |
| **held-out (61 q, never tuned)** | 59% | 82% | 89% | 93% | 100% | **95.1%** (58/61) | 1.1 s |

On the held-out set, 2 of the 3 misses are grader/question defects (a correct "2015" scored wrong; one "unanswerable"
question the document does answer), so real accuracy is closer to 97%. What is **not** at 95%: strict chunk hit@5
on unseen documents (89%; 93% at k=8). Answers reach 95% because the model often has enough context even when a
single chunk does not hold every evidence string. Weak spots: PDF tables (no extraction yet), numbers in
running text that share a page with near-duplicates, rare Italian proper-noun questions.

## Test

```bash
pytest
```
