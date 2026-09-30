# Benchmark

Real public documents, questions written from the source text with the evidence string next to each answer.
The documents are not committed (`docs*/` is git-ignored); download URLs are in the history of this folder's
`make_blind*_qa.py` docstrings and the README table below.

| set | files | questions | role |
|---|---|---|---|
| `docs/` + `qa.json` | 15 (arXiv PDFs, IT Wikipedia PDFs, 872-page DOCX, 369-slide PPTX, 4 XLSX) | 80 | development: parameters were tuned here |
| `docs_blind/` + `qa_blind.json` | 6 PDFs (BERT, GPT-3, Python, Photosynthesis, Fibonacci IT, Leonardo IT) | 61 | first blind run, then used to fix parser bugs: now a dev set |
| `docs_blind2/` + `qa_blind2.json` | 7 PDFs (ResNet, GAN, CoT, Galileo IT, Dante IT, Turing, Mars) | 61 | never tuned on: the reported number |

```bash
PYTHONPATH=src python eval/bench/run.py --tag t --docs docs_blind2 --qa qa_blind2.json \
  --embedder local --embed-model intfloat/multilingual-e5-small --rerank local --answer --llm http://127.0.0.1:8081/v1
```

Hit = one chunk contains **all** evidence strings. Answer = contains an accepted value; unanswerable questions
must be refused. Known grader limits: exact-substring matching (a correct paraphrase can score wrong) and a few
questions whose document text is itself ambiguous.
