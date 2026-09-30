"""Reference graph: which references become edges, and that retrieval follows them exactly one hop."""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from structrag import links
from structrag.app import build_services
from structrag.config import Settings
from structrag.llm.fake import FakeLLM

FILLER = "The measurements were repeated under identical conditions and the results were averaged carefully. " * 3


class OverlapReranker:
    """Stand-in cross-encoder: shared words minus a constant, so 'relevant' means 'shares vocabulary'."""

    def scores(self, query: str, texts: list[str]) -> list[float]:
        q = set(query.lower().split())
        return [len(q & set(t.lower().split())) - 1.0 for t in texts]


def write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def edges(store, doc_id: int) -> list[tuple[str, str]]:
    ids = [c["id"] for c in store.doc_chunks(doc_id)]
    return sorted((l["kind"], l["label"]) for l in store.links_out(ids))


@pytest.fixture
def paper(tmp_path: Path) -> Path:
    return write(tmp_path, "paper.md", f"""# Loss study

## 3 Method

{FILLER} The training curve is plotted in Figure 2 and the ablation in Table 1, details in Section 4.

## 4 Results

{FILLER} Results are summarised here.

Figure 2: Training loss over epochs, the loss falls below 0.1 after 40 epochs.

Table 1: Ablation of the learning rate schedule.
""")


# ---- pure functions ------------------------------------------------------------------------------
def test_identifiers_are_canonical():
    text = "See arXiv:1706.03762v5, doi 10.1145/3292500.3330701. and Legge n. 241/1990, D.Lgs. 196/2003, " \
           "decreto legislativo 30 giugno 2003, n. 196, Regolamento (UE) 2016/679."
    ids = links.identifiers(text)
    assert {"arxiv:1706.03762", "doi:10.1145/3292500.3330701", "act:241/1990", "act:196/2003", "act:2016/679"} <= ids
    assert links.identifiers("reported 12/2020 revenue") == set()      # a month/year is not an act


def test_aliases_drop_site_suffix_and_generic_titles():
    assert "alan turing" in links.aliases("Alan Turing - Wikipedia", "turing.pdf")[0]
    names, proper = links.aliases("Python (programming language) - Wikipedia", "wiki_en_python.pdf")
    assert proper == {"Python"} and "python" not in names
    assert links.aliases("Slide 1", "deck.pptx") == (set(), set())
    assert links.aliases("attention", "chain_of_thought.pdf")[0] == {"chain of thought"} and links.aliases("x", "gan.pdf") == (set(), set())   # single words are too vague


# ---- edges inside one document -------------------------------------------------------------------
def test_figure_table_and_section_references_resolve_inside_a_document(services, paper):
    doc = services.ingestor.ingest_file(paper).doc_id
    found = edges(services.store, doc)
    assert ("figure", "Figure 2") in found and ("table", "Table 1") in found and ("section", "Section 4") in found
    chunks = {c["id"]: c["text"] for c in services.store.doc_chunks(doc)}
    targets = {chunks[l["dst_chunk"]] for c in chunks for l in services.store.links_out([c]) if l["kind"] == "figure"}
    assert any("Figure 2: Training loss" in t for t in targets)


def test_a_chunk_never_links_to_itself(services, tmp_path):
    doc = services.ingestor.ingest_file(write(tmp_path, "self.md", "# T\n\nFigure 1: loss curve, see Figure 1.\n")).doc_id
    assert edges(services.store, doc) == []


# ---- edges between documents ---------------------------------------------------------------------
def _cited_pair(tmp_path: Path):
    cited = write(tmp_path, "attention.md", f"# Attention Is All You Need\n\narXiv:1706.03762\n\n{FILLER} "
                  "The Transformer relies on multi-head attention with eight heads.\n")
    citing = write(tmp_path, "bert.md", f"# BERT Pretraining\n\n## Introduction\n\n{FILLER}\n\n## Model\n\n{FILLER}\n\n"
                   "## Related work\n\nWe build on Attention Is All You Need (Vaswani et al.) and on arXiv:1706.03762.\n")
    return cited, citing


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_document_citations_are_found_whichever_document_is_ingested_first(services, tmp_path, order):
    files = _cited_pair(tmp_path)
    ids = {}
    for i in order:
        ids[i] = services.ingestor.ingest_file(files[i]).doc_id
    out = services.store.links_out([c["id"] for c in services.store.doc_chunks(ids[1])])
    assert [(l["kind"], l["dst_doc"], l["dst_chunk"]) for l in out] == [("doc", ids[0], None)]
    assert services.store.links_out([c["id"] for c in services.store.doc_chunks(ids[0])]) == []


def test_an_identifier_claimed_by_two_first_pages_is_not_trusted(services, tmp_path):
    a = write(tmp_path, "a.md", "# Alpha\n\nPreprint arXiv:1706.03762.\n")
    b = write(tmp_path, "b.md", "# Beta\n\nWe extend arXiv:1706.03762 in this note.\n")
    ids = [services.ingestor.ingest_file(f).doc_id for f in (a, b)]
    assert services.store.graph()["edges"] == [] and len(set(ids)) == 2


def test_deleting_a_document_removes_its_edges(services, tmp_path):
    cited, citing = _cited_pair(tmp_path)
    a, b = (services.ingestor.ingest_file(f).doc_id for f in (cited, citing))
    services.store.delete_document(a)
    assert services.store.links_out([c["id"] for c in services.store.doc_chunks(b)]) == []
    assert services.store.graph()["edges"] == []


def test_graph_endpoint_rolls_edges_up_per_document_pair(services, tmp_path):
    cited, citing = _cited_pair(tmp_path)
    a, b = (services.ingestor.ingest_file(f).doc_id for f in (cited, citing))
    g = services.store.graph()
    assert [(e["src"], e["dst"], e["kind"]) for e in g["edges"]] == [(b, a, "doc")]


# ---- retrieval: one hop ---------------------------------------------------------------------------
def _services(tmp_path: Path, **over) -> "build_services":
    s = Settings(data_dir=tmp_path / "data", profiles_dir=tmp_path / "profiles", embedder="hash", rerank="off",
                 ocr="off", **over)
    sv = build_services(s, llm=FakeLLM(reply="Answer [1]"))
    sv.retriever.reranker = OverlapReranker()
    return sv


def test_a_hit_pulls_in_the_figure_it_cites_but_only_when_relevant(tmp_path, paper):
    sv = _services(tmp_path)
    sv.ingestor.ingest_file(paper)
    hits = sv.retriever.search("training curve plotted in Figure 2 ablation", k=1)
    linked = sv.retriever.follow_links("how fast does the training loss fall over epochs", hits)
    assert any("Figure 2: Training loss" in h.text and 'cited as "Figure 2"' in h.via for h in linked)
    strict = _services(tmp_path / "b", link_min_relevance=99.0)
    strict.ingestor.ingest_file(paper)
    assert strict.retriever.follow_links("training loss over epochs", strict.retriever.search("Figure 2", k=1)) == []


def test_cited_document_contributes_its_best_passage_for_the_question(tmp_path):
    sv = _services(tmp_path)
    cited, citing = _cited_pair(tmp_path)
    sv.ingestor.ingest_file(cited)
    sv.ingestor.ingest_file(citing)
    question = "how many attention heads does the Transformer use"
    hits = [h for h in sv.retriever.search("We build on Attention Is All You Need", k=3) if "BERT" in h.section_path]
    linked = sv.retriever.follow_links(question, hits)
    assert linked and "eight heads" in linked[0].text and linked[0].doc_title == "Attention Is All You Need"
    ctx = sv.retriever.build_context(hits, linked=linked)
    assert ctx[-1].via == 'cited as "Attention Is All You Need" in [1]' and ctx[-1].n == len(ctx)


def test_links_are_followed_one_hop_never_chained(tmp_path):
    sv = _services(tmp_path)
    for name, title, cites, fact in (("a", "Alpha Report", "Beta Handbook", "alpha fact"),
                                    ("b", "Beta Handbook", "Gamma Manual", "beta fact"),
                                    ("c", "Gamma Manual", None, "gamma fact")):
        tail = f" It relies on the {cites}." if cites else ""
        sv.ingestor.ingest_file(write(tmp_path, f"{name}.md", f"# {title}\n\n{FILLER} The {fact} is stated here.{tail}\n"))
    hits = sv.retriever.search("Alpha Report relies on", k=1)
    linked = sv.retriever.follow_links("beta fact gamma fact alpha fact", hits)
    assert {h.doc_title for h in linked} == {"Beta Handbook"}


def test_graph_off_means_no_links_and_no_edges(tmp_path, paper):
    sv = _services(tmp_path, graph=False)
    doc = sv.ingestor.ingest_file(paper).doc_id
    assert edges(sv.store, doc) == []
    assert sv.retriever.follow_links("Figure 2", sv.retriever.search("Figure 2", k=1)) == []


def test_linked_context_needs_a_shown_source_to_hang_on(tmp_path, paper):
    sv = _services(tmp_path)
    sv.ingestor.ingest_file(paper)
    hits = sv.retriever.search("training curve plotted in Figure 2", k=1)
    linked = sv.retriever.follow_links("training loss epochs", hits)
    assert linked and sv.retriever.build_context([], linked=linked) == []      # seed not in the context: dropped


# ---- review regressions --------------------------------------------------------------------------
def test_text_whose_lowercase_differs_from_the_alias_key_does_not_crash():
    d = links.Directory({}, {"mina ilia": 1}, {})
    assert d.mentions("Mina \u0130lia") == {}          # 'İ'.lower() is two characters: no key, no crash


def test_generic_names_do_not_become_aliases():
    assert links.aliases("Overview", "overview.pdf") == (set(), set())
    assert links.aliases("Report - Q3", "report_q3.pdf")[1] == set()
    assert "project plan" not in links.aliases("Plan", "project_plan.md")[0]
    assert links.aliases("PyTorch", "pytorch.pdf")[1] == {"PyTorch"}


def test_short_linked_passage_survives_when_the_main_context_is_full(tmp_path, paper):
    sv = _services(tmp_path)
    sv.ingestor.ingest_file(paper)
    hits = sv.retriever.search("training curve plotted in Figure 2", k=1)
    linked = sv.retriever.follow_links("training loss epochs", hits)
    assert linked
    main = sv.retriever.build_context(hits)
    short = dataclasses.replace(linked[0], text="Figure 2: Training loss over epochs.")
    room = len(short.text)                        # far below MIN_CONTEXT_CHARS: a caption is short, not unwanted
    assert sv.retriever._linked_context(main, hits, [short], room)


def test_a_failing_graph_never_fails_the_ingest(tmp_path, paper, monkeypatch):
    sv = _services(tmp_path)
    monkeypatch.setattr(links, "refresh", lambda *a, **k: 1 / 0)
    assert sv.ingestor.ingest_file(paper).status == "ingested"


# ---- navigation: references the user can follow ----------------------------------------------------
def test_neighbours_name_where_a_reference_leads_and_who_cites_a_passage(services, paper):
    doc = services.ingestor.ingest_file(paper).doc_id
    chunks = {c["id"]: c["text"] for c in services.store.doc_chunks(doc)}
    citing = next(i for i, t in chunks.items() if "plotted in Figure 2" in t)
    caption = next(i for i, t in chunks.items() if "Figure 2: Training loss" in t)
    out = services.store.neighbours([citing])["out"]
    fig = next(r for r in out if r["kind"] == "figure")
    assert (fig["label"], fig["chunk_id"], fig["doc_id"]) == ("Figure 2", caption, doc) and fig["doc"] and fig["loc"] is not None
    back = services.store.neighbours([caption])["in"]
    assert [(r["kind"], r["chunk_id"]) for r in back if r["kind"] == "figure"] == [("figure", citing)]
    assert services.store.neighbours([citing, caption])["out"] == [r for r in out if r["chunk_id"] not in (citing, caption)]


def test_neighbours_of_a_citing_document_point_at_the_cited_one(services, tmp_path):
    cited, citing = _cited_pair(tmp_path)
    a, b = (services.ingestor.ingest_file(f).doc_id for f in (cited, citing))
    out = services.store.neighbours([c["id"] for c in services.store.doc_chunks(b)])["out"]
    assert [(r["kind"], r["doc_id"], r["chunk_id"]) for r in out] == [("doc", a, None)]


def test_sources_carry_their_references_and_the_chunk_endpoint_opens_a_node(tmp_path, paper):
    from fastapi.testclient import TestClient
    from structrag.api import create_app
    sv = _services(tmp_path)
    doc = sv.ingestor.ingest_file(paper).doc_id
    client = TestClient(create_app(sv))
    sid = sv.store.create_session()
    sources = next(e["sources"] for e in sv.chat.ask(sid, "training curve plotted in Figure 2 ablation") if e["type"] == "sources")
    ref = next(r for s in sources for r in s["refs"]["out"] if r["kind"] == "figure")
    node = client.get(f"/api/chunks/{ref['chunk_id']}").json()
    assert "Figure 2: Training loss" in node["text"] and node["doc_id"] == doc and node["refs"]["in"]
    assert client.get("/api/chunks/999999").status_code == 404
