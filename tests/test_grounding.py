from dataclasses import replace

import pytest

from structrag import grounding as g
from structrag.app import build_services
from structrag.llm.client import LLMError
from structrag.llm.fake import FakeLLM
from structrag.retrieve import Retriever

PASSAGES = [
    "The Transformer encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers.",
    "We trained the base models for 100,000 steps or 12 hours. The big models were trained for 300,000 steps (3.5 days).",
]
MIN = 0.5


# ---- pure functions ---------------------------------------------------------------------------
def test_numbers_are_canonical():
    assert g.numbers("1,720,320 hours") == g.numbers("1720320 hours") == {"1720320"}
    assert g.numbers("about 1,5 percent") == {"1.5"} == g.numbers("about 1.50 percent")
    assert g.numbers("see [12] and 28.4 BLEU in 2017") == {"28.4", "2017"}      # citation marker is not a number


def test_supported_sentence_keeps_citation_and_gets_evidence():
    r = g.check_answer("The encoder has a stack of 6 identical layers [1].", PASSAGES, MIN)
    assert [c.cites for c in r.claims] == [(1,)] and not r.dropped
    assert "N = 6 identical layers" in r.claims[0].evidence
    assert r.render() == "The encoder has a stack of 6 identical layers [1]."


def test_unsupported_sentence_is_dropped_and_supported_one_survives():
    ans = "The encoder has 6 identical layers [1]. It was designed by aliens from Mars [1]."
    r = g.check_answer(ans, PASSAGES, MIN)
    assert len(r.claims) == 1 and r.dropped[0][1] == "not supported by the sources"


def test_invented_number_is_dropped_even_if_words_match():
    r = g.check_answer("The big models were trained for 500,000 steps [2].", PASSAGES, MIN)
    assert not r.claims and r.dropped[0][1] == "number not in the sources"


def test_wrong_citation_is_repaired_and_missing_citation_attributed():
    r = g.check_answer("The big models were trained for 300,000 steps [1]. The base models ran 100,000 steps.",
                       PASSAGES, MIN)
    assert [c.cites for c in r.claims] == [(2,), (2,)]


def test_out_of_range_citation_is_ignored():
    r = g.check_answer("The encoder stack has 6 identical layers [9].", PASSAGES, MIN)
    assert r.claims[0].cites == (1,)


def test_language_and_refusal_text():
    assert g.language_of("Quanti strati ha il modello?") == "it"
    assert g.language_of("How many layers does the model have?") == "en"
    assert g.refusal("Dove è nato Dante?").startswith("Non ho trovato")
    assert g.is_refusal("NO_ANSWER") and g.is_refusal(g.refusal("x y z")) and not g.is_refusal("6 layers [1]")


# ---- chat flow --------------------------------------------------------------------------------
@pytest.fixture()
def rag(settings, tmp_path):
    f = tmp_path / "spec.md"
    f.write_text("# Encoder\n\nThe Transformer encoder is composed of a stack of N = 6 identical layers.\n\n"
                 "# Training\n\nThe base models were trained for 100,000 steps or 12 hours.\n", encoding="utf-8")

    def make(reply, verdict=None, **overrides):
        llm = FakeLLM(chooser=lambda u, labels: verdict or {"A": 0.95, "B": 0.05}, reply=reply)
        sv = build_services(replace(settings, **overrides), llm=llm)
        sv.ingestor.ingest_file(f)
        return sv, llm

    return make


def ask(sv, q):
    sid = sv.store.create_session()
    events = list(sv.chat.ask(sid, q))
    return next(e for e in events if e["type"] == "final"), events


def test_grounded_answer_is_returned_with_citations_and_quotes(rag):
    sv, _ = rag("The encoder has a stack of 6 identical layers [1].")
    final, events = ask(sv, "How many layers does the encoder have?")
    assert final["answered"] and "[1]" in final["text"] and final["verdict"]["verifier"] == 0.95
    cite = final["citations"][0]
    assert cite["cited"] and "6 identical layers" in cite["evidence"] and cite["doc"]
    assert {e["type"] for e in events} >= {"status", "sources", "final", "token", "done"}


def test_model_saying_no_answer_becomes_a_refusal(rag):
    sv, _ = rag("NO_ANSWER")
    final, _ = ask(sv, "Who invented the telephone?")
    assert not final["answered"] and final["citations"] == [] and final["text"].startswith("I could not find")


def test_verifier_rejection_blocks_an_answer_from_outside_knowledge(rag):
    sv, _ = rag("The encoder has a stack of 6 identical layers [1].", verdict={"A": 0.1, "B": 0.9})
    final, _ = ask(sv, "How many layers does the encoder have?")
    assert not final["answered"] and final["verdict"]["reason"] == "verifier rejected the answer"


def test_unsupported_answer_is_refused_without_reaching_the_verifier(rag):
    sv, llm = rag("Humans landed on Mars in 2035 [1].")
    final, _ = ask(sv, "Have humans landed on Mars?")
    assert not final["answered"] and final["verdict"]["dropped"] and llm.choose_calls == []


def test_italian_question_gets_italian_refusal(rag):
    sv, _ = rag("NO_ANSWER")
    final, _ = ask(sv, "Quanti abitanti ha Roma?")
    assert final["text"].startswith("Non ho trovato")


def test_low_relevance_refuses_without_calling_the_model(rag):
    sv, llm = rag("The encoder has a stack of 6 identical layers [1].")

    class Cold:                      # a reranker that finds nothing relevant
        def scores(self, query, passages):
            return [-12.0] * len(passages)

    sv.retriever = Retriever(sv.settings, sv.store, sv.embedder, Cold())
    sv.chat.retriever = sv.retriever
    calls = []
    llm.chat = lambda *a, **k: calls.append(1) or "x"
    final, _ = ask(sv, "How many layers does the encoder have?")
    assert not final["answered"] and final["verdict"]["reason"] == "no relevant passage found" and not calls


def test_verifier_outage_never_lets_an_unverified_answer_through(rag):
    sv, llm = rag("The encoder has a stack of 6 identical layers [1].")
    llm.choose = lambda *a, **k: (_ for _ in ()).throw(LLMError("down"))
    final, _ = ask(sv, "How many layers does the encoder have?")
    assert not final["answered"]


def test_grounding_off_passes_the_model_answer_through(rag):
    sv, _ = rag("Anything the model says.", grounding="off")
    final, _ = ask(sv, "How many layers does the encoder have?")
    assert final["answered"] and "Anything the model says." in final["text"]


def test_image_endpoint_serves_thumbnails_and_rejects_paths(settings):
    from fastapi.testclient import TestClient
    from structrag.api import create_app
    sv = build_services(settings)
    images = settings.data_dir / "images"
    images.mkdir(parents=True)
    (images / "0123456789abcdef.jpg").write_bytes(b"\xff\xd8jpeg")
    (settings.data_dir / "secret.jpg").write_bytes(b"secret")
    client = TestClient(create_app(sv))
    assert client.get("/api/images/0123456789abcdef").content == b"\xff\xd8jpeg"
    assert client.get("/api/images/..%2Fsecret").status_code == 404
    assert client.get("/api/images/ffffffffffffffff").status_code == 404
    assert client.get("/api/images/secret").status_code == 404


def test_chat_endpoint_streams_status_final_and_verdict(rag):
    from fastapi.testclient import TestClient
    from structrag.api import create_app
    sv, _ = rag("The encoder has a stack of 6 identical layers [1].")
    client = TestClient(create_app(sv))
    sid = client.post("/api/sessions", json={}, headers={"X-Requested-With": "t"}).json()["id"]
    body = client.post("/api/chat", json={"session_id": sid, "message": "How many layers does the encoder have?"},
                       headers={"X-Requested-With": "t"}).text
    assert '"type": "status"' in body and '"type": "final"' in body and '"answered": true' in body
    msgs = client.get(f"/api/sessions/{sid}").json()["messages"]
    assert msgs[1]["sources"][0]["evidence"]
