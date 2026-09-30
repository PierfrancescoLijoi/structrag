import time

from fastapi.testclient import TestClient

from conftest import heading_oracle, make_pdf
from structrag.api import create_app
from structrag.app import apply_review, build_services, scan_inbox
from structrag.llm.fake import FakeLLM

H = {"X-Requested-With": "test"}


def test_ingest_all_formats_and_skip_duplicates(services, md_file, docx_styled, pptx_file, xlsx_file, pdf_file):
    for f in (md_file, docx_styled, pptx_file, xlsx_file, pdf_file):
        r = services.ingestor.ingest_file(f)
        assert r.status == "ingested" and r.n_chunks > 0, f.name
    assert services.ingestor.ingest_file(md_file).status == "duplicate"
    assert len(services.store.list_documents()) == 5


def test_scanned_pdf_is_stored_as_needs_ocr_and_never_retrieved(services, scanned_pdf):
    r = services.ingestor.ingest_file(scanned_pdf)
    assert r.status == "needs_ocr"
    assert services.retriever.search("anything") == []


def test_retrieval_finds_the_right_section_across_documents(services, docx_styled, pptx_file, xlsx_file):
    for f in (docx_styled, pptx_file, xlsx_file):
        services.ingestor.ingest_file(f)
    hit = services.retriever.search("How much revenue was reached in 2025?")[0]
    assert "42 million" in hit.text and hit.section_path.endswith("Revenue")
    hit = services.retriever.search("Who is the Analyst and how old")[0]
    assert "Bob" in hit.text and hit.kind == "table"


def test_section_scoped_context_expands_small_sections(services, docx_styled):
    services.ingestor.ingest_file(docx_styled)
    ctx = services.retriever.build_context(services.retriever.search("revenue 2025"))
    assert any("Region" in c.text and "42 million" in c.text for c in ctx)   # whole section, table included


def test_agent_learns_layout_then_second_document_needs_no_llm(settings, tmp_path):
    from docx import Document
    from docx.shared import Pt

    def bold_doc(name: str, titles: list[str]):
        d = Document()
        for t in titles:
            d.add_paragraph().add_run(t).bold = True
            d.add_paragraph("Body text of the paragraph that follows the title. " * 6)
        d.save(tmp_path / name)
        return tmp_path / name

    llm = FakeLLM(heading_oracle("Alpha", "Beta", "Gamma", "Delta"))
    sv = build_services(settings, llm=llm)
    sv.ingestor.ingest_file(bold_doc("a.docx", ["Alpha", "Beta", "Gamma"]))
    calls_after_first = len(llm.choose_calls)
    assert calls_after_first == 3
    r = sv.ingestor.ingest_file(bold_doc("b.docx", ["Delta", "Alpha two", "Beta two"]))
    assert r.strategy == "profile" and len(llm.choose_calls) == calls_after_first   # zero new agent calls
    assert r.n_sections >= 3


def test_uncertain_agent_goes_to_review_and_human_answer_sticks(settings, tmp_path):
    from docx import Document
    d = Document()
    for t in ("Alpha", "Beta", "Gamma"):
        d.add_paragraph().add_run(t).bold = True
        d.add_paragraph("Body text of the paragraph that follows the title. " * 6)
    d.save(tmp_path / "u.docx")
    sv = build_services(settings, llm=FakeLLM(lambda u, l: {"A": 0.5, "B": 0.5}))
    r = sv.ingestor.ingest_file(tmp_path / "u.docx")
    assert r.queued_for_review == 3
    assert sv.store.list_documents()[0]["status"] == "review"
    for item in sv.store.pending_reviews():
        apply_review(sv, item["id"], "A")
        break
    titles = [s["title"] for s in sv.store.outline(sv.store.list_documents()[0]["id"])]
    assert "Alpha" in titles          # human decision applied via override + reingest


def test_inbox_scan_ingests_once(services):
    services.settings.inbox_dir.mkdir(parents=True, exist_ok=True)
    make_pdf(services.settings.inbox_dir / "d.pdf")
    assert scan_inbox(services)[0][1].status == "ingested"
    assert scan_inbox(services)[0][1].status == "duplicate"


def test_api_blocks_foreign_hosts_and_missing_csrf_header(services):
    client = TestClient(create_app(services))
    assert client.get("/api/docs", headers={"host": "evil.example"}).status_code == 403
    assert client.post("/api/sessions", json={}).status_code == 403
    assert client.post("/api/sessions", json={}, headers=H).status_code == 200


def test_api_upload_validates_type_and_chat_streams_with_sources(services, docx_styled):
    client = TestClient(create_app(services))
    bad = client.post("/api/upload", files={"files": ("evil.exe", b"MZ")}, headers=H)
    assert bad.status_code == 415
    ok = client.post("/api/upload", files={"files": ("../../x/report.docx", docx_styled.read_bytes())}, headers=H)
    assert ok.status_code == 200
    for _ in range(50):
        jobs = client.get("/api/jobs").json()
        if jobs and jobs[0]["state"] != "running":
            break
        time.sleep(0.1)
    assert jobs[0]["status"] == "ingested"
    assert all(".." not in p.name for p in services.settings.inbox_dir.iterdir())   # traversal neutralised
    sid = client.post("/api/sessions", json={}, headers=H).json()["id"]
    body = client.post("/api/chat", json={"session_id": sid, "message": "revenue in 2025?"}, headers=H).text
    assert '"type": "sources"' in body and '"type": "token"' in body and '"type": "done"' in body
    msgs = client.get(f"/api/sessions/{sid}").json()["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant"] and msgs[1]["sources"]


def test_pii_masking_replaces_identifiers_before_storage(services, tmp_path):
    def fake_detect(text):   # stands in for rizzo-pii: finds one name and one CF, plus a non-identifier
        found = []
        for label, value in (("FULLNAME", "Mario Rossi"), ("CF", "RSSMRA85M01H501Z"), ("DATE", "2025")):
            i = text.find(value)
            while i != -1:
                found.append({"entity_group": label, "start": i, "end": i + len(value), "score": 0.99})
                i = text.find(value, i + 1)
        return found

    f = tmp_path / "contratto.md"
    f.write_text("# Contratto\n\nMario Rossi, CF RSSMRA85M01H501Z, firma nel 2025. Mario Rossi accetta.\n",
                 encoding="utf-8")
    services.ingestor.pii_detector = fake_detect
    r = services.ingestor.ingest_file(f)
    assert r.status == "ingested" and r.pii == {"FULLNAME": 1, "CF": 1}
    assert services.store.list_documents()[0]["pii"] == {"FULLNAME": 1, "CF": 1}
    hit = services.retriever.search("contratto firma")[0]
    assert "Mario" not in hit.text and "RSSMRA" not in hit.text
    assert hit.text.count("[FULLNAME_1]") == 2 and "[CF_1]" in hit.text and "2025" in hit.text


def test_stats_endpoint_aggregates_documents_and_pii(services, md_file):
    services.ingestor.ingest_file(md_file)
    client = TestClient(create_app(services))
    s = client.get("/api/stats").json()
    assert s["documents"] == 1 and s["chunks"] > 0 and s["by_format"] == {"md": 1}
    assert s["pii"] == {"mode": "off", "model": services.settings.pii_model, "docs_with_pii": 0, "by_label": {}}
    assert client.get("/api/health").json()["pii_mode"] == "off"


def test_pii_ctx_and_text_stay_consistent(services, tmp_path):
    # detector that only fires when the value is preceded by a space: it would miss text-start values
    # if ctx and text were masked independently
    def detect(text):
        i = text.find("Anna Bianchi")
        return [{"entity_group": "FULLNAME", "start": i, "end": i + 12, "score": 0.9}] if i != -1 else []

    f = tmp_path / "note.md"
    f.write_text("# Note\n\nAnna Bianchi ha firmato il documento.\n", encoding="utf-8")
    services.ingestor.pii_detector = detect
    services.ingestor.ingest_file(f)
    rows = services.store.conn.execute("SELECT text, ctx FROM chunks").fetchall()
    assert rows and all("Anna" not in r["text"] and "Anna" not in r["ctx"] for r in rows)
    assert not services.store.conn.execute("SELECT 1 FROM chunk_fts WHERE chunk_fts MATCH 'Anna'").fetchall()
