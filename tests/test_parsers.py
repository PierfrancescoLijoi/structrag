from structrag.ir import LIST_ITEM, TABLE
from structrag.parsers import UnsupportedFormat, parse_file
import pytest


def test_markdown_headings_code_fence_table_and_setext(md_file):
    doc = parse_file(md_file)
    heads = [(b.text, b.level_hint) for b in doc.blocks if b.level_hint]
    assert heads == [("Guide", 1), ("Install", 2), ("Setext Title", 2)]
    assert not any("not a heading" == b.text for b in doc.blocks if b.level_hint)
    assert any(b.kind == LIST_ITEM for b in doc.blocks)
    table = next(b for b in doc.blocks if b.kind == TABLE)
    assert table.rows == (("a", "b"), ("1", "2"))


def test_docx_styles_become_explicit_levels_and_tables_are_kept(docx_styled):
    doc = parse_file(docx_styled)
    assert [(b.text, b.level_hint) for b in doc.blocks if b.level_hint] == [
        ("Annual Report", 1), ("Finance", 1), ("Revenue", 2)]
    assert any(b.kind == TABLE and b.rows[1] == ("North", "10") for b in doc.blocks)


def test_pptx_slide_titles_and_notes(pptx_file):
    doc = parse_file(pptx_file)
    titles = [b.text for b in doc.blocks if b.level_hint == 1]
    assert titles == ["Roadmap", "Risks"]
    assert any(b.meta.get("notes") and "budget" in b.text for b in doc.blocks)
    assert next(b for b in doc.blocks if "Ship v1" in b.text).loc == "slide 1"


def test_xlsx_splits_regions_and_detects_headers(xlsx_file):
    doc = parse_file(xlsx_file)
    tables = [b for b in doc.blocks if b.kind == TABLE]
    assert len(tables) == 2
    first = tables[0]
    assert first.meta["column_names"] == ["Region", "Q1", "Q2"]
    assert first.meta["range"] == "Sales!A2:C4"
    assert [b.text for b in doc.blocks if b.level_hint == 2] == ["Sales 2025", "Staff"]


def test_pdf_strips_boilerplate_and_reads_typography(pdf_file):
    doc = parse_file(pdf_file)
    texts = [b.text for b in doc.blocks]
    assert not any("Confidential" in t or t.startswith("Page ") for t in texts)
    chapter = next(b for b in doc.blocks if b.text == "Chapter 1")
    body = next(b for b in doc.blocks if b.text.startswith("Body line 0"))
    assert chapter.size > body.size and chapter.bold and not body.bold


def test_scanned_pdf_is_flagged_for_ocr(scanned_pdf):
    doc = parse_file(scanned_pdf)
    assert doc.needs_ocr and not doc.blocks


def test_unsupported_and_fake_office_files_are_rejected(tmp_path):
    bad = tmp_path / "x.docx"
    bad.write_text("not a zip")
    with pytest.raises(UnsupportedFormat):
        parse_file(bad)
    with pytest.raises(UnsupportedFormat):
        parse_file(tmp_path / "x.doc")


def test_docx_ragged_table_does_not_crash(tmp_path):
    from docx import Document
    from docx.oxml import parse_xml
    from docx.oxml.ns import nsdecls, qn
    from structrag.parsers import parse_file
    from structrag.config import Settings

    d = Document()
    t = d.add_table(rows=3, cols=3)
    for r, row in enumerate(t.rows):
        for c, cell in enumerate(row.cells):
            cell.text = f"r{r}c{c}"
    row = t.rows[1]._tr
    row.get_or_add_trPr().append(parse_xml(f'<w:gridBefore {nsdecls("w")} w:val="1"/>'))
    row.remove(row.findall(qn("w:tc"))[0])           # a row that starts late: python-docx `row.cells` raises here
    d.save(tmp_path / "ragged.docx")
    doc = parse_file(tmp_path / "ragged.docx", Settings())
    table = next(b for b in doc.blocks if b.kind == "table")
    assert table.rows[0] == ("r0c0", "r0c1", "r0c2") and table.rows[1] == ("r1c1", "r1c2")
