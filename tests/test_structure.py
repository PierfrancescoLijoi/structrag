import logging
from dataclasses import replace

from conftest import heading_oracle
from structrag.chunking import chunk_section
from structrag.ir import PARA, TABLE, Block, ParsedDoc, Section
from structrag.llm.client import LLMError
from structrag.llm.fake import FakeLLM
from structrag.parsers import parse_file
from structrag.structure.heuristics import infer
from structrag.structure.profiles import ProfileStore
from structrag.structure.resolver import resolve_headings, resolve_table_headers
from structrag.structure.tree import build_tree

BODY = "This is a long body paragraph that explains something in enough detail to be text. " * 3


def _doc(*blocks: Block, fmt: str = "docx") -> ParsedDoc:
    return ParsedDoc("t", fmt, tuple(blocks))


def test_explicit_levels_win_with_high_confidence(md_file):
    inf = infer(parse_file(md_file))
    assert inf.strategy == "explicit" and inf.confidence >= 0.9 and not inf.ambiguous


def test_pdf_typography_ranks_font_sizes_into_levels(pdf_file):
    doc = parse_file(pdf_file)
    inf = infer(doc)
    got = {doc.blocks[i].text: lv for i, lv in enumerate(inf.levels) if lv}
    assert got["Chapter 1"] == 1 and got["Topic 1.1"] == 2 and not inf.ambiguous


def test_numbering_gives_depth_and_lists_are_not_headings():
    blocks = [Block("1. Introduction", PARA), Block(BODY), Block("1.1 Scope", PARA), Block(BODY),
              Block("2. Method", PARA), Block(BODY),
              Block("1. buy milk", PARA), Block("2. buy eggs", PARA), Block("3. buy tea", PARA)]
    inf = infer(_doc(*blocks))
    assert inf.levels[:6] == [1, None, 2, None, 1, None]
    assert inf.levels[6:] == [None, None, None]


def test_bold_only_candidates_are_ambiguous_and_the_agent_decides(docx_bold_only):
    doc = parse_file(docx_bold_only)
    inf = infer(doc)
    assert len(inf.ambiguous) == 3
    llm = FakeLLM(heading_oracle("Safety rules", "Emergency exits"))
    levels, decisions = resolve_headings(doc, inf, llm, _settings())
    kept = [doc.blocks[i].text for i, lv in enumerate(levels) if lv]
    assert kept == ["Safety rules", "Emergency exits"]          # 'Maintenance' rejected
    assert all(d.accepted for d in decisions)


def test_unreachable_model_is_logged_and_keeps_the_heuristic_structure(docx_bold_only, caplog):
    class Down:
        def choose(self, *a, **k):
            raise LLMError("connection refused")
    doc = parse_file(docx_bold_only)
    inf = infer(doc)
    with caplog.at_level(logging.WARNING, logger="structrag.structure.resolver"):
        levels, decisions = resolve_headings(doc, inf, Down(), _settings())
    assert decisions == [] and levels == inf.levels
    assert "connection refused" in caplog.text


def test_low_confidence_agent_answer_is_not_applied_and_stays_reviewable(docx_bold_only):
    doc = parse_file(docx_bold_only)
    inf = infer(doc)
    llm = FakeLLM(lambda u, l: {"A": 0.55, "B": 0.45})           # below the 0.70 accept threshold
    levels, decisions = resolve_headings(doc, inf, llm, _settings())
    assert not any(levels) and not any(d.accepted for d in decisions)


def test_no_headings_in_long_pdf_text_falls_back_to_shape_candidates():
    blocks = []
    for i in range(12):
        blocks += [Block(f"Section number {i}", PARA), Block(BODY * 4)]
    inf = infer(_doc(*blocks, fmt="pdf"))
    assert inf.strategy == "shape" and len(inf.ambiguous) == 12


def test_agent_fixes_ambiguous_spreadsheet_header(tmp_path):
    from openpyxl import Workbook
    wb = Workbook()
    ws = wb.active
    for row in [["Name", "Qty", "Price"], ["Widget", "many", "cheap"], ["Gadget", "few", "dear"]]:
        ws.append(row)
    wb.save(tmp_path / "t.xlsx")
    doc = parse_file(tmp_path / "t.xlsx")
    table = next(b for b in doc.blocks if b.kind == TABLE)
    assert table.meta["header_ambiguous"]        # text-only rows: heuristics cannot tell
    # mechanism test: the agent picks row 2 and the table is re-headed without losing rows
    llm = FakeLLM(lambda u, l: {"A": 0.05, "B": 0.9, "C": 0.05})
    fixed, decisions = resolve_table_headers(doc, llm, _settings())
    t2 = next(b for b in fixed.blocks if b.kind == TABLE)
    assert t2.meta["header_row"] == 1 and t2.meta["column_names"] == ["Widget", "many", "cheap"]
    assert len(t2.rows) == 3 and decisions[0].accepted


def test_tree_nests_by_level_and_keeps_preamble():
    blocks = (Block("pre"), Block("A"), Block("a-body"), Block("A1"), Block("deep"), Block("B"))
    root = build_tree("T", blocks, [None, 1, None, 2, None, 1])
    assert [b.text for b in root.blocks] == ["pre"]
    assert [c.title for c in root.children] == ["A", "B"]
    assert root.children[0].children[0].blocks[0].text == "deep"


def test_chunks_stay_inside_sections_and_tables_repeat_header(settings):
    rows = tuple((f"item{i}", str(i)) for i in range(45))
    table = Block("x", TABLE, rows=(("name", "n"), *rows), loc="Sheet!A1:B46",
                  meta={"header_row": 0})
    sec = Section("Inventory", 1, [Block(BODY), table])
    chunks = chunk_section(sec, ("Doc", "Inventory"), replace(settings, table_rows_per_chunk=20))
    tables = [c for c in chunks if c.kind == "table"]
    assert len(tables) == 3 and all(c.text.startswith("| name | n |") for c in tables)
    assert all(c.ctx.startswith("Doc > Inventory") for c in chunks)


def test_profile_needs_two_agreeing_votes_and_human_outweighs(tmp_path):
    store = ProfileStore(tmp_path)
    store.learn("docx", "11|0|Normal", [("11|1|Normal", 1)])
    assert store.get("docx", "11|0|Normal") == {}
    store.learn("docx", "11|0|Normal", [("11|1|Normal", 1)])
    assert store.get("docx", "11|0|Normal") == {"11|1|Normal": 1}
    store.learn("docx", "11|0|Normal", [("11|1|Normal", 0)], human=True)   # 3 votes for body
    assert store.get("docx", "11|0|Normal") == {}                           # 2 vs 3 => below 0.8


def _settings():
    from structrag.config import Settings
    return Settings()
