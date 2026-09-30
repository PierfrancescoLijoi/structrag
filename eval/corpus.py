"""Synthetic evaluation corpus with known ground truth.

Every section is titled "<Device> <ID>" and states the same fact template with different values,
so a chunk is only answerable if the retriever knows which *section* it came from. That is exactly
what structure-aware chunking (breadcrumbs) is supposed to buy over flat chunking.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from docx import Document
from docx.shared import Pt
from pptx import Presentation
from openpyxl import Workbook
from openpyxl.styles import Font
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

DEVICES = ["Boiler", "Compressor", "Turbine", "Pump", "Chiller", "Furnace", "Reactor", "Mixer",
           "Conveyor", "Dryer", "Filter", "Generator"]
FILLER = [
    "Operators must log every inspection in the maintenance register before the end of the shift.",
    "The supplier guarantees replacement parts for a period of ten years after the purchase date.",
    "Noise levels were measured according to the standard procedure and remained within limits.",
    "Training sessions are scheduled quarterly and attendance is mandatory for all technicians.",
    "Any deviation from the documented procedure must be reported to the plant supervisor.",
]


@dataclass(frozen=True)
class Section:
    title: str
    facts: dict[str, int]      # attribute -> value
    paragraphs: list[str]


@dataclass(frozen=True)
class Question:
    text: str
    section_title: str
    answer: str


ATTRS = {"maximum pressure": "bar", "rated power": "kW", "service interval": "hours"}


def make_sections(n: int = 12, seed: int = 7) -> list[Section]:
    rng = random.Random(seed)
    out = []
    for i in range(n):
        title = f"{DEVICES[i % len(DEVICES)]} {chr(65 + i % 26)}{rng.randint(1, 9)}"
        facts = {a: rng.randint(3, 990) for a in ATTRS}
        sentences = [f"The {a} of this unit is {v} {ATTRS[a]}." for a, v in facts.items()]
        paras = [" ".join(sentences)] + [" ".join(rng.sample(FILLER, 3)) for _ in range(2)]
        out.append(Section(title, facts, paras))
    return out


def make_questions(sections: list[Section]) -> list[Question]:
    qs = []
    for s in sections:
        for attr, value in s.facts.items():
            qs.append(Question(f"What is the {attr} of the {s.title}?", s.title, f"{value} {ATTRS[attr]}"))
    return qs


# ---- layouts -------------------------------------------------------------------------------
def md(path: Path, secs: list[Section]) -> Path:
    path.write_text("# Plant Manual\n\n" + "\n\n".join(
        f"## {s.title}\n\n" + "\n\n".join(s.paragraphs) for s in secs), encoding="utf-8")
    return path


def flat_md(path: Path, secs: list[Section]) -> Path:
    """Same words, no structure at all: the baseline a structure-blind pipeline sees."""
    path.write_text("\n\n".join(f"{s.title}\n\n" + "\n\n".join(s.paragraphs) for s in secs), encoding="utf-8")
    return path


def docx_styled(path: Path, secs: list[Section]) -> Path:
    d = Document()
    d.add_heading("Plant Manual", 0)
    for s in secs:
        d.add_heading(s.title, 1)
        for p in s.paragraphs:
            d.add_paragraph(p)
    d.save(path)
    return path


def docx_bold(path: Path, secs: list[Section]) -> Path:
    d = Document()
    for s in secs:
        d.add_paragraph().add_run(s.title).bold = True
        for p in s.paragraphs:
            d.add_paragraph(p)
    d.save(path)
    return path


def docx_numbered(path: Path, secs: list[Section]) -> Path:
    d = Document()
    for i, s in enumerate(secs, 1):
        d.add_paragraph(f"{i}. {s.title}")
        for p in s.paragraphs:
            d.add_paragraph(p)
    d.save(path)
    return path


def docx_caps(path: Path, secs: list[Section]) -> Path:
    d = Document()
    for s in secs:
        d.add_paragraph(s.title.upper())
        for p in s.paragraphs:
            d.add_paragraph(p)
    d.save(path)
    return path


def _pdf(path: Path, secs: list[Section], title_font: str, title_size: int, numbered: bool = False) -> Path:
    c = canvas.Canvas(str(path), pagesize=A4)
    w, h = A4
    y, page = h - 60, 1

    def newpage():
        nonlocal y, page
        c.setFont("Helvetica", 8)
        c.drawString(w / 2, 20, f"{page}")
        c.showPage()
        y, page = h - 60, page + 1

    for i, s in enumerate(secs, 1):
        if y < 220:
            newpage()
        c.setFont(title_font, title_size)
        c.drawString(50, y, f"{i}. {s.title}" if numbered else s.title)
        y -= 26
        c.setFont("Helvetica", 10)
        for p in s.paragraphs:
            words, line = p.split(), ""
            for word in words:
                if c.stringWidth(f"{line} {word}", "Helvetica", 10) > w - 100:
                    c.drawString(50, y, line)
                    y, line = y - 13, word
                else:
                    line = f"{line} {word}".strip()
            c.drawString(50, y, line)
            y -= 22
        y -= 6
    newpage()
    c.save()
    return path


def pdf_sized(path, secs):
    return _pdf(path, secs, "Helvetica-Bold", 16)


def pdf_bold_only(path, secs):
    return _pdf(path, secs, "Helvetica-Bold", 10)


def pdf_numbered(path, secs):
    return _pdf(path, secs, "Helvetica", 10, numbered=True)


def pptx_deck(path: Path, secs: list[Section]) -> Path:
    prs = Presentation()
    for s in secs:
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = s.title
        slide.placeholders[1].text = s.paragraphs[0]
    prs.save(path)
    return path


def xlsx_book(path: Path, secs: list[Section]) -> Path:
    wb = Workbook()
    ws = wb.active
    ws.title = "Units"
    ws.append(["Plant equipment"])
    ws.append(["Unit", *ATTRS])
    for s in secs:
        ws.append([s.title, *s.facts.values()])
    for cell in ws[2]:
        cell.font = Font(bold=True)
    wb.save(path)
    return path


# layout name -> (builder, gold headings are the section titles, expected explicit structure?)
LAYOUTS = {
    "md": md, "docx_styled": docx_styled, "docx_bold": docx_bold, "docx_numbered": docx_numbered,
    "docx_caps": docx_caps, "pdf_sized": pdf_sized, "pdf_bold_only": pdf_bold_only,
    "pdf_numbered": pdf_numbered, "pptx": pptx_deck,
}
