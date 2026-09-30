"""Reference graph between chunks. Only explicit references become edges; retrieval follows them one hop.

Edges (no model involved):
  figure / table  "see Figure 3", "Tabella 2"        -> chunk captioned so (+ figures on the same page)
  section         "Section 3.2", "§4", "art. 5"      -> first chunk of the section numbered so
  doc             another document is cited by name (title, file name) or identifier (arXiv id, DOI,
                  act number "n. 196/2003")            -> no target chunk: retrieval picks that document's
                  best passage for the question, so the graph never has to guess which part is relevant
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePath

from .store import Store
from .textutil import STOPWORDS

MIN_ALIAS = 4
GENERIC_TITLE = re.compile(r"^(chapter|capitolo|slide|heading|sheet|foglio|untitled|document|documento|sezione|"
                           r"section|table|tabella|figure|figura)\b", re.I)
SITE_SUFFIX = re.compile(r"\s+[-–—|]\s+(?:wikipedia|wikipédia|treccani|britannica|arxiv)\b.*$", re.I)   # "Alan Turing - Wikipedia"
PARENS = re.compile(r"\s*\(([^)]*)\)")

_NAMES = r"(fig(?:ure|ura)?|tab(?:le|ella)?)"
XREF = re.compile(rf"\b{_NAMES}\.?\s*(\d{{1,3}}[a-z]?)\b", re.I)
CAPTION = re.compile(rf"^\W*{_NAMES}\.?\s*(\d{{1,3}}[a-z]?)\b", re.I | re.M)
_SEC_WORDS = (r"sections?|sec\.|sezion[ei]|paragraph|paragrafo|chapter|capitolo|cap\.|appendix|appendice|allegato|annex|"
              r"art(?:icolo|icle|\.)?")
_NUM = r"\d{1,3}(?:\.\d{1,3}){0,3}"
SEC_REF = re.compile(rf"(?:\b({_SEC_WORDS})|(§))\s*({_NUM}|(?-i:[A-Z]))(?!\w)", re.I)
SEC_HEAD = re.compile(rf"^\s*(?:({_SEC_WORDS})\s*({_NUM}|(?-i:[A-Z]))|()({_NUM}))(?:\s|[.:)]|$)", re.I)
ARXIV = re.compile(r"arxiv[:\s]*(\d{4}\.\d{4,5})", re.I)
DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+?)(?=[.,;)\]]*(?:\s|$))", re.I)
ACT = re.compile(r"(?:\b(?:legge|l\.|d\.?\s?lgs\.?|decreto|d\.?p\.?r\.?|regolamento|direttiva|reg\.|dir\.|n\.|n°|nr\.?|no\.)\s*"
                 r"(?:\((?:ue|ce|cee|eu|ec)\)\s*)?(?:n\.?\s*)?(\d{1,4}\s*/\s*\d{2,4}))|"
                 r"(?:\b\d{1,2}\s+[a-zà-ù]+\s+((?:19|20)\d{2}),?\s*n\.?\s*(\d{1,4}))", re.I)


@dataclass(frozen=True)
class Link:
    src: int
    dst_doc: int
    dst_chunk: int | None
    kind: str      # figure | table | section | doc
    label: str     # the text of the reference, shown to the user and the model


def _kind_of(word: str) -> str:
    return "figure" if word.lower().startswith("fig") else "table"


def _family(word: str | None) -> str:
    w = (word or "sec").lower()
    if w.startswith("art"):
        return "art"
    if w.startswith(("app", "all", "ann")):
        return "app"
    return "sec"


def identifiers(text: str) -> set[str]:
    """Canonical identifiers of published works: arXiv ids, DOIs, act numbers (n/year)."""
    out = {f"arxiv:{m.group(1)}" for m in ARXIV.finditer(text)}
    out |= {f"doi:{m.group(1).lower().rstrip('.')}" for m in DOI.finditer(text)}
    for m in ACT.finditer(text):
        num = m.group(1).replace(" ", "") if m.group(1) else f"{m.group(3)}/{m.group(2)}"
        out.add(f"act:{num}")
    return out


def aliases(title: str, path: str) -> tuple[set[str], set[str]]:
    """(case-insensitive names, exact-case single words) under which other documents may cite this one."""
    names: set[str] = set()
    proper: set[str] = set()

    def offer(raw: str) -> None:
        raw = " ".join(raw.split())
        if len(raw) < MIN_ALIAS or GENERIC_TITLE.match(raw) or "�" in raw or not any(c.isalpha() for c in raw):
            return
        if len(raw.split()) > 1:
            names.add(raw.lower())
        elif raw[0].isupper() and raw.lower() not in STOPWORDS and (qualified or raw != raw.capitalize()):
            proper.add(raw)      # a plain capitalised word ("Overview") starts sentences everywhere: not a name

    base = SITE_SUFFIX.sub("", title)
    qualified = base != title or "(" in title      # "Python (programming language)", "Roma - Wikipedia": a name
    offer(PARENS.sub("", base))
    for inner in PARENS.findall(base):
        if any(c.isupper() for c in inner):
            offer(inner)
    words = [w for w in re.split(r"[_\-\s]+", PurePath(path).stem) if w]
    if len(words) >= 3:       # "project_plan" would match any text that says "project plan"
        names.add(" ".join(words).lower())
    return names, proper


class Directory:
    """Who can be cited: identifiers and names of a set of documents, matched against arbitrary text."""

    def __init__(self, ids: dict[str, int], names: dict[str, int], proper: dict[str, int]):
        self.ids, self.names, self.proper = ids, names, proper
        self._names = self._compile(names, re.I)
        self._proper = self._compile(proper, 0)

    @classmethod
    def build(cls, docs: list[tuple[int, str, str, str]]) -> "Directory":
        """docs: (doc_id, title, path, head text). An identifier on the first page is the document's own,
        unless several first pages carry it (then one of them is just citing the other): those are dropped."""
        claims: dict[str, set[int]] = {}
        names: dict[str, int] = {}
        proper: dict[str, int] = {}
        for doc_id, title, path, head in docs:
            for ident in identifiers(f"{title}\n{head}"):
                claims.setdefault(ident, set()).add(doc_id)
            n, p = aliases(title, path)
            names.update({a: names.get(a, doc_id) for a in n})
            proper.update({a: proper.get(a, doc_id) for a in p})
        return cls({i: next(iter(d)) for i, d in claims.items() if len(d) == 1}, names, proper)

    def subset(self, keep) -> "Directory":
        """Only the documents for which keep(doc_id) is true."""
        pick = lambda table: {k: d for k, d in table.items() if keep(d)}
        return Directory(pick(self.ids), pick(self.names), pick(self.proper))

    @staticmethod
    def _compile(keys: dict[str, int], flags):
        if not keys:
            return None
        ordered = sorted(keys, key=len, reverse=True)
        return re.compile(r"(?<!\w)(" + "|".join(re.escape(k).replace(r"\ ", r"\s+") for k in ordered) + r")(?!\w)", flags)

    def __bool__(self) -> bool:
        return bool(self.ids or self.names or self.proper)

    def phrases(self) -> list[str]:
        """Search strings for the FTS prefilter (a full scan would be quadratic in corpus size)."""
        return [i.split(":", 1)[1] for i in self.ids] + list(self.names) + list(self.proper)

    def mentions(self, text: str) -> dict[int, str]:
        """doc_id -> the reference as written, for every document cited in `text`."""
        found: dict[int, str] = {}
        for rx, table, fold in ((self._names, self.names, True), (self._proper, self.proper, False)):
            for m in rx.finditer(text) if rx else ():
                key = " ".join(m.group(1).split()).lower() if fold else m.group(1)
                if (doc := table.get(key)) is not None:     # re.I can match text whose .lower() differs from the key
                    found.setdefault(doc, m.group(1))
        for ident in identifiers(text):     # a name reads better than an id as the label of the reference
            if ident in self.ids:
                found.setdefault(self.ids[ident], ident.split(":", 1)[1])
        return found


def _local_links(chunks: list[dict], sections: list[dict], doc_id: int) -> list[Link]:
    """Figure / table / section references inside one document."""
    captions: dict[tuple[str, str], list[int]] = {}
    for c in chunks:
        for m in CAPTION.finditer(c["text"]):
            captions.setdefault((_kind_of(m.group(1)), m.group(2).lower()), []).append(c["id"])
    figures_on = {}
    for c in chunks:
        if c["kind"] == "image" and c["loc"]:
            figures_on.setdefault(c["loc"], []).append(c["id"])
    loc_of = {c["id"]: c["loc"] for c in chunks}
    numbered: dict[tuple[str, str], int] = {}
    for s in sections:
        if (m := SEC_HEAD.match(s["title"])):
            numbered.setdefault((_family(m.group(1)), (m.group(2) or m.group(4)).lower()), s["first_chunk"])

    out: list[Link] = []
    for c in chunks:
        for m in XREF.finditer(c["text"]):
            kind, num = _kind_of(m.group(1)), m.group(2).lower()
            targets = [t for t in captions.get((kind, num), []) if t != c["id"]]
            targets += [f for t in list(targets) for f in figures_on.get(loc_of[t], []) if f != c["id"]]
            out += [Link(c["id"], doc_id, t, kind, m.group(0)) for t in dict.fromkeys(targets)]
        for m in SEC_REF.finditer(c["text"]):
            target = numbered.get((_family(m.group(1)), m.group(3).lower()))
            if target and target != c["id"]:
                out.append(Link(c["id"], doc_id, target, "section", m.group(0)))
    return out


def _dedupe(links: list[Link]) -> list[Link]:
    return list({(l.src, l.dst_doc, l.dst_chunk, l.kind): l for l in links}.values())


def refresh(store: Store, doc_id: int) -> int:
    """(Re)compute every edge that starts or ends in `doc_id`. Returns how many edges the document has."""
    store.delete_links(doc_id)
    chunks = store.doc_chunks(doc_id)
    links = _local_links(chunks, store.section_heads(doc_id), doc_id)

    everyone = Directory.build(store.directory_rows())
    outgoing = everyone.subset(lambda d: d != doc_id)
    if outgoing:
        for c in chunks:
            links += [Link(c["id"], dst, None, "doc", label) for dst, label in outgoing.mentions(c["text"]).items()]

    incoming = everyone.subset(lambda d: d == doc_id)
    if incoming:
        for row in store.chunks_matching(incoming.phrases(), doc_id):
            links += [Link(row["id"], dst, None, "doc", label) for dst, label in incoming.mentions(row["text"]).items()]
    links = _dedupe(links)
    store.add_links(links)
    return len(links)
