"""SQLite persistence: documents, sections, chunks (+FTS5), embeddings, review queue, chat memory.

Vectors are float32 blobs searched by brute-force numpy dot products over cached matrices.
# ponytail: brute force is fine up to ~200k chunks; swap in sqlite-vec/LanceDB beyond that.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .textutil import WORD, terms

REFS_PER_DIRECTION = 5   # references listed per source and direction: a pointer, not a bibliography
HEAD_CHUNKS = 2   # a document's own identifiers live on its first page

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents(
  id INTEGER PRIMARY KEY, path TEXT, sha256 TEXT UNIQUE, format TEXT, title TEXT, summary TEXT,
  emb BLOB, n_sections INTEGER, n_chunks INTEGER, words INTEGER, strategy TEXT,
  confidence REAL, status TEXT, warnings TEXT, ingested_at REAL, pii TEXT,
  mtime_ns INTEGER, size INTEGER, embed_id TEXT, index_sig TEXT);
CREATE TABLE IF NOT EXISTS sections(
  id INTEGER PRIMARY KEY, doc_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
  ordinal INTEGER, level INTEGER, title TEXT, path TEXT, summary TEXT, emb BLOB);
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY, doc_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
  section_id INTEGER REFERENCES sections(id) ON DELETE CASCADE, ordinal INTEGER,
  text TEXT, ctx TEXT, kind TEXT, loc TEXT, emb BLOB, ref TEXT);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);
CREATE INDEX IF NOT EXISTS chunks_section ON chunks(section_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(ctx, tokenize='unicode61 remove_diacritics 2');
CREATE TABLE IF NOT EXISTS links(
  src INTEGER REFERENCES chunks(id) ON DELETE CASCADE, dst_doc INTEGER REFERENCES documents(id) ON DELETE CASCADE,
  dst_chunk INTEGER REFERENCES chunks(id) ON DELETE CASCADE, kind TEXT, label TEXT);
CREATE INDEX IF NOT EXISTS links_src ON links(src);
CREATE INDEX IF NOT EXISTS links_dst_chunk ON links(dst_chunk);
CREATE INDEX IF NOT EXISTS links_dst_doc ON links(dst_doc);
CREATE TABLE IF NOT EXISTS review_queue(
  id INTEGER PRIMARY KEY, doc_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
  kind TEXT, block_idx INTEGER, text TEXT, state TEXT, options TEXT, probs TEXT,
  style_key TEXT, level INTEGER, body_key TEXT, format TEXT, resolved INTEGER DEFAULT 0, created REAL);
CREATE TABLE IF NOT EXISTS sessions(id INTEGER PRIMARY KEY, title TEXT, summary TEXT, created REAL);
CREATE TABLE IF NOT EXISTS messages(
  id INTEGER PRIMARY KEY, session_id INTEGER REFERENCES sessions(id) ON DELETE CASCADE,
  role TEXT, content TEXT, sources TEXT, created REAL);
"""


def _blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


@dataclass(frozen=True)
class Matrix:
    ids: list[int]
    vectors: np.ndarray   # (n, d), unit-normalised


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._local = threading.local()
        self._version = 0
        self._cache: dict[tuple, tuple[int, Matrix]] = {}
        self._lock = threading.RLock()
        with self.conn:
            self.conn.executescript(SCHEMA)
            have = {r["name"] for r in self.conn.execute("PRAGMA table_info(documents)")}
            chunk_cols = {r["name"] for r in self.conn.execute("PRAGMA table_info(chunks)")}
            if "ref" not in chunk_cols:
                self.conn.execute("ALTER TABLE chunks ADD COLUMN ref TEXT")
            for col, kind in (("pii", "TEXT"), ("mtime_ns", "INTEGER"), ("size", "INTEGER"),
                              ("embed_id", "TEXT"), ("index_sig", "TEXT")):
                if col not in have:                                   # DBs created by older versions
                    self.conn.execute(f"ALTER TABLE documents ADD COLUMN {col} {kind}")

    @property
    def conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn"):
            c = sqlite3.connect(self.path)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA foreign_keys=ON")
            self._local.conn = c
        return self._local.conn

    # ---- documents -------------------------------------------------------------------------
    def sha_of(self, doc_id: int) -> str:
        return self.conn.execute("SELECT sha256 FROM documents WHERE id=?", (doc_id,)).fetchone()[0]

    def find_by_hash(self, sha: str) -> sqlite3.Row | None:
        return self.conn.execute("SELECT * FROM documents WHERE sha256=?", (sha,)).fetchone()

    def save_document(self, meta: dict, doc_emb: np.ndarray | None, sections: list[dict]) -> int:
        """sections: dicts with section fields + 'emb' + 'chunks': [dict(text, ctx, kind, loc, emb)]."""
        with self._lock, self.conn as c:
            cur = c.execute(
                "INSERT INTO documents(path,sha256,format,title,summary,emb,n_sections,n_chunks,words,"
                "strategy,confidence,status,warnings,ingested_at,pii,mtime_ns,size,embed_id,index_sig) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (meta["path"], meta["sha256"], meta["format"], meta["title"], meta["summary"],
                 _blob(doc_emb) if doc_emb is not None else None, len(sections),
                 sum(len(s["chunks"]) for s in sections), meta["words"], meta["strategy"],
                 meta["confidence"], meta["status"], json.dumps(meta.get("warnings", [])), time.time(),
                 json.dumps(meta.get("pii", {})), meta.get("mtime_ns"), meta.get("size"),
                 meta.get("embed_id"), meta.get("index_sig")))
            doc_id = cur.lastrowid
            for ordinal, s in enumerate(sections):
                sc = c.execute(
                    "INSERT INTO sections(doc_id,ordinal,level,title,path,summary,emb) VALUES(?,?,?,?,?,?,?)",
                    (doc_id, ordinal, s["level"], s["title"], s["path"], s["summary"], _blob(s["emb"])))
                for k, ch in enumerate(s["chunks"]):
                    cc = c.execute(
                        "INSERT INTO chunks(doc_id,section_id,ordinal,text,ctx,kind,loc,emb,ref) VALUES(?,?,?,?,?,?,?,?,?)",
                        (doc_id, sc.lastrowid, k, ch["text"], ch["ctx"], ch["kind"], ch["loc"], _blob(ch["emb"]),
                         ch.get("ref", "")))
                    c.execute("INSERT INTO chunk_fts(rowid,ctx) VALUES(?,?)", (cc.lastrowid, ch["ctx"]))
            self._version += 1
        return doc_id

    def sync_rows(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT id,path,sha256,mtime_ns,size,embed_id,index_sig,status FROM documents").fetchall()

    def set_location(self, doc_id: int, path: str | None = None, mtime_ns: int | None = None,
                     size: int | None = None) -> None:
        with self._lock, self.conn as c:
            if path is not None:
                c.execute("UPDATE documents SET path=? WHERE id=?", (path, doc_id))
            if mtime_ns is not None:
                c.execute("UPDATE documents SET mtime_ns=?, size=? WHERE id=?", (mtime_ns, size, doc_id))

    def reusable_embeddings(self, doc_id: int) -> dict[str, np.ndarray]:
        """text -> vector for every chunk / section card / doc card of a document, so an edit only
        re-embeds what actually changed."""
        out: dict[str, np.ndarray] = {}
        for r in self.conn.execute("SELECT ctx, emb FROM chunks WHERE doc_id=?", (doc_id,)):
            if r["emb"]:
                out[r["ctx"]] = _vec(r["emb"])
        for r in self.conn.execute("SELECT path, summary, emb FROM sections WHERE doc_id=?", (doc_id,)):
            if r["emb"]:
                out[f"{r['path']}\n{r['summary']}"] = _vec(r["emb"])
        d = self.conn.execute("SELECT title, summary, emb FROM documents WHERE id=?", (doc_id,)).fetchone()
        if d and d["emb"]:
            out[f"{d['title']}\n{d['summary']}"] = _vec(d["emb"])
        return out

    def delete_document(self, doc_id: int) -> None:
        with self._lock, self.conn as c:
            c.execute("DELETE FROM chunk_fts WHERE rowid IN (SELECT id FROM chunks WHERE doc_id=?)", (doc_id,))
            c.execute("DELETE FROM documents WHERE id=?", (doc_id,))
            self._version += 1

    def list_documents(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id,path,format,title,n_sections,n_chunks,words,strategy,confidence,status,warnings,"
            "ingested_at,pii FROM documents ORDER BY ingested_at DESC").fetchall()
        return [dict(r) | {"warnings": json.loads(r["warnings"] or "[]"), "pii": json.loads(r["pii"] or "{}")}
                for r in rows]

    def outline(self, doc_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id,level,title,summary FROM sections WHERE doc_id=? ORDER BY ordinal", (doc_id,))
        return [dict(r) for r in rows]

    # ---- vectors + lexical search ---------------------------------------------------------
    def matrix(self, table: str, doc_ids: tuple[int, ...] | None = None) -> Matrix:
        key = (table, doc_ids)
        cached = self._cache.get(key)
        if cached and cached[0] == self._version:
            return cached[1]
        where, args = "", ()
        if doc_ids is not None:
            where = f"WHERE {'id' if table == 'documents' else 'doc_id'} IN ({','.join('?' * len(doc_ids))})"
            args = doc_ids
        rows = self.conn.execute(f"SELECT id, emb FROM {table} {where}", args).fetchall()
        rows = [r for r in rows if r["emb"]]
        m = Matrix([r["id"] for r in rows],
                   np.vstack([_vec(r["emb"]) for r in rows]) if rows else np.zeros((0, 1), np.float32))
        self._cache[key] = (self._version, m)
        return m

    def fts(self, query: str, doc_ids: list[int] | None, limit: int) -> list[int]:
        words = terms(query)
        if not words:
            return []
        match = " OR ".join(f'"{t}"' for t in dict.fromkeys(words))
        sql = ("SELECT f.rowid FROM chunk_fts f JOIN chunks c ON c.id=f.rowid "
               "WHERE chunk_fts MATCH ?")
        args: list = [match]
        if doc_ids is not None:
            sql += f" AND c.doc_id IN ({','.join('?' * len(doc_ids))})"
            args += doc_ids
        sql += " ORDER BY bm25(chunk_fts) LIMIT ?"
        return [r[0] for r in self.conn.execute(sql, [*args, limit])]

    def chunks_by_ids(self, ids: list[int]) -> dict[int, dict]:
        if not ids:
            return {}
        q = ("SELECT c.id,c.doc_id,c.section_id,c.text,c.ctx,c.kind,c.loc,c.ref,d.title AS doc_title,"
             "s.title AS section_title,s.path AS section_path FROM chunks c "
             "JOIN documents d ON d.id=c.doc_id JOIN sections s ON s.id=c.section_id "
             f"WHERE c.id IN ({','.join('?' * len(ids))})")
        return {r["id"]: dict(r) for r in self.conn.execute(q, ids)}

    def section_chunk_ids(self, section_ids: list[int]) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        q = f"SELECT id, section_id FROM chunks WHERE section_id IN ({','.join('?' * len(section_ids))})"
        for r in self.conn.execute(q, section_ids) if section_ids else []:
            out.setdefault(r["section_id"], []).append(r["id"])
        return out

    def section_text(self, section_id: int) -> str:
        rows = self.conn.execute("SELECT text FROM chunks WHERE section_id=? ORDER BY ordinal", (section_id,))
        return "\n".join(r[0] for r in rows)

    # ---- reference graph (links.py) ---------------------------------------------------------
    def doc_chunks(self, doc_id: int) -> list[dict]:
        rows = self.conn.execute("SELECT id,text,kind,loc FROM chunks WHERE doc_id=? ORDER BY id", (doc_id,))
        return [dict(r) for r in rows]

    def section_heads(self, doc_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT s.title, MIN(c.id) AS first_chunk FROM sections s JOIN chunks c ON c.section_id=s.id "
            "WHERE s.doc_id=? GROUP BY s.id ORDER BY s.ordinal", (doc_id,))
        return [dict(r) for r in rows]

    def directory_rows(self) -> list[tuple[int, str, str, str]]:
        """(doc_id, title, path, head text) of every searchable document: what other documents may cite."""
        out = []
        for d in self.conn.execute("SELECT id,title,path FROM documents WHERE status != 'needs_ocr'").fetchall():
            head = self.conn.execute("SELECT text FROM chunks WHERE doc_id=? ORDER BY id LIMIT ?",
                                     (d["id"], HEAD_CHUNKS)).fetchall()
            out.append((d["id"], d["title"] or "", d["path"] or "", "\n".join(r[0] for r in head)))
        return out

    def chunks_matching(self, phrases: list[str], exclude_doc: int, limit: int = 5000) -> list[dict]:
        """Chunks of other documents that contain any of the phrases (FTS prefilter; callers verify exactly)."""
        match = " OR ".join(f'"{" ".join(WORD.findall(p.lower()))}"' for p in phrases if WORD.findall(p.lower()))
        if not match:
            return []
        rows = self.conn.execute(
            "SELECT c.id,c.doc_id,c.text FROM chunk_fts f JOIN chunks c ON c.id=f.rowid "
            "WHERE chunk_fts MATCH ? AND c.doc_id != ? LIMIT ?", (match, exclude_doc, limit))
        return [dict(r) for r in rows]

    def delete_links(self, doc_id: int) -> None:
        """Edges leaving or entering the document (both directions are recomputed on refresh)."""
        with self._lock, self.conn as c:
            c.execute("DELETE FROM links WHERE dst_doc=? OR src IN (SELECT id FROM chunks WHERE doc_id=?)",
                      (doc_id, doc_id))

    def add_links(self, links) -> None:
        with self._lock, self.conn as c:
            c.executemany("INSERT INTO links(src,dst_doc,dst_chunk,kind,label) VALUES(?,?,?,?,?)",
                          [(l.src, l.dst_doc, l.dst_chunk, l.kind, l.label) for l in links])

    def links_out(self, chunk_ids: list[int]) -> list[dict]:
        if not chunk_ids:
            return []
        q = f"SELECT src,dst_doc,dst_chunk,kind,label FROM links WHERE src IN ({','.join('?' * len(chunk_ids))})"
        return [dict(r) for r in self.conn.execute(q, chunk_ids)]

    def links_in(self, chunk_ids: list[int]) -> list[dict]:
        """Chunks that point at these chunks (figure / table / section references only)."""
        if not chunk_ids:
            return []
        q = ("SELECT src,dst_doc,dst_chunk,kind,label FROM links "
             f"WHERE dst_chunk IN ({','.join('?' * len(chunk_ids))})")
        return [dict(r) for r in self.conn.execute(q, chunk_ids)]

    def neighbours(self, chunk_ids: list[int], limit: int = REFS_PER_DIRECTION) -> dict[str, list[dict]]:
        """Where the references of these chunks lead ("out") and who cites them ("in"), named for a reader:
        kind, label as written, document, page, target chunk id (None = a whole cited document)."""
        own = set(chunk_ids)
        out, back = self.links_out(chunk_ids), self.links_in(chunk_ids)
        ids = {r for l in out for r in (l["dst_chunk"],) if r} | {l["src"] for l in back}
        where = {r["id"]: r for r in self.conn.execute(
            f"SELECT c.id, c.loc, c.doc_id, d.title FROM chunks c JOIN documents d ON d.id=c.doc_id "
            f"WHERE c.id IN ({','.join('?' * len(ids))})", list(ids))} if ids else {}
        titles = {r["id"]: r["title"] for r in self.conn.execute("SELECT id, title FROM documents")}

        def card(l: dict, chunk: int | None, doc: int | None) -> dict | None:
            if chunk in own:
                return None
            w = where.get(chunk)
            return {"kind": l["kind"], "label": l["label"], "chunk_id": chunk, "doc_id": w["doc_id"] if w else doc,
                    "doc": w["title"] if w else titles.get(doc, ""), "loc": w["loc"] if w else None}

        def pick(cards) -> list[dict]:
            seen = {(c["kind"], c["label"], c["chunk_id"], c["doc_id"]): c for c in cards if c}
            return list(seen.values())[:limit]

        return {"out": pick(card(l, l["dst_chunk"], l["dst_doc"]) for l in out),
                "in": pick(card(l, l["src"], None) for l in back)}

    def chunk_node(self, chunk_id: int) -> dict | None:
        """One passage with its place in the document and its references, to open a node of the graph."""
        r = self.conn.execute(
            "SELECT c.id, c.doc_id, c.text, c.kind, c.loc, c.ref, d.title AS doc, s.path AS section "
            "FROM chunks c JOIN documents d ON d.id=c.doc_id JOIN sections s ON s.id=c.section_id WHERE c.id=?",
            (chunk_id,)).fetchone()
        return {**dict(r), "refs": self.neighbours([chunk_id])} if r else None

    def graph(self) -> dict:
        """Document-level view: nodes and edges (how many chunks cite the other document, by kind)."""
        edges = self.conn.execute(
            "SELECT c.doc_id AS src, l.dst_doc AS dst, l.kind, COUNT(*) AS n FROM links l "
            "JOIN chunks c ON c.id=l.src GROUP BY c.doc_id, l.dst_doc, l.kind").fetchall()
        return {"edges": [dict(e) for e in edges],
                "nodes": [{"id": d["id"], "title": d["title"]} for d in self.list_documents()]}

    def section_figures(self, section_ids: list[int], per_section: int = 3) -> dict[int, list[str]]:
        out: dict[int, list[str]] = {}
        q = ("SELECT section_id, ref FROM chunks WHERE kind='image' AND ref != '' AND section_id IN "
             f"({','.join('?' * len(section_ids))}) ORDER BY id")
        for r in self.conn.execute(q, section_ids) if section_ids else []:
            refs = out.setdefault(r["section_id"], [])
            if r["ref"] not in refs and len(refs) < per_section:
                refs.append(r["ref"])
        return out

    # ---- review queue ----------------------------------------------------------------------
    def queue_review(self, doc_id: int, item: dict) -> None:
        with self._lock, self.conn as c:
            c.execute(
                "INSERT INTO review_queue(doc_id,kind,block_idx,text,state,options,probs,style_key,level,"
                "body_key,format,created) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (doc_id, item["kind"], item["idx"], item["text"], item.get("state", ""),
                 json.dumps(item.get("options") or {}), json.dumps(item["probs"]), item.get("style_key", ""),
                 item.get("level", 0), item.get("body_key", ""), item.get("format", ""), time.time()))

    def pending_reviews(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT r.*, d.title AS doc_title FROM review_queue r JOIN documents d ON d.id=r.doc_id "
            "WHERE resolved=0 ORDER BY r.id").fetchall()
        return [dict(r) | {"options": json.loads(r["options"]), "probs": json.loads(r["probs"])} for r in rows]

    def get_review(self, review_id: int) -> dict | None:
        r = self.conn.execute("SELECT * FROM review_queue WHERE id=?", (review_id,)).fetchone()
        return dict(r) if r else None

    def resolve_review(self, review_id: int) -> None:
        with self._lock, self.conn as c:
            c.execute("UPDATE review_queue SET resolved=1 WHERE id=?", (review_id,))

    # ---- chat memory -----------------------------------------------------------------------
    def create_session(self, title: str = "New chat") -> int:
        with self._lock, self.conn as c:
            return c.execute("INSERT INTO sessions(title,summary,created) VALUES(?,?,?)",
                             (title, "", time.time())).lastrowid

    def list_sessions(self) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT id,title,created FROM sessions ORDER BY id DESC")]

    def get_session(self, session_id: int) -> dict | None:
        r = self.conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        return dict(r) if r else None

    def set_session(self, session_id: int, **fields: str) -> None:
        allowed = {k: v for k, v in fields.items() if k in ("title", "summary")}
        with self._lock, self.conn as c:
            for k, v in allowed.items():
                c.execute(f"UPDATE sessions SET {k}=? WHERE id=?", (v, session_id))

    def delete_session(self, session_id: int) -> None:
        with self._lock, self.conn as c:
            c.execute("DELETE FROM sessions WHERE id=?", (session_id,))

    def add_message(self, session_id: int, role: str, content: str, sources: list[dict] | None = None) -> None:
        with self._lock, self.conn as c:
            c.execute("INSERT INTO messages(session_id,role,content,sources,created) VALUES(?,?,?,?,?)",
                      (session_id, role, content, json.dumps(sources or []), time.time()))

    def messages(self, session_id: int) -> list[dict]:
        rows = self.conn.execute("SELECT role,content,sources FROM messages WHERE session_id=? ORDER BY id",
                                 (session_id,))
        return [dict(r) | {"sources": json.loads(r["sources"])} for r in rows]
