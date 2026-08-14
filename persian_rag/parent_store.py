"""Simple key-value store for parent chunk text, keyed by parent_id.

Parents are never embedded/indexed — they're only fetched by id once a
child chunk has already matched in retrieval. SQLite is plenty for a
single 625-page document (a few hundred parent rows).
"""
import sqlite3
from contextlib import contextmanager

from .chunking import ParentChunk


@contextmanager
def _conn(path: str):
    c = sqlite3.connect(path)
    try:
        yield c
    finally:
        c.close()


def init_db(path: str) -> None:
    with _conn(path) as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS parents (
                parent_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                page_start INTEGER,
                page_end INTEGER
            )
        """)
        c.commit()


def save_parents(path: str, parents: list[ParentChunk]) -> None:
    init_db(path)
    with _conn(path) as c:
        c.executemany(
            "INSERT OR REPLACE INTO parents (parent_id, text, page_start, page_end) VALUES (?, ?, ?, ?)",
            [(p.parent_id, p.text, p.page_start, p.page_end) for p in parents],
        )
        c.commit()


def get_parents(path: str, parent_ids: list[str]) -> dict[str, dict]:
    if not parent_ids:
        return {}
    init_db(path)
    placeholders = ",".join("?" for _ in parent_ids)
    with _conn(path) as c:
        rows = c.execute(
            f"SELECT parent_id, text, page_start, page_end FROM parents WHERE parent_id IN ({placeholders})",
            parent_ids,
        ).fetchall()
    return {
        r[0]: {"parent_id": r[0], "text": r[1], "page_start": r[2], "page_end": r[3]}
        for r in rows
    }
