# -*- coding: utf-8 -*-
"""懂你意思 · 梗库存储层（SQLite，零外部依赖）

表设计：
    memes       梗本体（标题 / 别名 / 释义 / 使用场景 / 情绪标签 / 风险 / 状态 / 来源）
    variants    同一梗的不同说法（用户原话）
    usages      回梗使用记录（用于统计与「同梗重复」衰减）
    persona_pool 视频里学到的人物人设素材（候选 → 审核 → 可注入）

状态机：candidate（候选）→ confirmed（确认可用）/ rejected（拒绝）
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS memes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    aliases TEXT DEFAULT '[]',
    meaning TEXT DEFAULT '',
    usage TEXT DEFAULT '',
    emotion_tags TEXT DEFAULT '[]',
    risk TEXT DEFAULT 'low',
    status TEXT DEFAULT 'candidate',
    source TEXT DEFAULT '',
    source_ref TEXT DEFAULT '',
    embedding TEXT DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(title)
);

CREATE TABLE IF NOT EXISTS variants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meme_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_variants_meme ON variants(meme_id);

CREATE TABLE IF NOT EXISTS usages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meme_id INTEGER NOT NULL,
    umo TEXT DEFAULT '',
    used_at REAL NOT NULL,
    context TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_usages_meme ON usages(meme_id);

CREATE TABLE IF NOT EXISTS persona_pool (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    source TEXT DEFAULT '',
    traits TEXT DEFAULT '[]',
    speech_style TEXT DEFAULT '',
    catchphrases TEXT DEFAULT '[]',
    summary TEXT DEFAULT '',
    status TEXT DEFAULT 'candidate',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(name)
);
"""

STATUSES = ("candidate", "confirmed", "rejected")


def _loads(value: Any, default):
    if isinstance(value, (list, dict)):
        return value
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


class MemeStore:
    def __init__(self, path: str):
        self.path = path

    # ------------------------------------------------------------ infra
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def init(self) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    # ------------------------------------------------------------ memes
    def count(self, status: str | None = None) -> int:
        with self._conn() as conn:
            if status:
                row = conn.execute("SELECT COUNT(*) FROM memes WHERE status=?", (status,)).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM memes").fetchone()
            return int(row[0])

    def stats(self) -> dict:
        with self._conn() as conn:
            total = int(conn.execute("SELECT COUNT(*) FROM memes").fetchone()[0])
            confirmed = int(conn.execute("SELECT COUNT(*) FROM memes WHERE status='confirmed'").fetchone()[0])
            candidate = int(conn.execute("SELECT COUNT(*) FROM memes WHERE status='candidate'").fetchone()[0])
            rejected = int(conn.execute("SELECT COUNT(*) FROM memes WHERE status='rejected'").fetchone()[0])
            usages = int(conn.execute("SELECT COUNT(*) FROM usages").fetchone()[0])
            personas = int(conn.execute("SELECT COUNT(*) FROM persona_pool").fetchone()[0])
            try:
                top = conn.execute(
                    "SELECT m.title, COUNT(u.id) AS c FROM memes m LEFT JOIN usages u ON u.meme_id=m.id"
                    " GROUP BY m.id ORDER BY c DESC LIMIT 5"
                ).fetchall()
                top_titles = [dict(r) for r in top]
            except sqlite3.Error:
                top_titles = []
        return {
            "total": total,
            "confirmed": confirmed,
            "candidate": candidate,
            "rejected": rejected,
            "usages": usages,
            "personas": personas,
            "top": top_titles,
        }

    def get(self, meme_id: int) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM memes WHERE id=?", (int(meme_id),)).fetchone()
        return dict(row) if row else None

    def get_by_title(self, title: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM memes WHERE title=?", (title,)).fetchone()
        return dict(row) if row else None

    def upsert(
        self,
        title: str,
        meaning: str = "",
        aliases: list[str] | None = None,
        usage: str = "",
        emotion_tags: list[str] | None = None,
        risk: str = "low",
        status: str = "candidate",
        source: str = "",
        source_ref: str = "",
    ) -> int:
        title = (title or "").strip()[:40]
        if not title:
            return 0
        if risk not in ("low", "mid", "high"):
            risk = "low"
        if status not in STATUSES:
            status = "candidate"
        now = time.time()
        with self._conn() as conn:
            row = conn.execute("SELECT id, status FROM memes WHERE title=?", (title,)).fetchone()
            if row:
                # 已确认的梗不被自动学习降级
                new_status = row["status"] if row["status"] == "confirmed" else status
                conn.execute(
                    "UPDATE memes SET meaning=?, aliases=?, usage=?, emotion_tags=?, risk=?,"
                    " status=?, updated_at=? WHERE id=?",
                    (
                        meaning or "",
                        json.dumps(aliases or [], ensure_ascii=False),
                        usage or "",
                        json.dumps(emotion_tags or [], ensure_ascii=False),
                        risk,
                        new_status,
                        now,
                        row["id"],
                    ),
                )
                return int(row["id"])
            cur = conn.execute(
                "INSERT INTO memes (title, aliases, meaning, usage, emotion_tags, risk, status,"
                " source, source_ref, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    title,
                    json.dumps(aliases or [], ensure_ascii=False),
                    meaning or "",
                    usage or "",
                    json.dumps(emotion_tags or [], ensure_ascii=False),
                    risk,
                    status,
                    source or "",
                    source_ref or "",
                    now,
                    now,
                ),
            )
            return int(cur.lastrowid or 0)

    def upsert_detection(self, memes: list[dict], source: str = "user_reply", source_ref: str = "") -> int:
        """把识别出来的梗批量入库（候选），返回新增/更新条数。"""
        n = 0
        for item in memes or []:
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            aliases = []
            surface = str(item.get("surface") or "").strip()
            if surface and surface != title:
                aliases.append(surface)
            self.upsert(
                title=title,
                meaning=str(item.get("meaning") or ""),
                aliases=aliases,
                risk=str(item.get("risk") or "low"),
                status="candidate",
                source=source,
                source_ref=source_ref,
            )
            n += 1
        return n

    def list(
        self,
        status: str = "",
        keyword: str = "",
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        sql = "SELECT * FROM memes WHERE 1=1"
        args: list[Any] = []
        if status in STATUSES:
            sql += " AND status=?"
            args.append(status)
        if keyword:
            sql += " AND (title LIKE ? OR aliases LIKE ? OR meaning LIKE ?)"
            pattern = f"%{keyword}%"
            args.extend([pattern, pattern, pattern])
        sql += " ORDER BY CASE status WHEN 'confirmed' THEN 0 WHEN 'candidate' THEN 1 ELSE 2 END," \
               " updated_at DESC LIMIT ? OFFSET ?"
        args.extend([max(1, min(200, int(limit))), max(0, int(offset))])
        with self._conn() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [self._meme_row(r) for r in rows]

    def search(self, text: str, limit: int = 3) -> list[dict]:
        """关键词 / 别名检索（embedding 语义检索留作后续）。"""
        text = (text or "").strip()
        if not text:
            return []
        pattern = f"%{text}%"
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM memes WHERE status!='rejected' AND"
                " (title LIKE ? OR aliases LIKE ? OR meaning LIKE ?)"
                " ORDER BY CASE status WHEN 'confirmed' THEN 0 ELSE 1 END, updated_at DESC LIMIT ?",
                (pattern, pattern, pattern, int(limit)),
            ).fetchall()
        return [self._meme_row(r) for r in rows]

    def top_for_prompt(self, limit: int = 12) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM memes WHERE status IN ('confirmed','candidate')"
                " ORDER BY CASE status WHEN 'confirmed' THEN 0 ELSE 1 END, updated_at DESC LIMIT ?",
                (max(1, min(50, int(limit))),),
            ).fetchall()
        return [self._meme_row(r) for r in rows]

    def set_status(self, meme_id: int, status: str) -> bool:
        if status not in STATUSES:
            return False
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE memes SET status=?, updated_at=? WHERE id=?",
                (status, time.time(), int(meme_id)),
            )
            return cur.rowcount > 0

    def update(self, meme_id: int, **fields) -> bool:
        allowed = {"title", "meaning", "usage", "risk", "status", "aliases", "emotion_tags"}
        sets, args = [], []
        for key, value in fields.items():
            if key not in allowed:
                continue
            if key in ("aliases", "emotion_tags"):
                value = json.dumps(value or [], ensure_ascii=False)
            sets.append(f"{key}=?")
            args.append(value)
        if not sets:
            return False
        sets.append("updated_at=?")
        args.append(time.time())
        args.append(int(meme_id))
        with self._conn() as conn:
            cur = conn.execute(f"UPDATE memes SET {', '.join(sets)} WHERE id=?", args)
            return cur.rowcount > 0

    def delete(self, meme_id: int) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM memes WHERE id=?", (int(meme_id),))
            return cur.rowcount > 0

    def record_usage(self, meme_id: int, umo: str = "", context: str = "") -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO usages (meme_id, umo, used_at, context) VALUES (?,?,?,?)",
                (int(meme_id), umo or "", time.time(), (context or "")[:500]),
            )

    def record_usage_by_title(self, umo: str, titles: list[str], context: str = "") -> None:
        for title in titles or []:
            row = self.get_by_title(title)
            if row:
                self.record_usage(row["id"], umo, context)

    def add_variant(self, meme_id: int, text: str) -> None:
        if not text:
            return
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO variants (meme_id, text, created_at) VALUES (?,?,?)",
                (int(meme_id), text[:200], time.time()),
            )

    # ------------------------------------------------------------ persona pool
    def add_persona(
        self,
        name: str,
        traits: list | None = None,
        speech_style: str = "",
        catchphrases: list | None = None,
        summary: str = "",
        source: str = "video",
        status: str = "candidate",
    ) -> int:
        name = (name or "").strip()[:40]
        if not name:
            return 0
        now = time.time()
        with self._conn() as conn:
            row = conn.execute("SELECT id FROM persona_pool WHERE name=?", (name,)).fetchone()
            if row:
                conn.execute(
                    "UPDATE persona_pool SET traits=?, speech_style=?, catchphrases=?, summary=?,"
                    " status=?, updated_at=? WHERE id=?",
                    (
                        json.dumps(traits or [], ensure_ascii=False),
                        speech_style or "",
                        json.dumps(catchphrases or [], ensure_ascii=False),
                        summary or "",
                        status,
                        now,
                        row["id"],
                    ),
                )
                return int(row["id"])
            cur = conn.execute(
                "INSERT INTO persona_pool (name, source, traits, speech_style, catchphrases,"
                " summary, status, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    name,
                    source or "video",
                    json.dumps(traits or [], ensure_ascii=False),
                    speech_style or "",
                    json.dumps(catchphrases or [], ensure_ascii=False),
                    summary or "",
                    status,
                    now,
                    now,
                ),
            )
            return int(cur.lastrowid or 0)

    def list_personas(self, status: str = "", limit: int = 50) -> list[dict]:
        sql = "SELECT * FROM persona_pool"
        args: list[Any] = []
        if status in STATUSES:
            sql += " WHERE status=?"
            args.append(status)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        args.append(max(1, min(200, int(limit))))
        with self._conn() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [self._persona_row(r) for r in rows]

    def set_persona_status(self, persona_id: int, status: str) -> bool:
        if status not in STATUSES:
            return False
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE persona_pool SET status=?, updated_at=? WHERE id=?",
                (status, time.time(), int(persona_id)),
            )
            return cur.rowcount > 0

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _meme_row(row: sqlite3.Row) -> dict:
        data = dict(row)
        data["aliases"] = _loads(data.get("aliases"), [])
        data["emotion_tags"] = _loads(data.get("emotion_tags"), [])
        return data

    @staticmethod
    def _persona_row(row: sqlite3.Row) -> dict:
        data = dict(row)
        data["traits"] = _loads(data.get("traits"), [])
        data["catchphrases"] = _loads(data.get("catchphrases"), [])
        return data
