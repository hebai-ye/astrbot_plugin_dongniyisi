# -*- coding: utf-8 -*-
"""把内置种子梗库导入「懂你意思」数据库。

用法（在 AstrBot 的数据目录/插件目录下均可运行）：
    python scripts/import_seed.py --db <插件数据目录>/meme_library.db
    python scripts/import_seed.py --db <...> --status confirmed
默认把种子条目写成 confirmed（可直接被注入使用）；--status candidate 则先入库待审。
已存在的同名梗会跳过，不会覆盖你自己编辑过的内容。
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time

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
"""


def seed_path() -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for rel in ("seed/memes_seed.json", "seed/meme_seed.json", "memes_seed.json"):
        p = os.path.join(here, rel)
        if os.path.isfile(p):
            return p
    raise SystemExit("找不到种子文件 seed/memes_seed.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="meme_library.db 路径")
    ap.add_argument("--status", default="confirmed", choices=["confirmed", "candidate"])
    ap.add_argument("--source", default="seed")
    args = ap.parse_args()

    data = json.load(open(seed_path(), encoding="utf-8"))
    os.makedirs(os.path.dirname(os.path.abspath(args.db)), exist_ok=True)
    con = sqlite3.connect(args.db)
    try:
        con.executescript(SCHEMA)
        now = time.time()
        added = skipped = variants_added = 0
        for item in data:
            title = (item.get("title") or "").strip()
            if not title:
                continue
            cur = con.execute("SELECT id FROM memes WHERE title = ?", (title,))
            row = cur.fetchone()
            if row:
                skipped += 1
                continue
            cur = con.execute(
                "INSERT INTO memes (title, aliases, meaning, usage, emotion_tags, risk, status,"
                " source, source_ref, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    title,
                    json.dumps(item.get("aliases") or [], ensure_ascii=False),
                    item.get("meaning") or "",
                    item.get("usage") or "",
                    json.dumps(item.get("emotion_tags") or [], ensure_ascii=False),
                    item.get("risk") or "low",
                    args.status,
                    args.source,
                    args.source,
                    now,
                    now,
                ),
            )
            meme_id = cur.lastrowid
            added += 1
            for text in item.get("variants") or []:
                text = str(text).strip()
                if text:
                    con.execute(
                        "INSERT INTO variants (meme_id, text, created_at) VALUES (?,?,?)",
                        (meme_id, text, now),
                    )
                    variants_added += 1
        con.commit()
    finally:
        con.close()
    print(f"导入完成：新增 {added} 条，跳过已存在 {skipped} 条，附带示例 {variants_added} 条")


if __name__ == "__main__":
    main()
