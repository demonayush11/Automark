"""Tiny SQLite storage: tests (with marking scheme) and student scripts (with text + marks)."""
import json
import os
import sqlite3
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
DB_PATH = os.path.join(DATA_DIR, "grader.db")
SETTINGS_PATH = os.path.join(DATA_DIR, "settings.json")

SCHEMA = """
CREATE TABLE IF NOT EXISTS tests (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,
  created REAL NOT NULL,
  scheme_json TEXT NOT NULL DEFAULT '{"questions": []}'
);
CREATE TABLE IF NOT EXISTS scripts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  test_id INTEGER NOT NULL REFERENCES tests(id) ON DELETE CASCADE,
  student TEXT NOT NULL,
  roll TEXT NOT NULL DEFAULT '',
  files_json TEXT NOT NULL DEFAULT '[]',
  previews_json TEXT NOT NULL DEFAULT '[]',
  text TEXT NOT NULL DEFAULT '',
  words INTEGER NOT NULL DEFAULT 0,
  low_conf INTEGER NOT NULL DEFAULT 0,
  result_json TEXT,
  status TEXT NOT NULL DEFAULT 'uploaded',
  created REAL NOT NULL
);
"""


def connect():
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    cols = {r[1] for r in con.execute("PRAGMA table_info(scripts)")}
    if "reader" not in cols:                       # upgrade databases made by the first version
        con.execute("ALTER TABLE scripts ADD COLUMN reader TEXT NOT NULL DEFAULT 'own'")
        con.commit()
    return con


def row_to_dict(row):
    if row is None:
        return None
    d = dict(row)
    for key in ("scheme_json", "files_json", "previews_json", "result_json"):
        if key in d:
            d[key[:-5]] = json.loads(d[key]) if d[key] else None
    return d


def now():
    return time.time()


ENGINE_NAMES = {"similarity": "Sentence-BERT", "keywords": "Keywords"}
READER_MODES = ("auto", "own", "trocr")


def load_settings():
    """engine = similarity | keywords (how answers are marked)
    reader = auto | own | trocr     (who reads the handwriting). Everything runs locally."""
    s = {"engine": "similarity", "reader": "auto"}
    if os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH, encoding="utf-8") as fh:
            saved = json.load(fh)
        s["engine"] = saved.get("engine", s["engine"])
        s["reader"] = saved.get("reader", s["reader"])
    if s["engine"] not in ENGINE_NAMES:          # old settings (gemini/groq/offline) -> local engine
        s["engine"] = "keywords" if s["engine"] == "offline" else "similarity"
    if s["reader"] not in READER_MODES:
        s["reader"] = "auto"
    s["engine_name"] = ENGINE_NAMES[s["engine"]]
    return s


def save_settings(engine, reader):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as fh:
        json.dump({"engine": engine, "reader": reader}, fh)   # old API keys are dropped
