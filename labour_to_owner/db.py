"""SQLite persistence: daily pay snapshots + saved comparisons. (Streamlit Cloud disk is ephemeral:
use the GitHub Action in .github/workflows to commit data/history.csv, or point this at Supabase/Postgres.)"""
from __future__ import annotations
import json, sqlite3
from datetime import date
from pathlib import Path
import pandas as pd

PATH = Path(__file__).parent / "data" / "ledger.db"

def _c() -> sqlite3.Connection:
    PATH.parent.mkdir(exist_ok=True)
    c = sqlite3.connect(PATH)
    c.execute("create table if not exists snap(day text, job text, pay real, primary key(day, job))")
    c.execute("create table if not exists cmp(name text primary key, jobs text, created text)")
    return c

def save_snapshot(day: date, pays: dict[str, float]) -> int:
    with _c() as c:
        have = c.execute("select count(*) from snap where day=?", (str(day),)).fetchone()[0]
        if have: return 0
        c.executemany("insert into snap values(?,?,?)", [(str(day), k, v) for k, v in pays.items()])
        return len(pays)

def days_stored() -> int:
    with _c() as c: return c.execute("select count(distinct day) from snap").fetchone()[0]

def history(job: str) -> pd.DataFrame:
    with _c() as c: return pd.read_sql("select day, pay from snap where job=? order by day", c, params=(job,))

def save_cmp(name: str, jobs: list[str]) -> None:
    with _c() as c: c.execute("insert or replace into cmp values(?,?,?)", (name, json.dumps(jobs), str(date.today())))

def list_cmp() -> dict[str, list[str]]:
    with _c() as c: return {n: json.loads(j) for n, j in c.execute("select name, jobs from cmp order by created desc")}
