"""Durable, process-safe job state for API and Celery on a shared local volume."""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3
import time

JOB_ID = re.compile(r"job_lunar_[0-9a-f]{32}\Z")
TERMINAL = {"complete", "review_required", "failed"}


class JobStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "jobs.sqlite3"
        with closing(self.connect()) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY, status TEXT NOT NULL, progress INTEGER NOT NULL,
                stage TEXT NOT NULL, updated REAL NOT NULL, params TEXT NOT NULL,
                result TEXT, error TEXT)""")

    def connect(self):
        db = sqlite3.connect(self.database, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def directory(self, job_id: str) -> Path:
        if not JOB_ID.fullmatch(job_id):
            raise KeyError(job_id)
        path = (self.root / job_id).resolve()
        if path.parent != self.root:
            raise KeyError(job_id)
        return path

    def create(self, job_id: str, params: dict):
        self.directory(job_id)
        with closing(self.connect()) as db, db:
            db.execute("INSERT INTO jobs VALUES (?, 'queued', 0, 'queued', ?, ?, NULL, NULL)",
                       (job_id, time.time(), json.dumps(params, allow_nan=False)))

    def get(self, job_id: str, *, include_result=False) -> dict:
        self.directory(job_id)
        columns = "*" if include_result else "job_id,status,progress,stage,error"
        with closing(self.connect()) as db:
            row = db.execute(f"SELECT {columns} FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        record = dict(row)
        record["progress_percent"] = record.pop("progress")
        if include_result:
            record["params"] = json.loads(record["params"])
            record["result"] = json.loads(record["result"]) if record["result"] else None
        return record

    def claim(self, job_id: str) -> bool:
        with closing(self.connect()) as db, db:
            return db.execute("""UPDATE jobs SET status='processing', progress=5,
                stage='ingestion', updated=? WHERE job_id=? AND status='queued'""",
                (time.time(), job_id)).rowcount == 1

    def progress(self, job_id: str, percent: int, stage: str):
        with closing(self.connect()) as db, db:
            db.execute("""UPDATE jobs SET progress=?, stage=?, updated=?
                WHERE job_id=? AND status='processing'""", (percent, stage, time.time(), job_id))

    def finish(self, job_id: str, status: str, result: dict | None = None, error: str | None = None):
        if status not in TERMINAL:
            raise ValueError("Invalid terminal status")
        serialized = json.dumps(result, allow_nan=False) if result is not None else None
        with closing(self.connect()) as db, db:
            db.execute("""UPDATE jobs SET status=?, progress=100, stage=?, updated=?, result=?, error=?
                WHERE job_id=? AND status IN ('queued','processing')""",
                (status, status, time.time(), serialized, error, job_id))

    def expire_stalled(self, job_id: str, timeout: float):
        # Covers a hard-killed worker or a broker with no consumer. Late results
        # cannot overwrite the terminal timeout state or expose partial artifacts.
        with closing(self.connect()) as db, db:
            db.execute("""UPDATE jobs SET status='failed', progress=100, stage='failed',
                error='Job timed out; submit a new job.' WHERE job_id=?
                AND status IN ('queued','processing') AND updated < ?""",
                (job_id, time.time() - timeout))
