"""Durable, process-safe job state for API and Celery on a shared local volume."""
from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3
import time
import shutil
from uuid import uuid4

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
            db.execute("""CREATE TABLE IF NOT EXISTS reservations (
                token TEXT PRIMARY KEY, bytes INTEGER NOT NULL, created REAL NOT NULL)""")

    def reserve(self, settings):
        """Cross-process admission before multipart spooling; conservative disk reservation."""
        # Multipart spool + isolated staged copy coexist until the request ends.
        amount = 2 * settings.max_request_bytes + settings.max_tile_pixels * 64
        with closing(self.connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            rows = db.execute('SELECT bytes FROM reservations').fetchall()
            if len(rows) >= settings.max_active_jobs:
                raise ValueError('Upload/inspection capacity reached; retry after current request finishes')
            used = 0
            for index, path in enumerate(self.root.rglob('*')):
                if index > 100000:
                    raise ValueError('Storage file-count quota reached; operator cleanup required')
                if path.is_file():
                    try:
                        used += path.stat().st_size
                    except FileNotFoundError:
                        pass  # A completed worker may have just removed its staging file.
            reserved = sum(r['bytes'] for r in rows)
            if used + reserved + amount > settings.max_storage_bytes or shutil.disk_usage(self.root).free < settings.min_free_bytes + reserved + amount:
                raise ValueError('Staging disk quota/free-space reserve exhausted; operator cleanup required')
            token = uuid4().hex
            db.execute('INSERT INTO reservations VALUES (?, ?, ?)', (token, amount, time.time()))
            return token

    def release(self, token):
        with closing(self.connect()) as db, db:
            db.execute('DELETE FROM reservations WHERE token=?', (token,))

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

    def create(self, job_id: str, params: dict, max_active=None):
        self.directory(job_id)
        with closing(self.connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            if max_active is not None and db.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','processing')").fetchone()[0] >= max_active:
                raise ValueError('Active job quota reached; wait or cancel a job')
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

    def cancel(self, job_id: str):
        with closing(self.connect()) as db, db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT status FROM jobs WHERE job_id=?', (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            previous = row['status']
            if previous not in TERMINAL:
                db.execute("UPDATE jobs SET status='failed', progress=100, stage='cancelled', updated=?, error='Registration cancelled by user' WHERE job_id=?", (time.time(),job_id))
            return previous

    def progress(self, job_id: str, percent: int, stage: str):
        with closing(self.connect()) as db, db:
            changed = db.execute("""UPDATE jobs SET progress=?, stage=?, updated=?
                WHERE job_id=? AND status='processing'""", (percent, stage, time.time(), job_id))
            if changed.rowcount != 1:
                raise ValueError('Registration cancelled or expired')

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
