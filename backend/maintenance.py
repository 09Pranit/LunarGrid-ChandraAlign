"""Offline cleanup for failed workers and expired jobs. Stop API and workers first."""
from __future__ import annotations
import argparse
from contextlib import closing
from pathlib import Path
import shutil
import time

from .job_models import Settings
from .job_store import JobStore


def cleanup(root: Path, older_than_hours=24):
    store=JobStore(root)
    cutoff=time.time()-older_than_hours*3600
    removed=[]
    with closing(store.connect()) as db, db:
        # This command is intentionally offline: active servers may own these reservations.
        db.execute('DELETE FROM reservations')
        for row in db.execute('SELECT job_id,status,updated FROM jobs').fetchall():
            if row['updated'] >= cutoff:continue
            directory=store.directory(row['job_id'])
            if directory.exists():shutil.rmtree(directory)
            db.execute('DELETE FROM jobs WHERE job_id=?',(row['job_id'],))
            removed.append(row['job_id'])
    for path in store.root.iterdir():
        if path.is_dir() and not path.is_symlink() and path.name.startswith(('.upload-','.inspect-')) and path.stat().st_mtime < cutoff:
            shutil.rmtree(path);removed.append(path.name)
    return removed


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline',action='store_true',help='Confirm API and workers are stopped')
    parser.add_argument('--older-than-hours',type=float,default=24)
    args=parser.parse_args()
    if not args.offline or args.older_than_hours < 0:
        parser.error('Stop API and workers, then pass --offline and a nonnegative retention age')
    print({'removed':cleanup(Settings().storage_dir,args.older_than_hours)})
