"""Celery entry point: celery --workdir backend -A tasks:celery_app worker.

Only IDs and the shared storage root travel through Redis. SQLite is the result
store, so local eager execution does not require a Redis result backend.
"""
from __future__ import annotations

import logging

from celery import Celery
from kombu.exceptions import OperationalError
from redis import Redis
from redis.backoff import NoBackoff
from redis.exceptions import RedisError
from redis.retry import Retry

if __package__:
    from .job_models import JobResult, Settings
    from .job_store import JobStore
else:
    from job_models import JobResult, Settings
    from job_store import JobStore

logger = logging.getLogger(__name__)
TASK_NAME = "lunar.registration"


def process_registration(job_id: str, storage_dir: str):
    store = JobStore(storage_dir)
    if not store.claim(job_id):
        return  # Atomic claim makes duplicate broker deliveries harmless.
    try:
        if __package__:
            from .pipeline import run_pipeline
        else:
            from pipeline import run_pipeline
        record = store.get(job_id, include_result=True)
        result = run_pipeline(store.directory(job_id), record["params"],
                              lambda percent, stage: store.progress(job_id, percent, stage))
        JobResult.model_validate(result)
        store.finish(job_id, result["status"], result)
    except Exception:
        logger.exception("Registration job %s failed", job_id)
        store.finish(job_id, "failed", error="Registration failed. Check worker logs using the job ID.")


def build_celery(settings: Settings) -> Celery:
    queue = Celery("lunar_registration", broker=settings.broker_url)
    queue.conf.update(
        task_always_eager=settings.always_eager,
        task_eager_propagates=True,
        task_ignore_result=True,
        task_store_eager_result=False,
        task_serializer="json", accept_content=["json"], result_serializer="json",
        task_default_queue="lunar.registration", worker_prefetch_multiplier=1,
        task_acks_late=True, task_reject_on_worker_lost=False,
        task_soft_time_limit=settings.job_timeout - 1,
        task_time_limit=settings.job_timeout,
        broker_connection_retry_on_startup=True,
        broker_connection_timeout=1, task_publish_retry=False,
        broker_transport_options={"socket_connect_timeout": 1, "socket_timeout": 1,
                                  "visibility_timeout": settings.job_timeout * 2},
    )
    queue.task(name=TASK_NAME)(process_registration)
    return queue


def redis_available(url: str) -> bool:
    try:
        with Redis.from_url(url, socket_connect_timeout=0.25, socket_timeout=0.25,
                            retry=Retry(NoBackoff(), 0)) as client:
            return bool(client.ping())
    except (RedisError, OSError):
        return False


def configure_dispatch(queue: Celery, settings: Settings):
    # Called at API startup, not at import and not by worker processes.
    if not settings.always_eager and settings.eager_fallback and not redis_available(settings.broker_url):
        queue.conf.task_always_eager = True
        logger.info("Redis unavailable: CELERY_TASK_ALWAYS_EAGER=True for local development")


def enqueue(queue: Celery, settings: Settings, job_id: str):
    task = queue.tasks[TASK_NAME]
    args = (job_id, str(settings.storage_dir.resolve()))
    try:
        return task.apply_async(args=args, task_id=job_id, retry=False)
    except (OperationalError, RedisError, OSError):
        if not settings.eager_fallback:
            raise
        # A publish can succeed before its acknowledgement is lost. The store's
        # atomic claim prevents a concurrent worker and fallback from running twice.
        logger.info("Broker publish failed; executing %s eagerly", job_id)
        return task.apply(args=args, task_id=job_id)


celery_app = build_celery(Settings())
