"""Celery configuration for MCP Server with scheduled tasks"""

from celery import Celery
from celery.schedules import crontab
from loguru import logger

from .core.constants import (
    CELERY_TASK_TIMEOUT_SECONDS,
    CELERY_TASK_SOFT_TIMEOUT_SECONDS,
    CELERY_WORKER_CONCURRENCY,
    CELERY_RESULT_EXPIRE_SECONDS,
)

# Create Celery app
app = Celery("mcp_server")

# Configuration - values from constants
app.conf.update(
    # Task settings
    task_track_started=True,
    task_time_limit=CELERY_TASK_TIMEOUT_SECONDS,
    task_soft_time_limit=CELERY_TASK_SOFT_TIMEOUT_SECONDS,
    task_acks_late=True,  # Only ack after task completes
    worker_prefetch_multiplier=1,  # Don't prefetch extra tasks

    # Result expiration - critical for memory management
    result_expires=CELERY_RESULT_EXPIRE_SECONDS,

    # Worker settings - will be overridden by CLI arg
    worker_concurrency=CELERY_WORKER_CONCURRENCY,

    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Celery Beat - periodic tasks
    beat_schedule={
        'cleanup-old-blacklisted-daily': {
            'task': 'tasks.periodic.cleanup_blacklist',
            'schedule': crontab(hour=2, minute=0),  # 2 AM daily
            'kwargs': {'days_old': 7},  # Remove domains older than 7 days
        },
        'cleanup-old-metrics-daily': {
            'task': 'tasks.periodic.cleanup_old_metrics',
            'schedule': crontab(hour=3, minute=0),  # 3 AM daily
            'kwargs': {'days': 7},  # Keep metrics for 7 days
        },
    },
)

# Import tasks explicitly (autodiscovery doesn't work with our structure)
# This MUST happen after app is created
from .tasks.scrape_tasks import scrape_task  # noqa: E402
from .tasks import periodic_tasks  # noqa: E402 - Register periodic tasks


@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """Setup periodic tasks - this is where Beat registers them"""
    logger.info("Celery Beat periodic tasks configured:")
    logger.info("  - cleanup_blacklist: Daily at 2 AM (removes blacklisted domains >7 days old)")
    logger.info("  - cleanup_old_metrics: Daily at 3 AM (removes scrape metrics >7 days old)")


if __name__ == "__main__":
    app.start()
