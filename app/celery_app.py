"""
Celery Application Configuration (optional; app runs realtime without Celery).
"""
from celery import Celery
from app.config import settings

celery_app = Celery(
    "grammar_correction",
    broker=getattr(settings, "CELERY_BROKER_URL", "memory://"),
    backend=getattr(settings, "CELERY_RESULT_BACKEND", "cache+memory://"),
    include=["app.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=getattr(settings, "CELERY_TASK_TRACK_STARTED", True),
    task_time_limit=getattr(settings, "CELERY_TASK_TIME_LIMIT", 600),
    task_soft_time_limit=getattr(settings, "CELERY_TASK_SOFT_TIME_LIMIT", 540),
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# Optional: Configure result backend settings
celery_app.conf.result_backend_transport_options = {
    'master_name': 'mymaster',
}

if __name__ == '__main__':
    celery_app.start()