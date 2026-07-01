"""
ClipForge FastAPI application.

Startup sequence:
    1. Initialize SQLite schema (create tables if missing)
    2. Run stale job recovery scan (marks abandoned jobs as failed)
    3. If REDIS_URL is set and Redis is reachable, log that RQ workers
       should be started separately via `rq worker`. If Redis is unavailable,
       log a warning — the app still starts but job queueing is degraded.

Job submission: POST /api/jobs enqueues process_job() via RQ.
If Redis is unavailable, the endpoint falls back to a daemon thread
(development mode only — production must have Redis running).
"""
import logging
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.models import init_db

logger = logging.getLogger(__name__)

# Configure logging early so startup messages are captured
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="ClipForge",
    description="AI-Powered Thumbnail Generator & Short-Form Clip Maker",
    version="0.1.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Startup handler
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup() -> None:
    """Initialize schema and run stale job recovery on every startup."""
    # 1. Create tables
    init_db()
    logger.info("Database schema initialized")

    # 2. Stale job recovery
    from backend.workers.job_processor import recover_stale_jobs
    recovered = recover_stale_jobs(db_path=str(settings.db_path))
    if recovered:
        logger.warning(f"Startup recovery: marked {recovered} stale job(s) as failed")
    else:
        logger.info("Startup recovery: no stale jobs found")

    # 3. Check Redis availability (informational only — workers are separate processes)
    if settings.redis_url:
        try:
            import redis as redis_lib
            r = redis_lib.from_url(settings.redis_url, socket_connect_timeout=2)
            r.ping()
            logger.info(f"Redis available at {settings.redis_url} — start RQ workers with: rq worker")
        except Exception as exc:
            logger.warning(
                f"Redis not reachable at {settings.redis_url}: {exc}. "
                "Job submission will fall back to in-process threads (dev mode only)."
            )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/")
@app.get("/api/health")
def health_check():
    """Liveness probe. Returns 200 when the app is running."""
    return {"status": "ok", "service": "clipforge"}


# ---------------------------------------------------------------------------
# Routers — imported after app is created to avoid circular imports
# ---------------------------------------------------------------------------
try:
    from backend.routers import jobs, thumbnails, clips, exports, posts
    app.include_router(jobs.router)
    app.include_router(thumbnails.router)
    app.include_router(clips.router)
    app.include_router(exports.router)
    app.include_router(posts.router)
except ImportError as exc:
    logger.warning(f"Could not import one or more routers: {exc}. Starting without them.")
