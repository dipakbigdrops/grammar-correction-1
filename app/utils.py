"""
Utility Functions
"""
import hashlib
import json
import logging
import os
import tempfile
from typing import Optional


def get_upload_dir() -> str:
    """Cross-platform upload directory (works on Windows and Linux)."""
    if os.name == "nt":
        return os.path.join(tempfile.gettempdir(), "grammar_uploads")
    return "/tmp/uploads"


def get_output_dir() -> str:
    """Cross-platform output directory."""
    if os.name == "nt":
        return os.path.join(tempfile.gettempdir(), "grammar_outputs")
    return "/tmp/outputs"

from app.config import settings

logger = logging.getLogger(__name__)

_memory_client = None


def get_redis_client():
    """Return in-memory cache client (FakeRedis). No Redis server required."""
    global _memory_client
    if _memory_client is None:
        try:
            import fakeredis
            _memory_client = fakeredis.FakeStrictRedis(decode_responses=True)
            logger.info("Using in-memory cache (no Redis)")
        except ImportError as e:
            logger.warning("FakeRedis not available: %s", e)
            _memory_client = None
    return _memory_client


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_cached_result(file_hash: str) -> Optional[dict]:
    """Get cached result from in-memory cache"""
    if not settings.ENABLE_CACHING:
        return None

    client = get_redis_client()
    if client is None:
        return None

    try:
        cached = client.get(f"result:{file_hash}")
        if cached:
            logger.info("Cache hit for hash: %s", file_hash)
            return json.loads(cached)
    except (Exception, json.JSONDecodeError) as e:
        logger.error("Error getting cached result: %s", e)

    return None


def set_cached_result(file_hash: str, result: dict):
    """Cache result in memory"""
    if not settings.ENABLE_CACHING:
        return

    client = get_redis_client()
    if client is None:
        return

    try:
        client.setex(
            f"result:{file_hash}",
            settings.CACHE_TTL,
            json.dumps(result)
        )
        logger.info("Cached result for hash: %s", file_hash)
    except (Exception, TypeError) as e:
        logger.error("Error caching result: %s", e)


def create_directories():
    """Create necessary directories"""
    directories = [
        get_upload_dir(),
        get_output_dir(),
        os.path.join(tempfile.gettempdir(), "grammar_cache") if os.name == "nt" else "/tmp/cache"
    ]

    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as e:
            logger.error("Failed to create directory %s: %s", directory, e)


def save_uploaded_file(content: bytes, filename: str) -> str:
    """Save uploaded file to temp directory and return path. Filename is sanitized to prevent path traversal."""
    base_dir = os.path.abspath(get_upload_dir())
    os.makedirs(base_dir, exist_ok=True)
    safe_basename = os.path.basename(filename.replace("\\", "/").strip()) if filename else "unnamed"
    if not safe_basename:
        safe_basename = "unnamed"
    file_path = os.path.join(base_dir, safe_basename)
    real_path = os.path.abspath(file_path)
    try:
        if os.path.commonpath([real_path, base_dir]) != base_dir:
            raise ValueError("Invalid filename: path would escape upload directory")
    except ValueError:
        raise ValueError("Invalid filename: path would escape upload directory")
    with open(real_path, "wb") as f:
        f.write(content)
    return real_path


def cleanup_old_files(directory: str, max_age_seconds: int):
    """Remove files older than max_age_seconds from directory. Skips paths that resolve outside directory."""
    import time
    try:
        base_abs = os.path.abspath(directory)
        for name in os.listdir(directory):
            path = os.path.join(directory, name)
            try:
                real_path = os.path.abspath(path)
            except OSError:
                continue
            try:
                if os.path.commonpath([real_path, base_abs]) != base_abs or real_path == base_abs:
                    continue
            except ValueError:
                continue
            if os.path.isfile(path):
                try:
                    if (time.time() - os.path.getmtime(path)) > max_age_seconds:
                        os.remove(path)
                        logger.debug("Removed old file: %s", path)
                except OSError:
                    pass
    except OSError as e:
        logger.debug("Cleanup skipped for %s: %s", directory, e)
