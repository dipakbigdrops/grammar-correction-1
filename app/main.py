"""
FastAPI Application
Main API endpoints and application setup
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Query
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
import hashlib
import uuid
import json
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import logging
from typing import Optional, Dict, Any
import time
from contextlib import asynccontextmanager

# Import custom middleware
from app.middleware import RateLimitMiddleware, CircuitBreakerMiddleware, RequestTrackingMiddleware

from app.config import settings
from app.models import (
    TaskResponse, TaskStatusResponse, ProcessResult,
    HealthResponse, ErrorResponse, ProcessingStatus, InputType
)
from app.utils import (
    get_redis_client, compute_file_hash, get_cached_result,
    set_cached_result, save_uploaded_file, cleanup_old_files,
    get_upload_dir, get_output_dir
)

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Helper functions for HTML preview storage in Redis
def store_html_preview(preview_id: str, html_content: str, filename: str, ttl: int = 3600) -> bool:
    """Store HTML preview in memory with TTL"""
    try:
        client = get_redis_client()
        if not client:
            return False
        preview_data = {
            'html': html_content,
            'timestamp': time.time(),
            'filename': filename
        }
        key = f"html_preview:{preview_id}"
        client.setex(key, ttl, json.dumps(preview_data))
        logger.debug("Stored HTML preview %s with TTL %d", preview_id, ttl)
        return True
    except Exception as e:
        logger.error("Error storing HTML preview: %s", e, exc_info=True)
        return False

def get_html_preview_data(preview_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve HTML preview from memory"""
    try:
        client = get_redis_client()
        if not client:
            return None
        key = f"html_preview:{preview_id}"
        cached_data = client.get(key)
        if cached_data:
            preview_data = json.loads(cached_data)
            logger.debug("Retrieved HTML preview %s", preview_id)
            return preview_data
        return None
    except Exception as e:
        logger.error("Error retrieving HTML preview: %s", e, exc_info=True)
        return None

def delete_html_preview(preview_id: str) -> bool:
    """Delete HTML preview from memory"""
    try:
        client = get_redis_client()
        if not client:
            return False
        key = f"html_preview:{preview_id}"
        client.delete(key)
        logger.debug("Deleted HTML preview %s", preview_id)
        return True
    except Exception as e:
        logger.error("Error deleting HTML preview: %s", e, exc_info=True)
        return False
from app.processor import get_processor
from app.universal_processor import get_universal_processor
from app.cache_manager import get_cache_manager

# HTML previews stored in-memory with 1-hour TTL


def _resolved_model_path() -> str:
    """Return MODEL_PATH resolved to project root when relative (matches processor and health)."""
    model_path = settings.MODEL_PATH
    if not os.path.isabs(model_path):
        if not os.path.exists(model_path):
            _app_dir = os.path.dirname(os.path.abspath(__file__))
            _project_root = os.path.dirname(_app_dir)
            _resolved = os.path.join(_project_root, model_path.replace("\\", "/").lstrip("./"))
            if os.path.exists(_resolved):
                return os.path.abspath(_resolved)
        return os.path.abspath(model_path)
    return os.path.abspath(model_path)


def _download_model_at_startup() -> None:
    """Download grammar model from Hugging Face if MODEL_ID is set and model dir is empty."""
    model_id = (getattr(settings, "MODEL_ID", "") or "").strip() or None
    if not model_id:
        return
    model_path = _resolved_model_path()
    if os.path.exists(os.path.join(model_path, "config.json")):
        return
    try:
        from huggingface_hub import snapshot_download
        hf_token = (getattr(settings, "HF_TOKEN", "") or "").strip() or None
        os.makedirs(model_path, exist_ok=True)
        logger.info("Downloading model from Hugging Face: %s into %s", model_id, model_path)
        snapshot_download(repo_id=model_id, local_dir=model_path, token=hf_token)
        logger.info("Model downloaded successfully")
    except Exception as e:
        logger.error("Startup model download failed: %s", e, exc_info=True)


_processing_semaphore: asyncio.Semaphore = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events"""
    global _processing_semaphore
    _processing_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_REQUESTS)
    logger.info(
        "Starting %s v%s (max concurrent requests: %d)",
        settings.APP_NAME, settings.APP_VERSION, settings.MAX_CONCURRENT_REQUESTS
    )

    try:
        os.makedirs(get_upload_dir(), exist_ok=True)
        os.makedirs(get_output_dir(), exist_ok=True)
        logger.info("Created necessary directories")
    except OSError as e:
        logger.error("Failed to create directories: %s", e)

    model_path = _resolved_model_path()
    config_path = os.path.join(model_path, "config.json")
    model_id = (getattr(settings, "MODEL_ID", "") or "").strip()
    if model_id and not os.path.exists(config_path):
        logger.info("Model not found at %s, downloading from Hugging Face (%s) ...", model_path, model_id)
        await asyncio.to_thread(_download_model_at_startup)
        if os.path.exists(config_path):
            logger.info("Startup model download completed; model ready at %s", model_path)
        else:
            logger.warning("Startup model download may have failed; config.json still missing at %s", model_path)

    logger.info("Application started successfully")

    async def _idle_unload_loop():
        while True:
            await asyncio.sleep(60)
            try:
                p = get_processor()
                p.unload_model_if_idle()
                p.unload_ocr_if_idle()
            except Exception as e:
                logger.debug("Idle unload check: %s", e)

    _idle_task = asyncio.create_task(_idle_unload_loop())

    yield

    _idle_task.cancel()
    try:
        await _idle_task
    except asyncio.CancelledError:
        pass
    logger.info("Shutting down application")


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="High-performance Grammar Correction API with OCR and HTML support",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# Add custom middleware for production
app.add_middleware(RequestTrackingMiddleware)
app.add_middleware(
    CircuitBreakerMiddleware,
    failure_threshold=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
    timeout=settings.CIRCUIT_BREAKER_TIMEOUT
)
app.add_middleware(
    RateLimitMiddleware,
    requests_per_minute=settings.RATE_LIMIT_PER_MINUTE,
    burst=settings.RATE_LIMIT_BURST
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to {}".format(settings.APP_NAME),
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    # Check if processor can be initialized
    model_loaded = False
    ocr_available = False
    beautifulsoup_available = False
    image_reconstruction_available = False
    html_reconstruction_available = False
    
    try:
        model_dir = _resolved_model_path()
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(model_dir) and os.path.exists(config_path):
            try:
                files = os.listdir(model_dir)
                model_loaded = "config.json" in files and any(
                    f.endswith((".bin", ".safetensors")) for f in files
                )
            except OSError:
                model_loaded = False
        else:
            model_loaded = False
        
        # Check OCR availability
        try:
            import easyocr
            ocr_available = True
        except ImportError:
            ocr_available = False
        except (OSError, RuntimeError):
            ocr_available = False

        # Check BeautifulSoup availability
        try:
            from bs4 import BeautifulSoup
            # Test with a simple HTML string
            soup = BeautifulSoup("<html><body>test</body></html>", 'html.parser')
            beautifulsoup_available = True
        except ImportError:
            beautifulsoup_available = False
        except (ValueError, AttributeError):
            beautifulsoup_available = False

        # Check image reconstruction capabilities
        try:
            from PIL import Image, ImageDraw, ImageFont
            import cv2
            import numpy as np
            # Test basic image operations
            test_img = Image.new('RGB', (100, 100), color='white')
            test_array = np.array(test_img)
            image_reconstruction_available = True
        except ImportError:
            image_reconstruction_available = False
        except (OSError, ValueError):
            image_reconstruction_available = False

        # Check HTML reconstruction capabilities
        try:
            from bs4 import BeautifulSoup
            from difflib import Differ
            # Test HTML parsing and text extraction
            test_html = "<html><body><p>Test content</p></body></html>"
            soup = BeautifulSoup(test_html, 'html.parser')
            text = soup.get_text()
            differ = Differ()
            html_reconstruction_available = True
        except ImportError:
            html_reconstruction_available = False
        except (ValueError, AttributeError):
            html_reconstruction_available = False

    except (OSError, ImportError) as e:
        logger.debug("Model/OCR health check failed: %s", e)
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        version=settings.APP_VERSION,
        grammar_model_loaded=model_loaded,
        ocr_available=ocr_available,
        beautifulsoup_available=beautifulsoup_available,
        image_reconstruction_available=image_reconstruction_available,
        html_reconstruction_available=html_reconstruction_available
    )


@app.post(
    "/process",
    response_model=TaskResponse,
    tags=["Processing"],
    responses={
        200: {
            "description": "Processing results",
            "content": {
                "application/json": {
                    "schema": {"$ref": "#/components/schemas/TaskResponse"},
                    "example": {
                        "task_id": "universal",
                        "status": "SUCCESS",
                        "message": "Processing completed",
                        "result": {
                            "input_type": "html",
                            "output_content": "<html>...</html>",
                            "corrections_count": 1
                        }
                    }
                },
                "text/html": {
                    "schema": {"type": "string"},
                    "example": "<html><body><p>This is a <u>test</u> sentence.</p></body></html>"
                }
            }
        },
        400: {"description": "Bad request - invalid file type or format parameter"},
        500: {"description": "Internal server error"}
    },
    summary="Process file for grammar correction",
    description="""
    Process uploaded file for grammar correction with support for multiple response formats.
    
    **Response Formats:**
    - `format=json` (default): Returns JSON with processing results
    - `format=html`: Returns HTML directly with `Content-Type: text/html` (HTML input only)
    
    **Supported Input Types:**
    - Images: .jpg, .jpeg, .png
    - HTML: .html, .htm
    - Archives: .zip (containing images/HTML)
    
    **HTML Response:**
    When `format=html` is used with HTML input, the response contains the corrected HTML
    with `<u>` tags wrapping corrected words. The response has `Content-Type: text/html`
    and can be rendered directly in a browser.
    
    **Preview:**
    For HTML responses, a `preview_id` is included in the response headers. Use
    `/process/preview/{preview_id}` to retrieve the HTML later.
    """
)
async def process_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    async_processing: bool = Form(default=True),
    format: Optional[str] = Query(default="json", description="Response format: 'json' or 'html' (for HTML input only)")
):
    file_path = None
    try:
        original_filename = file.filename
        if not original_filename:
            original_filename = "unnamed_file"
        
        # Validate file extension
        file_extension = os.path.splitext(original_filename)[1].lower()
        
        allowed_extensions = (
            settings.ALLOWED_IMAGE_EXTENSIONS + 
            settings.ALLOWED_HTML_EXTENSIONS + 
            settings.ALLOWED_ARCHIVE_EXTENSIONS
        )
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Read file content
        file_content = await file.read()
        
        # Check file size
        if len(file_content) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024)}MB"
            )
        
        # Save uploaded file
        file_path = save_uploaded_file(file_content, original_filename)
        background_tasks.add_task(cleanup_old_files, get_upload_dir(), 3600)
        background_tasks.add_task(cleanup_old_files, get_output_dir(), 3600)

        try:
            # Acquire a processing slot. On a 1-vCPU server MAX_CONCURRENT_REQUESTS=1
            # so inference is always sequential — no concurrent model calls, no CPU fight.
            # We wait up to 5 s for a free slot; after that we return 503 immediately
            # so clients know to retry rather than waiting indefinitely.
            try:
                await asyncio.wait_for(_processing_semaphore.acquire(), timeout=5.0)
            except asyncio.TimeoutError:
                raise HTTPException(
                    status_code=503,
                    detail="Server is busy processing another request. Please retry in a moment."
                )

            logger.info("Processing %s with universal processor", original_filename)
            universal_processor = get_universal_processor()
            try:
                result = await asyncio.to_thread(
                    universal_processor.process_any_input, file_path, get_output_dir()
                )
            finally:
                _processing_semaphore.release()

            stats = universal_processor.get_performance_stats()
            result['performance_stats'] = stats

            if format and format.lower() == "html":
                input_type = result.get('input_type')
                if input_type == 'html' and result.get('success'):
                    output_content = result.get('output_content')
                    if output_content and isinstance(output_content, str):
                        preview_id = str(uuid.uuid4())
                        store_html_preview(preview_id, output_content, original_filename, ttl=3600)
                        response = HTMLResponse(
                            content=output_content,
                            status_code=200,
                            media_type="text/html"
                        )
                        response.headers["X-Preview-ID"] = preview_id
                        return response
                raise HTTPException(
                    status_code=400,
                    detail=f"format=html is only available for HTML input files. Current input type: {input_type}"
                )

            return JSONResponse(content={
                "task_id": "universal",
                "status": "SUCCESS" if result.get('success') else "FAILURE",
                "message": "Processing completed with universal processor",
                "result": result,
                "estimated_completion_seconds": result.get('processing_time_seconds', 0)
            })
        except HTTPException:
            raise
        finally:
            try:
                if file_path and os.path.isfile(file_path):
                    os.remove(file_path)
            except OSError:
                pass

    except HTTPException:
        raise
    except ValueError as e:
        logger.error("Error processing file: %s", e, exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    except (OSError, RuntimeError, ImportError, AttributeError) as e:
        logger.error("Error processing file: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error("Error processing file: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/task/{task_id}", response_model=TaskStatusResponse, tags=["Tasks"])
async def get_task_status(task_id: str):
    """
    Get status of a processing task. Realtime mode: use /process for synchronous results.
    """
    if task_id == "universal" or task_id == "sync" or task_id == "cached":
        return TaskStatusResponse(
            task_id=task_id,
            status=ProcessingStatus.SUCCESS,
            progress=100,
            result={"message": "Realtime processing - results returned from /process"}
        )
    return TaskStatusResponse(
        task_id=task_id,
        status=ProcessingStatus.PENDING,
        progress=0,
        result={"message": "Realtime only - use POST /process for synchronous processing"}
    )


@app.get("/download/{filename}", tags=["Output"])
async def download_file(filename: str):
    """
    Download processed output file
    
    - **filename**: Name of the output file
    """
    # Sanitize filename to prevent path traversal attacks
    filename = os.path.basename(filename)
    
    output_dir = get_output_dir()
    file_path = os.path.join(output_dir, filename)
    if not os.path.abspath(file_path).startswith(os.path.abspath(output_dir)):
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )


@app.get(
    "/process/preview/{preview_id}",
    response_class=HTMLResponse,
    tags=["Processing"],
    responses={
        200: {
            "description": "HTML preview of processed file",
            "content": {
                "text/html": {
                    "schema": {"type": "string"},
                    "example": "<html><body><p>This is a <u>test</u> sentence.</p></body></html>"
                }
            }
        },
        404: {"description": "Preview not found or expired"}
    },
    summary="Get HTML preview of processed file",
    description="""
    Retrieve the HTML preview of a processed file using the preview ID.
    
    Preview IDs are returned in the `X-Preview-ID` header when using `format=html`
    with the `/process` endpoint. Previews are stored for 1 hour.
    
    This endpoint is useful for:
    - Opening HTML previews in a browser
    - Sharing processed HTML results
    - Testing HTML rendering without re-processing
    
    **Example:**
    ```bash
    # Process file and get preview ID
    curl -X POST "http://localhost:8000/process?format=html" -F "file=@example.html"
    # Response includes: X-Preview-ID: abc123-def456-...
    
    # Retrieve preview
    curl "http://localhost:8000/process/preview/abc123-def456-..."
    ```
    """
)
async def get_html_preview(preview_id: str):
    """
    Get HTML preview of processed file by preview ID
    
    - **preview_id**: Preview ID from X-Preview-ID header (returned when format=html)
    """
    # Retrieve preview from memory
    preview_data = get_html_preview_data(preview_id)
    
    if not preview_data:
        raise HTTPException(
            status_code=404,
            detail="HTML preview not found or expired. Previews are stored for 1 hour."
        )
    
    html_content = preview_data.get('html')
    if not html_content:
        raise HTTPException(
            status_code=404,
            detail="HTML preview data is invalid."
        )
    
    return HTMLResponse(
        content=html_content,
        status_code=200,
        media_type="text/html",
        headers={
            "X-Original-Filename": preview_data.get('filename', 'unknown')
        }
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus-compatible metrics endpoint
    """
    try:
        # Get universal processor stats
        universal_processor = get_universal_processor()
        processor_stats = universal_processor.get_performance_stats()
        
        # Get cache stats
        cache_manager = get_cache_manager()
        cache_stats = cache_manager.get_cache_stats()
        
        return {
            "status": "operational",
            "processor_stats": processor_stats,
            "cache_stats": cache_stats
        }
    except (OSError, AttributeError) as e:
        logger.error("Error getting metrics: %s", e)
        return {
            "status": "degraded",
            "error": "Metrics unavailable"
        }


@app.get("/performance", tags=["Monitoring"])
async def performance_stats():
    """
    Get detailed performance statistics
    """
    try:
        universal_processor = get_universal_processor()
        cache_manager = get_cache_manager()
        
        return {
            "processor_stats": universal_processor.get_performance_stats(),
            "cache_stats": cache_manager.get_cache_stats(),
            "cache_size": cache_manager.get_cache_size(),
            "timestamp": time.time()
        }
    except (OSError, AttributeError) as e:
        logger.error("Error getting performance stats: %s", e)
        return {
            "error": "Performance stats unavailable",
            "timestamp": time.time()
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        reload=settings.DEBUG
    )