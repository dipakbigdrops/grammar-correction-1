FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2

# Install system dependencies for ML libraries and image processing
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-dev \
    libxml2-dev \
    libxslt1-dev \
    libffi-dev \
    libssl-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install Rust (required for tokenizers and sentencepiece compilation)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /app

# Create virtual environment to avoid pip warnings
RUN python -m venv /app/venv
ENV VIRTUAL_ENV=/app/venv \
    PATH="/app/venv/bin:$PATH"

# Copy requirements and constraints first for better Docker layer caching
COPY requirements.txt constraints.txt ./

# Skip Git LFS during build (prevents LFS download failures)
ENV GIT_LFS_SKIP_SMUDGE=1

# Upgrade pip and install Python dependencies
# Install essential build tools first
RUN /app/venv/bin/pip install --upgrade pip setuptools wheel cython

# Install numpy first (critical, and some packages depend on it)
# Use constraints to prevent any upgrades
RUN /app/venv/bin/pip install --no-cache-dir --constraint constraints.txt "numpy==1.26.4"

# Force numpy version before any other ML packages
RUN /app/venv/bin/pip install --no-cache-dir --force-reinstall --constraint constraints.txt "numpy==1.26.4"

# Install Pydantic FIRST (FastAPI depends on it)
RUN /app/venv/bin/pip install --no-cache-dir --constraint constraints.txt \
    "pydantic>=2.5.0,<3.0.0" \
    "pydantic-settings>=2.1.0,<3.0.0"

# Install PyTorch dependencies first (required when using --no-deps)
# Install pillow and requests BEFORE torchvision (torchvision requires them)
RUN /app/venv/bin/pip install --no-cache-dir --constraint constraints.txt \
    "typing-extensions>=4.8.0" \
    "filelock>=3.9.0" \
    "networkx>=2.6.0" \
    "sympy>=1.12.0" \
    "jinja2>=3.1.2" \
    "fsspec>=2023.6.0" \
    "packaging>=21.3" \
    "pillow==10.4.0" \
    "requests>=2.31.0,<3.0.0"

# Install PyTorch CPU version (large package, install early)
# Using 2.2.0 for better compatibility with transformers
# Using --no-deps to prevent numpy version conflicts
RUN /app/venv/bin/pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.2.0 \
    torchvision==0.17.0 \
    --no-deps

# Force numpy version after PyTorch installation
RUN /app/venv/bin/pip install --no-cache-dir --force-reinstall --constraint constraints.txt "numpy==1.26.4"

# Install web framework dependencies (FastAPI needs Pydantic already installed)
RUN /app/venv/bin/pip install --no-cache-dir --constraint constraints.txt \
    "fastapi>=0.104.0,<1.0.0" \
    "uvicorn[standard]>=0.24.0,<1.0.0" \
    "python-multipart>=0.0.6,<1.0.0"

# Install async and HTTP clients
RUN /app/venv/bin/pip install --no-cache-dir --constraint constraints.txt \
    "aiofiles>=23.2.0,<24.0.0" \
    "aiohttp>=3.9.0,<4.0.0" \
    "requests>=2.31.0,<3.0.0" \
    "httpx>=0.25.0,<1.0.0"

# Install HTML processing (lxml can be problematic, install separately)
RUN /app/venv/bin/pip install --no-cache-dir --constraint constraints.txt "beautifulsoup4>=4.12.0,<5.0.0" && \
    /app/venv/bin/pip install --no-cache-dir --constraint constraints.txt "lxml>=4.9.0,<5.0.0"

# Install in-memory cache (no Redis)
RUN /app/venv/bin/pip install --no-cache-dir --constraint constraints.txt "fakeredis>=2.32.0,<3.0.0"

# Install ML/transformers dependencies (after PyTorch and numpy)
# Pin transformers to compatible version with PyTorch 2.2.0
# Use constraints to prevent numpy/opencv upgrades
RUN /app/venv/bin/pip install --no-cache-dir --constraint constraints.txt \
    "tokenizers>=0.13.0,<1.0.0" \
    "sentencepiece>=0.2.0,<1.0.0" \
    "safetensors>=0.3.0,<1.0.0" \
    "huggingface-hub>=0.16.0,<1.0.0" \
    "transformers>=4.37.0,<4.50.0" \
    "accelerate>=0.20.0,<1.0.0" \
    "pillow==10.4.0" \
    "opencv-python-headless==4.9.0.80" \
    "easyocr>=1.7.0,<2.0.0"

# Force opencv version after easyocr (easyocr may try to upgrade opencv)
RUN /app/venv/bin/pip install --no-cache-dir --force-reinstall --constraint constraints.txt "opencv-python-headless==4.9.0.80"

# Force protobuf version after transformers (transformers may pull different version)
RUN /app/venv/bin/pip install --no-cache-dir --force-reinstall --constraint constraints.txt "protobuf==4.25.3"

# Install remaining utilities
RUN /app/venv/bin/pip install --no-cache-dir --constraint constraints.txt \
    "python-dotenv>=1.0.0,<2.0.0" \
    "prometheus-client>=0.19.0,<1.0.0" \
    "pyspellchecker>=0.8.0,<1.0.0"

# Final numpy, protobuf, and opencv version enforcement after all dependencies
RUN /app/venv/bin/pip install --no-cache-dir --force-reinstall --constraint constraints.txt "numpy==1.26.4" && \
    /app/venv/bin/pip install --no-cache-dir --force-reinstall --constraint constraints.txt "protobuf==4.25.3" && \
    /app/venv/bin/pip install --no-cache-dir --force-reinstall --constraint constraints.txt "opencv-python-headless==4.9.0.80"

# Verify numpy version is correct (fail build if wrong version)
RUN /app/venv/bin/python -c "import numpy; assert numpy.__version__ == '1.26.4', f'NumPy version mismatch! Expected 1.26.4, got {numpy.__version__}'; print(f'✓ NumPy version verified: {numpy.__version__}')" || exit 1

# Copy application code
COPY . .

# Download grammar model at build time (default repo is public; no token required)
# For a private HF repo: docker build --secret id=HF_TOKEN,env=HF_TOKEN -t grammar-api .
ARG MODEL_ID=dipak-bigdrops/grammar-correction-model
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    mkdir -p ./model && \
    echo "=== Downloading grammar model: ${MODEL_ID} ===" && \
    export HF_TOKEN= && [ -f /run/secrets/HF_TOKEN ] && export HF_TOKEN="$(cat /run/secrets/HF_TOKEN)" ; \
    MODEL_ID="${MODEL_ID}" HF_TOKEN="${HF_TOKEN}" /app/venv/bin/python -c " \
from huggingface_hub import snapshot_download; \
import os; \
model_id = (os.environ.get('MODEL_ID') or 'dipak-bigdrops/grammar-correction-model').strip(); \
token = (os.environ.get('HF_TOKEN') or '').strip() or None; \
snapshot_download(repo_id=model_id, local_dir='./model', token=token); \
print('Model download complete') \
" && \
    echo "=== Model Download Complete ===" && \
    ls -lh ./model/ 2>/dev/null || true

# Pre-download EasyOCR models during build to prevent runtime downloads
# This ensures models are cached in the Docker image and persist across requests
RUN echo "=== Pre-downloading EasyOCR Models ===" && \
    mkdir -p /app/.EasyOCR/model && \
    /app/venv/bin/python -c "import easyocr; import os; import sys; model_dir = '/app/.EasyOCR/model'; os.makedirs(model_dir, exist_ok=True); print('Initializing EasyOCR Reader to pre-download models to', model_dir, '...'); reader = easyocr.Reader(['en'], model_storage_directory=model_dir, gpu=False); print('✓ EasyOCR models pre-downloaded successfully'); files = os.listdir(model_dir); print('✓ Found', len(files), 'model files') if files else print('⚠ Warning: No model files found'); sys.exit(0 if files else 1)" && \
    echo "=== EasyOCR Model Download Complete ===" && \
    ls -lh /app/.EasyOCR/model/ 2>/dev/null || echo "Note: EasyOCR model directory listing unavailable"

# Create non-root user for runtime security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app

# Create necessary directories with proper permissions
RUN mkdir -p /tmp/uploads /tmp/cache /tmp/outputs /app/.EasyOCR/model && \
    chmod -R 755 /tmp /app/.EasyOCR && \
    chown -R appuser:appuser /tmp/uploads /tmp/cache /tmp/outputs /app/.EasyOCR || true

# Switch to non-root user for security (after all build steps)
USER appuser

# Expose port (supports PORT env var for Cloud Run, defaults to 8000)
EXPOSE 8000
ENV PORT=8000
ENV MODEL_PATH=/app/model
ENV MODEL_ID=dipak-bigdrops/grammar-correction-model

# Health check (model is in image; no runtime download)
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Run: use --cpus=2 --memory=4g so the container is not killed (OOM or CPU throttling).
CMD ["sh", "-c", "/app/venv/bin/python -c \"import numpy; assert numpy.__version__.startswith('1.26'), f'CRITICAL: NumPy version {numpy.__version__} is incompatible! Expected 1.26.x'; print(f'Runtime NumPy check: {numpy.__version__}')\" && /app/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]