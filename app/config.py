"""
Configuration Management
Handles all environment variables and settings
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
import os
import json


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Application Settings
    APP_NAME: str = "Grammar Correction API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1  # Single worker; T5 model is large — multiple workers multiply RAM usage
    
    # Model Settings
    MODEL_PATH: str = "./model"
    MODEL_ID: str = "dipak-bigdrops/grammar-correction-model"
    HF_TOKEN: str = ""  # Hugging Face token (optional, only needed for private models)
    MODEL_MAX_LENGTH: int = 256
    MODEL_NUM_BEAMS: int = 2
    MODEL_CHUNK_MAX_TOKENS: int = 256
    MODEL_BATCH_CHUNKS: int = 1
    MODEL_IDLE_UNLOAD_SECONDS: int = 600
    SKIP_MODEL_TEST: bool = False

    # OCR Settings
    OCR_LANGUAGES: list = ["en"]
    OCR_CONFIDENCE_THRESHOLD: float = 0.5
    OCR_MODEL_DIR: str = "/app/.EasyOCR/model"
    OCR_MAX_DIMENSION: int = 1024
    OCR_IDLE_UNLOAD_SECONDS: int = 180

    
    # Processing Settings
    MAX_FILE_SIZE: int = 20 * 1024 * 1024  # 20MB
    ALLOWED_IMAGE_EXTENSIONS: list = [".jpg", ".jpeg", ".png"]
    ALLOWED_HTML_EXTENSIONS: list = [".html", ".htm"]
    ALLOWED_ARCHIVE_EXTENSIONS: list = [".zip"]
    MAX_ZIP_EXTRACT_SIZE: int = 50 * 1024 * 1024  # 50MB total extracted
    MAX_FILES_IN_ZIP: int = 100  # Maximum files to process from ZIP
    CONTEXT_WORDS: int = 3
    
    # Cache Settings — disabled; all requests are processed in real time
    CACHE_TTL: int = 0
    ENABLE_CACHING: bool = False
    
    # Monitoring Settings
    ENABLE_METRICS: bool = True
    
    # CORS Settings
    ALLOWED_ORIGINS: list = ["*"]
    CORS_ALLOW_CREDENTIALS: bool = False
    CORS_ALLOW_METHODS: list = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_ALLOW_HEADERS: list = ["*"]
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    RATE_LIMIT_BURST: int = 200

    # Circuit Breaker
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    CIRCUIT_BREAKER_TIMEOUT: int = 60
    
    # Autoscaling Thresholds
    MIN_REPLICAS: int = 3
    MAX_REPLICAS: int = 100
    TARGET_CPU_UTILIZATION: int = 70
    TARGET_MEMORY_UTILIZATION: int = 80
    TARGET_REQUESTS_PER_SECOND: int = 100
    
    # Optimized Settings for High Throughput
    # Batch Processing Optimization
    BATCH_PROCESSING_TIMEOUT: int = 600  # 10 minutes for large batches
    ENABLE_BATCH_OPTIMIZATION: bool = True
    STREAMING_PROCESSING: bool = True  # Process files as they're extracted
    
    # Resource Optimization
    WORKER_CPU_LIMIT: float = 0.5
    WORKER_MEMORY_LIMIT: int = 1024
    WORKER_CONCURRENCY: int = 1
    # How many /process requests can run simultaneously.
    # 1 vCPU → set to 1.  2 vCPUs → set to 2.
    # Requests beyond this limit receive HTTP 503 immediately.
    MAX_CONCURRENT_REQUESTS: int = 1
    
    # Multi-Level Caching
    CACHE_TTL_TEXT: int = 300   # 5 minutes
    CACHE_TTL_MODEL: int = 300  # 5 minutes
    CACHE_TTL_OCR: int = 300    # 5 minutes
    CACHE_TTL_PARTIAL: int = 300  # 5 minutes
    
    # Cache Hit Rate Optimization
    ENABLE_TEXT_CACHING: bool = True
    ENABLE_MODEL_CACHING: bool = True
    ENABLE_OCR_CACHING: bool = True
    ENABLE_PARTIAL_CACHING: bool = True
    
    # Performance Monitoring
    METRICS_INTERVAL: int = 60  # 1 minute
    CACHE_HIT_RATE_TRACKING: bool = True
    PERFORMANCE_TRACKING: bool = True
    
    # Context and Processing
    ENABLE_EARLY_TERMINATION: bool = True  # Skip processing if no text
    ENABLE_PARALLEL_OCR: bool = True  # Process multiple images simultaneously
    
    # Environment
    ENVIRONMENT: str = "production"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Parse list fields from JSON strings if they are strings
        # This handles cases where environment variables are set as JSON strings
        list_fields = [
            'ALLOWED_ORIGINS', 'CORS_ALLOW_METHODS', 'CORS_ALLOW_HEADERS',
            'OCR_LANGUAGES', 'ALLOWED_IMAGE_EXTENSIONS', 'ALLOWED_HTML_EXTENSIONS',
            'ALLOWED_ARCHIVE_EXTENSIONS'
        ]
        
        for field in list_fields:
            value = getattr(self, field, None)
            if isinstance(value, str):
                try:
                    # Try to parse as JSON
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        setattr(self, field, parsed)
                except (json.JSONDecodeError, TypeError):
                    # If not JSON, try splitting by comma
                    if value.strip() == "*":
                        setattr(self, field, ["*"])
                    else:
                        # Split by comma and strip whitespace
                        setattr(self, field, [item.strip() for item in value.split(",") if item.strip()])
        
        if self.MODEL_BATCH_CHUNKS > 4:
            self.MODEL_BATCH_CHUNKS = 4
        if self.MODEL_NUM_BEAMS > 3:
            self.MODEL_NUM_BEAMS = 3

        if self.MAX_FILES_IN_ZIP > 1000:
            self.MAX_FILES_IN_ZIP = 1000

        if self.MAX_ZIP_EXTRACT_SIZE > 500 * 1024 * 1024:
            self.MAX_ZIP_EXTRACT_SIZE = 500 * 1024 * 1024


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Create global settings instance
settings = get_settings()
