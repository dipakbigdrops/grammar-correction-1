"""
Pydantic Models for Request/Response Validation
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class InputType(str, Enum):
    """Supported input types"""
    IMAGE = "image"
    HTML = "html"


class ProcessingStatus(str, Enum):
    """Task processing status"""
    PENDING = "PENDING"
    STARTED = "STARTED"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    RETRY = "RETRY"


class CorrectionDetail(BaseModel):
    """Details of a single correction"""
    original_word: str = Field(..., description="Original word before correction")
    corrected_word: str = Field(..., description="Corrected word")
    original_context: str = Field(..., description="Context around original word")
    corrected_context: str = Field(..., description="Context around corrected word")


class ProcessRequest(BaseModel):
    """Request model for processing endpoint"""
    input_type: InputType = Field(..., description="Type of input (image or html)")
    async_processing: bool = Field(
        default=True,
        description="Whether to process asynchronously"
    )


class TaskResponse(BaseModel):
    """Response model for async task submission"""
    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Current task status")
    message: str = Field(..., description="Human-readable message")
    estimated_completion_seconds: int = Field(
        default=30,
        description="Estimated time to completion"
    )
    result: Optional[Dict[str, Any]] = Field(default=None, description="Processing result when status is SUCCESS")


class TaskStatusResponse(BaseModel):
    """Response model for task status check"""
    task_id: str
    status: ProcessingStatus
    progress: Optional[int] = Field(None, ge=0, le=100, description="Progress percentage")
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


class ProcessResult(BaseModel):
    """Result model for completed processing"""
    input_type: InputType
    original_text: str
    corrected_text: str
    corrections: List[CorrectionDetail]
    corrections_count: int
    output_file: Optional[str] = None
    processing_time_seconds: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    grammar_model_loaded: bool
    ocr_available: bool
    beautifulsoup_available: bool
    image_reconstruction_available: bool
    html_reconstruction_available: bool


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    task_id: Optional[str] = None