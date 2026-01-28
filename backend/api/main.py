"""
FastAPI main application for the Research Report Processor.

This module provides the REST API endpoints for:
- POST /api/upload: Upload PDF files for processing
- GET /api/tasks/{task_id}: Get task status
- GET /api/download/{task_id}/{file_type}: Download output files
- WS /ws/{task_id}: WebSocket progress updates
- GET /health: Health check endpoint

Requirements:
- 7.6: Generate download links valid for a configurable duration
- 8.2: Broadcast progress updates via WebSocket
- 8.7: Broadcast error messages via WebSocket
- 10.6: Provide health check endpoints for container orchestration
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from redis.asyncio import Redis

from backend.config import Settings, get_cached_settings
from backend.models import OutputFileType, TaskStatus
from backend.services.file_storage import FileStorage
from backend.services.file_validator import validate_batch, ValidationResult
from backend.services.task_manager import TaskManager
from backend.api.websocket import websocket_manager


logger = logging.getLogger(__name__)


# Response models
class UploadResponse(BaseModel):
    """Response model for file upload endpoint."""
    task_id: str
    message: str
    file_count: int


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Response model for error responses."""
    detail: str


# Global instances (initialized in lifespan)
_redis_client: Redis | None = None
_task_manager: TaskManager | None = None
_file_storage: FileStorage | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Initializes and cleans up resources:
    - Redis client connection
    - Task manager
    - File storage
    """
    global _redis_client, _task_manager, _file_storage
    
    settings = get_cached_settings()
    
    # Initialize Redis client
    _redis_client = Redis.from_url(settings.REDIS_URL)
    
    # Initialize file storage
    _file_storage = FileStorage(settings.STORAGE_PATH)
    
    # Initialize task manager
    _task_manager = TaskManager(
        redis_client=_redis_client,
        file_storage=_file_storage,
    )
    
    logger.info("Application started")
    
    yield
    
    # Cleanup
    if _redis_client:
        await _redis_client.close()
    
    logger.info("Application shutdown")


# Create FastAPI app
app = FastAPI(
    title="Research Report Processor",
    description="API for processing PDF research reports with translation and summarization",
    version="1.0.0",
    lifespan=lifespan,
)


# Dependency functions
def get_settings() -> Settings:
    """Get application settings."""
    return get_cached_settings()


def get_redis_client() -> Redis:
    """Get Redis client instance."""
    if _redis_client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis client not initialized",
        )
    return _redis_client


def get_task_manager() -> TaskManager:
    """Get task manager instance."""
    if _task_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Task manager not initialized",
        )
    return _task_manager


def get_file_storage() -> FileStorage:
    """Get file storage instance."""
    if _file_storage is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="File storage not initialized",
        )
    return _file_storage


@app.post(
    "/api/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Validation error"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Upload PDF files for processing",
    description="Upload one or more PDF files to be processed for translation and summarization.",
)
async def upload_files(
    files: Annotated[list[UploadFile], File(description="PDF files to upload")],
    background_tasks: BackgroundTasks,
    task_manager: Annotated[TaskManager, Depends(get_task_manager)],
    file_storage: Annotated[FileStorage, Depends(get_file_storage)],
) -> UploadResponse:
    """
    Upload PDF files for processing.
    
    This endpoint:
    1. Validates uploaded files (extension, size, batch size)
    2. Creates a new processing task
    3. Saves files to storage
    4. Starts background processing
    5. Returns the task_id for status tracking
    
    Args:
        files: List of PDF files to upload
        background_tasks: FastAPI background tasks
        task_manager: Task manager instance
        file_storage: File storage instance
        
    Returns:
        UploadResponse with task_id and file count
        
    Raises:
        HTTPException: If validation fails or task creation fails
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided",
        )
    
    # Prepare file info for validation
    file_info_list: list[tuple[str, int]] = []
    for file in files:
        # Read file content to get size
        content = await file.read()
        await file.seek(0)  # Reset for later reading
        
        filename = file.filename or "unknown.pdf"
        file_info_list.append((filename, len(content)))
    
    # Validate files using FileValidator
    validation_result: ValidationResult = validate_batch(file_info_list)
    
    if not validation_result.is_valid:
        # Collect all error messages
        error_messages = [error.message for error in validation_result.errors]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="; ".join(error_messages),
        )
    
    # Create task
    try:
        task_id = await task_manager.create_task(files)
    except Exception as e:
        logger.error("Failed to create task: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create task: {str(e)}",
        ) from e
    
    # Save uploaded files to storage
    for file in files:
        content = await file.read()
        filename = file.filename or "unknown.pdf"
        
        try:
            file_storage.save_upload(task_id, filename, content)
        except Exception as e:
            logger.error("Failed to save file %s: %s", filename, e)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save file {filename}: {str(e)}",
            ) from e
    
    # Start background processing
    background_tasks.add_task(task_manager.process_task, task_id)
    
    logger.info(
        "Created task %s with %d files",
        task_id,
        len(files),
    )
    
    return UploadResponse(
        task_id=task_id,
        message="Files uploaded successfully, processing started",
        file_count=len(files),
    )


@app.get(
    "/api/tasks/{task_id}",
    response_model=TaskStatus,
    responses={
        404: {"model": ErrorResponse, "description": "Task not found"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Get task status",
    description="Get the current status of a processing task.",
)
async def get_task_status(
    task_id: str,
    task_manager: Annotated[TaskManager, Depends(get_task_manager)],
) -> TaskStatus:
    """
    Get the current status of a task.
    
    Args:
        task_id: The unique task identifier
        task_manager: Task manager instance
        
    Returns:
        TaskStatus with current progress and stage
        
    Raises:
        HTTPException: If task is not found
    """
    task_status = await task_manager.get_task_status(task_id)
    
    if task_status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )
    
    return task_status


@app.get(
    "/api/download/{task_id}/{file_type}",
    response_class=FileResponse,
    responses={
        404: {"model": ErrorResponse, "description": "File not found"},
        400: {"model": ErrorResponse, "description": "Invalid file type"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    },
    summary="Download output file",
    description="Download a specific output file from a completed task.",
)
async def download_file(
    task_id: str,
    file_type: str,
    file_storage: Annotated[FileStorage, Depends(get_file_storage)],
    task_manager: Annotated[TaskManager, Depends(get_task_manager)],
) -> FileResponse:
    """
    Download an output file from a completed task.
    
    Args:
        task_id: The unique task identifier
        file_type: The type of output file (from OutputFileType enum)
        file_storage: File storage instance
        task_manager: Task manager instance
        
    Returns:
        FileResponse with the requested file
        
    Raises:
        HTTPException: If task not found, invalid file type, or file not found
        
    Requirement: 7.6 - Generate download links valid for a configurable duration
    """
    # Validate file_type
    valid_types = {t.value for t in OutputFileType}
    if file_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type '{file_type}'. Must be one of: {', '.join(valid_types)}",
        )
    
    # Check if task exists
    task_status = await task_manager.get_task_status(task_id)
    if task_status is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found",
        )
    
    # Get the output file path
    file_path = file_storage.get_output_file(task_id, file_type)
    
    if file_path is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Output file '{file_type}' not found for task {task_id}",
        )
    
    # Verify file exists
    path = Path(file_path)
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Output file '{file_type}' not found for task {task_id}",
        )
    
    # Determine media type based on file extension
    media_type = "application/octet-stream"
    suffix = path.suffix.lower()
    if suffix == ".md":
        media_type = "text/markdown"
    elif suffix == ".docx":
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    elif suffix == ".txt":
        media_type = "text/plain"
    
    # Get filename for download
    filename = path.name
    
    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=filename,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if the service is healthy and running.",
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint for container orchestration.
    
    Returns:
        HealthResponse indicating service status
        
    Requirement: 10.6 - Provide health check endpoints for container orchestration
    """
    return HealthResponse(
        status="healthy",
        version="1.0.0",
    )


@app.websocket("/ws/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """
    WebSocket endpoint for real-time progress updates.
    
    Clients can connect to this endpoint to receive real-time progress
    updates for a specific task. The connection remains open until
    the client disconnects or the task completes.
    
    Args:
        websocket: The WebSocket connection
        task_id: The unique task identifier to monitor
        
    Requirements:
        - 8.2: Broadcast progress updates via WebSocket
        - 8.7: Broadcast error messages via WebSocket
    """
    await websocket_manager.connect(task_id, websocket)
    try:
        # Keep the connection alive and handle incoming messages
        while True:
            # Wait for any message from client (ping/pong or close)
            # This keeps the connection alive
            try:
                data = await websocket.receive_text()
                # Client can send "ping" to keep connection alive
                if data == "ping":
                    await websocket.send_text("pong")
            except WebSocketDisconnect:
                break
    except Exception as e:
        logger.warning("WebSocket error for task %s: %s", task_id, e)
    finally:
        await websocket_manager.disconnect(task_id, websocket)


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )
