"""
Tests for the FastAPI main application.

Tests cover:
- POST /api/upload endpoint
- GET /api/tasks/{task_id} endpoint
- GET /api/download/{task_id}/{file_type} endpoint
- GET /health endpoint
"""

import io
import os
import tempfile
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, status
from fastapi.testclient import TestClient

from backend.models import (
    FileInfo,
    OutputFileType,
    ProgressUpdate,
    TaskStage,
    TaskStatus,
)


# Create a test app without the lifespan that requires configuration
@pytest.fixture
def test_app():
    """Create a test FastAPI app without lifespan."""
    from backend.api.main import (
        upload_files,
        get_task_status,
        download_file,
        health_check,
        UploadResponse,
        HealthResponse,
        ErrorResponse,
    )
    
    app = FastAPI()
    
    # Register routes manually without lifespan
    app.post("/api/upload", response_model=UploadResponse)(upload_files)
    app.get("/api/tasks/{task_id}", response_model=TaskStatus)(get_task_status)
    app.get("/api/download/{task_id}/{file_type}")(download_file)
    app.get("/health", response_model=HealthResponse)(health_check)
    
    return app


@pytest.fixture
def mock_redis_client():
    """Create a mock Redis client."""
    mock = AsyncMock()
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    mock.publish = AsyncMock(return_value=1)
    return mock


@pytest.fixture
def mock_task_manager(mock_redis_client):
    """Create a mock TaskManager."""
    mock = MagicMock()
    mock.redis_client = mock_redis_client
    mock.create_task = AsyncMock(return_value="task-test123")
    mock.get_task_status = AsyncMock(return_value=None)
    mock.process_task = AsyncMock()
    return mock


@pytest.fixture
def mock_file_storage():
    """Create a mock FileStorage."""
    mock = MagicMock()
    mock.save_upload = MagicMock(return_value="/tmp/test/uploads/test.pdf")
    mock.get_output_file = MagicMock(return_value=None)
    return mock


@pytest.fixture
def client(test_app, mock_redis_client, mock_task_manager, mock_file_storage):
    """Create a test client with mocked dependencies."""
    from backend.api.main import (
        get_redis_client,
        get_task_manager,
        get_file_storage,
    )
    
    # Override dependencies
    test_app.dependency_overrides[get_redis_client] = lambda: mock_redis_client
    test_app.dependency_overrides[get_task_manager] = lambda: mock_task_manager
    test_app.dependency_overrides[get_file_storage] = lambda: mock_file_storage
    
    with TestClient(test_app) as client:
        yield client
    
    # Clear overrides
    test_app.dependency_overrides.clear()


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""
    
    def test_health_check_returns_healthy(self, client):
        """Test that health check returns healthy status."""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
    
    def test_health_check_response_model(self, client):
        """Test that health check response matches expected model."""
        from backend.api.main import HealthResponse
        
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        # Validate response can be parsed as HealthResponse
        health = HealthResponse(**response.json())
        assert health.status == "healthy"


class TestUploadEndpoint:
    """Tests for POST /api/upload endpoint."""
    
    def test_upload_single_pdf_file(self, client, mock_task_manager, mock_file_storage):
        """Test uploading a single PDF file."""
        # Create a test PDF file
        pdf_content = b"%PDF-1.4 test content"
        files = [("files", ("test.pdf", io.BytesIO(pdf_content), "application/pdf"))]
        
        response = client.post("/api/upload", files=files)
        
        # Note: Test app returns 200, actual app returns 201
        assert response.status_code in (status.HTTP_200_OK, status.HTTP_201_CREATED)
        data = response.json()
        assert "task_id" in data
        assert data["file_count"] == 1
        assert "message" in data
    
    def test_upload_multiple_pdf_files(self, client, mock_task_manager, mock_file_storage):
        """Test uploading multiple PDF files."""
        pdf_content = b"%PDF-1.4 test content"
        files = [
            ("files", ("test1.pdf", io.BytesIO(pdf_content), "application/pdf")),
            ("files", ("test2.pdf", io.BytesIO(pdf_content), "application/pdf")),
        ]
        
        response = client.post("/api/upload", files=files)
        
        # Note: Test app returns 200, actual app returns 201
        assert response.status_code in (status.HTTP_200_OK, status.HTTP_201_CREATED)
        data = response.json()
        assert data["file_count"] == 2
    
    def test_upload_no_files_returns_error(self, client):
        """Test that uploading no files returns an error."""
        response = client.post("/api/upload", files=[])
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_upload_non_pdf_file_returns_error(self, client, mock_task_manager, mock_file_storage):
        """Test that uploading non-PDF files returns validation error."""
        txt_content = b"This is a text file"
        files = [("files", ("test.txt", io.BytesIO(txt_content), "text/plain"))]
        
        response = client.post("/api/upload", files=files)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "pdf" in response.json()["detail"].lower()


class TestTaskStatusEndpoint:
    """Tests for GET /api/tasks/{task_id} endpoint."""
    
    def test_get_task_status_found(self, client, mock_task_manager):
        """Test getting status of an existing task."""
        # Setup mock to return a task status
        task_status = TaskStatus(
            task_id="task-test123",
            stage=TaskStage.PARSING,
            files=[FileInfo(name="test.pdf", data_id="data-123", size=1024)],
            progress=ProgressUpdate(
                task_id="task-test123",
                stage=TaskStage.PARSING,
                progress=50,
                total=100,
                percentage=25.0,
                message="Parsing in progress",
                timestamp=datetime.now(),
            ),
            outputs=None,
            error=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        mock_task_manager.get_task_status = AsyncMock(return_value=task_status)
        
        response = client.get("/api/tasks/task-test123")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["task_id"] == "task-test123"
        assert data["stage"] == "parsing"
    
    def test_get_task_status_not_found(self, client, mock_task_manager):
        """Test getting status of a non-existent task."""
        mock_task_manager.get_task_status = AsyncMock(return_value=None)
        
        response = client.get("/api/tasks/nonexistent-task")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        assert "not found" in response.json()["detail"].lower()


class TestDownloadEndpoint:
    """Tests for GET /api/download/{task_id}/{file_type} endpoint."""
    
    def test_download_invalid_file_type(self, client, mock_task_manager, mock_file_storage):
        """Test downloading with invalid file type returns error."""
        response = client.get("/api/download/task-123/invalid_type")
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "invalid file type" in response.json()["detail"].lower()
    
    def test_download_task_not_found(self, client, mock_task_manager, mock_file_storage):
        """Test downloading from non-existent task returns error."""
        mock_task_manager.get_task_status = AsyncMock(return_value=None)
        
        response = client.get(f"/api/download/nonexistent/original_md")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_download_file_not_found(self, client, mock_task_manager, mock_file_storage):
        """Test downloading non-existent file returns error."""
        # Task exists but file doesn't
        task_status = TaskStatus(
            task_id="task-test123",
            stage=TaskStage.COMPLETED,
            files=[FileInfo(name="test.pdf", data_id="data-123", size=1024)],
            progress=ProgressUpdate(
                task_id="task-test123",
                stage=TaskStage.COMPLETED,
                progress=100,
                total=100,
                percentage=100.0,
                message="Complete",
                timestamp=datetime.now(),
            ),
            outputs=None,
            error=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        mock_task_manager.get_task_status = AsyncMock(return_value=task_status)
        mock_file_storage.get_output_file = MagicMock(return_value=None)
        
        response = client.get(f"/api/download/task-test123/original_md")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_download_valid_file(self, client, mock_task_manager, mock_file_storage):
        """Test downloading a valid output file."""
        # Create a temporary file to serve
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("# Test Markdown Content")
            temp_path = f.name
        
        try:
            # Task exists and file exists
            task_status = TaskStatus(
                task_id="task-test123",
                stage=TaskStage.COMPLETED,
                files=[FileInfo(name="test.pdf", data_id="data-123", size=1024)],
                progress=ProgressUpdate(
                    task_id="task-test123",
                    stage=TaskStage.COMPLETED,
                    progress=100,
                    total=100,
                    percentage=100.0,
                    message="Complete",
                    timestamp=datetime.now(),
                ),
                outputs={"original_md": temp_path},
                error=None,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            mock_task_manager.get_task_status = AsyncMock(return_value=task_status)
            mock_file_storage.get_output_file = MagicMock(return_value=temp_path)
            
            response = client.get(f"/api/download/task-test123/original_md")
            
            assert response.status_code == status.HTTP_200_OK
            assert "# Test Markdown Content" in response.text
        finally:
            # Cleanup
            os.unlink(temp_path)


class TestValidFileTypes:
    """Tests for valid file type values."""
    
    def test_all_output_file_types_are_valid(self, client, mock_task_manager, mock_file_storage):
        """Test that all OutputFileType values are accepted."""
        # Setup task status
        task_status = TaskStatus(
            task_id="task-test123",
            stage=TaskStage.COMPLETED,
            files=[FileInfo(name="test.pdf", data_id="data-123", size=1024)],
            progress=ProgressUpdate(
                task_id="task-test123",
                stage=TaskStage.COMPLETED,
                progress=100,
                total=100,
                percentage=100.0,
                message="Complete",
                timestamp=datetime.now(),
            ),
            outputs=None,
            error=None,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        mock_task_manager.get_task_status = AsyncMock(return_value=task_status)
        mock_file_storage.get_output_file = MagicMock(return_value=None)
        
        # Test each valid file type - should not return 400 (invalid type)
        for file_type in OutputFileType:
            response = client.get(f"/api/download/task-test123/{file_type.value}")
            # Should be 404 (file not found) not 400 (invalid type)
            assert response.status_code == status.HTTP_404_NOT_FOUND


class TestUploadResponseModel:
    """Tests for upload response model."""
    
    def test_upload_response_contains_required_fields(self, client, mock_task_manager, mock_file_storage):
        """Test that upload response contains all required fields."""
        pdf_content = b"%PDF-1.4 test content"
        files = [("files", ("test.pdf", io.BytesIO(pdf_content), "application/pdf"))]
        
        response = client.post("/api/upload", files=files)
        
        # Note: Test app returns 200, actual app returns 201
        assert response.status_code in (status.HTTP_200_OK, status.HTTP_201_CREATED)
        data = response.json()
        
        # Check all required fields are present
        assert "task_id" in data
        assert "message" in data
        assert "file_count" in data
        
        # Check field types
        assert isinstance(data["task_id"], str)
        assert isinstance(data["message"], str)
        assert isinstance(data["file_count"], int)
