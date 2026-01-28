"""
Unit tests for the MinerU client.

Tests cover:
- request_upload_urls method
- upload_file method
- get_batch_results method
- Error handling and notifications
- API parameter correctness
"""

import pytest
from aioresponses import aioresponses

from backend.models import FileInfo, MineruTaskState
from backend.services.mineru_client import MineruClient, MineruClientError


class TestMineruClientInit:
    """Tests for MineruClient initialization."""
    
    def test_init_with_valid_token(self):
        """Test client initialization with valid token."""
        client = MineruClient(api_token="test-token")
        assert client.api_token == "test-token"
        assert client.base_url == "https://mineru.net"
    
    def test_init_with_custom_base_url(self):
        """Test client initialization with custom base URL."""
        client = MineruClient(
            api_token="test-token",
            base_url="https://custom.api.com",
        )
        assert client.base_url == "https://custom.api.com"
    
    def test_init_with_empty_token_raises_error(self):
        """Test that empty token raises ValueError."""
        with pytest.raises(ValueError, match="api_token is required"):
            MineruClient(api_token="")
    
    def test_init_with_error_callback(self):
        """Test client initialization with error callback."""
        callback_called = []
        
        def error_callback(batch_id: str, error_msg: str):
            callback_called.append((batch_id, error_msg))
        
        client = MineruClient(
            api_token="test-token",
            error_callback=error_callback,
        )
        assert client.error_callback is not None


class TestGetApiParameters:
    """Tests for API parameter configuration."""
    
    def test_api_parameters_correctness(self):
        """
        Test that API parameters are correctly configured.
        
        **Validates: Requirements 2.5, 2.6, 2.7, 2.8**
        """
        client = MineruClient(api_token="test-token")
        params = client.get_api_parameters()
        
        # Requirement 2.5: enable_table=true
        assert params["enable_table"] is True
        
        # Requirement 2.6: enable_formula=true
        assert params["enable_formula"] is True
        
        # Requirement 2.7: extra_formats=["docx"]
        assert params["extra_formats"] == ["docx"]
        
        # Requirement 2.8: model_version="vlm"
        assert params["model_version"] == "vlm"


class TestRequestUploadUrls:
    """Tests for request_upload_urls method."""
    
    @pytest.fixture
    def client(self):
        """Create a MinerU client for testing."""
        return MineruClient(api_token="test-token")
    
    @pytest.fixture
    def sample_files(self):
        """Create sample file info list."""
        return [
            FileInfo(name="test1.pdf", data_id="data-123", size=1000),
            FileInfo(name="test2.pdf", data_id="data-456", size=2000),
        ]
    
    @pytest.mark.asyncio
    async def test_request_upload_urls_success(self, client, sample_files):
        """Test successful upload URL request."""
        with aioresponses() as m:
            m.post(
                "https://mineru.net/api/v4/file-urls/batch",
                payload={
                    "code": 0,
                    "data": {
                        "batch_id": "batch-abc123",
                        "file_urls": [
                            "https://storage.example.com/upload1",
                            "https://storage.example.com/upload2",
                        ],
                    },
                    "msg": "ok",
                },
            )
            
            response = await client.request_upload_urls(sample_files)
            
            assert response.batch_id == "batch-abc123"
            assert len(response.file_urls) == 2
            assert response.file_urls[0] == "https://storage.example.com/upload1"
    
    @pytest.mark.asyncio
    async def test_request_upload_urls_includes_correct_parameters(
        self, client, sample_files
    ):
        """
        Test that request includes correct API parameters.
        
        **Validates: Requirements 2.5, 2.6, 2.7, 2.8**
        """
        with aioresponses() as m:
            m.post(
                "https://mineru.net/api/v4/file-urls/batch",
                payload={
                    "code": 0,
                    "data": {
                        "batch_id": "batch-abc123",
                        "file_urls": ["url1", "url2"],
                    },
                    "msg": "ok",
                },
            )
            
            await client.request_upload_urls(sample_files)
            
            # Verify the request was made with correct parameters
            request = list(m.requests.values())[0][0]
            # The payload is in the request kwargs
            # We verify through the get_api_parameters method
            params = client.get_api_parameters()
            assert params["enable_table"] is True
            assert params["enable_formula"] is True
            assert params["extra_formats"] == ["docx"]
            assert params["model_version"] == "vlm"
    
    @pytest.mark.asyncio
    async def test_request_upload_urls_empty_files_raises_error(self, client):
        """Test that empty file list raises error."""
        with pytest.raises(MineruClientError, match="No files provided"):
            await client.request_upload_urls([])
    
    @pytest.mark.asyncio
    async def test_request_upload_urls_api_error(self, client, sample_files):
        """Test handling of API error response."""
        with aioresponses() as m:
            m.post(
                "https://mineru.net/api/v4/file-urls/batch",
                payload={
                    "code": -1,
                    "msg": "Invalid token",
                },
            )
            
            with pytest.raises(MineruClientError) as exc_info:
                await client.request_upload_urls(sample_files)
            
            assert "Invalid token" in str(exc_info.value)
            assert exc_info.value.error_code == "-1"
    
    @pytest.mark.asyncio
    async def test_request_upload_urls_url_count_mismatch(
        self, client, sample_files
    ):
        """Test handling of URL count mismatch."""
        with aioresponses() as m:
            m.post(
                "https://mineru.net/api/v4/file-urls/batch",
                payload={
                    "code": 0,
                    "data": {
                        "batch_id": "batch-abc123",
                        "file_urls": ["url1"],  # Only 1 URL for 2 files
                    },
                    "msg": "ok",
                },
            )
            
            with pytest.raises(MineruClientError, match="URL count mismatch"):
                await client.request_upload_urls(sample_files)
    
    @pytest.mark.asyncio
    async def test_request_upload_urls_error_callback_called(self, sample_files):
        """Test that error callback is called on API error."""
        callback_calls = []
        
        def error_callback(batch_id: str, error_msg: str):
            callback_calls.append((batch_id, error_msg))
        
        client = MineruClient(
            api_token="test-token",
            error_callback=error_callback,
        )
        
        with aioresponses() as m:
            m.post(
                "https://mineru.net/api/v4/file-urls/batch",
                payload={
                    "code": -1,
                    "msg": "Token expired",
                },
            )
            
            with pytest.raises(MineruClientError):
                await client.request_upload_urls(sample_files)
        
        assert len(callback_calls) == 1
        assert "Token expired" in callback_calls[0][1]


class TestUploadFile:
    """Tests for upload_file method."""
    
    @pytest.fixture
    def client(self):
        """Create a MinerU client for testing."""
        return MineruClient(api_token="test-token")
    
    @pytest.mark.asyncio
    async def test_upload_file_success(self, client):
        """
        Test successful file upload.
        
        **Validates: Requirements 2.2, 2.3**
        """
        with aioresponses() as m:
            m.put(
                "https://storage.example.com/upload",
                status=200,
            )
            
            result = await client.upload_file(
                upload_url="https://storage.example.com/upload",
                file_content=b"PDF content here",
            )
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_upload_file_empty_url_raises_error(self, client):
        """Test that empty URL raises error."""
        with pytest.raises(MineruClientError, match="upload_url is required"):
            await client.upload_file(
                upload_url="",
                file_content=b"content",
            )
    
    @pytest.mark.asyncio
    async def test_upload_file_empty_content_raises_error(self, client):
        """Test that empty content raises error."""
        with pytest.raises(MineruClientError, match="file_content is required"):
            await client.upload_file(
                upload_url="https://example.com/upload",
                file_content=b"",
            )
    
    @pytest.mark.asyncio
    async def test_upload_file_server_error(self, client):
        """Test handling of server error during upload."""
        with aioresponses() as m:
            m.put(
                "https://storage.example.com/upload",
                status=500,
            )
            
            with pytest.raises(MineruClientError, match="status 500"):
                await client.upload_file(
                    upload_url="https://storage.example.com/upload",
                    file_content=b"content",
                )
    
    @pytest.mark.asyncio
    async def test_upload_file_accepts_various_status_codes(self, client):
        """Test that various success status codes are accepted."""
        for status in [200, 201, 204]:
            with aioresponses() as m:
                m.put(
                    "https://storage.example.com/upload",
                    status=status,
                )
                
                result = await client.upload_file(
                    upload_url="https://storage.example.com/upload",
                    file_content=b"content",
                )
                
                assert result is True


class TestGetBatchResults:
    """Tests for get_batch_results method."""
    
    @pytest.fixture
    def client(self):
        """Create a MinerU client for testing."""
        return MineruClient(api_token="test-token")
    
    @pytest.mark.asyncio
    async def test_get_batch_results_success(self, client):
        """Test successful batch results retrieval."""
        with aioresponses() as m:
            m.get(
                "https://mineru.net/api/v4/extract-results/batch/batch-123",
                payload={
                    "code": 0,
                    "data": {
                        "batch_id": "batch-123",
                        "extract_result": [
                            {
                                "file_name": "test.pdf",
                                "data_id": "data-abc",
                                "state": "done",
                                "full_zip_url": "https://download.example.com/result.zip",
                                "err_msg": "",
                                "extract_progress": {
                                    "extracted_pages": 10,
                                    "total_pages": 10,
                                },
                            },
                        ],
                    },
                    "msg": "ok",
                },
            )
            
            response = await client.get_batch_results("batch-123")
            
            assert response.batch_id == "batch-123"
            assert len(response.extract_result) == 1
            
            result = response.extract_result[0]
            assert result.file_name == "test.pdf"
            assert result.data_id == "data-abc"
            assert result.state == MineruTaskState.DONE
            assert result.full_zip_url == "https://download.example.com/result.zip"
            assert result.extracted_pages == 10
            assert result.total_pages == 10
    
    @pytest.mark.asyncio
    async def test_get_batch_results_various_states(self, client):
        """Test parsing of various task states."""
        states_to_test = [
            ("waiting-file", MineruTaskState.WAITING_FILE),
            ("pending", MineruTaskState.PENDING),
            ("running", MineruTaskState.RUNNING),
            ("converting", MineruTaskState.CONVERTING),
            ("done", MineruTaskState.DONE),
            ("failed", MineruTaskState.FAILED),
        ]
        
        for state_str, expected_state in states_to_test:
            with aioresponses() as m:
                m.get(
                    "https://mineru.net/api/v4/extract-results/batch/batch-123",
                    payload={
                        "code": 0,
                        "data": {
                            "batch_id": "batch-123",
                            "extract_result": [
                                {
                                    "file_name": "test.pdf",
                                    "state": state_str,
                                },
                            ],
                        },
                        "msg": "ok",
                    },
                )
                
                response = await client.get_batch_results("batch-123")
                assert response.extract_result[0].state == expected_state
    
    @pytest.mark.asyncio
    async def test_get_batch_results_empty_batch_id_raises_error(self, client):
        """Test that empty batch_id raises error."""
        with pytest.raises(MineruClientError, match="batch_id is required"):
            await client.get_batch_results("")
    
    @pytest.mark.asyncio
    async def test_get_batch_results_api_error(self, client):
        """Test handling of API error response."""
        with aioresponses() as m:
            m.get(
                "https://mineru.net/api/v4/extract-results/batch/batch-123",
                payload={
                    "code": -1,
                    "msg": "Batch not found",
                },
            )
            
            with pytest.raises(MineruClientError) as exc_info:
                await client.get_batch_results("batch-123")
            
            assert "Batch not found" in str(exc_info.value)
            assert exc_info.value.batch_id == "batch-123"
    
    @pytest.mark.asyncio
    async def test_get_batch_results_with_error_message(self, client):
        """Test parsing of failed task with error message."""
        with aioresponses() as m:
            m.get(
                "https://mineru.net/api/v4/extract-results/batch/batch-123",
                payload={
                    "code": 0,
                    "data": {
                        "batch_id": "batch-123",
                        "extract_result": [
                            {
                                "file_name": "test.pdf",
                                "state": "failed",
                                "err_msg": "File corrupted",
                            },
                        ],
                    },
                    "msg": "ok",
                },
            )
            
            response = await client.get_batch_results("batch-123")
            
            result = response.extract_result[0]
            assert result.state == MineruTaskState.FAILED
            assert result.err_msg == "File corrupted"


class TestUploadFilesBatch:
    """Tests for upload_files_batch convenience method."""
    
    @pytest.fixture
    def client(self):
        """Create a MinerU client for testing."""
        return MineruClient(api_token="test-token")
    
    @pytest.fixture
    def sample_files(self):
        """Create sample file info list."""
        return [
            FileInfo(name="test1.pdf", data_id="data-123", size=1000),
            FileInfo(name="test2.pdf", data_id="data-456", size=2000),
        ]
    
    @pytest.mark.asyncio
    async def test_upload_files_batch_success(self, client, sample_files):
        """Test successful batch upload workflow."""
        with aioresponses() as m:
            # Mock the batch upload URL request
            m.post(
                "https://mineru.net/api/v4/file-urls/batch",
                payload={
                    "code": 0,
                    "data": {
                        "batch_id": "batch-abc123",
                        "file_urls": [
                            "https://storage.example.com/upload1",
                            "https://storage.example.com/upload2",
                        ],
                    },
                    "msg": "ok",
                },
            )
            
            # Mock the file uploads
            m.put("https://storage.example.com/upload1", status=200)
            m.put("https://storage.example.com/upload2", status=200)
            
            file_contents = [b"content1", b"content2"]
            
            response = await client.upload_files_batch(sample_files, file_contents)
            
            assert response.batch_id == "batch-abc123"
    
    def test_upload_files_batch_length_mismatch(self, client, sample_files):
        """Test that mismatched lengths raise ValueError."""
        with pytest.raises(ValueError, match="must have the same length"):
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                client.upload_files_batch(sample_files, [b"only one"])
            )
