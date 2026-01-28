"""
Unit tests for the file storage service.

Tests cover:
- Saving uploaded files
- Saving output files
- Getting output paths
- Downloading and extracting ZIP files
- Error handling

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 7.1, 7.2, 7.3, 7.4, 7.5
"""

import io
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.models import ExtractedFiles, OutputFileType
from backend.services.file_storage import FileStorage, FileStorageError


class TestFileStorageInit:
    """Tests for FileStorage initialization."""
    
    def test_init_creates_base_directory(self, tmp_path: Path):
        """Test that initialization creates the base directory."""
        base_path = tmp_path / "storage"
        assert not base_path.exists()
        
        storage = FileStorage(str(base_path))
        
        assert base_path.exists()
        assert storage.base_path == base_path
    
    def test_init_with_existing_directory(self, tmp_path: Path):
        """Test initialization with an existing directory."""
        base_path = tmp_path / "storage"
        base_path.mkdir()
        
        storage = FileStorage(str(base_path))
        
        assert base_path.exists()
        assert storage.base_path == base_path


class TestSaveUpload:
    """Tests for save_upload method."""
    
    def test_save_upload_creates_file(self, tmp_path: Path):
        """Test that save_upload creates the file correctly."""
        storage = FileStorage(str(tmp_path))
        task_id = "task-123"
        filename = "test.pdf"
        content = b"PDF content here"
        
        result = storage.save_upload(task_id, filename, content)
        
        assert Path(result).exists()
        assert Path(result).read_bytes() == content
        assert filename in result
    
    def test_save_upload_creates_task_directories(self, tmp_path: Path):
        """Test that save_upload creates necessary directories."""
        storage = FileStorage(str(tmp_path))
        task_id = "task-456"
        
        storage.save_upload(task_id, "test.pdf", b"content")
        
        task_dir = tmp_path / task_id
        assert (task_dir / "uploads").exists()
        assert (task_dir / "outputs").exists()
        assert (task_dir / "extracted").exists()
    
    def test_save_upload_sanitizes_filename(self, tmp_path: Path):
        """Test that save_upload sanitizes filenames to prevent path traversal."""
        storage = FileStorage(str(tmp_path))
        task_id = "task-789"
        malicious_filename = "../../../etc/passwd"
        
        result = storage.save_upload(task_id, malicious_filename, b"content")
        
        # Should only use the base name
        assert "passwd" in result
        assert "../" not in result
        # File should be in the uploads directory
        assert task_id in result
        assert "uploads" in result
    
    def test_save_upload_handles_special_characters(self, tmp_path: Path):
        """Test handling of filenames with special characters."""
        storage = FileStorage(str(tmp_path))
        task_id = "task-special"
        filename = "report (2024).pdf"
        
        result = storage.save_upload(task_id, filename, b"content")
        
        assert Path(result).exists()


class TestSaveOutput:
    """Tests for save_output method."""
    
    def test_save_output_creates_file(self, tmp_path: Path):
        """Test that save_output creates the file correctly."""
        storage = FileStorage(str(tmp_path))
        task_id = "task-output-1"
        file_type = OutputFileType.ORIGINAL_MD.value
        filename = "report.md"
        content = b"# Markdown content"
        
        result = storage.save_output(task_id, file_type, filename, content)
        
        assert Path(result).exists()
        assert Path(result).read_bytes() == content
    
    def test_save_output_prefixes_with_file_type(self, tmp_path: Path):
        """Test that output files are prefixed with file type."""
        storage = FileStorage(str(tmp_path))
        task_id = "task-output-2"
        file_type = OutputFileType.BILINGUAL_MD.value
        filename = "report.md"
        
        result = storage.save_output(task_id, file_type, filename, b"content")
        
        assert f"{file_type}_" in Path(result).name
    
    def test_save_output_all_file_types(self, tmp_path: Path):
        """Test saving all output file types."""
        storage = FileStorage(str(tmp_path))
        task_id = "task-output-all"
        
        for file_type in OutputFileType:
            result = storage.save_output(
                task_id,
                file_type.value,
                f"test.{file_type.value}",
                b"content",
            )
            assert Path(result).exists()
    
    def test_save_output_invalid_file_type(self, tmp_path: Path):
        """Test that invalid file type raises error."""
        storage = FileStorage(str(tmp_path))
        task_id = "task-invalid"
        
        with pytest.raises(FileStorageError) as exc_info:
            storage.save_output(task_id, "invalid_type", "test.txt", b"content")
        
        assert "Invalid file type" in str(exc_info.value)


class TestGetOutputPath:
    """Tests for get_output_path method."""
    
    def test_get_output_path_returns_directory(self, tmp_path: Path):
        """Test that get_output_path returns the outputs directory."""
        storage = FileStorage(str(tmp_path))
        task_id = "task-path-1"
        
        # Create task directories first
        storage.save_upload(task_id, "test.pdf", b"content")
        
        result = storage.get_output_path(task_id, OutputFileType.ORIGINAL_MD.value)
        
        assert "outputs" in result
        assert task_id in result


class TestGetOutputFile:
    """Tests for get_output_file method."""
    
    def test_get_output_file_finds_file(self, tmp_path: Path):
        """Test that get_output_file finds saved output files."""
        storage = FileStorage(str(tmp_path))
        task_id = "task-find-1"
        file_type = OutputFileType.ORIGINAL_MD.value
        
        # Save an output file
        storage.save_output(task_id, file_type, "report.md", b"content")
        
        result = storage.get_output_file(task_id, file_type)
        
        assert result is not None
        assert Path(result).exists()
    
    def test_get_output_file_returns_none_when_not_found(self, tmp_path: Path):
        """Test that get_output_file returns None when file doesn't exist."""
        storage = FileStorage(str(tmp_path))
        task_id = "task-notfound"
        
        result = storage.get_output_file(task_id, OutputFileType.ORIGINAL_MD.value)
        
        assert result is None


class TestDownloadAndExtractZip:
    """Tests for download_and_extract_zip method."""
    
    @pytest.fixture
    def sample_zip_content(self) -> bytes:
        """Create a sample ZIP file with MD, DOCX, and images."""
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add Markdown file with table structure (Requirement 4.4)
            md_content = """# Report Title

## Introduction

This is the introduction.

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
| Data 4   | Data 5   | Data 6   |

## Images

![Figure 1](images/figure1.png)

## Conclusion

This is the conclusion.
"""
            zf.writestr("report.md", md_content.encode())
            
            # Add DOCX file (simulated)
            zf.writestr("report.docx", b"DOCX content placeholder")
            
            # Add images in subdirectory (Requirement 4.3)
            zf.writestr("images/figure1.png", b"PNG image data")
            zf.writestr("images/figure2.jpg", b"JPG image data")
        
        return buffer.getvalue()
    
    @pytest.mark.asyncio
    async def test_extract_zip_extracts_markdown(
        self, tmp_path: Path, sample_zip_content: bytes
    ):
        """Test that ZIP extraction extracts Markdown files (Requirement 4.1)."""
        storage = FileStorage(str(tmp_path))
        task_id = "task-extract-md"
        
        # Create a local ZIP file for testing
        zip_path = tmp_path / "test.zip"
        zip_path.write_bytes(sample_zip_content)
        
        # Mock the download to use our local file
        async def mock_download(url: str, destination: Path):
            destination.write_bytes(sample_zip_content)
        
        with patch.object(storage, "_download_file", mock_download):
            result = await storage.download_and_extract_zip(
                "http://example.com/result.zip", task_id
            )
        
        assert result.markdown_path is not None
        assert Path(result.markdown_path).exists()
        assert Path(result.markdown_path).suffix == ".md"
    
    @pytest.mark.asyncio
    async def test_extract_zip_extracts_docx(
        self, tmp_path: Path, sample_zip_content: bytes
    ):
        """Test that ZIP extraction extracts DOCX files (Requirement 4.2)."""
        storage = FileStorage(str(tmp_path))
        task_id = "task-extract-docx"
        
        async def mock_download(url: str, destination: Path):
            destination.write_bytes(sample_zip_content)
        
        with patch.object(storage, "_download_file", mock_download):
            result = await storage.download_and_extract_zip(
                "http://example.com/result.zip", task_id
            )
        
        assert result.docx_path is not None
        assert Path(result.docx_path).exists()
        assert Path(result.docx_path).suffix == ".docx"
    
    @pytest.mark.asyncio
    async def test_extract_zip_preserves_images(
        self, tmp_path: Path, sample_zip_content: bytes
    ):
        """Test that ZIP extraction preserves images (Requirement 4.3)."""
        storage = FileStorage(str(tmp_path))
        task_id = "task-extract-images"
        
        async def mock_download(url: str, destination: Path):
            destination.write_bytes(sample_zip_content)
        
        with patch.object(storage, "_download_file", mock_download):
            result = await storage.download_and_extract_zip(
                "http://example.com/result.zip", task_id
            )
        
        assert len(result.images) == 2
        for image_path in result.images:
            assert Path(image_path).exists()
    
    @pytest.mark.asyncio
    async def test_extract_zip_preserves_directory_structure(
        self, tmp_path: Path, sample_zip_content: bytes
    ):
        """Test that extraction preserves directory structure for image references."""
        storage = FileStorage(str(tmp_path))
        task_id = "task-extract-structure"
        
        async def mock_download(url: str, destination: Path):
            destination.write_bytes(sample_zip_content)
        
        with patch.object(storage, "_download_file", mock_download):
            result = await storage.download_and_extract_zip(
                "http://example.com/result.zip", task_id
            )
        
        # Check that images are in the images subdirectory
        for image_path in result.images:
            assert "images" in image_path
    
    @pytest.mark.asyncio
    async def test_extract_zip_preserves_table_structure(
        self, tmp_path: Path, sample_zip_content: bytes
    ):
        """Test that extraction preserves table structures in Markdown (Requirement 4.4)."""
        storage = FileStorage(str(tmp_path))
        task_id = "task-extract-tables"
        
        async def mock_download(url: str, destination: Path):
            destination.write_bytes(sample_zip_content)
        
        with patch.object(storage, "_download_file", mock_download):
            result = await storage.download_and_extract_zip(
                "http://example.com/result.zip", task_id
            )
        
        # Read the extracted Markdown and verify table is preserved
        md_content = Path(result.markdown_path).read_text()
        assert "| Column 1 |" in md_content
        assert "|----------|" in md_content
        assert "| Data 1   |" in md_content
    
    @pytest.mark.asyncio
    async def test_extract_zip_no_markdown_raises_error(self, tmp_path: Path):
        """Test that extraction raises error when no Markdown file is found."""
        storage = FileStorage(str(tmp_path))
        task_id = "task-no-md"
        
        # Create ZIP without Markdown
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("report.docx", b"DOCX only")
        
        async def mock_download(url: str, destination: Path):
            destination.write_bytes(buffer.getvalue())
        
        with patch.object(storage, "_download_file", mock_download):
            with pytest.raises(FileStorageError) as exc_info:
                await storage.download_and_extract_zip(
                    "http://example.com/result.zip", task_id
                )
        
        assert "No Markdown file found" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_extract_zip_handles_nested_structure(self, tmp_path: Path):
        """Test extraction of ZIP with nested directory structure."""
        storage = FileStorage(str(tmp_path))
        task_id = "task-nested"
        
        # Create ZIP with nested structure
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("output/report.md", b"# Nested Report")
            zf.writestr("output/images/fig1.png", b"PNG data")
        
        async def mock_download(url: str, destination: Path):
            destination.write_bytes(buffer.getvalue())
        
        with patch.object(storage, "_download_file", mock_download):
            result = await storage.download_and_extract_zip(
                "http://example.com/result.zip", task_id
            )
        
        assert result.markdown_path is not None
        assert Path(result.markdown_path).exists()


class TestCleanupTask:
    """Tests for cleanup_task method."""
    
    def test_cleanup_removes_task_directory(self, tmp_path: Path):
        """Test that cleanup removes all task files."""
        storage = FileStorage(str(tmp_path))
        task_id = "task-cleanup"
        
        # Create some files
        storage.save_upload(task_id, "test.pdf", b"content")
        storage.save_output(task_id, OutputFileType.ORIGINAL_MD.value, "report.md", b"md")
        
        task_dir = tmp_path / task_id
        assert task_dir.exists()
        
        result = storage.cleanup_task(task_id)
        
        assert result is True
        assert not task_dir.exists()
    
    def test_cleanup_nonexistent_task(self, tmp_path: Path):
        """Test cleanup of non-existent task returns True."""
        storage = FileStorage(str(tmp_path))
        
        result = storage.cleanup_task("nonexistent-task")
        
        assert result is True


class TestListTaskOutputs:
    """Tests for list_task_outputs method."""
    
    def test_list_outputs_returns_all_files(self, tmp_path: Path):
        """Test that list_task_outputs returns all saved output files."""
        storage = FileStorage(str(tmp_path))
        task_id = "task-list"
        
        # Save multiple output files
        storage.save_output(task_id, OutputFileType.ORIGINAL_MD.value, "report.md", b"md")
        storage.save_output(task_id, OutputFileType.ORIGINAL_DOCX.value, "report.docx", b"docx")
        storage.save_output(task_id, OutputFileType.SUMMARY.value, "summary.md", b"summary")
        
        result = storage.list_task_outputs(task_id)
        
        assert len(result) == 3
        assert OutputFileType.ORIGINAL_MD.value in result
        assert OutputFileType.ORIGINAL_DOCX.value in result
        assert OutputFileType.SUMMARY.value in result
    
    def test_list_outputs_empty_task(self, tmp_path: Path):
        """Test list_task_outputs for task with no outputs."""
        storage = FileStorage(str(tmp_path))
        
        result = storage.list_task_outputs("empty-task")
        
        assert result == {}


class TestGetExtractedFiles:
    """Tests for get_extracted_markdown and get_extracted_docx methods."""
    
    @pytest.mark.asyncio
    async def test_get_extracted_markdown(self, tmp_path: Path):
        """Test getting extracted Markdown file path."""
        storage = FileStorage(str(tmp_path))
        task_id = "task-get-md"
        
        # Create ZIP with Markdown
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("report.md", b"# Report")
        
        async def mock_download(url: str, destination: Path):
            destination.write_bytes(buffer.getvalue())
        
        with patch.object(storage, "_download_file", mock_download):
            await storage.download_and_extract_zip("http://example.com/test.zip", task_id)
        
        result = storage.get_extracted_markdown(task_id)
        
        assert result is not None
        assert Path(result).exists()
    
    @pytest.mark.asyncio
    async def test_get_extracted_docx(self, tmp_path: Path):
        """Test getting extracted DOCX file path."""
        storage = FileStorage(str(tmp_path))
        task_id = "task-get-docx"
        
        # Create ZIP with both MD and DOCX
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as zf:
            zf.writestr("report.md", b"# Report")
            zf.writestr("report.docx", b"DOCX content")
        
        async def mock_download(url: str, destination: Path):
            destination.write_bytes(buffer.getvalue())
        
        with patch.object(storage, "_download_file", mock_download):
            await storage.download_and_extract_zip("http://example.com/test.zip", task_id)
        
        result = storage.get_extracted_docx(task_id)
        
        assert result is not None
        assert Path(result).exists()
