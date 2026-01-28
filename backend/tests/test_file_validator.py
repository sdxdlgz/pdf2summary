"""
Unit tests for the file validation module.

Tests cover:
- File extension validation (Requirement 1.1)
- File size validation (Requirement 1.2)
- Batch size validation (Requirement 1.4)
- Descriptive error messages (Requirement 1.5)

Note: Page count validation (Requirement 1.3) is NOT tested here.
Page count exceeding is handled by MinerU API returning error code -60006.
"""

import pytest

from backend.services.file_validator import (
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    MAX_BATCH_SIZE,
    ValidationError,
    ValidationErrorType,
    ValidationResult,
    FileValidationError,
    validate_file_extension,
    validate_file_size,
    validate_batch_size,
    validate_file,
    validate_files,
    validate_batch,
)


class TestValidateFileExtension:
    """Tests for validate_file_extension function.
    
    Validates: Requirement 1.1 - Accept files with .pdf extension only
    """
    
    def test_valid_pdf_extension_lowercase(self):
        """Test that .pdf extension is accepted."""
        result = validate_file_extension("document.pdf")
        assert result is None
    
    def test_valid_pdf_extension_uppercase(self):
        """Test that .PDF extension is accepted (case insensitive)."""
        result = validate_file_extension("document.PDF")
        assert result is None
    
    def test_valid_pdf_extension_mixed_case(self):
        """Test that .Pdf extension is accepted (case insensitive)."""
        result = validate_file_extension("document.Pdf")
        assert result is None
    
    def test_invalid_extension_txt(self):
        """Test that .txt extension is rejected."""
        result = validate_file_extension("document.txt")
        assert result is not None
        assert result.error_type == ValidationErrorType.INVALID_EXTENSION
        assert "document.txt" in result.message
    
    def test_invalid_extension_docx(self):
        """Test that .docx extension is rejected."""
        result = validate_file_extension("document.docx")
        assert result is not None
        assert result.error_type == ValidationErrorType.INVALID_EXTENSION
    
    def test_invalid_extension_no_extension(self):
        """Test that files without extension are rejected."""
        result = validate_file_extension("document")
        assert result is not None
        assert result.error_type == ValidationErrorType.INVALID_EXTENSION
    
    def test_invalid_extension_pdf_in_name(self):
        """Test that 'pdf' in filename but not as extension is rejected."""
        result = validate_file_extension("pdf_document.txt")
        assert result is not None
        assert result.error_type == ValidationErrorType.INVALID_EXTENSION
    
    def test_valid_pdf_with_multiple_dots(self):
        """Test that files with multiple dots ending in .pdf are accepted."""
        result = validate_file_extension("my.report.2024.pdf")
        assert result is None
    
    def test_invalid_extension_empty_filename(self):
        """Test that empty filename is rejected."""
        result = validate_file_extension("")
        assert result is not None
        assert result.error_type == ValidationErrorType.INVALID_EXTENSION


class TestValidateFileSize:
    """Tests for validate_file_size function.
    
    Validates: Requirement 1.2 - Validate that each file size does not exceed 200MB
    """
    
    def test_valid_size_small_file(self):
        """Test that small files are accepted."""
        result = validate_file_size(1024, "document.pdf")  # 1KB
        assert result is None
    
    def test_valid_size_at_limit(self):
        """Test that files exactly at 200MB limit are accepted."""
        result = validate_file_size(MAX_FILE_SIZE_BYTES, "document.pdf")
        assert result is None
    
    def test_valid_size_just_under_limit(self):
        """Test that files just under 200MB are accepted."""
        result = validate_file_size(MAX_FILE_SIZE_BYTES - 1, "document.pdf")
        assert result is None
    
    def test_invalid_size_over_limit(self):
        """Test that files over 200MB are rejected."""
        result = validate_file_size(MAX_FILE_SIZE_BYTES + 1, "document.pdf")
        assert result is not None
        assert result.error_type == ValidationErrorType.FILE_TOO_LARGE
        assert "document.pdf" in result.message
        assert str(MAX_FILE_SIZE_MB) in result.message
    
    def test_invalid_size_much_over_limit(self):
        """Test that very large files are rejected."""
        result = validate_file_size(MAX_FILE_SIZE_BYTES * 2, "document.pdf")
        assert result is not None
        assert result.error_type == ValidationErrorType.FILE_TOO_LARGE
    
    def test_valid_size_zero(self):
        """Test that zero-size files are accepted (edge case)."""
        result = validate_file_size(0, "empty.pdf")
        assert result is None


class TestValidateBatchSize:
    """Tests for validate_batch_size function.
    
    Validates: Requirement 1.4 - Process up to 200 files in a single batch
    """
    
    def test_valid_batch_size_single_file(self):
        """Test that single file batches are accepted."""
        result = validate_batch_size(1)
        assert result is None
    
    def test_valid_batch_size_at_limit(self):
        """Test that batches with exactly 200 files are accepted."""
        result = validate_batch_size(MAX_BATCH_SIZE)
        assert result is None
    
    def test_valid_batch_size_under_limit(self):
        """Test that batches under 200 files are accepted."""
        result = validate_batch_size(MAX_BATCH_SIZE - 1)
        assert result is None
    
    def test_invalid_batch_size_over_limit(self):
        """Test that batches over 200 files are rejected."""
        result = validate_batch_size(MAX_BATCH_SIZE + 1)
        assert result is not None
        assert result.error_type == ValidationErrorType.BATCH_TOO_LARGE
        assert str(MAX_BATCH_SIZE) in result.message
    
    def test_valid_batch_size_empty(self):
        """Test that empty batches are accepted (edge case)."""
        result = validate_batch_size(0)
        assert result is None


class TestValidateFile:
    """Tests for validate_file function.
    
    Validates: Requirements 1.1, 1.2, 1.5
    """
    
    def test_valid_file(self):
        """Test that valid PDF files pass all validations."""
        errors = validate_file("document.pdf", 1024 * 1024)  # 1MB
        assert len(errors) == 0
    
    def test_invalid_extension_stops_further_validation(self):
        """Test that invalid extension returns early without checking size."""
        errors = validate_file("document.txt", 1024)
        assert len(errors) == 1
        assert errors[0].error_type == ValidationErrorType.INVALID_EXTENSION
    
    def test_valid_extension_invalid_size(self):
        """Test that size error is returned for valid extension but oversized file."""
        errors = validate_file("document.pdf", MAX_FILE_SIZE_BYTES + 1)
        assert len(errors) == 1
        assert errors[0].error_type == ValidationErrorType.FILE_TOO_LARGE
    
    def test_valid_file_at_size_limit(self):
        """Test that file at exactly 200MB passes."""
        errors = validate_file("document.pdf", MAX_FILE_SIZE_BYTES)
        assert len(errors) == 0


class TestValidateFiles:
    """Tests for validate_files function.
    
    Validates: Requirements 1.1, 1.2, 1.4, 1.5
    """
    
    def test_valid_files_single(self):
        """Test that a single valid file returns no errors."""
        errors = validate_files([("document.pdf", 1024)])
        assert len(errors) == 0
    
    def test_valid_files_multiple(self):
        """Test that multiple valid files return no errors."""
        files = [(f"document{i}.pdf", 1024) for i in range(10)]
        errors = validate_files(files)
        assert len(errors) == 0
    
    def test_invalid_batch_too_many_files(self):
        """Test that too many files returns batch error."""
        files = [(f"document{i}.pdf", 1024) for i in range(MAX_BATCH_SIZE + 1)]
        errors = validate_files(files)
        assert any(e.error_type == ValidationErrorType.BATCH_TOO_LARGE for e in errors)
    
    def test_mixed_valid_and_invalid_files(self):
        """Test that errors are collected for all invalid files."""
        files = [
            ("valid.pdf", 1024),
            ("invalid.txt", 1024),
            ("another_valid.pdf", 1024),
            ("also_invalid.docx", 1024),
        ]
        errors = validate_files(files)
        assert len(errors) == 2
        assert all(e.error_type == ValidationErrorType.INVALID_EXTENSION for e in errors)
    
    def test_empty_files_list(self):
        """Test that empty file list returns no errors."""
        errors = validate_files([])
        assert len(errors) == 0


class TestValidateBatch:
    """Tests for validate_batch function.
    
    Validates: Requirements 1.1, 1.2, 1.4, 1.5
    """
    
    def test_valid_batch_single_file(self):
        """Test that a batch with one valid file passes."""
        result = validate_batch([("document.pdf", 1024)])
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_valid_batch_multiple_files(self):
        """Test that a batch with multiple valid files passes."""
        files = [(f"document{i}.pdf", 1024) for i in range(10)]
        result = validate_batch(files)
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_invalid_batch_too_many_files(self):
        """Test that a batch with too many files fails."""
        files = [(f"document{i}.pdf", 1024) for i in range(MAX_BATCH_SIZE + 1)]
        result = validate_batch(files)
        assert result.is_valid is False
        assert any(e.error_type == ValidationErrorType.BATCH_TOO_LARGE for e in result.errors)
    
    def test_invalid_batch_mixed_errors(self):
        """Test that a batch with multiple types of errors collects all errors."""
        files = [
            ("valid.pdf", 1024),
            ("invalid.txt", 1024),
            ("another_valid.pdf", 1024),
        ]
        result = validate_batch(files)
        assert result.is_valid is False
        assert any(e.error_type == ValidationErrorType.INVALID_EXTENSION for e in result.errors)
        assert any(e.filename == "invalid.txt" for e in result.errors)
    
    def test_empty_batch(self):
        """Test that an empty batch passes validation."""
        result = validate_batch([])
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_batch_with_oversized_file(self):
        """Test that batch validation catches oversized files."""
        files = [("large_file.pdf", MAX_FILE_SIZE_BYTES + 1)]
        result = validate_batch(files)
        assert result.is_valid is False
        assert any(e.error_type == ValidationErrorType.FILE_TOO_LARGE for e in result.errors)
    
    def test_batch_at_max_size(self):
        """Test that batch with exactly 200 files passes."""
        files = [(f"document{i}.pdf", 1024) for i in range(MAX_BATCH_SIZE)]
        result = validate_batch(files)
        assert result.is_valid is True
        assert len(result.errors) == 0


class TestErrorMessages:
    """Tests for descriptive error messages.
    
    Validates: Requirement 1.5 - Return descriptive error message indicating specific validation failure
    """
    
    def test_extension_error_message_contains_filename(self):
        """Test that extension error message contains the filename."""
        error = validate_file_extension("report.docx")
        assert error is not None
        assert "report.docx" in error.message
        assert ".pdf" in error.message
    
    def test_size_error_message_contains_details(self):
        """Test that size error message contains file size and limit."""
        error = validate_file_size(MAX_FILE_SIZE_BYTES + 1024 * 1024, "large_file.pdf")
        assert error is not None
        assert "large_file.pdf" in error.message
        assert str(MAX_FILE_SIZE_MB) in error.message
    
    def test_batch_error_message_contains_details(self):
        """Test that batch size error message contains count and limit."""
        error = validate_batch_size(MAX_BATCH_SIZE + 50)
        assert error is not None
        assert str(MAX_BATCH_SIZE) in error.message
        assert str(MAX_BATCH_SIZE + 50) in error.message


class TestFileValidationError:
    """Tests for FileValidationError exception class."""
    
    def test_exception_with_single_error(self):
        """Test that exception can be created with a single error."""
        error = ValidationError(
            filename="test.txt",
            error_type=ValidationErrorType.INVALID_EXTENSION,
            message="File 'test.txt' must have a .pdf extension"
        )
        exc = FileValidationError([error])
        assert len(exc.errors) == 1
        assert "test.txt" in str(exc)
    
    def test_exception_with_multiple_errors(self):
        """Test that exception can be created with multiple errors."""
        errors = [
            ValidationError(
                filename="test.txt",
                error_type=ValidationErrorType.INVALID_EXTENSION,
                message="File 'test.txt' must have a .pdf extension"
            ),
            ValidationError(
                filename="large.pdf",
                error_type=ValidationErrorType.FILE_TOO_LARGE,
                message="File 'large.pdf' exceeds size limit"
            ),
        ]
        exc = FileValidationError(errors)
        assert len(exc.errors) == 2
        assert "test.txt" in str(exc)
        assert "large.pdf" in str(exc)


class TestConstants:
    """Tests for validation constants."""
    
    def test_max_file_size_mb(self):
        """Test that MAX_FILE_SIZE_MB is 200."""
        assert MAX_FILE_SIZE_MB == 200
    
    def test_max_file_size_bytes(self):
        """Test that MAX_FILE_SIZE_BYTES is correctly calculated."""
        assert MAX_FILE_SIZE_BYTES == 200 * 1024 * 1024
    
    def test_max_batch_size(self):
        """Test that MAX_BATCH_SIZE is 200."""
        assert MAX_BATCH_SIZE == 200
