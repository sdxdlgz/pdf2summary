"""
File validation module for the Research Report Processor.

This module provides validation functions for PDF files including:
- File extension validation (.pdf only)
- File size validation (max 200MB)
- Batch size validation (max 200 files)

Note: Page count validation (600 pages limit, Requirement 1.3) is NOT implemented locally.
Page count exceeding is handled by MinerU API returning error code -60006.

Requirements: 1.1, 1.2, 1.4, 1.5
"""

from dataclasses import dataclass
from enum import Enum


class ValidationErrorType(str, Enum):
    """Types of validation errors."""
    INVALID_EXTENSION = "invalid_extension"
    FILE_TOO_LARGE = "file_too_large"
    BATCH_TOO_LARGE = "batch_too_large"


# Validation constants
MAX_FILE_SIZE_MB = 200
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024  # 200MB in bytes
MAX_BATCH_SIZE = 200


@dataclass
class ValidationError:
    """Represents a validation error for a file.
    
    Attributes:
        filename: The name of the file that failed validation
        error_type: The type of validation error (e.g., "invalid_extension", 
                    "file_too_large", "batch_too_large")
        message: A descriptive error message for the user
    """
    filename: str
    error_type: ValidationErrorType
    message: str


@dataclass
class ValidationResult:
    """Represents the result of validating one or more files.
    
    Attributes:
        is_valid: True if all validations passed, False otherwise
        errors: List of validation errors encountered
    """
    is_valid: bool
    errors: list[ValidationError]


class FileValidationError(Exception):
    """Exception raised when file validation fails.
    
    Attributes:
        errors: List of ValidationError objects describing the failures
    """
    def __init__(self, errors: list[ValidationError]):
        self.errors = errors
        messages = [e.message for e in errors]
        super().__init__("; ".join(messages))


def validate_file_extension(filename: str) -> ValidationError | None:
    """Validate that the file has a .pdf extension.
    
    Args:
        filename: The name of the file to validate
        
    Returns:
        ValidationError if the file doesn't have .pdf extension, None otherwise
        
    Requirement: 1.1 - Accept files with .pdf extension only
    """
    if not filename.lower().endswith('.pdf'):
        return ValidationError(
            filename=filename,
            error_type=ValidationErrorType.INVALID_EXTENSION,
            message=f"File '{filename}' must have a .pdf extension"
        )
    return None


def validate_file_size(size: int, filename: str) -> ValidationError | None:
    """Validate that the file size does not exceed the maximum limit.
    
    Args:
        size: The size of the file in bytes
        filename: The name of the file to validate
        
    Returns:
        ValidationError if the file exceeds 200MB, None otherwise
        
    Requirement: 1.2 - Validate that each file size does not exceed 200MB
    """
    if size > MAX_FILE_SIZE_BYTES:
        size_mb = size / (1024 * 1024)
        return ValidationError(
            filename=filename,
            error_type=ValidationErrorType.FILE_TOO_LARGE,
            message=f"File '{filename}' is {size_mb:.2f}MB, which exceeds the maximum allowed size of {MAX_FILE_SIZE_MB}MB"
        )
    return None


def validate_batch_size(file_count: int) -> ValidationError | None:
    """Validate that the batch size does not exceed the maximum limit.
    
    Args:
        file_count: The number of files in the batch
        
    Returns:
        ValidationError if the batch exceeds 200 files, None otherwise
        
    Requirement: 1.4 - Process up to 200 files in a single batch
    """
    if file_count > MAX_BATCH_SIZE:
        return ValidationError(
            filename="batch",
            error_type=ValidationErrorType.BATCH_TOO_LARGE,
            message=f"Batch contains {file_count} files, which exceeds the maximum allowed {MAX_BATCH_SIZE} files"
        )
    return None


def validate_file(filename: str, size: int) -> list[ValidationError]:
    """Validate a single file for extension and size.
    
    This function checks:
    1. File extension is .pdf
    2. File size does not exceed 200MB
    
    Note: Page count validation is handled by MinerU API (error code -60006).
    
    Args:
        filename: The name of the file to validate
        size: The size of the file in bytes
        
    Returns:
        List of ValidationError objects for any failed validations
        
    Requirements: 1.1, 1.2, 1.5
    """
    errors: list[ValidationError] = []
    
    # Validate file extension
    extension_error = validate_file_extension(filename)
    if extension_error:
        errors.append(extension_error)
        # If extension is invalid, skip size validation as it's not a PDF
        return errors
    
    # Validate file size
    size_error = validate_file_size(size, filename)
    if size_error:
        errors.append(size_error)
    
    return errors


def validate_files(files: list[tuple[str, int]]) -> list[ValidationError]:
    """Validate a list of files and return all validation errors.
    
    This function checks:
    1. Batch size does not exceed 200 files
    2. Each file passes individual validation (extension, size)
    
    Note: Page count validation is handled by MinerU API (error code -60006).
    
    Args:
        files: List of tuples containing (filename, size_in_bytes) for each file
        
    Returns:
        List of all ValidationError objects encountered
        
    Requirements: 1.1, 1.2, 1.4, 1.5
    """
    errors: list[ValidationError] = []
    
    # Validate batch size first
    batch_error = validate_batch_size(len(files))
    if batch_error:
        errors.append(batch_error)
    
    # Validate each file individually
    for filename, size in files:
        file_errors = validate_file(filename, size)
        errors.extend(file_errors)
    
    return errors


def validate_batch(files: list[tuple[str, int]]) -> ValidationResult:
    """Validate an entire batch of files.
    
    This function checks:
    1. Batch size does not exceed 200 files
    2. Each file passes individual validation (extension, size)
    
    Note: Page count validation is handled by MinerU API (error code -60006).
    
    Args:
        files: List of tuples containing (filename, size_in_bytes) for each file
        
    Returns:
        ValidationResult with is_valid=True if all validations pass,
        or is_valid=False with a list of all validation errors
        
    Requirements: 1.1, 1.2, 1.4, 1.5
    """
    errors = validate_files(files)
    
    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors
    )
