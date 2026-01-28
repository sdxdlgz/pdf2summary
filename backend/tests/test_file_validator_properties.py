"""
Property-based tests for file validation module.

**Property 1: File Validation Completeness**
*For any* uploaded file, the validation function SHALL reject files that do not
have .pdf extension, exceed 200MB in size, or when batch size exceeds 200 files,
and SHALL return a specific error message for each validation failure.

**Validates: Requirements 1.1, 1.2, 1.4, 1.5**

Note: Page count validation (Requirement 1.3) is NOT tested here.
Page count exceeding is handled by MinerU API returning error code -60006.

Uses Hypothesis for property-based testing with at least 100 iterations per test.
"""

import pytest
from hypothesis import given, settings, strategies as st, assume

from backend.services.file_validator import (
    MAX_FILE_SIZE_BYTES,
    MAX_FILE_SIZE_MB,
    MAX_BATCH_SIZE,
    ValidationErrorType,
    validate_file_extension,
    validate_file_size,
    validate_batch_size,
    validate_file,
    validate_batch,
)


# =============================================================================
# Strategies for generating test data
# =============================================================================

# Common file extensions that are NOT .pdf
NON_PDF_EXTENSIONS = [
    ".txt", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff",
    ".html", ".htm", ".xml", ".json", ".csv",
    ".zip", ".rar", ".7z", ".tar", ".gz",
    ".mp3", ".mp4", ".avi", ".mov", ".wav",
    ".exe", ".dll", ".so", ".py", ".js", ".ts",
    "", ".PDF.txt", ".pdf.bak", ".pdfx",
]

# Strategy for generating valid PDF filenames
valid_pdf_filename = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N"),
        whitelist_characters="_-.",
    ),
    min_size=1,
    max_size=50,
).map(lambda s: s.strip() + ".pdf").filter(lambda s: len(s) > 4)

# Strategy for generating filenames with non-PDF extensions
non_pdf_filename = st.one_of(
    # Filename with common non-PDF extension
    st.tuples(
        st.text(
            alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
            min_size=1,
            max_size=30,
        ),
        st.sampled_from(NON_PDF_EXTENSIONS),
    ).map(lambda t: t[0].strip() + t[1] if t[0].strip() else "file" + t[1]),
    # Filename without any extension
    st.text(
        alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
        min_size=1,
        max_size=30,
    ).filter(lambda s: "." not in s and s.strip()),
    # Filename with 'pdf' in name but not as extension
    st.text(
        alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
        min_size=1,
        max_size=20,
    ).map(lambda s: f"pdf_{s.strip()}.txt" if s.strip() else "pdf_file.txt"),
)

# Strategy for generating valid file sizes (0 to MAX_FILE_SIZE_BYTES)
valid_file_size = st.integers(min_value=0, max_value=MAX_FILE_SIZE_BYTES)

# Strategy for generating invalid file sizes (> MAX_FILE_SIZE_BYTES)
invalid_file_size = st.integers(
    min_value=MAX_FILE_SIZE_BYTES + 1,
    max_value=MAX_FILE_SIZE_BYTES * 10,  # Up to 2GB for testing
)

# Strategy for generating valid batch sizes (1 to MAX_BATCH_SIZE)
valid_batch_size = st.integers(min_value=1, max_value=MAX_BATCH_SIZE)

# Strategy for generating invalid batch sizes (> MAX_BATCH_SIZE)
invalid_batch_size = st.integers(
    min_value=MAX_BATCH_SIZE + 1,
    max_value=MAX_BATCH_SIZE * 5,  # Up to 1000 files for testing
)


# =============================================================================
# Property-based tests for File Extension Validation
# =============================================================================

class TestFileExtensionValidationProperty:
    """
    Property-based tests for file extension validation.
    
    **Property 1: File Validation Completeness (Extension part)**
    **Validates: Requirements 1.1, 1.5**
    """

    @settings(max_examples=100)
    @given(filename=non_pdf_filename)
    def test_non_pdf_extension_is_rejected(self, filename: str):
        """
        Property: For any filename that does not end with .pdf extension,
        the validation function SHALL reject it with INVALID_EXTENSION error.
        
        **Validates: Requirements 1.1, 1.5**
        """
        # Ensure the filename doesn't end with .pdf (case insensitive)
        assume(not filename.lower().endswith(".pdf"))
        assume(len(filename) > 0)
        
        result = validate_file_extension(filename)
        
        # SHALL reject files without .pdf extension
        assert result is not None, (
            f"File '{filename}' should be rejected (no .pdf extension)"
        )
        
        # SHALL return INVALID_EXTENSION error type
        assert result.error_type == ValidationErrorType.INVALID_EXTENSION, (
            f"Error type should be INVALID_EXTENSION, got {result.error_type}"
        )
        
        # SHALL return specific error message containing filename
        assert filename in result.message, (
            f"Error message should contain filename '{filename}'. Got: {result.message}"
        )

    @settings(max_examples=100)
    @given(filename=valid_pdf_filename)
    def test_pdf_extension_is_accepted(self, filename: str):
        """
        Property: For any filename that ends with .pdf extension (case insensitive),
        the validation function SHALL accept it.
        
        **Validates: Requirements 1.1**
        """
        result = validate_file_extension(filename)
        
        # SHALL accept files with .pdf extension
        assert result is None, (
            f"File '{filename}' should be accepted (has .pdf extension). "
            f"Got error: {result}"
        )

    @settings(max_examples=100)
    @given(
        base_name=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-."),
            min_size=1,
            max_size=30,
        ),
        case_variant=st.sampled_from([".pdf", ".PDF", ".Pdf", ".pDf", ".pdF", ".PDf", ".pDF", ".PdF"]),
    )
    def test_pdf_extension_case_insensitive(self, base_name: str, case_variant: str):
        """
        Property: For any case variation of .pdf extension, the validation
        function SHALL accept the file.
        
        **Validates: Requirements 1.1**
        """
        assume(base_name.strip())
        filename = base_name.strip() + case_variant
        
        result = validate_file_extension(filename)
        
        # SHALL accept any case variation of .pdf
        assert result is None, (
            f"File '{filename}' should be accepted (case insensitive .pdf). "
            f"Got error: {result}"
        )


# =============================================================================
# Property-based tests for File Size Validation
# =============================================================================

class TestFileSizeValidationProperty:
    """
    Property-based tests for file size validation.
    
    **Property 1: File Validation Completeness (Size part)**
    **Validates: Requirements 1.2, 1.5**
    """

    @settings(max_examples=100)
    @given(size=invalid_file_size, filename=valid_pdf_filename)
    def test_oversized_file_is_rejected(self, size: int, filename: str):
        """
        Property: For any file size exceeding 200MB, the validation function
        SHALL reject it with FILE_TOO_LARGE error.
        
        **Validates: Requirements 1.2, 1.5**
        """
        result = validate_file_size(size, filename)
        
        # SHALL reject files exceeding 200MB
        assert result is not None, (
            f"File with size {size} bytes ({size / (1024*1024):.2f}MB) should be rejected"
        )
        
        # SHALL return FILE_TOO_LARGE error type
        assert result.error_type == ValidationErrorType.FILE_TOO_LARGE, (
            f"Error type should be FILE_TOO_LARGE, got {result.error_type}"
        )
        
        # SHALL return specific error message containing filename
        assert filename in result.message, (
            f"Error message should contain filename '{filename}'. Got: {result.message}"
        )
        
        # SHALL return error message containing size limit
        assert str(MAX_FILE_SIZE_MB) in result.message, (
            f"Error message should contain size limit '{MAX_FILE_SIZE_MB}MB'. "
            f"Got: {result.message}"
        )

    @settings(max_examples=100)
    @given(size=valid_file_size, filename=valid_pdf_filename)
    def test_valid_size_is_accepted(self, size: int, filename: str):
        """
        Property: For any file size not exceeding 200MB, the validation
        function SHALL accept it.
        
        **Validates: Requirements 1.2**
        """
        result = validate_file_size(size, filename)
        
        # SHALL accept files not exceeding 200MB
        assert result is None, (
            f"File with size {size} bytes ({size / (1024*1024):.2f}MB) should be accepted. "
            f"Got error: {result}"
        )

    @settings(max_examples=100)
    @given(filename=valid_pdf_filename)
    def test_boundary_size_exactly_at_limit_is_accepted(self, filename: str):
        """
        Property: For any file with size exactly at 200MB limit, the validation
        function SHALL accept it.
        
        **Validates: Requirements 1.2**
        """
        result = validate_file_size(MAX_FILE_SIZE_BYTES, filename)
        
        # SHALL accept files exactly at the limit
        assert result is None, (
            f"File with size exactly at limit ({MAX_FILE_SIZE_BYTES} bytes) should be accepted. "
            f"Got error: {result}"
        )

    @settings(max_examples=100)
    @given(
        extra_bytes=st.integers(min_value=1, max_value=1024 * 1024 * 100),  # 1 byte to 100MB extra
        filename=valid_pdf_filename,
    )
    def test_boundary_size_just_over_limit_is_rejected(self, extra_bytes: int, filename: str):
        """
        Property: For any file with size just over 200MB limit, the validation
        function SHALL reject it.
        
        **Validates: Requirements 1.2, 1.5**
        """
        size = MAX_FILE_SIZE_BYTES + extra_bytes
        result = validate_file_size(size, filename)
        
        # SHALL reject files over the limit
        assert result is not None, (
            f"File with size {size} bytes (limit + {extra_bytes}) should be rejected"
        )
        assert result.error_type == ValidationErrorType.FILE_TOO_LARGE


# =============================================================================
# Property-based tests for Batch Size Validation
# =============================================================================

class TestBatchSizeValidationProperty:
    """
    Property-based tests for batch size validation.
    
    **Property 1: File Validation Completeness (Batch size part)**
    **Validates: Requirements 1.4, 1.5**
    """

    @settings(max_examples=100)
    @given(file_count=invalid_batch_size)
    def test_oversized_batch_is_rejected(self, file_count: int):
        """
        Property: For any batch with more than 200 files, the validation
        function SHALL reject it with BATCH_TOO_LARGE error.
        
        **Validates: Requirements 1.4, 1.5**
        """
        result = validate_batch_size(file_count)
        
        # SHALL reject batches exceeding 200 files
        assert result is not None, (
            f"Batch with {file_count} files should be rejected"
        )
        
        # SHALL return BATCH_TOO_LARGE error type
        assert result.error_type == ValidationErrorType.BATCH_TOO_LARGE, (
            f"Error type should be BATCH_TOO_LARGE, got {result.error_type}"
        )
        
        # SHALL return error message containing the actual count
        assert str(file_count) in result.message, (
            f"Error message should contain file count '{file_count}'. "
            f"Got: {result.message}"
        )
        
        # SHALL return error message containing the limit
        assert str(MAX_BATCH_SIZE) in result.message, (
            f"Error message should contain batch limit '{MAX_BATCH_SIZE}'. "
            f"Got: {result.message}"
        )

    @settings(max_examples=100)
    @given(file_count=valid_batch_size)
    def test_valid_batch_size_is_accepted(self, file_count: int):
        """
        Property: For any batch with 200 or fewer files, the validation
        function SHALL accept it.
        
        **Validates: Requirements 1.4**
        """
        result = validate_batch_size(file_count)
        
        # SHALL accept batches not exceeding 200 files
        assert result is None, (
            f"Batch with {file_count} files should be accepted. Got error: {result}"
        )

    @settings(max_examples=100)
    @given(extra_files=st.integers(min_value=1, max_value=500))
    def test_boundary_batch_just_over_limit_is_rejected(self, extra_files: int):
        """
        Property: For any batch with count just over 200 files, the validation
        function SHALL reject it.
        
        **Validates: Requirements 1.4, 1.5**
        """
        file_count = MAX_BATCH_SIZE + extra_files
        result = validate_batch_size(file_count)
        
        # SHALL reject batches over the limit
        assert result is not None, (
            f"Batch with {file_count} files (limit + {extra_files}) should be rejected"
        )
        assert result.error_type == ValidationErrorType.BATCH_TOO_LARGE


# =============================================================================
# Property-based tests for Combined Validation (validate_batch)
# =============================================================================

class TestCombinedValidationProperty:
    """
    Property-based tests for combined file and batch validation.
    
    **Property 1: File Validation Completeness**
    **Validates: Requirements 1.1, 1.2, 1.4, 1.5**
    """

    @settings(max_examples=100)
    @given(
        num_files=st.integers(min_value=1, max_value=50),
        file_size=valid_file_size,
    )
    def test_valid_batch_passes_validation(self, num_files: int, file_size: int):
        """
        Property: For any batch of valid PDF files (correct extension, valid size,
        within batch limit), the validation function SHALL accept all files.
        
        **Validates: Requirements 1.1, 1.2, 1.4**
        """
        # Generate a batch of valid files
        files = [(f"document_{i}.pdf", file_size) for i in range(num_files)]
        
        result = validate_batch(files)
        
        # SHALL accept valid batches
        assert result.is_valid is True, (
            f"Valid batch of {num_files} files should pass validation. "
            f"Errors: {[e.message for e in result.errors]}"
        )
        assert len(result.errors) == 0

    @settings(max_examples=100)
    @given(
        num_valid=st.integers(min_value=0, max_value=10),
        num_invalid_ext=st.integers(min_value=1, max_value=5),
        file_size=valid_file_size,
    )
    def test_batch_with_invalid_extensions_is_rejected(
        self, num_valid: int, num_invalid_ext: int, file_size: int
    ):
        """
        Property: For any batch containing files without .pdf extension,
        the validation function SHALL reject those files with specific errors.
        
        **Validates: Requirements 1.1, 1.5**
        """
        # Generate mix of valid and invalid files
        files = []
        files.extend([(f"valid_{i}.pdf", file_size) for i in range(num_valid)])
        invalid_names = [f"invalid_{i}.txt" for i in range(num_invalid_ext)]
        files.extend([(name, file_size) for name in invalid_names])
        
        result = validate_batch(files)
        
        # SHALL reject batch with invalid files
        assert result.is_valid is False, (
            f"Batch with {num_invalid_ext} invalid extension files should fail"
        )
        
        # SHALL have errors for each invalid file
        extension_errors = [
            e for e in result.errors
            if e.error_type == ValidationErrorType.INVALID_EXTENSION
        ]
        assert len(extension_errors) == num_invalid_ext, (
            f"Should have {num_invalid_ext} extension errors, got {len(extension_errors)}"
        )
        
        # Each error SHALL contain the filename
        for name in invalid_names:
            assert any(name in e.message for e in extension_errors), (
                f"Error message should mention '{name}'"
            )

    @settings(max_examples=100)
    @given(
        num_valid=st.integers(min_value=0, max_value=10),
        num_oversized=st.integers(min_value=1, max_value=5),
        valid_size=valid_file_size,
        invalid_size=invalid_file_size,
    )
    def test_batch_with_oversized_files_is_rejected(
        self, num_valid: int, num_oversized: int, valid_size: int, invalid_size: int
    ):
        """
        Property: For any batch containing files exceeding 200MB,
        the validation function SHALL reject those files with specific errors.
        
        **Validates: Requirements 1.2, 1.5**
        """
        # Generate mix of valid and oversized files
        files = []
        files.extend([(f"valid_{i}.pdf", valid_size) for i in range(num_valid)])
        oversized_names = [f"large_{i}.pdf" for i in range(num_oversized)]
        files.extend([(name, invalid_size) for name in oversized_names])
        
        result = validate_batch(files)
        
        # SHALL reject batch with oversized files
        assert result.is_valid is False, (
            f"Batch with {num_oversized} oversized files should fail"
        )
        
        # SHALL have errors for each oversized file
        size_errors = [
            e for e in result.errors
            if e.error_type == ValidationErrorType.FILE_TOO_LARGE
        ]
        assert len(size_errors) == num_oversized, (
            f"Should have {num_oversized} size errors, got {len(size_errors)}"
        )
        
        # Each error SHALL contain the filename
        for name in oversized_names:
            assert any(name in e.message for e in size_errors), (
                f"Error message should mention '{name}'"
            )

    @settings(max_examples=100)
    @given(
        extra_files=st.integers(min_value=1, max_value=100),
        file_size=valid_file_size,
    )
    def test_batch_exceeding_limit_is_rejected(self, extra_files: int, file_size: int):
        """
        Property: For any batch with more than 200 files, the validation
        function SHALL reject it with BATCH_TOO_LARGE error.
        
        **Validates: Requirements 1.4, 1.5**
        """
        file_count = MAX_BATCH_SIZE + extra_files
        files = [(f"document_{i}.pdf", file_size) for i in range(file_count)]
        
        result = validate_batch(files)
        
        # SHALL reject oversized batch
        assert result.is_valid is False, (
            f"Batch with {file_count} files should fail"
        )
        
        # SHALL have BATCH_TOO_LARGE error
        batch_errors = [
            e for e in result.errors
            if e.error_type == ValidationErrorType.BATCH_TOO_LARGE
        ]
        assert len(batch_errors) == 1, (
            f"Should have exactly 1 batch size error, got {len(batch_errors)}"
        )
        
        # Error SHALL contain the count and limit
        assert str(file_count) in batch_errors[0].message
        assert str(MAX_BATCH_SIZE) in batch_errors[0].message


# =============================================================================
# Property-based tests for Error Message Quality
# =============================================================================

class TestErrorMessageQualityProperty:
    """
    Property-based tests for error message quality and specificity.
    
    **Property 1: File Validation Completeness (Error message part)**
    **Validates: Requirements 1.5**
    """

    @settings(max_examples=100)
    @given(filename=non_pdf_filename)
    def test_extension_error_message_is_descriptive(self, filename: str):
        """
        Property: For any file with invalid extension, the error message
        SHALL contain the filename and indicate the required extension.
        
        **Validates: Requirements 1.5**
        """
        assume(not filename.lower().endswith(".pdf"))
        assume(len(filename) > 0)
        
        result = validate_file_extension(filename)
        assume(result is not None)
        
        # Error message SHALL contain filename
        assert filename in result.message, (
            f"Error message should contain filename '{filename}'"
        )
        
        # Error message SHALL indicate required extension
        assert ".pdf" in result.message.lower(), (
            f"Error message should mention .pdf extension"
        )

    @settings(max_examples=100)
    @given(size=invalid_file_size, filename=valid_pdf_filename)
    def test_size_error_message_is_descriptive(self, size: int, filename: str):
        """
        Property: For any oversized file, the error message SHALL contain
        the filename, actual size, and size limit.
        
        **Validates: Requirements 1.5**
        """
        result = validate_file_size(size, filename)
        assume(result is not None)
        
        # Error message SHALL contain filename
        assert filename in result.message, (
            f"Error message should contain filename '{filename}'"
        )
        
        # Error message SHALL contain size limit
        assert str(MAX_FILE_SIZE_MB) in result.message, (
            f"Error message should contain size limit '{MAX_FILE_SIZE_MB}'"
        )

    @settings(max_examples=100)
    @given(file_count=invalid_batch_size)
    def test_batch_error_message_is_descriptive(self, file_count: int):
        """
        Property: For any oversized batch, the error message SHALL contain
        the actual count and the batch limit.
        
        **Validates: Requirements 1.5**
        """
        result = validate_batch_size(file_count)
        assume(result is not None)
        
        # Error message SHALL contain actual count
        assert str(file_count) in result.message, (
            f"Error message should contain actual count '{file_count}'"
        )
        
        # Error message SHALL contain batch limit
        assert str(MAX_BATCH_SIZE) in result.message, (
            f"Error message should contain batch limit '{MAX_BATCH_SIZE}'"
        )
