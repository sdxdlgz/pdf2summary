"""
Property-based tests for file storage service, specifically ZIP content extraction.

**Property 8: ZIP Content Extraction**
*For any* MinerU result ZIP file, the extraction SHALL produce a Markdown file.
*For any* ZIP with DOCX content, the extraction SHALL produce a DOCX file.
Image and table references in Markdown SHALL be preserved in their original positions.

**Validates: Requirements 4.1, 4.2, 4.3, 4.4**

Uses Hypothesis for property-based testing with at least 100 iterations per test.
"""

import io
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck

from backend.models import ExtractedFiles
from backend.services.file_storage import FileStorage, FileStorageError


# =============================================================================
# Strategies for generating test data
# =============================================================================

# Strategy for generating valid image filenames
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp"]

# Windows reserved device names that cannot be used as filenames
WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
}


def is_valid_filename(name: str) -> bool:
    """Check if a filename is valid (not a Windows reserved name)."""
    base_name = name.split(".")[0].upper()
    return base_name not in WINDOWS_RESERVED_NAMES


image_filename = st.tuples(
    st.text(
        alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="_-"),
        min_size=1,
        max_size=20,
    ),
    st.sampled_from(IMAGE_EXTENSIONS),
).map(lambda t: (t[0].strip() or "image") + t[1]).filter(is_valid_filename)

# Strategy for generating image paths (flat or nested)
image_path = st.one_of(
    # Flat path: just filename
    image_filename,
    # Nested path: directory/filename
    st.tuples(
        st.sampled_from(["images", "figures", "assets", "img", "pics"]),
        image_filename,
    ).map(lambda t: f"{t[0]}/{t[1]}"),
)

# Strategy for generating lists of image paths
image_paths_list = st.lists(
    image_path,
    min_size=0,
    max_size=10,
    unique=True,
)

# Strategy for generating markdown table content
def generate_table(num_cols: int, num_rows: int) -> str:
    """Generate a markdown table with given dimensions."""
    if num_cols < 1 or num_rows < 1:
        return ""
    
    # Header row
    headers = [f"Column {i+1}" for i in range(num_cols)]
    header_row = "| " + " | ".join(headers) + " |"
    
    # Separator row
    separator = "| " + " | ".join(["---"] * num_cols) + " |"
    
    # Data rows
    data_rows = []
    for row_idx in range(num_rows):
        cells = [f"Data {row_idx+1}-{col_idx+1}" for col_idx in range(num_cols)]
        data_rows.append("| " + " | ".join(cells) + " |")
    
    return "\n".join([header_row, separator] + data_rows)


table_dimensions = st.tuples(
    st.integers(min_value=1, max_value=5),  # columns
    st.integers(min_value=1, max_value=10),  # rows
)


# Strategy for generating markdown content with image references
def generate_markdown_with_images(image_paths: list[str]) -> str:
    """Generate markdown content with image references."""
    content = "# Report with Images\n\n"
    content += "This is the introduction.\n\n"
    
    for i, img_path in enumerate(image_paths):
        content += f"## Section {i+1}\n\n"
        content += f"Some text before the image.\n\n"
        content += f"![Figure {i+1}]({img_path})\n\n"
        content += f"Some text after the image.\n\n"
    
    content += "## Conclusion\n\nThis is the conclusion.\n"
    return content


# Strategy for generating task IDs
task_id_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N"), whitelist_characters="-_"),
    min_size=5,
    max_size=20,
).map(lambda s: f"task-{s.strip() or 'test'}")


# =============================================================================
# Helper functions for creating test ZIP files
# =============================================================================

def create_zip_with_markdown(
    markdown_content: str,
    markdown_filename: str = "report.md",
    include_docx: bool = False,
    docx_content: bytes = b"DOCX content placeholder",
    image_paths: Optional[list[str]] = None,
    nested_structure: bool = False,
) -> bytes:
    """
    Create a ZIP file with the specified content.
    
    Args:
        markdown_content: Content for the markdown file
        markdown_filename: Name of the markdown file
        include_docx: Whether to include a DOCX file
        docx_content: Content for the DOCX file
        image_paths: List of image paths to include
        nested_structure: Whether to put files in a nested directory
        
    Returns:
        bytes: The ZIP file content
    """
    buffer = io.BytesIO()
    
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Determine base path for nested structure
        base_path = "output/" if nested_structure else ""
        
        # Add markdown file
        zf.writestr(f"{base_path}{markdown_filename}", markdown_content.encode("utf-8"))
        
        # Add DOCX file if requested
        if include_docx:
            zf.writestr(f"{base_path}report.docx", docx_content)
        
        # Add image files if provided
        if image_paths:
            for img_path in image_paths:
                # Create fake image data
                img_data = f"PNG image data for {img_path}".encode()
                full_path = f"{base_path}{img_path}" if not img_path.startswith(base_path) else img_path
                zf.writestr(full_path, img_data)
    
    return buffer.getvalue()


# =============================================================================
# Property-based tests for ZIP Content Extraction
# =============================================================================

class TestZIPMarkdownExtractionProperty:
    """
    Property-based tests for Markdown extraction from ZIP files.
    
    **Property 8: ZIP Content Extraction (Markdown part)**
    **Validates: Requirements 4.1**
    """

    @settings(max_examples=100)
    @given(
        markdown_content=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "S", "Z")),
            min_size=10,
            max_size=500,
        ),
        task_id=task_id_strategy,
        nested=st.booleans(),
    )
    def test_zip_with_markdown_produces_valid_markdown_path(
        self, markdown_content: str, task_id: str, nested: bool
    ):
        """
        Property: For any ZIP containing a .md file, the extraction SHALL
        produce a valid markdown_path in ExtractedFiles.
        
        **Validates: Requirements 4.1**
        """
        assume(markdown_content.strip())  # Ensure non-empty content
        
        # Create temp directory for this test iteration
        tmp_dir = tempfile.mkdtemp()
        try:
            tmp_path = Path(tmp_dir)
            storage = FileStorage(str(tmp_path))
            
            # Create ZIP with markdown
            zip_content = create_zip_with_markdown(
                markdown_content=f"# Report\n\n{markdown_content}",
                nested_structure=nested,
            )
            
            # Write ZIP to temp location
            zip_path = tmp_path / task_id / "extracted" / "result.zip"
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path.write_bytes(zip_content)
            
            # Extract the ZIP
            extract_dir = tmp_path / task_id / "extracted"
            result = storage._extract_zip(zip_path, extract_dir, task_id)
            
            # SHALL produce a valid markdown_path
            assert result.markdown_path is not None, (
                "Extraction should produce a markdown_path"
            )
            assert Path(result.markdown_path).exists(), (
                f"Markdown file should exist at {result.markdown_path}"
            )
            assert Path(result.markdown_path).suffix in {".md", ".markdown"}, (
                f"Markdown file should have .md or .markdown extension"
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @settings(max_examples=100)
    @given(
        markdown_content=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P", "S", "Z")),
            min_size=10,
            max_size=500,
        ),
        task_id=task_id_strategy,
        md_extension=st.sampled_from([".md", ".markdown"]),
    )
    def test_zip_extracts_markdown_with_various_extensions(
        self, markdown_content: str, task_id: str, md_extension: str
    ):
        """
        Property: For any ZIP containing a file with .md or .markdown extension,
        the extraction SHALL recognize and extract it as the markdown file.
        
        **Validates: Requirements 4.1**
        """
        assume(markdown_content.strip())
        
        tmp_dir = tempfile.mkdtemp()
        try:
            tmp_path = Path(tmp_dir)
            storage = FileStorage(str(tmp_path))
            
            # Create ZIP with markdown using specified extension
            zip_content = create_zip_with_markdown(
                markdown_content=f"# Report\n\n{markdown_content}",
                markdown_filename=f"report{md_extension}",
            )
            
            zip_path = tmp_path / task_id / "extracted" / "result.zip"
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path.write_bytes(zip_content)
            
            extract_dir = tmp_path / task_id / "extracted"
            result = storage._extract_zip(zip_path, extract_dir, task_id)
            
            # SHALL extract the markdown file
            assert result.markdown_path is not None
            assert Path(result.markdown_path).exists()
            
            # Content should be preserved
            extracted_content = Path(result.markdown_path).read_text(encoding="utf-8")
            assert markdown_content in extracted_content
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestZIPDocxExtractionProperty:
    """
    Property-based tests for DOCX extraction from ZIP files.
    
    **Property 8: ZIP Content Extraction (DOCX part)**
    **Validates: Requirements 4.2**
    """

    @settings(max_examples=100)
    @given(
        docx_content=st.binary(min_size=10, max_size=1000),
        task_id=task_id_strategy,
        nested=st.booleans(),
    )
    def test_zip_with_docx_produces_valid_docx_path(
        self, docx_content: bytes, task_id: str, nested: bool
    ):
        """
        Property: For any ZIP containing a .docx file, the extraction SHALL
        produce a valid docx_path in ExtractedFiles.
        
        **Validates: Requirements 4.2**
        """
        tmp_dir = tempfile.mkdtemp()
        try:
            tmp_path = Path(tmp_dir)
            storage = FileStorage(str(tmp_path))
            
            # Create ZIP with both markdown and docx
            zip_content = create_zip_with_markdown(
                markdown_content="# Report\n\nContent here.",
                include_docx=True,
                docx_content=docx_content,
                nested_structure=nested,
            )
            
            zip_path = tmp_path / task_id / "extracted" / "result.zip"
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path.write_bytes(zip_content)
            
            extract_dir = tmp_path / task_id / "extracted"
            result = storage._extract_zip(zip_path, extract_dir, task_id)
            
            # SHALL produce a valid docx_path
            assert result.docx_path is not None, (
                "Extraction should produce a docx_path when ZIP contains DOCX"
            )
            assert Path(result.docx_path).exists(), (
                f"DOCX file should exist at {result.docx_path}"
            )
            assert Path(result.docx_path).suffix == ".docx", (
                f"DOCX file should have .docx extension"
            )
            
            # Content should be preserved
            extracted_docx = Path(result.docx_path).read_bytes()
            assert extracted_docx == docx_content
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @settings(max_examples=100)
    @given(task_id=task_id_strategy)
    def test_zip_without_docx_has_none_docx_path(self, task_id: str):
        """
        Property: For any ZIP without a .docx file, the extraction SHALL
        produce docx_path=None in ExtractedFiles.
        
        **Validates: Requirements 4.2**
        """
        tmp_dir = tempfile.mkdtemp()
        try:
            tmp_path = Path(tmp_dir)
            storage = FileStorage(str(tmp_path))
            
            # Create ZIP with only markdown (no docx)
            zip_content = create_zip_with_markdown(
                markdown_content="# Report\n\nContent here.",
                include_docx=False,
            )
            
            zip_path = tmp_path / task_id / "extracted" / "result.zip"
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path.write_bytes(zip_content)
            
            extract_dir = tmp_path / task_id / "extracted"
            result = storage._extract_zip(zip_path, extract_dir, task_id)
            
            # docx_path SHALL be None when no DOCX in ZIP
            assert result.docx_path is None, (
                "docx_path should be None when ZIP contains no DOCX file"
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestZIPImageExtractionProperty:
    """
    Property-based tests for image extraction and path preservation.
    
    **Property 8: ZIP Content Extraction (Image part)**
    **Validates: Requirements 4.3**
    """

    @settings(max_examples=100)
    @given(
        image_paths=image_paths_list,
        task_id=task_id_strategy,
    )
    def test_zip_extracts_all_images_and_preserves_paths(
        self, image_paths: list[str], task_id: str
    ):
        """
        Property: For any ZIP containing images, the extraction SHALL extract
        all images and preserve their paths.
        
        **Validates: Requirements 4.3**
        """
        tmp_dir = tempfile.mkdtemp()
        try:
            tmp_path = Path(tmp_dir)
            storage = FileStorage(str(tmp_path))
            
            # Generate markdown with image references
            markdown_content = generate_markdown_with_images(image_paths)
            
            # Create ZIP with markdown and images
            zip_content = create_zip_with_markdown(
                markdown_content=markdown_content,
                image_paths=image_paths,
            )
            
            zip_path = tmp_path / task_id / "extracted" / "result.zip"
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path.write_bytes(zip_content)
            
            extract_dir = tmp_path / task_id / "extracted"
            result = storage._extract_zip(zip_path, extract_dir, task_id)
            
            # SHALL extract all images
            assert len(result.images) == len(image_paths), (
                f"Should extract {len(image_paths)} images, got {len(result.images)}"
            )
            
            # All extracted images SHALL exist
            for img_path in result.images:
                assert Path(img_path).exists(), (
                    f"Extracted image should exist at {img_path}"
                )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @settings(max_examples=100)
    @given(
        num_images=st.integers(min_value=1, max_value=5),
        task_id=task_id_strategy,
        nested=st.booleans(),
    )
    def test_image_directory_structure_is_preserved(
        self, num_images: int, task_id: str, nested: bool
    ):
        """
        Property: For any ZIP with images in subdirectories, the extraction
        SHALL preserve the directory structure.
        
        **Validates: Requirements 4.3**
        """
        tmp_dir = tempfile.mkdtemp()
        try:
            tmp_path = Path(tmp_dir)
            storage = FileStorage(str(tmp_path))
            
            # Create image paths in a subdirectory
            image_paths = [f"images/figure{i}.png" for i in range(num_images)]
            markdown_content = generate_markdown_with_images(image_paths)
            
            zip_content = create_zip_with_markdown(
                markdown_content=markdown_content,
                image_paths=image_paths,
                nested_structure=nested,
            )
            
            zip_path = tmp_path / task_id / "extracted" / "result.zip"
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path.write_bytes(zip_content)
            
            extract_dir = tmp_path / task_id / "extracted"
            result = storage._extract_zip(zip_path, extract_dir, task_id)
            
            # Directory structure SHALL be preserved
            for img_path in result.images:
                # Images should be in a subdirectory (images/)
                assert "images" in img_path or "output" in img_path, (
                    f"Image path should preserve directory structure: {img_path}"
                )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestMarkdownImageReferencePreservationProperty:
    """
    Property-based tests for image reference preservation in Markdown.
    
    **Property 8: ZIP Content Extraction (Image reference preservation)**
    **Validates: Requirements 4.3**
    """

    @settings(max_examples=100)
    @given(
        image_paths=st.lists(
            st.sampled_from([
                "images/fig1.png",
                "images/fig2.jpg",
                "figures/chart.png",
                "assets/diagram.svg",
            ]),
            min_size=1,
            max_size=4,
            unique=True,
        ),
        task_id=task_id_strategy,
    )
    def test_markdown_image_references_preserved_in_original_positions(
        self, image_paths: list[str], task_id: str
    ):
        """
        Property: Image references in Markdown SHALL be preserved in their
        original positions after extraction.
        
        **Validates: Requirements 4.3**
        """
        tmp_dir = tempfile.mkdtemp()
        try:
            tmp_path = Path(tmp_dir)
            storage = FileStorage(str(tmp_path))
            
            # Generate markdown with image references at specific positions
            markdown_content = generate_markdown_with_images(image_paths)
            
            zip_content = create_zip_with_markdown(
                markdown_content=markdown_content,
                image_paths=image_paths,
            )
            
            zip_path = tmp_path / task_id / "extracted" / "result.zip"
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path.write_bytes(zip_content)
            
            extract_dir = tmp_path / task_id / "extracted"
            result = storage._extract_zip(zip_path, extract_dir, task_id)
            
            # Read extracted markdown
            extracted_md = Path(result.markdown_path).read_text(encoding="utf-8")
            
            # All image references SHALL be preserved
            for img_path in image_paths:
                # Check that the image reference exists in markdown
                assert f"![" in extracted_md and f"]({img_path})" in extracted_md, (
                    f"Image reference to '{img_path}' should be preserved in markdown"
                )
            
            # Verify order is preserved by checking positions
            positions = []
            for img_path in image_paths:
                pos = extracted_md.find(f"]({img_path})")
                assert pos != -1, f"Image reference to '{img_path}' not found"
                positions.append(pos)
            
            # Positions should be in ascending order (original order preserved)
            assert positions == sorted(positions), (
                "Image references should be in their original order"
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestMarkdownTablePreservationProperty:
    """
    Property-based tests for table structure preservation in Markdown.
    
    **Property 8: ZIP Content Extraction (Table preservation)**
    **Validates: Requirements 4.4**
    """

    @settings(max_examples=100)
    @given(
        table_dims=table_dimensions,
        task_id=task_id_strategy,
    )
    def test_markdown_tables_preserved_exactly(
        self, table_dims: tuple[int, int], task_id: str
    ):
        """
        Property: Markdown content with tables SHALL be preserved exactly
        after extraction.
        
        **Validates: Requirements 4.4**
        """
        num_cols, num_rows = table_dims
        
        tmp_dir = tempfile.mkdtemp()
        try:
            tmp_path = Path(tmp_dir)
            storage = FileStorage(str(tmp_path))
            
            # Generate table content
            table_content = generate_table(num_cols, num_rows)
            markdown_content = f"# Report\n\n## Data Table\n\n{table_content}\n\n## Conclusion\n\nEnd of report."
            
            zip_content = create_zip_with_markdown(
                markdown_content=markdown_content,
            )
            
            zip_path = tmp_path / task_id / "extracted" / "result.zip"
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path.write_bytes(zip_content)
            
            extract_dir = tmp_path / task_id / "extracted"
            result = storage._extract_zip(zip_path, extract_dir, task_id)
            
            # Read extracted markdown
            extracted_md = Path(result.markdown_path).read_text(encoding="utf-8")
            
            # Table content SHALL be preserved exactly
            assert table_content in extracted_md, (
                f"Table content should be preserved exactly.\n"
                f"Expected table:\n{table_content}\n"
                f"Extracted content:\n{extracted_md}"
            )
            
            # Verify table structure elements
            # Header row
            assert "| Column 1 |" in extracted_md, "Table header should be preserved"
            # Separator row (may have spaces around ---)
            assert "| ---" in extracted_md or "|---" in extracted_md, "Table separator should be preserved"
            # Data rows
            assert "| Data 1-1 |" in extracted_md, "Table data should be preserved"
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    @settings(max_examples=100)
    @given(
        num_tables=st.integers(min_value=1, max_value=3),
        task_id=task_id_strategy,
    )
    def test_multiple_tables_preserved_in_order(
        self, num_tables: int, task_id: str
    ):
        """
        Property: Multiple tables in Markdown SHALL be preserved in their
        original positions.
        
        **Validates: Requirements 4.4**
        """
        tmp_dir = tempfile.mkdtemp()
        try:
            tmp_path = Path(tmp_dir)
            storage = FileStorage(str(tmp_path))
            
            # Generate markdown with multiple tables
            markdown_parts = ["# Report with Multiple Tables\n\n"]
            tables = []
            for i in range(num_tables):
                table = generate_table(2, 2)
                # Make each table unique by adding a marker
                unique_table = table.replace("Column 1", f"Table{i+1}Col1")
                tables.append(unique_table)
                markdown_parts.append(f"## Table {i+1}\n\n{unique_table}\n\n")
            
            markdown_content = "".join(markdown_parts)
            
            zip_content = create_zip_with_markdown(
                markdown_content=markdown_content,
            )
            
            zip_path = tmp_path / task_id / "extracted" / "result.zip"
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path.write_bytes(zip_content)
            
            extract_dir = tmp_path / task_id / "extracted"
            result = storage._extract_zip(zip_path, extract_dir, task_id)
            
            extracted_md = Path(result.markdown_path).read_text(encoding="utf-8")
            
            # All tables SHALL be preserved
            for i, table in enumerate(tables):
                assert f"Table{i+1}Col1" in extracted_md, (
                    f"Table {i+1} should be preserved"
                )
            
            # Tables SHALL be in original order
            positions = [extracted_md.find(f"Table{i+1}Col1") for i in range(num_tables)]
            assert positions == sorted(positions), (
                "Tables should be in their original order"
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestZIPStructureVariationsProperty:
    """
    Property-based tests for various ZIP structure variations.
    
    **Property 8: ZIP Content Extraction**
    **Validates: Requirements 4.1, 4.2, 4.3, 4.4**
    """

    @settings(max_examples=100)
    @given(
        nested_depth=st.integers(min_value=0, max_value=3),
        task_id=task_id_strategy,
        include_docx=st.booleans(),
        num_images=st.integers(min_value=0, max_value=3),
    )
    def test_various_zip_structures_extract_correctly(
        self,
        nested_depth: int,
        task_id: str,
        include_docx: bool,
        num_images: int,
    ):
        """
        Property: For any ZIP structure (flat or nested), the extraction
        SHALL correctly identify and extract all relevant files.
        
        **Validates: Requirements 4.1, 4.2, 4.3, 4.4**
        """
        tmp_dir = tempfile.mkdtemp()
        try:
            tmp_path = Path(tmp_dir)
            storage = FileStorage(str(tmp_path))
            
            # Build nested path prefix
            path_parts = ["level" + str(i) for i in range(nested_depth)]
            path_prefix = "/".join(path_parts) + "/" if path_parts else ""
            
            # Generate image paths
            image_paths = [f"images/fig{i}.png" for i in range(num_images)]
            
            # Generate markdown content
            markdown_content = generate_markdown_with_images(image_paths)
            table_content = generate_table(2, 2)
            markdown_content += f"\n\n## Data\n\n{table_content}\n"
            
            # Create ZIP with nested structure
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(f"{path_prefix}report.md", markdown_content.encode("utf-8"))
                
                if include_docx:
                    zf.writestr(f"{path_prefix}report.docx", b"DOCX content")
                
                for img_path in image_paths:
                    zf.writestr(f"{path_prefix}{img_path}", b"PNG data")
            
            zip_content = buffer.getvalue()
            
            zip_path = tmp_path / task_id / "extracted" / "result.zip"
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path.write_bytes(zip_content)
            
            extract_dir = tmp_path / task_id / "extracted"
            result = storage._extract_zip(zip_path, extract_dir, task_id)
            
            # Markdown SHALL always be extracted
            assert result.markdown_path is not None
            assert Path(result.markdown_path).exists()
            
            # DOCX SHALL be extracted if present
            if include_docx:
                assert result.docx_path is not None
                assert Path(result.docx_path).exists()
            else:
                assert result.docx_path is None
            
            # All images SHALL be extracted
            assert len(result.images) == num_images
            for img_path in result.images:
                assert Path(img_path).exists()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestZIPExtractionErrorHandling:
    """
    Tests for error handling in ZIP extraction.
    
    **Property 8: ZIP Content Extraction**
    **Validates: Requirements 4.1**
    """

    @settings(max_examples=100)
    @given(task_id=task_id_strategy)
    def test_zip_without_markdown_raises_error(self, task_id: str):
        """
        Property: For any ZIP without a Markdown file, the extraction
        SHALL raise a FileStorageError.
        
        **Validates: Requirements 4.1**
        """
        tmp_dir = tempfile.mkdtemp()
        try:
            tmp_path = Path(tmp_dir)
            storage = FileStorage(str(tmp_path))
            
            # Create ZIP without markdown
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w") as zf:
                zf.writestr("report.docx", b"DOCX only")
                zf.writestr("images/fig1.png", b"PNG data")
            
            zip_content = buffer.getvalue()
            
            zip_path = tmp_path / task_id / "extracted" / "result.zip"
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            zip_path.write_bytes(zip_content)
            
            extract_dir = tmp_path / task_id / "extracted"
            
            # SHALL raise error when no markdown found
            with pytest.raises(FileStorageError) as exc_info:
                storage._extract_zip(zip_path, extract_dir, task_id)
            
            assert "No Markdown file found" in str(exc_info.value)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
