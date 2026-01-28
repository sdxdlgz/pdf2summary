"""
File storage service for the Research Report Processor.

This module provides file storage functionality including:
- Saving uploaded PDF files
- Saving output files (original MD, original DOCX, bilingual MD, bilingual DOCX, summary)
- Downloading and extracting MinerU result ZIP files

Requirements:
- 4.1: Extract Markdown files from ZIP archive
- 4.2: Extract DOCX files from ZIP archive
- 4.3: Preserve image references in their original positions
- 4.4: Preserve table structures in their original positions
- 4.5: Store original MD and DOCX files for output
- 7.1: Provide original MD file for download
- 7.2: Provide original DOCX file for download
- 7.3: Provide bilingual translation MD file for download
- 7.4: Provide bilingual translation DOCX file for download
- 7.5: Provide bilingual summary file for download
"""

import os
import zipfile
from pathlib import Path
from typing import Optional

import aiofiles
import aiohttp

from backend.models import ExtractedFiles, OutputFileType


class FileStorageError(Exception):
    """Exception raised when file storage operations fail."""
    
    def __init__(self, message: str, task_id: Optional[str] = None):
        self.message = message
        self.task_id = task_id
        super().__init__(self.message)


class FileStorage:
    """
    File storage service for managing uploaded and output files.
    
    Directory structure:
    {base_path}/
        {task_id}/
            uploads/        # Uploaded PDF files
            outputs/        # Generated output files
            extracted/      # Extracted files from MinerU ZIP
                images/     # Extracted images
    
    Attributes:
        base_path: Base directory for all file storage
    """
    
    # Subdirectory names
    UPLOADS_DIR = "uploads"
    OUTPUTS_DIR = "outputs"
    EXTRACTED_DIR = "extracted"
    IMAGES_DIR = "images"
    
    # File extensions for extraction
    MARKDOWN_EXTENSIONS = {".md", ".markdown"}
    DOCX_EXTENSION = ".docx"
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp"}
    
    def __init__(self, base_path: str):
        """
        Initialize the file storage service.
        
        Args:
            base_path: Base directory path for all file storage.
                       Will be created if it doesn't exist.
        """
        self.base_path = Path(base_path)
        self._ensure_base_directory()
    
    def _ensure_base_directory(self) -> None:
        """Ensure the base directory exists."""
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _get_task_dir(self, task_id: str) -> Path:
        """Get the directory path for a specific task."""
        return self.base_path / task_id
    
    def _get_uploads_dir(self, task_id: str) -> Path:
        """Get the uploads directory path for a specific task."""
        return self._get_task_dir(task_id) / self.UPLOADS_DIR
    
    def _get_outputs_dir(self, task_id: str) -> Path:
        """Get the outputs directory path for a specific task."""
        return self._get_task_dir(task_id) / self.OUTPUTS_DIR
    
    def _get_extracted_dir(self, task_id: str) -> Path:
        """Get the extracted files directory path for a specific task."""
        return self._get_task_dir(task_id) / self.EXTRACTED_DIR
    
    def _ensure_task_directories(self, task_id: str) -> None:
        """
        Ensure all directories for a task exist.
        
        Creates:
        - {base_path}/{task_id}/uploads/
        - {base_path}/{task_id}/outputs/
        - {base_path}/{task_id}/extracted/
        - {base_path}/{task_id}/extracted/images/
        """
        self._get_uploads_dir(task_id).mkdir(parents=True, exist_ok=True)
        self._get_outputs_dir(task_id).mkdir(parents=True, exist_ok=True)
        extracted_dir = self._get_extracted_dir(task_id)
        extracted_dir.mkdir(parents=True, exist_ok=True)
        (extracted_dir / self.IMAGES_DIR).mkdir(parents=True, exist_ok=True)
    
    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a filename to prevent path traversal attacks.
        
        Args:
            filename: The original filename
            
        Returns:
            A sanitized filename with only the base name
        """
        # Get only the base name, removing any path components
        return Path(filename).name
    
    def save_upload(self, task_id: str, filename: str, content: bytes) -> str:
        """
        Save an uploaded file to the task's uploads directory.
        
        Args:
            task_id: The unique task identifier
            filename: The original filename
            content: The file content as bytes
            
        Returns:
            The full path to the saved file
            
        Raises:
            FileStorageError: If the file cannot be saved
        """
        self._ensure_task_directories(task_id)
        
        safe_filename = self._sanitize_filename(filename)
        file_path = self._get_uploads_dir(task_id) / safe_filename
        
        try:
            file_path.write_bytes(content)
            return str(file_path)
        except OSError as e:
            raise FileStorageError(
                message=f"Failed to save uploaded file '{filename}': {e}",
                task_id=task_id,
            ) from e
    
    def save_output(
        self,
        task_id: str,
        file_type: str,
        filename: str,
        content: bytes,
    ) -> str:
        """
        Save an output file to the task's outputs directory.
        
        Output files are organized by type:
        - original_md: Original Markdown from MinerU
        - original_docx: Original DOCX from MinerU
        - bilingual_md: Bilingual translation Markdown
        - bilingual_docx: Bilingual translation DOCX
        - summary: Bilingual summary
        
        Args:
            task_id: The unique task identifier
            file_type: The type of output file (from OutputFileType enum)
            filename: The filename for the output
            content: The file content as bytes
            
        Returns:
            The full path to the saved file
            
        Raises:
            FileStorageError: If the file cannot be saved
            
        Requirements: 4.5, 7.1, 7.2, 7.3, 7.4, 7.5
        """
        self._ensure_task_directories(task_id)
        
        # Validate file_type
        valid_types = {t.value for t in OutputFileType}
        if file_type not in valid_types:
            raise FileStorageError(
                message=f"Invalid file type '{file_type}'. Must be one of: {valid_types}",
                task_id=task_id,
            )
        
        safe_filename = self._sanitize_filename(filename)
        # Prefix with file_type to avoid name collisions
        output_filename = f"{file_type}_{safe_filename}"
        file_path = self._get_outputs_dir(task_id) / output_filename
        
        try:
            file_path.write_bytes(content)
            return str(file_path)
        except OSError as e:
            raise FileStorageError(
                message=f"Failed to save output file '{filename}': {e}",
                task_id=task_id,
            ) from e
    
    def get_output_path(self, task_id: str, file_type: str) -> str:
        """
        Get the path to an output file.
        
        Args:
            task_id: The unique task identifier
            file_type: The type of output file (from OutputFileType enum)
            
        Returns:
            The directory path where output files of this type are stored
            
        Note:
            This returns the outputs directory path. The actual filename
            will be prefixed with the file_type.
        """
        return str(self._get_outputs_dir(task_id))
    
    def get_output_file(self, task_id: str, file_type: str) -> Optional[str]:
        """
        Find and return the path to a specific output file.
        
        Args:
            task_id: The unique task identifier
            file_type: The type of output file (from OutputFileType enum)
            
        Returns:
            The full path to the output file, or None if not found
        """
        outputs_dir = self._get_outputs_dir(task_id)
        if not outputs_dir.exists():
            return None
        
        # Look for files prefixed with the file_type
        prefix = f"{file_type}_"
        for file_path in outputs_dir.iterdir():
            if file_path.name.startswith(prefix):
                return str(file_path)
        
        return None
    
    async def download_and_extract_zip(
        self,
        url: str,
        task_id: str,
    ) -> ExtractedFiles:
        """
        Download a ZIP file from URL and extract its contents.
        
        This method:
        1. Downloads the ZIP file from the given URL
        2. Extracts Markdown files (Requirement 4.1)
        3. Extracts DOCX files (Requirement 4.2)
        4. Extracts and preserves images (Requirement 4.3)
        5. Preserves table structures in Markdown (Requirement 4.4)
        
        The extracted files maintain their relative paths to preserve
        image references in Markdown files.
        
        Args:
            url: The URL to download the ZIP file from
            task_id: The unique task identifier
            
        Returns:
            ExtractedFiles model containing paths to extracted files
            
        Raises:
            FileStorageError: If download or extraction fails
            
        Requirements: 4.1, 4.2, 4.3, 4.4
        """
        self._ensure_task_directories(task_id)
        
        extracted_dir = self._get_extracted_dir(task_id)
        zip_path = extracted_dir / "result.zip"
        
        # Download the ZIP file
        try:
            await self._download_file(url, zip_path)
        except Exception as e:
            raise FileStorageError(
                message=f"Failed to download ZIP from '{url}': {e}",
                task_id=task_id,
            ) from e
        
        # Extract the ZIP file
        try:
            return self._extract_zip(zip_path, extracted_dir, task_id)
        except Exception as e:
            raise FileStorageError(
                message=f"Failed to extract ZIP file: {e}",
                task_id=task_id,
            ) from e
        finally:
            # Clean up the ZIP file after extraction
            if zip_path.exists():
                zip_path.unlink()
    
    async def _download_file(self, url: str, destination: Path) -> None:
        """
        Download a file from URL to the specified destination.
        
        Args:
            url: The URL to download from
            destination: The local path to save the file
        """
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                
                async with aiofiles.open(destination, "wb") as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
    
    def _extract_zip(
        self,
        zip_path: Path,
        extract_dir: Path,
        task_id: str,
    ) -> ExtractedFiles:
        """
        Extract contents from a ZIP file.
        
        Extracts:
        - Markdown files (.md, .markdown)
        - DOCX files (.docx)
        - Image files (.png, .jpg, .jpeg, .gif, .bmp, .svg, .webp)
        
        The directory structure is preserved to maintain image references
        in Markdown files (Requirement 4.3).
        
        Args:
            zip_path: Path to the ZIP file
            extract_dir: Directory to extract files to
            task_id: The task identifier for error reporting
            
        Returns:
            ExtractedFiles model with paths to extracted files
            
        Requirements: 4.1, 4.2, 4.3, 4.4
        """
        markdown_path: Optional[str] = None
        docx_path: Optional[str] = None
        images: list[str] = []
        
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                # Skip directories
                if member.endswith("/"):
                    continue
                
                member_path = Path(member)
                suffix = member_path.suffix.lower()
                
                # Determine file type and extract
                if suffix in self.MARKDOWN_EXTENSIONS:
                    # Extract Markdown file (Requirement 4.1)
                    # Markdown preserves table structures (Requirement 4.4)
                    extracted_path = self._extract_member(
                        zf, member, extract_dir
                    )
                    # Use the first Markdown file found as the main one
                    if markdown_path is None:
                        markdown_path = extracted_path
                
                elif suffix == self.DOCX_EXTENSION:
                    # Extract DOCX file (Requirement 4.2)
                    extracted_path = self._extract_member(
                        zf, member, extract_dir
                    )
                    # Use the first DOCX file found as the main one
                    if docx_path is None:
                        docx_path = extracted_path
                
                elif suffix in self.IMAGE_EXTENSIONS:
                    # Extract image files (Requirement 4.3)
                    # Preserve directory structure for image references
                    extracted_path = self._extract_member(
                        zf, member, extract_dir
                    )
                    images.append(extracted_path)
        
        if markdown_path is None:
            raise FileStorageError(
                message="No Markdown file found in ZIP archive",
                task_id=task_id,
            )
        
        return ExtractedFiles(
            markdown_path=markdown_path,
            docx_path=docx_path,
            images=images,
        )
    
    def _extract_member(
        self,
        zf: zipfile.ZipFile,
        member: str,
        extract_dir: Path,
    ) -> str:
        """
        Extract a single member from a ZIP file.
        
        Preserves the directory structure within the ZIP to maintain
        relative paths for image references.
        
        Args:
            zf: The ZipFile object
            member: The member name/path within the ZIP
            extract_dir: The base directory to extract to
            
        Returns:
            The full path to the extracted file
        """
        # Sanitize the member path to prevent path traversal
        member_path = Path(member)
        
        # Remove any leading slashes or parent directory references
        safe_parts = [
            part for part in member_path.parts
            if part not in ("", "..", "/", "\\")
        ]
        
        if not safe_parts:
            # Fallback to just the filename
            safe_parts = [member_path.name]
        
        safe_path = Path(*safe_parts)
        destination = extract_dir / safe_path
        
        # Ensure parent directories exist
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract the file
        with zf.open(member) as source:
            destination.write_bytes(source.read())
        
        return str(destination)
    
    def get_upload_path(self, task_id: str, filename: str) -> Optional[str]:
        """
        Get the path to an uploaded file.
        
        Args:
            task_id: The unique task identifier
            filename: The original filename
            
        Returns:
            The full path to the uploaded file, or None if not found
        """
        safe_filename = self._sanitize_filename(filename)
        file_path = self._get_uploads_dir(task_id) / safe_filename
        
        if file_path.exists():
            return str(file_path)
        return None
    
    def get_extracted_markdown(self, task_id: str) -> Optional[str]:
        """
        Get the path to the extracted Markdown file for a task.
        
        Args:
            task_id: The unique task identifier
            
        Returns:
            The path to the Markdown file, or None if not found
        """
        extracted_dir = self._get_extracted_dir(task_id)
        if not extracted_dir.exists():
            return None
        
        # Search for Markdown files
        for ext in self.MARKDOWN_EXTENSIONS:
            for md_file in extracted_dir.rglob(f"*{ext}"):
                return str(md_file)
        
        return None
    
    def get_extracted_docx(self, task_id: str) -> Optional[str]:
        """
        Get the path to the extracted DOCX file for a task.
        
        Args:
            task_id: The unique task identifier
            
        Returns:
            The path to the DOCX file, or None if not found
        """
        extracted_dir = self._get_extracted_dir(task_id)
        if not extracted_dir.exists():
            return None
        
        # Search for DOCX files
        for docx_file in extracted_dir.rglob(f"*{self.DOCX_EXTENSION}"):
            return str(docx_file)
        
        return None
    
    def cleanup_task(self, task_id: str) -> bool:
        """
        Remove all files associated with a task.
        
        Args:
            task_id: The unique task identifier
            
        Returns:
            True if cleanup was successful, False otherwise
        """
        import shutil
        
        task_dir = self._get_task_dir(task_id)
        if not task_dir.exists():
            return True
        
        try:
            shutil.rmtree(task_dir)
            return True
        except OSError:
            return False
    
    def list_task_outputs(self, task_id: str) -> dict[str, str]:
        """
        List all output files for a task.
        
        Args:
            task_id: The unique task identifier
            
        Returns:
            Dictionary mapping file types to their paths
        """
        outputs: dict[str, str] = {}
        outputs_dir = self._get_outputs_dir(task_id)
        
        if not outputs_dir.exists():
            return outputs
        
        for file_type in OutputFileType:
            file_path = self.get_output_file(task_id, file_type.value)
            if file_path:
                outputs[file_type.value] = file_path
        
        return outputs
