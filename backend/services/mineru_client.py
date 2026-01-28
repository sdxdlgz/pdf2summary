"""
MinerU API client for the Research Report Processor.

This module provides integration with the MinerU API for PDF parsing:
- Batch file upload API to get pre-signed URLs
- File upload via PUT to pre-signed URLs
- Batch results retrieval for parsing status and results

Requirements:
- 2.1: Always use batch file upload API (POST /api/v4/file-urls/batch)
- 2.2: Upload files using HTTP PUT to returned URLs
- 2.3: Send raw binary data without Content-Type header
- 2.4: Files are automatically submitted for parsing after upload
- 2.5: Set enable_table=true
- 2.6: Set enable_formula=true
- 2.7: Set extra_formats=["docx"]
- 2.8: Set model_version="vlm"
- 2.9: Store batch_id for subsequent polling
- 2.10: Log errors and notify Task_Manager on API errors
- 2.11: Complete uploads within 24 hours (URL expiry)
- 2.12: Do not use single file API
"""

import logging
from typing import Callable, Optional

import aiohttp

from backend.models import (
    BatchResultResponse,
    BatchUploadResponse,
    FileInfo,
    FileParseResult,
    MineruTaskState,
)


logger = logging.getLogger(__name__)


class MineruClientError(Exception):
    """
    Exception raised when MinerU API operations fail.
    
    Attributes:
        message: Human-readable error description
        error_code: MinerU API error code (if available)
        batch_id: The batch ID associated with the error (if available)
    """
    
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        batch_id: Optional[str] = None,
    ):
        self.message = message
        self.error_code = error_code
        self.batch_id = batch_id
        super().__init__(self.message)


class MineruClient:
    """
    Client for interacting with the MinerU API.
    
    This client handles:
    - Batch file upload URL requests
    - File uploads via PUT to pre-signed URLs
    - Batch result retrieval for parsing status
    
    The client always uses the batch API (Requirement 2.1, 2.12) and
    configures parsing with optimal settings (Requirements 2.5-2.8).
    
    Attributes:
        api_token: MinerU API authentication token
        base_url: Base URL for MinerU API
        error_callback: Optional callback for error notifications
    """
    
    # MinerU API endpoints
    BASE_URL = "https://mineru.net"
    BATCH_UPLOAD_ENDPOINT = "/api/v4/file-urls/batch"
    BATCH_RESULTS_ENDPOINT = "/api/v4/extract-results/batch"
    
    # Fixed API parameters (Requirements 2.5, 2.6, 2.7, 2.8)
    ENABLE_TABLE = True
    ENABLE_FORMULA = True
    EXTRA_FORMATS = ["docx"]
    MODEL_VERSION = "vlm"
    
    def __init__(
        self,
        api_token: str,
        base_url: Optional[str] = None,
        error_callback: Optional[Callable[[str, str], None]] = None,
    ):
        """
        Initialize the MinerU client.
        
        Args:
            api_token: MinerU API authentication token (Requirement 11.1)
            base_url: Optional custom base URL for testing
            error_callback: Optional callback function(batch_id, error_msg)
                           for notifying Task_Manager of errors (Requirement 2.10)
        """
        if not api_token:
            raise ValueError("api_token is required")
        
        self.api_token = api_token
        self.base_url = base_url or self.BASE_URL
        self.error_callback = error_callback
    
    def _get_headers(self) -> dict[str, str]:
        """
        Get the authorization headers for API requests.
        
        Returns:
            Dictionary with Authorization header
        """
        return {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
    
    def _notify_error(self, batch_id: Optional[str], error_msg: str) -> None:
        """
        Notify the error callback if configured.
        
        This implements Requirement 2.10: Log errors and notify Task_Manager.
        
        Args:
            batch_id: The batch ID associated with the error
            error_msg: The error message
        """
        logger.error(
            "MinerU API error: %s (batch_id=%s)",
            error_msg,
            batch_id or "N/A",
        )
        
        if self.error_callback:
            self.error_callback(batch_id or "", error_msg)
    
    async def request_upload_urls(
        self,
        files: list[FileInfo],
    ) -> BatchUploadResponse:
        """
        Request pre-signed upload URLs for a batch of files.
        
        This method calls the batch file upload API (Requirement 2.1) with
        the required parameters:
        - enable_table=true (Requirement 2.5)
        - enable_formula=true (Requirement 2.6)
        - extra_formats=["docx"] (Requirement 2.7)
        - model_version="vlm" (Requirement 2.8)
        
        The returned batch_id should be stored for subsequent polling
        (Requirement 2.9).
        
        Note: This method always uses the batch API regardless of file count
        (Requirement 2.12 - do not use single file API).
        
        Args:
            files: List of FileInfo objects containing name and data_id
            
        Returns:
            BatchUploadResponse containing batch_id and file_urls
            
        Raises:
            MineruClientError: If the API request fails
            
        Requirements: 2.1, 2.5, 2.6, 2.7, 2.8, 2.9, 2.10, 2.12
        """
        if not files:
            raise MineruClientError("No files provided for upload")
        
        # Build request payload with required parameters
        # Requirements 2.5, 2.6, 2.7, 2.8
        payload = {
            "files": [
                {"name": f.name, "data_id": f.data_id}
                for f in files
            ],
            "enable_table": self.ENABLE_TABLE,
            "enable_formula": self.ENABLE_FORMULA,
            "extra_formats": self.EXTRA_FORMATS,
            "model_version": self.MODEL_VERSION,
        }
        
        url = f"{self.base_url}{self.BATCH_UPLOAD_ENDPOINT}"
        
        logger.info(
            "Requesting upload URLs for %d files from MinerU API",
            len(files),
        )
        logger.debug("Request payload: %s", payload)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    headers=self._get_headers(),
                ) as response:
                    response_data = await response.json()
                    
                    # Check for API-level errors
                    code = response_data.get("code")
                    if code != 0:
                        error_msg = response_data.get("msg", "Unknown error")
                        self._notify_error(None, f"API error {code}: {error_msg}")
                        raise MineruClientError(
                            message=f"MinerU API error: {error_msg}",
                            error_code=str(code),
                        )
                    
                    # Extract response data
                    data = response_data.get("data", {})
                    batch_id = data.get("batch_id")
                    file_urls = data.get("file_urls", [])
                    
                    if not batch_id:
                        error_msg = "No batch_id in response"
                        self._notify_error(None, error_msg)
                        raise MineruClientError(message=error_msg)
                    
                    if len(file_urls) != len(files):
                        error_msg = (
                            f"URL count mismatch: expected {len(files)}, "
                            f"got {len(file_urls)}"
                        )
                        self._notify_error(batch_id, error_msg)
                        raise MineruClientError(
                            message=error_msg,
                            batch_id=batch_id,
                        )
                    
                    logger.info(
                        "Received batch_id=%s with %d upload URLs",
                        batch_id,
                        len(file_urls),
                    )
                    
                    # Requirement 2.9: Return batch_id for storage
                    return BatchUploadResponse(
                        batch_id=batch_id,
                        file_urls=file_urls,
                    )
        
        except aiohttp.ClientError as e:
            error_msg = f"Network error: {e}"
            self._notify_error(None, error_msg)
            raise MineruClientError(message=error_msg) from e
    
    async def upload_file(
        self,
        upload_url: str,
        file_content: bytes,
    ) -> bool:
        """
        Upload a file to a pre-signed URL using HTTP PUT.
        
        This method implements Requirements 2.2 and 2.3:
        - Uses HTTP PUT request (Requirement 2.2)
        - Sends raw binary data without Content-Type header (Requirement 2.3)
        
        After upload, MinerU automatically submits the parse task
        (Requirement 2.4 - no additional API call needed).
        
        Note: Upload URLs expire after 24 hours (Requirement 2.11).
        
        Args:
            upload_url: The pre-signed URL from request_upload_urls
            file_content: The raw file binary content
            
        Returns:
            True if upload was successful
            
        Raises:
            MineruClientError: If the upload fails
            
        Requirements: 2.2, 2.3, 2.4, 2.10, 2.11
        """
        if not upload_url:
            raise MineruClientError("upload_url is required")
        
        if not file_content:
            raise MineruClientError("file_content is required")
        
        logger.debug("Uploading file to pre-signed URL")
        
        try:
            async with aiohttp.ClientSession() as session:
                # Requirement 2.2: Use HTTP PUT
                # Requirement 2.3: Send raw binary, no Content-Type header
                async with session.put(
                    upload_url,
                    data=file_content,
                    # Do not set Content-Type header (Requirement 2.3)
                ) as response:
                    if response.status >= 400:
                        error_msg = (
                            f"Upload failed with status {response.status}"
                        )
                        self._notify_error(None, error_msg)
                        raise MineruClientError(message=error_msg)
                    
                    logger.debug(
                        "File uploaded successfully (status=%d)",
                        response.status,
                    )
                    
                    # Requirement 2.4: Parse task is automatically submitted
                    return True
        
        except aiohttp.ClientError as e:
            error_msg = f"Upload network error: {e}"
            self._notify_error(None, error_msg)
            raise MineruClientError(message=error_msg) from e
    
    async def get_batch_results(
        self,
        batch_id: str,
    ) -> BatchResultResponse:
        """
        Get the parsing results for a batch of files.
        
        This method retrieves the current status and results for all files
        in a batch. The response includes:
        - state: Current parsing state for each file
        - full_zip_url: Download URL when parsing is complete
        - err_msg: Error message if parsing failed
        - extract_progress: Pages parsed vs total pages
        
        Args:
            batch_id: The batch ID from request_upload_urls (Requirement 2.9)
            
        Returns:
            BatchResultResponse containing status for all files
            
        Raises:
            MineruClientError: If the API request fails
            
        Requirements: 2.9, 2.10
        """
        if not batch_id:
            raise MineruClientError("batch_id is required")
        
        url = f"{self.base_url}{self.BATCH_RESULTS_ENDPOINT}/{batch_id}"
        
        logger.debug("Getting batch results for batch_id=%s", batch_id)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=self._get_headers(),
                ) as response:
                    response_data = await response.json()
                    
                    # Check for API-level errors
                    code = response_data.get("code")
                    if code != 0:
                        error_msg = response_data.get("msg", "Unknown error")
                        self._notify_error(
                            batch_id,
                            f"API error {code}: {error_msg}",
                        )
                        raise MineruClientError(
                            message=f"MinerU API error: {error_msg}",
                            error_code=str(code),
                            batch_id=batch_id,
                        )
                    
                    # Extract response data
                    data = response_data.get("data", {})
                    extract_results = data.get("extract_result", [])
                    
                    # Parse individual file results
                    file_results = []
                    for result in extract_results:
                        # Parse extract_progress if present
                        progress = result.get("extract_progress", {})
                        extracted_pages = progress.get("extracted_pages")
                        total_pages = progress.get("total_pages")
                        
                        # Parse state string to enum
                        state_str = result.get("state", "pending")
                        try:
                            state = MineruTaskState(state_str)
                        except ValueError:
                            logger.warning(
                                "Unknown state '%s', defaulting to PENDING",
                                state_str,
                            )
                            state = MineruTaskState.PENDING
                        
                        file_result = FileParseResult(
                            file_name=result.get("file_name", ""),
                            data_id=result.get("data_id"),
                            state=state,
                            full_zip_url=result.get("full_zip_url"),
                            err_msg=result.get("err_msg"),
                            extracted_pages=extracted_pages,
                            total_pages=total_pages,
                        )
                        file_results.append(file_result)
                    
                    logger.debug(
                        "Batch %s: %d file results retrieved",
                        batch_id,
                        len(file_results),
                    )
                    
                    return BatchResultResponse(
                        batch_id=batch_id,
                        extract_result=file_results,
                    )
        
        except aiohttp.ClientError as e:
            error_msg = f"Network error: {e}"
            self._notify_error(batch_id, error_msg)
            raise MineruClientError(
                message=error_msg,
                batch_id=batch_id,
            ) from e
    
    async def upload_files_batch(
        self,
        files: list[FileInfo],
        file_contents: list[bytes],
    ) -> BatchUploadResponse:
        """
        Convenience method to request URLs and upload all files.
        
        This method combines request_upload_urls and upload_file for
        a complete batch upload workflow.
        
        Args:
            files: List of FileInfo objects
            file_contents: List of file contents (same order as files)
            
        Returns:
            BatchUploadResponse with batch_id for polling
            
        Raises:
            MineruClientError: If any step fails
            ValueError: If files and file_contents have different lengths
        """
        if len(files) != len(file_contents):
            raise ValueError(
                f"files ({len(files)}) and file_contents ({len(file_contents)}) "
                "must have the same length"
            )
        
        # Step 1: Request upload URLs
        upload_response = await self.request_upload_urls(files)
        
        # Step 2: Upload each file to its URL
        for i, (file_info, content, url) in enumerate(
            zip(files, file_contents, upload_response.file_urls)
        ):
            logger.info(
                "Uploading file %d/%d: %s",
                i + 1,
                len(files),
                file_info.name,
            )
            await self.upload_file(url, content)
        
        logger.info(
            "All %d files uploaded successfully (batch_id=%s)",
            len(files),
            upload_response.batch_id,
        )
        
        return upload_response
    
    def get_api_parameters(self) -> dict:
        """
        Get the fixed API parameters used for all requests.
        
        This is useful for testing to verify correct parameter configuration.
        
        Returns:
            Dictionary with enable_table, enable_formula, extra_formats,
            and model_version values
            
        Requirements: 2.5, 2.6, 2.7, 2.8
        """
        return {
            "enable_table": self.ENABLE_TABLE,
            "enable_formula": self.ENABLE_FORMULA,
            "extra_formats": self.EXTRA_FORMATS,
            "model_version": self.MODEL_VERSION,
        }
