"""
AI Service Client for the Research Report Processor.

This module provides AI-powered translation and summarization:
- Language detection (English or Japanese)
- Chunk-based translation with concurrent processing
- Bilingual document generation
- Bilingual summary generation
- Retry logic with exponential backoff

Requirements:
- 5.1: Detect source language (English or Japanese)
- 5.2: Call OpenAI-compatible API with model "gpt-5-nano"
- 5.3: Split content into chunks for large documents
- 5.4: Process chunks concurrently
- 5.5: Produce bilingual document with paragraph alignment
- 5.7: Retry up to 3 times before marking as failed
- 6.1: Generate summary in source language
- 6.2: Translate summary to Chinese
- 6.3: Call OpenAI-compatible API with model "gpt-5-nano"
- 6.4: Produce bilingual summary document
- 6.5: Retry up to 3 times before marking as failed
"""

import asyncio
import logging
import re
from typing import Callable, Optional

import aiohttp


logger = logging.getLogger(__name__)


# Retry configuration
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1.0  # seconds
RETRY_BACKOFF_MULTIPLIER = 2.0

# Chunk configuration
DEFAULT_CHUNK_SEPARATOR = "\n\n"  # Split by paragraphs (double newline)

# Language detection patterns
# Japanese character ranges: Hiragana, Katakana, CJK Ideographs (Kanji)
JAPANESE_PATTERN = re.compile(
    r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]'
)

# Default concurrency limit
DEFAULT_MAX_CONCURRENCY = 10


class AIClientError(Exception):
    """
    Exception raised when AI service operations fail.
    
    Attributes:
        message: Human-readable error description
        retry_count: Number of retries attempted before failure
        original_error: The underlying error that caused the failure
        status_code: HTTP status code if available
        retryable: Whether the error should be retried
    """
    
    def __init__(
        self,
        message: str,
        retry_count: int = 0,
        original_error: Optional[Exception] = None,
        status_code: Optional[int] = None,
        retryable: bool = True,
    ):
        self.message = message
        self.retry_count = retry_count
        self.original_error = original_error
        self.status_code = status_code
        self.retryable = retryable
        super().__init__(self.message)


def detect_language_heuristic(text: str) -> str:
    """
    Detect language using character-based heuristics.
    
    This function uses simple heuristics to detect whether text is
    English or Japanese:
    - If Japanese characters (Hiragana, Katakana, Kanji) are present
      and make up a significant portion, it's Japanese
    - Otherwise, it's English
    
    Args:
        text: The text to analyze
        
    Returns:
        "ja" for Japanese, "en" for English
        
    Validates: Requirement 5.1 - Detect source language
    """
    if not text or not text.strip():
        return "en"  # Default to English for empty text
    
    # Count Japanese characters
    japanese_chars = JAPANESE_PATTERN.findall(text)
    japanese_count = len(japanese_chars)
    
    # Count total non-whitespace characters
    non_whitespace = re.sub(r'\s', '', text)
    total_chars = len(non_whitespace)
    
    if total_chars == 0:
        return "en"
    
    # Calculate Japanese character ratio
    japanese_ratio = japanese_count / total_chars
    
    # If more than 10% Japanese characters, consider it Japanese
    # This threshold handles mixed content with some English terms
    if japanese_ratio > 0.1:
        return "ja"
    
    return "en"


def split_into_chunks(
    content: str,
    separator: str = DEFAULT_CHUNK_SEPARATOR,
) -> list[str]:
    """
    Split content into chunks by paragraphs.
    
    This function splits content by double newlines (paragraphs) to
    create manageable chunks for translation.
    
    Args:
        content: The content to split
        separator: The separator to use (default: double newline)
        
    Returns:
        List of non-empty chunks
        
    Validates: Requirement 5.3 - Split content into chunks
    """
    if not content:
        return []
    
    # Split by separator
    chunks = content.split(separator)
    
    # Filter out empty chunks and strip whitespace
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    
    return chunks


def merge_chunks(chunks: list[str], separator: str = DEFAULT_CHUNK_SEPARATOR) -> str:
    """
    Merge chunks back into a single document.
    
    Args:
        chunks: List of text chunks
        separator: The separator to use between chunks
        
    Returns:
        Merged content string
    """
    return separator.join(chunks)


def create_bilingual_content(
    original_chunks: list[str],
    translated_chunks: list[str],
    separator: str = DEFAULT_CHUNK_SEPARATOR,
) -> str:
    """
    Create bilingual content with original and translated text aligned.
    
    Each original paragraph is followed by its translation, creating
    a paragraph-by-paragraph bilingual document.
    
    Args:
        original_chunks: List of original text chunks
        translated_chunks: List of translated text chunks
        separator: The separator between paragraph pairs
        
    Returns:
        Bilingual content with aligned paragraphs
        
    Validates: Requirement 5.5 - Bilingual document with paragraph alignment
    """
    if len(original_chunks) != len(translated_chunks):
        logger.warning(
            "Chunk count mismatch: original=%d, translated=%d",
            len(original_chunks),
            len(translated_chunks),
        )
    
    bilingual_parts = []
    
    # Pair up original and translated chunks
    for i, (original, translated) in enumerate(
        zip(original_chunks, translated_chunks)
    ):
        # Format: original paragraph followed by translation
        bilingual_parts.append(original)
        bilingual_parts.append(translated)
    
    return separator.join(bilingual_parts)


class AIClient:
    """
    Client for AI-powered translation and summarization.
    
    This client provides:
    - Language detection (English or Japanese)
    - Chunk-based translation with concurrent processing
    - Bilingual document generation
    - Bilingual summary generation
    - Retry logic with exponential backoff
    
    The client uses an OpenAI-compatible API with the "gpt-5-nano" model
    as specified in the requirements.
    
    Attributes:
        endpoint: The OpenAI-compatible API endpoint URL
        api_key: The API key for authentication
        model: The model name (default: "gpt-5-nano")
        max_concurrency: Maximum concurrent API requests
    """
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        model: str = "gpt-5-nano",
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ):
        """
        Initialize the AI client.
        
        Args:
            endpoint: OpenAI-compatible API endpoint URL (Requirement 11.2)
            api_key: API key for authentication (Requirement 11.3)
            model: Model name (default: "gpt-5-nano" per Requirements 5.2, 6.3)
            max_concurrency: Maximum concurrent requests (Requirement 5.4)
            
        Raises:
            ValueError: If endpoint or api_key is empty
        """
        if not endpoint:
            raise ValueError("endpoint is required")
        if not api_key:
            raise ValueError("api_key is required")
        
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_concurrency = max_concurrency
        
        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrency)
    
    def _get_headers(self) -> dict[str, str]:
        """
        Get the authorization headers for API requests.
        
        Returns:
            Dictionary with Authorization and Content-Type headers
        """
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    async def _call_api(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
    ) -> str:
        """
        Make a single API call to the OpenAI-compatible endpoint.
        
        Args:
            messages: List of message objects with role and content
            temperature: Sampling temperature (lower = more deterministic)
            
        Returns:
            The assistant's response content
            
        Raises:
            AIClientError: If the API call fails
        """
        url = f"{self.endpoint}/chat/completions"
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        
        async with self._semaphore:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=payload,
                        headers=self._get_headers(),
                    ) as response:
                        if response.status >= 400:
                            error_text = await response.text()

                            if response.status == 429:
                                raise AIClientError(
                                    message="Rate limit exceeded",
                                    retry_count=0,
                                    status_code=response.status,
                                    retryable=True,
                                )

                            # Most 4xx errors are non-retryable (bad request, auth, etc.)
                            retryable = response.status >= 500

                            if response.status in (401, 403):
                                raise AIClientError(
                                    message=(
                                        f"AI API authentication failed (HTTP {response.status}). "
                                        "Check AI_API_ENDPOINT, AI_API_KEY, and AI_MODEL. "
                                        f"Provider response: {error_text}"
                                    ),
                                    retry_count=0,
                                    status_code=response.status,
                                    retryable=False,
                                )

                            raise AIClientError(
                                message=f"API error {response.status}: {error_text}",
                                retry_count=0,
                                status_code=response.status,
                                retryable=retryable,
                            )
                        
                        response_data = await response.json()
                        
                        # Extract response content
                        choices = response_data.get("choices", [])
                        if not choices:
                            raise AIClientError(
                                message="No choices in API response",
                                retry_count=0,
                                retryable=False,
                            )
                        
                        content = choices[0].get("message", {}).get("content", "")
                        return content
            
            except aiohttp.ClientError as e:
                raise AIClientError(
                    message=f"Network error: {e}",
                    retry_count=0,
                    original_error=e,
                    retryable=True,
                ) from e
            except asyncio.TimeoutError as e:
                raise AIClientError(
                    message="Request timeout",
                    retry_count=0,
                    original_error=e,
                    retryable=True,
                ) from e
    
    async def _call_api_with_retry(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
    ) -> str:
        """
        Make an API call with retry logic.
        
        Implements exponential backoff retry with up to MAX_RETRIES attempts.
        
        Args:
            messages: List of message objects
            temperature: Sampling temperature
            
        Returns:
            The assistant's response content
            
        Raises:
            AIClientError: If all retries fail
            
        Validates: Requirements 5.7, 6.5 - Retry up to 3 times
        """
        last_error: Optional[AIClientError] = None
        
        for attempt in range(MAX_RETRIES + 1):  # 1 initial + 3 retries = 4 total
            try:
                return await self._call_api(messages, temperature)
            
            except AIClientError as e:
                last_error = e

                if not e.retryable:
                    logger.error("Non-retryable AI API error: %s", e.message)
                    raise
                
                if attempt < MAX_RETRIES:
                    # Calculate backoff delay
                    delay = INITIAL_RETRY_DELAY * (RETRY_BACKOFF_MULTIPLIER ** attempt)
                    
                    logger.warning(
                        "API call failed (attempt %d/%d), retrying in %.1f seconds: %s",
                        attempt + 1,
                        MAX_RETRIES + 1,
                        delay,
                        e.message,
                    )
                    
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "API call failed after %d attempts: %s",
                        MAX_RETRIES + 1,
                        e.message,
                    )
        
        # All retries exhausted
        raise AIClientError(
            message=f"Failed after {MAX_RETRIES + 1} attempts: {last_error.message if last_error else 'Unknown error'}",
            retry_count=MAX_RETRIES,
            original_error=last_error,
            status_code=last_error.status_code if last_error else None,
            retryable=last_error.retryable if last_error else True,
        )
    
    async def detect_language(self, text: str) -> str:
        """
        Detect the language of the given text.
        
        Uses character-based heuristics to detect English or Japanese.
        This approach is fast and doesn't require an API call.
        
        Args:
            text: The text to analyze
            
        Returns:
            "en" for English, "ja" for Japanese
            
        Validates: Requirement 5.1 - Detect source language
        """
        return detect_language_heuristic(text)
    
    async def translate_chunk(
        self,
        text: str,
        source_lang: str,
        target_lang: str = "zh",
    ) -> str:
        """
        Translate a single text chunk with retry logic.
        
        Args:
            text: The text to translate
            source_lang: Source language code ("en" or "ja")
            target_lang: Target language code (default: "zh" for Chinese)
            
        Returns:
            The translated text
            
        Raises:
            AIClientError: If translation fails after retries
            
        Validates: Requirements 5.2, 5.7 - Translate with retry
        """
        if not text or not text.strip():
            return ""
        
        # Build language names for the prompt
        lang_names = {
            "en": "English",
            "ja": "Japanese",
            "zh": "Chinese",
        }
        
        source_name = lang_names.get(source_lang, source_lang)
        target_name = lang_names.get(target_lang, target_lang)
        
        # Create translation prompt
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a professional translator. Translate the following "
                    f"{source_name} text to {target_name}. "
                    f"Preserve the original formatting, including markdown syntax. "
                    f"Only output the translation, no explanations."
                ),
            },
            {
                "role": "user",
                "content": text,
            },
        ]
        
        logger.debug(
            "Translating chunk (%d chars) from %s to %s",
            len(text),
            source_lang,
            target_lang,
        )
        
        translated = await self._call_api_with_retry(messages)
        
        return translated.strip()
    
    async def translate_document(
        self,
        content: str,
        source_lang: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> str:
        """
        Translate an entire document with concurrent chunk processing.
        
        This method:
        1. Splits the document into chunks by paragraphs
        2. Translates chunks concurrently (up to max_concurrency)
        3. Creates a bilingual document with aligned paragraphs
        
        Args:
            content: The Markdown document content
            source_lang: Source language code ("en" or "ja")
            progress_callback: Optional callback(completed, total) for progress
            
        Returns:
            Bilingual Markdown document with original and translated aligned
            
        Raises:
            AIClientError: If translation fails
            
        Validates: Requirements 5.3, 5.4, 5.5 - Chunk, concurrent, bilingual
        """
        if not content or not content.strip():
            return ""
        
        # Split into chunks (Requirement 5.3)
        chunks = split_into_chunks(content)
        
        if not chunks:
            return ""
        
        total_chunks = len(chunks)
        completed_chunks = 0
        
        logger.info(
            "Translating document: %d chunks, source=%s",
            total_chunks,
            source_lang,
        )
        
        translated_chunks: list[Optional[str]] = [None] * total_chunks

        # Fail fast: if one chunk hits a non-retryable error (e.g., auth/model),
        # cancel remaining chunks to avoid spamming the provider.
        async def translate_with_progress(chunk: str, index: int) -> None:
            nonlocal completed_chunks

            translated = await self.translate_chunk(chunk, source_lang, "zh")
            translated_chunks[index] = translated

            completed_chunks += 1
            if progress_callback:
                progress_callback(completed_chunks, total_chunks)

        try:
            async with asyncio.TaskGroup() as tg:
                for i, chunk in enumerate(chunks):
                    tg.create_task(translate_with_progress(chunk, i))
        except* AIClientError as eg:
            first = eg.exceptions[0] if eg.exceptions else AIClientError("Unknown AI error")
            logger.error("Document translation failed: %s", getattr(first, "message", str(first)))
            raise AIClientError(
                message=getattr(first, "message", str(first)),
                retry_count=getattr(first, "retry_count", 0),
                original_error=first,
                status_code=getattr(first, "status_code", None),
                retryable=getattr(first, "retryable", False),
            ) from first
        except* Exception as eg:
            first = eg.exceptions[0] if eg.exceptions else Exception("Unknown error")
            logger.error("Document translation failed: %s", first)
            raise AIClientError(
                message=str(first),
                retry_count=0,
                original_error=first,
                retryable=False,
            ) from first
        
        # Ensure all chunks were translated
        final_translated = [t if t is not None else "" for t in translated_chunks]
        
        # Create bilingual document (Requirement 5.5)
        bilingual_content = create_bilingual_content(chunks, final_translated)
        
        logger.info(
            "Document translation complete: %d chunks translated",
            total_chunks,
        )
        
        return bilingual_content
    
    async def summarize(
        self,
        content: str,
        source_lang: str,
    ) -> str:
        """
        Generate a bilingual summary of the document.
        
        This method:
        1. Generates a summary in the source language (Requirement 6.1)
        2. Translates the summary to Chinese (Requirement 6.2)
        3. Returns a bilingual summary document (Requirement 6.4)
        
        Args:
            content: The document content to summarize
            source_lang: Source language code ("en" or "ja")
            
        Returns:
            Bilingual summary with source language and Chinese versions
            
        Raises:
            AIClientError: If summarization fails after retries
            
        Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5
        """
        if not content or not content.strip():
            return ""
        
        # Build language name for the prompt
        lang_names = {
            "en": "English",
            "ja": "Japanese",
        }
        source_name = lang_names.get(source_lang, source_lang)
        
        # Step 1: Generate summary in source language (Requirement 6.1)
        summary_messages = [
            {
                "role": "system",
                "content": (
                    f"You are a professional research analyst. "
                    f"Summarize the following research report in {source_name}. "
                    f"Focus on key findings, conclusions, and recommendations. "
                    f"Keep the summary concise but comprehensive."
                ),
            },
            {
                "role": "user",
                "content": content,
            },
        ]
        
        logger.info(
            "Generating summary in %s for document (%d chars)",
            source_lang,
            len(content),
        )
        
        # Generate source language summary with retry (Requirement 6.5)
        source_summary = await self._call_api_with_retry(summary_messages)
        source_summary = source_summary.strip()
        
        logger.debug(
            "Source summary generated: %d chars",
            len(source_summary),
        )
        
        # Step 2: Translate summary to Chinese (Requirement 6.2)
        chinese_summary = await self.translate_chunk(
            source_summary,
            source_lang,
            "zh",
        )
        
        logger.debug(
            "Chinese summary generated: %d chars",
            len(chinese_summary),
        )
        
        # Step 3: Create bilingual summary document (Requirement 6.4)
        # Format: Source language summary followed by Chinese translation
        bilingual_summary = (
            f"## Summary ({source_name})\n\n"
            f"{source_summary}\n\n"
            f"## 摘要 (中文)\n\n"
            f"{chinese_summary}"
        )
        
        logger.info("Bilingual summary generation complete")
        
        return bilingual_summary
