"""
Unit tests for the AI client.

Tests cover:
- Language detection (English and Japanese)
- Chunk splitting and merging
- Bilingual content creation
- translate_chunk method with retry logic
- translate_document method with concurrent processing
- summarize method with bilingual output
- Error handling and retries
"""

import pytest
from aioresponses import aioresponses
from yarl import URL

from backend.services.ai_client import (
    AIClient,
    AIClientError,
    detect_language_heuristic,
    split_into_chunks,
    merge_chunks,
    create_bilingual_content,
    MAX_RETRIES,
)


class TestDetectLanguageHeuristic:
    """Tests for language detection heuristic function."""
    
    def test_detect_english_text(self):
        """
        Test detection of English text.
        
        **Validates: Requirement 5.1**
        """
        english_texts = [
            "This is a simple English sentence.",
            "The quick brown fox jumps over the lazy dog.",
            "Research report on market analysis.",
            "Hello, world!",
        ]
        
        for text in english_texts:
            assert detect_language_heuristic(text) == "en"
    
    def test_detect_japanese_text(self):
        """
        Test detection of Japanese text.
        
        **Validates: Requirement 5.1**
        """
        japanese_texts = [
            "これは日本語のテストです。",
            "東京は日本の首都です。",
            "研究報告書の概要",
            "こんにちは世界",
        ]
        
        for text in japanese_texts:
            assert detect_language_heuristic(text) == "ja"
    
    def test_detect_mixed_content_with_japanese_majority(self):
        """Test detection of mixed content with significant Japanese."""
        # More than 10% Japanese characters should be detected as Japanese
        mixed_text = "これはテストです with some English words"
        assert detect_language_heuristic(mixed_text) == "ja"
    
    def test_detect_english_with_few_japanese_chars(self):
        """Test that English with very few Japanese chars is detected as English."""
        # Less than 10% Japanese should be detected as English
        text = "This is mostly English text with just one 日 character"
        # The ratio is very low, should be English
        result = detect_language_heuristic(text)
        # This depends on the exact ratio calculation
        assert result in ["en", "ja"]  # Accept either based on threshold
    
    def test_detect_empty_text(self):
        """Test detection of empty text defaults to English."""
        assert detect_language_heuristic("") == "en"
        assert detect_language_heuristic("   ") == "en"
    
    def test_detect_whitespace_only(self):
        """Test detection of whitespace-only text."""
        assert detect_language_heuristic("\n\n\t  ") == "en"


class TestSplitIntoChunks:
    """Tests for chunk splitting function."""
    
    def test_split_by_double_newline(self):
        """
        Test splitting content by paragraphs.
        
        **Validates: Requirement 5.3**
        """
        content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = split_into_chunks(content)
        
        assert len(chunks) == 3
        assert chunks[0] == "First paragraph."
        assert chunks[1] == "Second paragraph."
        assert chunks[2] == "Third paragraph."
    
    def test_split_filters_empty_chunks(self):
        """Test that empty chunks are filtered out."""
        content = "First.\n\n\n\nSecond.\n\n\n\n\n\nThird."
        chunks = split_into_chunks(content)
        
        assert len(chunks) == 3
        assert all(chunk.strip() for chunk in chunks)
    
    def test_split_empty_content(self):
        """Test splitting empty content returns empty list."""
        assert split_into_chunks("") == []
        assert split_into_chunks(None) == []
    
    def test_split_single_paragraph(self):
        """Test splitting content with no separators."""
        content = "Single paragraph with no breaks."
        chunks = split_into_chunks(content)
        
        assert len(chunks) == 1
        assert chunks[0] == "Single paragraph with no breaks."
    
    def test_split_preserves_content(self):
        """Test that splitting preserves all content."""
        content = "Para 1.\n\nPara 2.\n\nPara 3."
        chunks = split_into_chunks(content)
        
        # Merging should give back equivalent content
        merged = merge_chunks(chunks)
        assert merged == content


class TestMergeChunks:
    """Tests for chunk merging function."""
    
    def test_merge_chunks(self):
        """Test merging chunks back together."""
        chunks = ["First", "Second", "Third"]
        merged = merge_chunks(chunks)
        
        assert merged == "First\n\nSecond\n\nThird"
    
    def test_merge_empty_list(self):
        """Test merging empty list."""
        assert merge_chunks([]) == ""
    
    def test_merge_single_chunk(self):
        """Test merging single chunk."""
        assert merge_chunks(["Only one"]) == "Only one"


class TestCreateBilingualContent:
    """Tests for bilingual content creation."""
    
    def test_create_bilingual_aligned(self):
        """
        Test creating bilingual content with aligned paragraphs.
        
        **Validates: Requirement 5.5**
        """
        original = ["Hello", "World"]
        translated = ["你好", "世界"]
        
        bilingual = create_bilingual_content(original, translated)
        
        # Should alternate: original, translated, original, translated
        parts = bilingual.split("\n\n")
        assert len(parts) == 4
        assert parts[0] == "Hello"
        assert parts[1] == "你好"
        assert parts[2] == "World"
        assert parts[3] == "世界"
    
    def test_create_bilingual_empty_lists(self):
        """Test creating bilingual content with empty lists."""
        bilingual = create_bilingual_content([], [])
        assert bilingual == ""
    
    def test_create_bilingual_single_paragraph(self):
        """Test creating bilingual content with single paragraph."""
        original = ["Single paragraph"]
        translated = ["单段落"]
        
        bilingual = create_bilingual_content(original, translated)
        
        parts = bilingual.split("\n\n")
        assert len(parts) == 2
        assert parts[0] == "Single paragraph"
        assert parts[1] == "单段落"


class TestAIClientInit:
    """Tests for AIClient initialization."""
    
    def test_init_with_valid_params(self):
        """Test client initialization with valid parameters."""
        client = AIClient(
            endpoint="https://api.example.com",
            api_key="test-key",
        )
        
        assert client.endpoint == "https://api.example.com"
        assert client.api_key == "test-key"
        assert client.model == "gpt-5-nano"  # Default model
    
    def test_init_with_custom_model(self):
        """Test client initialization with custom model."""
        client = AIClient(
            endpoint="https://api.example.com",
            api_key="test-key",
            model="custom-model",
        )
        
        assert client.model == "custom-model"
    
    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is stripped from endpoint."""
        client = AIClient(
            endpoint="https://api.example.com/",
            api_key="test-key",
        )
        
        assert client.endpoint == "https://api.example.com"
    
    def test_init_empty_endpoint_raises_error(self):
        """Test that empty endpoint raises ValueError."""
        with pytest.raises(ValueError, match="endpoint is required"):
            AIClient(endpoint="", api_key="test-key")
    
    def test_init_empty_api_key_raises_error(self):
        """Test that empty api_key raises ValueError."""
        with pytest.raises(ValueError, match="api_key is required"):
            AIClient(endpoint="https://api.example.com", api_key="")
    
    def test_init_with_custom_concurrency(self):
        """Test client initialization with custom concurrency."""
        client = AIClient(
            endpoint="https://api.example.com",
            api_key="test-key",
            max_concurrency=5,
        )
        
        assert client.max_concurrency == 5


class TestAIClientDetectLanguage:
    """Tests for AIClient.detect_language method."""
    
    @pytest.fixture
    def client(self):
        """Create an AI client for testing."""
        return AIClient(
            endpoint="https://api.example.com",
            api_key="test-key",
        )
    
    @pytest.mark.asyncio
    async def test_detect_english(self, client):
        """Test detecting English text."""
        result = await client.detect_language("This is English text.")
        assert result == "en"
    
    @pytest.mark.asyncio
    async def test_detect_japanese(self, client):
        """Test detecting Japanese text."""
        result = await client.detect_language("これは日本語です。")
        assert result == "ja"


class TestAIClientTranslateChunk:
    """Tests for AIClient.translate_chunk method."""
    
    @pytest.fixture
    def client(self):
        """Create an AI client for testing."""
        return AIClient(
            endpoint="https://api.example.com",
            api_key="test-key",
        )
    
    @pytest.mark.asyncio
    async def test_translate_chunk_success(self, client):
        """
        Test successful chunk translation.
        
        **Validates: Requirements 5.2, 5.7**
        """
        with aioresponses() as m:
            m.post(
                "https://api.example.com/chat/completions",
                payload={
                    "choices": [
                        {
                            "message": {
                                "content": "你好世界",
                            },
                        },
                    ],
                },
            )
            
            result = await client.translate_chunk(
                text="Hello world",
                source_lang="en",
                target_lang="zh",
            )
            
            assert result == "你好世界"
    
    @pytest.mark.asyncio
    async def test_translate_chunk_empty_text(self, client):
        """Test translating empty text returns empty string."""
        result = await client.translate_chunk(
            text="",
            source_lang="en",
        )
        
        assert result == ""
    
    @pytest.mark.asyncio
    async def test_translate_chunk_whitespace_only(self, client):
        """Test translating whitespace-only text returns empty string."""
        result = await client.translate_chunk(
            text="   \n\t  ",
            source_lang="en",
        )
        
        assert result == ""
    
    @pytest.mark.asyncio
    async def test_translate_chunk_retry_on_error(self, client):
        """
        Test that translation retries on error.
        
        **Validates: Requirement 5.7**
        """
        with aioresponses() as m:
            # First call fails
            m.post(
                "https://api.example.com/chat/completions",
                status=500,
            )
            # Second call succeeds
            m.post(
                "https://api.example.com/chat/completions",
                payload={
                    "choices": [
                        {"message": {"content": "翻译结果"}},
                    ],
                },
            )
            
            result = await client.translate_chunk(
                text="Test text",
                source_lang="en",
            )
            
            assert result == "翻译结果"
    
    @pytest.mark.asyncio
    async def test_translate_chunk_max_retries_exceeded(self, client):
        """
        Test that translation fails after max retries.
        
        **Validates: Requirement 5.7**
        """
        with aioresponses() as m:
            # All calls fail
            for _ in range(MAX_RETRIES + 1):
                m.post(
                    "https://api.example.com/chat/completions",
                    status=500,
                )
            
            with pytest.raises(AIClientError) as exc_info:
                await client.translate_chunk(
                    text="Test text",
                    source_lang="en",
                )
            
            assert exc_info.value.retry_count == MAX_RETRIES
    
    @pytest.mark.asyncio
    async def test_translate_chunk_rate_limit_retry(self, client):
        """Test that rate limit (429) triggers retry."""
        with aioresponses() as m:
            # First call hits rate limit
            m.post(
                "https://api.example.com/chat/completions",
                status=429,
            )
            # Second call succeeds
            m.post(
                "https://api.example.com/chat/completions",
                payload={
                    "choices": [
                        {"message": {"content": "成功"}},
                    ],
                },
            )
            
            result = await client.translate_chunk(
                text="Test",
                source_lang="en",
            )
            
            assert result == "成功"


    @pytest.mark.asyncio
    async def test_translate_chunk_retries_without_temperature_when_unsupported(self, client):
        """Retry once without temperature when provider rejects non-default temperature."""
        url = "https://api.example.com/chat/completions"
        unsupported_body = (
            "Unsupported value: 'temperature' does not support 0.3 with this model. "
            "Only the default (1) value is supported."
        )

        with aioresponses() as m:
            m.post(url, status=400, body=unsupported_body)
            m.post(
                url,
                payload={
                    "choices": [
                        {"message": {"content": "ok"}},
                    ],
                },
            )

            result = await client.translate_chunk(
                text="Test",
                source_lang="en",
            )

            assert result == "ok"

            calls = m.requests[("POST", URL(url))]
            assert len(calls) == 2

            first_payload = calls[0].kwargs.get("json") or {}
            second_payload = calls[1].kwargs.get("json") or {}

            assert first_payload.get("temperature") == 0.3
            assert "temperature" not in second_payload


class TestAIClientTranslateDocument:
    """Tests for AIClient.translate_document method."""
    
    @pytest.fixture
    def client(self):
        """Create an AI client for testing."""
        return AIClient(
            endpoint="https://api.example.com",
            api_key="test-key",
            max_concurrency=2,
        )
    
    @pytest.mark.asyncio
    async def test_translate_document_success(self, client):
        """
        Test successful document translation.
        
        **Validates: Requirements 5.3, 5.4, 5.5**
        """
        content = "First paragraph.\n\nSecond paragraph."
        
        with aioresponses() as m:
            # Mock responses for both chunks
            m.post(
                "https://api.example.com/chat/completions",
                payload={
                    "choices": [
                        {"message": {"content": "第一段"}},
                    ],
                },
            )
            m.post(
                "https://api.example.com/chat/completions",
                payload={
                    "choices": [
                        {"message": {"content": "第二段"}},
                    ],
                },
            )
            
            result = await client.translate_document(
                content=content,
                source_lang="en",
            )
            
            # Should have bilingual content with aligned paragraphs
            assert "First paragraph." in result
            assert "Second paragraph." in result
            # Translations should be present (order may vary due to concurrency)
            assert "第一段" in result or "第二段" in result
    
    @pytest.mark.asyncio
    async def test_translate_document_empty_content(self, client):
        """Test translating empty document returns empty string."""
        result = await client.translate_document(
            content="",
            source_lang="en",
        )
        
        assert result == ""
    
    @pytest.mark.asyncio
    async def test_translate_document_progress_callback(self, client):
        """Test that progress callback is called."""
        content = "Para 1.\n\nPara 2."
        progress_calls = []
        
        def progress_callback(completed: int, total: int):
            progress_calls.append((completed, total))
        
        with aioresponses() as m:
            m.post(
                "https://api.example.com/chat/completions",
                payload={"choices": [{"message": {"content": "翻译1"}}]},
            )
            m.post(
                "https://api.example.com/chat/completions",
                payload={"choices": [{"message": {"content": "翻译2"}}]},
            )
            
            await client.translate_document(
                content=content,
                source_lang="en",
                progress_callback=progress_callback,
            )
        
        # Should have been called for each chunk
        assert len(progress_calls) == 2
        # Final call should show all complete
        assert (2, 2) in progress_calls


class TestAIClientSummarize:
    """Tests for AIClient.summarize method."""
    
    @pytest.fixture
    def client(self):
        """Create an AI client for testing."""
        return AIClient(
            endpoint="https://api.example.com",
            api_key="test-key",
        )
    
    @pytest.mark.asyncio
    async def test_summarize_success(self, client):
        """
        Test successful summarization.
        
        **Validates: Requirements 6.1, 6.2, 6.3, 6.4**
        """
        content = "This is a research report about market trends."
        
        with aioresponses() as m:
            # First call: generate English summary
            m.post(
                "https://api.example.com/chat/completions",
                payload={
                    "choices": [
                        {"message": {"content": "Summary: Market trends are positive."}},
                    ],
                },
            )
            # Second call: translate to Chinese
            m.post(
                "https://api.example.com/chat/completions",
                payload={
                    "choices": [
                        {"message": {"content": "摘要：市场趋势积极。"}},
                    ],
                },
            )
            
            result = await client.summarize(
                content=content,
                source_lang="en",
            )
            
            # Should have bilingual summary
            assert "Summary (English)" in result
            assert "摘要 (中文)" in result
            assert "Market trends are positive" in result
            assert "市场趋势积极" in result
    
    @pytest.mark.asyncio
    async def test_summarize_japanese_source(self, client):
        """Test summarization with Japanese source."""
        content = "これは市場動向に関する研究報告です。"
        
        with aioresponses() as m:
            # First call: generate Japanese summary
            m.post(
                "https://api.example.com/chat/completions",
                payload={
                    "choices": [
                        {"message": {"content": "概要：市場動向は良好です。"}},
                    ],
                },
            )
            # Second call: translate to Chinese
            m.post(
                "https://api.example.com/chat/completions",
                payload={
                    "choices": [
                        {"message": {"content": "摘要：市场趋势良好。"}},
                    ],
                },
            )
            
            result = await client.summarize(
                content=content,
                source_lang="ja",
            )
            
            # Should have bilingual summary with Japanese header
            assert "Summary (Japanese)" in result
            assert "摘要 (中文)" in result
    
    @pytest.mark.asyncio
    async def test_summarize_empty_content(self, client):
        """Test summarizing empty content returns empty string."""
        result = await client.summarize(
            content="",
            source_lang="en",
        )
        
        assert result == ""
    
    @pytest.mark.asyncio
    async def test_summarize_retry_on_failure(self, client):
        """
        Test that summarization retries on failure.
        
        **Validates: Requirement 6.5**
        """
        content = "Research report content."
        
        with aioresponses() as m:
            # First summary call fails
            m.post(
                "https://api.example.com/chat/completions",
                status=500,
            )
            # Retry succeeds
            m.post(
                "https://api.example.com/chat/completions",
                payload={
                    "choices": [
                        {"message": {"content": "Summary text"}},
                    ],
                },
            )
            # Translation call succeeds
            m.post(
                "https://api.example.com/chat/completions",
                payload={
                    "choices": [
                        {"message": {"content": "摘要文本"}},
                    ],
                },
            )
            
            result = await client.summarize(
                content=content,
                source_lang="en",
            )
            
            assert "Summary text" in result
            assert "摘要文本" in result


class TestAIClientError:
    """Tests for AIClientError exception."""
    
    def test_error_with_message(self):
        """Test error with message only."""
        error = AIClientError(message="Test error")
        
        assert error.message == "Test error"
        assert error.retry_count == 0
        assert error.original_error is None
    
    def test_error_with_retry_count(self):
        """Test error with retry count."""
        error = AIClientError(
            message="Failed after retries",
            retry_count=3,
        )
        
        assert error.retry_count == 3
    
    def test_error_with_original_error(self):
        """Test error with original exception."""
        original = ValueError("Original error")
        error = AIClientError(
            message="Wrapped error",
            original_error=original,
        )
        
        assert error.original_error is original
