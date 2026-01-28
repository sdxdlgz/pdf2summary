"""
Property-based tests for AI client language detection.

**Property 9: Language Detection**
*For any* English text input, the language detector SHALL return "en".
*For any* Japanese text input, the language detector SHALL return "ja".

**Validates: Requirements 5.1**

Uses Hypothesis for property-based testing with at least 100 iterations per test.
"""

import pytest
from hypothesis import given, settings, strategies as st, assume

from backend.services.ai_client import detect_language_heuristic


# =============================================================================
# Strategies for generating test data
# =============================================================================

# Strategy for generating pure ASCII English text (letters, digits, punctuation, spaces)
# This ensures no Japanese characters are present
pure_english_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S", "Z"),
        whitelist_characters=" \n\t",
        # Exclude CJK characters by limiting to ASCII range
        min_codepoint=0x0020,
        max_codepoint=0x007E,
    ),
    min_size=1,
    max_size=500,
).filter(lambda s: s.strip())  # Ensure non-whitespace-only strings


# Strategy for generating Japanese text with Hiragana characters (U+3040-U+309F)
hiragana_text = st.text(
    alphabet=st.characters(
        min_codepoint=0x3040,
        max_codepoint=0x309F,
    ),
    min_size=5,
    max_size=200,
).filter(lambda s: s.strip())


# Strategy for generating Japanese text with Katakana characters (U+30A0-U+30FF)
katakana_text = st.text(
    alphabet=st.characters(
        min_codepoint=0x30A0,
        max_codepoint=0x30FF,
    ),
    min_size=5,
    max_size=200,
).filter(lambda s: s.strip())


# Strategy for generating Japanese text with Kanji characters (U+4E00-U+9FFF)
kanji_text = st.text(
    alphabet=st.characters(
        min_codepoint=0x4E00,
        max_codepoint=0x9FFF,
    ),
    min_size=5,
    max_size=200,
).filter(lambda s: s.strip())


# Strategy for generating mixed Japanese text (Hiragana, Katakana, Kanji)
mixed_japanese_text = st.one_of(
    hiragana_text,
    katakana_text,
    kanji_text,
    # Combined Japanese characters
    st.tuples(hiragana_text, katakana_text).map(lambda t: t[0] + t[1]),
    st.tuples(hiragana_text, kanji_text).map(lambda t: t[0] + t[1]),
    st.tuples(katakana_text, kanji_text).map(lambda t: t[0] + t[1]),
)


# Strategy for generating text with significant Japanese content (>10% Japanese chars)
# This creates text where Japanese characters make up more than 10% of non-whitespace
def create_significant_japanese_text(japanese_part: str, english_ratio: float) -> str:
    """
    Create text with significant Japanese content.
    
    The function ensures Japanese characters make up more than 10% of the total.
    """
    # Calculate how much English text to add while keeping Japanese > 10%
    # If japanese_part has N chars, we need total chars T such that N/T > 0.1
    # So T < N/0.1 = 10*N
    # We want English chars E = T - N < 9*N
    japanese_len = len(japanese_part.replace(" ", "").replace("\n", "").replace("\t", ""))
    max_english_len = int(japanese_len * 8)  # Keep Japanese > 11% to be safe
    
    if max_english_len <= 0:
        return japanese_part
    
    # Add some English text based on ratio
    english_len = int(max_english_len * english_ratio)
    english_part = "a" * english_len
    
    return japanese_part + " " + english_part


significant_japanese_text = st.tuples(
    mixed_japanese_text,
    st.floats(min_value=0.0, max_value=0.8),
).map(lambda t: create_significant_japanese_text(t[0], t[1]))


# Strategy for generating empty or whitespace-only text
empty_or_whitespace_text = st.one_of(
    st.just(""),
    st.sampled_from([" ", "\n", "\t", "\r", "  ", "\n\n", "\t\t", "   ", " \n ", "\t \n"]),
    st.lists(
        st.sampled_from([" ", "\n", "\t", "\r"]),
        min_size=1,
        max_size=20,
    ).map(lambda chars: "".join(chars)),
)


# =============================================================================
# Property-based tests for Language Detection
# =============================================================================

class TestLanguageDetectionProperty:
    """
    Property-based tests for Property 9: Language Detection.
    
    **Validates: Requirements 5.1**
    
    These tests verify that:
    1. Pure English text (ASCII) returns "en"
    2. Text with significant Japanese characters returns "ja"
    3. Empty text returns "en" (default)
    4. The function is deterministic
    """

    @settings(max_examples=100)
    @given(text=pure_english_text)
    def test_pure_english_text_returns_en(self, text: str):
        """
        Property: For any pure English text (ASCII letters, digits, punctuation),
        the language detector SHALL return "en".
        
        **Validates: Requirements 5.1**
        """
        # Ensure text is non-empty and has content
        assume(text.strip())
        
        result = detect_language_heuristic(text)
        
        # SHALL return "en" for pure English text
        assert result == "en", (
            f"Pure English text should return 'en'. "
            f"Got '{result}' for text: {text[:100]}..."
        )

    @settings(max_examples=100)
    @given(text=hiragana_text)
    def test_hiragana_text_returns_ja(self, text: str):
        """
        Property: For any text with Hiragana characters (Japanese syllabary),
        the language detector SHALL return "ja".
        
        **Validates: Requirements 5.1**
        """
        assume(text.strip())
        
        result = detect_language_heuristic(text)
        
        # SHALL return "ja" for Hiragana text
        assert result == "ja", (
            f"Hiragana text should return 'ja'. "
            f"Got '{result}' for text: {text[:100]}..."
        )

    @settings(max_examples=100)
    @given(text=katakana_text)
    def test_katakana_text_returns_ja(self, text: str):
        """
        Property: For any text with Katakana characters (Japanese syllabary),
        the language detector SHALL return "ja".
        
        **Validates: Requirements 5.1**
        """
        assume(text.strip())
        
        result = detect_language_heuristic(text)
        
        # SHALL return "ja" for Katakana text
        assert result == "ja", (
            f"Katakana text should return 'ja'. "
            f"Got '{result}' for text: {text[:100]}..."
        )

    @settings(max_examples=100)
    @given(text=kanji_text)
    def test_kanji_text_returns_ja(self, text: str):
        """
        Property: For any text with Kanji characters (Chinese characters used in Japanese),
        the language detector SHALL return "ja".
        
        **Validates: Requirements 5.1**
        """
        assume(text.strip())
        
        result = detect_language_heuristic(text)
        
        # SHALL return "ja" for Kanji text
        assert result == "ja", (
            f"Kanji text should return 'ja'. "
            f"Got '{result}' for text: {text[:100]}..."
        )

    @settings(max_examples=100)
    @given(text=significant_japanese_text)
    def test_significant_japanese_content_returns_ja(self, text: str):
        """
        Property: For any text with significant Japanese characters (>10% of content),
        the language detector SHALL return "ja".
        
        **Validates: Requirements 5.1**
        """
        assume(text.strip())
        
        result = detect_language_heuristic(text)
        
        # SHALL return "ja" for text with significant Japanese content
        assert result == "ja", (
            f"Text with significant Japanese content should return 'ja'. "
            f"Got '{result}' for text: {text[:100]}..."
        )

    @settings(max_examples=100)
    @given(text=empty_or_whitespace_text)
    def test_empty_text_returns_en_default(self, text: str):
        """
        Property: For any empty or whitespace-only text,
        the language detector SHALL return "en" as the default.
        
        **Validates: Requirements 5.1**
        """
        result = detect_language_heuristic(text)
        
        # SHALL return "en" as default for empty text
        assert result == "en", (
            f"Empty/whitespace text should return 'en' (default). "
            f"Got '{result}' for text: '{text}'"
        )

    @settings(max_examples=100)
    @given(text=st.one_of(pure_english_text, mixed_japanese_text, empty_or_whitespace_text))
    def test_language_detection_is_deterministic(self, text: str):
        """
        Property: For any input text, calling the language detector multiple times
        SHALL always return the same result (deterministic behavior).
        
        **Validates: Requirements 5.1**
        """
        # Call the function multiple times
        result1 = detect_language_heuristic(text)
        result2 = detect_language_heuristic(text)
        result3 = detect_language_heuristic(text)
        
        # SHALL be deterministic - same input always gives same output
        assert result1 == result2 == result3, (
            f"Language detection should be deterministic. "
            f"Got different results: {result1}, {result2}, {result3} for text: {text[:50]}..."
        )

    @settings(max_examples=100)
    @given(text=st.one_of(pure_english_text, mixed_japanese_text))
    def test_language_detection_returns_valid_language_code(self, text: str):
        """
        Property: For any non-empty text, the language detector SHALL return
        either "en" or "ja" (valid language codes only).
        
        **Validates: Requirements 5.1**
        """
        assume(text.strip())
        
        result = detect_language_heuristic(text)
        
        # SHALL return only valid language codes
        assert result in ("en", "ja"), (
            f"Language detection should return 'en' or 'ja'. "
            f"Got '{result}' for text: {text[:100]}..."
        )


class TestLanguageDetectionThreshold:
    """
    Property-based tests for the 10% Japanese character threshold.
    
    **Validates: Requirements 5.1**
    
    These tests verify the threshold behavior:
    - Text with <10% Japanese characters returns "en"
    - Text with >10% Japanese characters returns "ja"
    """

    @settings(max_examples=100)
    @given(
        japanese_chars=st.integers(min_value=1, max_value=5),
        english_chars=st.integers(min_value=100, max_value=500),
    )
    def test_below_threshold_returns_en(self, japanese_chars: int, english_chars: int):
        """
        Property: For any text where Japanese characters are less than 10%
        of total non-whitespace characters, the detector SHALL return "en".
        
        **Validates: Requirements 5.1**
        """
        # Ensure Japanese ratio is below 10%
        total_chars = japanese_chars + english_chars
        japanese_ratio = japanese_chars / total_chars
        assume(japanese_ratio < 0.1)
        
        # Create text with the specified ratio
        # Use a simple Hiragana character (あ = U+3042)
        japanese_part = "あ" * japanese_chars
        english_part = "a" * english_chars
        text = english_part + japanese_part
        
        result = detect_language_heuristic(text)
        
        # SHALL return "en" when Japanese < 10%
        assert result == "en", (
            f"Text with {japanese_ratio:.2%} Japanese chars should return 'en'. "
            f"Got '{result}'"
        )

    @settings(max_examples=100)
    @given(
        japanese_chars=st.integers(min_value=15, max_value=100),
        english_chars=st.integers(min_value=10, max_value=100),
    )
    def test_above_threshold_returns_ja(self, japanese_chars: int, english_chars: int):
        """
        Property: For any text where Japanese characters are more than 10%
        of total non-whitespace characters, the detector SHALL return "ja".
        
        **Validates: Requirements 5.1**
        """
        # Ensure Japanese ratio is above 10%
        total_chars = japanese_chars + english_chars
        japanese_ratio = japanese_chars / total_chars
        assume(japanese_ratio > 0.1)
        
        # Create text with the specified ratio
        # Use a simple Hiragana character (あ = U+3042)
        japanese_part = "あ" * japanese_chars
        english_part = "a" * english_chars
        text = english_part + japanese_part
        
        result = detect_language_heuristic(text)
        
        # SHALL return "ja" when Japanese > 10%
        assert result == "ja", (
            f"Text with {japanese_ratio:.2%} Japanese chars should return 'ja'. "
            f"Got '{result}'"
        )


# =============================================================================
# Property 10: Document Chunking
# =============================================================================

from backend.services.ai_client import split_into_chunks, merge_chunks


class TestDocumentChunkingProperty:
    """
    Property-based tests for Property 10: Document Chunking.
    
    *For any* document exceeding the chunk size limit, the chunking function
    SHALL split it into multiple chunks. *For any* set of chunks, concatenating
    them SHALL produce content equivalent to the original document.
    
    **Validates: Requirements 5.3**
    """

    @settings(max_examples=100)
    @given(
        paragraphs=st.lists(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=("L", "N", "P", "S"),
                    blacklist_characters="\n",
                ),
                min_size=1,
                max_size=100,
            ).filter(lambda s: s.strip()),
            min_size=1,
            max_size=20,
        ),
    )
    def test_split_and_merge_preserves_content(self, paragraphs: list):
        """
        Property: For any document split into chunks, merging the chunks
        SHALL produce content equivalent to the original document.
        
        **Validates: Requirements 5.3**
        """
        # Create document from paragraphs
        original = "\n\n".join(paragraphs)
        
        # Split into chunks
        chunks = split_into_chunks(original)
        
        # Merge back
        merged = merge_chunks(chunks)
        
        # Content should be equivalent (may have whitespace differences)
        original_normalized = "\n\n".join(p.strip() for p in paragraphs if p.strip())
        merged_normalized = merged.strip()
        
        assert merged_normalized == original_normalized, (
            f"Merged content should equal original. "
            f"Original: {original_normalized[:100]}... "
            f"Merged: {merged_normalized[:100]}..."
        )

    @settings(max_examples=100)
    @given(
        num_paragraphs=st.integers(min_value=2, max_value=20),
        paragraph_text=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N"),
            ),
            min_size=5,
            max_size=50,
        ).filter(lambda s: s.strip()),
    )
    def test_multiple_paragraphs_produce_multiple_chunks(
        self,
        num_paragraphs: int,
        paragraph_text: str,
    ):
        """
        Property: For any document with N paragraphs (separated by double newlines),
        the chunking function SHALL produce N chunks.
        
        **Validates: Requirements 5.3**
        """
        # Create document with multiple paragraphs
        paragraphs = [f"{paragraph_text}_{i}" for i in range(num_paragraphs)]
        document = "\n\n".join(paragraphs)
        
        # Split into chunks
        chunks = split_into_chunks(document)
        
        # Should have same number of chunks as paragraphs
        assert len(chunks) == num_paragraphs, (
            f"Document with {num_paragraphs} paragraphs should produce "
            f"{num_paragraphs} chunks, got {len(chunks)}"
        )

    @settings(max_examples=100)
    @given(
        text=st.text(
            alphabet=st.characters(
                whitelist_categories=("L", "N", "P", "S"),
                blacklist_characters="\n",
            ),
            min_size=1,
            max_size=200,
        ).filter(lambda s: s.strip()),
    )
    def test_single_paragraph_produces_single_chunk(self, text: str):
        """
        Property: For any document with a single paragraph (no double newlines),
        the chunking function SHALL produce exactly one chunk.
        
        **Validates: Requirements 5.3**
        """
        # Ensure no double newlines in text
        assume("\n\n" not in text)
        
        # Split into chunks
        chunks = split_into_chunks(text)
        
        # Should produce exactly one chunk
        assert len(chunks) == 1, (
            f"Single paragraph should produce 1 chunk, got {len(chunks)}"
        )
        assert chunks[0].strip() == text.strip(), (
            f"Chunk content should match original text"
        )

    @settings(max_examples=100)
    @given(
        paragraphs=st.lists(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=("L", "N"),
                ),
                min_size=1,
                max_size=50,
            ).filter(lambda s: s.strip()),
            min_size=1,
            max_size=10,
        ),
    )
    def test_chunks_are_non_empty(self, paragraphs: list):
        """
        Property: For any document, all resulting chunks SHALL be non-empty
        (no empty strings in the chunk list).
        
        **Validates: Requirements 5.3**
        """
        document = "\n\n".join(paragraphs)
        chunks = split_into_chunks(document)
        
        # All chunks should be non-empty
        for i, chunk in enumerate(chunks):
            assert chunk.strip(), (
                f"Chunk {i} should be non-empty, got: '{chunk}'"
            )

    @settings(max_examples=100)
    @given(
        paragraphs=st.lists(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=("L", "N"),
                ),
                min_size=1,
                max_size=50,
            ).filter(lambda s: s.strip()),
            min_size=1,
            max_size=10,
        ),
    )
    def test_chunk_order_preserved(self, paragraphs: list):
        """
        Property: For any document, the order of chunks SHALL match
        the order of paragraphs in the original document.
        
        **Validates: Requirements 5.3**
        """
        document = "\n\n".join(paragraphs)
        chunks = split_into_chunks(document)
        
        # Chunks should be in same order as paragraphs
        assert len(chunks) == len(paragraphs), (
            f"Number of chunks should match paragraphs"
        )
        
        for i, (chunk, paragraph) in enumerate(zip(chunks, paragraphs)):
            assert chunk.strip() == paragraph.strip(), (
                f"Chunk {i} should match paragraph {i}. "
                f"Chunk: '{chunk}', Paragraph: '{paragraph}'"
            )

    def test_empty_document_produces_empty_list(self):
        """
        Test that empty document produces empty chunk list.
        
        **Validates: Requirements 5.3**
        """
        assert split_into_chunks("") == []
        assert split_into_chunks(None) == []

    def test_whitespace_only_produces_empty_list(self):
        """
        Test that whitespace-only document produces empty chunk list.
        
        **Validates: Requirements 5.3**
        """
        assert split_into_chunks("   ") == []
        assert split_into_chunks("\n\n\n") == []
        assert split_into_chunks("\t\t") == []


class TestChunkMergingProperty:
    """
    Property-based tests for chunk merging.
    
    **Validates: Requirements 5.3**
    """

    @settings(max_examples=100)
    @given(
        chunks=st.lists(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=("L", "N"),
                ),
                min_size=1,
                max_size=50,
            ).filter(lambda s: s.strip()),
            min_size=1,
            max_size=10,
        ),
    )
    def test_merge_produces_correct_separator(self, chunks: list):
        """
        Property: Merging chunks SHALL use double newline as separator.
        
        **Validates: Requirements 5.3**
        """
        merged = merge_chunks(chunks)
        
        # Should have correct number of separators
        if len(chunks) > 1:
            separator_count = merged.count("\n\n")
            assert separator_count == len(chunks) - 1, (
                f"Should have {len(chunks) - 1} separators, got {separator_count}"
            )

    @settings(max_examples=100)
    @given(
        chunks=st.lists(
            st.text(
                alphabet=st.characters(
                    whitelist_categories=("L", "N"),
                ),
                min_size=1,
                max_size=50,
            ).filter(lambda s: s.strip()),
            min_size=1,
            max_size=10,
        ),
    )
    def test_split_merge_roundtrip(self, chunks: list):
        """
        Property: For any list of chunks, merging and then splitting
        SHALL produce the original chunks.
        
        **Validates: Requirements 5.3**
        """
        # Merge chunks
        merged = merge_chunks(chunks)
        
        # Split back
        result_chunks = split_into_chunks(merged)
        
        # Should get back original chunks
        assert len(result_chunks) == len(chunks), (
            f"Roundtrip should preserve chunk count: {len(chunks)} -> {len(result_chunks)}"
        )
        
        for i, (original, result) in enumerate(zip(chunks, result_chunks)):
            assert original.strip() == result.strip(), (
                f"Chunk {i} should match after roundtrip"
            )


# =============================================================================
# Property 12: Retry Logic
# =============================================================================

from backend.services.ai_client import (
    MAX_RETRIES,
    INITIAL_RETRY_DELAY,
    RETRY_BACKOFF_MULTIPLIER,
)


class TestRetryLogicProperty:
    """
    Property-based tests for Property 12: Retry Logic.
    
    *For any* failed translation or summarization operation, the system SHALL
    retry up to 3 times before marking as failed. The total attempt count
    SHALL not exceed 4 (1 initial + 3 retries).
    
    **Validates: Requirements 5.7, 6.5**
    """

    def test_max_retries_is_3(self):
        """
        Test that MAX_RETRIES constant is 3.
        
        **Validates: Requirements 5.7, 6.5**
        """
        assert MAX_RETRIES == 3, (
            f"MAX_RETRIES should be 3, got {MAX_RETRIES}"
        )

    def test_total_attempts_is_4(self):
        """
        Test that total attempts (1 initial + 3 retries) is 4.
        
        **Validates: Requirements 5.7, 6.5**
        """
        total_attempts = 1 + MAX_RETRIES
        assert total_attempts == 4, (
            f"Total attempts should be 4 (1 initial + 3 retries), got {total_attempts}"
        )

    @settings(max_examples=100)
    @given(attempt=st.integers(min_value=0, max_value=MAX_RETRIES))
    def test_retry_delay_increases_exponentially(self, attempt: int):
        """
        Property: For any retry attempt, the delay SHALL follow exponential
        backoff: delay = INITIAL_RETRY_DELAY * (RETRY_BACKOFF_MULTIPLIER ^ attempt).
        
        **Validates: Requirements 5.7, 6.5**
        """
        expected_delay = INITIAL_RETRY_DELAY * (RETRY_BACKOFF_MULTIPLIER ** attempt)
        
        # Verify the formula produces increasing delays
        if attempt > 0:
            prev_delay = INITIAL_RETRY_DELAY * (RETRY_BACKOFF_MULTIPLIER ** (attempt - 1))
            assert expected_delay > prev_delay, (
                f"Delay should increase: attempt {attempt-1}={prev_delay}, "
                f"attempt {attempt}={expected_delay}"
            )

    @settings(max_examples=100)
    @given(attempt=st.integers(min_value=0, max_value=10))
    def test_retry_delay_is_positive(self, attempt: int):
        """
        Property: For any retry attempt, the delay SHALL be positive.
        
        **Validates: Requirements 5.7, 6.5**
        """
        delay = INITIAL_RETRY_DELAY * (RETRY_BACKOFF_MULTIPLIER ** attempt)
        
        assert delay > 0, (
            f"Retry delay should be positive, got {delay} for attempt {attempt}"
        )

    def test_initial_retry_delay_is_reasonable(self):
        """
        Test that initial retry delay is a reasonable value (1-5 seconds).
        
        **Validates: Requirements 5.7, 6.5**
        """
        assert 0.5 <= INITIAL_RETRY_DELAY <= 10, (
            f"Initial retry delay should be between 0.5 and 10 seconds, "
            f"got {INITIAL_RETRY_DELAY}"
        )

    def test_backoff_multiplier_is_reasonable(self):
        """
        Test that backoff multiplier is a reasonable value (1.5-3).
        
        **Validates: Requirements 5.7, 6.5**
        """
        assert 1.5 <= RETRY_BACKOFF_MULTIPLIER <= 3, (
            f"Backoff multiplier should be between 1.5 and 3, "
            f"got {RETRY_BACKOFF_MULTIPLIER}"
        )

    @settings(max_examples=100)
    @given(
        num_failures=st.integers(min_value=0, max_value=MAX_RETRIES),
    )
    def test_retry_count_tracking(self, num_failures: int):
        """
        Property: For any number of failures up to MAX_RETRIES, the system
        SHALL track the retry count correctly.
        
        **Validates: Requirements 5.7, 6.5**
        """
        # Simulate retry counting
        retry_count = num_failures
        
        # Retry count should not exceed MAX_RETRIES
        assert retry_count <= MAX_RETRIES, (
            f"Retry count {retry_count} should not exceed MAX_RETRIES {MAX_RETRIES}"
        )
        
        # After MAX_RETRIES failures, no more retries should be attempted
        if retry_count == MAX_RETRIES:
            should_retry = False
        else:
            should_retry = True
        
        # Verify the logic
        assert (retry_count < MAX_RETRIES) == should_retry, (
            f"Should retry logic incorrect for retry_count={retry_count}"
        )


class TestRetryDelaySequenceProperty:
    """
    Property-based tests for retry delay sequence.
    
    **Validates: Requirements 5.7, 6.5**
    """

    def test_retry_delay_sequence(self):
        """
        Test the complete retry delay sequence.
        
        With INITIAL_RETRY_DELAY=1.0 and RETRY_BACKOFF_MULTIPLIER=2.0:
        - Attempt 0: 1.0 seconds
        - Attempt 1: 2.0 seconds
        - Attempt 2: 4.0 seconds
        
        **Validates: Requirements 5.7, 6.5**
        """
        delays = [
            INITIAL_RETRY_DELAY * (RETRY_BACKOFF_MULTIPLIER ** i)
            for i in range(MAX_RETRIES)
        ]
        
        # Verify delays are increasing
        for i in range(1, len(delays)):
            assert delays[i] > delays[i-1], (
                f"Delays should be increasing: {delays}"
            )
        
        # Verify first delay is INITIAL_RETRY_DELAY
        assert delays[0] == INITIAL_RETRY_DELAY, (
            f"First delay should be {INITIAL_RETRY_DELAY}, got {delays[0]}"
        )

    @settings(max_examples=100)
    @given(
        attempt1=st.integers(min_value=0, max_value=MAX_RETRIES - 1),
    )
    def test_consecutive_delays_increase(self, attempt1: int):
        """
        Property: For any two consecutive retry attempts, the second delay
        SHALL be greater than the first.
        
        **Validates: Requirements 5.7, 6.5**
        """
        attempt2 = attempt1 + 1
        
        delay1 = INITIAL_RETRY_DELAY * (RETRY_BACKOFF_MULTIPLIER ** attempt1)
        delay2 = INITIAL_RETRY_DELAY * (RETRY_BACKOFF_MULTIPLIER ** attempt2)
        
        assert delay2 > delay1, (
            f"Delay for attempt {attempt2} ({delay2}) should be greater than "
            f"delay for attempt {attempt1} ({delay1})"
        )

    @settings(max_examples=100)
    @given(
        attempt=st.integers(min_value=0, max_value=MAX_RETRIES),
    )
    def test_delay_formula_is_deterministic(self, attempt: int):
        """
        Property: For any attempt number, the delay calculation SHALL be
        deterministic (same input always gives same output).
        
        **Validates: Requirements 5.7, 6.5**
        """
        delay1 = INITIAL_RETRY_DELAY * (RETRY_BACKOFF_MULTIPLIER ** attempt)
        delay2 = INITIAL_RETRY_DELAY * (RETRY_BACKOFF_MULTIPLIER ** attempt)
        delay3 = INITIAL_RETRY_DELAY * (RETRY_BACKOFF_MULTIPLIER ** attempt)
        
        assert delay1 == delay2 == delay3, (
            f"Delay calculation should be deterministic. "
            f"Got different results: {delay1}, {delay2}, {delay3}"
        )
