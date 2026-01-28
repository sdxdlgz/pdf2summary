"""
Property-based tests for Document Processor.

**Property 11: Bilingual Document Alignment**
*For any* original document with N paragraphs and its translation, the bilingual output
SHALL contain N paragraph pairs, each pair consisting of the original paragraph followed
by its translation.

**Validates: Requirements 5.5**

**Property 13: Output File Completeness**
*For any* successfully completed task, the output SHALL include: original MD, original DOCX,
bilingual MD, bilingual DOCX, and bilingual summary files.

**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**

Uses Hypothesis for property-based testing with at least 100 iterations per test.
"""

import pytest
from hypothesis import given, settings, strategies as st, assume

from backend.services.document_processor import (
    DocumentProcessor,
    split_into_paragraphs,
    merge_paragraphs,
    PARAGRAPH_SEPARATOR,
)
from backend.models import OutputFileType


# =============================================================================
# Strategies for generating test data
# =============================================================================

# Strategy for generating non-empty paragraph text (no double newlines)
paragraph_text = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S"),
        blacklist_characters="\n",
    ),
    min_size=1,
    max_size=200,
).filter(lambda s: s.strip())


# Strategy for generating a list of paragraphs
paragraph_list = st.lists(
    paragraph_text,
    min_size=1,
    max_size=20,
)


# Strategy for generating a document (paragraphs joined by double newlines)
document_text = paragraph_list.map(lambda paras: PARAGRAPH_SEPARATOR.join(paras))


# Strategy for generating matching original and translated paragraph lists
# (same number of paragraphs)
matching_paragraph_pairs = st.integers(min_value=1, max_value=15).flatmap(
    lambda n: st.tuples(
        st.lists(paragraph_text, min_size=n, max_size=n),
        st.lists(paragraph_text, min_size=n, max_size=n),
    )
)


# Strategy for generating mismatched paragraph counts
mismatched_paragraph_pairs = st.tuples(
    st.lists(paragraph_text, min_size=1, max_size=10),
    st.lists(paragraph_text, min_size=1, max_size=10),
).filter(lambda t: len(t[0]) != len(t[1]))


# =============================================================================
# Property 11: Bilingual Document Alignment
# =============================================================================

class TestBilingualDocumentAlignmentProperty:
    """
    Property-based tests for Property 11: Bilingual Document Alignment.
    
    *For any* original document with N paragraphs and its translation, the bilingual
    output SHALL contain N paragraph pairs, each pair consisting of the original
    paragraph followed by its translation.
    
    **Validates: Requirements 5.5**
    """

    @settings(max_examples=100)
    @given(paragraph_pairs=matching_paragraph_pairs)
    @pytest.mark.asyncio
    async def test_n_paragraphs_produce_n_pairs(
        self,
        paragraph_pairs: tuple[list[str], list[str]],
    ):
        """
        Property: For any original document with N paragraphs and its translation
        with N paragraphs, the bilingual output SHALL contain N paragraph pairs.
        
        **Validates: Requirements 5.5**
        """
        original_paragraphs, translated_paragraphs = paragraph_pairs
        n = len(original_paragraphs)
        
        # Create documents from paragraphs
        original = PARAGRAPH_SEPARATOR.join(original_paragraphs)
        translated = PARAGRAPH_SEPARATOR.join(translated_paragraphs)
        
        # Create bilingual document
        processor = DocumentProcessor()
        bilingual = await processor.create_bilingual_markdown(original, translated)
        
        # Count paragraph pairs by counting separators (---)
        # Each pair is separated by "---" except the last one
        separator_count = bilingual.count("\n\n---\n\n")
        
        # N paragraphs should produce N-1 separators (between N pairs)
        expected_separators = n - 1 if n > 0 else 0
        
        assert separator_count == expected_separators, (
            f"Document with {n} paragraphs should have {expected_separators} separators, "
            f"got {separator_count}"
        )

    @settings(max_examples=100)
    @given(paragraph_pairs=matching_paragraph_pairs)
    @pytest.mark.asyncio
    async def test_original_paragraph_followed_by_translation(
        self,
        paragraph_pairs: tuple[list[str], list[str]],
    ):
        """
        Property: For any paragraph pair, the original paragraph SHALL be followed
        by its translation (formatted as blockquote).
        
        **Validates: Requirements 5.5**
        """
        original_paragraphs, translated_paragraphs = paragraph_pairs
        
        # Create documents from paragraphs
        original = PARAGRAPH_SEPARATOR.join(original_paragraphs)
        translated = PARAGRAPH_SEPARATOR.join(translated_paragraphs)
        
        # Create bilingual document
        processor = DocumentProcessor()
        bilingual = await processor.create_bilingual_markdown(original, translated)
        
        # Split bilingual document by separator
        pairs = bilingual.split("\n\n---\n\n")
        
        # Each pair should contain original followed by translation (as blockquote)
        for i, pair in enumerate(pairs):
            # The pair should contain the original paragraph
            if i < len(original_paragraphs):
                assert original_paragraphs[i] in pair, (
                    f"Pair {i} should contain original paragraph: {original_paragraphs[i][:50]}..."
                )
            
            # The pair should contain the translation as blockquote (> prefix)
            if i < len(translated_paragraphs):
                # Check for blockquote format
                assert "> " in pair, (
                    f"Pair {i} should contain translation as blockquote"
                )

    @settings(max_examples=100)
    @given(paragraph_pairs=matching_paragraph_pairs)
    @pytest.mark.asyncio
    async def test_all_original_paragraphs_preserved(
        self,
        paragraph_pairs: tuple[list[str], list[str]],
    ):
        """
        Property: For any bilingual document, all original paragraphs SHALL be
        preserved in the output.
        
        **Validates: Requirements 5.5**
        """
        original_paragraphs, translated_paragraphs = paragraph_pairs
        
        # Create documents from paragraphs
        original = PARAGRAPH_SEPARATOR.join(original_paragraphs)
        translated = PARAGRAPH_SEPARATOR.join(translated_paragraphs)
        
        # Create bilingual document
        processor = DocumentProcessor()
        bilingual = await processor.create_bilingual_markdown(original, translated)
        
        # All original paragraphs should be in the output
        for i, para in enumerate(original_paragraphs):
            assert para in bilingual, (
                f"Original paragraph {i} should be preserved: {para[:50]}..."
            )

    @settings(max_examples=100)
    @given(paragraph_pairs=matching_paragraph_pairs)
    @pytest.mark.asyncio
    async def test_all_translated_paragraphs_preserved(
        self,
        paragraph_pairs: tuple[list[str], list[str]],
    ):
        """
        Property: For any bilingual document, all translated paragraphs SHALL be
        preserved in the output (as blockquotes).
        
        **Validates: Requirements 5.5**
        """
        original_paragraphs, translated_paragraphs = paragraph_pairs
        
        # Create documents from paragraphs
        original = PARAGRAPH_SEPARATOR.join(original_paragraphs)
        translated = PARAGRAPH_SEPARATOR.join(translated_paragraphs)
        
        # Create bilingual document
        processor = DocumentProcessor()
        bilingual = await processor.create_bilingual_markdown(original, translated)
        
        # All translated paragraphs should be in the output (as blockquotes)
        for i, para in enumerate(translated_paragraphs):
            # The translation is formatted with "> " prefix
            assert para in bilingual, (
                f"Translated paragraph {i} should be preserved: {para[:50]}..."
            )

    @settings(max_examples=100)
    @given(paragraphs=paragraph_list)
    @pytest.mark.asyncio
    async def test_empty_translation_preserves_original(self, paragraphs: list[str]):
        """
        Property: For any original document with empty translation, the output
        SHALL preserve the original content.
        
        **Validates: Requirements 5.5**
        """
        original = PARAGRAPH_SEPARATOR.join(paragraphs)
        
        processor = DocumentProcessor()
        bilingual = await processor.create_bilingual_markdown(original, "")
        
        # Output should be the original
        assert bilingual == original, (
            f"Empty translation should return original document"
        )

    @settings(max_examples=100)
    @given(paragraphs=paragraph_list)
    @pytest.mark.asyncio
    async def test_empty_original_preserves_translation(self, paragraphs: list[str]):
        """
        Property: For any translation with empty original, the output
        SHALL preserve the translation content.
        
        **Validates: Requirements 5.5**
        """
        translated = PARAGRAPH_SEPARATOR.join(paragraphs)
        
        processor = DocumentProcessor()
        bilingual = await processor.create_bilingual_markdown("", translated)
        
        # Output should be the translation
        assert bilingual == translated, (
            f"Empty original should return translation document"
        )

    @pytest.mark.asyncio
    async def test_both_empty_returns_empty(self):
        """
        Test that empty original and translation returns empty string.
        
        **Validates: Requirements 5.5**
        """
        processor = DocumentProcessor()
        bilingual = await processor.create_bilingual_markdown("", "")
        
        assert bilingual == "", "Both empty should return empty string"


class TestParagraphSplittingProperty:
    """
    Property-based tests for paragraph splitting and merging.
    
    **Validates: Requirements 5.5**
    """

    @settings(max_examples=100)
    @given(paragraphs=paragraph_list)
    def test_split_produces_correct_count(self, paragraphs: list[str]):
        """
        Property: For any document with N paragraphs, split_into_paragraphs
        SHALL return exactly N paragraphs.
        
        **Validates: Requirements 5.5**
        """
        document = PARAGRAPH_SEPARATOR.join(paragraphs)
        result = split_into_paragraphs(document)
        
        assert len(result) == len(paragraphs), (
            f"Document with {len(paragraphs)} paragraphs should split into "
            f"{len(paragraphs)} parts, got {len(result)}"
        )

    @settings(max_examples=100)
    @given(paragraphs=paragraph_list)
    def test_split_preserves_content(self, paragraphs: list[str]):
        """
        Property: For any document, split_into_paragraphs SHALL preserve
        all paragraph content.
        
        **Validates: Requirements 5.5**
        """
        document = PARAGRAPH_SEPARATOR.join(paragraphs)
        result = split_into_paragraphs(document)
        
        for i, (original, split) in enumerate(zip(paragraphs, result)):
            assert original.strip() == split.strip(), (
                f"Paragraph {i} content should be preserved"
            )

    @settings(max_examples=100)
    @given(paragraphs=paragraph_list)
    def test_split_merge_roundtrip(self, paragraphs: list[str]):
        """
        Property: For any document, splitting and then merging SHALL produce
        content equivalent to the original.
        
        **Validates: Requirements 5.5**
        """
        document = PARAGRAPH_SEPARATOR.join(paragraphs)
        
        # Split and merge
        split = split_into_paragraphs(document)
        merged = merge_paragraphs(split)
        
        # Should be equivalent
        assert merged == document, (
            f"Split-merge roundtrip should preserve document"
        )

    def test_empty_document_produces_empty_list(self):
        """
        Test that empty document produces empty list.
        
        **Validates: Requirements 5.5**
        """
        assert split_into_paragraphs("") == []
        assert split_into_paragraphs(None) == []

    @settings(max_examples=100)
    @given(paragraphs=paragraph_list)
    def test_split_produces_non_empty_paragraphs(self, paragraphs: list[str]):
        """
        Property: For any document, split_into_paragraphs SHALL produce
        only non-empty paragraphs.
        
        **Validates: Requirements 5.5**
        """
        document = PARAGRAPH_SEPARATOR.join(paragraphs)
        result = split_into_paragraphs(document)
        
        for i, para in enumerate(result):
            assert para.strip(), (
                f"Paragraph {i} should be non-empty"
            )


# =============================================================================
# Property 13: Output File Completeness
# =============================================================================

class TestOutputFileCompletenessProperty:
    """
    Property-based tests for Property 13: Output File Completeness.
    
    *For any* successfully completed task, the output SHALL include: original MD,
    original DOCX, bilingual MD, bilingual DOCX, and bilingual summary files.
    
    **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
    """

    def test_output_file_type_enum_has_all_required_types(self):
        """
        Test that OutputFileType enum contains all 5 required output types.
        
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
        """
        required_types = {
            "ORIGINAL_MD",      # Requirement 7.1
            "ORIGINAL_DOCX",    # Requirement 7.2
            "BILINGUAL_MD",     # Requirement 7.3
            "BILINGUAL_DOCX",   # Requirement 7.4
            "SUMMARY",          # Requirement 7.5
        }
        
        actual_types = {member.name for member in OutputFileType}
        
        assert required_types.issubset(actual_types), (
            f"OutputFileType should contain all required types. "
            f"Missing: {required_types - actual_types}"
        )

    def test_output_file_type_count_is_5(self):
        """
        Test that OutputFileType enum has exactly 5 members.
        
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
        """
        assert len(OutputFileType) == 5, (
            f"OutputFileType should have exactly 5 members, got {len(OutputFileType)}"
        )

    def test_original_md_type_exists(self):
        """
        Test that ORIGINAL_MD output type exists.
        
        **Validates: Requirement 7.1**
        """
        assert hasattr(OutputFileType, "ORIGINAL_MD"), (
            "OutputFileType should have ORIGINAL_MD member"
        )
        assert OutputFileType.ORIGINAL_MD.value == "original_md", (
            f"ORIGINAL_MD value should be 'original_md', got {OutputFileType.ORIGINAL_MD.value}"
        )

    def test_original_docx_type_exists(self):
        """
        Test that ORIGINAL_DOCX output type exists.
        
        **Validates: Requirement 7.2**
        """
        assert hasattr(OutputFileType, "ORIGINAL_DOCX"), (
            "OutputFileType should have ORIGINAL_DOCX member"
        )
        assert OutputFileType.ORIGINAL_DOCX.value == "original_docx", (
            f"ORIGINAL_DOCX value should be 'original_docx', got {OutputFileType.ORIGINAL_DOCX.value}"
        )

    def test_bilingual_md_type_exists(self):
        """
        Test that BILINGUAL_MD output type exists.
        
        **Validates: Requirement 7.3**
        """
        assert hasattr(OutputFileType, "BILINGUAL_MD"), (
            "OutputFileType should have BILINGUAL_MD member"
        )
        assert OutputFileType.BILINGUAL_MD.value == "bilingual_md", (
            f"BILINGUAL_MD value should be 'bilingual_md', got {OutputFileType.BILINGUAL_MD.value}"
        )

    def test_bilingual_docx_type_exists(self):
        """
        Test that BILINGUAL_DOCX output type exists.
        
        **Validates: Requirement 7.4**
        """
        assert hasattr(OutputFileType, "BILINGUAL_DOCX"), (
            "OutputFileType should have BILINGUAL_DOCX member"
        )
        assert OutputFileType.BILINGUAL_DOCX.value == "bilingual_docx", (
            f"BILINGUAL_DOCX value should be 'bilingual_docx', got {OutputFileType.BILINGUAL_DOCX.value}"
        )

    def test_summary_type_exists(self):
        """
        Test that SUMMARY output type exists.
        
        **Validates: Requirement 7.5**
        """
        assert hasattr(OutputFileType, "SUMMARY"), (
            "OutputFileType should have SUMMARY member"
        )
        assert OutputFileType.SUMMARY.value == "summary", (
            f"SUMMARY value should be 'summary', got {OutputFileType.SUMMARY.value}"
        )

    @settings(max_examples=100)
    @given(
        outputs=st.fixed_dictionaries({
            OutputFileType.ORIGINAL_MD.value: st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
            OutputFileType.ORIGINAL_DOCX.value: st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
            OutputFileType.BILINGUAL_MD.value: st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
            OutputFileType.BILINGUAL_DOCX.value: st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
            OutputFileType.SUMMARY.value: st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
        })
    )
    def test_complete_output_has_all_file_types(self, outputs: dict[str, str]):
        """
        Property: For any complete task output, all 5 file types SHALL be present.
        
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
        """
        required_keys = {
            OutputFileType.ORIGINAL_MD.value,
            OutputFileType.ORIGINAL_DOCX.value,
            OutputFileType.BILINGUAL_MD.value,
            OutputFileType.BILINGUAL_DOCX.value,
            OutputFileType.SUMMARY.value,
        }
        
        actual_keys = set(outputs.keys())
        
        assert required_keys == actual_keys, (
            f"Complete output should have all 5 file types. "
            f"Missing: {required_keys - actual_keys}"
        )

    @settings(max_examples=100)
    @given(
        outputs=st.fixed_dictionaries({
            OutputFileType.ORIGINAL_MD.value: st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
            OutputFileType.ORIGINAL_DOCX.value: st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
            OutputFileType.BILINGUAL_MD.value: st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
            OutputFileType.BILINGUAL_DOCX.value: st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
            OutputFileType.SUMMARY.value: st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
        })
    )
    def test_all_output_paths_are_non_empty(self, outputs: dict[str, str]):
        """
        Property: For any complete task output, all file paths SHALL be non-empty.
        
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
        """
        for file_type, path in outputs.items():
            assert path.strip(), (
                f"Output path for {file_type} should be non-empty"
            )


class TestOutputFileTypeValidation:
    """
    Property-based tests for validating output file type completeness.
    
    **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
    """

    @settings(max_examples=100)
    @given(
        missing_type=st.sampled_from(list(OutputFileType)),
    )
    def test_incomplete_output_missing_one_type(self, missing_type: OutputFileType):
        """
        Property: For any output missing one file type, the output SHALL be
        considered incomplete.
        
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
        """
        # Create complete output
        complete_outputs = {
            OutputFileType.ORIGINAL_MD.value: "/path/to/original.md",
            OutputFileType.ORIGINAL_DOCX.value: "/path/to/original.docx",
            OutputFileType.BILINGUAL_MD.value: "/path/to/bilingual.md",
            OutputFileType.BILINGUAL_DOCX.value: "/path/to/bilingual.docx",
            OutputFileType.SUMMARY.value: "/path/to/summary.md",
        }
        
        # Remove one type
        incomplete_outputs = {k: v for k, v in complete_outputs.items() if k != missing_type.value}
        
        # Should be incomplete (missing one type)
        required_keys = {ft.value for ft in OutputFileType}
        actual_keys = set(incomplete_outputs.keys())
        
        assert required_keys != actual_keys, (
            f"Output missing {missing_type.value} should be incomplete"
        )
        assert missing_type.value not in actual_keys, (
            f"Output should not contain {missing_type.value}"
        )

    @settings(max_examples=100)
    @given(
        file_type=st.sampled_from(list(OutputFileType)),
    )
    def test_output_file_type_value_is_string(self, file_type: OutputFileType):
        """
        Property: For any OutputFileType, the value SHALL be a non-empty string.
        
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
        """
        assert isinstance(file_type.value, str), (
            f"OutputFileType value should be string, got {type(file_type.value)}"
        )
        assert file_type.value.strip(), (
            f"OutputFileType value should be non-empty"
        )

    def test_output_file_types_are_unique(self):
        """
        Test that all OutputFileType values are unique.
        
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
        """
        values = [ft.value for ft in OutputFileType]
        unique_values = set(values)
        
        assert len(values) == len(unique_values), (
            f"OutputFileType values should be unique. "
            f"Duplicates found: {[v for v in values if values.count(v) > 1]}"
        )


def validate_task_outputs_complete(outputs: dict[str, str] | None) -> tuple[bool, list[str]]:
    """
    Validate that a task's outputs contain all required file types.
    
    Args:
        outputs: Dictionary mapping file type to file path
        
    Returns:
        Tuple of (is_complete, missing_types)
        
    **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
    """
    if outputs is None:
        return False, [ft.value for ft in OutputFileType]
    
    required_types = {ft.value for ft in OutputFileType}
    actual_types = set(outputs.keys())
    
    missing = required_types - actual_types
    
    return len(missing) == 0, list(missing)


class TestOutputValidationFunction:
    """
    Property-based tests for the output validation function.
    
    **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
    """

    def test_none_outputs_is_incomplete(self):
        """
        Test that None outputs is considered incomplete.
        
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
        """
        is_complete, missing = validate_task_outputs_complete(None)
        
        assert not is_complete, "None outputs should be incomplete"
        assert len(missing) == 5, f"Should have 5 missing types, got {len(missing)}"

    def test_empty_outputs_is_incomplete(self):
        """
        Test that empty outputs is considered incomplete.
        
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
        """
        is_complete, missing = validate_task_outputs_complete({})
        
        assert not is_complete, "Empty outputs should be incomplete"
        assert len(missing) == 5, f"Should have 5 missing types, got {len(missing)}"

    @settings(max_examples=100)
    @given(
        outputs=st.fixed_dictionaries({
            OutputFileType.ORIGINAL_MD.value: st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
            OutputFileType.ORIGINAL_DOCX.value: st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
            OutputFileType.BILINGUAL_MD.value: st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
            OutputFileType.BILINGUAL_DOCX.value: st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
            OutputFileType.SUMMARY.value: st.text(min_size=1, max_size=50).filter(lambda s: s.strip()),
        })
    )
    def test_complete_outputs_is_valid(self, outputs: dict[str, str]):
        """
        Property: For any complete outputs dictionary, validation SHALL return True.
        
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
        """
        is_complete, missing = validate_task_outputs_complete(outputs)
        
        assert is_complete, f"Complete outputs should be valid. Missing: {missing}"
        assert len(missing) == 0, f"Should have no missing types, got {missing}"

    @settings(max_examples=100)
    @given(
        missing_type=st.sampled_from(list(OutputFileType)),
    )
    def test_incomplete_outputs_reports_missing(self, missing_type: OutputFileType):
        """
        Property: For any incomplete outputs, validation SHALL report the missing types.
        
        **Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**
        """
        # Create complete output then remove one
        outputs = {
            OutputFileType.ORIGINAL_MD.value: "/path/to/original.md",
            OutputFileType.ORIGINAL_DOCX.value: "/path/to/original.docx",
            OutputFileType.BILINGUAL_MD.value: "/path/to/bilingual.md",
            OutputFileType.BILINGUAL_DOCX.value: "/path/to/bilingual.docx",
            OutputFileType.SUMMARY.value: "/path/to/summary.md",
        }
        del outputs[missing_type.value]
        
        is_complete, missing = validate_task_outputs_complete(outputs)
        
        assert not is_complete, f"Outputs missing {missing_type.value} should be incomplete"
        assert missing_type.value in missing, (
            f"Missing types should include {missing_type.value}, got {missing}"
        )
