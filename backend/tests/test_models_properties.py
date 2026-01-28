"""
Property-based tests for data models, specifically ID generation.

**Property 2: ID Uniqueness**
*For any* set of files uploaded in a batch, all generated Data_IDs SHALL be unique.
*For any* set of tasks created, all Task_IDs SHALL be unique.

**Validates: Requirements 1.6, 8.1**

Uses Hypothesis for property-based testing with at least 100 iterations per test.
"""

import re

import pytest
from hypothesis import given, settings, strategies as st

from backend.models import generate_data_id, generate_task_id


# Strategy for generating batch sizes from 1 to 1000
batch_size_strategy = st.integers(min_value=1, max_value=1000)


class TestIDUniquenessProperty:
    """
    Property-based tests for Property 2: ID Uniqueness.
    
    **Validates: Requirements 1.6, 8.1**
    
    These tests verify that:
    1. Generating N Data_IDs always produces N unique IDs
    2. Generating N Task_IDs always produces N unique IDs
    3. Data_IDs have the correct format (data-{uuid})
    4. Task_IDs have the correct format (task-{uuid})
    """

    @settings(max_examples=100)
    @given(n=batch_size_strategy)
    def test_data_ids_are_unique_for_any_batch_size(self, n: int):
        """
        Property: For any N from 1 to 1000, generating N Data_IDs SHALL
        produce N unique IDs.
        
        **Validates: Requirements 1.6**
        
        This ensures that when files are uploaded successfully, each file
        receives a unique Data_ID as required by Requirement 1.6.
        """
        # Generate N Data_IDs
        data_ids = [generate_data_id() for _ in range(n)]
        
        # All IDs SHALL be unique
        unique_ids = set(data_ids)
        assert len(unique_ids) == n, (
            f"Generated {n} Data_IDs but only {len(unique_ids)} are unique. "
            f"Expected all {n} to be unique."
        )

    @settings(max_examples=100)
    @given(n=batch_size_strategy)
    def test_task_ids_are_unique_for_any_batch_size(self, n: int):
        """
        Property: For any N from 1 to 1000, generating N Task_IDs SHALL
        produce N unique IDs.
        
        **Validates: Requirements 8.1**
        
        This ensures that when tasks are created, each task receives a
        unique Task_ID as required by Requirement 8.1.
        """
        # Generate N Task_IDs
        task_ids = [generate_task_id() for _ in range(n)]
        
        # All IDs SHALL be unique
        unique_ids = set(task_ids)
        assert len(unique_ids) == n, (
            f"Generated {n} Task_IDs but only {len(unique_ids)} are unique. "
            f"Expected all {n} to be unique."
        )

    @settings(max_examples=100)
    @given(n=batch_size_strategy)
    def test_data_ids_have_correct_format(self, n: int):
        """
        Property: For any generated Data_ID, the format SHALL be "data-{uuid4.hex}"
        where uuid4.hex is a 32-character hexadecimal string.
        
        **Validates: Requirements 1.6**
        """
        # Pattern for data-{uuid4.hex}: "data-" followed by exactly 32 hex characters
        data_id_pattern = re.compile(r"^data-[0-9a-f]{32}$")
        
        # Generate N Data_IDs and verify format
        for _ in range(n):
            data_id = generate_data_id()
            assert data_id_pattern.match(data_id), (
                f"Data_ID '{data_id}' does not match expected format 'data-{{uuid4.hex}}'. "
                f"Expected pattern: data-[0-9a-f]{{32}}"
            )

    @settings(max_examples=100)
    @given(n=batch_size_strategy)
    def test_task_ids_have_correct_format(self, n: int):
        """
        Property: For any generated Task_ID, the format SHALL be "task-{uuid4.hex}"
        where uuid4.hex is a 32-character hexadecimal string.
        
        **Validates: Requirements 8.1**
        """
        # Pattern for task-{uuid4.hex}: "task-" followed by exactly 32 hex characters
        task_id_pattern = re.compile(r"^task-[0-9a-f]{32}$")
        
        # Generate N Task_IDs and verify format
        for _ in range(n):
            task_id = generate_task_id()
            assert task_id_pattern.match(task_id), (
                f"Task_ID '{task_id}' does not match expected format 'task-{{uuid4.hex}}'. "
                f"Expected pattern: task-[0-9a-f]{{32}}"
            )

    @settings(max_examples=100)
    @given(n=st.integers(min_value=1, max_value=100))
    def test_data_ids_and_task_ids_are_distinct(self, n: int):
        """
        Property: Data_IDs and Task_IDs SHALL have distinct prefixes,
        ensuring they can never collide even when generated in the same batch.
        
        **Validates: Requirements 1.6, 8.1**
        """
        # Generate N of each type
        data_ids = [generate_data_id() for _ in range(n)]
        task_ids = [generate_task_id() for _ in range(n)]
        
        # All Data_IDs should start with "data-"
        for data_id in data_ids:
            assert data_id.startswith("data-"), (
                f"Data_ID '{data_id}' should start with 'data-'"
            )
        
        # All Task_IDs should start with "task-"
        for task_id in task_ids:
            assert task_id.startswith("task-"), (
                f"Task_ID '{task_id}' should start with 'task-'"
            )
        
        # No overlap between the two sets
        data_id_set = set(data_ids)
        task_id_set = set(task_ids)
        overlap = data_id_set & task_id_set
        assert len(overlap) == 0, (
            f"Data_IDs and Task_IDs should never overlap. Found overlap: {overlap}"
        )


class TestIDUniquenessEdgeCases:
    """
    Edge case tests for ID uniqueness to complement property-based tests.
    
    **Validates: Requirements 1.6, 8.1**
    """

    def test_single_data_id_is_valid(self):
        """
        Test that a single Data_ID is valid and correctly formatted.
        
        **Validates: Requirements 1.6**
        """
        data_id = generate_data_id()
        assert data_id.startswith("data-")
        assert len(data_id) == 37  # "data-" (5) + uuid hex (32)

    def test_single_task_id_is_valid(self):
        """
        Test that a single Task_ID is valid and correctly formatted.
        
        **Validates: Requirements 8.1**
        """
        task_id = generate_task_id()
        assert task_id.startswith("task-")
        assert len(task_id) == 37  # "task-" (5) + uuid hex (32)

    def test_maximum_batch_size_data_ids(self):
        """
        Test uniqueness at maximum batch size (200 files per Requirement 1.4).
        
        **Validates: Requirements 1.6**
        """
        max_batch_size = 200
        data_ids = [generate_data_id() for _ in range(max_batch_size)]
        unique_ids = set(data_ids)
        assert len(unique_ids) == max_batch_size

    def test_large_batch_data_ids(self):
        """
        Test uniqueness for a large batch of 1000 Data_IDs.
        
        **Validates: Requirements 1.6**
        """
        large_batch_size = 1000
        data_ids = [generate_data_id() for _ in range(large_batch_size)]
        unique_ids = set(data_ids)
        assert len(unique_ids) == large_batch_size

    def test_large_batch_task_ids(self):
        """
        Test uniqueness for a large batch of 1000 Task_IDs.
        
        **Validates: Requirements 8.1**
        """
        large_batch_size = 1000
        task_ids = [generate_task_id() for _ in range(large_batch_size)]
        unique_ids = set(task_ids)
        assert len(unique_ids) == large_batch_size

    def test_consecutive_data_ids_are_different(self):
        """
        Test that consecutively generated Data_IDs are different.
        
        **Validates: Requirements 1.6**
        """
        id1 = generate_data_id()
        id2 = generate_data_id()
        assert id1 != id2, "Consecutive Data_IDs should be different"

    def test_consecutive_task_ids_are_different(self):
        """
        Test that consecutively generated Task_IDs are different.
        
        **Validates: Requirements 8.1**
        """
        id1 = generate_task_id()
        id2 = generate_task_id()
        assert id1 != id2, "Consecutive Task_IDs should be different"
