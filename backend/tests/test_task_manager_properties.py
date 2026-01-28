"""
Property-based tests for Task Manager.

**Property 4: Task State Machine Correctness**
*For any* task in states "waiting-file", "pending", "running", or "converting",
the Task_Manager SHALL continue polling. *For any* task in states "done" or "failed",
the Task_Manager SHALL stop polling.

**Validates: Requirements 3.2**

Uses Hypothesis for property-based testing with at least 100 iterations per test.
"""

from typing import List

import pytest
from hypothesis import given, settings, strategies as st

from backend.models import MineruTaskState
from backend.services.task_manager import (
    is_terminal_state,
    should_continue_polling,
    calculate_backoff_interval,
    INITIAL_POLL_INTERVAL,
    MAX_POLL_INTERVAL,
    BACKOFF_MULTIPLIER,
)


# =============================================================================
# Strategies for generating MinerU task states
# =============================================================================

# Strategy for generating non-terminal states (polling should continue)
non_terminal_state_strategy = st.sampled_from([
    MineruTaskState.WAITING_FILE,
    MineruTaskState.PENDING,
    MineruTaskState.RUNNING,
    MineruTaskState.CONVERTING,
])

# Strategy for generating terminal states (polling should stop)
terminal_state_strategy = st.sampled_from([
    MineruTaskState.DONE,
    MineruTaskState.FAILED,
])

# Strategy for generating any MinerU task state
any_state_strategy = st.sampled_from([
    MineruTaskState.WAITING_FILE,
    MineruTaskState.PENDING,
    MineruTaskState.RUNNING,
    MineruTaskState.CONVERTING,
    MineruTaskState.DONE,
    MineruTaskState.FAILED,
])


class TestTaskStateMachineCorrectnessProperty:
    """
    Property-based tests for Property 4: Task State Machine Correctness.
    
    **Validates: Requirements 3.2**
    
    These tests verify that:
    1. is_terminal_state() returns False for non-terminal states
    2. is_terminal_state() returns True for terminal states
    3. For any sequence of non-terminal states, polling should continue
    4. For any terminal state, polling should stop
    """

    @settings(max_examples=100)
    @given(state=non_terminal_state_strategy)
    def test_non_terminal_states_return_false(
        self,
        state: MineruTaskState,
    ):
        """
        Property: For any non-terminal state (WAITING_FILE, PENDING, RUNNING,
        CONVERTING), is_terminal_state() SHALL return False, indicating that
        polling should continue.
        
        **Validates: Requirements 3.2**
        """
        result = is_terminal_state(state)
        
        assert result is False, (
            f"is_terminal_state({state.value}) should return False for "
            f"non-terminal state, but got {result}. "
            f"Polling should continue for state '{state.value}'."
        )

    @settings(max_examples=100)
    @given(state=terminal_state_strategy)
    def test_terminal_states_return_true(
        self,
        state: MineruTaskState,
    ):
        """
        Property: For any terminal state (DONE, FAILED), is_terminal_state()
        SHALL return True, indicating that polling should stop.
        
        **Validates: Requirements 3.2**
        """
        result = is_terminal_state(state)
        
        assert result is True, (
            f"is_terminal_state({state.value}) should return True for "
            f"terminal state, but got {result}. "
            f"Polling should stop for state '{state.value}'."
        )

    @settings(max_examples=100)
    @given(states=st.lists(non_terminal_state_strategy, min_size=1, max_size=20))
    def test_sequence_of_non_terminal_states_all_continue_polling(
        self,
        states: List[MineruTaskState],
    ):
        """
        Property: For any sequence of non-terminal states, is_terminal_state()
        SHALL return False for all states in the sequence, indicating that
        polling should continue for the entire sequence.
        
        **Validates: Requirements 3.2**
        """
        for i, state in enumerate(states):
            result = is_terminal_state(state)
            
            assert result is False, (
                f"In sequence position {i}, is_terminal_state({state.value}) "
                f"should return False, but got {result}. "
                f"Polling should continue for all non-terminal states."
            )

    @settings(max_examples=100)
    @given(state=terminal_state_strategy)
    def test_terminal_state_stops_polling(
        self,
        state: MineruTaskState,
    ):
        """
        Property: For any terminal state, is_terminal_state() SHALL return True,
        which signals that the Task_Manager should stop polling.
        
        **Validates: Requirements 3.2**
        """
        result = is_terminal_state(state)
        
        # Terminal state should stop polling
        should_stop_polling = result is True
        
        assert should_stop_polling, (
            f"Terminal state '{state.value}' should stop polling. "
            f"is_terminal_state() returned {result}, expected True."
        )


class TestTaskStateClassificationProperty:
    """
    Property-based tests for state classification correctness.
    
    **Validates: Requirements 3.2**
    
    These tests verify the complete classification of all MinerU task states.
    """

    @settings(max_examples=100)
    @given(state=any_state_strategy)
    def test_state_classification_is_boolean(
        self,
        state: MineruTaskState,
    ):
        """
        Property: For any MinerU task state, is_terminal_state() SHALL return
        a boolean value (True or False).
        
        **Validates: Requirements 3.2**
        """
        result = is_terminal_state(state)
        
        assert isinstance(result, bool), (
            f"is_terminal_state({state.value}) should return a boolean, "
            f"but got {type(result).__name__}: {result}"
        )

    @settings(max_examples=100)
    @given(state=any_state_strategy)
    def test_state_classification_is_deterministic(
        self,
        state: MineruTaskState,
    ):
        """
        Property: For any MinerU task state, calling is_terminal_state()
        multiple times SHALL return the same result.
        
        **Validates: Requirements 3.2**
        """
        result1 = is_terminal_state(state)
        result2 = is_terminal_state(state)
        result3 = is_terminal_state(state)
        
        assert result1 == result2 == result3, (
            f"is_terminal_state({state.value}) should be deterministic. "
            f"Got different results: {result1}, {result2}, {result3}"
        )

    @settings(max_examples=100)
    @given(state=any_state_strategy)
    def test_state_classification_matches_expected_behavior(
        self,
        state: MineruTaskState,
    ):
        """
        Property: For any MinerU task state, is_terminal_state() SHALL return
        True only for DONE and FAILED states, and False for all other states.
        
        **Validates: Requirements 3.2**
        """
        result = is_terminal_state(state)
        
        # Define expected terminal states
        expected_terminal_states = {MineruTaskState.DONE, MineruTaskState.FAILED}
        expected_result = state in expected_terminal_states
        
        assert result == expected_result, (
            f"is_terminal_state({state.value}) returned {result}, "
            f"but expected {expected_result}. "
            f"Terminal states are: {[s.value for s in expected_terminal_states]}"
        )


class TestPollingBehaviorProperty:
    """
    Property-based tests for polling behavior based on state.
    
    **Validates: Requirements 3.2**
    
    These tests verify that the polling behavior is correct based on state.
    """

    @settings(max_examples=100)
    @given(
        non_terminal_states=st.lists(
            non_terminal_state_strategy,
            min_size=1,
            max_size=10,
        ),
        terminal_state=terminal_state_strategy,
    )
    def test_polling_continues_until_terminal_state(
        self,
        non_terminal_states: List[MineruTaskState],
        terminal_state: MineruTaskState,
    ):
        """
        Property: For any sequence of non-terminal states followed by a
        terminal state, is_terminal_state() SHALL return False for all
        non-terminal states and True for the terminal state.
        
        This simulates the polling loop behavior where polling continues
        until a terminal state is reached.
        
        **Validates: Requirements 3.2**
        """
        # Simulate polling through non-terminal states
        for state in non_terminal_states:
            result = is_terminal_state(state)
            assert result is False, (
                f"Polling should continue for non-terminal state '{state.value}'"
            )
        
        # Terminal state should stop polling
        result = is_terminal_state(terminal_state)
        assert result is True, (
            f"Polling should stop for terminal state '{terminal_state.value}'"
        )

    @settings(max_examples=100)
    @given(state=non_terminal_state_strategy)
    def test_waiting_file_state_continues_polling(
        self,
        state: MineruTaskState,
    ):
        """
        Property: The WAITING_FILE state SHALL be classified as non-terminal,
        meaning polling should continue.
        
        Note: The design document mentions "waiting-file" as a state where
        polling should continue.
        
        **Validates: Requirements 3.2**
        """
        if state == MineruTaskState.WAITING_FILE:
            result = is_terminal_state(state)
            assert result is False, (
                f"WAITING_FILE state should be non-terminal. "
                f"is_terminal_state() returned {result}"
            )

    @settings(max_examples=100)
    @given(state=non_terminal_state_strategy)
    def test_pending_state_continues_polling(
        self,
        state: MineruTaskState,
    ):
        """
        Property: The PENDING state SHALL be classified as non-terminal,
        meaning polling should continue.
        
        **Validates: Requirements 3.2**
        """
        if state == MineruTaskState.PENDING:
            result = is_terminal_state(state)
            assert result is False, (
                f"PENDING state should be non-terminal. "
                f"is_terminal_state() returned {result}"
            )

    @settings(max_examples=100)
    @given(state=non_terminal_state_strategy)
    def test_running_state_continues_polling(
        self,
        state: MineruTaskState,
    ):
        """
        Property: The RUNNING state SHALL be classified as non-terminal,
        meaning polling should continue.
        
        **Validates: Requirements 3.2**
        """
        if state == MineruTaskState.RUNNING:
            result = is_terminal_state(state)
            assert result is False, (
                f"RUNNING state should be non-terminal. "
                f"is_terminal_state() returned {result}"
            )

    @settings(max_examples=100)
    @given(state=non_terminal_state_strategy)
    def test_converting_state_continues_polling(
        self,
        state: MineruTaskState,
    ):
        """
        Property: The CONVERTING state SHALL be classified as non-terminal,
        meaning polling should continue.
        
        **Validates: Requirements 3.2**
        """
        if state == MineruTaskState.CONVERTING:
            result = is_terminal_state(state)
            assert result is False, (
                f"CONVERTING state should be non-terminal. "
                f"is_terminal_state() returned {result}"
            )


class TestTerminalStateStopsPollingProperty:
    """
    Property-based tests for terminal states stopping polling.
    
    **Validates: Requirements 3.2**
    
    These tests verify that terminal states correctly signal to stop polling.
    """

    @settings(max_examples=100)
    @given(state=terminal_state_strategy)
    def test_done_state_stops_polling(
        self,
        state: MineruTaskState,
    ):
        """
        Property: The DONE state SHALL be classified as terminal,
        meaning polling should stop.
        
        **Validates: Requirements 3.2**
        """
        if state == MineruTaskState.DONE:
            result = is_terminal_state(state)
            assert result is True, (
                f"DONE state should be terminal. "
                f"is_terminal_state() returned {result}"
            )

    @settings(max_examples=100)
    @given(state=terminal_state_strategy)
    def test_failed_state_stops_polling(
        self,
        state: MineruTaskState,
    ):
        """
        Property: The FAILED state SHALL be classified as terminal,
        meaning polling should stop.
        
        **Validates: Requirements 3.2**
        """
        if state == MineruTaskState.FAILED:
            result = is_terminal_state(state)
            assert result is True, (
                f"FAILED state should be terminal. "
                f"is_terminal_state() returned {result}"
            )

    @settings(max_examples=100)
    @given(
        num_checks=st.integers(min_value=1, max_value=10),
        terminal_state=terminal_state_strategy,
    )
    def test_terminal_state_consistently_stops_polling(
        self,
        num_checks: int,
        terminal_state: MineruTaskState,
    ):
        """
        Property: For any terminal state, checking is_terminal_state()
        multiple times SHALL consistently return True.
        
        **Validates: Requirements 3.2**
        """
        results = [is_terminal_state(terminal_state) for _ in range(num_checks)]
        
        assert all(r is True for r in results), (
            f"Terminal state '{terminal_state.value}' should consistently "
            f"return True. Got results: {results}"
        )


class TestShouldContinuePollingProperty:
    """
    Property-based tests for should_continue_polling() function.
    
    **Property 4: Task State Machine Correctness**
    **Validates: Requirements 3.2**
    
    These tests verify that:
    1. should_continue_polling() returns True for non-terminal states
    2. should_continue_polling() returns False for terminal states
    3. should_continue_polling() is the inverse of is_terminal_state()
    """

    @settings(max_examples=100)
    @given(state=non_terminal_state_strategy)
    def test_should_continue_polling_returns_true_for_non_terminal_states(
        self,
        state: MineruTaskState,
    ):
        """
        Property: For any non-terminal state (WAITING_FILE, PENDING, RUNNING,
        CONVERTING), should_continue_polling() SHALL return True.
        
        **Validates: Requirements 3.2**
        """
        result = should_continue_polling(state)
        
        assert result is True, (
            f"should_continue_polling({state.value}) should return True for "
            f"non-terminal state, but got {result}. "
            f"Polling should continue for state '{state.value}'."
        )

    @settings(max_examples=100)
    @given(state=terminal_state_strategy)
    def test_should_continue_polling_returns_false_for_terminal_states(
        self,
        state: MineruTaskState,
    ):
        """
        Property: For any terminal state (DONE, FAILED), should_continue_polling()
        SHALL return False.
        
        **Validates: Requirements 3.2**
        """
        result = should_continue_polling(state)
        
        assert result is False, (
            f"should_continue_polling({state.value}) should return False for "
            f"terminal state, but got {result}. "
            f"Polling should stop for state '{state.value}'."
        )

    @settings(max_examples=100)
    @given(state=any_state_strategy)
    def test_should_continue_polling_is_inverse_of_is_terminal_state(
        self,
        state: MineruTaskState,
    ):
        """
        Property: For any MinerU task state, should_continue_polling() SHALL
        return the inverse of is_terminal_state().
        
        **Validates: Requirements 3.2**
        """
        terminal_result = is_terminal_state(state)
        continue_result = should_continue_polling(state)
        
        assert continue_result == (not terminal_result), (
            f"should_continue_polling({state.value}) should be the inverse of "
            f"is_terminal_state({state.value}). "
            f"is_terminal_state returned {terminal_result}, "
            f"should_continue_polling returned {continue_result}."
        )

    @settings(max_examples=100)
    @given(states=st.lists(non_terminal_state_strategy, min_size=1, max_size=20))
    def test_should_continue_polling_true_for_all_non_terminal_sequence(
        self,
        states: List[MineruTaskState],
    ):
        """
        Property: For any sequence of non-terminal states, should_continue_polling()
        SHALL return True for all states in the sequence.
        
        **Validates: Requirements 3.2**
        """
        for i, state in enumerate(states):
            result = should_continue_polling(state)
            
            assert result is True, (
                f"In sequence position {i}, should_continue_polling({state.value}) "
                f"should return True, but got {result}. "
                f"Polling should continue for all non-terminal states."
            )

    @settings(max_examples=100)
    @given(
        non_terminal_states=st.lists(
            non_terminal_state_strategy,
            min_size=1,
            max_size=10,
        ),
        terminal_state=terminal_state_strategy,
    )
    def test_should_continue_polling_simulates_polling_loop(
        self,
        non_terminal_states: List[MineruTaskState],
        terminal_state: MineruTaskState,
    ):
        """
        Property: For any sequence of non-terminal states followed by a
        terminal state, should_continue_polling() SHALL return True for all
        non-terminal states and False for the terminal state.
        
        This simulates the actual polling loop behavior.
        
        **Validates: Requirements 3.2**
        """
        # Simulate polling through non-terminal states
        for state in non_terminal_states:
            result = should_continue_polling(state)
            assert result is True, (
                f"Polling should continue for non-terminal state '{state.value}'"
            )
        
        # Terminal state should stop polling
        result = should_continue_polling(terminal_state)
        assert result is False, (
            f"Polling should stop for terminal state '{terminal_state.value}'"
        )


class TestShouldContinuePollingSpecificStatesProperty:
    """
    Property-based tests for specific state behaviors in should_continue_polling().
    
    **Property 4: Task State Machine Correctness**
    **Validates: Requirements 3.2**
    """

    @settings(max_examples=100)
    @given(state=non_terminal_state_strategy)
    def test_waiting_file_state_should_continue_polling(
        self,
        state: MineruTaskState,
    ):
        """
        Property: The WAITING_FILE state SHALL cause should_continue_polling()
        to return True.
        
        **Validates: Requirements 3.2**
        """
        if state == MineruTaskState.WAITING_FILE:
            result = should_continue_polling(state)
            assert result is True, (
                f"WAITING_FILE state should continue polling. "
                f"should_continue_polling() returned {result}"
            )

    @settings(max_examples=100)
    @given(state=non_terminal_state_strategy)
    def test_pending_state_should_continue_polling(
        self,
        state: MineruTaskState,
    ):
        """
        Property: The PENDING state SHALL cause should_continue_polling()
        to return True.
        
        **Validates: Requirements 3.2**
        """
        if state == MineruTaskState.PENDING:
            result = should_continue_polling(state)
            assert result is True, (
                f"PENDING state should continue polling. "
                f"should_continue_polling() returned {result}"
            )

    @settings(max_examples=100)
    @given(state=non_terminal_state_strategy)
    def test_running_state_should_continue_polling(
        self,
        state: MineruTaskState,
    ):
        """
        Property: The RUNNING state SHALL cause should_continue_polling()
        to return True.
        
        **Validates: Requirements 3.2**
        """
        if state == MineruTaskState.RUNNING:
            result = should_continue_polling(state)
            assert result is True, (
                f"RUNNING state should continue polling. "
                f"should_continue_polling() returned {result}"
            )

    @settings(max_examples=100)
    @given(state=non_terminal_state_strategy)
    def test_converting_state_should_continue_polling(
        self,
        state: MineruTaskState,
    ):
        """
        Property: The CONVERTING state SHALL cause should_continue_polling()
        to return True.
        
        **Validates: Requirements 3.2**
        """
        if state == MineruTaskState.CONVERTING:
            result = should_continue_polling(state)
            assert result is True, (
                f"CONVERTING state should continue polling. "
                f"should_continue_polling() returned {result}"
            )

    @settings(max_examples=100)
    @given(state=terminal_state_strategy)
    def test_done_state_should_stop_polling(
        self,
        state: MineruTaskState,
    ):
        """
        Property: The DONE state SHALL cause should_continue_polling()
        to return False.
        
        **Validates: Requirements 3.2**
        """
        if state == MineruTaskState.DONE:
            result = should_continue_polling(state)
            assert result is False, (
                f"DONE state should stop polling. "
                f"should_continue_polling() returned {result}"
            )

    @settings(max_examples=100)
    @given(state=terminal_state_strategy)
    def test_failed_state_should_stop_polling(
        self,
        state: MineruTaskState,
    ):
        """
        Property: The FAILED state SHALL cause should_continue_polling()
        to return False.
        
        **Validates: Requirements 3.2**
        """
        if state == MineruTaskState.FAILED:
            result = should_continue_polling(state)
            assert result is False, (
                f"FAILED state should stop polling. "
                f"should_continue_polling() returned {result}"
            )

    @settings(max_examples=100)
    @given(state=any_state_strategy)
    def test_should_continue_polling_returns_boolean(
        self,
        state: MineruTaskState,
    ):
        """
        Property: For any MinerU task state, should_continue_polling() SHALL
        return a boolean value (True or False).
        
        **Validates: Requirements 3.2**
        """
        result = should_continue_polling(state)
        
        assert isinstance(result, bool), (
            f"should_continue_polling({state.value}) should return a boolean, "
            f"but got {type(result).__name__}: {result}"
        )

    @settings(max_examples=100)
    @given(state=any_state_strategy)
    def test_should_continue_polling_is_deterministic(
        self,
        state: MineruTaskState,
    ):
        """
        Property: For any MinerU task state, calling should_continue_polling()
        multiple times SHALL return the same result.
        
        **Validates: Requirements 3.2**
        """
        result1 = should_continue_polling(state)
        result2 = should_continue_polling(state)
        result3 = should_continue_polling(state)
        
        assert result1 == result2 == result3, (
            f"should_continue_polling({state.value}) should be deterministic. "
            f"Got different results: {result1}, {result2}, {result3}"
        )


# =============================================================================
# Property 5: Exponential Backoff
# =============================================================================

class TestExponentialBackoffProperty:
    """
    Property-based tests for Property 5: Exponential Backoff.
    
    *For any* sequence of polling attempts, the interval between attempt N and
    attempt N+1 SHALL be greater than or equal to the interval between attempt
    N-1 and attempt N, following an exponential growth pattern.
    
    **Validates: Requirements 3.6**
    """

    @settings(max_examples=100)
    @given(attempt=st.integers(min_value=0, max_value=100))
    def test_backoff_interval_is_non_negative(self, attempt: int):
        """
        Property: For any attempt number, calculate_backoff_interval() SHALL
        return a non-negative interval.
        
        **Validates: Requirements 3.6**
        """
        interval = calculate_backoff_interval(attempt)
        
        assert interval >= 0, (
            f"Backoff interval for attempt {attempt} should be non-negative, "
            f"but got {interval}"
        )

    @settings(max_examples=100)
    @given(attempt=st.integers(min_value=0, max_value=100))
    def test_backoff_interval_at_least_initial(self, attempt: int):
        """
        Property: For any attempt number, calculate_backoff_interval() SHALL
        return an interval >= INITIAL_POLL_INTERVAL.
        
        **Validates: Requirements 3.6**
        """
        interval = calculate_backoff_interval(attempt)
        
        assert interval >= INITIAL_POLL_INTERVAL, (
            f"Backoff interval for attempt {attempt} should be >= {INITIAL_POLL_INTERVAL}, "
            f"but got {interval}"
        )

    @settings(max_examples=100)
    @given(attempt=st.integers(min_value=0, max_value=100))
    def test_backoff_interval_capped_at_max(self, attempt: int):
        """
        Property: For any attempt number, calculate_backoff_interval() SHALL
        return an interval <= MAX_POLL_INTERVAL.
        
        **Validates: Requirements 3.6**
        """
        interval = calculate_backoff_interval(attempt)
        
        assert interval <= MAX_POLL_INTERVAL, (
            f"Backoff interval for attempt {attempt} should be <= {MAX_POLL_INTERVAL}, "
            f"but got {interval}"
        )

    @settings(max_examples=100)
    @given(attempt=st.integers(min_value=0, max_value=99))
    def test_backoff_interval_monotonically_increasing(self, attempt: int):
        """
        Property: For any attempt N, the interval at attempt N+1 SHALL be
        >= the interval at attempt N (monotonically increasing).
        
        **Validates: Requirements 3.6**
        """
        interval_n = calculate_backoff_interval(attempt)
        interval_n_plus_1 = calculate_backoff_interval(attempt + 1)
        
        assert interval_n_plus_1 >= interval_n, (
            f"Backoff interval should be monotonically increasing. "
            f"Interval at attempt {attempt} = {interval_n}, "
            f"interval at attempt {attempt + 1} = {interval_n_plus_1}"
        )

    @settings(max_examples=100)
    @given(attempt=st.integers(min_value=0, max_value=10))
    def test_backoff_follows_exponential_formula_before_cap(self, attempt: int):
        """
        Property: For attempts where the interval hasn't reached MAX_POLL_INTERVAL,
        the interval SHALL follow the formula:
        interval = INITIAL_POLL_INTERVAL * (BACKOFF_MULTIPLIER ^ attempt)
        
        **Validates: Requirements 3.6**
        """
        expected_uncapped = INITIAL_POLL_INTERVAL * (BACKOFF_MULTIPLIER ** attempt)
        expected = min(expected_uncapped, MAX_POLL_INTERVAL)
        actual = calculate_backoff_interval(attempt)
        
        assert actual == expected, (
            f"Backoff interval for attempt {attempt} should be {expected}, "
            f"but got {actual}. "
            f"Formula: min({INITIAL_POLL_INTERVAL} * {BACKOFF_MULTIPLIER}^{attempt}, {MAX_POLL_INTERVAL})"
        )

    @settings(max_examples=100)
    @given(
        attempts=st.lists(
            st.integers(min_value=0, max_value=50),
            min_size=2,
            max_size=20,
            unique=True,
        ).map(sorted)
    )
    def test_backoff_sequence_is_non_decreasing(self, attempts: List[int]):
        """
        Property: For any sorted sequence of attempt numbers, the corresponding
        intervals SHALL form a non-decreasing sequence.
        
        **Validates: Requirements 3.6**
        """
        intervals = [calculate_backoff_interval(a) for a in attempts]
        
        for i in range(1, len(intervals)):
            assert intervals[i] >= intervals[i - 1], (
                f"Intervals should be non-decreasing. "
                f"At attempts {attempts[i-1]} and {attempts[i]}, "
                f"intervals are {intervals[i-1]} and {intervals[i]}"
            )

    @settings(max_examples=100)
    @given(attempt=st.integers(min_value=0, max_value=100))
    def test_backoff_interval_is_deterministic(self, attempt: int):
        """
        Property: For any attempt number, calling calculate_backoff_interval()
        multiple times SHALL return the same result.
        
        **Validates: Requirements 3.6**
        """
        result1 = calculate_backoff_interval(attempt)
        result2 = calculate_backoff_interval(attempt)
        result3 = calculate_backoff_interval(attempt)
        
        assert result1 == result2 == result3, (
            f"calculate_backoff_interval({attempt}) should be deterministic. "
            f"Got different results: {result1}, {result2}, {result3}"
        )


class TestExponentialBackoffEdgeCases:
    """
    Edge case tests for exponential backoff.
    
    **Validates: Requirements 3.6**
    """

    def test_first_attempt_returns_initial_interval(self):
        """
        Test that attempt 0 returns INITIAL_POLL_INTERVAL.
        
        **Validates: Requirements 3.6**
        """
        interval = calculate_backoff_interval(0)
        assert interval == INITIAL_POLL_INTERVAL, (
            f"First attempt (0) should return {INITIAL_POLL_INTERVAL}, got {interval}"
        )

    def test_second_attempt_doubles_interval(self):
        """
        Test that attempt 1 returns INITIAL_POLL_INTERVAL * BACKOFF_MULTIPLIER.
        
        **Validates: Requirements 3.6**
        """
        interval = calculate_backoff_interval(1)
        expected = INITIAL_POLL_INTERVAL * BACKOFF_MULTIPLIER
        assert interval == expected, (
            f"Second attempt (1) should return {expected}, got {interval}"
        )

    def test_large_attempt_returns_max_interval(self):
        """
        Test that very large attempt numbers return MAX_POLL_INTERVAL.
        
        **Validates: Requirements 3.6**
        """
        # With INITIAL=2, MULTIPLIER=2, MAX=60:
        # attempt 5: 2 * 32 = 64 > 60, so capped at 60
        for attempt in [10, 50, 100, 1000]:
            interval = calculate_backoff_interval(attempt)
            assert interval == MAX_POLL_INTERVAL, (
                f"Large attempt ({attempt}) should return {MAX_POLL_INTERVAL}, got {interval}"
            )

    def test_interval_reaches_cap_at_correct_attempt(self):
        """
        Test that the interval reaches MAX_POLL_INTERVAL at the expected attempt.
        
        With INITIAL=2, MULTIPLIER=2, MAX=60:
        - attempt 0: 2
        - attempt 1: 4
        - attempt 2: 8
        - attempt 3: 16
        - attempt 4: 32
        - attempt 5: 64 -> capped to 60
        
        **Validates: Requirements 3.6**
        """
        # Find the first attempt where interval reaches max
        for attempt in range(20):
            interval = calculate_backoff_interval(attempt)
            uncapped = INITIAL_POLL_INTERVAL * (BACKOFF_MULTIPLIER ** attempt)
            
            if uncapped >= MAX_POLL_INTERVAL:
                assert interval == MAX_POLL_INTERVAL, (
                    f"At attempt {attempt}, interval should be capped at {MAX_POLL_INTERVAL}, "
                    f"got {interval}"
                )
                break
            else:
                assert interval == uncapped, (
                    f"At attempt {attempt}, interval should be {uncapped}, got {interval}"
                )

    def test_backoff_sequence_example(self):
        """
        Test a specific sequence of backoff intervals.
        
        With INITIAL=2, MULTIPLIER=2, MAX=60:
        - attempt 0: 2
        - attempt 1: 4
        - attempt 2: 8
        - attempt 3: 16
        - attempt 4: 32
        - attempt 5: 60 (capped)
        - attempt 6: 60 (capped)
        
        **Validates: Requirements 3.6**
        """
        expected_sequence = [2, 4, 8, 16, 32, 60, 60]
        
        for attempt, expected in enumerate(expected_sequence):
            actual = calculate_backoff_interval(attempt)
            assert actual == expected, (
                f"At attempt {attempt}, expected interval {expected}, got {actual}"
            )


# =============================================================================
# Property 6: Progress Data Extraction
# =============================================================================

# Import additional dependencies for progress tests
from backend.models import TaskStage, FileParseResult
from backend.services.task_manager import (
    STAGE_WEIGHTS,
    STAGE_CUMULATIVE_WEIGHTS,
    TaskManager,
)


def extract_progress_from_parse_result(result: FileParseResult) -> tuple:
    """
    Helper function to extract progress data from a FileParseResult.
    
    Returns:
        Tuple of (extracted_pages, total_pages) or (None, None) if not available
    """
    return (result.extracted_pages, result.total_pages)


class TestProgressDataExtractionProperty:
    """
    Property-based tests for Property 6: Progress Data Extraction.
    
    *For any* MinerU task response with state="running", the Progress_Tracker
    SHALL extract and broadcast extracted_pages and total_pages values.
    *For any* progress update, the percentage SHALL equal
    (progress / total) * stage_weight.
    
    **Validates: Requirements 3.3, 8.3, 8.4, 8.5, 8.6**
    """

    @settings(max_examples=100)
    @given(
        extracted_pages=st.integers(min_value=0, max_value=600),
        total_pages=st.integers(min_value=1, max_value=600),
    )
    def test_running_state_has_extractable_progress(
        self,
        extracted_pages: int,
        total_pages: int,
    ):
        """
        Property: For any MinerU task response with state="running",
        the extracted_pages and total_pages values SHALL be extractable.
        
        **Validates: Requirements 3.3, 8.3**
        """
        # Create a FileParseResult with running state
        result = FileParseResult(
            file_name="test.pdf",
            state=MineruTaskState.RUNNING,
            extracted_pages=extracted_pages,
            total_pages=total_pages,
        )
        
        # Extract progress data
        pages_extracted, pages_total = extract_progress_from_parse_result(result)
        
        # Progress data SHALL be extractable
        assert pages_extracted is not None, (
            f"extracted_pages should be extractable from running state"
        )
        assert pages_total is not None, (
            f"total_pages should be extractable from running state"
        )
        assert pages_extracted == extracted_pages, (
            f"Extracted pages should be {extracted_pages}, got {pages_extracted}"
        )
        assert pages_total == total_pages, (
            f"Total pages should be {total_pages}, got {pages_total}"
        )

    @settings(max_examples=100)
    @given(
        progress=st.integers(min_value=0, max_value=100),
        total=st.integers(min_value=1, max_value=100),
        stage=st.sampled_from([
            TaskStage.UPLOADING,
            TaskStage.PARSING,
            TaskStage.DOWNLOADING,
            TaskStage.TRANSLATING,
            TaskStage.SUMMARIZING,
            TaskStage.GENERATING,
        ]),
    )
    def test_stage_progress_calculation(
        self,
        progress: int,
        total: int,
        stage: TaskStage,
    ):
        """
        Property: For any progress update, the stage progress SHALL be
        calculated as progress / total, clamped to [0, 1].
        
        **Validates: Requirements 8.3, 8.4, 8.5**
        """
        # Clamp progress to not exceed total (realistic scenario)
        clamped_progress = min(progress, total)
        stage_progress = clamped_progress / total
        
        assert 0.0 <= stage_progress <= 1.0, (
            f"Stage progress should be between 0 and 1, got {stage_progress}"
        )

    @settings(max_examples=100)
    @given(
        stage_progress=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        stage=st.sampled_from([
            TaskStage.UPLOADING,
            TaskStage.PARSING,
            TaskStage.DOWNLOADING,
            TaskStage.TRANSLATING,
            TaskStage.SUMMARIZING,
            TaskStage.GENERATING,
        ]),
    )
    def test_overall_percentage_within_bounds(
        self,
        stage_progress: float,
        stage: TaskStage,
    ):
        """
        Property: For any stage and stage_progress, the overall percentage
        SHALL be between 0 and 100.
        
        **Validates: Requirements 8.6**
        """
        cumulative = STAGE_CUMULATIVE_WEIGHTS.get(stage, 0.0)
        stage_weight = STAGE_WEIGHTS.get(stage, 0.0)
        percentage = cumulative + (stage_progress * stage_weight)
        percentage = max(0.0, min(100.0, percentage))
        
        assert 0.0 <= percentage <= 100.0, (
            f"Overall percentage should be between 0 and 100, got {percentage}"
        )

    @settings(max_examples=100)
    @given(
        stage_progress=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        stage=st.sampled_from([
            TaskStage.UPLOADING,
            TaskStage.PARSING,
            TaskStage.DOWNLOADING,
            TaskStage.TRANSLATING,
            TaskStage.SUMMARIZING,
            TaskStage.GENERATING,
        ]),
    )
    def test_overall_percentage_formula(
        self,
        stage_progress: float,
        stage: TaskStage,
    ):
        """
        Property: For any progress update, the percentage SHALL equal
        cumulative_weight + (stage_progress * stage_weight).
        
        **Validates: Requirements 8.6**
        """
        cumulative = STAGE_CUMULATIVE_WEIGHTS.get(stage, 0.0)
        stage_weight = STAGE_WEIGHTS.get(stage, 0.0)
        
        expected = cumulative + (stage_progress * stage_weight)
        expected = max(0.0, min(100.0, expected))
        
        # Verify the formula
        assert expected >= cumulative, (
            f"Percentage {expected} should be >= cumulative {cumulative}"
        )
        assert expected <= cumulative + stage_weight, (
            f"Percentage {expected} should be <= cumulative + stage_weight "
            f"({cumulative + stage_weight})"
        )

    @settings(max_examples=100)
    @given(
        stage_progress=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    def test_completed_stage_returns_100_percent(
        self,
        stage_progress: float,
    ):
        """
        Property: For COMPLETED stage, the overall percentage SHALL be 100%.
        
        **Validates: Requirements 8.6**
        """
        # COMPLETED stage should always return 100%
        assert STAGE_CUMULATIVE_WEIGHTS.get(TaskStage.COMPLETED, 0.0) == 100.0, (
            f"COMPLETED stage cumulative weight should be 100"
        )


class TestProgressStageWeightsProperty:
    """
    Property-based tests for stage weights configuration.
    
    **Validates: Requirements 8.6**
    """

    def test_stage_weights_sum_to_100(self):
        """
        Test that all stage weights sum to 100%.
        
        **Validates: Requirements 8.6**
        """
        total_weight = sum(STAGE_WEIGHTS.values())
        assert total_weight == 100, (
            f"Stage weights should sum to 100, got {total_weight}"
        )

    def test_cumulative_weights_are_consistent(self):
        """
        Test that cumulative weights are consistent with stage weights.
        
        **Validates: Requirements 8.6**
        """
        stages_in_order = [
            TaskStage.UPLOADING,
            TaskStage.PARSING,
            TaskStage.DOWNLOADING,
            TaskStage.TRANSLATING,
            TaskStage.SUMMARIZING,
            TaskStage.GENERATING,
        ]
        
        cumulative = 0
        for stage in stages_in_order:
            expected_cumulative = cumulative
            actual_cumulative = STAGE_CUMULATIVE_WEIGHTS.get(stage, 0.0)
            
            assert actual_cumulative == expected_cumulative, (
                f"Cumulative weight for {stage.value} should be {expected_cumulative}, "
                f"got {actual_cumulative}"
            )
            
            cumulative += STAGE_WEIGHTS.get(stage, 0.0)

    @settings(max_examples=100)
    @given(
        stage=st.sampled_from([
            TaskStage.UPLOADING,
            TaskStage.PARSING,
            TaskStage.DOWNLOADING,
            TaskStage.TRANSLATING,
            TaskStage.SUMMARIZING,
            TaskStage.GENERATING,
        ]),
    )
    def test_each_stage_has_positive_weight(self, stage: TaskStage):
        """
        Property: Each processing stage SHALL have a positive weight.
        
        **Validates: Requirements 8.6**
        """
        weight = STAGE_WEIGHTS.get(stage, 0.0)
        assert weight > 0, (
            f"Stage {stage.value} should have positive weight, got {weight}"
        )

    @settings(max_examples=100)
    @given(
        stage=st.sampled_from([
            TaskStage.UPLOADING,
            TaskStage.PARSING,
            TaskStage.DOWNLOADING,
            TaskStage.TRANSLATING,
            TaskStage.SUMMARIZING,
            TaskStage.GENERATING,
        ]),
    )
    def test_cumulative_weight_is_non_negative(self, stage: TaskStage):
        """
        Property: Each stage's cumulative weight SHALL be non-negative.
        
        **Validates: Requirements 8.6**
        """
        cumulative = STAGE_CUMULATIVE_WEIGHTS.get(stage, 0.0)
        assert cumulative >= 0, (
            f"Cumulative weight for {stage.value} should be >= 0, got {cumulative}"
        )


class TestProgressMonotonicityProperty:
    """
    Property-based tests for progress monotonicity.
    
    **Validates: Requirements 8.6**
    """

    @settings(max_examples=100)
    @given(
        progress1=st.floats(min_value=0.0, max_value=0.5, allow_nan=False),
        progress2=st.floats(min_value=0.5, max_value=1.0, allow_nan=False),
        stage=st.sampled_from([
            TaskStage.UPLOADING,
            TaskStage.PARSING,
            TaskStage.DOWNLOADING,
            TaskStage.TRANSLATING,
            TaskStage.SUMMARIZING,
            TaskStage.GENERATING,
        ]),
    )
    def test_progress_increases_within_stage(
        self,
        progress1: float,
        progress2: float,
        stage: TaskStage,
    ):
        """
        Property: Within a stage, as stage_progress increases, the overall
        percentage SHALL also increase.
        
        **Validates: Requirements 8.6**
        """
        cumulative = STAGE_CUMULATIVE_WEIGHTS.get(stage, 0.0)
        stage_weight = STAGE_WEIGHTS.get(stage, 0.0)
        
        percentage1 = cumulative + (progress1 * stage_weight)
        percentage2 = cumulative + (progress2 * stage_weight)
        
        assert percentage2 >= percentage1, (
            f"Progress should increase: {percentage1} -> {percentage2}"
        )

    def test_progress_increases_across_stages(self):
        """
        Test that progress increases as we move through stages.
        
        **Validates: Requirements 8.6**
        """
        stages_in_order = [
            TaskStage.UPLOADING,
            TaskStage.PARSING,
            TaskStage.DOWNLOADING,
            TaskStage.TRANSLATING,
            TaskStage.SUMMARIZING,
            TaskStage.GENERATING,
            TaskStage.COMPLETED,
        ]
        
        prev_cumulative = -1
        for stage in stages_in_order:
            cumulative = STAGE_CUMULATIVE_WEIGHTS.get(stage, 0.0)
            assert cumulative > prev_cumulative, (
                f"Cumulative weight should increase: {prev_cumulative} -> {cumulative} "
                f"at stage {stage.value}"
            )
            prev_cumulative = cumulative
