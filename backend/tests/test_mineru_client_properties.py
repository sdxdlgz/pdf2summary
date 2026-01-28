"""
Property-based tests for MinerU client.

**Property 3: MinerU API Parameter Correctness**
*For any* request to the MinerU batch file upload API, the request body SHALL
contain enable_table=true, enable_formula=true, extra_formats=["docx"], and
model_version="vlm".

**Validates: Requirements 2.5, 2.6, 2.7, 2.8**

**Property 7: Error Propagation**
*For any* failed MinerU task, the err_msg SHALL be recorded in TaskStatus.
*For any* error during processing, the error message SHALL be broadcast via WebSocket.

**Validates: Requirements 2.10, 3.5, 8.7**

Uses Hypothesis for property-based testing with at least 100 iterations per test.
"""

import copy
from typing import Any, Optional

import pytest
from hypothesis import given, settings, strategies as st, assume

from backend.models import FileParseResult, MineruTaskState
from backend.services.mineru_client import MineruClient, MineruClientError


# Strategy for generating valid API tokens (non-empty strings)
valid_api_token = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S"),
        blacklist_characters="\x00",
    ),
    min_size=1,
    max_size=100,
).filter(lambda s: s.strip())


# Strategy for generating valid base URLs
valid_base_url = st.sampled_from([
    "https://mineru.net",
    "https://api.mineru.net",
    "https://custom.api.example.com",
    "http://localhost:8080",
    "https://test.mineru.io",
])


# Strategy for generating arbitrary callback functions or None
callback_strategy = st.sampled_from([
    None,
    lambda batch_id, error_msg: None,
    lambda batch_id, error_msg: print(f"Error: {error_msg}"),
])


class TestMineruApiParameterCorrectnessProperty:
    """
    Property-based tests for Property 3: MinerU API Parameter Correctness.
    
    **Validates: Requirements 2.5, 2.6, 2.7, 2.8**
    
    These tests verify that:
    1. get_api_parameters() always returns the correct fixed values
    2. Parameters are immutable and cannot be changed after client creation
    3. Parameters are consistent across all client instances
    4. Class constants match the expected values
    """

    @settings(max_examples=100)
    @given(api_token=valid_api_token)
    def test_api_parameters_always_correct_for_any_token(
        self,
        api_token: str,
    ):
        """
        Property: For any valid API token, get_api_parameters() SHALL return
        enable_table=true, enable_formula=true, extra_formats=["docx"],
        and model_version="vlm".
        
        **Validates: Requirements 2.5, 2.6, 2.7, 2.8**
        """
        client = MineruClient(api_token=api_token)
        params = client.get_api_parameters()
        
        # Requirement 2.5: enable_table=true
        assert params["enable_table"] is True, (
            f"enable_table should be True, got {params['enable_table']}"
        )
        
        # Requirement 2.6: enable_formula=true
        assert params["enable_formula"] is True, (
            f"enable_formula should be True, got {params['enable_formula']}"
        )
        
        # Requirement 2.7: extra_formats=["docx"]
        assert params["extra_formats"] == ["docx"], (
            f"extra_formats should be ['docx'], got {params['extra_formats']}"
        )
        
        # Requirement 2.8: model_version="vlm"
        assert params["model_version"] == "vlm", (
            f"model_version should be 'vlm', got {params['model_version']}"
        )

    @settings(max_examples=100)
    @given(
        api_token=valid_api_token,
        base_url=valid_base_url,
    )
    def test_api_parameters_consistent_regardless_of_base_url(
        self,
        api_token: str,
        base_url: str,
    ):
        """
        Property: For any combination of API token and base URL,
        get_api_parameters() SHALL return the same fixed values.
        
        **Validates: Requirements 2.5, 2.6, 2.7, 2.8**
        """
        client = MineruClient(api_token=api_token, base_url=base_url)
        params = client.get_api_parameters()
        
        # Parameters should be identical regardless of base_url
        assert params["enable_table"] is True
        assert params["enable_formula"] is True
        assert params["extra_formats"] == ["docx"]
        assert params["model_version"] == "vlm"

    @settings(max_examples=100)
    @given(
        api_token1=valid_api_token,
        api_token2=valid_api_token,
    )
    def test_api_parameters_identical_across_different_instances(
        self,
        api_token1: str,
        api_token2: str,
    ):
        """
        Property: For any two MineruClient instances (with any tokens),
        get_api_parameters() SHALL return identical values.
        
        **Validates: Requirements 2.5, 2.6, 2.7, 2.8**
        """
        client1 = MineruClient(api_token=api_token1)
        client2 = MineruClient(api_token=api_token2)
        
        params1 = client1.get_api_parameters()
        params2 = client2.get_api_parameters()
        
        # Parameters should be identical across instances
        assert params1 == params2, (
            f"Parameters should be identical across instances. "
            f"Client1: {params1}, Client2: {params2}"
        )

    @settings(max_examples=100)
    @given(api_token=valid_api_token)
    def test_api_parameters_immutable_after_creation(
        self,
        api_token: str,
    ):
        """
        Property: For any MineruClient instance, the API parameters
        SHALL be immutable and cannot be changed after client creation.
        
        **Validates: Requirements 2.5, 2.6, 2.7, 2.8**
        """
        client = MineruClient(api_token=api_token)
        
        # Get parameters twice
        params_before = client.get_api_parameters()
        
        # Attempt to modify the returned dictionary (should not affect client)
        params_before["enable_table"] = False
        params_before["enable_formula"] = False
        params_before["extra_formats"] = ["pdf"]
        params_before["model_version"] = "other"
        
        # Get parameters again
        params_after = client.get_api_parameters()
        
        # Parameters should still be correct (immutable)
        assert params_after["enable_table"] is True, (
            "enable_table should remain True after external modification attempt"
        )
        assert params_after["enable_formula"] is True, (
            "enable_formula should remain True after external modification attempt"
        )
        assert params_after["extra_formats"] == ["docx"], (
            "extra_formats should remain ['docx'] after external modification attempt"
        )
        assert params_after["model_version"] == "vlm", (
            "model_version should remain 'vlm' after external modification attempt"
        )

    @settings(max_examples=100)
    @given(api_token=valid_api_token)
    def test_class_constants_match_api_parameters(
        self,
        api_token: str,
    ):
        """
        Property: For any MineruClient instance, the class constants
        SHALL match the values returned by get_api_parameters().
        
        **Validates: Requirements 2.5, 2.6, 2.7, 2.8**
        """
        client = MineruClient(api_token=api_token)
        params = client.get_api_parameters()
        
        # Class constants should match get_api_parameters() output
        assert MineruClient.ENABLE_TABLE == params["enable_table"], (
            f"ENABLE_TABLE constant ({MineruClient.ENABLE_TABLE}) should match "
            f"get_api_parameters() value ({params['enable_table']})"
        )
        assert MineruClient.ENABLE_FORMULA == params["enable_formula"], (
            f"ENABLE_FORMULA constant ({MineruClient.ENABLE_FORMULA}) should match "
            f"get_api_parameters() value ({params['enable_formula']})"
        )
        assert MineruClient.EXTRA_FORMATS == params["extra_formats"], (
            f"EXTRA_FORMATS constant ({MineruClient.EXTRA_FORMATS}) should match "
            f"get_api_parameters() value ({params['extra_formats']})"
        )
        assert MineruClient.MODEL_VERSION == params["model_version"], (
            f"MODEL_VERSION constant ({MineruClient.MODEL_VERSION}) should match "
            f"get_api_parameters() value ({params['model_version']})"
        )


class TestMineruApiParameterValuesProperty:
    """
    Property-based tests verifying the exact values of API parameters.
    
    **Validates: Requirements 2.5, 2.6, 2.7, 2.8**
    
    These tests verify that the class constants themselves have the
    correct values as specified in the requirements.
    """

    @settings(max_examples=100)
    @given(api_token=valid_api_token)
    def test_enable_table_is_boolean_true(
        self,
        api_token: str,
    ):
        """
        Property: For any MineruClient, ENABLE_TABLE SHALL be boolean True.
        
        **Validates: Requirement 2.5**
        """
        client = MineruClient(api_token=api_token)
        
        # Must be exactly boolean True, not truthy
        assert client.ENABLE_TABLE is True
        assert type(client.ENABLE_TABLE) is bool

    @settings(max_examples=100)
    @given(api_token=valid_api_token)
    def test_enable_formula_is_boolean_true(
        self,
        api_token: str,
    ):
        """
        Property: For any MineruClient, ENABLE_FORMULA SHALL be boolean True.
        
        **Validates: Requirement 2.6**
        """
        client = MineruClient(api_token=api_token)
        
        # Must be exactly boolean True, not truthy
        assert client.ENABLE_FORMULA is True
        assert type(client.ENABLE_FORMULA) is bool

    @settings(max_examples=100)
    @given(api_token=valid_api_token)
    def test_extra_formats_is_list_with_docx(
        self,
        api_token: str,
    ):
        """
        Property: For any MineruClient, EXTRA_FORMATS SHALL be ["docx"].
        
        **Validates: Requirement 2.7**
        """
        client = MineruClient(api_token=api_token)
        
        # Must be exactly ["docx"]
        assert client.EXTRA_FORMATS == ["docx"]
        assert type(client.EXTRA_FORMATS) is list
        assert len(client.EXTRA_FORMATS) == 1
        assert client.EXTRA_FORMATS[0] == "docx"

    @settings(max_examples=100)
    @given(api_token=valid_api_token)
    def test_model_version_is_vlm_string(
        self,
        api_token: str,
    ):
        """
        Property: For any MineruClient, MODEL_VERSION SHALL be "vlm".
        
        **Validates: Requirement 2.8**
        """
        client = MineruClient(api_token=api_token)
        
        # Must be exactly "vlm"
        assert client.MODEL_VERSION == "vlm"
        assert type(client.MODEL_VERSION) is str


class TestMineruApiParameterConsistencyProperty:
    """
    Property-based tests for parameter consistency across multiple calls.
    
    **Validates: Requirements 2.5, 2.6, 2.7, 2.8**
    
    These tests verify that parameters remain consistent across
    multiple calls and client instances.
    """

    @settings(max_examples=100)
    @given(
        api_token=valid_api_token,
        num_calls=st.integers(min_value=2, max_value=10),
    )
    def test_api_parameters_consistent_across_multiple_calls(
        self,
        api_token: str,
        num_calls: int,
    ):
        """
        Property: For any MineruClient instance, calling get_api_parameters()
        multiple times SHALL return identical values each time.
        
        **Validates: Requirements 2.5, 2.6, 2.7, 2.8**
        """
        client = MineruClient(api_token=api_token)
        
        # Get parameters multiple times
        all_params = [client.get_api_parameters() for _ in range(num_calls)]
        
        # All calls should return identical values
        first_params = all_params[0]
        for i, params in enumerate(all_params[1:], start=2):
            assert params == first_params, (
                f"Call {i} returned different parameters. "
                f"Expected: {first_params}, Got: {params}"
            )

    @settings(max_examples=100)
    @given(
        api_tokens=st.lists(valid_api_token, min_size=2, max_size=5),
    )
    def test_api_parameters_consistent_across_multiple_clients(
        self,
        api_tokens: list[str],
    ):
        """
        Property: For any list of MineruClient instances (with different tokens),
        all instances SHALL return identical API parameters.
        
        **Validates: Requirements 2.5, 2.6, 2.7, 2.8**
        """
        clients = [MineruClient(api_token=token) for token in api_tokens]
        all_params = [client.get_api_parameters() for client in clients]
        
        # All clients should return identical parameters
        first_params = all_params[0]
        for i, params in enumerate(all_params[1:], start=2):
            assert params == first_params, (
                f"Client {i} returned different parameters. "
                f"Expected: {first_params}, Got: {params}"
            )

    @settings(max_examples=100)
    @given(api_token=valid_api_token)
    def test_api_parameters_contain_all_required_keys(
        self,
        api_token: str,
    ):
        """
        Property: For any MineruClient, get_api_parameters() SHALL return
        a dictionary containing exactly the four required keys.
        
        **Validates: Requirements 2.5, 2.6, 2.7, 2.8**
        """
        client = MineruClient(api_token=api_token)
        params = client.get_api_parameters()
        
        required_keys = {"enable_table", "enable_formula", "extra_formats", "model_version"}
        
        # Should contain exactly the required keys
        assert set(params.keys()) == required_keys, (
            f"Parameters should contain exactly {required_keys}. "
            f"Got keys: {set(params.keys())}"
        )


# =============================================================================
# Property 7: Error Propagation Tests
# =============================================================================

# Strategy for generating non-empty error messages
error_message_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S", "Z"),
        blacklist_characters="\x00",
    ),
    min_size=1,
    max_size=500,
).filter(lambda s: s.strip())


# Strategy for generating error codes (can be numeric strings or alphanumeric)
error_code_strategy = st.one_of(
    st.integers(min_value=-99999, max_value=99999).map(str),
    st.text(
        alphabet=st.characters(whitelist_categories=("L", "N")),
        min_size=1,
        max_size=20,
    ).filter(lambda s: s.strip()),
)


# Strategy for generating batch IDs
batch_id_strategy = st.one_of(
    st.none(),
    st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P")),
        min_size=1,
        max_size=100,
    ).filter(lambda s: s.strip()),
)


# Strategy for generating file names
file_name_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N"),
        blacklist_characters="\x00/\\",
    ),
    min_size=1,
    max_size=100,
).map(lambda s: f"{s.strip() or 'file'}.pdf")


# Strategy for generating data IDs
data_id_strategy = st.one_of(
    st.none(),
    st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "P")),
        min_size=1,
        max_size=50,
    ).filter(lambda s: s.strip()),
)


class TestMineruClientErrorProperty:
    """
    Property-based tests for MineruClientError exception.
    
    **Property 7: Error Propagation**
    **Validates: Requirements 2.10, 3.5, 8.7**
    
    These tests verify that MineruClientError correctly stores and
    exposes error information for propagation to Task_Manager.
    """

    @settings(max_examples=100)
    @given(
        message=error_message_strategy,
    )
    def test_error_message_preserved_in_exception(
        self,
        message: str,
    ):
        """
        Property: For any error message, MineruClientError SHALL preserve
        the message in its message attribute.
        
        **Validates: Requirements 2.10, 3.5, 8.7**
        """
        error = MineruClientError(message=message)
        
        assert error.message == message, (
            f"Error message should be preserved. "
            f"Expected: {message!r}, Got: {error.message!r}"
        )
        # Also verify it's accessible via str()
        assert message in str(error), (
            f"Error message should be in string representation. "
            f"Message: {message!r}, str(error): {str(error)!r}"
        )

    @settings(max_examples=100)
    @given(
        message=error_message_strategy,
        error_code=error_code_strategy,
    )
    def test_error_code_preserved_in_exception(
        self,
        message: str,
        error_code: str,
    ):
        """
        Property: For any error code, MineruClientError SHALL preserve
        the error_code in its error_code attribute.
        
        **Validates: Requirements 2.10, 3.5, 8.7**
        """
        error = MineruClientError(message=message, error_code=error_code)
        
        assert error.error_code == error_code, (
            f"Error code should be preserved. "
            f"Expected: {error_code!r}, Got: {error.error_code!r}"
        )

    @settings(max_examples=100)
    @given(
        message=error_message_strategy,
        batch_id=batch_id_strategy,
    )
    def test_batch_id_preserved_in_exception(
        self,
        message: str,
        batch_id: Optional[str],
    ):
        """
        Property: For any batch_id (including None), MineruClientError SHALL
        preserve the batch_id in its batch_id attribute.
        
        **Validates: Requirements 2.10, 3.5, 8.7**
        """
        error = MineruClientError(message=message, batch_id=batch_id)
        
        assert error.batch_id == batch_id, (
            f"Batch ID should be preserved. "
            f"Expected: {batch_id!r}, Got: {error.batch_id!r}"
        )

    @settings(max_examples=100)
    @given(
        message=error_message_strategy,
        error_code=st.one_of(st.none(), error_code_strategy),
        batch_id=batch_id_strategy,
    )
    def test_all_error_attributes_preserved(
        self,
        message: str,
        error_code: Optional[str],
        batch_id: Optional[str],
    ):
        """
        Property: For any combination of message, error_code, and batch_id,
        MineruClientError SHALL preserve all attributes correctly.
        
        **Validates: Requirements 2.10, 3.5, 8.7**
        """
        error = MineruClientError(
            message=message,
            error_code=error_code,
            batch_id=batch_id,
        )
        
        assert error.message == message
        assert error.error_code == error_code
        assert error.batch_id == batch_id


class TestErrorCallbackProperty:
    """
    Property-based tests for error callback invocation.
    
    **Property 7: Error Propagation**
    **Validates: Requirements 2.10, 3.5, 8.7**
    
    These tests verify that the error callback is called with the
    correct error message when errors occur.
    """

    @settings(max_examples=100)
    @given(
        api_token=valid_api_token,
        batch_id=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N")),
            min_size=1,
            max_size=50,
        ).filter(lambda s: s.strip()),
        error_msg=error_message_strategy,
    )
    def test_error_callback_receives_error_message(
        self,
        api_token: str,
        batch_id: str,
        error_msg: str,
    ):
        """
        Property: For any error, the error callback SHALL be called with
        the error message.
        
        **Validates: Requirements 2.10, 3.5, 8.7**
        """
        callback_calls = []
        
        def error_callback(received_batch_id: str, received_error_msg: str):
            callback_calls.append((received_batch_id, received_error_msg))
        
        client = MineruClient(
            api_token=api_token,
            error_callback=error_callback,
        )
        
        # Directly call the internal _notify_error method
        client._notify_error(batch_id, error_msg)
        
        assert len(callback_calls) == 1, (
            f"Callback should be called exactly once. "
            f"Called {len(callback_calls)} times."
        )
        
        received_batch_id, received_error_msg = callback_calls[0]
        
        assert received_error_msg == error_msg, (
            f"Error message should be passed to callback. "
            f"Expected: {error_msg!r}, Got: {received_error_msg!r}"
        )
        assert received_batch_id == batch_id, (
            f"Batch ID should be passed to callback. "
            f"Expected: {batch_id!r}, Got: {received_batch_id!r}"
        )

    @settings(max_examples=100)
    @given(
        api_token=valid_api_token,
        error_msg=error_message_strategy,
    )
    def test_error_callback_receives_empty_batch_id_when_none(
        self,
        api_token: str,
        error_msg: str,
    ):
        """
        Property: When batch_id is None, the error callback SHALL receive
        an empty string for batch_id.
        
        **Validates: Requirements 2.10, 3.5, 8.7**
        """
        callback_calls = []
        
        def error_callback(received_batch_id: str, received_error_msg: str):
            callback_calls.append((received_batch_id, received_error_msg))
        
        client = MineruClient(
            api_token=api_token,
            error_callback=error_callback,
        )
        
        # Call with None batch_id
        client._notify_error(None, error_msg)
        
        assert len(callback_calls) == 1
        received_batch_id, received_error_msg = callback_calls[0]
        
        assert received_batch_id == "", (
            f"Batch ID should be empty string when None. "
            f"Got: {received_batch_id!r}"
        )
        assert received_error_msg == error_msg

    @settings(max_examples=100)
    @given(
        api_token=valid_api_token,
        error_msg=error_message_strategy,
    )
    def test_no_error_when_callback_not_configured(
        self,
        api_token: str,
        error_msg: str,
    ):
        """
        Property: When no error callback is configured, _notify_error
        SHALL not raise an exception.
        
        **Validates: Requirements 2.10, 3.5, 8.7**
        """
        client = MineruClient(
            api_token=api_token,
            error_callback=None,
        )
        
        # Should not raise any exception
        try:
            client._notify_error("batch-123", error_msg)
        except Exception as e:
            pytest.fail(
                f"_notify_error should not raise when callback is None. "
                f"Raised: {type(e).__name__}: {e}"
            )


class TestFileParseResultErrorProperty:
    """
    Property-based tests for error message preservation in FileParseResult.
    
    **Property 7: Error Propagation**
    **Validates: Requirements 2.10, 3.5, 8.7**
    
    These tests verify that for any failed task in batch results,
    the err_msg is preserved in FileParseResult.
    """

    @settings(max_examples=100)
    @given(
        file_name=file_name_strategy,
        err_msg=error_message_strategy,
    )
    def test_error_message_preserved_in_failed_result(
        self,
        file_name: str,
        err_msg: str,
    ):
        """
        Property: For any failed task, the err_msg SHALL be preserved
        in FileParseResult.
        
        **Validates: Requirements 2.10, 3.5, 8.7**
        """
        result = FileParseResult(
            file_name=file_name,
            state=MineruTaskState.FAILED,
            err_msg=err_msg,
        )
        
        assert result.err_msg == err_msg, (
            f"Error message should be preserved in FileParseResult. "
            f"Expected: {err_msg!r}, Got: {result.err_msg!r}"
        )
        assert result.state == MineruTaskState.FAILED, (
            f"State should be FAILED. Got: {result.state}"
        )

    @settings(max_examples=100)
    @given(
        file_name=file_name_strategy,
        data_id=data_id_strategy,
        err_msg=error_message_strategy,
    )
    def test_error_message_preserved_with_data_id(
        self,
        file_name: str,
        data_id: Optional[str],
        err_msg: str,
    ):
        """
        Property: For any failed task with any data_id, the err_msg
        SHALL be preserved in FileParseResult.
        
        **Validates: Requirements 2.10, 3.5, 8.7**
        """
        result = FileParseResult(
            file_name=file_name,
            data_id=data_id,
            state=MineruTaskState.FAILED,
            err_msg=err_msg,
        )
        
        assert result.err_msg == err_msg
        assert result.data_id == data_id
        assert result.state == MineruTaskState.FAILED

    @settings(max_examples=100)
    @given(
        file_name=file_name_strategy,
        err_msg=st.one_of(st.none(), error_message_strategy),
    )
    def test_error_message_can_be_none_or_string(
        self,
        file_name: str,
        err_msg: Optional[str],
    ):
        """
        Property: FileParseResult SHALL accept both None and string
        values for err_msg.
        
        **Validates: Requirements 2.10, 3.5, 8.7**
        """
        result = FileParseResult(
            file_name=file_name,
            state=MineruTaskState.FAILED if err_msg else MineruTaskState.DONE,
            err_msg=err_msg,
        )
        
        assert result.err_msg == err_msg, (
            f"Error message should be preserved (including None). "
            f"Expected: {err_msg!r}, Got: {result.err_msg!r}"
        )

    @settings(max_examples=100)
    @given(
        file_name=file_name_strategy,
        err_msg=error_message_strategy,
        extracted_pages=st.one_of(st.none(), st.integers(min_value=0, max_value=1000)),
        total_pages=st.one_of(st.none(), st.integers(min_value=1, max_value=1000)),
    )
    def test_error_message_preserved_with_progress_info(
        self,
        file_name: str,
        err_msg: str,
        extracted_pages: Optional[int],
        total_pages: Optional[int],
    ):
        """
        Property: For any failed task with progress information,
        the err_msg SHALL be preserved alongside progress data.
        
        **Validates: Requirements 2.10, 3.5, 8.7**
        """
        result = FileParseResult(
            file_name=file_name,
            state=MineruTaskState.FAILED,
            err_msg=err_msg,
            extracted_pages=extracted_pages,
            total_pages=total_pages,
        )
        
        assert result.err_msg == err_msg
        assert result.extracted_pages == extracted_pages
        assert result.total_pages == total_pages


class TestErrorPropagationConsistencyProperty:
    """
    Property-based tests for error propagation consistency.
    
    **Property 7: Error Propagation**
    **Validates: Requirements 2.10, 3.5, 8.7**
    
    These tests verify that error information is consistently
    propagated through the system.
    """

    @settings(max_examples=100)
    @given(
        api_token=valid_api_token,
        error_msg=error_message_strategy,
        num_notifications=st.integers(min_value=1, max_value=10),
    )
    def test_multiple_error_notifications_all_received(
        self,
        api_token: str,
        error_msg: str,
        num_notifications: int,
    ):
        """
        Property: For any number of error notifications, all SHALL be
        received by the callback.
        
        **Validates: Requirements 2.10, 3.5, 8.7**
        """
        callback_calls = []
        
        def error_callback(batch_id: str, msg: str):
            callback_calls.append((batch_id, msg))
        
        client = MineruClient(
            api_token=api_token,
            error_callback=error_callback,
        )
        
        # Send multiple notifications
        for i in range(num_notifications):
            client._notify_error(f"batch-{i}", f"{error_msg}-{i}")
        
        assert len(callback_calls) == num_notifications, (
            f"All {num_notifications} notifications should be received. "
            f"Got {len(callback_calls)} calls."
        )
        
        # Verify each notification was received correctly
        for i in range(num_notifications):
            batch_id, msg = callback_calls[i]
            assert batch_id == f"batch-{i}"
            assert msg == f"{error_msg}-{i}"

    @settings(max_examples=100)
    @given(
        api_token=valid_api_token,
        error_msgs=st.lists(
            error_message_strategy,
            min_size=2,
            max_size=5,
        ),
    )
    def test_different_error_messages_preserved_distinctly(
        self,
        api_token: str,
        error_msgs: list[str],
    ):
        """
        Property: For any list of different error messages, each SHALL
        be preserved distinctly in the callback.
        
        **Validates: Requirements 2.10, 3.5, 8.7**
        """
        callback_calls = []
        
        def error_callback(batch_id: str, msg: str):
            callback_calls.append(msg)
        
        client = MineruClient(
            api_token=api_token,
            error_callback=error_callback,
        )
        
        # Send different error messages
        for msg in error_msgs:
            client._notify_error("batch-123", msg)
        
        assert len(callback_calls) == len(error_msgs)
        
        # Verify each message was received in order
        for i, expected_msg in enumerate(error_msgs):
            assert callback_calls[i] == expected_msg, (
                f"Message {i} should be {expected_msg!r}, got {callback_calls[i]!r}"
            )

    @settings(max_examples=100)
    @given(
        file_names=st.lists(
            file_name_strategy,
            min_size=1,
            max_size=5,
        ),
        err_msgs=st.lists(
            error_message_strategy,
            min_size=1,
            max_size=5,
        ),
    )
    def test_multiple_failed_results_preserve_all_errors(
        self,
        file_names: list[str],
        err_msgs: list[str],
    ):
        """
        Property: For any batch with multiple failed files, each file's
        err_msg SHALL be preserved in its corresponding FileParseResult.
        
        **Validates: Requirements 2.10, 3.5, 8.7**
        """
        # Create results for each file with corresponding error
        results = []
        for i, file_name in enumerate(file_names):
            err_msg = err_msgs[i % len(err_msgs)]  # Cycle through error messages
            result = FileParseResult(
                file_name=file_name,
                state=MineruTaskState.FAILED,
                err_msg=err_msg,
            )
            results.append(result)
        
        # Verify each result preserves its error message
        for i, result in enumerate(results):
            expected_err = err_msgs[i % len(err_msgs)]
            assert result.err_msg == expected_err, (
                f"Result {i} should have err_msg {expected_err!r}, "
                f"got {result.err_msg!r}"
            )
            assert result.state == MineruTaskState.FAILED
