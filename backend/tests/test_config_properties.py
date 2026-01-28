"""
Property-based tests for configuration validation.

**Property 14: Environment Variable Validation**
*For any* startup with missing required environment variables (MINERU_API_TOKEN,
AI_API_ENDPOINT, AI_API_KEY), the application SHALL fail to start and SHALL
output an error message specifying which variable is missing.

**Validates: Requirements 11.1, 11.2, 11.3, 11.6**

Uses Hypothesis for property-based testing with at least 100 iterations per test.
"""

import os
from itertools import combinations
from unittest.mock import patch

import pytest
from hypothesis import given, settings, strategies as st, assume

from backend.config import (
    ConfigurationError,
    Settings,
    get_settings,
)


# Required environment variables that must be validated
# Note: STORAGE_PATH is also required but not part of Property 14's scope
REQUIRED_VARS = ["MINERU_API_TOKEN", "AI_API_ENDPOINT", "AI_API_KEY"]

# All required variables including STORAGE_PATH
ALL_REQUIRED_VARS = ["MINERU_API_TOKEN", "AI_API_ENDPOINT", "AI_API_KEY", "STORAGE_PATH"]


# Strategy for generating non-empty valid string values for environment variables
valid_env_value = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N", "P", "S"),
        blacklist_characters="\x00",
    ),
    min_size=1,
    max_size=100,
).filter(lambda s: s.strip())  # Ensure non-whitespace-only strings


# Strategy for generating subsets of required variables to be missing
missing_vars_subset = st.lists(
    st.sampled_from(REQUIRED_VARS),
    min_size=1,
    max_size=len(REQUIRED_VARS),
    unique=True,
)


class TestEnvironmentVariableValidationProperty:
    """
    Property-based tests for Property 14: Environment Variable Validation.
    
    **Validates: Requirements 11.1, 11.2, 11.3, 11.6**
    
    These tests verify that:
    1. Missing required variables cause startup failure
    2. Error messages specify which variables are missing
    3. All required variables present allows successful startup
    """

    @settings(max_examples=100)
    @given(
        missing_vars=missing_vars_subset,
        mineru_token=valid_env_value,
        ai_endpoint=valid_env_value,
        ai_key=valid_env_value,
        storage_path=valid_env_value,
    )
    def test_missing_required_vars_causes_failure_with_specific_error(
        self,
        missing_vars: list[str],
        mineru_token: str,
        ai_endpoint: str,
        ai_key: str,
        storage_path: str,
    ):
        """
        Property: For any subset of missing required variables, the system
        SHALL fail to start and SHALL output an error message specifying
        which variables are missing.
        
        **Validates: Requirements 11.1, 11.2, 11.3, 11.6**
        """
        # Build environment with all variables present
        all_vars = {
            "MINERU_API_TOKEN": mineru_token,
            "AI_API_ENDPOINT": ai_endpoint,
            "AI_API_KEY": ai_key,
            "STORAGE_PATH": storage_path,
        }
        
        # Remove the variables that should be missing
        env_vars = {k: v for k, v in all_vars.items() if k not in missing_vars}
        
        with patch.dict(os.environ, env_vars, clear=True):
            # System SHALL fail to start
            with pytest.raises(ConfigurationError) as exc_info:
                get_settings()
            
            error = exc_info.value
            
            # Error message SHALL specify which variables are missing
            for missing_var in missing_vars:
                assert missing_var in error.message, (
                    f"Error message should mention missing variable '{missing_var}'. "
                    f"Got: {error.message}"
                )
                assert missing_var in error.missing_vars, (
                    f"missing_vars should contain '{missing_var}'. "
                    f"Got: {error.missing_vars}"
                )

    @settings(max_examples=100)
    @given(
        mineru_token=valid_env_value,
        ai_endpoint=valid_env_value,
        ai_key=valid_env_value,
        storage_path=valid_env_value,
    )
    def test_all_required_vars_present_allows_startup(
        self,
        mineru_token: str,
        ai_endpoint: str,
        ai_key: str,
        storage_path: str,
    ):
        """
        Property: When all required variables are present with any valid
        string values, the system SHALL start successfully.
        
        **Validates: Requirements 11.1, 11.2, 11.3, 11.6**
        """
        env_vars = {
            "MINERU_API_TOKEN": mineru_token,
            "AI_API_ENDPOINT": ai_endpoint,
            "AI_API_KEY": ai_key,
            "STORAGE_PATH": storage_path,
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            # System SHALL start successfully
            settings = get_settings()
            
            # Verify all values are correctly loaded
            assert settings.MINERU_API_TOKEN == mineru_token
            assert settings.AI_API_ENDPOINT == ai_endpoint
            assert settings.AI_API_KEY == ai_key
            assert settings.STORAGE_PATH == storage_path

    @settings(max_examples=100)
    @given(
        present_vars=st.lists(
            st.sampled_from(REQUIRED_VARS),
            min_size=0,
            max_size=len(REQUIRED_VARS) - 1,
            unique=True,
        ),
        mineru_token=valid_env_value,
        ai_endpoint=valid_env_value,
        ai_key=valid_env_value,
        storage_path=valid_env_value,
    )
    def test_partial_required_vars_causes_failure(
        self,
        present_vars: list[str],
        mineru_token: str,
        ai_endpoint: str,
        ai_key: str,
        storage_path: str,
    ):
        """
        Property: For any proper subset of required variables (not all present),
        the system SHALL fail to start.
        
        **Validates: Requirements 11.1, 11.2, 11.3, 11.6**
        """
        # Build environment with only the present variables
        all_vars = {
            "MINERU_API_TOKEN": mineru_token,
            "AI_API_ENDPOINT": ai_endpoint,
            "AI_API_KEY": ai_key,
            "STORAGE_PATH": storage_path,
        }
        
        # Only include variables that are in present_vars, plus STORAGE_PATH
        # (STORAGE_PATH is always included since it's required but not in REQUIRED_VARS scope)
        env_vars = {"STORAGE_PATH": storage_path}
        for var in present_vars:
            env_vars[var] = all_vars[var]
        
        # Calculate which required vars are missing
        missing_vars = [v for v in REQUIRED_VARS if v not in present_vars]
        
        # Skip if all required vars happen to be present
        assume(len(missing_vars) > 0)
        
        with patch.dict(os.environ, env_vars, clear=True):
            # System SHALL fail to start
            with pytest.raises(ConfigurationError) as exc_info:
                get_settings()
            
            error = exc_info.value
            
            # All missing variables should be reported
            for missing_var in missing_vars:
                assert missing_var in error.missing_vars, (
                    f"missing_vars should contain '{missing_var}'. "
                    f"Got: {error.missing_vars}"
                )

    @settings(max_examples=100)
    @given(
        missing_var=st.sampled_from(REQUIRED_VARS),
        mineru_token=valid_env_value,
        ai_endpoint=valid_env_value,
        ai_key=valid_env_value,
        storage_path=valid_env_value,
    )
    def test_single_missing_var_error_message_is_descriptive(
        self,
        missing_var: str,
        mineru_token: str,
        ai_endpoint: str,
        ai_key: str,
        storage_path: str,
    ):
        """
        Property: For any single missing required variable, the error message
        SHALL be descriptive and SHALL specify the missing variable name.
        
        **Validates: Requirements 11.6**
        """
        all_vars = {
            "MINERU_API_TOKEN": mineru_token,
            "AI_API_ENDPOINT": ai_endpoint,
            "AI_API_KEY": ai_key,
            "STORAGE_PATH": storage_path,
        }
        
        # Remove the single missing variable
        env_vars = {k: v for k, v in all_vars.items() if k != missing_var}
        
        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                get_settings()
            
            error = exc_info.value
            
            # Error message SHALL be descriptive
            assert "Failed to start" in error.message, (
                f"Error message should indicate startup failure. Got: {error.message}"
            )
            assert "Missing required environment variables" in error.message, (
                f"Error message should mention missing variables. Got: {error.message}"
            )
            # Error message SHALL specify the missing variable
            assert missing_var in error.message, (
                f"Error message should specify '{missing_var}'. Got: {error.message}"
            )


class TestEnvironmentVariableValidationExhaustive:
    """
    Exhaustive tests for all combinations of missing required variables.
    
    **Validates: Requirements 11.1, 11.2, 11.3, 11.6**
    
    These tests complement the property-based tests by ensuring all
    possible combinations are covered.
    """

    @pytest.mark.parametrize(
        "missing_vars",
        [
            # Single missing variable
            ["MINERU_API_TOKEN"],
            ["AI_API_ENDPOINT"],
            ["AI_API_KEY"],
            # Two missing variables
            ["MINERU_API_TOKEN", "AI_API_ENDPOINT"],
            ["MINERU_API_TOKEN", "AI_API_KEY"],
            ["AI_API_ENDPOINT", "AI_API_KEY"],
            # All three missing
            ["MINERU_API_TOKEN", "AI_API_ENDPOINT", "AI_API_KEY"],
        ],
    )
    def test_all_missing_combinations_report_correct_vars(self, missing_vars: list[str]):
        """
        Test that all combinations of missing variables are correctly reported.
        
        **Validates: Requirements 11.1, 11.2, 11.3, 11.6**
        """
        all_vars = {
            "MINERU_API_TOKEN": "test_token",
            "AI_API_ENDPOINT": "https://api.example.com",
            "AI_API_KEY": "test_key",
            "STORAGE_PATH": "/tmp/storage",
        }
        
        env_vars = {k: v for k, v in all_vars.items() if k not in missing_vars}
        
        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                get_settings()
            
            error = exc_info.value
            
            # All missing variables should be in the error
            for var in missing_vars:
                assert var in error.missing_vars
                assert var in error.message
            
            # No extra variables should be reported as missing
            for var in ["MINERU_API_TOKEN", "AI_API_ENDPOINT", "AI_API_KEY"]:
                if var not in missing_vars:
                    assert var not in error.missing_vars
