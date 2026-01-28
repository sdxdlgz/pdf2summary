"""
Unit tests for the configuration management module.

Tests cover:
- Loading settings from environment variables
- Validation of required variables
- Default values for optional variables
- Error handling for missing/invalid variables
"""

import os
from unittest.mock import patch

import pytest

from backend.config import (
    ConfigurationError,
    Settings,
    get_settings,
    _format_validation_error,
)


class TestSettings:
    """Tests for the Settings class."""
    
    def test_settings_with_all_required_vars(self):
        """Test that Settings loads correctly when all required vars are set."""
        env_vars = {
            "MINERU_API_TOKEN": "test_token_123",
            "AI_API_ENDPOINT": "https://api.example.com/v1",
            "AI_API_KEY": "test_api_key_456",
            "STORAGE_PATH": "/tmp/storage",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()
            
            assert settings.MINERU_API_TOKEN == "test_token_123"
            assert settings.AI_API_ENDPOINT == "https://api.example.com/v1"
            assert settings.AI_API_KEY == "test_api_key_456"
            assert settings.STORAGE_PATH == "/tmp/storage"
    
    def test_settings_default_server_port(self):
        """Test that SERVER_PORT defaults to 8080 (Requirement 11.5)."""
        env_vars = {
            "MINERU_API_TOKEN": "test_token",
            "AI_API_ENDPOINT": "https://api.example.com",
            "AI_API_KEY": "test_key",
            "STORAGE_PATH": "/tmp/storage",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()
            assert settings.SERVER_PORT == 8080
    
    def test_settings_custom_server_port(self):
        """Test that SERVER_PORT can be customized."""
        env_vars = {
            "MINERU_API_TOKEN": "test_token",
            "AI_API_ENDPOINT": "https://api.example.com",
            "AI_API_KEY": "test_key",
            "STORAGE_PATH": "/tmp/storage",
            "SERVER_PORT": "3000",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()
            assert settings.SERVER_PORT == 3000
    
    def test_settings_optional_defaults(self):
        """Test that optional variables have correct defaults."""
        env_vars = {
            "MINERU_API_TOKEN": "test_token",
            "AI_API_ENDPOINT": "https://api.example.com",
            "AI_API_KEY": "test_key",
            "STORAGE_PATH": "/tmp/storage",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()
            
            assert settings.REDIS_URL == "redis://localhost:6379/0"
            assert settings.LOG_LEVEL == "INFO"
            assert settings.AI_MODEL == "gpt-5-nano"
            assert settings.AI_MAX_CONCURRENCY == 10
            assert settings.DOWNLOAD_LINK_EXPIRY == 86400
    
    def test_settings_custom_optional_values(self):
        """Test that optional variables can be customized."""
        env_vars = {
            "MINERU_API_TOKEN": "test_token",
            "AI_API_ENDPOINT": "https://api.example.com",
            "AI_API_KEY": "test_key",
            "STORAGE_PATH": "/tmp/storage",
            "REDIS_URL": "redis://redis:6379/1",
            "LOG_LEVEL": "DEBUG",
            "AI_MODEL": "gpt-4",
            "AI_MAX_CONCURRENCY": "20",
            "DOWNLOAD_LINK_EXPIRY": "3600",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()
            
            assert settings.REDIS_URL == "redis://redis:6379/1"
            assert settings.LOG_LEVEL == "DEBUG"
            assert settings.AI_MODEL == "gpt-4"
            assert settings.AI_MAX_CONCURRENCY == 20
            assert settings.DOWNLOAD_LINK_EXPIRY == 3600
    
    def test_settings_log_level_case_insensitive(self):
        """Test that LOG_LEVEL validation is case-insensitive."""
        env_vars = {
            "MINERU_API_TOKEN": "test_token",
            "AI_API_ENDPOINT": "https://api.example.com",
            "AI_API_KEY": "test_key",
            "STORAGE_PATH": "/tmp/storage",
            "LOG_LEVEL": "debug",  # lowercase
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            settings = Settings()
            assert settings.LOG_LEVEL == "DEBUG"  # Should be uppercased


class TestGetSettings:
    """Tests for the get_settings function."""
    
    def test_get_settings_success(self):
        """Test get_settings returns Settings when all required vars are set."""
        env_vars = {
            "MINERU_API_TOKEN": "test_token",
            "AI_API_ENDPOINT": "https://api.example.com",
            "AI_API_KEY": "test_key",
            "STORAGE_PATH": "/tmp/storage",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            settings = get_settings()
            assert isinstance(settings, Settings)
            assert settings.MINERU_API_TOKEN == "test_token"
    
    def test_get_settings_missing_mineru_token(self):
        """Test that missing MINERU_API_TOKEN raises ConfigurationError (Req 11.1, 11.6)."""
        env_vars = {
            "AI_API_ENDPOINT": "https://api.example.com",
            "AI_API_KEY": "test_key",
            "STORAGE_PATH": "/tmp/storage",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                get_settings()
            
            assert "MINERU_API_TOKEN" in exc_info.value.message
            assert "MINERU_API_TOKEN" in exc_info.value.missing_vars
    
    def test_get_settings_missing_ai_endpoint(self):
        """Test that missing AI_API_ENDPOINT raises ConfigurationError (Req 11.2, 11.6)."""
        env_vars = {
            "MINERU_API_TOKEN": "test_token",
            "AI_API_KEY": "test_key",
            "STORAGE_PATH": "/tmp/storage",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                get_settings()
            
            assert "AI_API_ENDPOINT" in exc_info.value.message
            assert "AI_API_ENDPOINT" in exc_info.value.missing_vars
    
    def test_get_settings_missing_ai_key(self):
        """Test that missing AI_API_KEY raises ConfigurationError (Req 11.3, 11.6)."""
        env_vars = {
            "MINERU_API_TOKEN": "test_token",
            "AI_API_ENDPOINT": "https://api.example.com",
            "STORAGE_PATH": "/tmp/storage",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                get_settings()
            
            assert "AI_API_KEY" in exc_info.value.message
            assert "AI_API_KEY" in exc_info.value.missing_vars
    
    def test_get_settings_missing_storage_path(self):
        """Test that missing STORAGE_PATH raises ConfigurationError (Req 11.4, 11.6)."""
        env_vars = {
            "MINERU_API_TOKEN": "test_token",
            "AI_API_ENDPOINT": "https://api.example.com",
            "AI_API_KEY": "test_key",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                get_settings()
            
            assert "STORAGE_PATH" in exc_info.value.message
            assert "STORAGE_PATH" in exc_info.value.missing_vars
    
    def test_get_settings_missing_multiple_vars(self):
        """Test that multiple missing vars are all reported (Req 11.6)."""
        env_vars = {}  # All required vars missing
        
        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                get_settings()
            
            error = exc_info.value
            assert "MINERU_API_TOKEN" in error.missing_vars
            assert "AI_API_ENDPOINT" in error.missing_vars
            assert "AI_API_KEY" in error.missing_vars
            assert "STORAGE_PATH" in error.missing_vars
            assert len(error.missing_vars) == 4
    
    def test_get_settings_descriptive_error_message(self):
        """Test that error message is descriptive (Req 11.6)."""
        env_vars = {
            "AI_API_ENDPOINT": "https://api.example.com",
            "AI_API_KEY": "test_key",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                get_settings()
            
            error_msg = exc_info.value.message
            assert "Failed to start" in error_msg
            assert "Missing required environment variables" in error_msg
    
    def test_get_settings_empty_string_treated_as_missing(self):
        """Test that empty strings for required vars are rejected."""
        env_vars = {
            "MINERU_API_TOKEN": "",  # Empty string
            "AI_API_ENDPOINT": "https://api.example.com",
            "AI_API_KEY": "test_key",
            "STORAGE_PATH": "/tmp/storage",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                get_settings()
            
            # Empty string should fail min_length validation
            assert "MINERU_API_TOKEN" in exc_info.value.message
    
    def test_get_settings_invalid_server_port(self):
        """Test that invalid SERVER_PORT raises ConfigurationError."""
        env_vars = {
            "MINERU_API_TOKEN": "test_token",
            "AI_API_ENDPOINT": "https://api.example.com",
            "AI_API_KEY": "test_key",
            "STORAGE_PATH": "/tmp/storage",
            "SERVER_PORT": "99999",  # Invalid port (> 65535)
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                get_settings()
            
            assert "SERVER_PORT" in exc_info.value.message
    
    def test_get_settings_invalid_log_level(self):
        """Test that invalid LOG_LEVEL raises ConfigurationError."""
        env_vars = {
            "MINERU_API_TOKEN": "test_token",
            "AI_API_ENDPOINT": "https://api.example.com",
            "AI_API_KEY": "test_key",
            "STORAGE_PATH": "/tmp/storage",
            "LOG_LEVEL": "INVALID",
        }
        
        with patch.dict(os.environ, env_vars, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                get_settings()
            
            assert "LOG_LEVEL" in exc_info.value.message


class TestConfigurationError:
    """Tests for the ConfigurationError exception."""
    
    def test_configuration_error_message(self):
        """Test ConfigurationError stores message correctly."""
        error = ConfigurationError("Test error message")
        assert error.message == "Test error message"
        assert str(error) == "Test error message"
    
    def test_configuration_error_missing_vars(self):
        """Test ConfigurationError stores missing_vars correctly."""
        error = ConfigurationError(
            "Missing vars",
            missing_vars=["VAR1", "VAR2"]
        )
        assert error.missing_vars == ["VAR1", "VAR2"]
    
    def test_configuration_error_default_missing_vars(self):
        """Test ConfigurationError defaults missing_vars to empty list."""
        error = ConfigurationError("Test error")
        assert error.missing_vars == []
