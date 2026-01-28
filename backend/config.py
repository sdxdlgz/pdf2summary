"""
Configuration management module for Research Report Processor.

Uses pydantic-settings to read environment variables and validates
that all required variables are present at startup.

Requirements:
- 11.1: Read MinerU API token from MINERU_API_TOKEN
- 11.2: Read AI service endpoint from AI_API_ENDPOINT
- 11.3: Read AI service API key from AI_API_KEY
- 11.4: Read file storage path from STORAGE_PATH
- 11.5: Read server port from SERVER_PORT with default 8765
- 11.6: Fail to start with descriptive error if required variable is missing
"""

from functools import lru_cache
from typing import Optional

from pydantic import Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Required variables (no defaults - must be set):
    - MINERU_API_TOKEN: MinerU API authentication token
    - AI_API_ENDPOINT: OpenAI-compatible API endpoint URL
    - AI_API_KEY: API key for AI service
    - STORAGE_PATH: Path for file storage
    
    Optional variables (have defaults):
    - SERVER_PORT: HTTP server port (default: 8765)
    - REDIS_URL: Redis connection URL
    - LOG_LEVEL: Logging level
    - AI_MODEL: AI model name
    - AI_MAX_CONCURRENCY: Max concurrent AI requests
    - DOWNLOAD_LINK_EXPIRY: Download link expiration in seconds
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )
    
    # Required environment variables (Requirement 11.1, 11.2, 11.3, 11.4)
    # These have no defaults and will cause validation errors if missing
    MINERU_API_TOKEN: str = Field(
        ...,
        description="MinerU API authentication token",
        min_length=1,
    )
    AI_API_ENDPOINT: str = Field(
        ...,
        description="OpenAI-compatible API endpoint URL",
        min_length=1,
    )
    AI_API_KEY: str = Field(
        ...,
        description="API key for AI service",
        min_length=1,
    )
    STORAGE_PATH: str = Field(
        ...,
        description="Path for file storage",
        min_length=1,
    )
    
    # Optional environment variables with defaults (Requirement 11.5)
    SERVER_PORT: int = Field(
        default=8765,
        description="HTTP server port",
        ge=1,
        le=65535,
    )
    
    # Additional optional configuration
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    AI_MODEL: str = Field(
        default="gpt-5-nano",
        description="AI model name for translation and summarization",
    )
    AI_MAX_CONCURRENCY: int = Field(
        default=10,
        description="Maximum concurrent AI requests",
        ge=1,
        le=100,
    )
    DOWNLOAD_LINK_EXPIRY: int = Field(
        default=86400,
        description="Download link expiration time in seconds (default: 24 hours)",
        ge=60,
    )
    
    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate that log level is one of the allowed values."""
        allowed_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in allowed_levels:
            raise ValueError(
                f"LOG_LEVEL must be one of {allowed_levels}, got '{v}'"
            )
        return upper_v


class ConfigurationError(Exception):
    """
    Exception raised when configuration validation fails.
    
    This exception provides a descriptive error message indicating
    which environment variables are missing or invalid.
    """
    
    def __init__(self, message: str, missing_vars: Optional[list[str]] = None):
        self.message = message
        self.missing_vars = missing_vars or []
        super().__init__(self.message)


def _format_validation_error(error: ValidationError) -> str:
    """
    Format a Pydantic ValidationError into a user-friendly message.
    
    Args:
        error: The ValidationError from Pydantic
        
    Returns:
        A formatted error message listing all missing/invalid variables
    """
    missing_vars = []
    invalid_vars = []
    
    for err in error.errors():
        field_name = err["loc"][0] if err["loc"] else "unknown"
        error_type = err["type"]
        
        if error_type == "missing":
            missing_vars.append(str(field_name))
        else:
            msg = err.get("msg", "invalid value")
            invalid_vars.append(f"{field_name}: {msg}")
    
    parts = []
    
    if missing_vars:
        parts.append(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )
    
    if invalid_vars:
        parts.append(
            f"Invalid environment variables: {'; '.join(invalid_vars)}"
        )
    
    return ". ".join(parts)


def get_settings() -> Settings:
    """
    Create and validate application settings from environment variables.
    
    This function attempts to load settings from environment variables
    and raises a ConfigurationError with a descriptive message if any
    required variables are missing or invalid.
    
    Returns:
        Settings: Validated application settings
        
    Raises:
        ConfigurationError: If required environment variables are missing
            or if any variable has an invalid value. The error message
            will specify which variables are problematic.
            
    Example:
        >>> try:
        ...     settings = get_settings()
        ...     print(f"Server running on port {settings.SERVER_PORT}")
        ... except ConfigurationError as e:
        ...     print(f"Configuration error: {e.message}")
        ...     sys.exit(1)
    """
    try:
        return Settings()
    except ValidationError as e:
        error_message = _format_validation_error(e)
        
        # Extract missing variable names for the exception
        missing_vars = [
            str(err["loc"][0])
            for err in e.errors()
            if err["type"] == "missing" and err["loc"]
        ]
        
        raise ConfigurationError(
            message=f"Failed to start: {error_message}",
            missing_vars=missing_vars,
        ) from e


@lru_cache
def get_cached_settings() -> Settings:
    """
    Get cached application settings (singleton pattern).
    
    This function caches the settings after the first successful load,
    which is useful for dependency injection in FastAPI.
    
    Returns:
        Settings: Validated and cached application settings
        
    Raises:
        ConfigurationError: If configuration validation fails
    """
    return get_settings()
