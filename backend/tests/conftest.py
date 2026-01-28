"""
Pytest configuration and shared fixtures for the Research Report Processor tests.
"""

import pytest
from hypothesis import settings

# Configure Hypothesis for property-based testing
# Each test should run at least 100 iterations as per design document
settings.register_profile("default", max_examples=100)
settings.register_profile("ci", max_examples=200)
settings.register_profile("dev", max_examples=50)
settings.load_profile("default")


@pytest.fixture
def sample_pdf_content() -> bytes:
    """Provide minimal valid PDF content for testing."""
    # Minimal PDF structure for testing purposes
    return b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"


@pytest.fixture
def temp_storage_path(tmp_path):
    """Provide a temporary storage path for file storage tests."""
    storage_path = tmp_path / "storage"
    storage_path.mkdir()
    return str(storage_path)
