"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def temp_config_dir(tmp_path):
    """Create a temporary directory for test configs."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    return config_dir
