import os
import pytest
from pathlib import Path


@pytest.fixture(autouse=True)
def _set_test_env(monkeypatch, tmp_path):
    """Set minimal env vars for all tests. Use tmp_path for output."""
    monkeypatch.setenv("SCALE", "0.001")
    monkeypatch.setenv("SEED", "42")
    monkeypatch.setenv("HISTORY_START", "2023-01-01")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
