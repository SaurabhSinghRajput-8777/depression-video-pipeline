"""
Tests for the config module.
"""

from pathlib import Path
import sys

# Ensure src is in the python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config

def test_config_dimensions():
    """Verify target dimensions are integers, positive, and size tuple is correct."""
    assert isinstance(config.TARGET_WIDTH, int)
    assert isinstance(config.TARGET_HEIGHT, int)
    assert config.TARGET_WIDTH > 0
    assert config.TARGET_HEIGHT > 0
    assert config.TARGET_SIZE == (config.TARGET_WIDTH, config.TARGET_HEIGHT)

def test_config_metadata():
    """Verify pipeline metadata strings are correctly set."""
    assert config.PIPELINE_PHASE == "phase_1"
    assert config.PIPELINE_NAME == "video_preprocessing"
    assert isinstance(config.PIPELINE_VERSION, str)

def test_config_extensions():
    """Verify supported extensions is a set of strings."""
    assert isinstance(config.SUPPORTED_VIDEO_EXTENSIONS, set)
    assert ".mp4" in config.SUPPORTED_VIDEO_EXTENSIONS

def test_config_paths():
    """Verify default paths are Path objects."""
    assert isinstance(config.DEFAULT_RAW_DIR, Path)
    assert isinstance(config.DEFAULT_PROCESSED_DIR, Path)
    assert isinstance(config.DEFAULT_METADATA_DIR, Path)
    assert isinstance(config.DEFAULT_LOGS_DIR, Path)
    assert isinstance(config.DEFAULT_LOG_FILE, Path)
