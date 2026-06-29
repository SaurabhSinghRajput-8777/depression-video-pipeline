"""
Configuration module for the Video Preprocessing Pipeline.

This module centralizes all configuration parameters, ensuring a single
source of truth for dimensions, paths, and pipeline metadata.

Usage Example:
    from config import TARGET_SIZE, PIPELINE_VERSION
    import cv2
    
    def resize_frame(frame):
        return cv2.resize(frame, TARGET_SIZE)
"""

from pathlib import Path
from typing import Tuple, Set

__all__ = [
    "PIPELINE_PHASE",
    "PIPELINE_NAME",
    "PIPELINE_VERSION",
    "TARGET_WIDTH",
    "TARGET_HEIGHT",
    "TARGET_SIZE",
    "SUPPORTED_VIDEO_EXTENSIONS",
    "DEFAULT_DATA_DIR",
    "DEFAULT_RAW_DIR",
    "DEFAULT_PROCESSED_DIR",
    "DEFAULT_METADATA_DIR",
    "DEFAULT_LOGS_DIR",
    "DEFAULT_LOG_FILE"
]

# Pipeline Metadata
PIPELINE_PHASE: str = "phase_1"
PIPELINE_NAME: str = "video_preprocessing"
PIPELINE_VERSION: str = "1.0.0"

# Target Dimensions for standardized outputs
TARGET_WIDTH: int = 224
TARGET_HEIGHT: int = 224
TARGET_SIZE: Tuple[int, int] = (TARGET_WIDTH, TARGET_HEIGHT)

# Supported Formats
SUPPORTED_VIDEO_EXTENSIONS: Set[str] = {".mp4"}

# Default directory structures (can be overridden via CLI)
DEFAULT_DATA_DIR: Path = Path("data")
DEFAULT_RAW_DIR: Path = DEFAULT_DATA_DIR / "raw"
DEFAULT_PROCESSED_DIR: Path = DEFAULT_DATA_DIR / "processed"
DEFAULT_METADATA_DIR: Path = DEFAULT_DATA_DIR / "metadata"

# Logging
DEFAULT_LOGS_DIR: Path = Path("logs")
DEFAULT_LOG_FILE: Path = DEFAULT_LOGS_DIR / "processing.log"
