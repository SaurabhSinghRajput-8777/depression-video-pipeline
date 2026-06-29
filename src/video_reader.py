"""
Video Reader module for the Video Preprocessing Pipeline.

Handles the safe opening and validation of video files, ensuring that only
supported, existing files are passed further down the pipeline.

Usage Example:
    from video_reader import read_video
    from pathlib import Path
    
    cap = read_video(Path("data/raw/sample.mp4"))
"""

import logging
import cv2
from pathlib import Path
from typing import Union

from config import SUPPORTED_VIDEO_EXTENSIONS
from exceptions import (
    VideoNotFoundError,
    InvalidVideoFormatError,
    VideoOpenError,
)

logger = logging.getLogger(__name__)

def read_video(video_path: Union[str, Path]) -> cv2.VideoCapture:
    """
    Safely opens a video file and returns an OpenCV VideoCapture object.
    
    Args:
        video_path: Path to the input video file.
        
    Returns:
        cv2.VideoCapture: The opened video capture object.
        
    Raises:
        VideoNotFoundError: If the file does not exist or is not a file.
        InvalidVideoFormatError: If the file extension is not supported.
        VideoOpenError: If OpenCV fails to open the video.
    """
    path = Path(video_path)
    
    logger.info("Attempting to read video: %s", path)
    
    if not path.exists() or not path.is_file():
        logger.error("Video file not found or is not a file: %s", path)
        raise VideoNotFoundError(f"Cannot find video at {path}")
        
    if path.suffix.lower() not in SUPPORTED_VIDEO_EXTENSIONS:
        logger.error("Unsupported video format: %s", path.suffix)
        raise InvalidVideoFormatError(
            f"Extension {path.suffix} not supported. "
            f"Supported extensions: {SUPPORTED_VIDEO_EXTENSIONS}"
        )
        
    # Open the video
    cap = cv2.VideoCapture(str(path))
    
    if not cap.isOpened():
        logger.error("OpenCV failed to open video: %s", path)
        cap.release()
        raise VideoOpenError(f"Failed to open video file: {path}")
        
    logger.info("Successfully opened video: %s", path)
    return cap
