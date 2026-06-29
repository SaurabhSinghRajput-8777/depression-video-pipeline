"""
FPS Extractor module for the Video Preprocessing Pipeline.

Extracts and strictly validates the Frames Per Second (FPS) metric from an 
open OpenCV VideoCapture object.

Usage Example:
    from video_reader import read_video
    from fps_extractor import extract_fps
    
    cap = read_video("video.mp4")
    fps = extract_fps(cap)
"""

import logging
import cv2
import math

from exceptions import InvalidFPSError

logger = logging.getLogger(__name__)

def extract_fps(cap: cv2.VideoCapture) -> float:
    """
    Extracts and validates the FPS from an OpenCV VideoCapture object.
    
    Args:
        cap: The opened video capture object.
        
    Returns:
        float: The validated frames per second.
        
    Raises:
        InvalidFPSError: If the FPS is <= 0, infinite, NaN, or otherwise invalid.
    """
    # Get FPS property
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Validate NaN
    if math.isnan(fps):
        logger.error("Detected NaN FPS value.")
        raise InvalidFPSError("Extracted FPS is NaN.")
        
    # Validate Infinity
    if not math.isfinite(fps):
        logger.error("Detected infinite FPS value: %s", fps)
        raise InvalidFPSError(f"Extracted FPS is infinite: {fps}")
        
    # Validate positive
    if fps <= 0:
        logger.error("Detected zero or negative FPS value: %s", fps)
        raise InvalidFPSError(f"Extracted FPS must be > 0. Got: {fps}")
        
    logger.info("Successfully extracted FPS: %s", fps)
    return float(fps)
