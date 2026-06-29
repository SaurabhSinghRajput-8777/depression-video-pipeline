"""
Grayscale Converter module for the Video Preprocessing Pipeline.

Transforms a multi-channel FrameRecord into a single-channel grayscale FrameRecord,
preserving metadata like timestamp and frame index.

Usage Example:
    from grayscale_converter import convert_to_grayscale
    
    gray_record = convert_to_grayscale(record)
"""

import logging
import cv2
import numpy as np

from models.frame_record import FrameRecord
from exceptions import GrayscaleConversionError

logger = logging.getLogger(__name__)

def convert_to_grayscale(record: FrameRecord) -> FrameRecord:
    """
    Converts a FrameRecord containing a multi-channel image to grayscale.
    
    Args:
        record: The input FrameRecord.
        
    Returns:
        FrameRecord: A new FrameRecord containing the grayscale image and
                     original metadata.
                     
    Raises:
        GrayscaleConversionError: If the input image is invalid or has wrong dimensions.
    """
    image = record.image
    
    if not isinstance(image, np.ndarray):
        raise GrayscaleConversionError("Input image must be a numpy.ndarray.")
        
    # Check if already grayscale (2D)
    if image.ndim == 2:
        raise GrayscaleConversionError("Input image is already single-channel.")
        
    if image.ndim != 3 or image.shape[2] != 3:
        raise GrayscaleConversionError(f"Expected 3-channel (HxWx3) image. Got shape: {image.shape}")
        
    logger.debug("Converting frame %s to grayscale", record.frame_index)
    
    # Assume BGR since it comes from OpenCV
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        logger.error("Failed to convert image to grayscale: %s", str(e))
        raise GrayscaleConversionError(f"OpenCV conversion failed: {e}") from e
        
    if gray_image.ndim != 2:
        raise GrayscaleConversionError(
            f"Grayscale conversion produced invalid output shape: {gray_image.shape}"
        )
        
    return FrameRecord(
        frame_index=record.frame_index,
        timestamp_seconds=record.timestamp_seconds,
        image=gray_image
    )
