"""
Image Preprocessor module for the Video Preprocessing Pipeline.

Applies center cropping and resizing to standard target dimensions
for incoming grayscale FrameRecords, ensuring inputs are ready for modeling.

Usage Example:
    from image_preprocessor import preprocess_frame
    
    processed_record = preprocess_frame(gray_record)
"""

import logging
import cv2
import numpy as np

from models.frame_record import FrameRecord
from exceptions import ImagePreprocessingError
from config import TARGET_SIZE

logger = logging.getLogger(__name__)

def center_crop(image: np.ndarray) -> np.ndarray:
    """
    Crops the center square from the image.
    
    Args:
        image: The 2D grayscale image.
        
    Returns:
        np.ndarray: The cropped center square.
        
    Raises:
        ImagePreprocessingError: If the image has zero area.
    """
    if image.size == 0:
        raise ImagePreprocessingError("Cannot crop an empty image.")
        
    h, w = image.shape
    min_dim = min(h, w)
    
    start_x = w // 2 - min_dim // 2
    start_y = h // 2 - min_dim // 2
    
    return image[start_y:start_y+min_dim, start_x:start_x+min_dim]

def preprocess_frame(record: FrameRecord) -> FrameRecord:
    """
    Takes a grayscale FrameRecord, applies center cropping, and resizes it to TARGET_SIZE.
    
    Args:
        record: The input grayscale FrameRecord.
        
    Returns:
        FrameRecord: A new FrameRecord containing the cropped, resized image.
        
    Raises:
        ImagePreprocessingError: If inputs are invalid or operations fail.
    """
    image = record.image
    
    if not isinstance(image, np.ndarray):
        raise ImagePreprocessingError("Input image must be a numpy.ndarray.")
        
    if image.ndim != 2:
        raise ImagePreprocessingError(f"Input image must be 2D grayscale. Got shape: {image.shape}")
        
    if image.dtype != np.uint8:
        raise ImagePreprocessingError(f"Input image must be uint8. Got: {image.dtype}")
        
    h, w = image.shape
    if h <= 0 or w <= 0:
        raise ImagePreprocessingError(f"Invalid image dimensions: {image.shape}")
        
    logger.debug("Preprocessing frame %s", record.frame_index)
    
    # Crop
    try:
        cropped = center_crop(image)
    except Exception as e:
        logger.error("Failed to center crop image: %s", str(e))
        if isinstance(e, ImagePreprocessingError):
            raise
        raise ImagePreprocessingError(f"Crop failure: {e}") from e
        
    # Smart interpolation: AREA for shrinking, LINEAR for enlarging
    target_w, target_h = TARGET_SIZE
    
    if cropped.shape[0] > target_h:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR
        
    # Resize
    try:
        resized = cv2.resize(cropped, TARGET_SIZE, interpolation=interpolation)
    except Exception as e:
        logger.error("Failed to resize image: %s", str(e))
        raise ImagePreprocessingError(f"Resize failure: {e}") from e
        
    # Post-condition
    if resized.shape != (target_h, target_w):
        raise ImagePreprocessingError(
            f"Resize result dimensions {resized.shape} do not match TARGET_SIZE {(target_h, target_w)}"
        )
        
    return FrameRecord(
        frame_index=record.frame_index,
        timestamp_seconds=record.timestamp_seconds,
        image=resized
    )
