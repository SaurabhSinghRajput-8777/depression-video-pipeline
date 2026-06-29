"""
Tests for the image_preprocessor module.
"""

import sys
from pathlib import Path
import numpy as np

# Ensure src is in the python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import cv2

from image_preprocessor import preprocess_frame, center_crop
from models.frame_record import FrameRecord
from exceptions import ImagePreprocessingError
from config import TARGET_SIZE

def test_center_crop_landscape():
    """Verify cropping of a landscape image (wider than tall)."""
    image = np.zeros((100, 200), dtype=np.uint8)
    image[0:100, 50:150] = 1
    
    cropped = center_crop(image)
    assert cropped.shape == (100, 100)
    assert np.all(cropped == 1)

def test_center_crop_portrait():
    """Verify cropping of a portrait image (taller than wide)."""
    image = np.zeros((200, 100), dtype=np.uint8)
    image[50:150, 0:100] = 1
    
    cropped = center_crop(image)
    assert cropped.shape == (100, 100)
    assert np.all(cropped == 1)

def test_center_crop_empty():
    """Verify cropping an empty image raises ImagePreprocessingError."""
    image = np.zeros((0, 0), dtype=np.uint8)
    with pytest.raises(ImagePreprocessingError, match="empty image"):
        center_crop(image)

def test_preprocess_frame_valid():
    """Verify a valid grayscale frame is cropped and resized correctly."""
    # 720p resolution
    image = np.ones((720, 1280), dtype=np.uint8)
    
    record = FrameRecord(frame_index=1, timestamp_seconds=0.1, image=image)
    
    processed = preprocess_frame(record)
    
    assert processed is not record
    assert processed.frame_index == 1
    assert processed.timestamp_seconds == 0.1
    
    target_w, target_h = TARGET_SIZE
    assert processed.shape == (target_h, target_w)

def test_preprocess_frame_tiny():
    """Verify preprocessing handles a tiny (1, 1) image correctly."""
    image = np.ones((1, 1), dtype=np.uint8)
    record = FrameRecord(frame_index=0, timestamp_seconds=0.0, image=image)
    
    processed = preprocess_frame(record)
    
    target_w, target_h = TARGET_SIZE
    assert processed.shape == (target_h, target_w)
    
def test_preprocess_frame_requires_2d():
    """Verify preprocessing fails for a 3-channel image."""
    image = np.ones((720, 1280, 3), dtype=np.uint8)
    record = FrameRecord(frame_index=1, timestamp_seconds=0.1, image=image)
    
    with pytest.raises(ImagePreprocessingError, match="must be 2D grayscale"):
        preprocess_frame(record)

def test_preprocess_frame_requires_uint8():
    """Verify preprocessing fails for non-uint8 dtype."""
    image = np.ones((100, 100), dtype=np.float32)
    record = FrameRecord(frame_index=1, timestamp_seconds=0.1, image=image)
    
    with pytest.raises(ImagePreprocessingError, match="must be uint8"):
        preprocess_frame(record)

def test_preprocess_frame_invalid_dimensions():
    """Verify preprocessing fails for 0-dimension images."""
    image = np.zeros((0, 0), dtype=np.uint8)
    record = FrameRecord(frame_index=1, timestamp_seconds=0.1, image=image)
    
    with pytest.raises(ImagePreprocessingError, match="Invalid image dimensions"):
        preprocess_frame(record)

def test_preprocess_frame_not_numpy():
    """Verify preprocessing fails if image is not a numpy array."""
    from unittest.mock import MagicMock
    record = MagicMock()
    record.image = "not_an_array"
    record.frame_index = 1
    record.timestamp_seconds = 0.1
    
    with pytest.raises(ImagePreprocessingError, match="must be a numpy.ndarray"):
        preprocess_frame(record)
