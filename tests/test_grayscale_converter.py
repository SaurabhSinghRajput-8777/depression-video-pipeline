"""
Tests for the grayscale_converter module.
"""

import sys
from pathlib import Path
import numpy as np

# Ensure src is in the python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from grayscale_converter import convert_to_grayscale
from models.frame_record import FrameRecord
from exceptions import GrayscaleConversionError

def test_convert_to_grayscale_valid_bgr():
    """Verify standard 3-channel BGR image is converted properly."""
    # Create a simple 10x10 RGB image (all ones)
    bgr_image = np.ones((10, 10, 3), dtype=np.uint8)
    
    input_record = FrameRecord(
        frame_index=5,
        timestamp_seconds=1.5,
        image=bgr_image
    )
    
    output_record = convert_to_grayscale(input_record)
    
    assert output_record is not input_record
    assert output_record.frame_index == 5
    assert output_record.timestamp_seconds == 1.5
    
    assert output_record.image.ndim == 2
    assert output_record.shape == (10, 10)

def test_convert_to_grayscale_single_channel_fails():
    """Verify that passing an already single-channel image raises GrayscaleConversionError."""
    gray_image = np.ones((10, 10), dtype=np.uint8)
    
    input_record = FrameRecord(
        frame_index=0,
        timestamp_seconds=0.0,
        image=gray_image
    )
    
    with pytest.raises(GrayscaleConversionError, match="already single-channel"):
        convert_to_grayscale(input_record)

def test_convert_to_grayscale_invalid_dimensions_fails():
    """Verify that passing a 4D array or 1D array raises GrayscaleConversionError."""
    invalid_image = np.ones((10, 10, 3, 2), dtype=np.uint8)
    
    input_record = FrameRecord(
        frame_index=0,
        timestamp_seconds=0.0,
        image=invalid_image
    )
    
    with pytest.raises(GrayscaleConversionError, match="Expected 3-channel"):
        convert_to_grayscale(input_record)

def test_convert_to_grayscale_invalid_channels_fails():
    """Verify that passing a 4-channel image (e.g. RGBA) raises GrayscaleConversionError."""
    rgba_image = np.ones((10, 10, 4), dtype=np.uint8)
    
    input_record = FrameRecord(
        frame_index=0,
        timestamp_seconds=0.0,
        image=rgba_image
    )
    
    with pytest.raises(GrayscaleConversionError, match="Expected 3-channel"):
        convert_to_grayscale(input_record)
