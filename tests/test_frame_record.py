"""
Tests for the FrameRecord data model.
"""

import sys
from pathlib import Path

# Ensure src is in the python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pytest
from models.frame_record import FrameRecord

def test_frame_record_creation():
    """Verify FrameRecord can be instantiated correctly."""
    mock_image = np.zeros((720, 1280, 3), dtype=np.uint8)
    record = FrameRecord(
        frame_index=10,
        timestamp_seconds=0.33,
        image=mock_image
    )
    
    assert record.frame_index == 10
    assert record.timestamp_seconds == 0.33
    assert np.array_equal(record.image, mock_image)

def test_frame_record_shape_property():
    """Verify the shape property returns the image shape."""
    mock_image = np.ones((224, 224), dtype=np.uint8)
    record = FrameRecord(
        frame_index=0,
        timestamp_seconds=0.0,
        image=mock_image
    )
    
    assert record.shape == (224, 224)

def test_frame_record_slots():
    """Verify that slots=True prevents dynamic attribute assignment."""
    record = FrameRecord(0, 0.0, np.zeros((10, 10)))
    
    with pytest.raises(AttributeError):
        record.new_attribute = "test"

def test_frame_record_validation():
    """Verify that __post_init__ correctly validates inputs."""
    valid_image = np.zeros((10, 10))
    
    with pytest.raises(ValueError, match="frame_index must be >= 0"):
        FrameRecord(-1, 0.0, valid_image)
        
    with pytest.raises(ValueError, match="timestamp_seconds must be >= 0"):
        FrameRecord(0, -1.0, valid_image)
        
    with pytest.raises(TypeError, match="image must be a numpy.ndarray"):
        FrameRecord(0, 0.0, "not_an_image")
