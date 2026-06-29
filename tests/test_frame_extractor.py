"""
Tests for the frame_extractor module.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np

# Ensure src is in the python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import cv2
from frame_extractor import extract_frames
from exceptions import FrameExtractionError

def test_extract_frames_empty_video():
    """Verify handling of an empty video (returns immediately)."""
    mock_cap = MagicMock(spec=cv2.VideoCapture)
    mock_cap.get.return_value = 0  # CAP_PROP_FRAME_COUNT
    mock_cap.read.return_value = (False, None)
    
    records = list(extract_frames(mock_cap, fps=30.0))
    
    assert len(records) == 0
    mock_cap.release.assert_called_once()

def test_extract_frames_single_frame():
    """Verify correct processing of a single frame."""
    mock_cap = MagicMock(spec=cv2.VideoCapture)
    mock_cap.get.return_value = 1  # CAP_PROP_FRAME_COUNT
    
    mock_image = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_cap.read.side_effect = [(True, mock_image), (False, None)]
    
    records = list(extract_frames(mock_cap, fps=10.0))
    
    assert len(records) == 1
    assert records[0].frame_index == 0
    assert records[0].timestamp_seconds == 0.0
    assert np.array_equal(records[0].image, mock_image)
    
    mock_cap.release.assert_called_once()

def test_extract_frames_multiple_frames_timestamp_correctness():
    """Verify correct sequential processing and timestamp calculation."""
    mock_cap = MagicMock(spec=cv2.VideoCapture)
    mock_cap.get.return_value = 3  # CAP_PROP_FRAME_COUNT
    
    mock_image = np.zeros((10, 10, 3), dtype=np.uint8)
    
    # 3 frames then EOF
    mock_cap.read.side_effect = [
        (True, mock_image),
        (True, mock_image),
        (True, mock_image),
        (False, None)
    ]
    
    fps = 2.0  # 0.5 seconds per frame
    records = list(extract_frames(mock_cap, fps=fps))
    
    assert len(records) == 3
    assert records[0].timestamp_seconds == 0.0
    assert records[1].timestamp_seconds == 0.5
    assert records[2].timestamp_seconds == 1.0
    
    mock_cap.release.assert_called_once()

def test_extract_frames_premature_eof():
    """Verify that ret=False before expected_frames raises FrameExtractionError."""
    mock_cap = MagicMock(spec=cv2.VideoCapture)
    mock_cap.get.return_value = 10  # CAP_PROP_FRAME_COUNT
    
    mock_image = np.zeros((10, 10, 3), dtype=np.uint8)
    mock_cap.read.side_effect = [
        (True, mock_image),
        (False, None)  # Fails at index 1 instead of 10
    ]
    
    generator = extract_frames(mock_cap, fps=30.0)
    
    record = next(generator)
    assert record.frame_index == 0
    
    with pytest.raises(FrameExtractionError, match="Failed to read frame at index 1"):
        next(generator)
        
    mock_cap.release.assert_called_once()

def test_extract_frames_none_frame_with_true_ret():
    """Verify that a None frame when ret is True raises FrameExtractionError."""
    mock_cap = MagicMock(spec=cv2.VideoCapture)
    mock_cap.get.return_value = 1
    
    mock_cap.read.return_value = (True, None)
    
    generator = extract_frames(mock_cap, fps=30.0)
    
    with pytest.raises(FrameExtractionError, match="Extracted frame is None"):
        next(generator)
        
    mock_cap.release.assert_called_once()
