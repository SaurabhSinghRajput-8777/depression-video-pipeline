"""
Tests for the fps_extractor module.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Ensure src is in the python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import cv2
import math

from fps_extractor import extract_fps
from exceptions import InvalidFPSError

def test_extract_fps_valid():
    """Verify valid FPS extraction."""
    mock_cap = MagicMock(spec=cv2.VideoCapture)
    mock_cap.get.return_value = 30.0
    
    fps = extract_fps(mock_cap)
    assert fps == 30.0
    mock_cap.get.assert_called_once_with(cv2.CAP_PROP_FPS)

def test_extract_fps_zero():
    """Verify zero FPS raises InvalidFPSError."""
    mock_cap = MagicMock(spec=cv2.VideoCapture)
    mock_cap.get.return_value = 0.0
    
    with pytest.raises(InvalidFPSError, match="must be > 0"):
        extract_fps(mock_cap)

def test_extract_fps_negative():
    """Verify negative FPS raises InvalidFPSError."""
    mock_cap = MagicMock(spec=cv2.VideoCapture)
    mock_cap.get.return_value = -15.0
    
    with pytest.raises(InvalidFPSError, match="must be > 0"):
        extract_fps(mock_cap)

def test_extract_fps_nan():
    """Verify NaN FPS raises InvalidFPSError."""
    mock_cap = MagicMock(spec=cv2.VideoCapture)
    mock_cap.get.return_value = float('nan')
    
    with pytest.raises(InvalidFPSError, match="is NaN"):
        extract_fps(mock_cap)

def test_extract_fps_infinite():
    """Verify infinite FPS raises InvalidFPSError."""
    mock_cap = MagicMock(spec=cv2.VideoCapture)
    mock_cap.get.return_value = float('inf')
    
    with pytest.raises(InvalidFPSError, match="is infinite"):
        extract_fps(mock_cap)

def test_extract_fps_small_positive():
    """Verify a very small positive FPS is accepted."""
    mock_cap = MagicMock(spec=cv2.VideoCapture)
    mock_cap.get.return_value = 0.001
    
    fps = extract_fps(mock_cap)
    assert fps == 0.001
