"""
Tests for the video_reader module.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure src is in the python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import cv2

from video_reader import read_video
from exceptions import (
    VideoNotFoundError,
    InvalidVideoFormatError,
    VideoOpenError
)

@patch("video_reader.Path.is_file")
@patch("video_reader.Path.exists")
def test_read_video_not_found(mock_exists, mock_is_file):
    """Verify that a missing file raises VideoNotFoundError."""
    mock_exists.return_value = False
    mock_is_file.return_value = False
    
    with pytest.raises(VideoNotFoundError, match="Cannot find video"):
        read_video(Path("missing.mp4"))

@patch("video_reader.Path.is_file")
@patch("video_reader.Path.exists")
def test_read_video_not_a_file(mock_exists, mock_is_file):
    """Verify that a directory raises VideoNotFoundError."""
    mock_exists.return_value = True
    mock_is_file.return_value = False
    
    with pytest.raises(VideoNotFoundError, match="Cannot find video"):
        read_video(Path("data/raw"))

@patch("video_reader.Path.is_file")
@patch("video_reader.Path.exists")
def test_read_video_invalid_format(mock_exists, mock_is_file):
    """Verify that an unsupported extension raises InvalidVideoFormatError."""
    mock_exists.return_value = True
    mock_is_file.return_value = True
    
    with pytest.raises(InvalidVideoFormatError, match="Extension .avi not supported"):
        read_video(Path("test.avi"))

@patch("video_reader.cv2.VideoCapture")
@patch("video_reader.Path.is_file")
@patch("video_reader.Path.exists")
def test_read_video_open_error(mock_exists, mock_is_file, mock_videocapture):
    """Verify that a failed cv2.VideoCapture raises VideoOpenError and releases handle."""
    mock_exists.return_value = True
    mock_is_file.return_value = True
    
    # Mock cap.isOpened() to return False
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_videocapture.return_value = mock_cap
    
    with pytest.raises(VideoOpenError, match="Failed to open video file"):
        read_video(Path("corrupt.mp4"))
        
    mock_cap.release.assert_called_once()

@patch("video_reader.cv2.VideoCapture")
@patch("video_reader.Path.is_file")
@patch("video_reader.Path.exists")
def test_read_video_success(mock_exists, mock_is_file, mock_videocapture):
    """Verify that a successful open returns the VideoCapture object."""
    mock_exists.return_value = True
    mock_is_file.return_value = True
    
    # Mock cap.isOpened() to return True
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_videocapture.return_value = mock_cap
    
    cap = read_video(Path("valid.mp4"))
    assert cap is mock_cap
