"""
Integration Tests for the Video Preprocessing Pipeline.
"""

import sys
from pathlib import Path
import json

# Ensure src is in the python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
import cv2
import numpy as np
from unittest.mock import patch, MagicMock

from pipeline import process_video
from exceptions import (
    VideoNotFoundError,
    InvalidVideoFormatError,
    FrameExtractionError,
    FrameProcessingError
)
from config import TARGET_SIZE, PIPELINE_NAME, PIPELINE_PHASE, PIPELINE_VERSION

@pytest.fixture
def synthetic_video(tmp_path):
    """Generates a real 3-frame synthetic MP4 video on disk."""
    video_path = tmp_path / "synthetic_3_frames.mp4"
    
    # 1280x720, 30 fps, 3 frames
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, 30.0, (1280, 720))
    
    for i in range(3):
        # Create a synthetic frame (changing colors so they aren't completely identical)
        frame = np.full((720, 1280, 3), (i * 50, i * 50, i * 50), dtype=np.uint8)
        out.write(frame)
        
    out.release()
    return video_path

def test_happy_path(synthetic_video, tmp_path):
    """Test 1 - Happy Path: 3-frame synthetic MP4 processing."""
    output_dir = tmp_path / "processed"
    metadata_dir = tmp_path / "metadata"
    
    metadata_path = process_video(
        video_path=synthetic_video,
        output_base_dir=output_dir,
        metadata_dir=metadata_dir
    )
    
    # Verify outputs
    assert metadata_path.exists()
    
    frames_dir = output_dir / synthetic_video.stem
    assert frames_dir.exists()
    
    # Verify 3 processed JPGs exist
    frame_files = list(frames_dir.glob("*.jpg"))
    assert len(frame_files) == 3
    
    # Verify metadata JSON
    with open(metadata_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        
    assert meta["processed_frames"] == 3
    assert meta["failed_frames"] == 0
    assert meta["skipped_frames"] == 0
    assert meta["fps"] == 30.0
    assert meta["total_frames"] == 3
    assert meta["output_width"] == TARGET_SIZE[0]
    assert meta["output_height"] == TARGET_SIZE[1]
    
    # Verify processing_duration_seconds exists and is reasonable
    assert "processing_duration_seconds" in meta
    assert meta["processing_duration_seconds"] >= 0

def test_invalid_extension(tmp_path):
    """Test 2 - Invalid Extension."""
    output_dir = tmp_path / "processed"
    metadata_dir = tmp_path / "metadata"
    
    fake_txt = tmp_path / "sample.txt"
    fake_txt.touch()
    
    with pytest.raises(InvalidVideoFormatError):
        process_video(
            video_path=fake_txt,
            output_base_dir=output_dir,
            metadata_dir=metadata_dir
        )

def test_missing_file(tmp_path):
    """Test 3 - Missing File."""
    output_dir = tmp_path / "processed"
    metadata_dir = tmp_path / "metadata"
    
    missing_file = tmp_path / "missing.mp4"
    
    with pytest.raises(VideoNotFoundError):
        process_video(
            video_path=missing_file,
            output_base_dir=output_dir,
            metadata_dir=metadata_dir
        )

@patch("frame_extractor.cv2.VideoCapture")
def test_corrupted_video_mocked(mock_videocapture, tmp_path):
    """Test 4 - Corrupted Video: FrameExtractionError on read failure."""
    output_dir = tmp_path / "processed"
    metadata_dir = tmp_path / "metadata"
    
    fake_mp4 = tmp_path / "corrupt.mp4"
    fake_mp4.touch()
    
    mock_cap = MagicMock()
    # Read video success, FPS success
    mock_cap.isOpened.return_value = True
    
    def mock_get(propId):
        if propId == cv2.CAP_PROP_FPS:
            return 30.0
        elif propId == cv2.CAP_PROP_FRAME_WIDTH:
            return 1280
        elif propId == cv2.CAP_PROP_FRAME_HEIGHT:
            return 720
        elif propId == cv2.CAP_PROP_FRAME_COUNT:
            return 10
        return 0
    mock_cap.get.side_effect = mock_get
    
    # Force read failure on frame 2
    mock_image = np.ones((720, 1280, 3), dtype=np.uint8)
    mock_cap.read.side_effect = [
        (True, mock_image),  # Frame 0: success
        (False, None)        # Frame 1: unexpected failure before 10 frames
    ]
    
    # We must patch video_reader.cv2.VideoCapture as well because read_video calls it
    # We will patch both using nested patch context or we can just patch `video_reader.cv2.VideoCapture` instead,
    # Actually wait. The pipeline orchestrator calls `read_video` which calls `cv2.VideoCapture(str(path))`
    # and returns cap. Then pipeline calls `extract_frames(cap)`.
    # Let's patch `video_reader.cv2.VideoCapture` so it returns mock_cap!
    # Wait, in pytest, patching `video_reader.cv2.VideoCapture` works perfectly.
    pass

@patch("video_reader.cv2.VideoCapture")
def test_corrupted_video_mocked_correctly(mock_videocapture, tmp_path):
    """Test 4 - Corrupted Video: FrameExtractionError on read failure."""
    output_dir = tmp_path / "processed"
    metadata_dir = tmp_path / "metadata"
    
    fake_mp4 = tmp_path / "corrupt.mp4"
    fake_mp4.touch()
    
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    
    def mock_get(propId):
        if propId == cv2.CAP_PROP_FPS:
            return 30.0
        elif propId == cv2.CAP_PROP_FRAME_WIDTH:
            return 1280
        elif propId == cv2.CAP_PROP_FRAME_HEIGHT:
            return 720
        elif propId == cv2.CAP_PROP_FRAME_COUNT:
            return 10
        return 0
    mock_cap.get.side_effect = mock_get
    
    mock_image = np.ones((720, 1280, 3), dtype=np.uint8)
    mock_cap.read.side_effect = [
        (True, mock_image),  
        (False, None)        
    ]
    
    mock_videocapture.return_value = mock_cap
    
    with pytest.raises(FrameExtractionError, match="Failed to read frame at index 1"):
        process_video(
            video_path=fake_mp4,
            output_base_dir=output_dir,
            metadata_dir=metadata_dir
        )
        
    # Check that metadata was still written despite failure
    metadata_path = metadata_dir / "corrupt_metadata.json"
    assert metadata_path.exists()
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        
    assert meta["processed_frames"] == 1
    assert meta["failed_frames"] == 0
    assert meta["total_frames"] == 10

@patch("pipeline.convert_to_grayscale")
def test_frame_processing_failure(mock_convert, synthetic_video, tmp_path):
    """Test 5 - Frame Processing Failure: Mock conversion failure on frame 2."""
    output_dir = tmp_path / "processed"
    metadata_dir = tmp_path / "metadata"
    
    def side_effect(record):
        if record.frame_index == 1:
            raise FrameProcessingError("Simulated processing error on frame 1")
        import numpy as np
        from models.frame_record import FrameRecord
        return FrameRecord(record.frame_index, record.timestamp_seconds, np.ones((720, 1280), dtype=np.uint8))
        
    mock_convert.side_effect = side_effect
    
    metadata_path = process_video(
        video_path=synthetic_video,
        output_base_dir=output_dir,
        metadata_dir=metadata_dir
    )
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        
    assert meta["processed_frames"] == 2
    assert meta["failed_frames"] == 1

def test_metadata_creation_fields(synthetic_video, tmp_path):
    """Test 6 - Metadata Creation Fields: Verify all required fields exist."""
    output_dir = tmp_path / "processed"
    metadata_dir = tmp_path / "metadata"
    
    metadata_path = process_video(
        video_path=synthetic_video,
        output_base_dir=output_dir,
        metadata_dir=metadata_dir
    )
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)
        
    required_fields = [
        "pipeline_name",
        "pipeline_phase",
        "pipeline_version",
        "video_name",
        "fps",
        "width",
        "height",
        "duration_seconds",
        "total_frames",
        "processed_frames",
        "failed_frames",
        "skipped_frames",
        "output_width",
        "output_height",
        "processing_started_at",
        "processing_completed_at",
        "processing_duration_seconds"
    ]
    
    for field in required_fields:
        assert field in meta
