"""
Tests for the metadata_logger module.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timezone

# Ensure src is in the python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest

from metadata_logger import save_metadata
from config import PIPELINE_NAME, PIPELINE_PHASE, PIPELINE_VERSION, TARGET_SIZE
from exceptions import MetadataWriteError

def test_save_metadata(tmp_path):
    """Verify that metadata is correctly structured and written to disk."""
    start_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    end_time = datetime(2025, 1, 1, 12, 1, 30, tzinfo=timezone.utc)
    
    json_path = save_metadata(
        video_name="test_video.mp4",
        fps=30.0,
        original_width=1920,
        original_height=1080,
        duration_seconds=60.0,
        total_frames=1800,
        processed_frames=1800,
        failed_frames=0,
        skipped_frames=0,
        start_time=start_time,
        end_time=end_time,
        output_dir=tmp_path
    )
    
    assert json_path.exists()
    assert json_path.name == "test_video_metadata.json"
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    assert data["pipeline_name"] == PIPELINE_NAME
    assert data["pipeline_phase"] == PIPELINE_PHASE
    assert data["pipeline_version"] == PIPELINE_VERSION
    assert data["video_name"] == "test_video.mp4"
    assert data["fps"] == 30.0
    assert data["width"] == 1920
    assert data["height"] == 1080
    assert data["duration_seconds"] == 60.0
    assert data["total_frames"] == 1800
    assert data["processed_frames"] == 1800
    assert data["failed_frames"] == 0
    assert data["skipped_frames"] == 0
    assert data["output_width"] == TARGET_SIZE[0]
    assert data["output_height"] == TARGET_SIZE[1]
    assert data["processing_started_at"] == start_time.isoformat()
    assert data["processing_completed_at"] == end_time.isoformat()
    assert data["processing_duration_seconds"] == 90.0

def test_save_metadata_invalid_values(tmp_path):
    """Verify that impossible metadata values raise MetadataWriteError."""
    start_time = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    end_time = datetime(2025, 1, 1, 12, 1, 30, tzinfo=timezone.utc)
    
    with pytest.raises(MetadataWriteError, match="Invalid duration_seconds"):
        save_metadata(
            video_name="test_video.mp4",
            fps=30.0,
            original_width=1920,
            original_height=1080,
            duration_seconds=-10.0,  # Invalid
            total_frames=1800,
            processed_frames=1800,
            failed_frames=0,
            skipped_frames=0,
            start_time=start_time,
            end_time=end_time,
            output_dir=tmp_path
        )
