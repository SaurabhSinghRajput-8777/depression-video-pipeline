"""
Metadata Logger module for the Video Preprocessing Pipeline.

Handles creating the final audit trail for processed videos, saving the execution 
metadata to JSON to ensure full reproducibility and tracing for Phase 1.

Usage Example:
    from metadata_logger import save_metadata
    from datetime import datetime, timezone
    
    save_metadata(
        video_name="sample.mp4",
        fps=30.0,
        original_width=1280,
        original_height=720,
        duration_seconds=15.0,
        total_frames=450,
        processed_frames=450,
        failed_frames=0,
        skipped_frames=0,
        start_time=datetime.now(timezone.utc),
        end_time=datetime.now(timezone.utc)
    )
"""

import json
import logging
from pathlib import Path
from datetime import datetime

from config import (
    PIPELINE_NAME,
    PIPELINE_PHASE,
    PIPELINE_VERSION,
    DEFAULT_METADATA_DIR,
    TARGET_SIZE
)
from exceptions import MetadataWriteError

logger = logging.getLogger(__name__)

def save_metadata(
    video_name: str,
    fps: float,
    original_width: int,
    original_height: int,
    duration_seconds: float,
    total_frames: int,
    processed_frames: int,
    failed_frames: int,
    skipped_frames: int,
    start_time: datetime,
    end_time: datetime,
    output_dir: Path = DEFAULT_METADATA_DIR
) -> Path:
    """
    Saves a comprehensive JSON audit trail of the preprocessing execution.
    
    Args:
        video_name: Name of the input video file (e.g., 'sample.mp4').
        fps: Validated frames per second.
        original_width: Original video width.
        original_height: Original video height.
        duration_seconds: Original video duration in seconds.
        total_frames: Total frames originally in the video.
        processed_frames: Frames successfully processed and saved.
        failed_frames: Frames that encountered an error during extraction or processing.
        skipped_frames: Frames that were deliberately skipped.
        start_time: Datetime object representing pipeline start.
        end_time: Datetime object representing pipeline completion.
        output_dir: Directory to save the metadata JSON file. Defaults to DEFAULT_METADATA_DIR.
        
    Returns:
        Path: The path to the saved JSON metadata file.
        
    Raises:
        MetadataWriteError: If metadata contains impossible values or write operations fail.
    """
    # Validation of impossible metadata
    if fps < 0:
        raise MetadataWriteError(f"Cannot save metadata: Invalid fps: {fps}")
    if duration_seconds < 0:
        raise MetadataWriteError(f"Cannot save metadata: Invalid duration_seconds: {duration_seconds}")
    if total_frames < 0:
        raise MetadataWriteError(f"Cannot save metadata: Invalid total_frames: {total_frames}")
    if processed_frames < 0:
        raise MetadataWriteError(f"Cannot save metadata: Invalid processed_frames: {processed_frames}")
    if failed_frames < 0:
        raise MetadataWriteError(f"Cannot save metadata: Invalid failed_frames: {failed_frames}")
    if skipped_frames < 0:
        raise MetadataWriteError(f"Cannot save metadata: Invalid skipped_frames: {skipped_frames}")
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stem = Path(video_name).stem
    json_path = output_dir / f"{stem}_metadata.json"
    tmp_path = json_path.with_suffix(".tmp")
    
    processing_duration_seconds = (end_time - start_time).total_seconds()
    
    metadata = {
        "pipeline_name": PIPELINE_NAME,
        "pipeline_phase": PIPELINE_PHASE,
        "pipeline_version": PIPELINE_VERSION,
        
        "video_name": video_name,
        
        "fps": fps,
        "width": original_width,
        "height": original_height,
        "duration_seconds": duration_seconds,
        
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "failed_frames": failed_frames,
        "skipped_frames": skipped_frames,
        
        "output_width": TARGET_SIZE[0],
        "output_height": TARGET_SIZE[1],
        
        "processing_started_at": start_time.isoformat(),
        "processing_completed_at": end_time.isoformat(),
        "processing_duration_seconds": processing_duration_seconds
    }
    
    try:
        # Atomic write pattern
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
            
        tmp_path.replace(json_path)
        logger.info("Successfully saved metadata audit trail to: %s", json_path)
        return json_path
        
    except Exception as e:
        logger.error("Failed to write metadata for %s: %s", video_name, str(e))
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise MetadataWriteError(f"Could not save metadata: {e}") from e
