import json
import logging
from pathlib import Path
from datetime import datetime

from config import (
    PIPELINE_NAME,
    PIPELINE_VERSION,
    DEFAULT_METADATA_DIR
)
from exceptions import MetadataWriteError

logger = logging.getLogger(__name__)

def save_face_metadata(
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
    phase2_stats: dict,
    output_dir: Path = DEFAULT_METADATA_DIR
) -> Path:
    """
    Saves a comprehensive JSON audit trail of the Phase 2 extraction execution.
    Maintains Phase 1 base keys but injects Phase 2 metrics.
    """
    if fps < 0 or duration_seconds < 0 or total_frames < 0 or processed_frames < 0:
        raise MetadataWriteError("Invalid Phase 1 base metadata constraints.")
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stem = Path(video_name).stem
    json_path = output_dir / f"{stem}_metadata_phase2.json"
    tmp_path = json_path.with_suffix(".tmp")
    
    processing_duration_seconds = (end_time - start_time).total_seconds()
    
    metadata = {
        "pipeline_name": PIPELINE_NAME,
        "pipeline_phase": 2,
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
        
        "output_width": 112,
        "output_height": 112,
        
        "processing_started_at": start_time.isoformat(),
        "processing_completed_at": end_time.isoformat(),
        "processing_duration_seconds": processing_duration_seconds
    }
    
    # Inject frozen Phase 2 schema
    metadata.update(phase2_stats)
    
    try:
        # Atomic write pattern
        with open(tmp_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
            
        tmp_path.replace(json_path)
        logger.info("Successfully saved Phase 2 metadata audit trail to: %s", json_path)
        return json_path
        
    except Exception as e:
        logger.error("Failed to write Phase 2 metadata for %s: %s", video_name, str(e))
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise MetadataWriteError(f"Could not save Phase 2 metadata: {e}") from e
