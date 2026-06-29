"""
Pipeline Orchestrator module for the Video Preprocessing Pipeline.

Coordinates the end-to-end processing of a single video, ensuring all metadata
is properly tracked and all modules are executed in the correct order.

Usage Example:
    from pipeline import process_video
    from pathlib import Path
    
    metadata_path = process_video(Path("data/raw/sample.mp4"))
"""

import logging
import cv2
from pathlib import Path
from datetime import datetime, timezone
from typing import Union

from config import DEFAULT_PROCESSED_DIR, DEFAULT_METADATA_DIR
from exceptions import PipelineError, FrameProcessingError, MetadataWriteError

from video_reader import read_video
from fps_extractor import extract_fps
from frame_extractor import extract_frames
from grayscale_converter import convert_to_grayscale
from image_preprocessor import preprocess_frame
from metadata_logger import save_metadata

logger = logging.getLogger(__name__)

def process_video(
    video_path: Union[str, Path],
    output_base_dir: Path = DEFAULT_PROCESSED_DIR,
    metadata_dir: Path = DEFAULT_METADATA_DIR
) -> Path:
    """
    Orchestrates the complete preprocessing of a video.
    
    Args:
        video_path: Path to the input video.
        output_base_dir: Where to save processed frames.
        metadata_dir: Where to save the JSON audit trail.
        
    Returns:
        Path: Path to the saved metadata JSON file.
        
    Raises:
        PipelineError: If any critical stage of the pipeline fails.
    """
    start_time = datetime.now(timezone.utc)
    path = Path(video_path)
    video_name = path.name
    
    logger.info("Starting pipeline for video: %s", video_name)
    
    processed_frames = 0
    failed_frames = 0
    skipped_frames = 0
    
    # Variables that will be needed for metadata
    fps = 0.0
    original_width = 0
    original_height = 0
    duration_seconds = 0.0
    total_frames = 0
    
    original_exception = None
    metadata_path = None
    
    try:
        # Step 1: Read Video
        cap = read_video(path)
        
        # Setup output directory for frames ONLY after reading succeeds
        video_output_dir = output_base_dir / path.stem
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 2: Extract FPS & Properties
        fps = extract_fps(cap)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / fps if fps > 0 else 0.0
        
        # Step 3: Extract and Process Frames
        for raw_record in extract_frames(cap, fps):
            try:
                # Convert to Grayscale
                gray_record = convert_to_grayscale(raw_record)
                
                # Preprocess (Crop & Resize)
                final_record = preprocess_frame(gray_record)
                
                # Save Frame
                frame_filename = f"frame_{final_record.frame_index:06d}.jpg"
                frame_path = video_output_dir / frame_filename
                
                success = cv2.imwrite(str(frame_path), final_record.image)
                if not success:
                    raise FrameProcessingError(f"Failed to save frame to {frame_path}")
                
                processed_frames += 1
                
            except FrameProcessingError as e:
                logger.warning("Failed to process frame %s: %s", raw_record.frame_index, e)
                failed_frames += 1
                # Research pipelines typically continue on individual frame failures
                continue
                
    except Exception as e:
        if isinstance(e, PipelineError):
            logger.error("Pipeline failure for video %s: %s", video_name, e)
            original_exception = e
        else:
            logger.error("Unexpected pipeline failure for video %s: %s", video_name, e)
            original_exception = PipelineError(f"Unexpected error: {e}")
            original_exception.__cause__ = e
    finally:
        end_time = datetime.now(timezone.utc)
        
        # We always attempt to write metadata, even if the pipeline failed halfway.
        # This preserves whatever stats we managed to collect.
        # Note: If pipeline failed before FPS was extracted, fps=0 will raise MetadataWriteError.
        try:
            metadata_path = save_metadata(
                video_name=video_name,
                fps=fps,
                original_width=original_width,
                original_height=original_height,
                duration_seconds=duration_seconds,
                total_frames=total_frames,
                processed_frames=processed_frames,
                failed_frames=failed_frames,
                skipped_frames=skipped_frames,
                start_time=start_time,
                end_time=end_time,
                output_dir=metadata_dir
            )
        except MetadataWriteError as meta_err:
            logger.exception("Critical failure writing metadata for %s: %s", video_name, meta_err)
            if original_exception is None:
                original_exception = PipelineError(f"Failed to save final metadata: {meta_err}")
                original_exception.__cause__ = meta_err
                
    if original_exception is not None:
        raise original_exception
        
    logger.info("Successfully completed pipeline for video: %s", video_name)
    return metadata_path
