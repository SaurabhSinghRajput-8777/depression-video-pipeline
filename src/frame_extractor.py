"""
Frame Extractor module for the Video Preprocessing Pipeline.

Yields video frames sequentially as `FrameRecord` objects, strictly streaming
from disk without loading the entire video into memory.

Usage Example:
    from video_reader import read_video
    from fps_extractor import extract_fps
    from frame_extractor import extract_frames
    
    cap = read_video("video.mp4")
    fps = extract_fps(cap)
    
    for record in extract_frames(cap, fps):
        print(record.frame_index, record.timestamp_seconds)
"""

import logging
import cv2
from typing import Generator

from models.frame_record import FrameRecord
from exceptions import FrameExtractionError

logger = logging.getLogger(__name__)

def extract_frames(cap: cv2.VideoCapture, fps: float) -> Generator[FrameRecord, None, None]:
    """
    Sequentially reads frames from a VideoCapture object and yields them as FrameRecords.
    
    Ensures memory efficiency by using a generator and computes timestamps
    based on the provided FPS. The capture object is released upon completion
    or when an error occurs.
    
    Args:
        cap: The OpenCV VideoCapture object.
        fps: The validated frames per second of the video.
        
    Yields:
        FrameRecord: A dataclass containing the frame index, timestamp, and image array.
        
    Raises:
        FrameExtractionError: If a frame fails to be read before expected EOF.
    """
    logger.info("Starting frame extraction with FPS: %s", fps)
    
    expected_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = 0
    yielded_frames = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # If we haven't reached the expected frame count, it's a read failure
                if expected_frames > 0 and frame_index < expected_frames:
                    logger.error(
                        "Unexpected read failure at index %s. Expected %s frames.", 
                        frame_index, expected_frames
                    )
                    raise FrameExtractionError(
                        f"Failed to read frame at index {frame_index}. "
                        f"Expected {expected_frames} total frames."
                    )
                # True End of File
                break
                
            if frame is None:
                logger.error("Extracted frame is None at index: %s", frame_index)
                raise FrameExtractionError(f"Extracted frame is None at index {frame_index}")
                
            timestamp_seconds = frame_index / fps
            
            record = FrameRecord(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                image=frame
            )
            
            yield record
            
            frame_index += 1
            yielded_frames += 1
            
    except FrameExtractionError:
        raise
    except Exception as e:
        logger.error("Unexpected error during frame extraction at index %s: %s", frame_index, str(e))
        raise FrameExtractionError(f"Unexpected error during extraction: {e}") from e
    finally:
        # Guarantee capture release
        logger.info("Capture released. Frames extracted: %s", yielded_frames)
        cap.release()
