import cv2
import logging
import traceback
from pathlib import Path
from datetime import datetime, timezone

from config import DEFAULT_METADATA_DIR
from video_reader import open_video
from fps_extractor import get_fps
from frame_extractor import extract_frames
from face_detector import InsightFaceDetector
from face_validator import validate_face
from quality_filter import filter_face_quality
from face_aligner import align_face
from debug_logger import DebugLogger
from metadata_logger_phase2 import save_face_metadata

from exceptions import (
    PipelineError,
    VideoError,
    FaceNotFoundError,
    FaceValidationError,
    FaceQualityError,
    FaceAlignmentError,
)

logger = logging.getLogger(__name__)

def process_video_phase2(
    video_path: Path, 
    output_dir: Path, 
    metadata_dir: Path = DEFAULT_METADATA_DIR,
    use_gpu: bool = False
) -> None:
    """
    Executes the Phase 2 facial extraction pipeline.
    This safely encapsulates the logic for face detection, validation, quality gating,
    and alignment, preserving Phase 1 behavior intact.
    """
    start_time = datetime.now(timezone.utc)
    
    if not video_path.exists():
        raise VideoError(f"Video file not found: {video_path}")
        
    stem = video_path.stem
    phase2_output_dir = output_dir / "phase2" / stem
    phase2_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Debug Logger
    debug_logger = DebugLogger(Path("data"))
    
    # Initialize Detector
    logger.info("Initializing InsightFace Detector...")
    try:
        detector = InsightFaceDetector(use_gpu=use_gpu)
    except Exception as e:
        raise PipelineError(f"Failed to initialize face detector: {e}") from e
    
    # Metrics
    stats = {
        "frames_seen": 0,
        "faces_detected": 0,
        "faces_validated": 0,
        "faces_quality_passed": 0,
        "faces_aligned": 0,
        "average_detection_confidence": 0.0,
        "min_face_area_ratio": float('inf'),
        "max_face_area_ratio": 0.0,
        "average_face_area_ratio": 0.0,
        "min_laplacian_variance": float('inf'),
        "max_laplacian_variance": 0.0,
        "average_laplacian_variance": 0.0,
        "face_detection_failures": 0,
        "face_validation_failures": 0,
        "face_quality_failures": 0,
        "face_alignment_failures": 0
    }
    
    sum_confidence = 0.0
    sum_area_ratio = 0.0
    sum_laplacian = 0.0
    
    cap = open_video(video_path)
    try:
        fps = get_fps(cap)
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0
        
        for frame_record in extract_frames(cap, total_frames):
            stats["frames_seen"] += 1
            
            try:
                # 1. Detection
                face_record = detector.detect_face(frame_record)
                stats["faces_detected"] += 1
                sum_confidence += face_record.detection_confidence
                
                # Update area ratio stats
                ar = face_record.face_area_ratio
                sum_area_ratio += ar
                if ar < stats["min_face_area_ratio"]: stats["min_face_area_ratio"] = ar
                if ar > stats["max_face_area_ratio"]: stats["max_face_area_ratio"] = ar
                
                # 2. Validation
                validated_record = validate_face(face_record)
                stats["faces_validated"] += 1
                
                # 3. Quality Filter
                filtered_record = filter_face_quality(validated_record)
                stats["faces_quality_passed"] += 1
                
                # Update laplacian stats
                lv = filtered_record.laplacian_variance
                sum_laplacian += lv
                if lv < stats["min_laplacian_variance"]: stats["min_laplacian_variance"] = lv
                if lv > stats["max_laplacian_variance"]: stats["max_laplacian_variance"] = lv
                
                # 4. Alignment
                aligned_record = align_face(filtered_record)
                stats["faces_aligned"] += 1
                
                # Save crop to processed/phase2/<video_name>/
                out_path = phase2_output_dir / f"face_{aligned_record.frame_index:06d}.jpg"
                cv2.imwrite(str(out_path), aligned_record.image)
                
            except FaceNotFoundError:
                stats["face_detection_failures"] += 1
                debug_logger.save_debug_sample(frame_record.image, "no_face", video_path.name, frame_record.frame_index)
                
            except FaceValidationError as e:
                stats["face_validation_failures"] += 1
                category = "validation_failure"
                if "confidence" in str(e).lower():
                    category = "low_confidence"
                elif "area ratio" in str(e).lower():
                    category = "tiny_face"
                
                # Save the detected crop if available, otherwise save the whole frame
                image_to_save = face_record.image if 'face_record' in locals() else frame_record.image
                debug_logger.save_debug_sample(image_to_save, category, video_path.name, frame_record.frame_index)
                
            except FaceQualityError:
                stats["face_quality_failures"] += 1
                debug_logger.save_debug_sample(validated_record.image, "blurry", video_path.name, frame_record.frame_index)
                
            except FaceAlignmentError:
                stats["face_alignment_failures"] += 1
                debug_logger.save_debug_sample(filtered_record.image, "alignment_failure", video_path.name, frame_record.frame_index)
                
            except Exception as e:
                logger.error("Unexpected error on frame %d: %s", frame_record.frame_index, traceback.format_exc())
                # Do not crash the dataset run for a single frame failure
                continue
                
    except PipelineError:
        raise
    except Exception as e:
        raise PipelineError(f"Unexpected error during Phase 2 processing: {e}") from e
    finally:
        cap.release()
        
        # Finalize averages and handle infinities if counters are zero
        if stats["faces_detected"] > 0:
            stats["average_detection_confidence"] = sum_confidence / stats["faces_detected"]
            stats["average_face_area_ratio"] = sum_area_ratio / stats["faces_detected"]
        else:
            stats["min_face_area_ratio"] = 0.0
            
        if stats["faces_quality_passed"] > 0:
            stats["average_laplacian_variance"] = sum_laplacian / stats["faces_quality_passed"]
        else:
            stats["min_laplacian_variance"] = 0.0
            
        end_time = datetime.now(timezone.utc)
        
        try:
            save_face_metadata(
                video_name=video_path.name,
                fps=fps if 'fps' in locals() else 0.0,
                original_width=original_width if 'original_width' in locals() else 0,
                original_height=original_height if 'original_height' in locals() else 0,
                duration_seconds=duration if 'duration' in locals() else 0.0,
                total_frames=total_frames if 'total_frames' in locals() else 0,
                processed_frames=stats["faces_aligned"], # Overload for Phase 1 compatibility
                failed_frames=stats["frames_seen"] - stats["faces_aligned"], 
                skipped_frames=0,
                start_time=start_time,
                end_time=end_time,
                phase2_stats=stats,
                output_dir=metadata_dir
            )
        except Exception as e:
            logger.exception("Final Phase 2 metadata write failed.")
            raise PipelineError(f"Failed to save final Phase 2 metadata: {e}") from e

if __name__ == "__main__":
    import argparse
    import sys
    
    # Configure root logger for the script execution
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(description="Run the Phase 2 Facial Extraction Pipeline.")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--output-dir", type=str, default="processed", help="Directory to save aligned faces")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for InsightFace detection")
    
    args = parser.parse_args()
    
    try:
        process_video_phase2(
            video_path=Path(args.video_path),
            output_dir=Path(args.output_dir),
            use_gpu=args.gpu
        )
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        sys.exit(1)
