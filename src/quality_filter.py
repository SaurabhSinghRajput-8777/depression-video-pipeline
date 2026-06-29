import cv2
import logging
from models.face_record import FaceRecord
from exceptions import FaceQualityError

logger = logging.getLogger(__name__)

def filter_face_quality(record: FaceRecord, blur_threshold: float = 100.0) -> FaceRecord:
    """
    Evaluates the image quality of a valid face crop.
    This module strictly assesses metrics like blur (Laplacian variance)
    and returns an updated FaceRecord with the computed metrics.
    
    Args:
        record (FaceRecord): The valid face record to filter.
        blur_threshold (float): Minimum acceptable Laplacian variance. Defaults to 100.0 (experimental).
        
    Returns:
        FaceRecord: A new FaceRecord containing the computed laplacian_variance.
        
    Raises:
        FaceQualityError: If the image fails quality standards (e.g., too blurry).
    """
    image = record.image
    
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    if gray.size == 0:
        raise FaceQualityError(f"Frame {record.frame_index}: Face crop is empty.")
        
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    if lap_var < blur_threshold:
        logger.debug(
            "Frame %s rejected. Laplacian variance %.2f",
            record.frame_index,
            lap_var
        )
        raise FaceQualityError(
            f"Frame {record.frame_index}: Face is too blurry. "
            f"Laplacian variance {lap_var:.2f} is below threshold {blur_threshold}."
        )
        
    logger.debug("Frame %s: Laplacian variance %.2f passed blur filter", record.frame_index, lap_var)
        
    # Return a new FaceRecord with the updated laplacian_variance
    return FaceRecord(
        frame_index=record.frame_index,
        timestamp_seconds=record.timestamp_seconds,
        image=record.image,
        face_bbox=record.face_bbox,
        detection_confidence=record.detection_confidence,
        landmarks=record.landmarks,
        face_area_ratio=record.face_area_ratio,
        laplacian_variance=float(lap_var)
    )
