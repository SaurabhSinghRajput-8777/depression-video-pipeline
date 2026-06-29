import cv2
import numpy as np
import logging
from models.face_record import FaceRecord
from exceptions import FaceAlignmentError
from constants.alignment import REFERENCE_POINTS, ALIGNED_FACE_SIZE

logger = logging.getLogger(__name__)

def align_face(record: FaceRecord) -> FaceRecord:
    """
    Performs a 5-point affine alignment to a canonical 112x112 template.
    
    Args:
        record (FaceRecord): The validated and quality-filtered FaceRecord.
        
    Returns:
        FaceRecord: A new FaceRecord containing the aligned 112x112 image and transformed landmarks.
        
    Raises:
        FaceAlignmentError: If alignment fails (e.g., missing landmarks or invalid transform matrix).
    """
    if record.landmarks is None:
        raise FaceAlignmentError(f"Frame {record.frame_index}: Missing landmarks for alignment.")
        
    if record.landmarks.shape != (5, 2):
        raise FaceAlignmentError(
            f"Frame {record.frame_index}: Invalid landmarks shape for alignment. "
            f"Expected (5, 2), got {record.landmarks.shape}."
        )
        
    try:
        # Compute optimal similarity transform (translation, rotation, uniform scale)
        # We use method=0 to compute regular least squares using all 5 points (no RANSAC needed for 5 stable points)
        M, _ = cv2.estimateAffinePartial2D(record.landmarks, REFERENCE_POINTS, method=0)
        
        if M is None:
            raise FaceAlignmentError(f"Frame {record.frame_index}: Failed to compute affine transform matrix.")
            
        # Warp the face crop to the canonical space
        aligned_image = cv2.warpAffine(record.image, M, ALIGNED_FACE_SIZE, borderValue=0.0)
        
        # Transform the landmarks to the new canonical space
        # landmarks is shape (5, 2). Convert to homogeneous coordinates (5, 3) for affine multiplication.
        ones = np.ones((record.landmarks.shape[0], 1))
        points_homogeneous = np.hstack([record.landmarks, ones])
        
        # M is (2, 3), points_homogeneous is (5, 3). 
        # Multiplication: (2x3) dot (3x5) = (2x5), transpose back to (5x2)
        aligned_landmarks = M.dot(points_homogeneous.T).T
        
        if aligned_landmarks.shape != (5, 2):
            raise FaceAlignmentError(f"Frame {record.frame_index}: Transformed landmarks have invalid shape {aligned_landmarks.shape}")
            
        if not np.isfinite(aligned_landmarks).all():
            raise FaceAlignmentError(f"Frame {record.frame_index}: Affine transform produced non-finite landmarks.")
        
    except Exception as e:
        raise FaceAlignmentError(f"Frame {record.frame_index}: Alignment warp failed: {str(e)}") from e
        
    logger.debug("Successfully aligned face for frame %s", record.frame_index)
        
    return FaceRecord(
        frame_index=record.frame_index,
        timestamp_seconds=record.timestamp_seconds,
        image=aligned_image,
        face_bbox=record.face_bbox,
        detection_confidence=record.detection_confidence,
        landmarks=aligned_landmarks,
        face_area_ratio=record.face_area_ratio,
        laplacian_variance=record.laplacian_variance
    )
