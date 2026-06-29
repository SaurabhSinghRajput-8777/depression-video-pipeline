import logging
from models.face_record import FaceRecord
from exceptions import FaceValidationError

logger = logging.getLogger(__name__)

def validate_face(record: FaceRecord) -> FaceRecord:
    """
    Validates a FaceRecord according to strict quality thresholds.
    This module strictly verifies if a detected face is "valid" 
    before expensive alignment or filtering operations take place.
    
    Args:
        record (FaceRecord): The face record to validate.
        
    Returns:
        FaceRecord: The unmodified face record if validation passes.
        
    Raises:
        FaceValidationError: If the face fails any threshold check.
    """
    # Rule 1: Detection confidence
    if record.detection_confidence < 0.8:
        raise FaceValidationError(
            f"Frame {record.frame_index}: Detection confidence {record.detection_confidence:.3f} is below 0.8 threshold."
        )
        
    # Rule 2: Face area ratio
    if record.face_area_ratio < 0.05:
        raise FaceValidationError(
            f"Frame {record.frame_index}: Face area ratio {record.face_area_ratio:.3f} is below 0.05 threshold."
        )
        
    # Rule 3: Landmarks shape
    if record.landmarks is None:
         raise FaceValidationError(f"Frame {record.frame_index}: Landmarks are missing.")
         
    if record.landmarks.shape != (5, 2):
        raise FaceValidationError(
            f"Frame {record.frame_index}: Invalid landmarks shape {record.landmarks.shape}. Expected (5, 2)."
        )
        
    logger.debug(
        "Validated face in frame %s",
        record.frame_index
    )
        
    return record
