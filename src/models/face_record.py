import numpy as np
from dataclasses import dataclass

@dataclass(slots=True)
class FaceRecord:
    """
    Data transport object for the Phase 2 facial extraction pipeline.
    Ensures memory efficiency and strict typing for face-specific metadata.
    """
    frame_index: int
    timestamp_seconds: float
    image: np.ndarray
    
    # Bounding box in (x, y, width, height) format
    face_bbox: tuple[int, int, int, int]
    
    detection_confidence: float
    landmarks: np.ndarray | None
    face_area_ratio: float
    laplacian_variance: float
    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the face crop image."""
        return self.image.shape

    def __post_init__(self) -> None:
        """Validates the inputs upon instantiation."""
        if self.frame_index < 0:
            raise ValueError("frame_index must be >= 0")
        
        if self.timestamp_seconds < 0.0:
            raise ValueError("timestamp_seconds must be >= 0")
            
        if not isinstance(self.image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")
            
        if self.image.ndim not in (2, 3):
            raise ValueError("image must be 2D (grayscale) or 3D (RGB) array")
            
        if not isinstance(self.face_bbox, tuple) or len(self.face_bbox) != 4:
            raise TypeError("face_bbox must be a tuple of 4 integers: (x, y, w, h)")
            
        x, y, w, h = self.face_bbox
        if not all(isinstance(v, int) for v in (x, y, w, h)):
            raise TypeError("face_bbox values must be integers")
            
        if w <= 0 or h <= 0:
            raise ValueError("face_bbox width and height must be strictly positive")
            
        if not (0.0 <= self.detection_confidence <= 1.0):
            raise ValueError("detection_confidence must be between 0.0 and 1.0")
            
        if not (0.0 <= self.face_area_ratio <= 1.0):
            raise ValueError("face_area_ratio must be between 0.0 and 1.0")
            
        if self.laplacian_variance < 0.0:
            raise ValueError("laplacian_variance cannot be negative")
            
        if self.landmarks is not None:
            if not isinstance(self.landmarks, np.ndarray):
                raise TypeError("landmarks must be a numpy.ndarray or None")
            if self.landmarks.shape != (5, 2):
                raise ValueError("landmarks must have shape (5, 2)")
