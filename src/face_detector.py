import cv2
import numpy as np
import logging
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)

from models.frame_record import FrameRecord
from models.face_record import FaceRecord
from exceptions import FaceNotFoundError

class InsightFaceDetector:
    """
    Wraps the InsightFace (RetinaFace) model for detecting the primary face
    in a frame. This module is strictly responsible for detection, 
    20% bounding box expansion, extracting landmarks, and cropping.
    It does not perform alignment or quality filtering.
    """
    def __init__(self, use_gpu: bool = False):
        # We only need the detection module for RetinaFace
        self.app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection'])
        ctx_id = 0 if use_gpu else -1
        # det_size is the input size to the detector network. 640x640 is InsightFace's recommended default for typical images.
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        
    def detect_face(self, record: FrameRecord) -> FaceRecord:
        """
        Detects faces in the FrameRecord, selects the largest detected face by area, 
        expands the bbox by 20%, crops the face, adjusts the landmarks to the cropped 
        region, and returns a FaceRecord.
        
        Args:
            record (FrameRecord): The input frame. Expected to be BGR or Grayscale.
            
        Returns:
            FaceRecord: Transport object containing the face crop and raw metadata.
            
        Raises:
            FaceNotFoundError: If no face is detected in the frame.
        """
        image = record.image
        
        # InsightFace expects BGR image (3 channels).
        if image.ndim == 2:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            image_bgr = image
            
        faces = self.app.get(image_bgr)
        
        if not faces:
            raise FaceNotFoundError(f"No faces detected in frame {record.frame_index}.")
            
        def bbox_area(face) -> float:
            fx1, fy1, fx2, fy2 = face.bbox
            return max(0.0, fx2 - fx1) * max(0.0, fy2 - fy1)
            
        # Select largest detected face by area (patient > clinician > background)
        primary_face = max(faces, key=bbox_area)
        
        logger.debug(
            "Frame %s: selected face with confidence %.3f",
            record.frame_index,
            primary_face.det_score
        )
        
        # Original bbox (x1, y1, x2, y2)
        x1, y1, x2, y2 = primary_face.bbox
        orig_w = x2 - x1
        orig_h = y2 - y1
        
        # 20% expansion margin around the face
        margin_x = orig_w * 0.2
        margin_y = orig_h * 0.2
        
        img_h, img_w = image.shape[:2]
        
        new_x1 = max(0, int(x1 - margin_x))
        new_y1 = max(0, int(y1 - margin_y))
        new_x2 = min(img_w, int(x2 + margin_x))
        new_y2 = min(img_h, int(y2 + margin_y))
        
        new_w = new_x2 - new_x1
        new_h = new_y2 - new_y1
        
        if new_w <= 0 or new_h <= 0:
            raise FaceNotFoundError(f"Frame {record.frame_index}: Face crop geometry is zero or negative.")
        
        # Crop region directly from the original record.image to preserve its color space
        crop = record.image[new_y1:new_y2, new_x1:new_x2]
        
        # Adjust landmarks relative to the cropped image
        landmarks = primary_face.kps
        adjusted_landmarks = landmarks - np.array([new_x1, new_y1])
        
        if adjusted_landmarks.shape != (5, 2):
            raise FaceNotFoundError(f"Frame {record.frame_index}: Detector returned invalid landmarks shape: {adjusted_landmarks.shape}")
        
        # Compute face area ratio based on the actual crop
        crop_area = new_w * new_h
        frame_area = img_w * img_h
        face_area_ratio = float(crop_area / frame_area)
        
        face_bbox = (int(new_x1), int(new_y1), int(new_w), int(new_h))
        
        # We do not perform blur analysis here. laplacian_variance is initialized to 0.0
        # and will be calculated subsequently in quality_filter.py or face_validator.py
        
        return FaceRecord(
            frame_index=record.frame_index,
            timestamp_seconds=record.timestamp_seconds,
            image=crop,
            face_bbox=face_bbox,
            detection_confidence=float(primary_face.det_score),
            landmarks=adjusted_landmarks,
            face_area_ratio=face_area_ratio,
            laplacian_variance=0.0
        )
