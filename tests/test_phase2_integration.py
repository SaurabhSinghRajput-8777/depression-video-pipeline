import pytest
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath('src'))

# Mock insightface before importing face_detector
sys.modules['insightface'] = MagicMock()
sys.modules['insightface.app'] = MagicMock()

from models.frame_record import FrameRecord
from face_detector import InsightFaceDetector
from face_validator import validate_face
from quality_filter import filter_face_quality
from face_aligner import align_face
from exceptions import (
    FaceNotFoundError,
    FaceValidationError,
    FaceQualityError,
    FaceAlignmentError,
)

class MockFace:
    def __init__(self, bbox, kps, det_score):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.kps = np.array(kps, dtype=np.float32)
        self.det_score = float(det_score)

@pytest.fixture
def mock_frame_record():
    # 640x480 RGB image
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    return FrameRecord(frame_index=1, timestamp_seconds=0.033, image=img)

@pytest.fixture
def happy_path_face():
    # 200x200 face bbox, well above area ratio threshold
    # Coordinates: x1=100, y1=100, x2=300, y2=300
    bbox = [100, 100, 300, 300]
    # Synthetic landmarks in the center
    kps = [
        [150, 150], [250, 150], [200, 200], [150, 250], [250, 250]
    ]
    return MockFace(bbox=bbox, kps=kps, det_score=0.95)

@patch("face_detector.FaceAnalysis")
def test_no_face_detected(mock_face_analysis, mock_frame_record):
    """Test 1: No face detected -> FaceNotFoundError"""
    mock_app = MagicMock()
    mock_app.get.return_value = []
    mock_face_analysis.return_value = mock_app
    
    detector = InsightFaceDetector(use_gpu=False)
    
    with pytest.raises(FaceNotFoundError):
        detector.detect_face(mock_frame_record)

@patch("face_detector.FaceAnalysis")
def test_low_confidence_face(mock_face_analysis, mock_frame_record, happy_path_face):
    """Test 2: Low confidence face -> FaceValidationError"""
    # Override confidence to fail validator
    happy_path_face.det_score = 0.5
    
    mock_app = MagicMock()
    mock_app.get.return_value = [happy_path_face]
    mock_face_analysis.return_value = mock_app
    
    detector = InsightFaceDetector(use_gpu=False)
    
    # Detection succeeds
    face_record = detector.detect_face(mock_frame_record)
    
    # Validator catches the low confidence
    with pytest.raises(FaceValidationError, match="below 0.8 threshold"):
        validate_face(face_record)

@patch("face_detector.FaceAnalysis")
def test_tiny_face(mock_face_analysis, mock_frame_record):
    """Test 3: Tiny face -> FaceValidationError"""
    # Create a tiny face, e.g. 10x10 bbox. Image is 640x480 (307,200 pixels)
    # Area ratio will be much less than 0.05
    bbox = [100, 100, 110, 110]
    kps = [[102, 102], [108, 102], [105, 105], [102, 108], [108, 108]]
    tiny_face = MockFace(bbox=bbox, kps=kps, det_score=0.99)
    
    mock_app = MagicMock()
    mock_app.get.return_value = [tiny_face]
    mock_face_analysis.return_value = mock_app
    
    detector = InsightFaceDetector(use_gpu=False)
    face_record = detector.detect_face(mock_frame_record)
    
    with pytest.raises(FaceValidationError, match="below 0.05 threshold"):
        validate_face(face_record)

@patch("face_detector.FaceAnalysis")
@patch("quality_filter.cv2.Laplacian")
def test_blurry_face(mock_laplacian, mock_face_analysis, mock_frame_record, happy_path_face):
    """Test 4: Blurry face -> FaceQualityError"""
    mock_app = MagicMock()
    mock_app.get.return_value = [happy_path_face]
    mock_face_analysis.return_value = mock_app
    
    # Mock laplacian variance to return 50.0 (below 100.0 threshold)
    mock_laplacian_result = MagicMock()
    mock_laplacian_result.var.return_value = 50.0
    mock_laplacian.return_value = mock_laplacian_result
    
    detector = InsightFaceDetector(use_gpu=False)
    face_record = detector.detect_face(mock_frame_record)
    validated_record = validate_face(face_record)
    
    with pytest.raises(FaceQualityError, match="below threshold"):
        filter_face_quality(validated_record, blur_threshold=100.0)

@patch("face_detector.FaceAnalysis")
@patch("face_aligner.cv2.estimateAffinePartial2D")
def test_alignment_failure(mock_estimate_affine, mock_face_analysis, mock_frame_record, happy_path_face):
    """Test 5: Alignment failure -> FaceAlignmentError"""
    mock_app = MagicMock()
    mock_app.get.return_value = [happy_path_face]
    mock_face_analysis.return_value = mock_app
    
    # Mock alignment to return None for the matrix
    mock_estimate_affine.return_value = (None, None)
    
    detector = InsightFaceDetector(use_gpu=False)
    face_record = detector.detect_face(mock_frame_record)
    validated_record = validate_face(face_record)
    
    # We bypass quality filter here because laplacian is mocked or irrelevant for alignment logic
    with pytest.raises(FaceAlignmentError, match="Failed to compute affine transform matrix"):
        align_face(validated_record)

@patch("face_detector.FaceAnalysis")
@patch("quality_filter.cv2.Laplacian")
@patch("face_aligner.cv2.estimateAffinePartial2D")
@patch("face_aligner.cv2.warpAffine")
def test_happy_path(mock_warp, mock_estimate_affine, mock_laplacian, mock_face_analysis, mock_frame_record, happy_path_face):
    """Test 6: Happy path through the entire Phase 2 sequence"""
    mock_app = MagicMock()
    mock_app.get.return_value = [happy_path_face]
    mock_face_analysis.return_value = mock_app
    
    # High laplacian variance
    mock_laplacian_result = MagicMock()
    mock_laplacian_result.var.return_value = 300.0
    mock_laplacian.return_value = mock_laplacian_result
    
    # Fake transform matrix
    M = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0]])
    mock_estimate_affine.return_value = (M, None)
    
    # Fake warped image
    mock_warp.return_value = np.zeros((112, 112, 3), dtype=np.uint8)
    
    # Execute Pipeline
    detector = InsightFaceDetector(use_gpu=False)
    
    # 1. Detect
    detected = detector.detect_face(mock_frame_record)
    
    # 2. Validate
    validated = validate_face(detected)
    
    # 3. Quality Filter
    filtered = filter_face_quality(validated)
    
    # 4. Align
    aligned = align_face(filtered)
    
    # Assertions
    assert aligned.image.shape == (112, 112, 3)
    assert aligned.landmarks.shape == (5, 2)
    assert aligned.laplacian_variance == 300.0
    assert np.isfinite(aligned.landmarks).all()
