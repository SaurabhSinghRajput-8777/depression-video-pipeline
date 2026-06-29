# Phase 2: Face Extraction Architecture

## 1. Objective
Produce a temporally stable sequence of aligned facial crops suitable for depression-related feature extraction. 
The goal is not simply to "detect faces," but to ensure that the bounding boxes are jitter-free and the facial landmarks are geometrically normalized (aligned). This pristine signal is critical because depression cues are subtle; any affine noise (pose variation) or tracking jitter will degrade the downstream neural network's performance.

## 2. Pipeline Diagram
```mermaid
graph LR
    A[FrameRecord] --> B[Face Detector]
    B --> C[Face Validator]
    C --> D[Face Aligner]
    D --> E[Quality Filter]
    E --> F[FaceRecord]
```
*(Note: The face detector is only one component of the full extraction pipeline.)*

## 3. Detector Evaluation
We evaluate models based on bounding-box stability across frames, difficult poses, partial occlusion, and consistency (accuracy > speed).

- **Haar Cascades**: Avoid. Outdated, highly sensitive to lighting and angles.
- **Dlib HOG**: Legacy. Robust baseline but struggles heavily with non-frontal faces.
- **MTCNN**: Acceptable. Multi-task cascaded architecture. Good alignment but computationally heavier and somewhat older.
- **RetinaFace**: ⭐ **Primary Choice**. State-of-the-art single-stage face detector. Stable bounding boxes, excellent occlusion handling, provides 5-point landmarks by default, and widely used in affective computing.
- **MediaPipe Face Detection**: ⭐ **Backup Choice**. Highly optimized and fast, but offline research prioritizes stability and consistency over real-time processing speed.

## 4. Selected Detector
**RetinaFace** is the primary choice. The pipeline bottleneck will ultimately be feature extraction and model training, not face detection. Thus, maximizing the stability of bounding boxes and extracting precise landmarks via RetinaFace takes priority.

## 5. FaceRecord Design
Phase 1's `FrameRecord` will not be mutated. Instead, a new `FaceRecord` dataclass will be introduced to transport the rich facial metadata.

```python
@dataclass(slots=True)
class FaceRecord:
    frame_index: int
    timestamp_seconds: float
    image: np.ndarray
    face_bbox: tuple[int, int, int, int]
    detection_confidence: float
    landmarks: np.ndarray | None
```

### Note on `FaceRecord.image` Semantics
The `FaceRecord.image` attribute dynamically represents the current state of the facial crop at any given point in the pipeline. Specifically:
- **After FaceDetector:** Contains the raw, unaligned 20% expanded face crop.
- **After FaceAligner:** Contains the 112x112 affine-aligned facial image.
Downstream components must understand that this abstraction represents the current highest-quality facial signal available in that stage.

## 6. Alignment Strategy
**Five-point alignment** is mandatory. 
RetinaFace natively provides the 5 canonical landmarks (Left Eye, Right Eye, Nose, Left Mouth Corner, Right Mouth Corner). We will compute an affine transform to normalize all faces to a canonical template before they enter any future CNNs. 2-point alignment will be skipped entirely.

## 7. Quality Filtering Rules
To prevent feeding corrupted or noisy data to downstream models, strict quality gating will be enforced:
- **Rule 1: Detection Confidence**: Must be `>= 0.8`. Else, skip frame.
- **Rule 2: Face Area**: `face_area` must be `>= 5%` of total `frame_area`. Tiny faces lack necessary textual detail. Else, skip frame.
- **Rule 3: Blur Detection**: Compute `cv2.Laplacian(image, cv2.CV_64F).var()`. If variance `< threshold`, skip frame.
- **Rule 4: Missing Face**: Do NOT interpolate yet. Just skip, log, and continue. Interpolation is a Phase 3 problem.

## 8. Metadata Extensions
The Phase 1 JSON audit trail will be expanded to include face-specific statistics:
```json
{
  "faces_detected": 420,
  "faces_aligned": 410,
  "faces_skipped": 10,
  "average_detection_confidence": 0.94,
  "average_face_area_ratio": 0.21
}
```

## 9. Testing Strategy
- **Unit Tests**: Mock `FaceRecord` generation and test isolated mathematical components (e.g., Laplacian blur calculation, Affine transform matrices).
- **Integration Tests**: Inject known "bad" frames (blurry, tiny face, no face) into the pipeline and verify that the `Quality Filter` correctly drops them and updates the tracking metadata accordingly.

## 10. Risks & Limitations
- **Computational Overhead**: RetinaFace + Affine transformations will be significantly slower than Phase 1's naive center crop.
- **Multiple Faces**: The pipeline assumes a single primary subject (e.g., patient in a clinical interview). If multiple faces are detected, the pipeline must implement a heuristic (e.g., largest bounding box) to consistently track the primary subject.
