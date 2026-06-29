# Phase 2 Metadata Schema

To preserve the reproducibility of the pipeline, Phase 2 maintains an extended audit trail that tracks specific facial extraction metrics alongside the original video processing statistics.

## Extensibility

Phase 2 will output a JSON file (e.g. `video_metadata.json`) containing the standard Phase 1 metrics (fps, total frames, etc.) and new keys strictly tracking facial signal quality.

## Frozen Schema

The following face-specific fields are strictly defined and must not be mutated in downstream pipelines without a formal version bump.

```json
{
  "frames_seen": 0,
  
  "faces_detected": 0,
  "faces_validated": 0,
  "faces_quality_passed": 0,
  "faces_aligned": 0,

  "average_detection_confidence": 0.0,
  
  "min_face_area_ratio": 0.0,
  "max_face_area_ratio": 0.0,
  "average_face_area_ratio": 0.0,
  
  "min_laplacian_variance": 0.0,
  "max_laplacian_variance": 0.0,
  "average_laplacian_variance": 0.0,

  "face_detection_failures": 0,
  "face_validation_failures": 0,
  "face_quality_failures": 0,
  "face_alignment_failures": 0
}
```

### Definitions

- **`frames_seen`**: Total frames attempted to be processed by the FaceDetector.
- **`faces_detected`**: Count of frames where the detector found at least one face.
- **`faces_validated`**: Count of frames that passed strict confidence, area ratio, and landmark-shape thresholds.
- **`faces_quality_passed`**: Count of frames that passed the image quality (e.g. Laplacian blur) threshold.
- **`faces_aligned`**: Count of frames successfully affine-warped to the 112x112 canonical template.

#### Statistics
- **`average_detection_confidence`**: Mean confidence across all `faces_detected`.
- **`min/max/average_face_area_ratio`**: Minimum, maximum, and mean crop-to-frame area ratio across all `faces_detected`.
- **`min/max/average_laplacian_variance`**: Minimum, maximum, and mean blur variance across all `faces_quality_passed` (metrics for rejected images are excluded).

#### Failure Counters
- **`face_detection_failures`**: Count of `FaceNotFoundError`.
- **`face_validation_failures`**: Count of `FaceValidationError`.
- **`face_quality_failures`**: Count of `FaceQualityError`.
- **`face_alignment_failures`**: Count of `FaceAlignmentError`.

These counters will accumulate independently and be finalized during the `save_metadata` invocation in Phase 2.
