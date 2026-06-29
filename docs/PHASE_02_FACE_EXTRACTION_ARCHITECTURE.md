# Phase 2: Face Extraction Architecture

## 1. Objective
Design and implement the facial extraction phase of the pipeline. Since the subject's face contains the primary signal for depression classification, this phase dictates downstream performance. 

## 2. Core Decisions to Evaluate

### A. Face Detector Choice
We must evaluate models based on accuracy, bounding-box stability across frames, and computational overhead.
- **Haar Cascades**: Outdated; highly sensitive to lighting and angles. We will avoid this except potentially as a strict baseline.
- **Dlib HOG**: Robust baseline but struggles heavily with non-frontal faces.
- **MTCNN**: Multi-task cascaded architecture. Good alignment but computationally heavier and somewhat older.
- **RetinaFace**: State-of-the-art single-stage face detector. Extremely robust to occlusion and scale variance. Highly recommended for clinical/research fidelity where stability and consistency are prioritized over raw speed.
- **MediaPipe Face Detection**: Highly optimized and incredibly fast.

**Decision:** 
- **Primary:** RetinaFace (prioritizing bounding box stability, difficult poses, and partial occlusion for offline research).
- **Fallback:** MediaPipe (for performance-constrained scenarios).

### B. Alignment Strategy
Facial alignment ensures that the eyes, nose, and mouth are registered to the same pixel coordinates across frames, stripping out affine noise (tilt/rotation) so the CNN focuses purely on expression/texture.
- **No Alignment**: Risk of high variance; CNN wastes capacity learning head pose invariants.
- **Eye-based Alignment (2-point)**: Rotates the image to ensure the eyes are horizontal. Simple and widely used.
- **Five-point Alignment (Eyes, Nose, Mouth corners)**: Computes an affine transform aligning to a canonical template. More rigid, but superior for spatial feature consistency.

**Decision:** Five-point alignment is **mandatory**. Depression cues are subtle; removing pose variation early is crucial and worth the extra computational cost.

### C. Quality Filtering
We must decide how to handle corrupted or noisy data to avoid feeding garbage to the final model. 
- **Face Not Detected**: Should we interpolate the bounding box from previous frames, or simply skip and record a `skipped_frames` stat?
- **Face Coverage Ratio**: Reject frames where the face area / frame area is `< 5%`. Tiny faces lack sufficient textural detail for analysis.
- **Heavily Occluded / Turned Away**: Utilize confidence scores from the detector. If confidence is `< 0.8`, drop the frame.
- **Motion Blur**: Apply a Laplacian variance filter. If variance is below a threshold, drop the frame.

## 3. Impact on Pipeline Architecture
Implementing this will require updating the core logic to transition from `image_preprocessor.py` (naive cropping) to a dedicated facial extraction architecture, likely creating a new `face_extractor.py` that executes between frame streaming and standardization.
