# Phase 1: Video Preprocessing Pipeline

## Objective

Build a reproducible video preprocessing pipeline for depression detection research.

The purpose of this phase is to convert raw video files into standardized frame sequences that can later be used for feature extraction, pretrained models, and depression detection systems.

This phase focuses only on video preprocessing and does not perform any machine learning inference.

---

# Problem Statement

Raw videos are not directly suitable for machine learning models because they differ in:

- Resolution
- Frame rate (FPS)
- Color format
- Frame dimensions
- Recording conditions

To ensure consistency across all videos, a preprocessing pipeline must be created.

---

# Scope of Phase 1

The following components will be implemented:

1. MP4 File Reading
2. FPS Extraction
3. Frame Extraction
4. RGB to Grayscale Conversion
5. Frame Cropping and Resizing

Anything beyond these steps is considered out of scope for this phase.

Examples of out-of-scope tasks:

- Face Detection
- Face Alignment
- Feature Extraction
- CNN Models
- Vision Transformers
- Depression Classification
- Multimodal Fusion

---

# Pipeline Architecture

```text
Raw MP4 Video
       │
       ▼
Video Reader
       │
       ▼
FPS Extraction
       │
       ▼
Frame Extraction
       │
       ▼
RGB → Grayscale Conversion
       │
       ▼
Frame Cropping / Resizing
       │
       ▼
Processed Frame Sequence
```

---

# Module 1: MP4 File Reading

## Purpose

Read and decode video files so frames can be accessed individually.

---

## Input

```text
video.mp4
```

---

## Output

```text
VideoCapture Object
```

---

## Selected Technology

### OpenCV

```python
cv2.VideoCapture()
```

---

## Why OpenCV?

- Widely used in computer vision research
- Supports most video formats
- Efficient frame-by-frame processing
- Simple API
- Direct FPS extraction support

---

## Expected Function

```python
read_video(video_path)
```

---

## Validation Requirements

The system must verify:

- File exists
- File format is supported
- Video opens successfully
- Video contains readable frames

---

# Module 2: FPS Extraction

## Purpose

Determine the temporal resolution of the video.

---

## Definition

FPS (Frames Per Second) represents the number of frames displayed each second.

Example:

```text
30 FPS
```

means:

```text
30 frames are recorded every second
```

---

## Why FPS Is Important

Temporal characteristics are important in depression analysis.

Examples:

- Eye blink frequency
- Head movement patterns
- Facial motion dynamics
- Expression duration

Incorrect FPS handling may distort temporal information.

---

## Implementation

```python
fps = cap.get(cv2.CAP_PROP_FPS)
```

---

## Output Example

```json
{
  "fps": 30
}
```

---

## Metadata Requirement

Store FPS information for every processed video.

Example:

```json
{
  "video_name": "sample.mp4",
  "fps": 30
}
```

---

# Module 3: Frame Extraction

## Purpose

Convert video streams into individual image frames.

---

## Input

```text
MP4 Video
```

---

## Output

```text
Frame_0001
Frame_0002
Frame_0003
...
```

---

## Design Decision

For Phase 1:

Extract every frame.

Reason:

- Preserve all temporal information
- Simplify validation
- Avoid introducing sampling bias

Frame selection strategies can be introduced in future phases.

---

## Output Format

Frames should be stored as NumPy arrays.

Example:

```python
(height, width, 3)
```

Example:

```python
(720, 1280, 3)
```

Where:

```text
3 = RGB color channels
```

---

# Module 4: RGB to Grayscale Conversion

## Purpose

Reduce computational complexity and standardize visual representation.

---

## Why Convert to Grayscale?

Many facial behavior studies rely more on:

- Facial structure
- Geometric changes
- Movement patterns
- Expression dynamics

rather than color information.

---

## Benefits

### Reduced Memory Usage

```text
RGB = 3 channels
Gray = 1 channel
```

Approximately three times less memory is required.

---

### Faster Processing

Smaller frame representations result in lower computational cost.

---

### Better Standardization

Less sensitivity to:

- Skin tone differences
- Lighting variations
- Camera color profiles

---

## Implementation

```python
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```

---

## Shape Transformation

Before:

```python
(H, W, 3)
```

After:

```python
(H, W)
```

---

# Module 5: Frame Cropping and Resizing

## Purpose

Ensure every frame has identical dimensions.

---

## Why Cropping Is Required

Pretrained models require fixed-size inputs.

Examples:

```text
224 × 224
112 × 112
256 × 256
```

Raw videos may have arbitrary dimensions:

```text
1920 × 1080
1280 × 720
640 × 480
```

These cannot be directly fed into most deep learning models.

---

# Design Decision

## Current Approach

### Center Crop

The initial implementation will use center cropping.

Advantages:

- Simple implementation
- No dependency on face detectors
- Easy to validate

---

## Future Upgrade

Replace center cropping with face-based cropping using:

- Haar Cascade
- Dlib
- MTCNN
- RetinaFace

Face-based cropping is expected to improve depression-related feature quality.

---

## Resize Target

Selected size:

```text
224 × 224
```

---

## Reason for Selection

224×224 is the most common input size for:

- ResNet
- EfficientNet
- MobileNet
- Vision Transformers

This choice maximizes compatibility with future models.

---

## Output Shape

```python
(224, 224)
```

---

# Folder Structure

```text
video-preprocessing/
│
├── data/
│   ├── raw_videos/
│   └── processed_frames/
│
├── src/
│   ├── video_reader.py
│   ├── fps_extractor.py
│   ├── frame_extractor.py
│   ├── grayscale_converter.py
│   └── frame_cropper.py
│
├── tests/
│
├── docs/
│   └── PHASE_01_VIDEO_PREPROCESSING_PIPELINE.md
│
├── requirements.txt
│
└── README.md
```

---

# Expected Output

Input:

```text
video.mp4
```

Output:

```text
processed_frames/
├── frame_0001.jpg
├── frame_0002.jpg
├── frame_0003.jpg
...
```

and

```json
{
  "video_name": "video.mp4",
  "fps": 30,
  "total_frames": 450
}
```

---

# Success Criteria

Phase 1 will be considered complete when the pipeline can:

- Read MP4 files successfully
- Extract FPS information
- Extract all frames
- Convert frames to grayscale
- Crop and resize frames to 224×224
- Save processed frames
- Generate metadata logs

---

# Next Phase

After successful completion of Phase 1:

Phase 2 will focus on:

- Face Detection
- Face Alignment
- Facial Region Extraction
- Quality Filtering

before integrating pretrained computer vision models for depression detection.