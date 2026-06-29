# Design Decisions

This document outlines the rationale behind the key technical choices made during Phase 1 of the Video Preprocessing Pipeline.

## 1. Config-Driven Architecture
Instead of hardcoding dimensions or parameters, the pipeline relies on a centralized `config.py`. 
* **Flexibility**: Researchers can easily switch target dimensions without modifying the core logic.
* **Maintainability**: A single source of truth prevents parameter drift.

## 2. Dataset-Agnostic Storage
The data folder structure separates `raw/`, `processed/`, and `metadata/` files.
* **Scalability**: Ensures the pipeline can cleanly handle diverse datasets.
* **Separation of Concerns**: Metadata files are kept isolated from processed frames for easier parsing.

## 3. Explicit Error Handling Strategy
The pipeline defines custom exceptions.
* **Debuggability**: Research code often processes thousands of videos. Explicit exceptions provide a clear audit trail.
* **Robustness**: Allows the pipeline to catch specific errors and potentially continue processing the next video.

## 4. FrameRecord Dataclass
Instead of passing raw `np.ndarray` objects between modules, the pipeline uses a `FrameRecord` dataclass.
* **Extensibility**: By Phase 2, frames will require metadata like timestamps, original dimensions, and bounding boxes. This structure allows seamless additions without breaking APIs.
* **Production Standard**: This is a common and highly effective pattern in production ML pipelines.

## 5. Comprehensive & Versioned Metadata Logging
The metadata schema includes detailed properties, processing statistics, and strict versioning (`pipeline_phase`, `pipeline_name`, `preprocessing_version`).
* **Reproducibility**: Captures the exact state of the video and the pipeline phase that produced it. Future phases generating their own metadata will not conflict or create ambiguity.
* **Future-Proofing**: Tracks failed and skipped frames, critical for later phases where face detectors may reject frames.

## 6. Renaming Cropper to Image Preprocessor
The cropping module is named `image_preprocessor.py`.
* **Evolution**: While Phase 1 uses a naive center crop, Phase 2 will introduce face-based cropping. A generalized module name prevents a future architectural refactor.

## 7. Why OpenCV?
OpenCV (`cv2`) was selected as the core computer vision library.
* **Popularity and Community**: It is the industry standard for computer vision.
* **Stability**: Highly optimized C++ backend.

## 8. Why Grayscale?
Frames are converted to grayscale immediately after extraction.
* **Memory Reduction**: Reduces the memory footprint per frame by 66%.
* **Standardization**: Removes color bias and illumination artifacts, aiding model generalization.
