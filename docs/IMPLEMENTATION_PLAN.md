# Implementation Plan

This document breaks down the Phase 1 implementation into manageable tasks.

## Implementation Order
As per the approved project plan, modules will be implemented, tested, and reviewed one at a time in the following order:
1. `config.py`
2. `exceptions.py`
3. `frame_record.py`
4. `video_reader.py`
5. `fps_extractor.py`
6. `frame_extractor.py`
7. `grayscale_converter.py`
8. `image_preprocessor.py`
9. `metadata_logger.py`
10. `pipeline.py`
11. `tests/`
12. `README.md`

## Module Details

### Task 1: Configuration & Error System
* **Files**: `config.py`, `exceptions.py`.
* **Details**: Define dimensions, paths, and custom exceptions (`VideoNotFoundError`, `VideoOpenError`, `InvalidFPSError`, `FrameProcessingError`, `MetadataWriteError`).

### Task 2: Data Models
* **Files**: `models/frame_record.py`.
* **Details**: Define `FrameRecord` dataclass with `frame_index`, `timestamp_seconds`, and `image`.

### Task 3: Video Reader
* **Objective**: Safely open a video file and return a capture object.
* **Validation Checks**: Raise `VideoNotFoundError` if missing. Raise `VideoOpenError` on cv2 failure.

### Task 4: FPS & Dimension Extractor
* **Objective**: Retrieve FPS, width, height, and calculate duration.
* **Validation Checks**: Raise `InvalidFPSError` if FPS <= 0.

### Task 5: Frame Extraction
* **Objective**: Provide a generator that yields frames sequentially.
* **Outputs**: Generator yielding `FrameRecord` objects.

### Task 6: Grayscale Conversion
* **Objective**: Convert a multi-channel frame to single-channel.
* **Inputs/Outputs**: Modifies or returns a `FrameRecord`.

### Task 7: Image Preprocessing (Standardization)
* **Objective**: Standardize the frame (center crop & resize).
* **Validation Checks**: Raise `FrameProcessingError` if frame boundaries are invalid.

### Task 8: Metadata & Audit Logging
* **Objective**: Generate a detailed JSON summary and maintain a research audit log.
* **Outputs**: Writes JSON metadata with fields: `pipeline_phase`, `pipeline_name`, `preprocessing_version`, `video_name`, `fps`, `total_frames`, `width`, `height`, `duration_seconds`, `processed_frames`, `failed_frames`, `skipped_frames`, `output_size`.

### Task 9: CLI Runner (Pipeline Orchestrator)
* **Objective**: Tie all modules together into an executable script.
* **CLI Interface**: `python src/pipeline.py --input ... --output ... --metadata ...`
