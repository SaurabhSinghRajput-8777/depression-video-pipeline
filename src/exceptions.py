"""
Exceptions module for the Video Preprocessing Pipeline.

This module defines a custom exception hierarchy for the pipeline, ensuring
granular error handling and clear debugging paths.

Usage Example:
    from exceptions import VideoNotFoundError
    
    if not video_path.exists():
        raise VideoNotFoundError(f"Cannot find video at {video_path}")
"""

__all__ = [
    "PipelineError",
    "VideoError",
    "VideoNotFoundError",
    "VideoOpenError",
    "InvalidVideoFormatError",
    "FPSError",
    "InvalidFPSError",
    "FrameProcessingError",
    "FrameExtractionError",
    "GrayscaleConversionError",
    "ImagePreprocessingError",
    "FaceProcessingError",
    "FaceNotFoundError",
    "FaceValidationError",
    "FaceQualityError",
    "FaceAlignmentError",
    "MetadataWriteError",
]

class PipelineError(Exception):
    """Base class for all pipeline-related exceptions."""
    pass

class VideoError(PipelineError):
    """Base class for video-related errors."""
    pass

class VideoNotFoundError(VideoError):
    """Raised when a video file does not exist on disk."""
    pass

class VideoOpenError(VideoError):
    """Raised when OpenCV fails to open or read a video file."""
    pass

class InvalidVideoFormatError(VideoError):
    """Raised when the video file extension is not supported."""
    pass

class FPSError(PipelineError):
    """Base class for FPS-related errors."""
    pass

class InvalidFPSError(FPSError):
    """Raised when the extracted FPS is <= 0 or invalid."""
    pass

class FrameProcessingError(PipelineError):
    """Base class for frame processing errors."""
    pass

class FrameExtractionError(FrameProcessingError):
    """Failure while extracting frames."""
    pass

class GrayscaleConversionError(FrameProcessingError):
    """Failure during grayscale conversion."""
    pass

class ImagePreprocessingError(FrameProcessingError):
    """Failure during crop/resize operations."""
    pass

class FaceProcessingError(PipelineError):
    """Base class for face extraction pipeline errors."""
    pass

class FaceNotFoundError(FaceProcessingError):
    """Raised when no face is detected in the frame."""
    pass

class FaceValidationError(FaceProcessingError):
    """Raised when a detected face fails validation rules."""
    pass

class FaceQualityError(FaceProcessingError):
    """Raised when a valid face fails image quality standards (e.g., blur)."""
    pass

class FaceAlignmentError(FaceProcessingError):
    """Raised when affine alignment fails."""
    pass

class MetadataWriteError(PipelineError):
    """Raised when the metadata JSON file fails to save."""
    pass
