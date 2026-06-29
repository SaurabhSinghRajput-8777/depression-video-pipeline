"""
Tests for the exceptions module.
"""

import sys
from pathlib import Path

# Ensure src is in the python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pytest
from exceptions import (
    PipelineError,
    VideoError,
    VideoNotFoundError,
    VideoOpenError,
    InvalidVideoFormatError,
    FPSError,
    InvalidFPSError,
    FrameProcessingError,
    FrameExtractionError,
    GrayscaleConversionError,
    ImagePreprocessingError,
    MetadataWriteError
)

def test_exception_hierarchy():
    """Verify that exceptions correctly inherit from the base PipelineError."""
    # Base error should inherit from Exception
    assert issubclass(PipelineError, Exception)
    
    # Sub-bases should inherit from PipelineError
    assert issubclass(VideoError, PipelineError)
    assert issubclass(FPSError, PipelineError)
    assert issubclass(FrameProcessingError, PipelineError)
    
    # Specific video errors
    assert issubclass(VideoNotFoundError, VideoError)
    assert issubclass(VideoOpenError, VideoError)
    assert issubclass(InvalidVideoFormatError, VideoError)
    
    # Specific FPS errors
    assert issubclass(InvalidFPSError, FPSError)
    
    # Specific frame processing errors
    assert issubclass(FrameExtractionError, FrameProcessingError)
    assert issubclass(GrayscaleConversionError, FrameProcessingError)
    assert issubclass(ImagePreprocessingError, FrameProcessingError)
    
    # Flat errors
    assert issubclass(MetadataWriteError, PipelineError)

def test_exception_raising():
    """Verify that exceptions can be raised and caught correctly."""
    with pytest.raises(PipelineError):
        raise VideoNotFoundError("Test error")
        
    with pytest.raises(FrameProcessingError):
        raise GrayscaleConversionError("Test error")
        
    with pytest.raises(ImagePreprocessingError, match="Invalid bounds"):
        raise ImagePreprocessingError("Invalid bounds")
