"""
Data model for frames flowing through the Video Preprocessing Pipeline.

This module defines the `FrameRecord` dataclass, encapsulating both
the image payload and its corresponding metadata (index, timestamp)
to provide a consistent interface between pipeline steps.

Usage Example:
    from models.frame_record import FrameRecord
    import numpy as np

    record = FrameRecord(
        frame_index=150,
        timestamp_seconds=5.0,
        image=np.zeros((720, 1280, 3), dtype=np.uint8)
    )
    print(record.shape)  # Outputs: (720, 1280, 3)
"""

from dataclasses import dataclass
import numpy as np
from typing import Any
from numpy.typing import NDArray

@dataclass(slots=True)
class FrameRecord:
    """
    Represents a single video frame along with its associated metadata.
    
    Attributes:
        frame_index (int): The 0-based sequential index of the frame.
        timestamp_seconds (float): The playback timestamp of the frame in seconds.
        image (NDArray[Any]): The actual image data (BGR, Grayscale, etc.).
    """
    frame_index: int
    timestamp_seconds: float
    image: NDArray[Any]

    def __post_init__(self) -> None:
        """Validates the inputs upon instantiation."""
        if self.frame_index < 0:
            raise ValueError("frame_index must be >= 0")
        
        if self.timestamp_seconds < 0:
            raise ValueError("timestamp_seconds must be >= 0")
            
        if not isinstance(self.image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the underlying image array."""
        return self.image.shape
