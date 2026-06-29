import cv2
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

MAX_DEBUG_SAMPLES = 50

class DebugLogger:
    """
    Handles saving diagnostic samples of rejected frames.
    Implements a strict global cap per category to prevent storage explosion.
    """
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir / "debug"
        self.category_counts = {}
        
    def save_debug_sample(self, image: np.ndarray, category: str, video_name: str, frame_index: int) -> None:
        """
        Saves a debug image if the global cap for its category has not been reached.
        """
        if category not in self.category_counts:
            category_dir = self.base_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
            # Initialize count based on existing files across pipeline runs
            self.category_counts[category] = len(list(category_dir.glob("*.jpg")))
            
        if self.category_counts[category] >= MAX_DEBUG_SAMPLES:
            return
            
        category_dir = self.base_dir / category
        stem = Path(video_name).stem
        filename = f"{stem}_frame_{frame_index:06d}.jpg"
        out_path = category_dir / filename
        
        try:
            cv2.imwrite(str(out_path), image)
            self.category_counts[category] += 1
            logger.debug("Saved debug sample %s (Total: %d/%d)", 
                         out_path, self.category_counts[category], MAX_DEBUG_SAMPLES)
        except Exception as e:
            logger.warning("Failed to save debug sample %s: %s", out_path, e)
