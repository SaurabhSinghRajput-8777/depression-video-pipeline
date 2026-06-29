import numpy as np

# Canonical ArcFace 112x112 alignment template
# This defines the optimal reference coordinates for 5-point facial landmarks.
REFERENCE_POINTS = np.array([
    [38.2946, 51.6963],  # left eye
    [73.5318, 51.5014],  # right eye
    [56.0252, 71.7366],  # nose
    [41.5493, 92.3655],  # left mouth
    [70.7299, 92.2041],  # right mouth
], dtype=np.float32)

# Output shape for affine alignment
ALIGNED_FACE_SIZE = (112, 112)
