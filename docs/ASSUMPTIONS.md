# Assumptions

The following assumptions constrain the Phase 1 implementation of the Video Preprocessing Pipeline.

1. **Input videos are MP4**: The pipeline expects `.mp4` video files. Other formats are not officially supported in this phase.
2. **Videos are readable by OpenCV**: The system assumes the host machine has the necessary codecs for OpenCV to successfully decode the input videos.
3. **Frames fit in memory during processing**: While frames are yielded one by one sequentially, it is assumed that individual frames easily fit into the available system RAM.
4. **Target Resolution Configurable**: Downstream models will expect specific spatial dimensions. The target size is fully configurable via `config.py`, with 224x224 as the initial default.
5. **Phase 1 uses center crop only**: Active face detection and alignment are explicitly out of scope for this phase. The `image_preprocessor.py` will extract the exact center of the video frame.
6. **Local File System**: Input and output operations assume a local file system layout (`data/raw/`, `data/processed/`, `data/metadata/`). Cloud storage integrations are not part of Phase 1.
