import logging
from typing import List, Optional

import cv2
import numpy as np

from .video_loader import VideoLoader

logger = logging.getLogger(__name__)

TARGET_SIZE = (256, 144)


class FrameExtractor:
    """Extracts uniformly sampled frames from a video for color analysis."""

    def __init__(
        self,
        num_frames: int = 75,
        target_size: tuple[int, int] = TARGET_SIZE,
    ) -> None:
        self.num_frames = num_frames
        self.target_size = target_size

    def extract(self, loader: VideoLoader) -> List[np.ndarray]:
        total = loader.frame_count
        if total <= 0:
            logger.warning("Video has no frames")
            return []

        actual_count = min(self.num_frames, total)
        indices = np.linspace(0, total - 1, actual_count, dtype=int)

        frames: List[np.ndarray] = []
        for idx in indices:
            try:
                frame = loader.read_frame_at(int(idx))
                if frame is not None:
                    resized = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
                    frames.append(np.asarray(resized, dtype=np.uint8))
            except Exception as e:
                logger.error("Failed to extract frame at index %d: %s", idx, e)

        if not frames:
            logger.warning("No frames could be extracted")
            return []

        logger.info("Extracted %d/%d frames sequentially", len(frames), actual_count)
        return frames
