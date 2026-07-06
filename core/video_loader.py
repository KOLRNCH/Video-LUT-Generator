import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoLoadError(Exception):
    """Raised when a video file cannot be loaded."""


class VideoLoader:
    """Loads video files and provides metadata and frame access."""

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 0.0
        self._frame_count: int = 0
        self._width: int = 0
        self._height: int = 0
        self._duration: float = 0.0
        self._open()

    def _open(self) -> None:
        if not self.file_path.exists():
            raise VideoLoadError(f"File not found: {self.file_path}")
        self._cap = cv2.VideoCapture(str(self.file_path))
        if not self._cap.isOpened():
            raise VideoLoadError(f"Cannot open video file: {self.file_path}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self._fps > 0:
            self._duration = self._frame_count / self._fps
        logger.info(
            "Loaded video: %s | %dx%d | %.2f fps | %d frames | %.1f sec",
            self.file_path.name,
            self._width,
            self._height,
            self._fps,
            self._frame_count,
            self._duration,
        )

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def duration(self) -> float:
        return self._duration

    def read_frame_at(self, position: int) -> Optional[np.ndarray]:
        if self._cap is None:
            return None
        if not self._cap.set(cv2.CAP_PROP_POS_FRAMES, position):
            logger.warning("Failed to seek to frame %d", position)
            return None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def read_frame_at_time(self, seconds: float) -> Optional[np.ndarray]:
        frame_idx = int(seconds * self._fps)
        return self.read_frame_at(frame_idx)

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.debug("VideoCapture released for %s", self.file_path.name)

    def __enter__(self) -> "VideoLoader":
        return self

    def __exit__(self, *args) -> None:
        self.release()
