import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class CubeExporter:
    """Exports a 3D LUT to .cube format compatible with major NLEs."""

    @staticmethod
    def export(lut: np.ndarray, file_path: str | Path, title: str = "Video LUT Generator") -> None:
        file_path = Path(file_path)
        size = lut.shape[0]

        if lut.shape != (size, size, size, 3):
            raise ValueError(
                f"LUT shape must be ({size}, {size}, {size}, 3), got {lut.shape}"
            )

        logger.info("Exporting %dx%dx%d LUT to %s", size, size, size, file_path)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"TITLE \"{title}\"\n")
            f.write(f"LUT_3D_SIZE {size}\n")
            f.write("DOMAIN_MIN 0.0 0.0 0.0\n")
            f.write("DOMAIN_MAX 1.0 1.0 1.0\n")
            f.write("\n")

            for b in range(size):
                for g in range(size):
                    for r in range(size):
                        r_val, g_val, b_val = lut[r, g, b]
                        f.write(f"{r_val:.6f} {g_val:.6f} {b_val:.6f}\n")

        logger.info("LUT exported to %s (%d entries)", file_path, size ** 3)

    @staticmethod
    def validate(file_path: str | Path) -> bool:
        file_path = Path(file_path)
        if not file_path.exists():
            return False

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            size = None
            data_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("LUT_3D_SIZE"):
                    size = int(stripped.split()[-1])
                elif stripped and not stripped.startswith("#") and not stripped.startswith("TITLE") and not stripped.startswith("DOMAIN"):
                    parts = stripped.split()
                    if len(parts) == 3:
                        data_lines.append(parts)

            if size is None:
                return False

            expected = size ** 3
            if len(data_lines) != expected:
                logger.warning(
                    "LUT validation: expected %d entries, got %d", expected, len(data_lines)
                )
                return False

            return True

        except Exception as e:
            logger.error("LUT validation failed: %s", e)
            return False
