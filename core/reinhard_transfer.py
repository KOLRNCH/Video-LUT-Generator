import logging

import cv2
import numpy as np
from numba import jit

logger = logging.getLogger(__name__)


class ReinhardTransfer:
    """Implements Reinhard et al. color transfer in LAB space."""

    @staticmethod
    def transfer(
        source_lab: np.ndarray,
        src_mean: np.ndarray,
        src_std: np.ndarray,
        tgt_mean: np.ndarray,
        tgt_std: np.ndarray,
    ) -> np.ndarray:
        result = np.zeros_like(source_lab, dtype=np.float32)

        for c in range(3):
            src_std_c = src_std[c] if src_std[c] > 1e-6 else 1.0
            tgt_std_c = tgt_std[c] if tgt_std[c] > 1e-6 else 1.0

            result[:, :, c] = (source_lab[:, :, c].astype(np.float32) - src_mean[c]) * (
                tgt_std_c / src_std_c
            ) + tgt_mean[c]

        return result

    @staticmethod
    def transfer_tonal_separated(
        source_lab: np.ndarray,
        ref_stats: dict,
        tgt_stats: dict,
        strength: float = 1.0,
    ) -> np.ndarray:
        result = source_lab.astype(np.float32).copy()
        h, w = source_lab.shape[:2]
        l_ch = result[:, :, 0]

        weights = ReinhardTransfer._compute_tonal_weights(l_ch)

        for region, weight_map in [("shadows", weights[0]), ("midtones", weights[1]), ("highlights", weights[2])]:
            src_mean = ref_stats[f"{region}_mean_lab"]
            src_std = ref_stats[f"{region}_std_lab"]
            tgt_mean = tgt_stats[f"{region}_mean_lab"]
            tgt_std = tgt_stats[f"{region}_std_lab"]

            for c in range(3):
                src_std_c = src_std[c] if src_std[c] > 1e-6 else 1.0
                tgt_std_c = tgt_std[c] if tgt_std[c] > 1e-6 else 1.0

                adjusted = (result[:, :, c] - src_mean[c]) * (tgt_std_c / src_std_c) + tgt_mean[c]
                blend = weight_map * strength
                result[:, :, c] = result[:, :, c] * (1.0 - blend) + adjusted * blend

        return result

    @staticmethod
    @jit(nopython=True)
    def _compute_tonal_weights(l_channel: np.ndarray) -> tuple:
        h, w = l_channel.shape
        shadows = np.zeros((h, w), dtype=np.float32)
        midtones = np.zeros((h, w), dtype=np.float32)
        highlights = np.zeros((h, w), dtype=np.float32)

        for y in range(h):
            for x in range(w):
                l = l_channel[y, x]
                if l < 30:
                    shadows[y, x] = 1.0
                elif l > 70:
                    highlights[y, x] = 1.0
                else:
                    mid_val = (l - 30) / 40.0
                    midtones[y, x] = mid_val
                    if l < 40:
                        shadows[y, x] = 1.0 - mid_val
                    elif l > 60:
                        highlights[y, x] = mid_val

        return shadows, midtones, highlights
