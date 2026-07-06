import logging

import numpy as np

logger = logging.getLogger(__name__)


class HistogramMatcher:
    """Matches histograms between images in LAB color space."""

    @staticmethod
    def match_histograms_lab(source_lab: np.ndarray, target_lab: np.ndarray) -> np.ndarray:
        source_float = source_lab.astype(np.float32)
        target_float = target_lab.astype(np.float32)

        result = np.zeros_like(source_float)
        for c in range(3):
            src_ch = source_float[:, :, c]
            tgt_ch = target_float[:, :, c]
            result[:, :, c] = HistogramMatcher._match_channel(src_ch, tgt_ch)

        return result

    @staticmethod
    def _match_channel(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        src_flat = source.ravel()
        tgt_flat = target.ravel()

        src_sorted = np.sort(src_flat)
        tgt_sorted = np.sort(tgt_flat)

        src_indices = np.argsort(src_flat)

        n = len(src_flat)
        tgt_positions = np.linspace(0, n - 1, n, dtype=int)

        matched = tgt_sorted[tgt_positions]

        result = np.empty_like(src_flat)
        result[src_indices] = matched

        return result.reshape(source.shape)

    @staticmethod
    def match_with_lut(source_channel: np.ndarray, target_channel: np.ndarray) -> np.ndarray:
        src_flat = source_channel.ravel().astype(np.int32)
        tgt_flat = target_channel.ravel().astype(np.int32)

        src_hist = np.bincount(src_flat, minlength=256)[:256]
        tgt_hist = np.bincount(tgt_flat, minlength=256)[:256]

        src_cdf = np.cumsum(src_hist).astype(np.float64)
        tgt_cdf = np.cumsum(tgt_hist).astype(np.float64)

        src_cdf /= src_cdf[-1] if src_cdf[-1] > 0 else 1
        tgt_cdf /= tgt_cdf[-1] if tgt_cdf[-1] > 0 else 1

        lut = np.zeros(256, dtype=np.uint8)
        tgt_idx = 0
        for src_val in range(256):
            while tgt_idx < 255 and tgt_cdf[tgt_idx] < src_cdf[src_val]:
                tgt_idx += 1
            lut[src_val] = tgt_idx

        return lut[src_flat].reshape(source_channel.shape)
