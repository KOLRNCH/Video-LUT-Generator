import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from numba import jit

logger = logging.getLogger(__name__)


@dataclass
class ColorStats:
    mean_rgb: np.ndarray = field(default_factory=lambda: np.zeros(3))
    std_rgb: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mean_lab: np.ndarray = field(default_factory=lambda: np.zeros(3))
    std_lab: np.ndarray = field(default_factory=lambda: np.zeros(3))
    brightness: float = 0.0
    contrast: float = 0.0
    saturation: float = 0.0
    color_temperature: float = 6500.0
    white_balance_r: float = 1.0
    white_balance_g: float = 1.0
    white_balance_b: float = 1.0
    hist_r: np.ndarray = field(default_factory=lambda: np.zeros(256))
    hist_g: np.ndarray = field(default_factory=lambda: np.zeros(256))
    hist_b: np.ndarray = field(default_factory=lambda: np.zeros(256))
    hist_l: np.ndarray = field(default_factory=lambda: np.zeros(256))
    hist_a: np.ndarray = field(default_factory=lambda: np.zeros(256))
    hist_b_lab: np.ndarray = field(default_factory=lambda: np.zeros(256))
    shadows_mean_lab: np.ndarray = field(default_factory=lambda: np.zeros(3))
    midtones_mean_lab: np.ndarray = field(default_factory=lambda: np.zeros(3))
    highlights_mean_lab: np.ndarray = field(default_factory=lambda: np.zeros(3))
    shadows_std_lab: np.ndarray = field(default_factory=lambda: np.zeros(3))
    midtones_std_lab: np.ndarray = field(default_factory=lambda: np.zeros(3))
    highlights_std_lab: np.ndarray = field(default_factory=lambda: np.zeros(3))


class ColorAnalyzer:
    """Performs comprehensive color analysis on a set of frames."""

    def __init__(self) -> None:
        self.stats: Optional[ColorStats] = None

    def analyze(self, frames: List[np.ndarray]) -> ColorStats:
        if not frames:
            raise ValueError("No frames to analyze")

        all_rgb = np.concatenate([f.reshape(-1, 3) for f in frames], axis=0).astype(np.float32)

        all_lab_list = []
        for f in frames:
            lab_u8 = cv2.cvtColor(f, cv2.COLOR_RGB2Lab).astype(np.float32)
            lab_u8[:, :, 0] = lab_u8[:, :, 0] * 100.0 / 255.0
            lab_u8[:, :, 1] -= 128.0
            lab_u8[:, :, 2] -= 128.0
            all_lab_list.append(lab_u8.reshape(-1, 3))
        all_lab = np.concatenate(all_lab_list, axis=0)

        mean_rgb = np.mean(all_rgb, axis=0)
        std_rgb = np.std(all_rgb, axis=0)
        mean_lab = np.mean(all_lab, axis=0)
        std_lab = np.std(all_lab, axis=0)

        brightness = float(mean_lab[0])
        contrast = float(std_lab[0])
        saturation = float(np.mean(np.sqrt(all_lab[:, 1] ** 2 + all_lab[:, 2] ** 2)))

        temp = self._estimate_color_temperature(mean_rgb)
        wb = self._estimate_white_balance(mean_rgb)

        hist_r = np.zeros(256, dtype=np.float32)
        hist_g = np.zeros(256, dtype=np.float32)
        hist_b = np.zeros(256, dtype=np.float32)
        hist_l = np.zeros(256, dtype=np.float32)
        hist_a = np.zeros(256, dtype=np.float32)
        hist_bl = np.zeros(256, dtype=np.float32)

        for f in frames:
            hist_r += self._compute_histogram(f[:, :, 0])
            hist_g += self._compute_histogram(f[:, :, 1])
            hist_b += self._compute_histogram(f[:, :, 2])
            lab_f = cv2.cvtColor(f, cv2.COLOR_RGBToLab)
            hist_l += self._compute_histogram(lab_f[:, :, 0])
            hist_a += self._compute_histogram(lab_f[:, :, 1])
            hist_bl += self._compute_histogram(lab_f[:, :, 2])

        def norm_hist(h):
            s = h.sum()
            return h / s if s > 0 else h

        hist_r = norm_hist(hist_r)
        hist_g = norm_hist(hist_g)
        hist_b = norm_hist(hist_b)
        hist_l = norm_hist(hist_l)
        hist_a = norm_hist(hist_a)
        hist_bl = norm_hist(hist_bl)

        shd_mean, mid_mean, hli_mean = self._compute_tonal_stats(all_lab)
        shd_std, mid_std, hli_std = self._compute_tonal_std(all_lab, shd_mean, mid_mean, hli_mean)

        self.stats = ColorStats(
            mean_rgb=mean_rgb,
            std_rgb=std_rgb,
            mean_lab=mean_lab,
            std_lab=std_lab,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            color_temperature=temp,
            white_balance_r=wb[0],
            white_balance_g=wb[1],
            white_balance_b=wb[2],
            hist_r=hist_r,
            hist_g=hist_g,
            hist_b=hist_b,
            hist_l=hist_l,
            hist_a=hist_a,
            hist_b_lab=hist_bl,
            shadows_mean_lab=shd_mean,
            midtones_mean_lab=mid_mean,
            highlights_mean_lab=hli_mean,
            shadows_std_lab=shd_std,
            midtones_std_lab=mid_std,
            highlights_std_lab=hli_std,
        )

        logger.info(
            "Color analysis complete: brightness=%.1f, contrast=%.1f, saturation=%.1f, temp=%.0fK",
            brightness,
            contrast,
            saturation,
            temp,
        )
        return self.stats

    @staticmethod
    @jit(nopython=True)
    def _compute_histogram(channel: np.ndarray) -> np.ndarray:
        hist = np.zeros(256, dtype=np.float32)
        h, w = channel.shape
        for y in range(h):
            for x in range(w):
                idx = int(channel[y, x])
                if 0 <= idx < 256:
                    hist[idx] += 1.0
        return hist

    @staticmethod
    def _estimate_color_temperature(mean_rgb: np.ndarray) -> float:
        r, g, b = mean_rgb
        if b == 0:
            b = 1
        ratio = r / b
        if ratio > 1.0:
            temp = 4500.0 + (ratio - 1.0) * 3000.0
        else:
            temp = 6500.0 + (1.0 - ratio) * 2000.0
        return float(np.clip(temp, 2000.0, 15000.0))

    @staticmethod
    def _estimate_white_balance(mean_rgb: np.ndarray) -> np.ndarray:
        max_val = np.max(mean_rgb)
        if max_val == 0:
            return np.ones(3)
        return max_val / mean_rgb

    @staticmethod
    def _compute_tonal_stats(lab_pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        l_channel = lab_pixels[:, 0]
        shd_mask = l_channel < 30
        mid_mask = (l_channel >= 30) & (l_channel <= 70)
        hli_mask = l_channel > 70

        shd_mean = np.mean(lab_pixels[shd_mask], axis=0) if np.any(shd_mask) else np.zeros(3)
        mid_mean = np.mean(lab_pixels[mid_mask], axis=0) if np.any(mid_mask) else np.zeros(3)
        hli_mean = np.mean(lab_pixels[hli_mask], axis=0) if np.any(hli_mask) else np.zeros(3)

        return shd_mean, mid_mean, hli_mean

    @staticmethod
    def _compute_tonal_std(
        lab_pixels: np.ndarray,
        shd_mean: np.ndarray,
        mid_mean: np.ndarray,
        hli_mean: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        l_channel = lab_pixels[:, 0]
        shd_mask = l_channel < 30
        mid_mask = (l_channel >= 30) & (l_channel <= 70)
        hli_mask = l_channel > 70

        def safe_std(pixels, mask, mean):
            m = mask
            if np.any(m):
                return np.std(pixels[m], axis=0)
            return np.zeros(3)

        shd_std = safe_std(lab_pixels, shd_mask, shd_mean)
        mid_std = safe_std(lab_pixels, mid_mask, mid_mean)
        hli_std = safe_std(lab_pixels, hli_mask, hli_mean)

        return shd_std, mid_std, hli_std
