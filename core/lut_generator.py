import logging
from typing import List, Optional

import cv2
import numpy as np
from numba import jit

from .color_analysis import ColorStats

logger = logging.getLogger(__name__)


class LUTGenerator:
    """Generates a 3D LUT for color style transfer using pixel-level analysis."""

    SUPPORTED_SIZES = [17, 33, 65]

    def __init__(self, size: int = 33) -> None:
        if size not in self.SUPPORTED_SIZES:
            raise ValueError(f"LUT size must be one of {self.SUPPORTED_SIZES}, got {size}")
        self.size = size
        self._lut: Optional[np.ndarray] = None
        self._last_quality: Optional[dict] = None

    @property
    def lut(self) -> Optional[np.ndarray]:
        return self._lut

    @property
    def last_quality(self) -> Optional[dict]:
        return self._last_quality

    def generate(
        self,
        ref_stats: ColorStats,
        tgt_stats: ColorStats,
        strength: float = 1.0,
        tgt_frames: Optional[List[np.ndarray]] = None,
        ref_frames: Optional[List[np.ndarray]] = None,
        mode: str = "natural",
    ) -> np.ndarray:
        logger.info("Generating %dx%dx%d LUT, strength=%.2f, mode=%s", self.size, self.size, self.size, strength, mode)
        s = self.size

        if tgt_frames is not None and len(tgt_frames) > 0:
            logger.info("Building LUT from pixel data (%d target frames)", len(tgt_frames))
            if mode == "aggressive":
                rgb_out = self._build_aggressive(tgt_frames, ref_frames, ref_stats, tgt_stats, strength, s)
            else:
                rgb_out = self._build_natural(tgt_frames, ref_frames, ref_stats, tgt_stats, strength, s)
        else:
            logger.info("Building LUT from stats (no frame data)")
            rgb_out = self._build_from_stats(ref_stats, tgt_stats, strength, s)

        lut_3d = rgb_out.reshape((s, s, s, 3))
        lut_3d[0, 0, 0] = 0.0
        lut_3d[-1, -1, -1] = 1.0

        self._last_quality = self._compute_quality(lut_3d)
        q = self._last_quality
        logger.info(
            "LUT quality: total=%.2f, sat_ratio=%.2f, gray_drift=%.2f, lum_dist=%.2f, gamut_excess=%.4f",
            q["total"], q["sat_ratio"], q["gray_drift"], q["lum_distortion"], q["gamut_excess"],
        )

        self._lut = lut_3d
        logger.info("LUT generation complete")
        return self._lut

    def _compute_quality(self, lut: np.ndarray) -> dict:
        s = lut.shape[0]
        mid = s // 2

        gray_line = lut[:, mid, mid]
        gray_drift = float(np.mean(np.abs(gray_line - np.linspace(0, 1, s)[:, None])).max())

        rg = lut[:, :, :, 0] - lut[:, :, :, 1]
        yb = (lut[:, :, :, 0] + lut[:, :, :, 1]) / 2.0 - lut[:, :, :, 2]
        sat = np.sqrt(rg ** 2 + yb ** 2)
        sat_mean = float(np.mean(sat))

        identity = np.linspace(0, 1, s, dtype=np.float32)
        rr, gg, bb = np.meshgrid(identity, identity, identity, indexing="ij")
        id_rgb = np.stack([rr, gg, bb], axis=-1)
        id_rg = id_rgb[:, :, :, 0] - id_rgb[:, :, :, 1]
        id_yb = (id_rgb[:, :, :, 0] + id_rgb[:, :, :, 1]) / 2.0 - id_rgb[:, :, :, 2]
        id_sat = np.sqrt(id_rg ** 2 + id_yb ** 2)
        id_sat_mean = float(np.mean(id_sat))

        sat_ratio = sat_mean / max(id_sat_mean, 1e-6)
        sat_quality = 1.0 - min(abs(sat_ratio - 1.0) * 2.0, 1.0)

        diff = lut - id_rgb
        lum_distortion = float(np.mean(np.abs(diff[:, :, :, 0])))

        gamut_mask = (lut < 0) | (lut > 1)
        gamut_excess = float(np.mean(np.abs(lut - np.clip(lut, 0, 1))[gamut_mask])) if np.any(gamut_mask) else 0.0

        gray_quality = 1.0 - min(gray_drift * 5.0, 1.0)
        lum_quality = 1.0 - min(lum_distortion * 5.0, 1.0)
        gamut_quality = 1.0 - min(gamut_excess * 10.0, 1.0)

        total = sat_quality * 0.35 + gray_quality * 0.30 + lum_quality * 0.20 + gamut_quality * 0.15

        return {
            "total": float(np.clip(total, 0, 1)),
            "sat_ratio": float(sat_ratio),
            "gray_drift": float(gray_drift),
            "lum_distortion": float(lum_distortion),
            "gamut_excess": float(gamut_excess),
        }

    def quality_text(self) -> str:
        if self._last_quality is None:
            return "Quality: N/A"
        q = self._last_quality
        total = q["total"]
        label = "Excellent" if total > 0.85 else "Good" if total > 0.7 else "Fair" if total > 0.5 else "Poor"
        return f"Quality: {label} ({total:.2f}) | Sat: {q['sat_ratio']:.2f}x | Gray drift: {q['gray_drift']:.3f}"

    def _pca_transfer(self, lab_pixels: np.ndarray, ref_lab: np.ndarray, tgt_lab: np.ndarray) -> np.ndarray:
        ref_mean = np.mean(ref_lab, axis=0)
        tgt_mean = np.mean(tgt_lab, axis=0)

        def robust_cov(data, mean):
            centered = data - mean
            cov = (centered.T @ centered) / max(data.shape[0] - 1, 1)
            cov += np.eye(3) * 1e-6
            return cov

        cov_t = robust_cov(tgt_lab, tgt_mean)
        cov_r = robust_cov(ref_lab, ref_mean)

        evals_t, evecs_t = np.linalg.eigh(cov_t)
        evals_r, evecs_r = np.linalg.eigh(cov_r)

        sqrt_evals_t = np.sqrt(np.maximum(evals_t, 1e-10))
        sqrt_evals_r = np.sqrt(np.maximum(evals_r, 1e-10))

        centered = lab_pixels - tgt_mean
        transformed = (
            centered @ evecs_t @ np.diag(1.0 / sqrt_evals_t)
            @ np.diag(sqrt_evals_r) @ evecs_r.T
            + ref_mean
        )
        return transformed

    @staticmethod
    def _soft_clip_gamut(rgb: np.ndarray) -> np.ndarray:
        gray = np.full_like(rgb, 0.5)
        max_excess = np.maximum(np.maximum(-rgb.min(axis=1, keepdims=True), (rgb - 1.0).max(axis=1, keepdims=True)), 0)
        factor = 1.0 / (1.0 + max_excess * 3.0)
        return gray + (rgb - gray) * factor

    @staticmethod
    def _adaptive_saturation_clamp(lab_out: np.ndarray, lab_in: np.ndarray, max_change: float = 30.0, max_ratio: float = 1.5) -> np.ndarray:
        delta_a = lab_out[:, 1] - lab_in[:, 1]
        delta_b = lab_out[:, 2] - lab_in[:, 2]
        chroma_in = np.sqrt(lab_in[:, 1] ** 2 + lab_in[:, 2] ** 2)
        chroma_out = np.sqrt(lab_out[:, 1] ** 2 + lab_out[:, 2] ** 2)
        ratio = np.ones_like(chroma_in)
        mask = chroma_in > 1.0
        if np.any(mask):
            ratio[mask] = chroma_out[mask] / chroma_in[mask]
        ratio = np.minimum(ratio, max_ratio)
        result = lab_out.copy()
        clamped_a = np.clip(delta_a, -max_change, max_change)
        clamped_b = np.clip(delta_b, -max_change, max_change)
        result[:, 1] = lab_in[:, 1] + clamped_a
        result[:, 2] = lab_in[:, 2] + clamped_b
        return result

    def _build_natural(
        self,
        tgt_frames: List[np.ndarray],
        ref_frames: Optional[List[np.ndarray]],
        ref_stats: ColorStats,
        tgt_stats: ColorStats,
        strength: float,
        s: int,
    ) -> np.ndarray:
        MAX_PIXELS = 50000

        tgt_pixels_rgb = np.concatenate([f.reshape(-1, 3) for f in tgt_frames], axis=0).astype(np.float32)
        rng = np.random.default_rng(42)
        if len(tgt_pixels_rgb) > MAX_PIXELS:
            idx = rng.choice(len(tgt_pixels_rgb), MAX_PIXELS, replace=False)
            tgt_pixels_rgb = tgt_pixels_rgb[idx]

        tgt_u8 = tgt_pixels_rgb.astype(np.uint8)
        lab_scaled = cv2.cvtColor(tgt_u8.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        tgt_lab = lab_scaled.copy()
        tgt_lab[:, 0] = lab_scaled[:, 0] * 100.0 / 255.0
        tgt_lab[:, 1] = lab_scaled[:, 1] - 128.0
        tgt_lab[:, 2] = lab_scaled[:, 2] - 128.0

        # --- Histogram matching (primary) ---
        hist_lut_l, hist_lut_a, hist_lut_b = self._build_histogram_luts(ref_stats, tgt_stats)
        lab_hm = tgt_lab.copy()
        for i in range(tgt_lab.shape[0]):
            li = int(np.clip(tgt_lab[i, 0] * 255.0 / 100.0, 0, 255))
            ai = int(np.clip(tgt_lab[i, 1] + 128.0, 0, 255))
            bi = int(np.clip(tgt_lab[i, 2] + 128.0, 0, 255))
            lab_hm[i, 0] = hist_lut_l[li] * 100.0 / 255.0
            lab_hm[i, 1] = hist_lut_a[ai] - 128.0
            lab_hm[i, 2] = hist_lut_b[bi] - 128.0

        # --- Gentle per-channel Reinhard per tonal zone ---
        lab_rh = tgt_lab.copy()
        l_vals = tgt_lab[:, 0]
        for c in range(3):
            s_in = tgt_stats.std_lab[c] if tgt_stats.std_lab[c] > 1e-6 else 1.0
            s_out = ref_stats.std_lab[c] if ref_stats.std_lab[c] > 1e-6 else 1.0
            ratio = np.clip(s_out / s_in, 0.5, 2.0)
            adj = 1.0 + (ratio - 1.0) * strength * 0.4
            lab_rh[:, c] = (tgt_lab[:, c] - tgt_stats.mean_lab[c]) * adj + ref_stats.mean_lab[c]

        # --- Tonal zone blend (trust HM more for midtones, Reinhard for extremes) ---
        shd_w = np.clip((20.0 - l_vals) / 20.0, 0.0, 1.0)
        hli_w = np.clip((l_vals - 80.0) / 20.0, 0.0, 1.0)
        mid_w = 1.0 - shd_w - hli_w
        lab_mixed = lab_hm * (0.6 + mid_w[:, None] * 0.3) + lab_rh * (0.4 - mid_w[:, None] * 0.3)

        # --- Luminance protection ---
        protect = self._luminance_protect_weights(tgt_lab[:, 0].reshape(-1, 1)).ravel()
        lab_mixed[:, 0] = tgt_lab[:, 0] * protect + lab_mixed[:, 0] * (1.0 - protect)

        # --- Conservative saturation clamp ---
        lab_mixed = self._adaptive_saturation_clamp(lab_mixed, tgt_lab, max_change=20.0)

        lab_mixed[:, 0] = np.clip(lab_mixed[:, 0], 0.0, 100.0)
        lab_mixed[:, 1] = np.clip(lab_mixed[:, 1], -128.0, 127.0)
        lab_mixed[:, 2] = np.clip(lab_mixed[:, 2], -128.0, 127.0)

        lab_mixed_32f = lab_mixed.astype(np.float32).reshape(-1, 1, 3)
        rgb_mapped = cv2.cvtColor(lab_mixed_32f, cv2.COLOR_LAB2RGB).reshape(-1, 3).astype(np.float32)

        rgb_mapped = self._soft_clip_gamut(rgb_mapped)
        rgb_mapped = np.clip(rgb_mapped, 0.0, 1.0)

        alpha = strength * 0.8
        rgb_mapped = tgt_pixels_rgb / 255.0 * (1.0 - alpha) + rgb_mapped * alpha

        src_rgb = tgt_pixels_rgb / 255.0
        lut = self._scatter_to_lut(src_rgb, rgb_mapped, s)

        fallback = self._build_from_stats(ref_stats, tgt_stats, strength, s)
        fallback_lut = fallback.reshape((s, s, s, 3))
        empty_mask = np.all(lut == 0.0, axis=3)
        for ri in range(s):
            for gi in range(s):
                for bi in range(s):
                    if empty_mask[ri, gi, bi] and (ri > 0 or gi > 0 or bi > 0):
                        lut[ri, gi, bi] = fallback_lut[ri, gi, bi]

        return lut.reshape(-1, 3)

    def _build_aggressive(
        self,
        tgt_frames: List[np.ndarray],
        ref_frames: Optional[List[np.ndarray]],
        ref_stats: ColorStats,
        tgt_stats: ColorStats,
        strength: float,
        s: int,
    ) -> np.ndarray:
        MAX_PIXELS = 50000

        tgt_pixels_rgb = np.concatenate([f.reshape(-1, 3) for f in tgt_frames], axis=0).astype(np.float32)
        rng = np.random.default_rng(42)
        if len(tgt_pixels_rgb) > MAX_PIXELS:
            idx = rng.choice(len(tgt_pixels_rgb), MAX_PIXELS, replace=False)
            tgt_pixels_rgb = tgt_pixels_rgb[idx]

        tgt_u8 = tgt_pixels_rgb.astype(np.uint8)
        lab_scaled = cv2.cvtColor(tgt_u8.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        tgt_lab = lab_scaled.copy()
        tgt_lab[:, 0] = lab_scaled[:, 0] * 100.0 / 255.0
        tgt_lab[:, 1] = lab_scaled[:, 1] - 128.0
        tgt_lab[:, 2] = lab_scaled[:, 2] - 128.0

        if ref_frames is not None and len(ref_frames) > 0:
            ref_pixels_rgb = np.concatenate([f.reshape(-1, 3) for f in ref_frames], axis=0).astype(np.float32)
            if len(ref_pixels_rgb) > MAX_PIXELS:
                idx = rng.choice(len(ref_pixels_rgb), MAX_PIXELS, replace=False)
                ref_pixels_rgb = ref_pixels_rgb[idx]
            ref_u8 = ref_pixels_rgb.astype(np.uint8)
            ref_lab_scaled = cv2.cvtColor(ref_u8.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
            ref_lab = ref_lab_scaled.copy()
            ref_lab[:, 0] = ref_lab_scaled[:, 0] * 100.0 / 255.0
            ref_lab[:, 1] = ref_lab_scaled[:, 1] - 128.0
            ref_lab[:, 2] = ref_lab_scaled[:, 2] - 128.0
        else:
            ref_lab = rng.normal(ref_stats.mean_lab, np.maximum(ref_stats.std_lab, 1.0), (MAX_PIXELS, 3)).astype(np.float32)

        # --- PCA + HM ---
        lab_pca = self._pca_transfer(tgt_lab, ref_lab, tgt_lab)

        hist_lut_l, hist_lut_a, hist_lut_b = self._build_histogram_luts(ref_stats, tgt_stats)
        lab_hm = tgt_lab.copy()
        for i in range(tgt_lab.shape[0]):
            li = int(np.clip(tgt_lab[i, 0] * 255.0 / 100.0, 0, 255))
            ai = int(np.clip(tgt_lab[i, 1] + 128.0, 0, 255))
            bi = int(np.clip(tgt_lab[i, 2] + 128.0, 0, 255))
            lab_hm[i, 0] = hist_lut_l[li] * 100.0 / 255.0
            lab_hm[i, 1] = hist_lut_a[ai] - 128.0
            lab_hm[i, 2] = hist_lut_b[bi] - 128.0

        lab_mixed = lab_pca * 0.3 + lab_hm * 0.7

        # --- Luminance protection (smooth curve) ---
        protect = self._luminance_protect_weights(tgt_lab[:, 0].reshape(-1, 1)).ravel()
        lab_mixed[:, 0] = tgt_lab[:, 0] * protect + lab_mixed[:, 0] * (1.0 - protect)

        # --- Adaptive saturation clamp (tighter) ---
        lab_mixed = self._adaptive_saturation_clamp(lab_mixed, tgt_lab, max_change=25.0)

        lab_mixed[:, 0] = np.clip(lab_mixed[:, 0], 0.0, 100.0)
        lab_mixed[:, 1] = np.clip(lab_mixed[:, 1], -128.0, 127.0)
        lab_mixed[:, 2] = np.clip(lab_mixed[:, 2], -128.0, 127.0)

        lab_mixed_32f = lab_mixed.astype(np.float32).reshape(-1, 1, 3)
        rgb_mapped = cv2.cvtColor(lab_mixed_32f, cv2.COLOR_LAB2RGB).reshape(-1, 3).astype(np.float32)

        # --- Soft gamut compression instead of hard clip ---
        rgb_mapped = self._soft_clip_gamut(rgb_mapped)
        rgb_mapped = np.clip(rgb_mapped, 0.0, 1.0)

        # --- Strength blend ---
        alpha = strength * 0.85
        rgb_mapped = tgt_pixels_rgb / 255.0 * (1.0 - alpha) + rgb_mapped * alpha

        # --- Build LUT via trilinear scatter ---
        src_rgb = tgt_pixels_rgb / 255.0
        lut = self._scatter_to_lut(src_rgb, rgb_mapped, s)

        # --- Fill empty grid points ---
        fallback = self._build_from_stats(ref_stats, tgt_stats, strength, s)
        fallback_lut = fallback.reshape((s, s, s, 3))
        empty_mask = np.all(lut == 0.0, axis=3)
        for ri in range(s):
            for gi in range(s):
                for bi in range(s):
                    if empty_mask[ri, gi, bi] and (ri > 0 or gi > 0 or bi > 0):
                        lut[ri, gi, bi] = fallback_lut[ri, gi, bi]

        return lut.reshape(-1, 3)

    @staticmethod
    @jit(nopython=True)
    def _scatter_to_lut(src_rgb: np.ndarray, rgb_mapped: np.ndarray, s: int) -> np.ndarray:
        n = src_rgb.shape[0]
        acc = np.zeros((s, s, s, 3), dtype=np.float64)
        wsum = np.zeros((s, s, s), dtype=np.float64)
        max_idx = s - 1

        for i in range(n):
            r, g, b = src_rgb[i]
            rf = r * max_idx
            gf = g * max_idx
            bf = b * max_idx
            ri = int(rf)
            gi = int(gf)
            bi = int(bf)
            ri = min(ri, max_idx - 1) if ri == max_idx else max(ri, 0)
            gi = min(gi, max_idx - 1) if gi == max_idx else max(gi, 0)
            bi = min(bi, max_idx - 1) if bi == max_idx else max(bi, 0)

            rw = rf - ri
            gw = gf - gi
            bw = bf - bi

            w000 = (1.0 - rw) * (1.0 - gw) * (1.0 - bw)
            w001 = rw * (1.0 - gw) * (1.0 - bw)
            w010 = (1.0 - rw) * gw * (1.0 - bw)
            w011 = rw * gw * (1.0 - bw)
            w100 = (1.0 - rw) * (1.0 - gw) * bw
            w101 = rw * (1.0 - gw) * bw
            w110 = (1.0 - rw) * gw * bw
            w111 = rw * gw * bw

            ri2 = ri + 1
            gi2 = gi + 1
            bi2 = bi + 1

            out = rgb_mapped[i]

            acc[ri, gi, bi] += out * w000
            acc[ri2, gi, bi] += out * w001
            acc[ri, gi2, bi] += out * w010
            acc[ri2, gi2, bi] += out * w011
            acc[ri, gi, bi2] += out * w100
            acc[ri2, gi, bi2] += out * w101
            acc[ri, gi2, bi2] += out * w110
            acc[ri2, gi2, bi2] += out * w111

            wsum[ri, gi, bi] += w000
            wsum[ri2, gi, bi] += w001
            wsum[ri, gi2, bi] += w010
            wsum[ri2, gi2, bi] += w011
            wsum[ri, gi, bi2] += w100
            wsum[ri2, gi, bi2] += w101
            wsum[ri, gi2, bi2] += w110
            wsum[ri2, gi2, bi2] += w111

        lut = np.zeros((s, s, s, 3), dtype=np.float32)
        for ri in range(s):
            for gi in range(s):
                for bi in range(s):
                    if wsum[ri, gi, bi] > 0:
                        lut[ri, gi, bi, 0] = acc[ri, gi, bi, 0] / wsum[ri, gi, bi]
                        lut[ri, gi, bi, 1] = acc[ri, gi, bi, 1] / wsum[ri, gi, bi]
                        lut[ri, gi, bi, 2] = acc[ri, gi, bi, 2] / wsum[ri, gi, bi]

        return lut

    def _build_from_stats(
        self,
        ref_stats: ColorStats,
        tgt_stats: ColorStats,
        strength: float,
        s: int,
    ) -> np.ndarray:
        r_vals = np.linspace(0, 1, s, dtype=np.float32)
        g_vals = np.linspace(0, 1, s, dtype=np.float32)
        b_vals = np.linspace(0, 1, s, dtype=np.float32)
        rr, gg, bb = np.meshgrid(r_vals, g_vals, b_vals, indexing="ij")
        rgb_flat = np.stack([rr.ravel(), gg.ravel(), bb.ravel()], axis=1)

        src_u8 = (rgb_flat * 255).astype(np.uint8).reshape(-1, 1, 3)
        lab_scaled = cv2.cvtColor(src_u8, cv2.COLOR_RGB2LAB).astype(np.float32).reshape(-1, 3)
        lab = lab_scaled.copy()
        lab[:, 0] = lab_scaled[:, 0] * 100.0 / 255.0
        lab[:, 1] = lab_scaled[:, 1] - 128.0
        lab[:, 2] = lab_scaled[:, 2] - 128.0

        # --- Reinhard per-channel ---
        lab_reinhard = lab.copy()
        for c in range(3):
            s_in = tgt_stats.std_lab[c] if tgt_stats.std_lab[c] > 1e-6 else 1.0
            s_out = ref_stats.std_lab[c] if ref_stats.std_lab[c] > 1e-6 else 1.0
            ratio = np.clip(s_out / s_in, 0.25, 4.0)
            adj = 1.0 + (ratio - 1.0) * strength
            lab_reinhard[:, c] = (lab[:, c] - tgt_stats.mean_lab[c]) * adj + ref_stats.mean_lab[c]

        # --- Histogram matching ---
        hist_lut_l, hist_lut_a, hist_lut_b = self._build_histogram_luts(ref_stats, tgt_stats)
        lab_hist = lab.copy()
        for i in range(lab.shape[0]):
            li = int(np.clip(lab[i, 0] * 255.0 / 100.0, 0, 255))
            ai = int(np.clip(lab[i, 1] + 128.0, 0, 255))
            bi = int(np.clip(lab[i, 2] + 128.0, 0, 255))
            lab_hist[i, 0] = hist_lut_l[li] * 100.0 / 255.0
            lab_hist[i, 1] = hist_lut_a[ai] - 128.0
            lab_hist[i, 2] = hist_lut_b[bi] - 128.0

        lab_mixed = lab_hist * 0.5 + lab_reinhard * 0.5

        alpha = strength * 0.85
        lab_out = lab * (1.0 - alpha) + lab_mixed * alpha

        protect = self._luminance_protect_weights(lab[:, 0].reshape(-1, 1)).ravel()
        lab_out[:, 0] = lab[:, 0] * protect + lab_out[:, 0] * (1.0 - protect)

        lab_out[:, 0] = np.clip(lab_out[:, 0], 0.0, 100.0)
        lab_out[:, 1] = np.clip(lab_out[:, 1], -128.0, 127.0)
        lab_out[:, 2] = np.clip(lab_out[:, 2], -128.0, 127.0)

        lab_32f = lab_out.astype(np.float32).reshape(-1, 1, 3)
        rgb_out = cv2.cvtColor(lab_32f, cv2.COLOR_LAB2RGB).reshape(-1, 3).astype(np.float32)
        rgb_out = np.clip(rgb_out, 0.0, 1.0)

        return rgb_out

    @staticmethod
    def _build_histogram_luts(ref_stats: ColorStats, tgt_stats: ColorStats):
        def cdf_from_hist(hist):
            c = np.cumsum(hist).astype(np.float64)
            if c[-1] > 0:
                c /= c[-1]
            return c

        def matching_lut(tgt_hist, ref_hist):
            t_cdf = cdf_from_hist(tgt_hist)
            r_cdf = cdf_from_hist(ref_hist)
            lut = np.zeros(len(tgt_hist), dtype=np.float32)
            ri = 0
            for ti in range(len(tgt_hist)):
                while ri < len(ref_hist) - 1 and r_cdf[ri] < t_cdf[ti]:
                    ri += 1
                lut[ti] = float(ri)
            return lut

        lut_l = matching_lut(tgt_stats.hist_l, ref_stats.hist_l)
        lut_a = matching_lut(tgt_stats.hist_a, ref_stats.hist_a)
        lut_b = matching_lut(tgt_stats.hist_b_lab, ref_stats.hist_b_lab)

        return lut_l, lut_a, lut_b

    @staticmethod
    @jit(nopython=True)
    def _luminance_protect_weights(l_channel: np.ndarray) -> np.ndarray:
        n = l_channel.shape[0]
        weights = np.zeros((n, 1), dtype=np.float32)
        for i in range(n):
            l = l_channel[i, 0]
            if l < 20.0:
                weights[i, 0] = 1.0 - l / 20.0
            elif l > 80.0:
                weights[i, 0] = (l - 80.0) / 20.0
            else:
                mid = (l - 20.0) / 60.0
                weights[i, 0] = 0.0
        return weights

    def apply_lut(self, image: np.ndarray) -> np.ndarray:
        if self._lut is None:
            raise RuntimeError("LUT not generated yet. Call generate() first.")
        return self._apply_lut_trilinear(image, self._lut)

    @staticmethod
    @jit(nopython=True)
    def _apply_lut_trilinear(image: np.ndarray, lut: np.ndarray) -> np.ndarray:
        h = image.shape[0]
        w = image.shape[1]
        result = np.zeros((h, w, 3), dtype=np.uint8)
        size = lut.shape[0]
        max_idx = size - 1

        for y in range(h):
            for x in range(w):
                ri = int(image[y, x, 0] / 255.0 * max_idx)
                gi = int(image[y, x, 1] / 255.0 * max_idx)
                bi = int(image[y, x, 2] / 255.0 * max_idx)

                rn = ri + 1
                gn = gi + 1
                bn = bi + 1
                if rn > max_idx:
                    rn = max_idx
                if gn > max_idx:
                    gn = max_idx
                if bn > max_idx:
                    bn = max_idx

                rd = image[y, x, 0] / 255.0 * max_idx - ri
                gd = image[y, x, 1] / 255.0 * max_idx - gi
                bd = image[y, x, 2] / 255.0 * max_idx - bi

                c000 = lut[ri, gi, bi]
                c001 = lut[rn, gi, bi]
                c010 = lut[ri, gn, bi]
                c011 = lut[rn, gn, bi]
                c100 = lut[ri, gi, bn]
                c101 = lut[rn, gi, bn]
                c110 = lut[ri, gn, bn]
                c111 = lut[rn, gn, bn]

                c00 = c000 * (1.0 - rd) + c001 * rd
                c01 = c010 * (1.0 - rd) + c011 * rd
                c10 = c100 * (1.0 - rd) + c101 * rd
                c11 = c110 * (1.0 - rd) + c111 * rd

                c0 = c00 * (1.0 - gd) + c01 * gd
                c1 = c10 * (1.0 - gd) + c11 * gd

                out_0 = c0[0] * (1.0 - bd) + c1[0] * bd
                out_1 = c0[1] * (1.0 - bd) + c1[1] * bd
                out_2 = c0[2] * (1.0 - bd) + c1[2] * bd

                val0 = int(out_0 * 255.0)
                val1 = int(out_1 * 255.0)
                val2 = int(out_2 * 255.0)

                if val0 < 0:
                    val0 = 0
                if val0 > 255:
                    val0 = 255
                if val1 < 0:
                    val1 = 0
                if val1 > 255:
                    val1 = 255
                if val2 < 0:
                    val2 = 0
                if val2 > 255:
                    val2 = 255

                result[y, x, 0] = val0
                result[y, x, 1] = val1
                result[y, x, 2] = val2

        return result
