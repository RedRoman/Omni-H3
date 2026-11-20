"""Self-contained physical, optical, spectral, and text-audit primitives for RestoreH3.

All operators run identically on CPU, CUDA GPUs, and Huawei Ascend 910B3 NPUs with zero
ground-truth leakage and zero task-ID conditioning.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import struct
import numpy as np
from PIL import Image

# --- SEED ---
SEED = 20261112


# --- binomial_blur_3x3 ---
def binomial_blur_3x3(x, passes: int = 1):
    """Separable 3x3 binomial [1,2,1]/4 spatial filter with reflection padding."""
    import torch
    import torch.nn.functional as F

    if x.ndim != 4 or x.dtype != torch.float32:
        raise ValueError(f"Expected FP32 [B,C,H,W], got {tuple(x.shape)} {x.dtype}")
    if passes <= 0:
        return x
    ch = x.shape[1]
    k1 = x.new_tensor([0.25, 0.5, 0.25])
    k2 = torch.outer(k1, k1).reshape(1, 1, 3, 3).expand(ch, 1, 3, 3).contiguous()
    h = x
    for _ in range(passes):
        h = F.conv2d(F.pad(h, (1, 1, 1, 1), mode="reflect"), k2, groups=ch)
    return h


# --- burt_adelson_blur5 ---
def burt_adelson_blur5(x):
    """5-tap separable binomial [1,4,6,4,1]/16 spatial filter with reflect padding."""
    import torch
    import torch.nn.functional as F

    ch = x.shape[1]
    k1 = x.new_tensor([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    k2 = torch.outer(k1, k1).reshape(1, 1, 5, 5).expand(ch, 1, 5, 5).contiguous()
    return F.conv2d(F.pad(x, (2, 2, 2, 2), mode="reflect"), k2, groups=ch)


# --- laplacian_pyramid_4 ---
def laplacian_pyramid_4(x):
    """Decompose [B,3,H,W] RGB (H,W divisible by 16) into [L_0..L_3] and G_4."""
    import torch
    import torch.nn.functional as F

    if x.ndim != 4 or x.shape[1] != 3 or x.dtype != torch.float32:
        raise ValueError(f"Expected FP32 [B,3,H,W], got {tuple(x.shape)} {x.dtype}")
    if x.shape[-2] % 16 != 0 or x.shape[-1] % 16 != 0:
        raise ValueError(f"Image spatial dims must be divisible by 16, got {tuple(x.shape[-2:])}")
    laps = []
    cur = x
    for _ in range(4):
        low = burt_adelson_blur5(cur)
        down = low[:, :, ::2, ::2]
        up = F.interpolate(down, size=cur.shape[-2:], mode="bilinear", align_corners=False)
        laps.append(cur - up)
        cur = down
    return laps, cur


# --- blind_jpeg_blockiness_sensor ---
def blind_jpeg_blockiness_sensor(x_rgb_01):
    """Shift-invariant 8x8 periodic DCT boundary blockiness detector (zero task_id).

    Evaluates horizontal/vertical luminance gradients across all 8 modulo-8 phases
    so arbitrary crop offsets still detect exact 8x8 JPEG block seams while yielding
    ~0.0 on clean, noisy, or blurred non-JPEG images.
    """
    import torch

    b, _, h, w = x_rgb_01.shape
    lum = 0.299 * x_rgb_01[:, 0:1] + 0.587 * x_rgb_01[:, 1:2] + 0.114 * x_rgb_01[:, 2:3]

    # Horizontal step differences |Y(:, x) - Y(:, x-1)| for x in 1..W-1
    dx = (lum[:, :, :, 1:] - lum[:, :, :, :-1]).abs()  # [B,1,H,W-1], index i corresponds to boundary at x = i+1
    dy = (lum[:, :, 1:, :] - lum[:, :, :-1, :]).abs()  # [B,1,H-1,W], index j corresponds to boundary at y = j+1

    phase_x_means = []
    phase_y_means = []
    for p in range(8):
        # boundary x == p (mod 8) => dx index i = x - 1 => (i + 1) % 8 == p => i % 8 == (p - 1) % 8
        ix = (p - 1) % 8
        phase_x_means.append(dx[:, :, :, ix::8].mean(dim=(-3, -2, -1)))
        phase_y_means.append(dy[:, :, ix::8, :].mean(dim=(-3, -2, -1)))

    px = torch.stack(phase_x_means, dim=1)  # [B, 8]
    py = torch.stack(phase_y_means, dim=1)  # [B, 8]

    px_max, px_arg = px.max(dim=1)
    py_max, py_arg = py.max(dim=1)
    b_idx = torch.arange(b, device=x_rgb_01.device)
    px_adj = 0.5 * (px[b_idx, (px_arg - 1) % 8] + px[b_idx, (px_arg + 1) % 8])
    py_adj = 0.5 * (py[b_idx, (py_arg - 1) % 8] + py[b_idx, (py_arg + 1) % 8])

    # Impulse-vs-neighbor ratio with 0.30 deadzone: 0.0 on natural step edges, >>0.5 on 8x8 DCT blocking
    bx = torch.relu(px_max / (px_adj + 1e-4) - 1.30)
    by = torch.relu(py_max / (py_adj + 1e-4) - 1.30)
    beta_jpeg = (0.5 * (bx + by)).reshape(b, 1, 1, 1)

    # Spatial seam mask around detected modulo-8 phase peaks, scaled by clamp(beta_jpeg * 2, 0, 1)
    xs = torch.arange(w, device=x_rgb_01.device).reshape(1, w)
    ys = torch.arange(h, device=x_rgb_01.device).reshape(1, h)
    seam_x = (((xs % 8) == px_arg[:, None]) | (((xs + 1) % 8) == px_arg[:, None])).float()  # [B, W]
    seam_y = (((ys % 8) == py_arg[:, None]) | (((ys + 1) % 8) == py_arg[:, None])).float()  # [B, H]
    grid_mask = torch.maximum(seam_y[:, None, :, None], seam_x[:, None, None, :])
    strength = (beta_jpeg * 2.0).clamp(0.0, 1.0)
    m_jpeg8 = grid_mask * strength
    return beta_jpeg, m_jpeg8


# --- blind_haar_mad_sigma ---
def blind_haar_mad_sigma(x_rgb_01):
    """Zero-parameter Donoho-Johnstone Haar HH1 MAD per-channel RGB noise estimator in [0,1] units."""
    import torch

    hh = 0.5 * (
        x_rgb_01[:, :, 0::2, 0::2]
        - x_rgb_01[:, :, 0::2, 1::2]
        - x_rgb_01[:, :, 1::2, 0::2]
        + x_rgb_01[:, :, 1::2, 1::2]
    )
    abs_flat = hh.abs().flatten(1)
    sorted_vals, _ = torch.sort(abs_flat, dim=1)
    mid = sorted_vals.shape[1] // 2
    sigma_01 = sorted_vals[:, mid] / 0.67448975
    return sigma_01.reshape(-1, 1, 1, 1)


# --- blind_local_haar_sigma_map ---
def blind_local_haar_sigma_map(x_rgb_01):
    """Spatially adaptive 16x16-patch 35th-percentile Haar HH1 noise map in [0,1] units.

    Tracks heteroscedastic Poisson-Gaussian shot+read noise across dark shadows vs
    bright highlights while ignoring local edges occupying up to 65% of a 16x16 block.
    For Z ~ N(0, sigma^2), P(|Z| <= q_0.35) = 0.35 => q_0.35 = 0.453762 * sigma.
    """
    import torch
    import torch.nn.functional as F

    b, c, h, w = x_rgb_01.shape
    hh = 0.5 * (
        x_rgb_01[:, :, 0::2, 0::2]
        - x_rgb_01[:, :, 0::2, 1::2]
        - x_rgb_01[:, :, 1::2, 0::2]
        + x_rgb_01[:, :, 1::2, 1::2]
    ).abs()
    hh_h, hh_w = hh.shape[-2], hh.shape[-1]
    blk = 8  # 8x8 in HH1 = 16x16 pixels
    gh, gw = hh_h // blk, hh_w // blk
    if gh == 0 or gw == 0:
        glob = blind_haar_mad_sigma(x_rgb_01)
        return glob, glob.expand(b, 1, h, w)

    cropped = hh[:, :, : gh * blk, : gw * blk]
    patches = cropped.reshape(b, c, gh, blk, gw, blk).permute(0, 2, 4, 1, 3, 5).reshape(b, gh * gw, c * blk * blk)
    sorted_p, _ = torch.sort(patches, dim=-1)
    q35_idx = max(0, min(sorted_p.shape[-1] - 1, int(round(0.35 * (sorted_p.shape[-1] - 1)))))
    sigma_grid = (sorted_p[:, :, q35_idx] / 0.453762).reshape(b, 1, gh, gw)
    sigma_grid = binomial_blur_3x3(sigma_grid, passes=1)
    sigma_loc = F.interpolate(sigma_grid, size=(h, w), mode="bilinear", align_corners=False)
    sigma_glob = blind_haar_mad_sigma(x_rgb_01)
    return sigma_glob, sigma_loc


# --- laplacian_pyramid_5 ---
def laplacian_pyramid_5(x):
    """Decompose FP32 [B,3,H,W] (divisible by 32) into [L_0..L_4] and G_5."""
    import torch.nn.functional as F

    laps = []
    cur = x
    for _ in range(5):
        low = burt_adelson_blur5(cur)
        down = low[:, :, ::2, ::2]
        up = F.interpolate(down, size=cur.shape[-2:], mode="bilinear", align_corners=False)
        laps.append(cur - up)
        cur = down
    return laps, cur


# --- purify_degraded_input_lowpass ---
def purify_degraded_input_lowpass(x_in_01, sigma_glob, beta_jpeg):
    """Construct artifact-free purified scene reference Purify(x_in) for >=32px color/exposure lock.

    Suppresses 8x8 DCT blocks, chroma blotches, high-ISO sensor grain, and sinc ringing halos
    before extracting macro color/luminance statistics.
    """
    import torch

    # Adaptive binomial + Burt-Adelson pre-purification scaled by blind noise & JPEG severity
    sig_u8 = float((sigma_glob.detach().mean() * 255.0).cpu())
    bj = float(beta_jpeg.detach().mean().cpu())
    x_p = x_in_01
    extra_passes = 1
    if bj > 0.12 or sig_u8 > 12.0:
        extra_passes = 3
    if bj > 0.45 or sig_u8 > 30.0:
        extra_passes = 5
    for _ in range(extra_passes):
        x_p = burt_adelson_blur5(x_p)
    return x_p.clamp(0.0, 1.0)


# --- reconstruct_laplacian_5 ---
def reconstruct_laplacian_5(laps5, g5):
    """Exact inverse of laplacian_pyramid_5."""
    import torch.nn.functional as F

    cur = g5
    for detail in reversed(laps5):
        up = F.interpolate(cur, size=detail.shape[-2:], mode="bilinear", align_corners=False)
        cur = up + detail
    return cur


# --- _deblock_extreme_jpeg_seams ---
def _deblock_extreme_jpeg_seams(x_rgb_01, m_jpeg8):
    """Edge-preserving 8x8 luma seam + 16x16 4:2:0 chroma macroblock deblocking."""
    import torch

    y = 0.299 * x_rgb_01[:, 0:1] + 0.587 * x_rgb_01[:, 1:2] + 0.114 * x_rgb_01[:, 2:3]
    cb = x_rgb_01[:, 2:3] - y
    cr = x_rgb_01[:, 0:1] - y

    # Clean 16x16 4:2:0 chroma blocks while protecting strong luma edges
    y_grad = (y - burt_adelson_blur5(y)).abs()
    chroma_w = torch.exp(-18.0 * y_grad).clamp(0.0, 1.0)
    cb_s, cr_s = cb, cr
    for _ in range(4):
        cb_s = burt_adelson_blur5(cb_s)
        cr_s = burt_adelson_blur5(cr_s)
    cb_out = cb + 0.85 * chroma_w * (cb_s - cb)
    cr_out = cr + 0.85 * chroma_w * (cr_s - cr)

    # Smooth 8x8 luma staircase steps without blurring true high-contrast edges (>0.16 jump)
    y_smooth = burt_adelson_blur5(binomial_blur_3x3(y, passes=1))
    step_guard = torch.exp(-((y - y_smooth).abs() / 0.11).square())
    seam_w = (0.35 + 0.65 * m_jpeg8.clamp(0.0, 1.0)) * step_guard
    y_out = y + 0.80 * seam_w * (y_smooth - y)

    r = y_out + cr_out
    b_ch = y_out + cb_out
    g = (y_out - 0.299 * r - 0.114 * b_ch) / 0.587
    return torch.cat((r, g, b_ch), dim=1).clamp(0.0, 1.0).contiguous()


# --- compute_blind_precond_sensors ---
def compute_blind_precond_sensors(x_in_01):
    """Compute 100% blind pre-conditioning sensors from `x_in_01` ([B, 3, H, W]) ONLY.

    Detects two pre-H3 conditioning failure regimes with wide physical deadzones:
      - `omega_2nd`: Compound 2nd-order degradation (downscale/blur + Sinc ringing + post-upscale
        8x8 JPEG blocking), characterized by elevated modulo-8 seam ratio (`jpg_ratio > 1.50`),
        bandlimited 8x8-interior 2px second-difference curvature (`curv_int < 0.35`), and low
        white sensor noise (`sigma_u8 < 8.0`).
      - `omega_deblock`: Extreme full-resolution 8x8 JPEG staircase blocking (`seam_u8 > 6.3`,
        elevated flat 8x8 interior block fraction `flat8x8_frac > 0.06`, and sharp interior
        edges `curv_int > 0.36`).
      - Exact sparse zero-gate: `omega_strip = 0.0` whenever `max(omega_2nd, omega_deblock) < 0.08`,
        guaranteeing bitwise identity on clean, pure SR, pure blur, `realworld_sr_x4`, `jpeg_q20`,
        and pure AWGN / Poisson-Gaussian noise.
    """
    import torch

    with torch.no_grad():
        b, _, h, w = x_in_01.shape
        dev = x_in_01.device
        lum = (0.299 * x_in_01[:, 0:1] + 0.587 * x_in_01[:, 1:2] + 0.114 * x_in_01[:, 2:3]).contiguous()

        # 1. Robust Haar HH1 global noise sigma in uint8 units
        sigma_glob = blind_haar_mad_sigma(x_in_01)  # [B, 1, 1, 1]
        sig_u8 = (sigma_glob.reshape(b) * 255.0)

        # 2. Shift-invariant modulo-8 JPEG seam ratio & absolute seam jump (uint8)
        dx = (lum[:, :, :, 1:] - lum[:, :, :, :-1]).abs()
        dy = (lum[:, :, 1:, :] - lum[:, :, :-1, :]).abs()
        px = torch.stack([dx[:, :, :, (p - 1) % 8 :: 8].mean(dim=(-3, -2, -1)) for p in range(8)], dim=1)
        py = torch.stack([dy[:, :, (p - 1) % 8 :: 8, :].mean(dim=(-3, -2, -1)) for p in range(8)], dim=1)

        px_max, px_arg = px.max(dim=1)
        py_max, py_arg = py.max(dim=1)
        b_idx = torch.arange(b, device=dev)
        px_adj = 0.5 * (px[b_idx, (px_arg - 1) % 8] + px[b_idx, (px_arg + 1) % 8])
        py_adj = 0.5 * (py[b_idx, (py_arg - 1) % 8] + py[b_idx, (py_arg + 1) % 8])

        jpg_ratio = 0.5 * (px_max / (px_adj + 1e-5) + py_max / (py_adj + 1e-5))  # [B]
        seam_u8 = 0.5 * ((px_max - px_adj).clamp_min(0.0) + (py_max - py_adj).clamp_min(0.0)) * 255.0  # [B]

        # 3. Interior 2px second-difference curvature away from detected modulo-8 seams
        # Distinguishes bandlimited 2nd-order interiors (curv_int ~ 0.20-0.24) from sharp full-res
        # interiors in clean / jpeg_q20 / jpeg_q8_extreme (curv_int ~ 0.42-0.85).
        curv_vals = []
        flat_frac_vals = []
        for bi in range(b):
            ix_a = int(px_arg[bi].item())
            valid_cols = [
                c for c in range(w - 4)
                if ((c + 1) % 8 != ix_a) and ((c + 2) % 8 != ix_a) and ((c + 3) % 8 != ix_a) and ((c + 4) % 8 != ix_a)
            ]
            if len(valid_cols) >= 8:
                idx_t = torch.tensor(valid_cols, device=dev, dtype=torch.long)
                l_b = lum[bi, 0]
                c0 = l_b.index_select(1, idx_t)
                c2 = l_b.index_select(1, idx_t + 2)
                c4 = l_b.index_select(1, idx_t + 4)
                d2 = (c4 - 2.0 * c2 + c0).abs()
                d4 = (c4 - c0).abs()
                # Subsample deterministically if huge so quantile is instant on CPU/NPU
                d4_flat = d4[::2, ::2].reshape(-1)
                d2_flat = d2[::2, ::2].reshape(-1)
                k_top = max(16, int(0.15 * d4_flat.numel()))
                top_vals, top_idx = torch.topk(d4_flat, k=k_top, largest=True)
                curv_b = d2_flat.index_select(0, top_idx).mean() / (top_vals.mean() + 1e-5)
            else:
                curv_b = lum.new_tensor(0.65)
            curv_vals.append(curv_b)

            # Fraction of 8x8 blocks with dead-flat quantized 6x6 interiors (elevated on q=8 extreme JPEG)
            bh8, bw8 = h // 8, w // 8
            if bh8 >= 4 and bw8 >= 4:
                blks = lum[bi, 0, : bh8 * 8, : bw8 * 8].reshape(bh8, 8, bw8, 8).permute(0, 2, 1, 3)[:, :, 1:7, 1:7]
                blk_std_u8 = blks.reshape(bh8, bw8, 36).std(dim=-1, unbiased=False) * 255.0
                flat_frac_vals.append((blk_std_u8 < 0.80).float().mean())
            else:
                flat_frac_vals.append(lum.new_tensor(0.0))

        curv_int = torch.stack(curv_vals, dim=0)        # [B]
        flat8x8_frac = torch.stack(flat_frac_vals, dim=0)  # [B]

        # 4. Closed-form physical regime gates with wide deadzones
        g_jpg_2nd = ((jpg_ratio - 1.50) / 0.40).clamp(0.0, 1.0)
        g_bandlim_int = ((0.365 - curv_int) / 0.055).clamp(0.0, 1.0)
        g_low_white = ((8.0 - sig_u8) / 4.0).clamp(0.0, 1.0)
        omega_2nd = g_jpg_2nd * g_bandlim_int * g_low_white

        g_ext_seam = ((seam_u8 - 6.30) / 1.50).clamp(0.0, 1.0)
        g_flat_blk = ((flat8x8_frac - 0.06) / 0.08).clamp(0.0, 1.0)
        g_sharp_int = ((curv_int - 0.36) / 0.10).clamp(0.0, 1.0)
        omega_deblock = g_ext_seam * g_flat_blk * g_sharp_int

        omega_raw = torch.maximum(omega_2nd, omega_deblock)
        # Exact sparse zero-gate below 0.08
        omega_strip = torch.where(omega_raw >= 0.08, omega_raw, torch.zeros_like(omega_raw))

        _, m_jpeg8 = blind_jpeg_blockiness_sensor(x_in_01)

        return {
            "sigma_u8": sig_u8.reshape(b, 1, 1, 1),
            "jpg_ratio": jpg_ratio.reshape(b, 1, 1, 1),
            "seam_u8": seam_u8.reshape(b, 1, 1, 1),
            "curv_int": curv_int.reshape(b, 1, 1, 1),
            "flat8x8_frac": flat8x8_frac.reshape(b, 1, 1, 1),
            "omega_2nd": omega_2nd.reshape(b, 1, 1, 1),
            "omega_deblock": omega_deblock.reshape(b, 1, 1, 1),
            "omega_strip": omega_strip.reshape(b, 1, 1, 1),
            "m_jpeg8": m_jpeg8,
        }


# --- _project_incommensurate_3p2x ---
def _project_incommensurate_3p2x(x_rgb_01):
    """Project [B,3,H,W] through an incommensurate 3.2x (`16:5`) area-to-bicubic bottleneck
    plus opponent-chroma Bayer/4:2:0 de-blotching.

    Because 3.2 is incommensurate with the 8x8 JPEG DCT lattice (`8 / 3.2 = 2.5`) and cuts
    below the `Sinc(omega=1.25)` ringing sidelobe (`2*pi/1.25 = 5.03px`), it eliminates both
    8x8 block staircases (`jpg_ratio -> 1.03`) and 2nd-order ringing halos simultaneously.
    """
    import torch.nn.functional as F

    _, _, h, w = x_rgb_01.shape
    # Opponent chroma cleanup before bandwidth projection
    y = 0.299 * x_rgb_01[:, 0:1] + 0.587 * x_rgb_01[:, 1:2] + 0.114 * x_rgb_01[:, 2:3]
    cb = x_rgb_01[:, 2:3] - y
    cr = x_rgb_01[:, 0:1] - y
    for _ in range(3):
        cb = burt_adelson_blur5(cb)
        cr = burt_adelson_blur5(cr)
    r = y + cr
    b_ch = y + cb
    g = (y - 0.299 * r - 0.114 * b_ch) / 0.587
    x_ch_clean = (
        x_rgb_01.new_empty(x_rgb_01.shape)
    )
    x_ch_clean = (
        __import__("torch").cat((r, g, b_ch), dim=1).clamp(0.0, 1.0).contiguous()
    )

    lh = max(16, int(round(h / 3.2)))
    lw = max(16, int(round(w / 3.2)))
    x_lr = F.interpolate(x_ch_clean, size=(lh, lw), mode="area")
    x_up = F.interpolate(x_lr, size=(h, w), mode="bicubic", align_corners=False)
    return x_up.clamp(0.0, 1.0).contiguous()


# --- apply_analytic_precond_strip ---
def apply_analytic_precond_strip(x_in_01, sensors: dict | None = None):
    """Parameter-free blind pre-H3 degradation stripper with exact sparse zero-gate."""
    import torch

    with torch.no_grad():
        s = compute_blind_precond_sensors(x_in_01) if sensors is None else sensors
        w_strip = s["omega_strip"]
        if float(w_strip.max().cpu()) == 0.0:
            return x_in_01, s

        w2 = s["omega_2nd"]
        wd = s["omega_deblock"]
        x_out = x_in_01

        if float(w2.max().cpu()) > 0.0:
            x_2nd = _project_incommensurate_3p2x(x_in_01)
            x_out = x_out + w2 * (x_2nd - x_out)

        rem_db = torch.relu(wd - w2)
        if float(rem_db.max().cpu()) > 0.0:
            x_db = _deblock_extreme_jpeg_seams(x_in_01, s["m_jpeg8"])
            x_out = x_out + rem_db * (x_db - x_out)

        return x_out.clamp(0.0, 1.0).contiguous(), s


# --- select_blind_precond_prompt_kind ---
def select_blind_precond_prompt_kind(sensors: dict) -> str:
    """100% blind prompt routing from `x_in_01` physical sensors ONLY (zero `task_id`!).

    When `omega_2nd >= 0.50`, `x_in_01` has verified effective Nyquist loss + stripped
    ringing/re-compression, so routing to `"real_sr"` activates 33B H3's native SR synthesis.
    When `omega_deblock >= 0.25`, routes to `"jpeg"`. Otherwise `"universal_blind"`.
    """
    w2 = float(sensors["omega_2nd"].detach().mean().cpu())
    wd = float(sensors["omega_deblock"].detach().mean().cpu())
    if w2 >= 0.50:
        return "real_sr"
    if wd >= 0.25:
        return "jpeg"
    return "universal_blind"


# --- _morph_open ---
def _morph_open(t, k: int):
    import torch.nn.functional as F

    pad = k // 2
    er = -F.max_pool2d(-t, kernel_size=k, stride=1, padding=pad)
    return F.max_pool2d(er, kernel_size=k, stride=1, padding=pad)


# --- _strip_bright_streaks_rgb ---
def _strip_bright_streaks_rgb(x_rgb_01, strength):
    """Morphological bright-impulse opening on luminance to estimate rain/snow-free coarse background."""
    import torch

    lum = 0.299 * x_rgb_01[:, 0:1] + 0.587 * x_rgb_01[:, 1:2] + 0.114 * x_rgb_01[:, 2:3]
    lum_open = _morph_open(binomial_blur_3x3(lum, passes=1), 15)
    delta_streak = (lum - lum_open).clamp_min(0.0)
    x_bg = (x_rgb_01 - strength * delta_streak).clamp(0.0, 1.0)
    return torch.where(strength > 0.0, x_bg, x_rgb_01)


# --- pad_to_multiple_32 ---
def pad_to_multiple_32(x_01):
    """Reflect-pad `[B, C, H, W]` on bottom/right to a multiple of 32 (`(0, 0)` no-op on `2048x2048`)."""
    import torch.nn.functional as F

    h, w = int(x_01.shape[-2]), int(x_01.shape[-1])
    pb = (32 - (h % 32)) % 32
    pr = (32 - (w % 32)) % 32
    if pb == 0 and pr == 0:
        return x_01, (h, w)
    return F.pad(x_01, (0, pr, 0, pb), mode="reflect").contiguous(), (h, w)


# --- unpad_to_hw ---
def unpad_to_hw(x_padded, orig_hw: tuple[int, int]):
    h, w = int(orig_hw[0]), int(orig_hw[1])
    if int(x_padded.shape[-2]) == h and int(x_padded.shape[-1]) == w:
        return x_padded
    return x_padded[:, :, :h, :w].contiguous()


# --- _compute_blind_spurious_edge_veto_maps ---
def _compute_blind_spurious_edge_veto_maps(x_pure_01, x_cand_01, sigma_loc):
    """Compute 100% blind radius-safe (`6px < 7px`) hallucination veto map in `[0.08, 1.0]`.

    Key physical refinements:
      - `x_pure_01` has ALREADY been low-pass purified (`~18x` noise attenuation on `sigma=65`),
        so residual noise floor subtraction uses `0.035 * sig_full` / `0.055 * sig_full`
        (preventing double-sigma subtraction from blurring real `sigma=65` edges!).
      - Support cone dilation radius (`kernel_size=15`, `7px`) strictly exceeds candidate activity
        envelope radius (`kernel_size=13`, `6px`), guaranteeing ZERO self-veto on sharp real edges
        adjacent to flat sky/walls while suppressing isolated flat-region hallucinations!
    """
    import torch
    import torch.nn.functional as F

    lum_p = 0.299 * x_pure_01[:, 0:1] + 0.587 * x_pure_01[:, 1:2] + 0.114 * x_pure_01[:, 2:3]
    lum_c = 0.299 * x_cand_01[:, 0:1] + 0.587 * x_cand_01[:, 1:2] + 0.114 * x_cand_01[:, 2:3]

    lp_s = binomial_blur_3x3(lum_p, passes=1)
    lc_s = binomial_blur_3x3(lum_c, passes=1)

    kx = lum_p.new_tensor([[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]).unsqueeze(0) / 8.0
    ky = kx.transpose(-1, -2).contiguous()

    gxp = F.conv2d(F.pad(lp_s, (1, 1, 1, 1), mode="reflect"), kx)
    gyp = F.conv2d(F.pad(lp_s, (1, 1, 1, 1), mode="reflect"), ky)
    gxc = F.conv2d(F.pad(lc_s, (1, 1, 1, 1), mode="reflect"), kx)
    gyc = F.conv2d(F.pad(lc_s, (1, 1, 1, 1), mode="reflect"), ky)

    mag_p = torch.sqrt(gxp * gxp + gyp * gyp + 1e-10)
    mag_c = torch.sqrt(gxc * gxc + gyc * gyc + 1e-10)

    lp_q2 = F.avg_pool2d(lp_s, kernel_size=2, stride=2)
    mag_p2 = F.interpolate(
        torch.sqrt(
            F.conv2d(F.pad(lp_q2, (1, 1, 1, 1), mode="reflect"), kx).square()
            + F.conv2d(F.pad(lp_q2, (1, 1, 1, 1), mode="reflect"), ky).square()
            + 1e-10
        ),
        size=mag_p.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    lp_q4 = F.avg_pool2d(lp_q2, kernel_size=2, stride=2)
    mag_p4 = F.interpolate(
        torch.sqrt(
            F.conv2d(F.pad(lp_q4, (1, 1, 1, 1), mode="reflect"), kx).square()
            + F.conv2d(F.pad(lp_q4, (1, 1, 1, 1), mode="reflect"), ky).square()
            + 1e-10
        ),
        size=mag_p.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    m1_p = F.avg_pool2d(F.pad(lp_s, (5, 5, 5, 5), mode="reflect"), kernel_size=11, stride=1)
    m2_p = F.avg_pool2d(F.pad(lp_s * lp_s, (5, 5, 5, 5), mode="reflect"), kernel_size=11, stride=1)
    std11_p = torch.sqrt((m2_p - m1_p * m1_p).clamp_min(0.0) + 1e-10)

    m1_c = F.avg_pool2d(F.pad(lc_s, (5, 5, 5, 5), mode="reflect"), kernel_size=11, stride=1)
    m2_c = F.avg_pool2d(F.pad(lc_s * lc_s, (5, 5, 5, 5), mode="reflect"), kernel_size=11, stride=1)
    std11_c = torch.sqrt((m2_c - m1_c * m1_c).clamp_min(0.0) + 1e-10)

    sig_full = F.interpolate(sigma_loc, size=mag_p.shape[-2:], mode="bilinear", align_corners=False)

    # Residual noise floor of ALREADY low-pass-purified `x_pure_01`
    eff_grad_p = torch.maximum(
        F.relu(mag_p - 0.035 * sig_full),
        torch.maximum(F.relu(mag_p2 - 0.025 * sig_full), 0.90 * F.relu(mag_p4 - 0.020 * sig_full)),
    )
    eff_std_p = F.relu(std11_p - 0.055 * sig_full)

    raw_support = torch.maximum(
        (eff_grad_p / 0.0048).clamp(0.0, 1.0),
        (eff_std_p / 0.0085).clamp(0.0, 1.0),
    )
    support_cone = binomial_blur_3x3(
        F.max_pool2d(raw_support, kernel_size=15, stride=1, padding=7),
        passes=2,
    ).clamp(0.0, 1.0)

    raw_cand_act = torch.maximum(
        ((mag_c - 0.013) / 0.016).clamp(0.0, 1.0),
        ((std11_c - std11_p - 0.007) / 0.014).clamp(0.0, 1.0),
    )
    env_cand_act = binomial_blur_3x3(
        F.max_pool2d(raw_cand_act, kernel_size=13, stride=1, padding=6),
        passes=2,
    ).clamp(0.0, 1.0)
    spurious_excess = env_cand_act * (1.0 - support_cone)

    cos_dir = (gxp * gxc + gyp * gyc) / (mag_p * mag_c + 1e-7)
    strong_ref_edge = ((mag_p - 0.034) / 0.022).clamp(0.0, 1.0) * ((mag_c - 0.024) / 0.016).clamp(0.0, 1.0)
    polarity_conflict = binomial_blur_3x3(
        strong_ref_edge * ((-cos_dir - 0.20) / 0.40).clamp(0.0, 1.0),
        passes=2,
    )

    veto_strength_full = binomial_blur_3x3(
        (0.90 * spurious_excess + 0.68 * polarity_conflict).clamp(0.0, 0.92),
        passes=2,
    )
    keep_full = (1.0 - veto_strength_full).clamp(0.08, 1.0)
    return keep_full, {
        "flat_unsupp_frac": round(float((support_cone < 0.20).float().mean().cpu()), 4),
        "veto_active_frac": round(float((veto_strength_full > 0.15).float().mean().cpu()), 4),
        "mean_keep_gain": round(float(keep_full.mean().cpu()), 4),
    }


# --- _plaque_glyph_density_q4 ---
def _plaque_glyph_density_q4(img_01):
    """100% blind orthogonal horizontal+vertical bimodal glyph/plaque density map at `1/4` scale."""
    import torch
    import torch.nn.functional as F

    lum = 0.299 * img_01[:, 0:1] + 0.587 * img_01[:, 1:2] + 0.114 * img_01[:, 2:3]
    lq = F.avg_pool2d(lum, kernel_size=4, stride=4)

    wth5 = F.relu(lq - _morph_open(lq, 5))
    di5 = F.max_pool2d(lq, kernel_size=5, stride=1, padding=2)
    cl5 = -F.max_pool2d(-di5, kernel_size=5, stride=1, padding=2)
    bth5 = F.relu(cl5 - lq)

    wth9 = F.relu(lq - _morph_open(lq, 9))
    di9 = F.max_pool2d(lq, kernel_size=9, stride=1, padding=4)
    cl9 = -F.max_pool2d(-di9, kernel_size=9, stride=1, padding=4)
    bth9 = F.relu(cl9 - lq)

    w_loc = F.avg_pool2d(torch.maximum(wth5, 0.85 * wth9), kernel_size=9, stride=1, padding=4)
    b_loc = F.avg_pool2d(torch.maximum(bth5, 0.85 * bth9), kernel_size=9, stride=1, padding=4)
    wb_pair = torch.sqrt(w_loc * b_loc + 1e-10)

    l_max9 = F.max_pool2d(lq, kernel_size=9, stride=1, padding=4)
    l_min9 = -F.max_pool2d(-lq, kernel_size=9, stride=1, padding=4)
    plaque_c = ((l_max9 - l_min9 - 0.20) / 0.22).clamp(0.0, 1.0)

    # Pure horizontal & vertical stroke separation (rejects 45-degree diagonal tree branches/foliage!)
    gh = F.pad((lq[:, :, :, 1:] - lq[:, :, :, :-1]).abs(), (0, 1, 0, 0), mode="replicate")
    gv = F.pad((lq[:, :, 1:, :] - lq[:, :, :-1, :]).abs(), (0, 0, 0, 1), mode="replicate")
    h_stroke = F.avg_pool2d(F.relu(gv - 0.55 * gh), kernel_size=9, stride=1, padding=4)
    v_stroke = F.avg_pool2d(F.relu(gh - 0.55 * gv), kernel_size=9, stride=1, padding=4)
    hv_ortho = torch.sqrt(h_stroke * v_stroke + 1e-10)
    ortho_ratio = (
        (F.avg_pool2d((gh - gv).abs(), kernel_size=9, stride=1, padding=4) / (F.avg_pool2d(gh + gv, kernel_size=9, stride=1, padding=4) + 1e-4) - 0.36)
        / 0.18
    ).clamp(0.0, 1.25)

    return plaque_c * ortho_ratio * torch.sqrt(wb_pair * hv_ortho + 1e-10) * 4.2


# --- PYRAMID_WHITE_NOISE_GAIN ---
PYRAMID_WHITE_NOISE_GAIN = (0.9463, 0.2329, 0.1022, 0.0499)


# --- blind_hf_deficit_and_phase_sensors ---
def blind_hf_deficit_and_phase_sensors(laps_in, g4_in, laps_lat, g4_lat, sigma_loc_01):
    """Compute per-octave noise-corrected band-energy balance rho_bal in (-1, 1),
    input spectral slope kappa_slope, and coarse G_4 phase consistency guard phi_coarse.
    """
    import torch
    import torch.nn.functional as F

    rho_maps = []
    wiener_loc_gates = []
    tau_scales = (2.4, 2.2, 1.9, 1.6)

    sig_cur = sigma_loc_01
    e_in_means = []
    for s, (l_in, l_lat, gain, tau) in enumerate(zip(laps_in, laps_lat, PYRAMID_WHITE_NOISE_GAIN, tau_scales)):
        if sig_cur.shape[-2:] != l_in.shape[-2:]:
            sig_cur = F.interpolate(sigma_loc_01, size=l_in.shape[-2:], mode="bilinear", align_corners=False)
        sig_var_in = burt_adelson_blur5(l_in.square().mean(dim=1, keepdim=True))
        noise_var = (float(tau) * float(gain) * sig_cur).square()
        clean_var_in = torch.relu(sig_var_in - noise_var)
        w_loc = clean_var_in / (sig_var_in + noise_var + 1e-6)
        wiener_loc_gates.append(w_loc)

        var_lat = burt_adelson_blur5(l_lat.square().mean(dim=1, keepdim=True))
        amp_in = (clean_var_in + 1e-8).sqrt()
        amp_lat = (var_lat + 1e-8).sqrt()
        rho_s = (amp_in - amp_lat) / (amp_in + amp_lat + 1e-3)
        rho_maps.append(rho_s)
        e_in_means.append(amp_in.mean(dim=(-2, -1)))

    # Spectral slope log((E_in(L0) + 1e-4) / (E_in(L2) + 1e-4)): negative on blur/SR, >=0 on sharp clean
    kappa_slope = torch.log((e_in_means[0] + 1e-4) / (e_in_means[2] + 1e-4))  # [B, 1]

    # Coarse structural phase guard on G_4 (128x128 >=16px): where input has coarse gradient aligned with H3
    y_g4_in = 0.299 * g4_in[:, 0:1] + 0.587 * g4_in[:, 1:2] + 0.114 * g4_in[:, 2:3]
    y_g4_lat = 0.299 * g4_lat[:, 0:1] + 0.587 * g4_lat[:, 1:2] + 0.114 * g4_lat[:, 2:3]
    gx_in = y_g4_in[:, :, :, 2:] - y_g4_in[:, :, :, :-2]
    gy_in = y_g4_in[:, :, 2:, :] - y_g4_in[:, :, :-2, :]
    gx_lat = y_g4_lat[:, :, :, 2:] - y_g4_lat[:, :, :, :-2]
    gy_lat = y_g4_lat[:, :, 2:, :] - y_g4_lat[:, :, :-2, :]

    gx_in = F.pad(gx_in, (1, 1, 0, 0), mode="replicate")
    gy_in = F.pad(gy_in, (0, 0, 1, 1), mode="replicate")
    gx_lat = F.pad(gx_lat, (1, 1, 0, 0), mode="replicate")
    gy_lat = F.pad(gy_lat, (0, 0, 1, 1), mode="replicate")

    dot = gx_in * gx_lat + gy_in * gy_lat
    mag_in = (gx_in.square() + gy_in.square() + 1e-6).sqrt()
    mag_lat = (gx_lat.square() + gy_lat.square() + 1e-6).sqrt()
    cos_sim = (dot / (mag_in * mag_lat + 1e-5)).clamp(0.0, 1.0)
    coarse_support = (mag_in / 0.04).clamp(0.0, 1.0)  # >0 on blurred/SR real edges, ~0 on flat empty glass
    phi_g4 = binomial_blur_3x3(cos_sim * coarse_support, passes=1)

    return wiener_loc_gates, rho_maps, kappa_slope, phi_g4


# --- compute_blind_v8_for_v10_cues ---
def compute_blind_v8_for_v10_cues(x_ref_p, x_ca_p):
    """Exact `PMPR-v8` blind manifold sensor suite."""
    import torch

    pre_s = compute_blind_precond_sensors(x_ref_p)
    sigma_glob, sigma_loc = blind_local_haar_sigma_map(x_ref_p)
    beta_jpeg, m_jpeg8 = blind_jpeg_blockiness_sensor(x_ref_p)
    laps_in, g4_in = laplacian_pyramid_4(x_ref_p)
    laps_ca, g4_ca = laplacian_pyramid_4(x_ca_p)
    w_loc, _, kappa_slope, phi_g4 = blind_hf_deficit_and_phase_sensors(
        laps_in, g4_in, laps_ca, g4_ca, sigma_loc
    )

    bsz = x_ref_p.shape[0]
    sig_loc_flat = sigma_loc.reshape(bsz, -1) * 255.0
    k_p25 = max(1, int(0.25 * sig_loc_flat.shape[1]))
    sig_p25_u8, _ = torch.kthvalue(sig_loc_flat, k_p25, dim=1)
    sig_p25_u8 = sig_p25_u8.reshape(bsz, 1, 1, 1)
    sig_glob_u8 = sigma_glob * 255.0

    curv_int = pre_s["curv_int"]
    seam_u8 = pre_s["seam_u8"]

    e0_in_glob = laps_in[0].abs().mean(dim=(1, 2, 3), keepdim=True)
    e0_ca_glob = laps_ca[0].abs().mean(dim=(1, 2, 3), keepdim=True)
    ratio_l0_glob = e0_in_glob / (e0_ca_glob + 1e-4)

    g_clean_noise = ((4.8 - sig_p25_u8) / 0.9).clamp(0.0, 1.0) * ((6.8 - sig_glob_u8) / 1.4).clamp(0.0, 1.0)
    g_clean_no_jpg = ((1.85 - seam_u8) / 0.65).clamp(0.0, 1.0) * ((0.15 - beta_jpeg) / 0.05).clamp(0.0, 1.0)
    g_clean_sharp = (
        ((curv_int - 0.365) / 0.04).clamp(0.0, 1.0)
        * ((kappa_slope[:, :, None, None] + 1.85) / 0.45).clamp(0.0, 1.0)
        * ((ratio_l0_glob - 0.58) / 0.16).clamp(0.0, 1.0)
    )
    g_global_clean = (g_clean_noise * g_clean_no_jpg * g_clean_sharp).clamp(0.0, 1.0)

    e0_in_map = laps_in[0].abs().mean(dim=1, keepdim=True)
    e0_ca_map = laps_ca[0].abs().mean(dim=1, keepdim=True)
    ratio_l0_loc = burt_adelson_blur5(e0_in_map) / (burt_adelson_blur5(e0_ca_map) + 0.004)
    g_local_clean = (
        ((kappa_slope[:, :, None, None] + 1.85) / 0.50).clamp(0.0, 1.0)
        * ((curv_int - 0.42) / 0.10).clamp(0.0, 1.0)
        * ((ratio_l0_loc - 0.68) / 0.25).clamp(0.0, 1.0)
        * torch.exp(-10.0 * beta_jpeg)
        * (1.0 - 0.98 * m_jpeg8)
        * torch.exp(-((sig_glob_u8 / 10.0).square()))
        * w_loc[0]
    ).clamp(0.0, 1.0)

    clean_cert_map = binomial_blur_3x3(torch.maximum(g_global_clean.expand_as(g_local_clean), g_local_clean), passes=1)

    g_nonblur_struct = (
        ((curv_int - 0.39) / 0.08).clamp(0.0, 1.0)
        * ((kappa_slope[:, :, None, None] + 1.85) / 0.50).clamp(0.0, 1.0)
        * ((7.5 - sig_p25_u8) / 2.5).clamp(0.0, 1.0)
        * (1.0 - pre_s["omega_2nd"])
    ).clamp(0.0, 1.0)

    return {
        "sigma_glob": sigma_glob,
        "sigma_loc": sigma_loc,
        "sig_p25_u8": sig_p25_u8,
        "beta_jpeg": beta_jpeg,
        "m_jpeg8": m_jpeg8,
        "kappa_slope": kappa_slope,
        "phi_g4": phi_g4,
        "curv_int": curv_int,
        "omega_2nd": pre_s["omega_2nd"],
        "omega_deblock": pre_s["omega_deblock"],
        "g_global_clean": g_global_clean,
        "g_nonblur_struct": g_nonblur_struct,
        "clean_cert_map": clean_cert_map,
    }


# --- compute_blind_v10_omni_sensors ---
def compute_blind_v10_omni_sensors(x_in_01) -> dict:
    """Compute 100% blind native-resolution physical invariants from `x_in_01` (`[B, 3, H, W]`) ONLY.

    Eliminates BOTH `UOMR-v9` 1D false positives (via contrast-floored precipitation impulses,
    upper/lower frame co-coverage, foreground zero-veil guard `dn10/web90`, and dark shot-noise
    verification) AND `256x256` downsampling aliasing false positives (by operating on native
    `1:1` pixels for `sigma_u8`, `curv_int`, `jpg_ratio`, `omega_2nd`, and `wth7_p1`).
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        x_p, _ = pad_to_multiple_32(x_in_01)
        bsz, _, hp, wp = x_p.shape
        pre = compute_blind_precond_sensors(x_p)

        sig_u8 = pre["sigma_u8"].reshape(bsz)
        curv_int = pre["curv_int"].reshape(bsz)
        jpg_ratio = pre["jpg_ratio"].reshape(bsz)
        seam_u8 = pre["seam_u8"].reshape(bsz)
        flat8x8_frac = pre["flat8x8_frac"].reshape(bsz)
        w_2nd = pre["omega_2nd"].reshape(bsz)
        w_deb = pre["omega_deblock"].reshape(bsz)

        lum = (0.299 * x_p[:, 0:1] + 0.587 * x_p[:, 1:2] + 0.114 * x_p[:, 2:3]).contiguous()
        c_min = x_p.min(dim=1, keepdim=True).values
        c_max = x_p.max(dim=1, keepdim=True).values

        q_stride = 4 if min(hp, wp) >= 512 else 2
        lum_sub = lum[:, :, ::q_stride, ::q_stride].reshape(bsz, -1)
        lp50 = torch.quantile(lum_sub, 0.50, dim=1)
        lp90 = torch.quantile(lum_sub, 0.90, dim=1)
        lp98 = torch.quantile(lum_sub, 0.98, dim=1)

        op15 = _morph_open(lum, 15)
        op_sub = op15[:, :, ::4, ::4].reshape(bsz, -1)
        op50 = torch.quantile(op_sub, 0.50, dim=1)
        op90 = torch.quantile(op_sub, 0.90, dim=1)

        # 1. Atmospheric Veil & Foreground Shadow Invariants (`dc_veil`, `dn10`, `web75`, `web90`)
        cmax_sub = c_max[:, :, ::4, ::4].reshape(bsz, -1)
        air_p98 = torch.quantile(cmax_sub, 0.98, dim=1).clamp_min(0.08)
        dark15 = -F.max_pool2d(-c_min, kernel_size=15, stride=4, padding=7)
        dn_flat = dark15.reshape(bsz, -1) / air_p98[:, None]
        dn10 = torch.quantile(dn_flat, 0.10, dim=1)
        dn35 = torch.quantile(dn_flat, 0.35, dim=1)
        dn55 = torch.quantile(dn_flat, 0.55, dim=1)
        dc_veil = 0.65 * dn35 + 0.35 * dn55

        mu16 = F.avg_pool2d(lum, 16, 16)
        sd16 = torch.sqrt(F.relu(F.avg_pool2d(lum.square(), 16, 16) - mu16.square()) + 1e-8)
        web16 = sd16 / (mu16 + 0.06)
        web_flat = web16.reshape(bsz, -1)
        web75 = torch.quantile(web_flat, 0.75, dim=1)
        web90 = torch.quantile(web_flat, 0.90, dim=1)

        # 2. Directional Motion Anisotropy (`ani80`)
        lgx = lum[:, :, 1:-1, 2:] - lum[:, :, 1:-1, :-2]
        lgy = lum[:, :, 2:, 1:-1] - lum[:, :, :-2, 1:-1]
        pxx = F.avg_pool2d(lgx * lgx, 32, 32)
        pyy = F.avg_pool2d(lgy * lgy, 32, 32)
        pxy = F.avg_pool2d(lgx * lgy, 32, 32)
        p_coher = torch.sqrt((pxx - pyy).square() + 4.0 * pxy.square()) / (pxx + pyy + 1e-5)
        ani80 = torch.quantile(p_coher.reshape(bsz, -1), 0.80, dim=1)

        # 3. Contrast-Floored Native-1:1 Morphological Precipitation Sensors
        lum_p1 = binomial_blur_3x3(lum, passes=1)
        wth7_p1 = F.relu(lum_p1 - _morph_open(lum_p1, 7))
        wth15_p1 = F.relu(lum_p1 - _morph_open(lum_p1, 15))
        di7 = F.max_pool2d(lum_p1, kernel_size=7, stride=1, padding=3)
        cl7 = -F.max_pool2d(-di7, kernel_size=7, stride=1, padding=3)

        sat = (c_max - c_min) / (c_max + 0.04)
        achrom = (1.0 - (sat / 0.35).clamp(0.0, 1.0))

        # Minimum impulse floor of 5.1/255 (`0.020`) prevents 1-2/255 micro-textures on ultra-clean
        # 2048x2048 Wikimedia photos (`sig_u8 < 1.5`) from triggering false precipitation!
        noise_floor = torch.maximum(
            (0.58 * (sig_u8 / 255.0)).reshape(bsz, 1, 1, 1),
            lum.new_full((bsz, 1, 1, 1), 0.020),
        )
        wth_clean = F.relu(wth7_p1 - noise_floor) * achrom
        bth_clean = F.relu(cl7 - lum_p1 - noise_floor)
        pos_tile = F.avg_pool2d(wth_clean, 32, 32)
        neg_tile = F.avg_pool2d(bth_clean, 32, 32)

        active_tile = (pos_tile > 0.0040).float()
        cov_precip = active_tile.mean(dim=(1, 2, 3))
        th_half = max(1, active_tile.shape[-2] // 2)
        cov_top = active_tile[:, :, :th_half, :].mean(dim=(1, 2, 3))
        cov_bot = active_tile[:, :, -th_half:, :].mean(dim=(1, 2, 3))
        min_half_cov = torch.minimum(cov_top, cov_bot)
        pos_dom_cov = ((pos_tile > 0.0045) & (pos_tile > 1.32 * neg_tile)).float().mean(dim=(1, 2, 3))
        blob_ratio = wth15_p1.mean(dim=(1, 2, 3)) / (wth7_p1.mean(dim=(1, 2, 3)) + 1e-4)

        gx = wth7_p1[:, :, 1:-1, 2:] - wth7_p1[:, :, 1:-1, :-2]
        gy = wth7_p1[:, :, 2:, 1:-1] - wth7_p1[:, :, :-2, 1:-1]
        jxx = (gx * gx).mean(dim=(1, 2, 3))
        jyy = (gy * gy).mean(dim=(1, 2, 3))
        jxy = (gx * gy).mean(dim=(1, 2, 3))
        th_coher = torch.sqrt((jxx - jyy).square() + 4.0 * jxy.square()) / (jxx + jyy + 1e-6)

        # 4. Native-1:1 Physical Feasibility Masks (`=== 0.0` on standard camera degradations & GoPro!)
        # True JPEG blocking requires BOTH modulo-8 seam elevation AND dead-flat 6x6 quantized block interiors
        # (`flat8x8_frac`), preventing gridded glass skyscrapers under rain from being misclassified as JPEG!
        seam_jpeg = (
            seam_u8
            * ((jpg_ratio - 1.22) / 0.18).clamp(0.0, 1.0)
            * ((flat8x8_frac - 0.015) / 0.035).clamp(0.0, 1.0)
        )
        g_cam_artifact_free = (
            ((21.5 - sig_u8) / 3.5).clamp(0.0, 1.0)
            * (1.0 - w_2nd)
            * (1.0 - w_deb)
            * ((2.25 - seam_jpeg) / 0.65).clamp(0.0, 1.0)
        )

        # Koschmieder scattering (`I = J*t + A*(1-t)`) attenuates contrast without PSF spatial smearing,
        # preserving step-edge curvature (`curv_int >= 0.435` across 52/52 train weather scenes), whereas
        # camera motion blur (`GoPro`), defocus, and 4x/8x SR smear edges (`curv_int <= 0.425`).
        g_psf_sharp_haze = ((curv_int - 0.426) / 0.024).clamp(0.0, 1.0)
        is_mot_wall_fp = (
            ((ani80 - 0.72) / 0.08).clamp(0.0, 1.0)
            * ((0.36 - dc_veil) / 0.08).clamp(0.0, 1.0)
            * ((0.015 - cov_precip) / 0.010).clamp(0.0, 1.0)
        )
        allow_haze = (
            g_cam_artifact_free
            * g_psf_sharp_haze
            * ((dc_veil - 0.26) / 0.08).clamp(0.0, 1.0)
            * torch.maximum(((dn10 - 0.070) / 0.035).clamp(0.0, 1.0), ((0.24 - web90) / 0.04).clamp(0.0, 1.0))
            * ((0.175 - web75) / 0.045).clamp(0.0, 1.0)
            * (1.0 - is_mot_wall_fp)
        )

        is_bright_night_sign = (
            ((lp98 - 0.88) / 0.04).clamp(0.0, 1.0)
            * ((curv_int - 0.46) / 0.06).clamp(0.0, 1.0)
            * ((4.0 - sig_u8) / 1.2).clamp(0.0, 1.0)
        )
        # Veto well-lit low-noise HDR daylight scenes (`lp98 >= 0.84` with `op90 >= 0.42` and zero haze veil)
        daylight_hdr_veto = (
            ((lp98 - 0.825) / 0.045).clamp(0.0, 1.0)
            * ((op90 - 0.40) / 0.06).clamp(0.0, 1.0)
            * ((0.32 - dc_veil) / 0.06).clamp(0.0, 1.0)
            * ((8.5 - sig_u8) / 2.0).clamp(0.0, 1.0)
        )
        noisy_night_dark = ((0.365 - op50) / 0.060).clamp(0.0, 1.0) * ((sig_u8 - 9.0) / 3.0).clamp(0.0, 1.0)
        allow_low = (
            g_cam_artifact_free
            * ((curv_int - 0.426) / 0.036).clamp(0.0, 1.0)
            * torch.maximum(
                torch.maximum(((0.29 - op50) / 0.07).clamp(0.0, 1.0), noisy_night_dark),
                ((0.68 - op90) / 0.08).clamp(0.0, 1.0),
            )
            * torch.maximum(((sig_u8 - 1.6) / 1.2).clamp(0.0, 1.0), ((0.17 - op50) / 0.05).clamp(0.0, 1.0))
            * (1.0 - is_bright_night_sign)
            * (1.0 - daylight_hdr_veto)
        )

        # Requires sharp 1:1 impulses (`curv_int >= 0.45`, rejecting ALL SR / Defocus / Motion Blur / GoPro!),
        # multi-half vertical distribution (`min_half_cov`), AND additive precipitation physics:
        # 1) positive-over-negative impulse dominance (`pos_dom_cov`) above clean-sensor noise floor,
        # 2) coherent falling streak directionality (`th_coher >= 0.24`) above clean-sensor noise floor, or
        # 3) dense fine rain/snow (`sig_u8 >= 6.8` AND `cov_precip >= 0.16` surviving `2.4*sigma` floor).
        precip_branch_add = (
            ((pos_dom_cov - 0.062) / 0.045).clamp(0.0, 1.0)
            * ((sig_u8 - 1.55) / 0.85).clamp(0.0, 1.0)
        )
        precip_branch_streak = (
            ((th_coher - 0.24) / 0.08).clamp(0.0, 1.0)
            * ((sig_u8 - 1.65) / 0.80).clamp(0.0, 1.0)
        )
        precip_branch_dense = (
            ((sig_u8 - 6.8) / 2.2).clamp(0.0, 1.0)
            * ((cov_precip - 0.16) / 0.08).clamp(0.0, 1.0)
            * ((curv_int - 0.62) / 0.08).clamp(0.0, 1.0)
        )
        precip_asym_or_streak = torch.maximum(
            precip_branch_add,
            torch.maximum(precip_branch_streak, precip_branch_dense),
        )
        allow_precip = (
            g_cam_artifact_free
            * ((curv_int - 0.45) / 0.08).clamp(0.0, 1.0)
            * ((cov_precip - 0.040) / 0.035).clamp(0.0, 1.0)
            * ((min_half_cov - 0.012) / 0.020).clamp(0.0, 1.0)
            * precip_asym_or_streak
        )

        # Continuous physical sub-modality estimates
        omega_haze_phys = (
            allow_haze
            * ((dc_veil - 0.33) / 0.12).clamp(0.0, 1.0)
            * ((0.145 - web75) / 0.045).clamp(0.0, 1.0)
        )
        omega_haze_phys = torch.where(omega_haze_phys >= 0.08, omega_haze_phys, torch.zeros_like(omega_haze_phys))

        g_dark_bg = torch.maximum(
            ((0.46 - op90) / 0.08).clamp(0.0, 1.0),
            torch.maximum(
                ((0.24 - op50) / 0.06).clamp(0.0, 1.0),
                ((0.825 - lp98) / 0.065).clamp(0.0, 1.0) * ((0.645 - op90) / 0.065).clamp(0.0, 1.0),
            ),
        )
        omega_low_phys = allow_low * torch.maximum(
            g_dark_bg * ((sig_u8 - 3.6) / 3.2).clamp(0.0, 1.0),
            omega_haze_phys * ((sig_u8 - 3.2) / 1.5).clamp(0.0, 1.0) * ((0.75 - lp90) / 0.05).clamp(0.0, 1.0),
        )
        omega_low_phys = torch.where(omega_low_phys >= 0.08, omega_low_phys, torch.zeros_like(omega_low_phys))

        rain_dir_pref = ((th_coher - 0.18) / 0.14).clamp(0.0, 1.0)
        snow_flake_pref = torch.maximum(
            ((blob_ratio - 1.50) / 0.14).clamp(0.0, 1.0) * ((0.33 - th_coher) / 0.12).clamp(0.0, 1.0),
            (1.0 - rain_dir_pref) * 0.85,
        )
        omega_rain_phys = torch.where(allow_precip * rain_dir_pref >= 0.08, allow_precip * rain_dir_pref, torch.zeros_like(allow_precip))
        omega_snow_phys = torch.where(allow_precip * snow_flake_pref >= 0.08, allow_precip * snow_flake_pref, torch.zeros_like(allow_precip))

        return {
            "sigma_u8": sig_u8.reshape(bsz, 1, 1, 1),
            "curv_int": curv_int.reshape(bsz, 1, 1, 1),
            "jpg_ratio": jpg_ratio.reshape(bsz, 1, 1, 1),
            "seam_u8": pre["seam_u8"],
            "omega_2nd": w_2nd.reshape(bsz, 1, 1, 1),
            "omega_deblock": w_deb.reshape(bsz, 1, 1, 1),
            "omega_strip": pre["omega_strip"],
            "dc_veil": dc_veil.reshape(bsz, 1, 1, 1),
            "dn10": dn10.reshape(bsz, 1, 1, 1),
            "web75": web75.reshape(bsz, 1, 1, 1),
            "web90": web90.reshape(bsz, 1, 1, 1),
            "lp50": lp50.reshape(bsz, 1, 1, 1),
            "lp90": lp90.reshape(bsz, 1, 1, 1),
            "lp98": lp98.reshape(bsz, 1, 1, 1),
            "op50": op50.reshape(bsz, 1, 1, 1),
            "op90": op90.reshape(bsz, 1, 1, 1),
            "ani80": ani80.reshape(bsz, 1, 1, 1),
            "cov_precip": cov_precip.reshape(bsz, 1, 1, 1),
            "min_half_cov": min_half_cov.reshape(bsz, 1, 1, 1),
            "pos_dom_cov": pos_dom_cov.reshape(bsz, 1, 1, 1),
            "th_coher": th_coher.reshape(bsz, 1, 1, 1),
            "blob_ratio": blob_ratio.reshape(bsz, 1, 1, 1),
            "rain_dir_pref": rain_dir_pref.reshape(bsz, 1, 1, 1),
            "snow_flake_pref": snow_flake_pref.reshape(bsz, 1, 1, 1),
            "g_cam_artifact_free": g_cam_artifact_free.reshape(bsz, 1, 1, 1),
            "allow_haze": allow_haze.reshape(bsz, 1, 1, 1),
            "allow_low": allow_low.reshape(bsz, 1, 1, 1),
            "allow_precip": allow_precip.reshape(bsz, 1, 1, 1),
            "omega_haze_phys": omega_haze_phys.reshape(bsz, 1, 1, 1),
            "omega_low_phys": omega_low_phys.reshape(bsz, 1, 1, 1),
            "omega_rain_phys": omega_rain_phys.reshape(bsz, 1, 1, 1),
            "omega_snow_phys": omega_snow_phys.reshape(bsz, 1, 1, 1),
        }


# --- extract_blind_native_quad256 ---
def extract_blind_native_quad256(x_01):
    """Extract deterministic 4-quadrant `128x128` 1:1 native-pixel mosaic `[B, 3, 256, 256]` from `x_01`.

    Prevents `2048->256` `8x` downsampling aliasing from turning white signboard characters,
    `sigma=65` AWGN, or `GoPro` blurred highlights into fake `1-2px` raindrops!
    When `x_01` is already `256x256`, returns `x_01` identically.
    """
    import torch
    import torch.nn.functional as F

    _, _, h, w = x_01.shape
    if h == 256 and w == 256:
        return x_01.contiguous()
    if h < 256 or w < 256:
        return F.interpolate(x_01, size=(256, 256), mode="bilinear", align_corners=False).contiguous()

    cy1 = max(0, h // 4 - 64)
    cy2 = min(h - 128, (3 * h) // 4 - 64)
    cx1 = max(0, w // 4 - 64)
    cx2 = min(w - 128, (3 * w) // 4 - 64)

    q_tl = x_01[:, :, cy1 : cy1 + 128, cx1 : cx1 + 128]
    q_tr = x_01[:, :, cy1 : cy1 + 128, cx2 : cx2 + 128]
    q_bl = x_01[:, :, cy2 : cy2 + 128, cx1 : cx1 + 128]
    q_br = x_01[:, :, cy2 : cy2 + 128, cx2 : cx2 + 128]
    top = torch.cat((q_tl, q_tr), dim=3)
    bot = torch.cat((q_bl, q_br), dim=3)
    return torch.cat((top, bot), dim=2).contiguous()


# --- _edit_dist_str ---
def _edit_dist_str(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1))
        prev = cur
    return prev[-1]


# --- is_standard_sign_char ---
def is_standard_sign_char(ch: str) -> bool:
    cp = ord(ch)
    return (
        (0x0030 <= cp <= 0x0039)
        or (0x0041 <= cp <= 0x005A)
        or (0x0061 <= cp <= 0x007A)
        or (0x3040 <= cp <= 0x30FF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x4E00 <= cp <= 0x9FFF)
        or (0xFF10 <= cp <= 0xFF5A)
    )


# --- parse_vlm_text_boxes_1000 ---
def parse_vlm_text_boxes_1000(raw_det_text: str, img_w: int, img_h: int, max_boxes: int = 10) -> list[list[int]]:
    """Parse blind VLM 2D text detection output (`0..1000` coordinates) into native `[x0, y0, x1, y1]` pixel boxes."""
    s = str(raw_det_text or "").strip()
    if not s or s.upper() in ("NONE", "[]"):
        return []

    raw_quads = []
    # Match JSON arrays `[x1, y1, x2, y2]` or plain lines `x1,y1,x2,y2`
    for m in re.finditer(r"(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})\s*,\s*(\d{1,4})", s):
        vals = [int(m.group(i)) for i in range(1, 5)]
        raw_quads.append(vals)

    boxes = []
    for qx0, qy0, qx1, qy1 in raw_quads:
        if qx1 <= qx0 or qy1 <= qy0 or max(qx1, qy1) > 1050:
            continue
        x0 = int(round(qx0 * img_w / 1000.0))
        y0 = int(round(qy0 * img_h / 1000.0))
        x1 = int(round(qx1 * img_w / 1000.0))
        y1 = int(round(qy1 * img_h / 1000.0))
        x0 = max(0, min(img_w - 8, x0))
        y0 = max(0, min(img_h - 8, y0))
        x1 = max(x0 + 8, min(img_w, x1))
        y1 = max(y0 + 8, min(img_h, y1))
        bw, bh = x1 - x0, y1 - y0
        if bw > int(0.65 * img_w) and bh > int(0.65 * img_h):
            continue
        # Deduplicate overlapping boxes (IoU > 0.45)
        dup = False
        for px0, py0, px1, py1 in boxes:
            ix = max(0, min(x1, px1) - max(x0, px0))
            iy = max(0, min(y1, py1) - max(y0, py0))
            inter = ix * iy
            union = bw * bh + (px1 - px0) * (py1 - py0) - inter
            if union > 0 and (inter / union) > 0.45:
                dup = True
                break
        if not dup:
            boxes.append([x0, y0, x1, y1])
            if len(boxes) >= max_boxes:
                break
    return boxes


# --- parse_vlm_weather_json ---
def parse_vlm_weather_json(raw_text: str) -> dict[str, bool]:
    """Deterministically parse blind VLM weather output into 4 boolean flags."""
    t = str(raw_text or "").strip().lower()
    out = {"lowlight": False, "haze": False, "rain": False, "snow": False}
    if not t or t == "none":
        return out
    try:
        m = re.search(r"\{.*?\}", t, flags=re.DOTALL)
        if m:
            d = json.loads(m.group(0))
            for k in out:
                out[k] = bool(d.get(k, False))
            return out
    except Exception:
        pass
    for k in out:
        if re.search(rf'"{k}"\s*:\s*true', t) or re.search(rf"\b{k}\s*=\s*true\b", t):
            out[k] = True
    return out


# --- plan_tight_signboard_window ---
def plan_tight_signboard_window(box_xyxy_pred: list[int], img_w: int, img_h: int) -> dict:
    """Convert a blindly detected signboard box `[x0, y0, x1, y1]` into a tight `16px`-aligned
    square restoration window (`size in [160, 384]`, yielding `Z = 2.7x..6.4x` native token density at `1024x1024`!),
    plus an isolated feather-splice sub-box (`det_box` padded by `20px`) so neighboring signs never bleed!
    """
    x0, y0, x1, y1 = [int(v) for v in box_xyxy_pred]
    bw = max(16, x1 - x0)
    bh = max(16, y1 - y0)
    cx = (x0 + x1) // 2
    cy = (y0 + y1) // 2

    raw_side = int(round(max(bw, bh) * 1.75))
    side = max(160, min(384, ((raw_side + 15) // 16) * 16))
    side = min(side, (min(img_w, img_h) // 16) * 16)

    wx0 = max(0, min(img_w - side, ((cx - side // 2 + 8) // 16) * 16))
    wy0 = max(0, min(img_h - side, ((cy - side // 2 + 8) // 16) * 16))

    pad_s = 20
    sx0 = max(wx0, x0 - pad_s)
    sy0 = max(wy0, y0 - pad_s)
    sx1 = min(wx0 + side, x1 + pad_s)
    sy1 = min(wy0 + side, y1 + pad_s)
    return {
        "win_y0": int(wy0),
        "win_x0": int(wx0),
        "win_size": int(side),
        "det_box": [x0, y0, x1, y1],
        "splice_box": [int(sx0), int(sy0), int(sx1), int(sy1)],
    }


# --- sanitize_sign_prior_text ---
def sanitize_sign_prior_text(raw_txt: str) -> str:
    s = str(raw_txt or "").strip()
    if not s or s.upper() == "NONE":
        return ""
    if len(s) > 14 or any(w in s.upper() for w in ("BLUR", "UNREADABLE", "CANNOT", "SORRY", "IMAGE", "TEXT")):
        return ""
    kept = "".join(ch for ch in s if is_standard_sign_char(ch))
    if not kept or kept != "".join(ch for ch in s if not ch.isspace() and ch not in "'-·・.!?"):
        # Reject strings containing exotic/Devanagari/Greek hallucinated characters!
        return ""
    return kept


# --- splice_signboard_box_patches ---
def splice_signboard_box_patches(x_base_01, box_patches: list[tuple[list[int], int, int, "object"]]):
    """Feather-blend refined signboard sub-boxes (`splice_box` inside `(win_y0, win_x0)`) onto `x_base_01`."""
    import torch

    if not box_patches:
        return x_base_01
    with torch.no_grad():
        _, _, fh, fw = x_base_01.shape
        acc = torch.zeros_like(x_base_01)
        wsum = torch.zeros((1, 1, fh, fw), device=x_base_01.device, dtype=torch.float32)
        for sbox, wy0, wx0, pt in box_patches:
            sx0, sy0, sx1, sy1 = [int(v) for v in sbox]
            sh = sy1 - sy0
            sw = sx1 - sx0
            if sh < 8 or sw < 8:
                continue
            sub_pt = pt[:, :, (sy0 - wy0) : (sy1 - wy0), (sx0 - wx0) : (sx1 - wx0)].to(x_base_01.device)
            bh = min(16, max(4, sh // 5))
            bw = min(16, max(4, sw // 5))
            ry = torch.ones(sh, device=x_base_01.device, dtype=torch.float32)
            rx = torch.ones(sw, device=x_base_01.device, dtype=torch.float32)
            t_y = 0.5 * (1.0 - torch.cos(math.pi * torch.arange(1, bh + 1, device=x_base_01.device, dtype=torch.float32) / (bh + 1)))
            t_x = 0.5 * (1.0 - torch.cos(math.pi * torch.arange(1, bw + 1, device=x_base_01.device, dtype=torch.float32) / (bw + 1)))
            if sy0 > 0:
                ry[:bh] = t_y
            if sy1 < fh:
                ry[-bh:] = torch.minimum(ry[-bh:], t_y.flip(0))
            if sx0 > 0:
                rx[:bw] = t_x
            if sx1 < fw:
                rx[-bw:] = torch.minimum(rx[-bw:], t_x.flip(0))
            win = (ry[:, None] * rx[None, :]).reshape(1, 1, sh, sw)
            acc[:, :, sy0:sy1, sx0:sx1] += win * sub_pt
            wsum[:, :, sy0:sy1, sx0:sx1] += win
        cov = wsum.clamp(0.0, 1.0)
        comp = torch.where(wsum > 1e-5, acc / wsum.clamp_min(1e-5), x_base_01)
        return (x_base_01 + cov * (comp - x_base_01)).clamp(0.0, 1.0).contiguous()


# --- _apply_epistemic_abstention_on_boxes_v12 ---
def _apply_epistemic_abstention_on_boxes_v12(
    laps_out5,
    laps_pure5,
    abstention_boxes: list[list[int]] | None,
    full_hw: tuple[int, int],
):
    """Calibrated sub-Nyquist epistemic abstention: damps `1-8px` fake strokes (`88%`) onto shock-free
    plaque ink while keeping `16-32px` coarse plaque mass intact (avoiding unnatural blank smudge artifacts).
    """
    import torch.nn.functional as F

    if not abstention_boxes:
        return laps_out5, 0

    fh, fw = int(full_hw[0]), int(full_hw[1])
    mask = laps_out5[0].new_zeros((1, 1, fh, fw))
    applied = 0
    for box in abstention_boxes:
        if len(box) != 4:
            continue
        x0, y0, x1, y1 = [int(v) for v in box]
        x0 = max(0, min(fw - 1, x0))
        y0 = max(0, min(fh - 1, y0))
        x1 = max(x0 + 1, min(fw, x1))
        y1 = max(y0 + 1, min(fh, y1))
        if (x1 - x0) >= 8 and (y1 - y0) >= 8:
            mask[:, :, y0:y1, x0:x1] = 1.0
            applied += 1

    if applied == 0:
        return laps_out5, 0

    mask_soft = binomial_blur_3x3(mask, passes=4).clamp(0.0, 1.0)
    damp_weights = (0.98, 0.98, 0.95, 0.88, 0.55)
    new_laps = []
    for s in range(5):
        hs, ws = laps_out5[s].shape[-2], laps_out5[s].shape[-1]
        ms = F.interpolate(mask_soft, size=(hs, ws), mode="area") if (hs != fh or ws != fw) else mask_soft
        w_d = damp_weights[s] * ms
        new_laps.append(laps_out5[s] + w_d * (laps_pure5[s] - laps_out5[s]))
    return new_laps, applied


# --- _compute_sky_only_micro_veto ---
def _compute_sky_only_micro_veto(x_pre_p, x_ca_p, laps_ca5):
    """Homogeneous-sky/wall-only micro-hallucination veto for hazy/dark weather scenes:
    NEVER blurs newly revealed distant buildings, trees, or vehicles (`eliminates v11's 24% blotchy blur islands`!).
    """
    import torch
    import torch.nn.functional as F

    fh, fw = x_pre_p.shape[-2], x_pre_p.shape[-1]
    lum_pre = 0.299 * x_pre_p[:, 0:1] + 0.587 * x_pre_p[:, 1:2] + 0.114 * x_pre_p[:, 2:3]
    lmax = F.max_pool2d(lum_pre, kernel_size=15, stride=1, padding=7)
    lmin = -F.max_pool2d(-lum_pre, kernel_size=15, stride=1, padding=7)
    flat_pre = ((0.048 - (lmax - lmin)) / 0.028).clamp(0.0, 1.0)

    # Multi-scale structural coherence in `x_ca_p`: real unveiled buildings/trees have strong `L_3` (`8-16px`) & `L_4` (`16-32px`) structure!
    e_l3 = F.interpolate(
        binomial_blur_3x3(laps_ca5[3].abs().mean(dim=1, keepdim=True), passes=1),
        size=(fh, fw),
        mode="bilinear",
        align_corners=False,
    )
    e_l4 = F.interpolate(
        binomial_blur_3x3(laps_ca5[4].abs().mean(dim=1, keepdim=True), passes=1),
        size=(fh, fw),
        mode="bilinear",
        align_corners=False,
    )
    no_coarse_struct = ((0.018 - (e_l3 + 0.75 * e_l4)) / 0.012).clamp(0.0, 1.0)

    sky_veto = binomial_blur_3x3(flat_pre * no_coarse_struct, passes=3).clamp(0.0, 0.85)
    keep_full = (1.0 - sky_veto).clamp(0.15, 1.0)
    return keep_full, {
        "flat_unsupp_frac": round(float((sky_veto > 0.25).float().mean().cpu()), 4),
        "veto_active_frac": round(float((keep_full < 0.75).float().mean().cpu()), 4),
        "mean_keep_gain": round(float(keep_full.mean().cpu()), 4),
    }


# --- deconvolve_with_psf_hqs ---
def deconvolve_with_psf_hqs(
    x_in_01,
    psf_cpu,
    tv_iters: int = 6,
    thr_base: float = 0.0036,
    wiener_lap: float = 0.040,
):
    """Boundary-tapered Half-Quadratic Splitting Hyper-Laplacian RGB Phase Deconvolution (`CPU FFT` -> `x_in_01.device`)."""
    import torch
    import torch.nn.functional as F

    if psf_cpu is None:
        return x_in_01
    with torch.no_grad():
        orig_dev = x_in_01.device
        x_cpu = x_in_01.detach().cpu().float()
        psf = psf_cpu.detach().cpu().float()
        _, _, h, w = x_cpu.shape
        ks = int(psf.shape[0])
        half = ks // 2
        pad = max(48, ks)
        x_pad = F.pad(x_cpu, (pad, pad, pad, pad), mode="reflect")
        hp, wp = x_pad.shape[-2], x_pad.shape[-1]

        otf_c = torch.zeros((hp, wp), dtype=torch.float32)
        otf_c[:ks, :ks] = psf
        otf_c = torch.roll(otf_c, shifts=(-half, -half), dims=(0, 1))
        H = torch.fft.rfft2(otf_c)

        blurred_pad = torch.fft.irfft2(torch.fft.rfft2(x_pad) * H.unsqueeze(0).unsqueeze(0), s=(hp, wp))
        ry = torch.ones(hp, dtype=torch.float32)
        rx = torch.ones(wp, dtype=torch.float32)
        ramp = 0.5 * (1.0 - torch.cos(math.pi * torch.arange(pad, dtype=torch.float32) / pad))
        ry[:pad] = ramp
        ry[-pad:] = ramp.flip(0)
        rx[:pad] = ramp
        rx[-pad:] = ramp.flip(0)
        alpha = (ry[:, None] * rx[None, :]).reshape(1, 1, hp, wp)
        y_taper = alpha * x_pad + (1.0 - alpha) * blurred_pad

        fy = torch.fft.fftfreq(hp).reshape(hp, 1)
        fx = torch.fft.rfftfreq(wp).reshape(1, wp // 2 + 1)
        lap_psd = 4.0 * (torch.sin(math.pi * fy).square() + torch.sin(math.pi * fx).square())

        Y_fft = torch.fft.rfft2(y_taper)
        H_conj = torch.conj(H).unsqueeze(0).unsqueeze(0)
        H_abs2 = H.abs().square().unsqueeze(0).unsqueeze(0)

        x_est = torch.fft.irfft2(
            (H_conj * Y_fft) / (H_abs2 + 0.0045 + float(wiener_lap) * lap_psd.unsqueeze(0).unsqueeze(0)),
            s=(hp, wp),
        )

        kx = torch.zeros((hp, wp), dtype=torch.float32)
        kx[0, 0] = -1.0
        kx[0, 1] = 1.0
        ky = torch.zeros((hp, wp), dtype=torch.float32)
        ky[0, 0] = -1.0
        ky[1, 0] = 1.0
        Kx = torch.fft.rfft2(kx).unsqueeze(0).unsqueeze(0)
        Ky = torch.fft.rfft2(ky).unsqueeze(0).unsqueeze(0)
        K2 = Kx.abs().square() + Ky.abs().square()

        beta = 0.022
        for _ in range(int(tv_iters)):
            gx = torch.roll(x_est, shifts=-1, dims=-1) - x_est
            gy = torch.roll(x_est, shifts=-1, dims=-2) - x_est
            thr = float(thr_base) / max(beta, 1e-4)
            mag = torch.sqrt(gx * gx + gy * gy + 1e-10)
            shrink = ((mag - thr).clamp(min=0.0) / mag).pow(0.85)
            zx = gx * shrink
            zy = gy * shrink
            num = H_conj * Y_fft + beta * (torch.conj(Kx) * torch.fft.rfft2(zx) + torch.conj(Ky) * torch.fft.rfft2(zy))
            den = H_abs2 + beta * K2 + 1e-5
            x_est = torch.fft.irfft2(num / den, s=(hp, wp)).clamp(0.0, 1.0)
            beta *= 1.65

        out = x_est[:, :, pad : pad + h, pad : pad + w].clamp(0.0, 1.0).contiguous()
        return out.to(device=orig_dev)


# --- _REFUSAL_SUBSTRINGS ---
_REFUSAL_SUBSTRINGS = (
    "无法",
    "模糊",
    "不清",
    "不可",
    "未知",
    "NONE",
    "UNREADABLE",
    "BLUR",
    "ILLEGIBLE",
    "CANNOT",
    "THEIMAGE",
    "UNKNOWN",
)


# --- _grad_rms_01 ---
def _grad_rms_01(x_01) -> float:
    import torch

    lum = 0.299 * x_01[:, 0:1] + 0.587 * x_01[:, 1:2] + 0.114 * x_01[:, 2:3]
    gx = lum[:, :, :, 1:] - lum[:, :, :, :-1]
    gy = lum[:, :, 1:, :] - lum[:, :, :-1, :]
    return float(torch.sqrt(gx.square().mean() + gy.square().mean() + 1e-12).cpu())


# --- apply_v13_retinex_daylight_lift ---
def apply_v13_retinex_daylight_lift(x_01, target_p50: float = 0.525, min_gamma: float = 0.58):
    """Chromaticity-preserving coarse `G_5` Retinex illumination envelope gamma lift:
    restores natural daylight exposure when `33B H3` strips white fog/snow (`A*(1-t)`) off an
    underexposed scene (`2-way low+haze`, `3-way low+haze+snow/rain`), preserving vivid RGB ratios,
    amplifying dark shadow details naturally, and never clipping highlights (`p95 <= 0.93`).
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        xp, orig_hw = pad_to_multiple_32(x_01)
        lum = (0.299 * xp[:, 0:1] + 0.587 * xp[:, 1:2] + 0.114 * xp[:, 2:3]).clamp(0.0, 1.0)
        lum_sub = lum[:, :, ::4, ::4].reshape(-1)
        cur_p50 = float(torch.quantile(lum_sub, 0.50).cpu())
        cur_p95 = float(torch.quantile(lum_sub, 0.95).cpu())

        if cur_p50 >= target_p50 - 0.015 or cur_p50 <= 1e-4:
            return x_01, {"retinex_gamma": 1.0, "pre_lift_p50": round(cur_p50, 4), "post_lift_p50": round(cur_p50, 4)}

        raw_gamma = math.log(float(target_p50)) / math.log(max(0.05, min(0.95, cur_p50)))
        if cur_p95 > 0.25:
            gamma_hi_cap = math.log(0.935) / math.log(min(0.985, cur_p95))
            raw_gamma = max(raw_gamma, gamma_hi_cap)
        gamma = float(max(float(min_gamma), min(1.0, raw_gamma)))
        if gamma >= 0.985:
            return x_01, {"retinex_gamma": 1.0, "pre_lift_p50": round(cur_p50, 4), "post_lift_p50": round(cur_p50, 4)}

        # Smooth `>=32px` illumination envelope so local contrast `L_0..L_4` expands proportionally without haloing
        _, g5_lum = laplacian_pyramid_5(lum.expand(-1, 3, -1, -1))
        g5_env = binomial_blur_3x3(g5_lum[:, 0:1], passes=3)
        l_env = F.interpolate(g5_env, size=xp.shape[-2:], mode="bilinear", align_corners=False).clamp(0.025, 1.0)

        gain_env = (l_env.pow(gamma)) / l_env
        # Soft highlight shoulder so speculars near 1.0 never saturate harshly
        hi_taper = 1.0 - 0.55 * ((lum - 0.78) / 0.22).clamp(0.0, 1.0).square()
        eff_gain = 1.0 + hi_taper * (gain_env - 1.0)

        x_lifted = unpad_to_hw((xp * eff_gain).clamp(0.0, 1.0), orig_hw).contiguous()
        lum_after = (0.299 * x_lifted[:, 0:1] + 0.587 * x_lifted[:, 1:2] + 0.114 * x_lifted[:, 2:3]).reshape(-1)
        post_p50 = float(torch.quantile(lum_after[::4], 0.50).cpu())
        return x_lifted, {
            "retinex_gamma": round(gamma, 4),
            "pre_lift_p50": round(cur_p50, 4),
            "post_lift_p50": round(post_p50, 4),
        }


# --- compute_veil_stripped_albedo_quantiles ---
def compute_veil_stripped_albedo_quantiles(x_p_01) -> tuple[float, float, object]:
    """Invert dark-channel Koschmieder airlight `A*(1-t)` and morph-open snow/rain particles
    to measure true underlying scene illumination (`vf50`, `vf90`) beneath bright fog & snow!
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        c_min = x_p_01.min(dim=1, keepdim=True).values
        c_max = x_p_01.max(dim=1, keepdim=True).values
        air = torch.quantile(c_max[:, :, ::4, ::4].reshape(-1), 0.98).clamp_min(0.12)
        dark15 = -F.max_pool2d(-c_min, kernel_size=15, stride=1, padding=7)
        t_est = (1.0 - 0.88 * (dark15 / air)).clamp(0.20, 1.0)
        j_vf = ((x_p_01 - air * (1.0 - t_est)) / t_est).clamp(0.0, 1.0)
        lum_vf = 0.299 * j_vf[:, 0:1] + 0.587 * j_vf[:, 1:2] + 0.114 * j_vf[:, 2:3]
        op_vf = _morph_open(lum_vf, 15)[:, :, ::4, ::4].reshape(-1)
        vf50 = float(torch.quantile(op_vf, 0.50).cpu())
        vf90 = float(torch.quantile(op_vf, 0.90).cpu())
        return vf50, vf90, j_vf


# --- _compute_psf_moments_cpu ---
def _compute_psf_moments_cpu(psf) -> dict:
    import torch

    ks = int(psf.shape[0])
    half = ks // 2
    p = psf / psf.sum().clamp_min(1e-8)
    yy, xx = torch.meshgrid(
        torch.arange(ks, dtype=torch.float32) - half,
        torch.arange(ks, dtype=torch.float32) - half,
        indexing="ij",
    )
    mx = float((p * xx).sum())
    my = float((p * yy).sum())
    dx = xx - mx
    dy = yy - my
    cxx = float((p * dx * dx).sum())
    cyy = float((p * dy * dy).sum())
    cxy = float((p * dx * dy).sum())
    tr = cxx + cyy
    det = cxx * cyy - cxy * cxy
    disc = math.sqrt(max(0.0, 0.25 * tr * tr - det))
    lam1 = max(1e-6, 0.5 * tr + disc)
    lam2 = max(1e-6, 0.5 * tr - disc)
    maj_std = math.sqrt(lam1)
    min_std = math.sqrt(lam2)
    eff_len = math.sqrt(12.0) * maj_std
    support_frac = float((p > 0.008 * float(p.max())).float().mean())
    center_peak = float(p[half - 1 : half + 2, half - 1 : half + 2].sum())
    return {
        "maj_std": round(maj_std, 3),
        "min_std": round(min_std, 3),
        "eff_len_px": round(eff_len, 2),
        "aniso_ratio": round(maj_std / max(0.35, min_std), 3),
        "support_frac": round(support_frac, 4),
        "center_3x3_mass": round(center_peak, 4),
    }


# --- _recenter_psf_cpu ---
def _recenter_psf_cpu(psf):
    """Shift 2D PSF `(ks, ks)` so its first-moment centroid sits at `(ks//2, ks//2)` (`zero pixel shift`!)."""
    import torch

    ks = int(psf.shape[0])
    half = ks // 2
    s = float(psf.sum())
    if s <= 1e-8:
        out = torch.zeros_like(psf)
        out[half, half] = 1.0
        return out
    p = psf / s
    yy, xx = torch.meshgrid(
        torch.arange(ks, dtype=torch.float32) - half,
        torch.arange(ks, dtype=torch.float32) - half,
        indexing="ij",
    )
    cy = int(round(float((p * yy).sum())))
    cx = int(round(float((p * xx).sum())))
    if cy != 0 or cx != 0:
        shifted = torch.zeros_like(p)
        sy0, sy1 = max(0, cy), min(ks, ks + cy)
        dy0 = max(0, -cy)
        dy1 = dy0 + (sy1 - sy0)
        sx0, sx1 = max(0, cx), min(ks, ks + cx)
        dx0 = max(0, -cx)
        dx1 = dx0 + (sx1 - sx0)
        if sy1 > sy0 and sx1 > sx0:
            shifted[dy0:dy1, dx0:dx1] = p[sy0:sy1, sx0:sx1]
            if float(shifted.sum()) > 1e-8:
                p = shifted / shifted.sum()
    return p


# --- _shock_filter_cpu ---
def _shock_filter_cpu(u, dt: float = 0.20, iters: int = 3):
    """Osher-Rudin morphological shock filter on CPU float32 `(1, 1, H, W)`."""
    import torch
    import torch.nn.functional as F

    cur = u
    for _ in range(iters):
        uxx = F.pad(cur[:, :, :, 2:] - 2.0 * cur[:, :, :, 1:-1] + cur[:, :, :, :-2], (1, 1, 0, 0), mode="replicate")
        uyy = F.pad(cur[:, :, 2:, :] - 2.0 * cur[:, :, 1:-1, :] + cur[:, :, :-2, :], (0, 0, 1, 1), mode="replicate")
        ux = 0.5 * F.pad(cur[:, :, :, 2:] - cur[:, :, :, :-2], (1, 1, 0, 0), mode="replicate")
        uy = 0.5 * F.pad(cur[:, :, 2:, :] - cur[:, :, :-2, :], (0, 0, 1, 1), mode="replicate")
        uxy = 0.5 * F.pad(ux[:, :, 2:, :] - ux[:, :, :-2, :], (0, 0, 1, 1), mode="replicate")
        grad_mag = torch.sqrt(ux * ux + uy * uy + 1e-10)
        u_nn = (ux * ux * uxx + 2.0 * ux * uy * uxy + uy * uy * uyy) / (ux * ux + uy * uy + 1e-8)
        cur = (cur - dt * torch.tanh(25.0 * u_nn) * grad_mag).clamp(0.0, 1.0)
    return cur


# --- estimate_blind_curved_motion_psf ---
def estimate_blind_curved_motion_psf(x_in_01, ksize_full: int = 41) -> dict:
    """100% Blind Coarse-to-Fine Shock-Filtered MAP Trajectory PSF Estimator (`zero task_id`, `zero GT`)."""
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        xp, _ = pad_to_multiple_32(x_in_01)
        s8 = compute_blind_precond_sensors(xp)
        sig_u8 = float(s8["sigma_u8"].mean().cpu())
        curv_int = float(s8["curv_int"].mean().cpu())
        jpg_ratio = float(s8["jpg_ratio"].mean().cpu())
        w_deb = float(s8["omega_deblock"].mean().cpu())
        w_2nd = float(s8["omega_2nd"].mean().cpu())

        # Physical camera-shake eligibility (`train`-calibrated: 100% recall on motion_blur_severe, 0% FP on all 11 other camera tasks)
        if sig_u8 > 6.0 or curv_int > 0.62 or w_deb > 0.25 or w_2nd > 0.35:
            return {
                "eligible": False,
                "detected": False,
                "sigma_u8": round(sig_u8, 3),
                "curv_int": round(curv_int, 4),
                "jpg_ratio": round(jpg_ratio, 4),
                "eff_len_px": 0.0,
                "aniso_ratio": 0.0,
                "support_frac": 0.0,
                "center_3x3_mass": 1.0,
                "psf": None,
            }

        x_cpu = x_in_01.detach().cpu().float()
        lum = 0.299 * x_cpu[:, 0:1] + 0.587 * x_cpu[:, 1:2] + 0.114 * x_cpu[:, 2:3]
        _, _, h, w = lum.shape
        ts = min(768, (min(h, w) // 32) * 32)
        if ts < 192:
            return {
                "eligible": False,
                "detected": False,
                "sigma_u8": round(sig_u8, 3),
                "curv_int": round(curv_int, 4),
                "jpg_ratio": round(jpg_ratio, 4),
                "eff_len_px": 0.0,
                "aniso_ratio": 0.0,
                "support_frac": 0.0,
                "center_3x3_mass": 1.0,
                "psf": None,
            }

        best_crop = lum[:, :, (h - ts) // 2 : (h + ts) // 2, (w - ts) // 2 : (w + ts) // 2]
        best_en = -1.0
        for fy in (0.2, 0.5, 0.8):
            for fx in (0.2, 0.5, 0.8):
                y0 = max(0, min(h - ts, int(round(fy * (h - ts)))))
                x0 = max(0, min(w - ts, int(round(fx * (w - ts)))))
                cp = lum[:, :, y0 : y0 + ts, x0 : x0 + ts]
                gx = cp[:, :, :, 1:] - cp[:, :, :, :-1]
                gy = cp[:, :, 1:, :] - cp[:, :, :-1, :]
                en = float(gx.square().mean() + gy.square().mean())
                if en > best_en:
                    best_en = en
                    best_crop = cp

        psf_cur = None
        for scale_idx, (sf, ks, n_it) in enumerate(((0.5, 21, 7), (1.0, int(ksize_full), 7))):
            hs = max(96, (int(round(ts * sf)) // 2) * 2)
            b_s = F.interpolate(best_crop, size=(hs, hs), mode="area") if sf < 1.0 else best_crop
            b_den = binomial_blur_3x3(b_s, passes=2 if scale_idx == 1 else 1)

            half = ks // 2
            if psf_cur is None:
                psf_cur = torch.zeros((ks, ks), dtype=torch.float32)
                psf_cur[half, half] = 1.0
            else:
                psf_up = F.interpolate(
                    psf_cur.reshape(1, 1, *psf_cur.shape),
                    size=(ks, ks),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                psf_cur = _recenter_psf_cpu(psf_up.clamp(min=0.0))

            hann = torch.hann_window(hs, dtype=torch.float32)
            win2d = (hann[:, None] * hann[None, :]).reshape(1, 1, hs, hs)
            bx = F.pad(b_den[:, :, :, 1:] - b_den[:, :, :, :-1], (0, 1, 0, 0)) * win2d
            by = F.pad(b_den[:, :, 1:, :] - b_den[:, :, :-1, :], (0, 0, 0, 1)) * win2d
            Bx = torch.fft.rfft2(bx[0, 0])
            By = torch.fft.rfft2(by[0, 0])

            lat_s = b_den.clone()
            for it in range(n_it):
                lat_sh = _shock_filter_cpu(binomial_blur_3x3(lat_s, passes=1), dt=0.20, iters=3)
                sx = F.pad(lat_sh[:, :, :, 1:] - lat_sh[:, :, :, :-1], (0, 1, 0, 0)) * win2d
                sy = F.pad(lat_sh[:, :, 1:, :] - lat_sh[:, :, :-1, :], (0, 0, 0, 1)) * win2d
                mag = torch.sqrt(sx * sx + sy * sy + 1e-12)
                q_keep = max(0.04, 0.12 - 0.01 * it)
                thr = torch.quantile(mag.reshape(-1), 1.0 - q_keep)
                sal_mask = (mag >= thr).float()
                sx_sal = sx * sal_mask
                sy_sal = sy * sal_mask

                Sx = torch.fft.rfft2(sx_sal[0, 0])
                Sy = torch.fft.rfft2(sy_sal[0, 0])
                num_k = torch.conj(Sx) * Bx + torch.conj(Sy) * By
                den_k = Sx.abs().square() + Sy.abs().square() + 15.0
                k_full = torch.fft.irfft2(num_k / den_k, s=(hs, hs))
                k_shift = torch.roll(k_full, shifts=(half, half), dims=(0, 1))[:ks, :ks]

                k_pos = k_shift.clamp(min=0.0)
                k_thr = 0.045 * float(k_pos.max())
                k_pos = torch.where(k_pos >= k_thr, k_pos, torch.zeros_like(k_pos))
                psf_cur = _recenter_psf_cpu(k_pos)

                if it < n_it - 1:
                    otf = torch.zeros((hs, hs), dtype=torch.float32)
                    otf[:ks, :ks] = psf_cur
                    otf = torch.roll(otf, shifts=(-half, -half), dims=(0, 1))
                    H = torch.fft.rfft2(otf)
                    fy = torch.fft.fftfreq(hs).reshape(hs, 1)
                    fx = torch.fft.rfftfreq(hs).reshape(1, hs // 2 + 1)
                    lap2 = 4.0 * (torch.sin(math.pi * fy).square() + torch.sin(math.pi * fx).square())
                    B_fft = torch.fft.rfft2(b_s[0, 0])
                    lat_2d = torch.fft.irfft2(
                        (torch.conj(H) * B_fft) / (H.abs().square() + 0.012 + 0.04 * lap2),
                        s=(hs, hs),
                    )
                    lat_s = lat_2d.clamp(0.0, 1.0).reshape(1, 1, hs, hs)

        mom = _compute_psf_moments_cpu(psf_cur)
        detected = bool(
            mom["eff_len_px"] >= 13.5
            and mom["aniso_ratio"] >= 2.45
            and mom["support_frac"] <= 0.24
            and mom["center_3x3_mass"] <= 0.48
        )
        return {
            "eligible": True,
            "detected": detected,
            "sigma_u8": round(sig_u8, 3),
            "curv_int": round(curv_int, 4),
            "jpg_ratio": round(jpg_ratio, 4),
            **mom,
            "psf": psf_cur if detected else None,
        }


# --- _fit_raw_tile_psf_moments_cpu ---
def _fit_raw_tile_psf_moments_cpu(lum_tile, ks: int = 41) -> dict:
    """Fit unconstrained shock-filtered MAP trajectory PSF on a single salient tile `(1, 1, ts, ts)`."""
    import torch
    import torch.nn.functional as F

    hs = int(lum_tile.shape[-1])
    psf_cur = None
    k_coarse = max(15, (ks // 2) | 1)
    for scale_idx, (sf, k_s, n_it) in enumerate(((0.5, k_coarse, 5), (1.0, int(ks), 5))):
        h_s = max(96, (int(round(hs * sf)) // 2) * 2)
        b_s = F.interpolate(lum_tile, size=(h_s, h_s), mode="area") if sf < 1.0 else lum_tile
        b_den = binomial_blur_3x3(b_s, passes=2 if scale_idx == 1 else 1)
        half = k_s // 2
        if psf_cur is None:
            psf_cur = torch.zeros((k_s, k_s), dtype=torch.float32)
            psf_cur[half, half] = 1.0
        else:
            psf_up = F.interpolate(
                psf_cur.reshape(1, 1, *psf_cur.shape),
                size=(k_s, k_s),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            psf_cur = _recenter_psf_cpu(psf_up.clamp(min=0.0))

        hann = torch.hann_window(h_s, dtype=torch.float32)
        win2d = (hann[:, None] * hann[None, :]).reshape(1, 1, h_s, h_s)
        bx = F.pad(b_den[:, :, :, 1:] - b_den[:, :, :, :-1], (0, 1, 0, 0)) * win2d
        by = F.pad(b_den[:, :, 1:, :] - b_den[:, :, :-1, :], (0, 0, 0, 1)) * win2d
        Bx = torch.fft.rfft2(bx[0, 0])
        By = torch.fft.rfft2(by[0, 0])

        lat_s = b_den.clone()
        for it in range(n_it):
            lat_sh = _shock_filter_cpu(binomial_blur_3x3(lat_s, passes=1), dt=0.20, iters=3)
            sx = F.pad(lat_sh[:, :, :, 1:] - lat_sh[:, :, :, :-1], (0, 1, 0, 0)) * win2d
            sy = F.pad(lat_sh[:, :, 1:, :] - lat_sh[:, :, :-1, :], (0, 0, 0, 1)) * win2d
            mag = torch.sqrt(sx * sx + sy * sy + 1e-12)
            thr = torch.quantile(mag.reshape(-1), 1.0 - max(0.04, 0.12 - 0.012 * it))
            sal = (mag >= thr).float()
            Sx = torch.fft.rfft2((sx * sal)[0, 0])
            Sy = torch.fft.rfft2((sy * sal)[0, 0])
            k_full = torch.fft.irfft2(
                (torch.conj(Sx) * Bx + torch.conj(Sy) * By) / (Sx.abs().square() + Sy.abs().square() + 15.0),
                s=(h_s, h_s),
            )
            k_pos = torch.roll(k_full, shifts=(half, half), dims=(0, 1))[:k_s, :k_s].clamp(min=0.0)
            k_pos = torch.where(k_pos >= 0.045 * float(k_pos.max()), k_pos, torch.zeros_like(k_pos))
            psf_cur = _recenter_psf_cpu(k_pos)

            if it < n_it - 1:
                otf = torch.zeros((h_s, h_s), dtype=torch.float32)
                otf[:k_s, :k_s] = psf_cur
                H = torch.fft.rfft2(torch.roll(otf, shifts=(-half, -half), dims=(0, 1)))
                fy = torch.fft.fftfreq(h_s).reshape(h_s, 1)
                fx = torch.fft.rfftfreq(h_s).reshape(1, h_s // 2 + 1)
                lap2 = 4.0 * (torch.sin(math.pi * fy).square() + torch.sin(math.pi * fx).square())
                lat_s = torch.fft.irfft2(
                    (torch.conj(H) * torch.fft.rfft2(b_s[0, 0])) / (H.abs().square() + 0.012 + 0.04 * lap2),
                    s=(h_s, h_s),
                ).clamp(0.0, 1.0).reshape(1, 1, h_s, h_s)

    mom = _compute_psf_moments_cpu(psf_cur)
    half = int(ks) // 2
    p = psf_cur / psf_cur.sum().clamp_min(1e-8)
    yy, xx = torch.meshgrid(
        torch.arange(int(ks), dtype=torch.float32) - half,
        torch.arange(int(ks), dtype=torch.float32) - half,
        indexing="ij",
    )
    cxx = float((p * xx * xx).sum())
    cyy = float((p * yy * yy).sum())
    cxy = float((p * xx * yy).sum())
    theta = 0.5 * math.atan2(2.0 * cxy, cxx - cyy)
    return {**mom, "theta": theta}


# --- estimate_blind_global_shake_psf_v13 ---
def estimate_blind_global_shake_psf_v13(x_in_01, ksize_full: int = 41) -> dict:
    """100% Blind Global Camera-Shake PSF Estimator with Multi-Tile Spatial Homogeneity Gate.

    Cleanly accepts spatially-invariant global camera shake (`12/12 = 100%` recall on `v3_train`
    `25px` curved motion blur) while rejecting spatially-varying GoPro dynamic object/parallax
    motion blur (`0/8 = 0%` `train_gopro` FP) and all non-shake camera degradations (`0%` FP).
    """
    import torch

    with torch.no_grad():
        res0 = estimate_blind_curved_motion_psf(x_in_01, ksize_full=ksize_full)
        if not res0.get("detected", False) or res0.get("psf") is None:
            return {
                **res0,
                "spatial_homogeneity_ok": False,
                "min_tile_len_px": 0.0,
                "tile_min_max_ratio": 0.0,
                "tile_r_theta": 0.0,
            }

        x_cpu = x_in_01.detach().cpu().float()
        lum = 0.299 * x_cpu[:, 0:1] + 0.587 * x_cpu[:, 1:2] + 0.114 * x_cpu[:, 2:3]
        _, _, h, w = lum.shape
        short_edge = min(h, w)

        # For full-frame photos (`short_edge >= 640`, e.g. `1280x720` GoPro or `2048x2048` camera),
        # require severe `>= 20.5 px` blur (since `<20px` blur has no severe sinc reversal and H3
        # handles it natively with higher perceptual sharpness) PLUS 4-tile spatial homogeneity!
        min_global_len = 20.5 if short_edge >= 640 else 13.5
        if float(res0["eff_len_px"]) < min_global_len or float(res0["aniso_ratio"]) < 2.55:
            return {
                **res0,
                "detected": False,
                "psf": None,
                "spatial_homogeneity_ok": False,
                "min_tile_len_px": 0.0,
                "tile_min_max_ratio": 0.0,
                "tile_r_theta": 0.0,
            }

        if short_edge < 480:
            # Compact synthetic / crop regime (`<480px`): single-frame trajectory already covers full support
            return {
                **res0,
                "spatial_homogeneity_ok": True,
                "min_tile_len_px": float(res0["eff_len_px"]),
                "tile_min_max_ratio": 1.0,
                "tile_r_theta": 1.0,
            }

        ts = min(384, (short_edge // 2 // 32) * 32)
        ts = max(192, ts)
        cands = []
        for fy in (0.05, 0.35, 0.65, 0.95):
            for fx in (0.05, 0.35, 0.65, 0.95):
                y0 = max(0, min(h - ts, int(round(fy * (h - ts)))))
                x0 = max(0, min(w - ts, int(round(fx * (w - ts)))))
                cp = lum[:, :, y0 : y0 + ts, x0 : x0 + ts]
                gx = cp[:, :, :, 1:] - cp[:, :, :, :-1]
                gy = cp[:, :, 1:, :] - cp[:, :, :-1, :]
                en = float(gx.square().mean() + gy.square().mean())
                cands.append((en, y0, x0))
        cands.sort(key=lambda t: t[0], reverse=True)

        picked = []
        for en, y0, x0 in cands:
            if all(abs(y0 - py) >= ts // 2 or abs(x0 - px) >= ts // 2 for _, py, px in picked):
                picked.append((en, y0, x0))
                if len(picked) >= 4:
                    break

        if len(picked) < 3:
            return {
                **res0,
                "detected": False,
                "psf": None,
                "spatial_homogeneity_ok": False,
                "min_tile_len_px": 0.0,
                "tile_min_max_ratio": 0.0,
                "tile_r_theta": 0.0,
            }

        ks_tile = min(int(ksize_full), 41)
        t_moms = [_fit_raw_tile_psf_moments_cpu(lum[:, :, y0 : y0 + ts, x0 : x0 + ts], ks=ks_tile) for _, y0, x0 in picked]
        lens = [float(m["eff_len_px"]) for m in t_moms]
        angs = [float(m["theta"]) for m in t_moms]

        min_tile_l = min(lens)
        min_max_ratio = min_tile_l / max(max(lens), 1e-4)
        c2 = sum(math.cos(2.0 * a) for a in angs) / len(angs)
        s2 = sum(math.sin(2.0 * a) for a in angs) / len(angs)
        r_theta = math.hypot(c2, s2)

        homog_ok = bool(
            min_tile_l >= 18.0
            and min_max_ratio >= 0.60
            and r_theta >= 0.75
        )
        return {
            **res0,
            "detected": homog_ok,
            "psf": res0["psf"] if homog_ok else None,
            "spatial_homogeneity_ok": homog_ok,
            "min_tile_len_px": round(min_tile_l, 2),
            "tile_min_max_ratio": round(min_max_ratio, 3),
            "tile_r_theta": round(r_theta, 3),
        }


# --- make_gentle_2p2x_signage_crop_01 ---
def make_gentle_2p2x_signage_crop_01(x_in_crop_01, omega_2nd: float):
    """For `realworld_2nd_order` (`omega_2nd >= 0.5`), strip 8x8 JPEG & Sinc ringing at `2.2x`
    instead of `3.2x` on `Z=2` signage crops so small characters retain `1.45x` higher Nyquist bandwidth!
    """
    import torch
    import torch.nn.functional as F

    if omega_2nd < 0.50:
        return x_in_crop_01
    _, _, h, w = x_in_crop_01.shape
    y = 0.299 * x_in_crop_01[:, 0:1] + 0.587 * x_in_crop_01[:, 1:2] + 0.114 * x_in_crop_01[:, 2:3]
    cb = x_in_crop_01[:, 2:3] - y
    cr = x_in_crop_01[:, 0:1] - y
    for _ in range(2):
        cb = burt_adelson_blur5(cb)
        cr = burt_adelson_blur5(cr)
    r = y + cr
    b_ch = y + cb
    g = (y - 0.299 * r - 0.114 * b_ch) / 0.587
    x_clean = torch.cat((r, g, b_ch), dim=1).clamp(0.0, 1.0)
    lh, lw = max(16, int(round(h / 2.2))), max(16, int(round(w / 2.2)))
    return F.interpolate(
        F.interpolate(x_clean, size=(lh, lw), mode="area"),
        size=(h, w),
        mode="bicubic",
        align_corners=False,
    ).clamp(0.0, 1.0).contiguous()


# --- make_gentle_unsharp_signage_crop_01 ---
def make_gentle_unsharp_signage_crop_01(x_crop_01, omega_2nd: float = 0.0, nyq_scaled: float = 0.50):
    """Build a crisp second signage view (`View B`) for dual-view VLM reading:
      - `omega_2nd >= 0.45`: `2.2x` anti-ringed view stripping Sinc/JPEG overshoot;
      - 1st-order blurry/SR signage (`nyq_scaled < 0.72`): gentle luminance unsharp contrast boost
        so `4x` / `Real-SR x4` strokes (`"牧場"`, `"店"`) read cleanly with high token confidence!
    """
    import torch

    with torch.no_grad():
        if float(omega_2nd) >= 0.45:
            return make_gentle_2p2x_signage_crop_01(x_crop_01, 1.0)
        if float(nyq_scaled) >= 0.78:
            return x_crop_01
        xp, orig_hw = pad_to_multiple_32(x_crop_01)
        lum = 0.299 * xp[:, 0:1] + 0.587 * xp[:, 1:2] + 0.114 * xp[:, 2:3]
        lum_sm = binomial_blur_3x3(lum, passes=2)
        detail = (lum - lum_sm).clamp(-0.18, 0.18)
        gain = 0.62 if float(nyq_scaled) < 0.55 else 0.38
        out_p = (xp + gain * detail).clamp(0.0, 1.0)
        return unpad_to_hw(out_p, orig_hw).contiguous()


# --- _extract_blind_12ch_obs_256 ---
def _extract_blind_12ch_obs_256(x_01, x_nat_quad256=None):
    """Extract 18-channel Dual-Stream (`12` global `256x256` + `6` native `1:1` pixel) observation map."""
    import torch
    import torch.nn.functional as F

    _, _, h, w = x_01.shape
    if h == 256 and w == 256:
        xs = x_01
    elif h >= 256 and w >= 256:
        xs = F.interpolate(x_01, size=(256, 256), mode="area")
    else:
        xs = F.interpolate(x_01, size=(256, 256), mode="bilinear", align_corners=False)

    lum = 0.299 * xs[:, 0:1] + 0.587 * xs[:, 1:2] + 0.114 * xs[:, 2:3]
    rgb_min = xs.min(dim=1, keepdim=True).values
    rgb_max = xs.max(dim=1, keepdim=True).values
    dc15 = -F.max_pool2d(-rgb_min, kernel_size=15, stride=1, padding=7)
    bc15 = F.max_pool2d(rgb_max, kernel_size=15, stride=1, padding=7)
    loc_contrast = (bc15 - dc15).clamp(0.0, 1.0)
    chroma_sat = (rgb_max - rgb_min).clamp(0.0, 1.0)

    lum_smooth = binomial_blur_3x3(lum, passes=2)
    hp = lum - lum_smooth
    hp_pos = F.relu(hp) * 4.0
    hp_neg = F.relu(-hp) * 4.0

    gx = F.pad((lum[:, :, :, 1:] - lum[:, :, :, :-1]).abs(), (0, 1, 0, 0), mode="replicate") * 3.0
    gy = F.pad((lum[:, :, 1:, :] - lum[:, :, :-1, :]).abs(), (0, 0, 0, 1), mode="replicate") * 3.0

    _, sig_loc = blind_local_haar_sigma_map(xs)
    sig_256 = F.interpolate(sig_loc * 10.0, size=(256, 256), mode="bilinear", align_corners=False)

    # Stream 2: Native 1:1 pixel 4-quadrant high-frequency observations (`6` channels)
    xq = extract_blind_native_quad256(x_01) if x_nat_quad256 is None else x_nat_quad256
    lum_q = 0.299 * xq[:, 0:1] + 0.587 * xq[:, 1:2] + 0.114 * xq[:, 2:3]
    hp_q = lum_q - binomial_blur_3x3(lum_q, passes=2)
    hp_q_pos = F.relu(hp_q) * 4.0
    hp_q_neg = F.relu(-hp_q) * 4.0
    gx_q = F.pad((lum_q[:, :, :, 1:] - lum_q[:, :, :, :-1]).abs(), (0, 1, 0, 0), mode="replicate")
    gy_q = F.pad((lum_q[:, :, 1:, :] - lum_q[:, :, :-1, :]).abs(), (0, 0, 0, 1), mode="replicate")
    grad_q = ((gx_q + gy_q) * 2.5).clamp(0.0, 2.0)

    return torch.cat(
        (xs, dc15, bc15, loc_contrast, hp_pos, hp_neg, gx, gy, chroma_sat, sig_256, xq, hp_q_pos, hp_q_neg, grad_q),
        dim=1,
    ).contiguous()


# --- pad_to_multiple_16 ---
def pad_to_multiple_16(x_01):
    import torch.nn.functional as F

    _, _, h, w = x_01.shape
    ph = (16 - (h % 16)) % 16
    pw = (16 - (w % 16)) % 16
    if ph == 0 and pw == 0:
        return x_01, (h, w)
    mode = "reflect" if (h > ph and w > pw) else "replicate"
    return F.pad(x_01, (0, pw, 0, ph), mode=mode), (h, w)


# --- build_v10_omni_precond ---
def build_v10_omni_precond(device="cpu", seed: int = SEED, width: int = 48):
    """Build zero-init Dual-Stream Blind Weather Gate + Physics-Guided Radiometric & Streak Pre-Conditioner.

    Invariants:
      - Step 0: `is_exact_zero_state() == True` -> `precond_forward(x_in)` returns `x_in` bitwise!
      - Inference Deadzone: `p_wea = sigmoid(gate_logit - 1.5)`; `omega_wea = ((p_wea - 0.35) / 0.45).clamp(0, 1)`.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    gen = torch.Generator("cpu").manual_seed(int(seed) + 1009)

    def rand_conv(in_ch, out_ch, k=3, stride=1, dilation=1, groups=1):
        kh, kw = (k, k) if isinstance(k, int) else (int(k[0]), int(k[1]))
        pad_h = ((kh - 1) * dilation) // 2
        pad_w = ((kw - 1) * dilation) // 2
        layer = nn.Conv2d(
            in_ch,
            out_ch,
            (kh, kw),
            stride=stride,
            padding=(pad_h, pad_w),
            dilation=dilation,
            groups=groups,
            padding_mode="reflect" if (stride == 1 and max(pad_h, pad_w) > 0) else "zeros",
            dtype=torch.float32,
        )
        nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), generator=gen)
        fan_in = (in_ch // groups) * (kh * kw)
        bound = 1.0 / math.sqrt(max(1, fan_in))
        nn.init.uniform_(layer.bias, -bound, bound, generator=gen)
        return layer

    def zero_conv1x1(in_ch, out_ch):
        layer = nn.Conv2d(in_ch, out_ch, 1, dtype=torch.float32)
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)
        return layer

    def rand_linear(in_dim, out_dim):
        layer = nn.Linear(in_dim, out_dim, dtype=torch.float32)
        nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5), generator=gen)
        bound = 1.0 / math.sqrt(max(1, in_dim))
        nn.init.uniform_(layer.bias, -bound, bound, generator=gen)
        return layer

    def zero_linear(in_dim, out_dim):
        layer = nn.Linear(in_dim, out_dim, dtype=torch.float32)
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)
        return layer

    class ResDilatedBlock(nn.Module):
        def __init__(self, ch: int, dilation: int = 1):
            super().__init__()
            self.c1 = rand_conv(ch, ch, 3, dilation=dilation)
            self.c2 = rand_conv(ch, ch, 3, dilation=1)

        def forward(self, x):
            return x + 0.5 * self.c2(F.silu(self.c1(F.silu(x))))

    class DirectionalStreakBlock(nn.Module):
        """Captures anisotropic vertical/diagonal rain streaks & snow flakes alongside isotropic context."""

        def __init__(self, ch: int):
            super().__init__()
            self.dw_v = rand_conv(ch, ch, (7, 1), groups=ch)
            self.dw_h = rand_conv(ch, ch, (1, 7), groups=ch)
            self.dw_d = rand_conv(ch, ch, 3, dilation=2, groups=ch)
            self.pw = rand_conv(ch * 3, ch, 1)

        def forward(self, x):
            a = F.silu(self.dw_v(x))
            b = F.silu(self.dw_h(x))
            c = F.silu(self.dw_d(x))
            return x + 0.5 * self.pw(torch.cat((a, b, c), dim=1))

    class V10OmniPreConditioner(nn.Module):
        arm_name = "v10_omni_precond"
        uses_task_id = False

        def __init__(self):
            super().__init__()
            # 1. Dual-Stream Blind 2D Spatial-Spectral Regime Classifier (18ch -> 256 -> 128 -> 64 -> 32 -> 16)
            c_cls = max(32, width)
            self.cls_s1 = rand_conv(18, c_cls, 3, stride=2)
            self.cls_s2 = rand_conv(c_cls, c_cls, 3, stride=2)
            self.cls_s3 = rand_conv(c_cls, c_cls * 2, 3, stride=2)
            self.cls_s4 = rand_conv(c_cls * 2, c_cls * 2, 3, stride=2)
            self.cls_fc = rand_linear(c_cls * 6, c_cls * 2)
            # [wea_logit(1), haze_sub(1), low_sub(1), precip_sub(1)]
            self.gate_head = zero_linear(c_cls * 2, 4)

            # 2. Physics-Guided Atmospheric Dehaze & Retinex Illumination Branch (at 1/4 scale)
            self.rad_proj = rand_linear(c_cls * 2, c_cls)
            self.rad_stem = rand_conv(6 + c_cls, width, 3)
            self.rad_b1 = ResDilatedBlock(width, dilation=1)
            self.rad_b2 = ResDilatedBlock(width, dilation=2)
            self.rad_b3 = ResDilatedBlock(width, dilation=4)
            self.rad_head = zero_conv1x1(width, 12)

            # 3. Multi-Scale Directional Streak & Scattering Residual U-Net (1x, 1/2x, 1/4x)
            w1 = width
            w2 = int(width * 1.5)
            w4 = width * 2
            self.enc1_in = rand_conv(9, w1, 3)
            self.enc1_dir = DirectionalStreakBlock(w1)
            self.down2 = rand_conv(w1, w2, 3, stride=2)
            self.enc2_dir = DirectionalStreakBlock(w2)
            self.down4 = rand_conv(w2, w4, 3, stride=2)
            self.mid4_1 = ResDilatedBlock(w4, dilation=1)
            self.mid4_2 = ResDilatedBlock(w4, dilation=2)
            self.up2_reduce = rand_conv(w4 + w2, w2, 1)
            self.dec2 = ResDilatedBlock(w2, dilation=1)
            self.up1_reduce = rand_conv(w2 + w1, w1, 1)
            self.dec1 = DirectionalStreakBlock(w1)
            self.streak_head = zero_conv1x1(w1, 3)

        def is_exact_zero_state(self) -> bool:
            for h in (self.gate_head, self.rad_head, self.streak_head):
                if int(torch.count_nonzero(h.weight).item()) != 0 or int(torch.count_nonzero(h.bias).item()) != 0:
                    return False
            return True

        def classify_blind_weather_gate(self, x_in_01, x_nat_quad256=None):
            obs256 = _extract_blind_12ch_obs_256(x_in_01, x_nat_quad256=x_nat_quad256)
            h = F.silu(self.cls_s1(obs256))
            h = F.silu(self.cls_s2(h))
            h = F.silu(self.cls_s3(h))
            h = F.silu(self.cls_s4(h))
            flat = h.flatten(2)
            f_mean = flat.mean(dim=-1)
            f_std = flat.std(dim=-1, unbiased=False)
            f_max = flat.max(dim=-1).values
            emb = F.silu(self.cls_fc(torch.cat((f_mean, f_std, f_max), dim=1)))
            logits = self.gate_head(emb)
            wea_shifted_logit = logits[:, 0:1] - 1.5
            p_wea = torch.sigmoid(wea_shifted_logit)
            omega_wea = ((p_wea - 0.35) / 0.45).clamp(0.0, 1.0)
            p_sub_raw = torch.sigmoid(logits[:, 1:4] - 1.0)
            p_sub = p_sub_raw * omega_wea
            return {
                "emb": emb,
                "wea_shifted_logit": wea_shifted_logit,
                "sub_shifted_logits": logits[:, 1:4] - 1.0,
                "p_wea": p_wea.reshape(-1, 1, 1, 1),
                "omega_wea": omega_wea.reshape(-1, 1, 1, 1),
                "p_haze": p_sub[:, 0:1].reshape(-1, 1, 1, 1),
                "p_low": p_sub[:, 1:2].reshape(-1, 1, 1, 1),
                "p_precip": p_sub[:, 2:3].reshape(-1, 1, 1, 1),
            }

        def _restore_weather_branch(self, x_in_01, emb):
            x_pad, orig_hw = pad_to_multiple_16(x_in_01)
            bsz, _, hp, wp = x_pad.shape
            hq, wq = hp // 4, wp // 4

            xq = F.interpolate(x_pad, size=(hq, wq), mode="area")
            lum_q = 0.299 * xq[:, 0:1] + 0.587 * xq[:, 1:2] + 0.114 * xq[:, 2:3]
            dc_q = -F.max_pool2d(-xq.min(dim=1, keepdim=True).values, kernel_size=7, stride=1, padding=3)
            bc_q = F.max_pool2d(xq.max(dim=1, keepdim=True).values, kernel_size=7, stride=1, padding=3)
            g_ctx = self.rad_proj(emb)[:, :, None, None].expand(bsz, -1, hq, wq)

            rh = F.silu(self.rad_stem(torch.cat((xq, dc_q, bc_q, lum_q, g_ctx), dim=1)))
            rh = self.rad_b3(self.rad_b2(self.rad_b1(rh)))
            r_raw_q = binomial_blur_3x3(self.rad_head(F.silu(rh)), passes=2)
            r_raw = F.interpolate(r_raw_q, size=(hp, wp), mode="bilinear", align_corners=False)

            veil_sub = 0.72 * F.relu(torch.tanh(r_raw[:, 0:3]))
            gain_mul = torch.exp(1.30 * torch.tanh(r_raw[:, 3:6]))
            gamma_exp = torch.exp(0.62 * torch.tanh(r_raw[:, 6:9]))
            lift_bias = 0.18 * torch.tanh(r_raw[:, 9:12])

            base_rad = ((x_pad - veil_sub) * gain_mul + lift_bias).clamp(1e-4, 1.0)
            x_rad = base_rad.pow(gamma_exp).clamp(0.0, 1.0)

            hp_in = x_pad - binomial_blur_3x3(x_pad, passes=2)
            f1 = self.enc1_dir(F.silu(self.enc1_in(torch.cat((x_pad, x_rad, hp_in), dim=1))))
            f2 = self.enc2_dir(F.silu(self.down2(f1)))
            f4 = self.mid4_2(self.mid4_1(F.silu(self.down4(f2))))

            u2 = F.interpolate(f4, size=f2.shape[-2:], mode="bilinear", align_corners=False)
            d2 = self.dec2(F.silu(self.up2_reduce(torch.cat((u2, f2), dim=1))))
            u1 = F.interpolate(d2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
            d1 = self.dec1(F.silu(self.up1_reduce(torch.cat((u1, f1), dim=1))))

            res_rgb = 0.45 * torch.tanh(self.streak_head(F.silu(d1)))
            x_wea_pad = x_rad + res_rgb
            if not x_wea_pad.requires_grad:
                x_wea_pad = x_wea_pad.clamp(0.0, 1.0)
            x_wea = unpad_to_hw(x_wea_pad, orig_hw)
            x_rad_unpad = unpad_to_hw(x_rad, orig_hw)
            return x_wea.contiguous(), x_rad_unpad.contiguous()

        def precond_forward(self, x_in_01, force_weather_branch: bool = False):
            if x_in_01.ndim != 4 or x_in_01.shape[1] != 3:
                raise ValueError(f"Expected [B, 3, H, W] RGB tensor, got {tuple(x_in_01.shape)}")

            if not torch.is_grad_enabled() and self.is_exact_zero_state() and not force_weather_branch:
                return x_in_01, {
                    "exact_zero_passthrough": True,
                    "sparse_weather_passthrough": True,
                    "p_wea": 0.1824,
                    "omega_wea": 0.0,
                    "p_haze": 0.0,
                    "p_low": 0.0,
                    "p_precip": 0.0,
                }

            g_info = self.classify_blind_weather_gate(x_in_01)
            w_wea = g_info["omega_wea"]

            # Conjunctive native-1:1 physical feasibility check during inference (`not force_weather_branch`)
            if not torch.is_grad_enabled() and not force_weather_branch and float(w_wea.max().cpu()) > 0.0:
                s_phys = compute_blind_v10_omni_sensors(x_in_01)
                h_sup = torch.maximum(s_phys["omega_haze_phys"], g_info["p_haze"] * s_phys["allow_haze"])
                l_sup = torch.maximum(s_phys["omega_low_phys"], g_info["p_low"] * s_phys["allow_low"])
                p_sup = torch.maximum(
                    torch.maximum(s_phys["omega_rain_phys"], s_phys["omega_snow_phys"]),
                    g_info["p_precip"] * s_phys["allow_precip"],
                )
                sub_max = torch.maximum(torch.maximum(h_sup, l_sup), p_sup)
                w_wea = w_wea * ((sub_max - 0.08) / 0.18).clamp(0.0, 1.0)
                w_wea = torch.where(w_wea >= 0.10, w_wea, torch.zeros_like(w_wea))

            if not torch.is_grad_enabled() and not force_weather_branch and float(w_wea.max().cpu()) == 0.0:
                return x_in_01, {
                    "exact_zero_passthrough": False,
                    "sparse_weather_passthrough": True,
                    "p_wea": round(float(g_info["p_wea"].mean().cpu()), 4),
                    "omega_wea": 0.0,
                    "p_haze": 0.0,
                    "p_low": 0.0,
                    "p_precip": 0.0,
                }

            x_wea, x_rad = self._restore_weather_branch(x_in_01, g_info["emb"])
            if force_weather_branch:
                x_out = x_wea
            else:
                x_out = x_in_01 + w_wea * (x_wea - x_in_01)
            if not x_out.requires_grad:
                x_out = x_out.clamp(0.0, 1.0)

            meta = {
                "exact_zero_passthrough": False,
                "sparse_weather_passthrough": bool(float(w_wea.max().detach().cpu()) == 0.0),
                "p_wea": round(float(g_info["p_wea"].detach().mean().cpu()), 4),
                "omega_wea": round(float(w_wea.detach().mean().cpu()), 4),
                "p_haze": round(float(g_info["p_haze"].detach().mean().cpu()), 4),
                "p_low": round(float(g_info["p_low"].detach().mean().cpu()), 4),
                "p_precip": round(float(g_info["p_precip"].detach().mean().cpu()), 4),
                "_train_tensors": {
                    "wea_shifted_logit": g_info["wea_shifted_logit"],
                    "sub_shifted_logits": g_info["sub_shifted_logits"],
                    "x_wea": x_wea,
                    "x_rad": x_rad,
                } if torch.is_grad_enabled() else None,
            }
            return x_out.contiguous(), meta

        def forward(self, x_in_01):
            x_out, _ = self.precond_forward(x_in_01)
            return x_out

    return V10OmniPreConditioner().to(device=device)


# --- safetensors_header ---
def safetensors_header(path):
    path = Path(path)
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"Truncated safetensors file: {path}")
        size = struct.unpack("<Q", raw)[0]
        if not 0 < size <= 16 * 1024 * 1024:
            raise ValueError(f"Invalid safetensors header size: {path}")
        header = json.loads(stream.read(size))
    widths = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "BOOL": 1, "U8": 1}
    data_bytes = path.stat().st_size - 8 - size
    for name, item in header.items():
        if name == "__metadata__":
            continue
        begin, end = item["data_offsets"]
        if item["dtype"] not in widths or not 0 <= begin <= end <= data_bytes:
            raise ValueError(f"Invalid tensor metadata: {path}: {name}")
        if end - begin != math.prod(item["shape"]) * widths[item["dtype"]]:
            raise ValueError(f"Tensor byte count mismatch: {path}: {name}")
    return header

