#!/usr/bin/env python3
"""Pure-CPU unit tests verifying all RestoreH3 physical, spectral, geometric, and epistemic invariants."""
from __future__ import annotations

import math
from pathlib import Path
import unittest

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from restoreh3 import (
    PRECOND_FILENAME,
    align_candidate_blind_affine_01,
    apply_stage1_blind_preconditioner,
    apply_stage2_physical_projection,
    build_weather_preconditioner,
    classify_signboard_prior_regime,
    compute_exact_parity_frame_spec,
    edge_preserving_ycbcr_denoise_01,
    frame_pil_exact_parity,
    gate_and_fuse_signboard_patch,
    load_weather_preconditioner,
    merge_dual_side_text_boxes,
    rgb01_to_ycbcr01,
    unframe_pil_exact_parity,
    ycbcr01_to_rgb01,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestRestoreH3CPU(unittest.TestCase):
    def test_exact_32px_lattice_parity_zero_shear(self):
        for w, h in [(481, 321), (1024, 680), (720, 1080), (550, 413), (1280, 720), (512, 512), (2048, 2048)]:
            spec = compute_exact_parity_frame_spec(w, h, min_generation_area=max(512 * 512, w * h))
            self.assertEqual(spec["gen_w"] % 32, 0)
            self.assertEqual(spec["gen_h"] % 32, 0)
            self.assertEqual(spec["ref_w"] % 32, 0)
            self.assertEqual(spec["ref_h"] % 32, 0)
            self.assertAlmostEqual(spec["ref_w"] / spec["ref_h"], spec["gen_w"] / spec["gen_h"], places=6)
            pil_in = Image.fromarray(np.full((h, w, 3), 128, dtype=np.uint8), mode="RGB")
            pil_gen = frame_pil_exact_parity(pil_in, spec)
            self.assertEqual(pil_gen.size, (spec["gen_w"], spec["gen_h"]))
            pil_back = unframe_pil_exact_parity(pil_gen, spec)
            self.assertEqual(pil_back.size, (w, h))

    def test_blind_affine_recovers_zoom_and_shift(self):
        yy, xx = torch.meshgrid(torch.linspace(0, 1, 256), torch.linspace(0, 1, 256), indexing="ij")
        img = (
            0.5
            + 0.25 * torch.sin(2.0 * math.pi * 6.0 * xx) * torch.cos(2.0 * math.pi * 5.0 * yy)
            + 0.15 * torch.sin(2.0 * math.pi * 13.0 * xx + 0.5)
        ).clamp(0.05, 0.95)
        x_ref = img.reshape(1, 1, 256, 256).repeat(1, 3, 1, 1)

        gx = 2.0 * (0.98 * xx + 5.0 / 256.0) - 1.0
        gy = 2.0 * (0.98 * yy + 4.0 / 256.0) - 1.0
        grid = torch.stack((gx, gy), dim=-1).unsqueeze(0)
        x_warped = F.grid_sample(x_ref, grid, mode="bicubic", padding_mode="border", align_corners=True)

        mse_before = float((x_warped[:, :, 16:-16, 16:-16] - x_ref[:, :, 16:-16, 16:-16]).square().mean())
        x_aligned, meta = align_candidate_blind_affine_01(x_ref, x_warped)
        mse_after = float((x_aligned[:, :, 16:-16, 16:-16] - x_ref[:, :, 16:-16, 16:-16]).square().mean())
        self.assertTrue(meta["affine_applied"], f"Expected affine alignment to trigger, got {meta}")
        self.assertLess(mse_after, mse_before * 0.15)

    def test_butterworth_ycbcr_chroma_lock_and_bw_guard(self):
        yy, xx = torch.meshgrid(torch.linspace(0, 1, 192), torch.linspace(0, 1, 192), indexing="ij")
        lum = (0.5 + 0.35 * torch.sign(torch.sin(2.0 * math.pi * 8.0 * xx))).clamp(0.1, 0.9)
        x_in = lum.reshape(1, 1, 192, 192).repeat(1, 3, 1, 1)
        x_cand = x_in.clone()
        x_cand[:, 0, 40:150, 40:150] = (x_cand[:, 0, 40:150, 40:150] + 0.14).clamp(0.0, 1.0)
        x_cand[:, 2, 40:150, 40:150] = (x_cand[:, 2, 40:150, 40:150] - 0.14).clamp(0.0, 1.0)

        x_out, _ = apply_stage2_physical_projection(x_in, x_in, x_cand, omega_wea=0.0, enable_blind_affine=False)
        _, cb_o, cr_o = rgb01_to_ycbcr01(x_out)
        self.assertLess(float(torch.hypot(cb_o, cr_o).max()), 0.012)

        y, cb, cr = rgb01_to_ycbcr01(x_cand)
        rec = ycbcr01_to_rgb01(y, cb, cr)
        self.assertLess(float((rec - x_cand).abs().max()), 1e-4)

    def test_wiener_denoise_and_epistemic_text_routing(self):
        torch.manual_seed(0)
        yy, xx = torch.meshgrid(torch.linspace(0, 1, 160), torch.linspace(0, 1, 160), indexing="ij")
        clean = torch.stack((
            (0.25 + 0.5 * (xx > 0.45).float()),
            (0.30 + 0.4 * (yy > 0.50).float()),
            (0.40 + 0.3 * ((xx + yy) > 1.0).float()),
        ), dim=0).unsqueeze(0)
        noisy = (clean + torch.randn_like(clean) * (35.0 / 255.0)).clamp(0.0, 1.0)
        den = edge_preserving_ycbcr_denoise_01(noisy, sigma_u8=35.0)
        self.assertLess(float((den - clean).square().mean()), float((noisy - clean).square().mean()) * 0.25)

        reg_med = classify_signboard_prior_regime(
            ocr_raw={"text": "Monday-By", "min_prob": 0.56, "mean_prob": 0.83, "mean_ent": 0.42},
            ocr_gentle={"text": "Monday-By", "min_prob": 0.62, "mean_prob": 0.86, "mean_ent": 0.38},
            ocr_z1={"text": "mondsy8y", "min_prob": 0.50, "mean_prob": 0.70, "mean_ent": 0.55},
            nyquist_ratio=0.12,
            from_cand_only=False,
        )
        self.assertEqual(reg_med["regime"], "zoom_with_glyph_prior")
        self.assertEqual(reg_med["prior_text"], "Monday-By")

        reg_keep = classify_signboard_prior_regime(
            ocr_raw={"text": "Too blurry to read", "min_prob": 0.15, "mean_prob": 0.45, "mean_ent": 1.8},
            ocr_gentle={"text": "", "min_prob": 0.10, "mean_prob": 0.40, "mean_ent": 1.9},
            ocr_z1={"text": "LIEMLATS", "min_prob": 0.50, "mean_prob": 0.70, "mean_ent": 0.9},
            nyquist_ratio=0.65,
            from_cand_only=False,
        )
        self.assertEqual(reg_keep["regime"], "keep_z1")

        reg_hard = classify_signboard_prior_regime(
            ocr_raw={"text": "The image is too blurred", "min_prob": 0.12, "mean_prob": 0.30, "mean_ent": 2.1},
            ocr_gentle={"text": "unreadable", "min_prob": 0.10, "mean_prob": 0.28, "mean_ent": 2.2},
            ocr_z1={"text": "12231003100100", "min_prob": 0.45, "mean_prob": 0.65, "mean_ent": 0.9},
            nyquist_ratio=0.05,
            from_cand_only=True,
        )
        self.assertEqual(reg_hard["regime"], "abstain_sub_nyquist")

        merged = merge_dual_side_text_boxes([], [[40, 150, 340, 220]], 384, 384)
        self.assertEqual(len(merged), 1)
        self.assertTrue(merged[0]["from_cand_only"])

    def test_clean_photo_and_utility_gate_guards(self):
        sky_plane = torch.full((1, 3, 192, 256), 0.72, dtype=torch.float32)
        sky_plane[:, :, 70:120, 60:190] = 0.08
        _, m_sky = apply_stage1_blind_preconditioner(
            sky_plane,
            vlm_wea_flags={"lowlight": False, "haze": False, "rain": False, "snow": False},
        )
        self.assertEqual(m_sky["weather_mode"], "camera")
        self.assertEqual(m_sky["omega_wea"], 0.0)

        yy, xx = torch.meshgrid(torch.linspace(0, 1, 160), torch.linspace(0, 1, 160), indexing="ij")
        base_sig = (0.5 + 0.30 * torch.sin(2.0 * math.pi * 6.0 * xx)).reshape(1, 1, 160, 160).repeat(1, 3, 1, 1)
        ref_blur = (0.5 + 0.06 * torch.sin(2.0 * math.pi * 6.0 * xx)).reshape(1, 1, 160, 160).repeat(1, 3, 1, 1)
        z1_noisy = (base_sig + 0.12 * torch.sin(2.0 * math.pi * 24.0 * xx)).clamp(0, 1)
        _, gm = gate_and_fuse_signboard_patch(
            ref_blur, z1_noisy, base_sig,
            prior_text="FORTIFLASH", full_vlm_text="F02TIFLASH", zoom_vlm_text="FORTIFLASH",
        )
        self.assertTrue(gm["used_z4"])

        torch.manual_seed(42)
        _, am = align_candidate_blind_affine_01(torch.rand((1, 3, 160, 160)), torch.rand((1, 3, 160, 160)))
        self.assertFalse(am["affine_applied"])

    def test_stage1_stage2_end_to_end(self):
        torch.manual_seed(7)
        clean = torch.linspace(0.15, 0.85, 192).reshape(1, 1, 1, 192).expand(1, 3, 192, 192).contiguous()
        noisy = (clean + torch.randn_like(clean) * (25.0 / 255.0)).clamp(0.0, 1.0)
        _, m_c = apply_stage1_blind_preconditioner(clean, vlm_wea_flags={"lowlight": False, "haze": False, "rain": False, "snow": False})
        self.assertEqual(m_c["weather_mode"], "camera")
        self.assertAlmostEqual(m_c["eta_step0"], 0.20, places=3)

        x_pre_n, m_n = apply_stage1_blind_preconditioner(noisy, vlm_wea_flags={"lowlight": False, "haze": False, "rain": False, "snow": False})
        self.assertEqual(m_n["prompt_route"], "denoise")
        self.assertTrue(m_n["denoise_regime"])
        self.assertAlmostEqual(m_n["eta_step0"], 0.28, places=3)

        x_out_n, st_n = apply_stage2_physical_projection(noisy, m_n["_x_rad_prior_01"], x_pre_n, omega_wea=0.0, weather_meta=m_n)
        self.assertEqual(tuple(x_out_n.shape), tuple(noisy.shape))
        self.assertGreater(st_n["wiener_noise_regime"], 0.5)

    def test_weather_preconditioner_092m_architecture_and_checkpoint(self):
        model = build_weather_preconditioner(device="cpu", width=48)
        n_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(n_params, 850_000)
        self.assertLess(n_params, 1_000_000)

        ckpt = REPO_ROOT / "weights" / PRECOND_FILENAME
        if ckpt.is_file():
            loaded = load_weather_preconditioner(ckpt, device="cpu", verify_sha256=True)
            x_hazy = torch.full((1, 3, 192, 192), 0.75, dtype=torch.float32)
            x_pre, meta = apply_stage1_blind_preconditioner(
                x_hazy,
                vlm_wea_flags={"lowlight": False, "haze": True, "rain": False, "snow": False},
                v13_precond=loaded,
            )
            self.assertEqual(tuple(x_pre.shape), (1, 3, 192, 192))
            self.assertTrue(bool(torch.isfinite(x_pre).all()))


if __name__ == "__main__":
    unittest.main()
