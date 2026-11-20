"""End-to-End RestoreH3 Blind All-in-One Image Restoration Pipeline.

Orchestrates Three-Stage Constrained Posterior Projection over unmodified 33B MiniMax-H3-Image
and Qwen3-VL-32B with the compact 0.92M auxiliary weather pre-conditioner:
  - Stage I:  Blind Physics-VLM Sensing, Pre-Conditioning & Exact-32px Lattice Framing
  - Stage II: Joint Geometric (NCC Affine), Spectral (Wiener Y), & Chromatic (Butterworth YCbCr) Projection
  - Stage III: Bidirectional ROI Discovery, 3-View Audit, Dimensionless Z=4 Utility Gate & Abstention
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image
import torch

from .core import (
    OCR_READ_PROMPT,
    TEXT_DET_PROMPT,
    WEATHER_SENSING_PROMPT,
    apply_stage1_blind_preconditioner,
    apply_stage2_physical_projection,
    classify_signboard_prior_regime,
    gate_and_fuse_signboard_patch,
    load_prompt_bank,
    merge_dual_side_text_boxes,
)
from .h3_adapter import RestoreH3Adapter
from .ops_primitives import (
    SEED,
    _apply_epistemic_abstention_on_boxes_v12,
    make_gentle_unsharp_signage_crop_01,
    parse_vlm_text_boxes_1000,
    parse_vlm_weather_json,
    plan_tight_signboard_window,
    splice_signboard_box_patches,
)
from .precond import load_weather_preconditioner
from .vlm_auditor import Qwen3VLAuditor


def _pil_to_t01(pil: Image.Image) -> torch.Tensor:
    arr = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def _t01_to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t[0].detach().float().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


class RestoreH3Pipeline:
    """Production single-image & batch inference pipeline for RestoreH3."""

    def __init__(
        self,
        model_dir: Union[str, Path],
        precond_path: Optional[Union[str, Path]] = None,
        device: str = "cuda:0",
        enable_text_stage3: bool = True,
        verify_precond_sha256: bool = True,
        seed: int = SEED,
    ):
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.device = torch.device(device)
        self.enable_text_stage3 = bool(enable_text_stage3)
        self.seed = int(seed)
        self.prompts = load_prompt_bank()

        self.precond = (
            load_weather_preconditioner(precond_path, device=self.device, verify_sha256=verify_precond_sha256)
            if precond_path is not None
            else None
        )
        self.vlm = Qwen3VLAuditor(self.model_dir, device=str(self.device))
        self.adapter = RestoreH3Adapter(model_dir=self.model_dir, device=str(self.device), seed=self.seed)

    @torch.inference_mode()
    def restore(
        self,
        image: Union[str, Path, Image.Image],
        scene_id: str = "image",
        return_metadata: bool = False,
    ) -> Union[Image.Image, tuple[Image.Image, dict]]:
        """Run 100%-blind RestoreH3 restoration on a single RGB image."""
        pil_in = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image.convert("RGB")
        x_in = _pil_to_t01(pil_in)
        h, w = pil_in.size[1], pil_in.size[0]

        # ---------------- Stage I: Blind Physics-VLM Sensing & Pre-Conditioning ----------------
        wea_raw = self.vlm.infer_batch([pil_in], WEATHER_SENSING_PROMPT, upscale_2x=False, max_tokens=48, batch_sz=1)[0]
        vlm_flags = parse_vlm_weather_json(wea_raw["text"])

        x_pre, stage1_meta = apply_stage1_blind_preconditioner(
            x_in,
            vlm_wea_flags=vlm_flags,
            v13_precond=self.precond,
        )
        omega_wea = float(stage1_meta["omega_wea"])
        prompt_key = str(stage1_meta.get("prompt_route", "universal_blind"))
        prompt_txt = self.prompts.get(prompt_key, self.prompts["universal_blind"])

        # Clean-Photo Early Exit Bypass (zero drift)
        if stage1_meta.get("early_exit_clean", False):
            meta = {"stage1": stage1_meta, "stage2": {"bypassed": True}, "stage3": {"bypassed": True}}
            return (pil_in.copy(), meta) if return_metadata else pil_in.copy()

        # ---------------- Frozen 33B MiniMax-H3-Image DiT Synthesis ----------------
        pil_cond = _t01_to_pil(x_pre)
        pil_cand, dit_meta = self.adapter.restore_pil_geolocked(
            pil_cond,
            scene_id=scene_id,
            input_kind="precond" if stage1_meta.get("precond_applied", False) else "raw",
            prompt=prompt_txt,
            min_generation_area=max(512 * 512, w * h),
            eta_step0=float(stage1_meta.get("eta_step0", 0.20)),
            eta_step1=float(stage1_meta.get("eta_step1", 0.08)),
            radiometric_free=(omega_wea > 0.0),
        )
        x_cand = _pil_to_t01(pil_cand)

        # ---------------- Stage II: Joint Geometric-Spectral-Chromatic Projection ----------------
        x_ref = stage1_meta.get("_x_rad_prior_01", x_in)
        x_z1, stage2_meta = apply_stage2_physical_projection(
            x_in,
            x_ref,
            x_cand,
            omega_wea=omega_wea,
            weather_meta=stage1_meta,
        )

        # ---------------- Stage III: Closed-Loop Epistemic Verification ----------------
        x_final = x_z1
        stage3_meta: dict = {"enabled": self.enable_text_stage3, "boxes": 0, "rescued_z4": 0, "abstained": 0}
        if self.enable_text_stage3 and omega_wea == 0.0:
            pil_z1 = _t01_to_pil(x_z1)
            det_res = self.vlm.infer_batch([pil_in, pil_z1], TEXT_DET_PROMPT, upscale_2x=False, max_tokens=256, batch_sz=2)
            boxes_in = parse_vlm_text_boxes_1000(det_res[0]["text"], w, h)
            boxes_z1 = parse_vlm_text_boxes_1000(det_res[1]["text"], w, h)
            merged = merge_dual_side_text_boxes(boxes_in, boxes_z1, w, h)
            stage3_meta["boxes"] = len(merged)

            abstain_boxes = []
            splice_entries = []
            for bi, b in enumerate(merged):
                x0, y0, x1, y1 = b["box_xyxy"]
                crop_in = x_in[:, :, y0:y1, x0:x1]
                crop_z1 = x_z1[:, :, y0:y1, x0:x1]
                crop_g = make_gentle_unsharp_signage_crop_01(crop_in)
                ocr_views = self.vlm.infer_batch(
                    [_t01_to_pil(crop_in), _t01_to_pil(crop_g), _t01_to_pil(crop_z1)],
                    OCR_READ_PROMPT,
                    upscale_2x=True,
                    max_tokens=20,
                    batch_sz=3,
                )
                reg = classify_signboard_prior_regime(
                    ocr_raw=ocr_views[0],
                    ocr_gentle=ocr_views[1],
                    ocr_z1=ocr_views[2],
                    nyquist_ratio=float(b.get("nyquist_ratio", 0.5)),
                    from_cand_only=bool(b.get("from_cand_only", False)),
                )
                if reg["regime"] == "abstain_sub_nyquist":
                    abstain_boxes.append(b["box_xyxy"])
                    stage3_meta["abstained"] += 1
                elif reg["regime"] == "zoom_with_glyph_prior" and reg.get("prior_text"):
                    win = plan_tight_signboard_window(b["box_xyxy"], w, h)
                    wx0, wy0, wx1, wy1 = win["win_xyxy"]
                    win_in = x_in[:, :, wy0:wy1, wx0:wx1]
                    win_z1 = x_z1[:, :, wy0:wy1, wx0:wx1]
                    prior_txt = reg["prior_text"]
                    z4_prompt = (
                        f"{self.prompts['text_zoom_generic']} "
                        f'The signboard clearly reads "{prior_txt}". Render exact sharp strokes for "{prior_txt}".'
                    )
                    pil_z4, _ = self.adapter.restore_pil_geolocked(
                        _t01_to_pil(win_in),
                        scene_id=f"{scene_id}_z4_{bi}",
                        input_kind="z4_sign",
                        prompt=z4_prompt,
                        min_generation_area=1024 * 1024,
                        eta_step0=0.18,
                        eta_step1=0.06,
                    )
                    win_z4 = _pil_to_t01(pil_z4.resize((wx1 - wx0, wy1 - wy0), Image.Resampling.LANCZOS))
                    z4_ocr = self.vlm.infer_batch([_t01_to_pil(win_z4)], OCR_READ_PROMPT, upscale_2x=True, max_tokens=20, batch_sz=1)[0]
                    fused, gm = gate_and_fuse_signboard_patch(
                        win_in,
                        win_z1,
                        win_z4,
                        prior_text=prior_txt,
                        full_vlm_text=ocr_views[2]["text"],
                        zoom_vlm_text=z4_ocr["text"],
                    )
                    if gm.get("used_z4", False):
                        splice_entries.append({"win_xyxy": win["win_xyxy"], "patch_01": fused})
                        stage3_meta["rescued_z4"] += 1

            if splice_entries:
                x_final = splice_signboard_box_patches(x_final, splice_entries)
            if abstain_boxes:
                x_final = _apply_epistemic_abstention_on_boxes_v12(x_final, x_ref, abstain_boxes)

        pil_out = _t01_to_pil(x_final)
        if return_metadata:
            return pil_out, {"stage1": {k: v for k, v in stage1_meta.items() if not k.startswith("_")}, "dit": dit_meta, "stage2": stage2_meta, "stage3": stage3_meta}
        return pil_out
