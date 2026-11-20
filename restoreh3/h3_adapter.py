"""Exact-32px Isotropic Lattice Framing & Early-Step Latent Phase Anchoring Adapter for MiniMax-H3-Image.

Eliminates upstream VAE 32px top-left cropping and independent canvas rounding (0.000% aspect shear,
0px truncation) while injecting gentle early-step latent phase anchoring (eta_0, eta_1) at steps t_0, t_1
and leaving steps t_2..t_5 completely unconstrained for photorealistic high-frequency synthesis.
Supports both NVIDIA CUDA GPUs and Huawei Ascend 910B3 NPUs.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time

import numpy as np
from PIL import Image, ImageOps
import torch

try:
    from diffusers.modular_pipelines.modular_pipeline import PipelineState
    from diffusers.modular_pipelines.minimax_h3_image.before_encoder import (
        MiniMaxH3ImageEditResizeStep,
        bound_short_edge,
    )
    from diffusers.modular_pipelines.minimax_h3_image.before_denoise import (
        MiniMaxH3ImageEditPrepareLatentsStep,
        MiniMaxH3ImageEditPrepareLayoutStep,
        MiniMaxH3ImagePrepareCanvasStep,
        MiniMaxH3ImagePrepareLatentsStep,
        MiniMaxH3ImageSetTimestepsStep,
        patchify_video_latents,
    )
    from diffusers.modular_pipelines.minimax_h3_image.decoders import (
        MiniMaxH3ImageAfterDenoiseStep,
        MiniMaxH3ImageDecodeStep,
    )
    from diffusers.modular_pipelines.minimax_h3_image.denoise import MiniMaxH3ImageDenoiseStep
    from diffusers.modular_pipelines.minimax_h3_image.encoders import (
        MiniMaxH3ImageEditTextEncoderStep,
        MiniMaxH3ImageEditVaeEncoderStep,
    )
except ImportError:
    PipelineState = None  # Pure-CPU framing helpers remain usable without H3 diffusers installed

SIGMAS = [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def seed_for(seed: int, *labels) -> int:
    payload = json.dumps([int(seed), *labels], separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


def require_finite(value: torch.Tensor, label: str):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"Non-finite tensor in {label}; refusing quantization")


def _unpack_rows(rows: torch.Tensor, lh: int, lw: int, channels: int = 24, ph: int = 2, pw: int = 2) -> torch.Tensor:
    r = rows.reshape(1, lh // ph, lw // pw, channels, ph, pw)
    return r.permute(0, 3, 1, 4, 2, 5).reshape(1, channels, lh, lw).contiguous()


def _pack_rows(lat: torch.Tensor, ph: int = 2, pw: int = 2) -> torch.Tensor:
    return patchify_video_latents(lat[:, :, None], (1, ph, pw))


def _binomial_blur_lat(x: torch.Tensor, passes: int = 2) -> torch.Tensor:
    import torch.nn.functional as F

    if passes <= 0:
        return x
    ch = x.shape[1]
    k1 = x.new_tensor([0.25, 0.5, 0.25])
    k2 = torch.outer(k1, k1).reshape(1, 1, 3, 3).expand(ch, 1, 3, 3).contiguous()
    h = x
    for _ in range(passes):
        h = F.conv2d(F.pad(h, (1, 1, 1, 1), mode="reflect"), k2, groups=ch)
    return h


def _compute_latent_edge_gate(z_ref: torch.Tensor) -> torch.Tensor:
    """Compute smooth [0, 1] spatial confidence gate on high-SNR reference latent boundaries."""
    z_smooth = _binomial_blur_lat(z_ref, passes=1)
    hf_energy = (z_ref - z_smooth).abs().mean(dim=1, keepdim=True)
    med = hf_energy.flatten(1).median(dim=1).values.reshape(-1, 1, 1, 1)
    snr = (hf_energy - 1.25 * med).clamp_min(0.0) / (med + 0.05)
    return _binomial_blur_lat((snr / (snr + 1.0)).clamp(0.0, 1.0), passes=1)


def compute_affine_invariant_struct_diff(
    z_ref_low: torch.Tensor,
    z_ref_dc: torch.Tensor,
    z0_low: torch.Tensor,
    z0_dc: torch.Tensor,
    max_contrast_gain: float = 3.2,
) -> torch.Tensor:
    """Affine-invariant local-RMS-matched latent phase alignment `(Koschmieder / Retinex invariant)`."""
    u_ref = z_ref_low - z_ref_dc
    u_0 = z0_low - z0_dc

    rms_ref = torch.sqrt(_binomial_blur_lat(u_ref.square(), passes=4).mean(dim=1, keepdim=True) + 1e-6)
    rms_0 = torch.sqrt(_binomial_blur_lat(u_0.square(), passes=4).mean(dim=1, keepdim=True) + 1e-6)

    # Gate contrast rescaling to regions with genuine reference structure (above flat sky noise floor)
    med_ref = rms_ref.flatten(1).median(dim=1).values.reshape(-1, 1, 1, 1)
    struct_conf = ((rms_ref - 0.65 * med_ref) / (0.85 * med_ref + 0.02)).clamp(0.0, 1.0)
    raw_gain = (rms_0 / (rms_ref + 0.015)).clamp(1.0, float(max_contrast_gain))
    eff_gain = 1.0 + struct_conf * (raw_gain - 1.0)

    u_ref_rescaled = eff_gain * u_ref
    return struct_conf * (u_ref_rescaled - u_0)


class H3V12AffinePhaseAdapter:
    """Deterministic native-aspect-ratio modular runner for frozen 33B MiniMax-H3-Image with Affine-Invariant phase anchoring."""

    def __init__(self, pipe, seed: int, device: str | torch.device, max_mem_cache: int = 16):
        self.pipe = pipe
        self.seed = int(seed)
        self.device = torch.device(device)
        self.max_mem_cache = int(max_mem_cache)
        self.resize_step = MiniMaxH3ImageEditResizeStep()
        self.text_encoder = MiniMaxH3ImageEditTextEncoderStep()
        self.vae_encoder = MiniMaxH3ImageEditVaeEncoderStep()
        self.decoder = MiniMaxH3ImageDecodeStep()

        self._ref_cache: dict[tuple, dict] = {}
        self._sem_cache: dict[tuple, dict] = {}
        assert tuple(pipe.patch_size) == (1, 2, 2)
        assert pipe.vae_spatial_compression_ratio == 16
        assert pipe.vae_latent_channels == 24
        assert list(pipe.config.sample_sigmas) == SIGMAS

    def sync(self):
        if self.device.type == "npu":
            torch.npu.synchronize(self.device)

    def state(self, **values) -> PipelineState:
        st = PipelineState()
        for k, v in values.items():
            st.set(k, v)
        return st

    def call(self, block, state: PipelineState) -> PipelineState:
        if isinstance(block, MiniMaxH3ImageDecodeStep):
            orig_decode = self.pipe.vae.decode

            def checked_decode(*args, **kwargs):
                res = orig_decode(*args, **kwargs)
                pixels = res[0] if isinstance(res, tuple) else res.sample
                require_finite(pixels, "raw VAE decode output before uint8 quantization")
                return res

            self.pipe.vae.decode = checked_decode
            try:
                _, ret = block(self.pipe, state)
            finally:
                self.pipe.vae.decode = orig_decode
        else:
            _, ret = block(self.pipe, state)
        assert ret is state
        return state

    def _trim_cache(self, d: dict):
        while len(d) > self.max_mem_cache:
            oldest = next(iter(d.keys()))
            d.pop(oldest, None)

    @torch.inference_mode()
    def encode_reference_and_vision(
        self,
        scene_id: str,
        input_kind: str,
        pil_rgb: Image.Image,
        min_ref_short_edge: int = 512,
    ) -> dict:
        img_bytes = pil_rgb.tobytes()
        in_sha = sha_bytes(img_bytes + f":{pil_rgb.size[0]}x{pil_rgb.size[1]}:r{min_ref_short_edge}".encode("utf-8"))
        key = (in_sha, scene_id, input_kind, int(min_ref_short_edge))
        if key in self._ref_cache:
            return self._ref_cache[key]

        st_rsz = self.state(image=pil_rgb)
        self.call(self.resize_step, st_rsz)
        vis_img = st_rsz.vision_images[0]
        req_min = int(min_ref_short_edge)
        cfg_min = int(self.pipe.config.reference_min_short_edge)
        cfg_max = int(self.pipe.config.reference_max_short_edge)
        if cfg_min <= req_min <= cfg_max:
            ref_img = st_rsz.reference_images[0]
        else:
            upright = ImageOps.exif_transpose(pil_rgb).convert("RGB")
            r_size = bound_short_edge(
                *upright.size,
                req_min,
                max(req_min, cfg_max),
                int(self.pipe.canvas_multiple),
            )
            ref_img = (
                upright
                if upright.size == r_size
                else self.pipe.image_processor.resize(upright, height=r_size[1], width=r_size[0])
            )
        rw, rh = ref_img.size

        post_seed = seed_for(self.seed, scene_id, "posterior", rw, rh)
        gen_post = torch.Generator(device="cpu").manual_seed(post_seed)
        st_vae = self.state(reference_images=[ref_img], generator=gen_post)
        self.call(self.vae_encoder, st_vae)
        self.sync()
        ref_lat = st_vae.reference_latents[0].detach().cpu()
        require_finite(ref_lat, "reference_latents")

        aug_seed = seed_for(self.seed, scene_id, "reference_augmentation", rw, rh)
        gen_aug = torch.Generator(device="cpu").manual_seed(aug_seed)
        ref_noise = torch.randn(ref_lat.shape, generator=gen_aug, dtype=torch.float32)

        entry = {
            "in_sha": in_sha,
            "vision_img": vis_img,
            "ref_size": (rw, rh),
            "reference_latents": ref_lat,
            "reference_noise": ref_noise,
        }
        self._ref_cache[key] = entry
        self._trim_cache(self._ref_cache)
        return entry

    @torch.inference_mode()
    def encode_semantic(self, in_sha: str, vision_img: Image.Image, prompt: str) -> dict:
        key = (in_sha, prompt)
        if key in self._sem_cache:
            return self._sem_cache[key]
        st_sem = self.state(prompt=prompt, vision_images=[vision_img])
        self.call(self.text_encoder, st_sem)
        self.sync()
        pe = st_sem.prompt_embeds.detach().cpu()
        tags = st_sem.text_token_tags.detach().cpu()
        require_finite(pe, "prompt_embeds")
        entry = {"prompt_embeds": pe, "text_token_tags": tags}
        self._sem_cache[key] = entry
        self._trim_cache(self._sem_cache)
        return entry

    @torch.inference_mode()
    def restore_pil(
        self,
        pil_in: Image.Image,
        scene_id: str,
        input_kind: str,
        prompt: str,
        min_generation_area: int = 512 * 512,
        min_ref_short_edge: int = 512,
        eta_step0: float = 0.0,
        eta_step1: float = 0.0,
        anchor_blur_passes: int = 2,
        edge_anchor_gain: float = 0.35,
        radiometric_free: bool = False,
        anchor_pil_override: Image.Image | None = None,
    ) -> tuple[Image.Image, dict]:
        """Run deterministic 6-step H3 restoration with optional affine-invariant early-step phase anchoring."""
        import torch.nn.functional as F

        t0 = time.monotonic()
        rgb_in = pil_in.convert("RGB")
        orig_w, orig_h = rgb_in.size

        ref_pack = self.encode_reference_and_vision(
            scene_id,
            input_kind,
            rgb_in,
            min_ref_short_edge=int(min_ref_short_edge),
        )
        sem_pack = self.encode_semantic(ref_pack["in_sha"], ref_pack["vision_img"], prompt)

        st = self.state(
            height=orig_h,
            width=orig_w,
            min_generation_area=int(min_generation_area),
            prompt_embeds=sem_pack["prompt_embeds"].to(self.device),
            text_token_tags=sem_pack["text_token_tags"],
            reference_latents=[ref_pack["reference_latents"]],
            reference_noise=[ref_pack["reference_noise"]],
            output_type="pil",
        )
        self.call(MiniMaxH3ImagePrepareCanvasStep(), st)
        canvas_h, canvas_w = int(st.height), int(st.width)
        lh, lw = canvas_h // 16, canvas_w // 16

        tgt_seed = seed_for(self.seed, scene_id, "target_noise", canvas_w, canvas_h)
        gen_tgt = torch.Generator(device="cpu").manual_seed(tgt_seed)
        tgt_noise = torch.randn((1, 24, lh, lw), generator=gen_tgt, dtype=torch.float32)
        st.set("latents", tgt_noise)

        for cls in (
            MiniMaxH3ImageEditPrepareLayoutStep,
            MiniMaxH3ImagePrepareLatentsStep,
            MiniMaxH3ImageEditPrepareLatentsStep,
            MiniMaxH3ImageSetTimestepsStep,
        ):
            self.call(cls(), st)

        require_finite(st.latents, "prepared packed image rows")
        num_ref_rows = int(st.num_reference_rows)
        e0 = float(eta_step0)
        e1 = float(eta_step1)
        bpasses = int(anchor_blur_passes)
        egain = float(edge_anchor_gain)
        rad_free = bool(radiometric_free)

        anchor_trace = {
            "eta_step0": e0,
            "eta_step1": e1,
            "anchor_blur_passes": bpasses,
            "radiometric_free": rad_free,
            "step0_dv_rms": 0.0,
            "step1_dv_rms": 0.0,
        }
        handles = []
        if e0 > 0.0 or e1 > 0.0:
            if anchor_pil_override is not None:
                anc_pack = self.encode_reference_and_vision(
                    scene_id,
                    f"{input_kind}_anc",
                    anchor_pil_override.convert("RGB"),
                    min_ref_short_edge=int(min_ref_short_edge),
                )
                anc_lat_src = anc_pack["reference_latents"]
            else:
                anc_lat_src = ref_pack["reference_latents"]

            ref_lat_dev = anc_lat_src.to(device=self.device, dtype=torch.float32)
            if ref_lat_dev.shape[-2:] != (lh, lw):
                ref_lat_aligned = F.interpolate(ref_lat_dev, size=(lh, lw), mode="bilinear", align_corners=False)
            else:
                ref_lat_aligned = ref_lat_dev

            z_ref_low = _binomial_blur_lat(ref_lat_aligned, passes=bpasses)
            z_ref_mid = _binomial_blur_lat(ref_lat_aligned, passes=max(0, bpasses - 1)) - z_ref_low
            z_ref_dc = _binomial_blur_lat(ref_lat_aligned, passes=12) if rad_free else None
            edge_gate = _compute_latent_edge_gate(ref_lat_aligned)
            step_x_holder: dict[int, torch.Tensor] = {}

            def pre_hook(module, args, kwargs):
                s_idx = int(kwargs.get("pdd_step_index", -1))
                if (s_idx == 0 and e0 > 0.0) or (s_idx == 1 and e1 > 0.0):
                    hs = kwargs["hidden_states"]
                    step_x_holder[s_idx] = hs[0, num_ref_rows:, :].detach().float()

            def post_hook(module, args, kwargs, output):
                s_idx = int(kwargs.get("pdd_step_index", -1))
                eta_cur = e0 if s_idx == 0 else (e1 if s_idx == 1 else 0.0)
                if eta_cur <= 0.0 or s_idx not in step_x_holder:
                    return None

                sigma_cur = float(SIGMAS[s_idx])
                vel_full = output[0] if isinstance(output, tuple) else output.sample
                xk_rows = step_x_holder.pop(s_idx)
                vk_rows = vel_full[0, num_ref_rows:, :].float()

                xk_lat = _unpack_rows(xk_rows, lh, lw)
                vk_lat = _unpack_rows(vk_rows, lh, lw)
                z0_hat = xk_lat + sigma_cur * vk_lat

                z0_low = _binomial_blur_lat(z0_hat, passes=bpasses)
                z0_mid = _binomial_blur_lat(z0_hat, passes=max(0, bpasses - 1)) - z0_low

                if rad_free and z_ref_dc is not None:
                    z0_dc = _binomial_blur_lat(z0_hat, passes=12)
                    struct_diff = compute_affine_invariant_struct_diff(z_ref_low, z_ref_dc, z0_low, z0_dc)
                    mid_diff = 0.65 * edge_gate * (1.35 * z_ref_mid - z0_mid)
                else:
                    struct_diff = z_ref_low - z0_low
                    mid_diff = egain * edge_gate * (z_ref_mid - z0_mid)

                dz0_lat = eta_cur * (struct_diff + mid_diff)
                dv_lat = dz0_lat / max(sigma_cur, 1e-4)
                dv_rows = _pack_rows(dv_lat).reshape_as(vk_rows)
                anchor_trace[f"step{s_idx}_dv_rms"] = round(float(dv_rows.square().mean().sqrt().cpu()), 6)

                new_vel = vel_full.clone()
                new_vel[0, num_ref_rows:, :] = (vk_rows + dv_rows).to(dtype=vel_full.dtype)
                if isinstance(output, tuple):
                    return (new_vel, *output[1:])
                new_out = copy.copy(output)
                if isinstance(new_out, dict):
                    new_out["sample"] = new_vel
                setattr(new_out, "sample", new_vel)
                return new_out

            handles.append(self.pipe.transformer.register_forward_pre_hook(pre_hook, with_kwargs=True))
            handles.append(self.pipe.transformer.register_forward_hook(post_hook, with_kwargs=True))

        t_den0 = time.monotonic()
        try:
            self.call(MiniMaxH3ImageDenoiseStep(), st)
            self.sync()
        finally:
            for h in handles:
                h.remove()
        den_sec = time.monotonic() - t_den0

        self.call(MiniMaxH3ImageAfterDenoiseStep(), st)
        require_finite(st.latents, "final denoised target latents")
        self.call(self.decoder, st)
        self.sync()
        out_pil = st.images[0]
        if out_pil.size != (orig_w, orig_h):
            out_pil = out_pil.resize((orig_w, orig_h), Image.Resampling.LANCZOS)

        for component in self.pipe.components.values():
            if hasattr(component, "_reset_stateful_cache"):
                component._reset_stateful_cache()

        meta = {
            "orig_size": [orig_w, orig_h],
            "canvas_size": [canvas_w, canvas_h],
            "ref_size": list(ref_pack["ref_size"]),
            "denoise_seconds": round(den_sec, 3),
            "total_seconds": round(time.monotonic() - t0, 3),
            **anchor_trace,
        }
        return out_pil, meta

def compute_exact_parity_frame_spec(
    orig_w: int,
    orig_h: int,
    min_generation_area: int = 512 * 512,
    min_ref_short_edge: int = 512,
    max_ref_short_edge: int = 1024,
    multiple: int = 32,
) -> dict:
    """Compute isotropic scaling + symmetric reflect-pad so `ref_size` and `canvas_size`
    have exact integer ratio (`1x` or `0.5x`), `0.000%` aspect distortion, and `0 px` VAE crop.
    """
    ow, oh = int(orig_w), int(orig_h)
    short_e = min(ow, oh)
    area = ow * oh
    # Scale isotropically if either short edge < min_ref_short_edge or area < min_generation_area
    s_short = float(min_ref_short_edge) / float(short_e) if short_e < int(min_ref_short_edge) else 1.0
    s_area = math.sqrt(float(min_generation_area) / float(max(1, area))) if area < int(min_generation_area) else 1.0
    scale = max(1.0, s_short, s_area)

    cont_w = max(multiple, int(round(ow * scale)))
    cont_h = max(multiple, int(round(oh * scale)))

    # If target canvas will exceed max_ref_short_edge (e.g. 2048x2048), round to `2*multiple=64`
    # so exact `0.5x` reference encoding is STILL an exact multiple of `32` with `0 px` crop!
    eff_mult = multiple * 2 if min(cont_w, cont_h) > int(max_ref_short_edge) else multiple
    gen_w = int(math.ceil(cont_w / float(eff_mult)) * eff_mult)
    gen_h = int(math.ceil(cont_h / float(eff_mult)) * eff_mult)

    pad_w = gen_w - cont_w
    pad_h = gen_h - cont_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    if min(gen_w, gen_h) > int(max_ref_short_edge):
        ref_w, ref_h = gen_w // 2, gen_h // 2
    else:
        ref_w, ref_h = gen_w, gen_h

    return {
        "orig_w": ow,
        "orig_h": oh,
        "scale": round(scale, 6),
        "cont_w": cont_w,
        "cont_h": cont_h,
        "gen_w": gen_w,
        "gen_h": gen_h,
        "ref_w": ref_w,
        "ref_h": ref_h,
        "pad_left": pad_left,
        "pad_right": pad_right,
        "pad_top": pad_top,
        "pad_bottom": pad_bottom,
    }


def frame_pil_exact_parity(pil_rgb: Image.Image, spec: dict) -> Image.Image:
    """Isotropically resize (if needed) and symmetrically reflect-pad `pil_rgb` onto `(gen_w, gen_h)`."""
    rgb = pil_rgb.convert("RGB")
    cw, ch = int(spec["cont_w"]), int(spec["cont_h"])
    if rgb.size != (cw, ch):
        rgb = rgb.resize((cw, ch), Image.Resampling.BICUBIC)
    arr = np.asarray(rgb, dtype=np.uint8)
    pt, pb = int(spec["pad_top"]), int(spec["pad_bottom"])
    pl, pr = int(spec["pad_left"]), int(spec["pad_right"])
    if pt == 0 and pb == 0 and pl == 0 and pr == 0:
        return Image.fromarray(arr, mode="RGB")
    mode = "reflect" if (min(arr.shape[0], arr.shape[1]) > max(pt, pb, pl, pr) + 1) else "edge"
    padded = np.pad(arr, ((pt, pb), (pl, pr), (0, 0)), mode=mode)
    return Image.fromarray(np.ascontiguousarray(padded, dtype=np.uint8), mode="RGB")


def unframe_pil_exact_parity(pil_gen: Image.Image, spec: dict) -> Image.Image:
    """Unpad symmetric border and downsample isotropically back to exact `(orig_w, orig_h)`."""
    gw, gh = int(spec["gen_w"]), int(spec["gen_h"])
    rgb = pil_gen.convert("RGB")
    if rgb.size != (gw, gh):
        rgb = rgb.resize((gw, gh), Image.Resampling.LANCZOS)
    arr = np.asarray(rgb, dtype=np.uint8)
    pt, pl = int(spec["pad_top"]), int(spec["pad_left"])
    cw, ch = int(spec["cont_w"]), int(spec["cont_h"])
    cropped = arr[pt : pt + ch, pl : pl + cw]
    out = Image.fromarray(np.ascontiguousarray(cropped, dtype=np.uint8), mode="RGB")
    ow, oh = int(spec["orig_w"]), int(spec["orig_h"])
    if out.size != (ow, oh):
        out = out.resize((ow, oh), Image.Resampling.LANCZOS)
    return out


class H3V14GeoLockedAdapter(H3V12AffinePhaseAdapter):
    """Geometry-locked exact-lattice runner for frozen 33B MiniMax-H3-Image (`PGAR-v14`)."""

    @torch.inference_mode()
    def encode_reference_and_vision_exact(
        self,
        scene_id: str,
        input_kind: str,
        pil_gen: Image.Image,
        ref_w: int,
        ref_h: int,
    ) -> dict:
        """Encode reference at exact `(ref_w, ref_h)` (both multiples of 32, zero VAE truncation!)."""
        rgb_gen = pil_gen.convert("RGB")
        img_bytes = rgb_gen.tobytes()
        in_sha = sha_bytes(img_bytes + f":{rgb_gen.size[0]}x{rgb_gen.size[1]}:exact_{ref_w}x{ref_h}".encode("utf-8"))
        key = (in_sha, scene_id, input_kind, int(ref_w), int(ref_h))
        if key in self._ref_cache:
            return self._ref_cache[key]

        st_rsz = self.state(image=rgb_gen)
        self.call(self.resize_step, st_rsz)
        vis_img = st_rsz.vision_images[0]

        if rgb_gen.size == (int(ref_w), int(ref_h)):
            ref_img = rgb_gen
        else:
            ref_img = rgb_gen.resize((int(ref_w), int(ref_h)), Image.Resampling.LANCZOS)
        rw, rh = ref_img.size

        post_seed = seed_for(self.seed, scene_id, "posterior", rw, rh)
        gen_post = torch.Generator(device="cpu").manual_seed(post_seed)
        st_vae = self.state(reference_images=[ref_img], generator=gen_post)
        self.call(self.vae_encoder, st_vae)
        self.sync()
        ref_lat = st_vae.reference_latents[0].detach().cpu()
        require_finite(ref_lat, "reference_latents_exact")

        aug_seed = seed_for(self.seed, scene_id, "reference_augmentation", rw, rh)
        gen_aug = torch.Generator(device="cpu").manual_seed(aug_seed)
        ref_noise = torch.randn(ref_lat.shape, generator=gen_aug, dtype=torch.float32)

        entry = {
            "in_sha": in_sha,
            "vision_img": vis_img,
            "ref_size": (rw, rh),
            "reference_latents": ref_lat,
            "reference_noise": ref_noise,
        }
        self._ref_cache[key] = entry
        self._trim_cache(self._ref_cache)
        return entry

    @torch.inference_mode()
    def restore_pil_geolocked(
        self,
        pil_in: Image.Image,
        scene_id: str,
        input_kind: str,
        prompt: str,
        min_generation_area: int = 512 * 512,
        min_ref_short_edge: int = 512,
        eta_step0: float = 0.0,
        eta_step1: float = 0.0,
        anchor_blur_passes: int = 2,
        edge_anchor_gain: float = 0.35,
        radiometric_free: bool = False,
        anchor_pil_override: Image.Image | None = None,
    ) -> tuple[Image.Image, dict]:
        """Run deterministic 6-step H3 restoration on an exact `32px`-parity lattice with
        optional affine-invariant early-step latent phase anchoring.
        """
        import torch.nn.functional as F
        from diffusers.modular_pipelines.minimax_h3_image.before_denoise import (
            MiniMaxH3ImageEditPrepareLatentsStep,
            MiniMaxH3ImageEditPrepareLayoutStep,
            MiniMaxH3ImagePrepareCanvasStep,
            MiniMaxH3ImagePrepareLatentsStep,
            MiniMaxH3ImageSetTimestepsStep,
        )
        from diffusers.modular_pipelines.minimax_h3_image.decoders import MiniMaxH3ImageAfterDenoiseStep
        from diffusers.modular_pipelines.minimax_h3_image.denoise import MiniMaxH3ImageDenoiseStep

        t0 = time.monotonic()
        rgb_in = ImageOps.exif_transpose(pil_in).convert("RGB")
        orig_w, orig_h = rgb_in.size
        cfg_max = int(self.pipe.config.reference_max_short_edge)
        spec = compute_exact_parity_frame_spec(
            orig_w,
            orig_h,
            min_generation_area=int(min_generation_area),
            min_ref_short_edge=int(min_ref_short_edge),
            max_ref_short_edge=cfg_max,
            multiple=int(self.pipe.canvas_multiple),
        )
        pil_gen = frame_pil_exact_parity(rgb_in, spec)
        gen_w, gen_h = int(spec["gen_w"]), int(spec["gen_h"])
        ref_w, ref_h = int(spec["ref_w"]), int(spec["ref_h"])

        ref_pack = self.encode_reference_and_vision_exact(scene_id, input_kind, pil_gen, ref_w, ref_h)
        sem_pack = self.encode_semantic(ref_pack["in_sha"], ref_pack["vision_img"], prompt)

        st = self.state(
            height=gen_h,
            width=gen_w,
            min_generation_area=gen_w * gen_h,
            prompt_embeds=sem_pack["prompt_embeds"].to(self.device),
            text_token_tags=sem_pack["text_token_tags"],
            reference_latents=[ref_pack["reference_latents"]],
            reference_noise=[ref_pack["reference_noise"]],
            output_type="pil",
        )
        self.call(MiniMaxH3ImagePrepareCanvasStep(), st)
        canvas_h, canvas_w = int(st.height), int(st.width)
        assert (canvas_w, canvas_h) == (gen_w, gen_h), f"Parity violated: {(canvas_w, canvas_h)} != {(gen_w, gen_h)}"
        lh, lw = canvas_h // 16, canvas_w // 16

        tgt_seed = seed_for(self.seed, scene_id, "target_noise", canvas_w, canvas_h)
        gen_tgt = torch.Generator(device="cpu").manual_seed(tgt_seed)
        tgt_noise = torch.randn((1, 24, lh, lw), generator=gen_tgt, dtype=torch.float32)
        st.set("latents", tgt_noise)

        for cls in (
            MiniMaxH3ImageEditPrepareLayoutStep,
            MiniMaxH3ImagePrepareLatentsStep,
            MiniMaxH3ImageEditPrepareLatentsStep,
            MiniMaxH3ImageSetTimestepsStep,
        ):
            self.call(cls(), st)

        require_finite(st.latents, "prepared packed image rows v14")
        num_ref_rows = int(st.num_reference_rows)
        e0 = float(eta_step0)
        e1 = float(eta_step1)
        bpasses = int(anchor_blur_passes)
        egain = float(edge_anchor_gain)
        rad_free = bool(radiometric_free)

        anchor_trace = {
            "eta_step0": e0,
            "eta_step1": e1,
            "anchor_blur_passes": bpasses,
            "radiometric_free": rad_free,
            "step0_dv_rms": 0.0,
            "step1_dv_rms": 0.0,
        }
        handles = []
        if e0 > 0.0 or e1 > 0.0:
            if anchor_pil_override is not None:
                anc_gen = frame_pil_exact_parity(anchor_pil_override.convert("RGB"), spec)
                anc_pack = self.encode_reference_and_vision_exact(
                    scene_id, f"{input_kind}_anc", anc_gen, ref_w, ref_h
                )
                anc_lat_src = anc_pack["reference_latents"]
            else:
                anc_lat_src = ref_pack["reference_latents"]

            ref_lat_dev = anc_lat_src.to(device=self.device, dtype=torch.float32)
            if ref_lat_dev.shape[-2:] != (lh, lw):
                ref_lat_aligned = F.interpolate(ref_lat_dev, size=(lh, lw), mode="bilinear", align_corners=False)
            else:
                ref_lat_aligned = ref_lat_dev

            z_ref_low = _binomial_blur_lat(ref_lat_aligned, passes=bpasses)
            z_ref_mid = _binomial_blur_lat(ref_lat_aligned, passes=max(0, bpasses - 1)) - z_ref_low
            z_ref_dc = _binomial_blur_lat(ref_lat_aligned, passes=12) if rad_free else None
            edge_gate = _compute_latent_edge_gate(ref_lat_aligned)
            step_x_holder: dict[int, torch.Tensor] = {}

            def pre_hook(module, args, kwargs):
                s_idx = int(kwargs.get("pdd_step_index", -1))
                if (s_idx == 0 and e0 > 0.0) or (s_idx == 1 and e1 > 0.0):
                    hs = kwargs["hidden_states"]
                    step_x_holder[s_idx] = hs[0, num_ref_rows:, :].detach().float()

            def post_hook(module, args, kwargs, output):
                s_idx = int(kwargs.get("pdd_step_index", -1))
                eta_cur = e0 if s_idx == 0 else (e1 if s_idx == 1 else 0.0)
                if eta_cur <= 0.0 or s_idx not in step_x_holder:
                    return None

                sigma_cur = float(SIGMAS[s_idx])
                vel_full = output[0] if isinstance(output, tuple) else output.sample
                xk_rows = step_x_holder.pop(s_idx)
                vk_rows = vel_full[0, num_ref_rows:, :].float()

                xk_lat = _unpack_rows(xk_rows, lh, lw)
                vk_lat = _unpack_rows(vk_rows, lh, lw)
                z0_hat = xk_lat + sigma_cur * vk_lat

                z0_low = _binomial_blur_lat(z0_hat, passes=bpasses)
                z0_mid = _binomial_blur_lat(z0_hat, passes=max(0, bpasses - 1)) - z0_low

                if rad_free and z_ref_dc is not None:
                    z0_dc = _binomial_blur_lat(z0_hat, passes=12)
                    struct_diff = compute_affine_invariant_struct_diff(z_ref_low, z_ref_dc, z0_low, z0_dc)
                    mid_diff = 0.65 * edge_gate * (1.35 * z_ref_mid - z0_mid)
                else:
                    struct_diff = z_ref_low - z0_low
                    mid_diff = egain * edge_gate * (z_ref_mid - z0_mid)

                dz0_lat = eta_cur * (struct_diff + mid_diff)
                dv_lat = dz0_lat / max(sigma_cur, 1e-4)
                dv_rows = _pack_rows(dv_lat).reshape_as(vk_rows)
                anchor_trace[f"step{s_idx}_dv_rms"] = round(float(dv_rows.square().mean().sqrt().cpu()), 6)

                new_vel = vel_full.clone()
                new_vel[0, num_ref_rows:, :] = (vk_rows + dv_rows).to(dtype=vel_full.dtype)
                if isinstance(output, tuple):
                    return (new_vel, *output[1:])
                new_out = copy.copy(output)
                if isinstance(new_out, dict):
                    new_out["sample"] = new_vel
                setattr(new_out, "sample", new_vel)
                return new_out

            handles.append(self.pipe.transformer.register_forward_pre_hook(pre_hook, with_kwargs=True))
            handles.append(self.pipe.transformer.register_forward_hook(post_hook, with_kwargs=True))

        t_den0 = time.monotonic()
        try:
            self.call(MiniMaxH3ImageDenoiseStep(), st)
            self.sync()
        finally:
            for h in handles:
                h.remove()
        den_sec = time.monotonic() - t_den0

        self.call(MiniMaxH3ImageAfterDenoiseStep(), st)
        require_finite(st.latents, "final denoised target latents v14")
        self.call(self.decoder, st)
        self.sync()
        out_pil = unframe_pil_exact_parity(st.images[0], spec)

        for component in self.pipe.components.values():
            if hasattr(component, "_reset_stateful_cache"):
                component._reset_stateful_cache()

        meta = {
            "orig_size": [orig_w, orig_h],
            "canvas_size": [canvas_w, canvas_h],
            "ref_size": list(ref_pack["ref_size"]),
            "parity_spec": spec,
            "denoise_seconds": round(den_sec, 3),
            "total_seconds": round(time.monotonic() - t0, 3),
            **anchor_trace,
        }
        return out_pil, meta

# Clean Public Alias
RestoreH3Adapter = H3V14GeoLockedAdapter
