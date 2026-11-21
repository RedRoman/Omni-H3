"""Compact 0.92M-Parameter Auxiliary Weather Pre-Conditioner for Omni-H3.

Represents <0.003% of the 33B MiniMax-H3-Image backbone parameters. Activated exclusively in
Stage I when VLM-Koschmieder consensus confirms severe atmospheric degradation (heavy rain,
snow, dense haze, or composite multi-weather storms).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union

import torch
from safetensors.torch import load_file

from .ops_primitives import SEED, build_v10_omni_precond

PRECOND_FILENAME = "restoreh3_weather_precond_0.92m.safetensors"
PRECOND_SHA256 = "cb8be30d0bda6f4b976ebd7e4471730d202d5eba0dc370abca4b0e9c32c120f8"
PRECOND_RELEASE_URL = (
    "https://github.com/RedRoman/Omni-H3/releases/download/v1.0.0/"
    "restoreh3_weather_precond_0.92m.safetensors"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_weather_preconditioner(device: Union[str, torch.device] = "cpu", width: int = 48, seed: int = SEED):
    """Instantiate the 0.92M auxiliary multi-scale residual weather pre-conditioner."""
    model = build_v10_omni_precond(device=device, seed=seed, width=width)
    model.arm_name = "restoreh3_weather_precond"
    return model


def load_weather_preconditioner(
    checkpoint_path: Union[str, Path],
    device: Union[str, torch.device] = "cpu",
    width: int = 48,
    verify_sha256: bool = True,
):
    """Load the frozen 0.92M auxiliary weather pre-conditioner weights with optional SHA256 audit."""
    ckpt = Path(checkpoint_path).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Missing RestoreH3 weather pre-conditioner checkpoint: {ckpt}\n"
            f"Download from: {PRECOND_RELEASE_URL}"
        )
    if verify_sha256:
        actual = sha256_file(ckpt)
        if actual != PRECOND_SHA256:
            raise ValueError(f"SHA256 mismatch for {ckpt.name}: got {actual}, expected {PRECOND_SHA256}")

    model = build_weather_preconditioner(device=device, width=width)
    state = load_file(str(ckpt), device="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device=torch.device(device)).eval().requires_grad_(False)
    return model
