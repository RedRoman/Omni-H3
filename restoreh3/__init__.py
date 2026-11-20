"""RestoreH3: Physics-Guided Blind All-in-One Image Restoration with MiniMax-H3-Image."""
from __future__ import annotations

__version__ = "1.0.0"

from .core import (
    OCR_READ_PROMPT,
    TEXT_DET_PROMPT,
    WEATHER_SENSING_PROMPT,
    align_candidate_blind_affine_01,
    apply_stage1_blind_preconditioner,
    apply_stage2_physical_projection,
    classify_signboard_prior_regime,
    edge_preserving_ycbcr_denoise_01,
    gate_and_fuse_signboard_patch,
    load_prompt_bank,
    merge_dual_side_text_boxes,
    rgb01_to_ycbcr01,
    ycbcr01_to_rgb01,
)
from .h3_adapter import (
    RestoreH3Adapter,
    compute_exact_parity_frame_spec,
    frame_pil_exact_parity,
    unframe_pil_exact_parity,
)
from .precond import (
    PRECOND_FILENAME,
    PRECOND_RELEASE_URL,
    PRECOND_SHA256,
    build_weather_preconditioner,
    load_weather_preconditioner,
)

__all__ = [
    "__version__",
    "RestoreH3Adapter",
    "build_weather_preconditioner",
    "load_weather_preconditioner",
    "PRECOND_FILENAME",
    "PRECOND_SHA256",
    "PRECOND_RELEASE_URL",
    "compute_exact_parity_frame_spec",
    "frame_pil_exact_parity",
    "unframe_pil_exact_parity",
    "apply_stage1_blind_preconditioner",
    "apply_stage2_physical_projection",
    "align_candidate_blind_affine_01",
    "edge_preserving_ycbcr_denoise_01",
    "rgb01_to_ycbcr01",
    "ycbcr01_to_rgb01",
    "merge_dual_side_text_boxes",
    "classify_signboard_prior_regime",
    "gate_and_fuse_signboard_patch",
    "load_prompt_bank",
    "TEXT_DET_PROMPT",
    "WEATHER_SENSING_PROMPT",
    "OCR_READ_PROMPT",
]
