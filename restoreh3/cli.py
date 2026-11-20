#!/usr/bin/env python3
"""Command-line interface for RestoreH3 blind all-in-one image restoration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import RestoreH3Pipeline

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="restoreh3",
        description="RestoreH3: Physics-Guided Blind All-in-One Image Restoration with MiniMax-H3-Image",
    )
    parser.add_argument("--input", "-i", required=True, type=Path, help="Input image file or directory")
    parser.add_argument("--output", "-o", required=True, type=Path, help="Output image file or directory")
    parser.add_argument("--model-dir", required=True, type=Path, help="Local path to MiniMax-H3-Image checkpoint")
    parser.add_argument(
        "--precond",
        type=Path,
        default=None,
        help="Path to restoreh3_weather_precond_0.92m.safetensors (optional; recommended for heavy weather)",
    )
    parser.add_argument("--device", default="cuda:0", help="Target accelerator device (e.g. cuda:0, npu:0)")
    parser.add_argument("--disable-stage3-text", action="store_true", help="Skip Stage III closed-loop text audit")
    parser.add_argument("--save-meta", action="store_true", help="Write per-image stage metadata JSON alongside output")
    args = parser.parse_args()

    pipe = RestoreH3Pipeline(
        model_dir=args.model_dir,
        precond_path=args.precond,
        device=args.device,
        enable_text_stage3=not args.disable_stage3_text,
    )

    if args.input.is_dir():
        args.output.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in args.input.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        for idx, fp in enumerate(files, start=1):
            out_pil, meta = pipe.restore(fp, scene_id=fp.stem, return_metadata=True)
            out_p = args.output / f"{fp.stem}.png"
            out_pil.save(out_p)
            if args.save_meta:
                (args.output / f"{fp.stem}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"[{idx}/{len(files)}] Saved {out_p}")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out_pil, meta = pipe.restore(args.input, scene_id=args.input.stem, return_metadata=True)
        out_pil.save(args.output)
        if args.save_meta:
            args.output.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
