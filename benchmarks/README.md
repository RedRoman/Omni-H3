# 925-Task Strictly Blind Benchmark Suite

All quantitative numbers reported in the Omni-H3 technical report are evaluated under a **100%-blind, zero-leakage native-pixel protocol**:

- Zero task labels, dataset tags, ROI annotations, or ground-truth leakage during inference.
- Identical `lpips_vgg`, DISTS, PSNR, SSIM, NIQE, MUSIQ, Sobel Edge-F1@1px, local `32x32` chromatic drift (`ChP95`), and `Qwen3-VL-32B` OCR scoring across all compared external omni-image and all-in-one SOTA baselines.

## Suite Breakdown (`N = 925`)

| Split | Tasks | Coverage | Verified Summary JSON |
|---|---:|---|---|
| **Public Standard Benchmarks** | 647 | RealSR-Val (100), DRealSR-Val (93), Urban100 4x (100), CBSD68 Clean/s25/s50 (204), TextZoom-150 (Easy/Med/Hard) | `results/public_647_summary.json` |
| **Dev Cross-Domain Suite** | 160 | External-64 (Rain100H, SOTS-In/Out, GoPro, CDD-11 1/2/3-way) + 2K Dev (96 native `2048x2048` tasks) + 42-ROI OCR + 20-ROI Geometry | `results/dev_160_summary.json` |
| **Untouched One-Shot `locked_test`** | 118 | External-Heldout-94 + Locked-Camera-24 (`2048x2048`) + 132-ROI Signage OCR + 48-ROI Geometry | `results/locked_test_118_summary.json` |
| **OOD Weather-Routing Study** | 807 | Strict train-only fitted routers (LinearLogOdds, Bilinear, TinyMLP) vs. VLM–Physics consensus across 754 non-weather & 53 external weather images | `results/ood_gating_807_summary.json` |
| **Extended 2024–2026 External SOTA** | 925 | Full blind evaluation of `OneRestore` (ECCV 2024), `DA-CLIP` (ICLR 2024), `FLUX.1-Kontext-dev` (2025), and `Qwen-Image-Edit-2511` (2025/2026) | `results/extra_sota_925_summary.json` |

All 11 external SOTA baselines (`Base H3-33B`, `GLM-Image`, `FLUX.2-klein-4B`, `FLUX.1-Kontext-dev`, `Qwen-Image-Edit-2511`, `AirNet`, `PromptIR`, `InstructIR`, `DiffBIR-v2.1`, `OneRestore`, and `DA-CLIP`) are evaluated under the identical zero-leakage protocol.
