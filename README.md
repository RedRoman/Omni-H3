<div align="center">

# Omni-H3: Unifying Blind All-in-One Image Restoration in Frozen Foundation Models via Constrained Posterior Projection

**Zihao He** &nbsp;·&nbsp; **Songhua Liu**<sup>*</sup>

School of Artificial Intelligence, Shanghai Jiao Tong University  
<sup>*</sup>Corresponding author: `liusonghua@sjtu.edu.cn`

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](pyproject.toml)
[![Weights](https://img.shields.io/badge/Release-v1.0.0%20Pre--Cond%20(0.92M)-0F766E.svg)](https://github.com/RedRoman/Omni-H3/releases/tag/v1.0.0)
[![Tests](https://img.shields.io/badge/CPU%20Unit%20Tests-Passing-1E644A.svg)](tests/test_restoreh3_cpu.py)

</div>

<p align="center">
  <img src="assets/teaser.jpg" width="100%" alt="Omni-H3 21-Case Blind All-in-One Restoration Teaser"/>
</p>

## Overview

Off-the-shelf unified omni-image foundation models such as 33B **`MiniMax-H3-Image`**, **`FLUX.1-Kontext`**, and **`Qwen-Image-Edit-2511`** synthesize sharp textures yet collapse in pixel and structural fidelity (11.33–16.74 dB PSNR across 429 degraded public scenes) when applied blindly to all-in-one image restoration due to **spatial zoom/lattice drift**, **mid-band chromatic hallucination**, and **high-entropy text fabrication**.

**Omni-H3** formulates blind all-in-one restoration as **Approximate Three-Stage Constrained Posterior Projection** over the **unmodified 33B `MiniMax-H3-Image` and `Qwen3-VL-32B` backbones**, adding only a compact **0.92M-parameter auxiliary weather pre-conditioner** (`<0.003%` of backbone weights):

- **Stage I — Blind Physics–VLM Sensing, Pre-Conditioning & Exact-32 px Lattice Framing ($\mathcal{R}_{\mathrm{phys}}$)**: Unifies VLM–Koschmieder consensus routing, clean-photo Early Exit bypass, compact 0.92M weather pre-stripping, spatial-homogeneity camera-shake deconvolution, Haar-MAD spectral noise anchoring, and exact-32 px isotropic reflect-padding with early-step latent phase anchoring $(\eta_0, \eta_1)$.
- **Stage II — Joint Geometric–Spectral–Chromatic Projection ($\Pi_{\mathcal{M}_{\mathrm{phys}}}$)**: Performs NCC-verified 4-DOF blind sub-pixel affine registration, BT.601 YCbCr luminance octave-band Wiener SNR locking, continuous 3rd-order Butterworth opponent-chroma locking, and an achromatic neutral guard (`-91.6%` Urban100 chromatic drift, eliminating false B&W yellowing).
- **Stage III — Closed-Loop Epistemic Text Verification & Abstention ($\mathcal{V}_{\mathrm{epi}}$)**: Runs bidirectional ROI discovery ($\mathcal{B}_{\mathrm{in}} \cup \mathcal{B}_{Z=1}$), 3-view readability/Nyquist auditing, dimensionless $Z{=}4$ utility gating $U(x)$, and precision sub-Nyquist abstention (`0.0%` alien-script hallucination on TextZoom-Hard and 42-ROI signage).

---

## Architecture

### End-to-End Three-Stage Pipeline
<p align="center">
  <img src="assets/architecture_overview.png" width="98%" alt="Omni-H3 End-to-End Architecture Overview"/>
</p>

### Stage II Detail: Geometric–Spectral–Chromatic Projection
<p align="center">
  <img src="assets/stage2_projection.png" width="98%" alt="Omni-H3 Stage II Projection Detail"/>
</p>

### Stage III Detail: Closed-Loop Epistemic Verification & Abstention
<p align="center">
  <img src="assets/stage3_epistemic.png" width="98%" alt="Omni-H3 Stage III Epistemic Verification Detail"/>
</p>

---

## Main Quantitative Highlights (925 Strictly Blind Tasks, 11 External SOTA Baselines)

Evaluated 100% blindly (zero task IDs, zero dataset names, zero GT leakage; SR benchmarks pre-upsampled to target resolution via bicubic interpolation per standard protocol) across **Public-647** (RealSR-Val, DRealSR-Val, Urban100, CBSD68, TextZoom-150), **Dev-160** (External-64 + Native 2K Dev-96), and the held-out **`locked_test` (118)** suite against **11 external SOTA baselines** spanning 5 unified omni-image models (`Base H3-33B`, `GLM-Image`, `FLUX.2-klein`, `FLUX.1-Kontext-dev`, `Qwen-Image-Edit-2511`) and 6 specialized all-in-one / generative restorers (`AirNet`, `PromptIR`, `InstructIR`, `DiffBIR-v2.1`, `OneRestore`, `DA-CLIP`), plus **`Stage I Ref Only (w/o 33B DiT)`** controls and **dual independent non-Qwen OCR (`RapidOCR PP-OCRv4` & `TrOCR-large-str`)**:

> Aggregate scene rows below report **strictly degraded tasks** (`Degraded-429` and `2K Dev Non-Clean 88`) to avoid clean-identity 99 dB capping bias; CBSD68 Clean-68 is reported in its own row (`97.1%` weather-router specificity, `69.1%` exact 99 dB pixel-identical Early Exit, `76.59 dB` mean PSNR).

| Benchmark Suite | Key Metric | Input | Stage I Ref (w/o DiT) | Base H3-33B | Qwen-2511 | PromptIR | InstructIR | **Omni-H3 (Ours)** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Public Degraded-429** | PSNR (dB) ↑ / DISTS ↓ / NIQE ↓ | 22.67 / 0.293 / 9.95 | 24.01 / 0.295 / 9.41 | 16.74 / 0.223 / 4.46 | 15.20 / 0.287 / 7.38 | 24.73 / 0.234 / 7.42 | **26.26** / 0.213 / 7.03 | **21.47 (#1 Omni) / 0.207 (#1) / 4.45 (#1)** |
| **Urban100 4× SR (100)** | ChP95 ↓ / LPIPS ↓ / DISTS ↓ | — / 0.378 / 0.270 | 0.22 / 0.378 / 0.270 | 9.50 / 0.402 / 0.176 | 17.51 / 0.490 / 0.217 | — / 0.380 / 0.270 | — / 0.359 / 0.247 | **0.80 / 0.288 / 0.163 (#1)** |
| **CBSD68 Clean ID (68)** | PSNR (dB) ↑ | 99.00 | 91.28 | 16.19 | 18.29 | 37.75 | 29.06 | **76.59 (#1 Overall)** |
| **TextZoom-Medium (50)** | Qwen / RapidOCR WordAcc ↑ | 34.0% / 40.0% | 0.0% / 0.0% | 28.0% / 24.0% | 32.0% / 30.0% | 36.0% / **36.0%** | 32.0% / 32.0% | **40.0% (#1) / 36.0% (#1)** |
| **Native 2K Dev (Non-Clean 88)** | PSNR (dB) ↑ / DISTS ↓ | 23.73 / 0.288 | 25.45 / 0.271 | 22.51 / **0.202** | 23.64 / 0.210 | 26.34 / 0.238 | **26.63** / 0.232 | **23.54 / 0.202 (#1)** |
| **Held-Out `locked_test` 2K** | All-24 / NonClean-22 DISTS ↓ | 0.264 / 0.288 | 0.225 / 0.245 | 0.176 / 0.177 | 0.173 / 0.176 | 0.214 / 0.232 | 0.216 / 0.232 | **0.167 (#1) / 0.173 (#1)** |
| **Held-Out 132-ROI (Qwen OCR)** | SubCER ↓ / Can-22 CER ↓ | 0.270 / 3.594 | 0.267 / 0.827 | 0.248 / 0.433 | 0.335 / 0.502 | 0.256 / 2.035 | 0.251 / 0.581 | **0.233 (#1) / 0.432 (#1)** |
| **Held-Out 132-ROI (`RapidOCR` Indep.)** | Exact ↑ / SubCER ↓ / Can-22 CER ↓ | 34.8% / 0.456 / 0.322 | 32.6% / 0.447 / 0.305 | 14.4% / 0.416 / 0.330 | 17.4% / 0.470 / 0.403 | 34.1% / 0.425 / 0.270 | 36.4% / 0.403 / 0.249 | **39.4% / 0.319 / 0.195 (#1 Overall)** |

Complete raw per-suite summary ledgers (including `extra_sota_925_summary.json`, `phys_ref_iqa_summary.json`, and `indep_ocr_summary.json`) are archived in [`benchmarks/results/`](benchmarks/results/).

---

## Installation & Quickstart

### 1. Install Package
```bash
git clone https://github.com/RedRoman/Omni-H3.git
cd Omni-H3
pip install -e .
```

### 2. Download Weights
Download the compact **0.92M auxiliary weather pre-conditioner** (3.5 MB) from [GitHub Releases (`v1.0.0`)](https://github.com/RedRoman/Omni-H3/releases/tag/v1.0.0):
```bash
curl -L -o weights/restoreh3_weather_precond_0.92m.safetensors \
  https://github.com/RedRoman/Omni-H3/releases/download/v1.0.0/restoreh3_weather_precond_0.92m.safetensors
(cd weights && shasum -a 256 -c SHA256SUMS.txt)
```
Place the unmodified open-weight `MiniMax-H3-Image` checkpoint in `./models/MiniMax-H3-Image` (see [`weights/README.md`](weights/README.md)).

### 3. Run Pure-CPU Verification Tests (Zero GPU Required)
```bash
python -m unittest discover -s tests -v
```

### 4. Restore Images via CLI
```bash
# Single image (CUDA or Ascend 910B3 via --device npu:0)
omni-h3 \
  --input examples/degraded.png \
  --output outputs/restored.png \
  --model-dir ./models/MiniMax-H3-Image \
  --precond weights/restoreh3_weather_precond_0.92m.safetensors \
  --device cuda:0 \
  --save-meta

# Batch restore an entire folder
omni-h3 \
  --input ./input_dir \
  --output ./output_dir \
  --model-dir ./models/MiniMax-H3-Image \
  --precond weights/restoreh3_weather_precond_0.92m.safetensors \
  --device cuda:0
```

### 5. Python API
```python
from omnih3.pipeline import OmniH3Pipeline

pipe = OmniH3Pipeline(
    model_dir="./models/MiniMax-H3-Image",
    precond_path="weights/restoreh3_weather_precond_0.92m.safetensors",
    device="cuda:0",  # or "npu:0" on Huawei Ascend 910B3
)
restored_pil, meta = pipe.restore("degraded.png", return_metadata=True)
restored_pil.save("restored.png")
```

---

## Repository Structure

```text
Omni-H3/
├── assets/                   # Teaser and Stage I-III architecture figures
├── omnih3/                   # Primary Omni-H3 package entrypoint (`OmniH3Pipeline`, `omni-h3` CLI)
├── restoreh3/                # Core physical projection operators, pre-conditioner, H3 adapter & VLM auditor
│   ├── core.py               # Stage I sensing/routing, Stage II YCbCr projection, Stage III epistemic audit
│   ├── ops_primitives.py     # Self-contained pyramid, Haar-MAD, Koschmieder, HQS deconv, and OCR helpers
│   ├── precond.py            # 0.92M auxiliary weather pre-conditioner model & SHA256-verified loader
│   ├── h3_adapter.py         # Exact-32px isotropic lattice framing & early-step latent phase anchoring
│   ├── vlm_auditor.py        # Qwen3-VL-32B weather sensor & token-entropy OCR auditor (CUDA & Ascend)
│   ├── pipeline.py           # End-to-end OmniH3Pipeline orchestrator
│   ├── cli.py                # Command-line entrypoint (`omni-h3`)
│   └── prompts.json          # Blind restoration & auditing prompt bank
├── benchmarks/               # 925-task zero-leakage protocol & verified benchmark JSON summaries
├── weights/                  # 0.92M pre-conditioner download guide & SHA256SUMS.txt
└── tests/                    # Pure-CPU unit test suite verifying all mathematical & physical invariants
```

---

## Citation

If you find Omni-H3 useful in your research, please consider citing:

```bibtex
@article{he2026omnih3,
  title   = {Omni-H3: Unifying Blind All-in-One Image Restoration in Frozen Foundation Models via Constrained Posterior Projection},
  author  = {He, Zihao and Liu, Songhua},
  journal = {Technical Report, Shanghai Jiao Tong University},
  year    = {2026},
  url     = {https://github.com/RedRoman/Omni-H3}
}
```

## License

This project is released under the [MIT License](LICENSE). Downstream foundation backbones (`MiniMax-H3-Image`, `Qwen3-VL-32B`) remain subject to their respective model licenses.
