# Model Weights & Checkpoints

Omni-H3 wraps **unmodified** foundation backbones with only one compact **0.92M-parameter** (`<0.003%` of backbone parameters) auxiliary weather pre-conditioner.

## 1. Omni-H3 Auxiliary Weather Pre-Conditioner (0.92M, 3.5 MB)

Download the frozen auxiliary weather pre-conditioner from [GitHub Releases (`v1.0.0`)](https://github.com/RedRoman/Omni-H3/releases/tag/v1.0.0):

```bash
curl -L -o weights/restoreh3_weather_precond_0.92m.safetensors \
  https://github.com/RedRoman/Omni-H3/releases/download/v1.0.0/restoreh3_weather_precond_0.92m.safetensors
(cd weights && shasum -a 256 -c SHA256SUMS.txt)
```

- **Expected SHA256**: `cb8be30d0bda6f4b976ebd7e4471730d202d5eba0dc370abca4b0e9c32c120f8`
- **Role**: Activated strictly during Stage I when VLM–Koschmieder consensus confirms heavy atmospheric degradation (rain, snow, haze, or composite multi-weather storms). Never alters clean natural photographs.

## 2. Unmodified Foundation Backbone (`MiniMax-H3-Image`)

Download the open-weight 33B `MiniMax-H3-Image` checkpoint (which bundles the frozen `Qwen3-VL-32B` vision-language encoder) into a local directory, e.g. `./models/MiniMax-H3-Image`, and pass `--model-dir ./models/MiniMax-H3-Image` to `omni-h3`. No fine-tuning or LoRA merging is required.
