# MLX-VLM Swift Port

This repository is now focused on replacing the Python `mlx-vlm` runtime with a Swift implementation built on top of `mlx-swift` and `mlx-swift-lm`.

## What Is Here

- `MLXVLMCore`: Swift compatibility contracts for model descriptors, request normalization, prompt/media planning, sampling, response rendering, and server API shapes.
- `MLXVLMMLXBackend`: the real MLX Swift backend bridge using upstream `mlx-swift-lm` model containers, tokenizers, VLM generation, embeddings, LoRA adapters, remote model resolution, and structured-output hooks.
- `mlx-vlm-swift`: a command line tool and compatibility server for Ollama/OpenAI-style generation routes.
- `scripts/verify_swift_port.sh`: fast compatibility verification.
- `scripts/verify_real_gemma4_smoke.sh`: real local Gemma4 smoke test when the model is available.

The old Python package, Python examples, Python CI, and Python packaging files have been removed so future work stays centered on the Swift runtime.

## Build

```bash
swift build --disable-sandbox --jobs 18
```

For the real MLX backend:

```bash
MLXVLM_ENABLE_MLX_BACKEND=1 \
MLXVLM_ENABLE_REAL_MLX_API=1 \
MLXVLM_ENABLE_TOKENIZER_INTEGRATIONS=1 \
swift build --disable-sandbox --jobs 18
```

## Run

Inspect a local MLX model directory:

```bash
.build/debug/mlx-vlm-swift inspect --model /path/to/mlx-model
```

Serve the Swift compatibility API:

```bash
.build/debug/mlx-vlm-swift serve \
  --model /path/to/mlx-model-or-hf-id \
  --host 127.0.0.1 \
  --port 11434 \
  --max-tokens 512
```

## Verify

```bash
SWIFT_BUILD_JOBS=18 scripts/verify_swift_port.sh
```

With a local Gemma4 MLX model:

```bash
SWIFT_BUILD_JOBS=18 scripts/verify_real_gemma4_smoke.sh
```

With a local Qwen VL MLX model for tokenizer, image, video, and streaming input:

```bash
MLXVLM_REAL_QWEN_VL_MODEL=/path/to/qwen-vl-model \
SWIFT_BUILD_JOBS=18 scripts/verify_real_qwen25vl_video_smoke.sh
```

Or let the smoke download the default MLX model first:

```bash
MLXVLM_REAL_QWEN_VL_ALLOW_REMOTE=1 \
SWIFT_BUILD_JOBS=18 scripts/verify_real_qwen25vl_video_smoke.sh
```

The same gate can target Qwen3-VL, for example:

```bash
MLXVLM_REAL_QWEN_VL_MODEL=mlx-community/Qwen3-VL-4B-Instruct-3bit \
MLXVLM_REAL_QWEN_VL_ALLOW_REMOTE=1 \
MLXVLM_REAL_QWEN_VL_DOWNLOAD_DIR="$HOME/Models/mlx-community-Qwen3-VL-4B-Instruct-3bit" \
MLXVLM_REAL_QWEN_VL_LABEL=Qwen3-VL \
SWIFT_BUILD_JOBS=18 scripts/verify_real_qwen25vl_video_smoke.sh
```

It can also target other video-capable upstream VLMs, such as SmolVLM2:

```bash
MLXVLM_REAL_QWEN_VL_MODEL=mlx-community/SmolVLM2-256M-Video-Instruct-mlx \
MLXVLM_REAL_QWEN_VL_ALLOW_REMOTE=1 \
MLXVLM_REAL_QWEN_VL_DOWNLOAD_DIR="$HOME/Models/mlx-community-SmolVLM2-256M-Video-Instruct-mlx" \
MLXVLM_REAL_QWEN_VL_LABEL=SmolVLM2 \
SWIFT_BUILD_JOBS=18 scripts/verify_real_qwen25vl_video_smoke.sh
```

For image-only VLM families such as FastVLM:

```bash
MLXVLM_REAL_IMAGE_MODEL=mlx-community/FastVLM-0.5B-bf16 \
MLXVLM_REAL_IMAGE_ALLOW_REMOTE=1 \
MLXVLM_REAL_IMAGE_DOWNLOAD_DIR="$HOME/Models/mlx-community-FastVLM-0.5B-bf16" \
MLXVLM_REAL_IMAGE_LABEL=FastVLM \
SWIFT_BUILD_JOBS=18 scripts/verify_real_vlm_image_smoke.sh
```

## Port Notes

See [docs/swift_port.md](docs/swift_port.md) for the running compatibility map, implemented parity areas, and known remaining gaps.
