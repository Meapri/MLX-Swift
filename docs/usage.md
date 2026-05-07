# Usage

Inspect a model:

```bash
.build/debug/mlx-vlm-swift inspect --model /path/to/mlx-model
```

Start the Swift compatibility server:

```bash
.build/debug/mlx-vlm-swift serve \
  --model /path/to/mlx-model-or-hf-id \
  --host 127.0.0.1 \
  --port 11434
```

Run the primary verification script:

```bash
SWIFT_BUILD_JOBS=18 scripts/verify_swift_port.sh
```

Run the real Gemma4 smoke when the local model is present:

```bash
SWIFT_BUILD_JOBS=18 scripts/verify_real_gemma4_smoke.sh
```
