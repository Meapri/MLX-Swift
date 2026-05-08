# Installation

Build the Swift package locally:

```bash
swift build --disable-sandbox --jobs 18
```

Enable the real MLX backend when you want to run against `mlx-swift-lm`:

```bash
MLXVLM_ENABLE_MLX_BACKEND=1 \
MLXVLM_ENABLE_REAL_MLX_API=1 \
MLXVLM_ENABLE_TOKENIZER_INTEGRATIONS=1 \
swift build --disable-sandbox --jobs 18
```

For the native Gemma4 MTP assistant loader, target text verifier loader/smoke commands, `draftBlock` smoke command, and Swift MTP session inspection, add:

```bash
MLXVLM_ENABLE_GEMMA4_ASSISTANT_RUNTIME=1
```

Then inspect the compiled MTP target adapter contract with:

```bash
.build/debug/mlx-vlm-swift inspect-gemma4-mtp-target-plan --model /path/to/gemma4-target --draft-model /path/to/gemma4-assistant
.build/debug/mlx-vlm-swift inspect-gemma4-mtp-target-adapter
.build/debug/mlx-vlm-swift smoke-gemma4-mtp-target-adapter
```
