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
