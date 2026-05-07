# Contributing to MLX-VLM Swift

This fork is a Swift-first replacement for the Python `mlx-vlm` runtime. Keep new work centered on Swift compatibility, upstream `mlx-swift-lm` integration, and local MLX model-directory behavior.

## Local Setup

Build the dependency-free Swift target first:

```shell
swift build --disable-sandbox --jobs 18
```

For real MLX generation, build with the backend flags:

```shell
MLXVLM_ENABLE_MLX_BACKEND=1 \
MLXVLM_ENABLE_REAL_MLX_API=1 \
MLXVLM_ENABLE_TOKENIZER_INTEGRATIONS=1 \
swift build --disable-sandbox --jobs 18
```

## Model Compatibility Work

Before adding Swift-owned model code, check whether upstream `mlx-swift-lm` already supports the architecture. This repository should own the compatibility layer around that engine: model inspection, config normalization, request adaptation, media planning, response rendering, adapters, diagnostics, and server behavior.

When a model family needs fork-owned compatibility work:

- Check that the local model directory has `config.json`, tokenizer/processor files, and safetensors weights.
- Preserve Python `mlx-vlm` behavior for config remapping, media placeholders, chat-template semantics, and request fields.
- Add Swift tests or self-test coverage for descriptor parsing, prompt/media planning, request normalization, response rendering, and any real-backend bridge behavior.

Python `mlx-vlm` source remains useful as a compatibility reference, especially for model-family behavior under [`mlx_vlm/models`](https://github.com/Blaizzy/mlx-vlm/tree/main/src/models), but Python code should not be added back to this repository.

## Verification

Run the fast Swift checks:

```shell
swift build --disable-sandbox --jobs 18
swift run mlx-vlm-swift self-test
```

When a local Gemma4-compatible MLX model is available, run `scripts/verify_real_gemma4_smoke.sh` for real-backend coverage.

## Pull Requests

1. Fork and submit pull requests to the repo.
2. If you've added code that should be tested, add tests.
3. Every PR should have passing tests and at least one review.
4. Keep formatting consistent with the surrounding Swift code and avoid unrelated churn.

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## License

By contributing, you agree that your contributions will be licensed under the LICENSE file in the root directory of this source tree.
