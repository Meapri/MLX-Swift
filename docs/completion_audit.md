# Swift mlx-vlm Replacement Completion Audit

Objective: replace the Python `mlx-vlm` runtime with a 100% Swift compatibility/runtime layer while preserving existing MLX model directory compatibility and using official `mlx-swift` / `mlx-swift-lm` as the primary engine.

Status: not complete. The Swift server is usable for the main Gemma4/OpenAI/Ollama path, but full Python `mlx-vlm` parity still has uncovered model-family and structured-output gaps.

## Success Criteria

| Requirement | Current evidence | Status |
| --- | --- | --- |
| Python runtime removed from the working tree | `aeff99d Remove legacy Python mlx-vlm runtime`; no Python server files are part of the Swift runtime path | Done |
| Swift package builds without Python runtime | `swift build --disable-sandbox --jobs 18` passes | Done |
| Official `mlx-swift-lm` is the primary real inference engine | `MLXVLMUpstreamBackend` loads `VLMModelFactory.shared.loadContainer`, maps requests into `MLXLMCommon.UserInput`, and streams upstream `Generation` events | Done |
| Local MLX model directories remain inspectable | `ModelStore`, descriptor/config/tokenizer/processor/weight catalog commands are covered by `scripts/verify_swift_port.sh` | Done |
| OpenAI-compatible chat/completions/responses routes work | `scripts/verify_swift_port.sh` and `scripts/verify_real_gemma4_smoke.sh` cover chat, completions compatibility, responses, streaming, usage, and model endpoints | Mostly done |
| Ollama-compatible generate/chat/show/tags/ps/model-management/blob routes work | `2d5eaaf` implements model/blob management; `ce6e02e` adds real Gemma4 smoke coverage for these endpoints | Done for current server-local semantics |
| Tokenizer compatibility does not require Python for common sidecars | `d8a75db` adds WordPiece, `vocab.txt`, `tokenizer.tiktoken`, ByteLevel BPE, sidecar BPE, and WordLevel fallback coverage; the current tree adds Unigram `tokenizer.json` plus binary SentencePiece `tokenizer.model` greedy fallback coverage | Mostly done |
| Gemma4 local MLX real backend works end-to-end | `scripts/verify_real_gemma4_smoke.sh` passes with the local Gemma4 E4B model, including chat, tools, image input, streaming, embeddings diagnostic, tokenize, and model/blob endpoints | Done |
| Audio/video/image metadata survives API normalization and upstream handoff | Self-test and `scripts/verify_swift_port.sh` cover metadata parsing/planning; mock upstream contracts include `UserInput.Video` | Partially verified |
| Video-capable model generation works end-to-end | Local cache currently contains Gemma4 models only; no video-capable MLX model smoke has passed | Missing |
| JSON mode / JSON Schema replacement is complete | Real backend has JSON root/start and first required-key prefix guidance | Partial |
| Full JSON Schema grammar/DFA constrained decoding is available | `ResponseFormatPlan` extracts schema constraints and `StructuredOutputValidator` post-validates generated JSON Schema outputs for required keys, types, enums, arrays, and `additionalProperties:false`; no full grammar/DFA token-state engine is implemented | Missing |
| Python `mlx-vlm` model-family parity is complete | Gemma4 real smoke passes; Qwen VL planning/metadata is broad; non-generative families are classified | Partial |
| RF-DETR/SAM3 and other non-generative model actual inference is Swift-backed | Capability planning prevents incorrect text generation, but actual predictor inference is not ported | Missing |
| SentencePiece/Unigram fallback tokenizers are Swift-executable without upstream tokenizer | Unigram `tokenizer.json` and binary SentencePiece `tokenizer.model` greedy fallbacks are covered by `scripts/verify_swift_port.sh`; non-Unigram SentencePiece variants still need explicit support or upstream delegation | Partial |

## Current Gates

- `swift build --disable-sandbox --jobs 18`
- `swift run mlx-vlm-swift self-test`
- `SWIFT_BUILD_JOBS=18 scripts/verify_swift_port.sh`
- `SWIFT_BUILD_JOBS=18 scripts/verify_mock_real_mlx_api.sh`
- `SWIFT_BUILD_JOBS=18 scripts/verify_real_gemma4_smoke.sh`
- `MLXVLM_REAL_QWEN25VL_MODEL=/path/to/qwen2.5-vl SWIFT_BUILD_JOBS=18 scripts/verify_real_qwen25vl_video_smoke.sh`, or `MLXVLM_REAL_QWEN25VL_ALLOW_REMOTE=1 SWIFT_BUILD_JOBS=18 scripts/verify_real_qwen25vl_video_smoke.sh` when remote Hugging Face downloads are acceptable

These gates are necessary but not sufficient for 100% completion because the video-capable real model gate still depends on a local Qwen2.5-VL directory or an explicit remote-download run, and the set still lacks full JSON Schema grammar decoding and non-generative model inference.

## Next Required Work

1. Add a real video-capable MLX model fixture or documented local model path and pass a video generation smoke through the Swift server.
2. Implement full JSON Schema grammar/DFA constrained decoding or adopt an official upstream equivalent using the extracted `JSONSchemaConstraintPlan`, then verify schema properties, enums, arrays, required keys, nesting, and streaming behavior.
3. Expand SentencePiece coverage beyond Unigram greedy fallback where needed, or explicitly delegate unsupported SentencePiece variants to the upstream tokenizer integrations.
4. Decide whether RF-DETR/SAM3 actual inference is in scope for this replacement; if yes, add Swift-backed predictor routes or explicit compatibility exclusions.
5. Expand real-model smoke beyond Gemma4 to at least one Qwen/Qwen2.5-VL model once a local MLX model directory is available.
