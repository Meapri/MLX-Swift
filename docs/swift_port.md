# MLX-VLM Swift Compatibility Map

This repository is now a Swift-first replacement for the Python `mlx-vlm` runtime. The Python package has been removed from the working tree; compatibility work is tracked here so local MLX model directories and Ollama/OpenAI client contracts continue to behave like the Python server while inference runs through Swift.

## Compatibility Contract

The Swift port must keep these inputs compatible with the Python package:

- Local MLX model directories containing `config.json`, tokenizer files, processor files, and `*.safetensors` weights.
- Sharded MLX weight directories containing `model.safetensors.index.json`.
- `generation_config.json` values that Python merges into model config, starting with `eos_token_id`.
- Hugging Face/MLX model metadata fields such as `model_type`, `text_config`, `llm_config`, `vision_config`, `audio_config`, `quantization`, and `dflash_config`.
- Python `MODEL_REMAPPING` behavior from `mlx_vlm/utils.py`.
- Server request shapes for `/chat/completions`, `/v1/chat/completions`, `/completions`, `/v1/completions`, `/responses`, `/v1/responses`, `/generate`, `/api/generate`, and `/api/chat`.
- Ollama/OpenAI discovery, embedding, tokenize/detokenize, model-show, residency, cache, and unload endpoints used by local clients.

Remote Hugging Face downloads stay out of the dependency-free Swift target. Hugging Face-style model IDs are resolved from an existing local HF cache (`HUGGINGFACE_HUB_CACHE`, `HF_HOME`, `TRANSFORMERS_CACHE`, or the default `~/.cache/huggingface/hub` layout) by default, and real MLX builds can opt into upstream `MLXHuggingFace` / `Downloader` downloads with `MLXVLM_ENABLE_HUGGINGFACE_DOWNLOADER=1`.

## Phase 0: Swift Compatibility Runtime

Implemented now:

- `Package.swift` defines a Swift package with `MLXVLMCore` and the `mlx-vlm-swift` executable.
- `HuggingFaceCacheResolver` maps cached `org/repo` model identifiers to local `models--org--repo/snapshots/<revision>` directories without performing network downloads.
- `MLXVLMCore` reads local `config.json`, detects safetensors, parses safetensors headers, checks tokenizer/chat-template presence, quantization metadata, text/vision/audio configs, and canonical model type.
- `ModelConfigNormalizationPlan` and normalized config loading mirror the Python loader's config handoff by popping `llm_config`, using it as `text_config` when needed, and supplying empty text/vision/audio config dictionaries for backend loading.
- `QuantizationMetadata` preserves raw `quantization` / `quantization_config` objects and normalizes common fields such as mode, bits, group size, strategy, format, and symmetry for backend loading and Ollama model info.
- `AdapterMetadata` detects Python mlx-vlm LoRA adapter directories/files (`adapter_config.json` plus `adapters.safetensors`) separately from base model weights, preserving rank/alpha/dropout and safetensors readability for the real upstream adapter bridge.
- `MLXVLMCore` merges `generation_config.json` `eos_token_id` into the model config like Python `load_config()`.
- `ProcessorMetadata` reads `processor_config.json`, `preprocessor_config.json`, and `video_preprocessor_config.json`, including Qwen-style `image_processor` overrides and size/min/max pixel settings.
- `MLXVLMCore` reads `model.safetensors.index.json` weight maps, validates referenced shard names, and summarizes per-shard dtype/tensor-count metadata.
- `TokenizerMetadata` reads tokenizer files, sidecar tokenizer assets (`tokenizer.tiktoken`, `vocab.json`, `merges.txt`, `vocab.txt`), chat template overrides (`chat_template.json`, `chat_template.jinja`, then `tokenizer_config.json`), Qwen VL special token IDs from `added_tokens_decoder`, Gemma4-style media/turn token IDs from `tokenizer.json` added tokens plus top-level `config.json`, and `tokenizer.json` summaries including tokenizer model type, vocab count, merge count, added-token count, and normalizer/pre-tokenizer/decoder type.
- `ChatTemplatePlan` reports whether the discovered chat template can use the native Swift prompt renderers (`qwen-chat`/plain fallback plus Gemma4, Llama 3, and Mistral coverage) or should be left to upstream tokenizer/template handling.
- The native prompt renderer now covers Gemma4 `<|turn>` / `<turn|>` chat templates, including system thinking prelude, media placeholders, assistant reasoning channels, Gemma4 tool-call markup, and tool response blocks for compatibility/preflight paths without invoking Python or a Jinja runtime.
- `TokenizerCatalog` maps `tokenizer.json` vocab, added tokens, and BPE merge ranks into deterministic token/id records for backend token lookup preflight.
- `TokenizerImplementationPlan` classifies the required tokenizer execution backend (`tokenizers-json-bpe`, SentencePiece/tokenizer model, `.tiktoken`, `vocab.json`+`merges.txt`, WordPiece `vocab.txt`, etc.), records normalizer/pre-tokenizer/decoder components, and makes the current full-tokenizer implementation gap explicit in model-load plans.
- `SimpleTokenizer` now executes local `tokenizer.json` WordLevel and common ByteLevel BPE vocabularies, including GPT-2/Hugging Face byte-to-Unicode mapping and a matching ByteLevel detokenizer for backend smoke tests.
- `GenerationTokenTextDecoder` turns sampled token IDs into incremental text deltas by reusing the simple detokenizer and tracking the decoded suffix, giving the local diagnostic decode loop a stable token-ID-to-chunk-text bridge.
- `TokenizerCatalogBuilder` also builds catalogs from sidecar `vocab.json` + `merges.txt`, `tokenizer.tiktoken`, and WordPiece `vocab.txt` tokenizers, using `tokenizer_config.json` to recover unknown and added special tokens when `tokenizer.json` is absent.
- Tokenizer plans now distinguish the required backend from dependency-free Swift execution support, so supported WordLevel and ByteLevel BPE tokenizers no longer appear as full-tokenizer blockers.
- Generation preflight now feeds supported `SimpleTokenizer` token IDs into the backend handoff instead of relying only on greedy catalog token matching.
- `SimpleTokenizer` executes dependency-free WordLevel `tokenizer.json`, WordPiece `tokenizer.json`/`vocab.txt`, `tokenizer.tiktoken` exact-piece catalogs, and limited catalog-backed BPE tokenization for whitespace-separated text with exact special-token matching and unknown-token fallback; production Unigram and SentencePiece remain classified for backend implementation.
- `TokenizationPreflight` greedily maps known catalog tokens in rendered prompts and flags unknown spans that still require the real tokenizer implementation.
- `ModelRegistry` mirrors Python's `MODEL_REMAPPING` and current model implementation directory names.
- `ModelCapabilityPlan` distinguishes generative VLMs from mlx-vlm custom predictor families such as RF-DETR and SAM3, so Ollama/OpenAI text generation compatibility is not assumed for non-generative detection/segmentation models.
- `QwenVLConfig` ports the config defaults and validation rules for `qwen2_vl` and `qwen2_5_vl`.
- `QwenVLArchitecturePlan` derives Qwen VL text/vision layer counts from config and checks coverage for core sanitized language/vision tensors before MLX arrays are loaded.
- `QwenVLProcessor` ports Qwen VL image/video placeholder expansion based on `image_grid_thw` and `video_grid_thw`.
- `QwenVLImageGrid` ports Qwen VL smart resize and image/video grid planning used to produce `image_grid_thw`/`video_grid_thw` and placeholder token counts.
- `QwenVLImageInputPlanner` decodes local/data-URI/base64 image dimensions with ImageIO and produces image grid/placeholder plans for normalized multimodal requests before MLX tensor conversion is wired.
- `QwenVLImagePixelPreflight` decodes loadable local/data-URI/base64 images with ImageIO, applies the Qwen resize plan, draws the resized image into an RGB-compatible buffer, applies Qwen patchification and normalization ordering, and reports RGB/patch tensor shapes, byte counts, and patch value stats without serializing image bytes.
- `QwenVLEmbeddingMerge` ports the placement plan behind Qwen VL `merge_input_ids_with_image_features`, validating image/video token counts before the MLX array write is implemented.
- `QwenVLPromptBuilder` ports the Python plain-prompt fallback for multimodal content, replacing image/video/audio payloads with model markers so base64 media is never tokenized as text.
- Native chat rendering now covers Qwen chat markers, Llama 3 header/eot templates, and Mistral `[INST]` templates without requiring a full Jinja runtime; unknown custom Jinja templates still fall back explicitly.
- `QwenVLPromptBuilder` also provides a Qwen chat prompt style with `<|im_start|>role` / `<|im_end|>` blocks for Qwen-family request summaries and backend preparation.
- `QwenVLWeights` mirrors Qwen VL Python `sanitize()` weight-key remapping from `visual.*`, `model.*`, and `lm_head.*` into Swift-side module prefixes.
- `WeightCatalog` maps safetensors header entries to backend-ready tensor records with original key, sanitized Swift key, shard, dtype, shape, offsets, index coverage, and Qwen VL role.
- `QwenVLWeightSanitizer` mirrors the Python Qwen2/Qwen2.5-VL key rewrite for `visual`, `model`, and `lm_head` prefixes and exposes single-key preflight for regression checks.
- `WeightDataCatalog` resolves safetensors tensor payload byte ranges, dtype byte widths, expected byte counts, and small explicit payload previews, including numeric decoding for common safetensors dtypes, for backend load validation without dumping model weights by default.
- `WeightTensorPayload` provides an exact byte-range reader for one safetensors tensor payload so MLX loading diagnostics and fallback backend work share a validated source for array allocation.
- `MLXWeightLoadPlan` maps readable safetensors payload handles to backend-facing MLX dtype names, loadability flags, byte totals, and unsupported dtype blockers before real MLX array allocation is implemented.
- `MLXWeightPreparer` turns selected load-plan entries into exact tensor payload bundles with dtype, shape, shard, checksum, and byte-limit guards. This is the last dependency-free handoff before the real backend converts payload bytes into `MLXArray` values.
- `WeightCatalog` reports duplicate original and sanitized tensor keys across shards so backend loading can reject ambiguous safetensors layouts before allocating arrays.
- `ModelMemoryEstimator` combines safetensors payload bytes, Qwen config, quantization metadata, KV-cache settings, and vision-cache settings into a conservative runtime memory estimate for backend planning.
- `APICompatibility` normalizes Ollama `/api/generate`, Ollama `/api/chat`, OpenAI `/v1/completions`, OpenAI `/v1/chat/completions`, and OpenAI `/v1/responses` request bodies into the shared `GenerationRequest` contract, preserving common generation controls such as max tokens, temperature, top-p, top-k, min-p, typical-p, tail-free sampling, seed, context length, repeat/presence/frequency penalties, Mirostat, newline penalty, KV-cache quantization preferences, vision-cache size, activation-quantization preference, stop sequences, Ollama `keep_alive`, and streaming intent.
- `GenerationRequestMetadata` preserves request-shape details that are not sampling knobs, including Ollama `format`, `raw`, `template`, `suffix`, legacy `context`, OpenAI response format metadata, and Ollama/OpenAI tool-calling metadata.
- `GenerationRequestMetadata` also preserves raw Ollama `options` so backend-relevant settings that the compatibility runtime does not execute directly, such as GPU/thread/mmap hints, remain available to Swift backend layers.
- OpenAI chat and Responses normalization preserves mlx-vlm server extensions such as `max_completion_tokens`, `top_k`, `min_p`, `repetition_penalty`, presence/frequency penalties, `logit_bias`, thinking controls, logprob controls, resize shape, adapter path, draft model fields, `n`, `stream_options`, `modalities`, `audio`, `prediction`, and user identifiers for backend handoff. The upstream Swift VLM bridge now forwards request resize shape through `UserInput.Processing` and forwards thinking, `logit_bias`, logprob, `tool_choice`, user, audio, video, draft, and response-shape kwargs through `UserInput.additionalContext` where upstream tokenizers/processors can consume them; `logit_bias` token-id keys are normalized the same way Python converts JSON object keys to integers.
- Chat message normalization preserves Python server message fields such as assistant reasoning, tool calls, tool-call IDs, function names, `output_text` content parts, nested `input_audio` payloads including `format`, Python `mlx-vlm` placeholder content items (`{"type":"image"}` / `{"type":"audio"}`), Python image references such as `{"type":"image","image":"..."}` plus image `detail`/`resized_height`/`resized_width`/`min_pixels`/`max_pixels` options, and Python `mlx-vlm` video content items with `video`, `video_url`, or `input_video` payload keys plus `min_pixels`/`max_pixels`/`fps`/`nframes`/`min_frames`/`max_frames` metadata.
- `OllamaPromptRenderer` applies common Ollama generate prompt controls during preflight, including raw prompt passthrough, suffix appending, and direct Go-template variables such as `{{ .System }}`, `{{ .Prompt }}`, `{{ .Suffix }}`, and `{{ .Response }}`.
- `EmbeddingRequest` normalizes Ollama `/api/embed`, legacy Ollama `/api/embeddings`, and OpenAI `/v1/embeddings` payloads into a shared embedding contract. Real MLX builds can load upstream `MLXEmbedders` for embedding-model directories and render Ollama/OpenAI embedding envelopes; VLM-only models return typed `501` unavailable reports with the explicit `diagnostic-501-no-generated-embedding` fallback policy and the embedding backend load/unavailable reason.
- Ollama model-management operations (`/api/create`, `/api/pull`, `/api/push`, `/api/copy`, `/api/delete`) are handled in Swift: pull resolves local or cached Hugging Face MLX descriptors, create/copy register server-local model aliases, push validates the referenced model descriptor, and delete unregisters aliases without destructively deleting model files.
- Ollama show requests accept the normal POST body shape with `model`/`name` and `verbose`, validate model mismatches, and still support simple GET `/api/show` for local inspection.
- Ollama blob operations are recognized at `/api/blobs/{digest}`: `HEAD` returns 404 or 200 based on server-local blob registration, and `POST` accepts a digest payload into the Swift compatibility server's in-memory blob registry.
- `StopSequenceMatcher` applies normalized stop sequences consistently to completed text and provides a small stream filter that withholds enough suffix text to catch stop sequences crossing chunk boundaries.
- Server-side default generation parameters can be supplied by CLI flags and are used only when a request omits that parameter, preserving per-request Ollama/OpenAI override behavior, including Ollama `num_ctx` and `keep_alive`.
- `MediaReferenceResolver` summarizes Ollama raw base64 images, OpenAI data URIs, local media paths, file URLs, common image/audio/video file extensions, and remote URLs without downloading remote data or echoing payloads back into prompts.
- `APIResponses` defines Ollama generate/chat, OpenAI chat completion, legacy OpenAI completions, OpenAI Responses, OpenAI model list/retrieve, streaming chunk response shapes, and JSONL/SSE stream framing helpers for Swift backend output.
- `GenerationOutputAssembler` collects backend token chunks into `CompletedGeneration`, applies streaming stop-sequence filtering, preserves finish reasons such as `stop`, `length`, and `tool_calls`, carries backend tool-call chunks forward, and feeds usage counts into Ollama/OpenAI response formatters. When a real backend emits final prompt/completion counts, those values override dependency-free chunk-count estimates.
- The shared generation contract now carries optional per-token logprob payloads from backend chunks through `CompletedGeneration` and renders them in OpenAI Chat-compatible `choices[].logprobs.content[]` and streaming `choices[].logprobs` shapes. Python `mlx-vlm` can compute these in its custom token loop; current upstream `mlx-swift-lm` public `Generation` events do not expose token logprobs yet, so the Swift response path is ready for a backend that supplies them while request metadata is still forwarded to upstream.
- `ThinkingOutputSplitter` mirrors Python server thinking-output cleanup for OpenAI response envelopes, splitting Qwen-style `<think>...</think>` and Gemma4-style `<|channel>thought...<channel|>` output into `reasoning` plus visible assistant content. Non-streaming OpenAI Chat/Responses use the completed-output splitter, and OpenAI Chat SSE uses a stateful streaming splitter so tag fragments are suppressed while reasoning deltas stream separately from content deltas. Ollama rendering still keeps the raw generated text.
- OpenAI Responses streaming now routes through the real generation backend for `/responses` and `/v1/responses` when `stream:true` is set. It emits Python-compatible named SSE events including `response.created`, `response.in_progress`, `response.output_item.added`, `response.content_part.added`, `response.output_text.delta`, `response.output_text.done`, `response.content_part.done`, `response.output_item.done`, and `response.completed`; deltas preserve raw generated text while final done/completed events use thinking-cleaned output text.
- `GenerationDecodeLoop` connects sampled token IDs plus detokenized text to the shared output assembler, enforcing EOS, stop-sequence, and max-completion-token termination while preserving usage counts for API response formatters.
- `VLMGenerationRunner` wraps backend token streams with the shared assembler, so backend implementations only need to emit `GenerationChunk` values while Core handles stop filtering, prompt/completion usage, final `CompletedGeneration` construction, and rendered Ollama/OpenAI response handoff.
- `BackendStatus` exposes both the dependency-free compatibility shell and the `mlx-swift-vlm` runtime status used when the real upstream MLX Swift VLM backend is linked and loaded.
- `BackendDependencyPlan` inspects the Swift package and common local vendor/checkouts locations to report whether `mlx-swift`, `mlx-swift-lm`, `swift-tokenizers-mlx`, a backend target, the optional MLX/tokenizer backend manifest toggles, and local dependency switches are available without breaking dependency-free compatibility builds.
- `Package.swift` keeps the default compatibility build dependency-free, but supports `MLXVLM_ENABLE_MLX_BACKEND=1` to attach `MLX`, `MLXLMCommon`, `MLXLLM`, and `MLXVLM` products, `MLXVLM_ENABLE_TOKENIZER_INTEGRATIONS=1` to attach `MLXLMTokenizers` once `swift-tokenizers-mlx` can be fetched or vendored, and `MLXVLM_ENABLE_HUGGINGFACE_DOWNLOADER=1` to attach upstream `MLXHuggingFace` plus `swift-huggingface` for remote model ID downloads.
- Network-restricted backend builds can set `MLXVLM_USE_LOCAL_MLX=1` to prefer local `vendor/`, `Vendor/`, `Dependencies/`, `.build/checkouts/`, or sibling checkouts, or set `MLXVLM_MLX_SWIFT_PATH`, `MLXVLM_MLX_SWIFT_LM_PATH`, and `MLXVLM_SWIFT_TOKENIZERS_MLX_PATH` explicitly. Because this workspace path has SwiftPM identity `mlx-swift`, alias directories such as `vendor/MLXSwift`, `vendor/MLXSwiftLM`, and `vendor/SwiftTokenizersMLX` are checked first to avoid colliding with upstream dependency identities.
- `MLXRuntimeProbe` lives inside the `MLXVLMMLXBackend` target and uses compile-time `canImport(MLX)`, `canImport(MLXLMCommon)`, `canImport(MLXLLM)`, `canImport(MLXVLM)`, and `canImport(MLXLMTokenizers)` checks so backend availability reports distinguish manifest readiness, module import readiness, and whether the guarded upstream real backend is compiled.
- `MLXArrayWeightLoader` contains the first real MLX API bridge behind `MLXVLM_ENABLE_REAL_MLX_API=1`: when real `mlx-swift` is available, it uses `MLXArray(data, shape, dtype:)` to convert prepared safetensors payload bytes into typed arrays. The flag is separate from `MLXVLM_ENABLE_MLX_BACKEND=1` so mock import checks keep validating manifest wiring without pretending the real API exists.
- `MLXBackendLoadPreflight` lives in the backend target and combines dependency availability, Qwen VL tensor binding coverage, and bounded `MLXWeightPreparer` payload reads into one report. Dependency-free builds still report the missing array/module/generation pieces honestly, while real builds can route generation through the guarded upstream backend.
- `MLXWeightBackedModelContainer` is the backend-side container boundary after metadata and weight payload loading. It exposes a summary of prepared tensors, loaded arrays, and local fallback module/generation blockers while the primary real path uses upstream `mlx-swift-lm` containers.
- `MLXQwenVLModuleConstructionPlan` maps loaded array keys and expected Qwen VL tensor bindings into backend module groups such as token embedding, language layers, vision patch embedding, vision blocks, and vision merger. It reports group-level missing keys, shape mismatches, loaded array counts, and whether a group is constructible.
- `MLXQwenVLForwardPlan` combines the backend container, module construction plan, and processed generation input to derive input ID shape, optional pixel shape, logits shape, tokenizer blockers, and next-token readiness before real MLX forward kernels are wired.
- `MLXQwenVLGenerationLoopPlan` maps the forward contract and runtime sampler settings into the expected prefill/decode loop, sampler feature requirements, stop/stream handling, upstream `mlx-swift-lm` `GenerateParameters`, and assembler handoff readiness.
- `MLXGenerateParametersBridge` conditionally instantiates real `MLXLMCommon.GenerateParameters` when `MLXVLM_ENABLE_REAL_MLX_API=1` and the MLX Swift LM products are importable; otherwise it reports the missing bridge dependency without breaking default builds.
- `MLXVLMModelFactoryBridge` conditionally references the upstream `MLXVLM.VLMModelFactory`, `MLXLMCommon.ModelContainer`, and `MLXLMTokenizers.TokenizersLoader` entrypoints that load local VLM directories and run `prepare`/`generate`.
- `MLXMetalLibrarySupport` prepares the MLX Metal runtime for SwiftPM debug executables by checking `MLXVLM_MLX_METALLIB_PATH`, common local Python `mlx` install locations, executable-adjacent paths, and SwiftPM bundle paths, then installing `mlx.metallib` as `default.metallib` before the real backend loads. If one destination is already populated, server startup can reuse it as the source and fill missing executable or bundle destinations. Its runtime package report also exposes the required build flags and runtime files for release/preflight checks.
- `MLXVLMUpstreamBackend` is the first real generation backend. When built with `MLXVLM_ENABLE_MLX_BACKEND=1`, `MLXVLM_ENABLE_TOKENIZER_INTEGRATIONS=1`, and `MLXVLM_ENABLE_REAL_MLX_API=1`, it loads a local model directory with `VLMModelFactory.shared.loadContainer(from:using:)`, converts normalized Swift chat requests into `MLXLMCommon.UserInput`, bridges OpenAI/Ollama-compatible tool schemas into upstream `ToolSpec` dictionaries, materializes OpenAI data URI and Ollama raw base64 media into temporary files for upstream `UserInput.Image.url`/`Video.url`, resolves ordinary relative local image/video paths before upstream handoff, maps runtime options into `GenerateParameters`, streams upstream `Generation.chunk` values into the shared `GenerationChunk` contract, maps upstream `Generation.toolCall` values into shared tool-call chunks, maps upstream `GenerateCompletionInfo` into exact prompt/completion token usage, and renders the existing Ollama/OpenAI response envelopes.
- When a request contains assistant `tool_calls`, tool `tool_call_id`, `name`, or reasoning metadata, the upstream bridge uses `UserInput(messages:)` raw dictionaries instead of lossy `Chat.Message` values so Jinja chat templates can still see Python server message metadata. Function-call `arguments` strings are normalized to JSON objects in Core before preflight/backend handoff, matching Python `mlx-vlm` server behavior for templates that iterate arguments.
- `ToolCallOutputParser` covers the Gemma4 Python tool-call format (`<|tool_call>call:name{key:<|"|>value<|"|>}<tool_call|>`) as a Swift fallback when upstream `mlx-swift-lm` does not emit parsed `Generation.toolCall` events for that newer parser family. It matches Python `mlx-vlm` parser behavior for hyphenated tool names, nested objects, arrays, escaped strings, and bare JSON-like literals. The upstream generator buffers Gemma4 tool-output text only for tool-enabled requests, converts parsed calls into shared `GenerationToolCall` chunks, and returns `finish_reason:"tool_calls"`. Audio/video request metadata is also preserved through `AudioReference`/`VideoReference`, media reports, raw-message `input_audio`/`video`, and upstream `additionalContext`.
- `MLXRemoteModelResolver` preserves the cache-only Core behavior but lets `mlx-vlm-swift serve --model org/repo` download an uncached Hugging Face model through upstream `VLMModelFactory.configuration(id:)`, `MLXLMCommon.resolve(configuration:from:)`, and `#hubDownloader(HubClient())` when the real backend is built with `MLXVLM_ENABLE_HUGGINGFACE_DOWNLOADER=1`.
- `MLXVLMAdapterBridge` applies per-request `adapter_path` in the real upstream backend. Native upstream adapter directories with `fine_tune_type` are loaded directly through `ModelAdapterFactory`, while Python `mlx-vlm` LoRA directories with `rank`/`alpha` and `adapters.safetensors` are converted into an upstream-compatible temporary adapter config, with Python `A`/`B` safetensors keys remapped to upstream `lora_a`/`lora_b` keys and `alpha / rank` used as the upstream scale.
- `MLXVLMMLXBackend` remains a separate availability-gated target, so dependency-free compatibility builds keep working until MLX Swift dependencies are vendored or fetchable.
- `ModelCompatibilityReport` validates a local model directory against the Swift compatibility shell and separates metadata readiness from real generation readiness, including safetensors payload byte-range checks, dtype/shape byte-count checks, MLX dtype loadability, index `total_size` consistency, chat-template renderer coverage, and Qwen VL core tensor coverage warnings.
- `GenerationPreflightPlanner` assembles the backend handoff: prompt style, rendered prompt, media summaries, processor-aware image grid plans, RGB pixel preflight, generation parameters, compatibility checks, and blocking reasons.
- `GenerationRuntimePlan` resolves context-window intent, model max-position limits, prompt/completion token budget, cache intent, KV-cache quantization preferences, vision-cache size, activation-quantization preference, stop sequences, streaming mode, and keep-alive into a backend-facing plan.
- `GenerationSamplingPlan` classifies Ollama/OpenAI sampling controls into greedy, temperature, or Mirostat sampler intent, records enabled probability filters and penalties, and now only flags Mirostat as requiring a separate advanced sampler.
- `GenerationLogitsSampler` provides a dependency-free Swift sampler for backend logits vectors, covering greedy argmax, temperature sampling, top-k, top-p, min-p, typical-p, tail-free sampling, logit-bias application, seeded randomness, repetition/presence/frequency penalties, and newline penalty handoff when a newline token ID is known.
- `GenerationLogitsDecodeExecutor` composes logits sampling, incremental token text decoding, and the decode loop into one dependency-free execution path for fallback backend experiments and deterministic diagnostics.
- `GenerationAPIResponseRenderer` turns completed generation output or stream chunks into Ollama/OpenAI-compatible JSON, NDJSON, or SSE bodies with explicit content-type metadata.
- `GenerationEndpointRenderer` collects backend generation chunks into completed text/token usage and routes them through the Ollama/OpenAI response renderer, so Swift generators can hand off token deltas without duplicating API response code.
- `ResponseFormatPlan` classifies Ollama/OpenAI structured-output requests, including JSON mode, JSON Schema guidance, missing schema warnings, and stream framing requirements.
- `MLXVLMUpstreamBackend` adds real-backend structured-output guidance for JSON mode and JSON Schema roots, including root-token masking and first required-key prefix guidance when a simple object schema is present.
- `ToolCallPlan` classifies OpenAI/Ollama-compatible tool requests, tool choice mode, forced function names, inferred parser hints, and backend requirements for tool-schema prompting and output parsing.
- `ModelLoadPlanner` assembles the model-level backend handoff: descriptor, Qwen VL config, tokenizer catalog, chat-template plan, weight catalog, weight data catalog, compatibility checks, and load/generation blockers.
- `MLXBackendBindingPlan` turns the Qwen VL architecture coverage into backend-oriented tensor binding phases such as token embedding, language layers, vision patch embedding, vision blocks, and vision merger, so missing/mismatched safetensors are visible before allocating MLX arrays.
- `BackendContracts` defines the Swift backend boundary for `ModelContainer`, `VLMProcessor`, `VLMModel`, `VLMGenerator`, and `VLMBackend`, plus compatibility-shell implementations that return typed unavailable errors instead of generated text and carry the runtime plan into `ProcessedGenerationInput`.
- `CompatibilityGenerationEngine` returns a typed `GenerationUnavailableReport` for generation endpoints while the real MLX Swift backend is absent.
- `mlx-vlm-swift inspect --model /path/to/model` prints a JSON descriptor.
- `mlx-vlm-swift inspect-qwen-vl --model /path/to/model` parses Qwen2-VL/Qwen2.5-VL config with Python-compatible defaults.
- `mlx-vlm-swift inspect-qwen-vl-weights --model /path/to/model` reports sanitized Qwen VL weight keys and their language/vision roles.
- `mlx-vlm-swift sanitize-qwen-vl-key --key ...` applies the Python-compatible Qwen VL weight-key rewrite to one key and reports its backend role.
- `mlx-vlm-swift inspect-qwen-vl-architecture --model /path/to/model` reports Qwen VL module counts and core tensor coverage from the sanitized weight catalog.
- `mlx-vlm-swift inspect-weight-catalog --model /path/to/model` reports tensor-level safetensors catalog entries for backend loading.
- `mlx-vlm-swift inspect-weight-data --model /path/to/model` validates tensor payload byte ranges and dtype/shape byte counts for backend loading.
- `mlx-vlm-swift plan-mlx-weight-load --model /path/to/model` reports which safetensors tensors can be loaded as MLX arrays, including safetensors-to-MLX dtype mapping and unsupported dtype blockers.
- `mlx-vlm-swift preview-weight-tensor --model /path/to/model --tensor <name> --bytes 32` reads a small hex prefix and decoded numeric values from one tensor payload to verify safetensors seek offsets and dtype handling.
- `mlx-vlm-swift read-weight-tensor-payload --model /path/to/model --tensor <name>` reads the full validated tensor payload and reports byte count/checksum without dumping the bytes.
- `mlx-vlm-swift inspect-safetensors --model /path/to/model` prints safetensors header readability, dtype, declared data byte, and tensor-count summaries.
- `mlx-vlm-swift inspect-tokenizer-catalog --model /path/to/model` prints tokenizer token/id records from `tokenizer.json`.
- `mlx-vlm-swift inspect-tokenizer-plan --model /path/to/model` prints the required tokenizer backend and whether catalog-only preflight can run.
- `mlx-vlm-swift inspect-processor --model /path/to/model` prints processor/preprocessor metadata used for image/video planning.
- `mlx-vlm-swift inspect-config-normalization --model /path/to/model` prints the Python-compatible config normalization plan for backend loading.
- `mlx-vlm-swift inspect-normalized-config --model /path/to/model` prints the backend-facing `config.json` shape after Python-compatible normalization.
- `mlx-vlm-swift preflight-tokenize --model /path/to/model --text ...` maps known catalog tokens and reports whether full tokenizer execution is still required.
- `mlx-vlm-swift tokenize-simple --model /path/to/model --text ...` runs the dependency-free WordLevel or limited catalog-backed BPE tokenizer path when supported and reports unsupported tokenizer families otherwise.
- `mlx-vlm-swift decode-token-stream --model /path/to/model --ids ...` simulates incremental token-ID detokenization and reports text deltas plus skipped special-token and unknown-token state.
- `mlx-vlm-swift plan-qwen-vl-merge --input-ids 1,151655,2 --feature-count 1` reports where vision feature rows would be inserted into input embeddings.
- `mlx-vlm-swift format-qwen-vl-prompt --api openai-chat --json ...` formats normalized chat requests into the Qwen VL plain prompt fallback.
- `mlx-vlm-swift format-qwen-vl-prompt --style qwen-chat --api openai-chat --json ...` formats normalized chat requests with Qwen chat block markers.
- `mlx-vlm-swift format-qwen-vl-prompt --api openai-responses --json ...` formats normalized Responses API input into the same prompt contract.
- `mlx-vlm-swift inspect-media --api ollama-generate --json ...` validates media references and reports source type, byte count, MIME guess, and loadability.
- `mlx-vlm-swift plan-qwen-vl-image-grid --height 224 --width 224` reports smart-resized dimensions, patch grid, and placeholder token count.
- `mlx-vlm-swift plan-qwen-vl-images --api openai-chat --json ...` resolves image references from a normalized API request and reports per-image Qwen VL grids and total placeholder token count.
- `mlx-vlm-swift plan-qwen-vl-pixels --api openai-chat --json ...` verifies image decode/resize/patchification viability and reports backend RGB plus Qwen patch tensor allocation shape, byte counts, and value stats.
- `mlx-vlm-swift expand-qwen-vl-placeholders --text ... --image-grid t,h,w` expands Qwen VL image/video placeholders the same way the Python processor does before tokenization.
- `mlx-vlm-swift apply-stop-sequences --text ... --stop END` reports the text truncation and matched stop sequence a backend should apply after generation.
- `mlx-vlm-swift list-supported` prints canonical model types known to this fork.
- `mlx-vlm-swift backend-status` prints whether real MLX Swift generation is linked.
- `mlx-vlm-swift backend-plan` reports package dependency readiness for enabling a separate MLX Swift backend target.
- `mlx-vlm-swift backend-availability` reports both package dependency readiness and compile-time MLX/MLXLMCommon/MLXLLM/MLXVLM module import readiness from the backend target.
- `mlx-vlm-swift preflight-mlx-backend-load --model /path/to/model [--tensor key] [--max-tensors n] [--max-total-bytes n] [--skip-weight-payloads]` runs the backend-target load preflight with bounded payload reads and reports the exact remaining blockers before real inference can run.
- `mlx-vlm-swift inspect-mlx-container --model /path/to/model [--tensor key] [--max-tensors n] [--max-total-bytes n] [--skip-weight-payloads]` prints the current `MLXWeightBackedModelContainer` summary, including whether arrays are loaded and whether module/generation construction can proceed.
- `mlx-vlm-swift inspect-mlx-module-plan --model /path/to/model [--tensor key] [--max-tensors n] [--max-total-bytes n] [--skip-weight-payloads]` prints the Qwen VL module construction plan derived from loaded arrays and binding coverage.
- `mlx-vlm-swift inspect-mlx-forward-plan --model /path/to/model --api openai-chat --json ... [--max-tensors n] [--max-total-bytes n] [--skip-weight-payloads]` prints the backend forward contract, including token/pixel input shape, logits shape, tokenizer readiness, and forward blockers.
- `mlx-vlm-swift inspect-mlx-generation-loop --model /path/to/model --api openai-chat --json ... [--max-tensors n] [--max-total-bytes n] [--skip-weight-payloads]` prints the prefill/decode/sampler/output-assembler contract for real generation.
- `mlx-vlm-swift inspect-mlx-decode-state --model /path/to/model --api openai-chat --json ... [--max-tensors n] [--max-total-bytes n] [--skip-weight-payloads]` prints the decode-state contract for KV-cache policy, next-token input shape, EOS IDs, stop-sequence handling, stream mode, and max-token termination.
- `mlx-vlm-swift inspect-mlx-generate-parameters --model /path/to/model --api openai-chat --json ...` maps normalized Ollama/OpenAI runtime options onto the upstream `mlx-swift-lm` `GenerateParameters` fields and reports options that still need separate backend support.
- `mlx-vlm-swift inspect-mlx-generate-parameters-bridge --model /path/to/model --api openai-chat --json ...` reports whether those mapped fields can be materialized as real `MLXLMCommon.GenerateParameters` in the current build.
- `mlx-vlm-swift inspect-mlx-model-factory-bridge --model /path/to/model` reports whether the current build can see upstream `MLXVLM.VLMModelFactory`, `MLXLMCommon.ModelContainer`, and generation entrypoint types for local model loading.
- `mlx-vlm-swift inspect-mlx-metal-library` reports the `default.metallib` locations checked by the real backend and candidate source paths without copying files.
- `mlx-vlm-swift inspect-runtime-package` combines backend dependency/import availability with `default.metallib` readiness, required real-backend build flags, required runtime files, and blocking reasons so release packages can be checked before starting the server.
- `mlx-vlm-swift inspect-mlx-pipeline --model /path/to/model --api openai-chat --json ... [--max-tensors n] [--max-total-bytes n] [--skip-weight-payloads]` prints the full MLX generation pipeline handoff report, combining load preflight, container summary, module construction, forward planning, `GenerateParameters` bridge readiness, upstream model-factory bridge readiness, decode-state planning, decode-loop readiness, request metadata, and blockers.
- `mlx-vlm-swift inspect-backend-context --model /path/to/model` prints the current `ModelLoadContext` object that a real MLX Swift backend will receive, including the normalized config and load plan.
- `mlx-vlm-swift inspect-ollama-show --model /path/to/model` prints the Ollama `/api/show` payload, including Swift backend readiness, normalized config keys, tokenizer/chat-template backend requirements, and current generation blockers in `model_info`.
- `mlx-vlm-swift preflight-ollama-show --json ...` validates Ollama `/api/show` request body decoding for `model`/`name` and `verbose`.
- `mlx-vlm-swift validate-model --model /path/to/model` prints compatibility checks for model metadata, config normalization, safetensors header readability, shard/index consistency, MLX weight loadability, tokenizer files, tokenizer JSON readability/vocab/catalog compatibility, chat-template renderer coverage, Qwen VL weight roles, and backend readiness.
- `mlx-vlm-swift plan-model-load --model /path/to/model` prints the complete model-load handoff plan for Swift backend execution.
- `mlx-vlm-swift plan-mlx-bindings --model /path/to/model` prints the backend tensor binding phases that the real MLX implementation must satisfy before generation can run.
- `mlx-vlm-swift prepare-mlx-weights --model /path/to/model [--tensor key] [--max-tensors n] [--max-total-bytes n] [--skip-weight-payloads]` reads exact safetensors payloads selected for MLX loading and prints a checksum summary without dumping raw weights. `--skip-weight-payloads` keeps diagnostics lightweight for large models and server settings UI readiness checks.
- `mlx-vlm-swift inspect-chat-template-plan --model /path/to/model` reports the selected chat template source and whether Swift can render it natively.
- `mlx-vlm-swift inspect-capabilities --model /path/to/model` reports the model's primary task and whether Ollama/OpenAI text generation endpoints are appropriate.
- `mlx-vlm-swift inspect-adapter --model /path/to/model` reports LoRA adapter metadata and whether `adapters.safetensors` is readable without mixing it into base model weight catalogs.
- `mlx-vlm-swift estimate-memory --model /path/to/model --context-length ...` prints estimated readable weight bytes, KV-cache bytes, vision-cache bytes, and assumptions for backend sizing.
- `mlx-vlm-swift preflight-generate --model /path/to/model --api openai-chat --json ...` prints the complete backend handoff plan and, in dependency-free builds, explicitly reports why generation cannot start yet; optional `--context-length` and `--keep-alive` flags set backend defaults.
- `mlx-vlm-swift preflight-embed --model /path/to/model --api ollama-embed --json ...` validates embedding request normalization and, in dependency-free builds, returns the typed unavailable embedding report.
- `mlx-vlm-swift preflight-model-operation --operation pull --json ...` previews the typed unavailable response used for Ollama model-management operations.
- `mlx-vlm-swift preflight-ollama-blob --digest sha256:...` previews the typed unavailable response used for Ollama blob upload/import operations.
- `mlx-vlm-swift sample-logits --logits 0.1,0.9,0.2 [--temperature n] [--top-k n] [--top-p n] [--min-p n] [--typical-p n] [--tfs-z n] [--seed n] [--recent-token-ids ids] [--logit-bias id:bias,...]` runs the pure Swift logits sampler used by local decode-loop diagnostics and fallback backend experiments.
- `mlx-vlm-swift simulate-decode-loop --model m --prompt-tokens n --max-tokens n --token id:text [--stop END] [--eos id]` runs the pure Swift decode-loop/output-assembler contract without requiring MLX logits.
- `mlx-vlm-swift simulate-logits-decode --model /path/to/model --logits ...` runs the composed logits sampler, incremental detokenizer, and decode-loop path against a local tokenizer without requiring MLX.
- `mlx-vlm-swift render-generation-response --api openai-chat --model m --text ok [--stream true] [--chunk o]` renders completed generation output or chunks into the target Ollama/OpenAI response envelope without requiring MLX logits.
- `mlx-vlm-swift render-generation-chunks --api openai-chat --model m --prompt-tokens n --chunk id:text [--stream true]` collects backend-style token chunks into completed text/usage and renders the target Ollama/OpenAI envelope.
- `mlx-vlm-swift serve --model /path/to/model --host 127.0.0.1 --port 11434` starts a server with `/health`, `/api/tags`, `/api/show`, `/models`, `/v1/models`, `/models/{id}`, and `/v1/models/{id}`; optional `--adapter-path`, `--draft-model`, `--draft-kind`, `--draft-block-size`, `--max-tokens`, `--temperature`, `--top-p`, `--top-k`, `--seed`, `--context-length`, `--keep-alive`, `--enable-thinking`, `--thinking-budget`, and `--thinking-start-token` flags set Python-server-style request defaults. Request-level `adapter_path`, draft, and thinking fields still override the server defaults. In a dependency-free build it returns typed generation diagnostics; in a real MLX build it loads `mlx-swift-vlm` and serves generated text through the same routes. With the Hugging Face downloader flag enabled, `--model org/repo` can download uncached models and `--use-latest` asks the upstream downloader to check for newer snapshots.
- For real MLX builds, `stream:true` generation requests on `/api/generate`, `/api/chat`, `/v1/completions`, `/v1/chat/completions`, and `/v1/responses` use socket-level streaming rather than a pre-buffered response body: the server writes NDJSON/SSE headers without `Content-Length`, then flushes each backend chunk frame as it arrives.
- The compatibility server normalizes query strings and trailing slashes before routing, so OpenAI clients can call paths such as `/v1/models?limit=20` and `/v1/models/{id}/` without falling through to a 404.
- `/health` preserves the Swift backend readiness fields and now also mirrors Python `mlx_vlm.server` fields such as `loaded_model`, `loaded_context_size`, `loaded_tool_parser`, `continuous_batching_enabled`, and `apc_enabled`.
- The server also exposes `/backend/status`, Ollama-style `/api/version`, `/api/ps`, `/api/unload`, Python-server-compatible `/unload`, APC cache diagnostics (`/v1/cache/stats`, `/cache/stats`, `/v1/cache/reset`, `/cache/reset`), Ollama/OpenAI embedding routes, tokenize/detokenize routes, Ollama model-management routes, and CORS `OPTIONS` responses for UI/client integration. If real generation is unavailable, generation `501` responses keep the existing top-level compatibility fields and add a compact `mlxPipeline` readiness summary for settings UI diagnostics. In real MLX builds, tokenize/detokenize routes use the upstream model tokenizer instead of the dependency-free `SimpleTokenizer` fallback, and `/health` also reports whether a separate embedding backend loaded for the selected model and includes the embedding backend unavailable reason when it did not.
- `OllamaModelResidency` tracks compatibility-server loaded/unloaded state so `/api/ps`, `/api/unload`, `/health`, and empty-prompt Ollama `/api/generate` requests with `keep_alive` reflect Ollama-like model residency even before real MLX arrays are resident.
- `mlx-vlm-settings` provides a small macOS SwiftUI settings window for choosing a local model directory, inspecting compatibility metadata, showing backend availability and Metal library readiness, setting server/generation defaults including context length, keep-alive, KV-cache quantization, max KV size, vision-cache size, activation quantization, and starting/stopping the compatibility server.
- `scripts/check_mlx_backend_dependencies.sh` checks for local `mlx-swift`, `mlx-swift-lm`, and `swift-tokenizers-mlx` checkouts, prints the backend plan when they are missing, and builds the `MLXVLMMLXBackend` target with local dependency environment variables when they are available.
- `scripts/verify_local_mlx_manifest.sh` creates temporary mock `MLX`, `MLXLMCommon`, `MLXLLM`, `MLXVLM`, and `MLXLMTokenizers` packages to verify the local dependency manifest path and compile-time `canImport` probe without requiring network access or real MLX packages.
- `scripts/verify_mock_real_mlx_api.sh` creates stricter mock `MLX`, `MLXLMCommon`, `MLXVLM`, and `MLXLMTokenizers` packages that expose `DType`, `MLXArray(data, shape, dtype:)`, `GenerateParameters`, `UserInput`, `ToolSpec`, `ToolCall`, `Generation`, `GenerateCompletionInfo`, `ModelContainer.prepare/generate`, `VLMModelFactory.loadContainer`, and `TokenizersLoader`, then builds with `MLXVLM_ENABLE_REAL_MLX_API=1` to exercise guarded upstream API bridges without downloading dependencies.
- `scripts/verify_real_gemma4_smoke.sh` is the local real-backend regression gate. When the Gemma4 E4B MLX model directory is present, it builds an identity-safe temporary copy with real upstream `mlx-swift`, `mlx-swift-lm`, and `swift-tokenizers-mlx`, starts the server, and verifies health, OpenAI chat, Swift model/blob management endpoints, tool-schema passthrough, actual Gemma4 tool-call generation parsed into OpenAI `tool_calls`, Ollama raw-base64 image input, OpenAI data-URI image input, OpenAI SSE streaming without `Content-Length`, Ollama NDJSON streaming without `Content-Length`, and the expected VLM embedding `501` diagnostic.

Generation endpoints now have two honest modes: dependency-free builds return typed `501 Not Implemented` compatibility reports, while real MLX builds route generation through upstream `mlx-swift-lm` instead of returning fake text.

## Phase 1: Official Swift Engine Adapter

The protocol boundary now exists in `BackendContracts`, and the primary real implementation path is `MLXVLMUpstreamBackend`. This path intentionally adopts upstream `mlx-swift-lm` as the Swift inference engine instead of reimplementing common LLM/VLM architectures in this repository. The fork-owned work is the Python `mlx-vlm` compatibility layer around that engine: model-directory inspection, Python-style metadata normalization, Ollama/OpenAI request adaptation, response rendering, settings UI, and diagnostics. Swift-owned model modules should be added only when Python `mlx-vlm` behavior or model coverage is missing upstream.

- `ModelContainer`: the upstream-backed path holds an `MLXLMCommon.ModelContainer` loaded by `MLXVLM.VLMModelFactory`, preserving compatibility with existing local MLX model directories.
- `VLMModel`: common multimodal forward is delegated to upstream `mlx-swift-lm` for covered architectures.
- `VLMProcessor`: this package still owns Python `mlx-vlm` request/media/preflight compatibility before handing normalized input to upstream `UserInput`.
- `VLMGenerator`: the upstream-backed generator emits real text/tool/info chunks; local native generation code remains a future fallback for uncovered Python `mlx-vlm` behavior.
- `MLXWeightPreparer`: dependency-free source for exact safetensors payload bytes; the backend should consume these summaries/payloads when creating typed `MLXArray` weights.

Use `mlx-swift`, `mlx-swift-lm`, and the tokenizer integration package `swift-tokenizers-mlx` as primary dependencies for real generation. Current upstream docs list MLX Swift `0.31.3`, MLX Swift LM `3.31.3`, and swift-tokenizers-mlx `0.3.0` as release anchors for this port.

Official reference adoption notes:

- `mlx-swift-lm` exposes remote model loading through the provider-agnostic `Downloader` protocol and `MLXHuggingFace` macros such as `#hubDownloader()` plus tokenizer-loader adapters. This project should adopt that path for remote Hugging Face model IDs instead of building a separate Python-style downloader.
- `ChatSession` is useful for a future stateful chat UX, but the current Ollama/OpenAI server path should remain stateless and continue using `ModelContainer.prepare`/`generate` directly so each API request maps cleanly to one normalized `GenerationRequest`.
- `UserInput.Image.url`, `UserInput.Video.url`, `ToolSpec`, `GenerateParameters`, and `Generation` are the upstream contracts this package should keep adapting to, rather than duplicating engine internals.

The current workspace does not vendor those packages. With network-restricted builds, the default package remains dependency-free. After local/vendor checkouts or network fetches are available, build with `MLXVLM_ENABLE_MLX_BACKEND=1 MLXVLM_ENABLE_TOKENIZER_INTEGRATIONS=1 swift build`. If network fetch is unavailable, prefer `vendor/MLXSwift`, `vendor/MLXSwiftLM`, and `vendor/SwiftTokenizersMLX` with `MLXVLM_USE_LOCAL_MLX=1`, or set explicit `MLXVLM_MLX_SWIFT_PATH` / `MLXVLM_MLX_SWIFT_LM_PATH` / `MLXVLM_SWIFT_TOKENIZERS_MLX_PATH` values.

After real dependencies are present, enable the guarded array and upstream VLM bridge with `MLXVLM_ENABLE_REAL_MLX_API=1` as well. That compiles `MLXArrayWeightLoader`, `MLXGenerateParametersBridge`, `MLXVLMModelFactoryBridge`, and `MLXVLMUpstreamBackend`, which are intentionally separated from basic dependency-free checks because those checks should not pretend the real MLX, MLXLMCommon, MLXVLM, or MLXLMTokenizers APIs exist. Add `MLXVLM_ENABLE_HUGGINGFACE_DOWNLOADER=1` only when the build can fetch `swift-huggingface` and should support remote `org/repo` downloads instead of local/cache-only resolution.

When running a SwiftPM debug executable directly, MLX may need `default.metallib` colocated with the executable or in the SwiftPM `mlx-swift_Cmlx.bundle` resource path. The server now prepares this automatically before real backend load when it can find a source library via `MLXVLM_MLX_METALLIB_PATH`, executable/workdir candidates, or common Python `mlx` install paths. In the fresh local Gemma4 E4B smoke test, no manual copy was required; the server installed `default.metallib` and loaded the Metal backend.

Verified local smoke test:

```sh
MODEL="$HOME/.cache/huggingface/hub/models--mlx-community--gemma-4-e4b-it-4bit/snapshots/cc3b666c01c20395e0dcebd53854504c7d9821f9"
MLXVLM_ENABLE_MLX_BACKEND=1 \
MLXVLM_ENABLE_TOKENIZER_INTEGRATIONS=1 \
MLXVLM_ENABLE_REAL_MLX_API=1 \
mlx-vlm-swift serve --model "$MODEL" --port 11469 --max-tokens 8 --temperature 0.0

curl -sS http://127.0.0.1:11469/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4","messages":[{"role":"user","content":"Say hi in one short sentence."}],"max_tokens":8,"temperature":0}'
```

Observed Swift response content: `Hello.` The same prompt through `/api/generate` also returned `Hello.`, matching the Python `mlx_vlm.generate` baseline for this local model. The current OpenAI-compatible response uses upstream completion info for usage accounting and reported `prompt_tokens:16`, `completion_tokens:2`. A second OpenAI-compatible request with a function-tool schema and `tool_choice:"auto"` was accepted by the upstream-backed Swift server and returned `Hi!`, confirming tool schemas are passed through instead of rejected at the bridge boundary.

Inline multimodal smoke tests against the same local Gemma4 E4B model also pass through the real backend:

- Ollama `/api/generate` with `images:["<raw base64 png>"]` returned the expected one-word color for the inline PNG.
- OpenAI `/v1/chat/completions` with `image_url.url:"data:image/png;base64,..."` returned the expected one-word color for the same inline PNG.
- OpenAI `stream:true` returned `Content-Type: text/event-stream` with per-chunk `data:` frames, final usage, and `data: [DONE]`, without a `Content-Length` header.
- Ollama `stream:true` returned `Content-Type: application/x-ndjson` with per-chunk JSON lines and final `prompt_eval_count:16`, `eval_count:2`, without a `Content-Length` header.
- The same checks are now codified in `scripts/verify_real_gemma4_smoke.sh`; set `MLXVLM_REAL_SMOKE_MODEL=/path/to/model` to run it against a different local Gemma4-compatible model directory.

Recommended backend dependency sketch:

```swift
.package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.31.3")),
.package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMajor(from: "3.31.3")),
```

The upstream-backed backend already replaces `UnavailableVLMGenerator` when real MLX modules are linked, server startup prepares the MLX Metal library for direct SwiftPM debug executables, tool schemas pass through to upstream `mlx-swift-lm`, inline image requests work for both Ollama raw base64 and OpenAI data URI payloads, Python-style audio/video/image-detail metadata plus raw/template/suffix/context/options payloads are preserved through prompt/media planning and upstream `additionalContext`, `seed` is applied through MLX random state before generation, Gemma4 tool-call text is parsed into Ollama/OpenAI Chat/Ollama/Responses response envelopes, final usage comes from upstream `GenerateCompletionInfo`, real generation streams now flush as NDJSON/SSE frames, legacy OpenAI completions honor `echo`, Ollama model/blob management no longer falls through to generic unavailable responses, and embedding routes can use upstream `MLXEmbedders` when the selected local model directory is an embedding model. VLM-only embedding requests now fail closed with explicit 501 fallback diagnostics rather than fake vectors. Runtime packaging now has an explicit preflight report for real-backend build flags plus `default.metallib` placement. The next backend hardening work is real-model video smoke coverage where a suitable local video-capable model is available.

## Phase 2: Compatibility Gaps Around The Official Engine

Keep upstream `mlx-swift-lm` as the default engine and close Python `mlx-vlm` compatibility gaps around it before porting architecture internals. Add Swift-owned architecture code only when a model or processor behavior exists in Python `mlx-vlm` but is not covered upstream. Recommended first fallback target if this becomes necessary:

- `qwen2_vl` or `qwen2_5_vl` because the Python repo has mature model, processor, tests, and common MLX community model weights.

Port order:

- Config structs.
- Vision encoder modules.
- Text model adapter or reuse from MLX Swift LM if compatible.
- Multimodal projector/merger.
- Processor image sizing and chat template path.
- Single-request `generate`.
- Streaming `/v1/chat/completions`.

## Phase 3: Server and Ollama UI

Now that real Swift generation is wired:

- Keep generation endpoint fallback diagnostics for dependency-free builds, but route real MLX builds through generated responses.
- Harden Ollama-compatible `/api/generate`, `/api/chat`, `/api/ps`, and model unload behavior against longer-running residency scenarios.
- Add a small macOS SwiftUI settings app or menu bar UI for model path, port, context length, quantization/KV cache options, and server start/stop.

## Verification Gates

Every ported architecture needs:

- Descriptor test for a local MLX model directory.
- Weight-key compatibility test against the Python expected keys.
- Prompt formatting parity test against Python processor output.
- One deterministic text-only generation smoke test.
- One image generation smoke test.
- OpenAI chat response shape test.
- OpenAI chat streaming chunk shape test.
- OpenAI Responses response shape test.
- Ollama `/api/generate` response shape test.
- Ollama streaming chunk response shape test.

The Swift runtime is the implementation source of truth. Python `mlx-vlm` behavior is used as a compatibility reference only when checking request normalization, prompt/media parity, or model-family behavior not yet covered upstream.
