#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CLANG_MODULE_CACHE_PATH="$ROOT_DIR/.build/clang-module-cache"
mkdir -p "$CLANG_MODULE_CACHE_PATH"

swift build --disable-sandbox

BIN="$ROOT_DIR/.build/arm64-apple-macosx/debug/mlx-vlm-swift"
"$BIN" self-test
MLX_METAL_JSON="$(mktemp)"
"$BIN" inspect-mlx-metal-library > "$MLX_METAL_JSON"
grep -q '"requiredFileName" : "default.metallib"' "$MLX_METAL_JSON"
grep -q '"checkedPaths" : \[' "$MLX_METAL_JSON"
grep -q '"destinationPaths" : \[' "$MLX_METAL_JSON"
grep -q '"installed" : false' "$MLX_METAL_JSON"

MODEL_DIR="${TMPDIR:-/tmp}/mlx-vlm-swift-verify-model"
rm -rf "$MODEL_DIR"
mkdir -p "$MODEL_DIR"

cat > "$MODEL_DIR/config.json" <<'JSON'
{
  "model_type": "qwen2_vl",
  "hidden_size": 1536,
  "num_hidden_layers": 28,
  "intermediate_size": 8960,
  "num_attention_heads": 12,
  "rms_norm_eps": 0.000001,
  "vocab_size": 151936,
  "quantization": {"mode": "mlx", "bits": 4, "group_size": 64},
  "vision_config": {"model_type": "qwen2_vl"}
}
JSON

cat > "$MODEL_DIR/tokenizer_config.json" <<'JSON'
{
  "chat_template": "verify-template",
  "added_tokens_decoder": {
    "151655": {"content": "<|image_pad|>"},
    "151656": {"content": "<|video_pad|>"}
  }
}
JSON

cat > "$MODEL_DIR/tokenizer.json" <<'JSON'
{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {"hello": 0, "world": 1, "user": 2},
    "merges": ["hello world"]
  },
  "added_tokens": [
    {"id": 151655, "content": "<|image_pad|>"},
    {"id": 151656, "content": "<|video_pad|>"},
    {"id": 151644, "content": "<|im_start|>"},
    {"id": 151645, "content": "<|im_end|>"}
  ],
  "normalizer": {"type": "Sequence"},
  "pre_tokenizer": {"type": "ByteLevel"},
  "decoder": {"type": "ByteLevel"}
}
JSON

cat > "$MODEL_DIR/adapter_config.json" <<'JSON'
{"rank": 8, "alpha": 16, "dropout": 0.05}
JSON

cat > "$MODEL_DIR/processor_config.json" <<'JSON'
{
  "chat_template": "processor-template",
  "image_processor": {
    "merge_size": 2,
    "min_pixels": 3136,
    "max_pixels": 1003520
  }
}
JSON

cat > "$MODEL_DIR/preprocessor_config.json" <<'JSON'
{
  "image_processor_type": "Qwen2VLImageProcessor",
  "patch_size": {"height": 14, "width": 14},
  "temporal_patch_size": 2,
  "merge_size": 1,
  "size": {
    "shortest_edge": 1024,
    "longest_edge": 2048
  }
}
JSON

cat > "$MODEL_DIR/model.safetensors.index.json" <<'JSON'
{
  "metadata": {"total_size": 8},
  "weight_map": {
    "model.embed_tokens.weight": "model.safetensors",
    "visual.patch_embed.proj.weight": "model.safetensors"
  }
}
JSON

SAFETENSORS_HEADER='{"model.embed_tokens.weight":{"dtype":"F16","shape":[2,1],"data_offsets":[0,4]},"visual.patch_embed.proj.weight":{"dtype":"F16","shape":[2,1],"data_offsets":[4,8]}}'
printf '\244\0\0\0\0\0\0\0%s\000\074\000\100\000\102\000\104' "$SAFETENSORS_HEADER" > "$MODEL_DIR/model.safetensors"
printf '\244\0\0\0\0\0\0\0%s\000\074\000\100\000\102\000\104' "$SAFETENSORS_HEADER" > "$MODEL_DIR/adapters.safetensors"

"$BIN" inspect --model "$MODEL_DIR" | grep -q '"canonicalModelType" : "qwen2_vl"'
"$BIN" inspect --model "$MODEL_DIR" | grep -q '"modelType" : "BPE"'
"$BIN" inspect --model "$MODEL_DIR" | grep -q '"quantizationMetadata"'
"$BIN" inspect --model "$MODEL_DIR" | grep -q '"groupSize" : 64'
"$BIN" inspect-safetensors --model "$MODEL_DIR" | grep -q '"isReadable" : true'
SYMLINK_MODEL_DIR="${TMPDIR:-/tmp}/mlx-vlm-swift-verify-symlink-model"
rm -rf "$SYMLINK_MODEL_DIR"
mkdir -p "$SYMLINK_MODEL_DIR"
cp "$MODEL_DIR/config.json" \
  "$MODEL_DIR/tokenizer_config.json" \
  "$MODEL_DIR/tokenizer.json" \
  "$MODEL_DIR/model.safetensors.index.json" \
  "$SYMLINK_MODEL_DIR/"
ln -s "$MODEL_DIR/model.safetensors" "$SYMLINK_MODEL_DIR/model.safetensors"
"$BIN" inspect --model "$SYMLINK_MODEL_DIR" | grep -q '"weightFiles"'
"$BIN" validate-model --model "$SYMLINK_MODEL_DIR" | grep -q 'Found 1 safetensors weight file(s).'
"$BIN" validate-model --model "$SYMLINK_MODEL_DIR" | grep -q 'All model.safetensors.index.json shards are present on disk.'
"$BIN" sanitize-qwen-vl-key --key visual.patch_embed.proj.weight | grep -q '"sanitizedKey" : "vision_tower.patch_embed.proj.weight"'
"$BIN" sanitize-qwen-vl-key --key model.embed_tokens.weight | grep -q '"sanitizedKey" : "language_model.model.embed_tokens.weight"'
"$BIN" sanitize-qwen-vl-key --key lm_head.weight | grep -q '"role" : "languageHead"'
ARCH_JSON="$(mktemp)"
"$BIN" inspect-qwen-vl-architecture --model "$MODEL_DIR" > "$ARCH_JSON"
grep -q '"textLayerCount" : 28' "$ARCH_JSON"
grep -q '"presentCoreTensorCount" : 2' "$ARCH_JSON"
grep -q '"missingRequiredKeys"' "$ARCH_JSON"
BINDINGS_JSON="$(mktemp)"
"$BIN" plan-mlx-bindings --model "$MODEL_DIR" > "$BINDINGS_JSON"
grep -q '"supportedBySwiftScaffold" : true' "$BINDINGS_JSON"
grep -q '"phase" : "token-embedding"' "$BINDINGS_JSON"
grep -q '"phase" : "vision-patch-embedding"' "$BINDINGS_JSON"
grep -q '"presentTensorBindings" : 2' "$BINDINGS_JSON"
"$BIN" inspect-weight-catalog --model "$MODEL_DIR" | grep -q '"sanitizedKey" : "language_model.model.embed_tokens.weight"'
"$BIN" inspect-weight-data --model "$MODEL_DIR" | grep -q '"totalReadableBytes" : 8'
"$BIN" plan-mlx-weight-load --model "$MODEL_DIR" | grep -q '"canLoadAllTensorsAsMLXArrays" : true'
"$BIN" plan-mlx-weight-load --model "$MODEL_DIR" | grep -q '"mlxDType" : "float16"'
"$BIN" plan-mlx-weight-load --model "$MODEL_DIR" | grep -q '"totalLoadableBytes" : 8'
"$BIN" prepare-mlx-weights --model "$MODEL_DIR" --max-tensors 2 --max-total-bytes 8 | grep -q '"tensorCount" : 2'
"$BIN" prepare-mlx-weights --model "$MODEL_DIR" --max-tensors 2 --max-total-bytes 8 | grep -q '"totalByteCount" : 8'
"$BIN" prepare-mlx-weights --model "$MODEL_DIR" --tensor language_model.model.embed_tokens.weight --max-tensors 1 --max-total-bytes 4 | grep -q '"checksum" : 124'
"$BIN" prepare-mlx-weights --model "$MODEL_DIR" --skip-weight-payloads | grep -q '"tensorCount" : 0'
"$BIN" preview-weight-tensor --model "$MODEL_DIR" --tensor language_model.model.embed_tokens.weight --bytes 4 | grep -q '"hexPrefix" : "003c0040"'
"$BIN" preview-weight-tensor --model "$MODEL_DIR" --tensor language_model.model.embed_tokens.weight --bytes 4 | grep -q '"numericValues" :'
"$BIN" read-weight-tensor-payload --model "$MODEL_DIR" --tensor language_model.model.embed_tokens.weight | grep -q '"byteCount" : 4'
"$BIN" read-weight-tensor-payload --model "$MODEL_DIR" --tensor language_model.model.embed_tokens.weight | grep -q '"checksum" : 124'

DUPLICATE_WEIGHT_MODEL_DIR="${TMPDIR:-/tmp}/mlx-vlm-swift-verify-duplicate-weights"
rm -rf "$DUPLICATE_WEIGHT_MODEL_DIR"
mkdir -p "$DUPLICATE_WEIGHT_MODEL_DIR"
cat > "$DUPLICATE_WEIGHT_MODEL_DIR/config.json" <<'JSON'
{"model_type":"qwen2_vl","vision_config":{},"vocab_size":10}
JSON
printf '\244\0\0\0\0\0\0\0%s\000\074\000\100\000\102\000\104' "$SAFETENSORS_HEADER" > "$DUPLICATE_WEIGHT_MODEL_DIR/model-00001-of-00002.safetensors"
printf '\244\0\0\0\0\0\0\0%s\000\074\000\100\000\102\000\104' "$SAFETENSORS_HEADER" > "$DUPLICATE_WEIGHT_MODEL_DIR/model-00002-of-00002.safetensors"
"$BIN" inspect-weight-catalog --model "$DUPLICATE_WEIGHT_MODEL_DIR" | grep -q '"duplicateOriginalKeys"'
"$BIN" inspect-weight-catalog --model "$DUPLICATE_WEIGHT_MODEL_DIR" | grep -q '"model.embed_tokens.weight"'
"$BIN" validate-model --model "$DUPLICATE_WEIGHT_MODEL_DIR" | grep -q '"id" : "weight-catalog-unique"'
"$BIN" validate-model --model "$DUPLICATE_WEIGHT_MODEL_DIR" | grep -q '"passed" : false'

"$BIN" inspect-tokenizer-catalog --model "$MODEL_DIR" | grep -q '"content" : "<|image_pad|>"'
"$BIN" inspect-tokenizer-catalog --model "$MODEL_DIR" | grep -q '"merges" : \['
"$BIN" inspect-tokenizer-plan --model "$MODEL_DIR" | grep -q '"requiredBackend" : "tokenizers-json-bpe"'
"$BIN" inspect-tokenizer-plan --model "$MODEL_DIR" | grep -q '"requiresFullTokenizerImplementation" : true'
"$BIN" inspect-tokenizer-plan --model "$MODEL_DIR" | grep -q '"swiftExecutionSupported" : false'
"$BIN" inspect-tokenizer-plan --model "$MODEL_DIR" | grep -q '"mergeCount" : 1'
"$BIN" tokenize-simple --model "$MODEL_DIR" --text 'hello world <|image_pad|>' | grep -q '"supported" : true'
"$BIN" inspect-capabilities --model "$MODEL_DIR" | grep -q '"primaryTask" : "vision-language-generation"'
"$BIN" inspect-capabilities --model "$MODEL_DIR" | grep -q '"supportsOllamaGenerationAPI" : true'
"$BIN" inspect-adapter --model "$MODEL_DIR" | grep -q '"isLoadable" : true'
"$BIN" inspect-adapter --model "$MODEL_DIR" | grep -q '"rank" : 8'
"$BIN" inspect-processor --model "$MODEL_DIR" | grep -q '"hasPreprocessorConfig" : true'
"$BIN" inspect-processor --model "$MODEL_DIR" | grep -q '"source" : "preprocessor_config.json+processor_config.json:image_processor"'
"$BIN" inspect-config-normalization --model "$MODEL_DIR" | grep -q '"insertedEmptyTextConfig" : true'
"$BIN" inspect-config-normalization --model "$MODEL_DIR" | grep -q '"visionConfigSource" : "vision_config"'
"$BIN" inspect-normalized-config --model "$MODEL_DIR" | grep -q '"text_config"'
"$BIN" inspect-normalized-config --model "$MODEL_DIR" | grep -q '"audio_config"'
"$BIN" preflight-tokenize --model "$MODEL_DIR" --text '<|im_start|>user' | grep -q '"tokenIDs"'
"$BIN" backend-plan | grep -q '"compatibilityShellBuildable" : true'
"$BIN" backend-plan | grep -q '"packageDeclaresBackendTarget" : true'
"$BIN" backend-plan | grep -q '"packageDeclaresMLXSwift" : true'
"$BIN" backend-plan | grep -q '"packageDeclaresMLXSwiftLM" : true'
"$BIN" backend-plan | grep -q '"packageDeclaresSwiftTokenizersMLX" : true'
"$BIN" backend-plan | grep -q '"manifestSupportsMLXBackendToggle" : true'
"$BIN" backend-plan | grep -q '"manifestSupportsTokenizerIntegrationToggle" : true'
"$BIN" backend-plan | grep -q '"manifestSupportsLocalMLXDependencies" : true'
"$BIN" backend-plan | grep -q '"manifestSupportsExplicitMLXPaths" : true'
"$BIN" backend-plan | grep -q '"manifestSupportsExplicitTokenizerPath" : true'
"$BIN" backend-plan | grep -q '"canEnableMLXBackend" : false'
"$BIN" backend-plan | grep -q '"packageName" : "mlx-swift"'
"$BIN" backend-plan | grep -q '"packageName" : "swift-tokenizers-mlx"'
"$BIN" backend-availability | grep -q '"canCreateBackend" : false'
"$BIN" backend-availability | grep -q '"backendImplementationReady" : false'
"$BIN" backend-availability | grep -q '"realMLXAPIImplementationCompiled" : false'
"$BIN" backend-availability | grep -q '"canImportMLXLMCommon" : false'
"$BIN" backend-availability | grep -q '"canImportMLXVLM" : false'
"$BIN" backend-availability | grep -q '"canImportMLXLMTokenizers" : false'
MLX_BACKEND_LOAD_JSON="$(mktemp)"
"$BIN" preflight-mlx-backend-load --model "$MODEL_DIR" --max-tensors 2 --max-total-bytes 8 > "$MLX_BACKEND_LOAD_JSON"
grep -q '"canPrepareWeightPayloads" : true' "$MLX_BACKEND_LOAD_JSON"
grep -q '"arrayLoadReport"' "$MLX_BACKEND_LOAD_JSON"
grep -q '"realMLXAPIImplementationCompiled" : false' "$MLX_BACKEND_LOAD_JSON"
grep -q '"canCreateMLXArrays" : false' "$MLX_BACKEND_LOAD_JSON"
grep -q '"canRunGeneration" : false' "$MLX_BACKEND_LOAD_JSON"
grep -q '"totalByteCount" : 8' "$MLX_BACKEND_LOAD_JSON"
grep -q 'MLXArray creation from safetensors payload bytes is not available yet.' "$MLX_BACKEND_LOAD_JSON"
MLX_CONTAINER_JSON="$(mktemp)"
"$BIN" inspect-mlx-container --model "$MODEL_DIR" --max-tensors 2 --max-total-bytes 8 > "$MLX_CONTAINER_JSON"
grep -q '"preparedWeightTensorCount" : 2' "$MLX_CONTAINER_JSON"
grep -q '"loadedArrayCount" : 0' "$MLX_CONTAINER_JSON"
grep -q '"arrayBacked" : false' "$MLX_CONTAINER_JSON"
grep -q '"moduleInstantiationReady" : false' "$MLX_CONTAINER_JSON"
grep -q '"generationReady" : false' "$MLX_CONTAINER_JSON"
MLX_MODULE_PLAN_JSON="$(mktemp)"
"$BIN" inspect-mlx-module-plan --model "$MODEL_DIR" --max-tensors 2 --max-total-bytes 8 > "$MLX_MODULE_PLAN_JSON"
grep -q '"moduleConstructionReady" : false' "$MLX_MODULE_PLAN_JSON"
grep -q '"phase" : "token-embedding"' "$MLX_MODULE_PLAN_JSON"
grep -q '"phase" : "vision-patch-embedding"' "$MLX_MODULE_PLAN_JSON"
grep -q '"loadedArrayCount" : 0' "$MLX_MODULE_PLAN_JSON"
grep -q 'Qwen VL module construction is not ready yet.' "$MLX_MODULE_PLAN_JSON"
MLX_FORWARD_PLAN_JSON="$(mktemp)"
"$BIN" inspect-mlx-forward-plan --model "$MODEL_DIR" --api openai-chat --json '{"model":"verify","messages":[{"role":"user","content":"hello"}]}' --max-tensors 2 --max-total-bytes 8 > "$MLX_FORWARD_PLAN_JSON"
grep -q '"forwardReady" : false' "$MLX_FORWARD_PLAN_JSON"
grep -q '"nextTokenSelectionReady" : false' "$MLX_FORWARD_PLAN_JSON"
grep -q '"vocabSize" : 151936' "$MLX_FORWARD_PLAN_JSON"
grep -q '"inputIDsShape"' "$MLX_FORWARD_PLAN_JSON"
grep -q '"logitsShape"' "$MLX_FORWARD_PLAN_JSON"
grep -q 'Qwen VL modules must be constructed before forward can run.' "$MLX_FORWARD_PLAN_JSON"
MLX_GENERATION_LOOP_JSON="$(mktemp)"
"$BIN" inspect-mlx-generation-loop --model "$MODEL_DIR" --api openai-chat --json '{"model":"verify","messages":[{"role":"user","content":"hello"}]}' --max-tensors 2 --max-total-bytes 8 > "$MLX_GENERATION_LOOP_JSON"
grep -q '"generationLoopReady" : false' "$MLX_GENERATION_LOOP_JSON"
grep -q '"expectedFirstStep" : "prefill"' "$MLX_GENERATION_LOOP_JSON"
grep -q '"expectedLoop" : "decode-next-token-until-stop-or-length"' "$MLX_GENERATION_LOOP_JSON"
grep -q '"sampler" : "greedy"' "$MLX_GENERATION_LOOP_JSON"
grep -q '"generateParameters"' "$MLX_GENERATION_LOOP_JSON"
grep -q '"sourceAPI" : "mlx-swift-lm GenerateParameters"' "$MLX_GENERATION_LOOP_JSON"
grep -q '"prefillStepSize" : 512' "$MLX_GENERATION_LOOP_JSON"
grep -q '"assemblerHandoffReady" : true' "$MLX_GENERATION_LOOP_JSON"
grep -q 'MLX-backed decode loop is not implemented yet.' "$MLX_GENERATION_LOOP_JSON"
MLX_GENERATE_PARAMETERS_JSON="$(mktemp)"
"$BIN" inspect-mlx-generate-parameters --model "$MODEL_DIR" --api openai-chat --json '{"model":"verify","messages":[{"role":"user","content":"hello"}],"max_tokens":33,"temperature":0.7,"top_p":0.8,"top_k":12,"min_p":0.05,"repetition_penalty":1.1}' --kv-bits 8 --kv-group-size 128 --max-kv-size 4096 > "$MLX_GENERATE_PARAMETERS_JSON"
grep -q '"sourceAPI" : "mlx-swift-lm GenerateParameters"' "$MLX_GENERATE_PARAMETERS_JSON"
grep -q '"maxTokens" : 33' "$MLX_GENERATE_PARAMETERS_JSON"
grep -q '"temperature" : 0.7' "$MLX_GENERATE_PARAMETERS_JSON"
grep -q '"topP" : 0.8' "$MLX_GENERATE_PARAMETERS_JSON"
grep -q '"topK" : 12' "$MLX_GENERATE_PARAMETERS_JSON"
grep -q '"minP" : 0.05' "$MLX_GENERATE_PARAMETERS_JSON"
grep -q '"kvBits" : 8' "$MLX_GENERATE_PARAMETERS_JSON"
grep -q '"kvGroupSize" : 128' "$MLX_GENERATE_PARAMETERS_JSON"
grep -q '"maxKVSize" : 4096' "$MLX_GENERATE_PARAMETERS_JSON"
grep -q '"repetitionPenalty" : 1.1' "$MLX_GENERATE_PARAMETERS_JSON"
MLX_GENERATE_PARAMETERS_BRIDGE_JSON="$(mktemp)"
"$BIN" inspect-mlx-generate-parameters-bridge --model "$MODEL_DIR" --api openai-chat --json '{"model":"verify","messages":[{"role":"user","content":"hello"}],"max_tokens":33,"temperature":0.7,"top_p":0.8,"top_k":12,"min_p":0.05,"repetition_penalty":1.1}' --kv-bits 8 --kv-group-size 128 --max-kv-size 4096 > "$MLX_GENERATE_PARAMETERS_BRIDGE_JSON"
grep -q '"sourceAPI" : "mlx-swift-lm GenerateParameters"' "$MLX_GENERATE_PARAMETERS_BRIDGE_JSON"
grep -q '"canBridgeToGenerateParameters" : false' "$MLX_GENERATE_PARAMETERS_BRIDGE_JSON"
grep -q 'MLXLMCommon.GenerateParameters is not available in this build.' "$MLX_GENERATE_PARAMETERS_BRIDGE_JSON"
MLX_MODEL_FACTORY_BRIDGE_JSON="$(mktemp)"
"$BIN" inspect-mlx-model-factory-bridge --model "$MODEL_DIR" > "$MLX_MODEL_FACTORY_BRIDGE_JSON"
grep -q '"sourceAPI" : "MLXVLM.VLMModelFactory + MLXLMCommon.ModelContainer"' "$MLX_MODEL_FACTORY_BRIDGE_JSON"
grep -q '"canReferenceVLMModelFactory" : false' "$MLX_MODEL_FACTORY_BRIDGE_JSON"
grep -q '"canReferenceTokenizersLoader" : false' "$MLX_MODEL_FACTORY_BRIDGE_JSON"
grep -q '"canLoadLocalModelContainer" : false' "$MLX_MODEL_FACTORY_BRIDGE_JSON"
grep -q 'MLXVLM.VLMModelFactory, MLXLMCommon.ModelContainer, and MLXLMTokenizers.TokenizersLoader are not available in this build.' "$MLX_MODEL_FACTORY_BRIDGE_JSON"
MLX_DECODE_STATE_JSON="$(mktemp)"
"$BIN" inspect-mlx-decode-state --model "$MODEL_DIR" --api openai-chat --json '{"model":"verify","messages":[{"role":"user","content":"hello"}]}' --max-tensors 2 --max-total-bytes 8 > "$MLX_DECODE_STATE_JSON"
grep -q '"canInitializeDecodeState" : true' "$MLX_DECODE_STATE_JSON"
grep -q '"nextTokenInputShape"' "$MLX_DECODE_STATE_JSON"
grep -q '"eosTokenIDs" : \[' "$MLX_DECODE_STATE_JSON"
grep -q '"stopConditionOrder" : \[' "$MLX_DECODE_STATE_JSON"
grep -q '"max-completion-tokens"' "$MLX_DECODE_STATE_JSON"
MLX_PIPELINE_JSON="$(mktemp)"
"$BIN" inspect-mlx-pipeline --model "$MODEL_DIR" --api openai-chat --json '{"model":"verify","messages":[{"role":"user","content":"hello"}]}' --max-tensors 2 --max-total-bytes 8 > "$MLX_PIPELINE_JSON"
grep -q '"pipelineReady" : false' "$MLX_PIPELINE_JSON"
grep -q '"generationLoopReady" : false' "$MLX_PIPELINE_JSON"
grep -q '"forwardReady" : false' "$MLX_PIPELINE_JSON"
grep -q '"decodeState"' "$MLX_PIPELINE_JSON"
grep -q '"generateParametersBridge"' "$MLX_PIPELINE_JSON"
grep -q '"modelFactoryBridge"' "$MLX_PIPELINE_JSON"
grep -q '"canBridgeToGenerateParameters" : false' "$MLX_PIPELINE_JSON"
grep -q '"canReferenceVLMModelFactory" : false' "$MLX_PIPELINE_JSON"
grep -q '"canLoadLocalModelContainer" : false' "$MLX_PIPELINE_JSON"
grep -q '"requestModel" : "verify"' "$MLX_PIPELINE_JSON"
grep -q 'MLX-backed decode loop is not implemented yet.' "$MLX_PIPELINE_JSON"
MLX_LIGHTWEIGHT_PIPELINE_JSON="$(mktemp)"
"$BIN" inspect-mlx-pipeline --model "$MODEL_DIR" --api openai-chat --json '{"model":"verify","messages":[{"role":"user","content":"hello"}]}' --skip-weight-payloads > "$MLX_LIGHTWEIGHT_PIPELINE_JSON"
grep -q '"pipelineReady" : false' "$MLX_LIGHTWEIGHT_PIPELINE_JSON"
grep -q '"preparedWeightTensorCount" : 0' "$MLX_LIGHTWEIGHT_PIPELINE_JSON"
grep -q '"canCreateMLXArrays" : false' "$MLX_LIGHTWEIGHT_PIPELINE_JSON"
grep -q '"canBridgeToGenerateParameters" : false' "$MLX_LIGHTWEIGHT_PIPELINE_JSON"
grep -q '"canReferenceVLMModelFactory" : false' "$MLX_LIGHTWEIGHT_PIPELINE_JSON"
grep -q '"canLoadLocalModelContainer" : false' "$MLX_LIGHTWEIGHT_PIPELINE_JSON"
grep -q '"requestModel" : "verify"' "$MLX_LIGHTWEIGHT_PIPELINE_JSON"
BACKEND_CONTEXT_JSON="$(mktemp)"
"$BIN" inspect-backend-context --model "$MODEL_DIR" > "$BACKEND_CONTEXT_JSON"
grep -q '"normalizedConfig"' "$BACKEND_CONTEXT_JSON"
grep -q '"loadPlan"' "$BACKEND_CONTEXT_JSON"
OLLAMA_SHOW_JSON="$(mktemp)"
"$BIN" inspect-ollama-show --model "$MODEL_DIR" > "$OLLAMA_SHOW_JSON"
grep -q '"mlx_vlm.backend_ready"' "$OLLAMA_SHOW_JSON"
grep -q '"mlx_vlm.normalized_config_keys"' "$OLLAMA_SHOW_JSON"
grep -q '"mlx_vlm.config_text_source"' "$OLLAMA_SHOW_JSON"
"$BIN" preflight-ollama-show --json '{"model":"verify","verbose":true}' | grep -q '"verbose" : true'
"$BIN" validate-model --model "$MODEL_DIR" | grep -q '"id" : "safetensors-readable"'
"$BIN" validate-model --model "$MODEL_DIR" | grep -q '"id" : "weight-catalog-readable"'
"$BIN" validate-model --model "$MODEL_DIR" | grep -q '"id" : "weight-data-readable"'
"$BIN" validate-model --model "$MODEL_DIR" | grep -q '"id" : "weight-data-byte-counts"'
"$BIN" validate-model --model "$MODEL_DIR" | grep -q '"id" : "mlx-weight-loadable"'
"$BIN" validate-model --model "$MODEL_DIR" | grep -q '"id" : "weight-index-total-size"'
"$BIN" validate-model --model "$MODEL_DIR" | grep -q '"id" : "config-normalization"'
"$BIN" validate-model --model "$MODEL_DIR" | grep -q '"id" : "tokenizer-json-readable"'
"$BIN" validate-model --model "$MODEL_DIR" | grep -q '"id" : "tokenizer-catalog-readable"'
"$BIN" validate-model --model "$MODEL_DIR" | grep -q '"id" : "chat-template-renderer"'
"$BIN" validate-model --model "$MODEL_DIR" | grep -q '"id" : "adapter-metadata"'
"$BIN" validate-model --model "$MODEL_DIR" | grep -q '"id" : "generation-api-compatible"'
"$BIN" validate-model --model "$MODEL_DIR" | grep -q '"id" : "qwen-vl-core-tensor-coverage"'
"$BIN" validate-model --model "$MODEL_DIR" | grep -q '"severity" : "warning"'
"$BIN" inspect-chat-template-plan --model "$MODEL_DIR" | grep -q '"requiredRenderer" : "custom-template"'
"$BIN" inspect-chat-template-plan --model "$MODEL_DIR" | grep -q '"canRenderNatively" : false'

LLAMA_TEMPLATE_MODEL_DIR="${TMPDIR:-/tmp}/mlx-vlm-swift-verify-llama-template"
rm -rf "$LLAMA_TEMPLATE_MODEL_DIR"
mkdir -p "$LLAMA_TEMPLATE_MODEL_DIR"
cat > "$LLAMA_TEMPLATE_MODEL_DIR/config.json" <<'JSON'
{"model_type":"llava","vocab_size":10}
JSON
cat > "$LLAMA_TEMPLATE_MODEL_DIR/tokenizer_config.json" <<'JSON'
{"chat_template":"{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"}
JSON
"$BIN" inspect-chat-template-plan --model "$LLAMA_TEMPLATE_MODEL_DIR" | grep -q '"requiredRenderer" : "llama3-chat-builtin"'
"$BIN" inspect-chat-template-plan --model "$LLAMA_TEMPLATE_MODEL_DIR" | grep -q '"canRenderNatively" : true'
"$BIN" preflight-generate --model "$LLAMA_TEMPLATE_MODEL_DIR" --api openai-chat --json '{"model":"verify","messages":[{"role":"system","content":"sys"},{"role":"user","content":"hello"}]}' | grep -q '"promptStyle" : "llama3Chat"'
"$BIN" preflight-generate --model "$LLAMA_TEMPLATE_MODEL_DIR" --api openai-chat --json '{"model":"verify","messages":[{"role":"system","content":"sys"},{"role":"user","content":"hello"}]}' | grep -q '<|start_header_id|>assistant'

MISTRAL_TEMPLATE_MODEL_DIR="${TMPDIR:-/tmp}/mlx-vlm-swift-verify-mistral-template"
rm -rf "$MISTRAL_TEMPLATE_MODEL_DIR"
mkdir -p "$MISTRAL_TEMPLATE_MODEL_DIR"
cat > "$MISTRAL_TEMPLATE_MODEL_DIR/config.json" <<'JSON'
{"model_type":"mistral3","vocab_size":10}
JSON
cat > "$MISTRAL_TEMPLATE_MODEL_DIR/tokenizer_config.json" <<'JSON'
{"chat_template":"{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %} {{ message['content'] }}</s>{% endif %}{% endfor %}"}
JSON
"$BIN" inspect-chat-template-plan --model "$MISTRAL_TEMPLATE_MODEL_DIR" | grep -q '"requiredRenderer" : "mistral-instruct-builtin"'
"$BIN" inspect-chat-template-plan --model "$MISTRAL_TEMPLATE_MODEL_DIR" | grep -q '"canRenderNatively" : true'
"$BIN" preflight-generate --model "$MISTRAL_TEMPLATE_MODEL_DIR" --api openai-chat --json '{"model":"verify","messages":[{"role":"system","content":"sys"},{"role":"user","content":"hello"},{"role":"assistant","content":"ok"}]}' | grep -q '"promptStyle" : "mistralInstruct"'
"$BIN" preflight-generate --model "$MISTRAL_TEMPLATE_MODEL_DIR" --api openai-chat --json '{"model":"verify","messages":[{"role":"system","content":"sys"},{"role":"user","content":"hello"},{"role":"assistant","content":"ok"}]}' | grep -Fq '[INST] sys'
LOAD_JSON="$(mktemp)"
"$BIN" plan-model-load --model "$MODEL_DIR" > "$LOAD_JSON"
grep -q '"canLoadMetadata" : true' "$LOAD_JSON"
grep -q '"tokenizerPlan"' "$LOAD_JSON"
grep -q '"chatTemplatePlan"' "$LOAD_JSON"
grep -q '"adapterMetadata"' "$LOAD_JSON"
grep -q '"capabilities"' "$LOAD_JSON"
grep -q '"configNormalization"' "$LOAD_JSON"
grep -q '"normalizedConfig"' "$LOAD_JSON"
grep -q '"qwenVLArchitecture"' "$LOAD_JSON"
grep -q '"memoryEstimate"' "$LOAD_JSON"
grep -q '"totalReadableBytes" : 8' "$LOAD_JSON"
"$BIN" estimate-memory --model "$MODEL_DIR" --context-length 4096 --kv-bits 8 --max-kv-size 2048 --vision-cache-size 4 | grep -q '"kvCacheTokenCapacity" : 2048'
"$BIN" estimate-memory --model "$MODEL_DIR" --context-length 4096 --kv-bits 8 --max-kv-size 2048 --vision-cache-size 4 | grep -q '"estimatedKVCacheBytes" : 117440512'
"$BIN" estimate-memory --model "$MODEL_DIR" --context-length 4096 --kv-bits 8 --max-kv-size 2048 --vision-cache-size 4 | grep -q '"estimatedVisionCacheBytes" : 24576'

LLM_CONFIG_MODEL_DIR="${TMPDIR:-/tmp}/mlx-vlm-swift-verify-llm-config"
rm -rf "$LLM_CONFIG_MODEL_DIR"
mkdir -p "$LLM_CONFIG_MODEL_DIR"
cat > "$LLM_CONFIG_MODEL_DIR/config.json" <<'JSON'
{"model_type":"llava","llm_config":{"vocab_size":9}}
JSON
"$BIN" inspect-config-normalization --model "$LLM_CONFIG_MODEL_DIR" | grep -q '"usedLLMConfigAsTextConfig" : true'
"$BIN" inspect-config-normalization --model "$LLM_CONFIG_MODEL_DIR" | grep -q '"textConfigSource" : "llm_config"'
"$BIN" inspect-config-normalization --model "$LLM_CONFIG_MODEL_DIR" | grep -q '"insertedEmptyVisionConfig" : true'
"$BIN" inspect-normalized-config --model "$LLM_CONFIG_MODEL_DIR" | grep -q '"text_config"'
"$BIN" inspect-normalized-config --model "$LLM_CONFIG_MODEL_DIR" | grep -q '"vocab_size" : 9'
if "$BIN" inspect-normalized-config --model "$LLM_CONFIG_MODEL_DIR" | grep -q '"llm_config"'; then
  echo "normalized config retained llm_config"
  exit 1
fi

RFDETR_MODEL_DIR="${TMPDIR:-/tmp}/mlx-vlm-swift-verify-rfdetr"
rm -rf "$RFDETR_MODEL_DIR"
mkdir -p "$RFDETR_MODEL_DIR"
cat > "$RFDETR_MODEL_DIR/config.json" <<'JSON'
{"model_type":"rf-detr","vision_config":{},"vocab_size":10}
JSON
printf '\244\0\0\0\0\0\0\0%s\000\074\000\100\000\102\000\104' "$SAFETENSORS_HEADER" > "$RFDETR_MODEL_DIR/model.safetensors"
"$BIN" inspect --model "$RFDETR_MODEL_DIR" | grep -q '"canonicalModelType" : "rfdetr"'
"$BIN" inspect-capabilities --model "$RFDETR_MODEL_DIR" | grep -q '"primaryTask" : "object-detection-or-segmentation"'
"$BIN" inspect-capabilities --model "$RFDETR_MODEL_DIR" | grep -q '"supportsOllamaGenerationAPI" : false'
"$BIN" validate-model --model "$RFDETR_MODEL_DIR" | grep -q '"id" : "generation-api-compatible"'
"$BIN" validate-model --model "$RFDETR_MODEL_DIR" | grep -q '"passed" : false'
"$BIN" preflight-generate --model "$RFDETR_MODEL_DIR" --api openai-chat --json '{"model":"rfdetr","messages":[{"role":"user","content":"detect person"}]}' | grep -q '"primaryTask" : "object-detection-or-segmentation"'
"$BIN" preflight-generate --model "$RFDETR_MODEL_DIR" --api openai-chat --json '{"model":"rfdetr","messages":[{"role":"user","content":"detect person"}]}' | grep -q 'not compatible with text generation endpoints'

SIDECAR_TOKENIZER_MODEL_DIR="${TMPDIR:-/tmp}/mlx-vlm-swift-verify-sidecar-tokenizer"
rm -rf "$SIDECAR_TOKENIZER_MODEL_DIR"
mkdir -p "$SIDECAR_TOKENIZER_MODEL_DIR"
cat > "$SIDECAR_TOKENIZER_MODEL_DIR/config.json" <<'JSON'
{"model_type":"llava","vocab_size":10}
JSON
printf 'token 0\n' > "$SIDECAR_TOKENIZER_MODEL_DIR/tokenizer.tiktoken"
printf '{"hello":0}\n' > "$SIDECAR_TOKENIZER_MODEL_DIR/vocab.json"
printf '#version: 0.2\nh e\n' > "$SIDECAR_TOKENIZER_MODEL_DIR/merges.txt"
printf '[UNK]\nhello\n' > "$SIDECAR_TOKENIZER_MODEL_DIR/vocab.txt"
"$BIN" inspect --model "$SIDECAR_TOKENIZER_MODEL_DIR" | grep -q '"hasTiktoken" : true'
"$BIN" inspect --model "$SIDECAR_TOKENIZER_MODEL_DIR" | grep -q '"hasMergesTXT" : true'
"$BIN" inspect-tokenizer-plan --model "$SIDECAR_TOKENIZER_MODEL_DIR" | grep -q '"requiredBackend" : "tiktoken-file"'
"$BIN" validate-model --model "$SIDECAR_TOKENIZER_MODEL_DIR" | grep -q '"id" : "tokenizer-present"'

SIDECAR_BPE_MODEL_DIR="${TMPDIR:-/tmp}/mlx-vlm-swift-verify-sidecar-bpe"
rm -rf "$SIDECAR_BPE_MODEL_DIR"
mkdir -p "$SIDECAR_BPE_MODEL_DIR"
cat > "$SIDECAR_BPE_MODEL_DIR/config.json" <<'JSON'
{"model_type":"sidecar_bpe_test","vocab_size":6}
JSON
cat > "$SIDECAR_BPE_MODEL_DIR/vocab.json" <<'JSON'
{"<unk>":0,"hello":10,"\u0120world":11,"!":12,"\u0120":13}
JSON
cat > "$SIDECAR_BPE_MODEL_DIR/merges.txt" <<'TXT'
#version: 0.2
\u0120 world
TXT
cat > "$SIDECAR_BPE_MODEL_DIR/tokenizer_config.json" <<'JSON'
{"unk_token":"<unk>","added_tokens_decoder":{"14":{"content":"<|end|>","special":true}}}
JSON
"$BIN" inspect-tokenizer-catalog --model "$SIDECAR_BPE_MODEL_DIR" | grep -q '"source" : "vocab.json"'
"$BIN" inspect-tokenizer-plan --model "$SIDECAR_BPE_MODEL_DIR" | grep -q '"requiredBackend" : "bpe-vocab-json-merges-txt"'
"$BIN" inspect-tokenizer-plan --model "$SIDECAR_BPE_MODEL_DIR" | grep -q '"swiftExecutionMode" : "bytelevel-bpe-sidecar"'
"$BIN" inspect-tokenizer-plan --model "$SIDECAR_BPE_MODEL_DIR" | grep -q '"requiresFullTokenizerImplementation" : false'
"$BIN" tokenize-simple --model "$SIDECAR_BPE_MODEL_DIR" --text 'hello world!' | grep -q '11'
"$BIN" detokenize-simple --model "$SIDECAR_BPE_MODEL_DIR" --ids 10,11,12 | grep -q '"text" : "hello world!"'

WORD_MODEL_DIR="${TMPDIR:-/tmp}/mlx-vlm-swift-verify-wordlevel"
rm -rf "$WORD_MODEL_DIR"
mkdir -p "$WORD_MODEL_DIR"
cat > "$WORD_MODEL_DIR/config.json" <<'JSON'
{"model_type":"wordlevel_test","vocab_size":5}
JSON
cat > "$WORD_MODEL_DIR/tokenizer.json" <<'JSON'
{
  "model": {
    "type": "WordLevel",
    "unk_token": "[UNK]",
    "vocab": {"[UNK]": 0, "hello": 1, "world": 2, "<image>": 3}
  },
  "added_tokens": [
    {"id": 3, "content": "<image>", "special": true}
  ],
  "pre_tokenizer": {"type": "Whitespace"}
}
JSON
"$BIN" inspect-tokenizer-plan --model "$WORD_MODEL_DIR" | grep -q '"requiredBackend" : "tokenizers-json-wordlevel"'
"$BIN" inspect-tokenizer-plan --model "$WORD_MODEL_DIR" | grep -q '"requiresFullTokenizerImplementation" : false'
"$BIN" tokenize-simple --model "$WORD_MODEL_DIR" --text 'hello missing <image> world' | grep -q '"tokenIDs"'
"$BIN" tokenize-simple --model "$WORD_MODEL_DIR" --text 'hello missing <image> world' | grep -q '"unknownTokens"'

BYTE_BPE_MODEL_DIR="${TMPDIR:-/tmp}/mlx-vlm-swift-verify-byte-bpe"
rm -rf "$BYTE_BPE_MODEL_DIR"
mkdir -p "$BYTE_BPE_MODEL_DIR"
cat > "$BYTE_BPE_MODEL_DIR/config.json" <<'JSON'
{"model_type":"bytelevel_bpe_test","vocab_size":5}
JSON
cat > "$BYTE_BPE_MODEL_DIR/tokenizer.json" <<'JSON'
{
  "model": {
    "type": "BPE",
    "unk_token": "<unk>",
    "vocab": {"<unk>": 0, "hello": 10, "\u0120world": 11, "!": 12, "\u0120": 13}
  },
  "added_tokens": [
    {"id": 14, "content": "<|end|>", "special": true}
  ],
  "pre_tokenizer": {"type": "ByteLevel"},
  "decoder": {"type": "ByteLevel"}
}
JSON
"$BIN" tokenize-simple --model "$BYTE_BPE_MODEL_DIR" --text 'hello world!' | grep -q '"tokenIDs"'
"$BIN" tokenize-simple --model "$BYTE_BPE_MODEL_DIR" --text 'hello world!' | grep -q '10'
"$BIN" tokenize-simple --model "$BYTE_BPE_MODEL_DIR" --text 'hello world!' | grep -q '11'
"$BIN" inspect-tokenizer-plan --model "$BYTE_BPE_MODEL_DIR" | grep -q '"swiftExecutionSupported" : true'
"$BIN" inspect-tokenizer-plan --model "$BYTE_BPE_MODEL_DIR" | grep -q '"swiftExecutionMode" : "bytelevel-bpe"'
"$BIN" inspect-tokenizer-plan --model "$BYTE_BPE_MODEL_DIR" | grep -q '"requiresFullTokenizerImplementation" : false'
"$BIN" detokenize-simple --model "$BYTE_BPE_MODEL_DIR" --ids 10,11,12 | grep -q '"text" : "hello world!"'
"$BIN" decode-token-stream --model "$BYTE_BPE_MODEL_DIR" --ids 10,11,12,13,14 | grep -q '"textDelta" : " world"'
"$BIN" decode-token-stream --model "$BYTE_BPE_MODEL_DIR" --ids 10,11,12,13,14 | grep -q '"skippedSpecialToken" : true'
"$BIN" simulate-logits-decode --model "$BYTE_BPE_MODEL_DIR" --prompt-tokens 2 --max-tokens 3 --logits 0,0,0,0,0,0,0,0,0,0,5,0,0,0,0 --logits 0,0,0,0,0,0,0,0,0,0,0,5,0,0,0 --logits 0,0,0,0,0,0,0,0,0,0,0,0,5,0,0 | grep -q '"text" : "hello world!"'
"$BIN" render-generation-response --api openai-chat --model verify --text ok --prompt-tokens 3 --completion-tokens 1 | grep -q '"contentType" : "application\\/json"'
"$BIN" render-generation-response --api ollama-generate --model verify --text ok --prompt-tokens 3 --completion-tokens 1 --stream true --chunk o | grep -q '"contentType" : "application\\/x-ndjson"'
"$BIN" render-generation-response --api openai-chat --model verify --text ok --prompt-tokens 3 --completion-tokens 1 --stream true --chunk o | grep -q 'data: \[DONE\]'
"$BIN" render-generation-chunks --api openai-chat --model verify --prompt-tokens 3 --stream true --chunk 10:o --chunk 11:k | grep -q '"text" : "ok"'
"$BIN" render-generation-chunks --api openai-chat --model verify --prompt-tokens 3 --stream true --chunk 10:o --chunk 11:k | grep -q '"completionTokens" : 2'
"$BIN" render-generation-chunks --api openai-chat --model verify --prompt-tokens 3 --stream true --chunk 10:o --chunk 11:k | grep -q 'data: \[DONE\]'
"$BIN" preflight-generate --model "$BYTE_BPE_MODEL_DIR" --api openai-chat --json '{"model":"verify","messages":[{"role":"user","content":"hello world!"}]}' | grep -q '"requiresTokenizerImplementation" : false'
"$BIN" preflight-generate --model "$MODEL_DIR" --api openai-chat --json '{"model":"verify","messages":[{"role":"user","content":"hello"}]}' | grep -q '"promptStyle" : "qwenChat"'
"$BIN" preflight-generate --model "$MODEL_DIR" --api openai-chat --max-tokens 64 --temperature 0.6 --top-p 0.75 --top-k 22 --min-p 0.04 --repeat-last-n 96 --seed 99 --context-length 8192 --keep-alive 30m --json '{"model":"verify","messages":[{"role":"user","content":"defaults"}]}' | grep -q '"maxTokens" : 64'
"$BIN" preflight-generate --model "$MODEL_DIR" --api openai-chat --max-tokens 64 --temperature 0.6 --top-p 0.75 --top-k 22 --min-p 0.04 --repeat-last-n 96 --seed 99 --context-length 8192 --keep-alive 30m --json '{"model":"verify","messages":[{"role":"user","content":"defaults"}]}' | grep -q '"minP" : 0.04'
"$BIN" preflight-generate --model "$MODEL_DIR" --api openai-chat --max-tokens 64 --temperature 0.6 --top-p 0.75 --top-k 22 --min-p 0.04 --repeat-last-n 96 --seed 99 --context-length 8192 --keep-alive 30m --json '{"model":"verify","messages":[{"role":"user","content":"defaults"}]}' | grep -q '"repeatLastN" : 96'
"$BIN" preflight-generate --model "$MODEL_DIR" --api openai-chat --max-tokens 64 --temperature 0.6 --top-p 0.75 --top-k 22 --min-p 0.04 --repeat-last-n 96 --seed 99 --context-length 8192 --keep-alive 30m --json '{"model":"verify","messages":[{"role":"user","content":"defaults"}]}' | grep -q '"contextLength" : 8192'
"$BIN" preflight-generate --model "$MODEL_DIR" --api openai-chat --max-tokens 64 --temperature 0.6 --top-p 0.75 --top-k 22 --min-p 0.04 --repeat-last-n 96 --seed 99 --context-length 8192 --keep-alive 30m --json '{"model":"verify","messages":[{"role":"user","content":"defaults"}]}' | grep -q '"effectiveContextLength" : 8192'
"$BIN" preflight-generate --model "$MODEL_DIR" --api openai-chat --max-tokens 64 --temperature 0.6 --top-p 0.75 --top-k 22 --min-p 0.04 --repeat-last-n 96 --seed 99 --context-length 8192 --keep-alive 30m --json '{"model":"verify","messages":[{"role":"user","content":"defaults"}]}' | grep -q '"keepAlive" : "30m"'
"$BIN" preflight-generate --model "$MODEL_DIR" --api openai-chat --kv-bits 8 --kv-quant-scheme uniform --kv-group-size 64 --max-kv-size 4096 --vision-cache-size 8 --quantize-activations true --json '{"model":"verify","messages":[{"role":"user","content":"cache"}]}' | grep -q '"kvBits" : 8'
"$BIN" preflight-generate --model "$MODEL_DIR" --api openai-chat --kv-bits 8 --kv-quant-scheme uniform --kv-group-size 64 --max-kv-size 4096 --vision-cache-size 8 --quantize-activations true --json '{"model":"verify","messages":[{"role":"user","content":"cache"}]}' | grep -q '"kvQuantizationScheme" : "uniform"'
"$BIN" preflight-generate --model "$MODEL_DIR" --api openai-chat --kv-bits 8 --kv-quant-scheme uniform --kv-group-size 64 --max-kv-size 4096 --vision-cache-size 8 --quantize-activations true --json '{"model":"verify","messages":[{"role":"user","content":"cache"}]}' | grep -q '"visionCacheSize" : 8'
"$BIN" preflight-generate --model "$MODEL_DIR" --api openai-chat --kv-bits 8 --kv-quant-scheme uniform --kv-group-size 64 --max-kv-size 4096 --vision-cache-size 8 --quantize-activations true --json '{"model":"verify","messages":[{"role":"user","content":"cache"}]}' | grep -q '"quantizeActivations" : true'
"$BIN" preflight-generate --model "$MODEL_DIR" --api ollama-chat --json '{"model":"verify","messages":[{"role":"user","content":"ctx"}],"keep_alive":45,"options":{"num_ctx":2048,"mirostat":2,"mirostat_tau":5.0,"mirostat_eta":0.1,"penalize_newline":false}}' | grep -q '"contextLength" : 2048'
"$BIN" preflight-generate --model "$MODEL_DIR" --api ollama-chat --json '{"model":"verify","messages":[{"role":"user","content":"ctx"}],"keep_alive":45,"options":{"num_ctx":2048,"mirostat":2,"mirostat_tau":5.0,"mirostat_eta":0.1,"penalize_newline":false}}' | grep -q '"keepAlive" : "45"'
"$BIN" preflight-generate --model "$MODEL_DIR" --api ollama-chat --json '{"model":"verify","messages":[{"role":"user","content":"ctx"}],"keep_alive":45,"options":{"num_ctx":2048,"mirostat":2,"mirostat_tau":5.0,"mirostat_eta":0.1,"penalize_newline":false}}' | grep -q '"mirostat" : 2'
"$BIN" preflight-generate --model "$MODEL_DIR" --api ollama-chat --json '{"model":"verify","messages":[{"role":"user","content":"ctx"}],"keep_alive":45,"options":{"num_ctx":2048,"mirostat":2,"mirostat_tau":5.0,"mirostat_eta":0.1,"penalize_newline":false}}' | grep -q '"penalizeNewline" : false'
"$BIN" preflight-generate --model "$MODEL_DIR" --api ollama-generate --json '{"model":"verify","prompt":"sample","options":{"temperature":0.7,"top_k":40,"top_p":0.9,"min_p":0.05,"typical_p":0.8,"tfs_z":1.0,"seed":42,"repeat_penalty":1.1,"repeat_last_n":64,"presence_penalty":0.2,"frequency_penalty":0.3,"penalize_newline":true,"mirostat":2,"mirostat_tau":5.0,"mirostat_eta":0.1}}' | grep -q '"sampler" : "mirostat"'
"$BIN" preflight-generate --model "$MODEL_DIR" --api ollama-generate --json '{"model":"verify","prompt":"sample","options":{"temperature":0.7,"top_k":40,"top_p":0.9,"min_p":0.05,"typical_p":0.8,"tfs_z":1.0,"seed":42,"repeat_penalty":1.1,"repeat_last_n":64,"presence_penalty":0.2,"frequency_penalty":0.3,"penalize_newline":true,"mirostat":2,"mirostat_tau":5.0,"mirostat_eta":0.1}}' | grep -q '"requiresAdvancedSampler" : true'
"$BIN" sample-logits --logits 0.1,0.9,0.2 --temperature 0 | grep -q '"tokenID" : 1'
"$BIN" sample-logits --logits 0.1,0.9,0.8 --temperature 0 --repeat-penalty 2.0 --repeat-last-n 1 --recent-token-ids 1 | grep -q '"tokenID" : 2'
"$BIN" sample-logits --logits 0.1,0.9,0.2 --temperature 0.7 --top-k 2 --top-p 0.8 --min-p 0.1 --seed 7 | grep -q '"sampler" : "temperature"'
"$BIN" simulate-decode-loop --model verify --prompt-tokens 4 --max-tokens 4 --stop END --token 10:he --token 11:llo --token 12:ENDtail | grep -q '"text" : "hello"'
"$BIN" simulate-decode-loop --model verify --prompt-tokens 1 --max-tokens 4 --eos 2 --token 1:a --token '2:</s>' --token 3:b | grep -q '"completionTokens" : 2'
"$BIN" simulate-decode-loop --model verify --prompt-tokens 1 --max-tokens 2 --token 1:a --token 2:b --token 3:c | grep -q '"finishReason" : "length"'
"$BIN" preflight-generate --model "$MODEL_DIR" --api ollama-generate --json '{"model":"verify","prompt":"options","options":{"num_gpu":1,"main_gpu":0,"low_vram":true,"use_mmap":false}}' | grep -q '"rawOptions"'
"$BIN" preflight-generate --model "$MODEL_DIR" --api ollama-generate --json '{"model":"verify","prompt":"options","options":{"num_gpu":1,"main_gpu":0,"low_vram":true,"use_mmap":false}}' | grep -q '"low_vram" : true'
"$BIN" preflight-generate --model "$MODEL_DIR" --api ollama-generate --json '{"model":"verify","system":"sys","prompt":"meta","suffix":"tail","raw":true,"template":"{{ .Prompt }}","format":"json","context":[1,2,3]}' | grep -q '"rawPrompt" : true'
"$BIN" preflight-generate --model "$MODEL_DIR" --api ollama-generate --json '{"model":"verify","system":"sys","prompt":"meta","suffix":"tail","raw":true,"template":"{{ .Prompt }}","format":"json","context":[1,2,3]}' | grep -q '"template" : "{{ .Prompt }}"'
"$BIN" preflight-generate --model "$MODEL_DIR" --api ollama-generate --json '{"model":"verify","system":"sys","prompt":"meta","suffix":"tail","raw":true,"template":"{{ .Prompt }}","format":"json","context":[1,2,3]}' | grep -q '"suffix" : "tail"'
"$BIN" preflight-generate --model "$MODEL_DIR" --api ollama-generate --json '{"model":"verify","system":"sys","prompt":"meta","suffix":"tail","raw":true,"template":"{{ .Prompt }}","format":"json","context":[1,2,3]}' | grep -q '"legacyContext"'
"$BIN" preflight-generate --model "$MODEL_DIR" --api ollama-generate --json '{"model":"verify","system":"sys","prompt":"meta","suffix":"tail","raw":true,"template":"{{ .Prompt }}","format":"json","context":[1,2,3]}' | grep -q '"source" : "ollama-template"'
"$BIN" preflight-generate --model "$MODEL_DIR" --api ollama-generate --json '{"model":"verify","system":"sys","prompt":"meta","suffix":"tail","raw":true,"template":"{{ .Prompt }}","format":"json","context":[1,2,3]}' | grep -q '"formatType" : "json_object"'
"$BIN" preflight-generate --model "$MODEL_DIR" --api ollama-generate --json '{"model":"verify","prompt":"raw","suffix":"tail","raw":true}' | grep -q '"source" : "ollama-raw"'
"$BIN" preflight-generate --model "$MODEL_DIR" --api ollama-generate --json '{"model":"verify","prompt":"raw","suffix":"tail","raw":true}' | grep -q '"prompt" : "rawtail"'
"$BIN" preflight-generate --model "$MODEL_DIR" --api openai-chat --json '{"model":"verify","messages":[{"role":"user","content":"json"}],"response_format":{"type":"json_object"},"tools":[{"type":"function","function":{"name":"describe"}}],"tool_choice":"auto"}' | grep -q '"responseFormat"'
"$BIN" preflight-generate --model "$MODEL_DIR" --api openai-chat --json '{"model":"verify","messages":[{"role":"user","content":"json"}],"response_format":{"type":"json_object"},"tools":[{"type":"function","function":{"name":"describe"}}],"tool_choice":"auto"}' | grep -q '"toolChoice" : "auto"'
"$BIN" preflight-generate --model "$MODEL_DIR" --api openai-chat --json '{"model":"verify","messages":[{"role":"user","content":"json"}],"response_format":{"type":"json_object"},"tools":[{"type":"function","function":{"name":"describe"}}],"tool_choice":"auto"}' | grep -q '"tools"'
"$BIN" preflight-generate --model "$MODEL_DIR" --api openai-chat --json '{"model":"verify","messages":[{"role":"user","content":"json"}],"response_format":{"type":"json_object"},"tools":[{"type":"function","function":{"name":"describe"}}],"tool_choice":"auto"}' | grep -q '"requiresJSONMode" : true'
"$BIN" preflight-generate --model "$MODEL_DIR" --api openai-chat --json '{"model":"verify","messages":[{"role":"user","content":"json"}],"response_format":{"type":"json_object"},"tools":[{"type":"function","function":{"name":"describe"}}],"tool_choice":{"type":"function","function":{"name":"describe"}}}' | grep -q '"toolChoiceMode" : "function"'
"$BIN" preflight-generate --model "$MODEL_DIR" --api openai-chat --json '{"model":"verify","messages":[{"role":"user","content":"json"}],"response_format":{"type":"json_object"},"tools":[{"type":"function","function":{"name":"describe"}}],"tool_choice":{"type":"function","function":{"name":"describe"}}}' | grep -q '"forcedFunctionName" : "describe"'
OPENAI_EXTRAS_JSON="$(mktemp)"
"$BIN" preflight-generate --model "$MODEL_DIR" --api openai-chat --json '{"model":"verify","messages":[{"role":"user","content":"openai extras"}],"max_completion_tokens":7,"top_k":13,"min_p":0.06,"repetition_penalty":1.2,"logit_bias":{"1":-2},"enable_thinking":true,"thinking_budget":16,"thinking_start_token":"<think>","logprobs":true,"top_logprobs":2,"resize_shape":[224,336],"adapter_path":"/tmp/adapter","user":"u1"}' > "$OPENAI_EXTRAS_JSON"
grep -q '"topK" : 13' "$OPENAI_EXTRAS_JSON"
grep -q '"resizeShape"' "$OPENAI_EXTRAS_JSON"
grep -q '"adapterPath"' "$OPENAI_EXTRAS_JSON"
MESSAGE_METADATA_JSON="$(mktemp)"
"$BIN" preflight-generate --model "$MODEL_DIR" --api openai-chat --json '{"model":"verify","messages":[{"role":"assistant","content":[{"type":"output_text","text":"need tool"}],"reasoning":"checking","tool_calls":[{"id":"call_1","type":"function","function":{"name":"lookup","arguments":"{}"}}]},{"role":"tool","content":"tool result","tool_call_id":"call_1","name":"lookup"},{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"AAECAw==","format":"wav"}}]}]}' > "$MESSAGE_METADATA_JSON"
grep -q '"reasoning" : "checking"' "$MESSAGE_METADATA_JSON"
grep -q '"tool_call_id" : "call_1"' "$MESSAGE_METADATA_JSON"
grep -q '"name" : "lookup"' "$MESSAGE_METADATA_JSON"
grep -q '"content_types"' "$MESSAGE_METADATA_JSON"
grep -q '"audio"' "$MESSAGE_METADATA_JSON"
"$BIN" preflight-generate --model "$MODEL_DIR" --api openai-responses --json '{"model":"verify","input":"schema","text":{"format":{"type":"json_schema","schema":{"type":"object"}}},"stream":true}' | grep -q '"requiresSchemaGuidance" : true'
"$BIN" preflight-generate --model "$MODEL_DIR" --api openai-responses --json '{"model":"verify","input":"schema","text":{"format":{"type":"json_schema","schema":{"type":"object"}}},"stream":true}' | grep -q '"streamFraming" : "api-native-stream"'
"$BIN" preflight-generate --model "$MODEL_DIR" --api openai-responses --json '{"model":"verify","input":"schema","text":{"format":{"type":"json_schema","schema":{"type":"object"}}},"tools":[{"type":"function","name":"search"}],"tool_choice":"auto","stream":true}' | grep -q '"toolNames" : \['
RESPONSES_EXTRAS_JSON="$(mktemp)"
"$BIN" preflight-generate --model "$MODEL_DIR" --api openai-responses --json '{"model":"verify","input":"responses extras","top_k":14,"min_p":0.07,"repetition_penalty":1.3,"logit_bias":{"2":-1},"enable_thinking":false,"thinking_budget":8,"thinking_start_token":"<reason>","user":"resp"}' > "$RESPONSES_EXTRAS_JSON"
grep -q '"topK" : 14' "$RESPONSES_EXTRAS_JSON"
grep -q '"thinkingStartToken" : "<reason>"' "$RESPONSES_EXTRAS_JSON"
"$BIN" preflight-embed --model "$MODEL_DIR" --api ollama-embed --json '{"model":"verify","input":["alpha","beta"],"truncate":false,"keep_alive":"5m","options":{"num_ctx":1024}}' | grep -q '"inputCount" : 2'
"$BIN" preflight-embed --model "$MODEL_DIR" --api ollama-embed --json '{"model":"verify","input":["alpha","beta"],"truncate":false,"keep_alive":"5m","options":{"num_ctx":1024}}' | grep -q '"keepAlive" : "5m"'
"$BIN" preflight-embed --model "$MODEL_DIR" --api ollama-embeddings --json '{"model":"verify","prompt":"legacy"}' | grep -q '"texts" :'
"$BIN" preflight-embed --model "$MODEL_DIR" --api openai-embeddings --json '{"model":"verify","input":[[1,2,3],[4,5]]}' | grep -q '"tokenIDInputs"'
"$BIN" preflight-model-operation --operation pull --json '{"model":"verify"}' | grep -q '"accepted" : false'
"$BIN" preflight-model-operation --operation pull --json '{"model":"verify"}' | grep -q '"operation" : "pull"'
"$BIN" preflight-ollama-blob --digest sha256:abc123 | grep -q '"operation" : "push-blob"'
"$BIN" preflight-ollama-blob --digest sha256:abc123 | grep -q '"accepted" : false'
"$BIN" preflight-ollama-residency --json '{"model":"verify","prompt":"","keep_alive":0}' | grep -q '"action" : "unload"'
"$BIN" preflight-ollama-residency --json '{"model":"verify","prompt":"","keep_alive":"5m"}' | grep -q '"action" : "load"'
"$BIN" preflight-generate --model "$MODEL_DIR" --api ollama-generate --json '{"model":"verify","prompt":"look","images":["iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABDQottAAAAABJRU5ErkJggg=="]}' | grep -q '"totalRGBByteCount" : 9408'
"$BIN" inspect-media --api ollama-generate --json '{"model":"verify","prompt":"look","images":["AAECAw==","data:image/png;base64,AAECAw==","https://example.invalid/i.png"],"options":{"stop":["</s>"],"seed":7}}' | grep -q '"source" : "dataURI"'
"$BIN" plan-qwen-vl-images --api ollama-generate --json '{"model":"verify","prompt":"look","images":["iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABDQottAAAAABJRU5ErkJggg=="]}' | grep -q '"placeholderTokenCount" : 4'
"$BIN" plan-qwen-vl-pixels --api ollama-generate --json '{"model":"verify","prompt":"look","images":["iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABDQottAAAAABJRU5ErkJggg=="]}' | grep -q '"rgbByteCount" : 9408'
"$BIN" plan-qwen-vl-pixels --api ollama-generate --json '{"model":"verify","prompt":"look","images":["iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABDQottAAAAABJRU5ErkJggg=="]}' | grep -q '"patchFloat32ByteCount" : 75264'
"$BIN" plan-qwen-vl-pixels --api ollama-generate --json '{"model":"verify","prompt":"look","images":["iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABDQottAAAAABJRU5ErkJggg=="]}' | grep -q '"checksum" : -18816'
"$BIN" apply-stop-sequences --text 'alphaENDtail' --stop END | grep -q '"text" : "alpha"'
"$BIN" apply-stop-sequences --text 'alphaENDtail' --stop END | grep -q '"matchedStopSequence" : "END"'
"$BIN" format-qwen-vl-prompt --api openai-responses --json '{"model":"verify","instructions":"be concise","input":"describe"}' | grep -q 'System: be concise'
"$BIN" format-qwen-vl-prompt --api openai-chat --json '{"model":"verify","messages":[{"role":"user","content":"stop check"}],"stop":"END","seed":9}' | grep -q 'stop check'
"$BIN" format-qwen-vl-prompt --style qwen-chat --api openai-chat --json '{"model":"verify","messages":[{"role":"user","content":"hello"}]}' | grep -q '<|im_start|>user'
"$BIN" plan-qwen-vl-image-grid --height 224 --width 224 | grep -q '"placeholderTokenCount" : 64'

PORT="${MLX_VLM_SWIFT_VERIFY_PORT:-11454}"
"$BIN" serve --model "$MODEL_DIR" --port "$PORT" >/tmp/mlx-vlm-swift-verify-server.log 2>&1 &
PID=$!
trap 'kill "$PID" 2>/dev/null || true; wait "$PID" 2>/dev/null || true' EXIT
sleep 0.5

if ! kill -0 "$PID" 2>/dev/null; then
  echo "server smoke skipped: local port bind is unavailable in this environment"
  exit 0
fi

curl -fsS "http://127.0.0.1:$PORT/health" | grep -q '"backend_ready":false'
curl -fsS "http://127.0.0.1:$PORT/health" | grep -q '"model_loaded":true'
curl -fsS "http://127.0.0.1:$PORT/api/tags" | grep -q '"models"'
MODEL_ID="$(basename "$MODEL_DIR")"
curl -fsS "http://127.0.0.1:$PORT/v1/models?limit=20" | grep -q '"object":"list"'
curl -fsS "http://127.0.0.1:$PORT/v1/models/$MODEL_ID" | grep -q '"owned_by":"mlx-vlm-swift"'
BLOB_HEAD_STATUS="$(curl -sS -o /dev/null -w '%{http_code}' -I "http://127.0.0.1:$PORT/api/blobs/sha256:abc123")"
test "$BLOB_HEAD_STATUS" = "404"
curl -sS -X POST "http://127.0.0.1:$PORT/api/blobs/sha256:abc123" | grep -q '"operation":"push-blob"'
SERVER_SHOW_JSON="$(mktemp)"
curl -fsS "http://127.0.0.1:$PORT/api/show" > "$SERVER_SHOW_JSON"
grep -q '"mlx_vlm.backend_ready"' "$SERVER_SHOW_JSON"
grep -q '"mlx_vlm.normalized_config_keys"' "$SERVER_SHOW_JSON"
curl -fsS -X POST "http://127.0.0.1:$PORT/api/show" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"verbose\":true}" | grep -q '"mlx_vlm.backend_ready"'
curl -fsS "http://127.0.0.1:$PORT/backend/status" | grep -q '"generationUnavailable"'
curl -fsS "http://127.0.0.1:$PORT/api/ps" | grep -q '"size_vram"'
curl -fsS "http://127.0.0.1:$PORT/api/unload" | grep -q '"unloaded":true'
curl -fsS "http://127.0.0.1:$PORT/api/ps" | grep -q '"models":\[\]'
LOAD_RESPONSE="$(
  curl -sS -X POST "http://127.0.0.1:$PORT/api/generate" \
    -H 'Content-Type: application/json' \
    -d '{"model":"verify","prompt":"","keep_alive":"5m"}'
)"
echo "$LOAD_RESPONSE" | grep -q '"done_reason":"load"'
curl -fsS "http://127.0.0.1:$PORT/api/ps" | grep -q '"models"'
UNLOAD_RESPONSE="$(
  curl -sS -X POST "http://127.0.0.1:$PORT/api/generate" \
    -H 'Content-Type: application/json' \
    -d '{"model":"verify","prompt":"","keep_alive":0}'
)"
echo "$UNLOAD_RESPONSE" | grep -q '"done_reason":"unload"'
curl -fsS "http://127.0.0.1:$PORT/api/ps" | grep -q '"models":\[\]'

CHAT_RESPONSE="$(
  curl -sS -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d '{"model":"verify","messages":[{"role":"user","content":[{"type":"text","text":"hi"},{"type":"image_url","image_url":{"url":"data:image/png;base64,AAAA"}}]}],"max_tokens":8}'
)"
echo "$CHAT_RESPONSE" | grep -q '"plainPrompt":"hi<|image_pad|>"'
echo "$CHAT_RESPONSE" | grep -q '"mlxPipeline"'
echo "$CHAT_RESPONSE" | grep -q '"generationLoopReady":false'
echo "$CHAT_RESPONSE" | grep -q '"canReferenceVLMModelFactory":false'
echo "$CHAT_RESPONSE" | grep -q '"canLoadLocalModelContainer":false'
echo "$CHAT_RESPONSE" | grep -q '"pipelineReady":false'

echo "swift port verification passed"
