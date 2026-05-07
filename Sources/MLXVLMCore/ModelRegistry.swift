import Foundation

public struct ModelRegistry: Sendable {
    public static let shared = ModelRegistry()

    public let remapping: [String: String] = [
        "llava_qwen2": "fastvlm",
        "llava-qwen2": "llava_bunny",
        "bunny-llama": "llava_bunny",
        "lfm2-vl": "lfm2_vl",
        "cohere2_vision": "aya_vision",
        "jvlm": "jina_vlm",
        "phi4-siglip": "phi4_siglip",
        "sam3_video": "sam3",
        "sam3.1_video": "sam3_1",
        "granite-vision": "granite_vision",
        "granite4-vision": "granite4_vision",
        "granite4_vision": "granite4_vision",
        "rf-detr": "rfdetr",
        "falcon-perception": "falcon_perception",
        "nemotronh_nano_omni_reasoning_v3": "nemotron_h_nano_omni",
    ]

    public let supportedModelTypes: Set<String> = [
        "aya_vision",
        "deepseek_vl_v2",
        "deepseekocr",
        "deepseekocr_2",
        "dots_ocr",
        "ernie4_5_moe_vl",
        "falcon_ocr",
        "falcon_perception",
        "fastvlm",
        "florence2",
        "gemma3",
        "gemma3n",
        "gemma4",
        "glm4v",
        "glm4v_moe",
        "glm_ocr",
        "granite4_vision",
        "granite_vision",
        "hunyuan_vl",
        "idefics2",
        "idefics3",
        "internvl_chat",
        "jina_vlm",
        "kimi_k25",
        "kimi_vl",
        "lfm2_vl",
        "llama4",
        "llava",
        "llava_bunny",
        "llava_next",
        "minicpmo",
        "mistral3",
        "mistral4",
        "mllama",
        "molmo",
        "molmo2",
        "molmo_point",
        "moondream3",
        "multi_modality",
        "nemotron_h_nano_omni",
        "paddleocr_vl",
        "paligemma",
        "phi3_v",
        "phi4_siglip",
        "phi4mm",
        "pixtral",
        "qwen2_5_vl",
        "qwen2_vl",
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_omni_moe",
        "qwen3_vl",
        "qwen3_vl_moe",
        "rfdetr",
        "sam3",
        "sam3_1",
        "sam3d_body",
        "smolvlm",
        "youtu_vl",
    ]

    public init() {}

    public func canonicalModelType(for rawModelType: String, dflashConfigPresent: Bool = false) -> String {
        var modelType = rawModelType.lowercased()
        modelType = remapping[modelType] ?? modelType
        if dflashConfigPresent {
            modelType += "_dflash"
        }
        return modelType
    }

    public func isSupported(_ canonicalModelType: String) -> Bool {
        supportedModelTypes.contains(canonicalModelType)
    }
}

