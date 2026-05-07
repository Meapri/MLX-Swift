# CLI Reference

The Swift executable is `mlx-vlm-swift`.

Common commands:

- `inspect --model /path/to/model`
- `validate-model --model /path/to/model`
- `plan-model-load --model /path/to/model`
- `inspect-mlx-generate-parameters --model /path/to/model --api openai-chat --json '{...}'`
- `preflight-predict --model /path/to/model --json '{...}'`
- `self-test`
- `serve --model /path/to/model-or-hf-id --host 127.0.0.1 --port 11434`

Run the executable without arguments to print the full command list.
