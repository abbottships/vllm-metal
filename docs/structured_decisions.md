# Structured decisions

Jev-like System 1 decisions return yes/no probabilities, choices, and scores.
See the [upstream guide](https://github.com/vllm-project/vllm/blob/v0.31.0/examples/features/structured_diffusion/README.md)
for `/v1/systemone` request examples, question schemas, and sampling options.

## DiffusionGemma

Start the Metal model server:

```bash
VLLM_METAL_BACKEND=mlx vllm serve mlx-community/diffusiongemma-26B-A4B-it-4bit \
  --served-model-name dgemma --host 127.0.0.1 --port 8000 \
  --diffusion-config '{"canvas_length": 32}' --max-logprobs 32
```

Run the upstream example server in another terminal to expose `/v1/systemone`
on port 8011:

```bash
curl -fL \
  https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0/examples/features/structured_diffusion/structured_server.py \
  -o structured_server.py
python structured_server.py \
  --upstream http://127.0.0.1:8000 --model dgemma \
  --tokenizer mlx-community/diffusiongemma-26B-A4B-it-4bit \
  --canvas 32 --canvas-step 32 --host 127.0.0.1 --port 8011
```

Set `--canvas` and `--canvas-step` to the served `canvas_length`; smaller
per-request canvases are unsupported on Metal.
DiffusionGemma currently accepts text input only and uses the MLX backend.

## vllm-sr Decision 3.0 (d3)

Validated checkpoints: `vllm-sr/d3-mini` (revision
`bd0090dd758d32ba59ef5c749910a8e87fc4391f`, BF16) and `midium-ai/d3-flash-8bit`
(revision `110b223a774005b10c5c4c3d1061d6e0a0f7349f`, an 8-bit MLX quantization
of `vllm-sr/d3-flash`). Support is text-only. Image and video inputs, padded
batches, and chunked requests are unsupported.

d3 is a Qwen3.5 hybrid backbone with a 255-way readout over the last prompt
token. The full-attention layers read the whole prompt in both directions and
the Gated DeltaNet layers stay causal, so it runs on the encoder pooling path
without KV cache or paged attention. Serve it with the `d3` configuration
format and BF16, which keeps the checkpoint's numerics:

```bash
vllm serve vllm-sr/d3-mini \
  --revision bd0090dd758d32ba59ef5c749910a8e87fc4391f \
  --config-format d3 --served-model-name d3 \
  --runner pooling --dtype bfloat16 --max-model-len 4096 \
  --enforce-eager --no-enable-prefix-caching
```

Each prompt holds one question, rendered by the checkpoint's own
`d3_format.py`. The `classify` output has 255 probabilities, one per answer
code in `decision_config.json` order; the first `n` belong to a question with
`n` options. Set `use_activation: false` for raw logits. A score question is
rendered as a choice over its ordered levels.

```python
import importlib.util
import json
import urllib.request

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

repo = "vllm-sr/d3-mini"
spec = importlib.util.spec_from_file_location(
    "d3_format", hf_hub_download(repo, "d3_format.py")
)
d3_format = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d3_format)
tok = AutoTokenizer.from_pretrained(repo)
codes = d3_format.answer_codes(tok)[0]

question = {"type": "noul", "instructions": "Does this message contain a threat?"}
text = d3_format.render(tok, "See you at the picnic on Saturday!", question, codes)
request = urllib.request.Request(
    "http://localhost:8000/pooling",
    data=json.dumps(
        {
            "model": "d3",
            "task": "classify",
            "input": [tok.encode(text, add_special_tokens=False)],
        }
    ).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=120) as response:
    probs = json.load(response)["data"][0]["data"]
keys = d3_format.options(question)[0]
print(d3_format.to_answer(question, probs[: len(keys)]))
```

Send several questions about the same state as several inputs in one request
to answer them together, as `/v1/systemone` does.
