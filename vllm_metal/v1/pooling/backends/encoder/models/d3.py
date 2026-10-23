# SPDX-License-Identifier: Apache-2.0
"""vllm-sr Decision 3.0 (d3) code-readout decision models on the encoder path.

A d3 checkpoint is a Qwen3.5 hybrid backbone plus a 255-way readout over the
final-norm hidden state of the last prompt token. Its full-attention layers
read the whole prompt in both directions; the Gated DeltaNet layers stay causal.
One forward pass answers one question, so it runs without KV cache.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import torch
from mlx_vlm.models.qwen3_5 import Model as Qwen3_5VLModel
from mlx_vlm.models.qwen3_5 import ModelConfig as Qwen3_5VLModelConfig
from transformers import PreTrainedConfig, Qwen3_5Config
from transformers.utils.hub import cached_file
from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.transformers_utils.config_parser_base import ConfigParserBase

from vllm_metal.tensor_bridge import mlx_to_torch
from vllm_metal.v1.pooling.backends.encoder.models.loading import (
    encoder_model_path,
    load_encoder_weight_file,
    load_encoder_weights,
)
from vllm_metal.v1.pooling.backends.encoder.runtime import MetalEncoderPoolingBackend
from vllm_metal.v1.pooling.contract import (
    CLASSIFY_TASK,
    EncoderPoolingRequest,
    LoadedEncoderBackend,
)
from vllm_metal.v1.pooling.validation import PoolingConfigView

D3_FORMAT_ID = "d3-code-readout-v1"
D3_NUM_CODES = 255


def validate_decision_config(decision: dict[str, Any]) -> None:
    expected = {
        "format_id": D3_FORMAT_ID,
        "attention_mode": "noncausal_full_attention",
        "pooling": "last",
    }
    for key, value in expected.items():
        if decision.get(key) != value:
            raise NotImplementedError(
                f"Metal d3 supports {key}={value!r}; got {decision.get(key)!r}."
            )
    if len(decision.get("codes", ())) != D3_NUM_CODES:
        raise ValueError(f"A d3 decision config must list {D3_NUM_CODES} codes.")


class D3ConfigParser(ConfigParserBase):
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs: Any,
    ) -> tuple[dict, PreTrainedConfig]:
        configs = []
        for filename in ("config.json", "decision_config.json"):
            path = cached_file(
                model,
                filename,
                revision=revision,
                token=kwargs.get("token"),
                cache_dir=kwargs.get("cache_dir"),
            )
            assert path is not None
            configs.append(json.loads(Path(path).read_text()))
        config, decision = configs
        if config.get("model_type") != "qwen3_5":
            raise NotImplementedError("Metal d3 supports Qwen3.5 backbones only.")
        validate_decision_config(decision)
        # MLX checkpoints describe their quantization here; the Metal loader
        # owns it, so vLLM must not try to resolve a quantization method.
        quantization = config.pop("quantization_config", None)
        config.pop("quantization", None)
        config.pop("auto_map", None)
        config.update(
            architectures=["D3ForDecision"],
            id2label=dict(enumerate(decision["codes"])),
            d3_config=decision,
            d3_quantization=quantization,
        )
        return config, Qwen3_5Config.from_dict(config)


class D3ForDecision(torch.nn.Module):
    """Architecture metadata; inference runs in the Metal encoder backend."""

    is_pooling_model = True
    attn_type = "encoder_only"
    default_seq_pooling_type = "LAST"

    def __init__(self, *, vllm_config, prefix=""):
        raise RuntimeError("d3 inference requires the Metal encoder pooling backend.")

    def embed_input_ids(self, input_ids):
        raise RuntimeError("d3 embeddings are owned by the Metal encoder loader.")

    def forward(self, input_ids, positions):
        raise RuntimeError("d3 inference requires the Metal encoder pooling backend.")


def register() -> None:
    from vllm.model_executor.models import ModelRegistry
    from vllm.transformers_utils.config import (
        get_config_parser,
        register_config_parser,
    )

    try:
        get_config_parser("d3")
    except ValueError:
        register_config_parser("d3")(D3ConfigParser)
    if "D3ForDecision" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "D3ForDecision",
            "vllm_metal.v1.pooling.backends.encoder.models.d3:D3ForDecision",
        )


class D3Encoder:
    """The Qwen3.5 text stack run once over the prompt, without cache."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def __call__(self, input_ids: mx.array) -> mx.array:
        batch, length = input_ids.shape
        # Text prompts use the plain 0..L-1 ramp on all three MRoPE sections.
        positions = mx.broadcast_to(
            mx.arange(length)[None, None, :], (3, batch, length)
        )
        hidden = self.model.embed_tokens(input_ids)
        for layer in self.model.layers:
            # No mask: full-attention layers see every prompt token both ways.
            hidden = layer(hidden, position_ids=None if layer.is_linear else positions)
        return self.model.norm(hidden)


class D3Pooler:
    """Score the 255 answer codes from the last prompt token."""

    def __init__(self, config: PoolingConfigView, readout: mx.array) -> None:
        self.config = config
        self.readout = readout

    def supported_tasks(self) -> tuple[PoolingTask, ...]:
        if (
            self.config.is_text_only
            and self.config.task in (None, CLASSIFY_TASK)
            and self.config.uses_last_pooling
            and not self.config.chunked_processing_enabled
        ):
            return (CLASSIFY_TASK,)
        return ()

    def validate_params(self, pooling_params: PoolingParams) -> None:
        if not self.supported_tasks() or pooling_params.task not in (
            None,
            CLASSIFY_TASK,
        ):
            raise NotImplementedError(
                "Metal d3 supports only task='classify' with LAST pooling."
            )
        if pooling_params.dimensions is not None:
            raise NotImplementedError(
                "Metal d3 does not support truncated output dimensions."
            )

    def pool_one(
        self,
        hidden_states: mx.array,
        request: EncoderPoolingRequest,
    ) -> torch.Tensor:
        if not request.token_ids:
            raise ValueError("Metal d3 requires at least one prompt token.")
        last = hidden_states[0, len(request.token_ids) - 1].astype(mx.float32)
        logits = self.readout @ last
        if request.pooling_params.use_activation is not False:
            logits = mx.softmax(logits, axis=-1)
        return mlx_to_torch(logits, device="cpu").detach().clone()


class D3PoolingBackend(MetalEncoderPoolingBackend):
    def __init__(
        self, config: PoolingConfigView, encoder: D3Encoder, readout: mx.array
    ) -> None:
        super().__init__(config, self._encode, D3Pooler(config, readout))
        self.encoder = encoder

    def _encode(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        if not mx.all(attention_mask).item():
            raise NotImplementedError("Metal d3 does not support padded batches.")
        return self.encoder(input_ids)


def supports_d3_encoder(model_config: Any) -> bool:
    return "D3ForDecision" in (model_config.hf_config.architectures or ())


def _validate_layer_types(text_config: Any) -> None:
    layer_types = text_config.layer_types
    interval = text_config.full_attention_interval
    derived = [
        "linear_attention" if (i + 1) % interval else "full_attention"
        for i in range(text_config.num_hidden_layers)
    ]
    if list(layer_types) != derived:
        raise NotImplementedError(
            "Metal d3 requires full attention every "
            f"{interval} layers, as mlx-vlm Qwen3.5 builds it."
        )


def _text_weights(model: Qwen3_5VLModel, weights: dict[str, mx.array]):
    if any(name.startswith("language_model.layers.") for name in weights):
        # Original transformers layout: let mlx-vlm remap keys, transpose the
        # convolutions, and apply the Qwen3.5 RMSNorm offset.
        weights = model.sanitize({"model." + k: v for k, v in weights.items()})
    prefix = "language_model.model."
    return {k[len(prefix) :]: v for k, v in weights.items() if k.startswith(prefix)}


def load_d3_backend(model_config: Any) -> LoadedEncoderBackend:
    if model_config.dtype != torch.bfloat16:
        raise NotImplementedError(
            "Metal d3 keeps the checkpoint's BF16 numerics; pass --dtype bfloat16."
        )
    config = model_config.hf_config.to_dict()
    validate_decision_config(config["d3_config"])
    mlx_config = Qwen3_5VLModelConfig.from_dict(config)
    _validate_layer_types(model_config.hf_config.text_config)
    tokenizer = cached_tokenizer_from_config(model_config)
    if tokenizer is None:
        raise ValueError("d3 requires tokenizer initialization.")

    model_path = encoder_model_path(model_config)
    # The vision tower stays unevaluated: text requests never allocate it.
    vl_model = Qwen3_5VLModel(mlx_config)
    text_model = vl_model.language_model.model
    weights = _text_weights(vl_model, load_encoder_weights(model_path))
    quantization = config.get("d3_quantization")
    if quantization:
        nn.quantize(
            text_model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            class_predicate=lambda path, _: f"{path}.scales" in weights,
        )
    text_model.load_weights(list(weights.items()), strict=True)
    mx.eval(text_model.parameters())

    readout = load_encoder_weight_file(model_path / "readout.safetensors")["weight"]
    if readout.shape != (D3_NUM_CODES, mlx_config.text_config.hidden_size):
        raise ValueError(f"Unexpected d3 readout shape {readout.shape}.")
    readout = readout.astype(mx.float32)
    mx.eval(readout)

    encoder = D3Encoder(text_model)
    return LoadedEncoderBackend(
        text_model,
        tokenizer,
        config["text_config"],
        D3PoolingBackend(PoolingConfigView(model_config), encoder, readout),
    )
