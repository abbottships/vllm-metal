# SPDX-License-Identifier: Apache-2.0
"""vllm-sr d3 decision models on the Metal encoder pooling path, with tiny random weights."""

import json
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
import torch
from mlx.utils import tree_flatten
from mlx_vlm.models.qwen3_5 import Model as Qwen3_5VLModel
from mlx_vlm.models.qwen3_5 import ModelConfig as Qwen3_5VLModelConfig
from transformers import Qwen3_5Config
from vllm.pooling_params import PoolingParams
from vllm.transformers_utils.config import get_config

from vllm_metal.v1.pooling.backends.encoder.models import d3
from vllm_metal.v1.pooling.contract import EncoderPoolingRequest

CODES = [chr(ord("A") + i) for i in range(26)] + [f"X{i}" for i in range(229)]
HIDDEN = 32


def tiny_config(**overrides):
    config = Qwen3_5Config(
        text_config={
            "vocab_size": 64,
            "hidden_size": HIDDEN,
            "intermediate_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "full_attention_interval": 2,
            "layer_types": ["linear_attention", "full_attention"] * 2,
            "max_position_embeddings": 128,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.25,
                "mrope_section": [1, 1, 0],
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": HIDDEN,
            "num_position_embeddings": 16,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
        },
    ).to_dict()
    config.update(overrides)
    return config


def decision_config(**overrides):
    decision = {
        "format_id": d3.D3_FORMAT_ID,
        "attention_mode": "noncausal_full_attention",
        "pooling": "last",
        "codes": CODES,
        "token_ids": list(range(len(CODES))),
    }
    decision.update(overrides)
    return decision


def text_model(config):
    return Qwen3_5VLModel(Qwen3_5VLModelConfig.from_dict(config)).language_model.model


def write_checkpoint(path, config, decision):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config))
    (path / "decision_config.json").write_text(json.dumps(decision))
    return path


def test_config_parser_maps_checkpoint_to_decision_architecture(tmp_path):
    quantization = {"group_size": 64, "bits": 8}
    config = tiny_config(quantization_config=quantization, quantization=quantization)
    path = write_checkpoint(tmp_path / "ckpt", config, decision_config())
    d3.register()

    hf_config = get_config(path, trust_remote_code=False, config_format="d3")

    assert hf_config.architectures == ["D3ForDecision"]
    assert hf_config.id2label[0] == "A" and len(hf_config.id2label) == 255
    # vLLM must not see an MLX quantization block it cannot resolve.
    assert getattr(hf_config, "quantization_config", None) is None
    assert hf_config.d3_quantization == quantization


@pytest.mark.parametrize(
    "overrides",
    [
        {"format_id": "d2-code-readout"},
        {"attention_mode": "causal"},
        {"pooling": "mean"},
        {"codes": CODES[:3]},
    ],
)
def test_config_parser_rejects_other_decision_formats(tmp_path, overrides):
    path = write_checkpoint(
        tmp_path / "ckpt", tiny_config(), decision_config(**overrides)
    )
    d3.register()
    with pytest.raises((NotImplementedError, ValueError)):
        get_config(path, trust_remote_code=False, config_format="d3")


def test_full_attention_layers_read_later_prompt_tokens():
    """d3 reads the prompt both ways; a causal pass would ignore later tokens."""
    model = text_model(tiny_config())
    first = mx.array([[1, 2, 3, 4, 5, 6]])
    changed = mx.array([[1, 2, 3, 4, 5, 9]])
    encoder = d3.D3Encoder(model)

    shift = mx.abs(encoder(first)[0, 0] - encoder(changed)[0, 0]).max().item()

    assert shift > 1e-4


def test_pooler_scores_the_last_prompt_token():
    readout = mx.random.normal((255, HIDDEN))
    hidden = mx.random.normal((1, 7, HIDDEN))
    config = SimpleNamespace()
    pooler = d3.D3Pooler(config, readout)
    expected = readout @ hidden[0, 4]

    def pool(**params):
        request = EncoderPoolingRequest("r", (1, 2, 3, 4, 5), PoolingParams(**params))
        return pooler.pool_one(hidden, request)

    probs = pool()
    raw = pool(use_activation=False)

    assert probs.shape == (255,) and probs.dtype == torch.float32
    torch.testing.assert_close(raw, torch.tensor(expected.tolist()))
    torch.testing.assert_close(probs, torch.softmax(raw, dim=-1))


def model_config_for(path, config, dtype=torch.bfloat16):
    hf_config = Qwen3_5Config.from_dict(
        {
            **config,
            "architectures": ["D3ForDecision"],
            "d3_config": decision_config(),
            "d3_quantization": config.get("quantization"),
        }
    )
    return SimpleNamespace(
        model=str(path),
        revision=None,
        dtype=dtype,
        hf_config=hf_config,
        served_model_name="d3",
    )


@pytest.mark.parametrize("bits", [None, 8])
def test_loader_reads_converted_mlx_checkpoints(tmp_path, monkeypatch, bits):
    config = tiny_config()
    if bits:
        config["quantization"] = {"group_size": 32, "bits": bits}
    reference = text_model(config)
    if bits:
        # Leave too-narrow layers unquantized, as mixed MLX checkpoints do.
        nn.quantize(
            reference,
            group_size=32,
            bits=bits,
            class_predicate=lambda _, m: (
                hasattr(m, "to_quantized") and m.weight.shape[-1] % 32 == 0
            ),
        )
    mx.eval(reference.parameters())
    path = tmp_path / "ckpt"
    path.mkdir()
    mx.save_safetensors(
        str(path / "model.safetensors"),
        {
            f"language_model.model.{k}": v
            for k, v in tree_flatten(reference.parameters())
        },
    )
    mx.save_safetensors(
        str(path / "readout.safetensors"),
        {"weight": mx.random.normal((255, HIDDEN)).astype(mx.float32)},
    )
    monkeypatch.setattr(d3, "cached_tokenizer_from_config", lambda _: object())

    loaded = d3.load_d3_backend(model_config_for(path, config))

    ids = mx.array([[3, 1, 4, 1, 5, 9]])
    expected = d3.D3Encoder(reference)(ids)
    actual = loaded.pooling_backend.forward_padded(ids, mx.ones_like(ids))
    assert mx.allclose(actual, expected).item()
    assert loaded.model_args["vocab_size"] == 64


def test_loader_rejects_non_bf16_before_reading_weights(tmp_path, monkeypatch):
    def fail(*_):
        pytest.fail("weights must not be read for an unsupported dtype")

    monkeypatch.setattr(d3, "load_encoder_weights", fail)
    with pytest.raises(NotImplementedError, match="bfloat16"):
        d3.load_d3_backend(
            model_config_for(tmp_path, tiny_config(), dtype=torch.float16)
        )


def test_loader_rejects_irregular_layer_patterns(tmp_path, monkeypatch):
    config = tiny_config()
    config["text_config"]["layer_types"] = [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    monkeypatch.setattr(d3, "cached_tokenizer_from_config", lambda _: object())
    with pytest.raises(NotImplementedError, match="full attention every"):
        d3.load_d3_backend(model_config_for(tmp_path, config))
