# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Nemotron-Parse's device-side normalisation path."""

import numpy as np
import pytest
import torch
from PIL import Image

from vllm.config import ModelConfig
from vllm.model_executor.layers.fusion.mm_input_norm import build_mm_input_norm
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.platforms import current_platform

from ...registry import HF_EXAMPLE_MODELS
from ...utils import build_model_context

_MODEL_ID = "nvidia/NVIDIA-Nemotron-Parse-2.0"


def _noise_image(width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


@pytest.mark.usefixtures("default_vllm_config")
@pytest.mark.parametrize(("width", "height"), [(1240, 1754), (310, 470)])
def test_mm_device_do_normalize(width: int, height: int):
    device = current_platform.device_type
    ctx = build_model_context(_MODEL_ID, limit_mm_per_prompt={"image": 1})
    assert ctx.model_config.multimodal_config.mm_device_do_normalize

    images = [_noise_image(width, height)]
    prompt = [0]

    ctx.model_config.multimodal_config.mm_device_do_normalize = False
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    mm_items = processor.info.parse_mm_data({"image": images})
    normalized_values = processor(prompt, mm_items=mm_items)["mm_kwargs"].get_data()[
        "pixel_values"
    ]

    ctx.model_config.multimodal_config.mm_device_do_normalize = True
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    raw_values = processor(prompt, mm_items=mm_items)["mm_kwargs"].get_data()[
        "pixel_values"
    ]
    assert raw_values.dtype == torch.uint8
    assert raw_values.shape == normalized_values.shape

    input_norm = build_mm_input_norm(ctx.model_config).to(device)
    output = input_norm(raw_values.flatten(1).to(device), normalized_values.dtype)
    torch.testing.assert_close(
        output, normalized_values.flatten(1).to(device), rtol=1.6e-2, atol=1e-5
    )


@pytest.mark.usefixtures("default_vllm_config")
@pytest.mark.parametrize("mm_device_do_normalize", [False, True])
@pytest.mark.parametrize("layout", ["ndarray_hwc", "tensor_hwc", "tensor_chw"])
def test_array_image_inputs(mm_device_do_normalize: bool, layout: str):
    ctx = build_model_context(_MODEL_ID, limit_mm_per_prompt={"image": 1})
    ctx.model_config.multimodal_config.mm_device_do_normalize = mm_device_do_normalize
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    image = _noise_image(310, 470)
    array = np.asarray(image)
    item = {
        "ndarray_hwc": array,
        "tensor_hwc": torch.from_numpy(array),
        "tensor_chw": torch.from_numpy(array).permute(2, 0, 1),
    }[layout]

    def pixel_values(img: object) -> torch.Tensor:
        mm_items = processor.info.parse_mm_data({"image": [img]})
        return processor([0], mm_items=mm_items)["mm_kwargs"].get_data()["pixel_values"]

    torch.testing.assert_close(pixel_values(item), pixel_values(image), rtol=0, atol=0)


@pytest.mark.usefixtures("default_vllm_config")
def test_explicit_kwargs_without_device_normalize():
    ctx = build_model_context(
        _MODEL_ID,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"do_rescale": False, "do_normalize": False},
    )
    ctx.model_config.multimodal_config.mm_device_do_normalize = False
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    mm_items = processor.info.parse_mm_data({"image": [_noise_image(310, 470)]})
    values = processor([0], mm_items=mm_items)["mm_kwargs"].get_data()["pixel_values"]
    # The remote processor ignores these kwargs and still normalises on the host.
    assert values.is_floating_point()


def _embeds_processor(image_limit: int):
    # `build_model_context` passes `enable_mm_embeds` positionally, so build the
    # config directly; keep the same online-availability skip behavior.
    HF_EXAMPLE_MODELS.find_hf_info(_MODEL_ID).check_available_online(on_fail="skip")
    model_config = ModelConfig(
        _MODEL_ID,
        runner="generate",
        trust_remote_code=True,
        limit_mm_per_prompt={"image": image_limit},
        enable_mm_embeds=True,
    )
    return MULTIMODAL_REGISTRY.create_processor(model_config)


@pytest.mark.parametrize("tensor_input", [False, True])
def test_image_embeds_size_encoder_prompt(tensor_input: bool):
    """The encoder prompt follows the embedding, not the vision tower's output."""
    processor = _embeds_processor(image_limit=1)
    hidden_size = processor.info.ctx.model_config.get_inputs_embeds_size()
    feature_size = 7
    embeddings = [torch.full((feature_size, hidden_size), 1.0)]
    image_input = torch.stack(embeddings) if tensor_input else embeddings

    mm_items = processor.info.parse_mm_data({"image": image_input})
    processed = processor([0], mm_items=mm_items)

    assert len(processed["encoder_prompt_token_ids"]) == feature_size
    assert [p.length for p in processed["mm_placeholders"]["image"]] == [feature_size]
    for item, expected in zip(processed["mm_kwargs"]["image"], embeddings):
        torch.testing.assert_close(item.get_data()["image_embeds"], expected)


def test_image_embeds_reject_oversized():
    """A feature count the cross-attention budget cannot admit is rejected."""
    processor = _embeds_processor(image_limit=1)
    hidden_size = processor.info.ctx.model_config.get_inputs_embeds_size()
    max_image_tokens = processor.info.get_num_image_tokens()
    embeddings = [torch.zeros(max_image_tokens + 1, hidden_size)]

    mm_items = processor.info.parse_mm_data({"image": embeddings})
    with pytest.raises(ValueError, match="at most"):
        processor([0], mm_items=mm_items)


def test_image_embeds_with_zero_image_limit():
    """Embedding-only inference works with the vision tower skipped."""
    processor = _embeds_processor(image_limit=0)
    hidden_size = processor.info.ctx.model_config.get_inputs_embeds_size()

    mm_items = processor.info.parse_mm_data({"image": [torch.zeros(5, hidden_size)]})
    processed = processor([0], mm_items=mm_items)

    assert len(processed["encoder_prompt_token_ids"]) == 5
