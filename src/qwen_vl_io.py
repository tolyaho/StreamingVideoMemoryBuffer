from __future__ import annotations

from typing import Any, Optional, Tuple, Type

from .summary_builder import _default_torch_device, _suppress_generate_warnings, _vlm_dtype_for_device

DEFAULT_VLM_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
TEXT_EVAL_EVENT_VLM = "Qwen/Qwen2.5-VL-3B-Instruct"


def resolve_vlm_class(model_name: str) -> Type[Any]:
    mn = model_name.lower()
    if "qwen3-vl" in mn or "qwen3_vl" in mn:
        from transformers import Qwen3VLForConditionalGeneration

        return Qwen3VLForConditionalGeneration
    if "qwen2.5-vl" in mn or "qwen2_5_vl" in mn:
        from transformers import Qwen2_5_VLForConditionalGeneration

        return Qwen2_5_VLForConditionalGeneration
    if "qwen2-vl" in mn or "qwen2_vl" in mn:
        from transformers import Qwen2VLForConditionalGeneration

        return Qwen2VLForConditionalGeneration
    raise ValueError(
        f"Unsupported VLM {model_name!r}; use Qwen2-VL, Qwen2.5-VL, or Qwen3-VL Instruct."
    )


def load_qwen_vl(
    model_name: str,
    *,
    device: Optional[str] = None,
    image_min_pixels: int = 64 * 28 * 28,
    image_max_pixels: int = 320 * 28 * 28,
) -> Tuple[Any, Any, str]:
    import torch
    from transformers import AutoProcessor

    if not hasattr(torch.compiler, "is_compiling"):
        torch.compiler.is_compiling = lambda: False

    vlm_cls = resolve_vlm_class(model_name)
    dev = _default_torch_device(device)
    dtype = _vlm_dtype_for_device(dev)
    processor = AutoProcessor.from_pretrained(
        model_name,
        min_pixels=image_min_pixels,
        max_pixels=image_max_pixels,
    )
    attn_impl = "sdpa" if dev == "cuda" else "eager"
    model = (
        vlm_cls.from_pretrained(
            model_name,
            dtype=dtype,
            attn_implementation=attn_impl,
        )
        .to(dev)
        .eval()
    )
    return model, processor, dev


def cast_vlm_inputs(inputs: dict, model: Any) -> dict:
    model_dtype = model.dtype
    for key in ("pixel_values", "pixel_values_videos"):
        if key in inputs and inputs[key].dtype != model_dtype:
            inputs[key] = inputs[key].to(model_dtype)
    return inputs


def decode_vlm_new_text(processor: Any, out: Any, inputs: dict) -> str:
    skip_special_tokens = True
    clean_up = False

    gen_list = processor.post_process_image_text_to_text(
        out,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=clean_up,
    )
    prompt_list = processor.post_process_image_text_to_text(
        inputs["input_ids"],
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=clean_up,
    )
    text_full = gen_list[0]
    text_prompt = prompt_list[0]
    i = text_full.find(text_prompt)
    if 0 <= i <= 2:
        return text_full[i + len(text_prompt) :].strip()

    in_len = inputs["input_ids"].shape[1]
    tail = out[0, in_len:]
    if tail.numel() == 0:
        return ""
    tail_list = processor.post_process_image_text_to_text(
        tail.unsqueeze(0),
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=clean_up,
    )
    return tail_list[0].strip()


def _images_from_messages(messages: list) -> list:
    from PIL import Image

    images: list[Image.Image] = []
    for turn in messages:
        content = turn.get("content")
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if (
                isinstance(block, dict)
                and block.get("type") == "image"
                and "image" in block
            ):
                images.append(block["image"])
    return images


def generate_vlm_text(
    model: Any,
    processor: Any,
    messages: list,
    *,
    max_new_tokens: int = 128,
) -> str:
    import torch

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images = _images_from_messages(messages)

    inputs = processor(
        text=[text],
        images=images or None,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    cast_vlm_inputs(inputs, model)

    with torch.no_grad(), _suppress_generate_warnings():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.05,
        )
    return decode_vlm_new_text(processor, out, inputs)
