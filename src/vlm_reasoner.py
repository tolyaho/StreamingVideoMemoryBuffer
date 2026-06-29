from __future__ import annotations

from typing import Any, Optional, Sequence

from PIL import Image

from .data_structures import RetrievalResult
from .prompts import REASONER_SYSTEM_PROMPT, build_reasoner_user_block
from .qwen_vl_io import DEFAULT_VLM_MODEL, generate_vlm_text, load_qwen_vl
from .reasoner_frames import collect_frames
from .summary_builder import _default_torch_device

DEFAULT_VLM_REASONER = DEFAULT_VLM_MODEL


class VLMReasoner:
    def __init__(
        self,
        model_name: str = DEFAULT_VLM_REASONER,
        device: Optional[str] = None,
        max_new_tokens: int = 128,
        max_frames: int = 8,
        *,
        model: Any = None,
        processor: Any = None,
        image_min_pixels: int = 64 * 28 * 28,
        image_max_pixels: int = 320 * 28 * 28,
    ):
        self.model_name = model_name
        self.device = _default_torch_device(device)
        self.max_new_tokens = int(max_new_tokens)
        self.max_frames = int(max_frames)

        if model is not None and processor is not None:
            self._model = model
            self._processor = processor
            print(f"VLM reasoner reusing {model_name}.")
        else:
            print(f"Loading VLM reasoner {model_name} on {self.device}…")
            self._model, self._processor, self.device = load_qwen_vl(
                model_name,
                device=self.device,
                image_min_pixels=image_min_pixels,
                image_max_pixels=image_max_pixels,
            )
            print("VLM reasoner ready.")

    def answer(
        self,
        llm_input: dict,
        *,
        options: Optional[Sequence[str]] = None,
        retrieval: Optional[RetrievalResult] = None,
        system: str = REASONER_SYSTEM_PROMPT,
    ) -> str:
        if retrieval is None:
            raise ValueError("VLMReasoner requires retrieval with window frames")

        evidence = collect_frames(retrieval, max_frames=self.max_frames)
        user_text = build_reasoner_user_block(llm_input, options)
        if evidence:
            frame_lines = "\n".join(
                f"  Image {i + 1}: [{ev.start_time:.1f}–{ev.end_time:.1f}s] ({ev.source})"
                for i, ev in enumerate(evidence)
            )
            user_text = (
                "Attached images are retrieved video frames (in order):\n"
                f"{frame_lines}\n\n{user_text}"
            )
        prompt = f"{system}\n\n{user_text}"
        content: list[dict] = [
            {"type": "image", "image": Image.fromarray(ev.frame).convert("RGB")}
            for ev in evidence
        ]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        return generate_vlm_text(
            self._model,
            self._processor,
            messages,
            max_new_tokens=self.max_new_tokens,
        )

    @classmethod
    def from_summary_builder(
        cls,
        summary_builder: Any,
        *,
        model_name: Optional[str] = None,
        **kwargs: Any,
    ) -> "VLMReasoner":
        vlm = getattr(summary_builder, "_vlm", None)
        proc = getattr(summary_builder, "_vlm_proc", None)
        if vlm is None or proc is None:
            raise ValueError("SummaryBuilder has no event VLM loaded")
        name = model_name or getattr(summary_builder, "vlm_model_name", DEFAULT_VLM_REASONER)
        return cls(model_name=name, model=vlm, processor=proc, **kwargs)
