"""generates text summaries at window, episode, and event granularity.

three modes (mixable):
  template only   — fast, no model load
  + captioner     — Florence <CAPTION> per window at ingest; episode = concat <DETAILED_CAPTION> per member rep frame at flush
  + vlm           — Qwen2.5-VL-3B-Instruct for event fusion with episode texts + frames
"""
from __future__ import annotations

import types
import warnings
from contextlib import contextmanager
from typing import List, Optional

import numpy as np


def _default_torch_device(passed: Optional[str]) -> str:
    import torch

    if passed:
        return passed
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype_for_device(device: str):
    import torch

    return torch.float16 if device in ("cuda", "mps") else torch.float32


def _patch_florence2_generation(model) -> None:
    """Florence-2 prepare_inputs_for_generation crashes when past_key_values[0][0]
    is None during the first decoding step. Treat it as no cache."""
    lm = getattr(model, "language_model", None)
    if lm is None:
        return
    _orig = lm.prepare_inputs_for_generation

    def _patched(self, decoder_input_ids, past_key_values=None, **kwargs):
        if past_key_values is not None and len(past_key_values) > 0:
            layer0 = past_key_values[0]
            if layer0 is None or layer0[0] is None:
                past_key_values = None
        return _orig(decoder_input_ids, past_key_values=past_key_values, **kwargs)

    lm.prepare_inputs_for_generation = types.MethodType(_patched, lm)


def _clear_invalid_early_stopping(model) -> None:
    """Florence-2's generation_config sets early_stopping=True which HF warns about in greedy mode."""
    for mod in (model, getattr(model, "language_model", None)):
        if mod is None:
            continue
        gc = getattr(mod, "generation_config", None)
        if gc is not None:
            gc.early_stopping = False


@contextmanager
def _suppress_generate_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="To copy construct from a tensor, it is recommended to use",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="The following generation flags are not valid",
            category=UserWarning,
        )
        yield


class SummaryBuilder:
    """generates text summaries for memory entries.

    Args:
        use_model: if True, load Florence-2 for visual captioning.
        use_vlm: if True, load Qwen2.5-VL for event fusion with frames.
        caption_model_name: HuggingFace id for the Florence-2 captioner.
        vlm_model_name: HuggingFace id for the event fusion VLM.
        task_prompt: Florence-2 task token used for per-window captions at ingest.
        episode_task_prompt: Florence task for each member window when building episode text.
        device: torch device. auto-detected if None.
        caption_num_beams: beam width for Florence-2 generate (CUDA only; MPS/CPU uses greedy).
    """

    def __init__(
        self,
        use_model: bool = False,
        use_vlm: bool = False,
        caption_model_name: str = "microsoft/Florence-2-base",
        vlm_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        task_prompt: str = "<CAPTION>",
        episode_task_prompt: str = "<DETAILED_CAPTION>",
        device: Optional[str] = None,
        caption_num_beams: int = 3,
    ):
        self.use_model = use_model
        self.use_vlm = use_vlm
        self.caption_model_name = caption_model_name
        self.vlm_model_name = vlm_model_name
        self.task_prompt = task_prompt
        self.episode_task_prompt = episode_task_prompt
        self.device: Optional[str] = device
        self.caption_num_beams = max(1, int(caption_num_beams))

        self._model = None
        self._processor = None
        self._vlm = None
        self._vlm_proc = None

        if use_model:
            self._load_captioner(caption_model_name, device)
        if use_vlm:
            self._load_vlm(vlm_model_name, device)

    def _load_captioner(self, model_name: str, device: Optional[str]) -> None:
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.device = _default_torch_device(device or self.device)
        dtype = _dtype_for_device(self.device)
        print(f"Loading captioner {model_name} on {self.device} ({dtype})…")
        self._processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self._model = (
            AutoModelForCausalLM.from_pretrained(
                model_name,
                trust_remote_code=True,
                dtype=dtype,
                attn_implementation="eager",
            )
            .to(self.device)
            .eval()
        )
        if "florence" in model_name.lower():
            _patch_florence2_generation(self._model)
            _clear_invalid_early_stopping(self._model)
        print("Captioner ready.")

    def _load_vlm(self, model_name: str, device: Optional[str]) -> None:
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.device = _default_torch_device(device or self.device)
        dtype = _dtype_for_device(self.device)
        print(f"Loading event VLM {model_name} on {self.device} ({dtype})…")
        self._vlm_proc = AutoProcessor.from_pretrained(model_name)
        self._vlm = (
            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=dtype,
                attn_implementation="eager",
            )
            .to(self.device)
            .eval()
        )
        print("Event VLM ready.")

    def caption_frame(
        self,
        frame: np.ndarray,
        task_prompt: Optional[str] = None,
        max_new_tokens: int = 96,
    ) -> str:
        """run Florence-2 on a single RGB uint8 frame and return the caption string."""
        if not self.use_model or self._model is None:
            return "visual scene"

        import torch
        from PIL import Image

        prompt = task_prompt or self.task_prompt
        pil = Image.fromarray(frame).convert("RGB")
        inputs = self._processor(text=prompt, images=pil, return_tensors="pt").to(self.device)

        pixel_values = inputs["pixel_values"]
        if pixel_values.dtype != self._model.dtype:
            pixel_values = pixel_values.to(self._model.dtype)

        num_beams = self.caption_num_beams if self.device == "cuda" else 1

        with torch.no_grad(), _suppress_generate_warnings():
            out = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                do_sample=False,
                early_stopping=False,
            )

        raw = self._processor.batch_decode(out, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(
            raw, task=prompt, image_size=(pil.width, pil.height)
        )
        return str(parsed.get(prompt, "")).strip() or "visual scene"

    def build_window_caption(self, raw_window) -> str:
        """Florence <CAPTION> on the window's representative frame; falls back to time template."""
        if not self.use_model or self._model is None:
            return self.build_window_note(raw_window.start_time, raw_window.end_time)
        frame = raw_window.representative_frame
        if frame is None:
            return self.build_window_note(raw_window.start_time, raw_window.end_time)
        return self.caption_frame(frame)

    def caption_episode(self, windows: list, start_time: float, end_time: float) -> str:
        """concatenate Florence episode_task_prompt captions for each window's stored rep frame."""
        if not windows:
            return f"Episode {start_time:.1f}–{end_time:.1f}s"

        if self.use_model and self._model is not None:
            detail_tokens = 256 if self.episode_task_prompt == "<DETAILED_CAPTION>" else 128
            parts: List[str] = []
            for w in windows:
                if w.frame is not None:
                    cap = self.caption_frame(
                        w.frame,
                        task_prompt=self.episode_task_prompt,
                        max_new_tokens=detail_tokens,
                    ).strip()
                else:
                    cap = (w.summary_text or self.build_window_note(w.start_time, w.end_time)).strip()
                parts.append(cap)
            body = "\n\n".join(parts)
            if len(windows) == 1:
                return body
            return f"Episode {start_time:.1f}–{end_time:.1f}s\n\n{body}"

        notes = [
            w.summary_text or self.build_window_note(w.start_time, w.end_time)
            for w in windows
        ]
        return self._stitch_episode(notes, start_time, end_time)

    def summarize_event(
        self,
        episode_summaries: List[str],
        episode_frames: Optional[List[List[np.ndarray]]],
        start_time: float,
        end_time: float,
        episode_time_ranges: Optional[List[tuple]] = None,
    ) -> str:
        """fuse episode summaries + representative frames into one event sentence."""
        if not episode_summaries:
            return f"Event {start_time:.1f}–{end_time:.1f}s"

        if self.use_vlm and self._vlm is not None and episode_frames:
            try:
                fused = self._fuse_with_vlm(
                    episode_summaries, episode_frames, episode_time_ranges
                )
                if fused:
                    return f"Event {start_time:.1f}–{end_time:.1f}s: {fused}"
            except Exception as exc:
                print(f"[SummaryBuilder] VLM fusion failed: {exc!r}; using template.")

        snippets = " → ".join(s.split(":")[0].strip() for s in episode_summaries[:5])
        return f"Event {start_time:.1f}–{end_time:.1f}s: {snippets}"

    def build_window_note(self, start_time: float, end_time: float) -> str:
        return f"Scene at {start_time:.1f}–{end_time:.1f}s"

    def __call__(
        self,
        entries: list,
        episode_frames: Optional[List[List[np.ndarray]]] = None,
    ) -> str:
        """dispatch: list[WindowEntry] → episode summary, list[EpisodeEntry] → event summary."""
        from .data_structures import EpisodeEntry, WindowEntry

        if not entries:
            return "empty"
        start, end = entries[0].start_time, entries[-1].end_time
        if isinstance(entries[0], WindowEntry):
            return self.caption_episode(entries, start, end)
        if isinstance(entries[0], EpisodeEntry):
            return self.summarize_event(
                [e.summary_text for e in entries],
                episode_frames,
                start,
                end,
                episode_time_ranges=[(e.start_time, e.end_time) for e in entries],
            )
        return f"Memory {start:.1f}–{end:.1f}s"

    def _fuse_with_vlm(
        self,
        episode_texts: List[str],
        episode_frames: List[List[np.ndarray]],
        episode_time_ranges: Optional[List[tuple]] = None,
    ) -> str:
        import torch
        from PIL import Image

        all_frames: List[Image.Image] = []
        for frames in episode_frames:
            for f in frames:
                if f is not None:
                    all_frames.append(Image.fromarray(f).convert("RGB"))

        if not all_frames:
            return ""

        lines = []
        for i, text in enumerate(episode_texts[:8]):
            if episode_time_ranges and i < len(episode_time_ranges):
                s, e = episode_time_ranges[i]
                lines.append(f"- [{s:.1f}s–{e:.1f}s] {text}")
            else:
                lines.append(f"- {text}")
        bulleted = "\n".join(lines)
        prompt = (
            "You are summarising a longer video event from several shorter scenes.\n"
            "Use the provided images together with the scene descriptions below.\n"
            "Write **2–3 short sentences** (plain prose, no bullet list) that capture the **overall** "
            "event: who/what is on screen, the main activity or setting change across scenes, and any "
            "clear continuity. Stay grounded: describe only what the images and text support; do not "
            "invent people, objects, places, or actions.\n"
            "Keep it tight—roughly **40–80 words** total across the 2–3 sentences.\n\n"
            f"Scene descriptions:\n{bulleted}\n\n"
            "Return only those 2–3 sentences, with no title, numbering, or preamble."
        )

        content = [{"type": "image"} for _ in all_frames]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]

        text = self._vlm_proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._vlm_proc(
            text=[text],
            images=all_frames,
            return_tensors="pt",
            padding=True,
        ).to(self._vlm.device)

        with torch.no_grad(), _suppress_generate_warnings():
            out = self._vlm.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                num_beams=1,
            )

        generated = out[0][inputs["input_ids"].shape[1]:]
        return self._vlm_proc.decode(generated, skip_special_tokens=True).strip().replace("\n", " ")

    @staticmethod
    def _stitch_episode(notes: List[str], start_time: float, end_time: float) -> str:
        if len(notes) == 1:
            return notes[0]
        return f"Episode {start_time:.1f}–{end_time:.1f}s: {'; '.join(notes)}"
