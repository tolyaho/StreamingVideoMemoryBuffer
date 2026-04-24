"""Text summaries per tier: Florence-2 on windows, Moondream2 on episodes, Qwen2.5-VL on events.
P.S. every tier is optional and falls back to a time-template when its model is off."""
from __future__ import annotations

import types
import warnings
from contextlib import contextmanager
from typing import List, Optional

import numpy as np

from .prompts import MOONDREAM_FRAME_PROMPT, build_event_vlm_prompt


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

    if device in ("cuda", "mps"):
        return torch.float16
    return torch.bfloat16


def _vlm_dtype_for_device(device: str):
    import torch

    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.bfloat16


def _patch_florence2_generation(model) -> None:
    """Florence-2 crashes on a None first-step KV cache — treat it as no cache."""
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
    for mod in (model, getattr(model, "language_model", None)):
        if mod is None:
            continue
        gc = getattr(mod, "generation_config", None)
        if gc is not None:
            gc.early_stopping = False


@contextmanager
def _suppress_generate_warnings():
    """mute the two noisy HF generate warnings."""
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
    """loads the three captioners (all optional) and exposes per-tier summary methods."""

    def __init__(
        self,
        use_model: bool = False,
        use_vlm: bool = False,
        use_moondream: bool = False,
        caption_model_name: str = "microsoft/Florence-2-base",
        vlm_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        moondream_model_name: str = "vikhyatk/moondream2",
        moondream_revision: Optional[str] = None,
        task_prompt: str = "<CAPTION>",
        episode_task_prompt: str = "<DETAILED_CAPTION>",
        device: Optional[str] = None,
        vlm_device: Optional[str] = None,
        moondream_device: Optional[str] = None,
        caption_num_beams: int = 3,
        vlm_max_frames: int = 10,
        vlm_image_max_pixels: int = 320 * 28 * 28,
        vlm_image_min_pixels: int = 64 * 28 * 28,
    ):
        self.use_model = use_model
        self.use_vlm = use_vlm
        self.use_moondream = use_moondream
        self.caption_model_name = caption_model_name
        self.vlm_model_name = vlm_model_name
        self.moondream_model_name = moondream_model_name
        self.moondream_revision = moondream_revision
        self.vlm_max_frames = max(1, int(vlm_max_frames))
        self.vlm_image_max_pixels = int(vlm_image_max_pixels)
        self.vlm_image_min_pixels = int(vlm_image_min_pixels)
        self.task_prompt = task_prompt
        self.episode_task_prompt = episode_task_prompt
        self.device: Optional[str] = device
        self._vlm_device_arg = vlm_device
        self._moondream_device_arg = moondream_device
        self.caption_num_beams = max(1, int(caption_num_beams))

        self._model = None
        self._processor = None
        self._vlm = None
        self._vlm_proc = None
        self._moondream = None
        self._moondream_tok = None
        self._moondream_device = None

        if use_model:
            self._load_captioner(caption_model_name, device)
        if use_vlm:
            self._load_vlm(vlm_model_name)
        if use_moondream:
            self._load_moondream(moondream_model_name, moondream_revision)

    def _load_captioner(self, model_name: str, device: Optional[str]) -> None:
        """Load Florence-2 (window tier), patching its generation quirks."""
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

    def _load_vlm(self, model_name: str) -> None:
        """Load Qwen2-VL / Qwen2.5-VL (event tier) with a capped per-image visual-token budget."""
        import torch
        from transformers import AutoProcessor

        if not hasattr(torch.compiler, "is_compiling"):
            torch.compiler.is_compiling = lambda: False

        mn = model_name.lower()
        if "qwen2.5-vl" in mn or "qwen2_5_vl" in mn:
            from transformers import Qwen2_5_VLForConditionalGeneration as VlmCls
        elif "qwen2-vl" in mn or "qwen2_vl" in mn:
            from transformers import Qwen2VLForConditionalGeneration as VlmCls
        else:
            raise ValueError(
                f"Unsupported VLM {model_name!r}; pass a Qwen2-VL-* or Qwen2.5-VL-* Instruct id."
            )

        dev = self._vlm_device_arg or _default_torch_device(self.device)
        dtype = _vlm_dtype_for_device(dev)
        print(f"Loading event VLM {model_name} on {dev} ({dtype})…")
        self._vlm_proc = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=self.vlm_image_min_pixels,
            max_pixels=self.vlm_image_max_pixels,
        )
        attn_impl = "sdpa" if dev == "cuda" else "eager"
        self._vlm = (
            VlmCls.from_pretrained(
                model_name,
                dtype=dtype,
                attn_implementation=attn_impl,
            )
            .to(dev)
            .eval()
        )
        print("Event VLM ready.")

    def _load_moondream(self, model_name: str, revision: Optional[str]) -> None:
        """Load Moondream2 (episode tier) — picked over Florence <DETAILED_CAPTION> because it hallucinates names far less."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dev = self._moondream_device_arg or _default_torch_device(self.device)
        dtype = _vlm_dtype_for_device(dev)
        print(f"Loading Moondream {model_name} on {dev} ({dtype})…")
        kwargs = {"trust_remote_code": True, "dtype": dtype}
        if revision:
            kwargs["revision"] = revision
        self._moondream = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(dev).eval()
        tok_kwargs = {}
        if revision:
            tok_kwargs["revision"] = revision
        self._moondream_tok = AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
        self._moondream_device = dev
        print("Moondream ready.")

    def caption_frame_moondream(self, frame: np.ndarray) -> str:
        """per-frame Moondream2 caption — tries query / caption / encode+answer depending on the model revision."""
        import torch
        from PIL import Image

        if self._moondream is None:
            return ""

        pil = Image.fromarray(frame).convert("RGB")
        last_err: Optional[BaseException] = None
        with torch.no_grad(), _suppress_generate_warnings():
            if hasattr(self._moondream, "query"):
                try:
                    out = self._moondream.query(pil, MOONDREAM_FRAME_PROMPT)
                    text = out.get("answer") if isinstance(out, dict) else out
                    if text:
                        return str(text).strip()
                except Exception as exc:
                    last_err = exc
            if hasattr(self._moondream, "caption"):
                try:
                    out = self._moondream.caption(pil, length="normal")
                    text = out.get("caption") if isinstance(out, dict) else out
                    if text:
                        return str(text).strip()
                except Exception as exc:
                    last_err = exc
            if hasattr(self._moondream, "encode_image") and hasattr(self._moondream, "answer_question"):
                try:
                    enc = self._moondream.encode_image(pil)
                    return str(
                        self._moondream.answer_question(
                            enc, MOONDREAM_FRAME_PROMPT, self._moondream_tok
                        )
                    ).strip()
                except Exception as exc:
                    last_err = exc
        if last_err is not None and not getattr(self, "_moondream_warned", False):
            print(f"[SummaryBuilder] Moondream failed ({type(last_err).__name__}); using Florence.")
            self._moondream_warned = True
        return ""

    def caption_frame(
        self,
        frame: np.ndarray,
        task_prompt: Optional[str] = None,
        max_new_tokens: int = 96,
    ) -> str:
        """single-frame Florence-2 caption, or a placeholder if the model isn't loaded."""
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
        """window-tier caption: Florence on the representative frame, time template otherwise."""
        if not self.use_model or self._model is None:
            return self.build_window_note(raw_window.start_time, raw_window.end_time)
        frame = raw_window.representative_frame
        if frame is None:
            return self.build_window_note(raw_window.start_time, raw_window.end_time)
        return self.caption_frame(frame)

    def caption_episode(self, windows: list, start_time: float, end_time: float) -> str:
        """episode-tier summary: Moondream per-frame if loaded, Florence fallback, else stitched window notes."""
        if not windows:
            return f"Episode {start_time:.1f}–{end_time:.1f}s"

        if self._moondream is not None:
            parts: List[str] = []
            total = len(windows)
            for i, w in enumerate(windows, start=1):
                if w.frame is not None:
                    cap = self.caption_frame_moondream(w.frame).strip()
                    if not cap:
                        cap = (w.summary_text or self.build_window_note(w.start_time, w.end_time)).strip()
                else:
                    cap = (w.summary_text or self.build_window_note(w.start_time, w.end_time)).strip()
                tag = f"[t={w.start_time:.1f}s | frame {i}/{total}]"
                parts.append(f"{tag} {cap}")
            body = "\n\n".join(parts)
            return body if len(windows) == 1 else f"Episode {start_time:.1f}–{end_time:.1f}s\n\n{body}"

        if self.use_model and self._model is not None:
            detail_tokens = 256 if self.episode_task_prompt == "<DETAILED_CAPTION>" else 128
            parts = []
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
        """Event-tier summary: Qwen2.5-VL multi-image fusion, template fallback on failure."""
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
                print(f"[SummaryBuilder] VLM fusion failed ({type(exc).__name__}); using template.")

        snippets = " → ".join(s.split(":")[0].strip() for s in episode_summaries[:5])
        return f"Event {start_time:.1f}–{end_time:.1f}s: {snippets}"

    def build_window_note(self, start_time: float, end_time: float) -> str:
        """time-only placeholder when no captioner is loaded."""
        return f"Scene at {start_time:.1f}–{end_time:.1f}s"

    def __call__(
        self,
        entries: list,
        episode_frames: Optional[List[List[np.ndarray]]] = None,
    ) -> str:
        """dispatch: WindowEntry list -> episode caption; EpisodeEntry list -> event summary."""
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
        """Run Qwen2.5-VL over subsampled episode frames + bulleted scene texts."""
        import torch
        from PIL import Image

        all_frames: List[Image.Image] = []
        for frames in episode_frames:
            for f in frames:
                if f is not None:
                    all_frames.append(Image.fromarray(f).convert("RGB"))

        if not all_frames:
            return ""

        if len(all_frames) > self.vlm_max_frames:
            step = len(all_frames) / self.vlm_max_frames
            all_frames = [all_frames[int(i * step)] for i in range(self.vlm_max_frames)]

        lines = []
        capped_texts = episode_texts[:8]
        for i, text in enumerate(capped_texts):
            if episode_time_ranges and i < len(episode_time_ranges):
                s, e = episode_time_ranges[i]
                lines.append(f"- Scene {i + 1} [{s:.1f}s–{e:.1f}s]: {text}")
            else:
                lines.append(f"- Scene {i + 1}: {text}")
        bulleted = "\n".join(lines)

        n_frames = len(all_frames)
        n_scenes = len(capped_texts)
        if n_scenes <= 1:
            target = "roughly **40–80 words**, one short paragraph"
        elif n_scenes <= 3:
            target = "roughly **80–140 words**, one paragraph"
        else:
            target = "roughly **120–220 words**, one or two paragraphs"

        prompt = build_event_vlm_prompt(
            bulleted=bulleted,
            n_frames=n_frames,
            n_scenes=n_scenes,
            target=target,
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

        model_dtype = self._vlm.dtype
        for key in ("pixel_values", "pixel_values_videos"):
            if key in inputs and inputs[key].dtype != model_dtype:
                inputs[key] = inputs[key].to(model_dtype)

        with torch.no_grad(), _suppress_generate_warnings():
            out = self._vlm.generate(
                **inputs,
                max_new_tokens=640,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.15,
                no_repeat_ngram_size=6,
            )

        return self._decode_vlm_new_text(out, inputs).replace("\n", " ")

    def _decode_vlm_new_text(self, out, inputs) -> str:
        """Strip the prompt prefix from Qwen-VL output — plain id-slicing is unsafe on multimodal inputs."""
        proc = self._vlm_proc
        skip_special_tokens = True
        clean_up = False

        gen_list = proc.post_process_image_text_to_text(
            out,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up,
        )
        prompt_list = proc.post_process_image_text_to_text(
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
        tail_list = proc.post_process_image_text_to_text(
            tail.unsqueeze(0),
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up,
        )
        return tail_list[0].strip()

    @staticmethod
    def _stitch_episode(notes: List[str], start_time: float, end_time: float) -> str:
        """Join window notes when no captioner is loaded."""
        if len(notes) == 1:
            return notes[0]
        return f"Episode {start_time:.1f}–{end_time:.1f}s: {'; '.join(notes)}"
