"""generates text summaries at window, episode, and event granularity.

Three models, chosen by tier (all optional; each can be toggled independently):
  window (ingest)      — Florence-2 <CAPTION> on the representative frame (`use_model`)
  episode (flush)      — Moondream2 per-member-frame grounded caption with
                         anti-hallucination prompt (`use_moondream`); falls back to
                         Florence <DETAILED_CAPTION> only if Moondream is disabled
  event (consolidation) — Qwen2.5-VL (default 3B) multi-image fusion over episode
                         texts + sampled frames (`use_vlm`)
When none are enabled, every tier falls through to a time-template string.
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

    if device in ("cuda", "mps"):
        return torch.float16
    return torch.bfloat16  # halves CPU memory vs float32; supported on modern CPUs


def _vlm_dtype_for_device(device: str):
    """Qwen2-VL / Qwen2.5-VL must run in bf16: fp16 overflows in the vision tower and
    the model generates repeated token 0 (rendered as ``!``). Fall back to fp16 only if
    bf16 is unsupported on this hardware."""
    import torch

    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device == "mps":
        # MPS bf16 lands in PyTorch 2.1+. Older builds must use fp16 (known-garbage risk).
        return torch.bfloat16
    return torch.bfloat16


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
    """Window / episode / event text summaries.

    Args:
        use_model: load Florence-2 for window captions.
        use_vlm: load Qwen2-VL or Qwen2.5-VL (default Qwen2.5-VL-3B-Instruct) for event fusion.
        use_moondream: load Moondream2 for grounded per-frame episode captions.
        device / vlm_device / moondream_device: torch devices, auto-detected when None;
            on MPS the secondary models default to CPU to keep both off the unified heap.
        caption_num_beams: Florence generate beam width (CUDA only; MPS/CPU uses greedy).
    """

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

    def _resolve_vlm_device(self) -> str:
        if self._vlm_device_arg is not None:
            return self._vlm_device_arg
        # Florence + Qwen on the same MPS heap often OOM; CPU for VLM is slower but stable.
        if self.device == "mps" and self._model is not None:
            print(
                "Placing event VLM on CPU (Florence on MPS — both on MPS usually exceeds "
                "unified memory). Pass vlm_device='mps' to force, or free GPU memory."
            )
            return "cpu"
        return _default_torch_device(self.device)

    def _load_vlm(self, model_name: str) -> None:
        import torch
        from transformers import AutoProcessor

        # Qwen2-VL / Qwen2.5-VL call torch.compiler.is_compiling() which was added in PyTorch 2.1
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

        dev = self._resolve_vlm_device()
        dtype = _vlm_dtype_for_device(dev)
        print(f"Loading event VLM {model_name} on {dev} ({dtype})…")
        # Cap the per-image visual-token budget. Qwen2-VL's default max_pixels ≈ 12.8 M
        # produces up to ~16k patches per image; eager attention over many frames blows up.
        self._vlm_proc = AutoProcessor.from_pretrained(
            model_name,
            min_pixels=self.vlm_image_min_pixels,
            max_pixels=self.vlm_image_max_pixels,
        )
        # SDPA avoids materialising the full N×N attention matrix that eager does.
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

    def _resolve_moondream_device(self) -> str:
        if self._moondream_device_arg is not None:
            return self._moondream_device_arg
        if self.device == "mps" and self._model is not None:
            return "cpu"
        return _default_torch_device(self.device)

    def _load_moondream(self, model_name: str, revision: Optional[str]) -> None:
        """Moondream2: single-image captioner + VQA. Used for episode-level
        captions where Florence's <DETAILED_CAPTION> tends to hallucinate
        named entities (teams, players, scores, seasons) from misread overlays.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dev = self._resolve_moondream_device()
        dtype = _vlm_dtype_for_device(dev)  # Moondream prefers bf16 on CUDA
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

    _MOONDREAM_PROMPT = (
        "You are looking at a single still frame extracted from a continuous video, so "
        "some motion may be frozen mid-action. Describe this frame in 2–4 sentences "
        "focused only on what is clearly visible, including any apparent motion or pose "
        "evident from the pixels (body position, blur, trajectory). Do NOT name specific "
        "people, organizations, locations, events, brands, or on-screen text unless you "
        "can read or identify them confidently and completely. When identities or text "
        "are unclear, use neutral generic descriptions instead of guessing. Prefer "
        "concrete visual detail (colours, clothing, positions, objects, apparent motion) "
        "over narrative speculation about what happened before or after this frame."
    )

    def caption_frame_moondream(self, frame: np.ndarray) -> str:
        """Grounded per-frame caption via Moondream2. Flexible across model revisions."""
        import torch
        from PIL import Image

        if self._moondream is None:
            return ""

        pil = Image.fromarray(frame).convert("RGB")
        last_err: Optional[BaseException] = None
        with torch.no_grad(), _suppress_generate_warnings():
            # Newest revisions: model.query(image, prompt) -> {"answer": str}
            if hasattr(self._moondream, "query"):
                try:
                    out = self._moondream.query(pil, self._MOONDREAM_PROMPT)
                    text = out.get("answer") if isinstance(out, dict) else out
                    if text:
                        return str(text).strip()
                except Exception as exc:
                    last_err = exc
            # Mid-generation revisions: model.caption(image, length=...) -> {"caption": str}
            if hasattr(self._moondream, "caption"):
                try:
                    out = self._moondream.caption(pil, length="normal")
                    text = out.get("caption") if isinstance(out, dict) else out
                    if text:
                        return str(text).strip()
                except Exception as exc:
                    last_err = exc
            # Legacy API: encode_image + answer_question
            if hasattr(self._moondream, "encode_image") and hasattr(self._moondream, "answer_question"):
                try:
                    enc = self._moondream.encode_image(pil)
                    return str(
                        self._moondream.answer_question(
                            enc, self._MOONDREAM_PROMPT, self._moondream_tok
                        )
                    ).strip()
                except Exception as exc:
                    last_err = exc
        if last_err is not None and not getattr(self, "_moondream_warned", False):
            print(
                f"[SummaryBuilder] Moondream call failed ({last_err!r}); falling back to "
                "Florence for episode captions. If this is a torch<2.5 / enable_gqa issue, "
                "upgrade torch or pass moondream_revision='2024-08-26'."
            )
            self._moondream_warned = True
        return ""

    def caption_frame(
        self,
        frame: np.ndarray,
        task_prompt: Optional[str] = None,
        max_new_tokens: int = 96,
    ) -> str:
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
        if not self.use_model or self._model is None:
            return self.build_window_note(raw_window.start_time, raw_window.end_time)
        frame = raw_window.representative_frame
        if frame is None:
            return self.build_window_note(raw_window.start_time, raw_window.end_time)
        return self.caption_frame(frame)

    def caption_episode(self, windows: list, start_time: float, end_time: float) -> str:
        if not windows:
            return f"Episode {start_time:.1f}–{end_time:.1f}s"

        # Preferred path: Moondream for episode captions — it hallucinates
        # far fewer named entities than Florence's <DETAILED_CAPTION>.
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

        # Fallback: Florence (original behaviour).
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

        # Subsample evenly if we exceed the configured per-event frame cap.
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
        # Adaptive word target — short for sparse/repetitive events, long only
        # when there is genuine variety to describe. Prevents the model from
        # padding with a duplicated closing sentence.
        if n_scenes <= 1:
            target = "roughly **40–80 words**, one short paragraph"
        elif n_scenes <= 3:
            target = "roughly **80–140 words**, one paragraph"
        else:
            target = "roughly **120–220 words**, one or two paragraphs"

        prompt = self._build_event_vlm_prompt(
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

        # Qwen2-VL/Qwen2.5-VL: pixel_values must match model dtype, otherwise the vision
        # tower returns garbage and the LM emits repeated token 0 ("!!!!!..."). ``.to(device)``
        # on a BatchFeature does not cast dtype.
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

    @staticmethod
    def _build_event_vlm_prompt(
        *,
        bulleted: str,
        n_frames: int,
        n_scenes: int,
        target: str,
    ) -> str:
        return (
            "You are producing a detailed summary of a single continuous video event that is made up "
            "of several shorter scenes.\n\n"
            "## Output language\n"
            "Write the entire summary in **English only**. Do **not** switch to Chinese or any other "
            "language at any point, even if on-screen text, captions, or image content contain "
            "non-English words. If you need to refer to visible foreign text, describe it in English "
            "(e.g. 'a sign with Chinese characters') rather than reproducing it.\n\n"
            "## What you receive\n"
            f"- **{n_frames} still images**, supplied in chronological order. They are the most "
            "representative frames sampled from the scenes that make up this event (roughly 1–2 "
            "frames per scene, evenly spread in time).\n"
            f"- **{n_scenes} scene descriptions** below, one per scene, each prefixed with its time "
            "range in seconds from the start of the video. The scenes are listed in chronological "
            "order and together cover the whole event.\n"
            "- The images are the **authoritative evidence**. The scene descriptions come from a "
            "small auto-captioner and are known to **hallucinate named entities** — specific team "
            "names, player names, scores, dates, seasons, leagues, brand names, and on-screen text. "
            "Use the captions only for general shape, action, and setting. If a caption names a "
            "specific team, player, league, match, score, date, or logo that you cannot **clearly "
            "verify from the images**, do NOT repeat that claim; describe it generically instead "
            "(e.g. 'a football match', 'a player in a red-and-blue striped jersey', 'a scoreboard').\n\n"
            "## Your task\n"
            f"Write a **detailed, cohesive summary** of the event as prose ({target}, no bullet "
            "lists, no headings, no numbered steps).\n"
            "Make it **high-information-density**: keep the rich context, but pack each sentence with "
            "distinct visual facts instead of broad atmosphere or commentary.\n"
            "Put the most query-useful facts early: who/what is present, where the event happens, the "
            "main actions, visible objects, and any clear state changes over time.\n"
            "Then cover, in natural narrative order:\n"
            "1. **Setting & context** — where the event takes place and any reliable on-screen "
            "graphics or text.\n"
            "2. **Subjects & objects** — the main people, animals, vehicles, or objects that appear, "
            "including recognisable clothing, colours, or distinctive features when visible.\n"
            "3. **What happens over time** — the key actions and how they unfold; refer to earlier vs. "
            "later moments only when it clarifies the progression.\n"
            "4. **Transitions & continuity** — how the focus, location, or activity shifts between "
            "scenes, and what stays the same.\n"
            "5. **Overall takeaway** — one short closing sentence naming what the event is about as a "
            "whole.\n\n"
            "## Rules\n"
            "- Stay **strictly grounded**: describe only what the images and scene descriptions "
            "actually support. If something is ambiguous, say so briefly ('appears to', 'likely') "
            "rather than inventing a specific identity, name, place, or action.\n"
            "- Prefer concrete visual detail over generic phrasing.\n"
            "- Do **not** list the scenes mechanically ('Scene 1 shows..., Scene 2 shows...'); weave "
            "them into one flowing narrative.\n"
            "- Do **not** repeat the scene descriptions verbatim; synthesise them.\n"
            "- **Avoid filler** such as mood, professionalism, intensity, or cinematic commentary "
            "unless it is directly visible and important for understanding the event.\n"
            "- If the scenes are visually repetitive or contain little new information, compress them "
            "briefly instead of restating similar actions.\n"
            "- The closing takeaway sentence must add a new synthesising observation. Do **not** "
            "re-describe the last scene or repeat what an earlier sentence already said.\n"
            "- Each sentence must introduce information not already stated. Avoid near-duplicate "
            "phrases.\n"
            "- Return **only** the summary prose — no title, preamble, bullet points, numbering, "
            "or meta-commentary about the task.\n\n"
            f"## Scene descriptions\n{bulleted}"
        )

    def _decode_vlm_new_text(self, out, inputs) -> str:
        """Strip the prompt from generate() output.

        Plain slicing ``out[:, len(prompt):]`` is unreliable for Qwen2-VL / Qwen2.5-VL:
        multimodal ``input_ids`` do not always align 1:1 with the produced sequence the way
        text-only models do, and decoding that slice can yield garbage (e.g. repeated ``!``).
        Hugging Face's image-text pipelines decode full output + prompt, then remove the
        decoded prompt prefix (see ``ImageTextToTextPipeline.postprocess``).
        """
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
        if len(notes) == 1:
            return notes[0]
        return f"Episode {start_time:.1f}–{end_time:.1f}s: {'; '.join(notes)}"
