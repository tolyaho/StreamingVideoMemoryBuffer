"""generates text summaries at window, episode, and event granularity.

three modes (mixable):
  template only       — fast, no model load
  + captioner         — Florence-2-base, "<DETAILED_CAPTION>" for episodes
  + captioner + llm   — adds Qwen2.5-1.5B-Instruct for event fusion
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class SummaryBuilder:
    """generates text summaries for memory entries.

    Args:
        use_model: if True, load Florence-2 for visual captioning.
        use_llm: if True, load Qwen2.5-1.5B-Instruct for event fusion.
        caption_model_name: HuggingFace id for the Florence-2 captioner.
        llm_model_name: HuggingFace id for the event fusion LLM.
        task_prompt: default Florence-2 task token for single frames.
        episode_task_prompt: Florence-2 task token for episode representative frames.
        device: torch device. auto-detected if None.
    """

    def __init__(
        self,
        use_model: bool = False,
        use_llm: bool = False,
        caption_model_name: str = "microsoft/Florence-2-base",
        llm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        task_prompt: str = "<CAPTION>",
        episode_task_prompt: str = "<DETAILED_CAPTION>",
        device: Optional[str] = None,
    ):
        self.use_model = use_model
        self.use_llm = use_llm
        self.caption_model_name = caption_model_name
        self.llm_model_name = llm_model_name
        self.task_prompt = task_prompt
        self.episode_task_prompt = episode_task_prompt
        self.device: Optional[str] = device

        self._model = None
        self._processor = None
        self._llm = None
        self._llm_tok = None

        if use_model:
            self._load_captioner(caption_model_name, device)
        if use_llm:
            self._load_llm(llm_model_name, device)

    def _load_captioner(self, model_name: str, device: Optional[str]) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self.device = device or self.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"Loading captioner {model_name} on {self.device} ({dtype})…")
        self._processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True
        )
        self._model = (
            AutoModelForCausalLM.from_pretrained(
                model_name, trust_remote_code=True, torch_dtype=dtype
            )
            .to(self.device)
            .eval()
        )
        print("Captioner ready.")

    def _load_llm(self, model_name: str, device: Optional[str]) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or self.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"Loading event LLM {model_name} on {self.device} ({dtype})…")
        self._llm_tok = AutoTokenizer.from_pretrained(model_name)
        self._llm = (
            AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
            .to(self.device)
            .eval()
        )
        print("Event LLM ready.")

    def caption_frame(
        self,
        frame: np.ndarray,
        task_prompt: Optional[str] = None,
    ) -> str:
        """generate a caption for a single RGB uint8 frame via Florence-2."""
        if not self.use_model or self._model is None:
            return "visual scene"

        import torch
        from PIL import Image

        prompt = task_prompt or self.task_prompt

        pil = Image.fromarray(frame).convert("RGB")
        inputs = self._processor(
            text=prompt, images=pil, return_tensors="pt"
        ).to(self.device)

        pixel_values = inputs["pixel_values"]
        if pixel_values.dtype != self._model.dtype:
            pixel_values = pixel_values.to(self._model.dtype)

        with torch.no_grad():
            out = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=pixel_values,
                max_new_tokens=96,
                num_beams=3,
                do_sample=False,
            )

        raw = self._processor.batch_decode(out, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(
            raw, task=prompt, image_size=(pil.width, pil.height)
        )
        caption = str(parsed.get(prompt, "")).strip()
        return caption or "visual scene"

    def _fuse_with_llm(self, episode_texts: List[str]) -> str:
        """fuse episode summaries into one event sentence via Qwen."""
        if not self.use_llm or self._llm is None:
            return ""

        import torch

        bulleted = "\n".join(f"- {t}" for t in episode_texts[:8])
        user_msg = (
            "You are summarising a stream of short scene descriptions. "
            "Fuse the bullet points below into ONE concise sentence (≤25 words) "
            "that describes the overall event. Use only information present in "
            "the bullets; do not invent entities, places, or actions. "
            "Return the sentence only, no preface.\n\n"
            f"Scenes:\n{bulleted}"
        )
        messages = [{"role": "user", "content": user_msg}]
        text = self._llm_tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._llm_tok(text, return_tensors="pt").to(self._llm.device)
        with torch.no_grad():
            out = self._llm.generate(
                **inputs,
                max_new_tokens=80,
                do_sample=False,
                num_beams=1,
            )
        generated = out[0][inputs["input_ids"].shape[1]:]
        response = self._llm_tok.decode(
            generated, skip_special_tokens=True
        ).strip()
        return response.replace("\n", " ").strip()

    def build_window_note(self, start_time: float, end_time: float) -> str:
        """one-line template note for a single window."""
        return f"Scene at {start_time:.1f}–{end_time:.1f}s"

    def build_episode_summary(
        self,
        window_notes: List[str],
        start_time: float,
        end_time: float,
    ) -> str:
        """template episode summary used when captioning isn't available."""
        if len(window_notes) == 1:
            return window_notes[0]
        joined = "; ".join(window_notes[:4])
        return f"Episode {start_time:.1f}–{end_time:.1f}s: {joined}"

    def build_event_summary(
        self,
        episode_summaries: List[str],
        start_time: float,
        end_time: float,
    ) -> str:
        """template event summary used when the LLM isn't available."""
        snippets = " → ".join(
            s.split(":")[0].strip() for s in episode_summaries[:5]
        )
        return f"Event {start_time:.1f}–{end_time:.1f}s: {snippets}"

    def caption_episode(
        self,
        windows: list,
        start_time: float,
        end_time: float,
    ) -> str:
        """caption the representative window with Florence-2 <DETAILED_CAPTION>."""
        if not windows:
            return f"Episode {start_time:.1f}–{end_time:.1f}s"

        if not self.use_model or self._model is None:
            notes = [self.build_window_note(w.start_time, w.end_time) for w in windows]
            return self.build_episode_summary(notes, start_time, end_time)

        rep = self._representative_window(windows)
        if rep is None or rep.frame is None:
            notes = [self.build_window_note(w.start_time, w.end_time) for w in windows]
            return self.build_episode_summary(notes, start_time, end_time)

        caption = self.caption_frame(rep.frame, task_prompt=self.episode_task_prompt)
        return f"Episode {start_time:.1f}–{end_time:.1f}s: {caption}"

    def summarize_event(
        self,
        episode_summaries: List[str],
        start_time: float,
        end_time: float,
    ) -> str:
        """fuse episode summaries into an event sentence via Qwen, with template fallback."""
        if not episode_summaries:
            return f"Event {start_time:.1f}–{end_time:.1f}s"

        if self.use_llm and self._llm is not None:
            try:
                fused = self._fuse_with_llm(episode_summaries)
            except Exception as exc:
                print(f"[SummaryBuilder] LLM fusion failed: {exc!r}; using template.")
                fused = ""
            if fused:
                return f"Event {start_time:.1f}–{end_time:.1f}s: {fused}"

        return self.build_event_summary(episode_summaries, start_time, end_time)

    def __call__(self, entries: list) -> str:
        """dispatch: list of WindowEntry -> episode summary, list of EpisodeEntry -> event summary."""
        from .data_structures import EpisodeEntry, WindowEntry

        if not entries:
            return "empty"

        start = entries[0].start_time
        end = entries[-1].end_time

        if isinstance(entries[0], WindowEntry):
            return self.caption_episode(entries, start, end)
        if isinstance(entries[0], EpisodeEntry):
            return self.summarize_event(
                [e.summary_text for e in entries], start, end
            )
        return f"Memory {start:.1f}–{end:.1f}s"

    @staticmethod
    def _representative_window(windows: list):
        """pick the window whose embedding is closest to the centroid of all members."""
        if not windows:
            return None
        if len(windows) == 1:
            return windows[0]
        embs = np.stack([w.visual_embedding for w in windows])
        centroid = embs.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-8
        sims = embs @ centroid
        return windows[int(np.argmax(sims))]
