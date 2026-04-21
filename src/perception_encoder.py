from __future__ import annotations

from typing import List, Optional

import numpy as np


def _sample_uniform(frames: List[np.ndarray], n: int) -> List[np.ndarray]:
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)
    return [frames[i] for i in indices]


def _chunk_token_ids(ids: List[int], chunk_size: int) -> List[List[int]]:
    """Split a flat id stream into fixed-size chunks, preserving order.

    Empty input yields ``[[]]`` so the caller can still emit one forward pass
    (wrapped with BOS/EOS) and get a defined embedding for an empty string.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not ids:
        return [[]]
    return [ids[i : i + chunk_size] for i in range(0, len(ids), chunk_size)]


class PerceptionEncoder:
    def __init__(
        self,
        model_name: str = "microsoft/xclip-base-patch32",
        device: Optional[str] = None,
        num_frames: int = 8,
    ):
        import torch
        from transformers import XCLIPModel, XCLIPProcessor

        self.num_frames = num_frames
        self.device = device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        print(f"Loading {model_name} on {self.device}...")
        self.model = XCLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = XCLIPProcessor.from_pretrained(model_name)
        self._dim = self.model.config.projection_dim
        print(f"Ready. Embedding dim = {self._dim}.")

    @property
    def dim(self) -> int:
        return self._dim

    def encode_window(self, raw_window) -> np.ndarray:
        return self.encode_frames(raw_window.frames)

    def encode_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        import torch
        from PIL import Image

        if not frames:
            raise ValueError("Empty frame list")

        clip = _sample_uniform(frames, self.num_frames)
        pil_frames = [Image.fromarray(f) for f in clip]

        pixel_values = self.processor.image_processor(pil_frames, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.device)   # (1, num_frames, C, H, W)

        with torch.no_grad():
            B, T, C, H, W = pixel_values.shape
            vision_out = self.model.vision_model(pixel_values=pixel_values.reshape(-1, C, H, W))
            video_embeds = self.model.visual_projection(vision_out.pooler_output)
            mit_out = self.model.mit(video_embeds.view(B, T, -1), return_dict=True)
            feats = mit_out.pooler_output
            feats = feats / feats.norm(dim=-1, keepdim=True)

        return feats.cpu().numpy()[0].astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        """Encode ``text`` into a single L2-normalised embedding.

        X-CLIP's text encoder inherits CLIP's 77-token position limit.
        Long event summaries (hundreds of tokens) blow past it, so we tokenize
        without truncation, split into ≤75-token content chunks (2 slots
        reserved for BOS/EOS), batch-forward, unit-normalise each chunk
        embedding, mean-pool, and renormalise the result.
        """
        import torch

        tokenizer = self.processor.tokenizer
        max_len = tokenizer.model_max_length
        bos_id = tokenizer.bos_token_id
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
        chunk_content = max_len - 2  # reserve 2 slots for BOS + EOS

        # HF emits a "sequence longer than model_max_length" warning for every
        # oversized input — noisy, and wrong in our context since we chunk.
        from transformers import logging as hf_logging
        prev_verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            raw_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        finally:
            hf_logging.set_verbosity(prev_verbosity)
        chunks = _chunk_token_ids(raw_ids, chunk_size=chunk_content)

        wrapped = [[bos_id] + c + [eos_id] for c in chunks]
        lengths = [len(x) for x in wrapped]
        L = max(lengths)
        padded = [x + [pad_id] * (L - len(x)) for x in wrapped]
        attn = [[1] * n + [0] * (L - n) for n in lengths]

        input_ids = torch.tensor(padded, device=self.device, dtype=torch.long)
        attention_mask = torch.tensor(attn, device=self.device, dtype=torch.long)

        with torch.no_grad():
            feats = self.model.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask
            )
            feats = feats / feats.norm(dim=-1, keepdim=True)
            pooled = feats.mean(dim=0)
            pooled = pooled / (pooled.norm() + 1e-8)

        return pooled.cpu().numpy().astype(np.float32)
