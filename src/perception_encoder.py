"""X-CLIP video-language encoder for window clips and text queries.

Both outputs are L2-normalised 512-dim vectors in the same joint space.
Each window is encoded as a clip of num_frames uniformly sampled from its
frame list, giving a temporally-aware embedding that captures short-term
motion rather than a single still-frame appearance.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


def _sample_uniform(frames: List[np.ndarray], n: int) -> List[np.ndarray]:
    """uniformly sample n frames, repeating boundary frames if the clip is shorter."""
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)
    return [frames[i] for i in indices]


class PerceptionEncoder:
    """wraps X-CLIP to encode short video clips and text into a shared embedding space."""

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
        """uniformly sample num_frames from the window and return a clip embedding."""
        return self.encode_frames(raw_window.frames)

    def encode_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """encode a list of RGB uint8 frames as a single clip; returns L2-normalised embedding."""
        import torch
        from PIL import Image

        if not frames:
            raise ValueError("Empty frame list")

        clip = _sample_uniform(frames, self.num_frames)
        pil_frames = [Image.fromarray(f) for f in clip]

        # XCLIPProcessor(videos=...) returns empty dict without text; use image_processor directly
        pixel_values = self.processor.image_processor(pil_frames, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(self.device)   # (1, num_frames, C, H, W)

        with torch.no_grad():
            # Inline get_video_features so we can pass return_dict=True to MIT.
            # get_video_features() calls self.mit(cls_features) without return_dict,
            # which makes MIT return a tuple; .pooler_output then fails.
            B, T, C, H, W = pixel_values.shape
            vision_out = self.model.vision_model(pixel_values=pixel_values.reshape(-1, C, H, W))
            video_embeds = self.model.visual_projection(vision_out.pooler_output)
            mit_out = self.model.mit(video_embeds.view(B, T, -1), return_dict=True)
            feats = mit_out.pooler_output
            feats = feats / feats.norm(dim=-1, keepdim=True)

        return feats.cpu().numpy()[0].astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        """encode a text string; returns an L2-normalised embedding."""
        import torch

        # XCLIPProcessor(text=...) also returns empty dict; use tokenizer directly
        inputs = self.processor.tokenizer([text], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            feats = self.model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        return feats.cpu().numpy()[0].astype(np.float32)
