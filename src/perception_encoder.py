"""SigLIP ViT-B/16 encoder for frames and text queries.

Both outputs are L2-normalised 768-dim vectors in the same joint space,
enabling cosine similarity for retrieval without any projection.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class PerceptionEncoder:
    """wraps SigLIP to encode frames and text into a shared embedding space."""

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
        device: Optional[str] = None,
        batch_size: int = 8,
    ):
        import torch
        from transformers import SiglipModel, SiglipProcessor

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        self.batch_size = batch_size
        self._dim: Optional[int] = None

        print(f"Loading {model_name} on {self.device}...")
        self.model = SiglipModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = SiglipProcessor.from_pretrained(model_name)
        with torch.no_grad():
            dummy = self.processor(text=["test"], return_tensors="pt", padding="max_length")
            dummy = {k: v.to(self.device) for k, v in dummy.items()}
            self._dim = self.model.get_text_features(**dummy).shape[-1]
        print(f"Ready. Embedding dim = {self._dim}.")

    @property
    def dim(self) -> int:
        return self._dim or 768

    def encode_frames(self, frames: List[np.ndarray]) -> np.ndarray:
        """encode a list of RGB uint8 frames; returns their mean L2-normalised embedding."""
        import torch
        from PIL import Image

        if not frames:
            raise ValueError("Empty frame list")

        pil_images = [Image.fromarray(f) for f in frames]
        all_embeds = []

        with torch.no_grad():
            for i in range(0, len(pil_images), self.batch_size):
                batch = pil_images[i: i + self.batch_size]
                inputs = self.processor(images=batch, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                feats = self.model.get_image_features(**inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                all_embeds.append(feats.cpu().numpy())

        embeddings = np.concatenate(all_embeds, axis=0)
        mean = embeddings.mean(axis=0)
        mean = mean / (np.linalg.norm(mean) + 1e-8)
        return mean.astype(np.float32)

    def encode_text(self, text: str) -> np.ndarray:
        """encode a text string; returns an L2-normalised embedding."""
        import torch

        with torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt", padding="max_length")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            feats = self.model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()[0].astype(np.float32)

    def encode_window(self, raw_window) -> np.ndarray:
        """convenience wrapper: encode a RawWindow's frame list."""
        return self.encode_frames(raw_window.frames)


class MockEncoder:
    """drop-in replacement for PerceptionEncoder using pre-designed scene embeddings.

    useful for demos and tests where downloading SigLIP is undesirable.
    default dim=768 matches SigLIP ViT-B/16.
    """

    def __init__(
        self,
        dim: int = 768,
        n_scenes: int = 4,
        noise_std: float = 0.08,
        seed: int = 0,
    ):
        self._dim = dim
        self.noise_std = noise_std
        self._rng = np.random.default_rng(seed)

        rng0 = np.random.default_rng(seed)
        raw = rng0.standard_normal((n_scenes, dim)).astype(np.float32)
        self._scene_centroids = raw / np.linalg.norm(raw, axis=1, keepdims=True)

        self._query_map: dict[str, int] = {}

    @property
    def dim(self) -> int:
        return self._dim

    def add_query(self, text: str, scene_idx: int) -> None:
        """map a query string to a scene index for controlled retrieval demos."""
        self._query_map[text] = scene_idx

    def encode_scene(self, scene_idx: int) -> np.ndarray:
        """return a noisy embedding for a given scene index."""
        base = self._scene_centroids[scene_idx % len(self._scene_centroids)]
        noise = self._rng.standard_normal(self._dim).astype(np.float32) * self.noise_std
        v = base + noise
        return v / (np.linalg.norm(v) + 1e-8)

    def encode_frames(self, frames: List[np.ndarray], scene_idx: int = 0) -> np.ndarray:
        return self.encode_scene(scene_idx)

    def encode_text(self, text: str) -> np.ndarray:
        if text in self._query_map:
            base = self._scene_centroids[self._query_map[text]]
            return (base / (np.linalg.norm(base) + 1e-8)).astype(np.float32)
        mean = self._scene_centroids.mean(axis=0)
        return (mean / (np.linalg.norm(mean) + 1e-8)).astype(np.float32)

    def encode_window(self, raw_window, scene_idx: int = 0) -> np.ndarray:
        return self.encode_scene(scene_idx)
