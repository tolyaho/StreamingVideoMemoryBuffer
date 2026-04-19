"""recency-only baseline: keep the last N windows, retrieve by cosine similarity."""
from __future__ import annotations

from collections import deque
from typing import List

import numpy as np

from .data_structures import WindowEntry
from .memory_writer import cosine_sim


class RecentWindowBaseline:
    """keeps only the last `n_windows` WindowEntries with no episodic memory.

    Args:
        n_windows: sliding window size.
    """

    def __init__(self, n_windows: int = 10):
        self.n_windows = n_windows
        self.recent: deque[WindowEntry] = deque(maxlen=n_windows)

    def update(self, window: WindowEntry) -> None:
        self.recent.append(window)

    def get_context(self) -> List[WindowEntry]:
        """return all retained windows in chronological order."""
        return list(self.recent)

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 5) -> List[WindowEntry]:
        """return the top-k most similar recent windows to the query embedding."""
        windows = list(self.recent)
        if not windows:
            return []
        sims = [cosine_sim(query_embedding, w.visual_embedding) for w in windows]
        top_idxs = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
        return [windows[i] for i in top_idxs]

    def stats(self) -> dict:
        return {"recent": len(self.recent), "capacity": self.n_windows}
