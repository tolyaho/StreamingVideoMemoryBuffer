from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class WindowEntry:
    """smallest stored memory unit — a short local time window of sampled frames."""

    entry_id: str
    start_time: float
    end_time: float
    visual_embedding: np.ndarray   # shape (D,), L2-normalised
    frame: Optional[np.ndarray] = None
    summary_text: Optional[str] = None
    summary_embedding: Optional[np.ndarray] = None
    tier: str = "recent"           # "recent" | "episodic"

    @classmethod
    def from_raw_window(
        cls,
        raw_window,
        visual_embedding: np.ndarray,
        summary_text: Optional[str] = None,
        summary_embedding: Optional[np.ndarray] = None,
        tier: str = "recent",
    ) -> "WindowEntry":
        """bridge from StreamReader.RawWindow to WindowEntry."""
        frame = None
        if hasattr(raw_window, "representative_frame"):
            frame = raw_window.representative_frame
        elif getattr(raw_window, "frames", None):
            frame = raw_window.frames[len(raw_window.frames) // 2]
            if hasattr(frame, "copy"):
                frame = frame.copy()

        return cls(
            entry_id=raw_window.window_id,
            start_time=raw_window.start_time,
            end_time=raw_window.end_time,
            visual_embedding=visual_embedding,
            frame=frame,
            summary_text=summary_text,
            summary_embedding=summary_embedding,
            tier=tier,
        )


@dataclass
class EpisodeEntry:
    """coherent action span built from consecutive novel windows."""

    entry_id: str
    start_time: float
    end_time: float
    visual_embedding: np.ndarray   # self-centrality pooled embedding of member windows
    member_window_ids: List[str]
    summary_text: str
    summary_embedding: Optional[np.ndarray] = None
    representative_window_ids: List[str] = field(default_factory=list)  # top-weight windows from pooling


@dataclass
class EventEntry:
    """high-level activity cluster built from consecutive episodes."""

    entry_id: str
    start_time: float
    end_time: float
    visual_embedding: np.ndarray   # centroid of member episode embeddings
    member_episode_ids: List[str]
    representative_window_ids: List[str]
    summary_text: str
    summary_embedding: Optional[np.ndarray] = None


@dataclass
class RetrievalResult:
    """bundle returned by the hierarchical retriever after a query."""

    query: str
    coarse_hits: List[EventEntry]
    episodic_hits: List[EpisodeEntry]
    grounded_windows: List[WindowEntry]
    scores: dict  # entry_id -> float score
