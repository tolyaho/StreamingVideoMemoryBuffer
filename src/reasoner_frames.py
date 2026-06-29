from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .data_structures import RetrievalResult, WindowEntry


@dataclass(frozen=True)
class FrameEvidence:
    frame: np.ndarray
    start_time: float
    end_time: float
    source: str
    summary: Optional[str] = None
    entry_id: Optional[str] = None


def _window_evidence(w: WindowEntry, source: str) -> Optional[FrameEvidence]:
    if w.frame is None:
        return None
    frame = w.frame.copy() if hasattr(w.frame, "copy") else w.frame
    return FrameEvidence(
        frame=frame,
        start_time=w.start_time,
        end_time=w.end_time,
        source=source,
        summary=w.summary_text,
        entry_id=w.entry_id,
    )


def collect_frames(
    result: RetrievalResult,
    *,
    max_frames: int = 8,
) -> List[FrameEvidence]:
    """Retrieved grounding frames first, then pinned recent — text context is separate."""
    max_frames = max(1, int(max_frames))
    picked: List[FrameEvidence] = []
    seen: set[str] = set()

    def add_window(w: WindowEntry, source: str) -> bool:
        if len(picked) >= max_frames or w.entry_id in seen:
            return False
        ev = _window_evidence(w, source)
        if ev is None:
            return False
        seen.add(w.entry_id)
        picked.append(ev)
        return True

    for w in result.grounded_windows:
        if add_window(w, "grounding_window"):
            if len(picked) >= max_frames:
                return picked

    pinned: List[FrameEvidence] = []
    for w in reversed(result.pinned_windows):
        if len(picked) + len(pinned) >= max_frames:
            break
        ev = _window_evidence(w, "pinned_recent")
        if ev is not None and w.entry_id not in seen:
            seen.add(w.entry_id)
            pinned.append(ev)
    pinned.reverse()
    picked.extend(pinned)

    return picked
