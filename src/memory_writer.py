"""three-tier hierarchical memory buffer updated online as windows arrive.

tier 1 (recent): fixed-capacity deque of the most recent WindowEntries.
tier 2 (episodic): coherent action spans built from novel promoted windows.
tier 3 (long-term): compressed EventEntries consolidated from the oldest episodes.

write path is query-agnostic — storage is driven by recency, novelty, and capacity.
"""
from __future__ import annotations

import uuid
from collections import deque
from typing import Callable, List, Optional

import numpy as np

from .data_structures import EpisodeEntry, EventEntry, WindowEntry


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


class HierarchicalMemoryWriter:
    """maintains three memory tiers and updates them on each incoming window.

    Args:
        recent_capacity: max WindowEntries in recent memory.
        episodic_capacity: max EpisodeEntries before consolidation is triggered.
        novelty_threshold: min cosine distance (1 - sim) to promote a window.
        episode_max_gap: max seconds between windows to merge into one episode.
        episode_min_sim: min cosine sim to the episode centroid to keep merging.
        episode_max_len: hard cap on windows per episode. None = no cap.
        event_max_gap: max seconds between episodes to join the same event.
        event_min_episode_sim: min cosine sim to the running event centroid.
        episodic_merge_batch: max episodes merged into one EventEntry.
        summary_fn: (entries: list) -> str. falls back to time templates if None.
        text_encode_fn: (text: str) -> np.ndarray. encodes summaries for retrieval.
    """

    def __init__(
        self,
        recent_capacity: int = 20,
        episodic_capacity: int = 50,
        novelty_threshold: float = 0.25,
        episode_max_gap: float = 10.0,
        episode_min_sim: float = 0.7,
        episode_max_len: Optional[int] = 8,
        event_max_gap: float = 45.0,
        event_min_episode_sim: float = 0.55,
        episodic_merge_batch: int = 10,
        summary_fn: Optional[Callable] = None,
        text_encode_fn: Optional[Callable[[str], np.ndarray]] = None,
    ):
        self.recent_capacity = recent_capacity
        self.episodic_capacity = episodic_capacity
        self.novelty_threshold = novelty_threshold
        self.episode_max_gap = episode_max_gap
        self.episode_min_sim = episode_min_sim
        self.episode_max_len = episode_max_len
        self.event_max_gap = event_max_gap
        self.event_min_episode_sim = event_min_episode_sim
        self.episodic_merge_batch = episodic_merge_batch
        self._summary_fn = summary_fn
        self._text_encode_fn = text_encode_fn

        self.recent: deque[WindowEntry] = deque()
        self.episodic: List[EpisodeEntry] = []
        self.long_term: List[EventEntry] = []
        self._window_archive: dict[str, WindowEntry] = {}

        # open episode buffer — flushed on scene break or finalize()
        self._pending_episode: List[WindowEntry] = []

        self._n_discarded = 0
        self._n_promoted = 0
        self._n_episodes_flushed = 0
        self._n_consolidated = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, window: WindowEntry) -> None:
        """process one new window: add to recent and propagate overflow upward."""
        self.recent.append(window)

        if len(self.recent) > self.recent_capacity:
            evicted = self.recent.popleft()
            if self._is_novel(evicted):
                self._add_to_current_episode(evicted)
                self._n_promoted += 1
            else:
                self._n_discarded += 1

        while len(self.episodic) > self.episodic_capacity:
            self._consolidate_episodic()
            self._n_consolidated += 1

    def finalize(self) -> None:
        """flush the open episode at end of stream."""
        self._flush_current_episode()
        while len(self.episodic) > self.episodic_capacity:
            self._consolidate_episodic()
            self._n_consolidated += 1

    def get_recent_windows(self) -> List[WindowEntry]:
        return list(self.recent)

    def get_searchable_episodes(self) -> List[EpisodeEntry]:
        """return flushed episodes plus a snapshot of the open pending episode."""
        episodes = list(self.episodic)
        pending = self._snapshot_current_episode()
        if pending is not None:
            episodes.append(pending)
        return episodes

    def get_grounding_windows(
        self,
        episode: EpisodeEntry,
        radius: int = 1,
    ) -> List[WindowEntry]:
        """return a small local window slice for an episodic hit."""
        windows = [
            self._window_archive[wid]
            for wid in episode.member_window_ids
            if wid in self._window_archive
        ]
        if not windows:
            return []
        if radius < 0:
            return windows

        max_keep = 2 * radius + 1
        if len(windows) <= max_keep:
            return windows

        center = len(windows) // 2
        lo = max(0, center - radius)
        hi = min(len(windows), center + radius + 1)
        return windows[lo:hi]

    def stats(self) -> dict:
        return {
            "recent": len(self.recent),
            "pending_episode": len(self._pending_episode),
            "episodic": len(self.episodic),
            "long_term": len(self.long_term),
            "n_promoted": self._n_promoted,
            "n_discarded": self._n_discarded,
            "n_episodes_flushed": self._n_episodes_flushed,
            "n_consolidated": self._n_consolidated,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_novel(self, entry: WindowEntry) -> bool:
        if not self.recent:
            return True
        sims = [
            cosine_sim(entry.visual_embedding, w.visual_embedding)
            for w in self.recent
        ]
        return (1.0 - max(sims)) > self.novelty_threshold

    def _add_to_current_episode(self, window: WindowEntry) -> None:
        """extend the current pending episode or flush and start a new one."""
        if not self._pending_episode:
            self._window_archive[window.entry_id] = window
            self._pending_episode.append(window)
            return

        last = self._pending_episode[-1]
        gap = window.start_time - last.end_time
        centroid = self._centroid(
            [w.visual_embedding for w in self._pending_episode]
        )
        sim = cosine_sim(window.visual_embedding, centroid)

        hit_length_cap = (
            self.episode_max_len is not None
            and len(self._pending_episode) >= self.episode_max_len
        )

        merge = (
            not hit_length_cap
            and gap <= self.episode_max_gap
            and sim >= self.episode_min_sim
        )

        if merge:
            self._window_archive[window.entry_id] = window
            self._pending_episode.append(window)
        else:
            self._flush_current_episode()
            self._window_archive[window.entry_id] = window
            self._pending_episode.append(window)

    def _flush_current_episode(self) -> None:
        """turn the pending buffer into one EpisodeEntry."""
        if not self._pending_episode:
            return

        windows = self._pending_episode
        self._pending_episode = []

        centroid = self._centroid([w.visual_embedding for w in windows])
        summary = (
            self._summary_fn(windows)
            if self._summary_fn
            else self._default_episode_summary(windows)
        )

        self.episodic.append(
            EpisodeEntry(
                entry_id=f"ep-{uuid.uuid4().hex[:8]}",
                start_time=windows[0].start_time,
                end_time=windows[-1].end_time,
                visual_embedding=centroid,
                member_window_ids=[w.entry_id for w in windows],
                summary_text=summary,
                summary_embedding=self._embed_summary(summary),
            )
        )
        self._n_episodes_flushed += 1

    def _snapshot_current_episode(self) -> Optional[EpisodeEntry]:
        """read-only EpisodeEntry view of the open pending buffer."""
        if not self._pending_episode:
            return None

        windows = self._pending_episode
        summary = (
            self._summary_fn(windows)
            if self._summary_fn
            else self._default_episode_summary(windows)
        )
        return EpisodeEntry(
            entry_id=(
                f"ep-pending-{windows[0].entry_id}-{windows[-1].entry_id}"
            ),
            start_time=windows[0].start_time,
            end_time=windows[-1].end_time,
            visual_embedding=self._centroid([w.visual_embedding for w in windows]),
            member_window_ids=[w.entry_id for w in windows],
            summary_text=summary,
            summary_embedding=self._embed_summary(summary),
        )

    @staticmethod
    def _centroid(vectors: List[np.ndarray]) -> np.ndarray:
        stacked = np.stack(vectors)
        mean = stacked.mean(axis=0)
        return (mean / (np.linalg.norm(mean) + 1e-8)).astype(np.float32)

    @staticmethod
    def _default_episode_summary(windows: List[WindowEntry]) -> str:
        start = windows[0].start_time
        end = windows[-1].end_time
        return f"Episode {start:.1f}–{end:.1f}s ({len(windows)} windows)"

    def _pop_similar_episode_cluster(self) -> List[EpisodeEntry]:
        """pop a prefix of self.episodic that forms one temporal-visual cluster."""
        if not self.episodic:
            return []
        cluster: List[EpisodeEntry] = [self.episodic[0]]
        for j in range(1, len(self.episodic)):
            if len(cluster) >= self.episodic_merge_batch:
                break
            nxt = self.episodic[j]
            gap = nxt.start_time - cluster[-1].end_time
            if gap > self.event_max_gap:
                break
            centroid = self._centroid([e.visual_embedding for e in cluster])
            if cosine_sim(nxt.visual_embedding, centroid) < self.event_min_episode_sim:
                break
            cluster.append(nxt)

        self.episodic = self.episodic[len(cluster):]
        return cluster

    def _consolidate_episodic(self) -> None:
        """merge one cluster of oldest episodes into an EventEntry."""
        batch = self._pop_similar_episode_cluster()
        if not batch:
            return

        centroid = self._centroid([e.visual_embedding for e in batch])

        # pick 1–3 representative windows (one per top episode, middle window each)
        sims = [cosine_sim(centroid, e.visual_embedding) for e in batch]
        rep_indices = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        rep_indices = rep_indices[: min(3, len(batch))]
        rep_window_ids = [
            batch[idx].member_window_ids[len(batch[idx].member_window_ids) // 2]
            for idx in rep_indices
        ]

        summary = (
            self._summary_fn(batch)
            if self._summary_fn
            else self._default_event_summary(batch)
        )

        event = EventEntry(
            entry_id=f"ev-{uuid.uuid4().hex[:8]}",
            start_time=batch[0].start_time,
            end_time=batch[-1].end_time,
            visual_embedding=centroid,
            member_episode_ids=[e.entry_id for e in batch],
            representative_window_ids=rep_window_ids,
            summary_text=summary,
            summary_embedding=self._embed_summary(summary),
        )
        self.long_term.append(event)

    @staticmethod
    def _default_event_summary(episodes: List[EpisodeEntry]) -> str:
        start = episodes[0].start_time
        end = episodes[-1].end_time
        snippets = " | ".join(e.summary_text for e in episodes[:3])
        return f"Event {start:.1f}–{end:.1f}s: {snippets}"

    def _embed_summary(self, text: Optional[str]) -> Optional[np.ndarray]:
        if self._text_encode_fn is None or not text:
            return None
        try:
            return self._text_encode_fn(text)
        except Exception as exc:
            print(f"[MemoryWriter] summary embedding failed: {exc!r}")
            return None
