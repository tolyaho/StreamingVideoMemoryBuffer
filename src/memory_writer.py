"""three-tier hierarchical memory buffer updated online as windows arrive.

tier 1 (recent): fixed-capacity deque of the most recent WindowEntries.
tier 2 (episodic): coherent action spans built from novel promoted windows.
              episode embedding: time-aware self-centrality pooling over member window embeddings.
tier 3 (long-term): compressed EventEntries consolidated from the oldest episodes.
              event embedding: L2-normalised centroid of member episode embeddings.

write path is query-agnostic — storage is driven by recency, novelty, and capacity.
"""
from __future__ import annotations

import uuid
from collections import deque
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import numpy as np

from .data_structures import EpisodeEntry, EventEntry, WindowEntry

if TYPE_CHECKING:
    from .memory_db import MemoryStore


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


class HierarchicalMemoryWriter:
    """maintains three memory tiers and updates them on each incoming window.

    Args:
        recent_capacity: max WindowEntries in recent memory.
        episodic_capacity: max EpisodeEntries before consolidation is triggered.
        novelty_threshold: min cosine distance (1 - sim) to promote a window.
        episode_max_gap: max seconds between window end and next window start
            to merge into one episode (smaller = stricter / shorter episodes).
        episode_min_sim: min cosine sim to the episode centroid to keep merging.
        episode_max_len: hard cap on windows per episode. None = no cap.
        event_max_gap: max seconds between episodes to cluster into one event
            (smaller = stricter / more events).
        event_min_episode_sim: min cosine sim to the running event centroid.
        episodic_merge_batch: max episodes merged into one EventEntry.
        sigma_center: std dev (seconds) of the temporal centrality Gaussian for episode pooling.
        sigma_time: time-decay constant (seconds) for the local consistency kernel.
        lambda_center: weight of temporal centrality term in self-centrality score.
        mu_consistency: weight of local consistency term in self-centrality score.
        n_rep_windows: number of top-weight windows to keep as episode representatives.
        summary_fn: callable that summarizes memory entries. It is called as
            ``summary_fn(entries)`` for episodes and may also be called as
            ``summary_fn(entries, episode_frames=episode_frames)`` for events.
            Falls back to time templates if None.
        text_encode_fn: (text: str) -> np.ndarray. encodes summaries for retrieval.
    """

    def __init__(
        self,
        recent_capacity: int = 20,
        episodic_capacity: int = 50,
        novelty_threshold: float = 0.25,
        episode_max_gap: float = 4.0,
        episode_min_sim: float = 0.7,
        episode_max_len: Optional[int] = 8,
        event_max_gap: float = 15.0,
        event_min_episode_sim: float = 0.55,
        episodic_merge_batch: int = 10,
        sigma_center: float = 3.0,
        sigma_time: float = 2.0,
        lambda_center: float = 0.5,
        mu_consistency: float = 0.5,
        n_rep_windows: int = 2,
        summary_fn: Optional[Callable] = None,
        text_encode_fn: Optional[Callable[[str], np.ndarray]] = None,
        store: Optional["MemoryStore"] = None,
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
        self.sigma_center = sigma_center
        self.sigma_time = sigma_time
        self.lambda_center = lambda_center
        self.mu_consistency = mu_consistency
        self.n_rep_windows = n_rep_windows
        self._summary_fn = summary_fn
        self._text_encode_fn = text_encode_fn
        self._store = store

        self.recent: deque[WindowEntry] = deque()
        self.episodic: List[EpisodeEntry] = []
        self.long_term: List[EventEntry] = []
        self._window_archive: dict[str, WindowEntry] = {}

        self._pending_episode: List[WindowEntry] = []

        self._n_discarded = 0
        self._n_promoted = 0
        self._n_episodes_flushed = 0
        self._n_consolidated = 0

    def update(self, window: WindowEntry) -> None:
        if window.summary_embedding is None and window.summary_text:
            window.summary_embedding = self._embed_summary(window.summary_text)

        self.recent.append(window)
        if self._store is not None:
            try:
                self._store.save_window(window)
            except Exception as exc:
                print(f"[MemoryWriter] DB save_window failed: {exc!r}")

        if len(self.recent) > self.recent_capacity:
            evicted = self.recent.popleft()
            if self._is_novel(evicted):
                evicted.tier = "episodic"
                self._add_to_current_episode(evicted)
                self._n_promoted += 1
                if self._store is not None:
                    try:
                        self._store.save_window(evicted)
                    except Exception as exc:
                        print(f"[MemoryWriter] DB tier update failed: {exc!r}")
            else:
                self._n_discarded += 1

        while len(self.episodic) > self.episodic_capacity:
            self._consolidate_episodic()
            self._n_consolidated += 1

    def finalize(self) -> None:
        # Drain whatever is still in the recent deque through the same
        # novelty/promotion path used during streaming. Without this the last
        # ``recent_capacity`` windows would be stranded with no episode
        # membership.
        while self.recent:
            evicted = self.recent.popleft()
            if self._is_novel(evicted):
                evicted.tier = "episodic"
                self._add_to_current_episode(evicted)
                self._n_promoted += 1
                if self._store is not None:
                    try:
                        self._store.save_window(evicted)
                    except Exception as exc:
                        print(f"[MemoryWriter] DB tier update failed: {exc!r}")
            else:
                self._n_discarded += 1

        self._flush_current_episode()

        # Force-consolidate residual episodes into events, even if below the
        # episodic capacity — otherwise the tail of the stream never makes it
        # into long-term memory.
        while self.episodic:
            before = len(self.episodic)
            self._consolidate_episodic()
            self._n_consolidated += 1
            if len(self.episodic) >= before:
                break  # safety: _pop_similar_episode_cluster didn't advance

    def flush_pending(self) -> None:
        """Force-close the currently-forming episode so it is searchable at query time.

        Only the in-progress episode is consolidated; completed episodes stay in the
        episodic tier so stage B (fine search) can still score them. Recent windows
        remain queryable and the writer can continue ingesting after the call.
        """
        self._flush_current_episode()

    def get_recent_windows(self) -> List[WindowEntry]:
        return list(self.recent)

    def get_searchable_episodes(self) -> List[EpisodeEntry]:
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

    def _is_novel(self, entry: WindowEntry) -> bool:
        if not self.recent:
            return True
        sims = [
            cosine_sim(entry.visual_embedding, w.visual_embedding)
            for w in self.recent
        ]
        return (1.0 - max(sims)) > self.novelty_threshold

    def _add_to_current_episode(self, window: WindowEntry) -> None:
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
        if not self._pending_episode:
            return

        windows = self._pending_episode
        self._pending_episode = []

        embs = [w.visual_embedding for w in windows]
        times = [w.start_time for w in windows]
        episode_emb, rep_indices = self._self_centrality_pool(
            embs, times,
            sigma_center=self.sigma_center,
            sigma_time=self.sigma_time,
            lambda_center=self.lambda_center,
            mu_consistency=self.mu_consistency,
            n_rep=self.n_rep_windows,
        )
        rep_ids = [windows[i].entry_id for i in rep_indices]

        summary = (
            self._summary_fn(windows)
            if self._summary_fn
            else self._default_episode_summary(windows)
        )

        episode = EpisodeEntry(
            entry_id=f"ep-{uuid.uuid4().hex[:8]}",
            start_time=windows[0].start_time,
            end_time=windows[-1].end_time,
            visual_embedding=episode_emb,
            member_window_ids=[w.entry_id for w in windows],
            summary_text=summary,
            summary_embedding=self._embed_summary(summary),
            representative_window_ids=rep_ids,
        )
        self.episodic.append(episode)
        self._n_episodes_flushed += 1
        if self._store is not None:
            try:
                self._store.save_episode(episode)
            except Exception as exc:
                print(f"[MemoryWriter] DB save_episode failed: {exc!r}")

    def _snapshot_current_episode(self) -> Optional[EpisodeEntry]:
        if not self._pending_episode:
            return None

        windows = self._pending_episode
        embs = [w.visual_embedding for w in windows]
        times = [w.start_time for w in windows]
        episode_emb, rep_indices = self._self_centrality_pool(
            embs, times,
            sigma_center=self.sigma_center,
            sigma_time=self.sigma_time,
            lambda_center=self.lambda_center,
            mu_consistency=self.mu_consistency,
            n_rep=self.n_rep_windows,
        )
        rep_ids = [windows[i].entry_id for i in rep_indices]

        summary = (
            self._summary_fn(windows)
            if self._summary_fn
            else self._default_episode_summary(windows)
        )
        return EpisodeEntry(
            entry_id=f"ep-pending-{windows[0].entry_id}-{windows[-1].entry_id}",
            start_time=windows[0].start_time,
            end_time=windows[-1].end_time,
            visual_embedding=episode_emb,
            member_window_ids=[w.entry_id for w in windows],
            summary_text=summary,
            summary_embedding=self._embed_summary(summary),
            representative_window_ids=rep_ids,
        )

    @staticmethod
    def _self_centrality_pool(
        embeddings: List[np.ndarray],
        timestamps: List[float],
        sigma_center: float = 3.0,
        sigma_time: float = 2.0,
        lambda_center: float = 0.5,
        mu_consistency: float = 0.5,
        n_rep: int = 2,
    ) -> Tuple[np.ndarray, List[int]]:
        """time-aware self-centrality pooling over an ordered sequence of window embeddings.

        For each window i:
          center_score_i    = -(t_i - t_center)^2 / (2 * sigma_center^2)
          consistency_score_i = sum_j exp(-|t_i - t_j| / sigma_time) * cos(w_i, w_j)
          score_i           = lambda_center * center_score_i + mu_consistency * consistency_score_i
          alpha_i           = softmax(score_i)
        episode_embedding = sum_i alpha_i * w_i   (L2-normalised)

        Returns the pooled embedding and the indices of the top-n_rep highest-weight windows
        in temporal order (useful as representative windows for summarisation).
        """
        n = len(embeddings)
        if n == 1:
            return embeddings[0].copy(), [0]

        t = np.array(timestamps, dtype=np.float64)
        t_center = (t[0] + t[-1]) / 2.0

        center_scores = -((t - t_center) ** 2) / (2.0 * sigma_center ** 2 + 1e-8)

        consistency_scores = np.zeros(n, dtype=np.float64)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                decay = np.exp(-abs(t[i] - t[j]) / (sigma_time + 1e-8))
                consistency_scores[i] += decay * cosine_sim(embeddings[i], embeddings[j])

        scores = lambda_center * center_scores + mu_consistency * consistency_scores
        scores_shifted = scores - scores.max()
        weights = np.exp(scores_shifted)
        weights /= weights.sum() + 1e-8
        weights = weights.astype(np.float32)

        stacked = np.stack(embeddings)
        pooled = (weights[:, None] * stacked).sum(axis=0)
        pooled = (pooled / (np.linalg.norm(pooled) + 1e-8)).astype(np.float32)

        n_rep_actual = min(n_rep, n)
        rep_indices = sorted(np.argsort(weights)[-n_rep_actual:].tolist())

        return pooled, rep_indices

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
        batch = self._pop_similar_episode_cluster()
        if not batch:
            return

        centroid = self._centroid([e.visual_embedding for e in batch])

        episode_rep_windows: List[List[WindowEntry]] = []
        rep_window_ids: List[str] = []
        for ep in batch:
            if ep.representative_window_ids:
                reps = [
                    self._window_archive[wid]
                    for wid in ep.representative_window_ids
                    if wid in self._window_archive
                ]
            else:
                ep_windows = [
                    self._window_archive[wid]
                    for wid in ep.member_window_ids
                    if wid in self._window_archive
                ]
                if not ep_windows:
                    episode_rep_windows.append([])
                    continue
                sims = [cosine_sim(ep.visual_embedding, w.visual_embedding) for w in ep_windows]
                top_idxs = sorted(
                    sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:2]
                )
                reps = [ep_windows[i] for i in top_idxs]
            if not reps:
                episode_rep_windows.append([])
                continue
            episode_rep_windows.append(reps)
            rep_window_ids.extend(w.entry_id for w in reps)

        episode_frames = [
            [w.frame for w in ws if w.frame is not None]
            for ws in episode_rep_windows
        ]

        summary = (
            self._summary_fn(batch, episode_frames=episode_frames)
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
        if self._store is not None:
            try:
                self._store.save_event(event)
            except Exception as exc:
                print(f"[MemoryWriter] DB save_event failed: {exc!r}")

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
