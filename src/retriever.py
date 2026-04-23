"""coarse-to-fine retrieval over the three-tier memory"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .data_structures import EpisodeEntry, EventEntry, RetrievalResult, WindowEntry
from .memory_writer import HierarchicalMemoryWriter, cosine_sim


class HierarchicalRetriever:
    def __init__(
        self,
        alpha: float = 0.70,
        beta: float = 0.30,
        tau_fraction: float = 0.50,
    ):
        self.alpha = alpha
        self.beta = beta
        self.tau_fraction = tau_fraction

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        memory: HierarchicalMemoryWriter,
        top_m: int = 3,
        top_k: int = 5,
        neighbor_radius: int = 1,
        query_summary_embedding: Optional[np.ndarray] = None,
        query_time: Optional[float] = None,
        recent_episodes: int = 5,
    ) -> RetrievalResult:
        """Stage 0 (recent) + Stage A (events) + Stage B (episodes) + Stage C (grounding windows)"""
        q_sum_emb = query_summary_embedding if query_summary_embedding is not None else query_embedding

        recent = memory.get_recent_windows()
        searchable_episodes = memory.get_searchable_episodes()
        span = self._unified_span(recent, searchable_episodes, memory.long_term)
        rep_vectors = self._build_rep_index(memory, memory.long_term)

        recent_hits: List[WindowEntry] = []
        recent_scores: dict = {}
        if recent:
            scored_recent = []
            for w in recent:
                decay = self._time_decay(
                    w.start_time,
                    w.end_time,
                    query_time,
                    max_time=span,
                )
                score = self._blended_score(
                    query_embedding,
                    q_sum_emb,
                    [w.visual_embedding],
                    w.summary_embedding,
                    decay=decay,
                )
                scored_recent.append((score, w))
            scored_recent.sort(key=lambda x: x[0], reverse=True)
            top = scored_recent[:top_k]
            recent_hits = [w for _, w in top]
            recent_scores = {w.entry_id: sc for sc, w in top}

        coarse_hits, coarse_scores, candidate_ranges = self._coarse_route(
            query_embedding, q_sum_emb, memory.long_term, top_m, span,
            rep_vectors, query_time=query_time,
        )

        episodic_hits, episodic_scores = self._fine_search(
            query_embedding,
            q_sum_emb,
            searchable_episodes,
            candidate_ranges,
            top_k,
            span,
            query_time=query_time,
            recent_n=recent_episodes,
        )

        archive_windows: List[WindowEntry] = []
        for ep in episodic_hits:
            archive_windows.extend(memory.get_grounding_windows(ep, radius=neighbor_radius))

        seen: set = set()
        grounded: List[WindowEntry] = []
        for w in recent_hits + archive_windows:
            if w.entry_id not in seen:
                seen.add(w.entry_id)
                grounded.append(w)

        scores = {**coarse_scores, **episodic_scores, **recent_scores}

        return RetrievalResult(
            query=query,
            coarse_hits=coarse_hits,
            episodic_hits=episodic_hits,
            grounded_windows=grounded,
            scores=scores,
        )

    def _blended_score(
        self,
        query_vis_emb: np.ndarray,
        query_txt_emb: np.ndarray,
        visual_embs: List[np.ndarray],
        summary_emb: Optional[np.ndarray],
        decay: float,
    ) -> float:
        """(alpha * vis_sim + beta * txt_sim) * decay, max-pool vis_sim over reps"""
        vis_sim = max(cosine_sim(query_vis_emb, v) for v in visual_embs)
        if summary_emb is not None:
            txt_sim = cosine_sim(query_txt_emb, summary_emb)
            semantic = self.alpha * vis_sim + self.beta * txt_sim
        else:
            semantic = vis_sim
        return semantic * decay

    @staticmethod
    def _build_rep_index(memory, events) -> dict:
        """event_id -> list of rep-window visual embeddings, for peak-over-reps scoring"""
        archive = memory._window_archive
        index: dict = {}
        for entry in events:
            rep_ids = getattr(entry, "representative_window_ids", None)
            if not rep_ids:
                continue
            vecs = [
                archive[wid].visual_embedding
                for wid in rep_ids
                if wid in archive
            ]
            if vecs:
                index[entry.entry_id] = vecs
        return index

    @staticmethod
    def _vectors_for(entry, rep_vectors: dict) -> List[np.ndarray]:
        """rep vectors for an entry, falling back to its own embedding"""
        reps = rep_vectors.get(entry.entry_id)
        return reps if reps else [entry.visual_embedding]

    def _coarse_route(
        self,
        query_vis_emb: np.ndarray,
        query_txt_emb: np.ndarray,
        long_term: List[EventEntry],
        top_m: int,
        span: float,
        rep_vectors: dict,
        query_time: Optional[float] = None,
    ) -> Tuple[List[EventEntry], dict, List[Tuple[float, float]]]:
        """stage A — rank events, force-include the newest one, return their time ranges"""
        if not long_term:
            return [], {}, []

        scored = []
        for ev in long_term:
            decay = self._time_decay(
                ev.start_time,
                ev.end_time,
                query_time,
                max_time=span,
            )
            score = self._blended_score(
                query_vis_emb,
                query_txt_emb,
                self._vectors_for(ev, rep_vectors),
                ev.summary_embedding,
                decay=decay,
            )
            scored.append((score, ev))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_m]
        if top_m > 0:
            latest = max(long_term, key=lambda ev: ev.end_time)
            if all(ev.entry_id != latest.entry_id for _, ev in top):
                latest_score = next((sc for sc, ev in scored if ev.entry_id == latest.entry_id), None)
                if latest_score is not None:
                    if len(top) < top_m:
                        top.append((latest_score, latest))
                    else:
                        top[-1] = (latest_score, latest)
        hits = [ev for _, ev in top]
        scores = {ev.entry_id: sc for sc, ev in top}
        ranges = [(ev.start_time, ev.end_time) for ev in hits]
        return hits, scores, ranges

    def _fine_search(
        self,
        query_vis_emb: np.ndarray,
        query_txt_emb: np.ndarray,
        episodic: List[EpisodeEntry],
        candidate_ranges: List[Tuple[float, float]],
        top_k: int,
        span: float,
        query_time: Optional[float] = None,
        recent_n: int = 5,
    ) -> Tuple[List[EpisodeEntry], dict]:
        """stage B — rank episodes inside stage-A ranges, plus the recent_n tail"""
        if not episodic:
            return [], {}

        if candidate_ranges:
            gated = [
                ep for ep in episodic
                if any(
                    ep.start_time <= end and ep.end_time >= start
                    for start, end in candidate_ranges
                )
            ]
            tail = episodic[-recent_n:] if recent_n > 0 else []
            seen: set = set()
            candidates: List[EpisodeEntry] = []
            for ep in gated + tail:
                if ep.entry_id not in seen:
                    seen.add(ep.entry_id)
                    candidates.append(ep)
            if not candidates:
                candidates = episodic
        else:
            candidates = episodic

        scored = []
        for ep in candidates:
            decay = self._time_decay(
                ep.start_time,
                ep.end_time,
                query_time,
                max_time=span,
            )
            score = self._blended_score(
                query_vis_emb,
                query_txt_emb,
                [ep.visual_embedding],
                ep.summary_embedding,
                decay=decay,
            )
            scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        hits = [ep for _, ep in scored[:top_k]]
        scores = {ep.entry_id: sc for sc, ep in scored[:top_k]}
        return hits, scores

    @staticmethod
    def _unified_span(*tiers) -> float:
        starts, ends = [], []
        for tier in tiers:
            for e in tier:
                starts.append(e.start_time)
                ends.append(e.end_time)
        if not starts:
            return 1.0
        return max(max(ends) - min(starts), 1.0)

    def _time_decay(
        self,
        start_time: float,
        end_time: float,
        query_time: Optional[float],
        *,
        max_time: float,
    ) -> float:
        if query_time is None:
            return 1.0
        if start_time <= query_time <= end_time:
            return 1.0
        dist = min(abs(query_time - start_time), abs(query_time - end_time))
        tau = max(max_time * self.tau_fraction, 1.0)
        return float(np.exp(-dist / tau))

