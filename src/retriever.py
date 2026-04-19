"""coarse-to-fine retrieval over the three-tier memory.

stage A: rank EventEntries by blended visual + summary similarity (top-M).
stage B: rank EpisodeEntries within selected time ranges (top-K).
stage C: attach nearby WindowEntries from the recent queue for local grounding.

blended score = alpha * visual_sim + beta * summary_sim + gamma * recency.
when summary_embedding is None, weights are renormalised onto the remaining terms.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .data_structures import EpisodeEntry, EventEntry, RetrievalResult, WindowEntry
from .memory_writer import HierarchicalMemoryWriter, cosine_sim


class HierarchicalRetriever:
    """retrieves the smallest useful subset of past memory for a text query.

    Args:
        alpha: weight for visual embedding similarity (dominant signal).
        beta: weight for summary text embedding similarity (auxiliary).
        gamma: weight for recency bonus.
    """

    def __init__(
        self,
        alpha: float = 0.65,
        beta: float = 0.30,
        gamma: float = 0.05,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def retrieve(
        self,
        query: str,
        query_embedding: np.ndarray,
        memory: HierarchicalMemoryWriter,
        top_m: int = 3,
        top_k: int = 5,
        neighbor_radius: int = 1,
        query_summary_embedding: Optional[np.ndarray] = None,
    ) -> RetrievalResult:
        """run full three-stage retrieval.

        Args:
            query: raw query text (stored in result for provenance).
            query_embedding: L2-normalised text embedding of the query.
            memory: the live memory object to search.
            top_m: number of coarse EventEntry hits.
            top_k: number of fine EpisodeEntry hits.
            neighbor_radius: window radius for local grounding.
            query_summary_embedding: separate summary-space embedding; defaults to query_embedding.
        """
        q_sum_emb = query_summary_embedding if query_summary_embedding is not None else query_embedding

        coarse_hits, coarse_scores, candidate_ranges = self._coarse_route(
            query_embedding, q_sum_emb, memory.long_term, top_m
        )

        episodic_hits, episodic_scores = self._fine_search(
            query_embedding,
            q_sum_emb,
            memory.get_searchable_episodes(),
            candidate_ranges,
            top_k,
        )
        scores = {**coarse_scores, **episodic_scores}

        grounded = self._local_grounding(episodic_hits, memory, neighbor_radius)

        recent = memory.get_recent_windows()
        if not episodic_hits and recent:
            sims = [cosine_sim(query_embedding, w.visual_embedding) for w in recent]
            top_idxs = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:top_k]
            grounded = [recent[i] for i in top_idxs]
            scores = {recent[i].entry_id: sims[i] for i in top_idxs}

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
        visual_emb: np.ndarray,
        summary_emb: Optional[np.ndarray],
        recency: float,
    ) -> float:
        vis_sim = cosine_sim(query_vis_emb, visual_emb)
        if summary_emb is not None:
            txt_sim = cosine_sim(query_txt_emb, summary_emb)
            return self.alpha * vis_sim + self.beta * txt_sim + self.gamma * recency
        total = self.alpha + self.gamma
        if total <= 0:
            return 0.0
        a = self.alpha / total
        g = self.gamma / total
        return a * vis_sim + g * recency

    def _coarse_route(
        self,
        query_vis_emb: np.ndarray,
        query_txt_emb: np.ndarray,
        long_term: List[EventEntry],
        top_m: int,
    ) -> Tuple[List[EventEntry], dict, List[Tuple[float, float]]]:
        if not long_term:
            return [], {}, []

        scored = []
        for ev in long_term:
            score = self._blended_score(
                query_vis_emb,
                query_txt_emb,
                ev.visual_embedding,
                ev.summary_embedding,
                recency=0.0,
            )
            scored.append((score, ev))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:top_m]
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
    ) -> Tuple[List[EpisodeEntry], dict]:
        if not episodic:
            return [], {}

        if candidate_ranges:
            candidates = [
                ep for ep in episodic
                if any(
                    ep.start_time <= end and ep.end_time >= start
                    for start, end in candidate_ranges
                )
            ]
            if not candidates:
                candidates = episodic
        else:
            candidates = episodic

        max_time = max((ep.end_time for ep in episodic), default=1.0)

        scored = []
        for ep in candidates:
            recency = ep.end_time / (max_time + 1e-8)
            score = self._blended_score(
                query_vis_emb,
                query_txt_emb,
                ep.visual_embedding,
                ep.summary_embedding,
                recency=recency,
            )
            scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        hits = [ep for _, ep in scored[:top_k]]
        scores = {ep.entry_id: sc for sc, ep in scored[:top_k]}
        return hits, scores

    def _local_grounding(
        self,
        episodic_hits: List[EpisodeEntry],
        memory: HierarchicalMemoryWriter,
        radius: int,
    ) -> List[WindowEntry]:
        """attach recent windows overlapping episodic hits plus the recent tail."""
        recent = memory.get_recent_windows()
        if not recent:
            return []

        nearby: List[WindowEntry] = []

        if episodic_hits:
            hit_ranges = [(ep.start_time, ep.end_time) for ep in episodic_hits]
            for i, w in enumerate(recent):
                for start, end in hit_ranges:
                    if w.end_time >= start and w.start_time <= end:
                        lo = max(0, i - radius)
                        hi = min(len(recent), i + radius + 1)
                        nearby.extend(recent[lo:hi])
                        break

        # always append the most recent tail
        tail_n = min(radius + 1, len(recent))
        nearby.extend(recent[-tail_n:])

        seen: set = set()
        result: List[WindowEntry] = []
        for w in nearby:
            if w.entry_id not in seen:
                seen.add(w.entry_id)
                result.append(w)
        return result
