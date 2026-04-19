"""converts retrieved evidence into a human-readable block and an LLM-ready dict."""
from __future__ import annotations

from typing import Optional

import numpy as np

from .data_structures import RetrievalResult


class ReasonerInputFormatter:
    """turns a RetrievalResult into a clean representation for downstream reasoning."""

    def format_text(self, result: RetrievalResult) -> str:
        """produce a readable evidence block suitable for display in a notebook."""
        lines = [
            "=" * 64,
            f"QUERY: {result.query}",
            "=" * 64,
        ]

        if result.coarse_hits:
            lines.append("\n[Coarse — event-level routing hits]")
            for ev in result.coarse_hits:
                score = result.scores.get(ev.entry_id, "—")
                score_str = f"{score:.3f}" if isinstance(score, float) else score
                lines.append(
                    f"  [{ev.start_time:.1f}–{ev.end_time:.1f}s] "
                    f"sim={score_str}  {ev.summary_text}"
                )

        if result.episodic_hits:
            lines.append("\n[Fine — episodic memory hits]")
            for ep in result.episodic_hits:
                score = result.scores.get(ep.entry_id, "—")
                score_str = f"{score:.3f}" if isinstance(score, float) else score
                lines.append(
                    f"  [{ep.start_time:.1f}–{ep.end_time:.1f}s] "
                    f"sim={score_str}  {ep.summary_text}"
                )

        if result.grounded_windows:
            lines.append("\n[Local grounding — window-level context]")
            for w in result.grounded_windows:
                score = result.scores.get(w.entry_id, "")
                score_str = f" sim={score:.3f}" if isinstance(score, float) else ""
                note = f"  {w.summary_text}" if w.summary_text else ""
                lines.append(
                    f"  [{w.start_time:.1f}–{w.end_time:.1f}s]{score_str}{note}"
                )

        lines.append("=" * 64)
        return "\n".join(lines)

    def format_for_llm(
        self,
        result: RetrievalResult,
        query_embedding: Optional[np.ndarray] = None,
    ) -> dict:
        """structured dict representing what would be passed to a full multimodal LLM.

        visual_context holds raw embeddings as placeholders — a real adapter
        would project these into the LM's token dimension.
        """
        visual_tokens = []

        for ep in result.episodic_hits:
            visual_tokens.append({
                "source": "episodic",
                "time_range": [round(ep.start_time, 2), round(ep.end_time, 2)],
                "embedding_dim": ep.visual_embedding.shape[0],
                "embedding": ep.visual_embedding.tolist(),
                "summary": ep.summary_text,
                "score": result.scores.get(ep.entry_id),
            })

        for w in result.grounded_windows:
            visual_tokens.append({
                "source": "grounding_window",
                "time_range": [round(w.start_time, 2), round(w.end_time, 2)],
                "embedding_dim": w.visual_embedding.shape[0],
                "embedding": w.visual_embedding.tolist(),
                "summary": w.summary_text,
                "score": result.scores.get(w.entry_id),
            })

        return {
            "query": result.query,
            "query_embedding": (
                query_embedding.tolist() if query_embedding is not None else None
            ),
            "visual_context": visual_tokens,
            "text_context": self.format_text(result),
            "n_visual_tokens": len(visual_tokens),
            "coarse_event_count": len(result.coarse_hits),
            "episodic_hit_count": len(result.episodic_hits),
        }

    def __call__(self, result: RetrievalResult) -> str:
        return self.format_text(result)
