import unittest
from types import SimpleNamespace

import numpy as np

from src.data_structures import EpisodeEntry, EventEntry
from src.retriever import HierarchicalRetriever


def _unit(vec):
    arr = np.array(vec, dtype=np.float32)
    return arr / np.linalg.norm(arr)


class _FakeMemory:
    def __init__(self, *, events, episodes, recent=None):
        self.long_term = events
        self._episodes = episodes
        self._recent = recent or []

    def get_recent_windows(self):
        return list(self._recent)

    def get_searchable_episodes(self):
        return list(self._episodes)

    def get_grounding_windows(self, ep, radius=1):
        return []


class RetrieverTemporalTests(unittest.TestCase):
    def test_latest_event_is_considered_even_if_not_best_semantic_match(self) -> None:
        retriever = HierarchicalRetriever()
        q = _unit([1.0, 0.0])
        old = EventEntry(
            entry_id="old",
            start_time=0.0,
            end_time=10.0,
            visual_embedding=q,
            member_episode_ids=[],
            representative_window_ids=[],
            summary_text="old",
            summary_embedding=q,
        )
        latest = EventEntry(
            entry_id="latest",
            start_time=100.0,
            end_time=110.0,
            visual_embedding=_unit([0.0, 1.0]),
            member_episode_ids=[],
            representative_window_ids=[],
            summary_text="latest",
            summary_embedding=_unit([0.0, 1.0]),
        )
        memory = _FakeMemory(events=[old, latest], episodes=[])

        result = retriever.retrieve(
            query="what is happening now",
            query_embedding=q,
            memory=memory,
            top_m=1,
            query_time=110.0,
        )

        self.assertEqual(1, len(result.coarse_hits))
        self.assertEqual("latest", result.coarse_hits[0].entry_id)

    def test_recent_episode_outside_event_ranges_is_still_considered(self) -> None:
        retriever = HierarchicalRetriever(alpha=1.0, beta=0.0)
        target = _unit([1.0, 0.0])
        other = _unit([0.0, 1.0])

        # Episode that sits inside the one coarse event range, but semantically far from query.
        inside = EpisodeEntry(
            entry_id="inside",
            start_time=2.0,
            end_time=6.0,
            visual_embedding=other,
            member_window_ids=[],
            summary_text="inside",
            summary_embedding=other,
        )
        # Tail episode that has not been consolidated into any event. Semantically matches.
        tail = EpisodeEntry(
            entry_id="tail",
            start_time=200.0,
            end_time=204.0,
            visual_embedding=target,
            member_window_ids=[],
            summary_text="tail",
            summary_embedding=target,
        )

        hits, _ = retriever._fine_search(
            query_vis_emb=target,
            query_txt_emb=target,
            episodic=[inside, tail],
            candidate_ranges=[(0.0, 10.0)],
            top_k=1,
            recent_n=2,
        )

        self.assertEqual(["tail"], [ep.entry_id for ep in hits])

    def test_query_time_proximity_beats_absolute_latest_episode(self) -> None:
        retriever = HierarchicalRetriever(alpha=1.0, beta=0.0)
        emb = _unit([1.0, 0.0])
        near = EpisodeEntry(
            entry_id="near",
            start_time=98.0,
            end_time=102.0,
            visual_embedding=emb,
            member_window_ids=[],
            summary_text="near",
            summary_embedding=emb,
        )
        far_future = EpisodeEntry(
            entry_id="far",
            start_time=300.0,
            end_time=304.0,
            visual_embedding=emb,
            member_window_ids=[],
            summary_text="far",
            summary_embedding=emb,
        )

        hits, _ = retriever._fine_search(
            query_vis_emb=emb,
            query_txt_emb=emb,
            episodic=[near, far_future],
            candidate_ranges=[],
            top_k=1,
            query_time=100.0,
        )

        self.assertEqual(["near"], [ep.entry_id for ep in hits])

    def test_multiplicative_decay_suppresses_semantically_strong_distant_entry(self) -> None:
        """Under multiplicative decay, semantic score is scaled by exp(-dt/tau):
        a temporally distant entry with a perfect semantic match can lose to a
        temporally close entry with only a partial semantic match. Under the
        previous additive ``gamma * bonus`` form this test would fail because
        the temporal contribution was capped at ``gamma``, leaving a strong but
        distant entry in the lead.
        """
        retriever = HierarchicalRetriever()
        query = _unit([1.0, 0.0])
        partial = _unit([0.5, np.sqrt(0.75)])

        strong_but_far = EpisodeEntry(
            entry_id="far",
            start_time=0.0,
            end_time=10.0,
            visual_embedding=query,
            member_window_ids=[],
            summary_text="far",
            summary_embedding=query,
        )
        partial_but_near = EpisodeEntry(
            entry_id="near",
            start_time=100.0,
            end_time=110.0,
            visual_embedding=partial,
            member_window_ids=[],
            summary_text="near",
            summary_embedding=partial,
        )

        hits, _ = retriever._fine_search(
            query_vis_emb=query,
            query_txt_emb=query,
            episodic=[strong_but_far, partial_but_near],
            candidate_ranges=[],
            top_k=1,
            query_time=110.0,
        )

        self.assertEqual(
            ["near"], [ep.entry_id for ep in hits],
            "multiplicative decay should let a partial-match near entry beat a perfect-match far entry",
        )


if __name__ == "__main__":
    unittest.main()
