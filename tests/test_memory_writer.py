import unittest
import uuid

import numpy as np

from src.data_structures import EpisodeEntry, WindowEntry
from src.memory_writer import HierarchicalMemoryWriter


def _unit(vec):
    arr = np.array(vec, dtype=np.float32)
    return arr / np.linalg.norm(arr)


def _window(t: float, emb, *, summary_text=None, summary_embedding=None) -> WindowEntry:
    return WindowEntry(
        entry_id=f"w-{uuid.uuid4().hex[:6]}",
        start_time=t,
        end_time=t + 1.0,
        visual_embedding=_unit(emb),
        summary_text=summary_text,
        summary_embedding=summary_embedding,
    )


def _episode(entry_id: str, start: float, end: float, emb) -> EpisodeEntry:
    return EpisodeEntry(
        entry_id=entry_id,
        start_time=start,
        end_time=end,
        visual_embedding=_unit(emb),
        member_window_ids=[],
        summary_text=f"episode {entry_id}",
    )


class FlushPendingSemanticsTests(unittest.TestCase):
    """flush_pending() must close the in-progress episode but leave the rest of the
    episodic tier intact; over-draining into events makes fine-tier retrieval
    (stage B) empty at query time.
    """

    def test_flush_pending_does_not_drain_completed_episodes_into_events(self) -> None:
        memory = HierarchicalMemoryWriter()
        memory.episodic = [
            _episode("ep1", 0.0, 3.0, [1.0, 0.0, 0.0]),
            _episode("ep2", 100.0, 103.0, [0.0, 1.0, 0.0]),
            _episode("ep3", 200.0, 203.0, [0.0, 0.0, 1.0]),
        ]

        memory.flush_pending()

        self.assertEqual(
            3, len(memory.episodic),
            "flush_pending should not consume completed episodes into events",
        )
        self.assertEqual(0, len(memory.long_term))

    def test_flush_pending_closes_in_progress_episode(self) -> None:
        memory = HierarchicalMemoryWriter()
        memory._pending_episode = [
            _window(10.0, [1.0, 0.0, 0.0]),
            _window(11.0, [1.0, 0.1, 0.0]),
        ]

        memory.flush_pending()

        self.assertEqual(0, len(memory._pending_episode))
        self.assertEqual(
            1, len(memory.episodic),
            "the in-progress episode should be closed into self.episodic",
        )
        self.assertEqual(0, len(memory.long_term))


class WindowSummaryEmbeddingTests(unittest.TestCase):
    """On update(), windows with a ``summary_text`` should be embedded so the
    retriever's β term can use them at scoring time. Without this, window-tier
    scoring silently falls through to visual-only cosine.
    """

    def test_update_embeds_summary_text_when_encoder_present(self) -> None:
        calls: list[str] = []

        def fake_encoder(text: str) -> np.ndarray:
            calls.append(text)
            return _unit([0.3, 0.4, 0.0])

        memory = HierarchicalMemoryWriter(text_encode_fn=fake_encoder)
        window = _window(0.0, [1.0, 0.0, 0.0], summary_text="a man in a kitchen")

        memory.update(window)

        self.assertEqual(["a man in a kitchen"], calls)
        self.assertIsNotNone(window.summary_embedding)
        np.testing.assert_allclose(window.summary_embedding, _unit([0.3, 0.4, 0.0]))

    def test_update_skips_embedding_when_no_encoder(self) -> None:
        memory = HierarchicalMemoryWriter()  # no text_encode_fn
        window = _window(0.0, [1.0, 0.0, 0.0], summary_text="a man in a kitchen")

        memory.update(window)

        self.assertIsNone(window.summary_embedding)

    def test_update_skips_embedding_when_summary_text_empty(self) -> None:
        calls: list[str] = []

        def fake_encoder(text: str) -> np.ndarray:
            calls.append(text)
            return _unit([0.1, 0.1, 0.1])

        memory = HierarchicalMemoryWriter(text_encode_fn=fake_encoder)
        window_no_text = _window(0.0, [1.0, 0.0, 0.0], summary_text=None)
        window_empty = _window(1.0, [1.0, 0.0, 0.0], summary_text="")

        memory.update(window_no_text)
        memory.update(window_empty)

        self.assertEqual([], calls)
        self.assertIsNone(window_no_text.summary_embedding)
        self.assertIsNone(window_empty.summary_embedding)

    def test_update_preserves_caller_provided_embedding(self) -> None:
        """If the caller already embedded the caption (e.g. batched upstream),
        the writer must not overwrite it. Encoding stays an optional fallback.
        """
        calls: list[str] = []

        def fake_encoder(text: str) -> np.ndarray:
            calls.append(text)
            return _unit([0.9, 0.0, 0.0])

        memory = HierarchicalMemoryWriter(text_encode_fn=fake_encoder)
        preset = _unit([0.0, 1.0, 0.0])
        window = _window(
            0.0, [1.0, 0.0, 0.0],
            summary_text="kitchen scene",
            summary_embedding=preset,
        )

        memory.update(window)

        self.assertEqual([], calls)  # encoder not called
        np.testing.assert_allclose(window.summary_embedding, preset)


if __name__ == "__main__":
    unittest.main()
