"""Unit tests for the text-side chunking helper in perception_encoder.

The helper exists so that X-CLIP's 77-token position limit does not reject
long event summaries. We chunk the raw (no-special-tokens) id stream into
pieces that each fit under the per-chunk content budget; the caller then
wraps each chunk with BOS/EOS, batch-forwards, and mean-pools.
"""
import unittest

from src.perception_encoder import _chunk_token_ids


class ChunkTokenIdsTests(unittest.TestCase):
    def test_empty_input_yields_single_empty_chunk(self) -> None:
        # Even empty input must produce one chunk so the caller can still
        # emit a [BOS, EOS] forward pass and get a defined embedding.
        self.assertEqual([[]], _chunk_token_ids([], chunk_size=5))

    def test_short_input_fits_in_one_chunk(self) -> None:
        self.assertEqual([[1, 2, 3]], _chunk_token_ids([1, 2, 3], chunk_size=5))

    def test_exact_fit_single_chunk(self) -> None:
        self.assertEqual([[1, 2, 3, 4, 5]], _chunk_token_ids([1, 2, 3, 4, 5], chunk_size=5))

    def test_one_over_splits_to_two(self) -> None:
        self.assertEqual(
            [[1, 2, 3, 4, 5], [6]],
            _chunk_token_ids([1, 2, 3, 4, 5, 6], chunk_size=5),
        )

    def test_many_chunks_preserve_order_and_coverage(self) -> None:
        ids = list(range(13))
        chunks = _chunk_token_ids(ids, chunk_size=5)
        self.assertEqual([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12]], chunks)
        # Round-trip: concatenating chunks must recover the original stream.
        self.assertEqual(ids, [tok for c in chunks for tok in c])

    def test_invalid_chunk_size_raises(self) -> None:
        with self.assertRaises(ValueError):
            _chunk_token_ids([1, 2, 3], chunk_size=0)


if __name__ == "__main__":
    unittest.main()
