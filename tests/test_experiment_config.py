import json
import tempfile
import unittest
from pathlib import Path

from src.experiment import (
    eval_config_from_mapping,
    flatten_summary_metrics,
    write_manifest_snapshot,
    write_metrics,
    write_run_config,
)


class ExperimentConfigTests(unittest.TestCase):
    def test_eval_config_uses_memory_and_retrieval_sections(self) -> None:
        cfg = eval_config_from_mapping(
            {
                "memory": {
                    "recent_capacity": 30,
                    "episodic_capacity": 80,
                    "novelty_threshold": 0.04,
                    "episode_max_gap": 6.0,
                    "event_max_gap": 40.0,
                    "episodic_merge_batch": 8,
                    "event_min_episode_sim": 0.58,
                },
                "retrieval": {
                    "top_m": 2,
                    "top_k": 6,
                    "baseline_top_k": 5,
                    "tau_fraction": 0.10,
                    "neighbor_radius": 1,
                    "recent_episodes": 8,
                    "pin_recent_n": 20,
                },
                "reasoner": {
                    "max_frames": 10,
                    "include_coarse_text": True,
                    "include_episodic_text": False,
                    "ground_episodic_archive": True,
                },
            }
        )

        self.assertEqual(30, cfg.recent_capacity)
        self.assertEqual(80, cfg.episodic_capacity)
        self.assertEqual(2, cfg.top_m)
        self.assertEqual(6, cfg.top_k)
        self.assertEqual(10, cfg.reasoner_max_frames)
        self.assertFalse(cfg.include_episodic_text)

    def test_eval_config_ignores_runner_only_reasoner_fields(self) -> None:
        cfg = eval_config_from_mapping(
            {
                "reasoner": {
                    "type": "text",
                    "model": "Qwen/Qwen2.5-3B-Instruct",
                    "share_event_vlm": True,
                }
            }
        )

        self.assertEqual(8, cfg.reasoner_max_frames)

    def test_write_run_config_persists_resolved_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            path = write_run_config(
                {
                    "experiment": "streamingbench_text",
                    "dataset": {"manifest": "data/eval_manifest_50.json"},
                },
                out_dir,
            )

            self.assertEqual(out_dir / "run_config.json", path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual("streamingbench_text", payload["experiment"])
            self.assertEqual("data/eval_manifest_50.json", payload["dataset"]["manifest"])

    def test_flatten_summary_metrics_keeps_core_numbers(self) -> None:
        metrics = flatten_summary_metrics(
            {
                "n_qas": 250,
                "videos_completed": 50,
                "hierarchical": {"accuracy": 0.516, "correct": 129},
                "baseline": {"accuracy": 0.448, "correct": 112},
                "delta_accuracy": 0.068,
            }
        )

        self.assertEqual(
            {
                "n_qas": 250,
                "videos_completed": 50,
                "hierarchical_accuracy": 0.516,
                "hierarchical_correct": 129,
                "baseline_accuracy": 0.448,
                "baseline_correct": 112,
                "delta_accuracy": 0.068,
            },
            metrics,
        )

    def test_write_metrics_persists_flat_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_metrics(
                {
                    "n_qas": 2,
                    "hierarchical": {"accuracy": 0.5, "correct": 1},
                    "baseline": {"accuracy": 0.0, "correct": 0},
                    "delta_accuracy": 0.5,
                },
                Path(tmp),
            )

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(0.5, payload["hierarchical_accuracy"])
            self.assertEqual(0.5, payload["delta_accuracy"])

    def test_write_manifest_snapshot_copies_entries(self) -> None:
        entries = [{"sample_id": 1, "video": "video.mp4", "qas": "qas.json"}]

        with tempfile.TemporaryDirectory() as tmp:
            path = write_manifest_snapshot(entries, Path(tmp))

            self.assertEqual(Path(tmp) / "manifest_snapshot.json", path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(entries, payload)


if __name__ == "__main__":
    unittest.main()
