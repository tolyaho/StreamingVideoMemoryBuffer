import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/eval_manifest_50.json"
SHARD = "data/streamingbench/Real_Time_Visual_Understanding/shard_1_50"
RUNS = ("streamingbench_text_3b_tuned", "streamingbench_text_3b_notebook", "streamingbench_vlm_8b")


class EvalManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.entries = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_lists_samples_1_to_50_with_relative_paths(self) -> None:
        self.assertEqual([e["sample_id"] for e in self.entries], list(range(1, 51)))
        for e in self.entries:
            sid = e["sample_id"]
            self.assertEqual(e["video"], f"{SHARD}/sample_{sid}/video.mp4")
            self.assertEqual(e["qas"], f"{SHARD}/sample_{sid}/qas.json")

    def test_matches_committed_annotations(self) -> None:
        by_id = {e["sample_id"]: e for e in self.entries}
        for qas_path in sorted((ROOT / SHARD).glob("sample_*/qas.json")):
            qas = json.loads(qas_path.read_text(encoding="utf-8"))
            entry = by_id[qas["sample_id"]]
            self.assertEqual(entry["video"], qas["video"])
            self.assertEqual(ROOT / entry["qas"], qas_path)

    def test_covers_the_committed_runs(self) -> None:
        ids = {e["sample_id"] for e in self.entries}
        for run in RUNS:
            log = ROOT / "results" / run / "run_log.jsonl"
            run_ids = {json.loads(line)["sample_id"] for line in log.read_text(encoding="utf-8").splitlines() if line}
            self.assertEqual(run_ids, ids, run)


if __name__ == "__main__":
    unittest.main()
