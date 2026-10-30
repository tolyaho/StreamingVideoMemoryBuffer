"""Deterministic artifact, source binding and passive-observer regressions. No model calls."""
import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

from demo.artifacts import ANNOTATIONS, ROOT, RUN, build_reference, digest, validate, validate_media
from demo.server import attach_annotation, byte_range, make_server
from demo.trace import TraceRecorder
from scripts.eval_common import EvalConfig, run_video
from src import HierarchicalRetriever, ReasonerInputFormatter
from src.stream_reader import RawWindow


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.session = build_reference()

    def test_predictions_are_verbatim_and_bound_to_annotations(self):
        originals = [json.loads(line) for line in (ROOT / RUN / "per_video/sample_36.jsonl").read_text().splitlines()]
        self.assertEqual(len(self.session["records"]), 5)
        for original, replay in zip(originals, self.session["records"]):
            for side in ("hierarchical", "baseline"):
                self.assertEqual(original[side]["prediction"], replay[side]["prediction"])
        focus = self.session["records"][3]
        self.assertEqual(focus["t_seconds"], 602)
        self.assertEqual(focus["stream_time"], 602.04)
        self.assertEqual(focus["hierarchical"]["pred_letter"], "A")
        self.assertEqual(focus["baseline"]["pred_letter"], "B")
        self.assertEqual(focus["hierarchical"]["citations"], [{"start": 533., "end": 536., "basis": "model_output_only"}])

    def test_changed_source_annotation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for source in self.session["provenance"]["source_files"]:
                p = root / source["path"]
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes((ROOT / source["path"]).read_bytes())
            path = root / ANNOTATIONS
            data = json.loads(path.read_text())
            data["qas"][3]["question"] = "A different video's question"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "annotation mismatch"):
                build_reference(root)

    def test_reference_does_not_fill_missing_trace(self):
        self.assertEqual(self.session["nodes"], {})
        self.assertEqual(self.session["retrievals"], [])
        self.assertEqual(self.session["snapshots"], [])
        self.assertFalse(self.session["media"]["available"])
        self.session["nodes"]["fake"] = {"id": "fake", "kind": "window", "start": 0, "end": 1}
        with self.assertRaisesRegex(ValueError, "invent"):
            validate(self.session)

    def test_timestamp_and_source_question_sets_cannot_drift(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for source in self.session["provenance"]["source_files"]:
                p = root / source["path"]
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes((ROOT / source["path"]).read_bytes())
            path = root / RUN / "per_video/sample_36.jsonl"
            records = [json.loads(line) for line in path.read_text().splitlines()]
            records[3]["t_seconds"] = 601.0
            path.write_text("\n".join(json.dumps(row) for row in records))
            with self.assertRaisesRegex(ValueError, "timestamp mismatch"):
                build_reference(root)
            path.write_text("\n".join(json.dumps(row) for row in records[:3]))
            with self.assertRaisesRegex(ValueError, "question set mismatch"):
                build_reference(root)

    def test_question_cannot_precede_evaluation_time(self):
        self.session["records"][3]["stream_time"] = 601
        with self.assertRaisesRegex(ValueError, "Question precedes"):
            validate(self.session)

    def test_source_video_must_match_trace_and_rights(self):
        with tempfile.TemporaryDirectory() as temp:
            video = Path(temp) / "unit-test-bytes.mp4"
            video.write_bytes(b"unit test bytes, not a demo video")
            h = digest(video)
            rights = {"source_sha256": h, "redistribution": "cleared", "basis": "self-created test fixture", "source_url": "local:test-fixture"}
            with self.assertRaisesRegex(ValueError, "Source alignment"):
                validate_media(self.session, video, rights)
            self.session["media"]["source_sha256"] = h
            validate_media(self.session, video, rights)
            with self.assertRaisesRegex(ValueError, "rights"):
                validate_media(self.session, video, {"source_sha256": h})
            rights["source_start_seconds"] = 30
            with self.assertRaisesRegex(ValueError, "full original"):
                validate_media(self.session, video, rights)


class TraceTests(unittest.TestCase):
    def make_trace(self, root):
        video, qas = root / "test-source.bin", root / "qas.json"
        video.write_bytes(b"unit test fixture; not exported to browser")
        qas.write_text(json.dumps({"qas": [{"question_id": "test_q1", "question": "Unit-test question", "time_stamp": "00:00:06", "answer": "A", "options": ["A. test", "B. other"]}]}))
        return video, qas, TraceRecorder(video, qas, "unit-test-stub", "unit-test-fixture")

    def run_fixture(self, video, qas, observer=None):
        calls = []

        class Encoder:
            def encode_window(self, raw):
                return np.array([1., 0.] if int(raw.start_time) % 2 else [0., 1.], dtype=np.float32)

            def encode_text(self, _text):
                return np.array([1., 0.], dtype=np.float32)

        class Summary:
            def build_window_caption(self, raw):
                calls.append(("window", raw.window_id))
                return "unit-test caption"

            def __call__(self, entries, **_kwargs):
                calls.append(("summary", len(entries)))
                return "unit-test summary"

        class Reasoner:
            def answer(self, *_args, **_kwargs):
                return "A. test\nUnit-test model stub."

        windows = [RawWindow(f"test_w{i}", i, i + 1., [np.zeros((2, 2, 3), dtype=np.uint8)]) for i in range(10)]
        with patch("scripts.eval_common.StreamReader.read_windows", return_value=iter(windows)):
            rows, meta = run_video(video, qas, "test", Encoder(), Summary(), HierarchicalRetriever(),
                                   ReasonerInputFormatter(), Reasoner(),
                                   cfg=EvalConfig(recent_capacity=1, episodic_capacity=1), observer=observer)
        return rows, meta["memory_stats"], calls

    def test_observer_preserves_predictions_memory_and_caption_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video, qas, recorder = self.make_trace(root)
            without = self.run_fixture(video, qas)
            with_trace = self.run_fixture(video, qas, recorder)
            self.assertEqual(without, with_trace)
            recorder.write(root / "trace.json")
            session = json.loads((root / "trace.json").read_text())
            validate(session)
            self.assertTrue(session["retrievals"])
            self.assertTrue(any(s["events"] for s in session["snapshots"]))
            self.assertFalse(any(n.get("thumbnail") for n in session["nodes"].values()))

    def test_every_eviction_decision_is_the_writers_own(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video, qas, recorder = self.make_trace(root)
            _, stats, _ = self.run_fixture(video, qas, recorder)
            decisions = recorder.session["decisions"]
            self.assertEqual(len(decisions), stats["n_promoted"] + stats["n_discarded"])
            self.assertEqual(sum(d["promoted"] for d in decisions), stats["n_promoted"])
            snaps = recorder.session["snapshots"]
            for d in decisions:
                # the logged similarity is exactly what the writer thresholds
                if d["max_similarity"] is None:  # empty recent buffer (finalize): the writer always keeps it
                    self.assertTrue(d["promoted"])
                else:
                    self.assertEqual(d["promoted"], (1 - d["max_similarity"]) > d["threshold"])
                if d["phase"] != "stream":
                    continue
                after = next(s for s in snaps if s["time"] == d["time"] and s["phase"] == "stream")
                self.assertNotIn(d["window_id"], after["recent"])
                kept = any(d["window_id"] in after[t] for t in ("pending",)) or any(
                    d["window_id"] in recorder.session["nodes"][e]["members"] for e in after["episodic"])
                self.assertEqual(kept, d["promoted"])

    def test_baseline_input_and_finalize_phase_are_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video, qas, recorder = self.make_trace(root)
            self.run_fixture(video, qas, recorder)
            s = recorder.session
            self.assertEqual([b["question_id"] for b in s["baseline_inputs"]], ["test_q1"])
            self.assertEqual(s["baseline_inputs"][0]["time"], s["records"][0]["stream_time"])
            self.assertEqual(s["snapshots"][-1]["phase"], "finalize")
            self.assertTrue(all(sn["phase"] in ("stream", "question", "finalize") for sn in s["snapshots"]))
            bad = copy.deepcopy(s)
            bad["decisions"][0]["time"] = bad["nodes"][bad["decisions"][0]["window_id"]]["end"] - 0.5
            with self.assertRaisesRegex(ValueError, "future memory decision"):
                validate(bad)

    def test_every_event_member_episode_is_recorded(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video, qas, recorder = self.make_trace(root)
            self.run_fixture(video, qas, recorder)
            nodes = recorder.session["nodes"]
            events = [n for n in nodes.values() if n["kind"] == "event"]
            self.assertTrue(events)
            for event in events:
                for member in event["members"]:
                    self.assertIn(member, nodes)  # even episodes consolidated within one update()
                    self.assertLessEqual(nodes[member]["created"], event["created"])

    def test_observer_attached_late_is_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, _, recorder = self.make_trace(root)

            class Writer:
                _n_promoted, _n_discarded = 1, 0

            with self.assertRaisesRegex(ValueError, "after the first eviction"):
                recorder._observe_writer(Writer())

    def test_future_memory_and_retrieval_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video, qas, recorder = self.make_trace(root)
            self.run_fixture(video, qas, recorder)
            data = recorder.session
            validate(data)
            bad = copy.deepcopy(data)
            node_id = bad["snapshots"][0]["recent"][0]
            bad["nodes"][node_id]["end"] = 1000
            with self.assertRaisesRegex(ValueError, "future memory"):
                validate(bad)
            bad = copy.deepcopy(data)
            hit = bad["retrievals"][0]["pinned"][0]
            bad["nodes"][hit["id"]]["end"] = 9
            bad["snapshots"] = []
            with self.assertRaisesRegex(ValueError, "future retrieval"):
                validate(bad)

    def test_questions_beyond_video_are_not_assigned_infinite_demo_time(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video, qas, recorder = self.make_trace(root)
            data = json.loads(qas.read_text())
            data["qas"][0]["time_stamp"] = "00:05:00"
            qas.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "beyond"):
                self.run_fixture(video, qas, recorder)


class ShowcaseBundleTests(unittest.TestCase):
    """The committed showcase records must stay internally consistent and bound to one clip."""
    SHOW = ROOT / "demo/showcase"

    def test_traces_validate_and_bind_to_the_rights_record(self):
        rights = json.loads((self.SHOW / "rights.json").read_text())
        annotation = json.loads((self.SHOW / "annotation.json").read_text())
        frozen = dict(reversed(line.split()) for line in (self.SHOW / "FROZEN_BEFORE_INFERENCE.sha256").read_text().splitlines() if "  " in line)
        self.assertEqual(frozen["qas.json"], digest(self.SHOW / "qas.json"))
        self.assertEqual(frozen["annotation.json"], digest(self.SHOW / "annotation.json"))
        self.assertEqual(frozen["bbb-excerpt-0033-0459.webm"], rights["source_sha256"])
        for name, config in (("trace-notebook.json", "notebook"), ("trace-short_clip_demo.json", "short_clip_demo")):
            session = json.loads((self.SHOW / name).read_text())
            validate(session)
            self.assertEqual(session["config"], config)
            self.assertEqual(session["media"]["source_sha256"], rights["source_sha256"])
            sources = {f["path"]: f["sha256"] for f in session["provenance"]["source_files"]}
            self.assertEqual(sources["source_qas"], digest(self.SHOW / "qas.json"))
            attach_annotation(session, annotation, digest(self.SHOW / "annotation.json"))
            self.assertTrue(all(n["kind"] != "window" or "thumbnail" in n for n in session["nodes"].values()))

    def test_recorded_showcase_results_are_unchanged(self):
        expected = {"trace-notebook.json": ("CBBD", "CDBD"), "trace-short_clip_demo.json": ("CCBD", "CDBD")}
        for name, (hier, base) in expected.items():
            rows = json.loads((self.SHOW / name).read_text())["records"]
            self.assertEqual("".join(r["hierarchical"]["pred_letter"] for r in rows), hier)
            self.assertEqual("".join(r["baseline"]["pred_letter"] for r in rows), base)
        apple = next(r for r in json.loads((self.SHOW / "trace-notebook.json").read_text())["records"] if r["question_id"] == "bbb_q2_apple")
        self.assertTrue(apple["hierarchical"]["prediction"].startswith("B. A red apple."))
        self.assertTrue(apple["baseline"]["prediction"].startswith("D. A carrot."))


class ServerTests(unittest.TestCase):
    def test_uncleared_trace_thumbnails_are_removed_from_served_data(self):
        session = build_reference()
        session["mode"] = "recorded_trace"
        session["nodes"] = {"w": {"id": "w", "kind": "window", "start": 0., "end": 1., "thumbnail": "data:image/jpeg;base64,AA=="}}
        session["snapshots"] = [{"time": 1., "recent": ["w"], "episodic": [], "events": [], "pending": []}]
        server = make_server(session, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urlopen(f"http://127.0.0.1:{server.server_port}/api/session") as response:
                served = json.load(response)
            self.assertNotIn("thumbnail", served["nodes"]["w"])
            self.assertIn("thumbnail", session["nodes"]["w"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_bound_media_is_seekable_with_real_http_ranges(self):
        with tempfile.TemporaryDirectory() as temp:
            media = Path(temp) / "test-bytes.mp4"
            media.write_bytes(bytes(range(100)))  # HTTP transport fixture, never demo footage.
            session = build_reference()
            session["media"]["source_sha256"] = digest(media)
            rights = {"source_sha256": digest(media), "redistribution": "cleared", "basis": "self-created unit bytes", "source_url": "local:unit-fixture"}
            server = make_server(session, media=media, rights=rights, port=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}/media/source"
            try:
                with urlopen(Request(url, headers={"Range": "bytes=95-200"})) as response:
                    self.assertEqual(response.status, 206)
                    self.assertEqual(response.headers["Content-Range"], "bytes 95-99/100")
                    self.assertEqual(response.read(), bytes(range(95, 100)))
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(url, headers={"Range": "bytes=100-"}))
                self.assertEqual(error.exception.code, 416)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_ranges_include_suffix_seek_and_reject_invalid_requests(self):
        self.assertEqual(byte_range("bytes=10-19", 100), (10, 19))
        self.assertEqual(byte_range("bytes=-10", 100), (90, 99))
        self.assertEqual(byte_range("bytes=98-", 100), (98, 99))
        for header in ("bytes=100-", "bytes=20-10", "bytes=1-2,4-5", "bytes=-0", "invalid"):
            with self.assertRaises(ValueError):
                byte_range(header, 100)

    def test_only_web_and_bound_data_are_served(self):
        server = make_server(build_reference(), port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base + "/api/session") as response:
                data = json.load(response)
                self.assertFalse(data["media"]["available"])
            for path in ("/.git/config", "/../README.md", "/%2e%2e/README.md", "/notes/cookies.txt", "/media/source"):
                with self.assertRaises(HTTPError) as raised:
                    urlopen(base + path)
                self.assertEqual(raised.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
