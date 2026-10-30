"""Passive observer for real eval_common.run_video runs. Never generates model content."""
from __future__ import annotations

import base64
import io
import json
import time as clock
from pathlib import Path

from demo.artifacts import digest, validate
from src.memory_writer import cosine_sim

SCHEMA_VERSION = 2


class TraceRecorder:
    def __init__(self, video: Path, qas: Path, model: str, config: str, *,
                 include_frames: bool = False, thumb_size: tuple[int, int] = (384, 216)):
        self.include_frames = include_frames
        self.thumb_size = thumb_size
        self._t0 = clock.perf_counter()
        self._observed_writer = None
        self._undated: list[dict] = []
        self._unstamped: list[str] = []
        self.session = {
            "schema_version": SCHEMA_VERSION, "mode": "recorded_trace", "case_id": f"local/{digest(video)[:12]}",
            "title": "Streaming Video Memory", "model": model, "config": config,
            "clock_end": 0.0, "records": [], "nodes": {}, "snapshots": [], "retrievals": [],
            "decisions": [], "baseline_inputs": [],
            "media": {"available": False, "source_sha256": digest(video), "status": "rights_unverified"},
            "provenance": {"prediction_kind": "newly_computed_model_output",
                           "memory_trace": "observed_during_run", "retrieval_trace": "observed_during_run",
                           "source_files": [{"path": "source_video", "sha256": digest(video)},
                                            {"path": "source_qas", "sha256": digest(qas)}]},
        }

    def _wall(self) -> float:
        return round(clock.perf_counter() - self._t0, 3)

    def _observe_writer(self, memory) -> None:
        """Record the writer's own promote/discard decision for every evicted window.

        Wraps the bound ``_is_novel`` of this one writer instance: the original method still makes
        the decision and its return value is passed through unchanged. The similarity logged next
        to it is the same quantity the writer thresholds (max cosine to the windows still in recent),
        read from the same embeddings at the same moment.
        """
        if self._observed_writer is memory:
            return
        if memory._n_promoted or memory._n_discarded:
            raise ValueError("Observer attached after the first eviction; decisions would be incomplete")
        original = memory._is_novel

        def observed(entry):
            sims = [cosine_sim(entry.visual_embedding, w.visual_embedding) for w in memory.recent]
            promoted = original(entry)
            self._undated.append({
                "window_id": entry.entry_id, "promoted": bool(promoted),
                "max_similarity": round(max(sims), 6) if sims else None,
                "threshold": memory.novelty_threshold,
            })
            return promoted

        original_flush = memory._flush_current_episode

        def observed_flush():
            # An episode can be flushed and consolidated inside one update(); record it as it is made.
            before = len(memory.episodic)
            original_flush()
            if len(memory.episodic) > before:
                self.node(memory.episodic[-1])
                self._unstamped.append(memory.episodic[-1].entry_id)

        memory._is_novel = observed
        memory._flush_current_episode = observed_flush
        self._observed_writer = memory

    def node(self, entry) -> str:
        if entry.entry_id in self.session["nodes"]:
            return entry.entry_id  # windows, flushed episodes and events are immutable once created
        if hasattr(entry, "member_episode_ids"):
            kind, members = "event", entry.member_episode_ids
        elif hasattr(entry, "member_window_ids"):
            kind, members = "episode", entry.member_window_ids
        else:
            kind, members = "window", []
        node = {"id": entry.entry_id, "kind": kind, "start": float(entry.start_time),
                "end": float(entry.end_time), "caption": entry.summary_text,
                "members": list(members), "representatives": list(getattr(entry, "representative_window_ids", []))}
        if self.include_frames and kind == "window" and entry.frame is not None:
            from PIL import Image
            image = Image.fromarray(entry.frame).convert("RGB")
            image.thumbnail(self.thumb_size)
            b = io.BytesIO()
            image.save(b, format="JPEG", quality=84)
            node["thumbnail"] = "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode("ascii")
        self.session["nodes"][entry.entry_id] = node
        return entry.entry_id

    def on_memory(self, time: float, memory, phase: str = "stream") -> None:
        self._observe_writer(memory)
        # Direct lists only: get_searchable_episodes() can caption a pending episode.
        # An observer must never add inference or change the memory writer's behavior.
        snapshot = {"time": float(time), "phase": phase, "wall_s": self._wall(),
                    "recent": [self.node(w) for w in memory.recent],
                    "episodic": [self.node(e) for e in memory.episodic],
                    "events": [self.node(e) for e in memory.long_term],
                    "pending": [self.node(w) for w in memory._pending_episode],
                    "stats": memory.stats()}
        for decision in self._undated:
            decision.update(time=float(time), phase=phase)
            self.session["decisions"].append(decision)
        self._undated = []
        # creation time = the stream time of the first observation after the node was written
        for node_id in self._unstamped + snapshot["events"] + snapshot["episodic"]:
            self.session["nodes"][node_id].setdefault("created", float(time))
        self._unstamped = []
        self.session["snapshots"].append(snapshot)
        self.session["clock_end"] = max(self.session["clock_end"], float(time))

    def on_retrieval(self, qa: dict, time: float, result, formatted: dict, memory) -> None:
        self.on_memory(time, memory, phase="question")
        record = {"question_id": qa["question_id"], "time": float(time), "wall_s": self._wall(),
                  "text_context": formatted["text_context"]}
        for key, entries in (("coarse", result.coarse_hits), ("episodic", result.episodic_hits),
                             ("grounded", result.grounded_windows), ("pinned", result.pinned_windows)):
            record[key] = [{"id": self.node(e), "score": float(result.scores[e.entry_id])
                            if e.entry_id in result.scores else None} for e in entries]
        self.session["retrievals"].append(record)

    def on_baseline(self, qa: dict, time: float, result, formatted: dict) -> None:
        record = {"question_id": qa["question_id"], "time": float(time), "text_context": formatted["text_context"]}
        for key, entries in (("grounded", result.grounded_windows), ("pinned", result.pinned_windows)):
            record[key] = [{"id": self.node(e), "score": float(result.scores[e.entry_id])
                            if e.entry_id in result.scores else None} for e in entries]
        self.session["baseline_inputs"].append(record)

    def on_question(self, row: dict) -> None:
        row = json.loads(json.dumps(row, allow_nan=False))
        row["wall_s"] = self._wall()
        self.session["records"].append(row)

    def write(self, path: Path, focus_question_id: str | None = None) -> None:
        records = self.session["records"]
        if not records:
            raise ValueError("No answered questions to replay")
        ids = [r["question_id"] for r in records]
        if focus_question_id is not None and focus_question_id not in ids:
            raise ValueError("Focus question was not answered in this run")
        self.session["focus_question_id"] = focus_question_id or ids[0]
        self.session["provenance"]["processing_wall_s"] = self._wall()
        validate(self.session)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.session, ensure_ascii=False, allow_nan=False, indent=1) + "\n")
