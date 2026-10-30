"""Build a replay from existing artifacts. No models, frames or inferred memory states."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = "results/streamingbench_text_3b_notebook"
ANNOTATIONS = "data/streamingbench/Real_Time_Visual_Understanding/shard_1_50/sample_36/qas.json"
REFERENCE_SOURCE_SHA256 = "120287733650bed8258e7a1af0f2737e14a3c6a855b7eecef7367117a4ed453f"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cited_ranges(text: str) -> list[dict]:
    """A citation in generated text is NOT a retrieval trace or verified observation."""
    return [{"start": float(a), "end": float(b), "basis": "model_output_only"}
            for a, b in re.findall(r"\[(\d+(?:\.\d+)?)\s*[–—-]\s*(\d+(?:\.\d+)?)s\]", text)]


def build_reference(root: Path = ROOT) -> dict:
    paths = [f"{RUN}/per_video/sample_36.jsonl", f"{RUN}/run_config.json",
             f"{RUN}/run_log.jsonl", ANNOTATIONS]
    rows = [json.loads(s) for s in (root / paths[0]).read_text().splitlines() if s.strip()]
    config = json.loads((root / paths[1]).read_text())
    annotation = json.loads((root / ANNOTATIONS).read_text())
    qas = {q["question_id"]: q for q in annotation["qas"]}
    logs = [json.loads(s) for s in (root / paths[2]).read_text().splitlines() if s.strip()]
    log = next(r for r in logs if r["sample_id"] == 36)
    if {r["question_id"] for r in rows} != set(qas):
        raise ValueError("Source question set mismatch")
    for row in rows:
        qa = qas[row["question_id"]]
        nominal_time = sum(float(part) * 60 ** i for i, part in enumerate(reversed(qa["time_stamp"].split(":"))))
        if abs(row["t_seconds"] - nominal_time) > 1e-6:
            raise ValueError("Source timestamp mismatch")
        for key in ("question", "options", "time_stamp"):
            if row[key] != qa[key]:
                raise ValueError(f"Source annotation mismatch: {key}")
        for side in ("hierarchical", "baseline"):
            if row[side]["ground_truth"] != qa["answer"]:
                raise ValueError("Source ground-truth mismatch")
            row[side]["citations"] = cited_ranges(row[side]["prediction"])
    session = {
        "schema_version": 1, "mode": "evaluation_reference", "case_id": "streamingbench/sample_36",
        "title": "Streaming Video Memory", "sample_id": 36,
        "model": config["reasoner_model"], "config": config["config"],
        "clock_end": max(r["stream_time"] for r in rows), "focus_question_id": rows[3]["question_id"],
        "records": rows, "nodes": {}, "snapshots": [], "retrievals": [],
        "media": {"available": False, "status": "rights_unverified", "source_sha256": REFERENCE_SOURCE_SHA256,
                  "reason": "Source footage is not cleared for redistribution. No video is bundled."},
        "provenance": {"prediction_kind": "previously_saved_model_output",
                       "memory_trace": "not_saved", "retrieval_trace": "not_saved",
                       "evidence_visual_check": "not_verified",
                       "source_files": [{"path": p, "sha256": digest(root / p)} for p in paths],
                       "citation_warning": "Model-cited times are unverified claims in the saved answer; they are not retrieved evidence.",
                       "run_totals": {k: log[k] for k in ("n_windows", "n_qas", "hier_correct", "base_correct")}},
    }
    review_path = root / "demo/source-review.json"
    if review_path.is_file():
        review = json.loads(review_path.read_text())
        if review["source_sha256"] != REFERENCE_SOURCE_SHA256:
            raise ValueError("Source review fingerprint mismatch")
        session["provenance"]["evidence_visual_check"] = "private_spot_check_not_a_retrieval_trace"
        session["provenance"]["visual_review"] = review["visual_finding"]
        session["clock_end"] = review["source_duration_seconds"]
        session["provenance"]["source_files"].append({"path": "demo/source-review.json", "sha256": digest(review_path)})
    validate(session)
    return session


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate(session: dict) -> None:
    if session.get("schema_version") not in (1, 2) or session.get("mode") not in ("evaluation_reference", "recorded_trace"):
        raise ValueError("Unsupported session schema")
    if not finite(session.get("clock_end")) or session["clock_end"] <= 0:
        raise ValueError("Invalid source clock")
    nodes = session["nodes"]
    for key, node in nodes.items():
        if key != node["id"] or node["kind"] not in ("window", "episode", "event"):
            raise ValueError("Invalid memory node")
        if not finite(node["start"]) or not finite(node["end"]) or not 0 <= node["start"] <= node["end"]:
            raise ValueError("Invalid memory time range")
        thumb = node.get("thumbnail")
        if thumb is not None and not re.fullmatch(r"data:image/jpeg;base64,[A-Za-z0-9+/=]+", thumb):
            raise ValueError("Invalid representative frame")
    previous = -1.0
    ids = set()
    for row in session["records"]:
        t = row["stream_time"]
        if not finite(t) or not finite(row["t_seconds"]) or not 0 <= row["t_seconds"] <= t <= session["clock_end"]:
            raise ValueError("Question precedes its evaluation time or is outside clock")
        if t < previous or row["question_id"] in ids:
            raise ValueError("Unordered or duplicate question")
        previous = t
        ids.add(row["question_id"])
        for side in ("hierarchical", "baseline"):
            a = row[side]
            if a["correct"] != (a["pred_letter"] == a["ground_truth"]):
                raise ValueError("Answer correctness mismatch")
    previous = -1.0
    for snapshot in session["snapshots"]:
        t = snapshot["time"]
        if not finite(t) or t < 0 or t < previous or t > session["clock_end"]:
            raise ValueError("Unordered memory snapshots")
        previous = t
        for tier in ("recent", "episodic", "events", "pending"):
            for node_id in snapshot[tier]:
                if node_id not in nodes or nodes[node_id]["end"] > t + 1e-6:
                    raise ValueError("Missing or future memory evidence")
    for retrieval in session["retrievals"]:
        row = next((r for r in session["records"] if r["question_id"] == retrieval["question_id"]), None)
        if row is None or retrieval["time"] != row["stream_time"]:
            raise ValueError("Retrieval/question clock mismatch")
        for group in ("coarse", "episodic", "grounded", "pinned"):
            for hit in retrieval[group]:
                n = nodes.get(hit["id"])
                if n is None or n["end"] > retrieval["time"] + 1e-6:
                    raise ValueError("Missing or future retrieval evidence")
                if hit["score"] is not None and not finite(hit["score"]):
                    raise ValueError("Invalid retrieval score")
    for decision in session.get("decisions", []):
        n = nodes.get(decision["window_id"])
        t = decision["time"]
        if n is None or n["kind"] != "window" or not finite(t) or n["end"] > t + 1e-6 or t > session["clock_end"]:
            raise ValueError("Missing or future memory decision")
        if not isinstance(decision["promoted"], bool):
            raise ValueError("Invalid memory decision")
    for baseline in session.get("baseline_inputs", []):
        row = next((r for r in session["records"] if r["question_id"] == baseline["question_id"]), None)
        if row is None or baseline["time"] != row["stream_time"]:
            raise ValueError("Baseline/question clock mismatch")
        for group in ("grounded", "pinned"):
            for hit in baseline[group]:
                n = nodes.get(hit["id"])
                if n is None or n["end"] > baseline["time"] + 1e-6:
                    raise ValueError("Missing or future baseline evidence")
    if session["mode"] == "evaluation_reference" and (nodes or session["snapshots"] or session["retrievals"]
                                                      or session.get("decisions") or session.get("baseline_inputs")):
        raise ValueError("Reference-only artifacts must not invent memory or retrieval")


def validate_media(session: dict, media: Path, rights: dict) -> None:
    """Bind this exact full-length source to its actual run, never an unrelated clip."""
    expected = session["media"].get("source_sha256")
    actual = digest(media)
    if not expected or expected != actual or rights.get("source_sha256") != actual:
        raise ValueError("Source alignment failed: video SHA-256 must match run and rights record")
    if rights.get("redistribution") != "cleared" or not rights.get("basis") or not rights.get("source_url"):
        raise ValueError("Redistribution rights are not documented")
    if rights.get("source_start_seconds", 0) != 0:
        raise ValueError("Use the full original source; retimed or trimmed clips are not supported")


if __name__ == "__main__":
    session = build_reference()
    out = ROOT / "demo/web/reference.json"
    out.write_text(json.dumps(session, ensure_ascii=False, indent=2) + "\n")
    print(f"Verified {len(session['records'])} saved questions; wrote {out}")
