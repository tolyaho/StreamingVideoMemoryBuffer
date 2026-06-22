from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from src import (
    HierarchicalMemoryWriter,
    HierarchicalRetriever,
    LLMReasoner,
    PerceptionEncoder,
    ReasonerInputFormatter,
    RecentWindowBaseline,
    RetrievalResult,
    StreamReader,
    SummaryBuilder,
    WindowEntry,
)
from src.memory_writer import cosine_sim


@dataclass(frozen=True)
class EvalConfig:
    fps: float = 1.0
    window_duration: float = 3.0
    recent_capacity: int = 20
    baseline_recent_capacity: int | None = None
    episodic_capacity: int = 100
    episode_max_gap: float = 8.0
    event_max_gap: float = 45.0
    novelty_threshold: float = 0.05
    episodic_merge_batch: int = 10
    event_min_episode_sim: float = 0.55
    top_m: int = 3
    top_k: int = 5
    baseline_top_k: int | None = None
    tau_fraction: float = 0.25
    neighbor_radius: int = 1
    recent_episodes: int = 5
    pin_recent_n: int | None = None

    @property
    def baseline_windows(self) -> int:
        if self.baseline_recent_capacity is not None:
            return self.baseline_recent_capacity
        return self.recent_capacity

    @property
    def baseline_k(self) -> int:
        if self.baseline_top_k is not None:
            return self.baseline_top_k
        return self.top_k

    @property
    def pin_n(self) -> int:
        if self.pin_recent_n is not None:
            return self.pin_recent_n
        return self.baseline_windows


NOTEBOOK_CONFIG = EvalConfig()


def hms_to_seconds(ts: str) -> float:
    parts = [float(p) for p in ts.strip().split(":")]
    if len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    return parts[0]


def load_qas(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    qas = payload.get("qas", [])
    for qa in qas:
        qa["t_seconds"] = hms_to_seconds(qa.get("time_stamp", "00:00:00"))
    qas.sort(key=lambda q: q["t_seconds"])
    return qas


def parse_letter(prediction: str) -> str:
    return next((c for c in prediction.strip() if c in "ABCD"), "?")


def answer_hierarchical(
    qa: dict,
    stream_time: float,
    memory: HierarchicalMemoryWriter,
    encoder: PerceptionEncoder,
    retriever: HierarchicalRetriever,
    formatter: ReasonerInputFormatter,
    reasoner: LLMReasoner,
    cfg: EvalConfig,
) -> dict:
    memory.flush_pending()
    q_emb = encoder.encode_text(qa["question"])
    result = retriever.retrieve(
        query=qa["question"],
        query_embedding=q_emb,
        memory=memory,
        top_m=cfg.top_m,
        top_k=cfg.top_k,
        neighbor_radius=cfg.neighbor_radius,
        query_time=stream_time,
        recent_episodes=cfg.recent_episodes,
        pin_recent_n=cfg.pin_n,
    )
    llm_input = formatter.format_for_llm(result, query_embedding=q_emb)
    prediction = reasoner.answer(llm_input, options=qa.get("options"))
    letter = parse_letter(prediction)
    gt = qa["answer"]
    return {
        "prediction": prediction,
        "pred_letter": letter,
        "ground_truth": gt,
        "correct": letter == gt,
    }


def answer_baseline(
    qa: dict,
    encoder: PerceptionEncoder,
    baseline: RecentWindowBaseline,
    formatter: ReasonerInputFormatter,
    reasoner: LLMReasoner,
    cfg: EvalConfig,
) -> dict:
    q_emb = encoder.encode_text(qa["question"])
    hits = baseline.retrieve(q_emb, top_k=cfg.baseline_k)
    scores = {w.entry_id: float(cosine_sim(q_emb, w.visual_embedding)) for w in hits}
    result = RetrievalResult(
        query=qa["question"],
        coarse_hits=[],
        episodic_hits=[],
        grounded_windows=hits,
        scores=scores,
    )
    llm_input = formatter.format_for_llm(result, query_embedding=q_emb)
    prediction = reasoner.answer(llm_input, options=qa.get("options"))
    letter = parse_letter(prediction)
    gt = qa["answer"]
    return {
        "prediction": prediction,
        "pred_letter": letter,
        "ground_truth": gt,
        "correct": letter == gt,
    }


def process_due_qas(
    qas: list[dict],
    cursor: int,
    stream_time: float,
    memory: HierarchicalMemoryWriter,
    baseline: RecentWindowBaseline,
    encoder: PerceptionEncoder,
    retriever: HierarchicalRetriever,
    formatter: ReasonerInputFormatter,
    reasoner: LLMReasoner,
    records: list[dict],
    cfg: EvalConfig,
) -> int:
    while cursor < len(qas) and qas[cursor]["t_seconds"] <= stream_time:
        qa = qas[cursor]
        row = {
            "question_id": qa.get("question_id"),
            "task_type": qa.get("task_type"),
            "question": qa.get("question"),
            "time_stamp": qa.get("time_stamp"),
            "t_seconds": qa.get("t_seconds"),
            "stream_time": stream_time,
            "options": qa.get("options"),
            "hierarchical": answer_hierarchical(
                qa, stream_time, memory, encoder, retriever, formatter, reasoner, cfg
            ),
            "baseline": answer_baseline(
                qa, encoder, baseline, formatter, reasoner, cfg
            ),
        }
        records.append(row)
        cursor += 1
    return cursor


def run_video(
    video_path: Path,
    qas_path: Path,
    clip_id: str,
    encoder: PerceptionEncoder,
    summary_builder: SummaryBuilder,
    retriever: HierarchicalRetriever,
    formatter: ReasonerInputFormatter,
    reasoner: LLMReasoner,
    cfg: EvalConfig = NOTEBOOK_CONFIG,
    extra_meta: dict | None = None,
) -> tuple[list[dict], dict]:
    qas = load_qas(qas_path)
    reader = StreamReader(fps=cfg.fps, window_duration=cfg.window_duration)
    memory = HierarchicalMemoryWriter(
        recent_capacity=cfg.recent_capacity,
        episodic_capacity=cfg.episodic_capacity,
        novelty_threshold=cfg.novelty_threshold,
        episode_max_gap=cfg.episode_max_gap,
        event_max_gap=cfg.event_max_gap,
        episodic_merge_batch=cfg.episodic_merge_batch,
        event_min_episode_sim=cfg.event_min_episode_sim,
        summary_fn=summary_builder,
        text_encode_fn=encoder.encode_text,
        store=None,
    )
    baseline = RecentWindowBaseline(n_windows=cfg.baseline_windows)

    records: list[dict] = []
    cursor = 0
    n_windows = 0
    t0 = time.perf_counter()

    for raw in reader.read_windows(str(video_path)):
        vis_emb = encoder.encode_window(raw)
        note = summary_builder.build_window_caption(raw)
        entry = WindowEntry.from_raw_window(
            raw, visual_embedding=vis_emb, summary_text=note
        )
        memory.update(entry)
        baseline.update(entry)
        n_windows += 1
        cursor = process_due_qas(
            qas, cursor, raw.end_time,
            memory, baseline, encoder, retriever, formatter, reasoner, records, cfg,
        )

    memory.finalize()
    cursor = process_due_qas(
        qas, cursor, float("inf"),
        memory, baseline, encoder, retriever, formatter, reasoner, records, cfg,
    )
    elapsed = time.perf_counter() - t0
    meta = {
        "clip_id": clip_id,
        "video": str(video_path),
        "n_qas": len(qas),
        "n_qas_answered": len(records),
        "n_windows": n_windows,
        "elapsed_s": round(elapsed, 1),
        "memory_stats": memory.stats(),
    }
    if extra_meta:
        meta.update(extra_meta)
    return records, meta


def aggregate(all_records: list[dict]) -> dict:
    n = len(all_records)
    if n == 0:
        return {"n_qas": 0}

    hier_correct = sum(1 for r in all_records if r["hierarchical"]["correct"])
    base_correct = sum(1 for r in all_records if r["baseline"]["correct"])

    by_task: dict = defaultdict(lambda: {"n": 0, "hier": 0, "base": 0})
    for r in all_records:
        task = r.get("task_type") or "unknown"
        by_task[task]["n"] += 1
        by_task[task]["hier"] += int(r["hierarchical"]["correct"])
        by_task[task]["base"] += int(r["baseline"]["correct"])

    hier_only = base_only = both = neither = 0
    for r in all_records:
        h, b = r["hierarchical"]["correct"], r["baseline"]["correct"]
        if h and not b:
            hier_only += 1
        elif b and not h:
            base_only += 1
        elif h and b:
            both += 1
        else:
            neither += 1

    return {
        "n_qas": n,
        "hierarchical": {
            "correct": hier_correct,
            "accuracy": round(hier_correct / n, 4),
        },
        "baseline": {
            "correct": base_correct,
            "accuracy": round(base_correct / n, 4),
        },
        "delta_accuracy": round((hier_correct - base_correct) / n, 4),
        "paired": {
            "hierarchical_only": hier_only,
            "baseline_only": base_only,
            "both_correct": both,
            "both_wrong": neither,
        },
        "by_task_type": {
            k: {
                "n": v["n"],
                "hierarchical_accuracy": round(v["hier"] / v["n"], 4),
                "baseline_accuracy": round(v["base"] / v["n"], 4),
            }
            for k, v in sorted(by_task.items())
        },
    }


def write_summary_md(summary: dict, path: Path, title: str = "Batch eval summary") -> None:
    lines = [
        f"# {title}",
        "",
        f"- **QAs**: {summary['n_qas']}",
        f"- **Hierarchical**: {summary['hierarchical']['correct']}/{summary['n_qas']} "
        f"({100 * summary['hierarchical']['accuracy']:.1f}%)",
        f"- **Baseline**: {summary['baseline']['correct']}/{summary['n_qas']} "
        f"({100 * summary['baseline']['accuracy']:.1f}%)",
        f"- **Δ (hier − base)**: {100 * summary['delta_accuracy']:+.1f} pp",
        "",
        "## Paired outcomes",
        "",
        "| Outcome | Count |",
        "|---------|-------|",
        f"| Hierarchical only | {summary['paired']['hierarchical_only']} |",
        f"| Baseline only | {summary['paired']['baseline_only']} |",
        f"| Both correct | {summary['paired']['both_correct']} |",
        f"| Both wrong | {summary['paired']['both_wrong']} |",
        "",
        "## By task type",
        "",
        "| Task type | N | Hier acc | Base acc |",
        "|-----------|---|----------|----------|",
    ]
    for task, row in summary.get("by_task_type", {}).items():
        lines.append(
            f"| {task} | {row['n']} | "
            f"{100 * row['hierarchical_accuracy']:.1f}% | "
            f"{100 * row['baseline_accuracy']:.1f}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_all_records(per_video_dir: Path, pattern: str = "*.jsonl") -> list[dict]:
    records = []
    for path in sorted(per_video_dir.glob(pattern)):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def save_video_results(
    out_dir: Path,
    per_video_dir: Path,
    out_path: Path,
    records: list[dict],
    meta: dict,
    *,
    records_glob: str = "*.jsonl",
    summary_title: str = "Batch eval summary",
) -> tuple[dict, int, int]:
    with out_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    h_ok = sum(1 for r in records if r["hierarchical"]["correct"])
    b_ok = sum(1 for r in records if r["baseline"]["correct"])
    meta = {**meta, "hier_correct": h_ok, "base_correct": b_ok}
    with (out_dir / "run_log.jsonl").open("a", encoding="utf-8") as log_f:
        log_f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    summary = refresh_summary(
        out_dir, per_video_dir, records_glob=records_glob, title=summary_title
    )
    return summary, h_ok, b_ok


def refresh_summary(
    out_dir: Path,
    per_video_dir: Path,
    *,
    records_glob: str = "*.jsonl",
    title: str = "Batch eval summary",
) -> dict:
    summary = aggregate(load_all_records(per_video_dir, records_glob))
    summary["videos_completed"] = len(list(per_video_dir.glob(records_glob)))
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_summary_md(summary, out_dir / "summary.md", title=title)
    return summary


def print_final_summary(summary: dict, summary_path: Path) -> None:
    print()
    print("=" * 60)
    print(
        f"FINAL  hier {summary['hierarchical']['correct']}/{summary['n_qas']} "
        f"({100 * summary['hierarchical']['accuracy']:.1f}%)"
    )
    print(
        f"       base {summary['baseline']['correct']}/{summary['n_qas']} "
        f"({100 * summary['baseline']['accuracy']:.1f}%)"
    )
    print(f"       Δ    {100 * summary['delta_accuracy']:+.1f} pp")
    print(f"written → {summary_path}")
    print("=" * 60)
