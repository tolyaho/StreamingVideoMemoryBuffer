from __future__ import annotations

import json
import shutil
import sys
import time
import textwrap
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (
    HierarchicalMemoryWriter,
    HierarchicalRetriever,
    MemoryStore,
    PerceptionEncoder,
    ReasonerInputFormatter,
    StreamReader,
    SummaryBuilder,
    WindowEntry,
)
from src.data_structures import EpisodeEntry, EventEntry

VIDEO_PATH        = ROOT / "data/streamingbench/Real_Time_Visual_Understanding/shard_1_50/sample_36/video.mp4"
QAS_PATH          = VIDEO_PATH.parent / "qas.json"
FPS               = 1.0
WINDOW_DURATION   = 3.0
RECENT_CAPACITY   = 20
EPISODIC_CAPACITY = 10
EPISODE_MAX_GAP   = 4.0
EVENT_MAX_GAP     = 15.0
NOVELTY_THRESHOLD = 0.05   # lower than default 0.25 — real X-CLIP embeddings are dense
OUTPUT_DIR        = ROOT / "outputs"
DB_PATH           = OUTPUT_DIR / "memory.db"
RETRIEVAL_OUT     = OUTPUT_DIR / "retrievals.md"
RETRIEVAL_TOP_M   = 3   # coarse: top events
RETRIEVAL_TOP_K   = 5   # fine: top episodes / recent hits

W = 68  # line width for decorations


def _clean_outputs_dir(out_dir: Path) -> None:
    """Remove previous run artifacts under ``outputs/`` (fresh DB + retrievals, etc.)."""
    if not out_dir.is_dir():
        return
    for child in out_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def _hms_to_seconds(ts: str) -> float:
    """'HH:MM:SS' / 'MM:SS' / 'SS' → seconds."""
    parts = [float(p) for p in ts.strip().split(":")]
    if len(parts) == 3:
        h, m, s = parts
        return h * 3600 + m * 60 + s
    if len(parts) == 2:
        m, s = parts
        return m * 60 + s
    return parts[0]


def _load_qas(path: Path) -> list[dict]:
    """StreamingBench QAs sorted by timestamp, with a ``t_seconds`` field added per QA."""
    if not path.exists():
        print(f"[main] no qas.json at {path} — retrieval step skipped.")
        return []
    payload = json.loads(path.read_text())
    qas = payload.get("qas", [])
    for qa in qas:
        qa["t_seconds"] = _hms_to_seconds(qa.get("time_stamp", "00:00:00"))
    qas.sort(key=lambda q: q["t_seconds"])
    return qas


def _rule(char: str = "─") -> None:
    print(char * W)


def _box_lines(text: str) -> list[str]:
    inner_width = W - 6
    lines: list[str] = []
    for para in text.splitlines() or [""]:
        wrapped = textwrap.wrap(para, width=inner_width) or [""]
        lines.extend(wrapped)
    return lines


def _print_window(n: int, raw, note: str) -> None:
    print(f"  [{raw.start_time:7.1f} – {raw.end_time:6.1f}s]  win {n:03d}  {note}", flush=True)


def _print_episode(ep: EpisodeEntry, idx: int) -> None:
    header = f"  episode {idx:02d}  [{ep.start_time:.1f} – {ep.end_time:.1f}s]  {len(ep.member_window_ids)} windows"
    print(f"\n  ┌{'─' * (W - 4)}┐")
    print(f"  │  {header:<{W - 6}}  │")
    for line in _box_lines(ep.summary_text):
        print(f"  │  {line:<{W - 6}}  │")
    print(f"  └{'─' * (W - 4)}┘")


def _print_event(ev: EventEntry, idx: int) -> None:
    header = f"  EVENT {idx:02d}  [{ev.start_time:.1f} – {ev.end_time:.1f}s]  {len(ev.member_episode_ids)} episodes"
    print(f"\n  ╔{'═' * (W - 4)}╗")
    print(f"  ║  {header:<{W - 6}}  ║")
    for line in _box_lines(ev.summary_text):
        print(f"  ║  {line:<{W - 6}}  ║")
    print(f"  ╚{'═' * (W - 4)}╝")


def _flush_new(memory: HierarchicalMemoryWriter, ep_seen: int, ev_seen: int) -> tuple[int, int]:
    while len(memory.episodic) > ep_seen:
        ep_seen += 1
        _print_episode(memory.episodic[ep_seen - 1], ep_seen)

    while len(memory.long_term) > ev_seen:
        ev_seen += 1
        _print_event(memory.long_term[ev_seen - 1], ev_seen)

    return ep_seen, ev_seen


def _render_retrieval(qa: dict, stream_time: float, result, formatter: ReasonerInputFormatter) -> str:
    """One QA → self-contained markdown block (QA metadata + formatted evidence)."""
    options_lines = "\n".join(f"  - {opt}" for opt in qa.get("options", []))
    options_block = f"- **options**:\n{options_lines}\n" if options_lines else ""
    evidence = formatter.format_text(result)
    return (
        f"## {qa.get('question_id', '?')}\n"
        f"- **fired at stream t** = {stream_time:.2f}s  "
        f"(qa timestamp = {qa['t_seconds']:.2f}s)\n"
        f"- **task_type** = {qa.get('task_type', '?')}\n"
        f"- **question** = {qa.get('question', '')}\n"
        f"{options_block}"
        f"- **ground_truth** = {qa.get('answer', '?')}\n\n"
        f"```\n{evidence}\n```\n"
    )


def _process_due_qas(
    qas: list[dict],
    cursor: int,
    stream_time: float,
    memory: HierarchicalMemoryWriter,
    encoder: PerceptionEncoder,
    retriever: HierarchicalRetriever,
    formatter: ReasonerInputFormatter,
    out_stream,
    ep_seen: int,
    ev_seen: int,
) -> tuple[int, int, int]:
    """Fire retrieval for every QA whose timestamp has been reached.

    If at least one QA is due, ``memory.flush_pending()`` is called first so the
    currently-forming episode/event becomes visible to coarse/fine routing. Any
    episodes/events produced by the flush are printed before the retrieval output.

    Returns ``(cursor, ep_seen, ev_seen)`` so the caller can keep its display counters
    in sync.
    """
    if cursor >= len(qas) or qas[cursor]["t_seconds"] > stream_time:
        return cursor, ep_seen, ev_seen

    memory.flush_pending()
    ep_seen, ev_seen = _flush_new(memory, ep_seen, ev_seen)

    while cursor < len(qas) and qas[cursor]["t_seconds"] <= stream_time:
        qa = qas[cursor]
        q_emb = encoder.encode_text(qa["question"])
        result = retriever.retrieve(
            query=qa["question"],
            query_embedding=q_emb,
            memory=memory,
            top_m=RETRIEVAL_TOP_M,
            top_k=RETRIEVAL_TOP_K,
            query_time=stream_time,
        )
        print()
        print(formatter.format_text(result))
        print(f"  (query: {qa.get('question_id', '?')} @ stream t={stream_time:.1f}s)")
        out_stream.write(_render_retrieval(qa, stream_time, result, formatter) + "\n")
        out_stream.flush()
        cursor += 1
    return cursor, ep_seen, ev_seen


def main() -> None:
    _clean_outputs_dir(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    encoder = PerceptionEncoder()
    summary = SummaryBuilder(use_model=True, use_vlm=True, use_moondream=True)
    reader  = StreamReader(fps=FPS, window_duration=WINDOW_DURATION)
    store   = MemoryStore(DB_PATH)
    memory  = HierarchicalMemoryWriter(
        recent_capacity=RECENT_CAPACITY,
        episodic_capacity=EPISODIC_CAPACITY,
        novelty_threshold=NOVELTY_THRESHOLD,
        episode_max_gap=EPISODE_MAX_GAP,
        event_max_gap=EVENT_MAX_GAP,
        summary_fn=summary,
        text_encode_fn=encoder.encode_text,
        store=store,
    )
    retriever = HierarchicalRetriever()
    formatter = ReasonerInputFormatter()
    qas       = _load_qas(QAS_PATH)

    retrieval_file = RETRIEVAL_OUT.open("w", encoding="utf-8")
    retrieval_file.write(f"# Retrievals for {VIDEO_PATH.name}\n\n")
    retrieval_file.write(f"- total QAs: {len(qas)}\n")
    retrieval_file.write(f"- top_m (events): {RETRIEVAL_TOP_M}\n")
    retrieval_file.write(f"- top_k (episodes / recent): {RETRIEVAL_TOP_K}\n\n")
    retrieval_file.flush()

    _rule("━")
    print(f" Streaming: {VIDEO_PATH.name}")
    print(f" QAs loaded: {len(qas)}  ·  retrieval → {RETRIEVAL_OUT}")
    _rule("━")
    print()

    t0 = time.perf_counter()
    n, ep_seen, ev_seen = 0, 0, 0
    qa_cursor = 0

    for raw in reader.read_windows(str(VIDEO_PATH)):
        vis_emb = encoder.encode_window(raw)
        note    = summary.build_window_caption(raw)
        entry   = WindowEntry.from_raw_window(raw, visual_embedding=vis_emb, summary_text=note)
        memory.update(entry)
        n += 1
        _print_window(n, raw, note)
        ep_seen, ev_seen = _flush_new(memory, ep_seen, ev_seen)
        qa_cursor, ep_seen, ev_seen = _process_due_qas(
            qas, qa_cursor, raw.end_time,
            memory, encoder, retriever, formatter, retrieval_file,
            ep_seen, ev_seen,
        )

    memory.finalize()
    ep_seen, ev_seen = _flush_new(memory, ep_seen, ev_seen)
    qa_cursor, ep_seen, ev_seen = _process_due_qas(
        qas, qa_cursor, float("inf"),
        memory, encoder, retriever, formatter, retrieval_file,
        ep_seen, ev_seen,
    )
    retrieval_file.close()
    elapsed = time.perf_counter() - t0

    s = memory.stats()
    print()
    _rule("━")
    print(
        f" {n} windows  ·  {s['n_episodes_flushed']} episodes"
        f"  ·  {s['long_term']} events  ·  {elapsed:.1f}s"
    )
    print(f" promoted {s['n_promoted']}  ·  discarded {s['n_discarded']}"
          f"  ·  recent {s['recent']}  ·  episodic {s['episodic']}")
    db_counts = store.counts()
    print(f" db → windows {db_counts['windows']}  ·  episodes {db_counts['episodes']}"
          f"  ·  events {db_counts['events']}  ·  {DB_PATH}")
    _rule("━")
    store.close()


if __name__ == "__main__":
    main()
