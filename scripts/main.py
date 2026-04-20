from __future__ import annotations

import sys
import time
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (
    HierarchicalMemoryWriter,
    PerceptionEncoder,
    StreamReader,
    SummaryBuilder,
    WindowEntry,
)
from src.data_structures import EpisodeEntry, EventEntry

VIDEO_PATH        = ROOT / "data/Real-Time%20Visual%20Understanding_1-50/sample_36/video.mp4"
FPS               = 1.0
WINDOW_DURATION   = 3.0
RECENT_CAPACITY   = 20
EPISODIC_CAPACITY = 5
EPISODE_MAX_GAP   = 4.0
EVENT_MAX_GAP     = 15.0
NOVELTY_THRESHOLD = 0.05   # lower than default 0.25 — real X-CLIP embeddings are dense

W = 68  # line width for decorations


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


def main() -> None:
    encoder = PerceptionEncoder()
    summary = SummaryBuilder(use_model=True, use_vlm=True)
    reader  = StreamReader(fps=FPS, window_duration=WINDOW_DURATION)
    memory  = HierarchicalMemoryWriter(
        recent_capacity=RECENT_CAPACITY,
        episodic_capacity=EPISODIC_CAPACITY,
        novelty_threshold=NOVELTY_THRESHOLD,
        episode_max_gap=EPISODE_MAX_GAP,
        event_max_gap=EVENT_MAX_GAP,
        summary_fn=summary,
    )

    _rule("━")
    print(f" Streaming: {VIDEO_PATH.name}")
    _rule("━")
    print()

    t0 = time.perf_counter()
    n, ep_seen, ev_seen = 0, 0, 0

    for raw in reader.read_windows(str(VIDEO_PATH)):
        vis_emb = encoder.encode_window(raw)
        note    = summary.build_window_caption(raw)
        entry   = WindowEntry.from_raw_window(raw, visual_embedding=vis_emb, summary_text=note)
        memory.update(entry)
        n += 1
        _print_window(n, raw, note)
        ep_seen, ev_seen = _flush_new(memory, ep_seen, ev_seen)

    memory.finalize()
    ep_seen, ev_seen = _flush_new(memory, ep_seen, ev_seen)
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
    _rule("━")


if __name__ == "__main__":
    main()
