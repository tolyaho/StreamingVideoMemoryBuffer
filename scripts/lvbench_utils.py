from __future__ import annotations

import json
import random
import re
from pathlib import Path

OPTION_BLOCK_RE = re.compile(
    r"\(([A-D])\)\s*(.*?)(?=\s*\([A-D]\)|\Z)",
    re.DOTALL,
)


def parse_question_and_options(raw: str) -> tuple[str, list[str]]:
    stem, _, tail = raw.partition("\n(A)")
    if not tail:
        return raw.strip(), []
    block = "(A)" + tail
    options = [
        f"{m.group(1)}. {m.group(2).strip()}"
        for m in OPTION_BLOCK_RE.finditer(block)
    ]
    return stem.strip(), options


def parse_time_reference(ref: str) -> str:
    ref = ref.strip()
    if "-" in ref:
        return ref.split("-", 1)[1].strip()
    return ref


def convert_qa(raw: dict) -> dict:
    question, options = parse_question_and_options(raw["question"])
    types = raw.get("question_type") or []
    return {
        "question_id": str(raw.get("uid", "")),
        "task_type": ", ".join(types) if types else "unknown",
        "question": question,
        "time_stamp": parse_time_reference(raw.get("time_reference", "00:00:00")),
        "answer": raw["answer"],
        "options": options,
    }


def load_meta(meta_path: Path) -> list[dict]:
    rows = []
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_clip_bundle(
    row: dict,
    data_root: Path,
) -> dict:
    video_key = row["key"]
    clip_dir = data_root / video_key
    clip_dir.mkdir(parents=True, exist_ok=True)

    qas = [convert_qa(q) for q in row.get("qa", [])]
    qas_path = clip_dir / "qas.json"
    qas_path.write_text(
        json.dumps(
            {
                "video_key": video_key,
                "type": row.get("type"),
                "video_info": row.get("video_info"),
                "qas": qas,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "video_key": video_key,
        "type": row.get("type"),
        "video": str(clip_dir / "video.mp4"),
        "qas": str(qas_path),
        "n_qas": len(qas),
        "duration_minutes": (row.get("video_info") or {}).get("duration_minutes"),
    }


def sample_videos(meta_path: Path, n: int, seed: int) -> list[dict]:
    rows = load_meta(meta_path)
    rng = random.Random(seed)
    if n >= len(rows):
        return rows
    return rng.sample(rows, n)
