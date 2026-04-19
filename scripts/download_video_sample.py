#!/usr/bin/env python3
"""download shard_1_50 of Real_Time_Visual_Understanding from StreamingBench."""
import ast
import json
import re
import shutil
import zipfile
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download

SUBSET = "Real_Time_Visual_Understanding"
ZIP_NAME = "Real-Time Visual Understanding_1-50.zip"
OUTPUT_DIR = Path("data/streamingbench") / SUBSET / "shard_1_50"


def parse_options(raw):
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
    except Exception:
        return [raw]
    return [str(x) for x in parsed] if isinstance(parsed, list) else [str(parsed)]


def sample_id_from_row(row):
    qid = str(row.get("question_id", ""))
    suffix = qid.split("_sample_", 1)[1]
    return int(suffix.split("_", 1)[0])


def find_video_in_zip(zf, sample_id):
    """find the mp4 for a given sample id inside the zip."""
    videos = [
        m for m in zf.namelist()
        if m.lower().endswith(".mp4") and not m.endswith("/") and "__MACOSX" not in m
    ]
    for m in videos:
        for part in Path(m).parts:
            if re.match(rf"^sample_0*{sample_id}$", part, re.IGNORECASE):
                return m
        if re.search(rf"sample_0*{sample_id}\.mp4$", m, re.IGNORECASE):
            return m
    return None


def extract_video(zf, member, dest_dir):
    tmp = Path(zf.extract(member, path=dest_dir))
    final = dest_dir / "video.mp4"
    if tmp != final:
        shutil.move(str(tmp), str(final))
        parent = tmp.parent
        while parent != dest_dir and parent.exists():
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    return final


def main():
    print("loading annotations...")
    ds = load_dataset("ngqtrung/StreamingBench", split=SUBSET)
    grouped = {}
    for row in ds:
        sid = sample_id_from_row(row)
        if 1 <= sid <= 50:
            grouped.setdefault(sid, []).append(dict(row))

    print(f"found {len(grouped)} samples, downloading zip...")
    zip_path = Path(hf_hub_download(
        repo_id="mjuicem/StreamingBench",
        repo_type="dataset",
        filename=ZIP_NAME,
    ))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        for sid in sorted(grouped):
            rows = grouped[sid]
            sample_dir = OUTPUT_DIR / f"sample_{sid}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            member = find_video_in_zip(zf, sid)
            if member is None:
                print(f"  sample_{sid}: video not found in zip, skipping")
                continue

            video_path = extract_video(zf, member, sample_dir)

            qas = [{
                "question_id": r.get("question_id"),
                "task_type": r.get("task_type"),
                "question": r.get("question"),
                "time_stamp": r.get("time_stamp"),
                "answer": r.get("answer"),
                "options": parse_options(r.get("options", [])),
            } for r in rows]

            (sample_dir / "qas.json").write_text(
                json.dumps({"sample_id": sid, "video": str(video_path), "qas": qas}, indent=2),
                encoding="utf-8",
            )
            print(f"  sample_{sid}: {video_path.name} ({len(rows)} QAs)")

    print("done.")


if __name__ == "__main__":
    main()
