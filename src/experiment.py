from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Mapping

from .perception_encoder import PerceptionEncoder
from .retriever import HierarchicalRetriever
from .formatter import ReasonerInputFormatter
from .summary_builder import SummaryBuilder
from .qwen_vl_io import TEXT_EVAL_EVENT_VLM

from scripts.eval_common import EvalConfig
from scripts.eval_common import print_final_summary, refresh_summary, run_video, save_video_results
from scripts.reasoner_factory import build_reasoner, default_reasoner_model


ROOT = Path(__file__).resolve().parents[1]


def _plain_mapping(cfg: Any) -> dict:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass
    return dict(cfg)


def eval_config_from_mapping(cfg: Mapping[str, Any]) -> EvalConfig:
    data = _plain_mapping(cfg)
    memory = data.get("memory", {})
    retrieval = data.get("retrieval", {})
    reasoner = data.get("reasoner", {})

    values = asdict(EvalConfig())
    values.update(memory)
    values.update(retrieval)

    aliases = {
        "max_frames": "reasoner_max_frames",
    }
    for key, value in reasoner.items():
        values[aliases.get(key, key)] = value

    allowed = {f.name for f in fields(EvalConfig)}
    return EvalConfig(**{k: v for k, v in values.items() if k in allowed})


def write_run_config(cfg: Mapping[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "run_config.json"
    path.write_text(
        json.dumps(_plain_mapping(cfg), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def flatten_summary_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "n_qas": summary.get("n_qas", 0),
        "videos_completed": summary.get("videos_completed", 0),
        "hierarchical_accuracy": summary.get("hierarchical", {}).get("accuracy"),
        "hierarchical_correct": summary.get("hierarchical", {}).get("correct"),
        "baseline_accuracy": summary.get("baseline", {}).get("accuracy"),
        "baseline_correct": summary.get("baseline", {}).get("correct"),
        "delta_accuracy": summary.get("delta_accuracy"),
    }


def write_metrics(summary: Mapping[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics.json"
    path.write_text(
        json.dumps(flatten_summary_metrics(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def write_manifest_snapshot(entries: list[Mapping[str, Any]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest_snapshot.json"
    path.write_text(
        json.dumps(list(entries), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_manifest(path: str | Path) -> list[dict]:
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def select_manifest_entries(entries: list[dict], dataset_cfg: Mapping[str, Any]) -> list[dict]:
    limit = dataset_cfg.get("limit")
    samples = dataset_cfg.get("samples") or []
    keys = dataset_cfg.get("keys") or []

    if samples:
        wanted = {int(x) for x in samples}
        entries = [e for e in entries if e.get("sample_id") in wanted]
    if keys:
        wanted = {str(x) for x in keys}
        entries = [e for e in entries if e.get("video_key") in wanted]
    if limit:
        entries = entries[: int(limit)]
    return entries


def run_experiment(cfg: Mapping[str, Any]) -> dict:
    data = _plain_mapping(cfg)
    dataset_cfg = data["dataset"]
    output_cfg = data["output"]
    reasoner_cfg = data["reasoner"]
    summary_cfg = data["summary"]

    out_dir = resolve_path(output_cfg["dir"])
    per_video_dir = out_dir / "per_video"
    per_video_dir.mkdir(parents=True, exist_ok=True)

    manifest = select_manifest_entries(load_manifest(dataset_cfg["manifest"]), dataset_cfg)
    if not manifest:
        raise SystemExit("no manifest entries to run")

    write_run_config(data, out_dir)
    write_manifest_snapshot(manifest, out_dir)

    eval_cfg = eval_config_from_mapping(data)
    reasoner_type = reasoner_cfg["type"]
    reasoner_model = reasoner_cfg.get("model") or default_reasoner_model(reasoner_type)
    event_vlm_name = reasoner_model if reasoner_type == "vlm" else TEXT_EVAL_EVENT_VLM

    encoder = PerceptionEncoder()
    summary_builder = SummaryBuilder(
        use_model=summary_cfg.get("use_model", True),
        use_vlm=summary_cfg.get("use_vlm", True),
        use_moondream=summary_cfg.get("use_moondream", True),
        vlm_model_name=event_vlm_name,
    )
    retriever = HierarchicalRetriever(tau_fraction=eval_cfg.tau_fraction)
    formatter = ReasonerInputFormatter()
    reasoner = build_reasoner(
        reasoner_type,
        reasoner_model,
        eval_cfg,
        summary_builder,
        share_event_vlm=reasoner_cfg.get("share_event_vlm", True),
    )

    records_glob = dataset_cfg.get("records_glob", "*.jsonl")
    id_field = dataset_cfg.get("id_field", "sample_id")
    title = output_cfg.get("summary_title", "Batch eval summary")

    for i, entry in enumerate(manifest, start=1):
        clip_id = str(entry[id_field])
        out_path = per_video_dir / f"{clip_id}.jsonl"
        if output_cfg.get("resume", False) and out_path.is_file():
            print(f"[{i}/{len(manifest)}] {clip_id} — skip")
            continue

        video_path = resolve_path(entry["video"])
        qas_path = resolve_path(entry["qas"])
        print(f"[{i}/{len(manifest)}] {clip_id} — streaming {video_path.name}")

        records, meta = run_video(
            video_path,
            qas_path,
            clip_id=clip_id,
            encoder=encoder,
            summary_builder=summary_builder,
            retriever=retriever,
            formatter=formatter,
            reasoner=reasoner,
            cfg=eval_cfg,
            extra_meta={
                id_field: entry[id_field],
                "reasoner_type": reasoner_type,
                "reasoner_model": reasoner_model,
                "experiment": data.get("experiment"),
            },
        )
        _, h_ok, b_ok = save_video_results(
            out_dir,
            per_video_dir,
            out_path,
            records,
            meta,
            records_glob=records_glob,
            summary_title=title,
        )
        print(f"         done · hier {h_ok}/{len(records)} · base {b_ok}/{len(records)}")

    summary = refresh_summary(
        out_dir,
        per_video_dir,
        records_glob=records_glob,
        title=title,
    )
    write_metrics(summary, out_dir)
    print_final_summary(summary, out_dir / "summary.md")
    return summary


def main() -> None:
    from hydra import main as hydra_main

    @hydra_main(version_base=None, config_path="../configs", config_name="config")
    def _main(cfg) -> None:
        run_experiment(cfg)

    _main()


if __name__ == "__main__":
    main()
