"""Construct text or VLM MCQ reasoners for batch eval."""
from __future__ import annotations

from typing import Any, Literal

from src import LLMReasoner, SummaryBuilder, VLMReasoner
from src.qwen_vl_io import DEFAULT_VLM_MODEL

from scripts.eval_common import EvalConfig

ReasonerType = Literal["text", "vlm"]

DEFAULT_TEXT_REASONER = "Qwen/Qwen2.5-3B-Instruct"


def default_reasoner_model(reasoner_type: ReasonerType) -> str:
    if reasoner_type == "vlm":
        return DEFAULT_VLM_MODEL
    return DEFAULT_TEXT_REASONER


def build_reasoner(
    reasoner_type: ReasonerType,
    model_name: str,
    cfg: EvalConfig,
    summary_builder: SummaryBuilder | None = None,
    *,
    share_event_vlm: bool = True,
) -> Any:
    if reasoner_type == "text":
        return LLMReasoner(model_name=model_name)

    vlm_kwargs = dict(
        model_name=model_name,
        max_frames=cfg.reasoner_max_frames,
    )
    if (
        share_event_vlm
        and summary_builder is not None
        and getattr(summary_builder, "use_vlm", False)
        and getattr(summary_builder, "vlm_model_name", None) == model_name
        and getattr(summary_builder, "_vlm", None) is not None
    ):
        return VLMReasoner.from_summary_builder(summary_builder, **vlm_kwargs)
    return VLMReasoner(**vlm_kwargs)
