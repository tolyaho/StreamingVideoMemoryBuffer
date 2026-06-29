from .data_structures import WindowEntry, EpisodeEntry, EventEntry, RetrievalResult
from .stream_reader import StreamReader, RawWindow
from .perception_encoder import PerceptionEncoder
from .memory_writer import HierarchicalMemoryWriter
from .summary_builder import SummaryBuilder
from .retriever import HierarchicalRetriever
from .formatter import ReasonerInputFormatter
from .baseline import RecentWindowBaseline
from .llm_reasoner import LLMReasoner, build_prompt as build_llm_prompt
from .qwen_vl_io import DEFAULT_VLM_MODEL
from .reasoner_frames import FrameEvidence, collect_frames
from .vlm_reasoner import VLMReasoner, DEFAULT_VLM_REASONER

__all__ = [
    "WindowEntry",
    "EpisodeEntry",
    "EventEntry",
    "RetrievalResult",
    "RawWindow",
    "StreamReader",
    "PerceptionEncoder",
    "HierarchicalMemoryWriter",
    "MemoryStore",
    "SummaryBuilder",
    "HierarchicalRetriever",
    "ReasonerInputFormatter",
    "RecentWindowBaseline",
    "LLMReasoner",
    "VLMReasoner",
    "DEFAULT_VLM_MODEL",
    "DEFAULT_VLM_REASONER",
    "FrameEvidence",
    "collect_frames",
    "build_llm_prompt",
]


def __getattr__(name: str):
    if name == "MemoryStore":
        from .memory_db import MemoryStore

        return MemoryStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
