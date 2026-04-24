# Memory Buffer for Streaming Video-LLM Systems

Hierarchical three-tier memory buffer (recent -> episodic -> long-term events) for streaming video QA, with coarse-to-fine retrieval and small-model captioners. Full walkthrough, math, and results live in `notebooks/solution.ipynb`.

## Layout

- `notebooks/solution.ipynb` — the main submission. Setup, stream reader, perception encoder, memory design, retrieval, LLM input, reasoner, baseline, 84-min LVBench run, conclusion, refs.
- `notebooks/*.svg` — architecture diagrams embedded in the notebook.
- `src/` — library code imported by the notebook.
  - `stream_reader.py` — OpenCV video → `RawWindow` generator (1 fps, 3 s windows).
  - `perception_encoder.py` — X-CLIP ViT-B/32 joint video/text encoder.
  - `data_structures.py` — `WindowEntry`, `EpisodeEntry`, `EventEntry`, `RetrievalResult`.
  - `memory_writer.py` — three-tier online writer: novelty gate, self-centrality pooling, event consolidation.
  - `summary_builder.py` — tier captioners: Florence-2 (window), Moondream2 (episode), Qwen2.5-VL-3B (event fusion).
  - `prompts.py` — all model prompts in one place.
  - `retriever.py` — coarse-to-fine retrieval with multiplicative time decay and tier-split visual scoring.
  - `formatter.py` — packs retrieval hits into an LLM-ready dict (`visual_context` + `text_context`).
  - `llm_reasoner.py` — Qwen2.5-3B-Instruct text-only MCQ head.
  - `baseline.py` — `RecentWindowBaseline` (flat deque-of-N) for head-to-head comparison.
  - `memory_db.py` — optional peewee/SQLite backing store for long runs.
- `scripts/` — optional CLI helpers (not needed to run the notebook).
  - `main.py` — standalone end-to-end pipeline + retrieval harness.
  - `download_video_sample.py` — fetch StreamingBench samples.
- `data/` — only `qas*.jsonl` annotations are tracked; videos are fetched locally.
- `requirements.txt` — pinned deps.

## Run

```bash
pip install -r requirements.txt
jupyter lab notebooks/solution.ipynb
```
