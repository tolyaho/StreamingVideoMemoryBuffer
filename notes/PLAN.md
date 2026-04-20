# Memory Buffer for Streaming Video-LLM — Project Plan

Legend: [x] done · [ ] to do · [~] partial / needs update

---

## SCOPE LOCK (do not cross these lines)

- [x] Do not train any model
- [x] Do not build a full VLM/LLM pipeline
- [x] Do not implement KV-cache or token-level visual memory
- [x] Keep visual embeddings as the primary memory representation
- [x] Do not caption every frame — only promoted/representative windows
- [x] Do not make summaries the only memory
- [x] Compare hierarchical memory against a recent-window baseline

---

## DESIGN DECISIONS

- [x] Architecture locked: hierarchical visual memory + multi-scale summaries
- [x] Write path is query-agnostic (store by recency + novelty + capacity)
- [x] Read path is query-aware (top-down retrieval: recent → events → episodes → grounding)
- [x] Encoder choice: X-CLIP ViT-B/32 (microsoft/xclip-base-patch32, 512-dim, temporally-aware clip embeddings)
- [x] Summary model choice: Florence-2-base (microsoft/Florence-2-base, 270M)
- [x] Event fusion model: Qwen2.5-VL-3B-Instruct (multimodal — episode texts + representative frames per episode)
- [x] Summary gradient: lower tier = more visual; higher tier = more text
- [x] Episode summary: concatenate per-window Florence <DETAILED_CAPTION> outputs at episode flush
- [x] Event summary: Qwen2.5-VL sees episode summaries (text) + representative frames from self-centrality pooling
- [x] Episode embedding: time-aware self-centrality pooling (replaces mean pooling)
- [x] Event embedding: L2-normalised centroid of member episode embeddings (unchanged)
- [x] Retrieval order: stage 0 always searches recent by cosine sim; stage A events; stage B episodes; stage C archive grounding
- [x] Evaluation style decided: qualitative + precision@k on synthetic + baseline comparison
- [x] Research notes written (related_work_notes.md)
- [x] Architecture document written (ARCHITECTURE.md)
- [x] Model decisions recorded (notes/models.md)

---

## INFRASTRUCTURE — src/ modules

### Data structures
- [x] WindowEntry (entry_id, start_time, end_time, visual_embedding, frame, summary_text, summary_embedding, tier)
- [x] EpisodeEntry (entry_id, time range, visual_embedding centroid, member_window_ids, summary_text, summary_embedding)
- [x] EventEntry (entry_id, time range, visual_embedding centroid, member_episode_ids, representative_window_ids, summary_text, summary_embedding)
- [x] RetrievalResult (query, coarse_hits, episodic_hits, grounded_windows, scores)
- [x] WindowEntry.from_raw_window() bridge method

### Stream ingestion (stream_reader.py)
- [x] RawWindow dataclass (window_id, start_time, end_time, frames, representative_frame)
- [x] StreamReader: 1 fps sampling, OpenCV-based, lazy generator
- [x] StreamReader.synthetic_stream() for demos without real video
- [x] Window duration: 5 sec library default (`scripts/main.py` overrides to 3 sec)

### Perception encoder (perception_encoder.py)
- [x] PerceptionEncoder wrapping X-CLIP ViT-B/32 (up to 8 frames sampled per window; short windows repeat boundary frames)
- [x] encode_frames(): temporal clip embedding via X-CLIP MIT, L2-normalised
- [x] encode_text(): L2-normalised text embedding in same joint space
- [x] encode_window() convenience wrapper

### Memory writer (memory_writer.py)
- [x] Tier 1 — Recent memory: fixed-capacity deque, dense storage
- [x] Tier 2 — Episodic memory: novelty-filtered, online episode building
  - [x] Novelty check: cosine distance vs recent neighbours (TAS-inspired)
  - [x] Episode building: gap + similarity + max_len checks
  - [x] Episode flush on scene break or capacity
  - [x] Pending episode snapshot for live queryability
  - [x] Episode embedding: time-aware self-centrality pooling (center_score + consistency_score → softmax weights → weighted sum)
  - [x] representative_window_ids stored on EpisodeEntry (top-weight windows from pooling)
- [x] Tier 3 — Long-term memory: greedy semantic-temporal clustering (SDC-inspired)
  - [x] Centroid visual embedding (L2-normalised mean of episode embeddings)
  - [x] Representative windows sourced from EpisodeEntry.representative_window_ids (pooling winners)
  - [x] Summary text + optional summary embedding
  - [x] episode_frames passed to summary_fn for VLM event fusion
- [x] update() online per-window call
- [x] finalize() end-of-stream flush
- [x] get_searchable_episodes() including in-progress episode
- [x] get_grounding_windows() from window archive (available helper; current retriever grounding uses recent queue)
- [x] text_encode_fn hook for summary embeddings
- [x] stats() diagnostics

### Summary builder (summary_builder.py)
- [x] Template mode (fast, no model)
- [x] Florence-2 captioning mode (use_model=True)
  - [x] caption_frame() for single frame
  - [x] caption_episode(): run `<DETAILED_CAPTION>` on each member window's representative frame and concatenate
- [x] Qwen2.5-VL event fusion mode (use_vlm=True)
  - [x] _fuse_with_vlm(): episode texts + 2 frames per episode → one event sentence
- [x] __call__(entries, episode_frames=None) dispatch
- [x] Resilient: all model failures fall back to template

### Retriever (retriever.py)
- [x] Stage 0: always similarity-search recent windows → top_k hits by cosine sim (guaranteed fresh context)
- [x] Stage A: coarse routing over long-term EventEntries (blended visual + summary similarity)
- [x] Stage B: fine search over episodic entries within candidate time ranges (blended visual + summary + recency)
- [x] Stage C: grounding from window archive via episode member IDs (get_grounding_windows)
- [x] Recent hits + archive windows merged and deduplicated into grounded_windows
- [x] _blended_score() with weight renormalisation when summary_embedding is None
- [x] query_summary_embedding optional param for separate summary space

### Formatter (formatter.py)
- [x] format_text(): human-readable evidence block (timestamps + scores + summaries)
- [x] format_for_llm(): structured dict with visual_context + text_context
- [x] Explains why visual tokens are embeddings not projected tokens (limitation note)

### Baseline (baseline.py)
- [x] RecentWindowBaseline: fixed deque, same encoder and similarity function
- [x] retrieve(query_embedding, top_k) cosine search over recent only
- [x] stats()

### Pipeline runner (scripts/main.py)
- [x] End-to-end streaming loop: StreamReader → PerceptionEncoder → SummaryBuilder → HierarchicalMemoryWriter
- [x] Per-window Florence-2 captioning with live logging
- [x] Box-formatted episode and event output as they are flushed, including wrapped multiline summaries
- [x] Final stats footer (windows, episodes, events, elapsed time)
- [x] This script is a write-path / summarization demo; retrieval is exercised separately via `HierarchicalRetriever`

### Data download (scripts/download_video_sample.py)
- [x] StreamingBench annotation + media download
- [x] Shard-mode: extracts all 50 videos in one zip
- [x] Sample-mode: extracts single video
- [x] Zip member matching: exact path → normalised basename → sample ID from directory name
- [x] __MACOSX entries filtered out
- [x] QA JSON saved per sample

---

## DATA & VIDEO

- [x] Download StreamingBench sample (sample_1 present at data/)
- [x] Verify at least one video loads correctly with StreamReader
- [x] Verify 1 fps sampling produces reasonable windows on real video
- [ ] Pick 1-3 representative videos for notebook demo (aim for 20-60 min each)
- [ ] Prepare 5-8 example queries matched to video content

---

## NOTEBOOK (solution.ipynb)

### Structure — sections needed
- [x] Title + architecture overview (approach rationale + architecture writeup)
- [x] Architecture diagram (architecture.svg referenced in notebook)
- [ ] Setup cell (imports, path, config flags)
- [ ] Data structures section with rationale
- [ ] Perception encoder section with rationale
- [ ] Stream ingestion section with rationale + synthetic demo
- [ ] Memory buffer design section with rationale
- [ ] Streaming update loop (synthetic)
- [ ] Memory tier size visualisation over time
- [ ] Baseline section with rationale
- [ ] Retrieval section with rationale
- [ ] Retrieval timeline visualisation per query
- [ ] LLM input formatter section with rationale
- [ ] Full pipeline demo (synthetic)
- [ ] Precision@k evaluation (synthetic)
- [ ] Baseline vs hierarchical comparison table
- [ ] Memory snapshot visualisation
- [ ] Design trade-offs discussion (markdown)

### Still needed in notebook
- [ ] Run full pipeline with real video (swap USE_REAL_CLIP=True, VIDEO_PATH=...)
- [ ] Show actual sampled frames from real video with timestamps
- [ ] Show real concatenated Florence-2 `<DETAILED_CAPTION>` episode summaries
- [ ] Show real X-CLIP similarity scores on real queries
- [ ] Example queries on real video content with retrieved thumbnails displayed
- [ ] Side-by-side: baseline retrieval vs hierarchical retrieval on same query
- [ ] Note obvious failure cases from real video
- [ ] Final conclusion section answering the assignment questions explicitly

### Assignment questions the notebook must answer
- [ ] What is stored in memory?
- [ ] How is memory size controlled?
- [ ] What stays and what gets removed?
- [ ] How are summaries generated (window → episode → event)?
- [ ] How is retrieval done (coarse-to-fine)?
- [ ] How are summaries used at query time?
- [ ] How is retrieved memory formatted for LLM input?
- [ ] When does hierarchical memory help vs recent-window baseline?
- [ ] What are the main limitations?

---

## EVALUATION

- [x] Evaluation style decided: qualitative + precision@k + baseline comparison
- [ ] Precision@k on synthetic stream (known ground-truth scene assignments)
- [ ] Baseline comparison table (earliest hit, hit count)
- [ ] Memory tier size plots over time
- [ ] Qualitative retrieval inspection on real video
- [ ] Side-by-side baseline vs hierarchical on real queries
- [ ] Note where recent-window is already enough
- [ ] Note where hierarchical memory clearly helps (queries about old content)
- [ ] Note where Florence-2 captions improve reranking vs visual-only

---

## POLISH (Day 7)

- [ ] Restart kernel, run notebook top-to-bottom without errors
- [ ] Remove dead code and stale cells
- [ ] Every code cell has a short explanation above it
- [ ] All section titles consistent
- [ ] Add architecture diagram
- [ ] Improve retrieval visualisation to show actual video thumbnails
- [ ] Final conclusion section
- [ ] Export clean copy

---

## STRETCH GOALS (only if ahead)

- [ ] FAISS instead of brute-force cosine search
- [ ] Scene-change detection signal to trigger episode flush earlier
- [ ] Summary confidence / quality flag on Florence-2 output
- [ ] Tiny interactive query demo cell (ipywidgets text input)

---

## CONCRETE DEFAULTS (locked)

| Parameter | Value |
|-----------|-------|
| Retrieval encoder | X-CLIP ViT-B/32 (512-dim, up to 8-frame clip) |
| Summary model | Florence-2-base |
| Event fusion | Qwen2.5-VL-3B-Instruct |
| Sampling rate | 1 fps |
| Window duration | 3 sec |
| Recent capacity | 20 windows |
| Episodic capacity | 50 entries |
| Episode max gap | 4 sec |
| Episode min sim | 0.70 |
| Event max gap | 15 sec |
| Event min sim | 0.55 |
| Episodic merge batch | 10 |
| Retrieval top-M (coarse) | 2-3 events |
| Retrieval top-K (fine) | 4-5 episodes |
| Neighbour radius | 1 |
| Scoring weights | α=0.65 visual, β=0.30 summary, γ=0.05 recency |
| Baseline window | 10 windows |
| Novelty threshold | 0.05 (real video; 0.25 default) |
