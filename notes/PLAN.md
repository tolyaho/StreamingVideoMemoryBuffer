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
- [x] Read path is query-aware (coarse-to-fine retrieval)
- [x] Encoder choice: SigLIP ViT-B/16 (google/siglip-base-patch16-224, 768-dim)
- [x] Summary model choice: Florence-2-base (microsoft/Florence-2-base, 270M)
- [x] Event fusion model: Qwen2.5-1.5B-Instruct (text-only, for event summary fusion)
- [x] Reasoning: text-only via Florence-2 captions → Qwen prompt (no visual projector)
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
- [x] Window duration: 5 sec default

### Perception encoder (perception_encoder.py)
- [x] PerceptionEncoder wrapping SigLIP ViT-B/16
- [x] encode_frames(): mean-pools frame embeddings, L2-normalised
- [x] encode_text(): L2-normalised text embedding in same space
- [x] encode_window() convenience wrapper
- [x] MockEncoder for fast demos (no model download needed)
- [x] MockEncoder.add_query() for controlled retrieval demos

### Memory writer (memory_writer.py)
- [x] Tier 1 — Recent memory: fixed-capacity deque, dense storage
- [x] Tier 2 — Episodic memory: novelty-filtered, online episode building
  - [x] Novelty check: cosine distance vs recent neighbours (TAS-inspired)
  - [x] Episode building: gap + similarity + max_len checks
  - [x] Episode flush on scene break or capacity
  - [x] Pending episode snapshot for live queryability
- [x] Tier 3 — Long-term memory: greedy semantic-temporal clustering (SDC-inspired)
  - [x] Centroid visual embedding
  - [x] 1-3 representative window IDs
  - [x] Summary text + optional summary embedding
- [x] update() online per-window call
- [x] finalize() end-of-stream flush
- [x] get_searchable_episodes() including in-progress episode
- [x] get_grounding_windows() from window archive (not recent queue)
- [x] text_encode_fn hook for summary embeddings
- [x] stats() diagnostics

### Summary builder (summary_builder.py)
- [x] Template mode (fast, no model)
- [x] Florence-2 captioning mode (use_model=True)
  - [x] caption_frame() for single frame
  - [x] caption_episode() on representative window using <DETAILED_CAPTION>
- [x] Qwen2.5-1.5B event fusion mode (use_llm=True)
  - [x] _fuse_with_llm() fuses episode summaries into one event sentence
- [x] __call__ dispatch: WindowEntry list → episode summary, EpisodeEntry list → event summary
- [x] Resilient: all model failures fall back to template

### Retriever (retriever.py)
- [x] Stage A: coarse routing over long-term EventEntries (visual + summary similarity, no recency)
- [x] Stage B: fine search over episodic entries within candidate time ranges (visual + summary + recency)
- [x] Stage C: local grounding from recent queue
  - [x] Recent windows temporally adjacent to episodic hits
  - [x] Always append recent tail (recency is always valuable per SimpleStream)
- [x] Fallback to direct recent search when episodic is empty
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

### Data download (scripts/download_video_sample.py)
- [x] StreamingBench annotation + media download
- [x] Shard-mode: extracts all 50 videos in one zip
- [x] Sample-mode: extracts single video
- [x] Zip member matching: exact path → normalised basename → sample ID from directory name
- [x] __MACOSX entries filtered out
- [x] QA JSON saved per sample, manifest saved per shard

---

## DATA & VIDEO

- [ ] Download StreamingBench shard (run: python scripts/download_video_sample.py --keep-zip)
- [ ] Verify at least one video loads correctly with StreamReader
- [ ] Pick 1-3 representative videos for notebook demo (aim for 20-60 min each)
- [ ] Prepare 5-8 example queries matched to video content
- [ ] Verify 1 fps sampling produces reasonable windows on real video

---

## NOTEBOOK (solution.ipynb)

### Structure — sections needed
- [ ] Title + architecture overview
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
- [ ] Show real Florence-2 captions on representative episode frames
- [ ] Show real SigLIP similarity scores on real queries
- [ ] Example queries on real video content with retrieved thumbnails displayed
- [ ] Side-by-side: baseline retrieval vs hierarchical retrieval on same query
- [ ] Note obvious failure cases from real video
- [ ] Architecture diagram (one figure showing the full pipeline)
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
- [x] Precision@k on synthetic stream (known ground-truth scene assignments)
- [x] Baseline comparison table (earliest hit, hit count)
- [x] Memory tier size plots over time
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
| Retrieval encoder | SigLIP ViT-B/16 (768-dim) |
| Summary model | Florence-2-base |
| Event fusion | Qwen2.5-1.5B-Instruct |
| Sampling rate | 1 fps |
| Window duration | 5 sec |
| Recent capacity | 15-20 windows |
| Episodic capacity | 30-50 entries |
| Episode max gap | 10 sec |
| Episode min sim | 0.70 |
| Event max gap | 45 sec |
| Event min sim | 0.55 |
| Episodic merge batch | 8-10 |
| Retrieval top-M (coarse) | 2-3 events |
| Retrieval top-K (fine) | 4-5 episodes |
| Neighbour radius | 1 |
| Scoring weights | α=0.65 visual, β=0.30 summary, γ=0.05 recency |
| Baseline window | 15 windows |
