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
- [x] Window captioner: Florence-2-base (microsoft/Florence-2-base, ~0.23B) — `<CAPTION>` per ingested window
- [x] Episode captioner: **Moondream2** (`vikhyatk/moondream2`, ~1.87B single-image VLM) — replaces Florence `<DETAILED_CAPTION>` at episode tier; anti-hallucination prompt + ground-truth `[t=Xs | frame i/N]` temporal tag per caption
- [x] Event fusion model: **Qwen2.5-VL-3B-Instruct** (multimodal — episode texts + up to 10 representative frames sampled evenly across the event)
- [x] Summary gradient: lower tier = more visual; higher tier = more text
- [x] Episode summary: concatenate per-window Moondream grounded captions at episode flush (each prefixed with its `[t=Xs | frame i/N]` tag)
- [x] Event summary: Qwen2.5-VL sees episode texts (treated as noisy hints) + frames (treated as authoritative) from self-centrality-pooling winners
- [x] Episode embedding: time-aware self-centrality pooling (replaces mean pooling)
- [x] Event embedding: L2-normalised centroid of member episode embeddings (unchanged)
- [x] Retrieval order: stage 0 always searches recent by cosine sim; stage A events; stage B episodes; stage C archive grounding
- [x] Persistence: peewee/SQLite (`MemoryStore` in `src/memory_db.py`) — windows/episodes/events + join tables, `float32` blob embeddings, JPEG blob frames, WAL journaling; optional (pass `store=None` for pure in-memory)
- [x] Evaluation style decided: qualitative + precision@k on synthetic + baseline comparison
- [x] Research notes written (related_work_notes.md)
- [x] Architecture document written (ARCHITECTURE.md, now incl. §14 persistence and §15 observed failure modes)
- [x] Model decisions recorded (notes/models.md, incl. three-tier captioning gradient + observed failure modes at event tier)

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
- [x] `update()` online per-window call — also writes to `MemoryStore` on tier flip (recent → episodic)
- [x] **`finalize()` tail-drain fix** — drains the recent deque through the same novelty path used during streaming, then force-consolidates residual episodes so the last ~1 capacity-worth of windows don't silently become stranded `tier=recent` rows
- [x] **`flush_pending()` on-demand synchronisation** — closes the currently-forming `_pending_episode` into `self.episodic` so the tail episode is queryable at QA time; does **not** drain `self.episodic` into events (that over-drain used to make stage-B fine search structurally empty). The recent-episodes bypass in `_fine_search` now covers tail-content visibility without requiring coarse-tier escalation.
- [x] get_searchable_episodes() including in-progress episode
- [x] get_grounding_windows() from window archive (available helper; current retriever grounding uses recent queue)
- [x] text_encode_fn hook for summary embeddings
- [x] Optional `store: MemoryStore` — on-the-fly persistence of windows/episodes/events (idempotent upserts + wipe-and-rewrite membership)
- [x] stats() diagnostics

### Summary builder (summary_builder.py)
- [x] Template mode (fast, no model)
- [x] Florence-2 window captioning mode (`use_model=True`)
  - [x] `caption_frame()` — `<CAPTION>` on one representative frame
  - [x] `build_window_caption()` dispatches per ingested window
- [x] **Moondream2 episode captioning mode** (`use_moondream=True`)
  - [x] `caption_frame_moondream()` — three-level API fallback (`query` → `caption` → `encode_image + answer_question`), each guarded by try/except with one-shot failure warning
  - [x] Anti-hallucination prompt: "single still frame from a continuous video", forbid naming teams/players/scores/brands/unreadable text, forbid speculating about what happened before/after
  - [x] `caption_episode()` prefixes each member caption with ground-truth `[t={start_time:.1f}s | frame i/N]` tag before concatenation
  - [x] Florence `<DETAILED_CAPTION>` kept as automatic fallback if Moondream fails
- [x] **Qwen2.5-VL event fusion mode** (`use_vlm=True`)
  - [x] Loader auto-detects family from HF id (`qwen2-vl` vs `qwen2.5-vl`) and picks the matching HF class
  - [x] bf16 dtype forced via `_vlm_dtype_for_device` (fp16 overflows vision tower → `!!!!…`); `pixel_values` / `pixel_values_videos` cast to `model.dtype` after `.to(device)`
  - [x] Processor built with `min_pixels=64·28²`, `max_pixels=320·28²`; CUDA forces `attn_implementation="sdpa"` (avoids 30 GB OOM from eager N² attention on default ~16k-patch images)
  - [x] `_fuse_with_vlm()`: episode texts + up to **10 representative frames** sampled evenly across the event → one detailed event summary
  - [x] Generation config: greedy, `max_new_tokens=640`, `repetition_penalty=1.15`, `no_repeat_ngram_size=6` (kills the "the Nth scene shows…" template loop)
  - [x] Adaptive word target 40–80 / 80–140 / 120–220 based on scene count (prevents padding when content is thin)
- [x] `__call__(entries, episode_frames=None)` dispatch
- [x] Resilient: any model failure falls back to the next tier / template without crashing the pipeline

### Retriever (retriever.py)
- [x] Stage 0: always similarity-search recent windows → top_k hits by cosine sim (guaranteed fresh context)
- [x] Stage A: coarse routing over long-term EventEntries (blended visual + summary similarity)
- [x] Stage B: fine search over episodic entries within candidate time ranges (blended visual + summary + recency)
- [x] **Recent-episodes bypass in stage B** — union the coarse-gated set with `episodic[-recent_episodes:]` (default 5, dedup by `entry_id`) so tail episodes not yet consolidated into any event stay searchable; mirrors the stage-0 recent-windows pass
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

### Persistence (memory_db.py)
- [x] **`MemoryStore`** — peewee/SQLite facade instantiated by `scripts/main.py`
- [x] Tables: `Window`, `Episode`, `EpisodeWindow`, `Event`, `EventEpisode`, `EventRepWindow`
- [x] Embeddings stored as raw `float32` blobs with `embedding_dim` column (recovered via `np.frombuffer` on read)
- [x] Representative frames stored as JPEG blobs (quality 90) when present
- [x] Idempotency: windows use `INSERT ... ON CONFLICT` preserving later tier/summary updates; episodes/events use `REPLACE` + wipe-and-rewrite membership
- [x] SQLite pragmas: WAL, `foreign_keys=1`, `synchronous=normal`, `cache_size=-64*1024`
- [x] `ON DELETE CASCADE` on parent side of all join tables
- [x] Context manager interface (`close()` flushes WAL)
- [x] `counts()` diagnostic for post-run inspection

### Pipeline runner (scripts/main.py)
- [x] End-to-end streaming loop: StreamReader → PerceptionEncoder → SummaryBuilder → HierarchicalMemoryWriter (+ `MemoryStore`)
- [x] `SummaryBuilder(use_model=True, use_vlm=True, use_moondream=True)` — all three captioners active
- [x] Per-window Florence-2 captioning with live logging
- [x] Box-formatted episode and event output as they are flushed, including wrapped multiline summaries
- [x] `MemoryStore` wired in; `outputs/memory.db` inspected via `sqlite3` for quality assessment
- [x] Final stats footer (windows, episodes, events, elapsed time, DB counts)
- [x] **Retrieval testing harness**: loads `qas.json` (HH:MM:SS timestamps) before the streaming loop; `_process_due_qas` fires `HierarchicalRetriever.retrieve()` whenever the stream clock passes a QA timestamp; before firing, calls `memory.flush_pending()` so boundary events become visible to coarse routing
- [x] Rendered coarse / fine / grounding hits appended to `outputs/retrievals.md` as self-contained markdown blocks (QA, options, ground truth, formatted evidence) — self-contained for offline scoring
- [x] No LLM reasoner is wired in; the pipeline stops at formatted evidence

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
- [ ] Show real concatenated Moondream episode summaries (with `[t=Xs | frame i/N]` tags)
- [ ] Show a Qwen2.5-VL event summary paragraph next to its source episode texts and sampled frames
- [ ] Show real X-CLIP similarity scores on real queries
- [ ] Example queries on real video content with retrieved thumbnails displayed
- [ ] Side-by-side: baseline retrieval vs hierarchical retrieval on same query
- [ ] Note obvious failure cases from real video (including the documented residual event-tier issues)
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

### Retrieval / infra
- [ ] FAISS instead of brute-force cosine search
- [ ] Scene-change detection signal to trigger episode flush earlier
- [ ] Tiny interactive query demo cell (ipywidgets text input)

### Event-tier quality (addresses residual failures documented in ARCHITECTURE §15)
- [ ] **Scoreboard pre-extraction**: regex `\d+-\d+` over Moondream episode captions → inject a timestamped list as a hard-constrained scoreboard trace in the Qwen fusion prompt (kills "scoring twice → 0-0" class errors that survive the 2B → 3B swap)
- [ ] **Previous-event carry-over**: thread the last event's summary into the next fusion call as stateful context (reconciles the running scoreline across adjacent events)
- [ ] **Post-filter pass**: strip flagged entity classes (brand names, stadium names, seasons, league IDs) from the VLM output unless they appear verbatim in the episode captions
- [ ] Summary confidence / quality flag on Florence-2 output (window tier)
- [ ] Try Qwen2.5-VL-7B as a drop-in for the event tier (auto-detect already handles it)

---

## CONCRETE DEFAULTS (locked)

| Parameter | Value |
|-----------|-------|
| Retrieval encoder | X-CLIP ViT-B/32 (512-dim, up to 8-frame clip) |
| Window captioner | Florence-2-base (`<CAPTION>`) |
| Episode captioner | Moondream2 (single-image, anti-hallucination prompt, `[t=Xs | frame i/N]` tag prepended) |
| Event fusion | Qwen2.5-VL-3B-Instruct |
| Persistence | peewee/SQLite (`outputs/memory.db`) — optional (`store=None` disables) |
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
| Recent-episodes bypass | 5 trailing episodes always added to fine candidate pool |
| Neighbour radius | 1 |
| Scoring formula | `score = (α · visual_sim + β · summary_sim) · exp(-dt / τ)` — multiplicative temporal prior |
| Scoring weights | α=0.70 visual, β=0.30 summary (α + β = 1.00); `τ = 0.50 · stream_span` |
| Baseline window | 10 windows |
| Novelty threshold | 0.05 (real video; 0.25 default) |
| **VLM dtype** | bf16 (fp16 overflows Qwen vision tower → `!!!!…`) |
| **VLM attention** | SDPA on CUDA (eager blows up VRAM on default 12.8 MP images) |
| **VLM image-pixel cap** | `min_pixels=64·28²`, `max_pixels=320·28²` (~64–320 patches/frame) |
| **VLM frames per event** | up to 10, sampled evenly across event |
| **VLM generation** | greedy, `max_new_tokens=640`, `repetition_penalty=1.15`, `no_repeat_ngram_size=6` |
| **VLM word-count target** | 40–80 / 80–140 / 120–220 (adaptive by scene count) |
| **torch / transformers pins** | `torch>=2.5,<2.9`, `transformers>=4.45,<5.0` (Moondream `enable_gqa` requires torch ≥ 2.5) |
