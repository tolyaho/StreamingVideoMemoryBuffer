# Memory Buffer for Streaming Video-LLM — Project Plan

Legend: [x] done · [ ] to do · [~] partial / needs update

---

## Status on 2026-04-22 (late — post §6 Reasoner + §7 Baseline cells)

**Implementation (src/)**: complete. All nine modules in `src/__init__.py`
are wired up, tested (4 test files under `tests/`), and exercised
end-to-end by `scripts/main.py`. The 3-min soccer clip (`sample_1`) has
been streamed through the full stack with Florence + Moondream +
Qwen2.5-VL all active, producing 63 windows · 16 episodes · 5 events in
82 s (see cell 10 / first streaming demo in the notebook). Persistence
to `memory.db` is working and the offline QA harness
(`outputs/retrievals.md`, `EVALUATION_REPORT.md`) reports 5/5 right-time
retrieval on sample_36.

**Notebook (solution.ipynb)**: 9 sections (§0 Setup → §8 Actually Long
Video, the last one is a stub heading for the remaining long-video run).
Assignment sections 1 (memory design) and 2 (storage strategy) are
covered with prose + a working streaming demo on `sample_1` (63
windows · 16 episodes · 5 events in 82 s). Assignment section 3
(query-based retrieval) is demonstrated live: §4 Retrieval writeup
covers query encoding, three-stage coarse-to-fine, multiplicative decay
with unified span, tier-split visual scoring; a retrieval helpers cell
wires `HierarchicalRetriever` + `ReasonerInputFormatter.format_text` +
`fire_due` (which calls `mem.flush_pending()` before each
`retriever.retrieve(...)`), and the downstream cell fires all 5 QAs from
`sample_36/qas.json` with full coarse/episodic/grounding evidence
printed per QA. **Assignment section 4 (LLM input integration) is now
also demonstrated live**: §5 writeup cells cover the full-VLM option
space (projector / raw-frames-into-VLM / Q-Former) and explicitly state
why this notebook ships the text-context path only, then the final
code cell calls `format_for_llm(result, query_embedding=q_emb)` on the
first sample_36 QA, prints the visual-context slot per hit, and prints
the exact string (system + evidence + user) that would be fed to the
LLM — the `[evidence]` block is the real `llm_input['text_context']`,
interpolated directly (not a placeholder).

**Shipped since last status**:
- **§6 LLM Reasoner** — `src/llm_reasoner.py` (Qwen2.5-3B-Instruct,
  text-only, bf16, greedy, MCQ letter + time-range citation prompt).
  Live end-to-end cell answers all 5 sample_36 QAs during streaming
  using `mem.flush_pending()` + `query_time=stream_time`. Scored
  **3/5** on sample_36.
- **§7 Baseline comparison** — `RecentWindowBaseline` now imported in
  the notebook. Same end-to-end loop as §6 with flat recent-frames
  retrieval packed into a `RetrievalResult` so the reasoner sees the
  same prompt shape. Scored **2/5** on sample_36. Description in
  `notes/answer.md`; conclusion paragraph in `notes/conclusion.md`.
- **`notes/answer.md`** rewritten with §6 + §7 prose in the first-person
  voice matching §5.

**Remaining work — one long-video run (§8):**
1. **§8 "Actually Long Video"** — the assignment suggests "e.g. 1h+".
   Longest tested so far: `sample_36` at 17 min. Architecture is
   bounded by construction (recent deque=20, episodic list=50, events
   slow-growing, no O(N²) retrieval), but scaling has never been
   empirically verified on hour-long footage. Expected wall time:
   ~3–4 h on CUDA with all three VLMs active (Florence ~1200 calls +
   Moondream ~100–200 + Qwen ~10–30). Biggest unknown: whether the
   documented intro-event dominance failure (ARCHITECTURE §15) scales
   linearly or exponentially with stream length. Dataset options:
   **LVBench** (1h avg, has memory-requiring QAs), **HourVideo**
   (egocentric 20–120 min, 12K QAs), **MLVU** long subset, or a
   single hand-curated MIT OCW lecture with auto-generated QAs
   verified manually. See `### Long-video run — preflight checklist`
   below.

Stretch (not required for the core deliverable): tier-size-over-time
plot, inline thumbnails of retrieved frames, a keyword-in-caption P@k
scorer against `qas.json`.

### Long-video run — preflight checklist

Keep this local to `solution.ipynb` — do not touch `scripts/main.py`.

- [ ] Pick one 1h+ clip (StreamingBench has nothing that long; pull from
      YouTube / local file and drop into `data/`)
- [ ] Before the stream loop: log `mem.stats()` + wall-clock every N
      windows (e.g. every 50) so tier growth is observable during the
      run, not just at the end
- [ ] Disable Qwen2.5-VL event fusion for the first pass (`use_vlm=False`)
      so you get event counts and retrieval quality without paying
      the ~3 h VLM cost
- [ ] Re-enable Qwen only if event count stays reasonable (< ~30 for
      1 h); cap `max_new_tokens` lower if needed
- [ ] Expected failure to watch for: intro-event dominance — if the
      first event's embedding wins coarse routing for queries about
      content 40+ min later, the multiplicative-decay τ may need a
      harder cap (currently `0.50 · stream_span` → 30 min τ on a 1 h
      stream, which is very permissive)
- [ ] Expected second failure: event tier grows without meaningful
      consolidation because `event_max_gap=15s` is tight → episodes
      flush into many short events. Consider relaxing to 30–60 s for
      long-form content
- [ ] Capture peak RSS — the window archive (`WindowEntry.frame`) is
      not bounded; 3600 windows × ~200 KB JPEG-equivalent ≈ 700 MB in
      memory if frames are not offloaded to disk. `MemoryStore` (SQLite)
      exists but is opt-in; consider enabling for the long run
- [ ] Run 3–5 text queries at the end over the populated memory and
      log whether coarse routing returns events from the right
      *section* of the hour, not just the dominant intro event

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
- [x] **Unified decay span** — single `stream_span` across `recent ∪ episodes ∪ events` passed into all three scoring passes, so straddling entries get `decay=1.0` at every tier and far-end entries compare like-for-like
- [x] **Tier-split visual scoring** — events: `max_{w ∈ reps(e)} cos(q_vis, w_vis)` (peak over `representative_window_ids`, undoes centroid-of-centroids blur); episodes: stored self-centrality-pooled centroid (pooling already peaks toward the typical frame); recent windows: own embedding directly
- [x] `_build_rep_index(memory, events)` — one archive lookup per `retrieve()` call, amortised across passes; `_vectors_for()` falls back to `[entry.visual_embedding]` when reps are absent

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
- [x] Notebook demo locked on `sample_1` (≈3 min soccer clip) — long enough to produce events, short enough to rerun cheaply
- [ ] Prepare 3–5 example queries matched to `sample_1` content for the retrieval cell (e.g. "when does someone score?", "show the crowd", "what is the scoreboard showing at the end?")

---

## NOTEBOOK (solution.ipynb)

Actual state snapshot (read on 2026-04-22 from `notebooks/solution.ipynb`,
**39 cells**. Sample_1 soccer streaming demo: 63 windows · 16 episodes ·
5 events · 82 s in cell 27. Sample_36 cooking retrieval demo: all 5 QAs
fired via `HierarchicalRetriever` + `ReasonerInputFormatter.format_text`
in cell 36 — executed output present, ~61 KB, covers coarse / episodic /
grounding hits per QA).

### Structure — sections needed
- [x] Title + architecture overview (approach rationale + architecture writeup) — cells 0–7
- [x] Architecture diagram — cells 7 (`architecture.svg`) + 15 (`MemoryBuffer.svg`)
- [x] Setup cell (imports, path, config flags) — cells 8–10 (pip install, sys.path fix)
- [x] Stream ingestion section with rationale — cells 11–12 (prose only; `StreamReader` is imported but never called standalone)
- [x] Perception encoder section with rationale — cells 13–14 (prose only; `PerceptionEncoder` imported but never shown encoding on its own)
- [x] Data structures section with rationale — folded into the Memory Buffer section (cell 15)
- [x] Memory buffer design section with rationale — cells 15–19 (what is stored, size control, novelty gate, self-centrality pooling with full math, summary gradient, motivation)
- [x] Streaming update loop on real video — cell 27 (end-to-end Florence + Moondream + Qwen2.5-VL, 63 windows, 16 episodes, 5 events in 82 s)
- [x] Real concatenated Moondream episode summaries visible in cell 27 output
- [x] Real Qwen2.5-VL event summaries visible in cell 27 output (5 event boxes rendered)
- [x] **Retrieval section with rationale + example queries** (cell 29 §4 writeup incl. query encoding, three-stage coarse-to-fine, multiplicative decay, unified span, tier-split visual scoring; cells 31–36 embed `sample_36` video, wire the retriever, and fire all 5 QAs at their QA timestamps via `fire_due(...)`)
- [x] Retrieval evidence printed inline per QA — coarse events + episodic hits + grounded windows with sim scores, via `ReasonerInputFormatter.format_text(result)` wrapped by `wrap_formatted` for readable line-width
- [x] **Baseline section with rationale + retrieve() demo** — §7 imports `RecentWindowBaseline` and runs the same 5 sample_36 QAs end-to-end through the same reasoner (MCQ accuracy 2/5 vs hierarchical's 3/5)
- [x] **Baseline vs hierarchical side-by-side comparison on the same query** — §7 output pairs naturally with §6 output; brief conclusion in `notes/conclusion.md`
- [x] **LLM input formatter section with rationale + `format_for_llm` demo** — §5 writeup covers the full-VLM option space (projector / raw-frames-into-VLM / Q-Former) and states why only the text-context path is shipped; the final code cell calls `format_for_llm(result, query_embedding=q_emb)` on the first sample_36 QA, prints the visual-context slot per hit, and prints the exact prompt (system + `[evidence] = llm_input['text_context']` + user) fed to the LLM
- [ ] Retrieval timeline / thumbnail visualisation per query (stretch)
- [ ] Memory tier size plot over time (stretch)
- [ ] Precision@k on the loaded video's QAs (stretch — `qas.json` is on disk; `scripts/main.py` already has the harness)
- [ ] Design trade-offs / limitations discussion (lightweight markdown cell pointing at ARCHITECTURE §15 residuals is enough)
- [ ] Final conclusion section answering the assignment questions explicitly

### Still needed in notebook
- [x] Real-video pipeline run (sample_1 soccer, cell 27) with Florence + Moondream + Qwen2.5-VL all active
- [x] Florence window captions visible inline
- [x] Moondream episode summaries visible inline
- [x] Qwen2.5-VL event summary paragraphs visible inline
- [x] Sample_36 cooking video embedded (cell 31, HTML5 `<video>` with controls) so graders can watch alongside the retrieval output
- [x] `HierarchicalRetriever.retrieve(...)` wired in (cell 36) and fired at every QA timestamp during the streaming loop (`fire_due(...)` from cell 33 calls `mem.flush_pending()` before each retrieval so boundary episodes are visible); all 5 QAs from `sample_36/qas.json` printed with coarse / episodic / grounding evidence via `ReasonerInputFormatter.format_text`
- [ ] Display actual sampled frames from the real video with timestamps (matplotlib grid — easy addition)
- [x] Run the same queries against `RecentWindowBaseline.retrieve(...)` and print both side-by-side — §7 cell (MCQ accuracy 2/5 baseline vs 3/5 hierarchical on sample_36)
- [x] Show `ReasonerInputFormatter.format_for_llm(result, query_embedding=...)` output for one query (dict with `visual_context`/`text_context`) to demonstrate the LLM integration — shipped in §5
- [ ] Note obvious failure cases (intro-event domination, named-entity hallucinations, colour captions) referencing ARCHITECTURE §15 + EVALUATION_REPORT
- [ ] Run on a 1h+ video (§8) — pick LVBench / HourVideo / MLVU long subset or a hand-curated OCW lecture; see preflight checklist in the Status section

### Assignment questions the notebook must answer
- [x] What is stored in memory? (§3 prose)
- [x] How is memory size controlled? (§3 prose)
- [x] What stays and what gets removed? (§3 prose)
- [x] How are summaries generated (window → episode → event)? (§3 prose)
- [x] How is retrieval done (coarse-to-fine)? — §4 markdown + live demo on sample_36 (5 QAs)
- [x] How are summaries used at query time? — §4 explains β·summary_sim term; live output shows summary text alongside each coarse / episodic hit with sim score
- [x] How is retrieved memory formatted for LLM input? — §5 writeup + `format_for_llm` demo on sample_36 QA 0, prints both the visual-context slot breakdown and the exact prompt (system / [evidence] = real `text_context` / user) fed to the LLM
- [x] When does hierarchical memory help vs recent-window baseline? — §7 side-by-side on sample_36 (3/5 hierarchical vs 2/5 baseline); flipped QA is the held-object question where the referenced item had rolled out of the recent deque but survived as an episode summary
- [x] What are the main limitations? — documented in ARCHITECTURE §15 (captioner ceiling, intro-event dominance, no cross-event state) and the `notes/conclusion.md` paragraph; longer-video scaling risks enumerated in ARCHITECTURE §17 and the preflight checklist above

---

## EVALUATION

- [x] Evaluation style decided: qualitative + precision@k + baseline comparison
- [x] Qualitative retrieval inspection done **offline** via `scripts/main.py` + `outputs/retrievals.md` + `notes/EVALUATION_REPORT.md` (5/5 right-time retrieval on cooking sample_36, 3/5 verbatim caption match) — but this exists only as markdown, not as notebook output
- [ ] Move at least a trimmed version of the eval into `solution.ipynb` so the graders see it without opening the markdown notes
- [ ] Precision@k on `qas.json` for the demo video (keyword-in-caption scorer — ~20 lines)
- [ ] Baseline vs hierarchical comparison table for the same 3–5 queries (earliest hit time, hit count)
- [ ] Memory tier size plot over time (`len(recent)` / `len(episodic)` / `len(long_term)` sampled every N windows during cell 27's loop)
- [ ] Note where recent-window is already enough (a "right now" query)
- [ ] Note where hierarchical memory clearly helps (a query about content that has already rolled out of the recent queue)
- [ ] Note where Florence-2 captions improve reranking vs visual-only (ablation is optional, but a one-sentence statement is mandatory for the trade-offs cell)

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
| Decay span | unified across `recent ∪ episodes ∪ events` (one ruler for all tiers) |
| Event visual_sim | `max` over `representative_window_ids` embeddings (peak-over-reps, undoes centroid-of-centroids blur) |
| Episode visual_sim | pooled centroid (self-centrality pooling already peaks toward typical frame) |
| Window visual_sim | own embedding (no pooling) |
| Baseline window | 10 windows |
| Novelty threshold | 0.05 (real video; 0.25 default) |
| **VLM dtype** | bf16 (fp16 overflows Qwen vision tower → `!!!!…`) |
| **VLM attention** | SDPA on CUDA (eager blows up VRAM on default 12.8 MP images) |
| **VLM image-pixel cap** | `min_pixels=64·28²`, `max_pixels=320·28²` (~64–320 patches/frame) |
| **VLM frames per event** | up to 10, sampled evenly across event |
| **VLM generation** | greedy, `max_new_tokens=640`, `repetition_penalty=1.15`, `no_repeat_ngram_size=6` |
| **VLM word-count target** | 40–80 / 80–140 / 120–220 (adaptive by scene count) |
| **torch / transformers pins** | `torch>=2.5,<2.9`, `transformers>=4.45,<5.0` (Moondream `enable_gqa` requires torch ≥ 2.5) |
