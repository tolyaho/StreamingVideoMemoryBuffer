# ARCHITECTURE.md

## Project goal

Build a lightweight streaming Video-LLM memory system that can watch a video stream over time, store useful past visual context under a fixed memory budget, retrieve the most relevant past evidence for a text query, and format that retrieved evidence for downstream reasoning. The design is intentionally inspired by recent streaming-video work, but simplified enough to be implemented cleanly in a notebook plus a few helper Python modules. In particular, the architecture follows a **perception → memory → reasoning** split, uses **hierarchical memory compression** for storage, and uses **coarse-to-fine retrieval** for reading, while always comparing against a strong **recent-window baseline**.

## Design principles

The system is built around five principles. First, **recent visual context is valuable and should be stored densely**, because a simple recent-window baseline already performs surprisingly well on current streaming benchmarks. Second, **older context should be stored more selectively and more compactly** to keep memory bounded. Third, **query answering should not read the entire stored history**; instead, it should retrieve a small relevant subset of memory. Fourth, **text summaries are auxiliary**, useful for routing and readability, but the primary memory representation remains visual. Fifth, **the lower the memory tier, the more the representation relies on raw vision; the higher the compression tier, the more it relies on language** — recent windows are pure visual embeddings; episodic entries add **short** per-window Florence captions for routing plus a **longer** episode string built at flush from **Moondream2** grounded captions on each member window’s representative frame; long-term events condense episode text together with a few real frames via a **Qwen2-VL** multi-image fusion step. These principles are directly motivated by SimpleStream’s recency result, FluxMem’s hierarchical compression, VideoTree’s coarse-to-fine retrieval, and IXC2.5-OmniLive’s perception–memory–reasoning split.

## Non-goals

This project is **not** a full production VLM system. It does not train a new model, does not implement KV-cache memory, does not reproduce VideoTree’s expensive LLM-in-the-loop search, and does not attempt end-to-end multimodal generation with projected visual tokens inside a large reasoning model. Instead, it focuses on the parts most aligned with the assignment: memory storage, retrieval, and interpretable formatting of retrieved evidence. FluxMem itself is training-free, and SimpleStream shows that complicated streaming systems should justify their complexity against a simple baseline, so a lightweight architecture is both practical and methodologically sound here.

## High-level architecture

The system has six components:

1. **Stream Ingestion**
2. **Perception Encoder**
3. **Hierarchical Memory Writer**
4. **Summary Builder**
5. **Hierarchical Retriever**
6. **Reasoner Input Formatter**

The high-level dataflow is:

`video stream -> sampled windows -> visual embeddings -> recent memory -> episodic memory -> long-term summary memory -> query-conditioned retrieval -> formatted evidence for reasoning`

This follows the same high-level decomposition used by IXC2.5-OmniLive, where perception, memory, and reasoning are separated instead of forcing one model to simultaneously perceive and answer over an ever-growing stream.

---

## 1. Stream Ingestion

### Purpose

The ingestion module converts a continuous video into a sequence of small time windows suitable for online memory updates.

### Input

- video file or stream
- sampling rate configuration

### Output

A sequence of short windows, each with:
- `window_id`
- `start_time`
- `end_time`
- sampled frames

### Default implementation

Start with **1 fps sampling** and group sampled frames into **local windows of 3–5 seconds**. The point is to avoid isolated frame reasoning while keeping the pipeline light enough for a notebook. This is consistent with the broader observation in streaming-video work that using every raw frame directly is too expensive, and that systems typically either sample, compress, or write memory online.

### Why windows instead of single frames

Single frames are too weak as semantic units: they often miss temporal continuity, action boundaries, and causal context. A short local window is a much better primitive for later memory building and summarization. VideoTree’s motivation is closely related: long-video reasoning benefits from hierarchical representations that preserve more temporal structure than flat, isolated captions.

---

## 2. Perception Encoder

### Purpose

Convert each local window into a compact visual representation suitable for storage and retrieval, while preserving short-term temporal dynamics.

### Main representation

Each window receives a **clip embedding** produced by a video-language model. This is the primary memory representation stored in all three memory tiers.

### Retrieval encoder — X-CLIP base-patch32

The retrieval encoder is **X-CLIP** (`microsoft/xclip-base-patch32`), a pretrained video-language model that produces a single 512-dim embedding from a short clip of frames. Unlike a static image encoder, X-CLIP integrates temporal information across multiple frames, making it sensitive to motion and short-term event dynamics such as actions, transitions, and scene changes.

Each window is encoded by uniformly sampling **up to 8 frames** from its frame list (the stream reader may yield fewer distinct frames per window; boundary frames are repeated as needed) and passing them together through X-CLIP’s video encoder. The result is one L2-normalised clip embedding per window. Text queries are encoded with X-CLIP’s matching text encoder, keeping both query and memory in the same joint retrieval space.

### Why X-CLIP instead of a static image encoder

A single still frame captures appearance but discards motion. Events such as scoring a goal, entering or leaving a scene, a fall, or the start of a celebration are poorly represented by any one frame because the discriminative signal lies in what changed across the window. X-CLIP was trained to align short video clips with text descriptions, so its video embeddings are inherently sensitive to these temporal patterns. This improves retrieval quality for action-based queries without requiring a generative VLM, a trained projector, or any fine-tuning.

### Summary encoders — tiered

Captioning uses a **three-model gradient**, matched to call volume and hallucination risk at each tier. No single captioner runs at every level.

- **Window tier — Florence-2-base** (`microsoft/Florence-2-base`, ~0.23B). At ingest, runs the short `<CAPTION>` task on each window’s representative (middle) frame and stores the result as `WindowEntry.summary_text`. Call volume is high (one per sampled window), so the cheapest model wins. The short captions back the retriever’s summary-similarity term and lightweight logging.
- **Episode tier — Moondream2** (`vikhyatk/moondream2`, ~1.87B single-image VLM). When an episode is flushed, Moondream2 runs on each member window’s stored representative frame with an explicit anti-hallucination prompt. The prompt tells the model the input is a **single still frame extracted from a continuous video** — so it should describe apparent motion or pose evident from the pixels (body position, blur, trajectory) while **never** naming teams, players, leagues, scores, brands, or unreadable on-screen text, and **never** speculating about what happened before or after the frame. Each output is stamped with a ground-truth temporal tag `[t={start_time:.1f}s | frame i/N]` (taken from `WindowEntry.start_time`, not inferred by the model) and the tagged captions are **concatenated** in temporal order into `EpisodeEntry.summary_text`. Moondream is a single-image model, so there is no temporal fusion here — that is the job of the event tier. All three Moondream API branches (`query` → `caption` → `encode_image + answer_question`) are wrapped in try/except with a one-shot warning; if every API fails (e.g. PyTorch < 2.5 missing `enable_gqa`) the pipeline cleanly falls back to Florence rather than crashing.
- **Event tier — Qwen2.5-VL-3B-Instruct** (`Qwen/Qwen2.5-VL-3B-Instruct`, ~3B; `Qwen2-VL-*` and larger `Qwen2.5-VL-*` variants also supported via the same loader). At consolidation, receives the concatenated episode summary strings plus up to **10 representative frames sampled evenly across the event** (drawn from each episode’s top-2 windows by cosine similarity to the episode centroid). Returns one detailed event summary in which the images are treated as **authoritative** and the upstream captions as **noisy hints explicitly known to hallucinate named entities**. Generation uses greedy decoding with `repetition_penalty=1.15` and `no_repeat_ngram_size=6` to prevent the "the Nth scene shows…" template loop that small instruction-tuned models fall into on repetitive multi-image inputs.

The architecture deliberately separates retrieval (X-CLIP clip embeddings) from summarisation (Florence-2 / Moondream2 / Qwen2.5-VL). Retrieval remains visual-first and temporally aware; text summaries are an auxiliary semantic index that grows more descriptive and more grounded at higher tiers.

### Summary pipeline

- **Per window (ingest)**: Florence-2 `<CAPTION>` on the representative frame → `WindowEntry.summary_text`; the RGB representative frame is also kept on the entry for later multimodal use.
- **Per episode (flush)**: for each member window in order, **Moondream2** grounded caption on that window’s representative frame → concatenate into `EpisodeEntry.summary_text` (falls back to the short Florence caption or a time template if a frame is missing).
- **Per event (consolidation)**: when `use_vlm=True`, **Qwen2.5-VL-3B-Instruct** receives the episode summary strings plus up to **10 representative frames** sampled evenly across the event (windows closest to each episode’s visual centroid, in time order) and returns one detailed event summary with adaptive word-count target (40–80 / 80–140 / 120–220 based on scene count); otherwise a template stitch over episode texts is used.

### Runtime notes for the event VLM

Qwen2.5-VL is loaded in **bf16** on CUDA/MPS/CPU via `_vlm_dtype_for_device` — fp16 overflows in its vision tower, which causes the LM to emit repeated token 0 (rendered as `!!!!…`). The processor is built with `min_pixels=64·28²` and `max_pixels=320·28²` so per-frame visual tokens stay in the 64–320 patch range (the default max of ~12.8 MP produces up to ~16k patches and blows up eager attention). CUDA additionally forces `attn_implementation="sdpa"` for the same reason. Generation uses greedy decoding (`do_sample=False`, `num_beams=1`) with `max_new_tokens=640`, `repetition_penalty=1.15`, and `no_repeat_ngram_size=6` — the n-gram constraint is what actually kills the "the Nth scene shows…" template loop that 2B-class instruction VLMs fall into on repetitive multi-image inputs (token-level repetition penalty alone is not enough). The loader auto-detects the model family from the HF id (`qwen2-vl` → `Qwen2VLForConditionalGeneration`, `qwen2.5-vl` → `Qwen2_5_VLForConditionalGeneration`), so swapping between Qwen2-VL-2B/7B and Qwen2.5-VL-3B/7B is a single-string change.

### Why not store raw pixels

The system stores encoder-produced embeddings as the **primary** evidence, not full video. One **representative RGB frame** per window is retained only where Florence, Moondream, or the event VLM needs pixels; that keeps footprint small compared to storing every sampled frame.

---

## 3. Hierarchical Memory Writer

### Purpose

Maintain a bounded memory over time by storing recent content densely and older content more selectively.

### Core idea

The writer has three tiers:

- **Recent memory**
- **Episodic memory**
- **Long-term summary memory**

This is the part of the architecture most strongly inspired by FluxMem and StreamBridge: memory is hierarchical, recent context is preserved more faithfully, and older context is compressed more aggressively.

### 3.1 Recent memory

#### Role

Store the most recent local windows densely for current-scene grounding.

#### Stored fields

- `entry_id`
- `start_time`
- `end_time`
- `visual_embedding`
- `tier = "recent"`

#### Policy

Recent memory is a fixed-size queue. New windows are appended. When it overflows, the oldest recent entry is evaluated for promotion or discard.

#### Why this tier exists

SimpleStream shows that recent context is a very strong baseline, so the architecture should not aggressively compress away the newest information.

### 3.2 Episodic memory

#### Role

Store only temporally informative and semantically meaningful past windows, compressed into coherent action spans.

#### Stored fields

- `entry_id`, `start_time`, `end_time`
- `visual_embedding` — self-centrality pooled embedding of member windows (see below)
- `member_window_ids`
- `representative_window_ids` — top-weight windows from pooling, used for summarisation
- `summary_text`, optional `summary_embedding`

#### Promotion rule

When recent memory overflows:
1. take the oldest recent entry
2. compare it with neighboring recent windows
3. compute a **novelty score**
4. if novelty is low, discard it
5. if novelty is high, promote it to episodic memory

This is the notebook-friendly analogue of FluxMem’s **Temporal Adjacency Selection (TAS)**, which keeps temporally informative tokens and drops redundant ones across adjacent frames.

#### Episode embedding — time-aware self-centrality pooling

Rather than mean pooling, the episode embedding is computed by a query-free, training-free pooling rule that favours windows which are both temporally central and locally consistent with their neighbours.

For each window `i` in the episode with embedding `w_i` and timestamp `t_i`:

```
center_score_i     = -(t_i - t_center)^2 / (2 * sigma_center^2)
consistency_score_i = sum_j  exp(-|t_i - t_j| / sigma_time) * cos(w_i, w_j)
score_i            = lambda_center * center_score_i + mu_consistency * consistency_score_i
alpha_i            = softmax(score_i)
episode_embedding  = L2_norm( sum_i  alpha_i * w_i )
```

- `center_score` gives higher weight to temporally central windows, downweighting edge transitions.
- `consistency_score` gives higher weight to windows that are similar to temporally nearby neighbours, downweighting noisy outliers.
- The temporal decay kernel `exp(-|t_i - t_j| / sigma_time)` ensures that only local context contributes to each window’s consistency score.
- Softmax normalization turns scores into a proper probability distribution over windows.

The top-weight windows are also stored as `representative_window_ids` on the episode, providing natural candidates for Florence-2 captioning and VLM event fusion without any additional selection step.

This method is strictly more expressive than mean pooling: it degrades to uniform mean pooling when all scores are equal, but actively upweights the most representative temporal region of the episode otherwise.

#### Why this tier exists

This tier captures medium-scale past context without keeping the entire stream. It preserves “interesting enough to remember” windows and avoids storing every near-duplicate frame.

### 3.3 Long-term summary memory

#### Role

Store compressed higher-level summaries of older content for coarse retrieval.

#### Stored fields

- `entry_id`
- `start_time`
- `end_time`
- `visual_embedding` or centroid embedding
- `summary_text`
- `summary_embedding`
- `representative_ids`
- `tier = "summary"`

#### Creation rule

When episodic memory overflows or enough episodic entries accumulate in a time span:
1. take a batch of older episodic entries
2. merge highly similar ones
3. compute a centroid visual embedding
4. choose 1–3 representative windows
5. generate one concise summary text
6. store the result as a long-term summary entry

This is inspired by the spirit of FluxMem’s **Spatial Domain Consolidation (SDC)** and by the general long-memory compression strategy in IXC2.5-OmniLive and StreamBridge, even though the notebook implementation is much simpler than their internal token-level mechanisms.

---

## 4. Summary Builder

### Purpose

Generate readable semantic notes at multiple temporal scales.

### Summary levels

- **Local note**: attached to promoted windows (Florence-2 `<CAPTION>`)
- **Episode summary**: attached to merged short action spans; built at flush by concatenating **Moondream2** grounded captions over each member representative frame (see **Summary pipeline** above)
- **Event summary**: attached to longer high-level activities; produced by **Qwen2.5-VL** multi-image fusion over episode text + sampled frames

### Important design choice

Summaries are **not** the main memory. They are:
- a semantic index
- a reranking signal
- an interpretability aid
- a readable proxy for notebook output

The primary evidence is still visual memory.

### Why isolated frame captions are avoided

A single frame rarely contains enough context to express a full action reliably. The summary builder therefore works over windows, then over groups of windows, then over groups of episodes. This is consistent with the motivation behind hierarchical long-video methods like VideoTree, which explicitly avoid flat independent descriptions in favor of hierarchical structure.

### Episode building

An **episode** is a short coherent action span built online from consecutive windows.

Windows are merged into the current episode when:
- time gap is small
- embedding similarity stays high
- no strong scene/action break is detected

Each episode stores:
- time range
- centroid visual embedding
- list of member windows
- episode summary text: concatenated **Moondream2** grounded captions over member representative frames (see **Summary pipeline** under Perception Encoder above)

### Event building

An **event** is a higher-level activity built from consecutive episodes.

Episodes are merged into the current event when:
- time gap stays moderate
- summaries remain semantically consistent
- no strong global break is detected

Each event stores:
- time range
- centroid visual embedding
- list of member episodes
- concise event summary

### Why this structure is useful

This gives the system memory at multiple granularities:
- windows for fine evidence
- episodes for local actions
- events for coarse semantic retrieval

It is the simplest implementable version of hierarchical memory informed by FluxMem’s compression and VideoTree’s hierarchy.

---

## 5. Hierarchical Retriever

### Purpose

Retrieve the smallest useful subset of past memory for a query.

### Core idea

Use **RAG-style retrieval over video memory**, not classical document RAG and not full-memory readout.

The retriever is inspired by:
- VideoTree’s **coarse-to-fine** search idea
- FlexMem’s emphasis on **recalling relevant memory fragments**
- the assignment’s explicit requirement for **query-based retrieval**

FluxMem is less suitable here because its read path is closer to concatenating the whole compressed hierarchy than to explicit query-conditioned top-k retrieval.

### Retrieval stages

#### Stage 0: recent window search (always)

Before any hierarchical search, **always** similarity-search the recent window queue.

Input:
- text query embedding

Search target:
- visual embeddings of all current recent windows

Output:
- top `K` recent windows by cosine similarity, always included in the evidence set

This guarantees that the most current visual context is never missed, consistent with SimpleStream's observation that recency is a very strong signal.

#### Stage A: coarse routing

Search over **long-term event embeddings**.

Input:
- text query embedding

Search target:
- blended (visual + summary) embeddings of EventEntries

Output:
- top `M` events → their time ranges as candidates for stage B

#### Stage B: fine retrieval

Search over **episode embeddings** inside the candidate time ranges, **unioned with the trailing `recent_episodes` entries of the episodic tier** (the in-progress episode included). The tail is always searchable regardless of whether it overlaps any top-M event range — this mirrors the recent-windows pass in stage 0 and closes the blind spot where tail episodes not yet consolidated into any event sit after the last event's `end_time` and never intersect a coarse-gate window.

Input:
- same query embedding
- episodic entries temporally overlapping stage A ranges ∪ last `recent_episodes` episodic entries (dedup by `entry_id`)

Output:
- top `K` episodic hits by blended score

#### Stage C: grounding

Return member windows from the top-K episodic hits via the window archive.

Purpose:
- recover fine-grained visual evidence at the window level
- provide the actual frames / embeddings for formatter and downstream reasoning

These archive windows are merged with the stage-0 recent hits and deduplicated into one evidence set.

### Scoring

Use **visual similarity as the primary signal**, with a **multiplicative temporal prior** on top.

Combined score:

`score = (alpha * visual_similarity + beta * summary_similarity) * exp(-dt / tau)`

with:
- `alpha` = 0.70 (dominant semantic signal)
- `beta`  = 0.30 (summary text, auxiliary); `alpha + beta = 1`
- `tau`   = `tau_fraction * stream_span`, default `tau_fraction = 0.50` (half-span-away → multiplier ~0.37; full-span-away → ~0.14)
- `dt`    = temporal distance between `query_time` and the entry's time range; 0 if the entry straddles `query_time`
- `query_time = None` collapses the decay to 1.0 (pure semantic ranking)
- `stream_span` = **unified** span across `recent ∪ episodes ∪ events` — one ruler for all tiers, so an event straddling `query_time` gets decay=1.0 just like a recent window. A tier-local span was always wider than the recent queue's, so recent windows got decay ≈ 1.0 for free while events were penalised by a much larger denominator — one of two causes of the sim-score gap between tiers.

Why multiplicative and not the older additive `+ gamma * recency_bonus`: the additive term caps the temporal contribution at `gamma` regardless of how far the entry is in time, so a long, "everything-centroid" event (e.g. a 68 s intro with a 1.4 k-char summary) that matches most kitchen queries could still out-score a correct mid-stream event because `gamma * bonus` could only claw back ≤ `gamma`. A multiplicative `exp(-dt/tau)` prior scales *semantic* score down proportionally — a temporally distant event loses in direct proportion to how strong its semantic match is, which is the correct behaviour for recency-sensitive QAs. This is the same shape used in news / social-feed ranking (Twitter, Reddit, freshness-aware BM25).

**Visual-similarity term splits by tier.** The naive form `cos(q_vis, entry.visual_embedding)` reads the *stored centroid* for all three tiers, but centroid blur is asymmetric across tiers:

- **Events** store a centroid-of-centroids (episode means averaged again). That 2-stage averaging smears peaked semantic content — the event vector drifts toward a generic "typical scene" that under-scores against any specific query. **Fix: peak-over-reps.** Score the query against each embedding in `representative_window_ids` and take the max cosine: `vis_sim(event) = max_{w ∈ reps(event)} cos(q_vis, w_vis)`. The rep windows are the pooling winners already persisted for grounding, so this is free at read time.
- **Episodes** store a time-aware self-centrality pooled vector (softmax over centrality + consistency scores). The pooling already weights members toward the central, visually-agreeing frame, so the episode vector sits on the "typical frame" rather than on the arithmetic mean. It does not suffer the centroids-of-centroids blur events do — we keep the stored centroid for episode scoring.
- **Recent windows** use their own embedding directly — no pooling, no blur.

The centroid is still fully used at write time for both tiers: novelty gate, episode-to-event clustering, and the persisted topic vector. Peak-over-reps is purely a read-time rescoring — storage is unchanged.

This keeps the system visual-first, consistent with multimodal-memory approaches like StreamBridge, while structurally suppressing length-bias centroid failures and closing the cross-tier sim-score gap.

### Retrieval output

Return:
- selected time ranges
- selected episodic entries
- neighboring recent windows
- optional summaries
- scores

This returned bundle is the **retrieved evidence set**.

---

## 6. Reasoner Input Formatter

### Purpose

Turn retrieved evidence into a clean representation for downstream reasoning.

### Notebook implementation

The notebook does **not** implement a full multimodal generator. Instead, it formats retrieved evidence as a readable evidence block:

- timestamps
- selected entries
- short summaries
- optional scores

This is a **human-readable serialization of retrieved memory**, not the low-level visual-token bridge itself.

### Conceptual full-system interpretation

In a full multimodal model, retrieved visual memory would be projected into the language model’s input space and combined with the text query. The notebook uses a readable proxy instead, because the goal is to demonstrate memory, retrieval, and reasoning support rather than fully reimplement a Video-LLM. IXC2.5-OmniLive’s high-level architecture is consistent with this separation between memory retrieval and reasoning.

---

## 7. Baseline System

### Purpose

Provide a strong recency-only baseline that the memory architecture must beat or complement.

### Baseline design

Use a **SimpleStream-style recent-window baseline**:
- keep only the last `N` windows
- no episodic memory
- no long-term summaries
- no retrieval over history
- answer using only recent context

### Why this baseline is required

SimpleStream shows that a plain sliding-window baseline with the most recent frames can already match or surpass many more complicated streaming methods on current benchmarks. Therefore, any memory-based architecture should be evaluated against a strong recent-window control rather than being assumed useful by construction.

---

## 8. Data structures

The system uses three main memory objects: **window entries**, **episode entries**, and **event entries**. Window entries are the smallest stored visual units and represent short local time ranges. Episode entries merge consecutive related windows into short coherent actions. Event entries merge consecutive related episodes into higher-level activities. This hierarchy is our implementation choice, designed to make storage and retrieval explicit while staying consistent with the general hierarchical-memory direction seen in recent streaming and long-video work.

```python
from dataclasses import dataclass

import numpy as np


@dataclass
class WindowEntry:
    entry_id: str
    start_time: float
    end_time: float
    visual_embedding: np.ndarray
    summary_text: str | None = None
    summary_embedding: np.ndarray | None = None


@dataclass
class EpisodeEntry:
    entry_id: str
    start_time: float
    end_time: float
    visual_embedding: np.ndarray
    member_window_ids: list[str]
    summary_text: str
    summary_embedding: np.ndarray | None = None


@dataclass
class EventEntry:
    entry_id: str
    start_time: float
    end_time: float
    visual_embedding: np.ndarray
    member_episode_ids: list[str]
    representative_window_ids: list[str]
    summary_text: str
    summary_embedding: np.ndarray | None = None
```

Window entries preserve fine local evidence, episode entries support medium-range action retrieval, and event entries support coarse semantic routing. This is the key structural choice that lets the retriever operate coarse-to-fine instead of searching a flat memory store.

---

## 9. Online update algorithm

On each new local window, the system first computes its visual embedding and appends it to recent memory. If recent memory exceeds capacity, the oldest recent entry is checked for novelty relative to adjacent recent windows. If it is too redundant, it is discarded. If it is sufficiently novel, it is promoted to episodic memory and optionally receives a short local summary. This is the notebook-level analogue of FluxMem’s idea that recent memory should be compressed by filtering temporal redundancy rather than stored exhaustively forever.

If episodic memory then exceeds capacity, older episodic entries are merged into a higher-level summary object. In practice, this means grouping temporally close and semantically related episodic entries, computing a centroid visual embedding from the episode embeddings, choosing representative windows (taken directly from each episode’s `representative_window_ids`), and creating a detailed summary text via **Qwen2.5-VL-3B-Instruct** over the episode texts plus a few representative frames. That compressed object is written into long-term summary memory.

This update rule is intentionally **query-agnostic on the write side**: the system does not know future questions in advance, so it stores memory based on recency, novelty, and bounded capacity rather than on a specific query.

```python
def update_memory(new_window):
    add_to_recent(new_window)
    if recent_overflow():
        old_window = pop_oldest_recent()
        if is_novel(old_window, neighbors=get_recent_neighbors()):
            episodic_entry = promote_to_episodic(old_window)
            # episode embedding: self-centrality pooling over member window embeddings
            episodic_entry.visual_embedding = self_centrality_pool(member_windows)
            episodic_entry.representative_window_ids = top_weight_windows
            add_to_episodic(episodic_entry)
    if episodic_overflow():
        old_batch = pop_oldest_episodic_batch()
        summary_entry = merge_into_summary(old_batch)  # centroid of episode embeddings
        add_to_long_term(summary_entry)
```

---

## 10. Query algorithm

When a user query arrives, the system encodes the text query into the same retrieval space used for memory search. Retrieval then happens in three stages. First, the system performs coarse routing over long-term summary memory to find the most relevant event-level regions. Second, it searches episodic memory only inside those selected time ranges. Third, it adds nearby recent windows around the best episodic hits for local grounding and temporal continuity. This is a simplified coarse-to-fine retrieval design inspired by VideoTree’s hierarchy and FlexMem’s focus on recalling only relevant memory fragments, but implemented with lightweight embedding search rather than expensive LLM-in-the-loop search.

The main retrieval score should be based on **visual similarity**. Summary similarity can be used as an auxiliary reranking signal, and recency can be used as a weak bonus. This keeps the system visual-first while still allowing summary text to help with semantic routing and interpretability.

```python
def retrieve(query, top_m=3, top_k=5, neighbor_radius=1, recent_episodes=5):
    q_emb = encode_query(query)

    # stage 0: always search recent windows
    recent_hits = similarity_search_recent(q_emb, top_k=top_k)

    # stage A: coarse routing over events
    coarse_hits = search_long_term_events(q_emb, top_m=top_m)
    candidate_ranges = [hit.time_range for hit in coarse_hits]

    # stage B: fine search over episodes in candidate ranges, unioned with the
    # trailing `recent_episodes` entries so tail episodes not yet consolidated
    # into any event remain searchable (mirrors the stage-0 recent-windows pass).
    episodic_hits = search_episodic_within_ranges(
        q_emb,
        ranges=candidate_ranges,
        top_k=top_k,
        recent_episodes=recent_episodes,
    )

    # stage C: grounding from episode member archive
    archive_windows = [
        get_grounding_windows(ep, radius=neighbor_radius)
        for ep in episodic_hits
    ]

    grounded_windows = deduplicate(recent_hits + flatten(archive_windows))
    return {
        "coarse_hits": coarse_hits,
        "episodic_hits": episodic_hits,
        "grounded_windows": grounded_windows,
    }
```

---

## 11. Why this architecture was chosen

This architecture is a deliberate hybrid. IXC2.5-OmniLive contributes the clean perception–memory–reasoning separation. FluxMem contributes the write-side idea that streaming memory should be hierarchical and should compress older content more aggressively by filtering redundancy. VideoTree contributes the read-side idea that retrieval should be coarse-to-fine rather than flat over the entire history. SimpleStream contributes the evaluation discipline that a strong recent-window baseline must always be included.

The resulting design is practical for a one-week notebook project because it separates concerns cleanly. The write path is simple enough to implement with embeddings, novelty checks, and summary merging. The read path is explicit and query-conditioned, which matches the assignment better than a design that simply concatenates the full compressed memory hierarchy at inference time. The baseline is strong enough to make any gains from memory meaningful rather than assumed.

---

## 12. What is intentionally simplified

Several parts of the architecture are intentionally simplified relative to the research papers. FluxMem’s exact token-level Temporal Adjacency Selection and Spatial Domain Consolidation are replaced by window-level novelty filtering and summary-level consolidation. VideoTree’s adaptive LLM-scored tree expansion is replaced by embedding-based hierarchical retrieval. IXC2.5-OmniLive’s full real-time multimodal interaction loop is reduced to a notebook-friendly perception–memory–reasoning prototype.

These simplifications preserve the main ideas while keeping the implementation manageable and interpretable. The system also does not implement KV-cache memory, end-to-end multimodal generation, or dense captioning for every frame. Those directions are interesting, but they are beyond the scope of this project and are not necessary to demonstrate the main assignment goals: storing useful visual information over time, retrieving it for later questions, and showing how the retrieved context would support reasoning.

---

## 13. Minimal component list for code organization

Even if the final deliverable is a notebook, the logic should be separated conceptually into a few small components:

- **StreamReader**
- **PerceptionEncoder**
- **MemoryWriter** (`HierarchicalMemoryWriter`)
- **SummaryBuilder** (wraps Florence-2 / Moondream2 / Qwen2.5-VL)
- **MemoryStore** (peewee/SQLite persistence facade — optional, pass `store=None` for pure in-memory)
- **Retriever**
- **RecentWindowBaseline**
- **ReasonerInputFormatter**

This separation keeps the architecture readable and mirrors the perception–memory–reasoning split used in recent systems. It also makes it easier to keep the notebook focused on experiments, visualizations, and explanations while moving reusable logic into helper Python modules if needed.

---

## 14. Persistence layer

`MemoryStore` (see `src/memory_db.py`) is a thin **peewee/SQLite** facade that the writer calls through as windows, episodes, and events are produced. It is optional — pass `store=None` to `HierarchicalMemoryWriter` for a pure in-memory run.

- **Tables**: `Window`, `Episode`, `EpisodeWindow`, `Event`, `EventEpisode`, `EventRepWindow`. Membership is represented by join tables with `ON DELETE CASCADE` on the parent side.
- **Embeddings** are stored as raw `float32` blobs with an `embedding_dim` column (recovered as `np.frombuffer` on read).
- **Representative frames** are stored as JPEG blobs (quality 90) when present.
- **Idempotency**: windows use `INSERT ... ON CONFLICT` preserving later tier/summary updates (a window may flip from `recent` → `episodic` during its lifetime); episodes and events use `REPLACE` plus a wipe-and-rewrite of their membership rows.
- **SQLite pragmas**: WAL journaling, `foreign_keys=1`, `synchronous=normal`, `cache_size=-64*1024`.

This keeps ingest observable (counts after a run, joinable tables for inspection) without coupling the writer to any specific storage layout.

---

## 15. Observed failure modes and mitigations

The pipeline has been debugged against real StreamingBench footage (a 3-minute sports clip). The failure classes below are documented here so future work knows what has been addressed and what remains.

### Resolved

- **Qwen vision-tower overflow producing `!!!!!…`.** The VLM was loaded in fp16 by default; fp16 overflows in the Qwen2-VL / Qwen2.5-VL vision tower and the LM emits repeated token 0. Fix: force **bf16** via `_vlm_dtype_for_device`, and cast `pixel_values` / `pixel_values_videos` to `model.dtype` after `.to(device)` (HF's `BatchFeature.to()` does not cast dtype).
- **~30 GB OOM on eager attention.** Default `max_pixels` on Qwen processors is ~12.8 MP, producing up to ~16 k patches per image and an N² attention allocation that exceeds VRAM. Fix: processor `min_pixels=64·28²`, `max_pixels=320·28²`, plus `attn_implementation="sdpa"` on CUDA.
- **Finalize-time tail loss.** The end of the stream left ~60 s as `tier=recent` with no episode membership, because the recent deque was never drained. Fix: `finalize()` now drains the deque through the same novelty path used during streaming, then force-consolidates residual episodes.
- **Event-tier repetition degeneracy.** With `repetition_penalty=1.05` and no n-gram constraint, greedy decoding on repetitive multi-image inputs (17 similar soccer scenes) locked into a self-reinforcing template ("the Nth scene shows…") for hundreds of characters. Fix: `repetition_penalty=1.15` plus `no_repeat_ngram_size=6` — the n-gram constraint is what actually breaks the loop at the phrase level.
- **Florence `<DETAILED_CAPTION>` hallucination.** At the episode tier Florence invented named entities (team names, player names, scores, leagues, seasons, logos) from misread overlays; those hallucinations then compounded through the event fusion step. Fix: episode tier moved to **Moondream2** with an explicit anti-hallucination prompt; Florence remains at the cheaper window tier where its short `<CAPTION>` is used only as a routing hint.
- **Self-contradicting arithmetic at 2B ("scoring twice, leading to 0-0").** Qwen2-VL-2B produced fluent but arithmetically incoherent narration under a word-count target. Mitigation: upgraded to **Qwen2.5-VL-3B** (larger capacity for multi-constraint instruction following) — the pathological cases disappear.
- **Coarse-tier flakiness near event boundaries.** Events only form when an `event_max_gap` break is observed, so a QA firing at `t` while an episode is still assembling finds only older, unrelated content. Fix: `HierarchicalMemoryWriter.flush_pending()` closes the currently-forming episode on demand so the tail episode becomes searchable at stage B; `scripts/main.py` calls it once per batch of due QAs before retrieval runs. Earlier implementation also force-consolidated the entire `episodic` tier into events — that over-drain made stage-B fine search structurally empty (every completed episode vanished into `long_term` at every QA). Current semantics: `flush_pending` only promotes `_pending_episode → episodic`; the recent-episodes bypass in `_fine_search` handles coarse-tier visibility of tail content. Cost: one Moondream call per QA that lands mid-episode; benefit: tail episode becomes queryable without sacrificing the fine tier.
- **Fine-tier gated-out by coarse time ranges.** With at least one `EventEntry` present, stage B filters episodes to those overlapping any top-M event range, falling back to "all episodes" only when the overlap set is empty. That silently hides any episode whose time range sits outside every top-M event — most commonly the **tail of episodes not yet consolidated into an event**, which sits after the last event's `end_time` and never intersects a coarse window. Fix: stage B now unions the gated candidates with the **last `recent_episodes` entries of the episodic tier** (default 5, dedup by `entry_id`) before scoring, mirroring the stage-0 recent-windows pass. Tail content is guaranteed searchable without losing the coarse-routing efficiency for older content.

### Residual at the event tier (not a model-capacity problem)

- **Score-side attribution is a coin flip.** The scoreboard reads as `X-Y` with no visible home/away label; the model picks a side. Captions of the form "0-1" → "1-1" are read as an equaliser *by some team*, but which team is assigned the goal is unreliable.
- **World-knowledge grafting.** Once a team name is visible, pretrained knowledge leaks in (stadium, league, typical scorelines) as if observed.
- **Brand leak.** Brand names visible in the frame (e.g. a sponsor logo) are occasionally named despite the anti-brand prompt — small instruction-tuned VLMs honour *negative* constraints less reliably than positive ones.
- **No cross-event state.** Each event fusion call is stateless; the closing score of event N is not piped forward as context for event N+1, so adjacent events can disagree on the running scoreline.

Addressing the residuals is architectural, not model-side: pre-extract scoreboard patterns (`\d+-\d+` plus timestamps) from the episode captions and inject them as a hard-constrained list in the fusion prompt; thread the previous event summary forward as carry-over context; post-filter flagged entity classes (brands, stadium names, seasons) from the VLM output. These are deferred beyond the current scope.

---

## 16. Process log — fixes and parameter switches

This section records the debugging journey of the last iteration — the order things were discovered, why each change was made, and what it unblocked. It reads chronologically so a future reader can reconstruct the reasoning.

1. **`TypeError: scaled_dot_product_attention() got an unexpected keyword argument 'enable_gqa'`.** Recent Moondream revisions call `F.scaled_dot_product_attention(..., enable_gqa=True)`, a kwarg added in PyTorch ≥ 2.5. The requirements pin was `torch>=2.0,<2.3`, producing the crash as soon as a query path hit Moondream. Resolved by bumping `torch>=2.5,<2.9`, `torchvision>=0.20`, `transformers>=4.45,<5.0` (option A, chosen over pinning to the older `moondream_revision='2024-08-26'` so that the rest of the HF stack could follow along). Moondream's three-level API (`query` → `caption` → `encode_image + answer_question`) was also wrapped in try/except with a one-shot warning so a future API break degrades gracefully to Florence instead of crashing.

2. **Episode-caption timestamp placement.** Initial intuition: put the time tag inside the Moondream prompt so the model is aware of temporal position. Pushback: that invites Moondream to speculate temporally ("the player is celebrating because earlier…"), re-introducing exactly the before/after fabrication the prompt is trying to forbid. Chosen design: motion-hint stays in the prompt (so the model knows frames are frozen mid-action and can describe pose/blur/trajectory), but the temporal tag `[t={start_time:.1f}s | frame i/N]` is **prepended to the caption output**, not asked from the model. Ground-truth tags come from `WindowEntry.start_time`, never the model — Qwen2.5-VL then sees reliable anchors downstream.

3. **Qwen-VL output of `!!!!…`.** The 2B VLM was loaded in fp16. fp16 overflows in Qwen's vision tower; the LM head then emits token 0 repeatedly. Moved VLM dtype selection out of `_dtype_for_device` (used by Florence on CUDA/MPS fp16) into a dedicated `_vlm_dtype_for_device` that returns bf16 on CUDA/MPS/CPU. Also added an explicit cast of `pixel_values` and `pixel_values_videos` to `model.dtype` after `.to(device)` because `BatchFeature.to(device)` moves tensors without casting dtype — easy to miss.

4. **~30 GB OOM on eager attention.** Qwen's default processor uses `max_pixels ≈ 12.8 MP` and `attn_implementation="eager"`, producing up to ~16 k patches per image and an N² attention allocation. Fix: `min_pixels=64·28²`, `max_pixels=320·28²` (64–320 patches per frame) and `attn_implementation="sdpa"` on CUDA. Relaxed the cap from an earlier `256·28²` to `320·28²` after confirming there was headroom on the target GPU — the extra patches give the model slightly more room on small on-screen text.

5. **"The Nth scene shows…" template loop.** With `repetition_penalty=1.05` and no n-gram constraint, greedy decoding on the 17-scene soccer match locked into a self-reinforcing template for hundreds of characters. Token-level repetition penalty alone did not break the loop because the repetition is at phrase granularity. Fix: `repetition_penalty=1.15` + `no_repeat_ngram_size=6` + `max_new_tokens=640`. The 6-gram constraint is what actually mattered.

6. **Self-contradicting arithmetic at 2B ("scoring twice, leading to 0-0").** After the loop was fixed, Qwen2-VL-2B still produced fluent but arithmetically incoherent narration when reconciling a sequence of scoreboard captions. This is a model-capacity issue, not a decoding issue. Switched to **Qwen2.5-VL-3B-Instruct**. The loader now auto-detects family (`qwen2-vl` → `Qwen2VLForConditionalGeneration`, `qwen2.5-vl` → `Qwen2_5_VLForConditionalGeneration`) from the HF id so swapping model sizes is one string change.

7. **Episode-tier hallucinations (Florence `<DETAILED_CAPTION>`).** Florence invented team names, player names, scoreline narratives, and league IDs from misread on-screen overlays; those fabrications then got laundered through the event-fusion VLM. Fix: moved episode captions to **Moondream2** with an explicit anti-hallucination prompt (single-frame framing, forbid naming entities, forbid before/after speculation, prefer generic descriptors like "a player in a red-and-blue striped jersey" when identity is unclear). Florence stays at the cheaper window tier (high call volume) where short `<CAPTION>` output is low-risk and used only as a routing hint.

8. **Finalize-time tail loss.** `finalize()` originally just called `_flush_current_episode()`; anything still sitting in the `recent` deque at end-of-stream never got promoted, so the last ≈ 60 s of a 3-minute video ended up with `tier=recent` and no episode membership. Fix: `finalize()` now drains the deque through the same novelty path used during streaming, then force-consolidates residual episodes into events even below the capacity threshold.

9. **DB duplication across re-runs.** Running `main.py` twice against the same SQLite file doubled every row because `MemoryStore.save_*` used `REPLACE`/`INSERT … ON CONFLICT`, but each run generated fresh UUID-based `entry_id`s — so "conflict on primary key" never fired. (The in-memory retriever was unaffected; the DB just grew.) Documented as a re-run hygiene issue; either wipe the db between runs or deterministic-hash the `entry_id`.

10. **Retrieval testing harness.** Added `qas.json` scheduling to `scripts/main.py`: `_load_qas` reads `HH:MM:SS` timestamps, `_process_due_qas` fires retrieval whenever the stream clock crosses a QA timestamp, and the rendered coarse/fine/grounding hits are appended to `outputs/retrievals.md` as markdown blocks (one per QA, with the QA, options, and ground-truth answer included so the file is self-contained for offline scoring). No LLM reasoner yet — this stops at the formatted evidence.

11. **Coarse-tier flakiness near event boundaries.** Caught on QA5 ("PRIME cooler colour" at `t=128s`): the correct episode had not yet been promoted from `_pending_episode` at query time because the gap/capacity thresholds hadn't fired. Coarse routing fell back to the noise event `[0–2s]` SUBSCRIBE at sim=0.096 while recent-tier grounding saved the answer. Fix: `HierarchicalMemoryWriter.flush_pending()` closes the in-progress episode on demand; `_process_due_qas` calls it once per batch of due QAs before retrieval runs. Follow-up bug caught while evaluating retrievals.md: the first iteration also wrapped a `while self.episodic: self._consolidate_episodic()` loop inside `flush_pending`, which drained every completed episode into events at every single QA call — so stage B (fine search) saw an empty `episodic` tier and the "Fine —" block was structurally absent from every rendered retrieval. Fix: drop the drain loop; `flush_pending` now only calls `_flush_current_episode()`. Tail-content searchability is provided by the recent-episodes bypass in `_fine_search` instead of by a coarse-tier promotion. Test: `tests/test_memory_writer.py::FlushPendingSemanticsTests`.

12. **Fine-tier gated-out by coarse time ranges.** Once at least one event existed, `_fine_search` restricted episode candidates to those overlapping any top-M event range (falling back to "all episodes" only when the overlap was empty). The tail of episodes that hadn't yet been rolled into any event sat after the last event's `end_time` and never intersected a coarse window, so those episodes were silently dropped from fine search even when they were the semantically best match. The symptom was symmetric to the recent-windows bypass in stage 0 — we already special-cased recent windows for exactly this reason; the same argument applies to recent episodes. Fix: union the gated candidates with `episodic[-recent_episodes:]` (dedup by `entry_id`) inside `_fine_search`, default `recent_episodes=5`. Caveat logged for future work: this closes the *tail* blind spot but does not address the adjacent failure in EVALUATION_REPORT §1 where the intro event dominates top-M and gates out *mid-stream* episodes too — that wants an event-duration cap or a top-M diversity rule and is tracked separately.

13. **Intro-event domination — additive γ cannot fix it.** EVALUATION_REPORT §1: the `[0–68 s]` intro event won a slot in all five coarse top-3s with sim 0.205–0.245. Root cause: it's simultaneously the longest event (68 s), has the longest summary (~1.4 k chars — more surface area for token overlap with kitchen queries), and has a visual embedding that reads as a "generic kitchen centroid". Under the additive `α·vis + β·txt + γ·recency` form, bumping γ (0.05 → 0.15 → 0.25) can only add at most γ to the temporal-near entry and subtract nothing from the temporal-far intro, so the intro keeps winning whenever its semantic lead exceeds γ. **Fix: switch the temporal term from additive bonus to multiplicative prior**, `score = (α·vis + β·txt) · exp(-dt/τ)`. A temporally distant entry is now scaled down in proportion to its semantic score, which structurally kills the centroid-wins-everything failure. `α=0.70, β=0.30, τ = 0.50 · stream_span` (initial 0.25 was too sharp — see parameter-history table); γ as a parameter is removed. Test: `tests/test_retriever.py::test_multiplicative_decay_suppresses_semantically_strong_distant_entry`.

14. **Unified decay span across tiers.** After the multiplicative-decay switch, sim-score printouts still showed a systematic gap: recent windows sat around 0.25–0.35, episodes around 0.05–0.08, events around 0.01–0.03. Part of the gap was the decay denominator: each tier computed its own `stream_span` locally, and since the recent queue spans at most `recent_capacity * window_duration` (~60 s) while events span the full stream (~minutes to hours), the recent-tier `tau` was always far tighter than the event-tier `tau`. A recent window straddling `query_time` got `decay=1.0`; an event also straddling `query_time` was punished by a much larger denominator somewhere else along the exponential curve. Fix: compute a single **unified span** across `recent ∪ episodes ∪ events` in `HierarchicalRetriever.retrieve()` and pass it down to all three scoring passes. The span is now one ruler — straddling entries get `decay=1.0` at every tier, and the far ends of the stream compare like-for-like.

15. **Tier-split visual scoring — peak-over-reps for events, centroid for episodes.** Unified span closed part of the gap but raw semantic cosines remained 3–4× lower at the event tier than at the window tier. Second root cause: **centroid blur**. Event embeddings are centroids-of-centroids — episode means averaged again — so a peaked semantic feature in one member window gets smeared across two pooling stages. Against any specific query, the event cosine drops toward "generic scene" rather than the strongest matching moment. Fix: at read time, score the query against each embedding in `representative_window_ids` (already persisted for grounding) and take `max_{w ∈ reps(e)} cos(q_vis, w_vis)`. A new `_build_rep_index(memory, events)` helper builds the lookup once per `retrieve()` call; `_vectors_for(entry, rep_vectors)` falls back to `[entry.visual_embedding]` when an entry has no reps. Episodes are deliberately kept on the stored centroid — time-aware self-centrality pooling already weights members by centrality and visual agreement, so the episode vector sits on the "typical frame" without the 2-stage averaging blur events have; peak-over-reps on episodes would fight that pooling rather than complement it. Recent windows pass their own embedding directly. `_blended_score` takes a `List[np.ndarray]` and `max()`s over it, so the episode/window path degenerates to a plain cosine. Storage and write-time semantics are unchanged: centroid still drives the novelty gate, episode-to-event clustering, and the stored topic vector.

16. **Window text channel silently dead at grounding stage.** After the weight rebalance to `α=0.70, β=0.30`, stage-C grounding was still scoring windows on visual similarity alone: `WindowEntry.summary_text` was populated by the per-window caption path, but `summary_embedding` stayed `None` because the writer never embedded it. The retriever's β term short-circuited to 0 for every window, so `β=0.30` was dead weight at the finest tier — the one that actually answers the QA. Episodes and events were unaffected (they go through `_embed_summary()` on consolidation). Fix: `HierarchicalMemoryWriter.update(window)` now calls `self._embed_summary(window.summary_text)` when `summary_embedding is None and summary_text` is non-empty; caller-provided embeddings are preserved (batched upstream paths keep working). No schema change — `Window.summary_embedding` was already persisted as a `BlobField`. No wiring change in `scripts/main.py` — `text_encode_fn=encoder.encode_text` was already plumbed through the constructor (process-log item before this one). Test: `tests/test_memory_writer.py::WindowSummaryEmbeddingTests`.

### Parameter history

| Parameter | Before | After | Reason |
|-----------|--------|-------|--------|
| `torch` pin | `>=2.0,<2.3` | `>=2.5,<2.9` | Moondream `enable_gqa` SDPA kwarg |
| `transformers` pin | `>=4.41,<5.0` | `>=4.45,<5.0` | Matches torch bump |
| VLM dtype | fp16 | bf16 | fp16 overflow → `!!!!…` |
| VLM attention | eager | sdpa (CUDA) | eager N² OOM on default pixels |
| VLM `max_pixels` | ~12.8 MP (default) | `320·28²` | token-budget cap |
| VLM `max_new_tokens` | 512 | 640 | room for adaptive word target |
| VLM `repetition_penalty` | 1.05 | 1.15 | template-loop suppression |
| VLM `no_repeat_ngram_size` | — | 6 | phrase-level loop suppression |
| VLM frames per event | 8 | 10 | small quality gain, cost negligible |
| Event-fusion model | Qwen2-VL-2B | Qwen2.5-VL-3B-Instruct | arithmetic / multi-constraint instruction following |
| Episode captioner | Florence `<DETAILED_CAPTION>` | Moondream2 + anti-hallucination prompt + `[t=Xs \| frame i/N]` tag | Florence fabricates named entities |
| `finalize()` | flushed pending episode only | drains recent + force-consolidates | tail windows were being stranded as `tier=recent` |
| Boundary synchronisation | none | `flush_pending()` on QA — promotes `_pending_episode` only | tail episode not queryable until next natural gap |
| `flush_pending` drain scope | also ran `while self.episodic: _consolidate_episodic()` | only `_flush_current_episode()` | over-drain emptied the episodic tier at every QA, killing stage-B fine search |
| Retrieval γ (recency) | 0.05 → 0.15 → 0.25 | **removed** | Even at γ=0.25, additive recency caps the temporal term at γ regardless of how far an entry is in time — a semantically strong distant entry still out-ranks a semantically weaker close one. Replaced entirely by a multiplicative prior below. |
| Retrieval scoring form | `α·vis + β·txt + γ·bonus` (additive, capped) | `(α·vis + β·txt) · exp(-dt/tau)` (multiplicative, uncapped decay) | Temporally distant entries are scaled down in proportion to their semantic score; structurally kills the "intro-event centroid wins every coarse top-3" failure. |
| Retrieval α (visual) | 0.65 → 0.60 → 0.80 | 0.70 | Semantic weights sum to 1 (γ removed). Dropped from 0.80 to 0.70 after X-CLIP text chunking (`PerceptionEncoder.encode_text`) made long-summary embeddings actually usable — text-to-text cosines are stronger and more discriminative than the earlier cross-modal (text ↔ visual-centroid) channel, so β earns a larger slice. |
| Retrieval β (summary) | 0.30 → 0.20 → 0.15 → 0.20 | 0.30 | Raised alongside α drop once `summary_embedding` actually carries signal end-to-end (text_encode_fn wired, long summaries chunked instead of truncated). |
| `tau_fraction` (new) | — → 0.25 | 0.50 | Decay timescale as a fraction of stream span. Initially picked 0.25 (quarter-span → multiplier ≈ 0.37) — too aggressive; events in the opposite *quarter* of a long stream got clipped to ~0.14, which could over-discount legitimately earlier content. Bumped to 0.50 so half-span-away is at ≈ 0.37 and full-span-away is at ≈ 0.14 — still a meaningful recency preference, but no longer wipes out older content. If "earlier in the video" queries start under-ranking their targets, raise further (0.75–1.0). |
| Fine-stage candidate pool | coarse-gated only (fallback to all on empty overlap) | coarse-gated ∪ `episodic[-recent_episodes:]` | tail episodes not yet consolidated into any event were silently gated out; bypass mirrors the recent-windows pass in stage 0 |
| `recent_episodes` (new) | — | 5 | trailing-episode bypass count; symmetric with `top_k` for recent-windows search |
| Window `summary_embedding` at ingest | not written (stayed `None`) | embedded in `HierarchicalMemoryWriter.update()` when `summary_text` is present and no embedding was provided | β·txt term was silently zeroed at stage-C grounding; β=0.30 of scoring weight was dead weight for windows. Caller-provided embeddings are preserved (batched upstream paths unaffected). |
| Decay span scope | per-tier local `stream_span` | **unified** across `recent ∪ episodes ∪ events` (computed once per `retrieve()` call) | tier-local spans gave recent-tier scoring a much tighter `tau` than the event tier, so a window straddling `query_time` got decay=1.0 while an event straddling `query_time` was penalised by a far larger denominator — one of two causes of the cross-tier sim-score gap |
| Event visual similarity | `cos(q_vis, event.visual_embedding)` (stored centroid) | `max_{w ∈ reps(event)} cos(q_vis, w_vis)` (peak over representative windows) | event embeddings are centroids-of-centroids (episode means averaged again); the 2-stage averaging smears peaked semantic content and drops cosines toward "generic scene". Peak-over-reps uses the grounding-pool winners already persisted on `EventEntry.representative_window_ids` — no extra storage. |
| Episode visual similarity | — | `cos(q_vis, episode.visual_embedding)` (stored centroid, unchanged) | considered peak-over-reps for symmetry, kept centroid instead: time-aware self-centrality pooling already weights members by centrality + visual agreement, so the episode vector sits on the "typical frame" without the centroids-of-centroids blur events have; peak-over-reps would fight that pooling rather than complement it |
| Rep-vector lookup | — | `_build_rep_index(memory, events)` built once per `retrieve()` call, maps `entry_id → List[w.visual_embedding]` via `_window_archive` | amortises archive lookups across the three scoring passes; falls back to `[entry.visual_embedding]` in `_vectors_for(...)` when no reps are stored |

---

## 17. Scaling to long (1h+) video

The assignment suggests "e.g. 1h+" as an example stream length. The
longest clip the system has actually been exercised on is `sample_36`
at **17 min**. Everything below is a scaling analysis — it explains
what the architecture predicts should happen at 1h+, what is bounded
by construction, and what has never been empirically verified.

### Per-tier growth bounds

| Tier | Data structure | Bound | 1h projection |
|------|----------------|-------|---------------|
| Recent windows | `collections.deque(maxlen=recent_capacity)` | **Hard cap 20** | 20 |
| Episodic | `list`, capped at `episodic_capacity=50` via consolidation | **Hard cap 50** (consolidates into events when full) | 50 |
| Long-term events | `list`, no cap | **Slow-growing** | ~5–30 at current `event_max_gap=15s` / `event_min_sim=0.55` |
| Window archive (`_window_archive: Dict[entry_id, WindowEntry]`) | dict | **Unbounded** (every ingested window stays addressable for grounding) | 3600 windows @ 1 fps / 1 window = 3 s |

The first three are bounded by construction and require no change for
longer streams. The fourth is the only structurally unbounded
container on the write path.

### Retrieval cost at stream length N windows

Per `retrieve()` call:
- **Stage 0 (recent)**: O(`recent_capacity`) = O(20), independent of N.
- **Stage A (coarse events)**: O(|events|) with `max`-over-reps
  replacing stored centroid similarity — per event costs O(|reps|),
  and `|reps|` is capped by the pooling at ~3–5 windows.
  Effectively O(|events| · 5).
- **Stage B (fine episodes)**: O(|coarse-gated episodes| ∪
  `recent_episodes`). Coarse-gated subset has an upper bound of
  `|episodes|` = 50 by the write-path cap; the bypass adds at most 5
  more. Effectively O(55).
- **Stage C (grounding windows)**: O(sum of `member_window_ids` across
  fine hits). Per-episode member counts are bounded by
  `episode_max_len`, which caps at the time-gap based episode-flush
  policy — typically 5–15 members.
- **`_build_rep_index`**: O(|events| · |reps|) dict build, amortised
  over the three scoring passes.

**No O(N²) operations.** Retrieval is linear in the number of *summary*
entries (events + episodes), both of which are capped or slow-growing,
not linear in raw window count N.

### Wall-clock projection for a 1h video

Rough per-stage costs, measured on the 3-min sample_1 run (63 windows,
16 episodes, 5 events in 82 s on CUDA with all three VLMs):

| Stage | 3-min cost | 1h projection (20× scaling) |
|-------|-----------|-----------------------------|
| X-CLIP window encoding | ~20 s | ~7 min |
| Florence-2 window caption (per window, fixed cost) | ~30 s (63 × 0.5 s) | ~30 min (3600 × 0.5 s) |
| Moondream2 episode caption (per promoted window, self-centrality winners only) | ~15 s (16 episodes × ~3 wins × 0.3 s) | ~20–40 min (5–15× more episodes at 1h) |
| Qwen2.5-VL event fusion (per event, up to 10 frames) | ~17 s (5 × ~3 s) | ~5–30 min (5–30 events × 1–3 s) |
| **Total** | **~82 s** | **~1.5–2 h** on a single CUDA GPU |

**Preflight recommendation**: run the first 1h pass with
`use_vlm=False` (no Qwen event fusion) to get event counts and
retrieval quality in ~1–1.5 h wall time, then re-enable Qwen only if
the event count stays below ~30.

### Known scaling risks (never empirically verified)

1. **Window archive memory.** `WindowEntry.frame` holds an RGB numpy
   array per window. 3600 windows × ~400 KB/frame ≈ **1.4 GB peak RSS**
   just for frames, without the embedding cache, Moondream image
   tensors, or Qwen activations. `MemoryStore` (SQLite, `src/memory_db.py`)
   exists and persists frames as JPEG blobs; opt-in via the
   `store=` kwarg. **Enable for 1h runs.**
2. **Intro-event dominance at stream length.** ARCHITECTURE §15
   documents that the first event's centroid tends to dominate coarse
   routing. The fix (multiplicative decay with `tau = 0.50 · stream_span`)
   was tuned on the 17-min sample_36 run. On a 1h stream, `tau = 30 min`
   — which means an event from the first 10 min still has
   decay ≈ `exp(-20/30) ≈ 0.51` when queried at 30 min in. This may or
   may not be enough; if the intro event still wins routing for
   mid-stream queries, `tau_fraction` should be lowered for long
   streams (or made absolute-time rather than span-relative).
3. **Event tier sparsity.** `event_max_gap=15s` is tight — long-form
   content with natural scene changes every 30–60 s will produce many
   short events rather than a few long, meaningful ones. Consider
   relaxing to 30–60 s for feature-length video.
4. **Retrieval output size.** At `top_k=5` episodic + grounding
   windows, the `text_context` string grows with episode summary
   length, which grows with `episode_max_len`. For a 1h stream with
   many long episodes, prompt length could push past 2–4k tokens —
   fine for modern LLMs but worth measuring.
5. **Florence latency.** Per-window captioning is the dominant fixed
   cost (~0.5 s × 3600 = 30 min just for window captions). If this
   becomes the bottleneck, Florence can be disabled for windows that
   fail the novelty gate — the summary is only required for windows
   that get promoted to episodic.

### What a 1h run would actually tell us

- **Whether intro-event dominance is linear or catastrophic in stream
  length.** This is the single biggest open question — the fix in §15
  works at 17 min; it may or may not survive 60 min.
- **Whether event-tier consolidation produces useful events at that
  scale.** 5 events / 17 min is already sparse; 1h could either
  produce ~15 meaningful events or ~60 useless ones depending on
  scene structure.
- **Whether `tau_fraction=0.50` is still the right default.** May need
  to be tuned per stream length, or made absolute.
- **Whether the window archive needs SQLite offloading by default.**
  Currently opt-in; 1h of in-memory frames will likely force the
  decision.

None of these are show-stoppers for the current design — they are the
*next* tuning pass, and they need real data before being worth
tuning at all.
