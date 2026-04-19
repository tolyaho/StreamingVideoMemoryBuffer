# ARCHITECTURE.md

## Project goal

Build a lightweight streaming Video-LLM memory system that can watch a video stream over time, store useful past visual context under a fixed memory budget, retrieve the most relevant past evidence for a text query, and format that retrieved evidence for downstream reasoning. The design is intentionally inspired by recent streaming-video work, but simplified enough to be implemented cleanly in a notebook plus a few helper Python modules. In particular, the architecture follows a **perception → memory → reasoning** split, uses **hierarchical memory compression** for storage, and uses **coarse-to-fine retrieval** for reading, while always comparing against a strong **recent-window baseline**.

## Design principles

The system is built around five principles. First, **recent visual context is valuable and should be stored densely**, because a simple recent-window baseline already performs surprisingly well on current streaming benchmarks. Second, **older context should be stored more selectively and more compactly** to keep memory bounded. Third, **query answering should not read the entire stored history**; instead, it should retrieve a small relevant subset of memory. Fourth, **text summaries are auxiliary**, useful for routing and readability, but the primary memory representation remains visual. Fifth, **the lower the memory tier, the more the representation relies on raw vision; the higher the compression tier, the more it relies on language** — recent windows are pure visual embeddings; episodic entries add **short** per-window Florence captions for routing plus a **longer** episode string built at flush from **`<DETAILED_CAPTION>`** on each member window’s representative frame; long-term events condense episode text together with a few real frames via a VLM. These principles are directly motivated by SimpleStream’s recency result, FluxMem’s hierarchical compression, VideoTree’s coarse-to-fine retrieval, and IXC2.5-OmniLive’s perception–memory–reasoning split.

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

### Summary encoder — Florence-2-base

A separate captioning model (**Florence-2-base**, `microsoft/Florence-2-base`) is used only for text, not retrieval. **At ingest**, it runs **`<CAPTION>`** on each window’s representative (middle) frame; the result is stored as `WindowEntry.summary_text` and supports the retriever’s summary-similarity term and lightweight logging. **When an episode is flushed**, the same model runs again with **`<DETAILED_CAPTION>`** on **each** member window’s stored representative frame; those strings are **concatenated** (with blank lines between windows) into `EpisodeEntry.summary_text`. That costs one extra Florence forward pass per window per episode at flush time, but yields a richer episodic text layer than stitching short captions alone.

The architecture deliberately separates retrieval (X-CLIP clip embeddings) from summarisation (Florence-2). Retrieval remains visual-first and temporally aware; text summaries are an auxiliary semantic index that grows more descriptive at the episodic tier.

### Summary pipeline

- **Per window (ingest)**: Florence-2 `<CAPTION>` on the representative frame → `WindowEntry.summary_text`; the RGB representative frame is also kept on the entry for later multimodal use.
- **Per episode (flush)**: for each member window in order, Florence-2 **`<DETAILED_CAPTION>`** on that window’s representative frame → concatenate into `EpisodeEntry.summary_text` (falls back to the short caption or a time template if a frame is missing).
- **Per event (consolidation)**: when `use_vlm=True`, **Qwen2.5-VL-3B-Instruct** receives the episode summary strings plus **two** representative frames per episode (windows closest to the episode’s visual centroid, in time order) and returns one concise event sentence; otherwise a template stitch over episode texts is used.

### Why not store raw pixels

The system stores encoder-produced embeddings as the **primary** evidence, not full video. One **representative RGB frame** per window is retained only where Florence or the event VLM needs pixels; that keeps footprint small compared to storing every sampled frame.

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

Store only temporally informative and semantically meaningful past windows.

#### Stored fields

- all recent-memory fields
- `summary_text` for promoted windows
- optional `summary_embedding`
- `tier = "episodic"`

#### Promotion rule

When recent memory overflows:
1. take the oldest recent entry
2. compare it with neighboring recent windows
3. compute a **novelty score**
4. if novelty is low, discard it
5. if novelty is high, promote it to episodic memory

This is the notebook-friendly analogue of FluxMem’s **Temporal Adjacency Selection (TAS)**, which keeps temporally informative tokens and drops redundant ones across adjacent frames.

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

- **Local note**: attached to promoted windows
- **Episode summary**: attached to merged short action spans; built from concatenated detailed Florence captions at flush (see **Summary pipeline** above)
- **Event summary**: attached to longer high-level activities

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
- episode summary text: concatenated **`<DETAILED_CAPTION>`** outputs over member representative frames (see **Summary pipeline** under Perception Encoder above)

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

#### Stage A: coarse routing

Search over **long-term summary embeddings** first.

Input:
- text query embedding

Search target:
- summary embeddings of long-term memory entries

Output:
- top `M` candidate event/time ranges

#### Stage B: fine retrieval

Search over **episodic memory embeddings** inside the selected time ranges.

Input:
- same query embedding
- episodic entries restricted to selected ranges

Output:
- top `K` episodic hits

#### Stage C: local grounding

Attach neighboring recent windows around the best episodic hits.

Purpose:
- recover local detail
- support before/after reasoning
- preserve fine temporal context

### Scoring

Use **visual similarity as the primary signal**.

Optional combined score:

`score = alpha * visual_similarity + beta * summary_similarity + gamma * recency_bonus`

with:
- `alpha` highest
- `beta` small but useful
- `gamma` optional and weak

This makes summaries helpful but keeps the system visual-first, which is more consistent with multimodal-memory approaches like StreamBridge than with text-only memory.

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

If episodic memory then exceeds capacity, older episodic entries are merged into a higher-level summary object. In practice, this means grouping temporally close and semantically related episodic entries, computing a centroid visual embedding, choosing representative members, and creating a concise summary text. That compressed object is written into long-term summary memory. This is simpler than FluxMem’s exact token-level TAS/SDC pipeline, but it preserves the same core idea: older content should be stored more compactly than recent content.

This update rule is intentionally **query-agnostic on the write side**: the system does not know future questions in advance, so it stores memory based on recency, novelty, and bounded capacity rather than on a specific query.

```python
def update_memory(new_window):
    add_to_recent(new_window)
    if recent_overflow():
        old_window = pop_oldest_recent()
        if is_novel(old_window, neighbors=get_recent_neighbors()):
            episodic_entry = promote_to_episodic(old_window)
            add_to_episodic(episodic_entry)
    if episodic_overflow():
        old_batch = pop_oldest_episodic_batch()
        summary_entry = merge_into_summary(old_batch)
        add_to_long_term(summary_entry)
```

---

## 10. Query algorithm

When a user query arrives, the system encodes the text query into the same retrieval space used for memory search. Retrieval then happens in three stages. First, the system performs coarse routing over long-term summary memory to find the most relevant event-level regions. Second, it searches episodic memory only inside those selected time ranges. Third, it adds nearby recent windows around the best episodic hits for local grounding and temporal continuity. This is a simplified coarse-to-fine retrieval design inspired by VideoTree’s hierarchy and FlexMem’s focus on recalling only relevant memory fragments, but implemented with lightweight embedding search rather than expensive LLM-in-the-loop search.

The main retrieval score should be based on **visual similarity**. Summary similarity can be used as an auxiliary reranking signal, and recency can be used as a weak bonus. This keeps the system visual-first while still allowing summary text to help with semantic routing and interpretability.

```python
def retrieve(query, top_m=3, top_k=5, neighbor_radius=1):
    q_emb = encode_query(query)
    coarse_hits = search_long_term_summaries(q_emb, top_m=top_m)
    candidate_ranges = [hit.time_range for hit in coarse_hits]
    episodic_hits = search_episodic_within_ranges(
        q_emb, ranges=candidate_ranges, top_k=top_k
    )
    grounded_hits = add_recent_neighbors(
        episodic_hits, radius=neighbor_radius
    )
    return {
        "coarse_hits": coarse_hits,
        "episodic_hits": episodic_hits,
        "grounded_hits": grounded_hits,
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
- **MemoryWriter**
- **SummaryBuilder**
- **Retriever**
- **RecentWindowBaseline**
- **ReasonerInputFormatter**

This separation keeps the architecture readable and mirrors the perception–memory–reasoning split used in recent systems. It also makes it easier to keep the notebook focused on experiments, visualizations, and explanations while moving reusable logic into helper Python modules if needed.
