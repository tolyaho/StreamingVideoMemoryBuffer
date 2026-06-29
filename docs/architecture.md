# Architecture

The system is split into perception, memory, retrieval, and reasoning. The write path does not know future questions. The read path is query-conditioned.

```text
video -> windows -> embeddings/captions -> memory tiers -> retrieval -> reasoner
```

## Stream Windows

`StreamReader` reads a video with OpenCV, samples frames at the configured fps, and yields `RawWindow` objects. A window stores an id, a start/end time, and sampled RGB frames.

`PerceptionEncoder` turns each window into one X-CLIP embedding. `SummaryBuilder` also captions the representative frame when the window captioner is enabled. `WindowEntry.from_raw_window` keeps the embedding, caption, timestamps, and one representative frame.

The full sampled frame list is not stored in memory. It is used for embedding and then discarded.

## Memory Tiers

### Recent

Recent memory is a fixed-size deque of `WindowEntry` objects. It stores the newest windows densely because recent context is a strong baseline.

When recent memory overflows, the oldest window is checked against the remaining recent queue. If it is redundant, it is discarded. If it is novel, it is promoted into the episode builder.

### Episodic

Episodic memory stores coherent spans of promoted windows. A promoted window joins the pending episode when the time gap is small, visual similarity stays high, and the episode length cap has not been hit. Otherwise the pending episode is flushed.

Flushing creates an `EpisodeEntry` with:

- time range
- self-centrality pooled visual embedding
- member window ids
- representative window ids
- episode summary
- optional summary embedding

The representative windows are the high-weight members from the pooling step, not a separate learned selector.

### Long-Term Events

When episodic memory exceeds capacity, the oldest similar adjacent episodes are merged into an `EventEntry`. The event stores a centroid embedding, member episode ids, representative window ids, and a fused summary.

Event summaries can use Qwen-VL over episode text plus representative frames. If the VLM is off or fails, the code falls back to a text template.

## Retrieval

`HierarchicalRetriever.retrieve` always starts with recent-window search. Then it routes through events, searches episodes inside the selected event ranges plus the recent episode tail, and grounds episodic hits back to representative windows.

The score is:

```text
(alpha * visual_similarity + beta * summary_similarity) * time_decay
```

Events use peak-over-representative-window scoring to avoid centroid blur. Episodes use their pooled embedding. Recent windows use their own embedding.

## Reasoning

`ReasonerInputFormatter` builds a readable evidence block and a structured `visual_context`. The text reasoner uses the evidence text. The VLM reasoner uses the retrieved frames plus the same evidence text.

The embedding arrays in `visual_context` are useful for inspection and future projector work, but the current text reasoner does not consume them directly.

## Current Limits

- The window archive keeps representative frames in memory unless SQLite persistence is used.
- SQLite persistence does not restore a memory state yet.
- Caption quality dominates many wrong answers.
- Long-video event counts still need better diagnostics and tuning.
