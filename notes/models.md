# Model Decisions

## Visual + Text Encoder
**`microsoft/xclip-base-patch32`** (X-CLIP ViT-B/32, ~200M params, 512-dim)

The main retrieval model. `PerceptionEncoder` uses X-CLIP to encode both video windows
and text queries into the same L2-normalised 512-dim embedding space, used for memory
storage and query-conditioned retrieval.

Unlike a static image encoder, X-CLIP takes a short clip of frames and produces one
temporally-aware embedding per window. Each window is encoded from up to 8 uniformly
sampled frames; if a window contains fewer distinct frames, boundary frames are repeated
by `_sample_uniform`. This captures short-term motion rather than a single still-frame
appearance.

The text encoder is shared: queries at retrieval time are encoded into the same joint
space for cosine similarity search. It can also be passed into `HierarchicalMemoryWriter`
as `text_encode_fn` so episode and event summaries receive `summary_embedding` vectors
for the retriever's auxiliary summary-similarity term.

## Summary / Captioning
**`microsoft/Florence-2-base`** (270M params)

Optional (`use_model=True`). `SummaryBuilder` uses Florence-2 twice at different
granularities. At ingest, it captions the representative frame of each window using
`<CAPTION>`, and the resulting short text is stored as `WindowEntry.summary_text`.

At episode flush, Florence runs again on each member window's stored representative frame
using `<DETAILED_CAPTION>`. Those detailed captions are concatenated in temporal order
to form `EpisodeEntry.summary_text`.

Default task prompts:
- `"<CAPTION>"` for per-window ingest captions
- `"<DETAILED_CAPTION>"` for per-episode concatenated captions at flush

## Event Summary Fusion
**`Qwen/Qwen2.5-VL-3B-Instruct`** (3B params, multimodal)

Optional (`use_vlm=True`). `SummaryBuilder` uses Qwen2.5-VL only for fusing a cluster
of episodes into one concise event summary. The model receives both the episode summary
texts and 2 representative frames per episode as actual images, giving it visual
grounding rather than relying on text alone.

Representative frames are the 2 windows per episode closest to the episode's centroid
embedding, selected by cosine similarity in `HierarchicalMemoryWriter._consolidate_episodic`.

## Summary gradient design principle

The lower the memory tier, the more the representation relies on raw visual information.
The higher the compression tier, the more it relies on text:

| Tier       | Primary signal                        | Text role           |
|------------|---------------------------------------|---------------------|
| Recent     | X-CLIP visual embedding                    | none                    |
| Episodic   | X-CLIP centroid + detailed Florence text   | descriptive text layer  |
| Long-term  | X-CLIP centroid + Qwen2.5-VL summary       | primary semantic handle |

At the long-term tier, the Qwen2.5-VL event summary is the main semantic handle for
coarse retrieval routing; the centroid embedding backs up visual similarity search.

## What Is Not Implemented

There is no separate reasoning or QA model wired into the end-to-end pipeline.
`ReasonerInputFormatter` only formats retrieved evidence into a readable block and a
structured dict showing what would be passed to a full multimodal or text-only reasoning
model later.
