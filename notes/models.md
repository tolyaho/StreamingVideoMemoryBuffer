# Model Decisions

## Visual + Text Encoder
**`google/siglip-base-patch16-224`** (SigLIP ViT-B/16, ~200M params, 768-dim)

This is the main retrieval model used in the code.
`PerceptionEncoder` uses SigLIP to encode both video windows and text queries
into the same L2-normalised embedding space, which is then used for memory
storage and query-conditioned retrieval.

The same text encoder can also be passed into `HierarchicalMemoryWriter` as
`text_encode_fn` so episode and event summaries receive `summary_embedding`
vectors for the retriever's auxiliary summary-similarity term.

## Summary / Captioning
**`microsoft/Florence-2-base`** (270M params)

This model is optional, not required for the base pipeline.
`SummaryBuilder` uses Florence-2 for visual captioning when `use_model=True`.
It captions the representative frame of an episode rather than every frame.

Default task prompts in the code:
- `"<CAPTION>"` for general frame captioning
- `"<DETAILED_CAPTION>"` for episode-level representative-frame captioning

## Event Summary Fusion
**`Qwen2.5-1.5B-Instruct`** (1.5B params)

This model is also optional.
`SummaryBuilder` uses Qwen only when `use_llm=True`, and only for fusing a list
of episode summaries into one concise event summary sentence.

It is not used as the main query-answering model in the current codebase.

## What Is Not Implemented

There is currently no separate reasoning or QA model wired into the end-to-end
pipeline. `ReasonerInputFormatter` only formats retrieved evidence into a
readable block and a structured dict showing what would be passed to a full
multimodal or text-only reasoning model later.
