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

## Per-window captioner
**`microsoft/Florence-2-base`** (~0.23B params)

Optional (`use_model=True`). Florence-2 runs **once per window at ingest** with the
`<CAPTION>` task token on the window's representative frame, and the resulting short
text is stored as `WindowEntry.summary_text`. At this volume (≈ one call per sampled
window — 63 calls for a 3-minute, 1-fps video) the small model is the right cost point:
fast, cheap, and the captions are short enough that their limited detail is not a
liability.

Florence is deliberately **not** used at the episode tier any more. Its
`<DETAILED_CAPTION>` task produces multi-sentence narrations that confidently fabricate
named entities from misread overlays (team names, player names, scores, seasons,
leagues, logos). Those hallucinations then compound across rep frames and get
laundered upward by the event-fusion VLM. The fix was to move the episode-tier job to
a model whose training encourages visual grounding over narrative speculation.

## Episode captioner
**`vikhyatk/moondream2`** (~1.87B params, single-image VLM)

Optional (`use_moondream=True`). Moondream2 runs **once per episode member window** at
flush time on each stored representative frame, and the resulting grounded captions are
concatenated in temporal order to form `EpisodeEntry.summary_text`. Call volume is
low (typically 10–40 per video), so the model's extra per-call cost is negligible.

Moondream is queried through its chat-style API with an explicit anti-hallucination
prompt. The prompt tells the model that the input is a **single still frame extracted
from a continuous video**, so motion may be frozen mid-action (running player, ball
in flight, gesture, vehicle in motion), and the caption should describe the apparent
motion or pose evident from the pixels (body position, blur, trajectory) — while
**never** naming specific teams, leagues, players, matches, scores, seasons, brands,
or on-screen text unless fully legible, and **never** speculating about what happened
before or after the frame. When identities or text are unclear, Moondream is steered
to generic descriptors (e.g. "a player in a red-and-blue striped jersey", "a
scoreboard with unreadable text").

The loader tries `model.query()` first (newest revisions), then falls back to
`model.caption(length="normal")`, then to the legacy `encode_image + answer_question`
API, so the same code survives Moondream revision updates. All three branches are
wrapped in try/except with a one-shot warning — if every API fails (e.g. PyTorch <
2.5 missing `enable_gqa`), the pipeline cleanly falls back to Florence's
`<DETAILED_CAPTION>` instead of crashing.

Each caption is stamped with a ground-truth temporal tag at episode-assembly time:
`[t={start_time:.1f}s | frame i/N] {moondream_caption}`. The timestamp and index
come directly from `WindowEntry.start_time` — never from the model — so Qwen2.5-VL
sees reliable temporal anchors when it later fuses captions with sampled frames.

Moondream is a **single-image** model: no multi-image input, no video/temporal
modelling. That is fine — temporal fusion across multiple frames is handled one tier
up by the event VLM.

## Event summary fusion
**`Qwen/Qwen2.5-VL-3B-Instruct`** (~3B params, multimodal; `Qwen2-VL-*` and larger
`Qwen2.5-VL-*` variants also supported via the same loader)

Optional (`use_vlm=True`). When episodic memory is consolidated, Qwen2.5-VL receives
the concatenated **episode summary strings** plus up to **10 representative frames
sampled evenly across the event** (chosen from each episode's top-2 windows by
cosine similarity to the episode centroid) and returns one detailed event summary.

The loader auto-detects the model family from the HF id (`qwen2-vl` vs `qwen2.5-vl`)
and picks the matching class (`Qwen2VLForConditionalGeneration` vs
`Qwen2_5_VLForConditionalGeneration`). dtype is forced to **bf16** on CUDA/MPS/CPU
via `_vlm_dtype_for_device` — fp16 overflows in Qwen's vision tower and the LM then
generates repeated token 0 (rendered as `!!!!!…`). The processor is loaded with
`min_pixels=64·28²` and `max_pixels=320·28²` to cap visual-token budget to ~64–320
patches per frame (default max_pixels ≈ 12.8 M produces up to ~16 k patches and
blows up eager attention). CUDA uses `attn_implementation="sdpa"` for the same
reason.

Generation: greedy (`do_sample=False`, `num_beams=1`), `max_new_tokens=640`,
`repetition_penalty=1.15`, and `no_repeat_ngram_size=6` — required because small
instruction-tuned VLMs (2B–3B) fall into a "the Nth scene shows…" template loop on
repetitive multi-image inputs under plain greedy decoding, producing several
hundred characters of near-identical clauses before the token budget is exhausted.
The 6-gram no-repeat constraint breaks that pattern at the phrase level rather
than the token level.

The fusion prompt treats the images as **authoritative** and the scene descriptions
as **noisy hints** explicitly known to hallucinate named entities. The VLM is asked
to omit any team/player/score/date/logo claim from the captions that it cannot
verify from the frames. Word-count target adapts to scene count (40–80 / 80–140 /
120–220) to prevent padding when content is thin.

At 2B the model still exhibited weak temporal / arithmetic / causal reasoning
over multi-image timelines (e.g. asserting "scoring twice, leading to 0-0" from
captions that read 0-1 → 1-1). The upgrade to **Qwen2.5-VL-3B** is specifically
to improve multi-constraint instruction following and cross-frame consistency at
the event tier, where the model must reconcile 5–17 Moondream captions with up
to 10 sampled frames and produce one internally-coherent paragraph.

## Summary gradient design principle

The lower the memory tier, the more the representation relies on raw visual
information. The higher the compression tier, the more it relies on text:

| Tier       | Primary signal                                     | Text role                          |
|------------|----------------------------------------------------|------------------------------------|
| Recent     | X-CLIP visual embedding                            | none                               |
| Episodic   | X-CLIP self-centrality pool + Moondream captions   | grounded descriptive text layer    |
| Long-term  | X-CLIP centroid + Qwen2.5-VL event summary         | primary semantic handle            |

At the long-term tier, the Qwen2.5-VL event summary is the main semantic handle for
coarse retrieval routing; the centroid embedding backs up visual similarity search.

## Captioning cost split — why three models

| tier     | model               | typical call count | cost justification                                                         |
|----------|---------------------|--------------------|----------------------------------------------------------------------------|
| window   | Florence-2-base     | ≈ N_windows        | high volume → cheapest model wins                                          |
| episode  | Moondream2          | ≈ N_episodes × 2   | low volume + hallucination-critical → mid-size grounded model              |
| event    | Qwen2.5-VL-3B-Instruct | ≈ N_events      | rare + multi-image temporal fusion → a proper multi-image VLM is required  |

## Persistence layer

**`peewee` over SQLite.** `MemoryStore` (see `src/memory_db.py`) is a thin facade the
writer calls through. Windows, episodes, and events are persisted as they are
produced; membership (episode ↔ windows, event ↔ episodes, event ↔ representative
windows) is stored in join tables with `ON DELETE CASCADE` on the parent side.
Embeddings are stored as raw `float32` blobs with an `embedding_dim` column;
representative frames are stored as JPEG blobs (quality 90) when present. SQLite is
opened with WAL journaling and foreign keys enabled. The hook is optional — pass
`store=None` to `HierarchicalMemoryWriter` for a pure in-memory run.

## Observed failure modes at the event tier

With Qwen2.5-VL-3B the pathological 2B behaviours are gone (repetition loops,
self-contradicting arithmetic like "scoring twice, leading to 0-0"), but a residual
class of errors persists and is well-understood:

- **Score-side attribution is a coin flip.** The scoreboard reads as `X-Y` with no
  visible home/away label; the model picks a side and commits. Captions of the form
  "scoreboard indicates 0-1" then "1-1" are read as an equaliser *by some team*, and
  Qwen2.5-VL may assign that goal to the wrong side. This is a data-availability
  problem, not a model-capacity problem.
- **World-knowledge grafting.** Once a team name or league logo is visible, the
  model's pretrained knowledge about that fixture leaks in as if observed (stadium
  name, competition, typical scorelines). 3B is better than 2B here but not
  bullet-proof.
- **Brand leak despite anti-brand prompt.** Brand names visible in the frame (e.g.
  a sponsor logo on a cooler) are occasionally named even when the prompt forbids
  it. Small instruction-tuned VLMs obey *negative* constraints less reliably than
  positive ones.
- **Cross-event consistency is not enforced.** Each event is summarised in
  isolation. The closing score of event N is not piped forward as context for
  event N+1, so adjacent events can disagree on the running scoreline.

These would be addressed by (a) pre-extracting scoreboard patterns from the
episode captions and injecting them as a hard-constrained list in the fusion
prompt, (b) threading the previous event's summary into the next fusion call as
carry-over context, and (c) a post-filter stripping flagged entity classes.
Those are architectural changes, not model changes.

## What Is Not Implemented

There is no separate reasoning or QA model wired into the end-to-end pipeline.
`ReasonerInputFormatter` only formats retrieved evidence into a readable block and a
structured dict showing what would be passed to a full multimodal or text-only reasoning
model later.
