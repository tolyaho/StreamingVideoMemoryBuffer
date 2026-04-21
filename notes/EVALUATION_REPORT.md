# Retrieval evaluation — sample_36 (Gordon-Ramsay-style cooking video, ~17 min, 5 QAs)

> **Post-eval changes (applied after this run):**
> 1. Stage B now unions the coarse-gated candidate set with the trailing
>    `recent_episodes=5` episodes so tail episodes not yet consolidated into any
>    event remain searchable.
> 2. `flush_pending()` no longer drains `self.episodic` into events at every QA —
>    that over-drain was making stage-B fine search structurally empty.
> 3. **Scoring switched from additive to multiplicative temporal prior:**
>    `α·vis + β·txt + γ·bonus` → `(α·vis + β·txt) · exp(-dt/τ)`, with
>    α=0.70, β=0.30, τ = 0.50 · stream_span. Bumping γ alone (0.05 → 0.15 →
>    0.25) could not de-throne the intro event (§1) because additive recency
>    caps at γ regardless of how far a distant "centroid" entry is; the
>    multiplicative form scales semantic score by the decay, so a far entry is
>    discounted in proportion to its semantic strength. γ is removed.
>
> The retrievals.md / memory.db below predate all three changes; re-run
> `scripts/main.py` to see the effect. See ARCHITECTURE §15 and §16 items 11–13.

Source data:

- `outputs/memory.db` — 343 windows · 110 episodes · **32 events** (last window end = 1026.3 s)
- `outputs/retrievals.md` — 5 QA firings at stream t = 203, 209, 404, 602, 800 s
- `data/.../sample_36/qas.json` — ground-truth answers B, D, A, A, C

Heads-up that the retrievals.md / memory.db in the repo are from an **older run against sample_36**, but `scripts/main.py` now points at **sample_1** (a soccer clip). Re-running `main.py` would overwrite the current outputs; the evaluation below is on what's actually on disk.

---

## Per-QA verdict

Legend:
- ✅ right timestamp retrieved *and* caption supports the correct option
- 🟡 right timestamp retrieved but caption does not resolve the answer
- ❌ wrong timestamp / irrelevant evidence

| # | QA / time | GT | Coarse top-3 | Window grounding | Verdict |
|---|-----------|----|--------------|------------------|---------|
| 1 | cupboard **color** @ 202 s | B. Light green | [0–68], [107–143], [80–83] — generic kitchens | [200–203] "grating eggs in kitchen"… | 🟡 right time, **all captions say "white" or "dark" — never "light green"** |
| 2 | box grater action @ 207 s | D. yellow ingredient | [0–68], [143–149], [92–101] | [200–203] "grating eggs", [203–206] "grating cheese", [197–200] "grating an orange" | 🟡 right time, captions pick random foods; the **one event that does say "a bright yellow ingredient"** ([188–224 s]) is not in coarse top-3 |
| 3 | blue shirt chopping @ 402 s | A. Red bell pepper | [287–344], [0–68], [224–287] | **[398–401] "cutting up a red pepper on a cutting board"** | ✅ exact text match in grounding |
| 4 | holding what @ 602 s | A. Wooden spoon | **[527–542] "uses a wooden spoon to stir a vibrant, golden mixture"**, [0–68], [92–101] | [599–602] "preparing food" (generic) | ✅ exact text match in coarse event summary |
| 5 | right hand @ 799 s | C. A pan | [689–740], [0–68], [107–143] | **[797–800] "A man holding a frying pan in a kitchen"** | ✅ exact text match in grounding |

**Summary**: retrieval lands on the correct time window in **5/5** QAs; the retrieved evidence contains a caption that *verbatim* answers the question in **3/5** QAs (QA3, QA4, QA5). The two "🟡" failures are caption-quality failures, not retrieval failures.

---

## Is retrieval "good"? — short answer

**Yes, directionally.** The three-tier design is doing its job: recent-window grounding catches "right now" questions (QA3, QA5), coarse events catch "earlier action" questions (QA4), and the time windows surfaced are always near the ground-truth moment. The stretch goal of beating a recent-window baseline is plausible — QA4 is the clear example (answer lives at t=527–542 s, a full minute before the query).

But **the coarse tier is noisy** and **the captions are the real bottleneck** — below.

---

## Systemic problems (ranked by impact)

### 1. The intro event [0–68 s] wins every coarse top-3

It appears in **all 5** top-3 lists with sim 0.205–0.245. Root cause: it's simultaneously

- the **longest event** (68 s vs typical 3–42 s, one massive outlier),
- the **longest summary** (~1.4k chars — exponentially more surface area for query-keyword overlap),
- a **mean-pooled kitchen centroid** (visual embedding looks like generic-kitchen, matches every cooking query).

This is a **length bias** problem: X-CLIP's text encoder, designed for short captions, scores long summaries higher on `<kitchen|cooking|...>`-type queries simply because more tokens hit the query.

### 2. Event duration is extremely skewed

Distribution of event durations on disk: nine 3-s singletons, several in the 6–15 s band, a handful in the 27–66 s band, and **two giants at 135 s and 147 s**. The giants hog coarse budget; the 3-s singletons waste it. The average hides this — median is OK (~15 s) but the tail is pathological.

Caused by `episodic_capacity=5` + `episodic_merge_batch=10`: when 5 episodes accumulate, consolidation fires and greedily agglomerates *up to* 10 contiguous episodes with `sim ≥ 0.55` and `gap ≤ 15 s`. Kitchen scenes easily hit both criteria continuously, so events balloon.

### 3. Named-entity hallucinations leak through the anti-hallucination prompt

- Event [0–68 s]: *"Chef Gordon Ramsay, introducing the series 'Ramsay in Ten'…"* — the explicit "do NOT name specific people, brands, on-screen text unless confidently readable" instruction to the VLM is being ignored.
- Window [797–800 s] (Moondream): *"A still frame shows a person, likely Gordon Ramsay…"* — same story at the episode tier.

Small instruction-tuned VLMs (3 B and under) honour **positive** instructions far better than **negative** ones. Known residual, documented in ARCHITECTURE §15, but it still poisons the event text used for coarse routing.

### 4. Caption models disagree with each other on obvious visuals

Around QA1 (cupboard colour):

- Moondream window [194–197 s]: *"dark cabinets, a marble countertop"*
- Moondream window [200–206 s]: *"dark cabinets, a white tiled backsplash"*
- Moondream window [197–200 s]: *"white cabinets, a marble countertop"*
- Qwen event [107–143 s]: *"white cabinetry"*

Ground truth is **light green**. Every caption is wrong, and they don't even agree with each other within-episode. This is a captioner-capacity problem — the cabinets probably look teal/sage under the given lighting and small VLMs snap to the nearest colour prior (white for light wood, dark for cabinetry shadows). Retrieval cannot rescue this; only a stronger VLM (or OCR-style colour sampling) would.

### 5. Per-window captions invent food identities

*"grating eggs"*, *"grating cherries"*, *"grating chicken"*, *"grating an orange"* around the same scene where the event VLM eventually says *"a bright yellow ingredient"*. The windows see one frozen frame and the model gambles. This is exactly the hallucination class that pushed episode captions from Florence → Moondream; Florence at the window tier still has it.

### 6. Query-time retrieval has a weak "right now" signal — resolved

Scoring weights *at the time of this run* were `α=0.65 · visual + β=0.30 · summary + γ=0.05 · recency_bonus`, and `query_time` was not passed into `retrieve()`. For "right now" questions, proximity to `stream_time` should carry more weight than 5%. **Post-run fixes**: (a) `query_time=stream_time` is now threaded through from `scripts/main.py`; (b) after a failed intermediate attempt to bump γ (0.05 → 0.15 → 0.25) that couldn't overcome the intro event's semantic lead because additive γ caps at γ, the scoring form itself was switched to a multiplicative prior — see EVALUATION_REPORT header and ARCHITECTURE §16 item 13. Current scoring: `(α·vis + β·txt) · exp(-dt/τ)` with α=0.70, β=0.30, τ = 0.50 · stream_span. A re-run is needed to measure the effect on the existing retrievals.md.

### 7. Coarse routing force-injects the latest event, sometimes harmfully

`_coarse_route` overwrites `top[-1]` with the latest-ending event even when its score is outside the natural top-M. For "right now" queries this is a useful safety net; for "earlier action" queries it consumes a slot with a likely-irrelevant recent event. Cleaner rule: inject only if the latest event intersects `query_time` within some window.

### 8. `flush_pending()` over-drained the episodic tier — fine search was structurally empty (FIXED)

Root cause of the missing `Fine — episodic memory hits` section in every retrieval: the previous `flush_pending()` implementation also ran `while self.episodic: self._consolidate_episodic()`, force-draining every completed episode into `long_term` at every single QA. Stage B (fine search) then saw `self.episodic == []` and the formatter's conditional `if result.episodic_hits` branch printed nothing. The coarse tier looked plausible because drained episodes reappeared as events, which is why the failure mode masqueraded as "just a captioner problem".

Separately, event `[188–224 s]` — *"grates a bright yellow ingredient onto a plate"* — still consolidated at the natural gap break at t≈224 s rather than at QA2's fire time of 209 s, because by then `_pending_episode` was empty (the last novelty break had already promoted it to `self.episodic`). That is fine under the new semantics: the episode was queryable at stage B via the recent-episodes bypass regardless of event formation.

**Fix applied**: drop the `while self.episodic` loop from `flush_pending`. New contract: flush only promotes the in-progress episode to `self.episodic`; the recent-episodes bypass in `_fine_search` guarantees stage-B visibility of tail content without coarse-tier consolidation. Regression test: `tests/test_memory_writer.py::FlushPendingSemanticsTests`.

### 9. Repo / config drift

- `scripts/main.py` points at `sample_1` (soccer), but `retrievals.md` and `memory.db` are from `sample_36` (cooking).
- `ARCHITECTURE.md §15/§16` extensively discusses soccer-video residuals (team names, PRIME cooler) that don't match the current on-disk evidence.
- `outputs/memory.db` isn't cleared between runs (entry_ids are fresh UUIDs, so REPLACE never fires → rerun doubles rows). §16 item 9 documents this but it's still unfixed.

### 10. No automated QA scoring

Evaluation is eyeball-only. The five ground-truth options are in `qas.json`; a tiny scorer could check whether any of the retrieved grounding windows' captions lexically contain the correct option's keyword. Would turn this eval from anecdotal to reproducible.

---

## Concrete suggestions (by tractability)

### Quick wins — 10 minutes each

1. **Cap event duration**. Add `event_max_duration` (e.g. 45 s) to the writer: force-consolidate when the *current* in-progress event would exceed it. Kills the 135/147-s giants and the [0–68 s] intro dominance.
2. **Filter 3-s singleton events from coarse top-M**. They are almost always noise. Easier: require `n_member_episodes ≥ 2`.
3. **Pass `query_time` in `_process_due_qas`**. One extra kwarg on `retriever.retrieve(..., query_time=stream_time)`; the retriever already handles it — just nobody passes it.
4. **Clear `outputs/memory.db` at the start of `main.py`** (or make entry_ids deterministic). One `DB_PATH.unlink(missing_ok=True)` line before `MemoryStore(...)`.
5. **Point main.py at the sample the notes reference**, or at least add a comment saying which sample the current `outputs/` was generated from.

### One-hour fixes

6. **Length-normalise the summary-similarity term**. Truncate every event summary to the first N tokens (say 128) before calling `text_encode_fn`, so [0–68 s] can't win by sheer verbosity. Or divide the raw similarity by `log(1 + n_chars)`.
7. **Entity post-filter on event summaries**. After VLM fusion, regex-strip proper nouns that don't appear in the concatenated Moondream captions (Moondream is already instructed not to name entities, so this is roughly "names the VLM hallucinated"). Tokens to scrub: "Gordon Ramsay", brand names, league names — the set from ARCHITECTURE §15.
8. **Temporal prior is now multiplicative (`exp(-dt/τ)`)** — applied. If "earlier in the video" queries surface later and start under-ranking the content they want, the right knob is `tau_fraction` (default 0.25) — raise to ~0.4 for a gentler decay, or detect temporal cue words (`{earlier, at the start, first, when ... started}`) in the query and dynamically widen `tau_fraction` for those calls.

### Bigger changes — worth the half-day

9. **Swap the summary text encoder**. X-CLIP's text tower is aligned to short video captions, not paragraphs. Use a small sentence encoder (BGE-small, E5-small) for summary embeddings; keep X-CLIP for visual. You already have separate slots in the scoring blend.
10. **Upgrade the window captioner**. Florence-2 at the window tier invents "grating eggs" / "grating cherries". Replacing with Moondream (already loaded) at the window tier would cost ~3× per window but kill the most damaging hallucinations for queries that rely on window captions (QA2, QA3, QA5). Alternative: run Florence *and* Moondream, take the intersection of named objects.
11. **Automated QA scorer**. For each QA, check whether the correct option's noun phrase appears in any retrieved caption. Hit count gives a P@k number and a recall-at-evidence metric.

### Architectural — future work

12. **Two-stage retriever**: visual top-K first (narrow by embedding), then text re-rank within. Avoids the length-bias trap that happens when you blend in one pass.
13. **Scene-change detection** for episode flush (already in PLAN.md stretch goals). Would tame both the giants and the 3-s singletons simultaneously.
14. **Verify `flush_pending()` end-to-end** with an integration test that streams windows, fires a mid-episode QA, and asserts the tail episode is present in `self.episodic` after `flush_pending` (not consumed into `long_term`). Unit-level coverage of the over-drain bug is in `tests/test_memory_writer.py`; the integration smoke test is still pending.

---

## Bottom line

- The **storage and retrieval pipeline works**: 5/5 on locating the right time window, which is the assignment's actual question.
- The **coarse tier's top-3 is noisy** (same giant intro event every time), but it doesn't matter much in practice because (a) window grounding rescues "right now" queries and (b) the blended score still floats the right event occasionally (QA4).
- The **captioners are the current ceiling** — when the retrieved captions are right (QA3, QA4, QA5), the answer is verbatim in the evidence; when they're wrong (QA1 colour, QA2 food identity), no amount of retrieval tuning can recover them.

If you only do two things before the notebook deliverable: **cap event duration (#1)** and **pass `query_time` through (#3)**. Those are a few lines of code and directly address the two most visible failures in the current retrievals.md.
