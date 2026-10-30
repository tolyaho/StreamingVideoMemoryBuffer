# SVMB showcase — run plan (written before inference)

Written 2026-10-30T09:56:05Z. Nothing below is changed after seeing results; any deviation is logged at the end of this file.

- Stimulus: ../stimulus/bbb-excerpt-0033-0459.webm (sha256 3435542701c6…4712), Big Buck Bunny film 0:33–4:59, CC BY 3.0.
- Questions and ground truth: ../stimulus/qas.json + annotation.json, frozen in ../stimulus/FROZEN_BEFORE_INFERENCE.sha256.
- Featured question for the walkthrough (fixed now): bbb_q3_butterfly. Secondary long-range question: bbb_q2_apple.
- Pipeline: repository code at release HEAD 513345e plus the uncommitted, passive demo observer (demo/trace.py, scripts/eval_common.py hooks).
- Models (text pipeline, as in the published text runs): X-CLIP base/32, Florence-2-base, Moondream2, Qwen2.5-VL-3B-Instruct (events), Qwen2.5-3B-Instruct (reasoner). Revisions in model-pins.json; the capture refuses to run if the local cache resolves elsewhere.
- Device: Apple M4 Pro, MPS. The published runs used CUDA; numerics can differ.
- Run A: --config notebook (unchanged benchmark preset).
- Run B: --config short_clip_demo = notebook preset with episodic_capacity 8 instead of 100, so that consolidation can occur within 266 s. Demo-only.
- Both runs are reported whatever the outcome. If a run fails, the failure is reported, not hidden.

## Deviation log

- 2026-10-30T10:18:24Z — **Featured question changed from bbb_q3_butterfly to bbb_q2_apple (Run A, notebook preset)**, after seeing results. Reason: Q3 is wrong for both systems in Run A; in Run B the hierarchical output is scored "D" (= butterfly) but its words say "D. A green leaf." (option A) — a letter-parse artifact, not treated as a success. Q2 in Run A is the only long-range question where the hierarchical answer is right in letter and text and the baseline is wrong; the apple description is in the hierarchical input (9 lines) and absent from the baseline input (0). All 8 outcomes (2 runs × 4 questions) are reported unchanged. The trace files are not edited; the tour question is a server presentation option.
- 2026-10-30T10:24:57Z — **Both runs repeated with a recorder fix** (episodes flushed and consolidated within a single update() were not recorded as nodes; the observer now records each episode at flush time). Same stimulus, questions, pins, configs and pipeline code; only demo/trace.py changed. The first traces are kept as trace-*.first-attempt.json and compared with the repeats.
- 2026-10-30T10:35:10Z — Repeat of Run A vs first attempt (compare-notebook.txt): identical window captions (89/89), promote/discard decisions, memory statistics and all answer letters; 17/24 Moondream2 episode texts and therefore some retrieval scores differ. Cause: Moondream2 `query()` samples by default (temperature 0.5, top_p 0.3 in the pinned revision) and src/summary_builder.py does not pass settings. Pipeline left unchanged; recommendation recorded.
- 2026-10-30T10:42:22Z — Repeat of Run B vs first attempt (compare-short_clip_demo.txt): identical window captions, decisions and memory statistics; Q3 hierarchical changed from "D. A green leaf." (letter D scored correct, words name option A) to "B. A small bird."; all other letters unchanged. Final traces (used by the viewer and the video): trace-notebook.json (SVMB 2/4, baseline 1/4) and trace-short_clip_demo.json (SVMB 1/4, baseline 1/4). First attempts kept for the record.
