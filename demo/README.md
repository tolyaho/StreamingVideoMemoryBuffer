# Streaming Video Memory — recorded-run viewer

A local browser app that replays a **recorded run of the actual SVMB pipeline**: the source video, every 3-second window the system processed, what its memory writer kept or dropped, the episodes and long-term events it built, what the retriever returned for each question (with scores), and both answers. The browser makes no model calls. Speeds, pauses, the jump back to the evidence frame and the guided tour are presentation choices, not inference time.

It needs only the Python 3.11+ standard library. It makes no network requests (no CDN) and uses system fonts.

## Two modes

**Recorded trace (the showcase).** These are traces captured by `demo/capture.py` while the real pipeline ran, served with the exact video they came from and its rights record. The recorded *Big Buck Bunny* runs ship in [`showcase/`](showcase/README.md). The 43 MB clip is a release asset.

```bash
python3 demo/fetch_showcase_media.py   # downloads it and checks its SHA-256
python3 demo/server.py --showcase      # both runs, guided tour on the apple question
```

Open <http://127.0.0.1:8765>. `--showcase` is shorthand for:

```bash
python3 demo/server.py --trace demo/showcase/trace-notebook.json --trace demo/showcase/trace-short_clip_demo.json \
  --media demo/showcase/media/bbb-excerpt-0033-0459.webm --rights demo/showcase/rights.json \
  --annotation demo/showcase/annotation.json --tour-question bbb_q2_apple
```

The server refuses a video whose SHA-256 differs from the one recorded in the trace and in the rights record, so the footage on screen is always the footage that was processed. You can serve several `--trace` files for the same source; the header switches between them.

**Saved evaluation outputs (no media, no trace).** `python3 demo/server.py` with no arguments replays StreamingBench sample 36 from `results/streamingbench_text_3b_notebook`. That evaluation did not save memory states or retrieval hits, and the footage is not cleared for redistribution, so those panels stay empty and say so.

## What is on screen

- **Video** — the bound source file. The caption under it is the actual Florence-2 caption of the newest processed window.
- **Ribbon** — one cell per window, coloured by where the recorded state puts it at the current time: recent (silver), kept in an episode (teal), consolidated into a long-term event (blue), discarded (dark). Bars below are the recorded episodes and events. Diamonds are question times. After a question, amber arcs go to the nodes the retriever actually returned; a thin green bar marks the human-annotated evidence. Nothing after the playhead is drawn.
- **Memory** — the actual recent thumbnails, episode cards (Moondream2 text) and event cards (Qwen2.5-VL text), plus the writer's last promote/discard decision with its similarity.
- **Ask** — the question, the returned events/episodes with their recorded scores, and both saved answers verbatim. The scorer reads only the first letter A–D; if the words after the letter spell out a different option, the card says so instead of counting it as a success.
- **Provenance** — models and revisions, device, settings, wall-clock, licence, and SHA-256 of every input.

Click a retrieved row to show that moment in the video; the memory panels stay at the question time.

## Capture a trace

Needs the repository's model environment (torch, transformers, OpenCV) and the five models: X-CLIP base/32, Florence-2-base, Moondream2 (and its tokenizer repo `moondream/starmie-v1`), Qwen2.5-VL-3B-Instruct, Qwen2.5-3B-Instruct.

```bash
HF_HUB_OFFLINE=1 python demo/capture.py --video clip.webm --qas qas.json \
  --config notebook --reasoner-type text --rights rights.json \
  --focus-question <question_id> --pins model-pins.json --output demo/local/run.json
```

`--pins` refuses to run if the local Hugging Face cache resolves a model to a different commit. `--config short_clip_demo` is the notebook preset with `episodic_capacity=8` instead of 100, so that long-term consolidation can happen inside a few-minute clip; it is a demo setting, not a benchmark one, and the viewer labels it.

The observer (`demo/trace.py`, hooked into `scripts/eval_common.run_video`) only reads what the pipeline already produced. It wraps two methods of the one writer instance — the novelty test and the episode flush — calling the originals and passing their results through, to log each promote/discard decision and every episode (including ones consolidated within the same update). It never calls a captioner or a model. With `observer=None` the evaluation behaves exactly as before; `demo/tests` checks answers, memory statistics and caption calls with and without it.

A rights record documents real permission for the exact file:

```json
{"source_sha256": "…", "redistribution": "cleared", "license": "CC BY 3.0",
 "attribution": "…", "basis": "…", "source_url": "…", "changes": "…", "source_start_seconds": 0}
```

## Record the tour, poster and checks

You need Node 18+, Playwright (run `npm ci` in `demo/`, then `npx playwright install chromium`) and FFmpeg.

```bash
S=demo/showcase
node demo/record.cjs --showcase --out /abs/out --python python3 \
  --trace $S/trace-notebook.json --trace $S/trace-short_clip_demo.json --media $S/media/bbb-excerpt-0033-0459.webm \
  --rights $S/rights.json --annotation $S/annotation.json --tour-question bbb_q2_apple
```

It first runs browser checks on every served run (question never visible early, answers verbatim, no future windows on the ribbon, backward seek, evidence jump keeps the memory clock, provenance dialog, no horizontal overflow at 390/768/1280/1920 px, no console errors, no external requests). It then renders the tour **deterministically**: for each output frame it calls `window.demo.renderTourFrame(t)`, waits until the `<video>` element has decoded the requested source time, takes a screenshot, and encodes the frames to 1920×1080 H.264 at 30 fps. `--check-only` skips the capture.

## Tests

```bash
python -m unittest discover -s tests
python -m unittest discover -s demo/tests
node --test demo/tests/timeline.test.mjs
ruff check --no-cache .
```

Footage: [*Big Buck Bunny*](https://peach.blender.org/) © 2008 Blender Foundation / www.bigbuckbunny.org, [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/). This is an edited excerpt (film 0:33–4:59, audio removed, re-encoded). It is not endorsed by the Blender Foundation.
