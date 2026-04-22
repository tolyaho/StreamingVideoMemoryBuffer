from src import RecentWindowBaseline, RetrievalResult
from src.memory_writer import cosine_sim

video36 = root / "data/streamingbench/Real_Time_Visual_Understanding/shard_1_50/sample_36/video.mp4"
qas = json.loads((video36.parent / "qas.json").read_text())["qas"]
for qa in qas:
    qa["t_seconds"] = hms_to_seconds(qa["time_stamp"])
qas.sort(key=lambda q: q["t_seconds"])

baseline = RecentWindowBaseline(n_windows=RC)
formatter = ReasonerInputFormatter()

correct_bl = 0


def answer_due_baseline(cursor, stream_time, qas, baseline, formatter, reasoner):
    global correct_bl
    while cursor < len(qas) and qas[cursor]["t_seconds"] <= stream_time:
        qa = qas[cursor]
        q_emb = enc.encode_text(qa["question"])
        hits = baseline.retrieve(q_emb, top_k=5)
        scores = {w.entry_id: float(cosine_sim(q_emb, w.visual_embedding)) for w in hits}
        result = RetrievalResult(
            query=qa["question"], coarse_hits=[], episodic_hits=[],
            grounded_windows=hits, scores=scores,
        )
        llm_input = formatter.format_for_llm(result, query_embedding=q_emb)
        prediction = reasoner.answer(llm_input, options=qa["options"])
        letter = next((c for c in prediction.strip() if c in "ABCD"), "?")
        correct_bl += int(letter == qa["answer"])

        rule("─")
        print(f"  QA {cursor + 1}  ·  t={stream_time:.1f}s  ·  Q: {qa['question']}")
        print(f"  prediction  : {prediction}")
        print("  ground truth: " + next(
            (o for o in qa["options"] if o.startswith(qa["answer"] + ".")), qa["answer"]
        ))
        cursor += 1
    return cursor


cursor = 0
for raw in reader.read_windows(str(video36)):
    baseline.update(
        WindowEntry.from_raw_window(
            raw, visual_embedding=enc.encode_window(raw), summary_text=summary.build_window_caption(raw)
        )
    )
    cursor = answer_due_baseline(cursor, raw.end_time, qas, baseline, formatter, reasoner)

rule("═")
print(f"  MCQ accuracy (baseline): {correct_bl}/{len(qas)}")
rule("═")
