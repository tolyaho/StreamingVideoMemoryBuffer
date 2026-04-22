# %% paths + QAs

from src import MemoryStore

video_key = "16Z-XQh9jhk"
video_path = root / f"data/lvbench/{video_key}/video.mp4"
qas_path = root / f"data/lvbench/{video_key}/qas.jsonl"
db_path = root / f"outputs/memory_{video_key}.db"
db_path.parent.mkdir(parents=True, exist_ok=True)
if db_path.exists():
    db_path.unlink()

qas = [json.loads(ln) for ln in qas_path.read_text().splitlines() if ln.strip()]
for qa in qas:
    qa["t_seconds"] = hms_to_seconds(qa["time_stamp"])
qas.sort(key=lambda q: q["t_seconds"])

print(f"video : {video_path.name}  ({qas[-1]['t_seconds']/60:.1f}+ min)")
print(f"qas   : {len(qas)}  ({qas[0]['time_stamp']} → {qas[-1]['time_stamp']})")
print(f"db    : {db_path}")

# %% memory + retrieval

store = MemoryStore(str(db_path))

mem = HierarchicalMemoryWriter(
    recent_capacity=RC, episodic_capacity=EC, novelty_threshold=NT,
    episode_max_gap=EG, event_max_gap=VG, summary_fn=summary,
    text_encode_fn=enc.encode_text, store=store,
)
retriever, formatter = HierarchicalRetriever(), ReasonerInputFormatter()

# %% per-QA answer loop

correct = 0


def answer_due(cursor, stream_time, qas, mem, retriever, formatter, reasoner):
    global correct
    while cursor < len(qas) and qas[cursor]["t_seconds"] <= stream_time:
        qa = qas[cursor]
        mem.flush_pending()
        q_emb = enc.encode_text(qa["question"])
        result = retriever.retrieve(
            query=qa["question"], query_embedding=q_emb, memory=mem,
            top_m=3, top_k=5, query_time=stream_time,
        )
        llm_input = formatter.format_for_llm(result, query_embedding=q_emb)
        prediction = reasoner.answer(llm_input, options=qa["options"])
        letter = next((c for c in prediction.strip() if c in "ABCD"), "?")
        correct += int(letter == qa["answer"])

        rule("─")
        print(f"  QA {cursor + 1:>2}/{len(qas)}  ·  t={stream_time:.1f}s  ·  uid={qa['uid']}")
        print(f"  Q: {qa['question']}")
        print(f"  prediction  : {prediction}")
        print("  ground truth: " + next(
            (o for o in qa["options"] if o.startswith(qa["answer"] + ".")), qa["answer"]
        ))
        cursor += 1
    return cursor


# %% streaming loop (long — expect a few hours on CUDA)

cursor = 0
for raw in reader.read_windows(str(video_path)):
    mem.update(
        WindowEntry.from_raw_window(
            raw, visual_embedding=enc.encode_window(raw),
            summary_text=summary.build_window_caption(raw),
        )
    )
    cursor = answer_due(cursor, raw.end_time, qas, mem, retriever, formatter, reasoner)

# %% final score

rule("═")
print(f"  MCQ accuracy : {correct}/{len(qas)}  ({100.0 * correct / len(qas):.1f}%)")
print(f"  db persisted : {db_path}  ({mem.stats()})")
rule("═")

store.close()
