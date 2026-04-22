## 7. Baseline comparison — does hierarchy actually help?

§6 ran the full hierarchical stack on the 5 sample_36 QAs. This section reruns exactly the same end-to-end loop, but with `RecentWindowBaseline` in place of `HierarchicalRetriever` — the SimpleStream-style [[1]](https://arxiv.org/abs/2604.02317) flat recent-frames buffer that most current streaming VQA systems use as their default, and a surprisingly strong one on short-horizon benchmarks.

Implemented in `baseline.py` as `RecentWindowBaseline`: a `deque(maxlen=N)` of `WindowEntry`, cosine-scored against `q_emb` at query time. No episodes, no events, no consolidation, no temporal decay — just the last N windows and nearest-neighbour search over them.

### What stays fixed

Everything downstream of retrieval. Same `ReasonerInputFormatter`, same `LLMReasoner` (Qwen2.5-3B-Instruct, greedy, same system prompt, same MCQ template), same 5 QAs at the same stream timestamps. The baseline's top-k windows are packed into a `RetrievalResult` with empty `coarse_hits` / `episodic_hits` and the windows as `grounded_windows`, so the `text_context` block the LLM sees has the same shape — just one tier instead of three.

Isolating the retriever is the whole point of §6's split: any delta in MCQ accuracy between §6 and §7 is attributable to the memory design, not the reasoner. The hypothesis is that the baseline is competitive when the answer lives in the last ~60 s of the stream and loses on QAs that reference earlier segments that have already rolled out of the deque.
