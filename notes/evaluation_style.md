# Evaluation Style

The evaluation should stay lightweight and practical. The main goal is to show
whether the memory system retrieves better long-range evidence than simpler
alternatives under a fixed memory budget.

The evaluation will use three components:

- A small set of qualitative retrieval examples that show what the system
  retrieves for representative queries.
- A few hand-checked retrieval sanity tests to verify that obvious queries map
  to the expected parts of the video.
- Baseline comparisons, first against a recent-window-only system, and, if
  feasible, against a visual-only retrieval variant without hierarchical memory.

This keeps the evaluation interpretable, easy to present in a notebook, and
focused on the real question: whether hierarchical memory adds value beyond
simple recency or flat retrieval.
