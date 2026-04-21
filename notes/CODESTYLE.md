## Very important style requirements

### Write code like a human researcher
Do **not** write bloated, over-defensive, tutorial-like, or “AI-ish” code.

Avoid:
- verbose docstrings everywhere,
- obvious comments for trivial lines,
- repetitive helper wrappers that add no value,
- excessive abstractions,
- giant config classes for no reason,
- unnecessary OOP,
- noisy logging,
- fake “production-grade” structure for a notebook assignment.

Prefer:
- clean, direct code,
- short meaningful functions,
- minimal but useful comments,
- readable variable names,
- compact implementations,
- natural notebook flow,
- code that feels like it was written by someone doing serious ML research.

### Comments
Use **only necessary comments**.

Good comments:
- explain a non-obvious design choice,
- clarify a tricky step,
- mark an important trade-off.

Bad comments:
- restate what the code literally does,
- explain basic Python,
- narrate every line.

### Clean up aggressively
Whenever you write or refactor code:
- remove redundant variables,
- remove dead code,
- merge duplicated logic,
- simplify control flow,
- reformat for clarity,
- keep only what materially helps readability.

Do not leave messy exploratory garbage in final code.