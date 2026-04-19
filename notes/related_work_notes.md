## What current systems actually do

Most current streaming video-understanding systems do **not** keep the full raw stream inside the active model context. Instead, they usually separate the pipeline into **perception**, **memory**, and **reasoning**, maintain some form of **compressed multimodal memory**, and answer queries using either **retrieved memory fragments** or a **compressed memory state**.

- **IXC2.5-OmniLive** is a clear example of a perception–memory–reasoning split.
- **FluxMem** focuses on hierarchical memory compression to reduce temporal and spatial redundancy.
- **VideoTree** uses query-adaptive coarse-to-fine retrieval over long videos.

---

## Why the recent-window baseline is important

A strong recent-window baseline matters because a very simple method — using only the last `N` frames with an off-the-shelf VLM and no memory, retrieval, compression, or extra training — already matches or outperforms many published streaming methods on **OVO-Bench** and **StreamingBench**.

**SimpleStream** shows that added memory complexity should not be assumed useful by default; it has to beat a strong recency baseline fairly.

---

## Why isolated frame captions are too weak

Isolated frame captions are too weak because they lose **temporal continuity**, **event structure**, and **query relevance**.

A single frame often cannot tell whether an action is:

- starting,
- continuing, or
- finishing.

Long videos also contain a lot of content that is either redundant or irrelevant to the current query.

That is why:

- methods like **VideoTree** retrieve information in a **coarse-to-fine** way over longer temporal structure instead of treating captions as flat independent notes, and
- memory-based systems **compress and organize context across time** rather than relying on one caption per frame.