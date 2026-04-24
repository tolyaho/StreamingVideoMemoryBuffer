from __future__ import annotations

from typing import Optional, Sequence


MOONDREAM_FRAME_PROMPT = (
    "You are looking at a single still frame extracted from a continuous video, so "
    "some motion may be frozen mid-action. Describe this frame in 2–4 sentences "
    "focused only on what is clearly visible, including any apparent motion or pose "
    "evident from the pixels (body position, blur, trajectory). Do NOT name specific "
    "people, organizations, locations, events, brands, or on-screen text unless you "
    "can read or identify them confidently and completely. When identities or text "
    "are unclear, use neutral generic descriptions instead of guessing. Prefer "
    "concrete visual detail (colours, clothing, positions, objects, apparent motion) "
    "over narrative speculation about what happened before or after this frame."
)


REASONER_SYSTEM_PROMPT = (
    "You are answering a question about a video.\n"
    "Use the evidence below. Each line includes a time range.\n"
    "When you answer, mention the time ranges you relied on."
)


def build_reasoner_user_block(
    llm_input: dict,
    options: Optional[Sequence[str]],
) -> str:
    """Evidence + question (+ optional MCQ options) as one user-turn string."""
    head = f"Evidence:\n{llm_input['text_context']}\n\nQuestion: {llm_input['query']}"
    if not options:
        return head
    return (
        f"{head}\n"
        f"Options:\n" + "\n".join(options) + "\n"
        "Answer with the single letter (A, B, C, or D) of the best option, "
        "followed by a one-sentence justification that cites the time range(s) you used."
    )


def build_event_vlm_prompt(
    *,
    bulleted: str,
    n_frames: int,
    n_scenes: int,
    target: str,
) -> str:
    """Event-tier VLM fusion prompt — grounded, English-only, anti-hallucination."""
    return (
        "You are producing a detailed summary of a single continuous video event that is made up "
        "of several shorter scenes.\n\n"
        "## Output language\n"
        "Write the entire summary in **English only**. Do **not** switch to Chinese or any other "
        "language at any point, even if on-screen text, captions, or image content contain "
        "non-English words. If you need to refer to visible foreign text, describe it in English "
        "(e.g. 'a sign with Chinese characters') rather than reproducing it.\n\n"
        "## What you receive\n"
        f"- **{n_frames} still images**, supplied in chronological order. They are the most "
        "representative frames sampled from the scenes that make up this event (roughly 1–2 "
        "frames per scene, evenly spread in time).\n"
        f"- **{n_scenes} scene descriptions** below, one per scene, each prefixed with its time "
        "range in seconds from the start of the video. The scenes are listed in chronological "
        "order and together cover the whole event.\n"
        "- The images are the **authoritative evidence**. The scene descriptions come from a "
        "small auto-captioner and are known to **hallucinate named entities** — specific team "
        "names, player names, scores, dates, seasons, leagues, brand names, and on-screen text. "
        "Use the captions only for general shape, action, and setting. If a caption names a "
        "specific team, player, league, match, score, date, or logo that you cannot **clearly "
        "verify from the images**, do NOT repeat that claim; describe it generically instead "
        "(e.g. 'a football match', 'a player in a red-and-blue striped jersey', 'a scoreboard').\n\n"
        "## Your task\n"
        f"Write a **detailed, cohesive summary** of the event as prose ({target}, no bullet "
        "lists, no headings, no numbered steps).\n"
        "Make it **high-information-density**: keep the rich context, but pack each sentence with "
        "distinct visual facts instead of broad atmosphere or commentary.\n"
        "Put the most query-useful facts early: who/what is present, where the event happens, the "
        "main actions, visible objects, and any clear state changes over time.\n"
        "Then cover, in natural narrative order:\n"
        "1. **Setting & context** — where the event takes place and any reliable on-screen "
        "graphics or text.\n"
        "2. **Subjects & objects** — the main people, animals, vehicles, or objects that appear, "
        "including recognisable clothing, colours, or distinctive features when visible.\n"
        "3. **What happens over time** — the key actions and how they unfold; refer to earlier vs. "
        "later moments only when it clarifies the progression.\n"
        "4. **Transitions & continuity** — how the focus, location, or activity shifts between "
        "scenes, and what stays the same.\n"
        "5. **Overall takeaway** — one short closing sentence naming what the event is about as a "
        "whole.\n\n"
        "## Rules\n"
        "- Stay **strictly grounded**: describe only what the images and scene descriptions "
        "actually support. If something is ambiguous, say so briefly ('appears to', 'likely') "
        "rather than inventing a specific identity, name, place, or action.\n"
        "- Prefer concrete visual detail over generic phrasing.\n"
        "- Do **not** list the scenes mechanically ('Scene 1 shows..., Scene 2 shows...'); weave "
        "them into one flowing narrative.\n"
        "- Do **not** repeat the scene descriptions verbatim; synthesise them.\n"
        "- **Avoid filler** such as mood, professionalism, intensity, or cinematic commentary "
        "unless it is directly visible and important for understanding the event.\n"
        "- If the scenes are visually repetitive or contain little new information, compress them "
        "briefly instead of restating similar actions.\n"
        "- The closing takeaway sentence must add a new synthesising observation. Do **not** "
        "re-describe the last scene or repeat what an earlier sentence already said.\n"
        "- Each sentence must introduce information not already stated. Avoid near-duplicate "
        "phrases.\n"
        "- Return **only** the summary prose — no title, preamble, bullet points, numbering, "
        "or meta-commentary about the task.\n\n"
        f"## Scene descriptions\n{bulleted}"
    )
