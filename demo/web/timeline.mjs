// Pure state derivation for recorded playback. Everything here reads the recorded session only:
// no element of memory, retrieval or answer is synthesised, and nothing after `time` is returned.

export function formatTime(seconds, precise = false) {
  const value = Math.max(0, Math.round(seconds * 100) / 100);
  const m = Math.floor(value / 60), s = Math.floor(value % 60);
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}${precise ? '.' + String(Math.round(value * 100) % 100).padStart(2, '0') : ''}`;
}

// The end-of-stream finalize() snapshot happens after the last question; it is not part of the
// streaming timeline (it would show every episode consolidated at once).
export function streamSnapshots(session) {
  return session.snapshots.filter(s => s.phase !== 'finalize');
}

export function stateAt(session, time) {
  const clock = Math.max(0, Math.min(session.clock_end, Number(time)));
  const available = session.records.filter(row => row.stream_time <= clock);
  const question = available.at(-1) ?? null;
  const snapshot = streamSnapshots(session).filter(row => row.time <= clock).at(-1) ?? null;
  const retrieval = question ? session.retrievals.find(row => row.question_id === question.question_id && row.time <= clock) ?? null : null;
  const baseline = question ? (session.baseline_inputs ?? []).find(row => row.question_id === question.question_id && row.time <= clock) ?? null : null;
  return {clock, question, snapshot, retrieval, baseline};
}

// Where every window seen so far sits at `time`, from the recorded snapshot and the writer's
// recorded promote/discard decisions. Windows that had not arrived yet are absent.
export function windowLedger(session, time) {
  const {snapshot, clock} = stateAt(session, time);
  if (!snapshot) return [];
  const nodes = session.nodes;
  const where = new Map();
  for (const id of snapshot.recent) where.set(id, {state: 'recent'});
  for (const id of snapshot.pending) where.set(id, {state: 'pending'});
  for (const ep of snapshot.episodic) for (const id of nodes[ep].members) where.set(id, {state: 'episodic', parent: ep});
  for (const ev of snapshot.events) for (const ep of nodes[ev].members) for (const id of (nodes[ep]?.members ?? [])) where.set(id, {state: 'event', parent: ev, episode: ep});
  const decided = new Map((session.decisions ?? []).filter(d => d.time <= clock + 1e-9).map(d => [d.window_id, d]));
  const seen = new Set();
  for (const s of streamSnapshots(session)) {
    if (s.time > clock + 1e-9) break;
    for (const id of s.recent) seen.add(id);
  }
  const out = [];
  for (const id of seen) {
    const n = nodes[id];
    const place = where.get(id) ?? (decided.get(id)?.promoted === false ? {state: 'discarded'} : {state: 'evicted'});
    out.push({id, start: n.start, end: n.end, ...place, decision: decided.get(id) ?? null});
  }
  return out.sort((a, b) => a.start - b.start);
}

// First stream time at which each node appears in any tier (for "fresh" highlighting).
export function firstSeen(session) {
  const first = {};
  for (const s of streamSnapshots(session)) {
    for (const tier of ['recent', 'pending', 'episodic', 'events']) for (const id of s[tier]) if (!(id in first)) first[id] = s.time;
  }
  return first;
}

export function windowAt(session, time) {
  const {snapshot} = stateAt(session, time);
  if (!snapshot || !snapshot.recent.length) return null;
  return session.nodes[snapshot.recent.at(-1)];
}

// Node ids the hierarchical retriever actually returned, grouped, with their recorded scores.
export function retrievedIds(retrieval) {
  if (!retrieval) return new Map();
  const hits = new Map();
  for (const group of ['coarse', 'episodic', 'grounded']) for (const h of retrieval[group]) hits.set(h.id, {group, score: h.score});
  return hits;
}

// The moment the walkthrough jumps back to. It is always one of the nodes the retriever actually
// returned and lies wholly before the recent window. If a human annotation for the question exists,
// the returned node overlapping the annotated evidence most is chosen (a recall check, not a model
// claim); otherwise the oldest such node. Model-cited time ranges are never used.
export function evidenceFor(session, retrieval, questionId) {
  if (!retrieval) return null;
  const snap = stateAt(session, retrieval.time).snapshot;
  const recentStart = snap?.recent.length ? Math.min(...snap.recent.map(id => session.nodes[id].start)) : retrieval.time;
  const candidates = [];
  for (const group of ['coarse', 'episodic', 'grounded']) for (const h of retrieval[group]) {
    const node = session.nodes[h.id];
    if (node.end <= recentStart + 1e-6) candidates.push({...h, group, node, recentStart});
  }
  if (!candidates.length) return null;
  const spans = session.annotation?.questions?.[questionId ?? retrieval.question_id]?.evidence_stimulus_seconds ?? [];
  const overlap = c => spans.reduce((sum, [a, b]) => sum + Math.max(0, Math.min(b, c.node.end) - Math.max(a, c.node.start)), 0);
  const annotated = candidates.filter(c => overlap(c) > 0);
  if (annotated.length) {
    const tier = {coarse: 0, episodic: 1, grounded: 2};
    annotated.sort((a, b) => overlap(b) - overlap(a) || tier[a.group] - tier[b.group] || (a.node.end - a.node.start) - (b.node.end - b.node.start));
    return {...annotated[0], annotated: true};
  }
  candidates.sort((a, b) => a.node.start - b.node.start);
  return {...candidates[0], annotated: false};
}

// Video position for a retrieved node: the middle of one of its own windows, preferring the
// node's representative windows (chosen by the writer) that overlap the annotated evidence most.
export function evidenceTime(session, ev, questionId) {
  if (!ev) return null;
  const node = ev.node ?? ev, nodes = session.nodes;
  const spans = session.annotation?.questions?.[questionId]?.evidence_stimulus_seconds ?? [];
  const overlap = w => spans.reduce((sum, [a, b]) => sum + Math.max(0, Math.min(b, w.end) - Math.max(a, w.start)), 0);
  const memberWindows = n => n.kind === 'window' ? [n] : n.kind === 'episode' ? n.members.map(id => nodes[id]).filter(Boolean)
    : n.members.flatMap(id => nodes[id] ? memberWindows(nodes[id]) : []);
  const reps = (node.representatives ?? []).map(id => nodes[id]).filter(Boolean);
  const pools = [reps, memberWindows(node)].filter(p => p.length);
  for (const pool of pools) {
    const best = [...pool].sort((a, b) => overlap(b) - overlap(a))[0];
    if (best && (overlap(best) > 0 || !spans.length)) return (best.start + best.end) / 2;
  }
  return (node.start + node.end) / 2;
}

export function buildStoryboard(session, questionId) {
  const focus = session.records.find(r => r.question_id === (questionId ?? session.focus_question_id)) ?? session.records[0];
  const qt = focus.stream_time;
  const retrieval = session.retrievals.find(r => r.question_id === focus.question_id) ?? null;
  const evidence = evidenceFor(session, retrieval, focus.question_id);
  const segs = [];
  const add = (dur, s0, s1, stage, chapter) => {
    const t0 = segs.length ? segs.at(-1).t1 : 0;
    segs.push({t0, t1: t0 + dur, s0, s1, stage, chapter});
  };
  const fill = Math.min(qt, 64);
  const early = Math.min(16, fill);
  const late = Math.max(fill, qt - 4);
  add(4, 0, early, 'intro', 'Streaming video, persistent memory.');
  add(7, early, fill, 'observe', 'Every 3 seconds, a new window enters recent memory.');
  add(Math.max(7, (late - fill) / 15), fill, late, 'pass', 'Windows leaving recent memory are kept if novel, then grouped.');
  add(3, late, qt, 'approach', 'Later, a question about the past.');
  add(3.5, qt, qt, 'question', '');
  add(3.5, qt, qt, 'retrieve', '');
  add(5.5, qt, qt, 'evidence', '');
  add(6, qt, qt, 'answer', '');
  add(4, qt, qt, 'outro', 'Remember the moments that matter.');
  return {segments: segs, duration: segs.at(-1).t1, focus, evidence, evidenceTime: evidenceTime(session, evidence, focus.question_id)};
}

export function presentationAt(board, elapsed) {
  const t = Math.max(0, Math.min(board.duration, elapsed));
  const seg = board.segments.find(s => t < s.t1) ?? board.segments.at(-1);
  const p = seg.t1 > seg.t0 ? (t - seg.t0) / (seg.t1 - seg.t0) : 1;
  const time = seg.s0 + (seg.s1 - seg.s0) * Math.min(1, p);
  const rate = seg.s1 > seg.s0 ? (seg.s1 - seg.s0) / (seg.t1 - seg.t0) : 0;
  const showVideoAt = ['evidence', 'answer', 'outro'].includes(seg.stage) ? board.evidenceTime ?? time : time;
  return {time, stage: seg.stage, chapter: seg.chapter, rate, progress: p, stageElapsed: t - seg.t0, showVideoAt};
}

// How to show a saved answer without rewriting it. `letter` is what the benchmark scorer reads
// (the first A-D character, as scripts/eval_common.parse_letter). `headline` is the verbatim
// opening of the output. `named` is the option whose text the output actually spells out after
// the letter, when it does; if that differs from `letter`, the display must say so.
export function answerReading(row, side) {
  const a = row[side], text = a.prediction.trim();
  const norm = s => s.toLowerCase().replace(/[^a-z0-9 ]+/g, ' ').replace(/\s+/g, ' ').trim();
  const after = norm(text.replace(/^[^A-D]*[A-D][.):]?\s*/, ''));
  let named = null;
  for (const o of row.options ?? []) {
    const body = norm(o.trim().slice(2));
    if (body && after.startsWith(body)) { named = o.trim()[0]; break; }
  }
  const first = text.split('\n')[0];
  const m = first.match(/^(\W*[A-D][.):]?\s*.{0,118}?[.!?])(\s|$)/) ?? first.match(/^(.{0,120}?[.!?])(\s|$)/);
  const headline = m ? m[1] : first.length > 120 ? first.slice(0, 117) + '…' : first;
  return {letter: a.pred_letter, headline, named, consistent: named === null || named === a.pred_letter, correct: a.correct};
}

// Where the human-annotated moment of a question sits in memory at `time`: derived from the
// recorded ledger (decisions + snapshot). Used for narration only; the system never sees it.
export function annotatedMomentAt(session, questionId, time) {
  const spans = session.annotation?.questions?.[questionId]?.evidence_stimulus_seconds ?? [];
  if (!spans.length) return null;
  const ledger = windowLedger(session, time);
  const inside = ledger.filter(w => spans.some(([a, b]) => Math.min(b, w.end) - Math.max(a, w.start) > 1));
  if (!inside.length) return {spans, state: 'future', windows: []};
  const states = new Set(inside.map(w => w.state));
  const parents = [...new Set(inside.map(w => w.parent).filter(Boolean))];
  const leftAt = inside.every(w => w.decision) ? Math.max(...inside.map(w => w.decision.time)) : null;
  const state = states.has('recent') ? (states.size === 1 ? 'recent' : 'leaving')
    : states.has('discarded') && states.size === 1 ? 'discarded' : 'kept';
  return {spans, state, windows: inside, parents, leftAt};
}
