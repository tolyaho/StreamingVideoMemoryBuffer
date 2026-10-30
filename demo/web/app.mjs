import {formatTime, stateAt, windowLedger, firstSeen, windowAt, retrievedIds, evidenceFor, buildStoryboard, presentationAt, answerReading, evidenceTime, annotatedMomentAt} from './timeline.mjs';

const $ = id => document.getElementById(id);
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const ease = t => 1 - Math.pow(1 - clamp(t), 3);
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
const SPEEDS = [1, 4, 8, 16];
const TIER_COLOR = {recent: 'var(--recent)', pending: 'var(--episodic)', episodic: 'var(--episodic)', event: 'var(--event)', discarded: 'var(--discard)', evicted: 'var(--discard)'};
const GROUP_TIER = {coarse: 'event', episodic: 'episodic', grounded: 'window'};

let runs = [], runIndex = 0, session, nodes, first = {}, board = null;
let clock = 0, playing = false, speed = 4;
let tour = null;                 // {startedAt, elapsed, capture}
let questionSince = null, lastQuestionId = null;
let evidenceView = null;         // {node, time} when the video shows an older retrieved moment
let tab = 'memory', tabPinned = false;
let lastSig = {};
const video = $('video');

function el(tag, cls, text) { const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = text; return n; }
function img(src, alt) { const i = el('img'); i.src = src; i.alt = alt; i.decoding = 'sync'; return i; }
function changed(key, value) { const s = JSON.stringify(value); if (lastSig[key] === s) return false; lastSig[key] = s; return true; }
function hasMedia() { return session.media.available; }
function isTrace() { return session.mode === 'recorded_trace'; }

// Episode/event captions carry generator scaffolding ("Episode a–bs", "[t=…| frame i/n]"); show the prose.
function prose(node) {
  const text = node?.caption ?? '';
  if (node.kind === 'event') return text.replace(/^Event [\d.]+–[\d.]+s:\s*/, '');
  if (node.kind === 'episode') return text.replace(/^Episode [\d.]+–[\d.]+s\s*/, '').split(/\n\s*\n/).map(p => p.replace(/^\[t=[^\]]+\]\s*/, '')).filter(Boolean)[0] ?? '';
  return text;
}
function thumbOf(node) {
  if (!node) return null;
  if (node.kind === 'window') return node.thumbnail ?? null;
  const ids = node.kind === 'event' ? node.representatives : (node.representatives?.length ? node.representatives : node.members);
  for (const id of ids ?? []) if (nodes[id]?.thumbnail) return nodes[id].thumbnail;
  return null;
}
function range(n) { return `${formatTime(n.start)}–${formatTime(n.end)}`; }
function optionText(row, letter) { return row.options?.find(o => o.trim().startsWith(letter + '.'))?.trim() ?? letter; }

// ------------------------------------------------------------------------------------------------
// view = {clock, videoAt, stage, stageElapsed, chapter, rate}
function currentView() {
  if (tour) {
    const p = presentationAt(board, tour.elapsed);
    return {clock: p.time, videoAt: p.showVideoAt, stage: p.stage, stageElapsed: p.stageElapsed, chapter: p.chapter, rate: p.rate, elapsed: tour.elapsed};
  }
  const st = stateAt(session, clock);
  if (st.question?.question_id !== lastQuestionId) { lastQuestionId = st.question?.question_id ?? null; questionSince = st.question ? performance.now() : null; }
  const since = questionSince === null ? 0 : (performance.now() - questionSince) / 1000;
  return {clock, videoAt: evidenceView ? evidenceView.time : clock, stage: 'explore', stageElapsed: since, chapter: '', rate: playing ? speed : 0};
}

// reveal fractions per element, from the stage clock only
function reveals(view) {
  const s = view.stage, e = view.stageElapsed;
  if (s === 'explore') {
    if (reduceMotion) return {hits: 1, answerH: 1, answerB: 1, arcs: 1, evidence: evidenceView ? 1 : 0, question: 1};
    return {hits: clamp(e / 0.6), answerH: clamp((e - 0.4) / 0.5), answerB: clamp((e - 0.7) / 0.5), arcs: clamp(e / 0.9), evidence: evidenceView ? 1 : 0, question: 1};
  }
  const order = ['intro', 'observe', 'pass', 'approach', 'question', 'retrieve', 'evidence', 'answer', 'outro'];
  const i = order.indexOf(s), at = name => order.indexOf(name);
  const after = (name, d = 0.6) => i > at(name) ? 1 : i === at(name) ? ease(e / d) : 0;
  return {
    question: after('question', 0.7),
    hits: i > at('retrieve') ? 1 : i === at('retrieve') ? clamp(e / 2.2) : 0,
    arcs: i > at('retrieve') ? 1 : i === at('retrieve') ? ease((e - 0.3) / 1.6) : 0,
    evidence: i === at('evidence') || i === at('answer') ? (i === at('evidence') ? ease(e / 0.6) : 1) : 0,
    answerH: i > at('answer') ? 1 : i === at('answer') ? ease((e - 0.2) / 0.7) : 0,
    answerB: i > at('answer') ? 1 : i === at('answer') ? ease((e - 1.6) / 0.7) : 0,
  };
}

// ------------------------------------------------------------------------------------------------
function renderViewport(view, st, rv) {
  $('tc').textContent = formatTime(view.videoAt, true);
  const rateLabel = view.rate > 1.05 ? ` · ${Math.round(view.rate)}×` : view.rate === 0 && tour ? ' · paused' : '';
  $('tc-label').textContent = (evidenceView || (tour && ['evidence', 'answer'].includes(view.stage)) ? 'Evidence' : 'Stream') + rateLabel;

  // caption ticker: the actual Florence-2 caption of the newest window processed by now
  const w = windowAt(session, view.clock);
  const qNow = st.question && !tour && view.clock - st.question.stream_time < 8;
  const showCap = isTrace() && w && !qNow && !['question', 'retrieve', 'evidence', 'answer', 'outro'].includes(view.stage);
  $('vp-caption').hidden = !showCap;
  if (showCap && changed('cap', w.id)) {
    $('cap-meta').textContent = `Window ${range(w)} · Florence-2 caption`;
    $('cap-text').textContent = w.caption ?? '';
  }

  // titles (tour only)
  const title = $('vp-title');
  if (tour && view.chapter) {
    const seg = board.segments.find(s => s.stage === view.stage);
    const dur = seg.t1 - seg.t0, e = view.stageElapsed;
    const big = view.stage === 'intro' || view.stage === 'outro';
    title.className = 'vp-title' + (big ? '' : ' note');
    const fade = Math.min(ease(e / 0.6), ease((dur - e) / 0.5));
    title.style.setProperty('--title', String(view.stage === 'question' || view.stage === 'retrieve' || view.stage === 'evidence' || view.stage === 'answer' ? 0 : fade));
    let text = view.chapter;
    if (view.stage === 'pass' && view.stageElapsed > dur * 0.45) {
      const m = annotatedMomentAt(session, board.focus.question_id, view.clock);
      if (m && m.state === 'kept' && m.parents.length) {
        const ep = nodes[m.parents[0]];
        text = `The annotated moment has left recent memory. It was kept in episode ${range(ep)}.`;
      } else if (m && m.state === 'kept') text = 'The annotated moment has left recent memory. Its windows were kept as novel.';
    }
    if (view.stage === 'outro') {
      const n = session.records.length, ok = side => session.records.filter(r => { const a = answerReading(r, side); return a.correct && a.consistent; }).length;
      text = `${view.chapter}\n`;
      $('vp-sub').textContent = `This clip: SVMB ${ok('hierarchical')}/${n}, recent-only baseline ${ok('baseline')}/${n} questions. One recorded example, not a benchmark.`;
    }
    $('vp-sub').hidden = view.stage !== 'outro';
    $('vp-sub').style.setProperty('--title', title.style.getPropertyValue('--title'));
    if (changed('title', text)) title.textContent = text.trim();
  } else { title.style.setProperty('--title', '0'); $('vp-sub').hidden = true; }
  $('vp-question').classList.toggle('compact', !!tour && ['evidence', 'answer', 'outro'].includes(view.stage));
  $('viewport').classList.toggle('dim', !!tour && (view.stage === 'outro' || view.stage === 'intro' && view.stageElapsed < 0.2));

  // question lower third
  const q = st.question;
  const showQ = !!q && (tour ? ['question', 'retrieve'].includes(view.stage) : view.clock - q.stream_time < 8);
  $('vp-question').hidden = !showQ;
  if (showQ) {
    $('vp-question').style.setProperty('--q', String(tour ? rv.question : 1));
    if (changed('vpq', q.question_id)) {
      $('vpq-meta').textContent = `Question · asked at ${formatTime(q.t_seconds)}`;
      $('vpq-text').textContent = q.question;
    }
  }

  // on-video verdicts during the tour's answer stage: verbatim headlines, flagged like the cards
  const showA = !!q && !!tour && view.stage === 'answer';
  $('vpq-answers').hidden = !showA;
  if (showA) {
    for (const [side, key, r] of [['hierarchical', 'h', rv.answerH], ['baseline', 'b', rv.answerB]]) {
      const a = answerReading(q, side);
      $(`vpa-${key}`).style.setProperty('--reveal', String(r));
      if (changed('vpa-' + key, [q.question_id, side])) {
        $(`vpa-${key}-t`).textContent = a.headline;
        const v = $(`vpa-${key}-v`);
        v.textContent = !a.consistent ? `scored ${a.letter}, text names ${a.named}` : a.correct ? 'matches label' : 'differs from label';
        v.className = 'verdict ' + (!a.consistent ? 'warn' : a.correct ? 'ok' : 'bad');
      }
    }
  }

  // evidence: the video shows an earlier moment the retriever actually returned
  const ev = tour ? (['evidence', 'answer'].includes(view.stage) ? board.evidence : null) : evidenceView;
  const evOn = ev ? rv.evidence : 0;
  $('viewport').style.setProperty('--evidence', String(evOn));
  $('vp-evidence').hidden = !ev;
  $('vp-evidence').style.setProperty('--evidence', String(evOn));
  if (ev && changed('vpe', [ev.node.id, view.clock])) {
    const n = ev.node, gap = Math.round(view.clock - n.end);
    const tier = ev.group === 'coarse' ? 'long-term memory' : ev.group === 'episodic' ? 'episodic memory' : 'an archived window';
    $('vp-evidence').replaceChildren(el('b', '', 'Retrieved evidence'), document.createTextNode(`${range(n)} from ${tier} · ${gap} s before the question`));
  }
}

// ------------------------------------------------------------------------------------------------
function ribbonGeometry() {
  const svg = $('ribbon'), W = svg.clientWidth || 1000, H = svg.clientHeight || 128;
  const narrow = W < 560, gutter = narrow ? 8 : 74;
  const x = t => gutter + (W - gutter - 10) * clamp(t / session.clock_end);
  const lanes = narrow ? {axis: 11, top: 16, win: 44, ep: 68, ev: 88, h: 16, bar: 12} : {axis: 12, top: 18, win: 50, ep: 76, ev: 100, h: 18, bar: 14};
  return {svg, W, H, x, gutter, narrow, ...lanes};
}

function renderRibbon(view, st, rv) {
  const g = ribbonGeometry();
  const ledger = isTrace() ? windowLedger(session, view.clock) : [];
  const hits = retrievedIds(st.retrieval);
  const showHits = !!st.retrieval && (tour ? rv.arcs > 0 : true);
  const arcP = clamp(tour || !reduceMotion ? rv.arcs : 1);
  const key = [g.W, !!tour, st.snapshot?.time ?? -1, st.retrieval?.question_id ?? null, showHits, Math.round(arcP * 40), evidenceView?.node.id ?? null,
    tour && ['evidence', 'answer'].includes(view.stage)];
  const svg = g.svg;
  svg.setAttribute('viewBox', `0 0 ${g.W} ${g.H}`);
  svg.setAttribute('aria-valuemax', String(Math.round(session.clock_end)));
  svg.setAttribute('aria-valuenow', String(Math.round(view.clock)));
  svg.setAttribute('aria-valuetext', `${formatTime(view.clock)} of ${formatTime(session.clock_end)}`);
  if (changed('ribbon', key)) {
    const parts = [];
    const step = [15, 30, 60, 120, 300, 600, 1200].find(st => (g.W - g.gutter) * st / session.clock_end >= 100) ?? 1200;
    for (let t = 0; t <= session.clock_end; t += step) {
      parts.push(`<line x1="${g.x(t)}" x2="${g.x(t)}" y1="${g.axis + 4}" y2="${g.H - 6}" stroke="#1d2127" />`);
      parts.push(`<text x="${g.x(t) + 3}" y="${g.axis}">${formatTime(t)}</text>`);
    }
    if (!g.narrow) for (const [y, label] of [[g.win, 'WINDOWS'], [g.ep, 'EPISODES'], [g.ev, 'EVENTS']]) parts.push(`<text class="lane-label" x="0" y="${y + 11}">${label}</text>`);
    // annotated evidence (human label, fixed before the run): shown only once the question is visible
    const annQ = tour ? board.focus.question_id : st.question?.question_id;
    const ann = annQ && session.annotation?.questions?.[annQ];
    if (ann && tour && !g.narrow) { const [, b] = ann.evidence_stimulus_seconds.at(-1); parts.push(`<text x="${g.x(b) + 5}" y="${g.H - 2}" style="fill:var(--ok);font-size:10px">annotated moment</text>`); }
    if (ann && (tour || showHits)) for (const [a, b] of ann.evidence_stimulus_seconds) parts.push(`<rect x="${g.x(a)}" y="${g.H - 7}" width="${Math.max(2, g.x(b) - g.x(a))}" height="3" rx="1.5" fill="var(--ok)"><title>Human-annotated evidence ${formatTime(a)}–${formatTime(b)}</title></rect>`);
    if (!isTrace() && !g.narrow) parts.push(`<text x="${g.x(0) + 4}" y="${g.ep + 11}" style="fill:var(--faint)">memory states were not saved by this evaluation</text>`);
    if (isTrace() && !(st.snapshot?.events.length) && !g.narrow) {
      const cap = session.provenance?.pipeline_config?.episodic_capacity;
      parts.push(`<text x="${g.x(0) + 4}" y="${g.ev + 11}" style="fill:var(--faint)">none yet${cap ? ` — consolidation starts above ${cap} episodes` : ''}</text>`);
    }
    // window cells
    let r0 = Infinity, r1 = -Infinity;
    for (const w of ledger) {
      const x0 = g.x(w.start), x1 = g.x(w.end), wd = Math.max(1, x1 - x0 - (g.narrow ? 0.6 : 1.4));
      const hit = showHits && hits.has(w.id);
      if (w.state === 'recent') { r0 = Math.min(r0, x0); r1 = Math.max(r1, x0 + wd); }
      const fill = w.state === 'pending' ? 'none' : TIER_COLOR[w.state];
      const stroke = hit ? 'var(--accent)' : w.state === 'pending' ? 'var(--episodic)' : 'none';
      parts.push(`<rect data-id="${w.id}" x="${x0}" y="${g.win}" width="${wd}" height="${g.h}" rx="2" fill="${fill}" stroke="${stroke}" stroke-width="${hit ? 2 : 1}"/>`);
    }
    if (r1 > r0) parts.push(`<rect x="${r0 - 3}" y="${g.win - 4}" width="${r1 - r0 + 6}" height="${g.h + 8}" rx="5" fill="none" stroke="var(--recent)" stroke-opacity=".5"/>`);
    // episode and event bars from the recorded snapshot
    const snap = st.snapshot;
    if (snap) {
      for (const id of snap.episodic) {
        const n = nodes[id], hit = showHits && hits.has(id);
        parts.push(`<rect data-id="${id}" x="${g.x(n.start)}" y="${g.ep}" width="${Math.max(2, g.x(n.end) - g.x(n.start) - 1.4)}" height="${g.bar}" rx="3" fill="var(--episodic)" fill-opacity="${hit ? 1 : .75}" stroke="${hit ? 'var(--accent)' : 'none'}" stroke-width="2"/>`);
      }
      if (snap.pending.length) {
        const a = nodes[snap.pending[0]].start, b = nodes[snap.pending.at(-1)].end;
        parts.push(`<rect x="${g.x(a)}" y="${g.ep}" width="${Math.max(2, g.x(b) - g.x(a) - 1.4)}" height="${g.bar}" rx="3" fill="none" stroke="var(--episodic)" stroke-dasharray="3 2"/>`);
      }
      for (const id of snap.events) {
        const n = nodes[id], hit = showHits && hits.has(id);
        parts.push(`<rect data-id="${id}" x="${g.x(n.start)}" y="${g.ev}" width="${Math.max(2, g.x(n.end) - g.x(n.start) - 1.4)}" height="${g.bar}" rx="3" fill="var(--event)" fill-opacity="${hit ? 1 : .8}" stroke="${hit ? 'var(--accent)' : 'none'}" stroke-width="2"/>`);
      }
    }
    // retrieval arcs: from the question time back to each returned memory node
    if (showHits && st.retrieval) {
      const qx = g.x(st.retrieval.time);
      const laneOf = h => h.group === 'coarse' ? g.ev : h.group === 'episodic' ? g.ep : g.win;
      for (const [id, h] of hits) {
        const n = nodes[id]; if (h.group === 'grounded') continue;
        const hx = (g.x(n.start) + g.x(n.end)) / 2, hy = laneOf(h);
        const len = Math.hypot(qx - hx, 40) * 1.6;
        parts.push(`<path d="M${qx},${g.win - 2} C${qx},${g.top - 6} ${hx},${g.top - 6} ${hx},${hy}" fill="none" stroke="var(--accent)" stroke-width="1.4" stroke-opacity=".85" stroke-dasharray="${len}" stroke-dashoffset="${len * (1 - arcP)}"/>`);
      }
    }
    // question markers
    for (const q of session.records) {
      const on = q.stream_time <= view.clock, cx = g.x(q.t_seconds), cy = g.top;
      parts.push(`<path d="M${cx},${cy - 5} l5,5 -5,5 -5,-5z" fill="${on ? 'var(--accent)' : 'var(--bg)'}" stroke="var(--accent)" stroke-width="1.3"><title>${q.question}</title></path>`);
    }
    parts.push(`<g id="playhead"><line y1="${g.top - 6}" y2="${g.H - 10}" stroke="var(--text)" stroke-width="1.5"/></g>`);
    if (evidenceView || (tour && ['evidence', 'answer'].includes(view.stage) && board.evidenceTime !== null)) {
      const et = evidenceView ? evidenceView.time : board.evidenceTime;
      parts.push(`<line x1="${g.x(et)}" x2="${g.x(et)}" y1="${g.top}" y2="${g.H - 10}" stroke="var(--accent)" stroke-width="2"/>`);
    }
    svg.innerHTML = parts.join('');
  }
  const ph = svg.querySelector('#playhead');
  if (ph) ph.setAttribute('transform', `translate(${g.x(view.clock)},0)`);
}

// ------------------------------------------------------------------------------------------------
function freshness(id, view, span = 3.5) {
  if (reduceMotion || !(id in first)) return 0;
  return clamp(1 - (view.clock - first[id]) / span);
}

function renderMemory(view, st) {
  const snap = st.snapshot;
  const hits = retrievedIds(st.retrieval);
  const reveal = st.retrieval && (!tour || ['retrieve', 'evidence', 'answer', 'outro'].includes(view.stage));
  if (!isTrace()) {
    if (changed('mem-empty', session.mode)) {
      for (const [id, label] of [['recent', 'Per-window frames and captions were not saved by this evaluation.'], ['episodic', 'Episode states were not saved.'], ['events', 'Event states were not saved.']]) {
        $(id).replaceChildren(el('div', 'empty-state', label));
        $(id + '-count').textContent = '—';
      }
      $('recent').classList.remove('film');
    }
    return;
  }
  // recent filmstrip
  const recent = snap?.recent ?? [];
  $('recent-count').textContent = `${recent.length} / ${session.provenance?.pipeline_config?.recent_capacity ?? '—'}`;
  if (changed('recent', [recent, reveal ? [...hits.keys()] : []])) {
    const film = $('recent'); film.replaceChildren();
    if (!recent.length) film.append(el('div', 'empty-state', 'Nothing has been processed yet.'));
    recent.forEach((id, i) => {
      const n = nodes[id], t = el('div', 'thumb' + (i === recent.length - 1 ? ' newest' : '') + (reveal && hits.has(id) ? ' hit' : ''));
      t.dataset.id = id;
      if (n.thumbnail) t.append(img(n.thumbnail, `Window ${range(n)}: ${n.caption}`)); else t.classList.add('empty');
      t.append(el('span', '', formatTime(n.start)));
      t.title = `${range(n)} · ${n.caption}`;
      film.append(t);
    });
  }
  for (const t of $('recent').children) if (t.dataset.id) t.style.setProperty('--fresh', String(freshness(t.dataset.id, view, 3)));

  // episodic: newest first, the forming episode, and every retrieved episode stays visible
  const eps = snap ? [...snap.episodic].reverse() : [];
  const pend = snap?.pending ?? [];
  const keepEp = new Set(eps.slice(0, 3));
  if (reveal) for (const id of eps) if (hits.has(id)) keepEp.add(id);
  $('episodic-count').textContent = String(eps.length) + (pend.length ? ` + forming` : '');
  if (changed('episodic', [eps, pend, [...keepEp], reveal])) {
    const box = $('episodic'); box.replaceChildren();
    if (pend.length) {
      const a = nodes[pend[0]], b = nodes[pend.at(-1)], c = el('div', 'card pending');
      const pic = el('div', 'pic'); const th = nodes[pend.at(-1)].thumbnail; if (th) pic.append(img(th, 'Newest promoted window'));
      const body = el('div'); const meta = el('div', 'meta'); meta.append(el('span', 'mono', `${formatTime(a.start)}–${formatTime(b.end)}`), el('span', '', `${pend.length} window${pend.length > 1 ? 's' : ''} · forming`));
      body.append(meta, el('p', '', 'Promoted windows wait here until the episode closes; its Moondream2 summary is written when it does.'));
      c.append(pic, body); box.append(c);
    }
    if (!eps.length && !pend.length) box.append(el('div', 'empty-state', 'No episodes yet. Windows are only promoted once they are evicted from recent memory (after 60 s).'));
    let hidden = 0;
    for (const id of eps) {
      if (!keepEp.has(id)) { hidden++; continue; }
      box.append(memoryCard(nodes[id], reveal && hits.has(id)));
    }
    if (hidden) box.append(el('div', 'more', `+ ${hidden} earlier episode${hidden > 1 ? 's' : ''} in memory`));
  }
  for (const c of $('episodic').querySelectorAll('.card[data-id]')) c.style.setProperty('--fresh', String(freshness(c.dataset.id, view, 6)));

  // long-term events
  const evs = snap ? [...snap.events].reverse() : [];
  $('events-count').textContent = String(evs.length);
  if (changed('events', [evs, reveal])) {
    const box = $('events'); box.replaceChildren();
    if (!evs.length) {
      const cap = session.provenance?.pipeline_config?.episodic_capacity;
      box.append(el('div', 'empty-state', cap ? `No long-term events yet. Consolidation starts once more than ${cap} episodes are held; this run's configuration${cap >= 50 ? ' (the benchmark preset) does not reach that within this clip' : ' reaches it once enough episodes accumulate'}.` : 'No long-term events.'));
    }
    for (const id of evs.slice(0, 3)) box.append(memoryCard(nodes[id], reveal && hits.has(id)));
    if (evs.length > 3) box.append(el('div', 'more', `+ ${evs.length - 3} earlier events`));
  }
  for (const c of $('events').querySelectorAll('.card[data-id]')) c.style.setProperty('--fresh', String(freshness(c.dataset.id, view, 8)));
  const recentDecision = (session.decisions ?? []).filter(d => d.phase === 'stream' && d.time <= view.clock).at(-1);
  $('flow-1').classList.toggle('pulse', !!recentDecision && view.clock - recentDecision.time < 3);
  $('flow-1-text').textContent = recentDecision
    ? `window ${formatTime(nodes[recentDecision.window_id].start)} evicted · max similarity ${recentDecision.max_similarity?.toFixed(3) ?? '—'} → ${recentDecision.promoted ? 'kept (novel)' : 'discarded'}`
    : 'evicted windows are kept if novel (similarity < 0.95)';
}

function memoryCard(n, hit) {
  const c = el('div', 'card fresh-glow' + (hit ? ' hit' : '')); c.dataset.id = n.id;
  const pic = el('div', 'pic' + (n.kind === 'event' ? ' multi' : ''));
  if (n.kind === 'event') {
    for (const id of n.representatives.slice(0, 4)) if (nodes[id]?.thumbnail) pic.append(img(nodes[id].thumbnail, `Representative window ${range(nodes[id])}`));
  } else { const th = thumbOf(n); if (th) pic.append(img(th, `Representative frame of episode ${range(n)}`)); }
  const body = el('div'), meta = el('div', 'meta');
  meta.append(el('span', 'mono', range(n)), el('span', '', n.kind === 'event' ? `${n.members.length} episodes · Qwen2.5-VL` : `${n.members.length} window${n.members.length > 1 ? 's' : ''} · Moondream2`));
  const p = el('p', '', prose(n)); p.title = n.caption;
  body.append(meta, p); c.append(pic, body);
  return c;
}

// ------------------------------------------------------------------------------------------------
function renderAsk(view, st, rv) {
  const q = st.question;
  $('ask-badge').hidden = !q; if (q) $('ask-badge').textContent = String(session.records.indexOf(q) + 1);
  $('ask-empty').hidden = !!q; $('ask-body').hidden = !q;
  if (!q) return;
  const ret = st.retrieval, ev = tour ? board.evidence : evidenceFor(session, ret, q.question_id);
  const ann = session.annotation?.questions?.[q.question_id];
  if (changed('ask', [q.question_id, !!ret])) {
    $('q-meta').textContent = `Q${session.records.indexOf(q) + 1} · asked ${formatTime(q.t_seconds)} · answered ${formatTime(q.stream_time, true)}`;
    $('q-meta').title = 'Questions are answered at the end of the first 3-second window that reaches their timestamp.';
    $('q-text').textContent = q.question;
    const ol = $('q-options'); ol.replaceChildren();
    for (const o of q.options ?? []) {
      const letter = o.trim()[0], li = el('li'); li.dataset.letter = letter;
      li.append(el('b', '', letter), el('span', '', o.trim().slice(2).trim()), el('span', 'who'));
      ol.append(li);
    }
    const list = $('ret-list'); list.replaceChildren();
    if (!ret) {
      list.append(el('div', 'empty-state', isTrace() ? 'No retrieval was recorded for this question.' : 'This evaluation did not save which memory entries were returned or their scores. A time range quoted inside a model answer is not a retrieval trace.'));
      $('ret-note').textContent = '';
    } else {
      const rows = [];
      for (const group of ['coarse', 'episodic', 'grounded']) for (const h of ret[group]) rows.push({...h, group, node: nodes[h.id]});
      const pinned = ret.pinned.length;
      const shown = rows.filter(r => r.group !== 'grounded');
      if (!shown.length) shown.push(...rows.slice(0, 4));
      const grounded = rows.filter(r => r.group === 'grounded').length;
      shown.forEach((r, i) => {
        const row = el('div', 'hit-row' + (ev && r.id === ev.id ? ' lead' : '')); row.style.setProperty('--i', String(i)); row.dataset.i = String(i);
        const dot = el('span', 'bar-dot'); dot.style.background = r.group === 'coarse' ? 'var(--event)' : r.group === 'episodic' ? 'var(--episodic)' : 'var(--recent)';
        const pic = el('div', 'pic'); const th = thumbOf(r.node); if (th) pic.append(img(th, `Retrieved ${r.node.kind} ${range(r.node)}`));
        const txt = el('div', 'txt'), top = el('div', 'top');
        top.append(el('span', 'mono', range(r.node)), el('span', '', `${r.group === 'coarse' ? 'event' : r.group === 'episodic' ? 'episode' : 'window'} · ${Math.max(0, Math.round(q.stream_time - r.node.end))} s before`));
        const p = el('p', '', prose(r.node)); p.title = r.node.caption;
        txt.append(top, p);
        const sc = el('span', 'score', r.score === null ? '—' : r.score.toFixed(3));
        sc.title = r.score === null ? 'The retriever returns no score for grounding windows' : 'Recorded retrieval score: (0.7·visual + 0.3·text similarity) × time decay';
        row.append(dot, pic, txt, sc);
        if (hasMedia()) { row.tabIndex = 0; row.setAttribute('role', 'button'); row.setAttribute('aria-label', `Show ${range(r.node)} in the video`);
          const go = () => showEvidence({...r, recentStart: ev?.recentStart}); row.addEventListener('click', go); row.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } }); }
        list.append(row);
      });
      $('ret-note').textContent = `+${grounded} frames · ${pinned} recent`;
      $('ret-note').title = `The reasoner also received ${grounded} grounding windows (representative frames' captions from the returned episodes, plus recent hits) and the ${pinned} most recent windows. The baseline received only the most recent windows.`;
    }
    for (const [side, key] of [['hierarchical', 'h'], ['baseline', 'b']]) {
      const a = q[side], r = answerReading(q, side);
      $(`ans-${key}-l`).textContent = r.headline;
      if (/^\W*[A-D]\W*$/.test(r.headline)) $(`ans-${key}-l`).append(el('span', 'opt', ` — letter only; option ${r.letter} reads “${optionText(q, r.letter).slice(3)}”`));
      const rest = a.prediction.trim().slice(r.headline.length).trim();
      $(`ans-${key}-why`).textContent = rest;
      $(`ans-${key}-why`).title = 'Verbatim model output: ' + a.prediction;
      const v = $(`ans-${key}-v`);
      if (!r.consistent) { v.textContent = `Scored ${r.letter} · text names ${r.named}`; v.className = 'verdict warn'; v.title = `The scorer reads only the first letter (${r.letter}), which ${a.correct ? 'matches' : 'differs from'} the label; the words that follow spell out option ${r.named}. Not counted as a genuine success here.`; }
      else { v.textContent = a.correct ? `${r.letter} · matches label` : `${r.letter} · differs from label`; v.className = 'verdict ' + (a.correct ? 'ok' : 'bad'); v.title = ''; }
    }
    const gtText = optionText(q, q.hierarchical.ground_truth);
    $('gt').textContent = ann ? `Ground truth ${gtText} Human annotation written before the run.` : `Ground truth (dataset label): ${gtText}`;
  }
  const hitsOn = tour ? rv.hits : 1;
  [...$('ret-list').children].forEach((row, i) => row.style.setProperty('--reveal', String(ease(hitsOn * 6 - i * 0.6))));
  $('ans-h').style.setProperty('--reveal', String(tour ? rv.answerH : 1));
  $('ans-b').style.setProperty('--reveal', String(tour ? rv.answerB : 1));
  $('gt').style.opacity = String(tour ? rv.answerB : 1);
  const showMarks = tour ? rv.answerB > 0.5 : true;
  for (const li of $('q-options').children) {
    const who = li.querySelector('.who'); who.replaceChildren();
    li.classList.toggle('is-gt', showMarks && li.dataset.letter === q.hierarchical.ground_truth);
    if (showMarks || (tour && rv.answerH > 0.5)) {
      const hr = answerReading(q, 'hierarchical');
      if (li.dataset.letter === q.hierarchical.pred_letter) who.append(el('i', 'h', hr.consistent ? 'SVMB' : 'SVMB letter'));
      if (!hr.consistent && li.dataset.letter === hr.named) who.append(el('i', 'h', 'SVMB text'));
      if (showMarks && li.dataset.letter === q.baseline.pred_letter) who.append(el('i', 'b', 'Baseline'));
    }
  }
}

// ------------------------------------------------------------------------------------------------
function setTab(name, pinned = false) {
  tab = name; if (pinned) tabPinned = true;
  $('tab-memory').setAttribute('aria-selected', String(name === 'memory'));
  $('tab-ask').setAttribute('aria-selected', String(name === 'ask'));
  $('pane-memory').hidden = name !== 'memory'; $('pane-ask').hidden = name !== 'ask';
}

function render() {
  const view = currentView();
  const st = stateAt(session, view.clock);
  const rv = reveals(view);
  if (tour) setTab(['question', 'retrieve', 'evidence', 'answer', 'outro'].includes(view.stage) ? 'ask' : 'memory');
  else if (!tabPinned) setTab(st.question && view.clock - st.question.stream_time < 8 ? 'ask' : 'memory');
  renderViewport(view, st, rv);
  renderRibbon(view, st, rv);
  if (tab === 'memory') renderMemory(view, st); else lastSig.recent = lastSig.episodic = lastSig.events = null;
  renderAsk(view, st, rv);
  if (tour?.capture) $('tour').textContent = `Guided tour · ${Math.floor(view.elapsed)} / ${Math.round(board.duration)} s`;
  $('play-icon').setAttribute('d', playing || (tour && !tour.capture) ? 'M5 4h3.5v12H5zM11.5 4H15v12h-3.5z' : 'M6 4l10 6-10 6z');
  $('play').setAttribute('aria-label', playing ? 'Pause' : 'Play');
  for (const c of $('chips').children) c.setAttribute('aria-current', String(st.question?.question_id === c.dataset.q));
  return view;
}

// ------------------------------------------------------------------------------------------------
function syncVideo(view) {
  if (!hasMedia()) return;
  const target = view.videoAt;
  if (tour) {
    const live = view.rate > 0 && view.rate <= 16 && !['evidence', 'answer'].includes(view.stage);
    if (live) {
      if (Math.abs(video.playbackRate - view.rate) > 0.01) video.playbackRate = view.rate;
      if (video.paused) video.play().catch(() => {});
      if (Math.abs(video.currentTime - target) > 0.6) video.currentTime = target;
    } else {
      if (!video.paused) video.pause();
      if (Math.abs(video.currentTime - target) > 0.05) video.currentTime = target;
    }
  }
}

let prevNow;
function tick(now) {
  const dt = prevNow === undefined ? 0 : Math.min(0.25, (now - prevNow) / 1000);
  prevNow = now;
  if (tour && !tour.capture) {
    tour.elapsed += dt;
    if (tour.elapsed >= board.duration) { tour.elapsed = board.duration; stopTour(false); }
  } else if (!tour && playing) {
    if (hasMedia() && !evidenceView) { clock = video.currentTime; if (video.ended) playing = false; }
    else { clock = Math.min(session.clock_end, clock + dt * speed); if (clock >= session.clock_end) playing = false; }
  }
  if (!(tour && tour.capture)) { const view = render(); syncVideo(view); }
  requestAnimationFrame(tick);
}

function seek(t) {
  stopTour(true); evidenceView = null;
  clock = clamp(t, 0, session.clock_end);
  if (hasMedia()) video.currentTime = clock;
  render();
}

function togglePlay() {
  if (tour) { stopTour(true); return; }
  if (evidenceView) { evidenceView = null; if (hasMedia()) video.currentTime = clock; }
  playing = !playing;
  if (playing && clock >= session.clock_end - 0.05) seek(0), playing = true;
  if (hasMedia()) { video.playbackRate = speed; playing ? video.play().catch(() => { playing = false; }) : video.pause(); }
  render();
}

function showEvidence(hit) {
  if (!hasMedia()) return;
  stopTour(true); playing = false; video.pause();
  const n = hit.node;
  const time = evidenceTime(session, hit, lastQuestionId);
  evidenceView = {node: n, group: hit.group, time};
  video.currentTime = time;
  render();
}

function startTour(capture = false) {
  playing = false; evidenceView = null; tabPinned = false;
  tour = {elapsed: 0, capture};
  if (hasMedia()) { video.pause(); video.currentTime = 0; }
  $('tour').textContent = 'Stop tour';
  document.body.classList.toggle('capture', capture);
}
function stopTour(keepClock) {
  if (!tour) return;
  const view = currentView();
  tour = null; $('tour').innerHTML = 'Guided tour <span aria-hidden="true">→</span>';
  if (hasMedia()) { video.pause(); video.playbackRate = speed; }
  clock = keepClock ? view.clock : clock;
  if (hasMedia()) video.currentTime = clock;
}

// Deterministic frame for recording: no wall-clock dependence; resolves once the video shows the frame.
async function renderTourFrame(elapsed) {
  if (!tour || !tour.capture) startTour(true);
  tour.elapsed = elapsed;
  const view = render();
  if (hasMedia() && Math.abs(video.currentTime - view.videoAt) > 0.5 / 24) {
    await new Promise(resolve => { video.addEventListener('seeked', resolve, {once: true}); video.currentTime = view.videoAt; });
  }
  await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
  return {clock: view.clock, videoAt: view.videoAt, stage: view.stage};
}

// ------------------------------------------------------------------------------------------------
function provenance() {
  const body = $('prov-body'); body.replaceChildren();
  const block = (title, ...children) => { const b = el('section', 'prov-block'); b.append(el('h3', '', title), ...children); body.append(b); };
  const kv = pairs => { const d = el('div', 'kv'); for (const [k, v] of pairs) d.append(el('span', '', k), el('span', '', v ?? '—')); return d; };
  const p = session.provenance ?? {};
  const legend = el('div', 'legend');
  for (const [c, t] of [['var(--recent)', 'recent window'], ['var(--episodic)', 'episodic (promoted, grouped)'], ['var(--event)', 'long-term event'], ['var(--discard)', 'discarded window'], ['var(--accent)', 'returned by retrieval'], ['var(--ok)', 'human-annotated evidence']]) { const s = el('span'); const i = el('i'); i.style.background = c; s.append(i, document.createTextNode(t)); legend.append(s); }
  block('Playback', el('p', '', 'This replays a recorded run. No model is called in the browser. Speeds, pauses, jumps, fades and the guided tour are presentation choices, not inference latency.'));
  if (isTrace()) {
    block('What is recorded, and what is UI', el('p', '', 'Recorded during the run: every window (its frame thumbnail and Florence-2 caption), the writer\'s promote/discard decision for each evicted window, episode and event membership, Moondream2 episode text, Qwen2.5-VL event summaries, the retriever\'s returned nodes and scores, the formatted evidence text, and both answers. UI only: colours, arcs, motion, layout, and which retrieved hit the tour jumps to (the one overlapping the human-annotated evidence).'), legend);
    block('Run', kv([['Configuration', session.config + (p.config_note ? ` — ${p.config_note}` : '')], ['Reasoner', session.model], ...Object.entries(p.models ?? {}).filter(([k]) => k !== 'revisions').map(([k, v]) => [k.replace(/_/g, ' '), `${v} @ ${(p.models.revisions?.[v] ?? '').slice(0, 10)}`]), ['Device', `${p.runtime?.device} · torch ${p.runtime?.torch} · transformers ${p.runtime?.transformers}`], ['Wall-clock to process the clip', p.processing_wall_s ? `${Math.round(p.processing_wall_s)} s for ${Math.round(session.clock_end)} s of video, including model loading (not real-time)` : '—']]));
    block('Settings', kv(Object.entries(p.pipeline_config ?? {}).map(([k, v]) => [k, String(v)])));
  } else {
    block('What is recorded', el('p', '', 'Saved answers from the published evaluation. Per-window memory states, returned retrieval nodes and scores were not saved, so those panels stay empty. A time range quoted inside a model answer is not a retrieval trace.'));
  }
  const r = session.media.rights;
  block('Source video', r ? kv([['License', `${r.license} — ${r.attribution}`], ['Source', r.source_url], ['Changes', r.changes], ['SHA-256 (exact file played and processed)', session.media.source_sha256]]) : el('p', '', session.media.reason ?? 'No cleared source video is attached; nothing is substituted.'));
  if (session.annotation) block('Ground truth', el('p', '', `Questions, options, answers and evidence intervals were written from frame inspection before inference (${session.annotation.written_before_inference}). Annotation SHA-256 ${session.annotation_sha256 ?? '—'}.`));
  block('Files · SHA-256', kv((p.source_files ?? []).concat(p.code_files ?? []).map(f => [f.path, f.sha256])));
  $('prov').showModal();
}

// ------------------------------------------------------------------------------------------------
async function loadRun(index) {
  const res = await fetch(`/api/session/${index}`); if (!res.ok) throw Error('The recorded run could not be loaded.');
  session = await res.json(); nodes = session.nodes; first = firstSeen(session); runIndex = index; lastSig = {};
  board = buildStoryboard(session, session.tour_question_id);
  for (const b of $('runs').children) b.setAttribute('aria-checked', String(Number(b.dataset.i) === index));
  $('nomedia').hidden = hasMedia(); video.hidden = !hasMedia();
  if (hasMedia()) { if (video.getAttribute('src') !== session.media.url) video.src = session.media.url; }
  else {
    $('nomedia-title').textContent = 'Video unavailable';
    $('nomedia-text').textContent = session.media.reason ?? 'The source video is not attached to this run. Captions, memory and answers are still the recorded ones.';
  }
  const r = session.media.rights;
  $('credit').textContent = '';
  $('foot-left').textContent = r
    ? `Footage: Big Buck Bunny ${r.attribution.replace('(c) copyright', '©')} · ${r.license} · excerpt 0:33–4:59, audio removed`
    : 'StreamingBench sample 36 · saved outputs only · source footage not redistributed';
  $('foot-right').textContent = isTrace()
    ? `${session.config === 'short_clip_demo' ? 'Demo-only config (episodic capacity 8, benchmark uses 100)' : 'Unmodified benchmark preset'} · ${session.model.split('/').at(-1)} reasoner · one recorded case study, not a benchmark`
    : 'Saved evaluation outputs · one example, not a benchmark';
  const chips = $('chips'); chips.replaceChildren();
  session.records.forEach((q, i) => {
    const c = el('button', 'chip'); c.type = 'button'; c.dataset.q = q.question_id;
    const mk = side => { const r = answerReading(q, side); return !r.consistent ? 'warn' : r.correct ? 'ok' : 'bad'; };
    const marks = el('span', 'marks'); marks.append(el('i', mk('hierarchical')), el('i', mk('baseline')));
    marks.title = `SVMB ${q.hierarchical.correct ? 'matches' : 'differs from'} label; baseline ${q.baseline.correct ? 'matches' : 'differs from'} label`;
    c.append(el('span', '', `Q${i + 1}`), el('span', 'mono', formatTime(q.t_seconds)), marks);
    c.setAttribute('aria-label', `Question ${i + 1} at ${formatTime(q.t_seconds)}: ${q.question}`);
    c.addEventListener('click', () => { playing = false; video.pause(); tabPinned = false; seek(q.stream_time); setTab('ask'); });
    chips.append(c);
  });
  const focus = board.focus;
  stopTour(false); playing = false; evidenceView = null;
  clock = isTrace() ? 0 : Math.max(0, focus.stream_time - 100);
  if (hasMedia()) video.currentTime = clock;
  render();
}

async function main() {
  runs = await (await fetch('/api/runs')).json();
  const box = $('runs');
  if (runs.length > 1) {
    box.hidden = false;
    for (const r of runs) {
      const b = el('button'); b.type = 'button'; b.setAttribute('role', 'radio'); b.dataset.i = r.index;
      b.append(document.createTextNode(r.config === 'short_clip_demo' ? 'Short-clip config' : r.config === 'notebook' ? 'Benchmark config' : r.config), el('small', '', r.config === 'short_clip_demo' ? 'all 3 tiers' : 'unchanged'));
      b.addEventListener('click', () => loadRun(r.index).catch(fail));
      box.append(b);
    }
  }
  const sp = $('speeds');
  for (const s of SPEEDS) { const b = el('button', '', `${s}×`); b.type = 'button'; b.setAttribute('role', 'radio'); b.setAttribute('aria-checked', String(s === speed)); b.addEventListener('click', () => { speed = s; for (const x of sp.children) x.setAttribute('aria-checked', String(x === b)); if (hasMedia()) video.playbackRate = s; }); sp.append(b); }
  $('play').addEventListener('click', togglePlay);
  $('tour').addEventListener('click', () => tour ? stopTour(true) : startTour(false));
  $('tab-memory').addEventListener('click', () => setTab('memory', true));
  $('tab-ask').addEventListener('click', () => setTab('ask', true));
  $('prov-open').addEventListener('click', provenance);
  $('prov-close').addEventListener('click', () => $('prov').close());
  const rib = $('ribbon');
  const toTime = e => { const g = ribbonGeometry(), r = rib.getBoundingClientRect(); return clamp((e.clientX - r.left - g.gutter) / (g.W - g.gutter - 10)) * session.clock_end; };
  rib.addEventListener('pointerdown', e => { rib.setPointerCapture(e.pointerId); playing = false; video.pause(); tabPinned = false; seek(toTime(e)); });
  rib.addEventListener('pointermove', e => {
    if (e.buttons) { seek(toTime(e)); return; }
    const t = toTime(e), w = isTrace() ? windowLedger(session, tour ? currentView().clock : clock).find(x => x.start <= t && t < x.end) : null;
    const tip = $('tip');
    if (!w) { tip.hidden = true; return; }
    const n = nodes[w.id]; tip.replaceChildren();
    if (n.thumbnail) tip.append(img(n.thumbnail, ''));
    tip.append(el('div', 'mono', `${range(n)} · ${w.state === 'event' ? 'in a long-term event' : w.state === 'episodic' ? 'in an episode' : w.state}`), el('div', '', n.caption));
    const r = rib.getBoundingClientRect(); tip.hidden = false;
    tip.style.left = `${clamp(e.clientX - r.left - 150, 0, r.width - 300)}px`; tip.style.bottom = `${r.height + 6}px`;
  });
  rib.addEventListener('pointerleave', () => { $('tip').hidden = true; });
  rib.addEventListener('keydown', e => { if (['ArrowLeft', 'ArrowRight'].includes(e.key)) { e.preventDefault(); seek(clock + (e.key === 'ArrowLeft' ? -3 : 3)); } });
  document.addEventListener('keydown', e => {
    if ($('prov').open || ['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
    if (e.code === 'Space' && !['BUTTON'].includes(e.target.tagName)) { e.preventDefault(); togglePlay(); }
  });
  video.addEventListener('ended', () => { playing = false; });
  window.addEventListener('resize', () => { lastSig.ribbon = null; });
  await loadRun(0);
  window.demo = {ready: true, loadRun, seek, startTour, stopTour, renderTourFrame, provenance, setTab,
    tourDuration: () => board.duration, state: () => ({clock, playing, tour: !!tour, ...stateAt(session, clock)}), session: () => session};
  requestAnimationFrame(tick);
}
function fail(error) { $('error').hidden = false; $('error').textContent = error.message; }
main().catch(fail);
