import test from 'node:test';
import assert from 'node:assert/strict';
import {stateAt, formatTime, windowLedger, evidenceFor, evidenceTime, buildStoryboard, presentationAt, answerReading, streamSnapshots, annotatedMomentAt} from '../web/timeline.mjs';

// Unit fixture (test data only, never shown in the app): windows w0..w5 of 3 s, recent capacity 2.
const W = (i, extra = {}) => ({id: `w${i}`, kind: 'window', start: i * 3, end: i * 3 + 3, caption: `window ${i}`, members: [], representatives: [], ...extra});
const nodes = Object.fromEntries([0, 1, 2, 3, 4, 5].map(i => [`w${i}`, W(i)]));
nodes.e1 = {id: 'e1', kind: 'episode', start: 0, end: 6, caption: 'episode 0-6', members: ['w0', 'w1'], representatives: ['w1']};
nodes.d2 = W(2); delete nodes.d2; // w2 is discarded below
const session = {
  clock_end: 18,
  nodes,
  records: [{question_id: 'q', t_seconds: 14, stream_time: 15, question: 'Q?', options: ['A. Alpha.', 'B. Beta.', 'C. Gamma.', 'D. Delta.'],
    hierarchical: {prediction: 'D. Alpha. I used [0.0–3.0s].', pred_letter: 'D', ground_truth: 'A', correct: false},
    baseline: {prediction: 'B. Beta.', pred_letter: 'B', ground_truth: 'A', correct: false}}],
  snapshots: [
    {time: 3, phase: 'stream', recent: ['w0'], pending: [], episodic: [], events: []},
    {time: 6, phase: 'stream', recent: ['w0', 'w1'], pending: [], episodic: [], events: []},
    {time: 9, phase: 'stream', recent: ['w1', 'w2'], pending: ['w0'], episodic: [], events: []},
    {time: 12, phase: 'stream', recent: ['w2', 'w3'], pending: ['w0', 'w1'], episodic: [], events: []},
    {time: 15, phase: 'stream', recent: ['w3', 'w4'], pending: ['w0', 'w1'], episodic: [], events: []},
    {time: 15, phase: 'question', recent: ['w3', 'w4'], pending: [], episodic: ['e1'], events: []},
    {time: 18, phase: 'stream', recent: ['w4', 'w5'], pending: [], episodic: ['e1'], events: []},
    {time: 18, phase: 'finalize', recent: [], pending: [], episodic: [], events: []},
  ],
  decisions: [
    {window_id: 'w0', promoted: true, max_similarity: 0.5, threshold: 0.05, time: 9, phase: 'stream'},
    {window_id: 'w1', promoted: true, max_similarity: 0.6, threshold: 0.05, time: 12, phase: 'stream'},
    {window_id: 'w2', promoted: false, max_similarity: 0.97, threshold: 0.05, time: 15, phase: 'stream'},
  ],
  retrievals: [{question_id: 'q', time: 15, coarse: [], episodic: [{id: 'e1', score: 0.12}], grounded: [{id: 'w1', score: null}, {id: 'w4', score: 0.3}], pinned: [{id: 'w3', score: null}, {id: 'w4', score: null}]}],
  baseline_inputs: [{question_id: 'q', time: 15, grounded: [{id: 'w4', score: 0.2}], pinned: []}],
  annotation: {questions: {q: {answer: 'A', evidence_stimulus_seconds: [[3, 6]]}}},
};
nodes.w2 = W(2);

test('question, retrieval and baseline input are not displayed early, including window-processing delay', () => {
  for (const t of [0, 14, 14.99]) {
    const s = stateAt(session, t);
    assert.equal(s.question, null); assert.equal(s.retrieval, null); assert.equal(s.baseline, null);
  }
  assert.equal(stateAt(session, 15).question.question_id, 'q');
  assert.equal(stateAt(session, 15).retrieval.question_id, 'q');
});

test('seeking backwards never retains future memory, answers or retrieval', () => {
  assert.equal(stateAt(session, 17).snapshot.time, 15);
  const past = stateAt(session, 7);
  assert.equal(past.snapshot.time, 6); assert.equal(past.question, null); assert.equal(past.retrieval, null);
});

test('end-of-stream finalize() is not part of the streaming timeline', () => {
  assert.ok(streamSnapshots(session).every(s => s.phase !== 'finalize'));
  assert.deepEqual(stateAt(session, 18).snapshot.recent, ['w4', 'w5']);
});

test('the window ledger only contains windows that had arrived, in their recorded tier', () => {
  const at7 = windowLedger(session, 7);
  assert.deepEqual(at7.map(w => w.id), ['w0', 'w1']);
  assert.ok(at7.every(w => w.end <= 7));
  const at16 = Object.fromEntries(windowLedger(session, 16).map(w => [w.id, w]));
  assert.equal(at16.w0.state, 'episodic'); assert.equal(at16.w0.parent, 'e1');
  assert.equal(at16.w2.state, 'discarded');
  assert.equal(at16.w3.state, 'recent');
  assert.equal(at16.w5, undefined);
});

test('evidence is a returned node outside recent memory, preferring the annotated moment', () => {
  const ev = evidenceFor(session, session.retrievals[0], 'q');
  assert.equal(ev.id, 'e1'); assert.equal(ev.annotated, true);
  assert.ok(ev.node.end <= ev.recentStart);
  const t = evidenceTime(session, ev, 'q');
  assert.ok(t >= 3 && t <= 6, 'evidence frame must come from the representative window overlapping the annotation');
  const noAnnotation = {...session, annotation: undefined};
  assert.equal(evidenceFor(noAnnotation, session.retrievals[0], 'q').id, 'e1');
  const nothingOld = {...session.retrievals[0], episodic: [], grounded: [{id: 'w4', score: 0.3}]};
  assert.equal(evidenceFor(session, nothingOld, 'q'), null, 'a recent hit is not long-range evidence');
});

test('the guided tour never shows the question before its evaluation time and keeps live speeds playable', () => {
  const board = buildStoryboard(session, 'q');
  for (let e = 0; e <= board.duration; e += 0.05) {
    const p = presentationAt(board, e);
    assert.ok(p.time <= 15 + 1e-9);
    if (['question', 'retrieve', 'evidence', 'answer', 'outro'].includes(p.stage)) assert.equal(p.time, 15);
    else assert.ok(stateAt(session, p.time - 1e-9).question === null || p.time >= 15);
    assert.ok(p.showVideoAt <= p.time + 1e-9, 'the video never shows the future');
  }
  for (const s of board.segments) if (s.s1 > s.s0) assert.ok((s.s1 - s.s0) / (s.t1 - s.t0) <= 16, `${s.stage} faster than the media element can play`);
});

test('answers are read verbatim, and a letter/text mismatch is detected', () => {
  const h = answerReading(session.records[0], 'hierarchical');
  assert.equal(h.letter, 'D'); assert.equal(h.named, 'A'); assert.equal(h.consistent, false);
  assert.equal(h.headline, 'D. Alpha.');
  const b = answerReading(session.records[0], 'baseline');
  assert.equal(b.consistent, true); assert.equal(b.headline, 'B. Beta.');
  const letterOnly = {...session.records[0], baseline: {...session.records[0].baseline, prediction: 'C\nBecause of [3.0–6.0s].', pred_letter: 'C'}};
  assert.equal(answerReading(letterOnly, 'baseline').headline, 'C');
});

test('annotated-moment narration is derived from the recorded ledger', () => {
  assert.equal(annotatedMomentAt(session, 'q', 2).state, 'future');
  assert.equal(annotatedMomentAt(session, 'q', 7).state, 'recent');
  const later = annotatedMomentAt(session, 'q', 16);
  assert.equal(later.state, 'kept'); assert.deepEqual(later.parents, ['e1']);
});

test('time formatting carries hundredths correctly', () => {
  assert.equal(formatTime(602.04, true), '10:02.04');
  assert.equal(formatTime(601.999, true), '10:02.00');
  assert.equal(formatTime(0), '00:00');
});
