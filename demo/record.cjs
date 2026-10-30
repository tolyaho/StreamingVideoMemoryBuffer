#!/usr/bin/env node
// Real browser QA and deterministic capture of the guided tour. Never synthesizes model output,
// never swaps footage: every frame is the actual app rendering the recorded run at tour time t,
// after the <video> element has decoded the bound source at the position the tour asks for.
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const cp = require('node:child_process');
const assert = require('node:assert/strict');
const {chromium} = require('playwright');
const args = process.argv.slice(2);
const option = (name, fallback) => args.includes(name) ? args[args.indexOf(name) + 1] : fallback;
const options = name => args.flatMap((a, i) => a === name ? [args[i + 1]] : []);
const root = path.resolve(__dirname, '..');
const out = path.resolve(option('--out', path.join(__dirname, 'recordings')));
const python = option('--python', 'python3');
const fps = Number(option('--fps', '30'));
const checkOnly = args.includes('--check-only');
const showcase = args.includes('--showcase');
let server, browser;

async function localServer() {
  const serverArgs = ['demo/server.py', '--port', '0'];
  for (const t of options('--trace')) serverArgs.push('--trace', path.resolve(t));
  for (const key of ['--media', '--rights', '--annotation']) if (args.includes(key)) serverArgs.push(key, path.resolve(option(key)));
  if (args.includes('--tour-question')) serverArgs.push('--tour-question', option('--tour-question'));
  server = cp.spawn(python, serverArgs, {cwd: root, env: {...process.env, PYTHONDONTWRITEBYTECODE: '1'}, stdio: ['ignore', 'pipe', 'pipe']});
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(Error('Local demo server did not start')), 15000);
    server.stdout.on('data', chunk => { const m = chunk.toString().match(/http:\/\/127\.0\.0\.1:\d+/); if (m) { clearTimeout(timer); resolve(m[0]); } });
    server.stderr.on('data', chunk => process.stderr.write(chunk));
    server.once('exit', code => { clearTimeout(timer); reject(Error('Local demo server exited: ' + code)); });
  });
}

async function open(context, url) {
  const page = await context.newPage(), errors = [], external = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push(m.text()); });
  page.on('request', r => { if (!r.url().startsWith(url) && !r.url().startsWith('data:')) external.push(new URL(r.url()).origin); });
  const response = await page.goto(url, {waitUntil: 'networkidle'}); assert.equal(response.status(), 200);
  await page.waitForFunction(() => window.demo?.ready);
  await page.waitForFunction(() => { const v = document.getElementById('video'); return v.hidden || v.readyState >= 2; });
  return {page, errors, external};
}

async function checks(context, url) {
  const {page, errors, external} = await open(context, url);
  const runs = await (await context.request.get(url + '/api/runs')).json();
  const results = {runs: []};
  for (const run of runs) {
    await page.evaluate(i => window.demo.loadRun(i), run.index);
    const data = await (await context.request.get(`${url}/api/session/${run.index}`)).json();
    assert.ok(data.records.length > 0);
    const q = data.records.find(r => r.question_id === (data.tour_question_id ?? data.focus_question_id)) ?? data.records[0];
    // nothing about a question is visible before its recorded evaluation time
    await page.evaluate(t => window.demo.seek(t), Math.max(0, q.stream_time - 0.01));
    await page.evaluate(() => window.demo.setTab('ask', true));
    const early = await page.locator('#q-text').innerText().catch(() => '');
    assert.notEqual(early.trim(), q.question, 'Question shown before its evaluation time');
    await page.evaluate(t => window.demo.seek(t), q.stream_time);
    assert.equal((await page.locator('#q-text').innerText()).trim(), q.question);
    // answers are the verbatim saved outputs
    for (const [side, key] of [['hierarchical', 'h'], ['baseline', 'b']]) {
      const shown = ((await page.locator(`#ans-${key}-l`).evaluate(n => n.firstChild?.textContent ?? '')) + ' ' + (await page.locator(`#ans-${key}-why`).innerText())).replace(/\s+/g, ' ').trim();
      assert.equal(shown, q[side].prediction.replace(/\s+/g, ' ').trim(), `${side} answer is not verbatim`);
    }
    if (data.mode === 'recorded_trace') {
      // the ribbon never draws a window that had not arrived yet
      for (const t of [5, q.stream_time, data.clock_end]) {
        await page.evaluate(x => window.demo.seek(x), t);
        const drawn = await page.locator('#ribbon rect[data-id]').evaluateAll(els => els.map(e => e.dataset.id));
        const future = drawn.filter(id => data.nodes[id] && data.nodes[id].end > t + 1e-6);
        assert.deepEqual(future, [], `future memory drawn at ${t}`);
      }
      // seeking backwards restores the earlier state
      await page.evaluate(t => window.demo.seek(t), q.stream_time);
      await page.evaluate(() => window.demo.seek(10));
      assert.equal(await page.locator('#ask-body').isVisible(), false, 'answer retained after seeking back');
      assert.equal(await page.locator('#ribbon path[stroke-dasharray]').count(), 0, 'retrieval arcs retained after seeking back');
      if (data.media.available) {
        await page.evaluate(t => window.demo.seek(t), q.stream_time);
        await page.evaluate(() => window.demo.setTab('ask', true));
        const lead = page.locator('.hit-row.lead');
        if (await lead.count()) {
          await lead.click();
          await page.waitForTimeout(400);
          const s = await page.evaluate(() => ({clock: window.demo.state().clock, video: document.getElementById('video').currentTime}));
          assert.ok(Math.abs(s.clock - q.stream_time) < 1e-6, 'memory clock moved when showing evidence');
          assert.ok(s.video < q.stream_time - 60, 'evidence jump did not reach back past the recent window');
        }
      }
    } else {
      assert.equal(await page.locator('#nomedia').isVisible(), true);
      assert.match(await page.locator('#ret-list').innerText(), /not save/);
    }
    results.runs.push({config: data.config, mode: data.mode, media: data.media.available, questions: data.records.length});
  }
  await page.evaluate(() => window.demo.provenance());
  assert.equal(await page.locator('dialog#prov').isVisible(), true);
  await page.getByRole('button', {name: 'Close provenance'}).click();
  await page.evaluate(() => window.demo.loadRun(0));
  const overflow = () => page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
  assert.equal(await overflow(), false, 'Desktop horizontal overflow');
  assert.equal(await page.evaluate(() => document.querySelector('footer').getBoundingClientRect().bottom <= innerHeight), true, 'Desktop composition must fit 1080p');
  await page.evaluate(() => { const s = window.demo.session(); const q = s.records.find(r => r.question_id === (s.tour_question_id ?? s.focus_question_id)) ?? s.records[0]; window.demo.seek(q.stream_time); window.demo.setTab('ask', true); });
  for (const [w, h, name] of [[390, 844, 'mobile'], [768, 1024, 'tablet'], [1280, 800, 'laptop']]) {
    await page.setViewportSize({width: w, height: h});
    await page.waitForTimeout(400);
    assert.equal(await overflow(), false, `Horizontal overflow at ${w}px`);
    await page.screenshot({path: path.join(out, `${name}.png`), fullPage: w < 1000});
  }
  assert.deepEqual(errors, []); assert.deepEqual(external, []);
  await page.close();
  return {...results, desktop: '1920x1080', mobile: '390x844', tablet: '768x1024', laptop: '1280x800', browserErrors: errors, externalRequests: external};
}

async function capture(context, url) {
  const {page, errors, external} = await open(context, url);
  const meta = await page.evaluate(() => { const s = window.demo.session(); return {duration: window.demo.tourDuration(), media: s.media.available, mode: s.mode, config: s.config, question: s.tour_question_id ?? s.focus_question_id, sha: s.media.source_sha256}; });
  if (showcase && (!meta.media || meta.mode !== 'recorded_trace')) throw Error('Showcase blocked: cleared, source-aligned media and an actual recorded trace are required.');
  const frames = fs.mkdtempSync(path.join(os.tmpdir(), 'svmb-frames-'));
  const total = Math.round(meta.duration * fps);
  console.log(`Rendering ${total} frames (${meta.duration.toFixed(1)} s at ${fps} fps)…`);
  const stages = {};
  for (let i = 0; i < total; i++) {
    const r = await page.evaluate(t => window.demo.renderTourFrame(t), i / fps);
    stages[r.stage] ??= i / fps;
    await page.screenshot({path: path.join(frames, `${String(i).padStart(5, '0')}.jpg`), type: 'jpeg', quality: 94});
    if (i % 150 === 0) console.log(`  frame ${i}/${total} · ${r.stage} · stream ${r.clock.toFixed(2)} s · video ${r.videoAt.toFixed(2)} s`);
  }
  const posterAt = Number(option('--poster-at', String(stages.answer + 4.6)));
  await page.evaluate(t => window.demo.renderTourFrame(t), posterAt);
  await page.screenshot({path: path.join(out, 'poster.png')});
  assert.deepEqual(errors, []); assert.deepEqual(external, []);
  await page.close();
  const name = showcase ? 'svmb-showcase.mp4' : 'svmb-tour.mp4', target = path.join(out, name);
  cp.execFileSync('ffmpeg', ['-y', '-v', 'error', '-framerate', String(fps), '-i', path.join(frames, '%05d.jpg'), '-an', '-vf', 'scale=in_range=full:out_range=tv,format=yuv420p', '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-color_range', 'tv', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-movflags', '+faststart', target], {stdio: 'inherit'});
  fs.rmSync(frames, {recursive: true, force: true});
  const probe = JSON.parse(cp.execFileSync('ffprobe', ['-v', 'error', '-show_format', '-show_streams', '-of', 'json', target], {encoding: 'utf8'}));
  const stream = probe.streams.find(s => s.codec_type === 'video');
  assert.equal(stream.width, 1920); assert.equal(stream.height, 1080); assert.equal(stream.codec_name, 'h264'); assert.equal(stream.pix_fmt, 'yuv420p');
  assert.ok(Number(probe.format.duration) >= 35 && Number(probe.format.duration) <= 50, 'duration outside 35–50 s');
  assert.equal(probe.streams.some(s => s.codec_type === 'audio'), false);
  fs.writeFileSync(path.join(out, 'recording.json'), JSON.stringify({
    kind: showcase ? 'source_aligned_showcase' : 'tour_capture', file: name, duration: Number(probe.format.duration), width: stream.width, height: stream.height,
    fps: stream.r_frame_rate, codec: stream.codec_name, frames: total, run_config: meta.config, tour_question: meta.question, source_sha256: meta.sha, stage_start_seconds: stages, poster_at_seconds: posterAt,
    method: 'Deterministic: each frame is window.demo.renderTourFrame(t) followed by a screenshot once the <video> element has decoded the requested source time.',
    notes: 'Playback speeds, pauses, the jump back to the evidence frame and reveal timing are presentation choices, not inference timing.'}, null, 2));
  console.log('Recorded:', target);
}

async function main() {
  fs.mkdirSync(out, {recursive: true});
  const url = await localServer();
  browser = await chromium.launch({headless: true});
  const qa = await browser.newContext({viewport: {width: 1920, height: 1080}, reducedMotion: 'reduce'});
  const results = await checks(qa, url); await qa.close();
  fs.writeFileSync(path.join(out, 'browser-checks.json'), JSON.stringify(results, null, 2));
  console.log('Browser checks passed:', JSON.stringify(results));
  if (checkOnly) return;
  const context = await browser.newContext({viewport: {width: 1920, height: 1080}, deviceScaleFactor: 1});
  await capture(context, url); await context.close();
}
main().catch(error => { console.error(error.stack || error.message); process.exitCode = 1; }).finally(async () => { if (browser) await browser.close(); if (server) server.kill('SIGTERM'); });
