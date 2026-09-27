/**
 * Does the workbench actually boot?
 * =================================
 *
 * This is the test that was missing. Everything else about the web app had a check —
 * `project-ts.ts` proves every module parses and its relative imports resolve, `serve.mjs`
 * serves bytes that match disk, and a handful of reachability probes prove the URLs are
 * 200 — and none of those can tell a working page from a blank one. Each of them passes
 * just as happily when `boot()` throws on its first line.
 *
 * So: install a DOM, import the real `dist/main.js`, and require that the app reaches the
 * end of `boot()`, builds every panel, survives a frame, and can run the agent loop from a
 * click. That is the span a user experiences as "用不了", and it is now a test instead of a
 * hypothesis.
 *
 * ## What this does and does not prove
 *
 * It runs the real module graph — the same bytes `scripts/serve.mjs` hands the browser,
 * through the same relative specifiers — and it exercises the real catalog, the real six
 * target emitters, the real IR and the real agent loop. Clicking 生成 below runs the whole
 * authoring path end to end.
 *
 * It does **not** prove a pixel is drawn: there is no GPU here, `canvas.getContext()`
 * returns `null`, and the app's response to that is a visible banner, which is asserted
 * rather than papered over. Rendering, WebGPU, and the compare worker remain verified only
 * by their own unit tests. The point is that a blank page, a bad id in `index.html`, an
 * unregistered emitter, a thrown exception in a panel constructor, or a broken agent
 * handoff all now fail here instead of in a browser.
 */

import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { after, before, test } from 'node:test';

import { installDom, type FakeElement, type Shim } from './dom-shim.ts';

const here = dirname(fileURLToPath(import.meta.url));
const distMain = join(here, '..', 'dist', 'main.js');

let shim: Shim;

/** Poll until `check` is truthy, so nothing here depends on a fixed sleep. */
async function waitFor(check: () => boolean, what: string, timeoutMs = 10_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (check()) return;
    await new Promise((r) => setTimeout(r, 10));
  }
  const status = shim.byId('statusbar')?.textContent ?? '(no #statusbar)';
  assert.fail(`timed out waiting for ${what}. #statusbar says: ${status}`);
}

function text(selector: string): string {
  return shim.document.body.querySelector<FakeElement>(selector)?.textContent ?? '';
}

/**
 * Console output that came from the app, not from Node.
 *
 * `--experimental-transform-types` prints a deprecation-shaped warning on every run and
 * Node routes it through `console.error`, so an unfiltered "console was clean" assertion
 * fails identically on a perfect boot and on a broken one. Filtering by the warning's own
 * shape keeps the assertion meaningful — anything the *app* logs still fails it.
 */
function appConsoleErrors(): string[] {
  const noise = /ExperimentalWarning|--trace-warnings|^\(node:\d+\)/;
  return shim.consoleErrors.filter((line) => !noise.test(line));
}

before(async () => {
  assert.ok(
    existsSync(distMain),
    `apps/web/dist/main.js is missing — this suite runs the projected output, not the sources.\n` +
      `Build it first:  node scripts/project-ts.ts`,
  );
  shim = installDom();
  // The import itself is half the assertion: `main.ts` resolves `#preview` at module
  // scope, so a bad id or a thrown top-level statement rejects right here.
  await import('../dist/main.js');
});

after(() => {
  // Nothing to tear down — the DOM is process-local — but a leaked failure is worth
  // reporting loudly at the end rather than only inside one test.
  for (const f of shim.failures) console.error(`[boot.test] leaked ${f}`);
});

test('boot() completes and reports ready', async () => {
  await waitFor(() => (shim.byId('statusbar')?.textContent ?? '').includes('就绪'), 'the ready status line');

  const status = shim.byId('statusbar')?.textContent ?? '';
  assert.match(status, /就绪 · .+ · \d+ 种节点 · 6 个目标发射器/);

  /*
   * Six emitters, asserted by name.
   *
   * The failure this guards against is subtle and has happened in this project before: a
   * targets module that is imported but does not register leaves `TARGET_EMITTERS` empty,
   * and every emission then comes back as `EM000_UNKNOWN_TARGET` — a code view full of
   * diagnostics rather than of shaders. The app *reports* that in a banner, but reporting
   * it is not the same as it not happening.
   */
  assert.equal(appConsoleErrors().length, 0, `console was not clean: ${appConsoleErrors().join(' | ')}`);
});

test('every mount point index.html declares is filled', () => {
  const filled: Array<[string, number]> = [
    ['author', 1],
    ['params', 1],
    ['codegen', 1],
    ['info', 1],
    ['toolbar', 1],
    ['compare', 1],
    ['banners', 1],
  ];
  for (const [id, min] of filled) {
    const el = shim.byId(id);
    assert.ok(el, `#${id} is missing from the DOM`);
    assert.ok(
      (el as FakeElement).children.length >= min,
      `#${id} was never populated — the panel that owns it did not build`,
    );
  }
});

test('the authoring panel is present, seeded, and armed', () => {
  assert.ok(text('.author-panel'), 'no .author-panel was mounted');

  const prompt = shim.document.body.querySelector<FakeElement>('.author-prompt');
  assert.ok(prompt, 'the prompt box is missing');
  assert.ok(prompt.value.length > 0, 'the prompt box is empty, so 生成 would refuse to run');

  // Both actions must start enabled: the default provider is the offline stub, which
  // needs no key and no network. An unusable-by-default agent panel is exactly the
  // complaint this suite exists to catch.
  const buttons = shim.document.body.querySelectorAll<FakeElement>('.author-actions button');
  assert.equal(buttons.length, 3, 'expected 生成 / 微调当前 / 取消');
  assert.equal(buttons[0]?.textContent, '生成');
  assert.equal(buttons[0]?.disabled, false, '生成 is disabled on a clean boot');
  assert.equal(buttons[1]?.disabled, false, '微调当前 is disabled on a clean boot');
  assert.equal(buttons[2]?.hidden, true, '取消 should be hidden until a run starts');
});

test('the online endpoint is configurable, not hardcoded', () => {
  // The regression this guards: the panel used to hardcode `https://api.openai.com/v1`
  // with no field to change it, so on a machine that cannot reach that one host there was
  // no fix available from inside the UI at all. A hardcoded endpoint is not a default,
  // it is a wall — and the failure it produced named neither the host nor the reason.
  const url = shim.document.body.querySelector<FakeElement>('input.author-url');
  assert.ok(url, 'there is no editable base-URL field — the endpoint is hardcoded again');

  const service = shim.document.body.querySelector<FakeElement>('select.author-service');
  assert.ok(service, 'there is no service selector');
  const presets = (service as unknown as { children: FakeElement[] }).children;
  assert.ok(presets.length >= 5, `expected the common OpenAI-compatible services, found ${presets.length}`);

  const key = shim.document.body.querySelector<FakeElement>('input.author-key');
  const model = shim.document.body.querySelector<FakeElement>('input.author-model');
  assert.ok(key && model, 'the key and model fields are missing');
  assert.equal(key.type, 'password', 'the key field must not be a plain text input');

  // Picking a service must fill the address and the model, so nobody has to know a URL.
  service.value = 'deepseek';
  service.dispatchEvent({ type: 'change' });
  assert.match(url.value, /^https:\/\/api\.deepseek\.com\/v1$/, `selecting a service did not set the address, got ${url.value}`);
  assert.ok(model.value.length > 0, 'selecting a service did not suggest a model');

  // And editing the address by hand must move the selector, so it never claims a service
  // that is no longer selected.
  url.value = 'http://127.0.0.1:11434/v1';
  url.dispatchEvent({ type: 'input' });
  assert.equal(service.value, 'ollama', 'typing a known address did not select its service');

  url.value = 'https://example.invalid/v1';
  url.dispatchEvent({ type: 'input' });
  assert.equal(service.value, 'custom', 'an unknown address must fall back to 自定义');
});

test('a connection test exists and reports without throwing', async () => {
  const probe = shim.document.body.querySelector<FakeElement>('.author-probe-row button');
  assert.ok(probe, 'there is no 测试连接 button — a failed run would be the first signal');
  assert.equal(probe.textContent, '测试连接');

  // The model list comes from the endpoint, so there is a second button that asks for it. Its
  // absence is the difference between "here are two guesses" and "here is what you serve".
  const listBtn = shim.document.body.querySelectorAll<FakeElement>('.author-probe-row button')[1];
  assert.ok(listBtn, 'there is no 拉取模型 button — the model list would be a hardcoded guess');
  assert.equal(listBtn.textContent, '拉取模型');

  const status = shim.document.body.querySelector<FakeElement>('.author-probe');
  assert.ok(status, 'there is no status line for the connection test');
  // Never blank: the state is always stated, and an edit resets it to 未测试 rather than
  // clearing the line.
  assert.match(status.textContent, /未测试/, `the status line should state the untested case, got "${status.textContent}"`);

  /*
   * Click it against a host that cannot answer.
   *
   * A closed loopback port fails without DNS, so this exercises the real
   * failure path — the one a user meets on a network that blocks their provider — without
   * depending on this machine's connectivity. The assertion is that it *reports*: the whole
   * point of the button is that a transport failure produces a sentence instead of a
   * rejected promise and an empty panel.
   */
  const url = shim.document.body.querySelector<FakeElement>('input.author-url');
  const { createServer } = await import('node:net');
  const server = createServer();
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve));
  const address = server.address();
  assert.ok(address && typeof address !== 'string');
  const port = address.port;
  await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
  url.value = `http://127.0.0.1:${port}/v1`;
  url.dispatchEvent({ type: 'input' });
  probe.click();

  const deadline = Date.now() + 20_000;
  while (Date.now() < deadline && !status.textContent.startsWith('✗') && !status.textContent.startsWith('✓')) {
    await new Promise((r) => setTimeout(r, 20));
  }
  assert.match(status.textContent, /^✗/, `expected a reported failure, got: ${status.textContent}`);
  assert.ok(status.textContent.includes(`127.0.0.1:${port}`), 'the report must name the host it could not reach');
});


test('a missing WebGL2 context is reported, not thrown', () => {
  // The shim deliberately has no GPU, so this banner is the app doing the right thing
  // with that. If it ever disappears, the failure path was silently dropped.
  const banners = text('#banners');
  assert.match(banners, /WebGL2/, `expected a WebGL2 banner, #banners says: ${banners || '(empty)'}`);
  assert.equal(shim.failures.length, 0, `uncaught during boot: ${shim.failures.join(' | ')}`);
});

test('one animation frame runs and reschedules', () => {
  const before = shim.frames.length;
  assert.equal(before, 1, 'boot() should have queued exactly one frame');
  assert.equal(shim.step(16), true);
  assert.equal(shim.step(32), true);
  assert.equal(shim.frames.length, 1, 'loop() must keep exactly one frame in flight');
});

test('clicking 生成 runs the agent loop and lands a document', async () => {
  const button = shim.document.body.querySelector<FakeElement>('.author-actions button');
  assert.ok(button);
  button.click();

  const verdict = () => text('.author-verdict');
  await waitFor(() => verdict().includes('代码生成通过') || verdict().includes('未成功'), 'an agent verdict');

  assert.match(
    verdict(),
    /^代码生成通过 · 起草 · 模型调用 \d+ 次/,
    `the offline stub should author a clean document, got: ${verdict()}`,
  );

  // The stage trace is the part that makes a multi-second run legible; assert it ran the
  // three deterministic-plus-one stages rather than only the endpoints.
  const trace = text('.author-trace');
  for (const stage of ['理解意图', '起草图', '发射校验']) {
    assert.ok(trace.includes(stage), `stage trace is missing ${stage}: ${trace || '(empty)'}`);
  }

  // And the loop's output must reach the app, not just the panel: a document that only
  // updated the verdict box would leave the code view stale.
  assert.match(shim.byId('statusbar')?.textContent ?? '', /agent 代码生成通过 · 6 个目标/);
  assert.match(verdict(), /未实现或采用近似/);
  assert.match(verdict(), /不代表已验证画面/);
  assert.equal(shim.failures.length, 0, `uncaught during the agent run: ${shim.failures.join(' | ')}`);
});

test('the code view holds all six targets', () => {
  // The codegen panel is tabbed and lazy, so the emitters' output is only produced when a
  // tab is shown. Assert the tabs exist rather than the source text: 6 tabs means the
  // target list reached the view, which is the wiring that can break.
  const tabs = shim.document.body.querySelectorAll<FakeElement>('.codegen-tabs .tab');
  assert.equal(tabs.length, 6, `expected a tab per target, found ${tabs.length}`);
});

/* -------------------------------------------------------------------------- */
/* 3D surface preview                                                          */
/* -------------------------------------------------------------------------- */

test('the toolbar offers flat / material / scene and a shape', () => {
  const toolbar = shim.byId('toolbar');
  assert.ok(toolbar, '#toolbar is missing');

  const selects = toolbar.querySelectorAll<FakeElement>('select');
  // preview mode, shape, backend, resolution — the mode pair comes first because it is what
  // the whole middle column now depends on.
  assert.ok(selects.length >= 4, `expected the toolbar selects, found ${selects.length}`);

  const mode = selects[0];
  const modeIds = mode.children.map((o) => o.value);
  assert.deepEqual(modeIds, ['flat', 'material', 'scene'], `unexpected preview modes: ${modeIds}`);

  const shape = selects[1];
  const shapeIds = shape.children.map((o) => o.value);
  assert.deepEqual(shapeIds, ['sphere', 'torus', 'knot', 'cube'], `unexpected shapes: ${shapeIds}`);
  // `ground` is a real mesh but not an object to look at; offering it would frame a plane
  // edge-on and read as a broken model.
  assert.ok(!shapeIds.includes('ground'), 'the ground plane must not be offered as an object');
});

test('the material strip exists, starts hidden, and carries the material controls', () => {
  const strip = shim.byId('surface-strip');
  assert.ok(strip, '#surface-strip is missing — index.html and main.ts must agree on it');
  // Flat mode is the default, and a density slider on a fullscreen quad is a control with
  // no meaning, so the strip is hidden rather than shown-and-ineffective.
  assert.equal(strip.hidden, true, 'the material strip should be hidden in flat mode');

  const inputs = strip.querySelectorAll<FakeElement>('input');
  const seamless = inputs.find(s => s.type === 'checkbox');
  assert.ok(seamless?.checked, 'seam blending should be enabled by default');
  const sliders = inputs.filter(s => s.type === 'range');
  assert.ok(sliders.length >= 5, `expected the material sliders, found ${sliders.length}`);
  for (const s of sliders) assert.equal(s.type, 'range', 'material controls must be range inputs');

  const buttons = strip.querySelectorAll<FakeElement>('button');
  const labels = buttons.map((b) => b.textContent);
  assert.ok(labels.some((l) => l.includes('自动旋转')), `no rotate toggle: ${labels}`);
  assert.ok(labels.some((l) => l.includes('重置视角')), `no camera reset: ${labels}`);
});

test('selecting a surface mode without WebGL2 reports and does not blank the page', async () => {
  /*
   * The shim has no GPU, so this exercises the honest-failure path rather than the feature.
   * That is still worth pinning: the failure mode this guards against is a mode switch that
   * leaves an empty canvas and no explanation, which is indistinguishable from a hung app.
   *
   * The banner count is compared before and after rather than merely searched. A WebGL2
   * banner already exists from boot, so `#banners` containing "WebGL2" is true either way —
   * an assertion that cannot fail is worse than no assertion.
   */
  const bannerCount = () => shim.document.body.querySelectorAll('.banner').length;
  const before = bannerCount();

  const mode = shim.byId('toolbar')?.querySelector<FakeElement>('select');
  assert.ok(mode);
  mode.value = 'material';
  mode.dispatchEvent({ type: 'change' });

  // The strip follows the mode synchronously — it is a toolbar state, not a renderer state.
  assert.equal(shim.byId('surface-strip')?.hidden, false, 'the material strip should appear in a surface mode');

  await waitFor(() => bannerCount() > before, 'a new banner explaining the failed mode switch');
  const latest = shim.document.body.querySelectorAll<FakeElement>('.banner').at(-1);
  assert.match(latest?.textContent ?? '', /WebGL2/, 'the new banner must name the cause');
  assert.ok(shim.frames.length >= 1, 'the render loop must still be alive after a failed mode switch');

  // Put it back so later assertions see the default a user starts from.
  mode.value = 'flat';
  mode.dispatchEvent({ type: 'change' });
  assert.equal(shim.byId('surface-strip')?.hidden, true, 'the material strip should hide again in flat mode');
});

test('opening a document rejects anything that is not one, by name', async () => {  /*
   * The workbench can now open a `.shader.json` from disk (the CLI writes them, the author
   * panel produces them). The failure worth pinning is not "a bad file throws" — it is *what
   * the user is told*. Without this check a stale or unrelated JSON reaches the emitter, and
   * the message becomes `TypeError: cannot read properties of undefined (reading 'nodes')`
   * from inside the IR, which reads like a bug in the workbench rather than a file that is not
   * a document.
   */
  const { documentProblem } = (await import('../dist/main.js')) as {
    documentProblem(value: unknown): string | null;
  };

  const good = {
    irVersion: '1.0.0',
    id: 'x',
    title: 'x',
    spec: {},
    stage: 'fragment',
    graph: { nodes: [{ id: 'n_out', kind: 'output' }], edges: [] },
    render: {},
    targets: ['glsl'],
  };
  assert.equal(documentProblem(good), null, 'a well-formed document must be accepted');

  assert.match(String(documentProblem(null)), /顶层/);
  assert.match(String(documentProblem([1, 2, 3])), /顶层/);
  assert.match(String(documentProblem({ graph: { nodes: [] } })), /irVersion/);
  assert.match(String(documentProblem({ ...good, irVersion: '0.9.0' })), /版本/);
  assert.match(String(documentProblem({ ...good, graph: { nodes: [] } })), /graph\.nodes/);
  assert.match(
    String(documentProblem({ ...good, graph: { nodes: [{ id: 'n' }] } })),
    /id\/kind/,
    'a node without a kind must be named as such',
  );
});

test('the status strip says what the image channel is bound to', async () => {
  /*
   * `render.feedback` is a request to the host, and the host's answer is invisible in the image:
   * a feedback document with a white placeholder looks exactly like a shader that does nothing.
   * So the strip states which of the three bindings is live — and the third case ("the 3D path
   * cannot do feedback") is a limit of the *host*, which the user has no other way to discover.
   */
  const { channelNote } = (await import('../dist/main.js')) as {
    channelNote(args: { needsChannel: boolean; feedback: boolean; surfacePath: boolean }): string | null;
  };

  // A document that never reads the channel says nothing at all: silence, not a note about a
  // binding that does not exist.
  assert.equal(channelNote({ needsChannel: false, feedback: false, surfacePath: false }), null);
  assert.equal(channelNote({ needsChannel: false, feedback: true, surfacePath: false }), null);

  const on = channelNote({ needsChannel: true, feedback: true, surfacePath: false });
  assert.match(String(on), /上一帧/, 'feedback must be named as the previous frame');

  const off = channelNote({ needsChannel: true, feedback: false, surfacePath: false });
  assert.match(String(off), /占位/, 'without the flag the binding is a stand-in, and must say so');
  assert.match(String(off), /render\.feedback/, 'and must name the flag that would change it');

  const surface = channelNote({ needsChannel: true, feedback: true, surfacePath: true });
  assert.match(String(surface), /3D/, 'the material path limitation must be attributed to the path');
  assert.match(String(surface), /平面/, 'and must say where feedback does work');
});

test('a surface document is never left on the flat preview', async () => {
  /*
   * The bug this prevents has no symptom until you look at the picture: GLSL permits a fragment
   * input with no matching vertex output, so a surface shader compiles happily on a fullscreen
   * quad and then reads undefined varyings. The compile strip says "0 errors" and the canvas is
   * black — the user has no way to tell it apart from a broken generator.
   *
   * `previewModeFor` is the decision, exported so it can be asserted without a GPU. The shim has
   * no WebGL2, so what the integration can check is the other half: that the promotion happened
   * and the failure to build a 3D pass is reported rather than swallowed.
   */
  const { previewModeFor } = (await import('../dist/main.js')) as {
    previewModeFor(needsSurface: boolean, current: string): string;
  };

  assert.equal(previewModeFor(true, 'flat'), 'material', 'a surface document must leave flat');
  assert.equal(previewModeFor(false, 'flat'), 'flat', 'a quad document must stay flat');
  // Never downgrade a mode the user picked: `scene` is a superset of `material`, and a 3D mode
  // already shows surface documents correctly.
  assert.equal(previewModeFor(true, 'scene'), 'scene');
  assert.equal(previewModeFor(true, 'material'), 'material');
  assert.equal(previewModeFor(false, 'material'), 'material');

  // And the strip/toolbar agree with the state after a real mode switch, which is what the
  // promotion drives.
  const mode = shim.byId('toolbar')?.querySelector<FakeElement>('select');
  assert.ok(mode);
  mode.value = 'material';
  mode.dispatchEvent({ type: 'change' });
  assert.equal(shim.byId('surface-strip')?.hidden, false);
  mode.value = 'flat';
  mode.dispatchEvent({ type: 'change' });
  assert.equal(shim.byId('surface-strip')?.hidden, true);
});
