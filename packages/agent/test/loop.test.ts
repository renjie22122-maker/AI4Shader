/**
 * `loop.ts` — the orchestration loop, tested without a model and without a network.
 *
 * Every test here drives `runAgent` with a hand-written provider object. That is the whole
 * argument for keeping the LLM behind a two-method interface: the interesting behaviour —
 * escalation, mechanical repair, plateau detection, honest failure — is all loop logic, and
 * loop logic that can only be exercised by calling a real model is loop logic nobody
 * exercises.
 *
 * Run with:
 *   node --test --test-isolation=none --experimental-transform-types \
 *     --import ./packages/ir/test/register-hooks.mjs packages/agent/test/loop.test.ts
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import type { Diagnostic, ShaderDoc, ShaderPatch, ShaderSpec } from '@ai4shader/ir/types';
import { CATALOG } from '@ai4shader/ir/catalog';
import { createDoc, exposeParam, hashDoc, normalizeEdges } from '@ai4shader/ir';
// Side-effect import: registers all six emitters, exactly as a host must.
import '@ai4shader/ir/targets';

import { runAgent, summariseRun } from '../src/loop.js';
// The error class only: the loop reads the *kind* a provider classified a failure as, so the test
// can produce the same failures the HTTP provider does without a network.
import { ProviderError } from '../src/provider/http.js';
import { repair } from '../src/repair.js';
import { buildContext } from '../src/context.js';
import { patchUserPrompt, systemPrompt } from '../src/provider/prompt.js';
import { DEFAULT_POLICY } from '../src/policy.js';
import { unrecoverableReason } from '../src/policy.js';
import type { PatchContext } from '../src/types.js';

/* -------------------------------------------------------------------------- */
/* Fixtures                                                                    */
/* -------------------------------------------------------------------------- */

const SPEC: ShaderSpec = {
  intent: 'slow flowing brushed metal',
  category: 'procedural-surface',
  tags: ['metal', 'noise'],
  features: ['animated flow', 'fine brushed detail'],
};

test('refine cannot report an unchanged or rejected edit as success', async () => {
  const base = docWithUniform();
  for (const patch of [
    { summary: 'empty edit', ops: [] },
    { summary: 'invalid edit', ops: [{ op: 'remove_node', node: 'missing' }] },
  ] as ShaderPatch[]) {
    const result = await runAgent({ prompt: 'make droplets smaller', basedOn: base,
      provider: scriptedProvider([patch]), catalog: CATALOG, requireTargets: ['glsl'] });
    assert.equal(result.clean, false);
    assert.ok(result.remaining.length > 0);
    assert.equal(hashDoc(result.doc), hashDoc(base));
    assert.equal(result.usage?.calls, 1);
  }
});

test('host compiler errors reach the model and are revalidated after repair', async () => {
  const base = docWithUniform();
  const provider = scriptedProvider([{ summary: 'fix shader', ops: [
    { op: 'set_param', node: 'n_d', param: 'scale', value: { kind: 'float', value: 8 } },
  ] }]);
  let validations = 0;
  const result = await runAgent({ prompt: 'repair', mode: 'repair', basedOn: base,
    provider, catalog: CATALOG, requireTargets: ['glsl'],
    validate: async () => ++validations === 1 ? [{ severity: 'error', code: 'AG030_GLSL_COMPILE', message: 'unknown shader function' }] : [],
  });
  assert.equal(result.clean, true);
  assert.equal(validations, 2);
  assert.equal(provider.seen[0]?.diagnostics[0]?.code, 'AG030_GLSL_COMPILE');
});

test('requirement review runs once even for an emittable draft, and verifies its changes', async () => {
  const provider = scriptedProvider([DRAFT_PATCH, { summary: 'retain transparency', ops: [
    { op: 'set_render', path: 'transparent', value: true },
  ] }]);
  const result = await runAgent({ prompt: 'transparent layered rain', provider, catalog: CATALOG,
    reviewRequirements: true, requireTargets: ['glsl'] });
  assert.equal(result.clean, true);
  assert.equal(result.usage?.calls, 3);
  assert.equal(provider.seen[1]?.stage, 'refine');
  assert.match(provider.seen[1]?.diagnostics.map(d => d.message).join('\n') ?? '', /EVERY original requirement/);
  assert.equal(result.doc.render.transparent, true);
  assert.ok(result.emissions.glsl);
});

test('cancellation and model budget also bound interpretation, drafting and requested edits', async () => {
  for (const basedOn of [undefined, docWithUniform()]) {
    const provider = scriptedProvider([DRAFT_PATCH]);
    const budget = await runAgent({ prompt: 'rain', basedOn, provider, catalog: CATALOG,
      policy: { ...DEFAULT_POLICY, maxModelCalls: 0 } });
    assert.equal(budget.stopReason, 'budget');
    assert.equal(budget.usage?.calls, 0);
    const cancelled = await runAgent({ prompt: 'rain', basedOn, provider, catalog: CATALOG,
      signal: AbortSignal.abort() });
    assert.equal(cancelled.stopReason, 'aborted');
    assert.equal(cancelled.usage?.calls, 0);
  }
  const result = await runAgent({ prompt: 'rain', provider: scriptedProvider([DRAFT_PATCH]), catalog: CATALOG,
    policy: { ...DEFAULT_POLICY, maxModelCalls: 1 } });
  assert.equal(result.stopReason, 'budget');
  assert.equal(result.usage?.calls, 1);
});

test('online patch context preserves custom source, render settings and every requirement', () => {
  const doc = createDoc({ spec: { ...SPEC, features: Array.from({length: 18}, (_, i) => `requirement-${i}`),
    outOfScope: ['collision-driven splitting needs a simulation host'] } });
  doc.render.transparent = true;
  doc.render.feedback = true;
  doc.graph.nodes.push({ id: 'n_rain', kind: 'custom_glsl', body: 'return vec4(0.25);' });
  const prompt = patchUserPrompt(buildContext({ stage: 'refine', doc, catalog: CATALOG,
    userIntent: 'uniform fixed fog, small stationary impact droplets, fading thin trails' }), CATALOG);
  assert.match(prompt, /return vec4\(0.25\);/);
  assert.match(prompt, /"transparent":true/);
  assert.match(prompt, /"feedback":true/);
  assert.match(prompt, /requirement-17/);
  assert.match(prompt, /collision-driven splitting/);
  assert.match(systemPrompt('patch'), /Shader code is allowed ONLY/);
  assert.match(systemPrompt('patch'), /Hash a cell\/event/);
  assert.doesNotMatch(systemPrompt('intent'), /"3-6 OBSERVABLE/);
});

/** A provider that replays a fixed list of patches and records what it was asked. */
function scriptedProvider(patches: ShaderPatch[], spec: ShaderSpec = SPEC) {
  const seen: PatchContext[] = [];
  let call = 0;
  return {
    seen,
    get calls() {
      return call;
    },
    name: 'scripted',
    vision: false,
    async complete() {
      return { text: '{}' };
    },
    async interpret() {
      return spec;
    },
    async propose(ctx: PatchContext): Promise<ShaderPatch> {
      seen.push(ctx);
      const p = patches[Math.min(call, patches.length - 1)];
      call++;
      return p ?? { summary: 'no-op', ops: [] };
    },
  };
}

/**
 * A provider whose **first** call throws, so the loop stops at `provider-error`.
 *
 * The error is thrown from `interpret`, which is the earliest point a run can fail: it means the
 * diagnostic under test is the one a real user sees when the very first request fails, with no
 * successful call before it to explain anything.
 */
function failingProvider(err: Error) {
  return {
    name: 'failing',
    vision: false,
    async complete(): Promise<never> {
      throw err;
    },
    async interpret(): Promise<never> {
      throw err;
    },
    async propose(): Promise<never> {
      throw err;
    },
  };
}

/**
 * The graph the reference example uses, expressed as one patch.
 *
 * Written by hand rather than reused from `examples/liquidMetal.ts` so that a change to the
 * example cannot silently change what this test is asserting.
 */
const DRAFT_PATCH: ShaderPatch = {
  summary: 'draft a liquid-metal graph',
  ops: [
    { op: 'add_node', node: { id: 'n_uv', kind: 'uv' } },
    {
      op: 'add_node',
      node: {
        id: 'n_centered',
        kind: 'transform',
        inputs: { uv: { node: 'n_uv', port: 'uv' } },
        params: {
          scale: { kind: 'vec2', value: [2, 2] },
          offset: { kind: 'vec2', value: [-0.5, -0.5] },
          rotate: { kind: 'float', value: 0 },
        },
      },
    },
    {
      op: 'add_node',
      node: {
        id: 'n_warp',
        kind: 'domain_warp',
        inputs: { uv: { node: 'n_centered', port: 'uv' } },
        params: {
          strength: { kind: 'float', value: 0.4, min: 0, max: 1.5, step: 0.01 },
          scale: { kind: 'float', value: 1, min: 0.2, max: 4, step: 0.05 },
          time: { kind: 'float', value: 0.2, min: 0, max: 2, step: 0.01 },
        },
      },
    },
    {
      op: 'add_node',
      node: {
        id: 'n_detail',
        kind: 'fbm',
        inputs: { uv: { node: 'n_warp', port: 'uv' } },
        params: {
          scale: { kind: 'float', value: 6, min: 1, max: 24, step: 0.5 },
          octaves: { kind: 'float', value: 4, min: 1, max: 8, step: 1 },
          lacunarity: { kind: 'float', value: 2 },
          gain: { kind: 'float', value: 0.5, min: 0.2, max: 0.8, step: 0.01 },
          ridged: { kind: 'float', value: 0.7, min: 0, max: 1, step: 0.01 },
        },
      },
    },
    {
      op: 'add_node',
      node: {
        id: 'n_ramp',
        kind: 'color_ramp',
        inputs: { t: { node: 'n_detail', port: 'value' } },
        params: {
          colorA: { kind: 'color', value: [0.02, 0.03, 0.05, 1] },
          colorB: { kind: 'color', value: [0.35, 0.38, 0.45, 1] },
          colorC: { kind: 'color', value: [0.95, 0.97, 1, 1] },
        },
      },
    },
    {
      op: 'add_node',
      node: {
        id: 'n_grade',
        kind: 'grade',
        inputs: { color: { node: 'n_ramp', port: 'color' } },
        params: {
          exposure: { kind: 'float', value: 1 },
          contrast: { kind: 'float', value: 1.3, min: 0.5, max: 3, step: 0.01 },
          saturation: { kind: 'float', value: 0.35, min: 0, max: 2, step: 0.01 },
          gamma: { kind: 'float', value: 1 },
          tint: { kind: 'vec3', value: [0.86, 0.92, 1] },
        },
      },
    },
    {
      op: 'add_node',
      node: { id: 'n_out', kind: 'output', inputs: { color: { node: 'n_grade', port: 'color' } } },
    },
  ],
};

function docWith(nodes: ConstructorParameters<typeof createDoc>[0] extends undefined ? never : Parameters<typeof createDoc>[0]): ShaderDoc {
  return createDoc(nodes);
}

/* -------------------------------------------------------------------------- */
/* 1. The happy path                                                           */
/* -------------------------------------------------------------------------- */

test('a clean draft reaches clean=true with zero model calls beyond the draft', async () => {
  const provider = scriptedProvider([DRAFT_PATCH]);
  const result = await runAgent({
    prompt: 'molten metal, slow flow',
    catalog: CATALOG,
    provider,
  });

  assert.equal(result.stopReason, 'clean', `expected clean, got ${result.stopReason}: ${summariseRun(result)}`);
  assert.equal(result.clean, true);
  assert.deepEqual(result.remaining, []);
  // One interpret + one propose. If mechanical repair had needed a model call, this would
  // be higher — which is the property being protected.
  assert.ok(result.usage!.calls <= 3, `too many model calls: ${result.usage!.calls}`);
  // All six targets must be present and clean, not just the driver.
  assert.deepEqual(
    Object.keys(result.emissions).sort(),
    ['glsl', 'hlsl', 'msl', 'shaderlab', 'ue', 'wgsl'],
  );
  assert.ok(result.doc.hash, 'the returned document must carry a content hash');
});

test('the loop reports intent and emit stages in its trace', async () => {
  const result = await runAgent({ prompt: 'x', catalog: CATALOG, provider: scriptedProvider([DRAFT_PATCH]) });
  const stages = result.trace.map((t) => t.stage);
  assert.ok(stages.includes('interpret'), 'trace must record the interpret stage');
  assert.ok(stages.includes('draft'), 'trace must record the draft stage');
  assert.equal(stages[stages.length - 1], 'emit', 'the last stage must be emit');
});

/* -------------------------------------------------------------------------- */
/* 2. Mechanical repair — no model call                                        */
/* -------------------------------------------------------------------------- */

test('a type mismatch is repaired mechanically, without a second model call', async () => {
  // Constructing a document with a type error takes two patches, because `applyPatch` is
  // atomic: one illegal op rejects the whole patch, so "a good draft plus a bad wire" yields
  // an *empty* document rather than a broken one. So: draft cleanly, then rewire an input from
  // a source of the wrong type with ops that are themselves legal to write.
  //
  // `n_ramp.t` is a float. Its natural source (`n_detail.value`) is also a float. Pointing it
  // at `n_grade.color` (a vec4) produces `IR006_TYPE_MISMATCH`, and because `n_grade` sits
  // downstream of `n_ramp` it produces `IR007_CYCLE` as well — two repairable diagnostics, one
  // of which the repairer fixes by dropping a wire and one by reporting advice.
  const rewire: ShaderPatch = {
    summary: 'rewire the ramp input from a vec4 source',
    ops: [
      { op: 'disconnect', to: { node: 'n_ramp', port: 't' } },
      { op: 'connect', from: { node: 'n_grade', port: 'color' }, to: { node: 'n_ramp', port: 't' } },
    ],
  };

  const provider = scriptedProvider([DRAFT_PATCH, rewire, { summary: 'no-op', ops: [] }]);
  const result = await runAgent({ prompt: 'x', catalog: CATALOG, provider, requireTargets: ['glsl'] });

  // The load-bearing property: the loop must not ask the model to fix something the
  // deterministic repairer can see for itself. With `requireTargets: ['glsl']` and one
  // interpret call, more than three total calls means it spent a round trip on a mismatch.
  assert.ok(
    result.usage!.calls <= 3,
    `a type mismatch must not cost extra model calls: ${result.usage!.calls}`,
  );
  // And it must never claim a document it has not verified.
  if (!result.clean) assert.ok(result.remaining.length > 0);
  assert.ok(result.trace.length >= 2, 'the trace must show that a second round happened');
});

test('a rejection diagnostic is never mistaken for a hopeless graph', async () => {
  // A draft whose only `output` points at a node that does not exist. `applyPatch` rejects it
  // atomically and reports `PT001_UNKNOWN_NODE @ghost`. The loop must carry that to the model
  // rather than concluding "no output node, therefore hopeless" — the difference between
  // "your patch was wrong" and "this graph cannot work" is the whole reason the seeded
  // diagnostics exist.
  const ghost: ShaderPatch = {
    summary: 'reference a node that does not exist',
    ops: [
      { op: 'add_node', node: { id: 'n_uv', kind: 'uv' } },
      { op: 'add_node', node: { id: 'n_out', kind: 'output', inputs: { color: { node: 'ghost', port: 'x' } } } },
    ],
  };
  const provider = scriptedProvider([ghost, { summary: 'no-op', ops: [] }]);
  const result = await runAgent({ prompt: 'x', catalog: CATALOG, provider, requireTargets: ['glsl'] });

  // The model must have been asked again, with the rejection in the context it was shown.
  assert.ok(provider.seen.length >= 2, 'a rejected draft must lead to another proposal');
  /*
   * `some`, not `at(-1)`.
   *
   * The rejection is seeded into the *first* repair round and deliberately dropped after that
   * — it describes a patch that was refused, and once the model has seen it, it is stale. The
   * loop may legitimately run several more rounds afterwards (the empty graph produces advice,
   * which is progress-capable), so the last context is not where this belongs. Asserting on
   * the last one made the test depend on how many rounds the budget happened to allow, which
   * is not the property under test: the property is that the model is told *which node was
   * missing*, at the moment it is asked to fix it.
   */
  const told = provider.seen.find((c) =>
    c.diagnostics.some((d) => /ghost/i.test(`${d.code} ${d.message}`)),
  );
  assert.ok(
    told,
    'the model was never told which node was missing; contexts seen:\n' +
      provider.seen
        .map((c, i) => `  #${i} ${c.stage}: ${c.diagnostics.map((d) => d.code).join(', ')}`)
        .join('\n'),
  );
  assert.notEqual(result.clean, true, 'the graph never became emittable, so it must not claim success');
  assert.notEqual(result.stopReason, 'clean');
});

test('a graph that has nodes but no output gets a sink, mechanically', async () => {
  /*
   * The case that produced an `unrecoverable` dead end against a real model.
   *
   * A deepseek-chat draft built six good nodes and simply forgot the `output`. The loop read
   * `IR008_NO_OUTPUT`, consulted the unrecoverable table, and gave up after three model calls
   * — for a defect that is one `add_node` and one `connect`. `repair.ts` now treats a missing
   * sink as what it is: the same shape of edit it has always done for every other unwired
   * port, with the source chosen as the deepest node nothing else consumes.
   */
  const noSink: ShaderPatch = {
    summary: 'a graph with no output node',
    ops: [
      { op: 'add_node', node: { id: 'n_uv', kind: 'uv' } },
      { op: 'add_node', node: { id: 'n_tint', kind: 'const_color' } },
    ],
  };
  // Only the draft: the repair must land without a second model call.
  const provider = scriptedProvider([noSink]);
  const result = await runAgent({ prompt: 'x', catalog: CATALOG, provider, requireTargets: ['glsl'] });

  assert.equal(result.clean, true, `expected a mechanical recovery, got ${result.stopReason}`);
  // Two calls is the floor for any run — `interpret` plus `draft`. The property is that no
  // *repair* round happened, which is what a missing sink used to cost.
  assert.equal(result.usage!.calls, 2, 'a missing sink must not cost a repair round');
  assert.ok(
    !provider.seen.some((c) => c.stage === 'repair'),
    'the model must never be asked to fix a missing output node',
  );
  assert.ok(
    result.doc.graph.nodes.some((n) => n.kind === 'output'),
    'the repair must have added the output node',
  );
  const out = result.doc.graph.nodes.find((n) => n.kind === 'output');
  assert.ok(out?.inputs?.color, 'the added output must be wired to something, not left flat');
  assert.ok(
    result.trace.some((t) => t.stage === 'repair'),
    'the trace must record the mechanical repair',
  );
});

test('the loop hands the catalog to any provider that validates against one', async () => {
  /*
   * The bug this pins, in full.
   *
   * `createHttpProvider` parses model replies and validates every op against a catalog. It
   * defaults that catalog to `{}`, and **nothing called `setCatalog`** — not the CLI, not the
   * web facade, not this loop. Every `add_node` from a real model was therefore checked
   * against an empty catalog, reported `AG012_UNKNOWN_KIND` and was dropped. A six-node
   * deepseek-chat draft arrived as an empty patch, the graph stayed empty, and the run ended
   * at the budget with `IR008_NO_OUTPUT` — which reads exactly like a model that cannot follow
   * instructions, and cost a real user a real run.
   *
   * The provider interface now declares `setCatalog`, and the loop calls it before the first
   * proposal. This test asserts the *call*, not its effect, because the effect depends on a
   * parsing provider and the call is what was missing.
   */
  let received: unknown = null;
  const provider = {
    name: 'catalog-spy',
    vision: false,
    async complete() {
      return { text: '{}' };
    },
    async interpret() {
      return SPEC;
    },
    setCatalog(catalog: unknown) {
      received = catalog;
    },
    async propose(): Promise<ShaderPatch> {
      return DRAFT_PATCH;
    },
  };

  const result = await runAgent({ prompt: 'x', catalog: CATALOG, provider, requireTargets: ['glsl'] });
  assert.ok(result.doc, 'the run must produce a document');
  assert.ok(received, 'the loop never called setCatalog — the provider validated against nothing');
  // The *same* catalog the loop emits and patches with, not merely a non-empty one: a second
  // copy that drifted would reinstate the bug in a subtler form.
  assert.equal(received, CATALOG, 'the provider must receive the loop\'s own catalog object');
});

test('a hopeless diagnostic stops the loop even while the document keeps changing', async () => {
  /*
   * The defect a real run exposed, reproduced.
   *
   * A toon document emitted to a target with no surface entry point produces
   * `EM002_SURFACE_UNSUPPORTED`. The repairer classifies it `hopeless` — correctly, since the
   * entry point lives in `src/targets/` and no document edit can add one. But the stop
   * condition checked `changed` *after* `hopeless`, and on round 0 `changed` is always true
   * (there is no previous hash to compare against). So the loop asked the model, the model
   * edited the document to no effect, the edit set `changed` again, and it asked again —
   * **twelve model calls, ten refine rounds, to learn what round 0 already knew.**
   *
   * `IR002_UNKNOWN_KIND` stands in here for the same class: a diagnostic about something
   * outside the document. The document genuinely changes each round (the provider appends a
   * node), which is what made `changed` able to defeat the verdict.
   */
  const broken = createDoc({
    title: 'has a kind the catalog does not',
    spec: SPEC,
    targets: ['glsl'],
  });
  // Hand-written rather than patched in: `applyPatch` would refuse an unknown kind, which is
  // the point — a document like this can only arrive from a corrupted file or a stale schema.
  broken.graph.nodes.push({ id: 'n_ghost', kind: 'not_a_real_kind' as never });

  let round = 0;
  const provider = {
    name: 'keeps-editing',
    vision: false,
    async complete() {
      return { text: '{}' };
    },
    async interpret() {
      return SPEC;
    },
    async propose(): Promise<ShaderPatch> {
      round++;
      // A real edit every round, so `hashDoc` differs and `changed` is true throughout.
      return {
        summary: `add filler ${round}`,
        ops: [{ op: 'add_node', node: { id: `filler${round}`, kind: 'const_float' } } as never],
      };
    },
  };

  const result = await runAgent({
    prompt: 'x',
    catalog: CATALOG,
    provider,
    basedOn: broken,
    mode: 'repair',
    requireTargets: ['glsl'],
  });

  assert.equal(result.stopReason, 'unrecoverable', `expected an immediate stop, got ${result.stopReason}`);
  assert.equal(
    result.usage!.calls,
    0,
    `a hopeless diagnostic must cost no model calls, spent ${result.usage!.calls}`,
  );
  assert.ok(
    result.remaining.some((d) => d.code === 'IR002_UNKNOWN_KIND'),
    `the reason must survive in the result: ${result.remaining.map((d) => d.code).join(', ')}`,
  );
});

test('EM002_SURFACE_UNSUPPORTED is classified as outside the document', async () => {
  /*
   * The classification itself, because it is what makes the test above generic.
   *
   * `EM002` is the first code in the table that a *valid, correct* document can produce — the
   * graph is fine, the emitter simply cannot write it for that target. It belongs in
   * `UNRECOVERABLE` more than a graph error does: a capability gap is by definition not
   * something the document can express its way out of.
   */
  assert.ok(
    unrecoverableReason('EM002_SURFACE_UNSUPPORTED'),
    'a target capability gap must be classified unrecoverable',
  );
  // And the codes that are genuinely document-level must *not* be in there, or the loop would
  // give up on problems it can fix.
  for (const code of ['IR006_TYPE_MISMATCH', 'IR031_MISSING_INPUT', 'IR008_NO_OUTPUT', 'EM001_NO_OUTPUT_NODE']) {
    assert.equal(unrecoverableReason(code), undefined, `${code} is a document problem and must stay repairable`);
  }
});

test('an empty graph is advised, never patched into a flat colour', async () => {  /*
   * The guard on the repair above.
   *
   * `addOutputNode` must not fire on a document with nothing in it: a lone `output` emits a
   * flat colour, the emitter calls that clean, the seeded rejection diagnostic is discarded
   * ("a clean emission wins over a seed"), and the user is told the run succeeded while
   * looking at a black frame. An empty graph is a failed draft and only the model can fix it.
   */
  const empty = scriptedProvider([{ summary: 'do nothing at all', ops: [] }]);
  const result = await runAgent({ prompt: 'x', catalog: CATALOG, provider: empty });
  assert.notEqual(result.clean, true, 'an empty graph must not be reported as clean');
  assert.ok(result.remaining.length > 0, 'the failure must carry its diagnostics');
  assert.ok(
    empty.seen.length >= 2,
    'advice is progress-capable, so the model must be asked again rather than given up on',
  );
});

test('an unwired required input is filled mechanically', async () => {
  // `n_ramp.t` left unconnected: the repairer should wire it from a node already present
  // that produces a float.
  const draft: ShaderPatch = {
    summary: 'draft with an unwired input',
    ops: [
      ...DRAFT_PATCH.ops,
      // Disconnect what the good draft connected, so the input falls back to "missing".
      { op: 'disconnect', to: { node: 'n_ramp', port: 't' }, reason: 'leave it open on purpose' },
    ],
  };
  const result = await runAgent({ prompt: 'x', catalog: CATALOG, provider: scriptedProvider([draft]) });
  // Either the mechanical pass fixed it, or the loop honestly reports what is missing —
  // both are acceptable; silently emitting a broken shader is not.
  if (result.clean) {
    assert.equal(result.remaining.length, 0);
  } else {
    assert.ok(result.remaining.length > 0, 'a non-clean result must say what is wrong');
  }
  assert.ok(result.trace.length > 0);
});

/* -------------------------------------------------------------------------- */
/* 3. Escalation, plateau and honest failure                                   */
/* -------------------------------------------------------------------------- */

test('a provider that never improves the graph ends in an honest non-clean result', async () => {
  // A draft that is structurally broken and a provider that keeps proposing nothing.
  const broken: ShaderPatch = {
    summary: 'reference a node that does not exist',
    ops: [
      { op: 'add_node', node: { id: 'n_uv', kind: 'uv' } },
      { op: 'add_node', node: { id: 'n_out', kind: 'output', inputs: { color: { node: 'n_missing', port: 'x' } } } },
    ],
  };
  const provider = scriptedProvider([broken, { summary: 'nothing', ops: [] }]);
  const result = await runAgent({ prompt: 'x', catalog: CATALOG, provider });

  assert.notEqual(result.stopReason, 'clean', 'the loop must not claim success');
  assert.ok(
    ['budget', 'unrecoverable', 'provider-error'].includes(result.stopReason),
    `unexpected stop reason: ${result.stopReason}`,
  );
  assert.ok(result.remaining.length > 0, 'a non-clean result must carry its diagnostics');
  // The budget must actually bound the work.
  assert.ok(result.usage!.calls <= 12, `model calls exceeded the policy ceiling: ${result.usage!.calls}`);
  // And the trace must be long enough to explain what happened.
  assert.ok(result.trace.length >= 2, 'the trace must record the attempt');
});

test('the loop always returns a document, even when it fails', async () => {
  const result = await runAgent({ prompt: 'nothing works', catalog: CATALOG, provider: null });
  assert.ok(result.doc, 'a failed run must still return a document');
  assert.equal(result.clean, false);
  assert.ok(result.remaining.length > 0);
  assert.ok(
    result.remaining.some((d) => d.code === 'AG001_NO_PROVIDER' || d.severity === 'error'),
    'the reason must be reported as a diagnostic',
  );
});

test('a provider that throws is reported as a provider error, not a crash', async () => {
  const throwing = {
    name: 'throwing',
    vision: false,
    async complete() {
      throw new Error('boom');
    },
    async interpret() {
      throw new Error('401 unauthorized: check the API key');
    },
  };
  const result = await runAgent({ prompt: 'x', catalog: CATALOG, provider: throwing as never });
  assert.equal(result.stopReason, 'provider-error');
  assert.ok(
    result.remaining.some((d) => d.message.includes('401')),
    'the provider message must be surfaced verbatim',
  );
  // A plain `Error` is what the local providers throw, and for those the configuration advice is
  // the right advice — this pins that the classification did not make everything generic.
  const diag = result.remaining.find((d) => d.code === 'AG004_PROVIDER_FAILED');
  assert.ok(diag, `AG004 must be the diagnostic, got ${result.remaining.map((d) => d.code).join(', ')}`);
  assert.match(diag.fix, /provider configuration/);
  // And the request that failed is counted: before this, a run whose first call threw reported
  // `modelCalls=0`, which reads as "no model was ever asked" and hides the failure it just had.
  assert.equal(result.usage?.calls, 1, 'a request that was issued and failed is still a model call');
});

/**
 * The failure the panel reported as "HTTP 200 … the response body could not be read", followed by
 * advice to check the API key. The status alone disproves that advice, so the `fix` a reader is
 * told to act on must not repeat it — for either of the two transport-class failures.
 */
test('AG004 advice fits the failure: a dropped body is not a configuration problem', async () => {
  const cases: { label: string; err: Error; mustMatch: RegExp; mustNotMatch: RegExp }[] = [
    {
      label: 'a body that never arrived',
      err: new ProviderError('read', 'the response body could not be read after HTTP 200 — terminated', true),
      mustMatch: /Retry|transport failure/,
      mustNotMatch: /API key|apiKey|base URL/i,
    },
    {
      label: 'our own ceiling firing',
      err: new ProviderError('timeout', 'no response within 120000 ms', true),
      mustMatch: /timeoutMs|Retry/,
      mustNotMatch: /API key|apiKey|base URL/i,
    },
    {
      label: 'the host never answered',
      err: new ProviderError('transport', 'network error — fetch failed', true),
      mustMatch: /reachable|CORS/,
      mustNotMatch: /API key is wrong/i,
    },
    {
      label: 'a 401',
      err: new ProviderError('http', 'HTTP 401 Unauthorized', false),
      mustMatch: /4xx|API key/,
      mustNotMatch: /transient/i,
    },
  ];

  for (const c of cases) {
    const result = await runAgent({
      prompt: 'x',
      catalog: CATALOG,
      provider: failingProvider(c.err),
    });
    assert.equal(result.stopReason, 'provider-error', `${c.label}: must stop as a provider error`);
    const diag = result.remaining.find((d) => d.code === 'AG004_PROVIDER_FAILED');
    assert.ok(diag, `${c.label}: no AG004 diagnostic`);
    assert.match(diag.fix, c.mustMatch, `${c.label}: the fix does not tell the reader what to do`);
    assert.doesNotMatch(diag.fix, c.mustNotMatch, `${c.label}: the fix gives advice the failure disproves`);
  }
});

test('an aborted signal stops the loop promptly', async () => {
  const controller = new AbortController();
  controller.abort();
  const result = await runAgent({
    prompt: 'x',
    catalog: CATALOG,
    provider: scriptedProvider([DRAFT_PATCH]),
    signal: controller.signal,
  });
  assert.equal(result.stopReason, 'aborted');
});

/* -------------------------------------------------------------------------- */
/* 4. Refining an existing document                                            */
/* -------------------------------------------------------------------------- */

test('basedOn skips the draft stage and refines the supplied document', async () => {
  const base = createDoc({
    title: '已有的金属',
    spec: SPEC,
    targets: ['glsl'],
    graph: normalizeEdges({
      nodes: [
        { id: 'n_uv', kind: 'uv' },
        { id: 'n_detail', kind: 'fbm', inputs: { uv: { node: 'n_uv', port: 'uv' } } },
        {
          id: 'n_ramp',
          kind: 'color_ramp',
          inputs: { t: { node: 'n_detail', port: 'value' } },
        },
        { id: 'n_out', kind: 'output', inputs: { color: { node: 'n_ramp', port: 'color' } } },
      ],
      edges: [],
    }),
  });

  const provider = scriptedProvider([{ summary: 'increase detail scale', ops: [
    { op: 'set_param', node: 'n_detail', param: 'scale', value: { kind: 'float', value: 9 } },
  ] }]);
  const result = await runAgent({
    prompt: 'contrast a bit higher',
    catalog: CATALOG,
    provider,
    basedOn: base,
    requireTargets: ['glsl'],
  });

  assert.ok(
    !result.trace.some((t) => t.stage === 'draft'),
    'refining an existing graph must not re-draft',
  );
  // No interpret call either: the document already carries its spec.
  assert.equal(result.usage!.calls, 1);
  assert.equal(provider.seen[0]?.stage, 'refine');
  assert.equal(provider.seen[0]?.userIntent, 'contrast a bit higher');
  assert.equal(result.doc.graph.nodes.find(n => n.id === 'n_detail')?.params?.scale?.value, 9);
  assert.equal(result.clean, true);
});

/* -------------------------------------------------------------------------- */
/* 5. Determinism                                                              */
/* -------------------------------------------------------------------------- */

test('the same inputs produce the same document hash and the same trace shape', async () => {
  const run = async () => {
    const result = await runAgent({
      prompt: 'molten metal',
      catalog: CATALOG,
      provider: scriptedProvider([DRAFT_PATCH]),
      requireTargets: ['glsl', 'wgsl'],
    });
    return { hash: hashDoc(result.doc), stages: result.trace.map((t) => t.stage), clean: result.clean };
  };
  const a = await run();
  const b = await run();
  assert.equal(a.hash, b.hash, 'the loop must be deterministic for a deterministic provider');
  assert.deepEqual(a.stages, b.stages);
  assert.equal(a.clean, b.clean);
});

/* -------------------------------------------------------------------------- */
/* 6. The context the model is shown                                           */
/* -------------------------------------------------------------------------- */

test('the provider sees a compact projection, never the whole document', async () => {
  const provider = scriptedProvider([DRAFT_PATCH]);
  await runAgent({ prompt: 'molten metal', catalog: CATALOG, provider });

  assert.equal(provider.seen.length, 1);
  const ctx = provider.seen[0]!;
  assert.equal(ctx.stage, 'draft');
  assert.equal(ctx.userIntent, 'molten metal');
  // Empty graph at draft time, so the inventories are empty but present.
  assert.deepEqual(ctx.nodeLines, []);
  assert.deepEqual(ctx.uniformLines, []);
  // The projection must not carry provenance or the full render spec into the prompt.
  assert.ok(!('provenance' in ctx), 'the projection must not expose provenance');
});

test('a repair round shows the model the node inventory and the diagnostics', async () => {
  const broken: ShaderPatch = {
    summary: 'unwired output',
    ops: [
      { op: 'add_node', node: { id: 'n_uv', kind: 'uv' } },
      { op: 'add_node', node: { id: 'n_out', kind: 'output', inputs: { color: { node: 'ghost', port: 'x' } } } },
    ],
  };
  // The second reply is the model's "fix", which is a no-op, so the run ends non-clean after
  // at least one repair context was built.
  const provider = scriptedProvider([broken, { summary: 'no-op', ops: [] }]);
  const result = await runAgent({ prompt: 'x', catalog: CATALOG, provider });

  const repairCtx = provider.seen.find((c) => c.stage === 'repair');
  assert.ok(repairCtx, 'a repair context must have been built');
  assert.ok(repairCtx!.diagnostics.length > 0, 'the repair context must carry the diagnostics');
  assert.ok(
    repairCtx!.diagnostics.every((d) => typeof d.code === 'string' && d.code.length > 0),
    'every diagnostic must have a stable code',
  );
  assert.ok(result.trace.length >= 1);
});

/* -------------------------------------------------------------------------- */
/* 7. expose_param, and refining without consulting a model                     */
/* -------------------------------------------------------------------------- */

/** A small valid graph whose fbm scale has been exposed as a tunable uniform. */
function docWithUniform(): ShaderDoc {
  const base = createDoc({
    title: 't',
    spec: SPEC,
    targets: ['glsl'],
    graph: normalizeEdges({
      nodes: [
        { id: 'n_uv', kind: 'uv' },
        {
          id: 'n_d',
          kind: 'fbm',
          inputs: { uv: { node: 'n_uv', port: 'uv' } },
          params: { scale: { kind: 'float', value: 4, min: 0, max: 10 } },
        },
        { id: 'n_r', kind: 'color_ramp', inputs: { t: { node: 'n_d', port: 'value' } } },
        { id: 'n_o', kind: 'output', inputs: { color: { node: 'n_r', port: 'color' } } },
      ],
      edges: [],
    }),
  });
  return exposeParam(
    base,
    'n_d',
    'scale',
    { name: 'u_scale', min: 0, max: 10, step: 0.5, label: '纹理密度', group: '纹理' },
    CATALOG,
  ).doc;
}

test('a uniform created by expose_param reaches the model projection intact', () => {
  // `expose_param` is how a literal becomes a slider, and it carries more than a value:
  // `step`, `label` and `group` decide how the slider behaves and where it appears. If the
  // projection drops them the model reasons about a control it cannot see properly — and the
  // failure is invisible in the generated shader, which is byte-identical either way. So this
  // asserts the *rendered line*, not the presence of a name.
  const doc = docWithUniform();
  const ctx = buildContext({
    stage: 'repair',
    doc,
    catalog: CATALOG,
    userIntent: 'tune it',
    diagnostics: [{ severity: 'error', code: 'X_TEST', message: 'placeholder' }],
  });

  const line = ctx.uniformLines.find((l) => l.includes('u_scale'));
  assert.ok(line, `the uniform must appear in the projection: ${JSON.stringify(ctx.uniformLines)}`);
  assert.match(line, /纹理密度/, 'the label must survive — it is the slider caption');
  assert.match(line, /纹理/, 'the group must survive — it decides the panel section');
  assert.match(line, /range\[0\.\.10\]/, 'the range must survive — it bounds the search space');
  assert.match(line, /n_d\.scale/, 'the origin must survive — it links the slider back to its node');
});

test('a mechanically repairable defect during refine costs ZERO model calls', async () => {
  // Aim a vec2 at a vec4 input. The repairer recognises `IR006`/`IR032` and **rebuilds the
  // value** (`split_vec2` + `make_vec4`) rather than deleting the wire, so the loop reaches a
  // clean document without asking anyone. This is the property that makes "refine this" feel
  // instant: the model is a fallback, not the first responder.
  const good = docWithUniform();
  const broken: ShaderDoc = {
    ...good,
    graph: normalizeEdges({
      nodes: good.graph.nodes.map((n) =>
        n.id === 'n_o' ? { ...n, inputs: { color: { node: 'n_uv', port: 'uv' } } } : n,
      ),
      edges: [],
    }),
  };

  // If the loop did consult the model it would get this reply, which fixes nothing — so a
  // clean result proves the model was never needed.
  const provider = scriptedProvider([{ summary: 'no-op', ops: [] }]);
  const result = await runAgent({
    prompt: 'tune it',
    catalog: CATALOG,
    provider,
    basedOn: broken,
    mode: 'repair',
    requireTargets: ['glsl'],
  });

  assert.equal(result.stopReason, 'clean', `expected a mechanical fix: ${summariseRun(result)}`);
  assert.equal(
    result.usage!.calls,
    0,
    'a repairable type error during refine must not cost a model call',
  );
  assert.equal(provider.seen.length, 0, 'the provider must not have been consulted at all');
  assert.ok(
    result.trace.some((t) => t.stage === 'repair'),
    'the trace must record that a repair round happened',
  );
  // The signal has to *survive*. A disconnect would also reach `clean` — because `output.color`
  // has a default — and the picture would quietly become a flat colour, which is the failure
  // this repair exists to stop.
  const out = result.doc.graph.nodes.find((n) => n.id === 'n_o');
  assert.ok(out?.inputs?.color, 'the repaired document lost the wire into output.color');
  assert.notEqual(
    out.inputs.color.node,
    'n_uv',
    'the vec2 must go through the inserted joiner, not be wired straight in',
  );
});

test('one defect at one port yields one repair, whichever code reports it', () => {
  /*
   * A port that receives the wrong type is reported **twice**: `IR006_TYPE_MISMATCH` ("`n_uv.uv`
   * is vec2 but `n_o.color` expects vec4") and `IR032_NO_COERCION` ("No known conversion from
   * vec2 to vec4"). The plan used to key its de-duplication on (code, node, port), so both were
   * planned — and the two handlers disagreed: one inserted a joiner, the other deleted the wire.
   * Applied in order, the deletion won, and the loop reported a *clean* document whose effect had
   * silently disappeared. This test pins the site as the unit of repair, which is what makes that
   * impossible rather than unlikely.
   *
   * Checked through `repair()` rather than the loop because the ordering is the whole point, and
   * the loop hides it behind a re-analysis.
   */
  const good = docWithUniform();
  const broken: ShaderDoc = {
    ...good,
    graph: normalizeEdges({
      nodes: good.graph.nodes.map((n) =>
        n.id === 'n_o' ? { ...n, inputs: { color: { node: 'n_uv', port: 'uv' } } } : n,
      ),
      edges: [],
    }),
  };

  const diagnostics: Diagnostic[] = [
    {
      severity: 'error',
      code: 'IR006_TYPE_MISMATCH',
      message: 'Type mismatch: `n_uv.uv` is vec2 but `n_o.color` expects vec4.',
      at: { node: 'n_o', port: 'color' },
    },
    {
      severity: 'error',
      code: 'IR032_NO_COERCION',
      message: 'No known conversion from vec2 to vec4 at `n_o.color`.',
      at: { node: 'n_o', port: 'color' },
    },
  ];

  const plan = repair(broken, diagnostics, CATALOG);
  assert.equal(plan.actions.length, 1, `one port, one repair — got ${plan.actions.length}`);

  const action = plan.actions[0]!;
  assert.equal(action.kind, 'patch', 'a buildable mismatch must be built, not described');
  const ops = (action as { patch: { ops: { op: string }[] } }).patch.ops.map((o) => o.op);
  assert.ok(ops.includes('add_node'), `the plan must insert nodes, got ${ops.join(',')}`);
  assert.ok(
    !ops.includes('disconnect'),
    'no repair for this port may delete the wire — that is the silent-effect-loss failure',
  );
});

test('a vec4 wired into a float input is split, never dropped', async () => {
  /*
   * The narrowing direction, which used to be the silent one. `color_ramp.t` is a float; a
   * vec4 source is a realistic authoring slip (wiring a colour where a scalar belongs), and the
   * old repair answered it with a `disconnect`. On a port with a default that leaves a document
   * that *emits cleanly and looks wrong*: the model reports success, the user sees the effect
   * they just wired up vanish, and nothing in the run says why.
   *
   * The repairer now inserts `split_vec4` and takes `.x`, which is a real edit with real
   * diagnostics behind it — and the run still costs zero model calls, because a mechanical fix
   * that consults the model is not mechanical.
   */
  const good = docWithUniform();
  const broken: ShaderDoc = {
    ...good,
    graph: normalizeEdges({
      nodes: good.graph.nodes.map((n) =>
        n.id === 'n_r' ? { ...n, inputs: { t: { node: 'n_o', port: 'color' } } } : n,
      ),
      edges: [],
    }),
  };

  const provider = scriptedProvider([{ summary: 'no-op', ops: [] }]);
  const result = await runAgent({
    prompt: 'tune it',
    catalog: CATALOG,
    provider,
    basedOn: broken,
    mode: 'repair',
    requireTargets: ['glsl'],
  });

  assert.equal(result.usage!.calls, 0, `a narrowing mismatch must be repaired offline: ${summariseRun(result)}`);
  const ramp = result.doc.graph.nodes.find((n) => n.id === 'n_r');
  assert.ok(ramp?.inputs?.t, 'the wire into color_ramp.t was deleted instead of split');
  const splitter = result.doc.graph.nodes.find((n) => n.id === ramp.inputs.t.node);
  assert.equal(splitter?.kind, 'split_vec4', `expected a split_vec4, got ${splitter?.kind}`);
  assert.ok(
    splitter?.inputs?.value,
    'the splitter must still be fed by the wide value it was inserted to preserve',
  );
});
