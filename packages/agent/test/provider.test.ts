/**
 * `packages/agent/src/provider/*` — unit tests
 * ============================================
 *
 * Scope: the three files this workstream owns — `parse.ts` (the defensive boundary),
 * `stub.ts` (the deterministic fixture) and `http.ts` (the transport). `prompt.ts` is
 * covered by the prompt-shape assertions at the bottom, which is where a prompt regression
 * would show up.
 *
 * Run with:
 *
 *     cd packages/agent
 *     node --test --test-isolation=none --experimental-strip-types \
 *       --import ./test/register-hooks.mjs test/provider.test.ts
 *
 * The `--import ./test/register-hooks.mjs` flag is not optional: the sources use `.js`
 * specifiers (moduleResolution: nodenext) and Node has no `.js` → `.ts` fallback. See
 * `test/ts-resolve.mjs`.
 *
 * `packages/agent/src/index.ts` re-exports `loop.ts` / `repair.ts` / `context.ts` /
 * `policy.ts`, which another workstream owns and which do not exist yet, so this suite
 * imports the provider modules **directly** and the package as a whole will not
 * type-check until those land. That is expected; see the report.
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import { CATALOG } from '@ai4shader/ir/catalog';
import '@ai4shader/ir/targets';
import { applyPatch, createDoc } from '@ai4shader/ir/patcher';
import { emit } from '@ai4shader/ir';
import { REWRITE_ALIASES } from '@ai4shader/ir/glslSubset';
import { SURFACE_HOST_SYMBOLS } from '@ai4shader/ir/types';
import type {
  Diagnostic,
  NodeSchema,
  PatchOp,
  ShaderDoc,
  ShaderPatch,
  TargetFormat,
} from '@ai4shader/ir/types';

import { AG, extractJson, parseIntentJson, parsePatch, parsePatchJson, parseSpec } from '../src/provider/parse.js';
import { createStubProvider, STUB_INTENTS } from '../src/provider/stub.js';
import { ProviderError, createHttpProvider, isTransientStatus } from '../src/provider/http.js';
import { patchUserPrompt, intentUserPrompt, systemPrompt } from '../src/provider/prompt.js';
import type { IntentInput, PatchContext } from '../src/types.js';

/* -------------------------------------------------------------------------- */
/* Helpers                                                                     */
/* -------------------------------------------------------------------------- */

const settle = (ds: readonly Diagnostic[]): Diagnostic[] => ds.filter((d) => d.severity === 'error');
const codes = (ds: readonly Diagnostic[]): string[] => settle(ds).map((d) => d.code);

/** Every one of the six targets, so a test cannot quietly check only glsl. */
const ALL_TARGETS: TargetFormat[] = ['glsl', 'wgsl', 'hlsl', 'msl', 'shaderlab', 'ue'];

test('online patches can enable and disable transparency and feedback without coercion', () => {
  for (const path of ['transparent', 'feedback'] as const) {
    for (const value of [true, false]) {
      const parsed = parsePatch({ summary: 'configure glass rendering', ops: [{ op: 'set_render', path, value }] }, CATALOG);
      assert.equal(parsed.value.ops.length, 1);
      assert.deepEqual(codes(parsed.diagnostics), []);
      const result = applyPatch(createDoc(), parsed.value, CATALOG);
      assert.equal(result.ok, true);
      assert.equal(result.doc.render[path], value);
    }
    for (const value of ['true', 1, null]) {
      const parsed = parsePatch({ summary: 'bad boolean', ops: [{ op: 'set_render', path, value }] }, CATALOG);
      assert.equal(parsed.value.ops.length, 0);
      assert.ok(codes(parsed.diagnostics).length > 0);
      const doc = createDoc();
      const result = applyPatch(doc, { summary: 'bad boolean', ops: [{ op: 'set_render', path, value }] } as never, CATALOG);
      assert.equal(result.ok, false);
      assert.equal(result.doc, doc);
    }
  }
});

function draftContext(over: Partial<PatchContext> = {}): PatchContext {
  return {
    stage: 'draft',
    doc: createDoc(),
    nodeLines: [],
    uniformLines: [],
    diagnostics: [],
    ...over,
  };
}

/**
 * A patch whose ops are all legal against the real catalog.
 *
 * Note the shape: `uv` is a pure generator with **no inputs**, so a `set_param` on it
 * would be a hallucinated port — the fixture sets a param on `fbm`, which has six real
 * inputs. Port names here are the catalog's, verbatim.
 */
function validPatch(): ShaderPatch {
  return {
    summary: 'two nodes, a wire and a tunable',
    ops: [
      { op: 'add_node', node: { id: 'n_uv', kind: 'uv' } },
      { op: 'add_node', node: { id: 'n_fbm', kind: 'fbm' } },
      { op: 'connect', from: { node: 'n_uv', port: 'uv' }, to: { node: 'n_fbm', port: 'uv' } },
      { op: 'set_param', node: 'n_fbm', param: 'scale', value: { kind: 'float', value: 6, min: 1, max: 24 } },
    ],
  };
}

/* ========================================================================== */
/* extractJson                                                                 */
/* ========================================================================== */

test('extractJson: a bare object', () => {
  const r = extractJson('{"a":1,"b":[2,3]}');
  assert.deepEqual(r.value, { a: 1, b: [2, 3] });
  assert.deepEqual(settle(r.diagnostics), []);
});

test('extractJson: a fenced block with a language tag', () => {
  const r = extractJson('```json\n{"intent":"x","nested":{"y":2}}\n```');
  assert.deepEqual(r.value, { intent: 'x', nested: { y: 2 } });
  assert.deepEqual(settle(r.diagnostics), []);
});

test('extractJson: a fenced block with no language tag', () => {
  const r = extractJson('```\n{"intent":"x"}\n```');
  assert.deepEqual(r.value, { intent: 'x' });
  assert.deepEqual(settle(r.diagnostics), []);
});

test('extractJson: prose before and after the object', () => {
  const r = extractJson(
    'Sure! Here is the spec you asked for.\n{"intent":"metal","tags":["a"]}\nLet me know if you want changes!',
  );
  assert.deepEqual(r.value, { intent: 'metal', tags: ['a'] });
  // Trailing prose is not worth a diagnostic: the value was found intact and the prose is
  // not a second value. (A *second value* is reported — see the test below.)
  assert.deepEqual(settle(r.diagnostics), []);
  assert.deepEqual(r.diagnostics, []);
});

test('extractJson: prose containing braces does not confuse the scan', () => {
  const r = extractJson('Before { not json } after\n{"ok":true}');
  assert.deepEqual(r.value, { ok: true });
  assert.deepEqual(settle(r.diagnostics), []);
});

test('extractJson: braces inside a JSON string do not change the depth', () => {
  const r = extractJson('prose {"note":"a { b } c","n":1} trailing');
  assert.deepEqual(r.value, { note: 'a { b } c', n: 1 });
  assert.deepEqual(settle(r.diagnostics), []);
});

test('extractJson: a trailing comma before } and before ] is repaired, and reported', () => {
  const obj = extractJson('{"a":1,"b":2,}');
  assert.deepEqual(obj.value, { a: 1, b: 2 });
  assert.deepEqual(codes(obj.diagnostics), [], 'a trailing comma is not an error');
  assert.ok(
    obj.diagnostics.some((d) => d.code === AG.TRAILING_COMMA),
    'the repair must be reported, not silent',
  );

  const arr = extractJson('{"a":[1,2,],}');
  assert.deepEqual(arr.value, { a: [1, 2] });
  assert.ok(arr.diagnostics.some((d) => d.code === AG.TRAILING_COMMA));
});

test('extractJson: truncated JSON is a hard error and nothing is invented', () => {
  const r = extractJson('{"a":1,"b":');
  assert.equal(r.value, undefined, 'a truncated reply must not yield a partial object');
  assert.deepEqual(codes(r.diagnostics), [AG.BAD_JSON]);
  assert.match(r.diagnostics[0]!.message, /does not parse/);
});

test('extractJson: an unbalanced quote is not repaired', () => {
  const r = extractJson('{"a":"unterminated}');
  assert.equal(r.value, undefined);
  assert.deepEqual(codes(r.diagnostics), [AG.BAD_JSON]);
});

test('extractJson: a missing value is reported, never filled in', () => {
  const r = extractJson('{"a":,}');
  assert.equal(r.value, undefined);
  assert.deepEqual(codes(r.diagnostics), [AG.BAD_JSON]);
});

test('extractJson: an empty reply is AG001', () => {
  const r = extractJson('   \n  ');
  assert.equal(r.value, undefined);
  assert.deepEqual(codes(r.diagnostics), [AG.NO_JSON]);
});

test('extractJson: prose with no JSON at all is AG001', () => {
  const r = extractJson('I cannot help with that request.');
  assert.equal(r.value, undefined);
  assert.deepEqual(codes(r.diagnostics), [AG.NO_JSON]);
});

test('extractJson: two objects in one reply — the FIRST is used, and the reply says so', () => {
  const r = extractJson('{"first":1}\n{"second":2}');
  assert.deepEqual(r.value, { first: 1 });
  assert.deepEqual(settle(r.diagnostics), []);
  assert.ok(
    r.diagnostics.some((d) => d.severity === 'warning' && /ignored/.test(d.message)),
    'dropping the second value silently would hide a real prompt problem',
  );
});

test('extractJson: a prose-wrapped brace that is not JSON does not shadow the real value', () => {
  // The first balanced candidate is `{ not json }`; it must be skipped, not "fixed".
  const r = extractJson('Before { not json } after\n{"ok":true}');
  assert.deepEqual(r.value, { ok: true });
  assert.deepEqual(settle(r.diagnostics), []);
});

test('extractJson: a trailing comma inside a string is left alone', () => {
  const r = extractJson('{"a":",}","b":1}');
  assert.deepEqual(r.value, { a: ',}', b: 1 });
  assert.deepEqual(r.diagnostics, []);
});

/* ========================================================================== */
/* parseSpec                                                                   */
/* ========================================================================== */

test('parseSpec: a complete spec round-trips through JSON with no diagnostics', () => {
  const r = parseIntentJson(JSON.stringify(STUB_INTENTS.liquidMetal));
  assert.deepEqual(r.diagnostics, []);
  assert.deepEqual(r.value, STUB_INTENTS.liquidMetal);
});

test('parseSpec: accepts a `{ "spec": {...} }` envelope', () => {
  const r = parseIntentJson(JSON.stringify({ spec: STUB_INTENTS.plasma }));
  assert.equal(r.value.intent, STUB_INTENTS.plasma.intent);
  assert.deepEqual(settle(r.diagnostics), []);
});

test('parseSpec: missing fields are defaulted AND reported (never invented silently)', () => {
  const r = parseSpec({ intent: 'something' });
  assert.equal(r.value.intent, 'something');
  assert.equal(r.value.category, 'other', 'the type requires a category, so a default is used');
  assert.deepEqual(codes(r.diagnostics), [], 'a missing optional-quality field is not fatal');
  assert.ok(r.diagnostics.some((d) => d.code === AG.SPEC_FIELD && /category/.test(d.message)));
  assert.ok(r.diagnostics.some((d) => d.code === AG.SPEC_NO_FEATURES));
});

test('parseSpec: a non-object is AG004', () => {
  const r = parseSpec('not an object');
  assert.deepEqual(codes(r.diagnostics), [AG.NOT_OBJECT]);
});

/* ========================================================================== */
/* parsePatch against the REAL catalog                                         */
/* ========================================================================== */

test('parsePatch: a valid multi-op patch parses clean against the real catalog', () => {
  const r = parsePatch(validPatch(), CATALOG);
  assert.deepEqual(settle(r.diagnostics), []);
  assert.equal(r.value.ops.length, 4);
  assert.equal(r.value.summary, 'two nodes, a wire and a tunable');
});

test('parsePatch: an illegal op is dropped alone; every other op survives', () => {
  const opCount = validPatch().ops.length;
  const broken: ShaderPatch = {
    summary: 'one bad op in the middle',
    ops: [
      ...validPatch().ops.slice(0, 2),
      { op: 'add_node', node: { id: 'n_bad', kind: 'voronoi_f2' as never } },
      ...validPatch().ops.slice(2),
    ],
  };
  const r = parsePatch(broken, CATALOG);
  assert.equal(r.value.ops.length, opCount, 'only the illegal op may be dropped');
  assert.deepEqual(codes(r.diagnostics), [AG.UNKNOWN_KIND]);
  assert.ok(!r.value.ops.some((op) => op.op === 'add_node' && op.node.id === 'n_bad'));
});

test('parsePatch: a hallucinated node KIND — exactly one AG012', () => {
  const r = parsePatch(
    { summary: 'x', ops: [{ op: 'add_node', node: { id: 'n_x', kind: 'super_noise' } }] },
    CATALOG,
  );
  assert.deepEqual(codes(r.diagnostics), [AG.UNKNOWN_KIND]);
  assert.equal(r.value.ops.length, 0);
  assert.match(r.diagnostics[0]!.fix!, /uv, time, mouse/, 'the fix must list the legal kinds');
});

test('parsePatch: a hallucinated INPUT port on connect — exactly one AG013', () => {
  const r = parsePatch(
    {
      summary: 'x',
      ops: [
        { op: 'add_node', node: { id: 'n_uv', kind: 'uv' } },
        { op: 'add_node', node: { id: 'n_out', kind: 'output' } },
        // `output` has one input, `color`; `centred` is the port the catalog removed.
        { op: 'connect', from: { node: 'n_uv', port: 'uv' }, to: { node: 'n_out', port: 'centred' } },
      ],
    },
    CATALOG,
  );
  assert.deepEqual(codes(r.diagnostics), [AG.UNKNOWN_INPUT_PORT]);
  assert.equal(r.value.ops.length, 2, 'the two adds survive');
  assert.match(r.diagnostics[0]!.fix!, /Valid inputs: color\.?$/);
});

test('parsePatch: a hallucinated OUTPUT port on connect — exactly one AG014', () => {
  const r = parsePatch(
    {
      summary: 'x',
      ops: [
        { op: 'add_node', node: { id: 'n_uv', kind: 'uv' } },
        { op: 'add_node', node: { id: 'n_out', kind: 'output' } },
        // `uv` has exactly ONE output port, and it is called `uv` — `centered` never existed.
        { op: 'connect', from: { node: 'n_uv', port: 'centered' }, to: { node: 'n_out', port: 'color' } },
      ],
    },
    CATALOG,
  );
  assert.deepEqual(codes(r.diagnostics), [AG.UNKNOWN_OUTPUT_PORT]);
  assert.equal(r.value.ops.length, 2);
  assert.match(r.diagnostics[0]!.fix!, /Valid outputs: uv\.?$/);
});

test('parsePatch: a wire to a node nobody created — exactly one AG015, and the source is named', () => {
  const r = parsePatch(
    {
      summary: 'x',
      ops: [
        { op: 'add_node', node: { id: 'n_out', kind: 'output' } },
        { op: 'connect', from: { node: 'n_ghost', port: 'value' }, to: { node: 'n_out', port: 'color' } },
      ],
    },
    CATALOG,
  );
  assert.deepEqual(codes(r.diagnostics), [AG.DANGLING_WIRE]);
  assert.equal(r.value.ops.length, 1);
  assert.ok(r.diagnostics[0]!.message.includes('n_ghost'));
});

test('parsePatch: a reserved uniform name — exactly one AG018 with a suggested replacement', () => {
  const r = parsePatch(
    {
      summary: 'x',
      ops: [
        { op: 'add_node', node: { id: 'n_out', kind: 'output' } },
        // `sample` is legal GLSL and fatal in MSL/HLSL/WGSL, which is exactly why the
        // parser rejects it rather than letting a driver error surface two stages later.
        { op: 'expose_param', node: 'n_out', param: 'color', name: 'sample' },
      ],
    },
    CATALOG,
  );
  assert.deepEqual(codes(r.diagnostics), [AG.RESERVED_UNIFORM_NAME]);
  assert.equal(r.value.ops.length, 1);
  assert.match(r.diagnostics[0]!.fix!, /`sample_v`/, 'the fix must name the safe replacement');
});

test('parsePatch: a non-identifier uniform name — exactly one AG017', () => {
  const r = parsePatch(
    {
      summary: 'x',
      ops: [
        { op: 'add_node', node: { id: 'n_out', kind: 'output' } },
        { op: 'expose_param', node: 'n_out', param: 'color', name: 'u-my scale' },
      ],
    },
    CATALOG,
  );
  assert.deepEqual(codes(r.diagnostics), [AG.BAD_UNIFORM_NAME]);
  assert.match(r.diagnostics[0]!.fix!, /u_my_scale/, 'the fix must be a usable identifier');
});

test('parsePatch: expose_param on a port this patch already wired — exactly one AG024', () => {
  const r = parsePatch(
    {
      summary: 'x',
      ops: [
        { op: 'add_node', node: { id: 'n_uv', kind: 'uv' } },
        { op: 'add_node', node: { id: 'n_fbm', kind: 'fbm' } },
        { op: 'connect', from: { node: 'n_uv', port: 'uv' }, to: { node: 'n_fbm', port: 'uv' } },
        // `fbm.uv` is now driven by a wire, so a uniform there would be silently ignored —
        // the worst outcome, because the model's edit looks applied and has no effect.
        { op: 'expose_param', node: 'n_fbm', param: 'uv', name: 'u_uvOffset' },
      ],
    },
    CATALOG,
  );
  assert.deepEqual(codes(r.diagnostics), [AG.EXPOSE_WIRED_PORT]);
  assert.equal(r.value.ops.length, 3, 'the three legal ops survive');
  assert.match(r.diagnostics[0]!.fix!, /disconnect/);
});

test('parsePatch: expose_param on a port that does not exist — exactly one AG016', () => {
  const r = parsePatch(
    {
      summary: 'x',
      ops: [
        { op: 'add_node', node: { id: 'n_fbm', kind: 'fbm' } },
        { op: 'expose_param', node: 'n_fbm', param: 'vibrance', name: 'u_vibrance' },
      ],
    },
    CATALOG,
  );
  assert.deepEqual(codes(r.diagnostics), [AG.EXPOSE_UNKNOWN_PORT]);
  assert.equal(r.value.ops.length, 1);
  assert.match(r.diagnostics[0]!.fix!, /Valid inputs: uv, scale, octaves/);
});

test('parsePatch: an op on a node that does not exist — exactly one AG011', () => {
  const r = parsePatch(
    { summary: 'x', ops: [{ op: 'set_param', node: 'n_nowhere', param: 'scale', value: { kind: 'float', value: 1 } }] },
    CATALOG,
  );
  assert.deepEqual(codes(r.diagnostics), [AG.UNKNOWN_NODE]);
  assert.equal(r.value.ops.length, 0);
});

test('parsePatch: an unknown op name — exactly one AG023', () => {
  const r = parsePatch({ summary: 'x', ops: [{ op: 'rewrite_everything' }] }, CATALOG);
  assert.deepEqual(codes(r.diagnostics), [AG.UNKNOWN_OP]);
});

test('parsePatch: a type mismatch on a wire — exactly one AG021', () => {
  const r = parsePatch(
    {
      summary: 'x',
      ops: [
        { op: 'add_node', node: { id: 'n_fbm', kind: 'fbm' } },
        { op: 'add_node', node: { id: 'n_out', kind: 'output' } },
        // fbm.value is a float; output.color is a vec4. Widening is allowed by canConnect,
        // so this one is LEGAL — the illegal case is the narrowing below.
        { op: 'connect', from: { node: 'n_fbm', port: 'value' }, to: { node: 'n_out', port: 'color' } },
      ],
    },
    CATALOG,
  );
  assert.deepEqual(settle(r.diagnostics), [], 'float -> vec4 is a legal widening');

  const bad = parsePatch(
    {
      summary: 'x',
      ops: [
        { op: 'add_node', node: { id: 'n_color', kind: 'const_color' } },
        { op: 'add_node', node: { id: 'n_fbm', kind: 'fbm' } },
        // const_color.color is a vec4; fbm.scale is a float. Narrowing without a node is
        // exactly the error the catalog exists to catch before emission.
        { op: 'connect', from: { node: 'n_color', port: 'color' }, to: { node: 'n_fbm', port: 'scale' } },
      ],
    },
    CATALOG,
  );
  assert.deepEqual(codes(bad.diagnostics), [AG.TYPE_MISMATCH]);
  assert.match(bad.diagnostics[0]!.fix!, /vec_op|swizzle/);
});

test('parsePatch: if NO op survives, the result is an empty patch plus diagnostics', () => {
  const r = parsePatch(
    {
      summary: 'all bad',
      ops: [
        { op: 'add_node', node: { id: 'a', kind: 'nope' } },
        { op: 'connect', from: { node: 'x' }, to: { node: 'y', port: 'z' } },
      ],
    },
    CATALOG,
  );
  assert.equal(r.value.ops.length, 0, 'the loop treats an empty patch as a failed round');
  assert.ok(codes(r.diagnostics).includes(AG.UNKNOWN_KIND));
  assert.ok(codes(r.diagnostics).includes(AG.DANGLING_WIRE));
  // Exactly one diagnostic per broken op — a summary error would double the count and
  // make "how many ops were dropped?" unanswerable from the diagnostics alone.
  assert.equal(r.diagnostics.length, 2);
});

test('parsePatch: a malformed envelope is AG010 and aborts the parse', () => {
  assert.deepEqual(codes(parsePatch({ ops: 'not an array' }, CATALOG).diagnostics), [AG.PATCH_SHAPE]);
  assert.deepEqual(codes(parsePatch([], CATALOG).diagnostics), [AG.PATCH_SHAPE]);
  assert.deepEqual(codes(parsePatch(null, CATALOG).diagnostics), [AG.PATCH_SHAPE]);
});

test('parsePatch: the simulation lets a patch add and wire in one go, in order', () => {
  const r = parsePatch(
    {
      summary: 'chain',
      ops: [
        { op: 'add_node', node: { id: 'n_uv', kind: 'uv' } },
        { op: 'add_node', node: { id: 'n_fbm', kind: 'fbm' } },
        { op: 'add_node', node: { id: 'n_out', kind: 'output' } },
        { op: 'connect', from: { node: 'n_uv' }, to: { node: 'n_fbm', port: 'uv' } },
        { op: 'connect', from: { node: 'n_fbm', port: 'value' }, to: { node: 'n_out', port: 'color' } },
      ],
    },
    CATALOG,
  );
  assert.deepEqual(r.diagnostics, []);
  assert.equal(r.value.ops.length, 5);
});

test('parsePatch: a partial catalog makes ports unverifiable, not silently valid', () => {
  const empty: Partial<Record<keyof typeof CATALOG, NodeSchema>> = {};
  const r = parsePatch(
    { summary: 'x', ops: [{ op: 'add_node', node: { id: 'n_uv', kind: 'uv' } }] },
    empty,
  );
  // A kind the catalog cannot describe is reported, never waved through: with no schema
  // its ports cannot be checked, and "cannot check" must not read as "fine".
  assert.deepEqual(codes(r.diagnostics), [AG.UNKNOWN_KIND]);
  assert.equal(r.value.ops.length, 0);
  assert.match(r.diagnostics[0]!.fix!, /Pass a catalog that covers every kind/);

  // With a catalog that knows `uv` but not the node a wire targets, the wire is the thing
  // reported — again nothing is assumed.
  const partial: Partial<Record<keyof typeof CATALOG, NodeSchema>> = { uv: CATALOG.uv };
  const r2 = parsePatch(
    {
      summary: 'x',
      ops: [
        { op: 'add_node', node: { id: 'n_uv', kind: 'uv' } },
        { op: 'add_node', node: { id: 'n_fbm', kind: 'fbm' } },
      ],
    },
    partial,
  );
  assert.deepEqual(codes(r2.diagnostics), [AG.UNKNOWN_KIND]);
  assert.equal(r2.value.ops.length, 1, 'only the checkable op survives');
  assert.match(r2.diagnostics[0]!.message, /no schema for `fbm`/);

  // The same failure seen from the wire, in a *complete* catalog: a connect to a node
  // whose schema is missing is reported too — but no cascade, because a single-node patch
  // has nothing else to go wrong.
  const r3 = parsePatch(
    {
      summary: 'x',
      ops: [{ op: 'set_param', node: 'n_nowhere', param: 'scale', value: { kind: 'float', value: 1 } }],
    },
    CATALOG,
  );
  assert.deepEqual(codes(r3.diagnostics), [AG.UNKNOWN_NODE]);
  assert.equal(r3.value.ops.length, 0);
});

test('parsePatchJson: a reply that is not JSON yields an empty patch and a diagnostic, never a throw', () => {
  const r = parsePatchJson('the model chose to write prose today', CATALOG);
  assert.equal(r.value.ops.length, 0);
  assert.deepEqual(codes(r.diagnostics), [AG.NO_JSON]);
});

/* ========================================================================== */
/* createStubProvider                                                          */
/* ========================================================================== */

test('stub: the default script drafts the liquid-metal spec', async () => {
  const p = createStubProvider();
  const spec = await p.interpret({ prompt: '液态金属' });
  assert.equal(spec.intent, STUB_INTENTS.liquidMetal.intent);
  assert.ok(spec.features.length >= 3);
});

test('stub: determinism — the same script produces byte-identical requests twice', async () => {
  const runOnce = async (): Promise<string> => {
    const p = createStubProvider({ intents: [STUB_INTENTS.plasma], patches: [] });
    await p.interpret({ prompt: '写一个等离子体' });
    await p.propose(draftContext({ stage: 'draft' }));
    await p.propose(draftContext({ stage: 'repair', diagnostics: [{ severity: 'error', code: 'IR006_TYPE_MISMATCH', message: 'x', fix: 'y' }] }));
    return JSON.stringify(p.calls);
  };
  const a = await runOnce();
  const b = await runOnce();
  assert.equal(a, b);
  assert.ok(a.length > 4_000, 'the recorded prompts must actually contain the prompts');
});

test('stub: `calls` records every request, in order, with its purpose and messages', async () => {
  const p = createStubProvider();
  assert.equal(p.calls.length, 0);

  await p.interpret({ prompt: '液态金属' });
  await p.propose(draftContext({ stage: 'draft' }));
  await p.propose(draftContext({ stage: 'repair', userIntent: '更慢一点' }));

  assert.equal(p.calls.length, 3);
  assert.deepEqual(p.calls.map((c) => c.purpose), ['intent', 'patch:draft', 'patch:repair']);
  assert.ok(p.calls.every((c) => c.json === true));
  assert.ok(p.calls.every((c) => c.messages.length >= 2));
  assert.equal(p.calls[0]!.messages[0]!.role, 'system');

  // The prompt actually carries the document projection and the user's words — that is
  // the whole reason `calls` exists.
  const repair = p.calls[2]!.messages[1]!.content;
  assert.ok(repair.includes('更慢一点'));
  assert.ok(repair.includes('STAGE: repair'));
});

test('stub: `failAtCall` throws on exactly call N, counting across both methods', async () => {
  const p = createStubProvider({ failAtCall: 1 });
  const first = await p.interpret({ prompt: 'x' });
  assert.equal(first.intent, STUB_INTENTS.liquidMetal.intent, 'call 0 must succeed');

  await assert.rejects(
    () => p.propose(draftContext()),
    /scripted failure at call 1/,
  );
  // The failing request is still recorded, so the test can see what was asked.
  assert.equal(p.calls.length, 2);
  assert.equal(p.calls[1]!.purpose, 'patch:draft');

  // Call 2 is past the scripted failure, so the provider keeps working: `failAtCall` is a
  // one-shot, not a latch.
  const spec = await p.interpret({ prompt: 'x' });
  assert.equal(typeof spec.intent, 'string');
});

test('stub: failAtCall 0 fails the very first call', async () => {
  const p = createStubProvider({ failAtCall: 0 });
  await assert.rejects(() => p.interpret({ prompt: 'x' }), /scripted failure at call 0 \(intent\)/);
  assert.equal(p.calls.length, 1);
});

test('stub: an aborted signal throws instead of producing a reply', async () => {
  const p = createStubProvider();
  const ctrl = new AbortController();
  ctrl.abort();
  await assert.rejects(() => p.interpret({ prompt: 'x' }, ctrl.signal), /aborted/);
});

test('stub: LlmReply.text is the JSON of the spec/patch, so it round-trips through parse', async () => {
  const p = createStubProvider();
  const spec = await p.interpret({ prompt: 'x' });
  const patch = await p.propose(draftContext());

  const back = parseIntentJson(JSON.stringify(spec));
  assert.deepEqual(back.value, spec);
  assert.deepEqual(back.diagnostics, []);

  const patchBack = parsePatch(JSON.parse(JSON.stringify(patch)), CATALOG);
  assert.deepEqual(settle(patchBack.diagnostics), []);
  assert.equal(patchBack.value.ops.length, patch.ops.length);
});

test('stub: replies are copies — mutating one cannot corrupt the fixture', async () => {
  const a = createStubProvider();
  const first = await a.interpret({ prompt: 'x' });
  first.tags.push('MUTATED');
  first.intent = 'MUTATED';

  const b = createStubProvider();
  const second = await b.interpret({ prompt: 'x' });
  assert.deepEqual(second, STUB_INTENTS.liquidMetal);
});

test('stub: scripted replies are consumed in order, and expire', async () => {
  const p = createStubProvider({
    intents: [STUB_INTENTS.plasma, STUB_INTENTS.waterCaustics],
    patches: [{ summary: 'one', ops: [] }, { summary: 'two', ops: [] }],
  });
  assert.equal((await p.interpret({ prompt: 'a' })).intent, STUB_INTENTS.plasma.intent);
  assert.equal((await p.interpret({ prompt: 'b' })).intent, STUB_INTENTS.waterCaustics.intent);
  // Past the end, a spec is always available: the loop has no "no intent" branch.
  assert.equal((await p.interpret({ prompt: 'c' })).intent, STUB_INTENTS.liquidMetal.intent);

  assert.equal((await p.propose(draftContext())).summary, 'one');
  assert.equal((await p.propose(draftContext())).summary, 'two');
  // Past the end of the PATCH script: an empty patch, which is how the loop learns the
  // stub has nothing further to say. Repeating would make a stuck loop look like progress.
  const exhausted = await p.propose(draftContext());
  assert.equal(exhausted.ops.length, 0);
  assert.match(exhausted.summary, /exhausted/);
});

/* ========================================================================== */
/* The end-to-end fixture: nothing -> a clean emission on all six targets      */
/* ========================================================================== */

test('stub default script: takes a document from NOTHING to a clean emission on all six targets', async () => {
  const p = createStubProvider();

  const spec = await p.interpret({ prompt: '液态金属表面，要慢速流动的拉丝高光' });
  const doc = createDoc({ spec, targets: [...ALL_TARGETS] });
  assert.equal(doc.graph.nodes.length, 0, 'the run must start from nothing');

  const draft = await p.propose(draftContext({ stage: 'draft', doc }));
  // The patch is a real graph build: add_node x N, connect x N, expose_param x N.
  const count = (op: PatchOp['op']): number => draft.ops.filter((o) => o.op === op).length;
  assert.equal(count('add_node'), 6);
  assert.equal(count('connect'), 5);
  assert.ok(count('expose_param') >= 4);

  // Round-trip through the parser, because that is the path a real provider takes and it
  // is the layer that would drop a hallucinated port.
  const parsed = parsePatch(JSON.parse(JSON.stringify(draft)) as unknown, CATALOG);
  assert.deepEqual(parsed.diagnostics, [], 'the default script must be legal against the real catalog');

  const applied = applyPatch(doc, parsed.value, CATALOG);
  assert.equal(applied.ok, true);
  assert.deepEqual(codes(applied.diagnostics), []);

  const ids = applied.doc.graph.nodes.map((n) => n.id);
  for (const id of ['n_uv', 'n_warp', 'n_detail', 'n_ramp', 'n_grade', 'n_out']) {
    assert.ok(ids.includes(id), `expected node ${id} in ${ids.join(', ')}`);
  }

  // The core assertion: this stub is a fixture, not a placeholder. If a port is renamed
  // in packages/ir/src/catalog/nodes.ts, this test fails — which is the point.
  for (const target of ALL_TARGETS) {
    const emission = emit(applied.doc, { catalog: CATALOG, target });
    const errors = emission.diagnostics.filter((d) => d.severity === 'error');
    assert.deepEqual(
      errors.map((d) => `${target} ${d.code}: ${d.message}`),
      [],
      `${target} must emit with zero error diagnostics`,
    );
    assert.ok(emission.source.length > 500, `${target} produced a suspiciously short source`);
  }

  // Eight tunables reach the uniform table, which is what makes the result tunable
  // without another model call.
  const glsl = emit(applied.doc, { catalog: CATALOG, target: 'glsl' });
  assert.ok(glsl.uniforms.length >= 6, `expected exposed uniforms, got ${glsl.uniforms.length}`);
  assert.equal(glsl.uniforms[0]!.name.startsWith('u_'), true);

  // The stub has nothing further to say, and says so with an empty patch rather than by
  // repeating the build.
  const again = await p.propose(draftContext({ stage: 'repair', doc: applied.doc }));
  assert.equal(again.ops.length, 0);
});

/* ========================================================================== */
/* http.ts with an injected fake fetch                                         */
/* ========================================================================== */

/**
 * One scripted response.
 *
 * The `init` argument is how a step can see the request's abort signal, which is the only way a
 * stub can model a *pending body read being cut off* rather than a connection that refused to
 * open. Steps that do not care simply ignore it, so the existing ones are unchanged.
 */
type FetchStep = (init?: RequestInit) => Response | Promise<Response>;

/** A fake `fetch` that replays `steps` and records every request body it was given. */
function fakeFetch(steps: FetchStep[]): { impl: typeof fetch; calls: () => number; bodies: Record<string, unknown>[]; urls: string[] } {
  let n = 0;
  const bodies: Record<string, unknown>[] = [];
  const urls: string[] = [];
  const impl = (async (url: string | URL | Request, init?: RequestInit): Promise<Response> => {
    urls.push(String(url));
    bodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>);
    const step = steps[Math.min(n, steps.length - 1)];
    n++;
    if (!step) throw new Error('fakeFetch: no steps');
    return step(init);
  }) as unknown as typeof fetch;
  return { impl, calls: () => n, bodies, urls };
}

const chatOk = (content: string, extra: Record<string, unknown> = {}): Response =>
  new Response(
    JSON.stringify({ model: 'fake-model', choices: [{ message: { content } }], usage: { prompt_tokens: 7, completion_tokens: 9 }, ...extra }),
    { status: 200, headers: { 'content-type': 'application/json' } },
  );

const httpOpts = (impl: typeof fetch, over: Partial<Parameters<typeof createHttpProvider>[0]> = {}) => ({
  baseUrl: 'https://api.example.test/v1',
  apiKey: 'test-key',
  model: 'fake-model',
  fetchImpl: impl,
  catalog: CATALOG,
  ...over,
});

function sseReply(text: string, fail = false): Response {
  const bytes = new TextEncoder().encode(text);
  let i = 0;
  return new Response(new ReadableStream<Uint8Array>({ pull(controller) {
    if (i < bytes.length) controller.enqueue(bytes.slice(i, ++i));
    else if (fail) controller.error(new Error('stream reset'));
    else controller.close();
  } }), { headers: { 'content-type': 'text/event-stream; charset=utf-8' } });
}

test('http SSE: byte-split UTF-8, CRLF, comments, usage and DONE produce one complete reply', async () => {
  const events = ': keepalive\r\n\r\ndata: {"model":"test","choices":[{"index":0,"delta":{"role":"assistant"}}]}\r\n\r\n' +
    'data: {"choices":[{"index":0,"delta":{"content":"雨滴"}}]}\n\n' +
    'data: {"choices":[{"index":0,"delta":{"content":"玻璃"},"finish_reason":"stop"}]}\n\n' +
    'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":8}}\n\ndata: [DONE]\n\n';
  const f = fakeFetch([() => sseReply(events)]);
  const p = createHttpProvider(httpOpts(f.impl));
  const reply = await p.complete({ messages: [], purpose: 'test' });
  assert.equal(reply.text, '雨滴玻璃');
  assert.equal(reply.model, 'test');
  assert.deepEqual(reply.usage, { promptTokens: 5, completionTokens: 8 });
  assert.equal(f.bodies[0].stream, true);
  assert.equal(f.bodies[0].stream_options, undefined);
});

test('http SSE: EOF and read failure discard partial content before retry', async () => {
  for (const fail of [false, true]) {
    const f = fakeFetch([() => sseReply('data: {"choices":[{"delta":{"content":"partial"}}]}\n\n', fail),
      () => sseReply('data: {"choices":[{"delta":{"content":"complete"}}]}\n\ndata: [DONE]\n\n')]);
    const reply = await createHttpProvider(httpOpts(f.impl)).complete({ messages: [], purpose: 'test' });
    assert.equal(reply.text, 'complete');
    assert.equal(f.calls(), 2);
  }
});

test('http SSE: malformed/error/length/no-content responses never become valid patches', async () => {
  for (const payload of ['broken', '{"error":{"message":"bad"}}',
    '{"choices":[{"delta":{"content":"{}"},"finish_reason":"length"}]}', '{"choices":[]}']) {
    const f = fakeFetch([() => sseReply(`data: ${payload}\n\ndata: [DONE]\n\n`)]);
    await assert.rejects(createHttpProvider(httpOpts(f.impl)).propose(draftContext()), (e: any) => e.kind === 'envelope');
    assert.equal(f.calls(), 1);
  }
});

test('http SSE: streaming can be disabled for a JSON-only gateway', async () => {
  const f = fakeFetch([() => chatOk('ok')]);
  assert.equal((await createHttpProvider(httpOpts(f.impl, { stream: false, includeUsage: true })).complete({ messages: [], purpose: 'test' })).text, 'ok');
  assert.equal(f.bodies[0].stream, false);
  assert.equal(f.bodies[0].stream_options, undefined);
  const stream = fakeFetch([() => sseReply('data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n')]);
  await createHttpProvider(httpOpts(stream.impl, { includeUsage: true })).complete({ messages: [], purpose: 'test' });
  assert.deepEqual(stream.bodies[0].stream_options, { include_usage: true });
});

test('http SSE: arriving chunks renew the idle deadline; cancellation never retries', async () => {
  for (const cancel of [false, true]) {
    const ctrl = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    const f = fakeFetch([(init) => new Response(new ReadableStream({ start(controller) {
      let n = 0;
      const send = () => {
        if (n++ === 5) {
          controller.enqueue(new TextEncoder().encode('data: [DONE]\n\n'));
          controller.close();
          return;
        }
        controller.enqueue(new TextEncoder().encode('data: {"choices":[{"delta":{"content":"x"}}]}\n\n'));
        timer = setTimeout(send, 20);
      };
      init?.signal?.addEventListener('abort', () => {
        clearTimeout(timer);
        controller.error(new Error('aborted'));
      }, { once: true });
      send();
    }, cancel() { clearTimeout(timer); } }), { headers: { 'content-type': 'text/event-stream' } })]);
    const pending = createHttpProvider(httpOpts(f.impl, { timeoutMs: 70 })).complete({ messages: [], purpose: 'test' }, ctrl.signal);
    if (cancel) {
      ctrl.abort();
      await assert.rejects(pending);
    } else assert.equal((await pending).text, 'xxxxx');
    assert.equal(f.calls(), 1);
  }
});

test('http SSE: an idle body times out with the original failure classification', async () => {
  const f = fakeFetch([(init) => new Response(new ReadableStream({ start(controller) {
    init?.signal?.addEventListener('abort', () => controller.error(new Error('aborted')), { once: true });
  } }), { headers: { 'content-type': 'text/event-stream' } })]);
  const keepAlive = setTimeout(() => {}, 500);
  try {
    await assert.rejects(createHttpProvider(httpOpts(f.impl, { timeoutMs: 15, maxRetries: 0 })).complete({ messages: [], purpose: 'test' }),
      (e: any) => e.kind === 'timeout');
  } finally { clearTimeout(keepAlive); }
});

test('http: refinement preserves edits and wiring to existing custom layers', async () => {
  const doc = createDoc({ graph: { nodes: [
    { id: 'drops', kind: 'custom_glsl', body: 'return vec4(0.0, 0.0, 0.0, 0.0);' },
    { id: 'out', kind: 'output' },
  ], edges: [] } });
  const ops: PatchOp[] = [
    { op: 'set_body', node: 'drops', body: 'return vec4(0.1, 0.2, 0.3, 0.4);' },
    { op: 'connect', from: { node: 'drops', port: 'value' }, to: { node: 'out', port: 'color' } },
  ];
  const f = fakeFetch([() => chatOk(JSON.stringify({ summary: 'edit rain layer', ops }))]);
  const p = createHttpProvider(httpOpts(f.impl));
  const patch = await p.propose(draftContext({ stage: 'refine', doc }));
  assert.deepEqual(patch.ops, ops);
  assert.deepEqual(settle(p.lastDiagnostics.flat()), []);
  const result = applyPatch(doc, patch, CATALOG);
  assert.equal(result.ok, true);
  assert.equal(result.doc.graph.nodes[0].body, 'return vec4(0.1, 0.2, 0.3, 0.4);');
  assert.deepEqual(result.doc.graph.nodes[1].inputs?.color, { node: 'drops', port: 'value' });
});

test('http: a successful interpret sends the OpenAI-compatible body and parses the reply', async () => {
  const f = fakeFetch([() => chatOk(JSON.stringify(STUB_INTENTS.plasma))]);
  const p = createHttpProvider(httpOpts(f.impl));

  const spec = await p.interpret({ prompt: '等离子体' });

  assert.equal(f.calls(), 1);
  assert.equal(f.urls[0], 'https://api.example.test/v1/chat/completions');
  assert.deepEqual(f.bodies[0]!['response_format'], { type: 'json_object' });
  assert.equal(f.bodies[0]!['model'], 'fake-model');
  const messages = f.bodies[0]!['messages'] as { role: string; content: string }[];
  assert.equal(messages[0]!.role, 'system');
  assert.equal(messages[1]!.role, 'user');
  assert.equal(spec.intent, STUB_INTENTS.plasma.intent);
  assert.deepEqual(p.lastDiagnostics[0], []);
});

test('http: `complete` surfaces model and token usage for provenance', async () => {
  const f = fakeFetch([() => chatOk('{"ok":true}')]);
  const p = createHttpProvider(httpOpts(f.impl));
  const reply = await p.complete({ messages: [{ role: 'user', content: 'hi' }], json: true, purpose: 'probe' });
  assert.equal(reply.model, 'fake-model');
  assert.deepEqual(reply.usage, { promptTokens: 7, completionTokens: 9 });
  assert.equal(reply.text, '{"ok":true}');
});

test('http: a 401 is NOT retried, and the error names the status and the provider body', async () => {
  const f = fakeFetch([
    () => new Response('{"error":{"message":"Incorrect API key provided","code":"invalid_api_key"}}', { status: 401, statusText: 'Unauthorized' }),
  ]);
  const p = createHttpProvider(httpOpts(f.impl));

  await assert.rejects(
    () => p.interpret({ prompt: 'x' }),
    (err: Error) => {
      assert.equal(f.calls(), 1, 'a bad API key must not be retried — it can only fail again');
      assert.match(err.message, /HTTP 401 Unauthorized/);
      assert.match(err.message, /Incorrect API key provided/, 'the provider body must survive verbatim');
      assert.match(err.message, /not transient/);
      assert.match(err.message, /\(intent\)/, 'the error must say which call failed');
      return true;
    },
  );
});

test('http: a 400 is not retried either', async () => {
  const f = fakeFetch([() => new Response('{"error":"bad model"}', { status: 400, statusText: 'Bad Request' })]);
  const p = createHttpProvider(httpOpts(f.impl));
  await assert.rejects(() => p.propose(draftContext()), /HTTP 400 Bad Request/);
  assert.equal(f.calls(), 1);
});

test('http: 429 then 200 — exactly one retry, then success, and it stays fast', async () => {
  const started = Date.now();
  const f = fakeFetch([
    () => new Response('{"error":{"message":"Rate limit reached"}}', { status: 429, statusText: 'Too Many Requests' }),
    () => chatOk(JSON.stringify(STUB_INTENTS.waterCaustics)),
  ]);
  const p = createHttpProvider(httpOpts(f.impl));

  const spec = await p.interpret({ prompt: '水' });

  assert.equal(f.calls(), 2, 'one retry, no more');
  assert.equal(spec.intent, STUB_INTENTS.waterCaustics.intent);
  const elapsed = Date.now() - started;
  assert.ok(elapsed < 1_000, `transport backoff must stay bounded; took ${elapsed}ms`);
});

test('http: a 500 is retried, then surfaced with its body when it keeps failing', async () => {
  const f = fakeFetch([() => new Response('upstream exploded', { status: 503, statusText: 'Service Unavailable' })]);
  const p = createHttpProvider(httpOpts(f.impl, { maxRetries: 2 }));

  await assert.rejects(
    () => p.interpret({ prompt: 'x' }),
    (err: Error) => {
      assert.equal(f.calls(), 3, 'initial attempt + 2 retries');
      assert.match(err.message, /HTTP 503 Service Unavailable/);
      assert.match(err.message, /upstream exploded/);
      assert.match(err.message, /transient/);
      return true;
    },
  );
});

test('http: a network error is retried, and a later success wins', async () => {
  const f = fakeFetch([
    () => {
      throw new TypeError('fetch failed');
    },
    () => chatOk('{"intent":"metal","category":"surface","tags":[],"features":["flows"]}'),
  ]);
  const p = createHttpProvider(httpOpts(f.impl));
  const spec = await p.interpret({ prompt: 'metal' });
  assert.equal(f.calls(), 2);
  assert.equal(spec.category, 'surface');
});

test('http: maxRetries = 0 means exactly one attempt', async () => {
  const f = fakeFetch([() => {
    throw new TypeError('fetch failed');
  }]);
  const p = createHttpProvider(httpOpts(f.impl, { maxRetries: 0 }));
  await assert.rejects(() => p.interpret({ prompt: 'x' }), /network error/);
  assert.equal(f.calls(), 1);
});

test('http: a reply whose content is not JSON produces a parse DIAGNOSTIC, not a crash', async () => {
  const f = fakeFetch([() => chatOk('I would rather describe the shader in prose.')]);
  const p = createHttpProvider(httpOpts(f.impl));

  const patch = await p.propose(draftContext({ stage: 'draft' }));

  assert.equal(patch.ops.length, 0, 'zero surviving ops is a valid result, not an error');
  const diags = p.lastDiagnostics[0]!;
  assert.deepEqual(codes(diags), [AG.NO_JSON]);
  assert.ok(diags[0]!.fix, 'every diagnostic carries an actionable fix');
});

test('http: a reply that is JSON but not a patch yields diagnostics, not a throw', async () => {
  const f = fakeFetch([() => chatOk('[1, 2, 3]')]);
  const p = createHttpProvider(httpOpts(f.impl));
  const patch = await p.propose(draftContext());
  assert.equal(patch.ops.length, 0);
  assert.ok(p.lastDiagnostics[0]!.length >= 1);
});

test('http: a non-JSON HTTP envelope is a loud error naming the status', async () => {
  const f = fakeFetch([() => new Response('<html>502 Bad Gateway</html>', { status: 200, statusText: 'OK' })]);
  const p = createHttpProvider(httpOpts(f.impl));
  await assert.rejects(() => p.complete({ messages: [{ role: 'user', content: 'x' }], purpose: 'probe' }), /non-JSON envelope/);
});

test('http: a 200 with no choices[0].message.content is a loud error', async () => {
  const f = fakeFetch([() => new Response(JSON.stringify({ choices: [] }), { status: 200 })]);
  const p = createHttpProvider(httpOpts(f.impl));
  await assert.rejects(() => p.complete({ messages: [{ role: 'user', content: 'x' }], purpose: 'probe' }), /choices\[0\]\.message\.content/);
});

test('http: a null JSON envelope produces a provider diagnostic, not a TypeError', async () => {
  const f = fakeFetch([() => new Response('null', { status: 200 })]);
  const p = createHttpProvider(httpOpts(f.impl));
  await assert.rejects(
    () => p.complete({ messages: [{ role: 'user', content: 'x' }], purpose: 'probe' }),
    (err: Error) => {
      assert.ok(err instanceof ProviderError);
      assert.equal(err.kind, 'envelope');
      assert.equal(err.retryable, false);
      assert.match(err.message, /HTTP 200/);
      return true;
    },
  );
  assert.equal(f.calls(), 1);
});

test('http: an image is sent as an image part only when vision is on', async () => {
  const image = 'data:image/png;base64,AAAA';
  const withVision = fakeFetch([() => chatOk(JSON.stringify(STUB_INTENTS.plasma))]);
  await createHttpProvider(httpOpts(withVision.impl, { vision: true })).interpret({ prompt: 'like this', image });
  const parts = (withVision.bodies[0]!['messages'] as { content: unknown }[])[2]!.content as { type: string }[];
  assert.equal(parts[0]!.type, 'text');
  assert.equal(parts[1]!.type, 'image_url');

  const noVision = fakeFetch([() => chatOk(JSON.stringify(STUB_INTENTS.plasma))]);
  await createHttpProvider(httpOpts(noVision.impl, { vision: false })).interpret({ prompt: 'like this', image });
  const msgs = noVision.bodies[0]!['messages'] as { content: unknown }[];
  assert.equal(msgs.length, 2, 'without vision the image is not attached');
  assert.equal(typeof msgs[1]!.content, 'string');
});

test('http: construction rejects a missing key, base URL, model, or fetch', () => {
  assert.throws(() => createHttpProvider(httpOpts(fakeFetch([]).impl, { apiKey: '' })), /`apiKey` is required/);
  assert.throws(() => createHttpProvider(httpOpts(fakeFetch([]).impl, { baseUrl: '' })), /`baseUrl` is required/);
  assert.throws(() => createHttpProvider(httpOpts(fakeFetch([]).impl, { model: '' })), /`model` is required/);
  const noFetch = { ...httpOpts(fakeFetch([]).impl), fetchImpl: undefined } as unknown as Parameters<typeof createHttpProvider>[0];
  const saved = globalThis.fetch;
  try {
    // @ts-expect-error — deliberately removing the global to prove the guard fires.
    delete globalThis.fetch;
    assert.throws(() => createHttpProvider(noFetch), /no fetch available/);
  } finally {
    globalThis.fetch = saved;
  }
});

test('http: a trailing slash on baseUrl does not produce a double slash', async () => {
  const f = fakeFetch([() => chatOk('{"intent":"x","category":"c","tags":[],"features":["f"]}')]);
  const p = createHttpProvider(httpOpts(f.impl, { baseUrl: 'https://api.example.test/v1/' }));
  await p.interpret({ prompt: 'x' });
  assert.equal(f.urls[0], 'https://api.example.test/v1/chat/completions');
});

test('http: an aborted caller signal is not retried', async () => {
  const ctrl = new AbortController();
  const f = fakeFetch([
    () => {
      ctrl.abort();
      throw new DOMException('aborted', 'AbortError');
    },
  ]);
  const p = createHttpProvider(httpOpts(f.impl));
  await assert.rejects(() => p.interpret({ prompt: 'x' }, ctrl.signal), /aborted by the caller/);
  assert.equal(f.calls(), 1);
});

/* ========================================================================== */
/* http.ts — what a failed round trip is, and whether it is worth repeating    */
/* ========================================================================== */

/**
 * A 200 whose body cannot be read — the failure this section exists for.
 *
 * The reported bug arrived as `HTTP 200 … (the response body could not be read)` followed by
 * advice to check the API key that the `200` had already proved correct. Three things were wrong
 * with the old handling, and each has a test below:
 *
 *   1. the cause was discarded and replaced with a sentence, so `JSON.parse` of that sentence
 *      reported a *content* problem and the export path blamed the emitter;
 *   2. a body that failed to arrive was not retried, while a connection that failed to open was
 *      retried twice — the same class of event, treated as two;
 *   3. every failure produced the same `fix`, including the ones that had just disproved it.
 */
function brokenBody(body: string, failWith: () => Error): Response {
  const res = new Response(body, { status: 200, statusText: 'OK' });
  let reads = 0;
  res.text = async (): Promise<string> => {
    reads += 1;
    if (reads === 1) throw failWith();
    return body;
  };
  return res;
}

test('http: a body read that fails once is retried, and the retry wins', async () => {
  const f = fakeFetch([
    () => brokenBody('{"choices":[{"message":{"content":"ignored"}}]}', () => new TypeError('terminated')),
    () => chatOk(JSON.stringify(STUB_INTENTS.plasma)),
  ]);
  const p = createHttpProvider(httpOpts(f.impl));

  const spec = await p.interpret({ prompt: 'x' });

  assert.equal(f.calls(), 2, 'a mid-response drop must be retried like any other transport failure');
  assert.equal(spec.intent, STUB_INTENTS.plasma.intent);
});

test('http: a body read that keeps failing names the cause, the status and the class of failure', async () => {
  const f = fakeFetch([() => brokenBody('', () => new TypeError('terminated'))]);
  const p = createHttpProvider(httpOpts(f.impl, { maxRetries: 2 }));

  await assert.rejects(
    () => p.interpret({ prompt: 'x' }),
    (err: Error) => {
      assert.equal(f.calls(), 3, 'initial attempt + 2 retries: the body is part of the round trip');
      assert.ok(err instanceof ProviderError, `expected a ProviderError, got ${err.name}`);
      assert.equal((err as ProviderError).kind, 'read');
      assert.equal((err as ProviderError).retryable, true);
      // The three facts that turn this from "check your key" into an actionable report.
      assert.match(err.message, /could not be read after HTTP 200/);
      assert.match(err.message, /terminated/, "the underlying error's own words must survive");
      assert.match(err.message, /not a configuration problem/);
      assert.doesNotMatch(err.message, /API key is|check the key/i);
      return true;
    },
  );
});

test('http: our own per-attempt ceiling firing mid-body is a timeout, and it names the ceiling', async () => {
  let signal: AbortSignal | undefined;
  const p = createHttpProvider(
    httpOpts(
      fakeFetch([
        (init) => {
          signal = init?.signal ?? undefined;
          const res = new Response('{"choices":[]}', { status: 200, statusText: 'OK' });
          // A real stack rejects a pending body read when the request signal aborts; a stub that
          // merely returned a resolved Response would never reach the timeout branch at all.
          res.text = () =>
            new Promise<string>((_resolve, reject) => {
              signal?.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
            });
          return res;
        },
      ]).impl,
      { timeoutMs: 30, maxRetries: 0 },
    ),
  );

  const started = Date.now();
  await assert.rejects(
    () => p.interpret({ prompt: 'x' }),
    (err: Error) => {
      assert.ok(err instanceof ProviderError);
      assert.equal((err as ProviderError).kind, 'timeout');
      assert.match(err.message, /per-attempt timeout of 30 ms/);
      assert.match(err.message, /already answered HTTP 200/, 'the 200 is what rules out configuration');
      // It *names* the API key only to say it is not the problem; it must not send the reader to
      // check configuration that the 200 has already proved correct.
      assert.match(err.message, /are all correct/);
      assert.doesNotMatch(err.message, /check the (api )?key/i);
      return true;
    },
  );
  assert.ok(Date.now() - started < 5_000, 'a 30 ms ceiling must not be enforced by a real clock wait');
});

test('http: an empty 200 body is an envelope failure that is worth retrying', async () => {
  const good = () => chatOk(JSON.stringify(STUB_INTENTS.plasma));
  const once = fakeFetch([() => new Response('', { status: 200, statusText: 'OK' }), good]);
  const spec = await createHttpProvider(httpOpts(once.impl)).interpret({ prompt: 'x' });
  assert.equal(once.calls(), 2, 'an empty 200 is a proxy swallowing the reply; retry it');
  assert.equal(spec.intent, STUB_INTENTS.plasma.intent);

  const always = fakeFetch([() => new Response('', { status: 200, statusText: 'OK' })]);
  await assert.rejects(
    () => createHttpProvider(httpOpts(always.impl, { maxRetries: 0 })).interpret({ prompt: 'x' }),
    /empty body/,
  );
});

test('http: a truncated reply is retried, a proxy error page is not', async () => {
  const truncated = fakeFetch([
    // Ends mid-structure: the connection was cut inside the JSON.
    () => new Response('{"choices":[{"message":{"content":"a shader', { status: 200, statusText: 'OK' }),
    () => chatOk(JSON.stringify(STUB_INTENTS.plasma)),
  ]);
  const p = createHttpProvider(httpOpts(truncated.impl));
  const spec = await p.interpret({ prompt: 'x' });
  assert.equal(truncated.calls(), 2, 'a prefix of a real reply deserves another attempt');
  assert.equal(spec.intent, STUB_INTENTS.plasma.intent);

  const page = fakeFetch([
    () => new Response('<html><body>502 Bad Gateway</body></html>', { status: 200, statusText: 'OK' }),
  ]);
  await assert.rejects(
    () => createHttpProvider(httpOpts(page.impl)).interpret({ prompt: 'x' }),
    (err: Error) => {
      assert.equal(page.calls(), 1, 'a captive portal answers the same way every time');
      assert.ok(err instanceof ProviderError);
      assert.equal((err as ProviderError).kind, 'envelope');
      assert.equal((err as ProviderError).retryable, false);
      return true;
    },
  );
});

test('isTransientStatus: 429 and 5xx only', () => {
  assert.equal(isTransientStatus(429), true);
  assert.equal(isTransientStatus(500), true);
  assert.equal(isTransientStatus(503), true);
  assert.equal(isTransientStatus(400), false);
  assert.equal(isTransientStatus(401), false);
  assert.equal(isTransientStatus(403), false);
  assert.equal(isTransientStatus(404), false);
  assert.equal(isTransientStatus(422), false);
});

/* ========================================================================== */
/* prompt.ts — the shape rules that keep a reply parseable                     */
/* ========================================================================== */

test('prompt: the patch system prompt states the hard rules in the imperative', () => {
  const s = systemPrompt('patch');
  assert.match(s, /Reply with ONE JSON object and nothing else/);
  assert.match(s, /Emit ONLY JSON matching the shape above/);
  assert.match(s, /Use ONLY node kinds and port names that appear in the CATALOG/);
  assert.match(s, /\[A-Za-z_\]\[A-Za-z0-9_\]\*/);
  assert.match(s, /Shader code is allowed ONLY in custom_glsl/);
  assert.match(s, /ONE patch per call/);
  assert.match(s, /Explain every op in its `reason`/);
  assert.ok(s.includes('"op": "add_node"'), 'the op shapes are shown verbatim');
  assert.match(s, /"kind": "float"/, 'the ParamValue shapes are shown verbatim');
});

test('prompt: the intent system prompt forbids graph details', () => {
  const s = systemPrompt('intent');
  assert.match(s, /Reply with ONE JSON object and nothing else/);
  assert.match(s, /Do not name nodes, ports, node kinds/);
  assert.match(s, /outOfScope/);
  assert.ok(!s.includes('add_node'), 'the intent call must not be shown the patch format');
});

test('prompt: the intent prompt no longer forbids a lighting model', () => {
  /*
   * It used to say "no geometry, no lighting model, no meshes" with the reason "the graph has
   * no 3D pass". That reason expired when the surface group landed, and a stale prohibition is
   * worse than a missing instruction: asked for 三渲二, a model that has been told lighting is
   * impossible builds a *picture* of cel shading out of screen coordinates — which looks
   * approximately right on a quad and is wrong on every surface. The rule now has to say what
   * is impossible (geometry, shadow maps, scene reflections) rather than a whole category that
   * became possible.
   */
  const s = systemPrompt('intent');
  assert.ok(
    !/no lighting model/i.test(s),
    'the intent prompt still forbids a lighting model; the surface inputs made that false',
  );
  assert.match(s, /surface normal, world position, view direction and light/);
  assert.match(s, /明暗分界随光照移动/, 'a surface example anchors the new capability');
  // The parts that are still impossible must stay forbidden, or the model promises them.
  assert.match(s, /shadow maps/);
  assert.match(s, /extra meshes/);
});

test('prompt: the patch prompt teaches the two quality fixes and the w convention', () => {
  /*
   * `antialias` and `tile` both existed in the catalog before this guide did, and neither was
   * reachable in practice: a model cannot use a capability whose *purpose* it has never been told,
   * and the requests that need them never name them — "the pattern shimmers when I zoom out", "the
   * texture shows a seam in Unity". The inventory lists the kind and the input; the guide is what
   * says when.
   *
   * The `w` convention is here for the same reason: it is the difference between a direction and a
   * point, it is invisible in the graph, and every op that reads `.xyz` silently ignores the
   * mistake.
   */
  const s = systemPrompt('patch');
  assert.match(s, /antialias/, 'the anti-aliasing node must be named by purpose');
  assert.match(s, /fwidth|derivative/, 'and its mechanism named, so the model can reason about it');
  assert.match(s, /shimmer|crawl|alias/, 'and the symptom it fixes, which is how the request arrives');
  assert.match(s, /\btile\b/, 'the tileable input must be named');
  assert.match(s, /tileable|seamless|seam/, 'with the request that calls for it');
  // The generators are named, so the model knows which inputs carry it.
  for (const kind of ['fbm', 'voronoi', 'domain_warp', 'raymarch']) {
    assert.ok(s.includes(kind), `${kind} samples noise and must be named as tileable`);
  }
  assert.match(s, /w\` is a TAG|`w` is a TAG/, 'the w convention must be stated as a rule');
});

test('prompt: a custom body is told never to write a matrix product with `*`', () => {
  /*
   * `RW014_MATRIX_PRODUCT` refuses such a body, so a model that writes one loses a round. Worse,
   * before that rule existed the body emitted cleanly on four targets and failed on two, because
   * HLSL's row-major constructors make `mul(matrix, vector)` a transpose — the rotation comes out
   * mirrored. Telling the model up front is cheaper than either.
   */
  const s = systemPrompt('patch');
  assert.match(s, /mat2_mul_/, 'the alias must be named');
  assert.match(s, /ROW-major|row-major/, 'and the reason, so the rule is not arbitrary');
  assert.match(s, /Never write a matrix product with/, 'stated as an instruction');
  // And the escape hatch can do surface shading directly, which a model has no way to guess:
  // naming a surface symbol in a body is what turns the document into a surface document.
  for (const symbol of SURFACE_HOST_SYMBOLS) {
    assert.ok(s.includes(symbol), `${symbol} is readable from a body and must be named as such`);
  }
  assert.match(s, /surface document automatically/, 'with the consequence stated');
});

test('prompt: the escape-hatch alias list matches the aliases the rewriter actually resolves', () => {
  /*
   * A model writes a snippet against this list. Missing an alias means the model avoids a
   * construct it could have used; naming one that does not exist means it writes a body the
   * rewriter leaves alone — which is a compile error on some target, and the one thing the subset
   * cannot check for it. Derived from `REWRITE_ALIASES`, so the two cannot drift apart.
   */
  const s = systemPrompt('patch');
  for (const alias of REWRITE_ALIASES) {
    assert.ok(s.includes(alias), `the alias \`${alias}\` is resolvable but the prompt never names it`);
  }
});

test('prompt: the patch prompt teaches the surface group and its consequences', () => {  const s = systemPrompt('patch');
  for (const kind of ['normal', 'world_pos', 'view_dir', 'light_dir', 'dot']) {
    assert.ok(s.includes(kind), `the surface kind \`${kind}\` is never named in the prompt`);
  }
  // The reason a model will not use these on its own: it does not know the effect class.
  assert.match(s, /三渲二|cel shading/);
  // And the two consequences that change what the patch may do.
  assert.match(s, /surface shader/, 'the entry-point consequence must be stated');
  assert.match(s, /uv. comes from the mesh/, 'the coordinate consequence must be stated');
  assert.match(s, /no shadow map/i, 'the limit that is still real must stay stated');
});

test('prompt: the catalog inventory is GENERATED from the catalog and is complete', () => {
  const prompt = patchUserPrompt(draftContext(), CATALOG);
  // Every kind, by name, in the authoritative whitelist line.
  for (const kind of Object.keys(CATALOG)) {
    assert.ok(prompt.includes(kind), `catalog kind \`${kind}\` is missing from the prompt`);
  }
  // Real port names with real types, for a relevant kind.
  assert.match(prompt, /• fbm \[noise\] "FBM"/);
  assert.match(prompt, /inputs:\s+uv: vec2, scale: float, octaves: float, lacunarity: float, gain: float, ridged: float/);
  assert.match(prompt, /outputs: value: float/);
  // The actual schema summary, not a hand-written paraphrase.
  assert.ok(prompt.includes('Fractal Brownian motion'));
});

test('prompt: the document is rendered from the projection, never dumped as IR JSON', () => {
  const doc: ShaderDoc = createDoc({ spec: STUB_INTENTS.liquidMetal });
  const ctx = draftContext({
    doc,
    nodeLines: ['n_uv (uv) — uv source', 'n_out (output) — final image'],
    uniformLines: ['u_gain: float = 0.5 [0.2..0.8] — 细节衰减 (纹理)'],
    userIntent: '更慢的流动',
  });
  const prompt = patchUserPrompt(ctx, CATALOG);

  assert.ok(prompt.includes('n_uv (uv) — uv source'));
  assert.ok(prompt.includes('u_gain: float = 0.5'));
  assert.ok(prompt.includes('更慢的流动'));
  // The cost/latency failure the projection exists to prevent: shipping the document.
  assert.ok(!prompt.includes(JSON.stringify(doc.graph)), 'the whole graph must never be serialised into the prompt');
  assert.ok(!prompt.includes(doc.id), 'not even the document id');
  assert.ok(prompt.length < 40_000, `prompt is ${prompt.length} chars; the projection is not doing its job`);
});

test('prompt: an empty document says "(none)" rather than silently rendering nothing', () => {
  const prompt = patchUserPrompt(draftContext(), CATALOG);
  assert.match(prompt, /nodes \(0\):\n\s+\(none\)/);
  assert.match(prompt, /tunable uniforms \(0\):\n\s+\(none\)/);
});

test('prompt: stage repair carries the diagnostics verbatim, fixes included', () => {
  const diagnostic: Diagnostic = {
    severity: 'error',
    code: 'IR006_TYPE_MISMATCH',
    message: 'Type mismatch: `n_fbm.value` is float but `n_out.color` expects vec4.',
    at: { node: 'n_out', port: 'color' },
    fix: 'Insert a `vec_op`/constructor node to widen float to vec4.',
  };
  const prompt = patchUserPrompt(draftContext({ stage: 'repair', diagnostics: [diagnostic] }), CATALOG);

  assert.ok(prompt.includes('DIAGNOSTICS TO FIX'));
  assert.ok(prompt.includes(diagnostic.code));
  assert.ok(prompt.includes(diagnostic.message), 'the message must be verbatim');
  assert.ok(prompt.includes(diagnostic.fix!), 'the fix is the field the repair loop actually consumes');
  assert.match(prompt, /node=n_out port=color/);
  assert.match(prompt, /Change AS LITTLE AS POSSIBLE/);
  assert.match(prompt, /Do not restructure/);
});

test('prompt: each stage gets its own instruction', () => {
  const draft = patchUserPrompt(draftContext({ stage: 'draft' }), CATALOG);
  assert.match(draft, /STAGE: draft/);
  assert.match(draft, /Build the graph/);

  const refine = patchUserPrompt(draftContext({ stage: 'refine' }), CATALOG);
  assert.match(refine, /STAGE: refine/);
  assert.match(refine, /prefer `set_param` \/ `expose_param`/);
});

test('prompt: the relevant kinds are ordered first but nothing is filtered out', () => {
  const ctx = draftContext({
    nodeLines: ['n_v (voronoi) — cells'],
    diagnostics: [{ severity: 'error', code: 'IR006_TYPE_MISMATCH', message: 'x', fix: 'Insert a `vec_op` node to widen float to vec4.' }],
  });
  const prompt = patchUserPrompt(ctx, CATALOG);
  // `voronoi` is in the document and `vec_op` is named by a fix, so both are promoted.
  const relevant = prompt.indexOf('Most relevant to this stage');
  assert.ok(relevant !== -1 && prompt.indexOf('voronoi', relevant) !== -1);
  assert.ok(prompt.indexOf('vec_op', relevant) !== -1);
  // …and the tail of the catalog is still present under the terser heading.
  assert.ok(prompt.includes('The rest of the catalog'));
});

test('prompt: the intent user prompt carries the request, the image note and the targets', () => {
  const input: IntentInput = {
    prompt: '液态金属',
    image: 'data:image/png;base64,AAAA',
    mustUseTargets: ['glsl', 'wgsl'],
  };
  const prompt = intentUserPrompt(input);
  assert.ok(prompt.includes('液态金属'));
  assert.ok(prompt.includes('reference image is attached'));
  assert.ok(prompt.includes('glsl, wgsl'));
});

test('prompt: a refinement intent prompt carries the previous spec forward', () => {
  const basedOn = createDoc({ spec: STUB_INTENTS.liquidMetal });
  const prompt = intentUserPrompt({ prompt: '再慢一点', basedOn });
  assert.ok(prompt.includes('REFINEMENT'), 'the model must know it is not starting over');
  assert.ok(prompt.includes(STUB_INTENTS.liquidMetal.intent));
  assert.ok(prompt.includes('再慢一点'));
});
