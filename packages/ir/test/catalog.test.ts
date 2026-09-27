/**
 * Catalog conformance tests
 * =========================
 *
 * Every rule in `src/catalog/nodes.ts` that can be machine-checked, checked. These tests
 * are deliberately written against the *frozen* modules (`types.ts`, `glslSubset.ts`,
 * `rewrite.ts`, `emitter.ts`) as they are.
 *
 * `emitBodies` now destructures multi-output nodes for real: a body may return a `vec4`
 * and its declared ports become `.x/.y/.z/.w` in order, but only when there are at most
 * four of them AND every port is scalar (`TYPE_ARITY[port.type] === 1`). Tests 1b/2c/2d
 * pin that contract here, so a schema edit that would only surface as `IR033` at emit
 * time fails in this file instead; tests 11a-11c wire each port of every multi-output
 * kind — including every non-`.x` port — through the emitter to prove it destructures.
 *
 * Run (from packages/ir):
 *
 *   node --test --experimental-strip-types --experimental-test-isolation=none \
 *     --import ./test/register-hooks.mjs test/catalog.test.ts
 *
 * `--experimental-test-isolation=none` keeps the runner in-process: the default
 * process-per-file isolation needs to spawn a child with piped stdio, which the
 * restricted sandbox denies (`spawn EPERM`).
 *
 * `--import ./test/register-hooks.mjs` teaches Node's type stripping to resolve the
 * `.js` specifiers that AGENT_CONTRACT.md §2 mandates (see test/ts-resolve.mjs).
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import { CATALOG, CATEGORIES } from '../src/catalog/index.js';
import {
  BANNED_BUILTIN_SPELLINGS,
  GLSL_LIBRARY,
  PORTABLE_FUNCTIONS,
  REWRITE_ALIASES,
} from '../src/glslSubset.js';
import { TARGET_EMITTERS, emit, emitBodies, analyzeGraph, octavesFor, registerTarget, splitReturn } from '../src/emitter.js';
import { lintSubset, rewriteSource, tokenize, type Token } from '../src/rewrite.js';
import {
  IR_VERSION,
  NODE_KINDS,
  TYPE_ARITY,
  type Edge,
  type NodeKind,
  type NodeSchema,
  type ParamValue,
  type ShaderDoc,
  type ShaderNode,
  type TargetFormat,
  type WireRef,
} from '../src/types.js';

/* -------------------------------------------------------------------------- */
/* Helpers                                                                     */
/* -------------------------------------------------------------------------- */

const KINDS = [...NODE_KINDS] as NodeKind[];
const ENTRIES: [NodeKind, NodeSchema][] = KINDS.map((k) => [k, CATALOG[k]]);
const TARGETS: TargetFormat[] = ['glsl', 'wgsl', 'hlsl', 'msl', 'shaderlab', 'ue'];

/** Comment-free copy of a body, layout preserved (comments become one space). */
function withoutComments(src: string): string {
  return tokenize(src)
    .map((t) => (t.kind === 'comment' ? ' ' : t.text))
    .join('');
}

/** Tokens that carry code: no whitespace, no comments. */
function codeTokens(src: string): Token[] {
  return tokenize(src).filter((t) => t.kind !== 'space' && t.kind !== 'comment');
}

/* -------------------------------------------------------------------------- */
/* 1. Shape                                                                    */
/* -------------------------------------------------------------------------- */

test('1. every kind in NODE_KINDS has a schema keyed by itself with a non-empty body', () => {
  const problems: string[] = [];
  for (const kind of KINDS) {
    const schema = CATALOG[kind];
    if (!schema) {
      problems.push(`${kind}: missing from CATALOG`);
      continue;
    }
    if (schema.kind !== kind) problems.push(`${kind}: schema.kind is ${schema.kind}`);
    if (typeof schema.glsl !== 'string' || schema.glsl.trim() === '') problems.push(`${kind}: empty glsl body`);
    if (typeof schema.title !== 'string' || schema.title === '') problems.push(`${kind}: empty title`);
    if (typeof schema.summary !== 'string' || schema.summary === '') problems.push(`${kind}: empty summary`);
    if (schema.outputs.length === 0) problems.push(`${kind}: no output ports`);
    if (schema.primaryOut !== undefined && !schema.outputs.some((p) => p.name === schema.primaryOut)) {
      problems.push(`${kind}: primaryOut ${schema.primaryOut} is not a declared output`);
    }
    // Inputs and outputs are separate namespaces: a pass-through node (transform, tile,
    // hsv, ...) legitimately exposes `uv`/`color` on both sides.
    for (const [side, ports] of [['input', schema.inputs], ['output', schema.outputs]] as const) {
      const seen = new Set<string>();
      for (const p of ports) {
        if (seen.has(p.name)) problems.push(`${kind}: duplicate ${side} port name ${p.name}`);
        seen.add(p.name);
      }
    }
  }
  assert.deepEqual(problems, []);
  assert.deepEqual(Object.keys(CATALOG).sort(), [...KINDS].sort(), 'CATALOG must cover exactly NODE_KINDS');
});

/**
 * THE multi-output contract, stated once, checked for every kind:
 *
 *   `portsOk = outs.length <= 4 && outs.every(p => TYPE_ARITY[p.type] === 1)`
 *
 * `emitBodies` reports `IR033_BAD_MULTI_OUTPUT` for anything else, so an edit that
 * violates it does not "work on my machine" — it fails this test loudly, at the schema,
 * instead of degrading the emitted shader (the fallback path aliases the whole `vec4`
 * into every port, which compiles into six different kinds of wrong).
 *
 * Why scalar-only: the ports are the components of ONE returned `vec4`, and a `vec2` port
 * would have to alias two of them. Nothing in the token rewriter can express that
 * identically in all six targets, so the emitter refuses rather than guessing.
 */
test('1b. every kind satisfies the multi-output contract (<= 4 scalar ports)', () => {
  const problems: string[] = [];
  const multi: string[] = [];
  for (const [kind, schema] of ENTRIES) {
    const outs = schema.outputs;
    if (outs.length > 4) problems.push(`${kind}: ${outs.length} output ports (max 4 — they are vec4 components)`);
    if (outs.length > 1) {
      multi.push(kind);
      for (const p of outs) {
        if (TYPE_ARITY[p.type] !== 1) {
          problems.push(`${kind}.${p.name}: ${p.type} output on a multi-output node (ports must be scalar)`);
        }
      }
    }
    // `primaryOut` is the port a bare `{"node": "id"}` wire resolves to, so it must exist.
    if (schema.primaryOut !== undefined && !outs.some((p) => p.name === schema.primaryOut)) {
      problems.push(`${kind}: primaryOut ${schema.primaryOut} is not a declared output`);
    }
  }
  assert.deepEqual(problems, []);
  // Guard the guard: the multi-output path is not dead code — these four use it, and the
  // two kinds that cannot (a pair of vec2 ports) are pinned by test 1d.
  assert.deepEqual(
    multi.sort(),
    ['polar', 'sdf', 'split_vec2', 'split_vec4', 'time', 'voronoi'],
    'the multi-output kinds changed; update the destructuring cases in test 11b',
  );
});

test('1c. every input carries a default so a freshly dropped node renders', () => {
  const problems: string[] = [];
  for (const [kind, schema] of ENTRIES) {
    for (const port of schema.inputs) {
      if (!port.default) {
        problems.push(`${kind}.${port.name}: no default (emitBodies would report IR031_MISSING_INPUT)`);
        continue;
      }
      const d: ParamValue = port.default;
      if (d.kind === 'float') {
        if (d.min === undefined || d.max === undefined || d.step === undefined) {
          problems.push(`${kind}.${port.name}: float default without min/max/step (the slider would be unusable)`);
          continue;
        }
        if (!(d.min < d.max)) problems.push(`${kind}.${port.name}: min ${d.min} >= max ${d.max}`);
        if (d.value < d.min || d.value > d.max) {
          problems.push(`${kind}.${port.name}: default ${d.value} outside [${d.min}, ${d.max}]`);
        }
        if (!(d.step > 0)) problems.push(`${kind}.${port.name}: step ${d.step} is not positive`);
      }
      const width: Partial<Record<ParamValue['kind'], number>> = { vec2: 2, vec3: 3, vec4: 4, color: 4 };
      const want = width[d.kind];
      if (want !== undefined) {
        const value = (d as { value: number[] }).value;
        if (!Array.isArray(value) || value.length !== want) {
          problems.push(`${kind}.${port.name}: ${d.kind} default needs ${want} components, got ${JSON.stringify(value)}`);
        }
      }
    }
  }
  assert.deepEqual(problems, []);
});

/**
 * `uv` and `tile` are the two kinds whose design calls for extra ports that this
 * contract cannot express, and the omission is a decision, not an oversight:
 *
 *   • `uv`   wants `uv: vec2` + `centered: vec2`
 *   • `tile` wants `uv: vec2` (in-cell) + `cell: vec2` (cell id)
 *
 * A two-`vec2` node cannot be emitted: the ports are `.x/.y/.z/.w` of one returned
 * `vec4` and must each be scalar (`TYPE_ARITY[port.type] === 1`), so exposing both halves
 * is impossible while either half stays a vec2 — and demoting one half to two scalars is
 * not enough on its own. This test is the tripwire: if the emitter ever learns to lower
 * composite ports, this fails and whoever made that change has to come back and add them
 * (the recipes live in each node's `summary` until then).
 */
test('1d. `uv` and `tile` stay single-output: a pair of vec2 ports cannot be emitted', () => {
  const unexpressible: Partial<Record<NodeKind, string>> = {
    uv: 'centered',
    tile: 'cell',
  };
  const problems: string[] = [];
  for (const [kind, missing] of Object.entries(unexpressible)) {
    const schema = CATALOG[kind as NodeKind];
    if (schema.outputs.length !== 1 || schema.outputs[0]!.type !== 'vec2') {
      problems.push(
        `${kind}: expected exactly one vec2 output, found ${schema.outputs.map((p) => `${p.name}:${p.type}`).join(', ')}`,
      );
    }
    // The limitation has to be *documented where a reader looks*, not just in this test:
    // the node's own summary names the missing port and the reason it cannot exist.
    if (!schema.summary.includes(`\`${missing}\``)) problems.push(`${kind}: summary no longer names the missing port ${missing}`);
    if (!/multi-output ports must be scalar/i.test(schema.summary)) {
      problems.push(`${kind}: summary no longer explains why \`${missing}\` is not a port`);
    }
  }
  assert.deepEqual(problems, [], 'if these grew a second port, the emitter contract must have changed — re-check test 1b');
});

test('2. every placeholder names a declared input, OCTAVE_LIMIT or node_id', () => {
  const problems: string[] = [];
  // `node_id` is the emitter's other built-in placeholder (expandTemplate resolves it to
  // the node's id): bodies use it to namespace their locals, which they must, because all
  // bodies share one flat scope. It is allowed here; anything else must be an input.
  const BUILT_IN = new Set(['OCTAVE_LIMIT', 'node_id']);
  for (const [kind, schema] of ENTRIES) {
    const declared = new Set(schema.inputs.map((p) => p.name));
    for (const m of schema.glsl.matchAll(/\$\{([^}]*)\}/g)) {
      const name = (m[1] ?? '').trim();
      if (BUILT_IN.has(name)) continue;
      if (!declared.has(name)) problems.push(`${kind}: unknown placeholder \${${name}}`);
    }
  }
  assert.deepEqual(problems, []);
});

test('2b. every local a body declares is namespaced with node_id', () => {
  // One flat scope for all bodies + the epilogue (see nodes.ts header): an un-namespaced
  // local makes two instances of the same kind redeclare it -> "redefinition of 'p'".
  const problems: string[] = [];
  for (const [kind, schema] of ENTRIES) {
    for (const m of schema.glsl.matchAll(/\b(?:float|vec2|vec3|vec4|mat2|mat3)\s+([A-Za-z_][A-Za-z0-9_]*)(\$\{node_id\})?\s*=/g)) {
      if (!m[2]) problems.push(`${kind}: local \`${m[1]}\` is not namespaced with \${node_id}`);
    }
  }
  assert.deepEqual(problems, []);
});

/* -------------------------------------------------------------------------- */
/* 2c-2f. The multi-output *body* side of the contract                         */
/* -------------------------------------------------------------------------- */

/** The kinds whose body must return a tuple, with the port order it must return. */
const MULTI_OUT: [NodeKind, string[]][] = KINDS.filter((k) => CATALOG[k].outputs.length > 1).map((k) => [
  k,
  CATALOG[k].outputs.map((p) => p.name),
]);
const COMPONENTS = 'xyzw';

test('2c. a multi-output body returns one 4-component value, in port order', () => {
  const problems: string[] = [];
  for (const [kind] of MULTI_OUT) {
    const schema = CATALOG[kind];
    const { resultExpr } = splitReturn(schema.glsl);
    /*
     * The emitter assigns the returned expression straight to `vec4 tmp_<id>`, so it has to be
     * four-component — there is no widening on this path.
     *
     * Two spellings satisfy that, and the second was added when `split_vec4` arrived:
     *
     *   • `vec4(...)`, the explicit construction every other multi-output kind uses;
     *   • a **passthrough of a vec4-typed input**, which is what `split_vec4` is: its whole
     *     job is to hand the same four components back out as four scalar ports, and
     *     `return vec4(${value}.x, ${value}.y, ${value}.z, ${value}.w)` would be the same
     *     program with a redundant constructor.
     *
     * The check is on the *type*, not the spelling, so it still catches the mistake it exists
     * for: a body that passes through a `vec2` or a scalar, or constructs a `vec3`, fails
     * either branch. `split_vec4`'s ports map to x/y/z/w of its input, which is the declared
     * order — so the "in port order" half of this test's name holds for it too.
     */
    const passthrough = /^\$\{(\w+)\}$/.exec(resultExpr.trim());
    if (/^vec4\s*\(/.test(resultExpr)) continue;
    if (passthrough) {
      const input = schema.inputs.find((p) => p.name === passthrough[1]);
      if (input?.type === 'vec4') continue;
      problems.push(
        `${kind}: multi-output body passes through \`${resultExpr}\`, whose input is ` +
          `${input ? input.type : 'not a declared input'} — the emitter assigns it to a vec4`,
      );
      continue;
    }
    problems.push(`${kind}: multi-output body returns \`${resultExpr}\` (must be vec4(...) so ports are .x/.y/.z/.w)`);
  }
  assert.deepEqual(problems, []);
  // Not just the 2-port cases: the mapping is exercised at three and four components too.
  // Order follows NODE_KINDS (the fixture's order), not the alphabet.
  assert.deepEqual(
    MULTI_OUT.map(([kind, ports]) => `${kind}:${ports.join('/')}`),
    [
      'time:time/sin/cos',
      'split_vec2:x/y',
      'split_vec4:x/y/z/w',
      'polar:angle/radius',
      'voronoi:f1/f2/edges/cell',
      'sdf:distance/mask',
    ],
  );
});

test('2d. no output port name is used as a bare identifier inside its own body', () => {
  // A port comes back as the local `L_<id>_<port>`, so a body may not name a local after
  // one of its own ports: `float sin_${node_id}` is fine, but a local called `sin` (or an
  // output name leaking in as `${sin}`) would collide with the port local once several
  // nodes are flattened into one scope. Inputs and outputs are separate namespaces, so
  // only the output side is checked here — 2b covers local declarations.
  const problems: string[] = [];
  for (const [kind, schema] of ENTRIES) {
    const body = withoutComments(schema.glsl);
    for (const p of schema.outputs) {
      if (schema.inputs.some((i) => i.name === p.name)) continue; // a passthrough port, by design
      if (body.includes(`\${${p.name}}`)) problems.push(`${kind}: \${${p.name}} is an output name, not an input`);
      const local = new RegExp(`\\b(?:float|vec2|vec3|vec4|int|bool)\\s+${p.name}\\s*=`);
      if (local.test(body)) problems.push(`${kind}: body declares a local named after its own output \`${p.name}\``);
    }
  }
  assert.deepEqual(problems, []);
});

test('2e. a non-first primaryOut is documented in the summary that a graph author reads', () => {
  // `sdf` declares `distance` first but keeps `mask` primary, which is the one port whose
  // ordering could surprise someone reading the port table. For single-output kinds the
  // port name is the whole story; for the multi-output ones the primary must be visible
  // from the node itself, not only from a table in docs/.
  const problems: string[] = [];
  let checked = 0;
  for (const [kind, schema] of ENTRIES) {
    if (schema.outputs.length < 2 || !schema.primaryOut) continue;
    if (schema.primaryOut === schema.outputs[0]!.name) continue;
    checked++;
    if (!schema.summary.includes(`\`${schema.primaryOut}\``)) {
      problems.push(`${kind}: primaryOut \`${schema.primaryOut}\` is not mentioned in the summary`);
    }
  }
  assert.deepEqual(problems, []);
  assert.equal(checked, 1, 'expected exactly one kind with a non-first primaryOut (sdf) — update this test if that changed');
});

test('2f. every declared input is referenced by the body or documented as inert', () => {
  // An input nobody reads is a port that does nothing when wired — allowed only when the
  // summary says so (currently just `shape.uv`), never silently.
  const problems: string[] = [];
  for (const [kind, schema] of ENTRIES) {
    for (const port of schema.inputs) {
      if (schema.glsl.includes(`\${${port.name}}`)) continue;
      // "documented as inert" is a documentation requirement, not a grammar one — a summary
      // may legitimately say the inputs "do not affect the result" or that one "does not".
      if (/do(?:es)? not affect the result|accepted for pipeline symmetry/i.test(schema.summary)) continue;
      problems.push(`${kind}.${port.name}: declared but never used in the body`);
    }
  }
  assert.deepEqual(problems, []);
});

/* -------------------------------------------------------------------------- */
/* 3. Single trailing return                                                   */
/* -------------------------------------------------------------------------- */

test('3. every body has exactly one return, as its last statement', () => {
  const problems: string[] = [];
  for (const [kind, schema] of ENTRIES) {
    const returns = codeTokens(schema.glsl).filter((t) => t.kind === 'ident' && t.text === 'return');
    if (returns.length !== 1) problems.push(`${kind}: ${returns.length} \`return\` tokens (must be exactly 1)`);

    const stripped = withoutComments(schema.glsl).trim();
    if (!/return\s+[^;]*;\s*$/.test(stripped)) problems.push(`${kind}: body does not end with \`return <expr>;\``);

    // splitReturn scans for the last occurrence of the word, so a `return` that only
    // lives in a trailing comment silently truncates the expression.
    const { resultExpr } = splitReturn(schema.glsl);
    if (/\breturn\b/.test(resultExpr)) problems.push(`${kind}: splitReturn picked up a return inside a comment`);
  }
  assert.deepEqual(problems, []);
});

/* -------------------------------------------------------------------------- */
/* 4. Restricted subset: no banned token, no lint/rewrite errors               */
/* -------------------------------------------------------------------------- */

const BANNED: [RegExp, string][] = [
  /*
   * The alias spellings are **derived from production**, not restated.
   *
   * `lintSubset` now reports these itself (`RW012_BUILTIN_SPELLING`), because the rule has to
   * hold for bodies that arrive at runtime — `custom_glsl` ended the assumption that a body is
   * a constant a test can enumerate. Keeping a second copy here would be two lists that must
   * agree, and the way that fails is the interesting one: someone adds an alias to the
   * rewriter, the production check learns it, this list does not, and the catalog quietly
   * ships a body the linter rejects.
   */
  ...Object.entries(BANNED_BUILTIN_SPELLINGS).map(
    ([builtin, alias]): [RegExp, string] => [
      new RegExp(`\\b${builtin}\\s*\\(`),
      `${builtin}(...) — the alias is ${alias}(`,
    ],
  ),
  [/\bdFdx\b|\bdFdy\b/, 'derivatives are unavailable in some WGSL contexts'],
  [/#/, 'preprocessor directive'],
  [/\bprecision\b/, 'precision qualifier'],
  [/\blayout\b/, 'layout qualifier'],
  [/\buniform\b/, 'uniform declaration'],
  [/\bvarying\b/, 'varying declaration'],
  [/\battribute\b/, 'attribute declaration'],
  [/\bsampler/, 'sampler'],
  [/\bgl_FragColor\b/, 'gl_FragColor'],
  [/\bgl_FragCoord\b/, 'gl_FragCoord'],
  [/\buint\b/, 'uint'],
  [/\bwhile\b/, 'while loop'],
  [/\bswitch\b/, 'switch'],
  [/\bstruct\b/, 'struct'],
  [/\bdiscard\b/, 'discard'],
  [/\bbreak\s*;/, 'raw break;'],
  [/[[\]]/, 'arrays / indexing'],
];

test('4a. no banned spelling appears in any body (comment-stripped)', () => {
  const problems: string[] = [];
  for (const [kind, schema] of ENTRIES) {
    const stripped = withoutComments(schema.glsl);
    for (const [re, why] of BANNED) {
      const m = re.exec(stripped);
      if (m) problems.push(`${kind}: ${why} (matched ${JSON.stringify(m[0])})`);
    }
  }
  assert.deepEqual(problems, []);
});

test('4c. the noise generators can tile, and say so through one honest input', () => {
  /*
   * A noise field is not periodic — `hash21(floor(p))` depends on the *absolute* cell — so two uv
   * squares side by side do not match and a texture made from one seams the moment an engine tiles
   * it. "Make me a seamless tileable texture" is one of the most common things a person asks a
   * shader tool for, and this catalog could not do it at all until the `tile` input existed.
   *
   * The flag lives on every generator that samples the shared helpers, because periodicity is a
   * property of the *lattice*, not of the document: a warped, tiled noise fed through a
   * non-tiling voronoi still seams. Three ways to get it wrong, all checked here:
   *
   *   - a generator that samples the helpers without offering the input (it can never tile);
   *   - a body that calls the non-`t` helper while exposing the flag (the input would do nothing,
   *     which is worse than not having it);
   *   - a period that is not derived from `scale`, which is the only quantity that knows how many
   *     cells fit in one uv unit.
   */
  const samplers = ENTRIES.filter(([, s]) => /\b(noise2|gnoise2|fbm2|voronoi2)t?\(/.test(withoutComments(s.glsl)));
  assert.ok(samplers.length >= 4, `expected the four noise generators, found ${samplers.length}`);

  for (const [kind, schema] of samplers) {
    const body = withoutComments(schema.glsl);
    const hasTile = schema.inputs.some((p) => p.name === 'tile');
    assert.ok(hasTile, `${kind}: samples the noise helpers but cannot be made tileable`);
    // The tiled helper is the only one that wraps the lattice, so the body must call it.
    assert.match(
      body,
      /\b(noise2t|gnoise2t|fbm2t|voronoi2t)\(/,
      `${kind}: exposes \`tile\` but calls the non-periodic helper, so the input does nothing`,
    );
    // The wrap period must be derived from the node's own frequency/scale input — the quantity
    // that says how many cells fit in one uv unit. `raymarch` calls it `frequency`, which is the
    // same thing under a different name; what is checked is that it is *the node's* scale input
    // and not a copied constant.
    const scaleInput = schema.inputs.some((p) => p.name === 'scale' || p.name === 'frequency');
    assert.ok(scaleInput, `${kind}: no scale/frequency input to derive the period from`);
    assert.ok(
      /vec2\(\\?\$\{(scale|frequency)\}, \\?\$\{(scale|frequency)\}\)/.test(schema.glsl),
      `${kind}: the wrap period must come from the node's scale input — it is what ties cells to uv`,
    );
    // The default is off, so no existing document changes behaviour when this lands.
    const tile = schema.inputs.find((p) => p.name === 'tile');
    assert.equal(tile?.default?.kind, 'float');
    assert.equal((tile?.default as { value: number }).value, 0, `${kind}: tile must default to 0`);
  }

  // And the helpers themselves exist in the library, with the wrap implemented once.
  assert.match(GLSL_LIBRARY, /vec2 wrap2\(vec2 i, vec2 period, float on\)/);
  for (const fn of ['noise2t', 'gnoise2t', 'fbm2t', 'voronoi2t']) {
    assert.ok(GLSL_LIBRARY.includes(`float ${fn}(`) || GLSL_LIBRARY.includes(`vec3 ${fn}(`), `${fn} missing`);
  }
  // The period has to scale with the octave loop, or only the first octave tiles.
  assert.match(
    GLSL_LIBRARY,
    /per = per \* lacunarity;/,
    'fbm2t must scale the wrap period per octave — otherwise the seam is in the higher octaves',
  );
});

test('4b. lintSubset and rewriteSource report zero errors for every body and target', () => {
  const problems: string[] = [];
  for (const [kind, schema] of ENTRIES) {
    for (const d of lintSubset(schema.glsl, `node ${kind}`)) {
      if (d.severity === 'error') problems.push(`${kind}: lint ${d.code} ${d.message}`);
    }
    for (const target of TARGETS) {
      const res = rewriteSource(schema.glsl, target, `node ${kind} (${kind})`);
      for (const d of res.diagnostics) {
        if (d.severity === 'error') problems.push(`${kind}/${target}: ${d.code} ${d.message}`);
      }
      // The rewriter is total: an unknown token passes through, so a body that survives
      // must still contain its own identifiers.
      if (!res.source.includes('return')) problems.push(`${kind}/${target}: rewrite lost the trailing return`);
    }
  }
  assert.deepEqual(problems, []);
});

test('4c. tokenize round-trips every body exactly', () => {
  for (const [kind, schema] of ENTRIES) {
    assert.equal(tokenize(schema.glsl).map((t) => t.text).join(''), schema.glsl, `${kind}: tokenize is lossy`);
  }
});

/* -------------------------------------------------------------------------- */
/* 5. Integer literals where a float is required                               */
/* -------------------------------------------------------------------------- */

/**
 * Two complementary checks, because "which integers are legitimate" is context
 * dependent and a naive `\b\d+\b` scan is all noise:
 *
 *   A. the general rule — every integer literal must sit in an explicitly allowed
 *      position: `int <name> = <n>`, `int(<n>)`, or a `for` header bound `<name> < <n> ;`
 *      followed by the same loop variable. Everything else is a float context.
 *   B. the specific failure mode GLSL ES 3.00 punishes — an integer literal directly
 *      mixed with a float literal through an operator (`2 * 0.5`).
 */
function integerLiteralProblems(src: string): string[] {
  const toks = tokenize(src).filter((t) => t.kind !== 'comment');
  /** The n-th non-space token before/after `idx` (n >= 1). */
  const step = (idx: number, dir: -1 | 1, n: number): Token | undefined => {
    let j = idx;
    for (let k = 0; k < n; k++) {
      j += dir;
      while (j >= 0 && j < toks.length && toks[j]!.kind === 'space') j += dir;
    }
    return toks[j];
  };
  const at = (i: number, dir: -1 | 1): Token | undefined => step(i, dir, 1);
  const problems: string[] = [];
  for (let i = 0; i < toks.length; i++) {
    const tok = toks[i]!;
    if (tok.kind !== 'number') continue;
    if (/[.eEfFuU]/.test(tok.text)) continue; // float literal (or a suffixed one)
    if (/^0[xX]/.test(tok.text)) {
      problems.push(`${tok.text}: hex literal`);
      continue;
    }
    const p1 = at(i, -1);
    const p2 = step(i, -1, 2);
    const p3 = step(i, -1, 3);
    const isIntInit = p1?.text === '=' && p2?.kind === 'ident' && p3?.text === 'int';
    const isIntCast = p1?.text === '(' && p2?.text === 'int';
    const n1 = at(i, 1);
    const n2 = step(i, 1, 2);
    const isLoopBound =
      p1?.text === '<' && p2?.kind === 'ident' && n1?.text === ';' && n2?.kind === 'ident' && n2.text === p2.text;
    if (isIntInit || isIntCast || isLoopBound) continue;
    problems.push(`${tok.text} @${tok.pos}: integer literal in a float context`);
  }
  return problems;
}

const MIXED_ARITHMETIC =
  /(?<![\w.])\d+(?![\d.])\s*(?:[-+*/%]|==|!=|<=|>=)\s*(?:\d+\.\d*|\.\d+)|(?:\d+\.\d*|\.\d+)\s*(?:[-+*/%]|==|!=|<=|>=)\s*(?<![\w.])\d+(?![\d.])/;

test('5. float literals always carry a decimal point (no implicit int -> float)', () => {
  const problems: string[] = [];
  for (const [kind, schema] of ENTRIES) {
    for (const p of integerLiteralProblems(schema.glsl)) problems.push(`${kind}: ${p}`);
    const mixed = MIXED_ARITHMETIC.exec(withoutComments(schema.glsl));
    if (mixed) problems.push(`${kind}: integer literal mixed with a float literal: ${JSON.stringify(mixed[0])}`);
  }
  assert.deepEqual(problems, []);
});

/* -------------------------------------------------------------------------- */
/* 6. Every called function exists (and takes the right number of arguments)    */
/* -------------------------------------------------------------------------- */

/** `float hash21(vec2 p) {` -> hash21 => 1, straight out of the shared library. */
const LIBRARY_ARITY = new Map<string, number>();
for (const m of GLSL_LIBRARY.matchAll(/^[ \t]*(?:float|vec[234]|mat[23])[ \t]+(\w+)[ \t]*\(([^)]*)\)/gm)) {
  const params = (m[2] ?? '').trim();
  LIBRARY_ARITY.set(m[1]!, params === '' ? 0 : params.split(',').length);
}

const TYPE_CONSTRUCTORS = new Set(['vec2', 'vec3', 'vec4', 'mat2', 'mat3']);
const KEYWORDS = new Set(['if', 'else', 'for', 'return', 'while', 'do', 'switch']);

/** Arity of the portable/aliased surface actually used by the catalog. */
const PORTABLE_ARITY: Record<string, number> = {
  abs: 1, sign: 1, cos: 1, sin: 1, tan: 1, floor: 1, ceil: 1, sqrt: 1, exp: 1, exp2: 1,
  log: 1, log2: 1, normalize: 1, length: 1, float: 1, int: 1, bool: 1, radians: 1, degrees: 1,
  inversesqrt_: 1, fract_: 1, tex2D_: 1,
  pow: 2, min: 2, max: 2, mod_: 2, atan2_: 2, dot: 2, distance: 2, step: 2, cross: 2, reflect: 2,
  clamp: 3, smoothstep: 3, mix_: 3, faceforward: 3, refract: 3,
};

function callsIn(src: string): { name: string; argc: number; pos: number }[] {
  const toks = codeTokens(src);
  const out: { name: string; argc: number; pos: number }[] = [];
  for (let i = 0; i < toks.length; i++) {
    const tok = toks[i]!;
    if (tok.kind !== 'ident') continue;
    if (toks[i + 1]?.text !== '(') continue;
    const prev = toks[i - 1];
    if (prev?.kind === 'punct' && prev.text === '.') continue; // member call, not a helper
    const empty = toks[i + 2]?.text === ')';
    let depth = 1;
    let commas = 0;
    for (let j = i + 2; j < toks.length && depth > 0; j++) {
      const tk = toks[j]!;
      if (tk.text === '(') depth++;
      else if (tk.text === ')') depth--;
      else if (tk.text === ',' && depth === 1) commas++;
    }
    out.push({ name: tok.text, argc: empty ? 0 : commas + 1, pos: tok.pos });
  }
  return out;
}

test('6. every function called by a body exists, with the declared arity', () => {
  const known = new Set<string>([...PORTABLE_FUNCTIONS, ...REWRITE_ALIASES, ...LIBRARY_ARITY.keys()]);
  const problems: string[] = [];
  for (const [kind, schema] of ENTRIES) {
    for (const call of callsIn(schema.glsl)) {
      if (TYPE_CONSTRUCTORS.has(call.name) || KEYWORDS.has(call.name)) continue;
      if (!known.has(call.name)) {
        problems.push(`${kind}: \`${call.name}(...)\` is not in PORTABLE_FUNCTIONS, REWRITE_ALIASES or GLSL_LIBRARY`);
        continue;
      }
      const want = LIBRARY_ARITY.get(call.name) ?? PORTABLE_ARITY[call.name];
      if (want !== undefined && want !== call.argc) {
        problems.push(`${kind}: ${call.name}() called with ${call.argc} args, declared with ${want}`);
      }
    }
  }
  assert.deepEqual(problems, []);
  // Guard the guard: the library really did declare the helpers the catalog leans on.
  for (const helper of ['fbm2', 'voronoi2', 'noise2', 'gnoise2', 'rot2', 'mat2_mul_', 'rgb2hsv', 'hsv2rgb', 'luma']) {
    assert.ok(LIBRARY_ARITY.has(helper), `GLSL_LIBRARY is expected to declare ${helper}`);
  }
  assert.equal(LIBRARY_ARITY.get('fbm2'), 5, 'fbm2 is (vec2 p, int octaves, float lacunarity, float gain, float ridged)');
});

/* -------------------------------------------------------------------------- */
/* Quality scaling / categories                                                */
/* -------------------------------------------------------------------------- */

test('7. only the quality-scaled nodes loop, and they use the mandated loop shape', () => {
  const problems: string[] = [];
  const looping: string[] = [];
  const derivativeUsers: string[] = [];
  const scaled = ENTRIES.filter(([, s]) => s.needs?.qualityScaled).map(([k]) => k);
  for (const [kind, schema] of ENTRIES) {
    const hasLoop = /\bfor\s*\(/.test(withoutComments(schema.glsl));
    if (hasLoop) {
      looping.push(kind);
      if (!schema.needs?.qualityScaled) problems.push(`${kind}: uses for(...) without needs.qualityScaled`);
      if (!schema.glsl.includes('${OCTAVE_LIMIT}')) problems.push(`${kind}: loop bound is not \${OCTAVE_LIMIT}`);
      if (!/for \(int i = 0; i < \$\{OCTAVE_LIMIT\}; i\+\+\)/.test(schema.glsl)) {
        problems.push(`${kind}: loop is not the mandated for (int i = 0; i < \${OCTAVE_LIMIT}; i++) shape`);
      }
    }
    if (schema.needs?.derivatives !== undefined || /\b(fwidth|dFdx|dFdy)\s*\(/.test(withoutComments(schema.glsl))) {
      // The flag and the call have to agree, in both directions: a body that reads a derivative
      // without declaring it would emit a shader whose host requirement is unstated, and a
      // declared flag with no call is a claim about a capability the node does not use.
      const calls = /\b(fwidth|dFdx|dFdy)\s*\(/.test(withoutComments(schema.glsl));
      const declares = Boolean(schema.needs?.derivatives);
      if (calls !== declares) {
        problems.push(
          `${kind}: needs.derivatives=${declares} but the body ${calls ? 'calls' : 'never calls'} a derivative`,
        );
      }
      // And a derivative is only legal in the subset through `fwidth`: `dFdx`/`dFdy` are not in
      // PORTABLE_FUNCTIONS, so a body that used one would already have failed test 6.
      if (calls) derivativeUsers.push(kind);
    }
  }
  assert.deepEqual(problems, []);
  // `fbm` is quality-scaled through the shared fbm2 loop rather than a loop of its own.
  assert.deepEqual(scaled.sort(), ['fbm', 'raymarch'], 'quality-scaled nodes are fbm and raymarch');
  assert.deepEqual(looping.sort(), ['raymarch'], 'only raymarch carries its own loop in the body');
  // `antialias` is the only derivative reader, and it is what the flag was reserved for: without
  // it `needs.derivatives` was a capability nothing could ask for.
  assert.deepEqual(derivativeUsers, ['antialias'], 'antialias is the only derivative user');
});

test('8. CATEGORIES partitions NODE_KINDS exactly once and matches every schema.category', () => {
  const listed = CATEGORIES.flatMap((c) => c.kinds);
  assert.deepEqual([...listed].sort(), [...KINDS].sort(), 'palette sections must cover every kind exactly once');
  assert.equal(new Set(listed).size, listed.length, 'a kind is listed in two sections');
  const ids = new Set(CATEGORIES.map((c) => c.id));
  for (const [kind, schema] of ENTRIES) {
    assert.ok(ids.has(schema.category), `${kind}: category ${schema.category} has no palette section`);
    assert.ok(
      CATEGORIES.find((c) => c.id === schema.category)!.kinds.includes(kind),
      `${kind}: not listed in its own category section`,
    );
  }
  for (const c of CATEGORIES) {
    assert.ok(c.title.length > 0 && c.summary.length > 0, `category ${c.id} needs a title and a summary`);
  }
});

test('9. needs flags are honest: matrix only where a mat2 helper is called', () => {
  for (const [kind, schema] of ENTRIES) {
    const usesMatrix = /\b(rot2|scale2|mat2_mul_|mat2_rot_|mat2)\b/.test(withoutComments(schema.glsl));
    assert.equal(Boolean(schema.needs?.matrix), usesMatrix, `${kind}: needs.matrix should be ${usesMatrix}`);
  }
});

test('9c. the vector ops whose meaning is 3D read .xyz and leave w alone', () => {
  /*
   * A vec4 in this IR carries its *kind* in `w`: 0 for a direction, 1 for a point (`v4()` and
   * `color()` differ in exactly that). `dot` has always honoured the convention —
   * `return dot(${a}.xyz, ${b}.xyz)` — but `vec_math`'s `normalize` did not: it normalised all
   * four components. The two disagree in a way that is invisible until it matters:
   *
   *   make_vec4(0.6, 0.8)        -> (0.6, 0.8, 0, 1)      // w defaults to 1, a *point*
   *   normalize(that)            -> (0.51, 0.68, 0, 0.85) // four components, length 1
   *   normalize(that.xyz)        -> (0.6, 0.8, 0)         // three components, length 1
   *
   * So the same "direction" normalised through `vec_math` and through `dot`/`reflect`/`length`
   * gave different vectors, and every term downstream of it was subtly wrong — with no
   * diagnostic, because both are valid GLSL. The fix is to make the odd one out agree.
   */
  const body = withoutComments(ENTRIES.find(([k]) => k === 'vec_math')![1].glsl);
  for (const raw of ['normalize(${a})', 'length(${a})', 'distance(${a},', 'cross(${a},']) {
    assert.ok(!body.includes(raw), `vec_math still uses the 4-component form \`${raw}\``);
  }
  // And the positive form, so a body edited to use something else entirely fails loudly.
  assert.ok(body.includes('normalize(${a}.xyz)'), 'normalize must read .xyz');
  assert.ok(body.includes('length(${a}.xyz)'), 'length must read .xyz');
  assert.ok(body.includes('distance(${a}.xyz'), 'distance must read .xyz');
  assert.ok(body.includes('cross(${a}.xyz'), 'cross must read .xyz');

  // The declaration must not do work: it was `normalize(${a})`, which ran a full 4-component
  // normalize on every call regardless of `op` and threw the result away.
  const firstAssign = /vec4\s+r_\$\{node_id\}\s*=\s*([^;]+);/.exec(body)?.[1] ?? '';
  assert.ok(
    !/normalize|length|distance/.test(firstAssign),
    `the declaration does real work (\`${firstAssign}\`); every branch assigns anyway`,
  );

  // The precedent this follows, asserted so the two cannot drift apart.
  const dotBody = withoutComments(ENTRIES.find(([k]) => k === 'dot')![1].glsl);
  assert.ok(dotBody.includes('.xyz'), '`dot` must keep ignoring w');
});

test('9b. needs.channel is declared by exactly the kinds that sample the host channel', () => {
  /*
   * The flag exists because a host has to *bind* something in the channel slot, and because
   * `render.feedback` is only meaningful when something reads it. Two ways to get it wrong, and
   * this test names both:
   *
   *   - a node samples `tex2D_` without declaring the flag → the host never learns it has to bind
   *     anything, and `render.feedback` on such a document is silently ignored;
   *   - a node declares the flag without sampling → every document containing it looks like it
   *     needs a channel, and the preview stops offering the honest "white stand-in" answer.
   *
   * Derived from the bodies rather than listed, so adding a second channel-reading kind is a
   * one-line catalog edit that this test then checks instead of a list somebody must remember.
   */
  for (const [kind, schema] of ENTRIES) {
    const samples = /\btex2D_\s*\(/.test(withoutComments(schema.glsl));
    assert.equal(
      Boolean(schema.needs?.channel),
      samples,
      `${kind}: needs.channel should be ${samples} (body samples the channel: ${samples})`,
    );
  }
});

/* -------------------------------------------------------------------------- */
/* Integration: one node of every kind, wired into one graph, emitted          */
/* -------------------------------------------------------------------------- */

/** node id -> input port -> source output port. Every kind is referenced at least once. */
const WIRES: Record<string, Record<string, WireRef>> = {
  // Non-primary ports are wired on purpose wherever a kind has them (`.cos`, `.radius`,
  // `.edges`, `.distance`): test 11b proves each one in isolation, this graph proves all
  // four in one emission, which is where a name collision between two nodes' port locals
  // would show up.
  n_gradient: { uv: { node: 'n_uv', port: 'uv' }, scale: { node: 'n_time', port: 'time' } },
  n_texture: { uv: { node: 'n_uv', port: 'uv' }, mix_amount: { node: 'n_checker', port: 'value' } },
  n_transform: { uv: { node: 'n_uv', port: 'uv' } },
  n_tile: { uv: { node: 'n_transform', port: 'uv' }, count: { node: 'n_resolution', port: 'resolution' } },
  n_kaleidoscope: { uv: { node: 'n_tile', port: 'uv' } },
  n_polar: { uv: { node: 'n_kaleidoscope', port: 'uv' } },
  n_domain_warp: { uv: { node: 'n_polar', port: 'radius' }, time: { node: 'n_time', port: 'cos' } },
  n_noise: { uv: { node: 'n_domain_warp', port: 'uv' } },
  n_fbm: { scale: { node: 'n_noise', port: 'value' } },
  n_voronoi: { scale: { node: 'n_fbm', port: 'value' } },
  n_stripe: { uv: { node: 'n_make_vec2', port: 'value' }, angle: { node: 'n_voronoi', port: 'edges' } },
  // NOTE: `checker.count` is fed from a vec2 producer, not from a float one. Wiring a
  // float into a vec2 port currently emits `vec2(x, x, x)` (three components) because of
  // a bug in the frozen emitter's `coerce()`; reported, not worked around in the catalog.
  n_checker: { count: { node: 'n_resolution', port: 'resolution' } },
  n_sdf: { shape: { node: 'n_checker', port: 'value' }, smoothness: { node: 'n_stripe', port: 'value' } },
  n_raymarch: { height: { node: 'n_sdf', port: 'distance' }, time: { node: 'n_time', port: 'time' } },
  n_math: { a: { node: 'n_raymarch', port: 'shade' }, b: { node: 'n_const_float', port: 'value' } },
  n_remap: { value: { node: 'n_math', port: 'value' } },
  n_smoothstep: {
    value: { node: 'n_remap', port: 'value' },
    edge1: { node: 'n_gradient', port: 'value' },
    edge0: { node: 'n_dot', port: 'value' },
  },
  // `antialias` sits in the chain rather than beside it, because a node whose output reaches
  // nothing is dead (IR011) — and because that is how it is actually used: softening a hard
  // threshold on the way to the colour stage.
  n_antialias: { value: { node: 'n_smoothstep', port: 'value' } },
  // `dot`'s output has to be consumed by something on the path to the output node, or the
  // whole surface chain is dead and never emitted — which is the failure this wiring exists
  // to prevent, and it caught exactly that when it was first written.
  // The surface group, wired so all five are reachable. They have no spatial *use* here — the
  // point is that a graph reading them emits a surface entry point, which is a different code
  // path in every target, and `dot` is the one node that consumes two of them at once.
  //
  // `n_dot`'s output goes to `n_smoothstep.edge0` rather than dangling: a node whose inputs
  // are wired but whose *output* reaches nothing is still dead, and the whole point of this
  // wiring is to make the surface path reach the output node.
  n_dot: { a: { node: 'n_normal', port: 'normal' }, b: { node: 'n_light_dir', port: 'direction' } },
  // The vector construction/decomposition group, wired as one chain that reaches the output:
  //
  //   output <- color_ramp <- grade <- hsv <- vec_op <- vec_math <- make_vec4 <- split_vec4 <- mouse
  //   output <- … <- sdf <- stripe <- make_vec2 <- split_vec4
  //
  // `split_vec4` reading `mouse` is the whole reason it exists: `mouse` is a bare vec4 and the
  // connection rule only widens, so before these nodes it could not reach any spatial input.
  // `vec_math` replaces `n_const_color` as `n_vec_op.b`; the constant stays reachable through
  // `n_blend_2`, so nothing is orphaned by the swap.
  n_split_vec4: { value: { node: 'n_mouse', port: 'mouse' } },
  n_split_vec2: { value: { node: 'n_resolution', port: 'resolution' } },
  n_make_vec2: { x: { node: 'n_split_vec4', port: 'x' }, y: { node: 'n_split_vec2', port: 'y' } },
  n_make_vec4: {
    x: { node: 'n_split_vec4', port: 'z' },
    y: { node: 'n_split_vec4', port: 'w' },
    z: { node: 'n_const_float', port: 'value' },
    w: { node: 'n_dot', port: 'value' },
  },
  n_vec_math: { a: { node: 'n_make_vec4', port: 'value' }, b: { node: 'n_view_dir', port: 'direction' } },
  // The escape hatch sits between them, so the fixture proves a *per-instance* body is lowered
  // by the same rewriter as a catalog body: `vec_math -> custom_glsl -> vec_op`. It takes
  // `n_vec_op.b`'s slot, and `vec_math` moves to its own input, keeping the chain intact.
  n_custom_glsl: { a: { node: 'n_vec_math', port: 'value' }, b: { node: 'n_make_vec4', port: 'value' } },
  n_vec_op: { a: { node: 'n_shape', port: 'value' }, b: { node: 'n_custom_glsl', port: 'value' } },
  n_mix: {
    a: { node: 'n_world_pos', port: 'position' },
    b: { node: 'n_view_dir', port: 'direction' },
    t: { node: 'n_antialias', port: 'value' },
  },
  n_blend: { a: { node: 'n_mix', port: 'value' }, b: { node: 'n_mouse', port: 'mouse' } },
  n_shape: { background: { node: 'n_blend', port: 'value' } },
  n_hsv: { color: { node: 'n_vec_op', port: 'value' } },
  n_grade: { color: { node: 'n_hsv', port: 'color' } },
  n_color_ramp: { colorA: { node: 'n_grade', port: 'color' }, colorB: { node: 'n_texture', port: 'color' } },
  n_output: { color: { node: 'n_color_ramp', port: 'color' } },
};

/**
 * Second instances of the kinds that declare locals. They exist to prove the
 * `${node_id}` namespacing: with un-namespaced locals this graph emits `vec2 p`
 * twice and no GLSL compiler would accept it.
 */
const EXTRA_NODES: ShaderNode[] = ['noise', 'fbm', 'math', 'kaleidoscope', 'sdf', 'raymarch', 'blend', 'grade'].map(
  (kind) => ({ id: `n_${kind}_2`, kind: kind as NodeKind }),
);

const EXTRA_WIRES: Record<string, Record<string, WireRef>> = {
  n_noise_2: { uv: { node: 'n_domain_warp', port: 'uv' } },
  n_fbm_2: { scale: { node: 'n_noise_2', port: 'value' } },
  n_math_2: { a: { node: 'n_fbm_2', port: 'value' }, b: { node: 'n_raymarch_2', port: 'shade' } },
  n_kaleidoscope_2: { uv: { node: 'n_kaleidoscope', port: 'uv' } },
  n_sdf_2: {},
  n_raymarch_2: { height: { node: 'n_sdf_2', port: 'mask' }, time: { node: 'n_time', port: 'time' } },
  n_blend_2: { a: { node: 'n_blend', port: 'value' }, b: { node: 'n_const_color', port: 'color' } },
  n_grade_2: { color: { node: 'n_grade', port: 'color' } },
};

/** Extra consumers, so every second instance is reachable from the output node. */
const EXTRA_SINKS: Record<string, Record<string, WireRef>> = {
  n_stripe: { width: { node: 'n_noise_2', port: 'value' } },
  n_voronoi: { randomness: { node: 'n_fbm_2', port: 'value' } },
  n_remap: { inMax: { node: 'n_math_2', port: 'value' } },
  n_polar: { center: { node: 'n_kaleidoscope_2', port: 'uv' } },
  n_raymarch: { frequency: { node: 'n_sdf_2', port: 'distance' } },
  n_shape: { color: { node: 'n_blend_2', port: 'value' } },
  n_color_ramp: { colorC: { node: 'n_grade_2', port: 'color' } },
};

const ALL_WIRES: Record<string, Record<string, WireRef>> = (() => {
  const merged: Record<string, Record<string, WireRef>> = {};
  for (const [id, ports] of Object.entries(WIRES)) merged[id] = { ...ports };
  for (const [id, ports] of Object.entries(EXTRA_WIRES)) merged[id] = { ...(merged[id] ?? {}), ...ports };
  for (const [id, ports] of Object.entries(EXTRA_SINKS)) merged[id] = { ...(merged[id] ?? {}), ...ports };
  return merged;
})();

function buildDoc(): ShaderDoc {
  const nodes: ShaderNode[] = [
    ...KINDS.map((kind) => {
      const wiring = ALL_WIRES[`n_${kind}`];
      const base: ShaderNode = wiring ? { id: `n_${kind}`, kind, inputs: wiring } : { id: `n_${kind}`, kind };
      // `custom_glsl` carries its body on the node, and a node without one is `IR034` by
      // design. The snippet is deliberately written in the restricted subset — `mix_`, not
      // `mix` — because this fixture is also the proof that a per-instance body is lowered by
      // the same rewriter as a catalog body.
      return kind === 'custom_glsl'
        ? { ...base, body: 'return mix_(${a}, ${b}, clamp(${c}.x + ${d}.y, 0.0, 1.0));' }
        : base;
    }),
    ...EXTRA_NODES.map((n) => {
      const wiring = ALL_WIRES[n.id];
      return wiring && Object.keys(wiring).length ? { ...n, inputs: wiring } : { ...n };
    }),
  ];
  const edges: Edge[] = [];
  for (const [to, ports] of Object.entries(ALL_WIRES)) {
    for (const [port, ref] of Object.entries(ports)) {
      edges.push({ from: { node: ref.node, port: ref.port ?? '' }, to: { node: to, port } });
    }
  }
  return {
    irVersion: IR_VERSION,
    id: 'catalog-conformance',
    title: 'Catalog conformance graph',
    spec: { intent: 'exercise every node kind once', category: 'test', tags: ['conformance'], features: [] },
    stage: 'fragment',
    graph: { nodes, edges },
    render: { resolution: [512, 512], qualityTier: 'medium', maxIterations: 8, transparent: false },
    targets: ['glsl'],
  };
}

test('10. a graph containing every kind emits with zero error diagnostics', () => {
  const previous = TARGET_EMITTERS.glsl;
  // Throwaway target: the rewritten shared library plus the emitted bodies, wrapped in
  // the minimum epilogue the bodies expect. `targets/` is NOT imported here — the real
  // emitters are another module's business (AGENT_CONTRACT.md §2).
  registerTarget('glsl', ({ doc, bodies, rw }) => {
    // NOTE: `OCTAVE_LIMIT` inside GLSL_LIBRARY is *not* expanded by the emitter
    // (expandTemplate only ever sees node bodies), so every target must substitute the
    // tier budget itself before rewriting the library.
    const library = rw(GLSL_LIBRARY.replace(/\bOCTAVE_LIMIT\b/g, String(octavesFor(doc))), 'library');
    const block = rw(bodies.lines.join('\n'), 'bodies');
    const source = [
      '#version 300 es',
      'precision highp float;',
      'uniform vec2 iResolution;',
      'uniform float iTime;',
      'uniform vec4 iMouse;',
      'uniform sampler2D iChannel0;',
      // The surface contract, mirrored from the real emitter: this graph reads `normal`, so a
      // fixture that omitted these would be asserting against source that could never compile.
      'uniform vec3 iCameraPos;',
      'uniform vec3 iLightDir;',
      'in vec3 vWorldPos;',
      'in vec3 vNormal;',
      'in vec2 vUv;',
      'out vec4 fragColor;',
      library.source,
      'void main() {',
      '  vec2 uv = vUv;',
      '  vec2 fragCoord = uv * iResolution.xy;',
      '  vec3 iWorldPos = vWorldPos;',
      '  vec3 iNormal = normalize(vNormal);',
      '  vec3 iViewDir = normalize(iCameraPos - vWorldPos);',
      block.source,
      `  fragColor = ${bodies.localName('n_output', 'color')};`,
      '}',
    ].join('\n');
    return { source };
  });

  try {
    const doc = buildDoc();
    const emission = emit(doc, { catalog: CATALOG, target: 'glsl' });
    const errors = emission.diagnostics.filter((d) => d.severity === 'error');
    assert.deepEqual(
      errors.map((d) => `${d.code} ${d.message}`),
      [],
    );

    // The only warnings allowed are lintSubset's loop-shape false positive: it tests a
    // regex against tokens joined with spaces, so `i ++` can never match `\w\+\+`.
    const unexpected = emission.diagnostics.filter((d) => d.severity !== 'error' && !d.code.startsWith('RW010'));
    assert.deepEqual(unexpected.map((d) => `${d.code} ${d.message}`), []);

    for (const kind of KINDS) {
      assert.ok(emission.source.includes(`L_n_${kind}_`), `no emitted local for ${kind}`);
    }
    for (const extra of EXTRA_NODES) {
      assert.ok(emission.source.includes(`L_${extra.id}_`), `no emitted local for the second ${extra.kind}`);
    }
    assert.ok(emission.source.includes('L_n_output_color'), 'the output node must expose a local for the target');
    assert.ok(!emission.source.includes('unresolved:'), 'a placeholder was not expanded');
    // Comments are stripped first: the library's own prose may legitimately contain the word
    // ("`mod_` by zero is undefined"), and an assertion that fails on documentation is an
    // assertion somebody will weaken instead of fixing. In *code* the word can only come from an
    // unexpanded placeholder, which is what this is looking for.
    assert.ok(
      !/\bundefined\b/.test(withoutComments(emission.source)),
      'the emitter wrote a literal `undefined`',
    );

    // The multi-output path is used by exactly the kinds that declare 2+ ports, and by no
    // one else: a `tmp_<id>` for a single-output kind would mean the schema grew a port
    // without this test being updated, and a *missing* one would mean the emitter fell
    // back to the IR033 branch (which is an error and would already have failed above).
    for (const [kind, schema] of ENTRIES) {
      const tmp = `tmp_n_${kind}`;
      if (schema.outputs.length > 1) {
        assert.ok(
          emission.source.includes(`vec4 ${tmp} = `),
          `${kind} declares ${schema.outputs.length} outputs but was not destructured`,
        );
        for (const [idx, port] of schema.outputs.entries()) {
          const want = `L_n_${kind}_${port.name} = ${tmp}.${COMPONENTS[idx]};`;
          assert.ok(emission.source.includes(want), `${kind}.${port.name} should be emitted as \`${want}\``);
        }
      } else {
        assert.ok(!emission.source.includes(tmp), `${kind} is single-output but took the destructuring path`);
      }
    }

    // One flat scope: no variable may be declared twice anywhere in the emitted bodies.
    const body = block0(emission.source);
    assert.ok(body.length > 1000, `emitted body block looks empty (${body.length} chars)`);
    const declared = new Map<string, number>();
    for (const m of body.matchAll(/(?:^|\s)(?:float|vec2|vec3|vec4|mat2|mat3)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=/g)) {
      const name = m[1]!;
      declared.set(name, (declared.get(name) ?? 0) + 1);
    }
    const duplicates = [...declared].filter(([, n]) => n > 1).map(([name, n]) => `${name} x${n}`);
    assert.deepEqual(duplicates, [], 'a local is declared more than once in the shared scope');
    assert.ok(declared.size > 30, `expected every local-heavy body to contribute declarations, saw ${declared.size}`);

    // Both quality-scaled bodies and the shared fbm2 must have a literal loop bound.
    const loops = emission.source.match(/for \(int i = 0; i < \d+; i\+\+\)/g) ?? [];
    assert.ok(loops.length >= 2, `expected >= 2 expanded quality-scaled loops, saw ${loops.length}`);

    // The emitted block must also satisfy the integer-literal rule once expanded.
    for (const problem of integerLiteralProblems(body)) {
      assert.fail(`emitted body: ${problem}`);
    }
  } finally {
    if (previous) registerTarget('glsl', previous);
    else delete TARGET_EMITTERS.glsl;
  }
});

/** The expanded node bodies, sliced out of the stub's source (between main() and fragColor). */
function block0(source: string): string {
  const start = source.indexOf('void main() {');
  const end = source.lastIndexOf('fragColor =');
  return start === -1 || end === -1 ? '' : source.slice(start, end);
}

/* -------------------------------------------------------------------------- */
/* 11. Destructuring: the port the wire names is the component it gets          */
/* -------------------------------------------------------------------------- */

/**
 * A one-producer graph whose only interesting feature is *which* port the wire names.
 * Wiring `.x` would pass even if destructuring were broken (`.x` is what the old broken
 * path happened to produce), so every case below names a later component on purpose.
 */
function docWiringPort(id: string, kind: NodeKind, port: string): ShaderDoc {
  return {
    irVersion: IR_VERSION,
    id: `destructure-${kind}`,
    title: `destructure ${kind}`,
    spec: { intent: `read \`${kind}.${port}\``, category: 'test', tags: ['destructuring'], features: [] },
    stage: 'fragment',
    graph: {
      nodes: [
        { id, kind },
        { id: 'n_out', kind: 'output', inputs: { color: { node: id, port } } },
      ],
      edges: [{ from: { node: id, port }, to: { node: 'n_out', port: 'color' } }],
    },
    render: { resolution: [256, 256], qualityTier: 'medium', maxIterations: 8, transparent: false },
    targets: ['glsl'],
  };
}

/** `emitBodies` for a graph, without needing a registered target. */
function bodiesFor(doc: ShaderDoc) {
  return emitBodies(doc, analyzeGraph(doc.graph, CATALOG), CATALOG);
}

test('11a. a single-output kind never takes the destructuring path', () => {
  // The negative half of the contract: `outs.length === 1` must stay on the direct
  // `float L_<id>_<port> = <expr>;` path, whose local is what every target's epilogue and
  // every downstream wire reads. Only a kind with 2+ ports may declare a `tmp_<id>`.
  const problems: string[] = [];
  for (const [kind, schema] of ENTRIES.filter(([, s]) => s.outputs.length === 1)) {
    const id = `n_${kind}`;
    const { lines } = bodiesFor(docWiringPort(id, kind, schema.outputs[0]!.name));
    const decl = lines.find((l) => l.includes(`L_${id}_`));
    if (!decl) problems.push(`${kind}: no local emitted`);
    else if (!decl.includes(`${schema.outputs[0]!.type} L_${id}_${schema.outputs[0]!.name} =`)) {
      problems.push(`${kind}: unexpected declaration \`${decl.trim()}\``);
    }
    if (lines.some((l) => l.includes(`tmp_${id}`))) problems.push(`${kind}: emitted a tmp_ local with one output port`);
  }
  assert.deepEqual(problems, []);
});

test('11b. every output port of every multi-output kind destructures to its own component', () => {
  const problems: string[] = [];
  const covered: string[] = [];
  for (const [kind, ports] of MULTI_OUT) {
    const id = `n_${kind}`;
    for (const [idx, port] of ports.entries()) {
      covered.push(`${kind}.${port}`);
      const doc = docWiringPort(id, kind, port);
      const analysis = analyzeGraph(doc.graph, CATALOG);
      const { lines, diagnostics } = emitBodies(doc, analysis, CATALOG);
      const errors = diagnostics.filter((d) => d.severity === 'error');
      if (errors.length) {
        problems.push(`${kind}.${port}: ${errors.map((d) => `${d.code} ${d.message}`).join('; ')}`);
        continue;
      }
      const joined = lines.join('\n');
      // The tuple is declared with the *body's* return type, never with a port type.
      if (!joined.includes(`vec4 tmp_${id} = `)) problems.push(`${kind}.${port}: no \`vec4 tmp_${id}\``);
      // Components are assigned in declaration order — port 0 is .x even when the wire
      // came in on .w — and each port gets its own typed local.
      for (const [i, p] of CATALOG[kind].outputs.entries()) {
        const comp = COMPONENTS[i]!;
        const want = `${p.type} L_${id}_${p.name} = tmp_${id}.${comp};`;
        if (!joined.includes(want)) problems.push(`${kind}.${port}: missing \`${want}\``);
      }
      // The wire really did read the named port: the consumer's local is that component.
      if (idx > 0 && !joined.includes(`tmp_${id}.${COMPONENTS[idx]}`)) {
        problems.push(`${kind}.${port}: the wired port .${COMPONENTS[idx]} is never referenced`);
      }
    }
  }
  assert.deepEqual(problems, []);
  // Guard the guard: every port of every multi-output kind is visited, and all but the
  // first port of each kind are wired through a non-`.x` case.
  const firstPorts = new Set(MULTI_OUT.map(([kind]) => `${kind}.${CATALOG[kind].outputs[0]!.name}`));
  assert.deepEqual(covered, MULTI_OUT.flatMap(([kind, ports]) => ports.map((p) => `${kind}.${p}`)));
  // time 2 + split_vec2 1 + split_vec4 3 + polar 1 + voronoi 3 + sdf 1 = 11. Spelled out
  // because the previous value was one too high and the failure (11 vs 12) says nothing about
  // which kind drifted.
  assert.equal(covered.filter((c) => !firstPorts.has(c)).length, 11, 'the non-.x cases went missing');});

test('11c. the designed ports land on the documented components (regression pin)', () => {
  // The exact mapping from the catalog header, asserted per kind so a reordering of
  // `outputs` cannot silently rewire what every existing graph reads.
  const expected: Record<string, string[]> = {
    time: ['time', 'sin', 'cos'],
    polar: ['angle', 'radius'],
    voronoi: ['f1', 'f2', 'edges', 'cell'],
    sdf: ['distance', 'mask'],
    // x/y/z/w, i.e. the passthrough order. Pinned for the same reason as the others: a
    // reordering would silently change which component every existing `split_vec4` wire reads,
    // and a graph that pulls `.w` out of `mouse` would start reading the press position.
    split_vec4: ['x', 'y', 'z', 'w'],
    split_vec2: ['x', 'y'],
  };
  const problems: string[] = [];
  for (const [kind, ports] of Object.entries(expected)) {
    const actual = CATALOG[kind as NodeKind].outputs.map((p) => p.name);
    if (actual.join(',') !== ports.join(',')) problems.push(`${kind}: ports are ${actual.join(', ')}, expected ${ports.join(', ')}`);
  }
  // `primaryOut` is what a bare wire resolves to; for the ports that existed before this
  // change it must still be the same *signal*, so old graphs keep their image.
  const primary: Record<string, string> = { time: 'time', polar: 'angle', voronoi: 'f1', sdf: 'mask' };
  for (const [kind, port] of Object.entries(primary)) {
    if (CATALOG[kind as NodeKind].primaryOut !== port) {
      problems.push(`${kind}: primaryOut is ${CATALOG[kind as NodeKind].primaryOut}, expected ${port}`);
    }
  }
  assert.deepEqual(problems, []);
});
