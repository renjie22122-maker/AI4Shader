/**
 * The capability matrix — every host-facing feature × every target
 * ===============================================================
 *
 * `targets.test.ts` builds its catalogs by hand (it is testing the *emitter*), and
 * `catalog.test.ts` emits one graph containing every kind. Neither states the thing a user is
 * actually promised: **for each kind of effect this tool claims to handle, all six targets emit
 * it.** That is the claim, so that is what this file checks — one small, realistic document per
 * feature, crossed with the six targets, asserting zero errors (and, stronger, zero *warnings*:
 * `IR011_DEAD_NODES` or a lint in a shipped document would mean the feature only half works).
 *
 * The features are not "one node each" — they are the shapes a request arrives in:
 *
 *   surface            a document that shades a 3D surface (needs a second entry point)
 *   channel            a document that samples the host image channel
 *   feedback           the same, with `render.feedback` (the channel is the previous frame)
 *   feedback+surface   **both**, which is legal in the IR and has no preview (see below)
 *   derivatives        anti-aliasing, the one node that reads `fwidth`
 *   escape-hatch       `custom_glsl`, whose body is written at runtime rather than shipped
 *   quality-scaled     the octave loop that the quality tier controls
 *   tileable           the wrap that makes a texture seamless
 *
 * The combination row matters most: `feedback+surface` is the one shape with **no host that can
 * show it** — the flat path cannot evaluate a surface shader, and the 3D path does not implement
 * feedback. It still has to *emit*, because the document is valid and the export is what a user
 * takes to Unity or UE; the preview limitation is the preview's, and the workbench says so.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { emit, type Emission } from '../src/emitter.js';
import { normalizeEdges } from '../src/patcher.js';
import { tokenize } from '../src/rewrite.js';
import { REWRITE_ALIASES } from '../src/glslSubset.js';
import { CATALOG } from '../src/catalog/index.js';
import '../src/targets/index.js';
import { SURFACE_HOST_SYMBOLS } from '../src/types.js';
import type { NodeKind, ShaderDoc, ShaderNode, TargetFormat } from '../src/types.js';

/** The emitted source with every comment replaced by a space — assertions about *code*. */
function withoutComments(src: string): string {
  return tokenize(src)
    .map((t) => (t.kind === 'comment' ? ' ' : t.text))
    .join('');
}

const TARGETS: TargetFormat[] = ['glsl', 'wgsl', 'hlsl', 'msl', 'shaderlab', 'ue'];

type Wire = { node: string; port?: string };
const wire = (node: string, port: string): Wire => ({ node, port });

const OUTPUT: ShaderNode = { id: 'n_out', kind: 'output' as NodeKind };

/**
 * One document per feature, built from the real catalog.
 *
 * Kept small on purpose: a failure should point at the feature, and a 40-node document would
 * make "which part of this target broke" a research question. The composition test is the pond
 * example, which the Python suite compiles.
 */
function docFor(feature: string): ShaderDoc {
  let nodes: ShaderNode[];
  let feedback = false;
  switch (feature) {
    case 'surface':
      nodes = [
        { id: 'n_a', kind: 'normal' as NodeKind },
        { id: 'n_b', kind: 'light_dir' as NodeKind },
        { id: 'n_c', kind: 'dot' as NodeKind, inputs: { a: wire('n_a', 'normal'), b: wire('n_b', 'direction') } },
        { id: 'n_d', kind: 'color_ramp' as NodeKind, inputs: { t: wire('n_c', 'value') } },
        { ...OUTPUT, inputs: { color: wire('n_d', 'color') } },
      ];
      break;
    case 'channel':
      nodes = [
        { id: 'n_a', kind: 'uv' as NodeKind },
        { id: 'n_b', kind: 'texture' as NodeKind, inputs: { uv: wire('n_a', 'uv') } },
        { ...OUTPUT, inputs: { color: wire('n_b', 'color') } },
      ];
      break;
    case 'feedback':
      feedback = true;
      nodes = [
        { id: 'n_a', kind: 'uv' as NodeKind },
        { id: 'n_b', kind: 'texture' as NodeKind, inputs: { uv: wire('n_a', 'uv') } },
        // Uses the alpha, which is what a feedback document does: the channel's fourth
        // component is the state, not a colour.
        { id: 'n_c', kind: 'split_vec4' as NodeKind, inputs: { value: wire('n_b', 'color') } },
        { id: 'n_d', kind: 'color_ramp' as NodeKind, inputs: { t: wire('n_c', 'w') } },
        { ...OUTPUT, inputs: { color: wire('n_d', 'color') } },
      ];
      break;
    case 'feedback+surface':
      feedback = true;
      nodes = [
        { id: 'n_a', kind: 'normal' as NodeKind },
        { id: 'n_b', kind: 'light_dir' as NodeKind },
        { id: 'n_c', kind: 'uv' as NodeKind },
        { id: 'n_d', kind: 'texture' as NodeKind, inputs: { uv: wire('n_c', 'uv') } },
        { id: 'n_e', kind: 'dot' as NodeKind, inputs: { a: wire('n_a', 'normal'), b: wire('n_b', 'direction') } },
        { id: 'n_f', kind: 'mix' as NodeKind, inputs: { a: wire('n_d', 'color'), t: wire('n_e', 'value') } },
        { ...OUTPUT, inputs: { color: wire('n_f', 'value') } },
      ];
      break;
    case 'derivatives':
      nodes = [
        { id: 'n_a', kind: 'uv' as NodeKind },
        { id: 'n_b', kind: 'sdf' as NodeKind, inputs: { uv: wire('n_a', 'uv') } },
        { id: 'n_c', kind: 'antialias' as NodeKind, inputs: { value: wire('n_b', 'distance') } },
        { id: 'n_d', kind: 'color_ramp' as NodeKind, inputs: { t: wire('n_c', 'value') } },
        { ...OUTPUT, inputs: { color: wire('n_d', 'color') } },
      ];
      break;
    case 'escape-hatch':
      nodes = [
        { id: 'n_a', kind: 'uv' as NodeKind },
        {
          id: 'n_b',
          kind: 'custom_glsl' as NodeKind,
          // Every alias the rewriter has to lower, in one snippet: `mod_` (floor semantics),
          // `mix_`, `fract_`, `atan2_`, a `mat2` product and a shared-library helper. A catalog
          // body is compiled from a constant that a test can enumerate; this text arrives at
          // runtime, which is the whole point of the escape hatch and the whole reason it needs
          // its own verification.
          body:
            'vec2 q_${node_id} = fract_(${uv} * 6.0); return vec4(mix_(q_${node_id}.x, mod_(q_${node_id}.x, 0.5), atan2_(q_${node_id}.y, q_${node_id}.x)), mat2_mul_(rot2(0.5), q_${node_id}), 1.0);',
          inputs: { uv: wire('n_a', 'uv') },
        },
        { ...OUTPUT, inputs: { color: wire('n_b', 'value') } },
      ];
      break;
    case 'quality-scaled':
      nodes = [
        { id: 'n_a', kind: 'uv' as NodeKind },
        { id: 'n_b', kind: 'fbm' as NodeKind, inputs: { uv: wire('n_a', 'uv') } },
        { id: 'n_c', kind: 'color_ramp' as NodeKind, inputs: { t: wire('n_b', 'value') } },
        { ...OUTPUT, inputs: { color: wire('n_c', 'color') } },
      ];
      break;
    case 'tileable':
      nodes = [
        { id: 'n_a', kind: 'uv' as NodeKind },
        {
          id: 'n_b',
          kind: 'fbm' as NodeKind,
          params: { tile: { kind: 'float', value: 1 } },
          inputs: { uv: wire('n_a', 'uv') },
        },
        { id: 'n_c', kind: 'color_ramp' as NodeKind, inputs: { t: wire('n_b', 'value') } },
        { ...OUTPUT, inputs: { color: wire('n_c', 'color') } },
      ];
      break;
    default:
      throw new Error(`unknown feature ${feature}`);
  }

  return {
    irVersion: '1.0.0',
    id: `matrix_${feature.replace(/\+/g, '_')}`,
    title: `matrix ${feature}`,
    spec: { intent: feature, category: 'test', tags: ['test'], features: [] },
    stage: 'fragment',
    graph: { nodes, edges: normalizeEdges({ nodes, edges: [] }).edges },
    render: { resolution: [512, 512], qualityTier: 'medium', maxIterations: 8, transparent: false, feedback },
    targets: TARGETS,
  };
}

const FEATURES = [
  'surface',
  'channel',
  'feedback',
  'feedback+surface',
  'derivatives',
  'escape-hatch',
  'quality-scaled',
  'tileable',
];

const emitFor = (feature: string, target: TargetFormat): Emission =>
  emit(docFor(feature), { catalog: CATALOG, target });

test('every feature emits on every target, with zero errors', () => {
  const failures: string[] = [];
  for (const feature of FEATURES) {
    for (const target of TARGETS) {
      const e = emitFor(feature, target);
      const errors = e.diagnostics.filter((d) => d.severity === 'error');
      if (errors.length) {
        failures.push(`${feature} on ${target}: ${errors.map((d) => `${d.code} ${d.message}`).join('; ')}`);
      }
      if (!e.source.length) failures.push(`${feature} on ${target}: empty source`);
    }
  }
  assert.deepEqual(failures, [], `the capability matrix has holes:\n${failures.join('\n')}`);
});

test('every feature emits on every target without warnings either', () => {
  /*
   * Stronger than "no errors", and deliberately so. A shipped feature that emits `IR011_DEAD_NODES`
   * would mean the document half-works: the source compiles and the effect is not in it. The only
   * warnings allowed here are the ones that describe the *document* rather than the emission —
   * and this matrix has none of those either, so the expected list is empty.
   */
  const warnings: string[] = [];
  for (const feature of FEATURES) {
    for (const target of TARGETS) {
      for (const d of emitFor(feature, target).diagnostics) {
        if (d.severity === 'warning') warnings.push(`${feature} on ${target}: ${d.code} ${d.message}`);
      }
    }
  }
  assert.deepEqual(warnings, []);
});

test('the surface matrix row really takes the surface entry point, and the others do not', () => {
  /*
   * A matrix that only counts diagnostics would pass if `needs.surface` silently stopped being
   * read — every row would emit the quad form and nothing would complain. So the flag and its
   * consequence are checked per row, in the direction that matters: the two surface rows must
   * change shape, and the rest must not (a stray `vNormal` in a quad shader would mean a
   * document started allocating vertex inputs it cannot use).
   */
  for (const feature of FEATURES) {
    const surface = feature.includes('surface');
    for (const target of TARGETS) {
      const e = emitFor(feature, target);
      assert.equal(
        e.needs.surface,
        surface,
        `${feature}/${target}: needs.surface should be ${surface}`,
      );
    }
    // And the GLSL projection is where the difference is observable without a driver.
    const glsl = emitFor(feature, 'glsl').source;
    if (surface) {
      assert.match(glsl, /in vec3 vNormal;/, `${feature}: the surface row must interpolate a normal`);
    } else {
      assert.ok(!glsl.includes('vNormal'), `${feature}: a quad row must not declare a normal`);
      assert.match(glsl, /gl_FragCoord\.xy/, `${feature}: a quad row must read gl_FragCoord`);
    }
  }
});

test('the channel rows declare the channel, and the feedback flag is on both of them', () => {
  for (const feature of FEATURES) {
    const usesChannel = feature.includes('channel') || feature.includes('feedback');
    const e = emitFor(feature, 'glsl');
    assert.equal(e.needs.channel, usesChannel, `${feature}: needs.channel should be ${usesChannel}`);
    // The flag lives on the document, so it is the same for every target — asserted once here
    // rather than six times above.
    assert.equal(docFor(feature).render.feedback === true, feature.startsWith('feedback'));
  }
});

test('a custom body that reads a surface symbol makes the document a surface document', () => {
  /*
   * The escape hatch is free text, and free text can name a host symbol that only the surface
   * entry point declares. Before this rule such a body emitted the **quad** header, reported
   * nothing, and failed to compile: `iNormal` is not an identifier a quad shader has. That is the
   * declared-and-silent failure mode — the file looks finished and the engine says otherwise.
   *
   * The fix treats the body as part of the reachable set's capability surface, exactly like a
   * `normal` node: reading `iNormal` *means* the document needs interpolated vertex outputs, and
   * the flag makes every target supply them.
   */
  const custom = (body: string): ShaderDoc => ({
    ...docFor('channel'),
    id: 'custom_surface',
    graph: normalizeEdges({
      nodes: [
        { id: 'n_a', kind: 'uv' as NodeKind },
        { id: 'n_b', kind: 'custom_glsl' as NodeKind, body, inputs: { uv: wire('n_a', 'uv') } },
        { ...OUTPUT, inputs: { color: wire('n_b', 'value') } },
      ],
      edges: [],
    }),
  });

  for (const symbol of SURFACE_HOST_SYMBOLS) {
    const body =
      symbol === 'iCameraPos' ? `return vec4(normalize(${symbol}), 1.0);` : `return vec4(${symbol}, 1.0);`;
    const e = emit(custom(body), { catalog: CATALOG, target: 'glsl' });
    assert.equal(e.needs.surface, true, `${symbol} in a body must set needs.surface`);
    assert.match(e.source, /in vec3 vNormal;/, `${symbol}: the surface header must be emitted`);
    assert.match(e.source, /uniform vec3 iCameraPos;/, `${symbol}: the camera must be declared`);
  }

  // A name that merely *starts* like a host symbol is not one: an author's own local must not
  // silently turn the document into a surface shader (which also moves `uv` to the mesh).
  const lookalike = emit(custom('float iNormalised = 0.5; return vec4(iNormalised, 0.0, 0.0, 1.0);'), {
    catalog: CATALOG,
    target: 'glsl',
  });
  assert.equal(lookalike.needs.surface, false, 'a lookalike local must not flip the flag');
  assert.match(lookalike.source, /gl_FragCoord\.xy/, 'and the quad entry must stay');

  // A body that reads none of them stays a quad document.
  const plain = emit(custom('return vec4(${uv}, 0.0, 1.0);'), { catalog: CATALOG, target: 'glsl' });
  assert.equal(plain.needs.surface, false);
});

test('every alias the rewriter promises is resolved on every target', () => {
  /*
   * `REWRITE_ALIASES` is a *contract* — "aliases the rewriter MUST resolve" — and a contract with
   * no test is a comment. Each alias is a name that means something different in each language (or
   * nothing at all), so one that is missing from a lowering path produces source that is valid on
   * five targets and a compile error on the sixth, with no diagnostic anywhere.
   *
   * The derivative family is the sharpest case and the reason this is not busywork: the four
   * spellings are `dFdx` (GLSL), `dpdx` (WGSL), `ddx` (HLSL) and `dfdx` (MSL) — no two alike.
   */
  const CALLS: Record<string, string> = {
    fract_: 'return vec4(fract_(${uv}.x), fract_(${uv}.y), 0.0, 1.0);',
    mod_: 'return vec4(mod_(${uv}.x, 0.5), mod_(${uv}.y, 0.5), 0.0, 1.0);',
    mix_: 'return vec4(mix_(${uv}.x, 0.2, 0.5), mix_(${uv}.y, 0.8, 0.5), 0.0, 1.0);',
    atan2_: 'return vec4(atan2_(${uv}.y, ${uv}.x), 0.0, 0.0, 1.0);',
    inversesqrt_: 'return vec4(inversesqrt_(${uv}.x + 0.5), 0.0, 0.0, 1.0);',
    dFdx_: 'return vec4(dFdx_(${uv}.x), 0.0, 0.0, 1.0);',
    dFdy_: 'return vec4(dFdy_(${uv}.y), 0.0, 0.0, 1.0);',
    tex2D_: 'return tex2D_(${uv});',
    mat2_mul_: 'return vec4(mat2_mul_(rot2(0.5), ${uv} - vec2(0.5, 0.5)), 0.0, 1.0);',
    mat3_mul_: 'return vec4(mat3_mul_(mat3_rot_(0.5), vec3(${uv}, 0.5)), 1.0);',
  };

  const problems: string[] = [];
  for (const alias of REWRITE_ALIASES) {
    const body = CALLS[alias];
    if (!body) {
      problems.push(`${alias}: no probe written, so nothing checks it`);
      continue;
    }
    for (const target of TARGETS) {
      const doc = {
        ...docFor('channel'), // shape only; the graph is replaced below
        id: `alias_${alias}`,
        graph: normalizeEdges({
          nodes: [
            { id: 'n_a', kind: 'uv' as NodeKind },
            { id: 'n_b', kind: 'custom_glsl' as NodeKind, body, inputs: { uv: wire('n_a', 'uv') } },
            { ...OUTPUT, inputs: { color: wire('n_b', 'value') } },
          ],
          edges: [],
        }),
      };
      const e = emit(doc, { catalog: CATALOG, target });
      const errors = e.diagnostics.filter((d) => d.severity === 'error');
      if (errors.length) problems.push(`${alias} on ${target}: ${errors.map((d) => d.code).join(', ')}`);
      // The alias itself must be gone from the *code* (comments name it on purpose).
      if (new RegExp(`\\b${alias}\\s*\\(`).test(withoutComments(e.source))) {
        problems.push(`${alias} on ${target}: the alias survived into the emitted code`);
      }
    }
  }
  assert.deepEqual(problems, []);

  // And the spellings that the aliases resolve to, pinned per target — a "resolved" alias that
  // resolves to the *wrong name* would pass the check above and still not compile.
  const derivative = (target: TargetFormat): string => {
    const doc = {
      ...docFor('channel'),
      id: 'alias_spelling',
      graph: normalizeEdges({
        nodes: [
          { id: 'n_a', kind: 'uv' as NodeKind },
          {
            id: 'n_b',
            kind: 'custom_glsl' as NodeKind,
            body: 'return vec4(dFdx_(${uv}.x), 0.0, 0.0, 1.0);',
            inputs: { uv: wire('n_a', 'uv') },
          },
          { ...OUTPUT, inputs: { color: wire('n_b', 'value') } },
        ],
        edges: [],
      }),
    };
    return withoutComments(emit(doc, { catalog: CATALOG, target }).source);
  };
  assert.match(derivative('glsl'), /\bdFdx\(/, 'GLSL spells it `dFdx`');
  assert.match(derivative('wgsl'), /\bdpdx\(/, 'WGSL spells it `dpdx`');
  assert.match(derivative('hlsl'), /\bddx\(/, 'HLSL spells it `ddx`');
  assert.match(derivative('msl'), /\bdfdx\(/, 'MSL spells it `dfdx`');
  assert.match(derivative('ue'), /\bddx\(/, 'UE is HLSL: `ddx`');
});

/* -------------------------------------------------------------------------- */
/* Doc plumbing                                                                */
/* -------------------------------------------------------------------------- */

test('the escape hatch reaches the source on every target, lowered', () => {
  /*
   * `custom_glsl` is the one place where the text is not a shipped constant, so "it emits" is not
   * enough: the body has to be *lowered* per target. The GLSL projection keeps `mod`/`mix`/`fract`
   * (its own builtins) and the HLSL projection must not — that is the rewriter's entire job, and
   * a pass-through would look identical in a diagnostic count.
   *
   * Negative assertions run on **comment-stripped** source. The emitted file keeps the shared
   * library's prose, and that prose names the aliases it is explaining; asserting on raw text
   * fails on documentation, and an assertion that fails on documentation is one somebody deletes.
   */
  const glsl = emitFor('escape-hatch', 'glsl').source;
  assert.ok(glsl.includes('mod('), 'GLSL keeps its own `mod`');
  assert.ok(!withoutComments(glsl).includes('mod_'), 'GLSL must not carry the subset alias');

  const hlsl = emitFor('escape-hatch', 'hlsl').source;
  const hlslCode = withoutComments(hlsl);
  assert.ok(!hlslCode.includes('mod_'), 'HLSL must not carry the subset alias');
  assert.match(hlslCode, /wf_mod\(/, 'HLSL needs the floor-form `wf_mod`, not `fmod`');
  assert.ok(hlslCode.includes('wf_mix('), 'HLSL spells `mix` as `lerp`, through the shim');
  assert.ok(!/fract\(/.test(hlslCode), 'HLSL has no `fract`; the rewriter must map it to `frac`');
  // Vector-first, because HLSL's matrix constructors are row-major: `mul(m, v)` would be the
  // transpose of the GLSL result and every rotation would come out mirrored.
  assert.match(hlslCode, /mul\(\s*q_n_b, rot2\(0\.5\)\)/, 'the `mat2_mul_` must lower to `mul(v, m)`');

  const wgsl = withoutComments(emitFor('escape-hatch', 'wgsl').source);
  assert.ok(!wgsl.includes('mod_'), 'WGSL must not carry the subset alias');
  assert.match(wgsl, /mix\(/, 'WGSL keeps `mix`');
  assert.match(wgsl, /atan2\(/, 'WGSL keeps the two-argument `atan2`');
});
