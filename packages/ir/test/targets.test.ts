/**
 * Target emitter tests
 * ====================
 *
 * These tests are the executable form of the per-target lowering contract documented in
 * `src/targets/README.md`. They assert the properties the rest of the system relies on:
 *
 *   1. determinism          — same IR ⇒ byte-identical source, twice in a row
 *   2. no leaks             — no restricted-GLSL type name or `*_` alias survives in a
 *                             non-GLSL target (the whole point of the token rewriter)
 *   3. clean diagnostics    — no `severity: 'error'` for a well-formed document
 *   4. uniform order        — declarations follow `uniforms[]` exactly, in every target
 *   5. quality tiers        — `octavesFor(doc)` drives the emitted loop bound
 *   6. the Y flip           — exactly once per target, in the entry point, never in a body
 *
 * The catalog is defined inline on purpose: `src/catalog/` is owned by another module and
 * these tests must be able to run (and fail) on their own.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { SURFACE_CAPABLE, analyzeGraph, emit, TARGET_EMITTERS, type Emission } from '../src/emitter.js';
import { normalizeEdges } from '../src/patcher.js';
import { rewriteSource, tokenize } from '../src/rewrite.js';
// Side-effect import: this is what registers all six emitters.
import '../src/targets/index.js';
import {
  IMPLEMENTED_TARGETS,
  assertWgslStructural,
  checkTargetLeaks,
  checkWgslStructural,
} from '../src/targets/_shared.js';
import { HLSLFAM_SHIMS, MSL_SHIMS } from '../src/rewrite.js';

/** The emitted source with every comment replaced by a space — assertions about *code*. */
function withoutComments(src: string): string {
  return tokenize(src)
    .map((t) => (t.kind === 'comment' ? ' ' : t.text))
    .join('');
}
import type {
  NodeKind,
  NodeSchema,
  ParamValue,
  PortSpec,
  QualityTier,
  ShaderDoc,
  ShaderNode,
  ShaderType,
  TargetFormat,
} from '../src/types.js';

/* -------------------------------------------------------------------------- */
/* Fixture: a hand-built catalog (no dependency on src/catalog)                */
/* -------------------------------------------------------------------------- */

function port(name: string, type: ShaderType, def?: ParamValue): PortSpec {  return def ? { name, type, default: def } : { name, type };
}

const f = (value: number): ParamValue => ({ kind: 'float', value });
const v3 = (value: [number, number, number]): ParamValue => ({ kind: 'vec3', value });

/**
 * Minimal catalog for the fixture graph. `outputColor` is parameterised so the
 * vec3→vec4 widening in the epilogue can be tested without inventing a second graph.
 */
function makeCatalog(outputColor: ShaderType): Record<NodeKind, NodeSchema> {
  const schemas: NodeSchema[] = [
    {
      kind: 'uv' as NodeKind,
      title: 'UV',
      category: 'generator',
      summary: 'Screen uv in [0,1], origin bottom-left.',
      inputs: [],
      outputs: [port('uv', 'vec2')],
      primaryOut: 'uv',
      glsl: 'return uv;',
    },
    {
      kind: 'const_color' as NodeKind,
      title: 'Color',
      category: 'generator',
      summary: 'A constant colour.',
      inputs: [port('color', 'vec3', v3([0.5, 0.5, 0.5]))],
      outputs: [port('color', 'vec3')],
      primaryOut: 'color',
      glsl: 'return ${color};',
    },
    {
      kind: 'fbm' as NodeKind,
      title: 'FBM',
      category: 'noise',
      summary: 'Fractal noise — exercises the shared library loop and OCTAVE_LIMIT.',
      inputs: [
        port('p', 'vec2'),
        port('octaves', 'int', { kind: 'int', value: 5 }),
        port('lacunarity', 'float', f(2)),
        port('gain', 'float', f(0.5)),
        port('ridged', 'float', f(0)),
        port('scale', 'float', f(1)),
      ],
      outputs: [port('n', 'float')],
      primaryOut: 'n',
      glsl: [
        'vec2 q = ${p} * ${scale} + vec2(iTime * 0.05, 0.0);',
        'return fbm2(q, ${octaves}, ${lacunarity}, ${gain}, ${ridged});',
      ].join('\n'),
      needs: { qualityScaled: true },
    },
    {
      kind: 'voronoi' as NodeKind,
      title: 'Voronoi',
      category: 'noise',
      summary: 'Worley noise — exercises a twice-nested loop in the library.',
      inputs: [port('p', 'vec2'), port('randomness', 'float', f(1))],
      outputs: [port('f1', 'float')],
      primaryOut: 'f1',
      glsl: 'return voronoi2(${p}, ${randomness}).x;',
    },
    {
      kind: 'color_ramp' as NodeKind,
      title: 'Color Ramp',
      category: 'color',
      summary: 'Cosine palette — exercises `palette` and vec3 literals.',
      inputs: [
        port('t', 'float', f(0.5)),
        port('a', 'vec3', v3([0.5, 0.5, 0.5])),
        port('b', 'vec3', v3([0.5, 0.5, 0.5])),
        port('c', 'vec3', v3([1, 1, 1])),
        port('d', 'vec3', v3([0, 0.33, 0.67])),
      ],
      outputs: [port('rgb', 'vec3')],
      primaryOut: 'rgb',
      glsl: 'return palette(${t}, ${a}, ${b}, ${c}, ${d});',
    },
    {
      kind: 'mix' as NodeKind,
      title: 'Mix',
      category: 'math',
      summary: 'Linear blend — exercises the `mix_` alias.',
      inputs: [port('a', 'vec3'), port('b', 'vec3'), port('t', 'float', f(0.5))],
      outputs: [port('color', 'vec4')],
      primaryOut: 'color',
      glsl: 'return vec4(mix_(${a}, ${b}, clamp(${t}, 0.0, 1.0)), 1.0);',
    },
    {
      kind: 'transform' as NodeKind,
      title: 'Transform',
      category: 'domain',
      summary: 'Exercises the matrix helpers, which the rewriter lowers per target.',
      inputs: [port('p', 'vec2'), port('angle', 'float', f(0.5))],
      outputs: [port('x', 'float')],
      primaryOut: 'x',
      // Mirrors the real catalog's `transform` body, including a type constructor
      // *inside* the matrix helper's arguments — the case `lowerMatrixMul` splices
      // verbatim and therefore does not rename on the first pass.
      glsl: [
        'mat2 m = mat2_rot_(${angle});',
        'return mat2_mul_(m, (${p} - vec2(0.5, 0.5)) * vec2(2.0, 2.0)).x;',
      ].join('\n'),
      needs: { matrix: true },
    },
    {
      kind: 'time' as NodeKind,      title: 'Time',
      category: 'generator',
      summary: 'Animation clock — exercises the `iFrame` (int) → float placeholder expansion.',
      inputs: [],
      outputs: [port('frame', 'float')],
      primaryOut: 'frame',
      glsl: 'return float(iFrame) * 0.001 + iTime;',
    },
    {
      kind: 'output' as NodeKind,
      title: 'Output',
      category: 'output',
      summary: 'Final colour.',
      inputs: [port('color', outputColor)],
      outputs: [port('color', outputColor)],
      primaryOut: 'color',
      glsl: 'return ${color};',
    },
  ];

  const catalog: Partial<Record<NodeKind, NodeSchema>> = {};
  for (const s of schemas) catalog[s.kind] = s;
  return catalog as Record<NodeKind, NodeSchema>;
}

/* -------------------------------------------------------------------------- */
/* Fixture: documents                                                          */
/* -------------------------------------------------------------------------- */

const U_TINT: ParamValue = {
  kind: 'param',
  name: 'u_tint',
  type: 'vec3',
  value: [0.02, 0.03, 0.05],
  min: 0,
  max: 1,
  label: 'Tint',
  group: 'Color',
};
const U_SCALE: ParamValue = {
  kind: 'param',
  name: 'u_scale',
  type: 'float',
  value: 3,
  min: 0.5,
  max: 8,
  step: 0.1,
  label: 'Scale',
  group: 'Noise',
};

type FixtureOptions = {
  tier?: QualityTier;
  transparent?: boolean;
  outputColor?: ShaderType;
  tags?: string[];
};

/**
 * uv ─┬─> fbm ──> color_ramp ─┐
 *     └─> voronoi ───────────┴─> mix ──> output
 *         const_color ────────┘
 *
 * Seven nodes, so every emitter path is exercised: a generator, two noise nodes that
 * pull in `fbm2`/`voronoi2`, the palette helper, the `mix_` alias, a vec3 literal, a
 * float uniform and a vec3 uniform.
 */
function makeDoc(o: FixtureOptions = {}): ShaderDoc {
  const nodes: ShaderNode[] = [
    { id: 'n_uv', kind: 'uv' as NodeKind },
    {
      id: 'n_base',
      kind: 'const_color' as NodeKind,
      params: { color: U_TINT },
    },
    {
      id: 'n_fbm',
      kind: 'fbm' as NodeKind,
      params: { scale: U_SCALE },
      inputs: { p: { node: 'n_uv', port: 'uv' } },
    },
    { id: 'n_vor', kind: 'voronoi' as NodeKind, inputs: { p: { node: 'n_uv', port: 'uv' } } },
    { id: 'n_ramp', kind: 'color_ramp' as NodeKind, inputs: { t: { node: 'n_fbm', port: 'n' } } },
    {
      id: 'n_mix',
      kind: 'mix' as NodeKind,
      inputs: {
        a: { node: 'n_ramp', port: 'rgb' },
        b: { node: 'n_base', port: 'color' },
        t: { node: 'n_vor', port: 'f1' },
      },
    },
    { id: 'n_out', kind: 'output' as NodeKind, inputs: { color: { node: 'n_mix', port: 'color' } } },
  ];

  return {
    irVersion: '1.0.0',
    id: 'doc_targets_test',
    title: 'Test Ripple',
    spec: {
      intent: 'exercise every target emitter',
      category: 'test',
      tags: o.tags ?? ['test'],
      features: [],
    },
    stage: 'fragment',
    graph: {
      nodes,
      edges: [
        { from: { node: 'n_uv', port: 'uv' }, to: { node: 'n_fbm', port: 'p' } },
        { from: { node: 'n_uv', port: 'uv' }, to: { node: 'n_vor', port: 'p' } },
        { from: { node: 'n_fbm', port: 'n' }, to: { node: 'n_ramp', port: 't' } },
        { from: { node: 'n_ramp', port: 'rgb' }, to: { node: 'n_mix', port: 'a' } },
        { from: { node: 'n_base', port: 'color' }, to: { node: 'n_mix', port: 'b' } },
        { from: { node: 'n_vor', port: 'f1' }, to: { node: 'n_mix', port: 't' } },
        { from: { node: 'n_mix', port: 'color' }, to: { node: 'n_out', port: 'color' } },
      ],
    },
    render: {
      resolution: [512, 512],
      qualityTier: o.tier ?? 'medium',
      maxIterations: 32,
      transparent: o.transparent ?? false,
    },
    targets: [...IMPLEMENTED_TARGETS],
  };
}

/** color_ramp ──> output(color: vec3): the epilogue must widen to vec4. */
function makeWidenDoc(): ShaderDoc {
  const doc = makeDoc();
  return {
    ...doc,
    graph: {
      nodes: [
        { id: 'n_ramp', kind: 'color_ramp' as NodeKind },
        { id: 'n_out', kind: 'output' as NodeKind, inputs: { color: { node: 'n_ramp', port: 'rgb' } } },
      ],
      edges: [{ from: { node: 'n_ramp', port: 'rgb' }, to: { node: 'n_out', port: 'color' } }],
    },
  };
}

/** time ──> output(color: vec4): the int `iFrame` host symbol used in float position. */
function makeFrameDoc(): ShaderDoc {
  const doc = makeDoc();
  return {
    ...doc,
    graph: {
      nodes: [
        { id: 'n_time', kind: 'time' as NodeKind },
        { id: 'n_out', kind: 'output' as NodeKind, inputs: { color: { node: 'n_time', port: 'frame' } } },
      ],
      edges: [{ from: { node: 'n_time', port: 'frame' }, to: { node: 'n_out', port: 'color' } }],
    },
  };
}

const CATALOG = makeCatalog('vec4');
const CATALOG_VEC3 = makeCatalog('vec3');
const NON_GLSL: TargetFormat[] = ['wgsl', 'hlsl', 'msl', 'shaderlab', 'ue'];

function build(doc: ShaderDoc, target: TargetFormat, catalog = CATALOG): Emission {
  return emit(doc, { target, catalog });
}

function errorsOf(e: Emission): string[] {
  return e.diagnostics.filter((d) => d.severity === 'error').map((d) => `${d.code}: ${d.message}`);
}

/* -------------------------------------------------------------------------- */
/* 1. registration + determinism                                               */
/* -------------------------------------------------------------------------- */

test('all six targets register themselves through targets/index.ts', () => {
  for (const t of IMPLEMENTED_TARGETS) {
    assert.equal(typeof TARGET_EMITTERS[t], 'function', `no emitter registered for ${t}`);
  }
  assert.equal(IMPLEMENTED_TARGETS.length, 6);
});

test('determinism: emitting twice is byte-identical, sidecars included', () => {
  const doc = makeDoc();
  for (const t of IMPLEMENTED_TARGETS) {
    const a = build(doc, t);
    const b = build(doc, t);
    assert.equal(a.source, b.source, `${t}: source is not deterministic`);
    assert.equal(a.hash, b.hash, `${t}: hash is not deterministic`);
    assert.deepEqual(
      (a.sidecars ?? []).map((s) => s.name),
      (b.sidecars ?? []).map((s) => s.name),
      `${t}: sidecar names differ between runs`,
    );
    assert.deepEqual(
      (a.sidecars ?? []).map((s) => s.content),
      (b.sidecars ?? []).map((s) => s.content),
      `${t}: sidecar content is not deterministic`,
    );
  }
});

/* -------------------------------------------------------------------------- */
/* 2. diagnostics                                                              */
/* -------------------------------------------------------------------------- */

test('no emitted target reports an error diagnostic for a well-formed document', () => {
  const doc = makeDoc();
  for (const t of IMPLEMENTED_TARGETS) {
    const e = build(doc, t);
    assert.deepEqual(errorsOf(e), [], `${t} produced error diagnostics`);
    for (const d of e.diagnostics) {
      assert.ok(d.severity === 'warning' || d.severity === 'info', `${t}: ${d.severity} ${d.code}`);
    }
  }
});

/* -------------------------------------------------------------------------- */
/* 3. uniform declaration order                                                */
/* -------------------------------------------------------------------------- */

test('uniform declarations follow uniforms[] array order in every target', () => {
  const doc = makeDoc();
  for (const t of IMPLEMENTED_TARGETS) {
    const e = build(doc, t);
    assert.deepEqual(
      e.uniforms.map((u) => u.name),
      ['u_tint', 'u_scale'],
      `${t}: uniform collection order changed`,
    );
    const tint = e.source.indexOf('u_tint');
    const scale = e.source.indexOf('u_scale');
    assert.ok(tint >= 0 && scale >= 0, `${t}: a uniform is missing from the source`);
    assert.ok(tint < scale, `${t}: u_tint must be declared before u_scale`);
  }
});

/* -------------------------------------------------------------------------- */
/* 4. no leaked GLSL names or aliases                                          */
/* -------------------------------------------------------------------------- */

test('non-GLSL targets contain no GLSL type names and no rewrite aliases', () => {
  const doc = makeDoc();
  for (const t of NON_GLSL) {
    const e = build(doc, t);
    assert.deepEqual(checkTargetLeaks(e.source, t), [], `${t} leaked restricted-GLSL tokens`);
  }
});

test('the fixture really does pull the shared library through the rewriter', () => {
  const doc = makeDoc();
  for (const t of IMPLEMENTED_TARGETS) {
    const src = build(doc, t).source;
    for (const fn of ['fbm2', 'voronoi2', 'palette', 'hsv2rgb', 'hash21']) {
      assert.ok(src.includes(fn), `${t}: shared library function ${fn} is missing`);
    }
  }
});

test('wgsl passes checkWgslStructural and assertWgslStructural', () => {
  const e = build(makeDoc(), 'wgsl');
  assert.deepEqual(checkWgslStructural(e.source), []);
  assertWgslStructural(e.source); // throwing form, must not throw

  // The checker has to be able to fail, or it proves nothing.
  assert.notDeepEqual(checkWgslStructural('fn ok() -> f32 { return 1.0; }\nfloat x = 1.0;'), []);
  assert.notDeepEqual(checkWgslStructural('fn ok() -> f32 { return mix_(1.0, 2.0, 0.5); }'), []);
  assert.notDeepEqual(checkWgslStructural('fn ok() -> f32 { return texture(uv); }'), []);
  assert.notDeepEqual(checkWgslStructural('fn ok() -> f32 { return 1.0;'), []);
  assert.notDeepEqual(checkWgslStructural('var x = 1.0;'), []);
  assert.notDeepEqual(checkWgslStructural('#version 300 es\nfn ok() -> f32 { return 1.0; }'), []);
});

test('wgsl structural promises hold: prologue, shims, entry point', () => {
  const src = build(makeDoc(), 'wgsl').source;
  assert.ok(
    src.includes('struct Uniforms { iResolution: vec3f, iTime: f32, iTimeDelta: f32, iFrame: i32, iMouse: vec4f, u_tint: vec3f, u_scale: f32 };'),
    'the mandatory uniform struct must be emitted verbatim, with IR uniforms appended',
  );
  assert.ok(src.includes('@group(0) @binding(0) var<uniform> U: Uniforms;'));
  assert.ok(src.includes('@group(0) @binding(1) var iChannel0: texture_2d<f32>;'));
  assert.ok(src.includes('@group(0) @binding(2) var iChannel0Sampler: sampler;'));
  assert.ok(src.includes('fn wf_mix(a: f32, b: f32, t: f32) -> f32'));
  assert.ok(src.includes('@fragment'));
  assert.ok(src.includes('fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {'));
});

test('the int host symbol `iFrame` is expanded into a float on every target', () => {
  const doc = makeFrameDoc();
  const expected: Partial<Record<TargetFormat, string>> = {
    glsl: 'float(iFrame)',
    wgsl: 'f32(f32(U.iFrame))',
    hlsl: 'float(iFrame)',
    msl: 'float(float(U.iFrame))',
    shaderlab: 'float(floor(_Time.w * 20.0))',
    ue: 'float(float(View.FrameNumber))',
  };
  for (const t of IMPLEMENTED_TARGETS) {
    const e = build(doc, t);
    assert.deepEqual(errorsOf(e), [], `${t}: iFrame document produced errors`);
    assert.ok(
      e.source.includes(expected[t]!),
      `${t}: expected the iFrame expansion ${expected[t]}`,
    );
    assert.ok(e.source.includes('iTime'), `${t}: iTime is missing`);
  }
  assert.ok(build(doc, 'wgsl').source.includes('iFrame: i32'), 'wgsl declares iFrame as i32');
  assert.deepEqual(checkWgslStructural(build(doc, 'wgsl').source), []);
});

/* -------------------------------------------------------------------------- */
/* 5. quality tiers                                                            */
/* -------------------------------------------------------------------------- */

test('octavesFor(doc) controls the emitted loop bound in every target', () => {
  const low = makeDoc({ tier: 'low' });
  const ultra = makeDoc({ tier: 'ultra' });
  for (const t of IMPLEMENTED_TARGETS) {
    const a = build(low, t).source;
    const b = build(ultra, t).source;
    assert.notEqual(a, b, `${t}: quality tier does not affect the source`);
    assert.ok(a.includes('i < 3'), `${t}: low tier loop bound missing`);
    assert.ok(b.includes('i < 8'), `${t}: ultra tier loop bound missing`);
  }
});

/* -------------------------------------------------------------------------- */
/* 6. the Y flip lives in exactly one place                                    */
/* -------------------------------------------------------------------------- */

test('each target flips Y exactly once, in the entry point', () => {
  const glsl = build(makeDoc(), 'glsl').source;
  assert.ok(glsl.includes('vec2 uv = fragCoord / iResolution.xy;'), 'glsl must not flip');
  assert.ok(!glsl.includes('1.0 - fragCoord'));

  const wgsl = build(makeDoc(), 'wgsl').source;
  assert.equal((wgsl.match(/1\.0 - pos\.y/g) ?? []).length, 1, 'wgsl must flip exactly once');

  const hlsl = build(makeDoc(), 'hlsl').source;
  assert.equal((hlsl.match(/1\.0 - input\.uv\.y/g) ?? []).length, 1, 'hlsl must flip exactly once');

  const msl = build(makeDoc(), 'msl').source;
  assert.equal((msl.match(/1\.0 - in\.position\.y/g) ?? []).length, 1, 'msl must flip exactly once');

  const lab = build(makeDoc(), 'shaderlab').source;
  assert.ok(lab.includes('float2 uv = i.uv;'), 'shaderlab uses the mesh uv: no flip');
  assert.ok(!lab.includes('1.0 - i.uv.y'));

  const ue = build(makeDoc(), 'ue').source;
  assert.ok(ue.includes('float2 uv = InUV;'), 'ue takes uv from a pin: no flip');
});

/* -------------------------------------------------------------------------- */
/* 7. glsl: the one target that must actually run                              */
/* -------------------------------------------------------------------------- */

test('glsl: header is contiguous at the top and split-stable', () => {
  const src = build(makeDoc(), 'glsl').source;
  const lines = src.split('\n');
  assert.deepEqual(lines.slice(0, 3), [
    '#version 300 es',
    'precision highp float;',
    'precision highp int;',
  ]);
  assert.equal(lines[3], '', 'line 4 must be blank so the split is unambiguous');
  assert.ok(lines[4]!.startsWith('//'));
  assert.ok(src.includes('uniform vec3 iResolution;'));
  assert.ok(src.includes('uniform float iTime;'));
  assert.ok(src.includes('uniform int iFrame;'));
  assert.ok(src.includes('out vec4 fragColor;'));
  assert.ok(src.includes('uniform float u_scale; // default 3.0'));
  assert.ok(src.includes('uniform vec3 u_tint; // default vec3(0.02, 0.03, 0.05)'));
  assert.ok(src.includes('void main() {'));
});

test('transparent:false forces alpha to 1.0; transparent:true keeps the graph alpha', () => {
  const opaque = build(makeDoc({ transparent: false }), 'glsl').source;
  const glass = build(makeDoc({ transparent: true }), 'glsl').source;
  // The graph's node bodies live *inside* the entry point, so the colour expression is
  // assigned straight to `fragColor` rather than to an intermediate `color` local.
  assert.ok(
    opaque.includes('fragColor = vec4(L_n_mix_color.xyz, 1.0);'),
    'opaque docs must force alpha to 1.0',
  );
  assert.ok(glass.includes('fragColor = L_n_mix_color;'), 'transparent docs keep the graph alpha');
});

test('a vec3 output colour is widened with an explicit 1.0 alpha', () => {
  const doc = makeWidenDoc();
  for (const t of IMPLEMENTED_TARGETS) {
    const e = build(doc, t, CATALOG_VEC3);
    assert.deepEqual(errorsOf(e), [], `${t}: widening the output colour must not error`);
    assert.deepEqual(checkTargetLeaks(e.source, t === 'glsl' ? 'glsl' : t), []);
  }
  const glsl = build(doc, 'glsl', CATALOG_VEC3).source;
  assert.ok(glsl.includes('fragColor = vec4(L_n_ramp_rgb, 1.0);'));
  const wgsl = build(doc, 'wgsl', CATALOG_VEC3).source;
  assert.ok(wgsl.includes('return vec4f(L_n_ramp_rgb, 1.0);'));
  const msl = build(doc, 'msl', CATALOG_VEC3).source;
  assert.ok(msl.includes('return float4(L_n_ramp_rgb, 1.0);'));
});

test('glsl declares the host sampler and lowers the texture fetch to a two-argument call', () => {
  const doc = makeDoc();
  const withTexture: ShaderDoc = {
    ...doc,
    graph: {
      ...doc.graph,
      nodes: doc.graph.nodes.map((n) =>
        n.id === 'n_base'
          ? { id: 'n_base', kind: 'texture' as NodeKind, inputs: { uv: { node: 'n_uv', port: 'uv' } } }
          : n,
      ),
    },
  };
  const catalog: Record<NodeKind, NodeSchema> = {
    ...CATALOG,
    texture: {
      kind: 'texture' as NodeKind,
      title: 'Texture',
      category: 'generator',
      summary: 'texel fetch',
      inputs: [port('uv', 'vec2')],
      outputs: [port('color', 'vec3')],
      primaryOut: 'color',
      glsl: 'return tex2D_(${uv}).rgb;',
    },
  };
  const glsl = emit(withTexture, { target: 'glsl', catalog }).source;
  assert.ok(glsl.includes('texture(iChannel0, L_n_uv_uv)'), 'glsl fetch must bind the sampler');
  for (const t of NON_GLSL) {
    const src = emit(withTexture, { target: t, catalog }).source;
    assert.deepEqual(checkTargetLeaks(src, t), [], `${t}: texture fetch leaked an alias`);
    assert.ok(!src.includes('tex2D_'), `${t}: tex2D_ alias survived`);
  }
  const lab = emit(withTexture, { target: 'shaderlab', catalog }).source;
  assert.ok(lab.includes('tex2D(_Channel0, L_n_uv_uv * _Channel0_ST.xy + _Channel0_ST.zw)'));
  assert.ok(lab.includes('sampler2D _Channel0;'));
});

test('matrix helpers are lowered per target and never survive as aliases', () => {
  const doc = makeDoc();
  const matrixDoc: ShaderDoc = {
    ...doc,
    graph: {
      nodes: [
        { id: 'n_uv', kind: 'uv' as NodeKind },
        {
          id: 'n_xf',
          kind: 'transform' as NodeKind,
          inputs: { p: { node: 'n_uv', port: 'uv' } },
        },
        { id: 'n_out', kind: 'output' as NodeKind, inputs: { color: { node: 'n_xf', port: 'x' } } },
      ],
      edges: [
        { from: { node: 'n_uv', port: 'uv' }, to: { node: 'n_xf', port: 'p' } },
        { from: { node: 'n_xf', port: 'x' }, to: { node: 'n_out', port: 'color' } },
      ],
    },
  };

  for (const t of IMPLEMENTED_TARGETS) {
    const e = build(matrixDoc, t);
    assert.deepEqual(errorsOf(e), [], `${t}: matrix helpers produced errors`);
    assert.ok(!e.source.includes('mat2_mul_'), `${t}: the mat2_mul_ alias survived`);
    assert.ok(!e.source.includes('mat2_rot_'), `${t}: the mat2_rot_ alias survived`);
    // The rotation must go through the portable helper: mapping it to a matrix
    // *constructor* would silently turn a rotation into a diagonal scale.
    assert.ok(e.source.includes('rot2('), `${t}: rot2() was not used`);
    assert.deepEqual(checkTargetLeaks(e.source, t), [], `${t}: matrix lowering leaked`);
    // The library's own definitions of the two lowered helpers must be *removed*, not
    // merely renamed: the rewriter cannot rewrite a definition's signature, so leaving
    // them in produces `vec2 (mat2 m * vec2 v) { … }`. Their bodies are the detector.
    assert.ok(
      !e.source.includes('m[0][0]'),
      `${t}: the lowered matrix helper definition survived in the output`,
    );
    assert.ok(
      e.source.includes('one matrix helper definition was lowered per target here'),
      `${t}: the removal should leave an auditable note behind`,
    );
  }

  /*
   * `mat2_mul_` is inlined per target: `(m * v)` where matrices are column-major, and
   * **`mul(v, m)` — arguments swapped — on HLSL**, whose constructors are row-major and whose `*`
   * is component-wise (so `matrix * vector` does not even compile there). The swap is what keeps
   * the two answers equal; without it every rotation is mirrored on HLSL/UE. Verified against the
   * compiler rather than recalled: `mul(float2x2(c,-s,s,c), float2(1,0))` disassembles to
   * `(c, s)`, while GLSL's `mat2(c,-s,s,c) * vec2(1,0)` is `(c, -s)` and `mul(vec2(1,0), m)` is
   * `(c, -s)` too.
   */
  const inlined = /\(\s*L_n_uv_uv\s*-\s*(vec2|float2)\(0\.5, 0\.5\)\)\s*\*\s*(vec2|float2)\(2\.0, 2\.0\)/;
  assert.match(build(matrixDoc, 'glsl').source, inlined);
  assert.match(build(matrixDoc, 'hlsl').source, inlined);
  const hlslMatrix = build(matrixDoc, 'hlsl').source;
  assert.match(
    hlslMatrix,
    /mul\(\s*\(L_n_uv_uv - float2\(0\.5, 0\.5\)\) \* float2\(2\.0, 2\.0\),\s*m\)/,
    'HLSL must multiply vector-first: `mul(v, m)`, not `mul(m, v)`',
  );
  assert.ok(
    !/mul\(\s*m,/.test(hlslMatrix),
    '`mul(m, v)` on HLSL is the transpose of the GLSL result — every rotation would mirror',
  );
  assert.match(build(matrixDoc, 'wgsl').source, /mul\b|\(m \*/);
  for (const t of NON_GLSL) {
    const src = build(matrixDoc, t).source;
    assert.ok(!/\bvec2\s*\(/.test(src), `${t}: a vec2 constructor survived inside inlined args`);
  }
  assert.ok(build(matrixDoc, 'glsl').source.includes('mat2 m = rot2(0.5);'));
  assert.ok(build(matrixDoc, 'hlsl').source.includes('float2x2 m = rot2(0.5);'));
  assert.ok(build(matrixDoc, 'msl').source.includes('float2x2 m = rot2(0.5);'));
  // WGSL has no `mat2x2f m = …` form: every local is declared with `var`, and the type is
  // optional because it is inferred. The WGSL lowering pass is what emits `var`, so this
  // assertion is really checking that the pass ran over the library and the node bodies
  // rather than only over the entry point. The local's name is namespaced with the node id
  // (`m_n_xf`) and the pass annotates the type explicitly, hence the two optional groups.
  assert.ok(
    /var \w+(?::\s*\w+)?\s*=\s*rot2\(0\.5\);/.test(build(matrixDoc, 'wgsl').source),
    'wgsl: the matrix local must be a `var` declaration, not a C-style typed declaration',
  );
  assert.ok(build(matrixDoc, 'hlsl').source.includes('float2(2.0, 2.0)'));
  assert.ok(build(matrixDoc, 'wgsl').source.includes('vec2f(2.0, 2.0)'));
});

/* -------------------------------------------------------------------------- */
/* 7b. graph analysis: a valid DAG must never be reported as a cycle           */
/* -------------------------------------------------------------------------- */

/**
 * Regression for a real bug: the topological sort de-duplicated out-edges by `from->to`
 * pair while the in-degrees had been counted per *wire*. A node feeding two inputs of the
 * same consumer therefore contributed two units of in-degree but only one decrement, so
 * the consumer's in-degree never reached zero — which reported a **false `IR007_CYCLE`**
 * and dropped that node and everything downstream. The result was a shader that compiled
 * (the graph simply vanished from it) — silent, total loss of the effect.
 *
 * Fan-in from one source to several ports of one consumer is an ordinary thing to build:
 * it is how you, for example, drive `mix` from a single noise field.
 */
test('a node feeding several inputs of one consumer is not a cycle', () => {
  const fanIn: ShaderDoc = {
    ...makeDoc(),
    // `analyzeGraph` reads `graph.edges`, which `normalizeEdges` derives from
    // `nodes[].inputs`. Building a graph from inputs alone and forgetting this call
    // yields an edgeless graph in which every node looks like a root — the assertion
    // below would then pass for entirely the wrong reason.
    graph: normalizeEdges({
      nodes: [
        { id: 'n_uv', kind: 'uv' as NodeKind },
        // `voronoi` has several outputs in this fixture, which is what makes it possible
        // to wire one *source node* into several *inputs* of one consumer.
        { id: 'n_src', kind: 'voronoi' as NodeKind, inputs: { p: { node: 'n_uv', port: 'uv' } } },
        {
          id: 'n_mix',
          kind: 'mix' as NodeKind,
          inputs: {
            // TWO wires from ONE source into ONE consumer. This is the exact shape that
            // used to be mis-reported as a cycle.
            a: { node: 'n_src', port: 'f1' },
            b: { node: 'n_src', port: 'f1' },
          },
        },
        { id: 'n_out', kind: 'output' as NodeKind, inputs: { color: { node: 'n_mix', port: 'color' } } },
      ],
      edges: [],
    }),
  };

  const analysis = analyzeGraph(fanIn.graph, CATALOG as Record<NodeKind, NodeSchema>);
  assert.ok(
    !analysis.diagnostics.some((d) => d.code === 'IR007_CYCLE'),
    'a fan-in DAG must not be reported as a cycle',
  );
  assert.equal(
    analysis.order.length,
    fanIn.graph.nodes.length,
    'every node must be ordered — a dropped node means a silently empty shader',
  );

  // The same shape must also survive a full emit: ordering correctly but producing no
  // code would be an equally silent failure.
  const emission = build(fanIn, 'glsl');
  assert.deepEqual(errorsOf(emission), [], 'a fan-in DAG must emit without errors');
  assert.ok(
    emission.source.includes('L_n_mix_color'),
    'the fan-in consumer must actually be emitted, not dropped',
  );

  // And a genuine cycle must still be caught, or the fix would have traded one silent
  // failure for another.
  const cyclic: ShaderDoc = {
    ...makeDoc(),
    graph: normalizeEdges({
      nodes: [
        { id: 'a', kind: 'mix' as NodeKind, inputs: { a: { node: 'b', port: 'color' } } },
        { id: 'b', kind: 'mix' as NodeKind, inputs: { a: { node: 'a', port: 'color' } } },
        { id: 'o', kind: 'output' as NodeKind, inputs: { color: { node: 'a', port: 'color' } } },
      ],
      edges: [],
    }),
  };
  const cyclicAnalysis = analyzeGraph(cyclic.graph, CATALOG as Record<NodeKind, NodeSchema>);
  assert.ok(
    cyclicAnalysis.diagnostics.some((d) => d.code === 'IR007_CYCLE'),
    'a real cycle must still be detected',
  );
});

test('a hex literal is reported and normalised on every target, including wgsl', () => {
  // The check used to sit after the `wgsl` early-return in `rewriteNumber`, so on that
  // target a hex literal silently became `0x10.0` (invalid WGSL) and the warning that
  // exists to flag it never fired. Ordering a diagnostic after a `return` deletes it.
  for (const t of IMPLEMENTED_TARGETS) {
    const r = rewriteSource('float x = 0x10;', t, `hex/${t}`);
    assert.ok(
      r.diagnostics.some((d) => d.code === 'RW002_HEX_LITERAL'),
      `${t}: the hex literal must be reported`,
    );
    assert.ok(!/0[xX]10\.0/.test(r.source), `${t}: a hex literal must not gain a decimal point`);
  }
});

/* -------------------------------------------------------------------------- */
/* 8. sidecars                                                                 */
/* -------------------------------------------------------------------------- */

test('hlsl emits a register-binding sidecar', () => {
  const e = build(makeDoc(), 'hlsl');
  assert.deepEqual(e.sidecars?.map((s) => s.name), ['hlsl-bindings.json']);
  const j = JSON.parse(e.sidecars![0]!.content) as {
    cbuffers: { name: string; register: string; members: { name: string }[] }[];
    textures: { name: string; register: string }[];
    samplers: { name: string; register: string }[];
  };
  assert.equal(j.cbuffers[0]!.name, 'AI4ShaderParams');
  assert.equal(j.cbuffers[0]!.register, 'b0');
  assert.deepEqual(
    j.cbuffers[0]!.members.map((m) => m.name),
    ['iResolution', 'iTime', 'iTimeDelta', 'iFrame', 'iMouse', 'u_tint', 'u_scale'],
  );
  assert.equal(j.textures[0]!.register, 't0');
  assert.equal(j.samplers[0]!.register, 's0');
  assert.ok(e.source.includes('VSOutput VSMain(uint id : SV_VertexID)'));
  assert.ok(e.source.includes('float4 PSMain(PSInput input) : SV_Target'));
});

test('msl emits an index sidecar', () => {
  const e = build(makeDoc(), 'msl');
  assert.deepEqual(e.sidecars?.map((s) => s.name), ['ai4shader.metal.json']);
  const j = JSON.parse(e.sidecars![0]!.content) as {
    entryPoint: string;
    buffers: { name: string; index: number }[];
    textures: { name: string; index: number }[];
    samplers: { name: string; index: number }[];
  };
  assert.equal(j.entryPoint, 'fs_main');
  assert.equal(j.buffers[0]!.index, 0);
  assert.equal(j.textures[0]!.index, 0);
  assert.equal(j.samplers[0]!.index, 0);
  assert.ok(e.source.includes('#include <metal_stdlib>'));
  assert.ok(e.source.includes('constant AI4ShaderUniforms& U [[buffer(0)]]'));
});

test('shaderlab emits a complete .shader file plus a material payload', () => {
  const e = build(makeDoc(), 'shaderlab');
  const src = e.source;
  assert.deepEqual(e.sidecars?.map((s) => s.name), ['Material.ai4shader.yaml']);
  assert.ok(src.includes('Shader "AI4Shader/Test Ripple" {'));
  assert.ok(src.includes('Properties {'));
  assert.ok(src.includes('_u_scale ("Scale", Range(0.5, 8)) = 3'));
  assert.ok(src.includes('_u_tint ("Tint", Vector) = (0.02, 0.03, 0.05, 0)'));
  assert.ok(src.includes('SubShader {'));
  assert.ok(src.includes('#pragma vertex vert'));
  assert.ok(src.includes('#pragma fragment frag'));
  assert.ok(src.includes('#include "UnityCG.cginc"'));
  assert.ok(src.includes('UnityObjectToClipPos(v.vertex)'));
  assert.ok(src.includes('fixed4 frag(v2f i) : SV_Target'));
  assert.ok(src.includes('_ScreenParams.xy'));
  assert.ok(src.includes('_Time.y'));
  assert.ok(src.includes('_MousePos'));
  assert.ok(src.includes('float4 _MousePos;'));
  assert.ok(src.includes('// NO UNITY EQUIVALENT'), 'the file must document its degradations');

  const yaml = e.sidecars![0]!.content;
  assert.ok(yaml.includes('_u_scale'));
  assert.ok(yaml.includes('_u_tint'));
});

test('shaderlab switches to URP includes when the doc mentions URP', () => {
  const e = build(makeDoc({ tags: ['urp', 'water'] }), 'shaderlab');
  assert.ok(e.source.includes('Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl'));
  assert.ok(e.source.includes('TransformObjectToHClip(v.vertex.xyz)'));
  assert.ok(e.source.includes('"RenderPipeline"="UniversalPipeline"'));
  assert.ok(!e.source.includes('UnityCG.cginc'));
});

test('ue emits a pin list sidecar and a build recipe sidecar', () => {
  const e = build(makeDoc(), 'ue');
  assert.deepEqual(e.sidecars?.map((s) => s.name), [
    'CustomNode.inputs.json',
    'MaterialGraph.ai4shader.md',
  ]);
  const pins = JSON.parse(e.sidecars![0]!.content) as {
    name: string;
    type: string;
    defaultValue: unknown;
    label: string;
  }[];
  assert.ok(Array.isArray(pins), 'the pin list sidecar is an ordered array');
  assert.deepEqual(pins.map((p) => p.name), ['UV', 'u_tint', 'u_scale']);
  assert.deepEqual(pins.map((p) => p.type), ['float2', 'float3', 'float']);
  for (const p of pins) {
    assert.deepEqual(Object.keys(p).sort(), ['defaultValue', 'label', 'name', 'type']);
  }
  assert.equal(pins[2]!.defaultValue, 3);

  assert.ok(e.source.includes('MaterialFloat4 ai4shader_main(float2 InUV, float3 u_tint, MaterialFloat u_scale)'));
  assert.ok(e.source.includes('View.ViewSizeAndInvSize.xy'));
  assert.ok(e.source.includes('View.GameTime'));
  assert.ok(e.source.includes('return ai4shader_main(UV, u_tint, u_scale);'));
  assert.ok(/is only valid/i.test(e.source), 'View.GameTime must be documented as context-limited');

  const md = e.sidecars![1]!.content;
  assert.ok(md.includes('CMOT Float 4'));
  assert.ok(md.includes('1. Add a **Custom** node'));
  assert.ok(md.includes('```hlsl'));
});

/* -------------------------------------------------------------------------- */
/* 9. cross-target sanity                                                      */
/* -------------------------------------------------------------------------- */

test('every target emits a non-empty, newline-terminated source', () => {
  const doc = makeDoc();
  for (const t of IMPLEMENTED_TARGETS) {
    const src = build(doc, t).source;
    assert.ok(src.length > 500, `${t}: suspiciously short output`);
    assert.ok(src.endsWith('\n'), `${t}: source must end with a newline`);
    assert.ok(!src.includes('\t'), `${t}: tabs make golden diffs noisy`);
  }
});

/* -------------------------------------------------------------------------- */
/* 10. surface documents                                                       */
/* -------------------------------------------------------------------------- */

/**
 * A catalog with exactly one surface input, built here rather than imported.
 *
 * `src/catalog/` is another module's business (this file's header), and the property under
 * test is not "does `normal` exist" but "does a node that declares `needs.surface` change the
 * shape of every emitted shader, and only when it is actually reachable".
 */
function surfaceCatalog(): Record<NodeKind, NodeSchema> {
  const schemas: NodeSchema[] = [
    {
      kind: 'const_color' as NodeKind,
      title: 'Color',
      category: 'generator',
      summary: 'A constant colour.',
      inputs: [port('color', 'vec4', { kind: 'color', value: [0.2, 0.4, 0.9, 1] })],
      outputs: [port('color', 'vec4')],
      primaryOut: 'color',
      glsl: 'return ${color};',
    },
    {
      kind: 'normal' as NodeKind,
      title: 'Normal',
      category: 'surface',
      summary: 'Unit surface normal — the input a fullscreen quad cannot supply.',
      inputs: [],
      outputs: [port('normal', 'vec4')],
      primaryOut: 'normal',
      needs: { surface: true },
      glsl: 'return vec4(iNormal, 0.0);',
    },
    {
      kind: 'mix' as NodeKind,
      title: 'Mix',
      category: 'math',
      summary: 'Linear blend.',
      // Defaults on `a`/`b` matter here: the quad fixture deliberately leaves `b` unwired, and
      // an input with no default is `IR031_MISSING_INPUT` — an error that would make the quad
      // case fail for a reason that has nothing to do with surfaces.
      inputs: [
        port('a', 'vec4', { kind: 'color', value: [0, 0, 0, 1] }),
        port('b', 'vec4', { kind: 'color', value: [1, 1, 1, 1] }),
        port('t', 'float', f(0.5)),
      ],
      outputs: [port('color', 'vec4')],
      primaryOut: 'color',
      glsl: 'return mix_(${a}, ${b}, clamp(${t}, 0.0, 1.0));',
    },
    {
      kind: 'output' as NodeKind,
      title: 'Output',
      category: 'output',
      summary: 'Final colour.',
      inputs: [port('color', 'vec4')],
      outputs: [port('color', 'vec4')],
      primaryOut: 'color',
      glsl: 'return ${color};',
    },
  ];
  const catalog: Partial<Record<NodeKind, NodeSchema>> = {};
  for (const s of schemas) catalog[s.kind] = s;
  return catalog as Record<NodeKind, NodeSchema>;
}

/** `withNormal: false` produces the same document minus the surface node. */
function surfaceDoc(withNormal: boolean): ShaderDoc {
  const nodes: ShaderNode[] = [
    { id: 'n_base', kind: 'const_color' as NodeKind },
    ...(withNormal ? [{ id: 'n_n', kind: 'normal' as NodeKind }] : []),
    {
      id: 'n_mix',
      kind: 'mix' as NodeKind,
      inputs: withNormal
        ? { a: { node: 'n_base', port: 'color' }, b: { node: 'n_n', port: 'normal' } }
        : { a: { node: 'n_base', port: 'color' } },
    },
    { id: 'n_out', kind: 'output' as NodeKind, inputs: { color: { node: 'n_mix', port: 'color' } } },
  ];
  const edges = withNormal
    ? [
        { from: { node: 'n_base', port: 'color' }, to: { node: 'n_mix', port: 'a' } },
        { from: { node: 'n_n', port: 'normal' }, to: { node: 'n_mix', port: 'b' } },
        { from: { node: 'n_mix', port: 'color' }, to: { node: 'n_out', port: 'color' } },
      ]
    : [
        { from: { node: 'n_base', port: 'color' }, to: { node: 'n_mix', port: 'a' } },
        { from: { node: 'n_mix', port: 'color' }, to: { node: 'n_out', port: 'color' } },
      ];
  return {
    irVersion: '1.0.0',
    id: withNormal ? 'doc_surface' : 'doc_quad',
    title: 'Surface fixture',
    spec: { intent: 'surface', category: 'test', tags: ['test'], features: [] },
    stage: 'fragment',
    graph: { nodes, edges: normalizeEdges({ nodes, edges }).edges },
    render: { resolution: [512, 512], qualityTier: 'medium', maxIterations: 8, transparent: false },
    targets: ['glsl'],
  };
}

const surfBuild = (withNormal: boolean, target: TargetFormat): Emission =>
  emit(surfaceDoc(withNormal), { catalog: surfaceCatalog(), target });

test('a quad document reports needs.surface = false and stays a quad shader', () => {  const e = surfBuild(false, 'glsl');
  assert.deepEqual(
    e.diagnostics.filter((d) => d.severity === 'error').map((d) => d.code),
    [],
  );
  assert.equal(e.needs.surface, false, 'no reachable node declares needs.surface');
  // The quad path is the one every existing document takes, so it must be untouched: a
  // `gl_FragCoord`-derived uv and no vertex-input declarations anywhere.
  assert.ok(e.source.includes('gl_FragCoord.xy'), 'the quad entry must still read gl_FragCoord');
  assert.ok(!e.source.includes('vNormal'), 'a quad shader must not declare a normal');
  assert.ok(!e.source.includes('iCameraPos'), 'a quad shader needs no camera position');
});

/* -------------------------------------------------------------------------- */
/* 10b. the image channel, and the feedback flag that depends on it            */
/* -------------------------------------------------------------------------- */

/**
 * A catalog with one channel-sampling node.
 *
 * Same reasoning as `surfaceCatalog()`: the property under test is not "does `texture` declare
 * the flag" (that is the catalog's own test) but "does a node that declares it make the emitter
 * report it, and only when it is reachable".
 */
function channelDoc(withTexture: boolean, feedback?: boolean): ShaderDoc {
  const nodes: ShaderNode[] = [
    { id: 'n_base', kind: 'const_color' as NodeKind },
    // `n_tex.uv` takes a vec2; the colour node is only here to have *something* upstream, and a
    // vec4 into a vec2 port would be an error that has nothing to do with what is under test.
    ...(withTexture ? [{ id: 'n_tex', kind: 'channel_reader' as NodeKind }] : []),
    {
      id: 'n_mix',
      kind: 'mix' as NodeKind,
      inputs: {
        a: { node: 'n_base', port: 'color' },
        ...(withTexture ? { b: { node: 'n_tex', port: 'color' } } : {}),
      },
    },
    { id: 'n_out', kind: 'output' as NodeKind, inputs: { color: { node: 'n_mix', port: 'color' } } },
  ];
  return {
    irVersion: '1.0.0',
    id: withTexture ? 'doc_channel' : 'doc_plain',
    title: 'Channel fixture',
    spec: { intent: 'channel', category: 'test', tags: ['test'], features: [] },
    stage: 'fragment',
    graph: { nodes, edges: normalizeEdges({ nodes, edges: [] }).edges },
    render: {
      resolution: [512, 512],
      qualityTier: 'medium',
      maxIterations: 8,
      transparent: false,
      ...(feedback === undefined ? {} : { feedback }),
    },
    targets: ['glsl'],
  };
}

/** `channel_reader` reads the host image channel; `mix` and `const_color` do not. */
function channelCatalog(): Record<NodeKind, NodeSchema> {
  const schemas: NodeSchema[] = [
    {
      kind: 'const_color' as NodeKind,
      title: 'Color',
      category: 'generator',
      summary: 'A constant colour.',
      inputs: [port('color', 'vec4', { kind: 'color', value: [0.2, 0.4, 0.9, 1] })],
      outputs: [port('color', 'vec4')],
      primaryOut: 'color',
      glsl: 'return ${color};',
    },
    {
      kind: 'channel_reader' as NodeKind,
      title: 'Channel',
      category: 'generator',
      summary: 'Samples the host image channel.',
      inputs: [port('uv', 'vec2', { kind: 'vec2', value: [0.5, 0.5] })],
      outputs: [port('color', 'vec4')],
      primaryOut: 'color',
      needs: { channel: true },
      glsl: 'return tex2D_(${uv});',
    },
    {
      kind: 'mix' as NodeKind,
      title: 'Mix',
      category: 'math',
      summary: 'Linear blend.',
      inputs: [
        port('a', 'vec4', { kind: 'color', value: [0, 0, 0, 1] }),
        port('b', 'vec4', { kind: 'color', value: [1, 1, 1, 1] }),
        port('t', 'float', f(0.5)),
      ],
      outputs: [port('color', 'vec4')],
      primaryOut: 'color',
      glsl: 'return mix_(${a}, ${b}, clamp(${t}, 0.0, 1.0));',
    },
    {
      kind: 'output' as NodeKind,
      title: 'Output',
      category: 'output',
      summary: 'Final colour.',
      inputs: [port('color', 'vec4')],
      outputs: [port('color', 'vec4')],
      primaryOut: 'color',
      glsl: 'return ${color};',
    },
  ];
  const catalog: Partial<Record<NodeKind, NodeSchema>> = {};
  for (const s of schemas) catalog[s.kind] = s;
  return catalog as Record<NodeKind, NodeSchema>;
}

const chanBuild = (withTexture: boolean, feedback?: boolean): Emission =>
  emit(channelDoc(withTexture, feedback), { catalog: channelCatalog(), target: 'glsl' });

test('needs.channel is reported only when a channel-sampling node is reachable', () => {
  const withChannel = chanBuild(true);
  assert.equal(withChannel.needs.channel, true, 'a reachable channel node must set the flag');
  // The evidence in the source is the channel symbol, not the alias: the rewriter renames
  // `tex2D_` to each target's own sampling call, so `tex2D_` is exactly what will *not* be there.
  assert.ok(
    withChannel.source.includes('iChannel0'),
    'the emitted GLSL must declare and read the image channel',
  );

  const without = chanBuild(false);
  assert.equal(without.needs.channel, false, 'nothing samples the channel here');
});

/* -------------------------------------------------------------------------- */
/* 10c. a wire into a port the body never reads                                */
/* -------------------------------------------------------------------------- */

/** A catalog with one node that declares an input its body ignores. */
function inertCatalog(): Record<NodeKind, NodeSchema> {
  const schemas: NodeSchema[] = [
    {
      kind: 'const_color' as NodeKind,
      title: 'Color',
      category: 'generator',
      summary: 'A constant colour.',
      inputs: [port('color', 'vec4', { kind: 'color', value: [0.2, 0.4, 0.9, 1] })],
      outputs: [port('color', 'vec4')],
      primaryOut: 'color',
      glsl: 'return ${color};',
    },
    {
      // The shape of the real `shape` node: a coordinate it accepts "for symmetry" and never
      // uses. `sink` is the output, so a wire into `dead` cannot possibly do anything.
      kind: 'symmetry' as NodeKind,
      title: 'Symmetry',
      category: 'color',
      summary: 'Composites two colours.',
      inputs: [
        port('uv', 'vec2', { kind: 'vec2', value: [0.5, 0.5] }),
        port('mask', 'float', f(0.5)),
        port('color', 'vec4', { kind: 'color', value: [1, 1, 1, 1] }),
      ],
      outputs: [port('value', 'vec4')],
      primaryOut: 'value',
      glsl: 'return mix_(${color}, vec4(1.0, 0.0, 0.0, 1.0), clamp(${mask}, 0.0, 1.0));',
    },
    {
      // The escape hatch, declared the way the real catalog declares it: slots a snippet *may*
      // use. A body that reads only `a` is the normal case, which is why this kind is exempt from
      // the inert-input check — and the exemption is tested rather than assumed.
      kind: 'custom_glsl' as NodeKind,
      title: 'Custom GLSL',
      category: 'math',
      summary: 'Per-instance body.',
      inputs: [
        port('uv', 'vec2', { kind: 'vec2', value: [0.5, 0.5] }),
        port('a', 'vec4', { kind: 'color', value: [1, 1, 1, 1] }),
        port('b', 'vec4', { kind: 'color', value: [1, 1, 1, 1] }),
      ],
      outputs: [port('value', 'vec4')],
      primaryOut: 'value',
      glsl: 'return ${a};',
    },
    {
      kind: 'output' as NodeKind,
      title: 'Output',
      category: 'output',
      summary: 'Final colour.',
      inputs: [port('color', 'vec4')],
      outputs: [port('color', 'vec4')],
      primaryOut: 'color',
      glsl: 'return ${color};',
    },
  ];
  const catalog: Partial<Record<NodeKind, NodeSchema>> = {};
  for (const s of schemas) catalog[s.kind] = s;
  return catalog as Record<NodeKind, NodeSchema>;
}

/** `wired` decides whether the inert `uv` port gets a wire. */
function inertDoc(wired: boolean): ShaderDoc {
  const nodes: ShaderNode[] = [
    { id: 'n_base', kind: 'const_color' as NodeKind },
    {
      id: 'n_sym',
      kind: 'symmetry' as NodeKind,
      inputs: {
        mask: { node: 'n_base', port: 'color' },
        ...(wired ? { uv: { node: 'n_base', port: 'color' } } : {}),
      },
    },
    { id: 'n_out', kind: 'output' as NodeKind, inputs: { color: { node: 'n_sym', port: 'value' } } },
  ];
  return {
    irVersion: '1.0.0',
    id: 'doc_inert',
    title: 'Inert fixture',
    spec: { intent: 'inert', category: 'test', tags: ['test'], features: [] },
    stage: 'fragment',
    graph: { nodes, edges: normalizeEdges({ nodes, edges: [] }).edges },
    render: { resolution: [512, 512], qualityTier: 'medium', maxIterations: 8, transparent: false },
    targets: ['glsl'],
  };
}

test('a wire into a port the body never reads is reported, not ignored', () => {
  /*
   * `shape.uv` is the shipped example: accepted "for pipeline symmetry", documented as ignored,
   * and completely silent when wired. From the user's side that is indistinguishable from a bug —
   * connect it, see nothing, and have no way to find out why. The check is derived from the body,
   * so a *future* node with a dead port is covered the moment it ships, and a body that stops
   * reading a port it still declares starts warning immediately.
   */
  const wired = emit(inertDoc(true), { catalog: inertCatalog(), target: 'glsl' });
  const found = wired.diagnostics.filter((d) => d.code === 'IR037_INERT_INPUT');
  assert.equal(found.length, 1, `expected one IR037, got ${JSON.stringify(wired.diagnostics.map((d) => d.code))}`);
  assert.equal(found[0]!.at?.node, 'n_sym', 'the diagnostic must name the node');
  assert.equal(found[0]!.at?.port, 'uv', 'and the port');
  assert.ok(found[0]!.fix, 'and carry an actionable fix');

  // Not wired: nothing to say. A warning that fires unconditionally is noise, and noise is how a
  // real finding gets ignored.
  const clean = emit(inertDoc(false), { catalog: inertCatalog(), target: 'glsl' });
  assert.deepEqual(
    clean.diagnostics.filter((d) => d.code === 'IR037_INERT_INPUT'),
    [],
  );

  // `custom_glsl` is exempt: its body is per instance, so a snippet that uses only some of its
  // slots is the normal case, not a defect.
  const customNodes: ShaderNode[] = [
    { id: 'n_base', kind: 'const_color' as NodeKind },
    {
      id: 'n_custom',
      kind: 'custom_glsl' as NodeKind,
      body: 'return ${a};',
      inputs: {
        a: { node: 'n_base', port: 'color' },
        // Wired but unused by the snippet above — and that must stay quiet.
        b: { node: 'n_base', port: 'color' },
      },
    },
    { id: 'n_out', kind: 'output' as NodeKind, inputs: { color: { node: 'n_custom', port: 'value' } } },
  ];
  const customDoc: ShaderDoc = {
    ...inertDoc(false),
    graph: { nodes: customNodes, edges: normalizeEdges({ nodes: customNodes, edges: [] }).edges },
  };
  const customEmit = emit(customDoc, { catalog: inertCatalog(), target: 'glsl' });
  assert.deepEqual(
    customEmit.diagnostics.filter((d) => d.code === 'IR037_INERT_INPUT'),
    [],
    'a partial custom_glsl snippet must not be warned about',
  );
});

test('render.feedback without a channel read is a warning, not silence', () => {  /*
   * The flag is a request to the *host*: set up an offscreen target and feed the previous frame
   * back in. If the document never samples the channel, that request cannot change a single pixel
   * — the host does real work, allocates two textures and a framebuffer, adds a pass per frame,
   * and the picture is identical. From the outside that is indistinguishable from "feedback is
   * broken", which is exactly why it has to be a diagnostic.
   */
  const flagged = chanBuild(false, true);
  const warnings = flagged.diagnostics.filter((d) => d.code === 'IR036_FEEDBACK_UNUSED');
  assert.equal(warnings.length, 1, `expected exactly one IR036, got ${JSON.stringify(flagged.diagnostics)}`);
  assert.equal(warnings[0]!.severity, 'warning', 'the document is still emittable');
  assert.match(warnings[0]!.fix ?? '', /texture/, 'the fix must name the node that reads the channel');

  // With a channel read, the flag is meaningful and must stay quiet.
  const ok = chanBuild(true, true);
  assert.deepEqual(ok.diagnostics.filter((d) => d.severity === 'error'), []);
  assert.equal(
    ok.diagnostics.filter((d) => d.code === 'IR036_FEEDBACK_UNUSED').length,
    0,
    'a document that reads the channel must not be warned about',
  );

  // And absent the flag, nothing is said at all: binding a stand-in is a legitimate host choice.
  assert.equal(chanBuild(true, undefined).diagnostics.filter((d) => d.code === 'IR036_FEEDBACK_UNUSED').length, 0);
  assert.equal(chanBuild(true, false).diagnostics.filter((d) => d.code === 'IR036_FEEDBACK_UNUSED').length, 0);
});

test('every surface-capable target emits a surface entry, and every other refuses', () => {  /*
   * One test, driven by the gate itself.
   *
   * Hardcoding "glsl is the only one" would go stale the moment a target gained its entry
   * point — and, worse, it would go stale *silently in the direction of over-claiming*: the
   * loop below would keep asserting a refusal that no longer happens. Reading `SURFACE_CAPABLE`
   * means converting a target automatically moves it from the refusal side to the emission
   * side, and the per-target assertions below are what prove the conversion is real.
   */
  for (const t of IMPLEMENTED_TARGETS) {
    const e = surfBuild(true, t);
    const codes = e.diagnostics.map((d) => d.code);
    const errorCodes = e.diagnostics.filter((d) => d.severity === 'error').map((d) => d.code);

    if (SURFACE_CAPABLE.has(t)) {
      assert.deepEqual(errorCodes, [], `${t}: a surface document must emit cleanly, saw ${errorCodes.join(', ')}`);
      assert.equal(e.needs.surface, true, `${t}: a reachable surface node must set the flag`);
      assert.ok(e.source.length > 500, `${t}: suspiciously short surface emission`);
      // Every capable target has to declare *something* for the normal it reads. The exact
      // spelling differs per language, so this checks the contract rather than a string.
      assert.ok(
        /normal/i.test(e.source),
        `${t}: a surface shader that never mentions a normal cannot be reading one`,
      );
    } else {
      assert.ok(
        codes.includes('EM002_SURFACE_UNSUPPORTED'),
        `${t}: a surface document must be refused, not emitted; saw ${codes.join(', ') || '(no diagnostics)'}`,
      );
      assert.equal(e.source, '', `${t}: a refused target must not return source`);
    }

    // Either way the quad form must still emit cleanly, or the refusal would be hiding a
    // broader breakage.
    const quad = surfBuild(false, t);
    assert.deepEqual(
      quad.diagnostics.filter((d) => d.severity === 'error').map((d) => d.code),
      [],
      `${t}: the quad form must still emit cleanly`,
    );
  }
});

test('the HLSL shim families are complete in *both* argument positions', () => {
  /*
   * The bug this pins, and why the check belongs here rather than at a call site.
   *
   * GLSL declares `mix(e1, e2, e3)` for a scalar **or vector** `e3`, and `mod(x, y)` for a
   * scalar or vector `y`. The HLSL shims declared only the scalar forms of both, so a vector
   * argument silently selected the scalar overload and was **truncated to its first
   * component** — HLSL's `lerp` had accepted either, so the shim simply had to say so.
   *
   * The cost was not theoretical. The shared library's `srgb2linear` reads
   *
   *     mix_(lo, hi, step(vec3(0.04045), c))
   *
   * which in GLSL converts each channel by its own threshold and in HLSL converted all three
   * by the red channel's. A colour shift on two of the six targets, from a library function
   * every document emits.
   *
   * Asserting this over *call sites* is not possible without a type checker — the failing
   * argument was itself a call (`step(...)`), so a regex cannot know its type. Asserting it
   * over the *shim family* is exact, cheap, and catches the omission where it happens. The
   * compiler remains the authority for call-site coverage, and
   * `python/tests/test_emitted_artifacts.py` now asserts `dxc` reports zero warnings — which
   * it had been reporting all along, into an assertion that only counted errors.
   */
  const widths = ['2', '3', '4'];
  for (const [name, shims] of [
    ['HLSLFAM_SHIMS', HLSLFAM_SHIMS],
    ['MSL_SHIMS', MSL_SHIMS],
  ] as const) {
    // `wf_mod`: the divisor (second argument) may be a vector.
    for (const w of widths) {
      assert.ok(
        new RegExp(`float${w}\\s+wf_mod\\s*\\(\\s*float${w}\\s+\\w+\\s*,\\s*float${w}\\s`).test(shims),
        `${name}: wf_mod has no (float${w}, float${w}) overload, so a vector divisor truncates`,
      );
      assert.ok(
        new RegExp(`float${w}\\s+wf_mod\\s*\\(\\s*float${w}\\s+\\w+\\s*,\\s*float\\s`).test(shims),
        `${name}: wf_mod has no (float${w}, float) overload`,
      );
    }
    // `wf_mix`: the factor (third argument) may be a vector — HLSL only.
    if (name === 'HLSLFAM_SHIMS') {
      for (const w of widths) {
        assert.ok(
          new RegExp(`float${w}\\s+wf_mix\\s*\\(\\s*float${w}\\s+\\w+\\s*,\\s*float${w}\\s+\\w+\\s*,\\s*float${w}\\s`).test(
            shims,
          ),
          `${name}: wf_mix has no vector-factor (float${w}, float${w}, float${w}) overload, ` +
            `so a per-component mix truncates to t.x — which is what broke srgb2linear`,
        );
        assert.ok(
          new RegExp(`float${w}\\s+wf_mix\\s*\\(\\s*float${w}\\s+\\w+\\s*,\\s*float${w}\\s+\\w+\\s*,\\s*float\\s`).test(
            shims,
          ),
          `${name}: wf_mix has no scalar-factor (float${w}, float${w}, float) overload`,
        );
      }
    }
  }
  // MSL deliberately keeps Metal's native `mix`, which already takes either factor arity —
  // so the absence of `wf_mix` there is correct, not an omission. Asserted so that "MSL has no
  // wf_mix" cannot quietly become "MSL lost its mix coverage".
  assert.ok(!/wf_mix/.test(MSL_SHIMS), 'MSL must keep the native `mix`, which is not scalar-only');
});

test('the GLSL surface shader carries the whole contract', () => {
  // The per-language checks above are necessarily loose; GLSL is the one target the workbench
  // actually runs, so its surface contract is pinned exactly.
  const glsl = surfBuild(true, 'glsl');
  assert.ok(glsl.source.includes('in vec3 vNormal;'), 'the fragment stage must declare the normal input');
  assert.ok(glsl.source.includes('in vec3 vWorldPos;'), 'and the world position');
  assert.ok(glsl.source.includes('uniform vec3 iCameraPos;'), 'and the camera uniform');
  assert.ok(glsl.source.includes('uniform vec3 iLightDir;'), 'and the light uniform');
  assert.ok(glsl.source.includes('vec3 iNormal = normalize(vNormal);'), 'the body local must be defined');
  assert.ok(glsl.source.includes('vec3 iViewDir = normalize(iCameraPos - vWorldPos);'), 'and the view vector');
  // There is no screen position on a mesh, so a surface shader must not read one.
  assert.ok(!glsl.source.includes('gl_FragCoord'), 'a surface shader has no gl_FragCoord to read');
  assert.ok(glsl.source.includes('vec2 uv = vUv;'), 'uv comes from the mesh, not the screen');
});

test('WGSL surface output survives the structural checker', () => {
  // `checkWgslStructural` is the only mechanical proof this target has, since there is no
  // naga/tint here. A second entry point and a vertex struct are exactly the kind of addition
  // that could slip past it, so it is run on the surface emission explicitly.
  const wgsl = surfBuild(true, 'wgsl');
  assert.deepEqual(checkWgslStructural(wgsl.source), [], 'the WGSL surface emission must be structurally sound');
  assert.ok(wgsl.source.includes('@vertex'), 'a surface document needs a vertex stage');
  assert.ok(wgsl.source.includes('fn vs_main('), 'named for the host to bind');
  assert.ok(wgsl.source.includes('@fragment'), 'and the fragment stage it feeds');
  assert.ok(wgsl.source.includes('worldViewProj: mat4x4f'), 'the transform block must be declared');
  // The quad form must NOT grow a vertex stage: a fullscreen triangle is generated by the
  // host, and emitting one here would be a second, competing definition of the same thing.
  const quad = surfBuild(false, 'wgsl');
  assert.ok(!quad.source.includes('@vertex'), 'a quad document must stay fragment-only');
});

/* -------------------------------------------------------------------------- */
/* 11. the escape hatch                                                        */
/* -------------------------------------------------------------------------- */

/**
 * A catalog with only `custom_glsl` in it.
 *
 * The point of the escape hatch is that the code is *per instance*, so a fixture that used the
 * real catalog would not distinguish "the node's body was used" from "the schema's body was
 * used". This one gives the schema an obvious placeholder and the node a different body, so
 * every assertion below is about which of the two reached the emission.
 */
function customCatalog(): Record<NodeKind, NodeSchema> {
  const schemas: NodeSchema[] = [
    {
      kind: 'const_color' as NodeKind,
      title: 'Color',
      category: 'generator',
      summary: 'A constant colour.',
      inputs: [port('color', 'vec4', { kind: 'color', value: [0.1, 0.2, 0.3, 1] })],
      outputs: [port('color', 'vec4')],
      primaryOut: 'color',
      glsl: 'return ${color};',
    },
    {
      kind: 'custom_glsl' as NodeKind,
      title: 'Custom GLSL',
      category: 'math',
      summary:
        'Escape hatch. The schema body is a pass-through placeholder; the node carries its own. ' +
        'The remaining inputs do not affect the result until a body is written.',
      inputs: [
        port('uv', 'vec2', { kind: 'vec2', value: [0.5, 0.5] }),
        port('a', 'vec4', { kind: 'color', value: [0, 0, 0, 1] }),
        port('b', 'vec4', { kind: 'color', value: [1, 1, 1, 1] }),
        port('c', 'vec4', { kind: 'color', value: [0.5, 0.5, 0.5, 1] }),
        port('d', 'vec4', { kind: 'color', value: [0, 1, 0, 1] }),
      ],
      outputs: [port('value', 'vec4')],
      primaryOut: 'value',
      glsl: 'return ${a};',
    },
    {
      kind: 'output' as NodeKind,
      title: 'Output',
      category: 'output',
      summary: 'Final colour.',
      inputs: [port('color', 'vec4')],
      outputs: [port('color', 'vec4')],
      primaryOut: 'color',
      glsl: 'return ${color};',
    },
  ];
  const catalog: Partial<Record<NodeKind, NodeSchema>> = {};
  for (const s of schemas) catalog[s.kind] = s;
  return catalog as Record<NodeKind, NodeSchema>;
}

/** A graph whose `custom_glsl` node carries `body` (or none, when `body` is undefined). */
function customDoc(body: string | undefined): ShaderDoc {
  const nodes: ShaderNode[] = [
    { id: 'n_src', kind: 'const_color' as NodeKind },
    { id: 'n_custom', kind: 'custom_glsl' as NodeKind, inputs: { a: { node: 'n_src', port: 'color' } }, ...(body !== undefined ? { body } : {}) },
    { id: 'n_out', kind: 'output' as NodeKind, inputs: { color: { node: 'n_custom', port: 'value' } } },
  ];
  const edges = [
    { from: { node: 'n_src', port: 'color' }, to: { node: 'n_custom', port: 'a' } },
    { from: { node: 'n_custom', port: 'value' }, to: { node: 'n_out', port: 'color' } },
  ];
  return {
    irVersion: '1.0.0',
    id: 'doc_custom',
    title: 'Custom fixture',
    spec: { intent: 'custom', category: 'test', tags: ['test'], features: [] },
    stage: 'fragment',
    graph: { nodes, edges: normalizeEdges({ nodes, edges }).edges },
    render: { resolution: [512, 512], qualityTier: 'medium', maxIterations: 8, transparent: false },
    targets: ['glsl'],
  };
}

/** A snippet using the alias vocabulary the rewriter is responsible for. */
const SNIPPET = 'float k_${node_id} = mod_(${a}.x * 6.0, 2.0);\nreturn vec4(${a}.xyz * k_${node_id}, 1.0);';

test('a per-instance body is emitted on every target, and lowered like a catalog body', () => {
  for (const t of IMPLEMENTED_TARGETS) {
    const e = emit(customDoc(SNIPPET), { catalog: customCatalog(), target: t });
    const errs = e.diagnostics.filter((d) => d.severity === 'error');
    assert.deepEqual(errs.map((d) => d.code), [], `${t}: the escape hatch must emit cleanly`);
    // The *node's* body is what reached the emission, not the schema's `return ${a};`
    // placeholder. Without this the whole feature could be inert and every other assertion
    // here would still pass.
    assert.ok(e.source.includes('6.0'), `${t}: the node body did not reach the emission`);
  }

  // And the alias really was rewritten per target rather than passed through: `mod_` is GLSL's
  // `mod` (floor semantics) on GLSL, and the floor-form `wf_mod` shim where the language's own
  // `fmod` truncates. That rename is the entire reason a snippet can be written once.
  //
  // The check runs on code without comments, because the emitted source *keeps* the shared
  // library's prose — and that prose explains the aliases by name. An assertion that fails on
  // documentation is an assertion somebody eventually deletes.
  const glsl = emit(customDoc(SNIPPET), { catalog: customCatalog(), target: 'glsl' });
  assert.ok(
    !/mod_|wf_mod/.test(withoutComments(glsl.source)),
    'GLSL must use its own `mod`, not an alias or a shim',
  );
  const hlsl = emit(customDoc(SNIPPET), { catalog: customCatalog(), target: 'hlsl' });
  assert.ok(hlsl.source.includes('wf_mod'), 'HLSL must use the floor-form shim, not fmod');
  const wgsl = emit(customDoc(SNIPPET), { catalog: customCatalog(), target: 'wgsl' });
  assert.deepEqual(checkWgslStructural(wgsl.source), [], 'a custom body must survive the WGSL structural checker');
});

test('a custom_glsl node with no body is an error, not a silent pass-through', () => {
  /*
   * The fallback in the schema (`return ${a}`) exists so that a half-built graph still emits —
   * "nothing happened" rather than a syntax error. But a pass-through reported as success is
   * precisely the false-success this project keeps hunting, so the state is an error with a
   * fix. Without this check a user could ask for an effect, get their input colour back
   * unchanged, and be told the run was clean.
   */
  const e = emit(customDoc(undefined), { catalog: customCatalog(), target: 'glsl' });
  const codes = e.diagnostics.map((d) => d.code);
  assert.ok(codes.includes('IR034_CUSTOM_BODY_EMPTY'), `expected IR034, saw ${codes.join(', ') || '(none)'}`);
  const d = e.diagnostics.find((x) => x.code === 'IR034_CUSTOM_BODY_EMPTY');
  assert.equal(d?.at?.node, 'n_custom', 'the diagnostic must name the node');
  assert.ok(d?.fix, 'and carry a fix');
  // It still emits something — the error is a report, not a refusal to produce a document.
  assert.ok(e.source.length > 500);
});

test('a banned construct in a custom body is reported against its node', () => {
  // `mix(` with the builtin spelling is the mistake a model makes most often here, and the
  // alias exists because `mix` is the one name that needs a different lowering per target.
  const bad = emit(customDoc('return mix(${a}, ${b}, 0.5);'), { catalog: customCatalog(), target: 'glsl' });
  const codes = bad.diagnostics.map((d) => d.code);
  assert.ok(
    codes.some((c) => c.startsWith('RW0')),
    `a banned alias must be reported, saw ${codes.join(', ') || '(none)'}`,
  );
  // Attributed to the node, so a reader can tell which snippet produced it when the graph has
  // more than one.
  assert.ok(
    bad.diagnostics.some((x) => x.at?.node === 'n_custom'),
    'the lint diagnostic must name the node that carries the body',
  );
});

test('a body on a kind that does not use one is reported', () => {
  // `body` is `custom_glsl`'s mechanism. A body on another kind silently replaces a catalog
  // template — which is the rewriter's proof obligation — so it is at least worth saying.
  const doc = customDoc(undefined);
  doc.graph.nodes = doc.graph.nodes.map((n) => (n.id === 'n_src' ? { ...n, body: 'return ${color};' } : n));
  const e = emit(doc, { catalog: customCatalog(), target: 'glsl' });
  assert.ok(
    e.diagnostics.some((d) => d.code === 'IR035_STRAY_BODY'),
    'a body on a non-custom kind must be reported',
  );
});

test('removing the last surface node returns the emission to the quad form exactly', () => {
  // The flag is computed over the *reachable* set, so a surface node that is not wired to the
  // output must not change anything. Without this, adding and deleting a `normal` node would
  // leave a document emitting a shader it cannot run on a quad — and the difference would be
  // invisible until someone noticed the preview had stopped working.
  const plain = surfBuild(false, 'glsl').source;
  const withOrphan = (() => {
    const doc = surfaceDoc(false);
    doc.graph.nodes.push({ id: 'n_orphan', kind: 'normal' as NodeKind });
    return emit(doc, { catalog: surfaceCatalog(), target: 'glsl' });
  })();
  assert.equal(withOrphan.needs.surface, false, 'an unreachable surface node must not set the flag');
  assert.equal(withOrphan.source, plain, 'an orphan must not perturb a single byte of the output');
});
