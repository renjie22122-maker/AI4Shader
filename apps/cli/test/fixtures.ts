/**
 * Test fixtures — a self-contained catalog + emitter.
 *
 * `packages/ir/src/catalog/` and `.../targets/` are authored by other agents and must not
 * be imported here (they may not exist yet, and their content is not this module's
 * contract). Every test that needs a catalog builds one inline, exactly as a unit test
 * should: a handful of node kinds, hand-written schemas, and a stub emitter registered
 * through the public `registerTarget` hook.
 *
 * Kept in one module so the patcher tests and the CLI tests exercise the *same* graph.
 */

import type { NodeKind, NodeSchema, ShaderDoc, TargetFormat } from '../../../packages/ir/src/types.js';
import { createDoc, wire } from '../../../packages/ir/src/patcher.js';
import { registerTarget } from '../../../packages/ir/src/emitter.js';

/* -------------------------------------------------------------------------- */
/* Nodes                                                                       */
/* -------------------------------------------------------------------------- */

function schema(
  kind: NodeKind,
  category: NodeSchema['category'],
  inputs: NodeSchema['inputs'],
  outputs: NodeSchema['outputs'],
  glsl: string,
  primaryOut: string,
): NodeSchema {
  return { kind, title: kind, category, summary: `stub ${kind}`, inputs, outputs, primaryOut, glsl };
}

/**
 * A catalog containing *only* the kinds these fixtures use. `uv`, `noise`, `fbm` and
 * `output` are real `NODE_KINDS`, so the port/type shapes under test are the real shapes;
 * the bodies are trivial because no test here compiles a shader for real.
 */
export const STUB_CATALOG: Partial<Record<NodeKind, NodeSchema>> = {
  uv: schema('uv', 'generator', [], [{ name: 'uv', type: 'vec2' }], 'return vec2(0.0, 0.0);', 'uv'),

  noise: schema(
    'noise',
    'noise',
    [{ name: 'uv', type: 'vec2' }],
    [{ name: 'value', type: 'float' }],
    'return uv.x + uv.y;',
    'value',
  ),

  fbm: schema(
    'fbm',
    'noise',
    [
      { name: 'uv', type: 'vec2' },
      { name: 'scale', type: 'float' },
      { name: 'octaves', type: 'float' },
      { name: 'lacunarity', type: 'float' },
      { name: 'gain', type: 'float' },
      { name: 'ridged', type: 'float' },
    ],
    [{ name: 'value', type: 'float' }],
    'return uv.x * scale;',
    'value',
  ),

  output: schema(
    'output',
    'output',
    [{ name: 'color', type: 'vec4' }],
    [{ name: 'color', type: 'vec4' }],
    'return color;',
    'color',
  ),
};

/* -------------------------------------------------------------------------- */
/* Document                                                                    */
/* -------------------------------------------------------------------------- */

export type FixtureOptions = {
  title?: string;
  intent?: string;
  qualityTier?: ShaderDoc['render']['qualityTier'];
  maxIterations?: number;
};

/**
 * A small, *clean* document.
 *
 * Every node kind here exists in the real catalog (`packages/ir/src/catalog`) with exactly
 * these port names, so this document validates with zero error-severity diagnostics against
 * the real catalog once it lands — and degrades to a structural check before that. The one
 * deliberate wrinkle is the unreachable `noise_1` node, which produces exactly one
 * `IR011_DEAD_NODES` warning so the diagnostics path is always exercised.
 */
export function makeDoc(opts: FixtureOptions = {}): ShaderDoc {
  const base = createDoc({
    title: opts.title ?? 'Fixture noise shader',
    spec: {
      intent: opts.intent ?? 'a soft procedural noise field',
      category: 'noise',
      tags: ['procedural', 'noise'],
      features: ['continuous', 'deterministic'],
    },
    render: {
      resolution: [640, 360],
      qualityTier: opts.qualityTier ?? 'medium',
      maxIterations: opts.maxIterations ?? 64,
      transparent: false,
    },
    targets: ['glsl', 'wgsl', 'hlsl', 'msl', 'shaderlab', 'ue'],
  });

  // Real kinds, real port names: `uv.uv -> fbm.uv`, `fbm.value -> output.color`.
  const graph = wire(
    {
      nodes: [
        { id: 'uv_1', kind: 'uv' },
        {
          id: 'fbm_1',
          kind: 'fbm',
          params: {
            // Three uniforms, not two: `tune` has to be able to exceed its 256-variant cap to
            // exercise the refusal path (3 axes x 7 steps = 343), and a two-uniform fixture
            // cannot get there without a step count that makes the arithmetic opaque.
            scale: { kind: 'param', name: 'u_scale', type: 'float', value: 4, min: 0.1, max: 64 },
            octaves: { kind: 'param', name: 'u_octaves', type: 'float', value: 5, min: 1, max: 8 },
            gain: { kind: 'param', name: 'u_gain', type: 'float', value: 0.5, min: 0, max: 1 },
          },
        },
        { id: 'noise_1', kind: 'noise' },
        { id: 'out_1', kind: 'output' },
      ],
      edges: [],
    },
    [
      ['uv_1', 'fbm_1', 'uv', 'uv'],
      ['fbm_1', 'out_1', 'color', 'value'],
    ],
  );

  return { ...base, graph };
}

/** A document whose only error-level problem is a dangling wire. */
export function makeBrokenDoc(): ShaderDoc {
  const base = makeDoc({ title: 'Fixture broken shader' });
  return {
    ...base,
    graph: {
      nodes: base.graph.nodes.map((n) =>
        n.id === 'out_1' ? { ...n, inputs: { color: { node: 'ghost_1', port: 'value' } } } : n,
      ),
      edges: base.graph.edges,
    },
  };
}

/**
 * A document every possible catalog can check: one `output` node with a literal and no
 * wires at all.
 *
 * The tests that must run with the real catalog *absent* use this, because the CLI's
 * fallback structural catalog knows exactly one kind — `output`.
 */
export function makeStructuralDoc(): ShaderDoc {
  return createDoc({
    title: 'Structural only',
    graph: {
      nodes: [{ id: 'out_1', kind: 'output', params: { color: { kind: 'color', value: [0, 0, 0, 1] } } }],
      edges: [],
    },
  });
}

/* -------------------------------------------------------------------------- */
/* Stub target                                                                 */
/* -------------------------------------------------------------------------- */

/**
 * A complete stand-in for a real target emitter: it proves the CLI's plumbing
 * (registration, sidecars, per-target file names, README assembly, overwrite policy)
 * without depending on the real `targets/` emitters.
 *
 * Registered for `wgsl` by default so a test can exercise the stub *and* the real GLSL
 * emitter in the same run without one overwriting the other in `TARGET_EMITTERS`.
 */
export function registerStubTarget(target: TargetFormat = 'wgsl'): void {
  registerTarget(target, ({ doc, bodies, uniforms, analysis }) => ({
    source:
      [
        `// stub ${target} emitter — ${doc.title}`,
        `// doc hash ${doc.hash ?? '(unhashed)'}`,
        `// order ${analysis.order.join(' ')}`,
        ...bodies.lines,
        '// end',
      ].join('\n') + '\n',
    sidecars: [{ name: `${target}.meta`, content: `{"target":"${target}","uniforms":${uniforms.length}}\n` }],
    diagnostics: [],
  }));
}
