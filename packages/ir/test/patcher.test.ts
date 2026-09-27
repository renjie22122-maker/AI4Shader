/**
 * `patcher.ts` — unit tests.
 *
 * These run against an inline stub catalog (four node kinds) registered directly, never
 * against `packages/ir/src/catalog/`, which another agent owns and which does not exist
 * yet. That is the point: the patcher's contract is *purity, atomicity and determinism*,
 * none of which depends on the real node set.
 *
 * Run with:  node --experimental-strip-types --test packages/ir/test/patcher.test.ts
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import type {
  Diagnostic,
  NodeSchema,
  ParamValue,
  ShaderDoc,
  ShaderGraph,
  ShaderNode,
  ShaderPatch,
} from '../src/types.js';
import { analyzeGraph, emit, type Emission, type TargetEmitter } from '../src/emitter.js';
import {
  addNode,
  applyPatch,
  canonicalJson,
  connect,
  createDoc,
  disconnect,
  exposeParam,
  forkDoc,
  hashDoc,
  inlineParam,
  makeNodeId,
  normalizeEdges,
  removeNode,
  setParam,
  wire,
  PT,
  type Catalog,
} from '../src/patcher.js';

/* -------------------------------------------------------------------------- */
/* Fixture: an inline catalog and a matching stub emitter                      */
/* -------------------------------------------------------------------------- */

function schema(
  kind: ShaderNode['kind'],
  inputs: NodeSchema['inputs'],
  outputs: NodeSchema['outputs'],
  primaryOut: string,
  body: string,
  category: NodeSchema['category'] = 'math',
): NodeSchema {
  return { kind, title: kind, category, summary: kind, inputs, outputs, primaryOut, glsl: body };
}

const CATALOG: Partial<Record<ShaderNode['kind'], NodeSchema>> = {
  uv: schema('uv', [], [{ name: 'uv', type: 'vec2' }], 'uv', 'return vec2(0.5, 0.5);', 'generator'),
  const_float: schema('const_float', [{ name: 'value', type: 'float' }], [{ name: 'out', type: 'float' }], 'out', 'return ${value};', 'generator'),
  const_color: schema('const_color', [{ name: 'color', type: 'vec4' }], [{ name: 'out', type: 'vec4' }], 'out', 'return ${color};', 'generator'),
  math: schema('math', [{ name: 'a', type: 'float' }, { name: 'b', type: 'float' }], [{ name: 'out', type: 'float' }], 'out', 'return ${a} + ${b};'),
  noise: schema('noise', [{ name: 'uv', type: 'vec2' }], [{ name: 'value', type: 'float' }], 'value', 'return ${uv}.x;', 'noise'),
  fbm: schema(
    'fbm',
    [
      { name: 'uv', type: 'vec2' },
      { name: 'scale', type: 'float' },
    ],
    [{ name: 'value', type: 'float' }],
    'value',
    'return ${uv}.x * ${scale};',
    'noise',
  ),
  mix: schema('mix', [{ name: 'a', type: 'vec4' }, { name: 'b', type: 'vec4' }, { name: 't', type: 'float' }], [{ name: 'out', type: 'vec4' }], 'out', 'return ${a};', 'color'),
  // The escape hatch. Its schema body is a pass-through placeholder; the node carries its own,
  // which is why `set_body` exists and why this stub has to declare the kind.
  custom_glsl: schema(
    'custom_glsl',
    [
      { name: 'uv', type: 'vec2' },
      { name: 'a', type: 'vec4' },
      { name: 'b', type: 'vec4' },
      { name: 'c', type: 'vec4' },
      { name: 'd', type: 'vec4' },
    ],
    [{ name: 'value', type: 'vec4' }],
    'value',
    'return ${a};',
    'math',
  ),
  output: schema('output', [{ name: 'color', type: 'vec4' }], [{ name: 'color', type: 'vec4' }], 'color', 'return ${color};', 'output'),
};

/** A stub target so `emit()` can run end-to-end without `targets/`. */
const STUB_GLSL: TargetEmitter = ({ doc, bodies }) => ({
  source: `// stub glsl ${doc.hash ?? ''}\n${bodies.lines.join('\n')}\n`,
  diagnostics: [],
});

/**
 * `emit` against the stub, **injected per call** rather than registered globally.
 *
 * `TARGET_EMITTERS` is a process-global table and `--test-isolation=none` shares one process
 * across every suite, so a module-scope `registerTarget('glsl', …)` here silently replaced the
 * real GLSL emitter for whatever ran next. Whether that mattered depended on which test files the
 * runner overlapped, so it was invisible until one more file joined the suite — and then sixteen
 * tests in `targets.test.ts` and `wgsl.test.ts` failed with "the quad entry must still read
 * gl_FragCoord", which points at the emitter and not at the test that broke it.
 *
 * `EmitOptions.emitter` exists so that no test (and no host) has to swap a global to do this.
 */
const emitStub = (doc: ShaderDoc, catalog: Record<NodeKind, NodeSchema>): Emission =>
  emit(doc, { catalog, target: 'glsl', emitter: STUB_GLSL });

const floatLit = (value: number, min?: number, max?: number): ParamValue => ({
  kind: 'float',
  value,
  ...(min !== undefined ? { min } : {}),
  ...(max !== undefined ? { max } : {}),
});

function chainDoc(): ShaderDoc {
  return createDoc({
    title: 'chain',
    graph: wire(
      {
        nodes: [
          { id: 'uv_1', kind: 'uv' },
          { id: 'n_1', kind: 'noise' },
          { id: 'fbm_1', kind: 'fbm', params: { scale: floatLit(4, 0, 10) } },
          { id: 'out_1', kind: 'output' },
        ],
        edges: [],
      },
      [
        ['uv_1', 'n_1', 'uv', 'uv'],
        ['uv_1', 'fbm_1', 'uv', 'uv'],
        ['fbm_1', 'out_1', 'color', 'value'],
      ],
    ),
  });
}

/** Recursively freeze — proves every mutator is pure, because a write would throw. */
function deepFreeze<T>(value: T): T {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.freeze(value);
    for (const key of Object.keys(value as object)) {
      deepFreeze((value as Record<string, unknown>)[key]);
    }
  }
  return value;
}

const codes = (ds: readonly Diagnostic[]): string[] => ds.map((d) => d.code);
const errors = (ds: readonly Diagnostic[]): Diagnostic[] => ds.filter((d) => d.severity === 'error');

/**
 * The `normalizeEdges` invariant, asserted in both directions:
 * every `inputs[port]` has exactly one matching edge, and every edge has a matching ref.
 */
function assertWireInvariant(graph: ShaderGraph, where: string): void {
  const byId = new Map(graph.nodes.map((n) => [n.id, n] as const));
  let refs = 0;
  for (const node of graph.nodes) {
    for (const [port, w] of Object.entries(node.inputs ?? {})) {
      refs++;
      const hits = graph.edges.filter((e) => e.to.node === node.id && e.to.port === port);
      assert.equal(hits.length, 1, `${where}: expected exactly 1 edge for ${node.id}.${port}, got ${hits.length}`);
      assert.equal(hits[0]!.from.node, w.node, `${where}: edge source node mismatch for ${node.id}.${port}`);
      assert.equal(hits[0]!.from.port, w.port ?? '', `${where}: edge source port mismatch for ${node.id}.${port}`);
      assert.ok(byId.has(w.node), `${where}: ${node.id}.${port} points at missing node ${w.node}`);
    }
  }
  assert.equal(graph.edges.length, refs, `${where}: edges (${graph.edges.length}) != input refs (${refs})`);
}

/* -------------------------------------------------------------------------- */
/* makeNodeId                                                                  */
/* -------------------------------------------------------------------------- */

test('makeNodeId: deterministic, collision-free, no randomness', () => {
  assert.equal(makeNodeId('noise', []), 'noise');
  assert.equal(makeNodeId('noise', ['uv', 'output']), 'noise');

  // Same inputs, same output — called 50 times.
  const many = new Set(Array.from({ length: 50 }, () => makeNodeId('fbm', ['fbm', 'fbm_2', 'fbm_3'])));
  assert.deepEqual([...many], ['fbm_4']);

  // Density of the suffix space.
  assert.equal(makeNodeId('fbm', ['fbm']), 'fbm_2');
  assert.equal(makeNodeId('fbm', ['fbm', 'fbm_2']), 'fbm_3');
  assert.equal(makeNodeId('fbm', ['fbm', 'fbm_2', 'fbm_3']), 'fbm_4');
  // Gaps are filled from the bottom, so ids stay dense across a long editing session.
  assert.equal(makeNodeId('fbm', ['fbm', 'fbm_3']), 'fbm_2');
  assert.equal(makeNodeId('fbm', new Set(['fbm'])), 'fbm_2');
});

/* -------------------------------------------------------------------------- */
/* addNode / removeNode / connect / disconnect                                 */
/* -------------------------------------------------------------------------- */

test('addNode: auto id, structural sharing, invariant holds, input untouched', () => {
  const doc = chainDoc();
  const before = canonicalJson(doc);

  const r = addNode(doc, { kind: 'noise' }, CATALOG);
  assert.deepEqual(errors(r.diagnostics), []);
  assert.equal(r.id, 'noise', 'first node of a kind takes the bare kind as its id');
  assert.equal(r.doc.graph.nodes.length, 5);
  assertWireInvariant(r.doc.graph, 'addNode');

  // Structural sharing: untouched nodes are the same objects, not copies.
  assert.equal(r.doc.graph.nodes[0], doc.graph.nodes[0]);
  assert.notEqual(r.doc, doc);
  assert.equal(canonicalJson(doc), before, 'input document was mutated');

  // A second node of the same kind collides deterministically.
  const r2 = addNode(r.doc, { kind: 'noise' }, CATALOG);
  assert.equal(r2.id, 'noise_2');

  // An explicit, free id is honored; a taken one is an error, not a silent rename.
  const r3 = addNode(r.doc, { kind: 'math', id: 'gain' }, CATALOG);
  assert.equal(r3.id, 'gain');
  const dup = addNode(r.doc, { kind: 'math', id: 'n_1' }, CATALOG);
  assert.equal(dup.doc, r.doc, 'rejected addNode returns the very same document object');
  assert.deepEqual(codes(dup.diagnostics), [PT.DUPLICATE_ID]);
  assert.ok(dup.diagnostics[0]!.fix?.includes('math'), `fix should suggest a free id, got: ${dup.diagnostics[0]!.fix}`);

  // Unknown kind, and a kind the catalog does not cover, are both reported.
  const bad = addNode(doc, { kind: 'nope' as ShaderNode['kind'] }, CATALOG);
  assert.deepEqual(codes(bad.diagnostics), [PT.UNKNOWN_KIND]);
  const uncovered = addNode(doc, { kind: 'voronoi' }, CATALOG);
  assert.deepEqual(codes(uncovered.diagnostics), [PT.UNKNOWN_KIND]);
});

test('removeNode: drops the node and every reference to it', () => {
  const doc = chainDoc();
  const before = canonicalJson(doc);

  const r = removeNode(doc, 'n_1');
  assert.deepEqual(errors(r.diagnostics), []);
  assert.deepEqual(r.doc.graph.nodes.map((n) => n.id), ['uv_1', 'fbm_1', 'out_1']);
  assert.equal(r.doc.graph.edges.length, 2, 'the uv_1 -> n_1 wire must be gone; the fbm chain must not be touched');
  // Other wires are untouched: only references to the removed node are dropped.
  // `wire()` records an omitted source port as absent, and `analyzeGraph` resolves it
  // against the source kind's `primaryOut` at emit time.
  assert.deepEqual(r.doc.graph.nodes.find((n) => n.id === 'out_1')!.inputs, { color: { node: 'fbm_1', port: 'value' } });
  assertWireInvariant(r.doc.graph, 'removeNode');
  assert.equal(canonicalJson(doc), before);

  const missing = removeNode(doc, 'ghost');
  assert.equal(missing.doc, doc);
  assert.deepEqual(codes(missing.diagnostics), [PT.UNKNOWN_NODE]);

  // Removing the output node leaves the generators without a sink.
  const r2 = removeNode(doc, 'out_1');
  assert.deepEqual(r2.doc.graph.nodes.map((n) => n.id), ['uv_1', 'n_1', 'fbm_1']);
  assertWireInvariant(r2.doc.graph, 'removeNode(output)');
});

test('connect / disconnect: wires stay consistent and validate every rule', () => {
  const doc = createDoc({
    graph: {
      nodes: [
        { id: 'uv_1', kind: 'uv' },
        { id: 'n_1', kind: 'noise' },
        { id: 'out_1', kind: 'output' },
      ],
      edges: [],
    },
  });
  const before = canonicalJson(doc);

  // --- happy path: the primary output port is resolved and pinned into the edge ---
  const c1 = connect(doc, { node: 'uv_1' }, { node: 'n_1', port: 'uv' }, CATALOG);
  assert.deepEqual(errors(c1.diagnostics), []);
  assert.deepEqual(c1.doc.graph.edges, [{ from: { node: 'uv_1', port: 'uv' }, to: { node: 'n_1', port: 'uv' } }]);
  assertWireInvariant(c1.doc.graph, 'connect');

  // --- float -> vec4 is a legal widening on the wire -------------------------------
  const down = addNode(c1.doc, { kind: 'const_float', params: { value: floatLit(1) } }, CATALOG);
  const widened = connect(down.doc, { node: down.id }, { node: 'out_1', port: 'color' }, CATALOG);
  assert.deepEqual(errors(widened.diagnostics), [], 'float must be splattable into a vec4 input');

  // --- vec4 -> float is NOT: this is the type-mismatch rejection ---------------
  const color = addNode(doc, { kind: 'const_color', params: { color: { kind: 'color', value: [1, 0, 0, 1] } } }, CATALOG);
  const mismatch = connect(color.doc, { node: color.id }, { node: 'n_1', port: 'uv' }, CATALOG);
  assert.deepEqual(codes(mismatch.diagnostics), [PT.TYPE_MISMATCH]);
  assert.equal(mismatch.doc, color.doc, 'a rejected connect returns the input doc');
  assert.ok(
    mismatch.diagnostics[0]!.fix && /vec_op|swizzle/i.test(mismatch.diagnostics[0]!.fix),
    `fix must be actionable, got: ${mismatch.diagnostics[0]!.fix}`,
  );
  // ... but vec4 -> vec4 into the output port is fine.
  assert.deepEqual(errors(connect(color.doc, { node: color.id }, { node: 'out_1', port: 'color' }, CATALOG).diagnostics), []);

  // --- unknown node / port / output -------------------------------------------
  assert.deepEqual(codes(connect(doc, { node: 'ghost' }, { node: 'n_1', port: 'uv' }, CATALOG).diagnostics), [PT.DANGLING_WIRE]);
  assert.deepEqual(codes(connect(doc, { node: 'uv_1' }, { node: 'ghost', port: 'uv' }, CATALOG).diagnostics), [PT.UNKNOWN_NODE]);
  assert.deepEqual(codes(connect(doc, { node: 'uv_1' }, { node: 'n_1', port: 'nope' }, CATALOG).diagnostics), [PT.UNKNOWN_PORT]);
  assert.deepEqual(codes(connect(doc, { node: 'uv_1', port: 'nope' }, { node: 'n_1', port: 'uv' }, CATALOG).diagnostics), [PT.UNKNOWN_OUTPUT]);

  // --- cycles: the IR is a DAG ------------------------------------------------
  const cyc = connect(c1.doc, { node: 'n_1' }, { node: 'uv_1', port: 'uv' }, CATALOG);
  assert.deepEqual(codes(cyc.diagnostics), [PT.UNKNOWN_PORT], 'uv has no inputs, so a cycle cannot even be expressed on it');
  const selfLoop = connect(c1.doc, { node: 'n_1' }, { node: 'n_1', port: 'uv' }, CATALOG);
  assert.deepEqual(codes(selfLoop.diagnostics), [PT.CYCLE]);
  assert.equal(selfLoop.doc, c1.doc);

  // --- reconnecting an occupied input replaces the wire (editor semantics) ----
  const back = addNode(c1.doc, { kind: 'uv', id: 'uv_2' }, CATALOG);
  const rewired = connect(back.doc, { node: 'uv_2' }, { node: 'n_1', port: 'uv' }, CATALOG);
  assert.deepEqual(errors(rewired.diagnostics), []);
  assert.equal(rewired.doc.graph.edges.filter((e) => e.to.node === 'n_1').length, 1, 'an input holds at most one wire');
  assert.equal(rewired.doc.graph.edges[0]!.from.node, 'uv_2');
  assertWireInvariant(rewired.doc.graph, 'reconnect');

  // --- disconnect -------------------------------------------------------------
  const d1 = disconnect(rewired.doc, { node: 'n_1', port: 'uv' });
  assert.equal(d1.doc.graph.edges.length, 0);
  assert.equal(d1.doc.graph.nodes.find((n) => n.id === 'n_1')!.inputs, undefined);
  assertWireInvariant(d1.doc.graph, 'disconnect');
  // Idempotent: disconnecting an unwired port is a no-op, not an error.
  const d2 = disconnect(d1.doc, { node: 'n_1', port: 'uv' });
  assert.equal(d2.doc, d1.doc);
  assert.deepEqual(d2.diagnostics, []);
  assert.deepEqual(codes(disconnect(doc, { node: 'ghost', port: 'uv' }).diagnostics), [PT.UNKNOWN_NODE]);

  assert.equal(canonicalJson(doc), before, 'the original document was mutated by the wire helpers');
});

/* -------------------------------------------------------------------------- */
/* normalizeEdges / wire                                                       */
/* -------------------------------------------------------------------------- */

test('normalizeEdges: derived, deterministic, idempotent', () => {
  const nodes: ShaderNode[] = [
    { id: 'a', kind: 'noise', inputs: { uv: { node: 'b' } } },
    { id: 'b', kind: 'uv' },
    { id: 'c', kind: 'output', inputs: { color: { node: 'a', port: 'value' }, uv: { node: 'b', port: 'uv' } } },
  ];

  const once = normalizeEdges({ nodes, edges: [{ from: { node: 'stale', port: 'x' }, to: { node: 'stale', port: 'y' } }] });
  // The stale edge is gone: `inputs` is the authority.
  assert.deepEqual(once.edges, [
    { from: { node: 'b', port: '' }, to: { node: 'a', port: 'uv' } },
    { from: { node: 'a', port: 'value' }, to: { node: 'c', port: 'color' } },
    { from: { node: 'b', port: 'uv' }, to: { node: 'c', port: 'uv' } },
  ]);
  assert.deepEqual(normalizeEdges(once), once, 'normalizeEdges must be idempotent');

  // Key insertion order must not leak into the edge order: `c` declares `uv` before
  // `color` here, and its edges still come out sorted by port name.
  const reordered = normalizeEdges({
    nodes: [
      { id: 'c', kind: 'output', inputs: { uv: { node: 'b', port: 'uv' }, color: { node: 'a', port: 'value' } } },
      { id: 'b', kind: 'uv' },
      { id: 'a', kind: 'noise', inputs: { uv: { node: 'b' } } },
    ],
    edges: [],
  });
  assert.deepEqual(
    reordered.edges.map((e) => `${e.from.node}->${e.to.node}.${e.to.port}`),
    ['a->c.color', 'b->c.uv', 'b->a.uv'],
  );

  // A wire to a node that does not exist produces no edge (analyzeGraph reports it).
  assert.deepEqual(normalizeEdges({ nodes: [{ id: 'a', kind: 'noise', inputs: { uv: { node: 'ghost' } } }], edges: [] }).edges, []);
});

test('wire: pairs are last-write-wins and the edge list is always derived', () => {
  const g = wire({ nodes: [{ id: 'a', kind: 'uv' }, { id: 'b', kind: 'uv' }, { id: 'n', kind: 'noise' }], edges: [] }, [
    ['a', 'n', 'uv', 'uv'],
    ['b', 'n', 'uv'],
    ['ghost', 'n', 'uv'],
  ]);
  assert.equal(g.edges.length, 1);
  assert.deepEqual(g.edges[0], { from: { node: 'b', port: '' }, to: { node: 'n', port: 'uv' } });
  assert.deepEqual(g.nodes.find((x) => x.id === 'n')!.inputs, { uv: { node: 'b' } }, 'no explicit port when none was given');
  assertWireInvariant(g, 'wire');

  // A pair naming a node that is not in the graph is not a wire. It must not be written:
  // a dangling reference here would only surface as IR004 two stages later.
  assert.deepEqual(wire({ nodes: [{ id: 'n', kind: 'noise' }], edges: [] }, [['ghost', 'n', 'uv']]).edges, []);
  // ... and a pair naming a missing *destination* is equally inert.
  assert.deepEqual(wire({ nodes: [{ id: 'a', kind: 'uv' }], edges: [] }, [['a', 'ghost', 'uv']]).edges, []);
});

/* -------------------------------------------------------------------------- */
/* setParam                                                                    */
/* -------------------------------------------------------------------------- */

test('setParam: type-checks, refuses wired ports, is pure', () => {
  const doc = addNode(createDoc(), { kind: 'math' }, CATALOG);
  const before = canonicalJson(doc.doc);

  const ok = setParam(doc.doc, 'math', 'a', floatLit(0.5, 0, 1), CATALOG);
  assert.deepEqual(errors(ok.diagnostics), []);
  assert.deepEqual(ok.doc.graph.nodes[0]!.params, { a: floatLit(0.5, 0, 1) });
  assert.equal(canonicalJson(doc.doc), before);

  // Unknown node / unknown port.
  assert.deepEqual(codes(setParam(doc.doc, 'ghost', 'a', floatLit(1), CATALOG).diagnostics), [PT.UNKNOWN_NODE]);
  assert.deepEqual(codes(setParam(doc.doc, 'math', 'nope', floatLit(1), CATALOG).diagnostics), [PT.UNKNOWN_PORT]);

  // Literal that is the wrong type for the port: a vec4 literal on a float input.
  const bad = setParam(doc.doc, 'math', 'a', { kind: 'vec4', value: [0, 0, 0, 1] }, CATALOG);
  assert.deepEqual(codes(bad.diagnostics), [PT.TYPE_MISMATCH]);
  assert.equal(bad.doc, doc.doc);

  // Non-finite numbers are rejected: they would reach the emitter as `NaN`.
  assert.deepEqual(codes(setParam(doc.doc, 'math', 'a', floatLit(Number.NaN), CATALOG).diagnostics), [PT.BAD_RANGE]);

  // A `param` binding whose type cannot feed the port is a mismatch too.
  const wrongUniform = setParam(doc.doc, 'math', 'a', { kind: 'param', name: 'u_c', type: 'vec4', value: [0, 0, 0, 1] }, CATALOG);
  assert.deepEqual(codes(wrongUniform.diagnostics), [PT.TYPE_MISMATCH]);

  // A wired input cannot take a literal — the wire would silently win at emit time.
  // `doc` only knows `math`; add a `uv` generator and a `noise` to wire together.
  const uv = addNode(doc.doc, { kind: 'uv' }, CATALOG);
  const noise = addNode(uv.doc, { kind: 'noise' }, CATALOG);
  assert.deepEqual(errors(noise.diagnostics), []);
  const wired = connect(noise.doc, { node: 'uv' }, { node: 'noise', port: 'uv' }, CATALOG);
  assert.deepEqual(errors(wired.diagnostics), []);
  const clash = setParam(wired.doc, 'noise', 'uv', { kind: 'param', name: 'u_k', type: 'vec2', value: [1, 1] }, CATALOG);
  assert.deepEqual(codes(clash.diagnostics), [PT.WIRED_PORT]);
  assert.ok(clash.diagnostics[0]!.fix?.includes("op: 'disconnect'"), 'fix must name the op to send');
  // The document's own unwired node is still writable.
  const sibling = setParam(wired.doc, 'math', 'b', floatLit(2, 0, 8), CATALOG);
  assert.deepEqual(errors(sibling.diagnostics), []);
  assert.deepEqual(sibling.doc.graph.nodes.find((n) => n.id === 'math')!.params, { b: floatLit(2, 0, 8) });
  assert.equal(canonicalJson(doc.doc), before, 'none of the helpers mutated the input document');
});

/* -------------------------------------------------------------------------- */
/* applyPatch — atomicity, hashing, provenance                                  */
/* -------------------------------------------------------------------------- */

test('applyPatch is atomic: an invalid middle op rejects the whole patch', () => {
  const doc = chainDoc();
  const snapshot = structuredClone(doc);

  // Three ops, where the middle one is invalid. The first op is a *legal* change, so the
  // test proves it does not survive the rejection.
  const patch: ShaderPatch = {
    summary: 'three ops, the second is broken',
    author: { kind: 'llm', model: 'test-model', round: 3 },
    ops: [
      { op: 'disconnect', to: { node: 'fbm_1', port: 'uv' } },
      { op: 'set_param', node: 'ghost_1', param: 'uv', value: floatLit(1) },
      { op: 'set_spec', spec: { category: 'never-applied' } },
    ],
  };

  const r = applyPatch(doc, patch, CATALOG);
  assert.equal(r.ok, false);
  assert.equal(r.applied, 0, 'no op may be reported as applied');
  assert.deepEqual(r.doc, snapshot, 'the returned document must be the original, byte for byte');
  assert.deepEqual(r.doc, doc);

  const failures = errors(r.diagnostics);
  assert.deepEqual(failures.map((d) => d.code), [PT.UNKNOWN_NODE, PT.NOT_ATOMIC]);
  assert.ok(r.diagnostics.find((d) => d.code === PT.NOT_ATOMIC)!.message.includes('op 2'), 'the rejection must name the failing op');
  // Op 1 was a legal wire removal; it must not survive anywhere.
  assert.equal(r.doc.graph.edges.length, 3, 'op 1 was legal and must still have been rolled back');
  assert.equal(r.doc.spec.category, snapshot.spec.category, 'op 3 must not have run at all');
  assert.equal(r.doc.hash, doc.hash);
  assert.equal(r.doc.updatedAt, doc.updatedAt);
  assert.deepEqual(r.doc.provenance, doc.provenance, 'a rejected patch must not touch provenance');
});

test('applyPatch: a valid patch bumps the hash and appends exactly one provenance step', () => {
  const doc = chainDoc();
  const stepsBefore = doc.provenance?.steps?.length ?? 0;

  const patch: ShaderPatch = {
    summary: 'expose the noise scale',
    author: { kind: 'llm', model: 'deepseek-test', round: 7 },
    ops: [
      { op: 'set_param', node: 'fbm_1', param: 'scale', value: { kind: 'float', value: 6, min: 0, max: 10 } },
      { op: 'set_spec', spec: { intent: 'a swirling noise field' } },
      { op: 'set_render', path: 'qualityTier', value: 'high' },
    ],
  };

  const r = applyPatch(doc, patch, CATALOG);
  assert.equal(r.ok, true, JSON.stringify(errors(r.diagnostics)));
  assert.equal(r.applied, 3);
  assert.notEqual(r.doc.hash, doc.hash);
  assert.equal(r.doc.provenance!.steps!.length, stepsBefore + 1);
  const step = r.doc.provenance!.steps!.at(-1)!;
  assert.equal(step.summary, 'expose the noise scale');
  assert.equal(step.actor, 'llm:deepseek-test:round 7');
  assert.ok(!Number.isNaN(Date.parse(step.at)), 'step.at must be an ISO timestamp');
  assert.equal(r.doc.spec.intent, 'a swirling noise field');
  assert.equal(r.doc.render.qualityTier, 'high');
  assert.equal(r.doc.updatedAt, step.at);

  // The hash is a function of content, not of the patch: applying the *same* content
  // change through a different route must land on the same hash.
  const viaOps = applyPatch(doc, { summary: 'one op', ops: [{ op: 'set_render', path: 'qualityTier', value: 'high' }] }, CATALOG);
  const viaOps2 = applyPatch(doc, { summary: 'one op again', ops: [{ op: 'set_render', path: 'qualityTier', value: 'high' }] }, CATALOG);
  assert.equal(viaOps.doc.hash, viaOps2.doc.hash, 'summary/author must not affect the hash');
  assert.notEqual(viaOps.doc.hash, doc.hash);

  // An empty patch is a successful no-op.
  const noop = applyPatch(doc, { summary: 'nothing', ops: [] }, CATALOG);
  assert.equal(noop.ok, true);
  assert.equal(noop.applied, 0);
  assert.equal(noop.doc, doc, 'a no-op must return the very same object');
});

test('applyPatch: normalizeEdges runs last, so edges and inputs cannot drift', () => {
  const doc = chainDoc();
  // Corrupt the edge list deliberately: a stale edge plus a missing one.
  const dirty: ShaderDoc = {
    ...doc,
    graph: { nodes: doc.graph.nodes, edges: [{ from: { node: 'n_1', port: 'value' }, to: { node: 'out_1', port: 'bogus' } }] },
  };

  const r = applyPatch(dirty, { summary: 'touch the spec', ops: [{ op: 'set_spec', spec: { category: 'water' } }] }, CATALOG);
  assert.equal(r.ok, true, JSON.stringify(errors(r.diagnostics)));
  assertWireInvariant(r.doc.graph, 'applyPatch');
  assert.deepEqual(r.doc.graph.edges, normalizeEdges(r.doc.graph).edges);
  assert.equal(canonicalJson(dirty.graph.edges).includes('bogus'), true, 'the input must not be repaired in place');

  // A wiring op lands in `inputs` and the derived edges follow within the same call.
  const wired = applyPatch(
    createDoc({ graph: { nodes: [{ id: 'uv_1', kind: 'uv' }, { id: 'n_1', kind: 'noise' }], edges: [] } }),
    { summary: 'wire it', ops: [{ op: 'connect', from: { node: 'uv_1' }, to: { node: 'n_1', port: 'uv' } }] },
    CATALOG,
  );
  assert.equal(wired.ok, true);
  assertWireInvariant(wired.doc.graph, 'applyPatch(connect)');
});

test('applyPatch: runs analyzeGraph once, so graph-level problems surface', () => {
  const doc = createDoc({ graph: { nodes: [{ id: 'uv_1', kind: 'uv' }, { id: 'n_1', kind: 'noise' }], edges: [] } });

  const noOutput = applyPatch(doc, { summary: 'rename', ops: [{ op: 'set_spec', spec: { category: 'x' } }] }, CATALOG);
  assert.equal(noOutput.ok, true, 'a missing output is a document problem, not a patch failure');
  assert.ok(codes(noOutput.diagnostics).includes('IR008_NO_OUTPUT'), `expected IR008, got ${codes(noOutput.diagnostics).join(',')}`);

  // A patch that removes the output node reports the consequence immediately.
  const removed = applyPatch(chainDoc(), { summary: 'drop the output', ops: [{ op: 'remove_node', node: 'out_1' }] }, CATALOG);
  assert.equal(removed.ok, true, 'the ops were all legal; the *document* is what became invalid');
  assert.ok(codes(removed.diagnostics).includes('IR008_NO_OUTPUT'), `expected IR008, got ${codes(removed.diagnostics).join(',')}`);
  assert.equal(removed.applied, 1);

  // A fully valid patch of a fully valid graph reports nothing at all.
  const clean = applyPatch(
    applyPatch(chainDoc(), { summary: 'noop', ops: [] }, CATALOG).doc,
    { summary: 'spec only', ops: [{ op: 'set_spec', spec: { category: 'noise' } }] },
    CATALOG,
  );
  assert.deepEqual(errors(clean.diagnostics), []);
  assert.equal(clean.applied, 1);

  // A `set_render` op that writes the value the document already has has changed
  // nothing, and `applied` says so rather than counting ops seen.
  const same = applyPatch(chainDoc(), { summary: 'same resolution', ops: [{ op: 'set_render', path: 'resolution', value: [1280, 720] }] }, CATALOG);
  assert.equal(same.ok, true);
  assert.equal(same.applied, 0, 'the default resolution is already [1280, 720]');
  const diff = applyPatch(chainDoc(), { summary: 'new resolution', ops: [{ op: 'set_render', path: 'resolution', value: [800, 600] }] }, CATALOG);
  assert.equal(diff.applied, 1);
  assert.deepEqual(diff.doc.render.resolution, [800, 600]);
  // The array written into the doc is a copy, so mutating the op afterwards is inert.
  assert.notEqual(diff.doc.render.resolution, chainDoc().render.resolution);
});

test('set_body: the escape hatch, and the three ways it refuses', () => {
  /*
   * `custom_glsl` is the one kind whose body is per-instance, and `set_body` is the only way
   * to write one. The refusals matter as much as the success: `body` replaces a catalog
   * template, and a catalog template is what the token rewriter's correctness is proven
   * against — so letting a patch rewrite one would hand the six-target guarantee to whatever
   * the model wrote that round.
   */
  const doc = createDoc({
    graph: {
      nodes: [
        { id: 'src', kind: 'uv' },
        { id: 'cust', kind: 'custom_glsl' },
        { id: 'out', kind: 'output' },
      ],
      edges: [],
    },
  });

  const ok = applyPatch(
    doc,
    { summary: 'write the body', ops: [{ op: 'set_body', node: 'cust', body: 'return vec4(${uv}, 0.0, 1.0);' }] },
    CATALOG,
  );
  assert.equal(ok.ok, true, JSON.stringify(errors(ok.diagnostics)));
  assert.equal(ok.doc.graph.nodes.find((n) => n.id === 'cust')?.body, 'return vec4(${uv}, 0.0, 1.0);');
  // Provenance is stamped once by `applyPatch`, not by the op handler — checked because the
  // handler for this op returns an edited graph without a hash, and getting that wrong would
  // leave a document whose `hash` field disagrees with its contents.
  assert.notEqual(ok.doc.hash, doc.hash);

  const refusals: [PatchOp, string, string][] = [
    [
      { op: 'set_body', node: 'ghost', body: 'return vec4(1.0);' },
      PT.UNKNOWN_NODE,
      'a node that does not exist',
    ],
    [
      { op: 'set_body', node: 'src', body: 'return vec4(1.0);' },
      PT.BAD_OP,
      'a kind that does not carry a body',
    ],
    [{ op: 'set_body', node: 'cust', body: '   ' }, PT.BAD_OP, 'an empty body'],
  ];
  for (const [op, code, what] of refusals) {
    const r = applyPatch(doc, { summary: 'x', ops: [op] }, CATALOG);
    assert.equal(r.ok, false, `${what} must be refused`);
    assert.ok(
      r.diagnostics.some((d) => d.code === code),
      `${what}: expected ${code}, saw ${r.diagnostics.map((d) => d.code).join(', ')}`,
    );
    // Atomic: the document is untouched, not partially edited.
    assert.equal(r.doc.graph.nodes.find((n) => n.id === 'cust')?.body, undefined);
  }

  // A body the restricted subset rejects is refused at patch time, not at scene-compile time
  // in whichever of the six targets happens to disagree first.
  const banned = applyPatch(
    doc,
    { summary: 'x', ops: [{ op: 'set_body', node: 'cust', body: 'return mix(${a}, ${b}, 0.5);' }] },
    CATALOG,
  );
  assert.equal(banned.ok, false, 'a banned builtin spelling must be refused');
  assert.ok(
    banned.diagnostics.some((d) => d.code === 'RW012_BUILTIN_SPELLING'),
    `expected the alias diagnostic, saw ${banned.diagnostics.map((d) => d.code).join(', ')}`,
  );

  /*
   * And the same treatment for a construct that is valid in five targets and invalid in one.
   *
   * `vec4(0.0)` is a scalar splat: fine in GLSL, WGSL and MSL, and a hard error in HLSL
   * ("too few elements in vector initialization"). It is refused here rather than at the far end
   * of a Unity import, and the fix names the exact replacement so a model can act on it. This is
   * the escape hatch's real risk in one line — arbitrary text in a six-language pipeline — and
   * the reason `lintSubset` runs on every body whether it came from the catalog or a patch.
   */
  const splat = applyPatch(
    doc,
    { summary: 'x', ops: [{ op: 'set_body', node: 'cust', body: 'return vec4(0.0);' }] },
    CATALOG,
  );
  assert.equal(splat.ok, false, 'a scalar-splat vector constructor must be refused');
  const splatDiag = splat.diagnostics.find((d) => d.code === 'RW013_SCALAR_SPLAT_CONSTRUCTOR');
  assert.ok(splatDiag, `expected RW013, saw ${splat.diagnostics.map((d) => d.code).join(', ')}`);
  assert.match(splatDiag.fix ?? '', /vec4\(0\.0, 0\.0, 0\.0, 0\.0\)/, 'the fix must be the exact literal');

  // The long form passes, and so does a copy constructor — the rule is about a bare *number*, not
  // about one-argument constructors in general (typed analysis would reject valid source).
  for (const body of ['return vec4(0.0, 0.0, 0.0, 1.0);', 'return vec4(${a});']) {
    const ok = applyPatch(doc, { summary: 'x', ops: [{ op: 'set_body', node: 'cust', body }] }, CATALOG);
    assert.equal(ok.ok, true, `${body} must be accepted, saw ${ok.diagnostics.map((d) => d.code).join(', ')}`);
  }

  /*
   * A matrix product written with `*` is refused too, and this one is not a style rule: HLSL
   * rejects `matrix * vector` outright, so the snippet would compile on four targets and fail on
   * two. `mat2_mul_` is the alias that lowers correctly everywhere — vector-first on HLSL, because
   * its row-major constructors make the naive `mul(matrix, vector)` a transpose.
   */
  const aliasProduct = applyPatch(
    doc,
    { summary: 'x', ops: [{ op: 'set_body', node: 'cust', body: 'return mat2_mul_(rot2(0.5), vec2(${a}.x, ${a}.y));' }] },
    CATALOG,
  );
  assert.equal(
    aliasProduct.ok,
    true,
    `the alias form must be accepted, saw ${aliasProduct.diagnostics.map((d) => d.code).join(', ')}`,
  );

  const rawProduct = applyPatch(
    doc,
    { summary: 'x', ops: [{ op: 'set_body', node: 'cust', body: 'return rot2(0.5) * vec2(${a}.x, ${a}.y);' }] },
    CATALOG,
  );
  assert.equal(rawProduct.ok, false, 'a raw matrix product must be refused');
  const rawDiag = rawProduct.diagnostics.find((d) => d.code === 'RW014_MATRIX_PRODUCT');
  assert.ok(rawDiag, `expected RW014, saw ${rawProduct.diagnostics.map((d) => d.code).join(', ')}`);
  assert.match(rawDiag.fix ?? '', /mat2_mul_/, 'the fix must name the alias');

  // A local declared as `mat2` is caught as well — that is the other shape the mistake takes.
  const localMatrix = applyPatch(
    doc,
    {
      summary: 'x',
      ops: [
        {
          op: 'set_body',
          node: 'cust',
          body: 'mat2 m = rot2(0.5); return vec4(m * vec2(${a}.x, ${a}.y), 0.0, 1.0);',
        },
      ],
    },
    CATALOG,
  );
  assert.equal(localMatrix.ok, false, 'a matrix local used with `*` must be refused');
  assert.ok(localMatrix.diagnostics.some((d) => d.code === 'RW014_MATRIX_PRODUCT'));

  // And a component-wise product of two ordinary vectors is untouched: the rule is about
  // matrices, and a linter that fired on `vec2 * vec2` would be turned off within a day.
  const componentwise = applyPatch(
    doc,
    { summary: 'x', ops: [{ op: 'set_body', node: 'cust', body: 'return vec4(${a}.xy * ${b}.xy, 0.0, 1.0);' }] },
    CATALOG,
  );
  assert.equal(
    componentwise.ok,
    true,
    `a component-wise product must be accepted, saw ${componentwise.diagnostics.map((d) => d.code).join(', ')}`,
  );
});

test('applyPatch: every op kind round-trips, and malformed ops are rejected', () => {
  const doc = createDoc({
    graph: {
      nodes: [
        { id: 'uv_1', kind: 'uv' },
        { id: 'n_1', kind: 'noise' },
        { id: 'out_1', kind: 'output' },
      ],
      edges: [],
    },
  });

  const full: ShaderPatch = {
    summary: 'build a small graph',
    author: { kind: 'optimizer', round: 1 },
    ops: [
      { op: 'add_node', node: { id: 'gain_1', kind: 'math', params: { a: { kind: 'float', value: 2, min: 0, max: 4 } } } },
      { op: 'connect', from: { node: 'uv_1' }, to: { node: 'n_1', port: 'uv' } },
      { op: 'expose_param', node: 'gain_1', param: 'a', name: 'u_gain', min: 0, max: 4, label: 'Gain', group: 'Math' },
      { op: 'set_param', node: 'gain_1', param: 'a', value: { kind: 'param', name: 'u_gain', type: 'float', value: 1.5, min: 0, max: 4 } },
      { op: 'inline_param', node: 'gain_1', param: 'a' },
      { op: 'set_render', path: 'resolution', value: [800, 600] },
      { op: 'set_render', path: 'maxIterations', value: 96 },
      { op: 'set_spec', spec: { features: ['one'], tags: ['two'] } },
      { op: 'disconnect', to: { node: 'n_1', port: 'uv' } },
      { op: 'remove_node', node: 'gain_1' },
    ],
  };

  const r = applyPatch(doc, full, CATALOG);
  assert.equal(r.ok, true, JSON.stringify(errors(r.diagnostics)));
  assert.equal(r.applied, 10);
  assert.deepEqual(r.doc.graph.nodes.map((n) => n.id), ['uv_1', 'n_1', 'out_1']);
  assert.deepEqual(r.doc.render.resolution, [800, 600]);
  assert.deepEqual(r.doc.spec.features, ['one']);

  // Malformed payloads.
  const badOps: [PatchOp, string][] = [
    [{ op: 'set_render', path: 'maxIterations', value: -1 }, PT.BAD_OP],
    [{ op: 'set_render', path: 'maxIterations', value: 1.5 }, PT.BAD_OP],
    [{ op: 'set_render', path: 'qualityTier', value: 'extreme' }, PT.BAD_OP],
    [{ op: 'set_render', path: 'resolution', value: [0, 10] }, PT.BAD_OP],
    [{ op: 'set_spec', spec: { features: 'not an array' as unknown as string[] } }, PT.BAD_OP],
    [{ op: 'wat' as unknown as PatchOp['op'], node: 'x' } as unknown as PatchOp, PT.BAD_OP],
  ];
  for (const [op, expected] of badOps) {
    const res = applyPatch(doc, { summary: 'bad', ops: [op] }, CATALOG);
    assert.equal(res.ok, false, `${JSON.stringify(op)} should have been rejected`);
    assert.ok(codes(res.diagnostics).includes(expected), `${JSON.stringify(op)} -> ${codes(res.diagnostics).join(',')}`);
    assert.equal(res.doc, doc);
  }

  // A patch that is not a patch at all.
  const notAPatch = applyPatch(doc, undefined as unknown as ShaderPatch, CATALOG);
  assert.equal(notAPatch.ok, false);
  assert.equal(notAPatch.doc, doc);
});

test('applyPatch: a chain of patches accumulates provenance in order', () => {
  let doc = chainDoc();
  for (let i = 1; i <= 3; i++) {
    const r = applyPatch(
      doc,
      { summary: `step ${i}`, author: { kind: 'human' }, ops: [{ op: 'set_spec', spec: { category: `c${i}` } }] },
      CATALOG,
    );
    assert.equal(r.ok, true);
    doc = r.doc;
  }
  assert.deepEqual(doc.provenance!.steps!.map((s) => s.summary), ['step 1', 'step 2', 'step 3']);
  assert.deepEqual(doc.provenance!.steps!.map((s) => s.actor), ['human', 'human', 'human']);
});

/* -------------------------------------------------------------------------- */
/* hashDoc / canonicalJson                                                     */
/* -------------------------------------------------------------------------- */

test('hashDoc: stable under key insertion order, sensitive to content', () => {
  const a = chainDoc();
  const shuffled: ShaderDoc = {
    targets: [...a.targets],
    render: { ...a.render },
    stage: a.stage,
    graph: {
      edges: [...a.graph.edges],
      nodes: a.graph.nodes.map((n) => ({ kind: n.kind, params: n.params, inputs: n.inputs, id: n.id })),
    },
    spec: { features: [...a.spec.features], tags: [...a.spec.tags], category: a.spec.category, intent: a.spec.intent },
    title: a.title,
    id: a.id,
    irVersion: a.irVersion,
  };
  assert.notEqual(canonicalJson(a), canonicalJson(shuffled), 'the two constructions must really differ textually');
  assert.equal(hashDoc(a), hashDoc(shuffled), 'key order must not affect the hash');

  // Content changes must change the hash.
  const bumped = applyPatch(a, { summary: 'scale', ops: [{ op: 'set_param', node: 'fbm_1', param: 'scale', value: floatLit(9, 0, 10) }] }, CATALOG).doc;
  assert.notEqual(hashDoc(bumped), hashDoc(a));

  // A *vector* param is a content change too — this is the case a naive hash that
  // stringified arrays by reference would miss.
  const vecBumped = applyPatch(
    a,
    { summary: 'retarget', ops: [{ op: 'add_node', node: { id: 'c_1', kind: 'const_color', params: { color: { kind: 'color', value: [1, 0, 0, 1] } } } }] },
    CATALOG,
  ).doc;
  const vecBumped2 = applyPatch(
    vecBumped,
    { summary: 'recolor', ops: [{ op: 'set_param', node: 'c_1', param: 'color', value: { kind: 'color', value: [0, 1, 0, 1] } }] },
    CATALOG,
  ).doc;
  assert.notEqual(hashDoc(vecBumped2), hashDoc(vecBumped), 'a colour change must move the hash');
  assert.notEqual(hashDoc(vecBumped), hashDoc(a));

  const retargeted: ShaderDoc = { ...a, targets: ['wgsl', 'hlsl'] };
  assert.notEqual(hashDoc(retargeted), hashDoc(a), 'targets are part of codegen');

  // ... but identity, bookkeeping and editor layout are NOT part of codegen.
  assert.equal(hashDoc({ ...a, id: 'other-id' }), hashDoc(a));
  assert.equal(hashDoc({ ...a, title: 'a different title' }), hashDoc(a));
  assert.equal(hashDoc({ ...a, createdAt: '1999-01-01T00:00:00.000Z' }), hashDoc(a));
  assert.equal(hashDoc({ ...a, updatedAt: '1999-01-01T00:00:00.000Z' }), hashDoc(a));
  assert.equal(hashDoc({ ...a, provenance: { model: 'x', steps: [] } }), hashDoc(a));
  assert.equal(
    hashDoc({ ...a, graph: { ...a.graph, nodes: a.graph.nodes.map((n) => ({ ...n, pos: { x: 10, y: 20 } })) } }),
    hashDoc(a),
    'node position is presentational and must not invalidate a compile cache',
  );
  assert.equal(hashDoc({ ...a, hash: 'deadbeefdeadbeef' }), hashDoc(a), 'the hash field is not its own input');
});

test('canonicalJson: sorted keys, preserved array order, JSON-compatible', () => {
  assert.equal(canonicalJson({ b: 1, a: 2 }), '{"a":2,"b":1}');
  assert.equal(canonicalJson({ a: { d: 1, c: [3, 1, 2] } }), '{"a":{"c":[3,1,2],"d":1}}');
  assert.equal(canonicalJson([1, 'two', true, null]), '[1,"two",true,null]');
  assert.equal(canonicalJson({ u: undefined, a: 1 }), '{"a":1}', 'undefined values are dropped, not spelled');
  assert.equal(canonicalJson(-0), '0');
  assert.equal(canonicalJson(Number.POSITIVE_INFINITY), 'null', 'non-finite numbers must not produce invalid JSON');
  assert.doesNotThrow(() => JSON.parse(canonicalJson(chainDoc())));
  const cyc: Record<string, unknown> = {};
  cyc.self = cyc;
  assert.throws(() => canonicalJson(cyc), /circular/);
});

test('createDoc: every default the emitter reads is present and self-consistent', () => {
  const doc = createDoc();
  assert.equal(doc.irVersion, '1.0.0');
  assert.equal(doc.stage, 'fragment');
  assert.deepEqual(doc.graph, { nodes: [], edges: [] });
  assert.deepEqual(doc.spec, { intent: '', category: 'other', tags: [], features: [] });
  assert.equal(doc.render.qualityTier, 'medium');
  assert.ok(doc.render.maxIterations > 0);
  assert.equal(doc.targets.length > 0, true);
  assert.equal(doc.hash, hashDoc(doc));
  assert.equal(typeof doc.id, 'string');
  assert.notEqual(createDoc().id, createDoc().id, 'ids are unique even though they are not hashed');

  const partial = createDoc({ title: 't', spec: { intent: 'i' }, render: { qualityTier: 'ultra' } });
  assert.equal(partial.title, 't');
  assert.equal(partial.spec.intent, 'i');
  assert.equal(partial.spec.category, 'other', 'unspecified spec fields keep their defaults');
  assert.equal(partial.render.qualityTier, 'ultra');
  assert.equal(partial.render.maxIterations, doc.render.maxIterations);
});

test('forkDoc: content copy under a new title, prior provenance preserved', () => {
  const parent = applyPatch(chainDoc(), { summary: 'first', ops: [{ op: 'set_spec', spec: { category: 'water' } }] }, CATALOG).doc;
  const fork = forkDoc(parent, 'Fork — experiment A', { parents: [parent.hash!] });

  assert.equal(fork.title, 'Fork — experiment A');
  assert.notEqual(fork.id, parent.id);
  assert.equal(fork.hash, undefined, 'a fork is not hashed until it is patched or emitted');
  assert.equal(hashDoc(fork), hashDoc(parent), 'a fork that only renames itself still compiles to the same bytes');
  assert.deepEqual(fork.provenance!.steps, parent.provenance!.steps, 'the clone keeps the history it came from');
  assert.deepEqual(fork.provenance!.parents, [parent.hash!]);

  // Deep copy: editing the fork must not touch the original.
  const edited = applyPatch(fork, { summary: 'tweak', ops: [{ op: 'set_spec', spec: { category: 'fire' } }] }, CATALOG).doc;
  assert.equal(edited.spec.category, 'fire');
  assert.equal(parent.spec.category, 'water', 'the parent was mutated through the fork');
  assert.notEqual(edited.graph.nodes[0], parent.graph.nodes[0], 'fork must deep-copy the graph');

  // Forks can be forked; the lineage accumulates.
  const grand = forkDoc(edited, 'Fork B', { parents: [edited.hash!] });
  assert.deepEqual(grand.provenance!.parents, [parent.hash!, edited.hash!]);
});

/* -------------------------------------------------------------------------- */
/* exposeParam / inlineParam                                                   */
/* -------------------------------------------------------------------------- */

test('exposeParam then inlineParam round-trips the hash', () => {
  const base = addNode(createDoc(), { kind: 'const_float', params: { value: floatLit(0.25, 0, 1) } }, CATALOG);
  const doc = base.doc;

  const exposed = exposeParam(doc, 'const_float', 'value', { name: 'u_gain', min: 0, max: 1, step: 0.01, label: 'Gain', group: 'Tone' });
  assert.deepEqual(errors(exposed.diagnostics), []);
  assert.deepEqual(exposed.doc.graph.nodes[0]!.params!.value, {
    kind: 'param', name: 'u_gain', type: 'float', value: 0.25, min: 0, max: 1, step: 0.01, label: 'Gain', group: 'Tone',
  });
  // `updatedAt` and `provenance.steps` are intentionally *not* part of the hash, so a
  // fresh `hashDoc` (pure function of content) is the right comparison here.
  assert.notEqual(hashDoc(exposed.doc), hashDoc(doc), 'a uniform is codegen-relevant');

  const inlined = inlineParam(exposed.doc, 'const_float', 'value');
  assert.deepEqual(errors(inlined.diagnostics), []);
  assert.deepEqual(inlined.doc.graph.nodes[0]!.params!.value, { kind: 'float', value: 0.25, min: 0, max: 1, step: 0.01 });
  // The folded literal carries the uniform's slider metadata. `min`/`max` were on the
  // original literal, so they round-trip; `step` was introduced by the exposure and is
  // therefore additive. The *value* is what codegen reads, and it is identical.
  assert.notEqual(hashDoc(inlined.doc), hashDoc(doc), 'the folded literal legitimately gains the step hint');

  // ... and with no `step` supplied, the fold-back is byte-identical to the literal that
  // was exposed: this is the round-trip the tuning loop performs on every iteration.
  const plainExposed = exposeParam(doc, 'const_float', 'value', { name: 'u_gain', min: 0, max: 1 });
  const plainInlined = inlineParam(plainExposed.doc, 'const_float', 'value');
  assert.equal(
    canonicalJson(plainInlined.doc.graph),
    canonicalJson(doc.graph),
    'expose -> inline must restore the exact literal, slider bounds included',
  );
  assert.equal(hashDoc(plainInlined.doc), hashDoc(doc));

  // A 4-component literal: an RGBA vector is indistinguishable from a colour on the wire,
  // so the fold-back restores the general `vec4` form and the explicit `color` spelling is
  // not preserved. This is the one round-trip that is not byte-exact, and it is recorded
  // here deliberately rather than papered over: both emit identically (`paramExpr` sends
  // both through `literalOf('vec4', ...)`), only the literal's `kind` tag differs.
  const colored = addNode(createDoc(), { kind: 'const_color', params: { color: { kind: 'color', value: [0.1, 0.2, 0.3, 1] } } }, CATALOG).doc;
  const colorExposed = exposeParam(colored, 'const_color', 'color', { name: 'u_tint' });
  assert.deepEqual(errors(colorExposed.diagnostics), []);
  assert.deepEqual(colorExposed.doc.graph.nodes[0]!.params!.color, {
    kind: 'param', name: 'u_tint', type: 'vec4', value: [0.1, 0.2, 0.3, 1],
  });
  const colorInlined = inlineParam(colorExposed.doc, 'const_color', 'color');
  assert.deepEqual(errors(colorInlined.diagnostics), []);
  assert.deepEqual(colorInlined.doc.graph.nodes[0]!.params!.color, { kind: 'vec4', value: [0.1, 0.2, 0.3, 1] });
  // A document that already used the `vec4` spelling does round-trip exactly.
  const vecDoc = addNode(createDoc(), { kind: 'const_color', params: { color: { kind: 'vec4', value: [0.1, 0.2, 0.3, 1] } } }, CATALOG).doc;
  const vecExposed = exposeParam(vecDoc, 'const_color', 'color', { name: 'u_tint' }).doc;
  const vecInlined = inlineParam(vecExposed, 'const_color', 'color').doc;
  assert.deepEqual(vecInlined.graph.nodes[0]!.params!.color, { kind: 'vec4', value: [0.1, 0.2, 0.3, 1] });
  assert.equal(hashDoc(vecInlined), hashDoc(vecDoc));

  // The uniform's value and bounds survive the fold, so a slider added by a patch and then
  // removed still leaves the document in the state the slider described.
  const pushed = setParam(plainExposed.doc, 'const_float', 'value', { kind: 'param', name: 'u_gain', type: 'float', value: 0.75, min: 0, max: 1 }, CATALOG);
  const folded = inlineParam(pushed.doc, 'const_float', 'value');
  assert.deepEqual(errors(folded.diagnostics), []);
  assert.deepEqual(folded.doc.graph.nodes[0]!.params!.value, { kind: 'float', value: 0.75, min: 0, max: 1 });

  // A uniform clamped outside its own range folds back to a legal literal.
  const clamped = exposeParam(doc, 'const_float', 'value', { name: 'u_gain', min: 0, max: 1 });
  const outOfRange = setParam(clamped.doc, 'const_float', 'value', { kind: 'param', name: 'u_gain', type: 'float', value: 7, min: 0, max: 1 }, CATALOG);
  const clampedBack = inlineParam(outOfRange.doc, 'const_float', 'value');
  assert.deepEqual(clampedBack.doc.graph.nodes[0]!.params!.value, { kind: 'float', value: 1, min: 0, max: 1 });
  // ... and with no bounds on the uniform, the value passes through untouched.
  const free = exposeParam(doc, 'const_float', 'value', { name: 'u_free' });
  const freeSet = setParam(free.doc, 'const_float', 'value', { kind: 'param', name: 'u_free', type: 'float', value: 99 }, CATALOG);
  assert.deepEqual(inlineParam(freeSet.doc, 'const_float', 'value').doc.graph.nodes[0]!.params!.value, { kind: 'float', value: 99 });
});

test('exposeParam: rejects bad, reserved and conflicting names with a usable fix', () => {
  const doc = addNode(createDoc(), { kind: 'const_float', params: { value: floatLit(1) } }, CATALOG).doc;

  const bad = exposeParam(doc, 'const_float', 'value', { name: 'u-gain scale' });
  assert.deepEqual(codes(bad.diagnostics), [PT.BAD_NAME]);
  assert.ok(bad.diagnostics[0]!.fix?.includes('u_gain_scale'), `expected a safe suggestion, got ${bad.diagnostics[0]!.fix}`);

  const numeric = exposeParam(doc, 'const_float', 'value', { name: '2fast' });
  assert.deepEqual(codes(numeric.diagnostics), [PT.BAD_NAME]);
  assert.equal(numeric.doc, doc);

  const reserved = exposeParam(doc, 'const_float', 'value', { name: 'sample' });
  assert.deepEqual(codes(reserved.diagnostics), [PT.RESERVED_NAME]);
  assert.ok(reserved.diagnostics[0]!.fix?.includes('sample_v'), `expected sample_v, got ${reserved.diagnostics[0]!.fix}`);
  // Case-insensitive, because MSL/HLSL keywords are matched case-insensitively by the linter.
  assert.deepEqual(codes(exposeParam(doc, 'const_float', 'value', { name: 'Matrix' }).diagnostics), [PT.RESERVED_NAME]);

  // Same name, different type: this is the bug that would otherwise surface as a
  // linker error in exactly one target.
  const first = exposeParam(doc, 'const_float', 'value', { name: 'u_x' }).doc;
  const vecAdd = addNode(first, { kind: 'const_color', params: { color: { kind: 'color', value: [1, 1, 1, 1] } } }, CATALOG);
  const conflict = exposeParam(vecAdd.doc, 'const_color', 'color', { name: 'u_x' });
  assert.deepEqual(codes(conflict.diagnostics), [PT.NAME_TAKEN]);
  assert.ok(conflict.diagnostics[0]!.fix?.includes('u_x2'), `expected u_x2, got ${conflict.diagnostics[0]!.fix}`);
  assert.equal(conflict.doc, vecAdd.doc);

  // Same name, same type: allowed, but reported — a shared slider is usually a bug.
  const second = addNode(first, { kind: 'const_float', id: 'f2', params: { value: floatLit(2) } }, CATALOG);
  const shared = exposeParam(second.doc, 'f2', 'value', { name: 'u_x' });
  assert.deepEqual(errors(shared.diagnostics), [], 'sharing a uniform must not be an error');
  assert.deepEqual(codes(shared.diagnostics), [PT.NAME_TAKEN]);
  assert.equal(shared.diagnostics[0]!.severity, 'warning');
  assert.equal(shared.doc.graph.nodes.find((n) => n.id === 'f2')!.params!.value.kind, 'param');
});

test('exposeParam / inlineParam: refuse the states that have no meaning', () => {
  const doc = addNode(createDoc(), { kind: 'const_float', params: { value: floatLit(1) } }, CATALOG).doc;

  assert.deepEqual(codes(exposeParam(doc, 'ghost', 'value', { name: 'u_a' }).diagnostics), [PT.UNKNOWN_NODE]);
  assert.deepEqual(codes(exposeParam(doc, 'const_float', 'nope', { name: 'u_a' }).diagnostics), [PT.NO_VALUE]);
  assert.deepEqual(codes(inlineParam(doc, 'ghost', 'value').diagnostics), [PT.UNKNOWN_NODE]);
  assert.deepEqual(codes(inlineParam(doc, 'const_float', 'nope').diagnostics), [PT.NO_VALUE]);
  // Already a literal: nothing to inline.
  assert.deepEqual(codes(inlineParam(doc, 'const_float', 'value').diagnostics), [PT.NO_VALUE]);
  // Already a uniform: re-exposing is a `set_param`, not a rename.
  const exposed = exposeParam(doc, 'const_float', 'value', { name: 'u_a' }).doc;
  assert.deepEqual(codes(exposeParam(exposed, 'const_float', 'value', { name: 'u_b' }).diagnostics), [PT.NAME_TAKEN]);

  // A wired input has no literal, so there is nothing to expose (the disconnect op comes
  // first, exactly as the fix string says).
  const chain = chainDoc();
  const wiredPort = exposeParam(chain, 'n_1', 'uv', { name: 'u_a' });
  assert.deepEqual(codes(wiredPort.diagnostics), [PT.WIRED_PORT]);
  assert.ok(wiredPort.diagnostics[0]!.fix?.includes("op: 'disconnect'"));
  // ... and once disconnected *and given a literal*, the same call succeeds: a free port
  // with no value has nothing to bind, which is reported as NO_VALUE.
  const freed = disconnect(chain, { node: 'n_1', port: 'uv' }).doc;
  assert.deepEqual(codes(exposeParam(freed, 'n_1', 'uv', { name: 'u_a' }).diagnostics), [PT.NO_VALUE]);
  const seeded = setParam(freed, 'n_1', 'uv', { kind: 'vec2', value: [1, 2] }, CATALOG);
  assert.deepEqual(errors(seeded.diagnostics), []);
  assert.deepEqual(errors(exposeParam(seeded.doc, 'n_1', 'uv', { name: 'u_a' }).diagnostics), []);
});

/* -------------------------------------------------------------------------- */
/* Immutability                                                                */
/* -------------------------------------------------------------------------- */

test('every mutator is pure: a deeply frozen document survives all of them', () => {
  const frozen = deepFreeze(chainDoc());

  const added = addNode(frozen, { kind: 'noise' }, CATALOG);
  const removed = removeNode(added.doc, 'noise');
  const connected = connect(frozen, { node: 'uv_1' }, { node: 'n_1', port: 'uv' }, CATALOG);
  const disconnected = disconnect(connected.doc, { node: 'n_1', port: 'uv' });
  const param = setParam(frozen, 'fbm_1', 'scale', floatLit(2, 0, 10), CATALOG);
  const exposed = exposeParam(added.doc, 'noise', 'uv', { name: 'u_g' });
  // `noise.uv` arrived without a literal, so there is nothing to bind yet: seed one.
  const seeded = setParam(added.doc, 'noise', 'uv', { kind: 'vec2', value: [1, 2] }, CATALOG);
  assert.deepEqual(errors(seeded.diagnostics), []);
  const exposed2 = exposeParam(seeded.doc, 'noise', 'uv', { name: 'u_g' });
  const inlined = inlineParam(exposed2.doc, 'noise', 'uv');
  const normalized = normalizeEdges(frozen.graph);
  const wired = wire(frozen.graph, [['uv_1', 'n_1', 'uv', 'uv']]);
  const patched = applyPatch(
    frozen,
    {
      summary: 'everything at once',
      ops: [
        // `n_1.uv` is wired, so the patch has to free it before writing a literal there.
        { op: 'disconnect', to: { node: 'n_1', port: 'uv' } },
        { op: 'set_param', node: 'n_1', param: 'uv', value: { kind: 'param', name: 'u_k', type: 'vec2', value: [1, 1] } },
        { op: 'set_spec', spec: { tags: ['frozen'] } },
        { op: 'set_render', path: 'qualityTier', value: 'low' },
      ],
    },
    CATALOG,
  );
  const forked = forkDoc(frozen, 'Forked while frozen');

  for (const [name, result] of Object.entries({ added, removed, connected, disconnected, param, seeded, exposed: exposed2, inlined, patched })) {
    assert.deepEqual(errors(result.diagnostics), [], `${name} produced an unexpected error`);
  }
  assert.equal(patched.ok, true);
  assert.equal(Object.isFrozen(frozen.graph.nodes[0]), true, 'sanity: the input really is frozen');
  assert.equal(frozen.graph.nodes.find((n) => n.id === 'n_1')!.params, undefined, 'the frozen input was mutated');
  assert.equal(normalized.nodes, frozen.graph.nodes);
  assert.equal(wired.nodes.length, frozen.graph.nodes.length);
  // Re-declaring the wires of a frozen graph yields the same edge list, and reading the
  // frozen graph afterwards still finds its original edges (nothing was written through).
  assert.deepEqual(wired.edges, frozen.graph.edges);
  assert.equal(frozen.graph.edges.length, 3, 'the frozen input still has its original edges');
  assert.equal(forked.title, 'Forked while frozen');
});

test('ops never alias the caller: mutating a patch after it is applied cannot affect the doc', () => {
  const doc = chainDoc();
  const value: ParamValue = { kind: 'float', value: 1, min: 0, max: 10 };
  const patch: ShaderPatch = { summary: 'alias check', ops: [{ op: 'set_param', node: 'n_1', param: 'scale', value }] };
  assert.equal(patch.ops.length, 1);

  // A doc produced by `addNode` must own its params, not share the caller's object.
  const additive = addNode(doc, { kind: 'noise', params: { uv: { kind: 'param', name: 'u_b', type: 'vec2', value: [3, 4] } } }, CATALOG);
  const params = additive.doc.graph.nodes.at(-1)!.params!;
  (params.uv as Extract<ParamValue, { kind: 'param' }>).value = [9, 9];
  assert.deepEqual(additive.doc.graph.nodes.at(-1)!.params!.uv, { kind: 'param', name: 'u_b', type: 'vec2', value: [9, 9] });
  assert.deepEqual(value, { kind: 'float', value: 1, min: 0, max: 10 });

  const wireRef = { node: 'uv_1' };
  const c = connect(doc, wireRef, { node: 'n_1', port: 'uv' }, CATALOG);
  wireRef.node = 'mutated';
  assert.equal(c.doc.graph.edges[0]!.from.node, 'uv_1', 'connect must copy the WireRef it is given');
});

/* -------------------------------------------------------------------------- */
/* Integration: the patcher output is exactly what the emitter consumes         */
/* -------------------------------------------------------------------------- */

test('a patched document emits cleanly through the real emit() pipeline', () => {
  const r = applyPatch(
    createDoc(),
    {
      summary: 'build the whole graph',
      ops: [
        { op: 'add_node', node: { id: 'uv_1', kind: 'uv' } },
        { op: 'add_node', node: { id: 'n_1', kind: 'noise' } },
        { op: 'add_node', node: { id: 'fbm_1', kind: 'fbm' } },
        { op: 'add_node', node: { id: 'out_1', kind: 'output' } },
        { op: 'connect', from: { node: 'uv_1' }, to: { node: 'n_1', port: 'uv' } },
        { op: 'connect', from: { node: 'uv_1' }, to: { node: 'fbm_1', port: 'uv' } },
        { op: 'connect', from: { node: 'fbm_1', port: 'value' }, to: { node: 'out_1', port: 'color' } },
        // A literal, then promoted to a slider — the two-step the tuning loop uses.
        { op: 'set_param', node: 'fbm_1', param: 'scale', value: { kind: 'float', value: 3, min: 0, max: 8 } },
        { op: 'expose_param', node: 'fbm_1', param: 'scale', name: 'u_scale', min: 0, max: 8, label: 'Scale' },
      ],
    },
    CATALOG,
  );
  assert.equal(r.ok, true, JSON.stringify(errors(r.diagnostics)));
  assert.equal(r.applied, 9);

  const emission = emitStub(r.doc, CATALOG as Record<ShaderNode['kind'], NodeSchema>);
  assert.deepEqual(errors(emission.diagnostics), [], 'a patch that reports ok must emit without errors');
  assert.equal(emission.uniforms.length, 1);
  assert.equal(emission.uniforms[0]!.name, 'u_scale');
  assert.equal(emission.uniforms[0]!.type, 'float');
  assert.deepEqual(emission.uniforms[0]!.origin, { node: 'fbm_1', param: 'scale' });
  assert.ok(emission.source.length > 0);
  assert.equal(emission.hash.length, 16);

  // analyzeGraph (which applyPatch runs) and emit must agree on the diagnostic set for the
  // same document — otherwise the patch loop and the build loop disagree about whether a
  // document is good, and the agent optimises against the wrong signal.
  const analysis = analyzeGraph(r.doc.graph, CATALOG as Record<ShaderNode['kind'], NodeSchema>);
  assert.deepEqual(codes(analysis.diagnostics), codes(emission.diagnostics));
  assert.equal(hashDoc(r.doc), r.doc.hash, 'the stored hash must equal a fresh hash of the content');
  assert.equal(hashDoc(applyPatch(r.doc, { summary: 'noop', ops: [] }, CATALOG).doc), r.doc.hash, 'a no-op patch must not invalidate the cache key');
});

/* -------------------------------------------------------------------------- */
/* expose_param carries the full ParamValue shape                              */
/* -------------------------------------------------------------------------- */

/**
 * `expose_param` must be able to produce *exactly* the uniform a human would get by
 * exposing the same literal through the UI. The failure this guards against is invisible
 * in the generated shader — the source is identical either way — and shows up only as a
 * slider with the wrong granularity sitting in the wrong panel. That is the "the tool
 * seems broken" class of bug, and it is worth a test of its own.
 */
test('applyPatch forwards step and group from expose_param into the uniform', () => {
  const before = chainDoc();
  const patched = applyPatch(
    before,
    {
      summary: 'expose the fbm scale for tuning',
      ops: [
        {
          op: 'expose_param',
          node: 'fbm_1',
          param: 'scale',
          name: 'u_scale',
          min: 0.5,
          max: 24,
          step: 0.5,
          label: '纹理密度',
          group: '纹理',
        },
      ],
    },
    CATALOG,
  );

  assert.equal(patched.ok, true, 'the patch must apply');
  assert.deepEqual(errors(patched.diagnostics), []);

  const node = patched.doc.graph.nodes.find((n) => n.id === 'fbm_1')!;
  const value = node.params!.scale as Extract<ParamValue, { kind: 'param' }>;
  assert.equal(value.kind, 'param');
  assert.equal(value.name, 'u_scale');
  assert.equal(value.step, 0.5, 'step must survive the patch — it sizes the slider');
  assert.equal(value.group, '纹理', 'group must survive the patch — it places the slider');
  assert.equal(value.label, '纹理密度');
  assert.equal(value.min, 0.5);
  assert.equal(value.max, 24);

  // The round trip is an identity **when the exposure does not change the bounds**.
  // This matters more than it looks: without it, every tune-then-revert cycle would
  // produce a new hash and fork the version tree, so a user could never get back to
  // where they started. `chainDoc()` comes from `createDoc()`, which does not compute a
  // hash, so the baseline is a fresh `hashDoc` of the content.
  const baseline = hashDoc(before);
  const reverted = applyPatch(
    patched.doc,
    { summary: 'fold it back', ops: [{ op: 'inline_param', node: 'fbm_1', param: 'scale' }] },
    CATALOG,
  );
  assert.equal(reverted.ok, true);
  // The fixture literal is `floatLit(4, 0, 10)`, so re-exposing with the same window is
  // what a no-op exposure looks like.
  const tight = applyPatch(
    before,
    {
      summary: 'expose with the literal  bounds',
      ops: [{ op: 'expose_param', node: 'fbm_1', param: 'scale', name: 'u_s2', min: 0, max: 10 }],
    },
    CATALOG,
  );
  assert.equal(tight.ok, true);
  const back = applyPatch(
    tight.doc,
    { summary: 'fold it back', ops: [{ op: 'inline_param', node: 'fbm_1', param: 'scale' }] },
    CATALOG,
  );
  assert.equal(back.ok, true);
  assert.equal(
    hashDoc(back.doc),
    baseline,
    'expose_param → inline_param must be an identity on the content hash when bounds are unchanged',
  );

  // The one documented non-identity: `inline_param` folds the uniform's *own* bounds back
  // onto the literal, so widening a slider window is a real content change and correctly
  // earns a new hash. Asserting this pins the semantics rather than leaving them implied.
  assert.notEqual(
    hashDoc(reverted.doc),
    baseline,
    'a widened window must produce a different hash — it changes the document',
  );
});

test('Catalog accepts a partial record (the stub really is a legal catalog)', () => {
  const partial: Catalog = { uv: CATALOG.uv! };
  assert.deepEqual(errors(addNode(createDoc(), { kind: 'uv' }, partial).diagnostics), []);
  // ... and a kind outside the partial catalog is reported, not crashed on.
  assert.deepEqual(codes(addNode(createDoc(), { kind: 'noise' }, partial).diagnostics), [PT.UNKNOWN_KIND]);
});
