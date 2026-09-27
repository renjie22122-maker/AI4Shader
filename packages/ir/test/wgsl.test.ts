/**
 * WGSL target tests
 * =================
 *
 * There is no WGSL compiler in this environment, so these tests are the strongest
 * available substitute for one. They are written against the *whole* emitted module for
 * a document that contains **every node kind in the shipped catalog** (`CATALOG`), with
 * every kind reachable from the output node so that `emitBodies` actually emits its body:
 *
 *   1. zero `severity: 'error'` diagnostics from the lowering pass;
 *   2. no C declaration shape survives (`float x = …`, `int i = 0`, `void f() { … }`);
 *   3. every `for` is `for (var …)`, none is `for (int …)`;
 *   4. every `fn` has typed parameters and an explicit return arrow (or returns nothing);
 *   5. no parameter is ever assigned — the property WGSL enforces and GLSL does not;
 *   6. the shims and the entry point appear exactly once; `@fragment`/`@builtin(position)`/
 *      `@location(0)` are present;
 *   7. no GLSL type name survives outside a comment;
 *   8. every bracket in the whole emission balances (`structuralBalanceProblems`).
 *
 * What these tests canNOT prove is listed at the bottom of `wgslLower.ts` and in the
 * module header of the emitter: they are structural, not semantic. A `tint`/`naga`
 * compile is still the only way to be sure — see `targets/README.md`.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { emit, canConnect, type Emission } from '../src/emitter.js';
// Side-effect import: this is what registers all six emitters.
import '../src/targets/index.js';
import { CATALOG } from '../src/catalog/index.js';
import { stripComments } from '../src/targets/_shared.js';
import { lowerSubsetToWgsl, structuralBalanceProblems } from '../src/targets/wgslLower.js';
import { rewriteSource, tokenize } from '../src/rewrite.js';
import {
  IR_VERSION,
  NODE_KINDS,
  type Edge,
  type NodeKind,
  type ShaderDoc,
  type ShaderNode,
  type ShaderType,
  type WireRef,
} from '../src/types.js';

/* -------------------------------------------------------------------------- */
/* Fixture: one node of every catalog kind, all reachable from `output`        */
/* -------------------------------------------------------------------------- */

type Produced = { id: string; port: string; type: ShaderType };

function primaryOut(kind: NodeKind): { port: string; type: ShaderType } {
  const schema = CATALOG[kind];
  const name = schema.primaryOut ?? schema.outputs[0]!.name;
  const spec = schema.outputs.find((p) => p.name === name) ?? schema.outputs[0]!;
  return { port: name, type: spec.type };
}

/**
 * Ports left as literals on purpose, so the emitted source keeps an integer-literal
 * conversion (`i32(5)`) as well as the uniform one (`i32(U.u_octaves)`).
 */
const KEEP_LITERAL: Record<string, ReadonlySet<string>> = {
  n_fbm: new Set(['octaves']),
  /*
   * `shape.uv` is accepted by the node "for pipeline symmetry" and ignored by its body, so wiring
   * it is a no-op — and the emitter now says so (`IR037_INERT_INPUT`). This fixture wires every
   * declared input of every kind, which is exactly the shape that diagnostic exists to catch, so
   * the port is left on its literal here rather than having the warning whitelisted. Whitelisting
   * would have been the smaller edit and the worse one: it would make the fixture's own document
   * contain a wire that cannot do anything, and silence the check that noticed.
   */
  n_shape: new Set(['uv']),
};

/**
 * Every catalog kind whose schema does **not** declare `needs.surface`.
 *
 * This fixture's job is to put *every kind the target can carry* through the WGSL lowering pass
 * at once, and WGSL has no surface entry point yet: a document reading a normal is refused with
 * `EM002_SURFACE_UNSUPPORTED` rather than emitted (see `SURFACE_CAPABLE` in `emitter.ts`). So
 * the surface inputs are filtered out here — and, crucially, **the refusal itself is asserted**
 * in `targets.test.ts`, so the gap is a pinned contract rather than a silent exclusion.
 *
 * Derived from the schemas rather than listed, so a future surface input node is excluded
 * automatically instead of turning this fixture red for the wrong reason.
 */
const CARRIABLE_KINDS = NODE_KINDS.filter((k) => !CATALOG[k]?.needs?.surface);

/**
 * Build the conformance document.
 *
 * Wiring strategy, in three steps:
 *
 *   1. one node per catalog kind, in catalog order (generators → domain → noise → math →
 *      color → output), then best-effort wiring of every input to an *earlier* node of a
 *      compatible type (`canConnect`, the emitter's own rule). "Earlier" makes the result
 *      a DAG by construction, and every port that is left unwired has a catalog default.
 *   2. the producers of each value type are folded through a chain of like-typed nodes
 *      (`vec_op` for vec4, `math` for float, `tile` for vec2) and the three chains are
 *      fanned into a `mix` → `vec_op` pair. Without this, most kinds would be dead
 *      branches: `emitBodies` skips anything not reachable from the output, so their
 *      bodies would never be lowered and never tested.
 *   3. two extra nodes exist only to cover the uniform paths: `u_octaves` reaches
 *      `int(…)` (→ `i32(U.u_octaves)`) and `u_tint` reaches a vec4 input.
 */
function buildCatalogDoc(): ShaderDoc {
  const nodes: ShaderNode[] = [];
  const edges: Edge[] = [];
  const inputs = new Map<string, Record<string, WireRef>>();
  const produced = new Map<string, Produced>();

  const wire = (to: string, port: string, from: Produced): void => {
    const map = inputs.get(to) ?? {};
    map[port] = { node: from.id, port: from.port };
    inputs.set(to, map);
    edges.push({ from: { node: from.id, port: from.port }, to: { node: to, port } });
  };
  const sink = (kind: NodeKind, id: string, wires: Record<string, Produced>): Produced => {
    nodes.push({ id, kind });
    for (const [port, from] of Object.entries(wires)) wire(id, port, from);
    const out = primaryOut(kind);
    produced.set(id, { id, port: out.port, type: out.type });
    return produced.get(id)!;
  };

  for (const kind of CARRIABLE_KINDS) {
    const id = `n_${kind}`;
    const params =
      kind === 'shape'
        ? { color: { kind: 'param' as const, name: 'u_tint', type: 'vec4' as ShaderType, value: [1, 0.5, 0.2, 1], label: 'Tint' } }
        : undefined;
    // `custom_glsl` keeps its body on the node; without one it is `IR034` by design. The
    // snippet runs through this target's lowering pass like any catalog body, which is the
    // property the fixture exists to exercise.
    const body = kind === 'custom_glsl' ? 'return mix_(${a}, ${b}, clamp(${c}.x, 0.0, 1.0));' : undefined;
    nodes.push({
      id,
      kind,
      ...(params ? { params } : {}),
      ...(body ? { body } : {}),
    });
    const out = primaryOut(kind);
    produced.set(id, { id, port: out.port, type: out.type });
  }

  for (const node of nodes) {
    if (node.kind === 'output') continue; // wired last, to the fan-in below
    // Two ports of one node may not be fed by the *same* source node: `analyzeGraph`
    // counts an in-degree per wire but decrements it once per distinct edge pair
    // (`seenEdge`), so a parallel edge leaves the consumer stuck at in-degree 1 and is
    // reported as `IR007_CYCLE` (see the report accompanying this change). Distinct
    // sources per node also exercise more of the float→vecN coercion paths.
    const used = new Set<string>();
    for (const port of CATALOG[node.kind].inputs) {
      if (node.params?.[port.name]) continue; // a bound uniform stays visible
      if (KEEP_LITERAL[node.id]?.has(port.name)) continue;
      const from = nodes.find((cand) => {
        if (cand.id === node.id || cand.kind === 'output' || used.has(cand.id)) return false;
        if (NODE_KINDS.indexOf(cand.kind) >= NODE_KINDS.indexOf(node.kind)) return false;
        return canConnect(produced.get(cand.id)!.type, port.type);
      });
      if (!from) continue;
      used.add(from.id);
      wire(node.id, port.name, produced.get(from.id)!);
    }
  }

  const byType = (type: ShaderType): Produced[] =>
    [...produced.values()].filter((p) => p.type === type && p.id !== 'n_output');

  const chain = (kind: NodeKind, aPort: string, bPort: string, items: Produced[]): Produced => {
    let acc = items[0]!;
    items.slice(1).forEach((item, n) => {
      acc = sink(kind, `c_${kind}_${n}`, { [aPort]: acc, [bPort]: item });
    });
    return acc;
  };

  // A second FBM, whose octave count is a uniform: `i32(U.u_octaves)` must be emitted.
  const uniformFbm = sink('fbm', 'n_fbm_u', { uv: produced.get('n_uv')! });
  nodes[nodes.length - 1] = {
    ...nodes[nodes.length - 1]!,
    params: { octaves: { kind: 'param', name: 'u_octaves', type: 'float', value: 5, min: 1, max: 8, step: 1, label: 'Octaves' } },
  };

  const floats = chain('math', 'a', 'b', byType('float'));
  const vec2s = chain('tile', 'uv', 'count', byType('vec2'));
  const vec4s = chain('vec_op', 'a', 'b', byType('vec4'));

  const asFloat = sink('noise', 'c_to_float', { uv: vec2s });
  const combined = sink('smoothstep', 'c_combine', { edge0: floats, value: asFloat });
  const asColor = sink('mix', 'c_as_color', { t: combined });
  const finalColor = sink('vec_op', 'c_final', { a: vec4s, b: asColor });
  wire('n_output', 'color', finalColor);

  return {
    irVersion: IR_VERSION,
    id: 'wgsl-conformance',
    title: 'WGSL conformance graph',
    spec: { intent: 'lower every catalog body to WGSL', category: 'test', tags: ['wgsl'], features: [] },
    stage: 'fragment',
    graph: {
      nodes: nodes.map((n) => {
        const wiring = inputs.get(n.id);
        return wiring ? { ...n, inputs: wiring } : { ...n };
      }),
      edges,
    },
    render: { resolution: [512, 512], qualityTier: 'medium', maxIterations: 8, transparent: false },
    targets: ['wgsl'],
  };
}

const DOC = buildCatalogDoc();
const EMISSION: Emission = emit(DOC, { catalog: CATALOG, target: 'wgsl' });
const SOURCE = EMISSION.source;
/** The same module with comments blanked (newlines kept): what a compiler would read. */
const CODE = stripComments(SOURCE);

/* -------------------------------------------------------------------------- */
/* Small helpers                                                               */
/* -------------------------------------------------------------------------- */

function countOf(src: string, re: RegExp): number {
  return (src.match(re) ?? []).length;
}

function errorsOf(e: Emission): string[] {
  return e.diagnostics.filter((d) => d.severity === 'error').map((d) => `${d.code}: ${d.message}`);
}

type FnDecl = {
  name: string;
  header: string;
  params: { name: string; type: string }[];
  /** Return type text (attributes included), or `null` when there is no `->`. */
  ret: string | null;
  body: string;
};

/** A parameter may carry an attribute (`@builtin(position) pos: vec4f`). */
function parseParam(text: string): { name: string; type: string } {
  const cleaned = text.replace(/@[A-Za-z_]\w*(?:\([^)]*\))?/g, '').trim();
  const m = /^([A-Za-z_]\w*)\s*:\s*(.+)$/.exec(cleaned);
  return m ? { name: m[1]!, type: m[2]!.trim() } : { name: cleaned, type: '' };
}

/**
 * Every `fn` in the module, with its parameter list and body. A character scanner rather
 * than a regex: the entry point's `@builtin(position)` attribute contains parentheses, so
 * `fn\s+\w+\(([^)]*)\)` silently *skips* `fs_main` — the one function that matters most.
 */
function fnDeclarations(code: string): FnDecl[] {
  const out: FnDecl[] = [];
  const re = /\bfn\s+([A-Za-z_]\w*)\s*\(/g;
  for (let m = re.exec(code); m; m = re.exec(code)) {
    const open = m.index + m[0].length - 1;
    let depth = 0;
    let close = -1;
    for (let i = open; i < code.length; i++) {
      if (code[i] === '(') depth++;
      else if (code[i] === ')') {
        depth--;
        if (depth === 0) {
          close = i;
          break;
        }
      }
    }
    if (close < 0) continue;
    let i = close + 1;
    while (i < code.length && /\s/.test(code[i]!)) i++;
    let ret: string | null = null;
    if (code.startsWith('->', i)) {
      const start = i + 2;
      let j = start;
      while (j < code.length && code[j] !== '{') j++;
      ret = code.slice(start, j).trim();
      i = j;
    }
    if (code[i] !== '{') continue;
    const bodyOpen = i;
    depth = 0;
    let bodyClose = -1;
    for (let k = bodyOpen; k < code.length; k++) {
      if (code[k] === '{') depth++;
      else if (code[k] === '}') {
        depth--;
        if (depth === 0) {
          bodyClose = k;
          break;
        }
      }
    }
    const paramsText = code.slice(open + 1, close);
    out.push({
      name: m[1]!,
      header: code.slice(m.index, bodyOpen),
      params: paramsText.trim()
        ? paramsText.split(',').map((p) => parseParam(p.trim())).filter((p) => p.name.length > 0)
        : [],
      ret,
      body: bodyClose < 0 ? code.slice(bodyOpen) : code.slice(bodyOpen + 1, bodyClose),
    });
    re.lastIndex = bodyClose < 0 ? code.length : bodyClose;
  }
  return out;
}

const ASSIGN_OPS = new Set(['=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=', '++', '--']);

/**
 * Parameter writes in a function body, found on the token stream rather than with a
 * regex: the lvalue may be `p`, `p.x` or `p.xy`, and `p == x` must not count.
 */
function parameterWrites(body: string, params: readonly string[]): string[] {
  const toks = tokenize(body).filter((t) => t.kind !== 'space' && t.kind !== 'comment');
  const bad: string[] = [];
  for (let i = 0; i < toks.length; i++) {
    const t = toks[i]!;
    if (t.kind !== 'ident' || !params.includes(t.text)) continue;
    let j = i + 1;
    while (toks[j]?.text === '.') j += 2; // `p.x`, `p.xy`
    const op = toks[j]?.text ?? '';
    if (ASSIGN_OPS.has(op)) bad.push(`${t.text}${op}`);
  }
  return bad;
}

const FNS = fnDeclarations(CODE);

/* -------------------------------------------------------------------------- */
/* 1. the fixture is what it claims to be                                      */
/* -------------------------------------------------------------------------- */

test('wgsl fixture: the document contains every kind the target can carry, and emits all of them', () => {
  assert.equal(EMISSION.target, 'wgsl');
  assert.ok(SOURCE.length > 5000, `suspiciously short emission (${SOURCE.length} chars)`);
  // Every kind must survive reachability analysis, or its body was never lowered.
  for (const kind of CARRIABLE_KINDS) {
    assert.ok(SOURCE.includes(`L_n_${kind}_`), `no emitted local for catalog kind \`${kind}\``);
    assert.ok(
      DOC.graph.nodes.some((n) => n.kind === kind),
      `the fixture is missing a node of kind \`${kind}\``,
    );
  }
  // The excluded set is exactly the surface inputs, and it is not empty — an empty exclusion
  // would mean this filter had quietly stopped excluding anything and the loop above was
  // silently covering less than it claims.
  const excluded = NODE_KINDS.filter((k) => !CARRIABLE_KINDS.includes(k));
  assert.deepEqual(
    [...excluded].sort(),
    ['light_dir', 'normal', 'view_dir', 'world_pos'],
    'the excluded kinds changed; update this test and SURFACE_CAPABLE together',
  );
  assert.ok(SOURCE.includes('L_n_output_color'), 'the output node must expose a local');
  assert.ok(!SOURCE.includes('unresolved:'), 'a node placeholder was not expanded');
  assert.ok(FNS.length > 20, `expected the whole library, saw ${FNS.length} functions`);
});

/* -------------------------------------------------------------------------- */
/* 2. diagnostics                                                              */
/* -------------------------------------------------------------------------- */

test('wgsl: the lowering pass reports zero error diagnostics', () => {
  assert.deepEqual(errorsOf(EMISSION), []);
});

test('wgsl: the only tolerated non-error diagnostics are the documented warnings', () => {
  const allowed = new Set(['IR011_DEAD_NODES', 'RW010_LOOP_SHAPE']);
  const unexpected = EMISSION.diagnostics.filter((d) => d.severity !== 'error' && !allowed.has(d.code));
  assert.deepEqual(
    unexpected.map((d) => `${d.code}: ${d.message}`),
    [],
    'an undocumented diagnostic appeared; it is either a real finding or a missing whitelist entry',
  );
});

/* -------------------------------------------------------------------------- */
/* 3. no C declaration shape survives                                          */
/* -------------------------------------------------------------------------- */

test('wgsl: no C-style typed declaration survives (`float x = …`, `vec2f q = …`)', () => {
  const cDecl = /^\s*(?:float|int|bool|vec2f|vec3f|vec4f|mat2x2f|mat3x3f)\s+\w+\s*=/gm;
  const hits = (CODE.match(cDecl) ?? []).map((s) => s.trim());
  assert.deepEqual(hits, [], 'every local must be declared with `var <name>: <type> = …`');

  // The same rule, not anchored: a declaration after another statement on one line
  // (`var a: f32 = 1.0; float b = 2.0;`) would slip past the anchored form.
  const inline = /\b(?:f32|i32|bool|vec2f|vec3f|vec4f|mat2x2f|mat3x3f)\s+\w+\s*=/g;
  const inlineHits = (CODE.match(inline) ?? []).map((s) => s.trim());
  assert.deepEqual(inlineHits, [], 'no `type name =` form may appear outside a `var` declaration');
});

test('wgsl: no function signature is left in C form (`name(args) {` without `->`)', () => {
  // A `for`/`if` header block is the only place an identifier may be followed by `(` and
  // `{` without an intervening `->`, and those headers never contain a C type name.
  for (const m of CODE.matchAll(/\b[A-Za-z_]\w*\s*\([^;{}]*\)\s*\{/g)) {
    const head = m[0];
    assert.match(head, /^\s*(?:if|else if|for)\b/, `possible C-style signature or stray block: \`${head.trim()}\``);
  }
  for (const fn of FNS) {
    assert.match(fn.header, /^fn\s+[A-Za-z_]\w*\s*\(/, `malformed definition header: ${fn.header.split('\n')[0]}`);
  }
});

/* -------------------------------------------------------------------------- */
/* 4. loop headers                                                             */
/* -------------------------------------------------------------------------- */

test('wgsl: no `for (int …)` survives and every `for (` is `for (var …)`', () => {
  assert.equal(countOf(CODE, /for\s*\(\s*int\b/g), 0, 'a GLSL `for (int …)` header survived');
  assert.ok(countOf(CODE, /for\s*\(/g) > 0, 'the fixture must contain loops to make this meaningful');
  for (const m of CODE.matchAll(/for\s*\(/g)) {
    const rest = CODE.slice(m.index, m.index + 24);
    assert.match(rest, /^for \(var /, `loop header is not WGSL: \`${rest.split('{')[0]!.trim()}\``);
  }
  assert.match(CODE, /for \(var i: i32 = 0; i < 5; i = i \+ 1\) \{/, 'fbm2 loop header');
  assert.match(CODE, /for \(var j: i32 = -1; j <= 1; j = j \+ 1\) \{/, 'voronoi2 outer loop header');
  assert.match(CODE, /for \(var i: i32 = -1; i <= 1; i = i \+ 1\) \{/, 'voronoi2 inner loop header');
});

/* -------------------------------------------------------------------------- */
/* 5. function signatures and immutable parameters                             */
/* -------------------------------------------------------------------------- */

test('wgsl: every `fn` has typed parameters and an explicit return type or returns nothing', () => {
  assert.ok(FNS.length > 0);
  for (const fn of FNS) {
    for (const p of fn.params) {
      assert.ok(p.type.length > 0, `\`${fn.name}\`: parameter \`${p.name}\` has no type annotation`);
      assert.match(p.type, /^[A-Za-z_]\w*(?:<[^<>]*>)?$/, `\`${fn.name}\`: parameter \`${p.name}\` type \`${p.type}\``);
    }
    if (fn.ret === null) {
      assert.ok(!/\breturn\b/.test(fn.body), `\`${fn.name}\` has no return arrow but returns a value`);
    } else {
      assert.ok(fn.ret.length > 0, `\`${fn.name}\` has an empty return type`);
    }
  }
  assert.ok(FNS.some((f) => f.ret !== null), 'the library functions must all carry a return type');
  // The `void` (no arrow) case does not occur in the shared library; it is covered by the
  // `void g(float a)` unit test below.
  assert.ok(FNS.every((f) => f.ret !== null || !/\breturn\b/.test(f.body)), 'a `void` function must not return a value');
  // Spot checks for the shapes the task calls out.
  assert.match(CODE, /fn hash11\(p: f32\) -> f32 \{/);
  assert.match(CODE, /fn fbm2\(p: vec2f, octaves: i32, lacunarity: f32, gain: f32, ridged: f32\) -> f32 \{/);
  assert.match(CODE, /fn sd_circle\(p: vec2f, r: f32\) -> f32 \{ return length\(p\) - r; \}/);
});

test('wgsl: no function assigns to any of its parameters (WGSL parameters are immutable)', () => {
  for (const fn of FNS) {
    const names = fn.params.map((p) => p.name);
    const writes = parameterWrites(fn.body, names);
    assert.deepEqual(writes, [], `\`${fn.name}\` writes to its parameter(s)`);
    // The literal regex form from the specification, as an independent second check.
    for (const name of names) {
      const re = new RegExp(`^\\s*${name}\\s*[+\\-*/]?=`, 'm');
      assert.ok(!re.test(fn.body), `\`${fn.name}\`: body assigns \`${name}\``);
    }
  }
});

test('wgsl: every parameter a body writes is copied into a `var` at the top of the body', () => {
  /*
   * WGSL parameters are immutable, so a function that assigns to one has to copy it into a `var`
   * first. The lowering pass derives that itself; this test checks it did, and the list below is
   * the *consequence* of which functions write their parameters — not the rule.
   *
   * It went stale once, in the instructive direction: making `fbm2` a one-line wrapper around
   * `fbm2t` (so the tiled and untiled lattices cannot drift) removed its parameter writes, and
   * this test failed claiming the emission was wrong when the emission had in fact got *simpler*.
   * The rule moved to `fbm2t` with the code, which is exactly what a rule pinned to a name
   * cannot do.
   */
  const copies: [string, string][] = [
    ['hash11', 'var p_v: f32 = p;'],
    ['hash31', 'var p_v: vec3f = p;'],
    ['fbm2t', 'var p_v: vec2f = p;'],
    ['sd_hexagon', 'var p_v: vec2f = p;'],
    ['sd_triangle', 'var p_v: vec2f = p;'],
    ['sd_star', 'var p_v: vec2f = p;'],
  ];
  for (const [fn, copy] of copies) {
    const decl = FNS.find((f) => f.name === fn);
    assert.ok(decl, `\`${fn}\` is missing from the emission`);
    assert.ok(decl.body.includes(copy), `\`${fn}\` must copy its parameter: expected \`${copy}\``);
    // …and the body must use the copy, never the parameter.
    assert.match(decl.body, new RegExp(`${fn === 'sd_triangle' ? 'p_v\\.x' : 'p_v'} = `), `\`${fn}\` must write through the copy`);
  }
  // A function that does not write a parameter must not get a pointless copy — including the
  // wrappers, which pass their arguments straight through.
  for (const pure of ['noise2', 'gnoise2', 'fbm2', 'voronoi2', 'noise2t', 'voronoi2t']) {
    const decl = FNS.find((f) => f.name === pure);
    assert.ok(decl, `\`${pure}\` is missing from the emission`);
    assert.ok(!decl.body.includes('var p_v'), `${pure} never writes \`p\`: no copy should be emitted`);
  }
  assert.ok(FNS.find((f) => f.name === 'sd_triangle')!.body.includes('p_v.x = abs(p_v.x) - r;'), 'component writes use the copy too');
});

/* -------------------------------------------------------------------------- */
/* 6. prologue, shims, entry point                                             */
/* -------------------------------------------------------------------------- */

test('wgsl: shims and entry point appear exactly once, with the required attributes', () => {
  assert.equal(countOf(SOURCE, /\bfn wf_mix\(/g), 1, 'wf_mix shim');
  assert.equal(countOf(SOURCE, /\bfn wf_mod\(/g), 1, 'wf_mod shim');
  assert.equal(countOf(SOURCE, /\bfn wf_atan2\(/g), 1, 'wf_atan2 shim');
  assert.equal(countOf(SOURCE, /@fragment/g), 1, '@fragment');
  assert.equal(countOf(SOURCE, /\bfn fs_main\(/g), 1, 'fs_main');
  assert.ok(SOURCE.includes('@builtin(position)'), '@builtin(position)');
  assert.ok(SOURCE.includes('@location(0)'), '@location(0)');
  assert.ok(SOURCE.includes('struct Uniforms {'), 'the uniform struct');
  assert.ok(SOURCE.includes('@group(0) @binding(1) var iChannel0: texture_2d<f32>;'), 'the texture binding');
  assert.ok(SOURCE.includes('@group(0) @binding(2) var iChannel0Sampler: sampler;'), 'the sampler binding');
});

test('wgsl: the scalar shims are never called (WGSL has no overloading; they are inlined)', () => {
  // The library calls `mix_`/`mod_` with vec2/vec3/vec4 arguments, which the scalar shim
  // cannot serve. Every call site must therefore be a component-wise expression.
  assert.equal(countOf(CODE, /\bwf_mix\s*\(/g), 1, 'only the shim definition may mention wf_mix');
  assert.equal(countOf(CODE, /\bwf_mod\s*\(/g), 1, 'only the shim definition may mention wf_mod');
  assert.match(CODE, /fn wf_atan2\(y: f32, x: f32\) -> f32 \{ return atan2\(vec2f\(y, x\)\); \}/);
  assert.match(CODE, /wf_atan2\(/, 'atan2_ is scalar-only and stays a call');
});

/* -------------------------------------------------------------------------- */
/* 7. names, numbers, texture fetches, matrices                                */
/* -------------------------------------------------------------------------- */

test('wgsl: no GLSL type name survives outside a comment', () => {
  const glsl = CODE.match(/\b(vec2|vec3|vec4|mat2|mat3|float)\b/g) ?? [];
  assert.deepEqual(glsl, [], 'GLSL type names must be renamed (`vec2f`, `f32`, …)');
  // `vec2f`/`mat2x2f` must not be mistaken for `vec2`/`mat2` by the check above.
  assert.ok(CODE.includes('vec2f('), 'the WGSL spellings must be present');
});

test('wgsl: integer literals stay integers in integer position', () => {
  assert.match(CODE, /i32\(5\)/, 'the literal octave count must lower to `i32(5)`, not `i32(5.0)`');
  assert.match(CODE, /i32\(U\.u_octaves\)/, 'a uniform octave count must lower to `i32(U.u_octaves)`');
  assert.equal(countOf(CODE, /i32\(\s*-?\d+\.\d+\s*\)/g), 0, 'no integer conversion may take a float literal');
  assert.equal(countOf(CODE, /\bi < \d+\.0\b/g), 0, 'no integer loop bound may be a float literal');
});

test('wgsl: the texture fetch is bound to the declared sampler', () => {
  assert.match(CODE, /textureSample\(iChannel0, iChannel0Sampler, [A-Za-z_]\w*\)/);
  assert.equal(countOf(CODE, /\btexture\s*\(/g), 0, 'a bare `texture(...)` call must not survive');
  assert.equal(countOf(CODE, /tex2D_/g), 0, 'the one-argument alias must not survive');
});

test('wgsl: GLSL-only mixed-arity builtin calls are broadcast to the vector arity', () => {
  // GLSL declares `clamp(genType, float, float)`; WGSL's table does not (gpuweb/gpuweb#4815),
  // and `hsv2rgb` in the shared library relies on it.
  assert.match(CODE, /clamp\(abs\([\s\S]{0,600}?vec3f\(0\.0\), vec3f\(1\.0\)\);/);
  assert.equal(
    countOf(CODE, /vec3f\(0\.0\), vec3f\(1\.0\)\)/g),
    1,
    'exactly one call site (hsv2rgb) needs the broadcast',
  );
});

test('wgsl: the matrix helpers use WGSL column-major constructors', () => {
  assert.match(CODE, /fn rot2\(a: f32\) -> mat2x2f \{/);
  // Column-major order is the same in GLSL and WGSL, so the argument order is unchanged:
  // `mat2(c, -s, s, c)` and `mat2x2f(c, -s, s, c)` are the same matrix.
  assert.match(CODE, /return mat2x2f\(c, -s, s, c\);/);
  assert.match(CODE, /return mat2x2f\(s\.x, 0\.0, 0\.0, s\.y\);/);
  // `mat2_mul_`/`mat2_rot_` are lowered at their call sites; the inlined arguments must
  // still be WGSL (the rewriter's fixpoint pass is what renames them).
  assert.match(CODE, /rot2\(/, 'the rotation helper must be used');
  assert.equal(countOf(CODE, /mat2_mul_|mat2_rot_/g), 0, 'a matrix alias survived');
  assert.equal(countOf(CODE, /\bm\[0\]\[0\]/g), 0, 'the lowered helper definition survived');
});

/* -------------------------------------------------------------------------- */
/* 8. structure                                                                */
/* -------------------------------------------------------------------------- */

test('wgsl: every bracket in the emission balances (structural checker)', () => {
  assert.deepEqual(structuralBalanceProblems(SOURCE), []);
  assert.deepEqual(structuralBalanceProblems(CODE), []);
  // The checker has to be able to fail, or it proves nothing.
  assert.deepEqual(structuralBalanceProblems('fn a() -> f32 { return 1.0; }'), []);
  assert.match(structuralBalanceProblems('fn a() -> f32 {\n  return 1.0;\n').join(' '), /line 1 is never closed/);
  assert.match(structuralBalanceProblems('fn a() -> f32 { return 1.0; }\n}').join(' '), /unmatched `}` on line 2/);
  assert.match(structuralBalanceProblems('fn a() -> f32 { return f(1.0; }').join(' '), /closed by/);
});

/** Number of top-level (depth-0) comma-separated items in `text`. */
function argCount(text: string): number {
  const body = text.trim();
  if (body.length === 0) return 0;
  let depth = 0;
  let n = 1;
  for (const c of body) {
    if (c === '(' || c === '[' || c === '<') depth++;
    else if (c === ')' || c === ']' || c === '>') depth--;
    else if (c === ',' && depth === 0) n++;
  }
  return n;
}

test('wgsl: every call to a module function has the right arity', () => {
  // A text-splicing lowering can silently drop or duplicate an argument; the arity check
  // is what catches that without a compiler.
  const arity = new Map<string, number>(FNS.map((f) => [f.name, f.params.length]));
  assert.ok(arity.size > 20);
  const problems: string[] = [];
  for (const m of CODE.matchAll(/(^|[^\w.])([A-Za-z_]\w*)\s*\(/g)) {
    const name = m[2]!;
    const want = arity.get(name);
    if (want === undefined) continue;
    const open = m.index + m[0].length - 1;
    let depth = 0;
    let close = -1;
    for (let i = open; i < CODE.length; i++) {
      if (CODE[i] === '(') depth++;
      else if (CODE[i] === ')') {
        depth--;
        if (depth === 0) {
          close = i;
          break;
        }
      }
    }
    if (close < 0) {
      problems.push(`${name}: unterminated argument list`);
      continue;
    }
    const got = argCount(CODE.slice(open + 1, close));
    if (got !== want) problems.push(`${name} called with ${got} argument(s), declared with ${want}`);
  }
  assert.deepEqual(problems, []);
});

/** `var` names declared directly in a function's own scope (not in a block or a header). */
function ownScopeVarNames(body: string): string[] {
  const names: string[] = [];
  let braces = 0;
  let parens = 0;
  for (let i = 0; i < body.length; i++) {
    const c = body[i]!;
    if (c === '{') braces++;
    else if (c === '}') braces--;
    else if (c === '(') parens++;
    else if (c === ')') parens--;
    else if (braces === 0 && parens === 0 && body.startsWith('var', i)) {
      if (/[\w.]/.test(body[i - 1] ?? '') || /\w/.test(body[i + 3] ?? '')) continue;
      const m = /^var\s+([A-Za-z_]\w*)\s*:/.exec(body.slice(i));
      if (m) names.push(m[1]!);
    }
  }
  return names;
}

test('wgsl: no function declares the same local twice in its own scope', () => {
  // The node bodies are spliced into ONE scope (`fs_main`), which is why every local
  // carries its `${node_id}` suffix. A duplicate declaration is a WGSL error, and the
  // WGPU preview would refuse the module.
  for (const fn of FNS) {
    const names = ownScopeVarNames(fn.body);
    const seen = new Set<string>();
    const dupes = names.filter((n) => (seen.has(n) ? true : (seen.add(n), false)));
    assert.deepEqual(dupes, [], `\`${fn.name}\` declares a local twice`);
  }
  const fsMain = FNS.find((f) => f.name === 'fs_main');
  assert.ok(fsMain && ownScopeVarNames(fsMain.body).length > 30, 'the entry point must carry all node bodies');
});

test('wgsl: no construct WGSL lacks survives (ternary, bitwise, arrays, `%`)', () => {
  assert.equal(countOf(CODE, /\?/g), 0, '`?:`');
  assert.equal(countOf(CODE, /%/g), 0, '`%`');
  assert.equal(countOf(CODE, /[[\]]/g), 0, 'array indexing');
  assert.equal(countOf(CODE, /(?<![&|])&(?![&])/g), 0, 'bitwise `&`');
  assert.equal(countOf(CODE, /#/g), 0, 'preprocessor');
});

test('wgsl: emission is deterministic and byte-stable across runs', () => {
  const again = emit(DOC, { catalog: CATALOG, target: 'wgsl' });
  assert.equal(again.source, SOURCE);
  assert.equal(again.hash, EMISSION.hash);
});

/* -------------------------------------------------------------------------- */
/* 9. the lowering pass in isolation                                           */
/* -------------------------------------------------------------------------- */

/**
 * Lower one restricted-GLSL snippet through the pass with the real WGSL renames.
 *
 * The `rewrite` callback mirrors the emitter's contract exactly: rename, then re-run
 * until the text stops changing (`lowerMatrixMul` splices its arguments verbatim, so one
 * pass can leave a `vec2(` inside an inlined matrix product).
 */
function wgslRewrite(src: string): string {
  let out = rewriteSource(src, 'wgsl').source;
  for (let pass = 0; pass < 3; pass++) {
    const again = rewriteSource(out, 'wgsl');
    if (again.source === out) break;
    out = again.source;
  }
  return out;
}

function buildWgslFor(
  src: string,
  immutableNames: string[] = [],
  symbols: Record<string, 1 | 2 | 3 | 4> = {},
): { source: string; diagnostics: ReturnType<typeof lowerSubsetToWgsl>['diagnostics'] } {
  return lowerSubsetToWgsl(src, { context: 'unit', immutableNames, symbols, rewrite: wgslRewrite });
}

test('wgsl lowering: scalar bounds of a vector builtin are broadcast, and only when proven', () => {
  const r = buildWgslFor(
    [
      'float f(vec3 v, float lo, float hi) {',
      '  vec3 a = clamp(v, lo, hi);',
      '  vec3 b = clamp(v, 0.0, 1.0);',
      '  vec3 c = min(v, hi);',
      '  vec3 d = max(v, 0.5);',
      '  vec3 e = step(0.5, v);',
      '  vec3 g = smoothstep(0.0, 1.0, v);',
      '  float h = clamp(lo, 0.0, 1.0);',
      '  vec3 i = clamp(w, 0.0, 1.0);',
      '  return a.x + b.x + c.x + d.x + e.x + g.x + h + i.x;',
      '}',
    ].join('\n'),
  );
  assert.deepEqual(r.diagnostics, []);
  assert.match(r.source, /clamp\(v, vec3f\(lo\), vec3f\(hi\)\)/);
  assert.match(r.source, /clamp\(v, vec3f\(0\.0\), vec3f\(1\.0\)\)/);
  assert.match(r.source, /min\(v, vec3f\(hi\)\)/);
  assert.match(r.source, /max\(v, vec3f\(0\.5\)\)/);
  assert.match(r.source, /step\(vec3f\(0\.5\), v\)/);
  assert.match(r.source, /smoothstep\(vec3f\(0\.0\), vec3f\(1\.0\), v\)/);
  // Unchanged: an all-scalar call, and a call whose first argument's arity is unknown
  // (`w` is not declared anywhere). A guess here could emit invalid WGSL, so there is none.
  assert.match(r.source, /clamp\(lo, 0\.0, 1\.0\)/);
  assert.match(r.source, /clamp\(w, 0\.0, 1\.0\)/);

  // A uniform's declared type is enough to prove the arity (the emitter passes these in).
  const uniform = buildWgslFor('vec3 f(vec3 u_dir) { return clamp(u_dir, 0.0, 1.0); }', [], { u_dir: 3 });
  assert.deepEqual(uniform.diagnostics, []);
  assert.match(uniform.source, /clamp\(u_dir, vec3f\(0\.0\), vec3f\(1\.0\)\)/);

  // A local declared with `vec3` is proven too, and `vec3f(vec3f(…))` is never emitted.
  const local = buildWgslFor('vec3 f(float x) {\n  vec3 c = vec3(x, x, x);\n  return clamp(c, 0.0, 1.0);\n}');
  assert.deepEqual(local.diagnostics, []);
  assert.match(local.source, /clamp\(c, vec3f\(0\.0\), vec3f\(1\.0\)\)/);
});

test('wgsl lowering: an inlined matrix product is fully renamed', () => {
  const r = buildWgslFor('vec2 f(vec2 uv, float a) { return mat2_mul_(rot2(a), (uv - vec2(0.5, 0.5)) * vec2(2.0, 2.0)); }');
  assert.deepEqual(r.diagnostics, []);
  assert.match(r.source, /return \(rot2\(a\) \*\s+\(uv - vec2f\(0\.5, 0\.5\)\) \* vec2f\(2\.0, 2\.0\)\);/);
});

test('wgsl lowering: signatures, declarations, loops and parameter copies', () => {
  const r = buildWgslFor(
    [
      'float f(vec2 p, int n) {',
      '  vec2 q = p * 2.0;',
      '  p = q + vec2(1.0, 1.0);',
      '  for (int i = 0; i < 8; i++) {',
      '    q = q + float(i) * p;',
      '  }',
      '  return q.x + float(n);',
      '}',
    ].join('\n'),
  );
  assert.deepEqual(r.diagnostics, []);
  assert.match(r.source, /fn f\(p: vec2f, n: i32\) -> f32 \{/);
  assert.match(r.source, /var p_v: vec2f = p;/);
  assert.match(r.source, /var q: vec2f = p_v \* 2\.0;/);
  assert.match(r.source, /p_v = q \+ vec2f\(1\.0, 1\.0\);/);
  assert.match(r.source, /for \(var i: i32 = 0; i < 8; i = i \+ 1\) \{/);
  assert.match(r.source, /q = q \+ f32\(i\) \* p_v;/);
  assert.match(r.source, /return q\.x \+ f32\(n\);/);
});

test('wgsl lowering: `void` functions lose the return arrow', () => {
  const r = buildWgslFor('void g(float a) {\n  float b = a;\n}');
  assert.deepEqual(r.diagnostics, []);
  assert.match(r.source, /fn g\(a: f32\) \{/);
  assert.match(r.source, /var b: f32 = a;/);
  assert.equal(countOf(r.source, /->/g), 0);
});

test('wgsl lowering: component writes to a parameter are detected too', () => {
  const r = buildWgslFor('float h(vec2 p) {\n  p.x = p.y;\n  return p.x;\n}');
  assert.deepEqual(r.diagnostics, []);
  assert.match(r.source, /var p_v: vec2f = p;/);
  assert.match(r.source, /p_v\.x = p_v\.y;/);
});

test('wgsl lowering: shim arguments are parenthesised before substitution', () => {
  // `a - b * floor(a / b)` with a multi-token `b` must not flatten into
  // `a - b * floor(a / b * c)`; this is the precedence trap of text-level inlining.
  const mod = buildWgslFor('float f(float x, float y, float z) { return mod_(x * 2.0, y * z); }');
  assert.deepEqual(mod.diagnostics, []);
  assert.match(mod.source, /\(x \* 2\.0\) - \(y \* z\) \* floor\(\(x \* 2\.0\) \/ \(y \* z\)\)/);

  const mix = buildWgslFor('vec3 g(vec3 a, vec3 b, float t) { return mix_(a, b, t + 1.0); }');
  assert.deepEqual(mix.diagnostics, []);
  assert.match(mix.source, /\(a \+ \(b - a\) \* \(t \+ 1\.0\)\)/);
});

test('wgsl lowering: unsupported constructs are reported, never emitted', () => {
  const ternary = buildWgslFor('float f(float a) { return a > 0.5 ? 1.0 : 0.0; }');
  const ternaryError = ternary.diagnostics.find((d) => d.code === 'TGT005_UNSUPPORTED_TERNARY');
  assert.ok(ternaryError, 'a ternary must be reported');
  assert.equal(ternaryError.severity, 'error');
  assert.match(ternaryError.fix ?? '', /mix_|step|if/);

  const loop = buildWgslFor('float f(int n) {\n  for (int i = 0; i < n; i++) {\n  }\n  return 0.0;\n}');
  const loopError = loop.diagnostics.find((d) => d.code === 'TGT003_UNSUPPORTED_LOOP');
  assert.ok(loopError, 'a non-literal loop bound must be reported, not guessed at');
  assert.match(loopError.fix ?? '', /literal/);

  const braceless = buildWgslFor('float f(float a) {\n  if (a > 0.5) return 1.0;\n  return 0.0;\n}');
  assert.ok(
    braceless.diagnostics.some((d) => d.code === 'TGT004_UNSUPPORTED_CONDITIONAL'),
    'a braceless if body must be reported (WGSL requires a block)',
  );

  const whileLoop = buildWgslFor('float f(float a) {\n  while (a > 0.5) { a = a - 0.1; }\n  return a;\n}');
  assert.ok(
    whileLoop.diagnostics.some((d) => d.code === 'TGT009_UNSUPPORTED_STATEMENT'),
    '`while` must be reported',
  );

  const array = buildWgslFor('float f(float a) { return a[0]; }');
  assert.ok(
    array.diagnostics.some((d) => d.code === 'TGT006_UNSUPPORTED_OPERATOR'),
    'array indexing must be reported',
  );

  // A node body runs in the entry point's scope, where `uv` is a `let` and the uniforms
  // are members of `U`: writing to either is invalid WGSL and must be reported.
  const immutable = buildWgslFor('vec2 o = uv;\nuv = vec2(0.0, 0.0);', ['uv']);
  const immutableError = immutable.diagnostics.find((d) => d.code === 'TGT007_IMMUTABLE_ASSIGNMENT');
  assert.ok(immutableError, 'assigning to the entry point `uv` must be reported');
  assert.match(immutableError.fix ?? '', /uv/);
  // …but a *parameter* that happens to share the name is a local, and is copied instead.
  const shadowing = buildWgslFor('vec2 f(vec2 uv) { uv = vec2(0.0, 0.0); return uv; }', ['uv']);
  assert.deepEqual(shadowing.diagnostics, []);
  assert.match(shadowing.source, /var uv_v: vec2f = uv;/);

  for (const r of [ternary, loop, braceless, whileLoop, array, immutable]) {
    for (const d of r.diagnostics) {
      assert.equal(d.severity, 'error', `${d.code} must be an error`);
      assert.match(d.code, /^TGT\d{3}_/, `diagnostic code \`${d.code}\` is outside the TGT0xx range`);
      assert.ok((d.fix ?? '').length > 0, `${d.code} must carry a \`fix\` naming the construct`);
      assert.ok((d.at?.line ?? 0) > 0, `${d.code} must point at a line`);
    }
  }
});

test('wgsl lowering: comments and blank lines pass through untouched', () => {
  const src = [
    '// ---- section ----',
    '',
    '// explains the next function',
    'float f(float a) {',
    '  // inside the body',
    '  float b = a * 2.0; // trailing',
    '  return b;',
    '}',
    '',
    '// tail comment',
  ].join('\n');
  const r = buildWgslFor(src);
  assert.deepEqual(r.diagnostics, []);
  for (const line of ['// ---- section ----', '// explains the next function', '  // inside the body', '// tail comment']) {
    assert.ok(r.source.includes(line), `comment lost: ${line}`);
  }
  assert.equal(r.source.split('\n').length, src.split('\n').length, 'line count changed');
});

test('wgsl lowering: the structural self-check reports on its own output', () => {
  // An unterminated chunk cannot be lowered into balanced WGSL; the pass must say so
  // rather than return something that looks plausible.
  const r = buildWgslFor('float f(float a) {\n  return a;');
  assert.ok(
    r.diagnostics.some((d) => d.code === 'TGT090_OUTPUT_STRUCTURE'),
    'an unbalanced chunk must be reported by the pass itself',
  );
  assert.notDeepEqual(structuralBalanceProblems(r.source), []);
});
