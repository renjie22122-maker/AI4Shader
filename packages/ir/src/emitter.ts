/**
 * AI4Shader — Emitter core
 * ========================
 *
 * IR -> source, for every target. This module owns everything target-independent:
 *
 *   1. graph validation + deterministic topological order
 *   2. static type checking of every wire (before a compiler is ever invoked)
 *   3. uniform collection & de-duplication (the basis of both sliders and presets)
 *   4. placeholder expansion of node bodies from the catalog
 *   5. per-target assembly, delegated to `targets/*.ts`
 *
 * Determinism is a hard requirement: the same `ShaderDoc` must produce byte-identical
 * source for a given target, forever. That is what makes `doc.hash` a valid cache key
 * for compiled binaries and golden images. Consequently: no `Date.now()`, no `Math.random()`,
 * no iteration over object keys where insertion order is not guaranteed by construction,
 * and every generated identifier is derived from node ids.
 */

import {
  type Diagnostic,
  type NodeSchema,
  type NodeKind,
  type ParamValue,
  type ShaderDoc,
  type ShaderGraph,
  type ShaderNode,
  type ShaderType,
  type TargetFormat,
  type UniformDecl,
  TYPE_ARITY,
  SURFACE_HOST_SYMBOLS,
} from './types.js';
import { lintSubset, rewriteSource, type RewriteResult } from './rewrite.js';
import { OCTAVE_LIMIT } from './glslSubset.js';

/* -------------------------------------------------------------------------- */
/* Public shapes                                                               */
/* -------------------------------------------------------------------------- */

export type EmitOptions = {
  /** Catalog of node definitions. Injected so the IR package stays free of content. */
  catalog: Record<NodeKind, NodeSchema>;
  /** Which target to emit. */
  target: TargetFormat;
  /**
   * Node output ports that are still consumed but should nevertheless be treated as
   * the result. Defaults to the single `output` node.
   */
  includeLibrary?: boolean;
  /**
   * Use this emitter instead of the one registered for `target`.
   *
   * The registry (`registerTarget`) is a process-global table, which is right for the six shipped
   * targets and wrong for everything else: a test with a stub emitter, a host that wants to try a
   * modified lowering pass, or two callers in one process that disagree. Without this option the
   * only way to do any of those is to *replace* the global and put it back — and a global that is
   * swapped and restored is a global that is observed mid-swap by whatever else is running. That
   * is not hypothetical: a test file did exactly this and made sixteen tests in two other files
   * fail, depending on which files the runner happened to overlap.
   */
  emitter?: TargetEmitter;
};

export type Emission = {
  target: TargetFormat;
  /** Complete, compilable source (or the closest thing the platform allows). */
  source: string;
  /** Additional files: `.meta`, UE node recipe, MSL header, etc. */
  sidecars?: { name: string; content: string }[];
  /** Diagnostics from validation + rewriting. `error` means `source` is not trustworthy. */
  diagnostics: Diagnostic[];
  /** Everything the runtime must bind, in a stable order. */
  uniforms: UniformDecl[];
  /** Identifier -> GLSL type, for the reflection panel. */
  symbols: { name: string; type: ShaderType; origin?: string }[];
  /** Hash of `source`; used to key compile caches. */
  hash: string;
  /**
   * What this shader needs from its host, over the reachable graph.
   *
   * `surface: true` means the source is a **surface shader**: it reads interpolated vertex
   * outputs and cannot be evaluated on a fullscreen quad. A host that only has a quad (the
   * flat preview) must either synthesise a stand-in or refuse; one that has a mesh (the 3D
   * preview, Unity, UE) supplies them directly.
   */
  needs: {
    matrix: boolean;
    derivatives: boolean;
    qualityScaled: boolean;
    surface: boolean;
    /** A reachable node samples the host image channel (iChannel0). */
    channel: boolean;
  };
};

/* -------------------------------------------------------------------------- */
/* Hashing                                                                     */
/* -------------------------------------------------------------------------- */

/**
 * FNV-1a 64-bit (as two 32-bit halves) — dependency-free and stable across JS engines
 * and platforms, unlike `crypto.subtle` which is async and unavailable in some hosts.
 */
export function stableHash(input: string): string {
  let h1 = 0x811c9dc5;
  let h2 = 0x01000193;
  for (let i = 0; i < input.length; i++) {
    const c = input.charCodeAt(i);
    h1 ^= c;
    h1 = Math.imul(h1, 0x01000193) >>> 0;
    h2 ^= c + i;
    h2 = Math.imul(h2, 0x85ebca6b) >>> 0;
  }
  return (h1 >>> 0).toString(16).padStart(8, '0') + (h2 >>> 0).toString(16).padStart(8, '0');
}

/* -------------------------------------------------------------------------- */
/* Graph analysis                                                              */
/* -------------------------------------------------------------------------- */

export type GraphAnalysis = {
  /** Node ids in a deterministic execution order (Kahn, ties broken by first appearance). */
  order: string[];
  /** node id -> resolved input expressions, computed lazily per wire. */
  byId: Map<string, ShaderNode>;
  diagnostics: Diagnostic[];
  /** The node whose output is the final image. */
  outputNode?: string;
  /** Ports that are wired more than once (last write wins, but it is a smell). */
  duplicateInputs: string[];
  /**
   * Schema capabilities required by the nodes that actually reach the output.
   *
   * Computed over the *reachable* set, not the whole graph, for the same reason dead nodes are
   * reported rather than emitted: a `normal` sitting in a disconnected branch must not change
   * the shape of the emitted shader. Deleting the last one has to return the emission to the
   * quad form exactly, or every existing document would start emitting a surface shader it
   * cannot run.
   */
  needs: {
    matrix: boolean;
    derivatives: boolean;
    qualityScaled: boolean;
    surface: boolean;
    /** A reachable node samples the host image channel (iChannel0). */
    channel: boolean;
  };
};
export function analyzeGraph(
  graph: ShaderGraph,
  catalog: Record<NodeKind, NodeSchema>,
): GraphAnalysis {
  const diagnostics: Diagnostic[] = [];
  const byId = new Map<string, ShaderNode>();
  const order: string[] = [];
  const duplicateInputs: string[] = [];

  for (const n of graph.nodes) {
    if (byId.has(n.id)) {
      diagnostics.push({
        severity: 'error',
        code: 'IR001_DUPLICATE_ID',
        message: `Duplicate node id \`${n.id}\`.`,
        at: { node: n.id },
        fix: 'Node ids must be unique; regenerate with addNode().',
      });
      continue;
    }
    if (!catalog[n.kind]) {
      diagnostics.push({
        severity: 'error',
        code: 'IR002_UNKNOWN_KIND',
        message: `Unknown node kind \`${n.kind}\` on node \`${n.id}\`.`,
        at: { node: n.id },
        fix: `Known kinds: ${Object.keys(catalog).join(', ')}`,
      });
      continue;
    }
    /*
     * --- a node that carries its own body ------------------------------------
     *
     * `custom_glsl` is the only kind that does, and it is checked here rather than at emit
     * time for two reasons, both of which are the "silent failure" pattern this file keeps
     * running into:
     *
     *   • `emitBodies` skips nodes that are not reachable from the output, so a broken body on
     *     a disconnected node would never be linted at all — the diagnostic would appear the
     *     moment someone wired it up, and not before.
     *   • a `custom_glsl` node with **no** body falls back to the schema's pass-through
     *     (`return ${a}`). That is a deliberate fallback — a half-built graph should still
     *     emit, so the failure is "nothing happened" rather than a syntax error — but a
     *     pass-through presented as success is exactly the false-success this project hunts.
     *     So it is an error with a fix, not a shrug.
     */
    if (n.body !== undefined || n.kind === 'custom_glsl') {
      if (n.kind !== 'custom_glsl') {
        diagnostics.push({
          severity: 'warning',
          code: 'IR035_STRAY_BODY',
          message: `Node \`${n.id}\` (${n.kind}) carries a \`body\`, which the emitter will use instead of the catalog's.`,
          at: { node: n.id },
          fix: 'Only `custom_glsl` is meant to carry a body. Remove it to use the catalog template.',
        });
      }
      const body = n.body?.trim();
      if (n.kind === 'custom_glsl' && !body) {
        diagnostics.push({
          severity: 'error',
          code: 'IR034_CUSTOM_BODY_EMPTY',
          message: `\`custom_glsl\` node \`${n.id}\` has no \`body\`, so it emits a pass-through and does nothing.`,
          at: { node: n.id },
          fix: 'Set `node.body` to restricted-GLSL ending in `return vec4(...)`, or remove the node.',
        });
      } else if (body) {
        // The same lint the catalog bodies get, run on a per-instance body. Warnings are
        // attributed to the node so a reader can tell which snippet produced them.
        for (const d of lintSubset(body, `node ${n.id} (\`body\`)`)) {
          diagnostics.push({ ...d, at: { node: n.id, ...(d.at ?? {}) } });
        }
      }
    }

    byId.set(n.id, n);
  }

  // --- port / type checking -------------------------------------------------
  for (const n of byId.values()) {
    const schema = catalog[n.kind];

    for (const [key, val] of Object.entries(n.params ?? {})) {
      const port = schema.inputs.find((p) => p.name === key);
      if (!port) {
        diagnostics.push({
          severity: 'error',
          code: 'IR003_UNKNOWN_PORT',
          message: `Node \`${n.id}\` (${n.kind}) has no input \`${key}\`.`,
          at: { node: n.id, port: key },
          fix: `Valid inputs: ${schema.inputs.map((p) => p.name).join(', ') || '(none)'}`,
        });
        continue;
      }
      const d = checkParamAgainstType(n.id, key, val, port.type);
      if (d) diagnostics.push(d);
    }

    for (const [key, wire] of Object.entries(n.inputs ?? {})) {
      const port = schema.inputs.find((p) => p.name === key);
      if (!port) {
        diagnostics.push({
          severity: 'error',
          code: 'IR003_UNKNOWN_PORT',
          message: `Node \`${n.id}\` (${n.kind}) has no input \`${key}\`.`,
          at: { node: n.id, port: key },
        });
        continue;
      }
      const src = byId.get(wire.node);
      if (!src) {
        diagnostics.push({
          severity: 'error',
          code: 'IR004_DANGLING_WIRE',
          message: `Node \`${n.id}\` input \`${key}\` reads from missing node \`${wire.node}\`.`,
          at: { node: n.id, port: key },
          fix: 'Remove the wire or re-create the source node.',
        });
        continue;
      }
      const srcSchema = catalog[src.kind];
      const outName = wire.port ?? srcSchema.primaryOut ?? srcSchema.outputs[0]?.name;
      const outPort = srcSchema.outputs.find((p) => p.name === outName);
      if (!outPort) {
        diagnostics.push({
          severity: 'error',
          code: 'IR005_UNKNOWN_OUTPUT',
          message: `Node \`${wire.node}\` (${src.kind}) has no output \`${outName}\`.`,
          at: { node: n.id, port: key },
          fix: `Valid outputs: ${srcSchema.outputs.map((p) => p.name).join(', ')}`,
        });
        continue;
      }
      if (!canConnect(outPort.type, port.type)) {
        diagnostics.push({
          severity: 'error',
          code: 'IR006_TYPE_MISMATCH',
          message: `Type mismatch: \`${wire.node}.${outName}\` is ${outPort.type} but \`${n.id}.${key}\` expects ${port.type}.`,
          at: { node: n.id, port: key },
          fix: mismatchFix(outPort.type, port.type),
        });
      }
    }
  }

  // --- cycle-safe deterministic topological sort ----------------------------
  const indeg = new Map<string, number>();
  const adj = new Map<string, string[]>();
  for (const id of byId.keys()) {
    indeg.set(id, 0);
    adj.set(id, []);
  }
  for (const n of byId.values()) {
    for (const wire of Object.values(n.inputs ?? {})) {
      if (!byId.has(wire.node)) continue;
      adj.get(wire.node)!.push(n.id);
      indeg.set(n.id, (indeg.get(n.id) ?? 0) + 1);
    }
  }
  // Seed with the original declaration order so emission is stable.
  const queue = graph.nodes.map((n) => n.id).filter((id) => byId.has(id) && indeg.get(id) === 0);
  while (queue.length) {
    const id = queue.shift()!;
    order.push(id);
    // Every out-edge decrements its target's in-degree. There is deliberately **no**
    // `seenEdge` de-duplication here: `indeg` was built by counting each wire, so a node
    // that feeds two inputs of the same consumer contributes two units of in-degree and
    // must be decremented twice. Collapsing those into one edge (by `from->to` pair)
    // leaves the consumer's in-degree permanently above zero, which then reports a
    // **false `IR007_CYCLE`** and drops the node and everything downstream of it — a
    // silently empty shader from a perfectly valid DAG. Kahn's algorithm requires the
    // edge multiset, not the edge set.
    for (const next of adj.get(id) ?? []) {
      const d = (indeg.get(next) ?? 0) - 1;
      indeg.set(next, d);
      if (d === 0) queue.push(next);
    }
  }
  if (order.length !== byId.size) {
    const stuck = [...byId.keys()].filter((id) => !order.includes(id));
    diagnostics.push({
      severity: 'error',
      code: 'IR007_CYCLE',
      message: `Graph contains a cycle among: ${stuck.join(', ')}.`,
      fix: 'Break the cycle — the IR is a DAG by definition; feedback needs an explicit ping-pong pass.',
    });
  }

  // --- output node ----------------------------------------------------------
  const outputs = graph.nodes.filter((n) => n.kind === 'output' && byId.has(n.id));
  if (outputs.length === 0) {
    diagnostics.push({
      severity: 'error',
      code: 'IR008_NO_OUTPUT',
      message: 'Graph has no `output` node.',
      fix: 'Add an `output` node and wire a vec4 into its `color` input.',
    });
  } else if (outputs.length > 1) {
    diagnostics.push({
      severity: 'warning',
      code: 'IR009_MULTIPLE_OUTPUTS',
      message: `${outputs.length} output nodes found; using \`${outputs[0]!.id}\`.`,
      fix: 'Remove the extra output nodes or split them into separate documents.',
    });
  }

  // --- duplicate input writes ----------------------------------------------
  const edgeCount = new Map<string, number>();
  for (const e of graph.edges) {
    const k = `${e.to.node}.${e.to.port}`;
    edgeCount.set(k, (edgeCount.get(k) ?? 0) + 1);
  }
  for (const [k, c] of edgeCount) if (c > 1) duplicateInputs.push(k);
  if (duplicateInputs.length) {
    diagnostics.push({
      severity: 'warning',
      code: 'IR010_DUPLICATE_WIRES',
      message: `Multiple edges target the same input: ${duplicateInputs.join(', ')}.`,
      fix: 'Keep graph.edges in sync with nodes[].inputs — call normalizeEdges() after patching.',
    });
  }

  // --- reachability: report dead branches rather than silently dropping them --
  if (outputs.length) {
    const needed = new Set<string>();
    const walk = (id: string) => {
      if (needed.has(id)) return;
      needed.add(id);
      for (const w of Object.values(byId.get(id)?.inputs ?? {})) walk(w.node);
    };
    walk(outputs[0]!.id);
    const dead = [...byId.keys()].filter((id) => !needed.has(id));
    if (dead.length) {
      diagnostics.push({
        severity: 'warning',
        code: 'IR011_DEAD_NODES',
        message: `Nodes not reachable from the output: ${dead.join(', ')}. They will not be emitted.`,
        fix: 'Connect them, or remove them to keep the graph honest about what it does.',
      });
    }
  }

  /*
   * --- required capabilities, over the reachable set ------------------------
   *
   * Union of every `needs` flag declared by a node that actually reaches the output. The
   * reachable set is the same one the dead-node walk just computed, and it has to be: a
   * `normal` in a disconnected branch would otherwise flip the whole emission to a surface
   * shader that the connected graph does not need and cannot run as a quad.
   */
  const needs = {
    matrix: false,
    derivatives: false,
    qualityScaled: false,
    surface: false,
    channel: false,
  };
  if (outputs.length) {
    const reachable = new Set<string>();
    const walk = (id: string) => {
      if (reachable.has(id)) return;
      reachable.add(id);
      for (const w of Object.values(byId.get(id)?.inputs ?? {})) walk(w.node);
    };
    walk(outputs[0]!.id);
    for (const id of reachable) {
      const node = byId.get(id);
      const schema = catalog[node?.kind as NodeKind];
      /*
       * A **per-instance body** is part of the reachable set's capability surface, not just its
       * text: a `custom_glsl` snippet that reads `iNormal` needs the surface entry point every bit
       * as much as a `normal` node does, and nothing else would say so. Before this, such a
       * document emitted the quad header, gave no diagnostic, and failed to compile — the
       * declared-and-silent failure this project spends most of its effort hunting.
       *
       * Checked textually because that is what the body is: free text in the restricted subset,
       * whose only contract is the alias list. Word-boundary match, so an author's own
       * `iNormalised` local does not trip it.
       */
      const body = node?.body;
      if (body && SURFACE_HOST_SYMBOLS.some((s) => new RegExp(`\\b${s}\\b`).test(body))) {
        needs.surface = true;
      }
      if (!schema?.needs) continue;
      if (schema.needs.matrix) needs.matrix = true;
      if (schema.needs.derivatives) needs.derivatives = true;
      if (schema.needs.qualityScaled) needs.qualityScaled = true;
      if (schema.needs.surface) needs.surface = true;
      if (schema.needs.channel) needs.channel = true;
    }
  }

  return { order, byId, diagnostics, outputNode: outputs[0]?.id, duplicateInputs, needs };
}

/** Widening rules: a float may feed a vecN input (splat), never the reverse. */
export function canConnect(from: ShaderType, to: ShaderType): boolean {
  if (from === to) return true;
  if (from === 'float' && (to === 'vec2' || to === 'vec3' || to === 'vec4')) return true;
  if (from === 'int' && (to === 'float' || to === 'vec2' || to === 'vec3' || to === 'vec4')) return true;
  if (from === 'bool' && to === 'float') return true;
  return false;
}

function mismatchFix(from: ShaderType, to: ShaderType): string {
  const a = TYPE_ARITY[from];
  const b = TYPE_ARITY[to];
  if (a > b) return `Insert a swizzle (e.g. .${'xyzw'.slice(0, b)}) or a \`vec_op\` node to reduce ${from} to ${to}.`;
  return `Insert a \`vec_op\`/constructor node to widen ${from} to ${to}.`;
}

function checkParamAgainstType(
  node: string,
  key: string,
  val: ParamValue,
  expect: ShaderType,
): Diagnostic | null {
  if (val.kind === 'param') {
    if (val.type !== expect && !canConnect(val.type, expect)) {
      return {
        severity: 'error',
        code: 'IR012_PARAM_TYPE',
        message: `Uniform \`${val.name}\` is ${val.type} but \`${node}.${key}\` expects ${expect}.`,
        at: { node, port: key },
      };
    }
    return null;
  }
  const map: Record<string, ShaderType> = {
    float: 'float', int: 'int', bool: 'bool', vec2: 'vec2', vec3: 'vec3', vec4: 'vec4', color: 'vec4',
  };
  const actual = map[val.kind]!;
  if (!canConnect(actual, expect)) {
    return {
      severity: 'error',
      code: 'IR013_LITERAL_TYPE',
      message: `Literal is ${actual} but \`${node}.${key}\` expects ${expect}.`,
      at: { node, port: key },
      fix: mismatchFix(actual, expect),
    };
  }
  if ((val.kind === 'float' || val.kind === 'int') && val.min !== undefined && val.max !== undefined && val.min > val.max) {
    return {
      severity: 'warning',
      code: 'IR014_BAD_RANGE',
      message: `\`${node}.${key}\` has min > max (${val.min} > ${val.max}).`,
      at: { node, port: key },
    };
  }
  return null;
}

/* -------------------------------------------------------------------------- */
/* Uniforms                                                                    */
/* -------------------------------------------------------------------------- */

/**
 * Collect every `expose_param` binding into a de-duplicated, stably ordered list.
 * Duplicate names with conflicting types are an error — that is a real bug that
 * would otherwise surface as a cryptic linker failure in one target only.
 */
export function collectUniforms(graph: ShaderGraph, order: string[], byId: Map<string, ShaderNode>): {
  uniforms: UniformDecl[];
  diagnostics: Diagnostic[];
} {
  const uniforms: UniformDecl[] = [];
  const byName = new Map<string, UniformDecl>();
  const diagnostics: Diagnostic[] = [];

  for (const id of order) {
    const n = byId.get(id);
    if (!n) continue;
    for (const [key, val] of Object.entries(n.params ?? {})) {
      if (val.kind !== 'param') continue;
      const existing = byName.get(val.name);
      if (existing) {
        if (existing.type !== val.type) {
          diagnostics.push({
            severity: 'error',
            code: 'IR020_UNIFORM_TYPE_CONFLICT',
            message: `Uniform \`${val.name}\` declared as both ${existing.type} (${existing.origin.node}.${existing.origin.param}) and ${val.type} (${id}.${key}).`,
            at: { node: id, port: key },
            fix: 'Give one of them a distinct uniform name.',
          });
        }
        continue;
      }
      const decl: UniformDecl = {
        name: val.name,
        type: val.type,
        value: val.value,
        min: val.min,
        max: val.max,
        step: val.step,
        label: val.label,
        group: val.group,
        origin: { node: id, param: key },
      };
      byName.set(val.name, decl);
      uniforms.push(decl);
    }
  }
  return { uniforms, diagnostics };
}

/** Serialize a numeric value into GLSL source with the right arity and decimal points. */
export function literalOf(type: ShaderType, value: number | boolean | number[]): string {
  const arr = Array.isArray(value) ? value : [value];
  const num = (x: number | boolean): string => {
    if (typeof x === 'boolean') return x ? '1.0' : '0.0';
    if (Number.isInteger(x) && type !== 'int') return x.toFixed(1);
    if (Number.isInteger(x)) return String(x);
    // Avoid exponent notation: some targets reject `1e-7` without a decimal point.
    let s = x.toString();
    if (s.includes('e') || s.includes('E')) s = x.toFixed(6).replace(/0+$/, '');
    return s.includes('.') ? s : `${s}.0`;
  };
  switch (type) {
    case 'float':
      return num(arr[0]!);
    case 'int':
      return String(Math.round(Number(arr[0])));
    case 'bool':
      return arr[0] ? 'true' : 'false';
    default: {
      const n = TYPE_ARITY[type];
      const parts = Array.from({ length: n }, (_, i) => num(arr[i] ?? arr[0] ?? 0));
      return `${type}(${parts.join(', ')})`;
    }
  }
}

/* -------------------------------------------------------------------------- */
/* Placeholder expansion                                                       */
/* -------------------------------------------------------------------------- */

export type ExpandContext = {
  node: ShaderNode;
  schema: NodeSchema;
  /** input name -> already-rendered expression string (in GLSL form). */
  inputs: Record<string, string>;
  octaveLimit: number;
};

/**
 * Substitute `${name}` in a node body.
 *
 * Deliberately a plain string replace over a closed set of names: the catalog is
 * authored by hand (or by an LLM) and a template engine with expressions would be a
 * second language to get wrong. Unknown placeholders are reported, not ignored.
 */
export function expandTemplate(body: string, ctx: ExpandContext): { code: string; diagnostics: Diagnostic[] } {
  const diagnostics: Diagnostic[] = [];
  const code = body.replace(/\$\{([A-Za-z_][A-Za-z0-9_]*)\}/g, (_m, name: string) => {
    if (name === 'OCTAVE_LIMIT') return String(ctx.octaveLimit);
    if (name === 'node_id') return ctx.node.id;
    const v = ctx.inputs[name];
    if (v === undefined) {
      diagnostics.push({
        severity: 'error',
        code: 'IR030_UNKNOWN_PLACEHOLDER',
        message: `Node \`${ctx.node.id}\` (${ctx.schema.kind}) body references \`\${${name}}\`, which is not one of its inputs.`,
        at: { node: ctx.node.id },
        fix: `Declared inputs: ${ctx.schema.inputs.map((p) => p.name).join(', ') || '(none)'}`,
      });
      return `/* unresolved:${name} */ 0.0`;
    }
    return v;
  });
  return { code, diagnostics };
}

/** Quality tier -> iteration budget. Also the perf guard rail surfaced in the UI. */
export function octavesFor(doc: ShaderDoc): number {
  const tierBase: Record<string, number> = { low: 3, medium: 5, high: 7, ultra: OCTAVE_LIMIT };
  return Math.min(tierBase[doc.render.qualityTier] ?? 5, OCTAVE_LIMIT, doc.render.maxIterations);
}

/* -------------------------------------------------------------------------- */
/* Body emission (target-independent)                                          */
/* -------------------------------------------------------------------------- */

export type EmittedBody = {
  /** Lines of GLSL, in execution order, using `L_<node>_<port>` locals. */
  lines: string[];
  /** Local name for a given node output port. */
  localName: (nodeId: string, port: string) => string;
  diagnostics: Diagnostic[];
};

/**
 * Turn the graph into a straight-line GLSL block. Every node output becomes a local,
 * which keeps the generated code readable (people *will* read it), makes stack traces
 * and shader-debugger line numbers meaningful, and means the rewriter only ever sees
 * simple statements.
 */
export function emitBodies(
  doc: ShaderDoc,
  analysis: GraphAnalysis,
  catalog: Record<NodeKind, NodeSchema>,
): EmittedBody {
  const lines: string[] = [];
  const diagnostics: Diagnostic[] = [...analysis.diagnostics];
  const localName = (nodeId: string, port: string) =>
    `L_${sanitizeId(nodeId)}_${sanitizeId(port)}`;
  const octaveLimit = octavesFor(doc);

  const reachable = new Set<string>();
  if (analysis.outputNode) {
    const walk = (id: string) => {
      if (reachable.has(id)) return;
      reachable.add(id);
      for (const w of Object.values(analysis.byId.get(id)?.inputs ?? {})) walk(w.node);
    };
    walk(analysis.outputNode);
  }

  for (const id of analysis.order) {
    const node = analysis.byId.get(id)!;
    const schema = catalog[node.kind];
    if (!reachable.has(id)) continue;

    // Resolve each declared input to an expression.
    const inputExprs: Record<string, string> = {};
    for (const port of schema.inputs) {
      const wire = node.inputs?.[port.name];
      const lit = node.params?.[port.name];
      if (wire && analysis.byId.has(wire.node)) {
        const srcSchema = catalog[analysis.byId.get(wire.node)!.kind];
        const outName = wire.port ?? srcSchema.primaryOut ?? srcSchema.outputs[0]!.name;
        const srcType = srcSchema.outputs.find((p) => p.name === outName)?.type ?? 'float';
        let expr = localName(wire.node, outName);
        // Splat/widen on the wire (float -> vecN), or narrow with a swizzle.
        if (srcType !== port.type) expr = coerce(expr, srcType, port.type, diagnostics, id, port.name);
        inputExprs[port.name] = expr;
      } else if (lit) {
        inputExprs[port.name] = paramExpr(lit);
      } else if (port.default) {
        inputExprs[port.name] = paramExpr(port.default);
      } else {
        diagnostics.push({
          severity: 'error',
          code: 'IR031_MISSING_INPUT',
          message: `Node \`${id}\` (${node.kind}) input \`${port.name}\` (${port.type}) is neither wired nor given a literal.`,
          at: { node: id, port: port.name },
          fix: `Provide \`params.${port.name}\` or wire \`inputs.${port.name}\`.`,
        });
        inputExprs[port.name] = literalOf(port.type, 0);
      }
    }

    /*
     * `node.body` wins over the schema template.
     *
     * Exactly one kind uses it — `custom_glsl` — but the branch lives here rather than in a
     * special case inside that kind, because this is the single point where "the code for this
     * node" is decided. Every other kind has a constant body, which is what makes the rewriter
     * provable; the escape hatch is the acknowledgement that a catalog cannot be complete, and
     * it is confined to this one line.
     */
    const expanded = expandTemplate(node.body ?? schema.glsl, {
      node,
      schema,
      inputs: inputExprs,
      octaveLimit,
    });
    diagnostics.push(...expanded.diagnostics);

    // The body's last `return <expr>;` becomes the port assignments.
    const { statements, resultExpr } = splitReturn(expanded.code);
    // Independent custom layers routinely use the same locals (p, hi, normal, etc.).
    // Keep only the output in graph scope; all snippet temporaries belong to the node.
    if (node.kind === 'custom_glsl' && (schema.outputs.length <= 1 ||
        (schema.outputs.length <= 4 && schema.outputs.every(p => TYPE_ARITY[p.type] === 1)))) {
      for (const port of schema.outputs) {
        lines.push(`    ${glslType(port.type)} ${localName(id, port.name)} = ${literalOf(port.type, 0)};`);
      }
      lines.push(`    { // custom_glsl ${sanitizeId(id)}`);
      const nonempty = statements.filter(s => s.trim());
      const margin = nonempty.length ? Math.min(...nonempty.map(s => s.length - s.trimStart().length)) : 0;
      for (const s of statements) lines.push(`      ${s.slice(margin)}`);
      if (schema.outputs.length > 1) lines.push(`      vec4 nodeResult = ${resultExpr};`);
      schema.outputs.forEach((port, i) => lines.push(`      ${localName(id, port.name)} = ${schema.outputs.length === 1 ? resultExpr : `nodeResult.${'xyzw'[i]}`};`));
      if (!schema.outputs.length) lines.push(`      ${resultExpr};`);
      lines.push('    }');
      continue;
    }
    for (const s of statements) lines.push(`    ${s}`);

    const outs = schema.outputs;
    if (outs.length === 0) {
      // A node with no ports is a statement, not a value — `output` is the only one in
      // the catalog. Emitting nothing keeps a hand-written IR from crashing the emitter.
      lines.push(`    ${resultExpr};`);
      continue;
    }

    if (outs.length === 1) {
      lines.push(`    ${glslType(outs[0]!.type)} ${localName(id, outs[0]!.name)} = ${resultExpr};`);
      continue;
    }

    // --- multiple outputs -----------------------------------------------------
    //
    // Contract: a multi-output node's body returns a **vec4**, and the ports are its
    // `.x/.y/.z/.w` in declaration order. A fixed vec4 is the only shape the token
    // rewriter can lower identically to all six targets — a struct or an out-parameter
    // would each need per-language handling. Declaring more than four ports, or ports
    // whose arity is not 1, is therefore a schema bug and is reported as one instead of
    // emitting code that silently aliases the whole container into every port.
    const tmp = `tmp_${sanitizeId(id)}`;
    const portsOk = outs.length <= 4 && outs.every((p) => TYPE_ARITY[p.type] === 1);
    if (!portsOk) {
      diagnostics.push({
        severity: 'error',
        code: 'IR033_BAD_MULTI_OUTPUT',
        message: `Node \`${id}\` (${node.kind}) declares ${outs.length} outputs (${outs.map((p) => `${p.name}: ${p.type}`).join(', ')}).`,
        at: { node: id },
        fix: 'Multi-output nodes must return a vec4 and declare at most four scalar ports; split the node otherwise.',
      });
      lines.push(`    vec4 ${tmp} = vec4(${resultExpr});`);
    } else {
      lines.push(`    vec4 ${tmp} = ${resultExpr};`);
    }
    outs.forEach((port, idx) => {
      const comp = 'xyzw'[idx] ?? 'x';
      lines.push(`    ${glslType(port.type)} ${localName(id, port.name)} = ${tmp}.${comp};`);
    });
  }

  return { lines, localName, diagnostics };
}

/** Node ids are already tame, but a hand-written or LLM-authored IR might not be. */
export function sanitizeId(id: string): string {
  const s = id.replace(/[^A-Za-z0-9_]/g, '_');
  return /^[0-9]/.test(s) ? `n${s}` : s;
}

function glslType(t: ShaderType): string {
  return t;
}

/** float/vecN conversions, GLSL spelling. */
function coerce(
  expr: string,
  from: ShaderType,
  to: ShaderType,
  diagnostics: Diagnostic[],
  node: string,
  port: string,
): string {
  if (from === to) return expr;
  if (from === 'int' && to === 'float') return `float(${expr})`;
  if (from === 'bool' && to === 'float') return `float(${expr})`;
  if (from === 'float' || from === 'int') {
    // Splat a scalar across a vector. The component count must match the destination
    // exactly: `vec2(x, x, x)` is a GLSL compile error, and getting this wrong is
    // invisible until a shader compiler sees it.
    if (to === 'vec2' || to === 'vec3' || to === 'vec4') {
      const n = TYPE_ARITY[to];
      const arg = from === 'int' ? `float(${expr})` : expr;
      return `${to}(${Array.from({ length: n }, () => arg).join(', ')})`;
    }
  }
  const order = 'xyzw';
  const n = TYPE_ARITY[to];
  if (TYPE_ARITY[from] > n && (to === 'float' || to === 'vec2' || to === 'vec3')) {
    return n === 1 ? `${expr}.x` : `${expr}.${order.slice(0, n)}`;
  }
  diagnostics.push({
    severity: 'error',
    code: 'IR032_NO_COERCION',
    message: `No known conversion from ${from} to ${to} at \`${node}.${port}\`.`,
    at: { node, port },
  });
  return expr;
}

/** Render a ParamValue as a GLSL expression (uniform reference or literal). */
export function paramExpr(v: ParamValue): string {
  if (v.kind === 'param') {
    if (v.type === 'float' || v.type === 'int' || v.type === 'bool') return v.name;
    return v.name;
  }
  if (v.kind === 'color') return literalOf('vec4', v.value);
  return literalOf(v.kind === 'vec2' ? 'vec2' : v.kind === 'vec3' ? 'vec3' : v.kind === 'vec4' ? 'vec4' : v.kind, v.value as never);
}

/**
 * Split a node body into `statements` + the returned expression.
 * Node bodies always end in a single `return <expr>;` — enforced here so catalog
 * authors get a precise error instead of a mysterious syntax error six stages later.
 */
export function splitReturn(body: string): { statements: string[]; resultExpr: string } {
  const trimmed = body.trim();
  const retIdx = trimmed.lastIndexOf('return');
  if (retIdx === -1) {
    return { statements: trimmed.split('\n').map((l) => l.trim()).filter(Boolean), resultExpr: '0.0' };
  }
  const stmts = trimmed.slice(0, retIdx).split('\n').map((l) => l.trim()).filter(Boolean);
  let rest = trimmed.slice(retIdx + 'return'.length).trim();
  if (rest.endsWith(';')) rest = rest.slice(0, -1);
  // A return spanning multiple lines (a wrapped vec4(...) call) is fine — keep it as one expr.
  return { statements: stmts, resultExpr: rest.replace(/\s*\n\s*/g, ' ') };
}

/* -------------------------------------------------------------------------- */
/* Top-level emit                                                              */
/* -------------------------------------------------------------------------- */

export type TargetEmitter = (args: {
  doc: ShaderDoc;
  analysis: GraphAnalysis;
  catalog: Record<NodeKind, NodeSchema>;
  bodies: EmittedBody;
  uniforms: UniformDecl[];
  /** Rewrite a GLSL fragment into the target language. */
  rw: (src: string, context: string) => RewriteResult;
}) => { source: string; sidecars?: { name: string; content: string }[]; diagnostics?: Diagnostic[] };

/**
 * Emit source for one target.
 *
 * Diagnostics from *every* phase are merged and returned together: a caller should
 * never have to run the pipeline twice to learn about a type error and a rewrite
 * limitation. `emission.diagnostics` is the single place the agent loop looks.
 */
export function emit(doc: ShaderDoc, opts: EmitOptions): Emission {
  const diagnostics: Diagnostic[] = [];
  const analysis = analyzeGraph(doc.graph, opts.catalog);
  const { uniforms, diagnostics: uDiags } = collectUniforms(doc.graph, analysis.order, analysis.byId);
  diagnostics.push(...uDiags);

  const bodies = emitBodies(doc, analysis, opts.catalog);
  diagnostics.push(...bodies.diagnostics);

  /*
   * `render.feedback` asks the host to feed the previous frame into the image channel. If no
   * reachable node reads that channel, the request cannot do anything: the emitted source is
   * identical, the host sets up an offscreen target and a ping-pong pair, and the picture does
   * not change. That is a mistake worth naming — it costs the host real work and it looks, from
   * the outside, exactly like feedback being broken.
   *
   * A warning, not an error: the document still emits and still renders, and the intent behind
   * the flag (a trail, an accumulation) may be one `texture` node away from working.
   */
  if (doc.render.feedback && !analysis.needs.channel) {
    diagnostics.push({
      severity: 'warning',
      code: 'IR036_FEEDBACK_UNUSED',
      message:
        '`render.feedback` is true, but no reachable node samples the image channel ' +
        '(`iChannel0`), so feeding the previous frame back cannot change the result.',
      fix:
        'Wire a `texture` node into the graph (that is the node that reads the channel), or ' +
        'set `render.feedback` to false.',
    });
  }

  /*
   * A wire into a port the node's body never reads.
   *
   * `shape.uv` is the one such port in the shipped catalog, and it is deliberate — accepted "for
   * pipeline symmetry" and documented as ignored, because every other node that takes a
   * coordinate takes one. Deliberate or not, the user experience is identical to a bug: connect
   * it, see nothing change, and have no way to tell. So the wire is reported.
   *
   * Derived from the body rather than listed, which means a future node with a dead port is
   * covered the moment it ships — and a body that *stops* reading a port it declares starts
   * warning immediately instead of quietly dropping the signal.
   *
   * `custom_glsl` is exempt on purpose: its body is per instance, so its ports are a fixed slot
   * set and a snippet that uses only `a` is completely normal.
   */
  for (const node of doc.graph.nodes) {
    const schema = opts.catalog[node.kind];
    if (!schema || node.kind === ('custom_glsl' as NodeKind)) continue;
    for (const port of schema.inputs) {
      if (!node.inputs?.[port.name]) continue;
      if (schema.glsl.includes(`\${${port.name}}`)) continue;
      diagnostics.push({
        severity: 'warning',
        code: 'IR037_INERT_INPUT',
        message:
          `\`${node.id}.${port.name}\` is wired, but the \`${node.kind}\` node never reads that ` +
          `input — the wire cannot change the result.`,
        at: { node: node.id, port: port.name },
        fix:
          `Remove the wire, or use a node that does read a coordinate (for \`shape\`, the mask it ` +
          `composites comes from \`sdf.mask\` — the shape is positioned by feeding \`sdf.uv\`).`,
      });
    }
  }

  const rw = (src: string, context: string): RewriteResult => {
    const lint = lintSubset(src, context);
    const res = rewriteSource(src, opts.target, context);
    diagnostics.push(...lint, ...res.diagnostics);
    return res;
  };

  // Lazily imported to keep the IR package free of target-specific code paths at
  // module-eval time (the CLI may only ever need one of them). An explicit `emitter` wins over
  // the registry so that a caller can emit to its own lowering pass without touching global state.
  const emitter = opts.emitter ?? TARGET_EMITTERS[opts.target];
  if (!emitter) {
    diagnostics.push({
      severity: 'error',
      code: 'EM000_UNKNOWN_TARGET',
      message: `No emitter registered for target \`${opts.target}\`.`,
    });
    return { target: opts.target, source: '', diagnostics, uniforms, symbols: [], hash: '', needs: analysis.needs };
  }

  /*
   * A surface document for an emitter that has no surface entry point.
   *
   * Refused rather than attempted. The alternative is worse in a way that is hard to notice:
   * the node body mentions `iNormal`, the header never declares it, `emit()` returns zero
   * errors, and the failure surfaces later as a compiler error inside Unity or UE — where the
   * trail back to "the emitter was not updated" is cold. An error here is actionable now.
   */
  if (analysis.needs.surface && !SURFACE_CAPABLE.has(opts.target)) {
    diagnostics.push({
      severity: 'error',
      code: 'EM002_SURFACE_UNSUPPORTED',
      message:
        `This graph reads a surface input (normal / world_pos / view_dir / light_dir), but the ` +
        `\`${opts.target}\` emitter does not emit a surface entry point yet — it would produce a ` +
        `shader that references symbols it never declares.`,
      fix:
        `Emit this document to one of: ${[...SURFACE_CAPABLE].join(', ')}. ` +
        `Or drop the surface nodes to make it a plain 2D pattern, which every target supports.`,
    });
    return { target: opts.target, source: '', diagnostics, uniforms, symbols: [], hash: '', needs: analysis.needs };
  }

  const { source, sidecars, diagnostics: tDiags } = emitter({
    doc, analysis, catalog: opts.catalog, bodies, uniforms, rw,
  });
  diagnostics.push(...(tDiags ?? []));

  return {
    target: opts.target,
    source,
    sidecars,
    diagnostics,
    uniforms,
    symbols: bodies.lines
      .map((l) => /(\w+)\s+(L_\w+)\s*=/.exec(l))
      .filter((m): m is RegExpExecArray => !!m)
      .map((m) => ({ name: m[2]!, type: m[1]! as ShaderType })),
    hash: stableHash(`${opts.target}\u0000${source}`),
    needs: analysis.needs,
  };
}

/** Registered by the target modules; kept here so `emit` has no import cycle. */
export const TARGET_EMITTERS: Partial<Record<TargetFormat, TargetEmitter>> = {};

/**
 * Targets whose emitter supplies a **surface entry point** when a graph needs one.
 *
 * A surface document reads `iNormal` / `iWorldPos` / `iViewDir` / `iLightDir`. Those are not
 * numbers the graph can invent: each target has to declare them and wire them to interpolated
 * vertex outputs. An emitter that has not been updated would happily write the node body
 * (`vec4(iNormal, 0.0)`) into a shader that never declares `iNormal` — source that looks
 * finished, reports zero errors, and fails to compile in the engine it was written for.
 *
 * So this set is a **gate, not a note**: a surface document for a target outside it is an
 * error rather than a silent piece of broken code. All six targets implement it now, which is
 * why the set is complete — the check stays because a *seventh* target would otherwise
 * reintroduce the hole, and because it documents the contract at the point where it is
 * enforced. `targets.test.ts` drives its per-target assertions from this set, so adding a
 * target moves it from the refusal side to the emission side automatically.
 */
export const SURFACE_CAPABLE: ReadonlySet<TargetFormat> = new Set<TargetFormat>([
  'glsl',
  'hlsl',
  'wgsl',
  'msl',
  'shaderlab',
  'ue',
]);

export function registerTarget(target: TargetFormat, emitter: TargetEmitter): void {
  TARGET_EMITTERS[target] = emitter;
}
