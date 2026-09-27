/**
 * AI4Shader —?Patcher
 * ===================
 *
 * The **only** sanctioned way to modify a `ShaderDoc`.
 *
 * The LLM never rewrites the IR wholesale. It emits a `ShaderPatch` —?a short list of
 * ops —?which is applied to a *copy* of the document, validated against the catalog,
 * and either accepted as a whole or rejected as a whole. That is what makes agent edits
 * reviewable (the patch is the diff), revertible (keep the previous doc), and structurally
 * incapable of corrupting a document (a rejected patch returns the input untouched).
 *
 * Invariants this module is responsible for:
 *
 *   1. PURITY.        Every function returns a new document; the input is never mutated,
 *                     not even transiently. Callers may freely `Object.freeze` a doc.
 *   2. DETERMINISM.   No randomness, no clocks in hashed content, no `Object.keys`
 *                     iteration whose order is not pinned by construction. The same
 *                     inputs give the same output on every run and every platform.
 *   3. ATOMICITY.     `applyPatch` is all-or-nothing. A half-applied patch is worse
 *                     than a rejected one: it costs an extra LLM round-trip to diagnose.
 *   4. NO DRIFT.      `graph.edges` is *derived* from `graph.nodes[].inputs`. Every
 *                     mutator ends with `normalizeEdges`, so the two can never disagree.
 *   5. HASH = CODEGEN. `doc.hash` covers exactly the fields the emitter reads. Editing a
 *                     title or a node's screen position must not invalidate a compile
 *                     cache; editing a param value must.
 *
 * `provenance.steps` is the lineage log: each accepted patch appends exactly one entry
 * built from `patch.summary` and `patch.author`.
 */

import {
  IR_VERSION,
  NODE_KINDS,
  type Diagnostic,
  type Edge,
  type NodeKind,
  type NodeSchema,
  type ParamValue,
  type PatchOp,
  type QualityTier,
  type ShaderDoc,
  type ShaderGraph,
  type ShaderNode,
  type ShaderPatch,
  type ShaderType,
  type WireRef,
} from './types.js';
import { analyzeGraph, canConnect, stableHash } from './emitter.js';
import { isReservedAnywhere, safeIdentifier } from './rewrite.js';

/* -------------------------------------------------------------------------- */
/* Catalog access                                                              */
/* -------------------------------------------------------------------------- */

/**
 * The catalog is injected, never imported: the IR package stays free of content and
 * stays unit-testable with a three-node stub. Accepted as a loose record because a
 * partial catalog (a test fixture, a future plugin set) is a legitimate input —?the
 * missing kinds are reported as `PT002_UNKNOWN_KIND`, not as a crash.
 */
export type Catalog = Partial<Record<NodeKind, NodeSchema>>;

/**
 * Diagnostics are *values*, not exceptions. Every failure in this module is reported as
 * a `Diagnostic` with enough context for the agent to fix it without a second pass.
 */
export type PatchResult = {
  doc: ShaderDoc;
  /** Number of ops that actually changed the document. 0 whenever `ok` is false. */
  applied: number;
  diagnostics: Diagnostic[];
  ok: boolean;
};

export type NodeInit = Omit<ShaderNode, 'id'> & { id?: string };

export type WireEndpoint = { node: string; port: string };
export type WireSource = { node: string; port?: string };

export type ExposeOptions = {
  name: string;
  min?: number;
  max?: number;
  step?: number;
  label?: string;
  group?: string;
};

export type ForkOptions = { parents?: string[] };

/* -------------------------------------------------------------------------- */
/* Diagnostic codes                                                            */
/* -------------------------------------------------------------------------- */

/**
 * Patcher diagnostic codes.
 *
 * `AGENT_CONTRACT.md` §5 fixes the namespaces: `IR0xx` = graph/type validation,
 * `PT0xx` = patch application. *Every* failure this module produces is a patch-application
 * failure, so every code here is `PT0xx` —?including the ones the emitter calls
 * `IR003/IR004/IR005/IR006/IR007`. Reusing the `IR` codes would make failure attribution
 * ambiguous: an agent reading `IR006_TYPE_MISMATCH` could not tell whether the document
 * was already broken or whether the patch just broke it, which is exactly the distinction
 * its retry policy depends on. (`IR014_BAD_RANGE` is the one exception and is documented
 * at its use site: it is a *warning*, and it is byte-identical to what `analyzeGraph`
 * emits for the same document, so duplicating it would double-report.)
 */
export const PT = {
  /** The op names a node that does not exist in the document. */
  UNKNOWN_NODE: 'PT001_UNKNOWN_NODE',
  /** The catalog has no schema for the kind, so ports cannot be checked. */
  UNKNOWN_KIND: 'PT002_UNKNOWN_KIND',
  /** `add_node` reused an id that is already taken. */
  DUPLICATE_ID: 'PT003_DUPLICATE_ID',
  /** `inline_param`/`expose_param` on a port that has no literal. */
  NO_VALUE: 'PT004_NO_VALUE',
  /** The port is wired, so a literal there would be silently ignored. */
  WIRED_PORT: 'PT005_WIRED_PORT',
  /** Uniform name is not a valid identifier. */
  BAD_NAME: 'PT006_BAD_NAME',
  /** Uniform name is already declared with a different type. */
  NAME_TAKEN: 'PT007_NAME_TAKEN',
  /** Uniform name is reserved in at least one target language. */
  RESERVED_NAME: 'PT008_RESERVED_NAME',
  /** A number that is not finite, or a min/max that cannot make a slider. */
  BAD_RANGE: 'PT009_BAD_RANGE',
  /** Malformed op: unknown `op`, missing field, wrong payload shape. */
  BAD_OP: 'PT010_BAD_OP',
  /** The patch was rejected as a whole; the document is unchanged. */
  NOT_ATOMIC: 'PT011_NOT_ATOMIC',
  /** Target port does not exist on the destination node's schema. */
  UNKNOWN_PORT: 'PT012_UNKNOWN_PORT',
  /** Source output port does not exist on the source node's schema. */
  UNKNOWN_OUTPUT: 'PT013_UNKNOWN_OUTPUT',
  /** Intrinsic type mismatch (this module's own `canConnect` check). */
  TYPE_MISMATCH: 'PT014_TYPE_MISMATCH',
  /** The wire would create a cycle; the IR is a DAG by definition. */
  CYCLE: 'PT015_CYCLE',
  /** A wire references a node that is not in the document. */
  DANGLING_WIRE: 'PT016_DANGLING_WIRE',
} as const;

/**
 * The emitter's code for an inverted slider range. Kept because it is a warning that
 * `analyzeGraph` will emit for the same document anyway —?inventing a second code for one
 * fact would show the user the same problem twice under two names.
 */
const IR014 = 'IR014_BAD_RANGE';

const KNOWN_KINDS: readonly string[] = NODE_KINDS;
const IDENT_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;

function err(code: string, message: string, at?: Diagnostic['at'], fix?: string): Diagnostic {
  return { severity: 'error', code, message, ...(at ? { at } : {}), ...(fix ? { fix } : {}) };
}

function warn(code: string, message: string, at?: Diagnostic['at'], fix?: string): Diagnostic {
  return { severity: 'warning', code, message, ...(at ? { at } : {}), ...(fix ? { fix } : {}) };
}

/* -------------------------------------------------------------------------- */
/* Internal helpers —?all pure                                                 */
/* -------------------------------------------------------------------------- */

function hasErr(ds: readonly Diagnostic[]): boolean {
  return ds.some((d) => d.severity === 'error');
}

function schemaOf(catalog: Catalog, kind: NodeKind): NodeSchema | undefined {
  return catalog[kind];
}

function nodeOf(doc: ShaderDoc, id: string): ShaderNode | undefined {
  return doc.graph.nodes.find((n) => n.id === id);
}

function inputPort(catalog: Catalog, kind: NodeKind, port: string) {
  return schemaOf(catalog, kind)?.inputs.find((p) => p.name === port);
}

function outputPort(catalog: Catalog, kind: NodeKind, port?: string) {
  const s = schemaOf(catalog, kind);
  if (!s) return undefined;
  const name = port ?? s.primaryOut ?? s.outputs[0]?.name;
  if (name === undefined) return undefined;
  return s.outputs.find((p) => p.name === name);
}

/** The concrete port name a `WireRef` resolves to. */
function resolvedPort(catalog: Catalog, kind: NodeKind, port?: string): string {
  return outputPort(catalog, kind, port)?.name ?? port ?? schemaOf(catalog, kind)?.primaryOut ?? '';
}

/** Replace one key of a `Record<string, T>`, dropping it when the value is `undefined`. */
function withKey<T>(rec: Record<string, T> | undefined, key: string, val: T | undefined): Record<string, T> {
  const out: Record<string, T> = { ...(rec ?? {}) };
  if (val === undefined) delete out[key];
  else out[key] = val;
  return out;
}

function withNode(doc: ShaderDoc, node: ShaderNode): ShaderDoc {
  return {
    ...doc,
    graph: { ...doc.graph, nodes: doc.graph.nodes.map((n) => (n.id === node.id ? node : n)) },
  };
}

/**
 * Deep structural equality for the shapes a patch can touch: plain objects, arrays and
 * primitives. Used to decide whether an op actually *changed* anything so `applied` is
 * an honest count rather than a count of ops seen.
 */
export function deepEqual(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (typeof a !== typeof b) return false;
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
    return a.every((v, i) => deepEqual(v, b[i]));
  }
  if (a && b && typeof a === 'object') {
    const ka = Object.keys(a as object).sort();
    const kb = Object.keys(b as object).sort();
    if (ka.length !== kb.length || ka.some((k, i) => k !== kb[i])) return false;
    return ka.every((k) => deepEqual((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]));
  }
  return false;
}

/* -------------------------------------------------------------------------- */
/* Canonical JSON + hashing                                                    */
/* -------------------------------------------------------------------------- */

/**
 * Canonical, key-sorted JSON. Object key order is *not* semantically meaningful in the
 * IR, so it must not be able to change a hash: two documents built by different code
 * paths (or by different LLM rounds) that mean the same thing hash the same.
 *
 * Arrays keep their order —?`graph.nodes`, `graph.edges`, `spec.features` and
 * `targets` are all ordered on purpose.
 */
export function canonicalJson(value: unknown): string {
  return stringify(value, new Set<unknown>());
}

function stringify(v: unknown, seen: Set<unknown>): string {
  if (v === null) return 'null';
  switch (typeof v) {
    case 'number':
      // Normalize -0 and non-finite values so the hash cannot depend on float spelling.
      if (!Number.isFinite(v)) return 'null';
      return Object.is(v, -0) ? '0' : String(v);
    case 'boolean':
      return v ? 'true' : 'false';
    case 'string':
      return JSON.stringify(v);
    case 'undefined':
      return 'null';
    case 'bigint':
      return JSON.stringify(String(v));
    case 'object':
      break;
    default:
      return 'null';
  }
  const obj = v as object;
  if (seen.has(obj)) throw new TypeError('canonicalJson: circular structure');
  seen.add(obj);
  try {
    if (Array.isArray(obj)) {
      let out = '[';
      for (let i = 0; i < obj.length; i++) {
        out += (i ? ',' : '') + stringify(obj[i], seen);
      }
      return `${out}]`;
    }
    const rec = obj as Record<string, unknown>;
    const keys = Object.keys(rec)
      .filter((k) => rec[k] !== undefined)
      .sort();
    const parts: string[] = [];
    for (const k of keys) parts.push(`${JSON.stringify(k)}:${stringify(rec[k], seen)}`);
    return `{${parts.join(',')}}`;
  } finally {
    seen.delete(obj);
  }
}

/**
 * The projection of a document that affects codegen —?i.e. everything `emit()` reads.
 *
 * Deliberately absent:
 *   —?`id`, `title`, `createdAt`, `updatedAt` —?identity and bookkeeping, not pixels
 *   —?`provenance`                            —?how we got here, not what we built
 *   —?`node.pos`                              —?node-editor layout, ignored by codegen
 *
 * Including any of those would mean dragging a node in the GUI invalidates every cached
 * compile and every golden image. Excluding them means `hash` is a real content address.
 */
function hashedShape(doc: ShaderDoc): Record<string, unknown> {
  return {
    irVersion: doc.irVersion,
    spec: doc.spec,
    stage: doc.stage,
    graph: {
      nodes: doc.graph.nodes.map((n) => {
        const { pos: _pos, ...rest } = n;
        return rest;
      }),
      edges: doc.graph.edges,
    },
    render: doc.render,
    targets: doc.targets,
    ...(doc.targetNotes ? { targetNotes: doc.targetNotes } : {}),
  };
}

/** Content hash of everything that affects codegen. The cache key for builds + goldens. */
export function hashDoc(doc: ShaderDoc): string {
  return stableHash(canonicalJson(hashedShape(doc)));
}

/* -------------------------------------------------------------------------- */
/* Graph normalization                                                         */
/* -------------------------------------------------------------------------- */

/**
 * Rebuild `graph.edges` from `graph.nodes[].inputs`.
 *
 * `edges` exists for diffing and for the UI, but `inputs` is the authority: it is what
 * the emitter walks. Keeping two representations of one fact in sync by hand is the
 * single most reliable way to produce a document that renders one way and diffs another.
 * So the edges are *derived*, deterministically:
 *
 *   —?node declaration order, then sorted input name (object key order is not trusted)
 *   —?a wire with no explicit port resolves to the source kind's `primaryOut`
 *
 * Idempotent: `normalizeEdges(normalizeEdges(g))` deep-equals `normalizeEdges(g)`.
 * A `WireRef` pointing at a node that does not exist is skipped —?edges reference real
 * nodes only; the dangling wire itself is reported by `analyzeGraph` as `IR004`.
 */
export function normalizeEdges(graph: ShaderGraph): ShaderGraph {
  const byId = new Map<string, ShaderNode>();
  for (const n of graph.nodes) if (!byId.has(n.id)) byId.set(n.id, n);

  const edges: Edge[] = [];
  for (const n of graph.nodes) {
    const inputs = n.inputs;
    if (!inputs) continue;
    for (const port of Object.keys(inputs).sort()) {
      const wire = inputs[port];
      if (!wire) continue;
      const src = byId.get(wire.node);
      if (!src) continue;
      // No catalog here: `port` is used verbatim when given. An omitted port stays
      // omitted only when we cannot resolve it —?`wire()` always fills it in.
      edges.push({
        from: { node: wire.node, port: wire.port ?? '' },
        to: { node: n.id, port },
      });
    }
  }

  return { nodes: graph.nodes, edges };
}

/**
 * Declare wires as `[fromId, toId, toPort, fromPort?]` and let `normalizeEdges` derive
 * the edge list. Hand-authoring (and LLM output) gets a form that cannot desync, and
 * writing the same `toPort` twice is a well-defined last-write-wins.
 */
export function wire(graph: ShaderGraph, pairs: [string, string, string, string?][]): ShaderGraph {
  const nodes = graph.nodes.map((n) => ({ ...n }));
  const byId = new Map(nodes.map((n) => [n.id, n] as const));
  for (const [fromId, toId, toPort, fromPort] of pairs) {
    const target = byId.get(toId);
    // A wire is only written when BOTH endpoints name a node that is in the graph. A wire
    // to a node that does not exist is not a wire, it is a dangling reference that
    // `analyzeGraph` reports as IR004 — silently writing one here would turn a typo in a
    // hand-authored graph into a document that only fails two stages later.
    if (!target || !byId.has(fromId)) continue;
    const ref: WireRef = fromPort === undefined ? { node: fromId } : { node: fromId, port: fromPort };
    target.inputs = withKey(target.inputs, toPort, ref);
  }
  return normalizeEdges({ nodes, edges: [] });
}

/* -------------------------------------------------------------------------- */
/* Ids                                                                         */
/* -------------------------------------------------------------------------- */

/**
 * Deterministic node id: `<kind>` if free, else `<kind>_2`, `<kind>_3`, —? *
 * No randomness and no clock: ids end up inside generated identifiers, so they are part
 * of codegen. Two runs over the same graph must produce byte-identical source, which is
 * impossible if an id can be a uuid. (Ids are emitted as `L_<id>_<port>` locals, so they
 * are also always valid GLSL identifiers —?kinds are `[a-z_]` only.)
 *
 * Starts at `_2` because the unsuffixed name reads as the first instance.
 */
export function makeNodeId(kind: NodeKind, taken: Iterable<string>): string {
  const used = new Set(taken);
  const base = String(kind);
  if (!used.has(base)) return base;
  for (let n = 2; ; n++) {
    const candidate = `${base}_${n}`;
    if (!used.has(candidate)) return candidate;
  }
}

function allIds(doc: ShaderDoc): string[] {
  return doc.graph.nodes.map((n) => n.id);
}

/**
 * Deep-copy a `ParamValue`. Vectors are the only mutable part of the shape.
 *
 * Every entry point that accepts a `ParamValue` runs it through this. Without it a
 * document can alias an object the caller still holds — an LLM loop that reuses one op
 * template across proposals would then mutate documents it never touched, and a patch's
 * arrays would appear in a document whose hash was computed for different numbers.
 */
function copyParamValue(v: ParamValue): ParamValue {
  if (v.kind === 'param') {
    return {
      kind: 'param',
      name: v.name,
      type: v.type,
      value: Array.isArray(v.value) ? [...v.value] : v.value,
      ...(v.min !== undefined ? { min: v.min } : {}),
      ...(v.max !== undefined ? { max: v.max } : {}),
      ...(v.step !== undefined ? { step: v.step } : {}),
      ...(v.label !== undefined ? { label: v.label } : {}),
      ...(v.group !== undefined ? { group: v.group } : {}),
    };
  }
  const value = Array.isArray(v.value) ? [...v.value] : v.value;
  return {
    kind: v.kind,
    value,
    ...(v.kind === 'float' || v.kind === 'int' ? (v.min !== undefined ? { min: v.min } : {}) : {}),
    ...(v.kind === 'float' || v.kind === 'int' ? (v.max !== undefined ? { max: v.max } : {}) : {}),
    ...(v.kind === 'float' ? (v.step !== undefined ? { step: v.step } : {}) : {}),
  } as ParamValue;
}

/* -------------------------------------------------------------------------- */
/* addNode / removeNode                                                        */
/* -------------------------------------------------------------------------- */

/**
 * Add a node. The caller may suggest an `id` (an LLM reproducing a graph it saw will),
 * but a collision is an error rather than a silent rename: silently renaming would make
 * every subsequent op in the same patch target the wrong node.
 */
export function addNode(
  doc: ShaderDoc,
  node: NodeInit,
  catalog: Catalog,
): { doc: ShaderDoc; id: string; diagnostics: Diagnostic[] } {
  const diagnostics: Diagnostic[] = [];

  if (!KNOWN_KINDS.includes(node.kind)) {
    diagnostics.push(err(
      PT.UNKNOWN_KIND,
      `\`${node.kind}\` is not a node kind.`,
      { node: node.id },
      `Known kinds: ${KNOWN_KINDS.join(', ')}`,
    ));
    return { doc, id: node.id ?? String(node.kind), diagnostics };
  }
  const schema = schemaOf(catalog, node.kind);
  if (!schema) {
    diagnostics.push(err(
      PT.UNKNOWN_KIND,
      `The catalog has no schema for \`${node.kind}\`, so its ports cannot be checked.`,
      { node: node.id },
      'Pass a catalog that covers every kind the document uses (see CATALOG in @ai4shader/ir/catalog).',
    ));
    return { doc, id: node.id ?? String(node.kind), diagnostics };
  }

  const id = node.id ?? makeNodeId(node.kind, allIds(doc));
  if (node.id !== undefined && allIds(doc).includes(node.id)) {
    diagnostics.push(err(
      PT.DUPLICATE_ID,
      `Node id \`${node.id}\` is already used.`,
      { node: node.id },
      `Use a free id such as \`${makeNodeId(node.kind, allIds(doc))}\`, or omit the id and let the patcher pick one.`,
    ));
    return { doc, id, diagnostics };
  }

  const built: ShaderNode = { id, kind: node.kind };
  if (node.pos) built.pos = { x: node.pos.x, y: node.pos.y };
  if (node.note !== undefined) built.note = node.note;

  // Params: unknown names and mismatched literal types are rejected here rather than
  // discovered by the emitter two stages later.
  if (node.params) {
    const params: Record<string, ParamValue> = {};
    for (const key of Object.keys(node.params).sort()) {
      const val = node.params[key]!;
      const port = schema.inputs.find((p) => p.name === key);
      if (!port) {
        diagnostics.push(err(
          PT.UNKNOWN_PORT,
          `Node \`${id}\` (${node.kind}) has no input \`${key}\`.`,
          { node: id, port: key },
          `Valid inputs: ${schema.inputs.map((p) => p.name).join(', ') || '(none)'}`,
        ));
        continue;
      }
      const d = checkLiteral(id, key, val, port.type);
      if (d) diagnostics.push(d);
      else params[key] = copyParamValue(val);
    }
    if (Object.keys(params).length) built.params = params;
  }

  // Inputs: same story, plus the source must exist (it cannot yet —?a node is added
  // before it is wired) so unknown sources are a hard error, not a dangling wire.
  if (node.inputs) {
    const inputs: Record<string, WireRef> = {};
    for (const key of Object.keys(node.inputs).sort()) {
      const src = node.inputs[key]!;
      const port = schema.inputs.find((p) => p.name === key);
      if (!port) {
        diagnostics.push(err(
          PT.UNKNOWN_PORT,
          `Node \`${id}\` (${node.kind}) has no input \`${key}\`.`,
          { node: id, port: key },
          `Valid inputs: ${schema.inputs.map((p) => p.name).join(', ') || '(none)'}`,
        ));
        continue;
      }
      if (!nodeOf(doc, src.node)) {
        diagnostics.push(err(
          PT.DANGLING_WIRE,
          `Node \`${id}\` input \`${key}\` reads from missing node \`${src.node}\`.`,
          { node: id, port: key },
          'Add the source node first, then wire it with a `connect` op.',
        ));
        continue;
      }
      const srcSchema = schemaOf(catalog, nodeOf(doc, src.node)!.kind);
      const out = srcSchema ? outputPort(catalog, srcSchema.kind, src.port) : undefined;
      if (srcSchema && !out) {
        diagnostics.push(err(
          PT.UNKNOWN_OUTPUT,
          `Node \`${src.node}\` (${srcSchema.kind}) has no output \`${src.port}\`.`,
          { node: id, port: key },
          `Valid outputs: ${srcSchema.outputs.map((p) => p.name).join(', ')}`,
        ));
        continue;
      }
      if (out && !canConnect(out.type, port.type)) {
        diagnostics.push(err(
          PT.TYPE_MISMATCH,
          `Type mismatch: \`${src.node}.${out.name}\` is ${out.type} but \`${id}.${key}\` expects ${port.type}.`,
          { node: id, port: key },
          mismatchFix(out.type, port.type),
        ));
        continue;
      }
      inputs[key] = out ? { node: src.node, port: out.name } : { ...src };
    }
    if (Object.keys(inputs).length) built.inputs = inputs;
  }

  if (hasErr(diagnostics)) return { doc, id, diagnostics };

  const graph: ShaderGraph = { nodes: [...doc.graph.nodes, built], edges: doc.graph.edges };
  return { doc: { ...doc, graph: normalizeEdges(graph) }, id, diagnostics };
}

/**
 * Remove a node and every reference to it.
 *
 * Wires *into* the node are dropped; wires *out of* it are dropped from every downstream
 * `inputs` map. Leaving those behind would produce a document whose edges point at
 * nothing —?the exact class of corruption a patch is supposed to make impossible.
 */
export function removeNode(
  doc: ShaderDoc,
  id: string,
): { doc: ShaderDoc; diagnostics: Diagnostic[] } {
  if (!nodeOf(doc, id)) {
    return {
      doc,
      diagnostics: [err(
        PT.UNKNOWN_NODE,
        `Cannot remove \`${id}\`: no such node.`,
        { node: id },
        `Existing nodes: ${allIds(doc).join(', ') || '(none)'}`,
      )],
    };
  }
  const nodes = doc.graph.nodes
    .filter((n) => n.id !== id)
    .map((n) => {
      const inputs = n.inputs;
      if (!inputs) return n;
      const keep: Record<string, WireRef> = {};
      for (const k of Object.keys(inputs).sort()) {
        const w = inputs[k]!;
        if (w.node !== id) keep[k] = w;
      }
      if (Object.keys(keep).length === Object.keys(inputs).length) return n;
      const copy: ShaderNode = { ...n };
      if (Object.keys(keep).length) copy.inputs = keep;
      else delete copy.inputs;
      return copy;
    });
  return { doc: { ...doc, graph: normalizeEdges({ nodes, edges: [] }) }, diagnostics: [] };
}

/* -------------------------------------------------------------------------- */
/* connect / disconnect                                                        */
/* -------------------------------------------------------------------------- */

function mismatchFix(from: ShaderType, to: ShaderType): string {
  const order = 'xyzw';
  const arity: Record<ShaderType, number> = { float: 1, vec2: 2, vec3: 3, vec4: 4, mat2: 4, mat3: 9, int: 1, bool: 1 };
  if (arity[from] > arity[to]) {
    return `Insert a \`vec_op\` node to reduce ${from} to ${to} (or swizzle .${order.slice(0, arity[to])}).`;
  }
  return `Insert a \`vec_op\`/constructor node to widen ${from} to ${to}.`;
}

/**
 * Every wire reachable backwards from `fromId`, including `fromId` itself.
 * Cycle detection needs the *upstream* set: a new edge `from -> to` is legal iff `to`
 * is not already upstream of `from`.
 */
function upstreamOf(doc: ShaderDoc, fromId: string): Set<string> {
  const byId = new Map(doc.graph.nodes.map((n) => [n.id, n] as const));
  const seen = new Set<string>([fromId]);
  const stack = [fromId];
  while (stack.length) {
    const id = stack.pop()!;
    for (const w of Object.values(byId.get(id)?.inputs ?? {})) {
      if (seen.has(w.node)) continue;
      seen.add(w.node);
      stack.push(w.node);
    }
  }
  return seen;
}

/**
 * Wire `from`'s output port into `to`'s input port.
 *
 * Validated against the same rules `analyzeGraph` applies, so a successful `connect`
 * can never introduce a diagnostic: unknown node, unknown port, intrinsic type mismatch,
 * and the one thing a per-node check cannot see —?a cycle.
 *
 * An existing wire on the same input port is replaced (counted as a change), which is
 * what an editor dragging a new wire expects.
 */
export function connect(
  doc: ShaderDoc,
  from: WireSource,
  to: WireEndpoint,
  catalog: Catalog,
): { doc: ShaderDoc; diagnostics: Diagnostic[] } {
  const diagnostics: Diagnostic[] = [];
  const fromNode = nodeOf(doc, from.node);
  const toNode = nodeOf(doc, to.node);

  if (!fromNode) {
    diagnostics.push(err(
      PT.DANGLING_WIRE,
      `Wire reads from missing node \`${from.node}\`.`,
      { node: to.node, port: to.port },
      `Existing nodes: ${allIds(doc).join(', ') || '(none)'}`,
    ));
  }
  if (!toNode) {
    diagnostics.push(err(
      PT.UNKNOWN_NODE,
      `Wire targets missing node \`${to.node}\`.`,
      { node: to.node, port: to.port },
      `Existing nodes: ${allIds(doc).join(', ') || '(none)'}`,
    ));
  }
  if (!fromNode || !toNode) return { doc, diagnostics };

  const srcSchema = schemaOf(catalog, fromNode.kind);
  const dstSchema = schemaOf(catalog, toNode.kind);
  if (!srcSchema || !dstSchema) {
    diagnostics.push(err(
      PT.UNKNOWN_KIND,
      `The catalog is missing a schema for \`${!srcSchema ? fromNode.kind : toNode.kind}\`, so the wire cannot be type-checked.`,
      { node: !srcSchema ? from.node : to.node },
      'Pass a catalog that covers every kind the document uses.',
    ));
    return { doc, diagnostics };
  }

  const out = outputPort(catalog, fromNode.kind, from.port);
  if (!out) {
    diagnostics.push(err(
      PT.UNKNOWN_OUTPUT,
      `Node \`${from.node}\` (${fromNode.kind}) has no output \`${from.port ?? srcSchema.primaryOut ?? ''}\`.`,
      { node: to.node, port: to.port },
      `Valid outputs: ${srcSchema.outputs.map((p) => p.name).join(', ')}`,
    ));
  }
  const inPort = inputPort(catalog, toNode.kind, to.port);
  if (!inPort) {
    diagnostics.push(err(
      PT.UNKNOWN_PORT,
      `Node \`${to.node}\` (${toNode.kind}) has no input \`${to.port}\`.`,
      { node: to.node, port: to.port },
      `Valid inputs: ${dstSchema.inputs.map((p) => p.name).join(', ') || '(none)'}`,
    ));
  }
  if (!out || !inPort) return { doc, diagnostics };

  if (!canConnect(out.type, inPort.type)) {
    diagnostics.push(err(
      PT.TYPE_MISMATCH,
      `Type mismatch: \`${from.node}.${out.name}\` is ${out.type} but \`${to.node}.${to.port}\` expects ${inPort.type}.`,
      { node: to.node, port: to.port },
      mismatchFix(out.type, inPort.type),
    ));
  }

  // The IR is a DAG by definition. A cycle is a *graph-level* error that per-node
  // validation cannot see, so it is caught here rather than after a wasted emit.
  if (upstreamOf(doc, from.node).has(to.node)) {
    diagnostics.push(err(
      PT.CYCLE,
      `Connecting \`${from.node}\` -> \`${to.node}.${to.port}\` would create a cycle.`,
      { node: to.node, port: to.port },
      'Break the cycle —?feedback needs an explicit ping-pong pass, not a wire.',
    ));
  }

  if (hasErr(diagnostics)) return { doc, diagnostics };

  const next: ShaderNode = { ...toNode, inputs: withKey(toNode.inputs, to.port, { node: from.node, port: out.name }) };
  const nodes = doc.graph.nodes.map((n) => (n.id === next.id ? next : n));
  return { doc: { ...doc, graph: normalizeEdges({ nodes, edges: [] }) }, diagnostics };
}

/**
 * Remove the wire feeding `to.port`. Idempotent: disconnecting an unwired port is not an
 * error, it is a no-op —?a patch that says "make sure this is a literal" should not fail
 * because it already was one.
 */
export function disconnect(
  doc: ShaderDoc,
  to: WireEndpoint,
): { doc: ShaderDoc; diagnostics: Diagnostic[] } {
  const node = nodeOf(doc, to.node);
  if (!node) {
    return {
      doc,
      diagnostics: [err(
        PT.UNKNOWN_NODE,
        `Cannot disconnect \`${to.node}.${to.port}\`: no such node.`,
        { node: to.node, port: to.port },
        `Existing nodes: ${allIds(doc).join(', ') || '(none)'}`,
      )],
    };
  }
  if (node.inputs?.[to.port] === undefined) return { doc, diagnostics: [] };

  const inputs = withKey(node.inputs, to.port, undefined);
  const next: ShaderNode = { ...node };
  if (Object.keys(inputs).length) next.inputs = inputs;
  else delete next.inputs;

  const nodes = doc.graph.nodes.map((n) => (n.id === to.node ? next : n));
  return { doc: { ...doc, graph: normalizeEdges({ nodes, edges: [] }) }, diagnostics: [] };
}

/* -------------------------------------------------------------------------- */
/* setParam                                                                    */
/* -------------------------------------------------------------------------- */

function checkLiteral(
  node: string,
  port: string,
  val: ParamValue,
  expect: ShaderType,
): Diagnostic | null {
  const map: Record<string, ShaderType> = {
    float: 'float', int: 'int', bool: 'bool', vec2: 'vec2', vec3: 'vec3', vec4: 'vec4', color: 'vec4',
  };
  const actual: ShaderType = val.kind === 'param' ? val.type : map[val.kind] ?? 'float';
  if (!canConnect(actual, expect)) {
    return err(
      PT.TYPE_MISMATCH,
      `Value is ${actual} but \`${node}.${port}\` expects ${expect}.`,
      { node, port },
      mismatchFix(actual, expect),
    );
  }
  if (val.kind === 'param') return null;

  const nums = Array.isArray(val.value) ? val.value : [val.value];
  if (nums.some((x) => typeof x === 'number' && !Number.isFinite(x))) {
    return err(PT.BAD_RANGE, `\`${node}.${port}\` was given a non-finite number.`, { node, port });
  }
  const arity: Record<string, number> = { float: 1, int: 1, bool: 1, vec2: 2, vec3: 3, vec4: 4, color: 4 };
  const want = arity[val.kind] ?? 1;
  if (nums.length !== want) {
    return err(
      PT.BAD_OP,
      `\`${node}.${port}\` got a ${val.kind} with ${nums.length} component(s); ${want} required.`,
      { node, port },
    );
  }
  if ((val.kind === 'float' || val.kind === 'int') && val.min !== undefined && val.max !== undefined && val.min > val.max) {
    return warn(
      IR014,
      `\`${node}.${port}\` has min > max (${val.min} > ${val.max}).`,
      { node, port },
      'Swap the bounds —?a slider cannot be built from an inverted range.',
    );
  }
  return null;
}

/**
 * Set an inline literal (or a uniform binding) on one input port.
 *
 * Rejects an input that is currently *wired*: the wire wins at emit time, so a param set
 * there is silently ignored —?the worst possible outcome, because the agent's edit looks
 * applied and has no effect. It is an explicit `disconnect` first, or nothing.
 */
export function setParam(
  doc: ShaderDoc,
  nodeId: string,
  param: string,
  value: ParamValue,
  catalog: Catalog,
): { doc: ShaderDoc; diagnostics: Diagnostic[] } {
  const node = nodeOf(doc, nodeId);
  if (!node) {
    return {
      doc,
      diagnostics: [err(
        PT.UNKNOWN_NODE,
        `Cannot set \`${nodeId}.${param}\`: no such node.`,
        { node: nodeId, port: param },
        `Existing nodes: ${allIds(doc).join(', ') || '(none)'}`,
      )],
    };
  }
  const schema = schemaOf(catalog, node.kind);
  if (!schema) {
    return {
      doc,
      diagnostics: [err(
        PT.UNKNOWN_KIND,
        `The catalog has no schema for \`${node.kind}\`, so \`${param}\` cannot be type-checked.`,
        { node: nodeId, port: param },
        'Pass a catalog that covers every kind the document uses.',
      )],
    };
  }
  const port = schema.inputs.find((p) => p.name === param);
  if (!port) {
    return {
      doc,
      diagnostics: [err(
        PT.UNKNOWN_PORT,
        `Node \`${nodeId}\` (${node.kind}) has no input \`${param}\`.`,
        { node: nodeId, port: param },
        `Valid inputs: ${schema.inputs.map((p) => p.name).join(', ') || '(none)'}`,
      )],
    };
  }
  if (node.inputs?.[param]) {
    return {
      doc,
      diagnostics: [err(
        PT.WIRED_PORT,
        `\`${nodeId}.${param}\` is wired from \`${node.inputs[param]!.node}\`; a literal there would be ignored.`,
        { node: nodeId, port: param },
        `Send \`{ op: 'disconnect', to: { node: '${nodeId}', port: '${param}' } }\` in the same patch first.`,
      )],
    };
  }
  const d = checkLiteral(nodeId, param, value, port.type);
  if (d && d.severity === 'error') return { doc, diagnostics: [d] };

  const next: ShaderNode = { ...node, params: withKey(node.params, param, copyParamValue(value)) };
  const diagnostics = d ? [d] : [];
  return { doc: withNode(doc, next), diagnostics };
}

/* -------------------------------------------------------------------------- */
/* exposeParam / inlineParam                                                   */
/* -------------------------------------------------------------------------- */

/** Every uniform name currently declared anywhere in the graph, with its type. */
export function uniformNames(graph: ShaderGraph): Map<string, { type: ShaderType; nodes: string[] }> {
  const out = new Map<string, { type: ShaderType; nodes: string[] }>();
  for (const n of graph.nodes) {
    for (const [key, val] of Object.entries(n.params ?? {})) {
      if (val.kind !== 'param') continue;
      const hit = out.get(val.name);
      if (hit) hit.nodes.push(`${n.id}.${key}`);
      else out.set(val.name, { type: val.type, nodes: [`${n.id}.${key}`] });
    }
  }
  return out;
}

/**
 * Bind an inline literal to a new named uniform so it becomes a tunable slider.
 *
 * This is the operation the whole fine-tuning story rests on: after it, the optimizer
 * searches a *number* instead of asking an LLM for a new graph, and every target gets
 * the value from its natural uniform path.
 *
 * Rejected when the name
 *   —?is not a valid identifier in every target (`u-my scale`, `2x`),
 *   —?is reserved in any target (`sample`, `matrix`, `fragment` —?legal in GLSL, fatal in
 *     MSL/HLSL/WGSL), or
 *   —?is already declared with a *different* type.
 * Each rejection carries a `fix` with a concrete safe name, so the agent can retry
 * without a round-trip to a human.
 */
export function exposeParam(
  doc: ShaderDoc,
  nodeId: string,
  param: string,
  opts: ExposeOptions,
): { doc: ShaderDoc; diagnostics: Diagnostic[] } {
  const node = nodeOf(doc, nodeId);
  if (!node) {
    return {
      doc,
      diagnostics: [err(
        PT.UNKNOWN_NODE,
        `Cannot expose \`${nodeId}.${param}\`: no such node.`,
        { node: nodeId, port: param },
        `Existing nodes: ${allIds(doc).join(', ') || '(none)'}`,
      )],
    };
  }
  if (node.inputs?.[param]) {
    return {
      doc,
      diagnostics: [err(
        PT.WIRED_PORT,
        `\`${nodeId}.${param}\` is wired from \`${node.inputs[param]!.node}\`; it has no literal to expose.`,
        { node: nodeId, port: param },
        `Send \`{ op: 'disconnect', to: { node: '${nodeId}', port: '${param}' } }\` first if the wire is a mistake.`,
      )],
    };
  }
  const current = node.params?.[param];
  if (current === undefined) {
    return {
      doc,
      diagnostics: [err(
        PT.NO_VALUE,
        `\`${nodeId}.${param}\` has no literal to expose.`,
        { node: nodeId, port: param },
        `Send \`{ op: 'set_param', node: '${nodeId}', param: '${param}', value: {...} }\` in the same patch first.`,
      )],
    };
  }
  if (current.kind === 'param') {
    return {
      doc,
      diagnostics: [err(
        PT.NAME_TAKEN,
        `\`${nodeId}.${param}\` is already bound to uniform \`${current.name}\`.`,
        { node: nodeId, port: param },
        'Set its value with `set_param` instead, or `inline_param` first to start over.',
      )],
    };
  }

  const taken = uniformNames(doc.graph);
  const suggested = safeIdentifier(opts.name, new Set(taken.keys()));
  if (!IDENT_RE.test(opts.name)) {
    return {
      doc,
      diagnostics: [err(
        PT.BAD_NAME,
        `\`${opts.name}\` is not a valid identifier.`,
        { node: nodeId, port: param },
        `Use \`${suggested}\` (must match [A-Za-z_][A-Za-z0-9_]*).`,
      )],
    };
  }
  if (isReservedAnywhere(opts.name)) {
    return {
      doc,
      diagnostics: [err(
        PT.RESERVED_NAME,
        `\`${opts.name}\` is reserved in at least one target language.`,
        { node: nodeId, port: param },
        `Use \`${suggested}\` —?a name that compiles identically in GLSL, WGSL, HLSL and MSL.`,
      )],
    };
  }

  const bindType = paramKindToType(current);
  if (!bindType) {
    return {
      doc,
      diagnostics: [err(
        PT.BAD_OP,
        `\`${nodeId}.${param}\` is a ${current.kind} literal and cannot back a uniform.`,
        { node: nodeId, port: param },
        'Only float, int, bool, vec2/3/4 and color literals can be exposed.',
      )],
    };
  }

  const existing = taken.get(opts.name);
  if (existing && existing.type !== bindType) {
    return {
      doc,
      diagnostics: [err(
        PT.NAME_TAKEN,
        `Uniform \`${opts.name}\` is already declared as ${existing.type} at ${existing.nodes.join(', ')}; exposing a ${bindType} under the same name would fail to link.`,
        { node: nodeId, port: param },
        `Use \`${suggested}\` instead, or give the existing uniform a different name.`,
      )],
    };
  }
  if (existing) {
    // Same name and same type: re-binding is legal, but the document already has one
    // shared uniform. Report it loudly —?a shared slider is occasionally intended and
    // usually a bug.
    const shared: ParamValue = {
      kind: 'param',
      name: opts.name,
      type: existing.type,
      value: paramValueOf(current),
      ...(opts.min !== undefined ? { min: opts.min } : {}),
      ...(opts.max !== undefined ? { max: opts.max } : {}),
      ...(opts.step !== undefined ? { step: opts.step } : {}),
      ...(opts.label !== undefined ? { label: opts.label } : {}),
      ...(opts.group !== undefined ? { group: opts.group } : {}),
    };
    const next: ShaderNode = { ...node, params: withKey(node.params, param, shared) };
    return {
      doc: withNode(doc, next),
      diagnostics: [warn(
        PT.NAME_TAKEN,
        `Uniform \`${opts.name}\` is now shared by ${[...existing.nodes, `${nodeId}.${param}`].join(', ')}.`,
        { node: nodeId, port: param },
        'Two nodes driven by one slider is a legitimate design; it is also the most common accidental uniform collision.',
      )],
    };
  }

  const bound: ParamValue = {
    kind: 'param',
    name: opts.name,
    type: bindType,
    value: paramValueOf(current),
    ...(opts.min !== undefined ? { min: opts.min } : {}),
    ...(opts.max !== undefined ? { max: opts.max } : {}),
    ...(opts.step !== undefined ? { step: opts.step } : {}),
    ...(opts.label !== undefined ? { label: opts.label } : {}),
    ...(opts.group !== undefined ? { group: opts.group } : {}),
  };
  const next: ShaderNode = { ...node, params: withKey(node.params, param, bound) };
  return { doc: withNode(doc, next), diagnostics: [] };
}

/** The uniform type an inline literal lowers to. `color` is a vec4 in every target. */
function paramKindToType(v: ParamValue): ShaderType | null {
  switch (v.kind) {
    case 'float':
      return 'float';
    case 'int':
      return 'int';
    case 'bool':
      return 'bool';
    case 'vec2':
      return 'vec2';
    case 'vec3':
      return 'vec3';
    case 'vec4':
      return 'vec4';
    case 'color':
      return 'vec4';
    default:
      return null;
  }
}

/** The uniform's current value, as the loose `number | boolean | number[]` shape. */
function paramValueOf(v: ParamValue): number | boolean | number[] {
  if (v.kind === 'param') return Array.isArray(v.value) ? [...v.value] : v.value;
  return Array.isArray(v.value) ? [...v.value] : (v.value as number | boolean);
}

/**
 * Inverse of `exposeParam`: fold a uniform back into an inline literal.
 *
 * The slider disappears from the inspector and the value is baked into the graph. Used
 * when an optimizer has converged (the winning value becomes a constant, so the shader
 * no longer needs a uniform slot) and to keep a one-off experiment out of the public
 * parameter list.
 *
 * Round-trips: `inlineParam(exposeParam(doc, ...), ...)` hashes identically to `doc`,
 * because the literal is reconstructed from the uniform's own type and value.
 */
export function inlineParam(
  doc: ShaderDoc,
  nodeId: string,
  param: string,
): { doc: ShaderDoc; diagnostics: Diagnostic[] } {
  const node = nodeOf(doc, nodeId);
  if (!node) {
    return {
      doc,
      diagnostics: [err(
        PT.UNKNOWN_NODE,
        `Cannot inline \`${nodeId}.${param}\`: no such node.`,
        { node: nodeId, port: param },
        `Existing nodes: ${allIds(doc).join(', ') || '(none)'}`,
      )],
    };
  }
  const current = node.params?.[param];
  if (current === undefined) {
    return {
      doc,
      diagnostics: [err(
        PT.NO_VALUE,
        `\`${nodeId}.${param}\` has no value to inline.`,
        { node: nodeId, port: param },
        'Nothing to do —?the port is already a literal or unwired.',
      )],
    };
  }
  if (current.kind !== 'param') {
    return {
      doc,
      diagnostics: [err(
        PT.NO_VALUE,
        `\`${nodeId}.${param}\` is a ${current.kind} literal, not a uniform.`,
        { node: nodeId, port: param },
        '`inline_param` only applies to literal values bound with `expose_param`.',
      )],
    };
  }

  const lit = literalFromUniform(current);
  const next: ShaderNode = { ...node, params: withKey(node.params, param, lit) };
  return { doc: withNode(doc, next), diagnostics: [] };
}

/**
 * Fold a uniform binding back into the literal it was exposed from.
 *
 * `min`/`max` are carried over for numeric literals, because they are what makes the
 * round-trip an identity: `exposeParam(doc, 'float' -> param)` then `inlineParam` has to
 * reproduce the original literal exactly, including its slider bounds, or every
 * expose/preview/inline cycle would silently drift the document hash.
 *
 * The value is clamped into its own range first: an out-of-range uniform (a slider driven
 * past its end by an optimizer, or a hand-edited `params.json`) must not fold back into an
 * illegal literal.
 */
function literalFromUniform(u: Extract<ParamValue, { kind: 'param' }>): ParamValue {
  const num = (v: number): number => {
    let out = v;
    if (u.min !== undefined) out = Math.max(u.min, out);
    if (u.max !== undefined) out = Math.min(u.max, out);
    return out;
  };
  switch (u.type) {
    case 'float':
      return {
        // `min`/`max` are what make the round-trip an identity, so they are always carried
        // back. `step` is the one field that can be *added* by an exposure that did not
        // exist before: a uniform authored directly by a patch (or by the tuning loop) may
        // carry a step the original literal never had. A folded literal with one extra
        // slider hint is still the same shader, and the hint is worth having.
        kind: 'float',
        value: num(Number(u.value)),
        ...(u.min !== undefined ? { min: u.min } : {}),
        ...(u.max !== undefined ? { max: u.max } : {}),
        ...(u.step !== undefined ? { step: u.step } : {}),
      };
    case 'int':
      return {
        kind: 'int',
        value: Math.round(num(Number(u.value))),
        ...(u.min !== undefined ? { min: u.min } : {}),
        ...(u.max !== undefined ? { max: u.max } : {}),
      };
    case 'bool':
      return { kind: 'bool', value: Boolean(u.value) };
    case 'vec2':
      return { kind: 'vec2', value: vec2(u.value) };
    case 'vec3':
      return { kind: 'vec3', value: vec3(u.value) };
    default:
      // `mat2`/`mat3` never reach here (they cannot be exposed as a uniform literal today),
      // so the remaining case is the RGBA one: `vec4`, and the `color` literal that is
      // indistinguishable from it on the wire.
      return { kind: 'vec4', value: vec4(u.value) };
  }
}

function vec2(v: number | boolean | number[]): [number, number] {
  const a = Array.isArray(v) ? v : [v];
  return [Number(a[0] ?? 0), Number(a[1] ?? a[0] ?? 0)];
}
function vec3(v: number | boolean | number[]): [number, number, number] {
  const a = Array.isArray(v) ? v : [v];
  return [Number(a[0] ?? 0), Number(a[1] ?? a[0] ?? 0), Number(a[2] ?? a[0] ?? 0)];
}
function vec4(v: number | boolean | number[]): [number, number, number, number] {
  const a = Array.isArray(v) ? v : [v];
  return [Number(a[0] ?? 0), Number(a[1] ?? a[0] ?? 0), Number(a[2] ?? a[0] ?? 0), Number(a[3] ?? a[0] ?? 0)];
}

/* -------------------------------------------------------------------------- */
/* applyPatch                                                                  */
/* -------------------------------------------------------------------------- */

/** Human/agent-readable actor string for the provenance log. */
function actorOf(author: ShaderPatch['author']): string {
  if (!author) return 'human';
  const bits: string[] = [author.kind];
  if (author.model) bits.push(author.model);
  if (author.round !== undefined) bits.push(`round ${author.round}`);
  return bits.join(':');
}

/** Reject a patch caught mid-flight; the caller gets the *original* doc back. */
function reject(doc: ShaderDoc, opIndex: number, diagnostics: Diagnostic[]): PatchResult {
  const failures = diagnostics.filter((d) => d.severity === 'error');
  const reason = failures.map((d) => d.code).join(', ') || 'a rejected operation';
  return {
    doc,
    applied: 0,
    diagnostics: [
      ...diagnostics,
      err(
        PT.NOT_ATOMIC,
        `Patch rejected in full at op ${opIndex + 1} (${reason}); the document is unchanged.`,
        undefined,
        'Fix the failing op and resend; no earlier op was applied, so the op indices still match.',
      ),
    ],
    ok: false,
  };
}

function applyOne(doc: ShaderDoc, op: PatchOp, catalog: Catalog): { doc: ShaderDoc; diagnostics: Diagnostic[] } {
  switch (op.op) {
    case 'add_node': {
      if (!op.node || typeof op.node !== 'object') {
        return { doc, diagnostics: [err(PT.BAD_OP, '`add_node` is missing its node.', undefined, 'Send `{ op: \'add_node\', node: { id, kind, ... } }`.')] };
      }
      const added = addNode(doc, op.node, catalog);
      return { doc: added.doc, diagnostics: added.diagnostics };
    }
    case 'remove_node':
      return removeNode(doc, op.node);
    case 'set_body': {
      const target = doc.graph.nodes.find((n) => n.id === op.node);
      if (!target) {
        return {
          doc,
          diagnostics: [err(PT.UNKNOWN_NODE, `\`set_body\` names a node that does not exist: \`${op.node}\`.`, undefined, 'Add the node first, or use an id from the node list.')],
        };
      }
      /*
       * Refused for every kind but `custom_glsl`.
       *
       * Not a style rule. A catalog body is a constant precisely so the token rewriter can be
       * proven correct against it; letting a patch replace one would hand the six-target
       * guarantee to whatever the model wrote that round. `custom_glsl` is the single kind
       * whose body is *meant* to be per-instance — that is what makes it an escape hatch
       * rather than a hole.
       */
      if (target.kind !== 'custom_glsl') {
        return {
          doc,
          diagnostics: [
            err(
              PT.BAD_OP,
              `\`set_body\` targets \`${op.node}\` of kind \`${target.kind}\`; only \`custom_glsl\` carries a body.`,
              { node: op.node },
              'Add a `custom_glsl` node and set its body, or edit the graph with the normal ops.',
            ),
          ],
        };
      }
      const body = op.body.trim();
      if (!body) {
        return {
          doc,
          diagnostics: [err(PT.BAD_OP, '`set_body` has an empty body.', { node: op.node }, 'Send the restricted-GLSL text, or remove the node.')],
        };
      }
      // Diagnostics come from `analyzeGraph`, which lints every body — including on nodes that
      // nothing reaches yet. Re-running it here is what makes an invalid snippet a rejected
      // patch rather than a document that fails later.
      const next: ShaderDoc = {
        ...doc,
        graph: {
          ...doc.graph,
          nodes: doc.graph.nodes.map((n) => (n.id === op.node ? { ...n, body } : n)),
        },
      };
      // The catalog may be partial here (a caller can pass a subset), which is why the cast
      // matches the one `applyPatch` itself uses a few hundred lines down.
      const analysis = analyzeGraph(next.graph, catalog as Record<NodeKind, NodeSchema>);
      const fatal = analysis.diagnostics.filter((d) => d.severity === 'error' && d.at?.node === op.node);
      if (fatal.length) return { doc, diagnostics: fatal };
      // `applyPatch` stamps `hash` and appends the provenance step once *all* ops succeed, so
      // this handler returns the edited document and nothing else.
      return { doc: next, diagnostics: [] };
    }
    case 'connect':
      return connect(doc, { ...op.from }, { ...op.to }, catalog);
    case 'disconnect':
      return disconnect(doc, { ...op.to });
    case 'set_param':
      return setParam(doc, op.node, op.param, op.value, catalog);
    case 'expose_param':
      return exposeParam(doc, op.node, op.param, {
        name: op.name,
        ...(op.min !== undefined ? { min: op.min } : {}),
        ...(op.max !== undefined ? { max: op.max } : {}),
        // `step` and `group` are forwarded for the same reason `min`/`max` are: a uniform
        // created by a patch must be indistinguishable from one a human exposed through
        // the UI, or `inline_param` ceases to be its inverse and the slider a user sees
        // depends on which route created the uniform. Dropping them here would be
        // invisible in the generated shader — only in the tuning panel.
        ...(op.step !== undefined ? { step: op.step } : {}),
        ...(op.label !== undefined ? { label: op.label } : {}),
        ...(op.group !== undefined ? { group: op.group } : {}),
      });
    case 'inline_param':
      return inlineParam(doc, op.node, op.param);
    case 'set_render':
      return setRender(doc, op.path, op.value);
    case 'set_spec':
      return setSpec(doc, op.spec);
    default: {
      // Exhaustiveness: an unknown op is a malformed patch (an LLM hallucinating an op
      // name is common enough to deserve its own message).
      const unknown = op as { op?: unknown };
      return {
        doc,
        diagnostics: [err(
          PT.BAD_OP,
          `Unknown patch op \`${String(unknown.op)}\`.`,
          undefined,
          'Valid ops: add_node, remove_node, connect, disconnect, set_param, set_render, set_spec, expose_param, inline_param.',
        )],
      };
    }
  }
}

function setRender(
  doc: ShaderDoc,
  path: 'qualityTier' | 'maxIterations' | 'resolution' | 'transparent' | 'feedback',
  value: QualityTier | number | [number, number] | boolean,
): { doc: ShaderDoc; diagnostics: Diagnostic[] } {
  const tiers: QualityTier[] = ['low', 'medium', 'high', 'ultra'];
  switch (path) {
    case 'transparent':
    case 'feedback': {
      if (typeof value !== 'boolean') {
        return { doc, diagnostics: [err(PT.BAD_OP, `set_render.${path} expects a boolean.`)] };
      }
      return { doc: { ...doc, render: { ...doc.render, [path]: value } }, diagnostics: [] };
    }
    case 'qualityTier': {
      if (typeof value !== 'string' || !tiers.includes(value as QualityTier)) {
        return {
          doc,
          diagnostics: [err(
            PT.BAD_OP,
            `\`set_render.qualityTier\` got ${JSON.stringify(value)}.`,
            undefined,
            `Expected one of: ${tiers.join(', ')}.`,
          )],
        };
      }
      return { doc: { ...doc, render: { ...doc.render, qualityTier: value as QualityTier } }, diagnostics: [] };
    }
    case 'maxIterations': {
      if (typeof value !== 'number' || !Number.isInteger(value) || value < 1) {
        return {
          doc,
          diagnostics: [err(
            PT.BAD_OP,
            `\`set_render.maxIterations\` got ${JSON.stringify(value)}.`,
            undefined,
            'Expected a positive integer (the hard ceiling on emitted loop iterations).',
          )],
        };
      }
      return { doc: { ...doc, render: { ...doc.render, maxIterations: value } }, diagnostics: [] };
    }
    case 'resolution': {
      if (!Array.isArray(value) || value.length !== 2 || value.some((n) => !Number.isFinite(n) || n <= 0)) {
        return {
          doc,
          diagnostics: [err(
            PT.BAD_OP,
            `\`set_render.resolution\` got ${JSON.stringify(value)}.`,
            undefined,
            'Expected [width, height] with both > 0.',
          )],
        };
      }
      return { doc: { ...doc, render: { ...doc.render, resolution: [value[0], value[1]] } }, diagnostics: [] };
    }
    default:
      return {
        doc,
        diagnostics: [err(
          PT.BAD_OP,
          `\`set_render\` cannot write \`${String(path)}\`.`,
          undefined,
          'Valid paths: qualityTier, maxIterations, resolution, transparent, feedback.',
        )],
      };
  }
}

function setSpec(doc: ShaderDoc, spec: Partial<ShaderDoc['spec']>): { doc: ShaderDoc; diagnostics: Diagnostic[] } {
  if (!spec || typeof spec !== 'object') {
    return { doc, diagnostics: [err(PT.BAD_OP, '`set_spec` is missing its spec.', undefined, 'Send `{ op: \'set_spec\', spec: { intent?: ..., features?: [...] } }`.')] };
  }
  for (const key of ['features', 'tags', 'antiFeatures', 'outOfScope'] as const) {
    const v = spec[key];
    if (v !== undefined && (!Array.isArray(v) || v.some((x) => typeof x !== 'string'))) {
      return {
        doc,
        diagnostics: [err(PT.BAD_OP, `\`set_spec.${key}\` must be an array of strings.`, undefined, `Got ${JSON.stringify(v)}.`)],
      };
    }
  }
  // Field-by-field merge (not `{...old, ...new}`) so an explicit `undefined` cannot
  // erase a field: JSON round-trips drop undefineds, and "unset" is not a patch op.
  const next = { ...doc.spec };
  if (spec.intent !== undefined) next.intent = spec.intent;
  if (spec.category !== undefined) next.category = spec.category;
  if (spec.tags !== undefined) next.tags = [...spec.tags];
  if (spec.features !== undefined) next.features = [...spec.features];
  if (spec.antiFeatures !== undefined) next.antiFeatures = [...spec.antiFeatures];
  if (spec.outOfScope !== undefined) next.outOfScope = [...spec.outOfScope];
  return { doc: { ...doc, spec: next }, diagnostics: [] };
}

/**
 * Apply a patch **atomically**.
 *
 * If any op fails to validate, the original `doc` is returned unchanged with `ok: false`
 * and `applied: 0`. Partial application is what produces half-broken documents that then
 * cost an extra LLM round-trip to diagnose —?and, worse, a partially applied patch is no
 * longer the diff the user reviewed.
 *
 * On success the document is re-hashed and one provenance step is appended. The sequence
 * is: apply ops -> `normalizeEdges` (so `edges` and `inputs` can never drift) -> `analyzeGraph`
 * once (so the returned diagnostics include any *graph-level* problem the ops introduced,
 * not just the per-op checks) -> stamp `updatedAt` + `hash` + provenance.
 *
 * `emitter.ts` has no inverse for `edges`: this is the last place the two representations
 * of the wiring are reconciled before the document is hashed and cached.
 */
export function applyPatch(doc: ShaderDoc, patch: ShaderPatch, catalog: Catalog): PatchResult {
  if (!patch || typeof patch !== 'object' || !Array.isArray(patch.ops)) {
    return {
      doc,
      applied: 0,
      diagnostics: [err(
        PT.BAD_OP,
        'A patch must be an object with an `ops` array.',
        undefined,
        'Send `{ summary: "...", ops: [...] }`.',
      )],
      ok: false,
    };
  }
  if (patch.ops.length === 0) {
    return { doc, applied: 0, diagnostics: [], ok: true };
  }

  let work = doc;
  const diagnostics: Diagnostic[] = [];
  let applied = 0;

  for (let i = 0; i < patch.ops.length; i++) {
    const op = patch.ops[i]!;
    const res = applyOne(work, op, catalog);
    if (hasErr(res.diagnostics)) {
      return reject(doc, i, [
        ...diagnostics,
        // Attribute the failure to the op's own node when it has one, so the agent's
        // repair loop has somewhere to look.
        ...res.diagnostics,
      ]);
    }
    diagnostics.push(...res.diagnostics);
    if (!deepEqual(work, res.doc)) applied++;
    work = res.doc;
  }

  // --- reconcile, then re-validate the whole graph --------------------------
  const graph = normalizeEdges(work.graph);
  const stable: ShaderDoc = { ...work, graph };
  const analysis = analyzeGraph(graph, catalog as Record<NodeKind, NodeSchema>);

  const now = new Date().toISOString();
  const step = { at: now, actor: actorOf(patch.author), summary: patch.summary };

  const stamped: ShaderDoc = {
    ...stable,
    updatedAt: now,
    provenance: {
      ...(stable.provenance ?? {}),
      steps: [...(stable.provenance?.steps ?? []), step],
    },
  };

  return {
    doc: { ...stamped, hash: hashDoc(stamped) },
    applied,
    diagnostics: [...diagnostics, ...analysis.diagnostics],
    ok: true,
  };
}

/* -------------------------------------------------------------------------- */
/* Document construction                                                       */
/* -------------------------------------------------------------------------- */

/**
 * A fresh, valid document. Everything the emitter reads has a defensible default, so a
 * half-specified doc from an LLM still round-trips through the patcher.
 *
 * `crypto.randomUUID` is used for `id` only —?`id` is excluded from `hashDoc`, so
 * identity may be random while content stays deterministic.
 */
export function createDoc(partial: Partial<ShaderDoc> = {}): ShaderDoc {
  const base: ShaderDoc = {
    irVersion: IR_VERSION,
    id: globalThis.crypto?.randomUUID ? globalThis.crypto.randomUUID() : `doc_${stableHash(String(Math.random()))}`,
    title: 'Untitled shader',
    createdAt: new Date().toISOString(),
    spec: { intent: '', category: 'other', tags: [], features: [] },
    stage: 'fragment',
    graph: { nodes: [], edges: [] },
    render: {
      resolution: [1280, 720],
      qualityTier: 'medium',
      maxIterations: 128,
      transparent: false,
    },
    targets: ['glsl'],
  };
  const render = { ...base.render, ...(partial.render ?? {}) };
  const merged: ShaderDoc = {
    ...base,
    ...partial,
    spec: { ...base.spec, ...(partial.spec ?? {}) },
    graph: partial.graph
      ? { nodes: partial.graph.nodes.map((n) => ({ ...n })), edges: partial.graph.edges.map((e) => ({ ...e })) }
      : base.graph,
    // `resolution` is a tuple, so it must be copied field by field: spreading `render`
    // alone would leave the document aliasing whatever array the caller passed in, and a
    // later mutation of that array would change the document without changing its hash.
    render: { ...render, resolution: [render.resolution[0], render.resolution[1]] },
    targets: partial.targets ? [...partial.targets] : base.targets,
  };
  return { ...merged, hash: hashDoc(merged) };
}

/**
 * Clone a document under a new title —?the "branch a version before an experiment" move.
 *
 * The clone is a *content* clone: it hashes identically to its parent, because a fork that
 * only renames itself still compiles to the same bytes and must still hit the same cache
 * entry. `hash` is nevertheless cleared, because a fork is by definition the opening of an
 * experiment: the next `applyPatch` stamps a fresh one, and carrying the parent's hash
 * around invites a host to key the fork's first build under the parent's entry.
 */
export function forkDoc(doc: ShaderDoc, title: string, opts: ForkOptions = {}): ShaderDoc {
  const cloned = structuredClone(doc) as ShaderDoc;
  const now = new Date().toISOString();
  const next: ShaderDoc = {
    ...cloned,
    id: globalThis.crypto?.randomUUID ? globalThis.crypto.randomUUID() : `doc_${stableHash(title + now)}`,
    title,
    createdAt: now,
    updatedAt: now,
    provenance: {
      ...(cloned.provenance ?? {}),
      // Appending rather than replacing keeps the clone's own step log; the fork point
      // itself is `parents`, which is what the DAG walker follows.
      ...(opts.parents ? { parents: [...(cloned.provenance?.parents ?? []), ...opts.parents] } : {}),
    },
  };
  delete next.hash;
  return next;
}
