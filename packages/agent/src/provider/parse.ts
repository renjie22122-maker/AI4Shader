/**
 * AI4Shader — agent layer: the defensive parsing boundary
 * =========================================================
 *
 * This file sits between **raw model text** and **the IR types**, and it is the last line
 * of defence for the one failure mode the whole IR design exists to prevent: a model that
 * emits a node kind, a port name, or a wire that does not exist.
 *
 * The rule this module follows, everywhere, without exception:
 *
 *   **Report and fail. Never guess.**
 *
 * A parser that "repairs" a missing key invents a document nobody wrote; the user then
 * sees an effect they did not ask for and cannot tell whether the model was wrong or the
 * tool was. Removing a trailing comma is the single tolerated exception, and it is a
 * *syntactic* fix with exactly one reading (`{"a":1,}` has no other meaning), it is
 * reported as a diagnostic, and it is confined to `extractJson`. Nothing here balances
 * braces, invents keys, coerces types across families, or fixes an unbalanced quote.
 *
 * ## Diagnostic codes (`AG0xx`)
 *
 * `AG0xx` is this package's namespace — the parsing boundary between "the model said
 * something" and "the IR can act on it" (see docs/AGENT_CONTRACT.md §5 for the other
 * namespaces). Codes are stable and each one carries an actionable `fix`.
 *
 * ## Why a bad op is dropped instead of rejecting the patch
 *
 * `applyPatch` is atomic — it is all-or-nothing — and that is correct for *application*:
 * a half-applied patch is no longer the diff anyone reviewed. Parsing is a different
 * moment. A model that emits nine good ops and one hallucinated port has produced nine
 * ops' worth of value, and rejecting the lot costs a round trip to re-derive what we
 * already have. So this layer returns the patch **with the illegal ops removed** and one
 * diagnostic per removal, and the caller can see exactly what was dropped. The patch that
 * reaches `applyPatch` is still applied atomically — the softening is confined to the
 * boundary, and it is only safe because the removals are visible in `diagnostics`.
 *
 * Deliberate consequence: if *no* op survives, the result is an empty-ops patch plus
 * diagnostics. That is not an error either — `applyPatch` on an empty patch is a no-op,
 * and the loop treats an empty patch as a failed round (it has the budget and the
 * diagnostics to decide what that means).
 */

import type { CatalogLike } from '../types.js';
import {
  NODE_KINDS,
  isReservedAnywhere,
  safeIdentifier,
  type Diagnostic,
  type NodeKind,
  type NodeSchema,
  type ParamValue,
  type PatchOp,
  type ShaderDoc,
  type ShaderNode,
  type ShaderPatch,
  type ShaderSpec,
  type ShaderType,
  type WireRef,
} from '@ai4shader/ir';

/* -------------------------------------------------------------------------- */
/* Public shapes                                                               */
/* -------------------------------------------------------------------------- */

/**
 * The catalog this layer validates against. Deliberately the *loose* record, matching
 * `patcher.Catalog`: a partial catalog (a test fixture, a plugin node set) is a legal
 * input, and a missing kind is reported as a diagnostic rather than a crash.
 *
 * The type itself lives in `types.ts`; this is a re-export so callers of `parse.ts` keep one
 * name for it.
 */
export type { CatalogLike };

/** Every diagnostic this module can produce. Stable strings; never renumber. */
export const AG = {
  /** No JSON value at all could be located in the reply. */
  NO_JSON: 'AG001_NO_JSON',
  /** JSON was located but does not parse, and the problem is not a trailing comma. */
  BAD_JSON: 'AG002_BAD_JSON',
  /** A trailing comma was removed. Not a guess — the input has exactly one reading. */
  TRAILING_COMMA: 'AG003_TRAILING_COMMA',
  /** The JSON parsed but the top-level value is not an object. */
  NOT_OBJECT: 'AG004_NOT_OBJECT',

  /** A required spec field is missing or of the wrong type; a default was substituted. */
  SPEC_FIELD: 'AG005_SPEC_FIELD',
  /** `features` is empty, so nothing downstream can check "did we build what was asked". */
  SPEC_NO_FEATURES: 'AG006_SPEC_NO_FEATURES',

  /** The patch is not `{ summary?, ops: [...] }`. */
  PATCH_SHAPE: 'AG010_PATCH_SHAPE',
  /** The op names a node the graph does not have and the patch did not add. */
  UNKNOWN_NODE: 'AG011_UNKNOWN_NODE',
  /** `add_node` (or a port check) names a kind the catalog does not have. */
  UNKNOWN_KIND: 'AG012_UNKNOWN_KIND',
  /** An input port that does not exist on the node's kind. */
  UNKNOWN_INPUT_PORT: 'AG013_UNKNOWN_INPUT_PORT',
  /** An output port that does not exist on the source node's kind. */
  UNKNOWN_OUTPUT_PORT: 'AG014_UNKNOWN_OUTPUT_PORT',
  /** A wire whose source is neither in the document nor added by this patch. */
  DANGLING_WIRE: 'AG015_DANGLING_WIRE',
  /** `expose_param`/`set_param` on a port the kind does not declare. */
  EXPOSE_UNKNOWN_PORT: 'AG016_EXPOSE_UNKNOWN_PORT',
  /** A uniform name that is not a legal identifier in every target. */
  BAD_UNIFORM_NAME: 'AG017_BAD_UNIFORM_NAME',
  /** A uniform name reserved in at least one target language (`sample`, `matrix`, …). */
  RESERVED_UNIFORM_NAME: 'AG018_RESERVED_UNIFORM_NAME',
  /** A uniform name already declared with a different type in this patch. */
  UNIFORM_NAME_CLASH: 'AG019_UNIFORM_NAME_CLASH',
  /** `add_node` reused an id already present in the document or earlier in the patch. */
  DUPLICATE_NODE_ID: 'AG020_DUPLICATE_NODE_ID',
  /** A wire whose source type cannot feed the destination port's type. */
  TYPE_MISMATCH: 'AG021_TYPE_MISMATCH',
  /** The op is not an object, or is missing a field the op kind requires. */
  BAD_OP: 'AG022_BAD_OP',
  /** The op kind is not in `PatchOp` at all. */
  UNKNOWN_OP: 'AG023_UNKNOWN_OP',
  /** `expose_param` names a port that this patch (or the document) already wires. */
  EXPOSE_WIRED_PORT: 'AG024_EXPOSE_WIRED_PORT',
} as const;

/** The result of validating a patch: the surviving patch, plus what was dropped and why. */
export type Parsed<T> = { value: T; diagnostics: Diagnostic[] };

/* -------------------------------------------------------------------------- */
/* Small helpers                                                               */
/* -------------------------------------------------------------------------- */

const IDENT_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;
const KNOWN_KINDS: readonly string[] = NODE_KINDS;

function err(code: string, message: string, at?: Diagnostic['at'], fix?: string): Diagnostic {
  return { severity: 'error', code, message, ...(at ? { at } : {}), ...(fix ? { fix } : {}) };
}

function warn(code: string, message: string, at?: Diagnostic['at'], fix?: string): Diagnostic {
  return { severity: 'warning', code, message, ...(at ? { at } : {}), ...(fix ? { fix } : {}) };
}

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

/** `"a, b, c"`, or `"(none)"` — the shape every `fix` string below uses for a port list. */
function list(names: readonly string[]): string {
  return names.length ? names.join(', ') : '(none)';
}

function inputsOf(schema: NodeSchema | undefined): string[] {
  return (schema?.inputs ?? []).map((p) => p.name);
}

function outputsOf(schema: NodeSchema | undefined): string[] {
  return (schema?.outputs ?? []).map((p) => p.name);
}

function describe(v: unknown): string {
  if (v === null) return 'null';
  if (Array.isArray(v)) return `an array of ${v.length}`;
  return typeof v;
}

/* -------------------------------------------------------------------------- */
/* extractJson                                                                 */
/* -------------------------------------------------------------------------- */

/**
 * Find the first complete JSON value in a reply and parse it.
 *
 * Tolerated wrappers, each of which a real model produces routinely:
 *
 *   — a bare object/array                       `{"a":1}`
 *   — a fenced block, with or without a tag     ` ```json\n{…}\n``` `
 *   — prose before and/or after the value       `Sure! Here it is:\n{…}\nLet me know!`
 *   — a trailing comma before `}` / `]`         `{"a":1,}`     (reported: `AG003`)
 *
 * Not tolerated, on purpose: truncated JSON, an unbalanced quote, two values where one was
 * asked for (the **first** is taken and `AG001` says so), or any structural damage. See
 * the file header.
 *
 * The search is four *explicit* steps — whole text, de-fenced text, de-fenced text with
 * trailing commas removed, then each balanced candidate in turn. Each step parses a
 * concrete string with `JSON.parse` and nothing else, so "what was repaired" is always a
 * fact about a string rather than a flag travelling through the code.
 */
export function extractJson(text: string): { value: unknown; diagnostics: Diagnostic[] } {
  const diagnostics: Diagnostic[] = [];
  const trimmed = typeof text === 'string' ? text.trim() : '';

  if (trimmed === '') {
    diagnostics.push(err(
      AG.NO_JSON,
      'The reply was empty; there is no JSON to parse.',
      undefined,
      'Ask the model for a single JSON object and nothing else (system prompt rule 1).',
    ));
    return { value: undefined, diagnostics };
  }

  const unfenced = stripFence(trimmed);

  // Steps 1 and 2: the reply (or its de-fenced body) is the value, verbatim.
  const strict = parseStrict(unfenced);
  if (strict.ok) return { value: strict.value, diagnostics };

  // Step 3: the same, with a trailing comma removed. See the file header for why this is
  // the only tolerated repair.
  const relaxedText = stripTrailingCommas(unfenced);
  if (relaxedText !== unfenced) {
    const relaxed = parseStrict(relaxedText);
    if (relaxed.ok) {
      diagnostics.push(trailingCommaDiagnostic());
      return { value: relaxed.value, diagnostics };
    }
  }

  // Step 4: the value is embedded in prose. Each balanced candidate is parsed as-is.
  const found = [...candidates(unfenced)];
  if (found.length === 0) {
    diagnostics.push(err(
      AG.NO_JSON,
      `No JSON object or array found in the reply (${trimmed.length} chars).`,
      undefined,
      'The reply must contain one JSON value; a truncated or prose-only reply cannot be used.',
    ));
    return { value: undefined, diagnostics };
  }

  let firstError = '';
  for (let i = 0; i < found.length; i++) {
    const candidate = found[i]!;
    // Values the caller will never see because this one came first. Counted, not assumed:
    // prose after the object is not a second answer, it is a sentence, and warning about
    // it would train people to ignore the warning that matters. A bare scalar (`42`,
    // `"done"`) is checked for explicitly because the scanner only tracks brackets.
    const extraValues =
      found.length - 1 - i + (candidate.trailing && !found[i + 1] && hasScalarValue(unfenced.slice(candidate.end)) ? 1 : 0);

    const parsed = parseStrict(candidate.text);
    if (!parsed.ok) {
      const repaired = parseStrict(stripTrailingCommas(candidate.text));
      if (!repaired.ok) {
        if (!firstError) firstError = parsed.error;
        continue;
      }
      // A candidate needed the comma repair: it is the value, and the repair is reported.
      diagnostics.push(trailingCommaDiagnostic());
      if (extraValues > 0) diagnostics.push(extraValuesDiagnostic(extraValues));
      return { value: repaired.value, diagnostics };
    }
    // Anything after the chosen value was dropped. Saying so is the difference between
    // "the model answered once" and "the model answered twice and we took the first" —
    // and only the second of those is a prompt problem worth fixing.
    if (extraValues > 0) diagnostics.push(extraValuesDiagnostic(extraValues));
    return { value: parsed.value, diagnostics };
  }

  diagnostics.push(err(
    AG.BAD_JSON,
    `The JSON in the reply does not parse: ${firstError}`,
    undefined,
    'Resend the same request; if it repeats, lower the temperature or shorten the requested output.',
  ));
  return { value: undefined, diagnostics };
}

type ParseOk = { ok: true; value: unknown } | { ok: false; error: string };

/** `JSON.parse` and nothing else. No repair happens here, ever. */
function parseStrict(s: string): ParseOk {
  try {
    return { ok: true, value: JSON.parse(s.trim()) as unknown };
  } catch (e) {
    return { ok: false, error: (e as Error).message };
  }
}

function trailingCommaDiagnostic(): Diagnostic {
  return warn(
    AG.TRAILING_COMMA,
    'The JSON had a trailing comma before a closing bracket, which strict JSON forbids; it was removed.',
    undefined,
    'This has exactly one reading, so it is repaired rather than rejected — but tell the model "no trailing commas" if it repeats.',
  );
}

function extraValuesDiagnostic(count: number): Diagnostic {
  return warn(
    AG.NO_JSON,
    `The reply contains ${count} further JSON value(s) after the one used; they were ignored.`,
    undefined,
    'Ask for exactly one JSON object per reply (system prompt rule 1).',
  );
}

/** ` ```json … ``` `, ` ``` … ``` `, or the input unchanged. */
function stripFence(s: string): string {
  const m = /^```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n([\s\S]*?)\r?\n?[ \t]*```\s*$/.exec(s);
  if (m) return (m[2] ?? '').trim();
  // An unterminated fence is a truncated reply; strip the opening so the scan can still
  // find a balanced value, and let the syntax check decide the rest.
  const open = /^```[ \t]*[A-Za-z0-9_+-]*[ \t]*\r?\n?/.exec(s);
  if (open) return s.slice(open[0].length).trim();
  return s;
}

/**
 * Remove `,` immediately before a closing `}`/`]`, leaving string contents alone.
 *
 * A regex would be wrong here: `{"a": ",}"}` is legal JSON and `/,(\s*[}\]])/` mangles
 * it. Scanning with string state costs ten lines and is exactly right.
 */
function stripTrailingCommas(s: string): string {
  let out = '';
  let inString = false;
  let escaped = false;
  for (let i = 0; i < s.length; i++) {
    const c = s[i]!;
    if (inString) {
      out += c;
      if (escaped) escaped = false;
      else if (c === '\\') escaped = true;
      else if (c === '"') inString = false;
      continue;
    }
    if (c === '"') {
      inString = true;
      out += c;
      continue;
    }
    if (c === ',') {
      // Look ahead past whitespace for a closer; if that is what follows, drop the comma.
      let j = i + 1;
      while (j < s.length && /\s/.test(s[j]!)) j++;
      const next = s[j];
      if (next === '}' || next === ']') continue;
    }
    out += c;
  }
  return out;
}

/**
 * Every balanced `{…}` / `[…]` in `s`, string-aware, in order of appearance.
 *
 * Yields each candidate with `trailing` (was there anything after it?) and `end` (the
 * index just past it), so the caller can both count what came later and look at it.
 * A candidate that is never closed is still yielded, once, so the caller reports a syntax
 * error rather than pretending no JSON was present at all.
 */
function* candidates(s: string): Generator<{ text: string; trailing: boolean; end: number }> {
  let i = 0;
  while (i < s.length) {
    const start = firstOpen(s, i);
    if (start === -1) return;
    const open = s[start]!;
    const close = open === '{' ? '}' : ']';

    let depth = 0;
    let inString = false;
    let escaped = false;
    let end = -1;
    for (let k = start; k < s.length; k++) {
      const c = s[k]!;
      if (inString) {
        if (escaped) escaped = false;
        else if (c === '\\') escaped = true;
        else if (c === '"') inString = false;
        continue;
      }
      if (c === '"') {
        inString = true;
        continue;
      }
      if (c === open) depth++;
      else if (c === close) {
        depth--;
        if (depth === 0) {
          end = k;
          break;
        }
      }
    }

    if (end === -1) {
      yield { text: s.slice(start), trailing: false, end: s.length };
      return;
    }
    yield { text: s.slice(start, end + 1), trailing: s.slice(end + 1).trim() !== '', end: end + 1 };
    i = end + 1;
  }
}

/**
 * Is there a bare JSON scalar (`42`, `true`, `"done"`, `null`) in this trailing text?
 *
 * The candidate scanner only tracks `{}` and `[]`, so a reply like `{"a":1} 42` would
 * otherwise look like "one value and some prose". This is a deliberately narrow check —
 * it answers "did the model emit a second answer?" and nothing else.
 */
function hasScalarValue(tail: string): boolean {
  const words = tail.trim().replace(/[.,;:!?]+$/, '');
  if (words === '') return false;
  return /^(?:-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null|"(?:[^"\\]|\\.)*")$/.test(words);
}

function firstOpen(s: string, from: number): number {
  const a = s.indexOf('{', from);
  const b = s.indexOf('[', from);
  if (a === -1) return b;
  if (b === -1) return a;
  return Math.min(a, b);
}

/* -------------------------------------------------------------------------- */
/* parseSpec                                                                   */
/* -------------------------------------------------------------------------- */

const SPEC_ARRAY_FIELDS = ['tags', 'features', 'antiFeatures', 'outOfScope'] as const;

/**
 * Validate a `ShaderSpec`.
 *
 * `intent`/`category`/`tags`/`features` are required by the type, so a missing one is
 * reported and defaulted (`AG005`) rather than dropped: a spec with an empty `intent` is
 * still a usable starting point, and refusing it would strand the whole run on a field no
 * deterministic code can supply. `features` is singled out (`AG006`) because it is what
 * the critic checks the graph against — an empty one silently disables that check.
 */
export function parseSpec(value: unknown): Parsed<ShaderSpec> {
  const diagnostics: Diagnostic[] = [];
  const src = unwrapSpec(value);
  if (src === null) {
    diagnostics.push(err(
      AG.NOT_OBJECT,
      `Expected a spec object, got ${describe(value)}.`,
      undefined,
      'Send `{ "intent": "...", "category": "...", "tags": [], "features": [] }`.',
    ));
    return { value: { intent: '', category: 'other', tags: [], features: [] }, diagnostics };
  }

  const out: ShaderSpec = { intent: '', category: 'other', tags: [], features: [] };

  for (const key of ['intent', 'category'] as const) {
    const v = src[key];
    if (typeof v === 'string' && v.trim() !== '') out[key] = v.trim();
    else {
      diagnostics.push(warn(
        AG.SPEC_FIELD,
        `\`spec.${key}\` is missing or not a non-empty string; defaulted to ${JSON.stringify(out[key])}.`,
        undefined,
        `Send \`"${key}": "..."\` — the IR requires it, so the default is a placeholder, not an answer.`,
      ));
    }
  }

  for (const key of SPEC_ARRAY_FIELDS) {
    const v = src[key];
    if (v === undefined) continue;
    if (!Array.isArray(v)) {
      diagnostics.push(warn(
        AG.SPEC_FIELD,
        `\`spec.${key}\` is ${describe(v)}, not an array of strings; it was dropped.`,
        undefined,
        `Send \`"${key}": ["...", "..."]\`.`,
      ));
      continue;
    }
    const strings = v.filter((x): x is string => typeof x === 'string' && x.trim() !== '').map((x) => x.trim());
    if (strings.length !== v.length) {
      diagnostics.push(warn(
        AG.SPEC_FIELD,
        `\`spec.${key}\` contained ${v.length - strings.length} non-string or empty entr(ies); they were dropped.`,
        undefined,
        `Every entry of \`${key}\` must be a non-empty string.`,
      ));
    }
    out[key] = strings;
  }

  if (out.features.length === 0) {
    diagnostics.push(warn(
      AG.SPEC_NO_FEATURES,
      'The spec lists no observable `features`, so "did we build what was asked for?" cannot be checked.',
      undefined,
      'List every requested observable behavior, including temporal behavior, e.g. "droplets stay fixed before fading".',
    ));
  }

  return { value: out, diagnostics };
}

/**
 * Accept either a bare spec or a `{ "spec": { … } }` envelope.
 *
 * Models wrap. The system prompt asks for the bare object, and a wrapper that carries a
 * real spec is not an error worth burning a round trip on — but the unwrap is only
 * performed when the outer object has no spec fields of its own, so a spec that happens
 * to contain a `spec` key can never be mistaken for an envelope.
 */
function unwrapSpec(value: unknown): Record<string, unknown> | null {
  if (!isRecord(value)) return null;
  const looksLikeSpec = ['intent', 'category', 'tags', 'features'].some((k) => k in value);
  if (!looksLikeSpec && isRecord(value['spec'])) return value['spec'];
  return value;
}

/* -------------------------------------------------------------------------- */
/* parsePatch                                                                  */
/* -------------------------------------------------------------------------- */

const ALL_OPS = new Set([
  'add_node', 'remove_node', 'set_body', 'connect', 'disconnect', 'set_param', 'set_render', 'set_spec',
  'expose_param', 'inline_param',
]);

/**
 * Validate a `ShaderPatch` op by op against the catalog and the document it will be
 * applied to. Illegal ops are **removed** and reported; the rest survive. See the file
 * header for why that softening is deliberate and what makes it safe.
 *
 * The validator simulates the patch's *structure* as it goes — which ids exist, which
 * ports are wired — so an op is judged against the document the patch is building, not
 * only against the document as it was. That is what lets the default stub script add
 * nodes and wire them in one patch, and it is also what catches a `connect` whose source
 * the patch itself forgot to add.
 */
export function parsePatch(value: unknown, catalog: CatalogLike): Parsed<ShaderPatch> {
  return validatePatch(value, new PatchState(catalog), false);
}

/**
 * Re-validate a patch against the *document* it will be applied to.
 *
 * Seed existing nodes before checking edits and connections. Online providers have the
 * document in PatchContext and must use this path for repairs and refinements too.
 */
export function parsePatchFor(doc: ShaderDoc, value: unknown, catalog: CatalogLike): Parsed<ShaderPatch> {
  return validatePatch(value, PatchState.forDoc(doc, catalog), true);
}

/** The shared body of `parsePatch` / `parsePatchFor`. */
function validatePatch(value: unknown, state: PatchState, seeded: boolean): Parsed<ShaderPatch> {
  const diagnostics: Diagnostic[] = [];
  const rejected = (summary: string): Parsed<ShaderPatch> => ({ value: { summary, ops: [] }, diagnostics });

  if (!isRecord(value)) {
    diagnostics.push(err(
      AG.PATCH_SHAPE,
      `A patch must be an object with an \`ops\` array; got ${describe(value)}.`,
      undefined,
      'Send `{ "summary": "...", "ops": [ { "op": "add_node", ... } ] }`.',
    ));
    return rejected('malformed patch');
  }
  if (!Array.isArray(value['ops'])) {
    diagnostics.push(err(
      AG.PATCH_SHAPE,
      'The patch has no `ops` array.',
      undefined,
      'Send `{ "summary": "...", "ops": [...] }`; an empty `ops` array is legal and means "no change".',
    ));
    return rejected('malformed patch');
  }

  const rawOps = value['ops'] as unknown[];
  const summary = typeof value['summary'] === 'string' && value['summary'].trim() !== ''
    ? value['summary'].trim()
    : `${rawOps.length} op(s)`;

  const kept: PatchOp[] = [];
  rawOps.forEach((raw, index) => {
    const verdict = state.check(raw, index);
    if (verdict.diagnostics.length) diagnostics.push(...verdict.diagnostics);
    if (verdict.op) kept.push(verdict.op);
  });

  // Deliberately NO aggregate "all ops were dropped" error here.
  //
  // One broken op must produce exactly one diagnostic: that is what makes the count a
  // useful signal to the caller, and a summary error would double it for the single-op
  // patch that is by far the most common shape. "Nothing survived" is already fully
  // visible — `ops` is empty and every removal is reported — so the extra code would add
  // no information a reader could act on.
  return { value: { summary, ops: kept }, diagnostics };
}

type Verdict = { op?: PatchOp; diagnostics: Diagnostic[] };

/**
 * The state a patch builds as it is validated.
 *
 * Only what is needed to judge the *next* op is tracked: which node ids will exist and
 * with which kind, which input ports are wired, and which uniform names are spoken for.
 * ParamValues are not interpreted — `applyPatch` owns value semantics.
 */
class PatchState {
  /** node id -> kind. Seeded from the document, mutated by surviving ops. */
  private readonly nodes = new Map<string, NodeKind>();
  /** `node.port` -> true for inputs that the document or a surviving op wires. */
  private readonly wired = new Set<string>();
  /** uniform name -> declared type, for the clash check. */
  private readonly uniforms = new Map<string, ShaderType>();

  /**
   * Not a TypeScript *parameter property* (`constructor(private readonly catalog: …)`):
   * that syntax is stripped only in transform mode, and these sources must run under
   * plain `--experimental-strip-types` (the test runner's mode). Assigning in the body
   * keeps the package loadable in both.
   */
  private readonly catalog: CatalogLike;

  constructor(catalog: CatalogLike) {
    this.catalog = catalog;
  }

  /** Seed from the document the patch will be applied to. */
  static forDoc(doc: ShaderDoc, catalog: CatalogLike): PatchState {
    const s = new PatchState(catalog);
    for (const n of doc.graph.nodes) s.nodes.set(n.id, n.kind);
    for (const n of doc.graph.nodes) {
      for (const port of Object.keys(n.inputs ?? {})) s.wired.add(`${n.id}.${port}`);
      for (const [port, val] of Object.entries(n.params ?? {})) {
        if (val.kind === 'param') s.uniforms.set(val.name, val.type);
      }
    }
    return s;
  }

  describeNodes(): string {
    return list([...this.nodes.keys()]);
  }

  // --- the catalog checks -------------------------------------------------

  private schema(kind: NodeKind): NodeSchema | undefined {
    return this.catalog[kind];
  }

  private inPort(id: string, port: string) {
    const kind = this.nodes.get(id);
    const schema = kind ? this.schema(kind) : undefined;
    return schema?.inputs.find((p) => p.name === port);
  }

  private outPort(id: string, port?: string) {
    const kind = this.nodes.get(id);
    const schema = kind ? this.schema(kind) : undefined;
    if (!schema) return undefined;
    const name = port ?? schema.primaryOut ?? schema.outputs[0]?.name;
    if (name === undefined) return undefined;
    return schema.outputs.find((p) => p.name === name);
  }

  // --- the per-op entry point --------------------------------------------

  /** Validate one raw op. Returns the op to keep (absent => dropped) plus its diagnostics. */
  check(raw: unknown, index: number): Verdict {
    if (!isRecord(raw)) {
      return { diagnostics: [badOp(index, `op ${index + 1} is ${describe(raw)}, not an object`)] };
    }
    const kind = raw['op'];
    if (typeof kind !== 'string') {
      return { diagnostics: [badOp(index, `op ${index + 1} has no \`op\` field`)] };
    }
    if (!ALL_OPS.has(kind)) {
      return {
        diagnostics: [err(
          AG.UNKNOWN_OP,
          `op ${index + 1}: \`${kind}\` is not a patch op.`,
          undefined,
          'Valid ops: add_node, remove_node, connect, disconnect, set_param, set_render, set_spec, expose_param, inline_param.',
        )],
      };
    }
    switch (kind) {
      case 'add_node':
        return this.addNode(raw, index);
      case 'remove_node':
        return this.removeNode(raw, index);
      case 'set_body':
        return this.setBody(raw, index);
      case 'connect':
        return this.connect(raw, index);
      case 'disconnect':
        return this.disconnect(raw, index);
      case 'set_param':
        return this.setParam(raw, index);
      case 'expose_param':
        return this.exposeParam(raw, index);
      case 'inline_param':
        return this.inlineParam(raw, index);
      case 'set_render':
        return this.setRender(raw, index);
      case 'set_spec':
        return this.setSpec(raw, index);
      default:
        return { diagnostics: [badOp(index, `op ${index + 1}: unhandled op \`${kind}\``)] };
    }
  }

  // --- op validators ------------------------------------------------------

  private addNode(raw: Record<string, unknown>, index: number): Verdict {
    const node = raw['node'];
    if (!isRecord(node)) {
      return { diagnostics: [badOp(index, '`add_node` is missing its `node` object')] };
    }
    const kind = node['kind'];
    if (typeof kind !== 'string' || !KNOWN_KINDS.includes(kind)) {
      return {
        diagnostics: [err(
          AG.UNKNOWN_KIND,
          `op ${index + 1}: \`${String(kind)}\` is not a node kind.`,
          { node: typeof node['id'] === 'string' ? node['id'] : undefined },
          `Known kinds: ${list(KNOWN_KINDS)}.`,
        )],
      };
    }
    const nodeKind = kind as NodeKind;
    const schema = this.schema(nodeKind);
    if (!schema) {
      return {
        diagnostics: [err(
          AG.UNKNOWN_KIND,
          `op ${index + 1}: the catalog has no schema for \`${nodeKind}\`, so its ports cannot be checked.`,
          { node: typeof node['id'] === 'string' ? node['id'] : undefined },
          'Pass a catalog that covers every kind the document uses (see CATALOG in @ai4shader/ir/catalog).',
        )],
      };
    }

    const id = node['id'];
    if (typeof id !== 'string' || !IDENT_RE.test(id)) {
      return {
        diagnostics: [err(
          AG.BAD_OP,
          `op ${index + 1}: \`add_node\` needs an \`id\` matching [A-Za-z_][A-Za-z0-9_]*; got ${describe(id)}.`,
          undefined,
          'Ids become GLSL identifiers, so they may not contain `-`, `.` or start with a digit.',
        )],
      };
    }
    if (this.nodes.has(id)) {
      return {
        diagnostics: [err(
          AG.DUPLICATE_NODE_ID,
          `op ${index + 1}: node id \`${id}\` is already taken.`,
          { node: id },
          `Use a free id — the existing ids are: ${this.describeNodes()}.`,
        )],
      };
    }

    // Ports declared inside `add_node` are checked for the same reason `connect` is: a
    // node added with a hallucinated port parameter would fail inside `applyPatch`, and
    // the whole atomic patch would be rejected for one bad key.
    for (const key of Object.keys(node['params'] ?? {})) {
      if (!schema.inputs.some((p) => p.name === key)) {
        return {
          diagnostics: [err(
            AG.UNKNOWN_INPUT_PORT,
            `op ${index + 1}: \`${id}\` (${nodeKind}) has no input \`${key}\`.`,
            { node: id, port: key },
            `Valid inputs: ${list(inputsOf(schema))}.`,
          )],
        };
      }
    }
    for (const key of Object.keys(node['inputs'] ?? {})) {
      if (!schema.inputs.some((p) => p.name === key)) {
        return {
          diagnostics: [err(
            AG.UNKNOWN_INPUT_PORT,
            `op ${index + 1}: \`${id}\` (${nodeKind}) has no input \`${key}\`.`,
            { node: id, port: key },
            `Valid inputs: ${list(inputsOf(schema))}.`,
          )],
        };
      }
    }

    this.nodes.set(id, nodeKind);
    for (const key of Object.keys(node['inputs'] ?? {})) this.wired.add(`${id}.${key}`);

    const op: PatchOp = { op: 'add_node', node: node as unknown as ShaderNode };
    if (typeof raw['reason'] === 'string') op.reason = raw['reason'];
    return { op, diagnostics: [] };
  }

  private removeNode(raw: Record<string, unknown>, index: number): Verdict {
    const id = raw['node'];
    if (typeof id !== 'string') return { diagnostics: [badOp(index, '`remove_node` needs a `node` id')] };
    if (!this.nodes.has(id)) {
      return {
        diagnostics: [err(
          AG.UNKNOWN_NODE,
          `op ${index + 1}: cannot remove \`${id}\`; no such node.`,
          { node: id },
          `Existing nodes: ${this.describeNodes()}.`,
        )],
      };
    }
    this.nodes.delete(id);
    const op: PatchOp = { op: 'remove_node', node: id };
    if (typeof raw['reason'] === 'string') op.reason = raw['reason'];
    return { op, diagnostics: [] };
  }

  /**
   * `set_body` — the escape hatch's only entry point.
   *
   * Validated against three things a model gets wrong, each with its own message because each
   * needs a different correction: a node that does not exist, a node whose kind has no body,
   * and a body that is empty or not a string. The *snippet* is not validated here — the IR
   * lints it when the patch is applied, which is where the restricted-subset rules live and
   * where the diagnostic can name a line.
   */
  private setBody(raw: Record<string, unknown>, index: number): Verdict {
    const id = raw['node'];
    const body = raw['body'];
    if (typeof id !== 'string' || !id) {
      return { diagnostics: [badOp(index, '`set_body` is missing its `node` id')] };
    }
    if (typeof body !== 'string' || !body.trim()) {
      return {
        diagnostics: [
          badOp(
            index,
            '`set_body` needs a non-empty `body` string',
            'Send the restricted-GLSL text, ending in `return vec4(...)`.',
          ),
        ],
      };
    }
    const kind = this.nodes.get(id);
    if (kind === undefined) {
      return {
        diagnostics: [
          err(
            AG.UNKNOWN_NODE,
            `op ${index + 1}: cannot set the body of \`${id}\`; no such node.`,
            { node: id },
            `Existing nodes: ${this.describeNodes()}.`,
          ),
        ],
      };
    }
    if (kind !== 'custom_glsl') {
      return {
        diagnostics: [
          err(
            AG.BAD_OP,
            `op ${index + 1}: \`${id}\` is a \`${kind}\` node; only \`custom_glsl\` carries a body.`,
            { node: id },
            'Add a `custom_glsl` node (`add_node`) and set its body instead.',
          ),
        ],
      };
    }
    const op: PatchOp = { op: 'set_body', node: id, body };
    if (typeof raw['reason'] === 'string') op.reason = raw['reason'];
    return { op, diagnostics: [] };
  }

  private connect(raw: Record<string, unknown>, index: number): Verdict {    const from = raw['from'];
    const to = raw['to'];
    if (!isRecord(from) || typeof from['node'] !== 'string' || !isRecord(to) || typeof to['node'] !== 'string' || typeof to['port'] !== 'string') {
      return {
        diagnostics: [badOp(
          index,
          '`connect` needs `from: { node, port? }` and `to: { node, port }`',
        )],
      };
    }
    const fromId = from['node'];
    const toId = to['node'];
    const toPort = to['port'];
    const fromPort = typeof from['port'] === 'string' ? from['port'] : undefined;

    // A source that is neither in the document nor in this patch's own `add_node` set is
    // the single most damaging hallucination: it produces a document that references a
    // node nobody created.
    if (!this.nodes.has(fromId)) {
      return {
        diagnostics: [err(
          AG.DANGLING_WIRE,
          `op ${index + 1}: wire reads from \`${fromId}\`, which is neither in the document nor added by this patch.`,
          { node: toId, port: toPort },
          `Add it first (\`add_node\`) or wire from one of: ${this.describeNodes()}.`,
        )],
      };
    }
    if (!this.nodes.has(toId)) {
      return {
        diagnostics: [err(
          AG.UNKNOWN_NODE,
          `op ${index + 1}: wire targets \`${toId}\`, which is neither in the document nor added by this patch.`,
          { node: toId, port: toPort },
          `Nodes available to this patch: ${this.describeNodes()}.`,
        )],
      };
    }

    const srcKind = this.nodes.get(fromId)!;
    const dstKind = this.nodes.get(toId)!;
    const srcSchema = this.schema(srcKind);
    const dstSchema = this.schema(dstKind);
    if (!srcSchema || !dstSchema) {
      const missing = !srcSchema ? srcKind : dstKind;
      return {
        diagnostics: [err(
          AG.UNKNOWN_KIND,
          `op ${index + 1}: the catalog has no schema for \`${missing}\`, so the wire cannot be checked.`,
          { node: toId, port: toPort },
          'Pass a catalog that covers every kind in the document.',
        )],
      };
    }

    const out = this.outPort(fromId, fromPort);
    if (!out) {
      return {
        diagnostics: [err(
          AG.UNKNOWN_OUTPUT_PORT,
          `op ${index + 1}: \`${fromId}\` (${srcKind}) has no output \`${fromPort ?? srcSchema.primaryOut ?? ''}\`.`,
          { node: fromId, port: fromPort },
          `Valid outputs: ${list(outputsOf(srcSchema))}.`,
        )],
      };
    }
    const inPort = this.inPort(toId, toPort);
    if (!inPort) {
      return {
        diagnostics: [err(
          AG.UNKNOWN_INPUT_PORT,
          `op ${index + 1}: \`${toId}\` (${dstKind}) has no input \`${toPort}\`.`,
          { node: toId, port: toPort },
          `Valid inputs: ${list(inputsOf(dstSchema))}.`,
        )],
      };
    }
    if (!canConnect(out.type, inPort.type)) {
      return {
        diagnostics: [err(
          AG.TYPE_MISMATCH,
          `op ${index + 1}: \`${fromId}.${out.name}\` is ${out.type} but \`${toId}.${toPort}\` expects ${inPort.type}.`,
          { node: toId, port: toPort },
          mismatchFix(out.type, inPort.type),
        )],
      };
    }

    this.wired.add(`${toId}.${toPort}`);
    const op: PatchOp = { op: 'connect', from: { node: fromId, ...(fromPort ? { port: fromPort } : {}) }, to: { node: toId, port: toPort } };
    if (typeof raw['reason'] === 'string') op.reason = raw['reason'];
    return { op, diagnostics: [] };
  }

  private disconnect(raw: Record<string, unknown>, index: number): Verdict {
    const to = raw['to'];
    if (!isRecord(to) || typeof to['node'] !== 'string' || typeof to['port'] !== 'string') {
      return { diagnostics: [badOp(index, '`disconnect` needs `to: { node, port }`')] };
    }
    const id = to['node'];
    const port = to['port'];
    if (!this.nodes.has(id)) {
      return {
        diagnostics: [err(
          AG.UNKNOWN_NODE,
          `op ${index + 1}: cannot disconnect \`${id}.${port}\`; no such node.`,
          { node: id, port },
          `Existing nodes: ${this.describeNodes()}.`,
        )],
      };
    }
    if (!this.inPort(id, port)) {
      const kind = this.nodes.get(id)!;
      return {
        diagnostics: [err(
          AG.UNKNOWN_INPUT_PORT,
          `op ${index + 1}: \`${id}\` (${kind}) has no input \`${port}\`.`,
          { node: id, port },
          `Valid inputs: ${list(inputsOf(this.schema(kind)))}.`,
        )],
      };
    }
    this.wired.delete(`${id}.${port}`);
    const op: PatchOp = { op: 'disconnect', to: { node: id, port } };
    if (typeof raw['reason'] === 'string') op.reason = raw['reason'];
    return { op, diagnostics: [] };
  }

  private setParam(raw: Record<string, unknown>, index: number): Verdict {
    const id = raw['node'];
    const param = raw['param'];
    if (typeof id !== 'string' || typeof param !== 'string' || !isRecord(raw['value'])) {
      return { diagnostics: [badOp(index, '`set_param` needs `node`, `param` and a `value` object')] };
    }
    const known = this.checkPort(id, param, index, 'set_param');
    if (!known.ok) return { diagnostics: [known.diagnostic] };
    const op: PatchOp = { op: 'set_param', node: id, param, value: raw['value'] as unknown as ParamValue };
    if (typeof raw['reason'] === 'string') op.reason = raw['reason'];
    return { op, diagnostics: [] };
  }

  private inlineParam(raw: Record<string, unknown>, index: number): Verdict {
    const id = raw['node'];
    const param = raw['param'];
    if (typeof id !== 'string' || typeof param !== 'string') {
      return { diagnostics: [badOp(index, '`inline_param` needs `node` and `param`')] };
    }
    const known = this.checkPort(id, param, index, 'inline_param');
    if (!known.ok) return { diagnostics: [known.diagnostic] };
    const op: PatchOp = { op: 'inline_param', node: id, param };
    if (typeof raw['reason'] === 'string') op.reason = raw['reason'];
    return { op, diagnostics: [] };
  }

  private exposeParam(raw: Record<string, unknown>, index: number): Verdict {
    const id = raw['node'];
    const param = raw['param'];
    const name = raw['name'];
    if (typeof id !== 'string' || typeof param !== 'string') {
      return { diagnostics: [badOp(index, '`expose_param` needs `node` and `param`')] };
    }

    // `expose_param` is the op that turns a literal into a slider, i.e. into a *named
    // uniform* that must be a legal identifier in all six targets. These three checks
    // are the reason this parser imports `isReservedAnywhere`/`safeIdentifier`: a name
    // like `sample` compiles in GLSL and dies in MSL, and the failure would surface as a
    // driver error two stages later with no hint about which uniform caused it.
    if (typeof name !== 'string' || name === '') {
      return {
        diagnostics: [err(
          AG.BAD_UNIFORM_NAME,
          `op ${index + 1}: \`expose_param\` needs a uniform \`name\`.`,
          { node: id, port: param },
          `Send \`"name": "u_${param}"\` — a \`u_\`-prefixed name matching [A-Za-z_][A-Za-z0-9_]*.`,
        )],
      };
    }
    if (!IDENT_RE.test(name)) {
      const suggested = safeIdentifier(name, new Set(this.uniforms.keys()));
      return {
        diagnostics: [err(
          AG.BAD_UNIFORM_NAME,
          `op ${index + 1}: \`${name}\` is not a valid uniform name.`,
          { node: id, port: param },
          `Use \`${suggested}\` (must match [A-Za-z_][A-Za-z0-9_]*).`,
        )],
      };
    }
    if (isReservedAnywhere(name)) {
      const suggested = safeIdentifier(name, new Set(this.uniforms.keys()));
      return {
        diagnostics: [err(
          AG.RESERVED_UNIFORM_NAME,
          `op ${index + 1}: \`${name}\` is reserved in at least one target language.`,
          { node: id, port: param },
          `Use \`${suggested}\` — a name that compiles identically in GLSL, WGSL, HLSL and MSL.`,
        )],
      };
    }

    // The port must exist on the kind. This is the `expose_param`-specific port check
    // the design calls out; it runs before the uniform-name checks so that a hallucinated
    // *port* is reported as a port problem, not as a missing node.
    const known = this.checkPort(id, param, index, 'expose_param');
    if (!known.ok) return { diagnostics: [known.diagnostic] };

    if (this.wired.has(`${id}.${param}`)) {
      return {
        diagnostics: [err(
          AG.EXPOSE_WIRED_PORT,
          `op ${index + 1}: \`${id}.${param}\` is wired, so it has no literal to expose.`,
          { node: id, port: param },
          `Turn the wire into a literal first (\`disconnect\` + \`set_param\`), then expose it — a uniform on a wired port is silently ignored.`,
        )],
      };
    }

    const existing = this.uniforms.get(name);
    const declared = paramKindToType(raw);
    if (existing !== undefined && declared !== null && existing !== declared) {
      return {
        diagnostics: [err(
          AG.UNIFORM_NAME_CLASH,
          `op ${index + 1}: uniform \`${name}\` is already declared as ${existing}, but \`${id}.${param}\` is ${declared}.`,
          { node: id, port: param },
          `Use \`${safeIdentifier(name, new Set(this.uniforms.keys()))}\` instead, or reuse the existing uniform deliberately.`,
        )],
      };
    }
    return this.finishExpose(raw, index, known.type, declared ?? existing);
  }

  private finishExpose(
    raw: Record<string, unknown>,
    index: number,
    portType: ShaderType,
    declared: ShaderType | undefined,
  ): Verdict {
    const name = raw['name'] as string;
    // The uniform's declared type is the *literal's* type when the model told us one
    // (`raw.kind`), else the port's — the same rule `exposeParam` applies via
    // `paramKindToType`. Recording it is what makes the clash check above able to fire
    // for two exposures that disagree.
    this.uniforms.set(name, declared ?? portType);
    const op: PatchOp = {
      op: 'expose_param',
      node: raw['node'] as string,
      param: raw['param'] as string,
      name,
    };
    for (const key of ['min', 'max', 'step'] as const) {
      if (typeof raw[key] === 'number') op[key] = raw[key] as number;
    }
    for (const key of ['label', 'group'] as const) {
      if (typeof raw[key] === 'string') op[key] = raw[key] as string;
    }
    if (typeof raw['reason'] === 'string') op.reason = raw['reason'];
    return { op, diagnostics: [] };
  }

  private setRender(raw: Record<string, unknown>, index: number): Verdict {
    const path = raw['path'];
    const value = raw['value'];
    if (path === 'qualityTier') {
      if (!['low', 'medium', 'high', 'ultra'].includes(String(value))) {
        return { diagnostics: [badOp(index, `\`set_render.qualityTier\` got ${JSON.stringify(value)}`, 'Expected one of: low, medium, high, ultra.')] };
      }
    } else if (path === 'maxIterations') {
      if (typeof value !== 'number' || !Number.isInteger(value) || value < 1) {
        return { diagnostics: [badOp(index, `\`set_render.maxIterations\` got ${JSON.stringify(value)}`, 'Expected a positive integer.')] };
      }
    } else if (path === 'transparent' || path === 'feedback') {
      if (typeof value !== 'boolean') {
        return { diagnostics: [badOp(index, `set_render.${path} expects a boolean`, 'Use true or false, not a string or number.')] };
      }
    } else if (path === 'resolution') {
      if (!Array.isArray(value) || value.length !== 2 || value.some((n) => typeof n !== 'number' || !Number.isFinite(n) || n <= 0)) {
        return { diagnostics: [badOp(index, `\`set_render.resolution\` got ${JSON.stringify(value)}`, 'Expected [width, height] with both > 0.')] };
      }
    } else {
      return { diagnostics: [badOp(index, `\`set_render\` cannot write \`${String(path)}\``, 'Valid paths: qualityTier, maxIterations, resolution, transparent, feedback.')] };
    }
    const op = { op: 'set_render', path, value } as PatchOp;
    if (typeof raw['reason'] === 'string') op.reason = raw['reason'];
    return { op, diagnostics: [] };
  }

  private setSpec(raw: Record<string, unknown>, index: number): Verdict {
    const spec = raw['spec'];
    if (!isRecord(spec)) {
      return { diagnostics: [badOp(index, '`set_spec` is missing its `spec` object')] };
    }
    const op: PatchOp = { op: 'set_spec', spec: spec as Partial<ShaderSpec> };
    if (typeof raw['reason'] === 'string') op.reason = raw['reason'];
    return { op, diagnostics: [] };
  }

  /**
   * Shared node/port existence check for the four ops that name an input port.
   *
   * Returns the port's declared type on success, because two of the four callers
   * (`expose_param`, `set_param`) need it to reason about uniform types.
   */
  private checkPort(id: string, port: string, index: number, opName: string): PortCheck {
    if (!this.nodes.has(id)) {
      return {
        ok: false,
        diagnostic: err(
          AG.UNKNOWN_NODE,
          `op ${index + 1}: \`${opName}\` names \`${id}\`, which is neither in the document nor added by this patch.`,
          { node: id, port },
          `Nodes available to this patch: ${this.describeNodes()}.`,
        ),
      };
    }
    const kind = this.nodes.get(id)!;
    const schema = this.schema(kind);
    if (!schema) {
      return {
        ok: false,
        diagnostic: err(
          AG.UNKNOWN_KIND,
          `op ${index + 1}: the catalog has no schema for \`${kind}\`, so \`${port}\` cannot be checked.`,
          { node: id, port },
          'Pass a catalog that covers every kind in the document.',
        ),
      };
    }
    const spec = schema.inputs.find((p) => p.name === port);
    if (!spec) {
      return {
        ok: false,
        diagnostic: err(
          AG.EXPOSE_UNKNOWN_PORT,
          `op ${index + 1}: \`${id}\` (${kind}) has no input \`${port}\`.`,
          { node: id, port },
          `Valid inputs: ${list(inputsOf(schema))}.`,
        ),
      };
    }
    return { ok: true, type: spec.type };
  }
}

type PortCheck = { ok: true; type: ShaderType } | { ok: false; diagnostic: Diagnostic };

/** Widening rules, identical to `canConnect` in the IR package. */
function canConnect(from: ShaderType, to: ShaderType): boolean {
  if (from === to) return true;
  if (from === 'float' && (to === 'vec2' || to === 'vec3' || to === 'vec4')) return true;
  if (from === 'int' && (to === 'float' || to === 'vec2' || to === 'vec3' || to === 'vec4')) return true;
  if (from === 'bool' && to === 'float') return true;
  return false;
}

function mismatchFix(from: ShaderType, to: ShaderType): string {
  const arity: Record<ShaderType, number> = { float: 1, vec2: 2, vec3: 3, vec4: 4, mat2: 4, mat3: 9, int: 1, bool: 1 };
  if (arity[from] > arity[to]) {
    return `Insert a \`vec_op\` node to reduce ${from} to ${to} (or swizzle .${'xyzw'.slice(0, arity[to])}).`;
  }
  return `Insert a \`vec_op\`/constructor node to widen ${from} to ${to}.`;
}

function badOp(index: number, what: string, fix = 'Send a well-formed op: `{ "op": "...", "reason": "..." }`.'): Diagnostic {
  return err(AG.BAD_OP, `op ${index + 1}: ${what}.`, undefined, fix);
}

/**
 * The uniform type an `expose_param`'s sibling literal would lower to, read off the raw
 * op. `expose_param` does not carry the literal, so the model usually says nothing and
 * this returns `null` — the port's type is then used instead, which is the same rule
 * `exposeParam` applies via its own `paramKindToType`.
 */
function paramKindToType(raw: Record<string, unknown>): ShaderType | null {
  switch (raw['kind'] ?? raw['valueKind']) {
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
    case 'color':
      return 'vec4';
    default:
      return null;
  }
}

/* -------------------------------------------------------------------------- */
/* Text -> typed object                                                        */
/* -------------------------------------------------------------------------- */

/** `extractJson` + `parseSpec`. The one call a provider needs for an intent reply. */
export function parseIntentJson(text: string): Parsed<ShaderSpec> {
  const extracted = extractJson(text);
  if (extracted.value === undefined) return { value: { intent: '', category: 'other', tags: [], features: [] }, diagnostics: extracted.diagnostics };
  const parsed = parseSpec(extracted.value);
  return { value: parsed.value, diagnostics: [...extracted.diagnostics, ...parsed.diagnostics] };
}

/** `extractJson` + `parsePatch`. The one call a provider needs for a patch reply. */
export function parsePatchJson(text: string, catalog: CatalogLike, doc?: ShaderDoc): Parsed<ShaderPatch> {
  const extracted = extractJson(text);
  if (extracted.value === undefined) {
    return {
      value: { summary: 'no JSON in reply', ops: [] },
      diagnostics: extracted.diagnostics,
    };
  }
  const parsed = doc ? parsePatchFor(doc, extracted.value, catalog) : parsePatch(extracted.value, catalog);
  return { value: parsed.value, diagnostics: [...extracted.diagnostics, ...parsed.diagnostics] };
}

/**
 * How many ops were dropped by a parse, for the caller's log line. `diagnostics.length`
 * is not the same number — a parse also emits warnings about the JSON wrapper.
 */
export function droppedOps(diagnostics: readonly Diagnostic[]): number {
  return diagnostics.filter((d) => d.severity === 'error').length;
}

/** Re-exported so a caller can name the shape without a second import. */
export type { WireRef };
