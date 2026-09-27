/**
 * AI4Shader — Restricted-GLSL → WGSL lowering pass
 * =================================================
 *
 * `rewrite.ts` is a **token** transform: it renames types (`float` → `f32`) and functions
 * (`mix_` → `wf_mix`) and nothing else. That is sufficient for GLSL, HLSL and MSL, whose
 * grammar is C-like, but WGSL is not C-like in exactly the places the subset uses most:
 *
 *     float hash11(float p) {            fn hash11(p: f32) -> f32 {
 *       p = fract_(p * 0.1031);            var p_v: f32 = p;          // params are immutable
 *       return fract_(p * (p + p));        p_v = fract(p_v * 0.1031);
 *     }                                    return fract(p_v * (p_v + p_v));
 *                                        }
 *
 *     vec2 i = floor(p);                 var i: vec2f = floor(p);     // no C declarations
 *     for (int k = 0; k < 8; k++)        for (var k: i32 = 0; k < 8; k = k + 1)
 *
 * This module is the second half of the WGSL target: it consumes one chunk of restricted
 * GLSL (the shared helper library, or the emitted node-body block), asks `opts.rewrite`
 * for the WGSL *spellings*, and then **restructures** the result — signatures,
 * declarations, loop headers, parameter mutation — at token granularity. Everything in
 * between (every expression, every comment, every blank line) is passed through
 * byte-for-byte, which is what keeps the emitted library readable and auditable.
 *
 * ── WHAT THE SUBSET GUARANTEES, AND WHY THAT IS ENOUGH ────────────────────────
 *
 * The grammar this pass has to cover is closed (see `glslSubset.ts`):
 *
 *   • statements: declarations, assignments, `if`/`else` with braced blocks,
 *     `for (int i = 0; i < N; i++)` with a literal N, `return`;
 *   • expressions: literals, constructors, operators, swizzles, function calls, no
 *     ternary, no arrays, no bitwise ops;
 *   • no global state, no structs, no recursion, no samplers.
 *
 * Anything outside that surface is reported as an `error` diagnostic in the `TGT0xx`
 * range rather than lowered "plausibly": a diagnostic is a correct outcome, silently
 * emitting source a WGSL compiler rejects is not.
 *
 * ── KNOWN LIMITS (honest list, see also the WGSL section of targets/README.md) ─
 *
 *   1. `?:` has no lowering here (WGSL spells it `select`, which needs the condition's
 *      type and an argument swap). Reported as `TGT005`, never emitted verbatim.
 *   2. `wf_mix`/`wf_mod` are *inlined* rather than called. `rewrite.ts` maps `mix_`/`mod_`
 *      to scalar-only shims, but WGSL has no overloading and the library calls them with
 *      `vec3`/`vec4`/`vec2` arguments (`hsv2rgb`, `rgb2hsv`, `srgb2linear`, `sd_triangle`,
 *      the `grade` node). `a + (b - a) * t` and `a - b * floor(a / b)` are component-wise
 *      and identical to the shims, so one text form serves every arity.
 *   3. Loop bounds must be literals (`ALLOWED` says they are). An identifier bound would
 *      need the identifier's type to choose `i32`/`f32`; reported as `TGT003`.
 *   4. Numbers are formatted here, not by `rewriteNumber`: a literal that is the argument
 *      of `i32(...)` or the init/bound of an integer loop stays an integer literal.
 *   5. **Scalar bounds are broadcast by hand.** GLSL overloads `clamp`/`min`/`max`/`step`/
 *      `smoothstep` for a vector with *scalar* bounds; WGSL's builtin table does not
 *      (`clamp(e: T, low: T, high: T)` only — the mixed overload is an open request,
 *      gpuweb/gpuweb#4815), and the shared library relies on the GLSL form:
 *
 *          clamp(abs(mod_(…, 6.0) - 3.0) - 1.0, 0.0, 1.0)   // hsv2rgb, a vec3
 *
 *      `vecNf(scalar)` is a legal WGSL broadcast, so the scalar argument is wrapped with
 *      the arity of the *proven* vector argument. `shapeOf()` is a deliberately small,
 *      conservative shape analysis — it never guesses, and an unknown shape leaves the
 *      call untouched.
 *
 * ── ORDER OF OPERATIONS ───────────────────────────────────────────────────────
 *
 *      src (restricted GLSL)
 *        → opts.rewrite          token renames, once, whole chunk (owns all lint/RW diags)
 *        → inlineScalarShims     wf_mix/wf_mod call sites → component-wise expressions
 *        → normaliseIntLiterals  i32(5.0) → i32(5)
 *        → broadcastScalarBounds clamp(v3, 0.0, 1.0) → clamp(v3, vec3f(0.0), vec3f(1.0))
 *        → structural scan       edits: signatures, `var`s, for-headers, param copies
 *        → structuralBalanceProblems   final self-check; a failure becomes TGT090
 */

import { type Diagnostic } from '../types.js';
import { tokenize, type Token } from '../rewrite.js';

/* -------------------------------------------------------------------------- */
/* Public shapes                                                               */
/* -------------------------------------------------------------------------- */

/** Proven component count of an expression: `1` is a scalar, `2..4` a vector. */
export type WgslShape = 1 | 2 | 3 | 4;

export type WgslLowerOptions = {
  /**
   * Rename restricted-GLSL identifiers into WGSL spellings — i.e. `rewriteSource(src,
   * 'wgsl').source` (plus whatever fixpoint the caller needs). Called **once**, on the
   * whole chunk, so the caller's lint diagnostics are reported exactly once.
   */
  rewrite: (src: string) => string;
  /** Human label used in diagnostics, e.g. `wgsl: shared library`. */
  context?: string;
  /**
   * Names the epilogue declares immutably (`let uv = …`, uniform members). An assignment
   * to one of them is not valid WGSL and is reported as `TGT007` instead of being
   * passed through.
   */
  immutableNames?: readonly string[];
  /**
   * Component count of every name the chunk does not declare itself — host symbols
   * (`iResolution: vec3f`) and uniforms. Used *only* to prove an argument's arity before
   * broadcasting scalar bounds (see note 5); an unknown name simply disables the rewrite.
   */
  symbols?: Readonly<Record<string, WgslShape>>;
  /**
   * Return shape of functions defined in *another* chunk — the shared library, when the
   * node bodies are being lowered. Feed back what `WgslLowerResult.signatures` returned.
   */
  signatures?: Readonly<Record<string, WgslShape>>;
};

export type WgslLowerResult = {
  source: string;
  diagnostics: Diagnostic[];
  /** Return shape of every function this chunk defined, for the next chunk's `signatures`. */
  signatures: Record<string, WgslShape>;
};

/* -------------------------------------------------------------------------- */
/* Vocabulary                                                                  */
/* -------------------------------------------------------------------------- */

/** WGSL type spellings `WGSL_TABLES` produces. */
const WGSL_TYPES = [
  'f32', 'i32', 'bool',
  'vec2f', 'vec3f', 'vec4f',
  'mat2x2f', 'mat3x3f',
] as const;

const TYPE_SET = new Set<string>(WGSL_TYPES);

/** `void` is the one GLSL spelling the rewriter leaves alone: it is not a *value* type. */
const VOID = 'void';

/** Assignment operators. `++`/`--` are handled separately (postfix mutation statements). */
const ASSIGN_OPS = new Set([
  '=', '+=', '-=', '*=', '/=', '%=', '&=', '|=', '^=', '<<=', '>>=',
]);

/** Punctuation WGSL either lacks or the subset bans; each one is reported, never kept. */
const UNSUPPORTED_PUNCT = new Set([
  '%', '%=', '&', '&=', '|', '|=', '^', '^=', '<<', '>>', '<<=', '>>=', '~', '#', '[', ']',
]);

/**
 * Call sites of these are inlined, because the shims that replace them are scalar-only
 * and WGSL has no function overloading. See note 2 in the header.
 *
 * Every argument is parenthesised when it is not atomic. `a - b * floor(a / b)` with
 * `b = 2.0 * an` is *not* `a - 2.0 * an * floor(a / 2.0 * an)`: splicing argument text
 * without thinking about precedence is the classic way a text-level lowering produces
 * plausible-looking, numerically wrong source.
 */
const SCALAR_SHIMS: Record<string, { arity: number; build: (args: string[]) => string }> = {
  wf_mix: {
    arity: 3,
    build: ([a, b, t]) => {
      const x = atomic(a!);
      const y = atomic(b!);
      const k = atomic(t!);
      return `(${x} + (${y} - ${x}) * ${k})`;
    },
  },
  wf_mod: {
    arity: 2,
    build: ([a, b]) => {
      const x = atomic(a!);
      const y = atomic(b!);
      return `(${x} - ${y} * floor(${x} / ${y}))`;
    },
  },
};

/** Parenthesise `text` unless it is already a single primary expression. */
function atomic(text: string): string {
  const t = text.trim();
  if (t.length === 0) return t;
  if (t.includes('//') || t.includes('/*')) return `(${t})`; // comments hide brackets
  if (/^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$/.test(t)) return t; // x, u.x, p_v.y
  if (/^-?(?:\d[\w.]*|\.\d[\w]*)$/.test(t)) return t; // 1.0, .5, 2
  if (isSingleGroup(t)) return t; // ( … ) and f( … )
  return `(${t})`;
}

/** True when the text is `[name]( … )` whose bracket closes on the very last character. */
function isSingleGroup(t: string): boolean {
  const open = t.indexOf('(');
  if (open < 0) return false;
  const head = t.slice(0, open);
  if (head.length > 0 && !/^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$/.test(head)) return false;
  let depth = 0;
  for (let i = open; i < t.length; i++) {
    const c = t[i]!;
    if (c === '(') depth++;
    else if (c === ')') {
      depth--;
      if (depth === 0) return i === t.length - 1;
    }
  }
  return false;
}

/* -------------------------------------------------------------------------- */
/* Scanner state                                                               */
/* -------------------------------------------------------------------------- */

type Edit = { start: number; end: number; text: string; why: string };

/** One function body being lowered: what its parameters are and which ones are written. */
type FnCtx = {
  name: string;
  /** Parameter name → declared WGSL type, in declaration order. */
  params: Map<string, string>;
  /** Parameters the body assigns to (directly, component-wise, or with `++`). */
  assigned: Set<string>;
  bodyOpen: number;
  bodyClose: number;
};

type State = {
  src: string;
  toks: Token[];
  edits: Edit[];
  diags: Diagnostic[];
  context: string;
  immutable: Set<string>;
};

/* -------------------------------------------------------------------------- */
/* Entry point                                                                 */
/* -------------------------------------------------------------------------- */

export function lowerSubsetToWgsl(src: string, opts: WgslLowerOptions): WgslLowerResult {
  const context = opts.context ?? 'wgsl lowering';
  const diags: Diagnostic[] = [];

  // 1. The token renames. Everything below works on WGSL spellings.
  const renamed = opts.rewrite(src);

  // 2. Scalar shims that the library calls with vector arguments.
  const inlined = inlineScalarShims(renamed, diags, context);

  // 3. Integer literals in integer position.
  const normalised = normaliseIntLiterals(inlined, diags, context);

  // 4. Scalar bounds of vector builtins, broadcast to the vector's arity.
  const primed: State = {
    src: normalised,
    toks: tokenize(normalised),
    edits: [],
    diags,
    context,
    immutable: new Set(opts.immutableNames ?? []),
  };
  const { source: broadcast, signatures } = broadcastScalarBounds(primed, opts);

  // 5. Structure.
  const state: State = {
    src: broadcast,
    toks: tokenize(broadcast),
    edits: [],
    diags,
    context,
    immutable: new Set(opts.immutableNames ?? []),
  };
  checkUnsupportedTokens(state);
  scanStatements(state, 0, state.toks.length, null);
  const source = applyEdits(state);

  // 6. A lowering pass that silently returns unbalanced source is worse than one that
  //    says it failed, so the output is re-derived and checked here, not only in tests.
  for (const problem of structuralBalanceProblems(source)) {
    diags.push({
      severity: 'error',
      code: 'TGT090_OUTPUT_STRUCTURE',
      message: `${context}: ${problem}`,
      fix: 'This is a bug in the WGSL lowering pass, not in the input: report it with the node/library body that produced it.',
    });
  }

  return { source, diagnostics: diags, signatures };
}

/* -------------------------------------------------------------------------- */
/* Pass 2 — scalar shim call sites                                             */
/* -------------------------------------------------------------------------- */

/**
 * Expand `wf_mix(a, b, t)` → `(a + (b - a) * t)` and `wf_mod(a, b)` → `(a - b * floor(a / b))`.
 *
 * Leftmost-first with the arguments expanded recursively *before* substitution, so a
 * nested call is expanded exactly once (the naive "expand and restart" loop duplicates
 * arguments and can grow exponentially on nested calls).
 */
function inlineScalarShims(src: string, diags: Diagnostic[], context: string): string {
  const toks = tokenize(src);
  for (let i = 0; i < toks.length; i++) {
    const t = toks[i]!;
    if (t.kind !== 'ident') continue;
    const shim = SCALAR_SHIMS[t.text];
    if (!shim || isMemberAccess(toks, i)) continue;
    const open = nextMeaningful(toks, i);
    if (toks[open]?.text !== '(') continue;
    const close = matchingClose(toks, open, '(', ')');
    if (close < 0) {
      report(diags, context, src, 'TGT010_SHIM_ARITY', `\`${t.text}\` has an unterminated argument list.`, `Call as ${t.text}(…) with ${shim.arity} arguments.`, t.pos);
      return src;
    }
    const args = splitTopLevel(toks, open + 1, close).map((a) =>
      inlineScalarShims(sliceRange(toks, src, a[0], a[1]), diags, context),
    );
    if (args.length !== shim.arity) {
      report(diags, context, src, 'TGT010_SHIM_ARITY', `\`${t.text}\` called with ${args.length} argument(s); the shim takes ${shim.arity}.`, `Call it with exactly ${shim.arity} arguments.`, t.pos);
      return src;
    }
    const replacement = shim.build(args);
    return inlineScalarShims(`${src.slice(0, t.pos)}${replacement}${src.slice(tokenEnd(toks[close]!))}`, diags, context);
  }
  return src;
}

/* -------------------------------------------------------------------------- */
/* Pass 3 — integer literals in integer position                               */
/* -------------------------------------------------------------------------- */

/**
 * `i32(5.0)` → `i32(5)`. `rewriteNumber` (see `rewrite.ts`) appends `.0` to every integer
 * literal for the WGSL target, which is right for float position and wrong here. Only a
 * *literal* argument is rewritten: `i32(x)` and `i32(2.5)` are left alone (`i32` of a
 * float is a legal WGSL conversion, it just is not what a literal integer means).
 */
function normaliseIntLiterals(src: string, diags: Diagnostic[], context: string): string {
  const toks = tokenize(src);
  const edits: Edit[] = [];
  for (let i = 0; i < toks.length; i++) {
    const t = toks[i]!;
    if (t.kind !== 'ident' || t.text !== 'i32' || isMemberAccess(toks, i)) continue;
    const open = nextMeaningful(toks, i);
    if (toks[open]?.text !== '(') continue;
    const close = matchingClose(toks, open, '(', ')');
    if (close < 0) continue;
    const inner = meaningfulIndices(toks, open + 1, close);
    const sign = inner.length === 2 && toks[inner[0]!]!.text === '-' ? '-' : '';
    const numIdx = inner[inner.length - 1];
    if (numIdx === undefined) continue;
    if (inner.length !== 1 && sign === '') continue;
    const num = toks[numIdx]!;
    if (num.kind !== 'number') continue;
    const asInt = integerLiteralText(num.text);
    if (asInt === null) continue;
    edits.push({
      start: toks[inner[0]!]!.pos,
      end: tokenEnd(num),
      text: `${sign}${asInt}`,
      why: 'integer conversion argument',
    });
  }
  return applyEditList(src, edits, diags, context);
}

/* -------------------------------------------------------------------------- */
/* Pass 4 — scalar bounds of vector builtins                                   */
/* -------------------------------------------------------------------------- */

/**
 * GLSL overloads `clamp`/`min`/`max`/`step`/`smoothstep` for "vector with scalar bounds";
 * WGSL's builtin table does not (see note 5 in the module header). The fix is mechanical:
 * wrap the scalar argument in a `vecNf(...)` broadcast of the vector argument's arity,
 * which is both a legal constructor and exactly GLSL's component-wise semantics.
 *
 * `shapeOf` is the smallest analysis that can prove an arity, and it never guesses:
 * anything it cannot prove returns `null`, and a `null` disables the rewrite for that
 * call. A wrong *vector* claim could emit invalid WGSL, so every rule below is one that
 * holds for any well-typed input rather than a heuristic.
 */
const SCALAR_BROADCAST_ARGS: Record<string, number[]> = {
  // name -> argument positions that must carry the vector's arity
  clamp: [0, 1, 2],
  min: [0, 1],
  max: [0, 1],
  step: [0], // `step(edge, x)`: only the edge may be a scalar in GLSL
  smoothstep: [0, 1], // `smoothstep(e0, e1, x)`
};

/** Builtins whose result has the shape of one argument (component-wise functions). */
const SHAPE_OF_ARG0 = new Set([
  'abs', 'acos', 'asin', 'atan', 'ceil', 'cos', 'degrees', 'exp', 'exp2', 'floor',
  'fract', 'inverseSqrt', 'log', 'log2', 'max', 'min', 'clamp', 'mix', 'wf_mix',
  'wf_mod', 'mod', 'normalize', 'pow', 'radians', 'reflect', 'refract', 'saturate',
  'sign', 'sin', 'smoothstep', 'sqrt', 'step', 'tan', 'trunc', 'round', 'faceforward',
]);

/** Builtins whose result is always one scalar. */
const SHAPE_SCALAR = new Set(['f32', 'i32', 'bool', 'dot', 'length', 'distance', 'wf_atan2']);

function shapeOfTypeName(type: string): WgslShape | null {
  switch (type) {
    case 'f32':
    case 'i32':
    case 'bool':
      return 1;
    case 'vec2f':
      return 2;
    case 'vec3f':
      return 3;
    case 'vec4f':
      return 4;
    default:
      return null; // mat2x2f / mat3x3f / anything else is not a vector
  }
}

/** Swizzle → arity: `.x` is scalar, `.xyzw`/`.rgba` is the number of letters. */
function shapeOfSwizzle(member: string): WgslShape | null {
  if (!/^[xyzwrgba]{1,4}$/.test(member)) return null;
  const unique = new Set(member);
  if (unique.size !== member.length) return null; // `.xx` is a broadcast, not a type
  return member.length as WgslShape;
}

/**
 * The proven component count of the expression in `[from, to)`, or `null` when it cannot
 * be proven. Precedence is irrelevant here: for arithmetic a scalar operand is broadcast,
 * so the result's shape is the shape of any vector operand, whatever the tree looks like.
 */
function shapeOfExpr(
  toks: Token[],
  from: number,
  to: number,
  symbols: ReadonlyMap<string, WgslShape>,
): WgslShape | null {
  const idx = meaningfulIndices(toks, from, to);
  if (idx.length === 0) return null;
  return shapeOfRange(toks, idx, 0, idx.length - 1, symbols, 0);
}

const ARITH_OPS = new Set(['+', '-', '*', '/']);
const LOGICAL_OPS = new Set(['&&', '||', '==', '!=', '<', '>', '<=', '>=']);

function shapeOfRange(
  toks: Token[],
  idx: number[],
  lo: number,
  hi: number,
  symbols: ReadonlyMap<string, WgslShape>,
  depth: number,
): WgslShape | null {
  if (depth > 32 || lo > hi) return null;
  // `( … )` wrapping the whole expression is not an operand.
  while (toks[idx[lo]!]!.text === '(' && matchingClose(toks, idx[lo]!, '(', ')') === idx[hi]) {
    lo++;
    hi--;
    if (lo > hi) return null;
  }
  // Logical/comparison operators yield a scalar bool; stop at the first one found.
  for (let i = lo; i <= hi; i++) {
    const t = toks[idx[i]!]!;
    if (t.kind === 'punct' && LOGICAL_OPS.has(t.text)) return 1;
  }
  // Otherwise split on the last top-level arithmetic operator (rightmost is enough:
  // the merge below is associative in shape terms).
  for (let i = hi; i > lo; i--) {
    const t = toks[idx[i]!]!;
    if (t.kind !== 'punct' || !ARITH_OPS.has(t.text)) continue;
    const left = shapeOfRange(toks, idx, lo, i - 1, symbols, depth + 1);
    const right = shapeOfRange(toks, idx, i + 1, hi, symbols, depth + 1);
    if (left === null && right === null) return null;
    if (left === null) return right;
    if (right === null) return left;
    if (left === 1) return right;
    if (right === 1) return left;
    return left === right ? left : null;
  }
  // Unary prefix.
  const first = toks[idx[lo]!]!;
  if (first.kind === 'punct' && first.text === '!') return 1;
  if (first.kind === 'punct' && (first.text === '-' || first.text === '+')) {
    return shapeOfRange(toks, idx, lo + 1, hi, symbols, depth + 1);
  }
  // A call or a bare name, optionally followed by member accesses.
  if (first.kind === 'ident') {
    const i = lo + 1;
    if (i <= hi && toks[idx[i]!]!.text === '(') {
      const open = idx[i]!;
      const close = matchingClose(toks, open, '(', ')');
      if (close < 0) return null;
      const afterClose = idx.indexOf(close) + 1;
      const name = first.text;
      const args = splitTopLevel(toks, open + 1, close);
      let shape: WgslShape | null;
      if (SHAPE_SCALAR.has(name)) shape = 1;
      else if (name === 'vec2f') shape = 2;
      else if (name === 'vec3f') shape = 3;
      else if (name === 'vec4f') shape = 4;
      else if (name === 'cross') shape = 3;
      else if (name === 'textureSample') shape = 4;
      else if (name === 'atan2') {
        // `atan2(y, x) -> T` but `atan2(vec2f) -> f32`.
        const only = args.length === 1 ? shapeOfExpr(toks, args[0]![0], args[0]![1], symbols) : null;
        if (only === 2) shape = 1;
        else shape = args.length > 0 ? shapeOfExpr(toks, args[0]![0], args[0]![1], symbols) : null;
      } else if (SHAPE_OF_ARG0.has(name) && args.length > 0) {
        shape = shapeOfExpr(toks, args[0]![0], args[0]![1], symbols);
      } else {
        shape = symbols.get(name) ?? null;
      }
      // `.xy` on a call result, or any further member access.
      if (shape !== null && afterClose <= hi && toks[idx[afterClose]!]!.text === '.') {
        const member = toks[idx[afterClose + 1] ?? -1];
        if (!member || member.kind !== 'ident') return null;
        return shapeOfSwizzle(member.text);
      }
      return shape;
    }
    // `name.member` (swizzle) or a bare name.
    if (i + 1 <= hi && toks[idx[i]!]!.text === '.') {
      const member = toks[idx[i + 1]!]!;
      if (member.kind !== 'ident' || i + 1 !== hi) return null;
      return shapeOfSwizzle(member.text);
    }
    if (i <= hi) return null;
    if (first.text === 'true' || first.text === 'false') return 1;
    return symbols.get(first.text) ?? null;
  }
  if (first.kind === 'number' && lo === hi) return 1;
  return null;
}

/** First and last token offset of `[from, to)`, ignoring trivia. */
function rangeSpan(toks: Token[], from: number, to: number): { start: number; end: number } | null {
  const idx = meaningfulIndices(toks, from, to);
  if (idx.length === 0) return null;
  return { start: toks[idx[0]!]!.pos, end: tokenEnd(toks[idx[idx.length - 1]!]!) };
}

/**
 * Collect function headers: parameter shapes for the body scan, and the return shape to
 * hand to the next chunk.
 *
 * This pass runs *before* the structural one, so the signatures it sees are still
 * GLSL-shaped with the types already renamed (`f32 hash11(f32 p) {`). The WGSL spelling
 * (`fn hash11(p: f32) -> f32 {`) is accepted too, so the pass stays correct if a caller
 * runs it over already-lowered text.
 */
type FnHeader = {
  name: string;
  params: Map<string, WgslShape>;
  returns: WgslShape | null;
  bodyOpen: number;
  bodyClose: number;
};

function collectFnHeaders(toks: Token[]): FnHeader[] {
  const out: FnHeader[] = [];
  for (let i = 0; i < toks.length; i++) {
    const t = toks[i]!;
    if (t.kind !== 'ident') continue;

    let nameIdx = -1;
    let returns: WgslShape | null = null;
    let wgslSpelling = false;
    if (t.text === 'fn') {
      nameIdx = nextMeaningful(toks, i);
      wgslSpelling = true;
    } else {
      // `TYPE name(` — a definition, not a declaration (`TYPE name =`) or a call.
      const maybeName = nextMeaningful(toks, i);
      const maybeOpen = toks[maybeName]?.kind === 'ident' ? nextMeaningful(toks, maybeName) : -1;
      if (toks[maybeName]?.kind === 'ident' && toks[maybeOpen]?.text === '(') {
        nameIdx = maybeName;
        returns = shapeOfTypeName(t.text);
      } else {
        continue;
      }
    }

    const nameTok = toks[nameIdx];
    const openIdx = nameTok ? nextMeaningful(toks, nameIdx) : -1;
    if (!nameTok || nameTok.kind !== 'ident' || toks[openIdx]?.text !== '(') continue;
    const closeIdx = matchingClose(toks, openIdx, '(', ')');
    if (closeIdx < 0) continue;

    const params = new Map<string, WgslShape>();
    for (const [a, b] of splitTopLevel(toks, openIdx + 1, closeIdx)) {
      const items = meaningfulIndices(toks, a, b);
      // `TYPE name` (GLSL) or `name: TYPE` (WGSL).
      const typeIdx = wgslSpelling ? items[2] : items[0];
      const paramIdx = wgslSpelling ? items[0] : items[1];
      if (items.length < 2 || typeIdx === undefined || paramIdx === undefined) continue;
      const typeTok = toks[typeIdx]!;
      const paramTok = toks[paramIdx]!;
      if (typeTok.kind !== 'ident' || paramTok.kind !== 'ident') continue;
      const shape = shapeOfTypeName(typeTok.text);
      if (shape !== null) params.set(paramTok.text, shape);
    }

    // `-> T` when the structural pass has already run.
    let bodyIdx = nextMeaningful(toks, closeIdx);
    if (toks[bodyIdx]?.text === '-') {
      const typeIdx = nextMeaningful(toks, nextMeaningful(toks, bodyIdx));
      const retTok = toks[typeIdx];
      if (retTok && retTok.kind === 'ident') returns = shapeOfTypeName(retTok.text);
      bodyIdx = nextMeaningful(toks, typeIdx);
    }
    if (toks[bodyIdx]?.text !== '{') continue;
    const bodyClose = matchingClose(toks, bodyIdx, '{', '}');
    if (bodyClose < 0) continue;
    out.push({ name: nameTok.text, params, returns, bodyOpen: bodyIdx, bodyClose });
    i = bodyIdx;
  }
  return out;
}

/**
 * Wrap the scalar bounds of a vector `clamp`/`min`/`max`/`step`/`smoothstep` call.
 *
 * Runs over the whole chunk once; the symbol table starts from `opts.symbols` (host
 * symbols and uniforms), gains every function's parameters as its body is entered, and
 * gains every local at its declaration (GLSL requires declaration before use, so a single
 * forward pass sees every name that can appear at a call site).
 */
function broadcastScalarBounds(
  s: State,
  opts: WgslLowerOptions,
): { source: string; signatures: Record<string, WgslShape> } {
  const headers = collectFnHeaders(s.toks);
  const signatures: Record<string, WgslShape> = {};
  for (const h of headers) if (h.returns !== null) signatures[h.name] = h.returns;

  const symbols = new Map<string, WgslShape>();
  for (const [name, shape] of Object.entries(opts.symbols ?? {})) symbols.set(name, shape);
  for (const [name, shape] of Object.entries(opts.signatures ?? {})) symbols.set(name, shape);

  const edits: Edit[] = [];
  const local = new Map<string, WgslShape>(symbols);
  let active: FnHeader | null = null;

  for (let i = 0; i < s.toks.length; i++) {
    const t = s.toks[i]!;
    if (t.kind === 'comment') continue;

    // Enter / leave a function: parameters come into scope, locals go out of it.
    const entering = headers.find((h) => h.bodyOpen === i);
    if (entering) {
      local.clear();
      for (const [n, sh] of symbols) local.set(n, sh);
      for (const [n, sh] of entering.params) local.set(n, sh);
      active = entering;
      continue;
    }
    if (active && i === active.bodyClose) {
      local.clear();
      for (const [n, sh] of symbols) local.set(n, sh);
      active = null;
      continue;
    }

    // Record a declaration (`f32 x = …`, `vec3f x = …`, `i32 i = …`) as its own type.
    if (t.kind === 'ident') {
      const shape = shapeOfTypeName(t.text);
      if (shape !== null) {
        const nameIdx = nextMeaningful(s.toks, i);
        const nameTok = s.toks[nameIdx];
        const afterTok = nameTok ? s.toks[nextMeaningful(s.toks, nameIdx)] : undefined;
        if (
          nameTok?.kind === 'ident' &&
          afterTok?.kind === 'punct' &&
          (afterTok.text === '=' || afterTok.text === ';')
        ) {
          local.set(nameTok.text, shape);
        }
      }
    }

    // A broadcast candidate.
    const positions = t.kind === 'ident' ? SCALAR_BROADCAST_ARGS[t.text] : undefined;
    if (!positions || isMemberAccess(s.toks, i)) continue;
    const openIdx = nextMeaningful(s.toks, i);
    if (s.toks[openIdx]?.text !== '(') continue;
    const closeIdx = matchingClose(s.toks, openIdx, '(', ')');
    if (closeIdx < 0) continue;
    const args = splitTopLevel(s.toks, openIdx + 1, closeIdx);
    const shapes = args.map((a) => shapeOfExpr(s.toks, a[0], a[1], local));

    // The vector argument decides the arity: `clamp`/`min`/`max` take it first, `step`
    // and `smoothstep` last (that is the only mixed form GLSL declares for them).
    const vectorAt = t.text === 'step' || t.text === 'smoothstep' ? args.length - 1 : 0;
    const vector = shapes[vectorAt];
    if (vector === undefined || vector === null || vector === 1) continue;

    for (const position of positions) {
      if (position === vectorAt || position >= args.length) continue;
      if (shapes[position] !== 1) continue; // unknown or already a vector: leave it alone
      const [a, b] = args[position]!;
      const span = rangeSpan(s.toks, a, b);
      if (!span) continue;
      edits.push({
        start: span.start,
        end: span.end,
        text: `vec${vector}f(${sliceRange(s.toks, s.src, a, b)})`,
        why: `${t.text} scalar bound broadcast`,
      });
    }
  }

  return { source: applyEditList(s.src, edits, s.diags, s.context), signatures };
}

/* -------------------------------------------------------------------------- */
/* Pass 5 — function signatures, declarations, loops, immutable parameters      */
/* -------------------------------------------------------------------------- */

/**
 * Statements that WGSL does not accept at all, reported up front so the structural scan
 * can keep going. Expressions are otherwise passed through untouched, which is only safe
 * because these tokens cannot appear in a valid lowering.
 */
function checkUnsupportedTokens(s: State): void {
  for (let i = 0; i < s.toks.length; i++) {
    const t = s.toks[i]!;
    if (t.kind === 'punct' && t.text === '?') {
      report(s.diags, s.context, s.src, 'TGT005_UNSUPPORTED_TERNARY', '`?:` appears in this chunk.', 'WGSL has no ternary operator: rewrite the expression with `mix_`/`step` (branchless) or an `if` that assigns a local.', t.pos);
      continue;
    }
    if (t.kind === 'punct' && UNSUPPORTED_PUNCT.has(t.text)) {
      report(s.diags, s.context, s.src, 'TGT006_UNSUPPORTED_OPERATOR', `\`${t.text}\` is not available in the WGSL target.`, `\`${t.text}\` is outside the restricted subset (no arrays, no bitwise ops, no \`%\`); rewrite using the portable scalar helpers.`, t.pos);
    }
  }
}

function scanStatements(s: State, from: number, to: number, fn: FnCtx | null): void {
  let i = from;
  while (i < to) {
    let k = i;
    while (k < to && isTrivia(s.toks[k])) k++;
    if (k >= to) return;
    const t = s.toks[k]!;
    // A closing token belongs to the caller's block.
    if (t.kind === 'punct' && (t.text === '}' || t.text === ')')) return;
    if (t.kind === 'punct' && t.text === '{') {
      const close = matchingClose(s.toks, k, '{', '}');
      if (close < 0 || close >= to) {
        report(s.diags, s.context, s.src, 'TGT004_UNSUPPORTED_CONDITIONAL', 'An isolated block is not closed.', 'Balance the block braces.', t.pos);
        return;
      }
      scanStatements(s, k + 1, close, fn);
      i = close + 1;
      continue;
    }

    if (t.kind === 'ident') {
      switch (t.text) {
        case 'if':
          i = scanIf(s, k, to, fn);
          continue;
        case 'for':
          i = scanFor(s, k, to, fn);
          continue;
        case 'return':
        case 'break':
        case 'continue':
        case 'discard': {
          const semi = findSemicolon(s, k, to);
          i = semi < 0 ? to : semi + 1;
          continue;
        }
        case 'while':
        case 'do':
        case 'switch': {
          report(s.diags, s.context, s.src, 'TGT009_UNSUPPORTED_STATEMENT', `\`${t.text}\` has no WGSL lowering in the restricted subset.`, `Replace the \`${t.text}\` statement with \`if\`/\`else\` or the per-iteration mask form used by \`fbm2\`.`, t.pos);
          const semi = findSemicolon(s, k, to);
          i = semi < 0 ? to : semi + 1;
          continue;
        }
        case 'else': {
          report(s.diags, s.context, s.src, 'TGT004_UNSUPPORTED_CONDITIONAL', 'A stray `else` without a matching `if` was found.', 'Fix the if/else structure; WGSL requires `else` to follow a braced `if` body.', t.pos);
          i = nextMeaningful(s.toks, k);
          continue;
        }
        default:
          break;
      }
      if (t.text === VOID || TYPE_SET.has(t.text)) {
        i = scanTyped(s, k, to, fn);
        continue;
      }
    }

    i = scanExpressionStatement(s, k, to, fn);
  }
}

/** A statement that starts with a type name: a function definition, or a declaration. */
function scanTyped(s: State, typeIdx: number, to: number, fn: FnCtx | null): number {
  const typeTok = s.toks[typeIdx]!;
  const nameIdx = nextMeaningful(s.toks, typeIdx);
  const nameTok = nameIdx < to ? s.toks[nameIdx] : undefined;

  if (!nameTok) {
    report(s.diags, s.context, s.src, 'TGT002_UNSUPPORTED_DECLARATION', `\`${typeTok.text}\` at the end of the chunk declares nothing.`, 'Write a declaration (`float x = 0.0;`) or a function definition.', typeTok.pos);
    return to;
  }
  // `vec2f(1.0);` — a constructor used as a statement. Already valid WGSL; pass it through.
  if (nameTok.kind === 'punct' && nameTok.text === '(') {
    const semi = findSemicolon(s, typeIdx, to);
    return semi < 0 ? to : semi + 1;
  }
  if (nameTok.kind !== 'ident') {
    report(s.diags, s.context, s.src, 'TGT002_UNSUPPORTED_DECLARATION', `\`${typeTok.text} ${nameTok.text}\` is not a declaration this pass can lower.`, 'Declare a named local (`float x = 0.0;`) — arrays, structs and unnamed declarators are outside the subset.', nameTok.pos);
    const semi = findSemicolon(s, typeIdx, to);
    return semi < 0 ? to : semi + 1;
  }

  const after = nextMeaningful(s.toks, nameIdx);
  const afterTok = after < to ? s.toks[after] : undefined;
  if (afterTok?.kind === 'punct' && afterTok.text === '(') {
    return scanFunction(s, typeIdx, to, fn);
  }
  if (typeTok.text === VOID) {
    report(s.diags, s.context, s.src, 'TGT002_UNSUPPORTED_DECLARATION', '`void` can only introduce a function, and this is not one.', 'Remove the declaration, or write the function signature as `void name(…) { … }`.', typeTok.pos);
    const semi = findSemicolon(s, typeIdx, to);
    return semi < 0 ? to : semi + 1;
  }
  if (afterTok?.kind === 'punct' && (afterTok.text === '=' || afterTok.text === ';')) {
    if (fn && fn.params.has(nameTok.text)) {
      report(s.diags, s.context, s.src, 'TGT008_PARAM_SHADOWED', `\`${fn.name}\` declares a local named \`${nameTok.text}\`, which is also one of its parameters.`, 'Rename the local: the pass renames parameter reads inside the body and cannot tell the two apart.', nameTok.pos);
    }
    s.edits.push({
      start: typeTok.pos,
      end: tokenEnd(nameTok),
      text: `var ${nameTok.text}: ${typeTok.text}`,
      why: 'local declaration',
    });
    const semi = findSemicolon(s, after, to);
    return semi < 0 ? to : semi + 1;
  }

  report(s.diags, s.context, s.src, 'TGT002_UNSUPPORTED_DECLARATION', `\`${typeTok.text} ${nameTok.text}\` is not followed by \`=\`, \`;\` or a parameter list.`, 'The subset only declares plain named locals; anything else (arrays, qualifiers, multiple declarators) must be rewritten.', nameTok.pos);
  const semi = findSemicolon(s, typeIdx, to);
  return semi < 0 ? to : semi + 1;
}

/**
 * `float hash11(vec2 p, int n) { … }` →
 *
 *     fn hash11(p: vec2f, n: i32) -> f32 { … }
 *
 * plus, for every parameter the body writes to, a `var <name>_v: T = <name>;` copy at the
 * top of the body and a rename of that parameter inside the body (WGSL parameters are
 * `let`-like: they cannot be assigned, and neither can their components).
 */
function scanFunction(s: State, typeIdx: number, to: number, outer: FnCtx | null): number {
  const typeTok = s.toks[typeIdx]!;
  const nameIdx = nextMeaningful(s.toks, typeIdx);
  const nameTok = s.toks[nameIdx]!;
  if (outer) {
    report(s.diags, s.context, s.src, 'TGT001_UNSUPPORTED_SIGNATURE', `\`${nameTok.text}\` is a function defined inside \`${outer.name}\`.`, 'Move the nested function to module scope: WGSL has no closures or nested functions.', nameTok.pos);
  }
  const openIdx = nextMeaningful(s.toks, nameIdx);
  if (s.toks[openIdx]?.text !== '(') {
    report(s.diags, s.context, s.src, 'TGT001_UNSUPPORTED_SIGNATURE', `\`${typeTok.text} ${nameTok.text}\` has no parameter list.`, 'Write the signature as `float name(type arg, …)`.', nameTok.pos);
    return to;
  }
  const closeIdx = matchingClose(s.toks, openIdx, '(', ')');
  if (closeIdx < 0) {
    report(s.diags, s.context, s.src, 'TGT001_UNSUPPORTED_SIGNATURE', `The parameter list of \`${nameTok.text}\` is not closed.`, 'Balance the parentheses of the signature.', openIdx >= 0 ? s.toks[openIdx]!.pos : nameTok.pos);
    return to;
  }

  // --- parameters ---------------------------------------------------------
  const params = new Map<string, string>();
  const paramEdits: Edit[] = [];
  for (const [a, b] of splitTopLevel(s.toks, openIdx + 1, closeIdx)) {
    const items = meaningfulIndices(s.toks, a, b);
    if (items.length === 0) continue; // `()`: no parameters
    const pType = s.toks[items[0]!]!;
    const pName = items.length > 1 ? s.toks[items[1]!]! : undefined;
    if (items.length !== 2 || pType.kind !== 'ident' || !TYPE_SET.has(pType.text) || !pName || pName.kind !== 'ident') {
      const raw = s.src.slice(pType.pos, tokenEnd(s.toks[items[items.length - 1]!]!));
      report(s.diags, s.context, s.src, 'TGT001_UNSUPPORTED_SIGNATURE', `Parameter \`${raw.trim()}\` of \`${nameTok.text}\` is not \`type name\`.`, 'The subset only has plain `type name` parameters (no qualifiers, no arrays, no defaults).', pType.pos);
      continue;
    }
    if (params.has(pName.text)) {
      report(s.diags, s.context, s.src, 'TGT001_UNSUPPORTED_SIGNATURE', `\`${nameTok.text}\` declares the parameter \`${pName.text}\` twice.`, 'Give each parameter a distinct name.', pName.pos);
      continue;
    }
    params.set(pName.text, pType.text);
    paramEdits.push({ start: pType.pos, end: tokenEnd(pName), text: `${pName.text}: ${pType.text}`, why: 'parameter' });
  }

  // --- signature ----------------------------------------------------------
  s.edits.push({
    start: typeTok.pos,
    end: tokenEnd(nameTok),
    text: `fn ${nameTok.text}`,
    why: 'function signature',
  });
  s.edits.push(...paramEdits);
  if (typeTok.text !== VOID) {
    const afterParen = tokenEnd(s.toks[closeIdx]!);
    s.edits.push({ start: afterParen, end: afterParen, text: ` -> ${typeTok.text}`, why: 'return type' });
  }

  // --- body ---------------------------------------------------------------
  const bodyOpen = nextMeaningful(s.toks, closeIdx);
  if (s.toks[bodyOpen]?.text !== '{') {
    report(s.diags, s.context, s.src, 'TGT001_UNSUPPORTED_SIGNATURE', `\`${nameTok.text}\` has no braced body.`, 'Give the function a `{ … }` body; WGSL has no forward declarations or prototypes.', nameTok.pos);
    return closeIdx + 1;
  }
  const bodyClose = matchingClose(s.toks, bodyOpen, '{', '}');
  if (bodyClose < 0) {
    report(s.diags, s.context, s.src, 'TGT001_UNSUPPORTED_SIGNATURE', `The body of \`${nameTok.text}\` is not closed.`, 'Balance the braces of the function body.', s.toks[bodyOpen]!.pos);
    return to;
  }

  const fn: FnCtx = {
    name: nameTok.text,
    params,
    assigned: new Set(),
    bodyOpen,
    bodyClose,
  };
  scanStatements(s, bodyOpen + 1, bodyClose, fn);
  insertParameterCopies(s, fn);
  return bodyClose + 1;
}

/** The `var p_v: T = p;` block, plus the in-body rename, for every written parameter. */
function insertParameterCopies(s: State, fn: FnCtx): void {
  if (fn.assigned.size === 0) return;

  const used = new Set<string>(fn.params.keys());
  for (let i = fn.bodyOpen + 1; i < fn.bodyClose; i++) {
    const t = s.toks[i]!;
    if (t.kind === 'ident') used.add(t.text);
  }

  const indent = bodyIndent(s, fn.bodyOpen);
  const lines: string[] = [];
  for (const [name, type] of fn.params) {
    if (!fn.assigned.has(name)) continue;
    let copy = `${name}_v`;
    for (let n = 2; used.has(copy); n++) copy = `${name}_v${n}`;
    used.add(copy);
    lines.push(`var ${copy}: ${type} = ${name};`);
    renameParameterUses(s, fn, name, copy);
  }
  if (lines.length === 0) return;

  const at = tokenEnd(s.toks[fn.bodyOpen]!);
  s.edits.push({
    start: at,
    end: at,
    text: `\n${indent}${lines.join(`\n${indent}`)}`,
    why: 'immutable parameter copies',
  });
}

/** Rename every *use* of `name` inside `fn`'s body. Members (`x.name`) are never touched. */
function renameParameterUses(s: State, fn: FnCtx, name: string, copy: string): void {
  for (let i = fn.bodyOpen + 1; i < fn.bodyClose; i++) {
    const t = s.toks[i]!;
    if (t.kind !== 'ident' || t.text !== name) continue;
    if (isMemberAccess(s.toks, i)) continue; // `something.name`
    const prev = prevMeaningful(s.toks, i);
    const prevTok = prev >= 0 ? s.toks[prev]! : undefined;
    if (prevTok && prevTok.kind === 'ident' && TYPE_SET.has(prevTok.text) && !isMemberAccess(s.toks, prev)) {
      report(s.diags, s.context, s.src, 'TGT008_PARAM_SHADOWED', `\`${fn.name}\` declares a local named \`${name}\`, which is also a parameter.`, 'Rename the local: the pass cannot distinguish the two, so the parameter copy would shadow it.', t.pos);
      continue;
    }
    s.edits.push({ start: t.pos, end: tokenEnd(t), text: copy, why: 'parameter use' });
  }
}

/** Verified `if`/`else`: WGSL's syntax is already identical, so nothing is rewritten here. */
function scanIf(s: State, ifIdx: number, to: number, fn: FnCtx | null): number {
  const openIdx = nextMeaningful(s.toks, ifIdx);
  if (s.toks[openIdx]?.text !== '(') {
    report(s.diags, s.context, s.src, 'TGT004_UNSUPPORTED_CONDITIONAL', '`if` without a parenthesised condition.', 'Write `if (condition) { … }`; the subset always parenthesises the condition.', s.toks[ifIdx]!.pos);
    return skipStatement(s, ifIdx, to);
  }
  const closeIdx = matchingClose(s.toks, openIdx, '(', ')');
  if (closeIdx < 0) {
    report(s.diags, s.context, s.src, 'TGT004_UNSUPPORTED_CONDITIONAL', 'The `if` condition is not closed.', 'Balance the condition parentheses.', s.toks[openIdx]!.pos);
    return to;
  }
  const bodyOpen = nextMeaningful(s.toks, closeIdx);
  if (s.toks[bodyOpen]?.text !== '{') {
    report(s.diags, s.context, s.src, 'TGT004_UNSUPPORTED_CONDITIONAL', 'An `if`/`else` body is not a `{ … }` block.', 'WGSL requires a braced block: write `if (c) { … } else { … }`.', bodyOpen < s.toks.length ? s.toks[bodyOpen]!.pos : s.toks[closeIdx]!.pos);
    return skipStatement(s, closeIdx, to);
  }
  const bodyClose = matchingClose(s.toks, bodyOpen, '{', '}');
  if (bodyClose < 0) {
    report(s.diags, s.context, s.src, 'TGT004_UNSUPPORTED_CONDITIONAL', 'An `if` body is not closed.', 'Balance the braces of the `if` body.', s.toks[bodyOpen]!.pos);
    return to;
  }
  scanStatements(s, bodyOpen + 1, bodyClose, fn);

  const elseIdx = nextMeaningful(s.toks, bodyClose);
  if (s.toks[elseIdx]?.kind === 'ident' && s.toks[elseIdx]!.text === 'else') {
    const thenIdx = nextMeaningful(s.toks, elseIdx);
    const thenTok = s.toks[thenIdx];
    if (thenTok?.kind === 'ident' && thenTok.text === 'if') return scanIf(s, thenIdx, to, fn);
    if (thenTok?.text === '{') {
      const close2 = matchingClose(s.toks, thenIdx, '{', '}');
      if (close2 < 0) {
        report(s.diags, s.context, s.src, 'TGT004_UNSUPPORTED_CONDITIONAL', 'An `else` body is not closed.', 'Balance the braces of the `else` body.', thenTok.pos);
        return to;
      }
      scanStatements(s, thenIdx + 1, close2, fn);
      return close2 + 1;
    }
    report(s.diags, s.context, s.src, 'TGT004_UNSUPPORTED_CONDITIONAL', '`else` is not followed by `if` or a `{ … }` block.', 'WGSL requires a braced block after `else`.', thenTok ? thenTok.pos : s.toks[elseIdx]!.pos);
    return thenIdx;
  }
  return bodyClose + 1;
}

/**
 * `for (int i = 0; i < 8; i++)` →
 *
 *     for (var i: i32 = 0; i < 8; i = i + 1)
 *
 * The bound must be an integer literal for an `i32` loop (the subset says loop bounds are
 * literals): an identifier bound would need that identifier's type to pick `i32` vs `f32`,
 * which this pass deliberately does not infer. `f32` loop variables are supported too and
 * keep float literals.
 */
function scanFor(s: State, forIdx: number, to: number, fn: FnCtx | null): number {
  const openIdx = nextMeaningful(s.toks, forIdx);
  if (s.toks[openIdx]?.text !== '(') {
    report(s.diags, s.context, s.src, 'TGT003_UNSUPPORTED_LOOP', '`for` without a parenthesised header.', 'Write `for (int i = 0; i < N; i++) { … }`.', s.toks[forIdx]!.pos);
    return skipStatement(s, forIdx, to);
  }
  const closeIdx = matchingClose(s.toks, openIdx, '(', ')');
  if (closeIdx < 0) {
    report(s.diags, s.context, s.src, 'TGT003_UNSUPPORTED_LOOP', 'The `for` header is not closed.', 'Balance the header parentheses.', s.toks[openIdx]!.pos);
    return to;
  }

  const header = parseForHeader(s, openIdx, closeIdx);
  if (header) {
    s.edits.push({
      start: s.toks[openIdx]!.pos,
      end: tokenEnd(s.toks[closeIdx]!),
      text: `(var ${header.name}: ${header.type} = ${header.init}; ${header.name} ${header.cmp} ${header.bound}; ${header.name} = ${header.name} + 1)`,
      why: 'loop header',
    });
  }

  const bodyOpen = nextMeaningful(s.toks, closeIdx);
  if (s.toks[bodyOpen]?.text !== '{') {
    report(s.diags, s.context, s.src, 'TGT003_UNSUPPORTED_LOOP', 'The `for` body is not a `{ … }` block.', 'WGSL requires a braced loop body.', bodyOpen < s.toks.length ? s.toks[bodyOpen]!.pos : s.toks[closeIdx]!.pos);
    return skipStatement(s, closeIdx, to);
  }
  const bodyClose = matchingClose(s.toks, bodyOpen, '{', '}');
  if (bodyClose < 0) {
    report(s.diags, s.context, s.src, 'TGT003_UNSUPPORTED_LOOP', 'The `for` body is not closed.', 'Balance the braces of the loop body.', s.toks[bodyOpen]!.pos);
    return to;
  }
  scanStatements(s, bodyOpen + 1, bodyClose, fn);
  return bodyClose + 1;
}

type ForHeader = { name: string; type: string; init: string; cmp: string; bound: string };

/**
 * Parse `TYPE NAME = [-]<lit> ; NAME (<|<=) [-]<lit> ; NAME ++` from the header tokens.
 *
 * A cursor rather than fixed indices, because a leading `-` shifts everything (the
 * Voronoi neighbourhood scan really is `for (int j = -1; j <= 1; j++)`).
 */
function parseForHeader(s: State, openIdx: number, closeIdx: number): ForHeader | null {
  const headerText = s.src.slice(s.toks[openIdx]!.pos, tokenEnd(s.toks[closeIdx]!));
  const fail = (why: string): null => {
    report(s.diags, s.context, s.src, 'TGT003_UNSUPPORTED_LOOP', `Unsupported \`for\` header: \`${headerText}\` (${why}).`, 'Only `for (int i = 0; i < N; i++)` with a literal N is in the subset; N must be a literal so the loop variable can be typed `i32`.', s.toks[openIdx]!.pos);
    return null;
  };

  const items = meaningfulIndices(s.toks, openIdx + 1, closeIdx).map((i) => s.toks[i]!);
  let p = 0;
  const take = (): Token | undefined => items[p++];
  const isPunct = (t: Token | undefined, text: string): boolean => !!t && t.kind === 'punct' && t.text === text;

  const typeTok = take();
  const nameTok = take();
  /** `[ - ] <number>` with the sign folded into the returned text. */
  const readLiteral = (what: string): { text: string } | string => {
    const first = take();
    if (!first) return `the ${what} is missing`;
    const negated = isPunct(first, '-');
    const num = negated ? take() : first;
    if (!num || num.kind !== 'number') return `the ${what} is not a numeric literal`;
    return { text: `${negated ? '-' : ''}${num.text}` };
  };

  if (!typeTok || typeTok.kind !== 'ident' || (typeTok.text !== 'i32' && typeTok.text !== 'f32')) {
    return fail('the loop variable is not declared `int` or `float`');
  }
  if (!nameTok || nameTok.kind !== 'ident') return fail('the loop variable has no name');
  if (!isPunct(take(), '=')) return fail('the initialiser is missing');
  const init = readLiteral('initialiser');
  if (typeof init === 'string') return fail(init);
  if (!isPunct(take(), ';')) return fail('the initialiser clause is not terminated by `;`');
  const condName = take();
  const cmp = take();
  if (!condName || condName.kind !== 'ident' || condName.text !== nameTok.text) {
    return fail('the condition does not test the loop variable');
  }
  if (!isPunct(cmp, '<') && !isPunct(cmp, '<=')) return fail('the condition is not `<` or `<=`');
  const bound = readLiteral('bound');
  if (typeof bound === 'string') return fail(bound);
  if (!isPunct(take(), ';')) return fail('the condition clause is not terminated by `;`');
  const updateName = take();
  if (!updateName || updateName.kind !== 'ident' || updateName.text !== nameTok.text) {
    return fail('the update does not write the loop variable');
  }
  if (!isPunct(take(), '++')) return fail('the update is not `i++`');
  if (p !== items.length) return fail('there are unexpected tokens in the header');

  const isInt = typeTok.text === 'i32';
  const initText = isInt ? integerLiteralText(init.text) : floatLiteralText(init.text);
  const boundText = isInt ? integerLiteralText(bound.text) : floatLiteralText(bound.text);
  if (initText === null || boundText === null) {
    report(s.diags, s.context, s.src, 'TGT011_NON_INTEGRAL_LOOP_LITERAL', `The \`for\` header \`${headerText}\` uses a non-integral literal for an \`i32\` loop variable.`, 'Use an integer literal (the subset writes float literals with a decimal point and integers without).', s.toks[openIdx]!.pos);
    return null;
  }
  return { name: nameTok.text, type: typeTok.text, init: initText, cmp: cmp!.text, bound: boundText };
}

/** Assignments: the only place a parameter can be mutated, and the immutable-name check. */
function scanExpressionStatement(s: State, k: number, to: number, fn: FnCtx | null): number {
  const t = s.toks[k]!;
  if (t.kind === 'punct' && (t.text === '++' || t.text === '--')) {
    report(s.diags, s.context, s.src, 'TGT012_UNSUPPORTED_EXPRESSION', `Prefix \`${t.text}\` is not WGSL.`, 'Write the mutation as a statement (`i = i + 1;`).', t.pos);
  }
  if (t.kind === 'ident') {
    // Walk the lvalue chain (`p`, `p.x`, `p.xy`) to the assignment operator.
    let j = nextMeaningful(s.toks, k);
    while (j < to && s.toks[j]!.kind === 'punct' && s.toks[j]!.text === '.') {
      const member = nextMeaningful(s.toks, j);
      if (member >= to || s.toks[member]!.kind !== 'ident') break;
      j = nextMeaningful(s.toks, member);
    }
    const op = j < to ? s.toks[j]! : undefined;
    const isAssign = !!op && op.kind === 'punct' && ASSIGN_OPS.has(op.text);
    const isIncDec = !!op && op.kind === 'punct' && (op.text === '++' || op.text === '--');
    if (isAssign || isIncDec) noteAssignment(s, fn, t);
  }

  const semi = findSemicolon(s, k, to);
  if (semi < 0) {
    report(s.diags, s.context, s.src, 'TGT012_UNSUPPORTED_EXPRESSION', 'A statement is not terminated by `;`.', 'Terminate every statement with `;`.', t.pos);
    return to;
  }
  return semi + 1;
}

function noteAssignment(s: State, fn: FnCtx | null, root: Token): void {
  if (fn && fn.params.has(root.text)) {
    fn.assigned.add(root.text);
    return;
  }
  if (s.immutable.has(root.text)) {
    report(s.diags, s.context, s.src, 'TGT007_IMMUTABLE_ASSIGNMENT', `\`${root.text}\` is assigned, but the WGSL epilogue declares it immutably (\`let\`/uniform member).`, 'Node bodies must not write to `uv`, a host symbol or a uniform; compute into a local instead.', root.pos);
  }
}

/* -------------------------------------------------------------------------- */
/* Edits                                                                       */
/* -------------------------------------------------------------------------- */

function applyEdits(s: State): string {
  return applyEditList(s.src, s.edits, s.diags, s.context);
}

function applyEditList(src: string, edits: Edit[], diags: Diagnostic[], context: string): string {
  if (edits.length === 0) return src;
  const sorted = [...edits].sort((a, b) => a.start - b.start || a.end - b.end);
  let out = src;
  let rightBoundary = Number.POSITIVE_INFINITY;
  for (let i = sorted.length - 1; i >= 0; i--) {
    const e = sorted[i]!;
    if (e.end > rightBoundary) {
      report(diags, context, src, 'TGT080_OVERLAPPING_EDITS', `Two lowering edits overlap at offset ${e.start} (${e.why}).`, 'This is a bug in the WGSL lowering pass: report the input that produced it. The overlapping edit was skipped, so the output is incomplete.');
      continue;
    }
    out = `${out.slice(0, e.start)}${e.text}${out.slice(e.end)}`;
    rightBoundary = e.start;
  }
  return out;
}

/* -------------------------------------------------------------------------- */
/* Structural checker                                                          */
/* -------------------------------------------------------------------------- */

/**
 * Balance the brackets of a WGSL chunk and report the **first** imbalance with the line
 * number it occurs on. Comments are blanked (not removed) so line numbers stay exact.
 *
 * This is the substitute for the compiler this repository does not have: it cannot prove
 * a module compiles, but it does prove the lowering pass did not tear a block in half —
 * which is the failure mode a text-splicing transform actually has.
 */
export function structuralBalanceProblems(src: string): string[] {
  const code = blankComments(src);
  // One pass for the line starts: the naive `lineAt()` per bracket is quadratic, and the
  // full emission is tens of kilobytes.
  const lineStart: number[] = [0];
  for (let i = 0; i < code.length; i++) if (code[i] === '\n') lineStart.push(i + 1);
  const lineAt = (i: number): number => {
    let lo = 0;
    let hi = lineStart.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (lineStart[mid]! <= i) lo = mid;
      else hi = mid - 1;
    }
    return lo + 1;
  };

  const problems: string[] = [];
  const stack: { ch: string; line: number }[] = [];
  for (let i = 0; i < code.length; i++) {
    const c = code[i]!;
    if (c === '{' || c === '(' || c === '[') {
      stack.push({ ch: c, line: lineAt(i) });
      continue;
    }
    if (c !== '}' && c !== ')' && c !== ']') continue;
    const want = c === '}' ? '{' : c === ')' ? '(' : '[';
    const top = stack.pop();
    if (!top) {
      problems.push(`unmatched \`${c}\` on line ${lineAt(i)}`);
      break;
    }
    if (top.ch !== want) {
      problems.push(`\`${top.ch}\` opened on line ${top.line} is closed by \`${c}\` on line ${lineAt(i)}`);
      break;
    }
  }
  if (problems.length === 0 && stack.length) {
    const top = stack[stack.length - 1]!;
    problems.push(`\`${top.ch}\` opened on line ${top.line} is never closed`);
  }
  return problems;
}

/* -------------------------------------------------------------------------- */
/* Token helpers                                                               */
/* -------------------------------------------------------------------------- */

function isTrivia(t: Token | undefined): boolean {
  return t !== undefined && (t.kind === 'space' || t.kind === 'comment');
}

function tokenEnd(t: Token): number {
  return t.pos + t.text.length;
}

function nextMeaningful(toks: Token[], i: number): number {
  let j = i + 1;
  while (j < toks.length && isTrivia(toks[j])) j++;
  return j;
}

function prevMeaningful(toks: Token[], i: number): number {
  let j = i - 1;
  while (j >= 0 && isTrivia(toks[j])) j--;
  return j;
}

function isMemberAccess(toks: Token[], i: number): boolean {
  const p = prevMeaningful(toks, i);
  return p >= 0 && toks[p]!.kind === 'punct' && toks[p]!.text === '.';
}

/** Token index of the bracket matching the one at `openIdx`, or `-1`. */
function matchingClose(toks: Token[], openIdx: number, open: string, close: string): number {
  let depth = 0;
  for (let i = openIdx; i < toks.length; i++) {
    const t = toks[i]!;
    if (t.kind !== 'punct') continue;
    if (t.text === open) depth++;
    else if (t.text === close) {
      depth--;
      if (depth === 0) return i;
    }
  }
  return -1;
}

/** Meaningful token indices in `[from, to)`. */
function meaningfulIndices(toks: Token[], from: number, to: number): number[] {
  const out: number[] = [];
  for (let i = from; i < to; i++) if (!isTrivia(toks[i])) out.push(i);
  return out;
}

/** The source text of `[from, to)`, trimmed to its first and last meaningful token. */
function sliceRange(toks: Token[], src: string, from: number, to: number): string {
  const idx = meaningfulIndices(toks, from, to);
  if (idx.length === 0) return '';
  return src.slice(toks[idx[0]!]!.pos, tokenEnd(toks[idx[idx.length - 1]!]!));
}

/** Ranges `[startIdx, endIdx)` of the top-level (depth-0) comma-separated items. */
function splitTopLevel(toks: Token[], from: number, to: number): [number, number][] {
  const out: [number, number][] = [];
  let depth = 0;
  let start = from;
  for (let i = from; i < to; i++) {
    const t = toks[i]!;
    if (t.kind !== 'punct') continue;
    if (t.text === '(' || t.text === '[') depth++;
    else if (t.text === ')' || t.text === ']') depth--;
    else if (t.text === ',' && depth === 0) {
      out.push([start, i]);
      start = i + 1;
    }
  }
  if (start < to || out.length === 0) out.push([start, to]);
  if (out.length === 1 && meaningfulIndices(toks, out[0]![0], out[0]![1]).length === 0) return [];
  return out;
}

/** Index of the `;` that ends the statement starting at `from`, or `-1`. */
function findSemicolon(s: State, from: number, to: number): number {
  let depth = 0;
  for (let i = from; i < to; i++) {
    const t = s.toks[i]!;
    if (t.kind !== 'punct') continue;
    if (t.text === '(' || t.text === '[') depth++;
    else if (t.text === ')' || t.text === ']') depth--;
    else if (t.text === ';' && depth === 0) return i;
  }
  return -1;
}

/** Best effort recovery: consume one statement without lowering it. */
function skipStatement(s: State, from: number, to: number): number {
  const brace = nextMeaningful(s.toks, from);
  if (s.toks[brace]?.text === '{') {
    const close = matchingClose(s.toks, brace, '{', '}');
    if (close >= 0) return close + 1;
  }
  const semi = findSemicolon(s, from, to);
  return semi < 0 ? to : semi + 1;
}

/** The indentation to use for statements inserted at the top of a braced body. */
function bodyIndent(s: State, openIdx: number): string {
  const openPos = s.toks[openIdx]!.pos;
  const following = s.toks[openIdx + 1];
  if (following && following.kind === 'space' && following.text.includes('\n')) {
    const tail = following.text.slice(following.text.lastIndexOf('\n') + 1);
    if (tail.trim().length === 0 && tail.length > 0) return tail;
  }
  const lineStart = s.src.lastIndexOf('\n', openPos) + 1;
  const linePrefix = s.src.slice(lineStart, openPos);
  const leading = /^[ \t]*/.exec(linePrefix);
  return `${leading ? leading[0] : ''}  `;
}

/* -------------------------------------------------------------------------- */
/* Literals                                                                    */
/* -------------------------------------------------------------------------- */

/** A decimal literal (optionally signed, optionally exponent, optionally `f`-suffixed). */
const NUMBER_RE = /^-?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?$/;

/** The integer spelling of a literal, or `null` when it is not an integral number. */
function integerLiteralText(raw: string): string | null {
  const text = raw.replace(/[fFuU]$/, '');
  if (!NUMBER_RE.test(text)) return null;
  const value = Number(text);
  if (!Number.isFinite(value) || !Number.isInteger(value)) return null;
  const out = String(Math.trunc(value));
  // `1e21` is a valid float literal and not a valid integer one.
  return /[eE.+]/.test(out) ? null : out;
}

/** The float spelling of a literal: always with a decimal point, never with a suffix. */
function floatLiteralText(raw: string): string | null {
  const text = raw.replace(/[fFuU]$/, '');
  if (!NUMBER_RE.test(text)) return null;
  if (/[.eE]/.test(text)) return text;
  return `${text}.0`;
}

/* -------------------------------------------------------------------------- */
/* Diagnostics / text helpers                                                  */
/* -------------------------------------------------------------------------- */

function lineOf(src: string, pos: number): number {
  let line = 1;
  for (let i = 0; i < pos && i < src.length; i++) if (src[i] === '\n') line++;
  return line;
}

function report(
  diags: Diagnostic[],
  context: string,
  src: string,
  code: string,
  message: string,
  fix: string,
  pos?: number,
): void {
  diags.push({
    severity: 'error',
    code,
    message: `${message} (${context})`,
    ...(pos !== undefined ? { at: { line: lineOf(src, pos) } } : {}),
    fix,
  });
}

/** Comments → spaces, newlines kept, so offsets and line numbers are unchanged. */
function blankComments(src: string): string {
  let out = '';
  for (const t of tokenize(src)) {
    out += t.kind === 'comment' ? t.text.replace(/[^\n]/g, ' ') : t.text;
  }
  return out;
}
