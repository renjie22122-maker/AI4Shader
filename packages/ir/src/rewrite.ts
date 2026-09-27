/**
 * AI4Shader — Multi-target source rewriter
 * ========================================
 *
 * Takes a body of restricted GLSL (see `glslSubset.ts`) and produces the equivalent
 * source in GLSL / WGSL / HLSL / MSL / UE-HLSL.
 *
 * Why a token rewriter and not a real parser: writing five parallel AST emitters is
 * five places for the noise function to drift, and drift between targets is exactly
 * the bug class this project exists to eliminate. Instead there is ONE authored
 * source, and this module is the only place a language difference may be expressed.
 * The restricted subset is the contract that makes the transform sound.
 *
 * The rewriter is *total*: any token it does not recognise passes through unchanged,
 * and any token it knows it cannot lower correctly raises a `RW` diagnostic rather
 * than emitting plausible-looking wrong code.
 */

import { type Diagnostic, type TargetFormat } from './types.js';
import { BANNED_BUILTIN_SPELLINGS, BANNED_TOKENS, TYPE_TOKENS } from './glslSubset.js';

/* -------------------------------------------------------------------------- */
/* Tokenizer                                                                   */
/* -------------------------------------------------------------------------- */

export type TokenKind = 'ident' | 'number' | 'punct' | 'space' | 'comment';

export type Token = {
  kind: TokenKind;
  text: string;
  /** Offset in the original source — used to report diagnostics with line numbers. */
  pos: number;
};

const IDENT_START = /[A-Za-z_]/;
const IDENT_BODY = /[A-Za-z0-9_]/;
const DIGIT = /[0-9]/;

/** Longest-first so `<<=`/`<=`/`<<` lex correctly (bitwise is banned but must still lex). */
const PUNCT = [
  '<<=', '>>=',
  '==', '!=', '<=', '>=', '&&', '||', '++', '--', '+=', '-=', '*=', '/=', '%=',
  '<<', '>>',
  '+', '-', '*', '/', '%', '=', '<', '>', '!', '?', ':', '(', ')', '{', '}',
  '[', ']', ',', '.', ';', '&', '|', '^', '~', '#',
];

export function tokenize(src: string): Token[] {
  const out: Token[] = [];
  let i = 0;
  while (i < src.length) {
    const c = src[i]!;
    if (c === ' ' || c === '\t' || c === '\n' || c === '\r') {
      let j = i;
      while (j < src.length && /\s/.test(src[j]!)) j++;
      out.push({ kind: 'space', text: src.slice(i, j), pos: i });
      i = j;
      continue;
    }
    if (c === '/' && src[i + 1] === '/') {
      let j = i;
      while (j < src.length && src[j] !== '\n') j++;
      out.push({ kind: 'comment', text: src.slice(i, j), pos: i });
      i = j;
      continue;
    }
    if (c === '/' && src[i + 1] === '*') {
      let j = src.indexOf('*/', i + 2);
      j = j === -1 ? src.length : j + 2;
      out.push({ kind: 'comment', text: src.slice(i, j), pos: i });
      i = j;
      continue;
    }
    if (IDENT_START.test(c)) {
      let j = i;
      while (j < src.length && IDENT_BODY.test(src[j]!)) j++;
      out.push({ kind: 'ident', text: src.slice(i, j), pos: i });
      i = j;
      continue;
    }
    if (DIGIT.test(c) || (c === '.' && DIGIT.test(src[i + 1] ?? ''))) {
      let j = i;
      // Hex form (banned later, but must lex as one token so errors read well).
      if (c === '0' && (src[i + 1] === 'x' || src[i + 1] === 'X')) {
        j = i + 2;
        while (j < src.length && /[0-9a-fA-F]/.test(src[j]!)) j++;
      } else {
        while (j < src.length && /[0-9.eE]/.test(src[j]!)) {
          // Stop at a `.` that is a swizzle/member access, e.g. `1.0.x` or `arr.length`.
          if (src[j] === '.' && !DIGIT.test(src[j + 1] ?? '') && !/[eE]/.test(src[j - 1] ?? '')) break;
          if ((src[j] === 'e' || src[j] === 'E') && !/[0-9+-]/.test(src[j + 1] ?? '')) break;
          j++;
        }
        while (j < src.length && /[uUfF]/.test(src[j]!)) j++;
      }
      out.push({ kind: 'number', text: src.slice(i, j), pos: i });
      i = j;
      continue;
    }
    const p = PUNCT.find((x) => src.startsWith(x, i));
    if (p) {
      out.push({ kind: 'punct', text: p, pos: i });
      i += p.length;
      continue;
    }
    // Unknown character: keep it, let the target compiler complain.
    out.push({ kind: 'punct', text: c, pos: i });
    i += 1;
  }
  return out;
}

/* -------------------------------------------------------------------------- */
/* Identifier tables                                                           */
/* -------------------------------------------------------------------------- */

type Tables = {
  /** Renamed types. */
  types: Record<string, string>;
  /** Renamed plain functions (identical call shape, different name). */
  fns: Record<string, string>;
  /** Member access renames (rare). */
  members: Record<string, string>;
};

const GLSL_TABLES: Tables = {
  types: {},
  fns: {
    fract_: 'fract', mod_: 'mod', mix_: 'mix', atan2_: 'atan', inversesqrt_: 'inversesqrt',
    dFdx_: 'dFdx', dFdy_: 'dFdy', tex2D_: 'texture',
  },
  members: {},
};

const HLSL_TABLES: Tables = {
  types: { vec2: 'float2', vec3: 'float3', vec4: 'float4', mat2: 'float2x2', mat3: 'float3x3' },
  fns: {
    fract_: 'frac', mod_: 'wf_mod', mix_: 'wf_mix', atan2_: 'wf_atan2', inversesqrt_: 'rsqrt',
    dFdx_: 'ddx', dFdy_: 'ddy', tex2D_: 'tex2D',
  },
  members: {},
};

const MSL_TABLES: Tables = {
  types: { vec2: 'float2', vec3: 'float3', vec4: 'float4', mat2: 'float2x2', mat3: 'float3x3' },
  fns: {
    fract_: 'fract', mod_: 'wf_mod', mix_: 'mix', atan2_: 'wf_atan2', inversesqrt_: 'rsqrt',
    dFdx_: 'dfdx', dFdy_: 'dfdy', tex2D_: 'sample',
  },
  members: {},
};

const WGSL_TABLES: Tables = {
  types: {
    vec2: 'vec2f', vec3: 'vec3f', vec4: 'vec4f',
    mat2: 'mat2x2f', mat3: 'mat3x3f',
    float: 'f32', int: 'i32', bool: 'bool',
  },
  fns: {
    fract_: 'fract', mod_: 'wf_mod', mix_: 'wf_mix', atan2_: 'wf_atan2',
    inversesqrt_: 'inverseSqrt', dFdx_: 'dpdx', dFdy_: 'dpdy', tex2D_: 'textureSample',
  },
  members: {},
};

function tablesFor(target: TargetFormat): Tables {
  switch (target) {
    case 'glsl':
      return GLSL_TABLES;
    case 'hlsl':
    case 'ue':
    case 'shaderlab':
      return HLSL_TABLES;
    case 'msl':
      return MSL_TABLES;
    case 'wgsl':
      return WGSL_TABLES;
  }
}

/**
 * WGSL prelude shim.
 *
 * Only `mod` genuinely needs one. The earlier comment here also claimed WGSL lacks a
 * scalar `mix`, which is **wrong**: the spec's table has `mix(e1: T, e2: T, e3: T)` with
 * `T` scalar-or-vector, plus `mix(vecN<T>, vecN<T>, T)`. That mistake made a vector-arity
 * problem look like it needed a shim at all. `wf_mod` is real — WGSL has no `mod`
 * builtin, and `%` is integer-only.
 *
 * `wf_mix` and `wf_atan2` are kept because the other targets' shim blocks define the same
 * three names, and a per-target difference in *which* shims exist would be a second place
 * for the languages to diverge.
 */
export const WGSL_SHIMS = /* wgsl */ `
fn wf_mix(a: f32, b: f32, t: f32) -> f32 { return a + (b - a) * t; }
fn wf_mod(a: f32, b: f32) -> f32 { return a - b * floor(a / b); }
fn wf_atan2(y: f32, x: f32) -> f32 { return atan2(vec2f(y, x)); }
`;

/**
 * Scalar shims for HLSL / UE-HLSL / MSL.
 *
 * `fmod` — the obvious mapping for GLSL's `mod` — **truncates toward zero** while GLSL's
 * `mod` **floors**. The two agree only for a positive divisor and a non-negative
 * dividend, so `mod_(-0.3, 1.0)` is `0.7` in GLSL and `-0.3` in HLSL. `hsv2rgb` feeds
 * `mod_` a hue that can legitimately be negative, so this is not a theoretical
 * difference: it is a hue discontinuity that only appears in two of the six targets,
 * which is the worst possible shape for a rendering bug. The floor form is used
 * everywhere instead.
 *
 * `atan2` genuinely has no vector form in either language, so `wf_atan2` is scalar-only by
 * necessity — and its body must be the **scalar** `atan2(y, x)`. It used to read
 * `atan2(float2(y, x))`, which is a compile error in HLSL: `atan2` there is scalar-only, so
 * the vector argument has no overload and dxc reports `use of undeclared identifier 'atan2'`.
 * WGSL is the odd one out — its `atan2` takes a `vec2` — which is why the WGSL shim keeps the
 * vector form and this one does not. The two languages' `atan2` have the same argument order
 * and range, so this is a plain rename rather than a wrapper.
 *
 * ## The overload family has to be *complete*, and completing it is not optional
 *
 * GLSL's `mix(e1, e2, e3)` is declared for `e3` scalar **or vector** — the vector form mixes
 * per component. This shim originally had only the scalar-`t` overloads, so a vector `t`
 * silently selected `wf_mix(float3, float3, float)` and **truncated to `t.x`**: the red
 * channel's decision applied to green and blue. HLSL's own `lerp` accepts either, so the shim
 * only had to say so.
 *
 * It was not a latent hole. `srgb2linear` in the shared library reads
 *
 *     mix_(lo, hi, step(vec3(0.04045), c))
 *
 * which in GLSL converts *each channel* by its own threshold and in HLSL converted all three
 * by the red one — a colour shift on exactly two of the six targets, with no compile error to
 * point at it. `dxc -Wconversion` did warn (`implicit truncation of vector type`), and the
 * warning was in the emitted library of *every* document, which is how it was finally caught.
 *
 * `wf_mod` gets the vector-divisor forms for the same reason: GLSL declares
 * `mod(genType, genType)` as well as `mod(genType, float)`, so a call with a vector divisor
 * would have truncated in exactly the same way.
 */
export const HLSLFAM_SHIMS = /* hlsl */ `
float  wf_mod(float a, float b) { return a - b * floor(a / b); }
float2 wf_mod(float2 a, float b) { return a - b * floor(a / b); }
float3 wf_mod(float3 a, float b) { return a - b * floor(a / b); }
float4 wf_mod(float4 a, float b) { return a - b * floor(a / b); }
float2 wf_mod(float2 a, float2 b) { return a - b * floor(a / b); }
float3 wf_mod(float3 a, float3 b) { return a - b * floor(a / b); }
float4 wf_mod(float4 a, float4 b) { return a - b * floor(a / b); }

float  wf_mix(float a, float b, float t) { return a + (b - a) * t; }
float2 wf_mix(float2 a, float2 b, float t) { return a + (b - a) * t; }
float3 wf_mix(float3 a, float3 b, float t) { return a + (b - a) * t; }
float4 wf_mix(float4 a, float4 b, float t) { return a + (b - a) * t; }
float2 wf_mix(float2 a, float2 b, float2 t) { return a + (b - a) * t; }
float3 wf_mix(float3 a, float3 b, float3 t) { return a + (b - a) * t; }
float4 wf_mix(float4 a, float4 b, float4 t) { return a + (b - a) * t; }

float wf_atan2(float y, float x) { return atan2(y, x); }
`;

/**
 * MSL needs the same overload family for `wf_mod`, for the same reason: a scalar-only shim
 * turns every vector call into an implicit truncation, and Metal diagnoses that too. MSL does
 * have a scalar `mix`, so it needs no `wf_mix`.
 *
 * The vector-divisor `wf_mod` overloads are here for the same completeness argument as HLSL's:
 * GLSL declares `mod(genType, genType)`, so the scalar-divisor-only family would truncate a
 * vector divisor on this target and nowhere else.
 */
export const MSL_SHIMS = /* metal */ `
float  wf_mod(float a, float b) { return a - b * floor(a / b); }
float2 wf_mod(float2 a, float b) { return a - b * floor(a / b); }
float3 wf_mod(float3 a, float b) { return a - b * floor(a / b); }
float4 wf_mod(float4 a, float b) { return a - b * floor(a / b); }
float2 wf_mod(float2 a, float2 b) { return a - b * floor(a / b); }
float3 wf_mod(float3 a, float3 b) { return a - b * floor(a / b); }
float4 wf_mod(float4 a, float4 b) { return a - b * floor(a / b); }

float wf_atan2(float y, float x) { return atan2(y, x); }
`;

/** HLSL-family preamble, emitted at module scope before the helper library. */
export function hlslShimsFor(target: TargetFormat): string {
  return target === 'hlsl' || target === 'ue' || target === 'shaderlab' ? HLSLFAM_SHIMS : '';
}

/** GLSL needs none of this: every shimmed operation already exists with GLSL semantics. */
export const EMPTY_SHIMS = '';

export function shimsFor(target: TargetFormat): string {
  switch (target) {
    case 'wgsl':
      return WGSL_SHIMS;
    case 'hlsl':
    case 'ue':
    case 'shaderlab':
      return HLSLFAM_SHIMS;
    case 'msl':
      return MSL_SHIMS;
    case 'glsl':
      return EMPTY_SHIMS;
  }
}

/* -------------------------------------------------------------------------- */
/* Rewriter                                                                    */
/* -------------------------------------------------------------------------- */

export type RewriteResult = {
  source: string;
  diagnostics: Diagnostic[];
};

/** Functions whose argument order differs per target and therefore need a wrapper. */
const ARG_ORDER_WRAPPED = new Set(['atan2_']);

function lineOf(src: string, pos: number): number {
  let line = 1;
  for (let i = 0; i < pos && i < src.length; i++) if (src[i] === '\n') line++;
  return line;
}

/**
 * Rewrite one body of restricted GLSL into `target`.
 *
 * @param src      restricted GLSL (node body, library function, or whole program)
 * @param target   destination language
 * @param context  human label used in diagnostics, e.g. `node n_fbm_1 (fbm)`
 */
export function rewriteSource(src: string, target: TargetFormat, context = 'source'): RewriteResult {
  const diags: Diagnostic[] = [];
  const toks = tokenize(src);
  const t = tablesFor(target);
  const out: string[] = [];

  // Track whether we are inside a comment so we never rewrite prose.
  let parenDepth = 0;

  for (let i = 0; i < toks.length; i++) {
    const tok = toks[i]!;

    if (tok.kind === 'comment' || tok.kind === 'space') {
      out.push(tok.text);
      continue;
    }
    if (tok.kind === 'number') {
      out.push(rewriteNumber(tok.text, target, diags, src, tok.pos, context));
      continue;
    }
    if (tok.kind === 'punct') {
      if (tok.text === '(') parenDepth++;
      if (tok.text === ')') parenDepth = Math.max(0, parenDepth - 1);
      out.push(tok.text);
      continue;
    }

    // ---- identifier ----
    const name = tok.text;

    if ((BANNED_TOKENS as readonly string[]).includes(name)) {
      // `texture`, `in`, `out` etc. are legal *members* and *locals* in some targets,
      // so only reject the ones that are never valid in the restricted subset.
      const hardBanned = new Set([
        'discard', 'switch', 'while', 'struct', 'sampler2D', 'gl_FragColor',
        'gl_FragCoord', 'precision', 'layout', 'uniform', 'varying', 'attribute',
        'uint',
      ]);
      if (hardBanned.has(name)) {
        diags.push({
          severity: 'error',
          code: 'RW001_BANNED_TOKEN',
          message: `\`${name}\` is not allowed in the restricted GLSL subset (${context}).`,
          at: { line: lineOf(src, tok.pos) },
          fix: 'Rewrite using the portable subset; see glslSubset.ts for the allowed surface.',
        });
      }
    }

    // Type rename.
    if (t.types[name] && !isMemberAccess(toks, i)) {
      out.push(t.types[name]!);
      continue;
    }

    // Helper / alias rename.
    if (t.fns[name]) {
      const mapped = t.fns[name]!;
      if (ARG_ORDER_WRAPPED.has(name) && target === 'wgsl') {
        // Already handled by the `wf_atan2` shim, which takes (y, x) — same as GLSL.
      }
      out.push(mapped);
      continue;
    }

    // Matrix helpers become direct multiplications per target.
    if (name === 'mat2_mul_' || name === 'mat3_mul_') {
      const expanded = lowerMatrixMul(toks, i, target, src, context, diags);
      if (expanded) {
        out.push(expanded.text);
        i = expanded.lastIndex;
        continue;
      }
    }

    // `mat2_rot_(a)` is a **rotation**, so it must lower to the library's `rot2`
    // function — not to a constructor. `mat2(a)` builds a diagonal matrix `[a,0,0,a]`,
    // which is a uniform scale, and the two are indistinguishable until you look at the
    // rendered image: a rotation silently becomes a zoom. `rot2` is spelled identically
    // in all six targets, so this rename is the whole lowering.
    if (name === 'mat2_rot_') {
      out.push('rot2');
      continue;
    }

    if (t.members[name] && isMemberAccess(toks, i)) {
      out.push(t.members[name]!);
      continue;
    }

    out.push(name);
  }

  // MSL: `vec4(1.0)` style scalar splat is fine, but `float4` splat is NOT allowed in
  // MSL — `float4(0.5)` is actually valid (scalar splat), so nothing to do here.
  // WGSL: a bare `vec4f(0.0)` is valid too. Kept as a comment because both languages
  // diverge from GLSL in *some* constructors and the guard belongs here if it appears.

  return { source: out.join(''), diagnostics: diags };
}

function isMemberAccess(toks: Token[], i: number): boolean {
  for (let j = i - 1; j >= 0; j--) {
    const p = toks[j]!;
    if (p.kind === 'space' || p.kind === 'comment') continue;
    return p.kind === 'punct' && p.text === '.';
  }
  return false;
}

/**
 * Number literal portability.
 *  - WGSL: `1.0` is fine, but a bare integer literal used as `f32` is not — and `1.`
 *    is illegal. Normalise to always have a decimal point and no trailing `f`.
 *  - HLSL/UE: append `f` to float literals is optional; leaving them bare keeps the
 *    source readable and lets overload resolution pick `float`.
 */
function rewriteNumber(
  text: string,
  target: TargetFormat,
  diags: Diagnostic[],
  src: string,
  pos: number,
  context: string,
): string {
  const isHex = /^0[xX]/.test(text);
  const isFloat = /[.eE]/.test(text) || /[fF]$/.test(text);
  let n = text.replace(/[fF]$/, '');

  // The hex check comes FIRST, before any per-target formatting. It used to sit after the
  // `wgsl` early-return, which made it unreachable on that target: a hex literal would
  // silently become `0x10.0` (invalid) and the diagnostic intended to warn about exactly
  // that never fired. Ordering a diagnostic after a `return` is a silent way to delete it.
  if (isHex) {
    diags.push({
      severity: 'warning',
      code: 'RW002_HEX_LITERAL',
      message: `Hex literal \`${text}\` found in ${context}; bit-exact integer behaviour differs per target.`,
      at: { line: lineOf(src, pos) },
      fix: 'Use the sin-based hash helpers in the shared library instead of integer hashing.',
    });
    // Rewrite it to a decimal literal so at least the output is well-formed. Hex is
    // banned by the subset, so reaching here means an author or an LLM went around it.
    const value = Number.parseInt(n, 16);
    return Number.isFinite(value) ? String(value) : n;
  }

  if (target === 'wgsl') {
    if (!isFloat) {
      // An integer literal appearing where the subset implies a float. WGSL requires
      // explicit `f32(...)`; the safest general form is a float literal.
      return `${n}.0`;
    }
    if (n.startsWith('.')) n = `0${n}`;
    if (n.endsWith('.')) n = `${n}0`;
    return n;
  }
  if (n.startsWith('.')) n = `0${n}`;
  if (n.endsWith('.')) n = `${n}0`;
  return n;
}

/**
 * `mat2_mul_(m, v)` → the target's own matrix×vector form.
 *
 * GLSL, WGSL and MSL are **column-major**: `mat2(c, -s, s, c)` stores (c,−s) as its first
 * *column*, so `m * v` is the ordinary product and needs no help.
 *
 * HLSL is **row-major**, and that is not a storage detail — it changes the answer. The same
 * constructor stores (c,−s) as its first *row*, so the matrix is the transpose of the GLSL one.
 * Two consequences, both verified against the real compiler rather than recalled:
 *
 *   * `m * v` does not compile at all (`dxc`: "cannot convert from 'float2' to 'float2x2'`),
 *     because `*` on a matrix is component-wise;
 *   * `mul(m, v)` compiles and is **wrong** — it multiplies by the transpose, which mirrors every
 *     rotation in the document. Disassembling `mul(float2x2(cos,-sin,sin,cos), float2(1,0))`
 *     yields `(cos, sin)`; GLSL's `mat2(cos,-sin,sin,cos) * vec2(1,0)` yields `(cos, -sin)`.
 *     `mul(v, m)` yields `(cos, -sin)` on both, which is why the arguments are swapped here.
 *
 * Every example happened to use `rotate = 0`, so nothing in the repository showed it: the failure
 * mode is "the pattern rotates the other way in Unity than in the workbench", which is exactly
 * the class of difference this whole rewriter exists to eliminate.
 */
function lowerMatrixMul(
  toks: Token[],
  i: number,
  target: TargetFormat,
  src: string,
  context: string,
  diags: Diagnostic[],
): { text: string; lastIndex: number } | null {
  // Expect: name ( expr , expr )
  let j = i + 1;
  const skipWs = () => {
    while (j < toks.length && (toks[j]!.kind === 'space' || toks[j]!.kind === 'comment')) j++;
  };
  skipWs();
  if (toks[j]?.text !== '(') return null;
  const open = j;
  j++;
  let depth = 1;
  const commas: number[] = [];
  for (; j < toks.length; j++) {
    const tk = toks[j]!;
    if (tk.text === '(') depth++;
    else if (tk.text === ')') {
      depth--;
      if (depth === 0) break;
    } else if (tk.text === ',' && depth === 1) commas.push(j);
  }
  if (depth !== 0 || commas.length !== 1) {
    diags.push({
      severity: 'error',
      code: 'RW003_MATMUL_ARITY',
      message: `mat*_mul_ requires exactly two arguments in ${context}.`,
      at: { line: lineOf(src, toks[i]!.pos) },
      fix: 'Call as mat2_mul_(matrix, vector).',
    });
    return null;
  }
  const close = j;
  // The arguments are spliced into the output **verbatim**, so they must themselves be
  // lowered for the target. Renaming them here rather than re-entering `rewriteSource`
  // keeps this a single pass and avoids duplicating lint diagnostics: `rewriteSource`
  // already recursed into the argument tokens on its way to this call, but the tokens
  // it emitted were discarded when we took over. Missing this is how
  // `mat2_mul_(rot2(a), (uv - vec2(0.5,0.5)) * scale)` leaks a `vec2(` into HLSL/WGSL/MSL.
  const lowerTokens = (from: number, to: number): string =>
    rewriteSource(toks.slice(from, to).map((x) => x.text).join(''), target, context).source;
  const argA = lowerTokens(open + 1, commas[0]!);
  const argB = lowerTokens(commas[0]! + 1, close);
  const isHlslFamily = target === 'hlsl' || target === 'ue' || target === 'shaderlab';
  /*
   * HLSL: `mul(vector, matrix)` — arguments **swapped**, and the swap is the whole point (see the
   * doc comment: row-major construction makes `mul(m, v)` the transpose of the GLSL result).
   * Everything else: the plain product, because those languages agree with GLSL.
   */
  const text = isHlslFamily ? `mul(${argB}, ${argA})` : `(${argA} * ${argB})`;
  return { text, lastIndex: close };
}

/* -------------------------------------------------------------------------- */
/* Structural checks                                                           */
/* -------------------------------------------------------------------------- */

/**
 * Cheap, target-independent validation of the restricted subset, run *before*
 * rewriting. Catches the mistakes an LLM makes most often, with better messages than
 * a real shader compiler would give, and without needing a compiler round-trip.
 */
export function lintSubset(src: string, context = 'source'): Diagnostic[] {
  const diags: Diagnostic[] = [];
  const toks = tokenize(src).filter((t) => t.kind !== 'space' && t.kind !== 'comment');

  for (let i = 0; i < toks.length; i++) {
    const tk = toks[i]!;
    if (tk.kind !== 'ident') continue;
    if (tk.text !== 'for') continue;

    // Re-tokenize the loop header without the whitespace separators. Joining tokens
    // with a space (the obvious implementation) makes `i ++` unmatchable against `i\+\+`
    // and fires this warning on every well-formed loop — a false positive that trains
    // people to ignore the diagnostics, which is worse than having none.
    //
    // The bound may be signed (`for (int j = -1; j <= 1; j++)`, which the Voronoi
    // neighbourhood scan uses) and the comparison may be `<=`, so both are accepted
    // here. This is a portability *warning*, not a validator: the point is to catch the
    // shapes the subset forbids (non-constant bounds, compound updates), not to be a
    // grammar.
    const header = toks.slice(i, i + 24).map((x) => x.text).join('');
    const ok =
      /^for\(int[A-Za-z_$][A-Za-z0-9_$]*=-?[0-9]+;[A-Za-z_$][A-Za-z0-9_$]*(?:<=?|>=?)[A-Za-z_0-9]+;[A-Za-z_$][A-Za-z0-9_$]*\+\+\)/.test(
        header,
      );
    if (!ok) {
      diags.push({
        severity: 'warning',
        code: 'RW010_LOOP_SHAPE',
        message: `Unusual \`for\` shape in ${context}: ${header.slice(0, 60)}…`,
        at: { line: lineOf(src, tk.pos) },
        fix: 'Use `for (int i = 0; i < N; i++)` with N an identifier or integer literal.',
      });
    }
  }

  // Trailing `break_`/`return_` usage is fine, but raw `break` inside the library
  // bodies would not be portable; flag it so catalog authors use the mask form.
  if (/\bbreak\s*;/.test(src)) {
    diags.push({
      severity: 'warning',
      code: 'RW011_BREAK',
      message: `Raw \`break;\` in ${context}.`,
      fix: 'Prefer a per-iteration weight mask (see fbm2) — it is uniformly legal across all six targets.',
    });
  }

  /*
   * A builtin whose per-target spelling differs, written as the builtin.
   *
   * An **error**, not a warning: `mix(a,b,t)` is correct GLSL and a compile error in HLSL, so
   * this is the "works in one target of six" class the subset exists to eliminate.
   *
   * This check lives here rather than only in `catalog.test.ts` because the rule now has to
   * hold for text that arrives at runtime. The test scans the shipped catalog bodies, which is
   * sound exactly as long as bodies are constants a test can enumerate — `custom_glsl` ended
   * that, and a rule enforced only against the inputs you already have is not a rule.
   */
  for (let i = 0; i < toks.length; i++) {
    const tk = toks[i]!;
    if (tk.kind !== 'ident' || isMemberAccess(toks, i)) continue;
    const alias = BANNED_BUILTIN_SPELLINGS[tk.text];
    if (!alias) continue;
    // Only a *call* is a problem. `texture` as a local name is handled elsewhere, and a bare
    // identifier that happens to share the name is not calling the builtin.
    if (toks[i + 1]?.text !== '(') continue;
    // The alias itself contains the builtin as a prefix (`mod_` starts with `mod`), but the
    // tokenizer reads `mod_` as one identifier, so the map is never reached for it.
    diags.push({
      severity: 'error',
      code: 'RW012_BUILTIN_SPELLING',
      message: `\`${tk.text}(…)\` in ${context} must be written \`${alias}(…)\`.`,
      at: { line: lineOf(src, tk.pos) },
      fix:
        `\`${tk.text}\` is spelled differently in at least one target (see BANNED_BUILTIN_SPELLINGS). ` +
        `Use the alias \`${alias}\` and the rewriter emits the right call per language.`,
    });
  }

  /*
   * A vector constructor called with a **single numeric literal**.
   *
   * `vec4(0.0)` is a scalar splat: valid GLSL, valid WGSL (`vec4f(0.0)`), valid MSL — and a hard
   * error in HLSL, where a vector constructor must be given exactly as many components as it has:
   *
   *     error: too few elements in vector initialization (expected 4 elements, have 1)
   *
   * The rewriter renames the type (`vec4` -> `float4`) and does not touch the argument list, so
   * this is the "works in five targets of six" class the subset exists to eliminate — and it is
   * not hypothetical: a body written this way emitted clean GLSL, passed every structural check,
   * and was caught only when `dxc` was pointed at the result.
   *
   * Only a **numeric literal** argument is flagged, which makes the rule exact rather than
   * conservative: `vec4(v)` (a copy) and `vec4(a, b, c, d)` are legal in every target, and a bare
   * number can never be a vector. Typed analysis would catch more and would also reject valid
   * source, and the subset's whole premise is that its rules are checkable by looking.
   */
  for (let i = 0; i < toks.length; i++) {
    const tk = toks[i]!;
    if (tk.kind !== 'ident' || isMemberAccess(toks, i)) continue;
    if (!(TYPE_TOKENS as readonly string[]).includes(tk.text)) continue;
    if (toks[i + 1]?.text !== '(') continue;
    const arg = toks[i + 2];
    const close = toks[i + 3];
    if (arg?.kind !== 'number' || close?.text !== ')') continue;
    diags.push({
      severity: 'error',
      code: 'RW013_SCALAR_SPLAT_CONSTRUCTOR',
      message: `\`${tk.text}(${arg.text})\` in ${context} is a scalar splat, which HLSL rejects.`,
      at: { line: lineOf(src, tk.pos) },
      fix:
        `Write all components: \`${tk.text}(${Array.from({ length: Number(tk.text.slice(3)) }, () => arg.text).join(', ')})\`.` +
        ` GLSL/WGSL/MSL accept the short form and HLSL does not, so the long form is the only ` +
        `portable one.`,
    });
  }

  /*
   * A matrix product written with `*`.
   *
   * `mat2`-returning helpers are the only matrices a body can build, and HLSL rejects
   * `matrix * vector` outright ("cannot convert from 'float2' to 'float2x2'") — so a body that
   * gets this wrong does not merely differ, it fails to compile on two of the six targets. The
   * alias exists for exactly this reason, and nothing was enforcing it: `transform` happens to
   * use `mat2_mul_`, but a `custom_glsl` snippet is written by a person or a model and has no
   * such habit.
   *
   * Detected at token level, which is precise enough to be useful and honest about its limits:
   *
   *   * a call to a known matrix-returning helper (`rot2`, `scale2`, `mat2_rot_`) on either side
   *     of a `*`;
   *   * an identifier that this body declared as `mat2`/`mat3`.
   *
   * What it cannot see is a matrix that arrives from a helper it does not know about, which is
   * why the message names the fix rather than assuming the author meant a product.
   */
  const matrixLocals = new Set<string>();
  for (let i = 0; i + 1 < toks.length; i++) {
    if (toks[i]!.kind !== 'ident') continue;
    if (toks[i]!.text !== 'mat2' && toks[i]!.text !== 'mat3') continue;
    const name = toks[i + 1]!;
    if (name.kind === 'ident') matrixLocals.add(name.text);
  }
  const MATRIX_CALLS = new Set(['rot2', 'scale2', 'mat2_rot_', 'mat3_rot_']);
  /** Is the token at `i` the start of something matrix-valued? */
  const isMatrixStart = (i: number): boolean => {
    const tk = toks[i];
    if (!tk) return false;
    if (tk.kind === 'ident' && MATRIX_CALLS.has(tk.text) && toks[i + 1]?.text === '(') return true;
    return tk.kind === 'ident' && matrixLocals.has(tk.text);
  };
  /**
   * Is the token at `i` the **end** of something matrix-valued?
   *
   * Needed because `rot2(0.5) * v` puts a `)` next to the `*`, not the name. Walking back over the
   * balanced call is the whole difference between catching this and missing it — the first version
   * of this check only looked at the adjacent token and let the exact case it was written for
   * through, which the test caught.
   */
  const isMatrixEnd = (i: number): boolean => {
    let k = i;
    if (toks[k]?.text === ')') {
      let depth = 0;
      for (; k >= 0; k--) {
        if (toks[k]!.text === ')') depth++;
        else if (toks[k]!.text === '(') {
          depth--;
          if (depth === 0) break;
        }
      }
      k--;
    }
    return isMatrixStart(k);
  };
  for (let i = 1; i + 1 < toks.length; i++) {
    if (toks[i]!.text !== '*') continue;
    if (!isMatrixEnd(i - 1) && !isMatrixStart(i + 1)) continue;
    diags.push({
      severity: 'error',
      code: 'RW014_MATRIX_PRODUCT',
      message:
        `\`*\` with a matrix operand in ${context}: HLSL rejects \`matrix * vector\` outright ` +
        `("cannot convert"), so this would fail to compile on two of six targets.`,
      at: { line: lineOf(src, toks[i]!.pos) },
      fix:
        'Write `mat2_mul_(matrix, vector)`. The rewriter lowers it per target: a plain product ' +
        'where matrices are column-major (GLSL/WGSL/MSL) and `mul(vector, matrix)` on HLSL, ' +
        'where the constructor order makes the naive `mul(matrix, vector)` a transpose.',
    });
  }

  return diags;
}

/* -------------------------------------------------------------------------- */
/* Keyword safety                                                              */
/* -------------------------------------------------------------------------- */

/**
 * Identifiers that are reserved in at least one target. A node parameter or uniform
 * named `sample` or `in` compiles fine in GLSL and explodes in MSL/HLSL. Renaming at
 * the source is the only reliable fix, so the IR validator calls this.
 */
const RESERVED_ANYWHERE = new Set([
  // MSL / Metal
  'sample', 'constant', 'device', 'thread', 'threadgroup', 'kernel', 'vertex',
  'fragment', 'metal', 'texture2d', 'half', 'half2', 'half3', 'half4',
  // HLSL
  'cbuffer', 'tbuffer', 'register', 'packoffset', 'matrix', 'vector', 'float2x2',
  'linear', 'centroid', 'nointerpolation', 'snorm', 'unorm',
  // WGSL
  'var', 'let', 'fn', 'loop', 'alias', 'override', 'enable', 'diagnostic', 'vec2f',
  'vec3f', 'vec4f', 'mat2x2f', 'f32', 'i32', 'u32',
  // GLSL
  'sampler', 'flat', 'noperspective', 'smooth', 'buffer', 'shared',
  // TypeScript-ish (the GUI uses these names as object keys)
  'constructor', 'prototype', 'default',
]);

export function isReservedAnywhere(name: string): boolean {
  return RESERVED_ANYWHERE.has(name.toLowerCase());
}

/** Suggest a safe replacement for a reserved identifier. */
export function safeIdentifier(name: string, taken: Set<string> = new Set()): string {
  let candidate = isReservedAnywhere(name) ? `${name}_v` : name;
  candidate = candidate.replace(/[^A-Za-z0-9_]/g, '_');
  if (/^[0-9]/.test(candidate)) candidate = `_${candidate}`;
  let n = 2;
  const base = candidate;
  while (taken.has(candidate) || isReservedAnywhere(candidate)) {
    candidate = `${base}${n++}`;
  }
  return candidate;
}
