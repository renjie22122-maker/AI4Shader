/**
 * AI4Shader — Restricted GLSL subset
 * ==================================
 *
 * Every node body in the catalog is written in this subset. It is deliberately tiny,
 * because the multi-target rewriter is a *token-level* transform (see `rewrite.ts`)
 * rather than a full parser per language. A subset that is closed under the rewriter
 * is what lets one authored graph project to GLSL/WGSL/HLSL/MSL/ShaderLab/UE with a
 * single source of truth.
 *
 * ── ALLOWED ────────────────────────────────────────────────────────────────────
 *   • Literals: float (`1.0` — ALWAYS with a decimal point), int, bool
 *   • Types: float vec2 vec3 vec4 mat2 mat3, bool, int
 *   • Constructors: any arity, including nested (`vec3(vec2(a,b), c)`, `vec4(v, 1.0)`)
 *   • Operators: + - * / unary-, comparisons, `&&` `||` `!`, ternary `?:`
 *   • Swizzles: single-component (`.x`) or 4-char (`.xyzw`, `.rgba`)
 *   • Control flow: `if`/`else`, `for (int i = 0; i < N; i++)` with N a literal
 *   • The function whitelist below
 *   • Local `float`/`vec*` temporaries and `const`-free declarations
 *
 * ── BANNED (the rewriter cannot prove these correct) ───────────────────────────
 *   • `#` preprocessor, `precision`, `layout`, attribute/varying/uniform decls
 *   • samplers, `texture(...)` (use the `tex2D_` alias), `discard`, `return` in nodes
 *   • arrays, structs, function pointers, recursion, `switch`, `while`, `do`
 *   • bitwise ops & `uint` (WebGL2 GLSL ES 3.00 supports them but MSL/WGSL integer
 *     promotion rules differ; the hash functions use the sin-based trick instead)
 *   • implicit int→float conversion: write `float(i)`, never `i * 0.5`
 *   • non-constant loop bounds
 *   • multi-component swizzles that drop the w channel (`.xyz` on a `vec4` is fine,
 *     `.zyx` is fine — but never rely on `.w` in a 2- or 3-component swizzle)
 *   • matrix multiplication where operand order matters (MSL/HLSL differ) — use
 *     `mat2_mul_` / `mat3_mul_` helpers instead of raw `*` when multiplying matrices
 *
 * ── ALIASES ────────────────────────────────────────────────────────────────────
 * Functions that have no common name across languages are written with a trailing
 * underscore and resolved per target:
 *
 *   fract_(x)          GLSL fract      HLSL/MSL frac
 *   mod_(x, y)         GLSL mod        HLSL/MSL fmod      WGSL (x - y*floor(x/y))
 *   mix_(a, b, t)      GLSL mix        HLSL/MSL lerp      WGSL mix
 *   atan2_(y, x)       GLSL atan(y,x)  HLSL/MSL atan2     WGSL atan2
 *   inversesqrt_(x)    GLSL inversesqrt HLSL/MSL rsqrt
 *   dFdx_/dFdy_        GLSL dFdx       HLSL ddx/ddy       MSL dfdx/dfdy
 *   tex2D_(uv)         GLSL texture    HLSL tex2D         MSL sample         WGSL textureSample
 *   mat2_mul_(m, v)    order-safe matrix*vector for every target
 *   mat3_mul_(m, v)
 *
 * `rewrite.ts` asserts that no banned token survives; violations are reported as
 * `RW0xx` diagnostics pointing at the node that produced them.
 */

/** Function names that exist with identical semantics in all six targets. */
export const PORTABLE_FUNCTIONS = [
  'abs', 'acos', 'asin', 'ceil', 'clamp', 'cos', 'cross', 'degrees', 'distance',
  'dot', 'exp', 'exp2', 'faceforward', 'floor', 'length', 'log', 'log2', 'max',
  'min', 'normalize', 'pow', 'radians', 'reflect', 'refract', 'sign', 'sin',
  'smoothstep', 'sqrt', 'step', 'tan', 'float', 'int', 'bool',
  // Derivative — fragment stage only, and spelled the same in GLSL ES 3.00, WGSL, HLSL, MSL and
  // therefore in ShaderLab and UE. It is here because the `antialias` node reads it: a derivative
  // is the only way to know how wide one screen pixel is in the units of the value being
  // thresholded, which is what analytic anti-aliasing needs.
  'fwidth',
] as const;

/** Aliases the rewriter MUST resolve. Every one is rewritten by `rewrite.ts`. */
export const REWRITE_ALIASES = [
  'fract_', 'mod_', 'mix_', 'atan2_', 'inversesqrt_', 'dFdx_', 'dFdy_',
  'tex2D_', 'mat2_mul_', 'mat3_mul_',
] as const;

/**
 * The builtin spelling each alias replaces, and why using the builtin is not merely untidy.
 *
 * `mix_` exists because HLSL spells it `lerp`; `mod_` because HLSL/MSL `fmod` **truncates**
 * while GLSL `mod` **floors**; `fract_` because HLSL says `frac`; `atan2_` because HLSL has no
 * vector overload; `tex2D_` because the sampling call differs in all six languages.
 *
 * So a body that writes `mix(a, b, t)` instead of `mix_(a, b, t)` is correct GLSL and a
 * compile error in HLSL — one target of six, no diagnostic. That is the exact shape of bug
 * this subset exists to prevent, and until now the rule was enforced only by *a unit test over
 * the catalog*: `catalog.test.ts` scans the shipped bodies for these spellings, which works
 * because catalog bodies are constants that a test can enumerate.
 *
 * `custom_glsl` broke that assumption — a body can now arrive at runtime — so the rule has to
 * be enforced in production. `lintSubset` reports these, which means it holds for every body,
 * wherever it came from.
 */
export const BANNED_BUILTIN_SPELLINGS: Readonly<Record<string, string>> = {
  fract: 'fract_',
  mod: 'mod_',
  mix: 'mix_',
  atan: 'atan2_',
  inversesqrt: 'inversesqrt_',
  texture: 'tex2D_',
};

/** Type names that get renamed per target. */
export const TYPE_TOKENS = ['vec2', 'vec3', 'vec4', 'mat2', 'mat3'] as const;

/** Tokens the rewriter rejects outright. */
export const BANNED_TOKENS = [
  'discard', 'switch', 'while', 'do', 'struct', 'return', 'uint', 'sampler2D',
  'texture', 'textureLod', 'gl_FragColor', 'gl_FragCoord', 'precision', 'layout',
  'uniform', 'varying', 'attribute', 'in', 'out', 'inout', 'const',
] as const;

/**
 * Canonical GLSL implementations of the shared helper library.
 *
 * These are emitted once per shader, above the node bodies, and are themselves
 * written in the restricted subset (they are held to the same rewriter). Keeping
 * them in one place means the noise in the preview is bit-identical to the noise in
 * the Unity build — the #1 source of "it looked different in the tool" bugs is a
 * re-implemented noise function per target.
 *
 * **Editing note.** Everything below is one template literal, so a backtick inside a comment has
 * to be escaped (`\``) or the file stops parsing — the failure is `Expected a semicolon` from the
 * projection step, which points at a line *after* the mistake and is easy to misread as a code
 * error. Function *order* also matters here in a way TypeScript does not enforce: GLSL and MSL
 * require a declaration before its first call, so helpers must be defined above their callers.
 */
export const GLSL_LIBRARY = /* glsl */ `
// ---- hashing -------------------------------------------------------------
// sin-based rather than integer bit-twiddling: identical source in all 6 targets,
// no uint, no bitwise ops, and stable enough for procedural effects.
float hash11(float p) {
  p = fract_(p * 0.1031);
  p = p * (p + 33.33);
  return fract_(p * (p + p));
}
float hash21(vec2 p) {
  vec3 p3 = fract_(vec3(p.x, p.y, p.x) * 0.1031);
  p3 = p3 + dot(p3, vec3(p3.y, p3.z, p3.x) + 33.33);
  return fract_((p3.x + p3.y) * p3.z);
}
vec2 hash22(vec2 p) {
  vec3 p3 = fract_(vec3(p.x, p.y, p.x) * vec3(0.1031, 0.1030, 0.0973));
  p3 = p3 + dot(p3, vec3(p3.y, p3.z, p3.x) + 33.33);
  return fract_(vec2(p3.x, p3.y) * vec2(p3.z, p3.x));
}
float hash31(vec3 p) {
  p = fract_(p * 0.1031);
  p = p + dot(p, vec3(p.y, p.z, p.x) + 33.33);
  return fract_((p.x + p.y) * p.z);
}

// ---- value / perlin noise ------------------------------------------------
/*
 * ---- tileable, and the only implementation ---------------------------------
 *
 * A noise field is not periodic: \`hash21(floor(p))\` depends on the *absolute* cell, so uv square
 * N+1 does not agree with square 1 and a texture made from it seams the moment an engine repeats
 * it. "Make me a seamless tileable texture" is one of the commonest things anyone asks a shader
 * tool for, and this library could not do it at all before \`wrap2\`.
 *
 * The fix is one line — hash the cell index **modulo the period** — and the geometry stays
 * unwrapped; only the *lookup* wraps. Keeping the geometry unwrapped is what makes the result
 * continuous rather than merely similar on both sides: the grid point of cell \`period\` is the
 * grid point of cell 0 translated by \`period\`, so the interpolation across the seam agrees
 * exactly. Hashing the wrapped index while building the point from the wrapped index too is the
 * tempting version, and it produces a visible crease.
 *
 * There is **one** lattice implementation here, not a tiled copy of an untiled one. \`noise2\`,
 * \`gnoise2\`, \`fbm2\` and \`voronoi2\` are one-line wrappers that pass \`on = 0\`, so the two can
 * never drift apart — and \`on = 0\` is exact rather than approximate, because
 * \`mix_(i, mod_(i, per), 0.0)\` is \`i\` for every finite \`i\`. \`per\` is floored at 1 because
 * \`mod_\` by zero is undefined, and a document with \`scale = 0\` must not produce NaN.
 */
vec2 wrap2(vec2 i, vec2 period, float on) {
  vec2 per = max(period, vec2(1.0, 1.0));
  return mix_(i, mod_(i, per), on);
}
float noise2t(vec2 p, vec2 period, float on) {
  vec2 i = floor(p);
  vec2 f = fract_(p);
  vec2 u = f * f * (vec2(3.0, 3.0) - 2.0 * f);
  float a = hash21(wrap2(i, period, on));
  float b = hash21(wrap2(i + vec2(1.0, 0.0), period, on));
  float c = hash21(wrap2(i + vec2(0.0, 1.0), period, on));
  float d = hash21(wrap2(i + vec2(1.0, 1.0), period, on));
  return mix_(mix_(a, b, u.x), mix_(c, d, u.x), u.y);
}
float noise2(vec2 p) {
  return noise2t(p, vec2(1.0, 1.0), 0.0);
}
float gnoise2t(vec2 p, vec2 period, float on) {
  vec2 i = floor(p);
  vec2 f = fract_(p);
  vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  vec2 ga = hash22(wrap2(i, period, on)) * 2.0 - 1.0;
  vec2 gb = hash22(wrap2(i + vec2(1.0, 0.0), period, on)) * 2.0 - 1.0;
  vec2 gc = hash22(wrap2(i + vec2(0.0, 1.0), period, on)) * 2.0 - 1.0;
  vec2 gd = hash22(wrap2(i + vec2(1.0, 1.0), period, on)) * 2.0 - 1.0;
  float va = dot(ga, f);
  float vb = dot(gb, f - vec2(1.0, 0.0));
  float vc = dot(gc, f - vec2(0.0, 1.0));
  float vd = dot(gd, f - vec2(1.0, 1.0));
  return 0.5 + 0.5 * mix_(mix_(va, vb, u.x), mix_(vc, vd, u.x), u.y);
}
float gnoise2(vec2 p) {
  return gnoise2t(p, vec2(1.0, 1.0), 0.0);
}
float noise3(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract_(p);
  vec3 u = f * f * (vec3(3.0, 3.0, 3.0) - 2.0 * f);
  float n000 = hash31(i);
  float n100 = hash31(i + vec3(1.0, 0.0, 0.0));
  float n010 = hash31(i + vec3(0.0, 1.0, 0.0));
  float n110 = hash31(i + vec3(1.0, 1.0, 0.0));
  float n001 = hash31(i + vec3(0.0, 0.0, 1.0));
  float n101 = hash31(i + vec3(1.0, 0.0, 1.0));
  float n011 = hash31(i + vec3(0.0, 1.0, 1.0));
  float n111 = hash31(i + vec3(1.0, 1.0, 1.0));
  float x00 = mix_(n000, n100, u.x);
  float x10 = mix_(n010, n110, u.x);
  float x01 = mix_(n001, n101, u.x);
  float x11 = mix_(n011, n111, u.x);
  return mix_(mix_(x00, x10, u.y), mix_(x01, x11, u.y), u.z);
}
// Fractal Brownian motion. The loop bound is a literal (per-target \`OCTAVE_LIMIT\`)
// and the live octave count is applied as a per-iteration weight mask rather than a
// \`break\`, because WGSL/MSL/HLSL place different restrictions on loop-carried control
// flow. Multiplying by a 0.0/1.0 mask is uniformly legal, branch-free, and lets the
// optimizer collapse the dead iterations at every tier.
//
// The tiled form comes **first**, and that ordering is load-bearing: GLSL and MSL require a
// function to be declared before it is called, and the wrapper below calls it. Reversing the two
// is invisible to every structural check — the emitter is happy, the subset linter is happy, the
// six-target emission is byte-for-byte plausible — and it fails the moment a real compiler reads
// the file. That is what happened, and glslangValidator is what said so.
/*
 * \`period\` is in the *input* space of the octave loop, so it has to scale with the loop: octave
 * \`i\` samples at \`p * lacunarity^i\`, its cell lattice is \`lacunarity^i\` times denser, and its wrap
 * period has to grow by the same factor. Get that wrong and the *first* octave tiles while the
 * rest do not — a faint seam plus a strange grain, which is the sort of thing that ships because
 * the low frequencies cover it up.
 */
float fbm2t(vec2 p, int octaves, float lacunarity, float gain, float ridged, vec2 period, float on) {
  float sum = 0.0;
  float amp = 0.5;
  float norm = 0.0;
  vec2 per = period;
  for (int i = 0; i < OCTAVE_LIMIT; i++) {
    float live = step(float(i) + 0.5, float(octaves));
    float n = noise2t(p, per, on);
    float r = 1.0 - abs(n * 2.0 - 1.0);
    n = mix_(n, r * r, clamp(ridged, 0.0, 1.0));
    sum = sum + amp * n * live;
    norm = norm + amp * live;
    p = p * lacunarity;
    per = per * lacunarity;
    amp = amp * gain;
  }
  return sum / max(norm, 0.00001);
}
float fbm2(vec2 p, int octaves, float lacunarity, float gain, float ridged) {
  return fbm2t(p, octaves, lacunarity, gain, ridged, vec2(1.0, 1.0), 0.0);
}
// Worley / Voronoi. Returns (F1, F2, cellHash) so callers can pick edges, cells or
// distance fields without needing a second traversal.
/*
 * Periodic Worley — first, for the same declaration-order reason as \`fbm2t\`. Only the *hash
 * lookups* wrap; the neighbour offsets \`g\` stay unwrapped, so the feature point of cell \`period\`
 * is the feature point of cell 0 translated by one period. That is what makes the field continuous
 * across the seam rather than merely similar on both sides — and it is also why the jitter cannot
 * simply be hashed from the wrapped index while the point is built from the unwrapped one, which
 * is the tempting version and produces a visible crease.
 */
vec3 voronoi2t(vec2 p, float randomness, vec2 period, float on) {
  vec2 n = floor(p);
  vec2 f = fract_(p);
  float f1 = 8.0;
  float f2 = 8.0;
  float id = 0.0;
  for (int j = -1; j <= 1; j++) {
    for (int i = -1; i <= 1; i++) {
      vec2 g = vec2(float(i), float(j));
      vec2 cell = wrap2(n + g, period, on);
      vec2 o = hash22(cell);
      o = (o - 0.5) * randomness + 0.5;
      vec2 r = g + o - f;
      float d = dot(r, r);
      if (d < f1) {
        f2 = f1;
        f1 = d;
        id = hash21(cell);
      } else if (d < f2) {
        f2 = d;
      }
    }
  }
  return vec3(sqrt(f1), sqrt(f2), id);
}
vec3 voronoi2(vec2 p, float randomness) {
  return voronoi2t(p, randomness, vec2(1.0, 1.0), 0.0);
}

// ---- signed distance fields ---------------------------------------------
float sd_circle(vec2 p, float r) { return length(p) - r; }
float sd_box(vec2 p, vec2 b) {
  vec2 d = abs(p) - b;
  return length(max(d, vec2(0.0, 0.0))) + min(max(d.x, d.y), 0.0);
}
float sd_hexagon(vec2 p, float r) {
  vec2 k = vec2(-0.8660254, 0.5);
  p = abs(p);
  p = p - 2.0 * min(dot(k, p), 0.0) * k;
  p = p - vec2(clamp(p.x, -r * 1.1547005, r * 1.1547005), r);
  return length(p) * sign(p.y);
}
float sd_triangle(vec2 p, float r) {
  float k = 1.7320508;
  p.x = abs(p.x) - r;
  p.y = p.y + r / k;
  float flip = step(0.0, p.x + k * p.y);
  p = mix_(p, vec2(p.x - k * p.y, -k * p.x - p.y) / 2.0, vec2(flip, flip));
  p.x = p.x - clamp(p.x, -2.0 * r, 0.0);
  return -length(p) * sign(p.y);
}
float sd_star(vec2 p, float r, float points) {
  float an = 3.14159265 / points;
  float en = 3.14159265 / 2.0;
  vec2 acs = vec2(cos(an), sin(an));
  vec2 ecs = vec2(cos(en), sin(en));
  float bn = mod_(atan2_(p.x, p.y), 2.0 * an) - an;
  p = length(p) * vec2(cos(bn), abs(sin(bn)));
  p = p - r * acs;
  p = p + ecs * clamp(-dot(p, ecs), 0.0, r * acs.y / ecs.y);
  return length(p) * sign(p.x);
}
float sd_ring(vec2 p, float r, float thickness) {
  return abs(length(p) - r) - thickness;
}
float sd_segment(vec2 p, vec2 a, vec2 b) {
  vec2 pa = p - a;
  vec2 ba = b - a;
  float h = clamp(dot(pa, ba) / max(dot(ba, ba), 0.00001), 0.0, 1.0);
  return length(pa - ba * h);
}

// ---- color ---------------------------------------------------------------
vec3 hsv2rgb(vec3 c) {
  vec3 rgb = clamp(abs(mod_(c.x * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
  return c.z * mix_(vec3(1.0, 1.0, 1.0), rgb, c.y);
}
vec3 rgb2hsv(vec3 c) {
  vec4 K = vec4(0.0, -0.3333333, 0.6666667, -1.0);
  vec4 p = mix_(vec4(c.z, c.y, K.w, K.z), vec4(c.y, c.z, K.x, K.y), step(c.z, c.y));
  vec4 q = mix_(vec4(p.x, p.y, p.w, c.x), vec4(c.x, p.y, p.z, p.x), step(p.x, c.x));
  float d = q.x - min(q.w, q.y);
  float e = 0.0000001;
  return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}
// Inigo Quilez cosine gradient — the workhorse for "make it look designed" palettes.
vec3 palette(float t, vec3 a, vec3 b, vec3 c, vec3 d) {
  return a + b * cos(6.28318530718 * (c * t + d));
}
float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
vec3 srgb2linear(vec3 c) {
  vec3 lo = c / 12.92;
  vec3 hi = pow((c + vec3(0.055, 0.055, 0.055)) / 1.055, vec3(2.4, 2.4, 2.4));
  return mix_(lo, hi, step(vec3(0.04045, 0.04045, 0.04045), c));
}
vec3 linear2srgb(vec3 c) {
  vec3 lo = c * 12.92;
  vec3 hi = 1.055 * pow(c, vec3(0.4166667, 0.4166667, 0.4166667)) - 0.055;
  return mix_(lo, hi, step(vec3(0.0031308, 0.0031308, 0.0031308), c));
}
vec3 tonemap_aces(vec3 x) {
  float a = 2.51; float b = 0.03; float c = 2.43; float d = 0.59; float e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0));
}

// ---- geometry helpers ----------------------------------------------------
mat2 rot2(float a) {
  float s = sin(a);
  float c = cos(a);
  return mat2(c, -s, s, c);
}
mat2 scale2(vec2 s) {
  return mat2(s.x, 0.0, 0.0, s.y);
}
// Order-safe matrix*vector. HLSL packs matrices the other way round, so raw \`m * v\`
// cannot be rewritten by a token transform; these helpers are lowered per target.
vec2 mat2_mul_(mat2 m, vec2 v) {
  return vec2(m[0][0] * v.x + m[1][0] * v.y, m[0][1] * v.x + m[1][1] * v.y);
}
mat2 mat2_rot_(float a) { return rot2(a); }
`;

/** Loop bound for quality-scaled nodes; the emitter substitutes a per-tier literal. */
export const OCTAVE_LIMIT = 8;
