/**
 * AI4Shader — Target framework + the six concrete emitters
 * =========================================================
 *
 * Everything in this file is *lowering*: the one place where a language difference is
 * allowed to be visible. The IR itself never names a shading language, a binding, a
 * precision qualifier or an entry point; that is decided here, per target, once.
 *
 * Assembly is identical for every target and is expressed once, in `assemble()`:
 *
 *     [prologue]     version/pragma, binding struct or cbuffer, samplers, declarations
 *     [shim block]   rw(shimsFor(target)) — only WGSL has shims today
 *     [library]      rw(GLSL_LIBRARY), OCTAVE_LIMIT already substituted
 *     [bodies]       rw(bodies.lines.join('\n'))
 *     [epilogue]     the entry point: computes `uv`, evaluates the graph, writes out
 *
 * Three invariants hold for every target and are asserted by `test/targets.test.ts`:
 *
 *  1. DETERMINISM.  Output is a pure function of the IR. No clocks, no randomness, no
 *     iteration over unordered objects. `emit()` twice ⇒ byte-identical `source`.
 *  2. NO LEAKS.  Not one restricted-GLSL type name or `*_` alias may survive outside a
 *     comment in a non-GLSL target. `checkTargetLeaks()` is the executable form of that
 *     rule and every emitter's output is run through it in the tests.
 *  3. ONE Y FLIP.  The IR is GLSL-like (origin bottom-left, uv in [0,1]). Every target
 *     whose framebuffer origin is top-left flips exactly once, in the entry-point
 *     epilogue, never inside a node body, never in the helper library.
 *
 * Known gaps — places where the emitted source is deliberately not compiler-valid — are
 * documented per target in `targets/README.md` and summarised at the bottom of it. They
 * are *not* hidden: a clean `diagnostics` array means "the rewriter had nothing to
 * complain about", not "a driver will accept this".
 */

import {
  HOST_SYMBOLS,
  TYPE_ARITY,
  type Diagnostic,
  type NodeSchema,
  type ParamValue,
  type ShaderDoc,
  type ShaderType,
  type TargetFormat,
  type UniformDecl,
} from '../types.js';
import {
  literalOf,
  octavesFor,
  paramExpr,
  registerTarget,
  type TargetEmitter,
} from '../emitter.js';
import { GLSL_LIBRARY, REWRITE_ALIASES, TYPE_TOKENS } from '../glslSubset.js';
import { rewriteSource, shimsFor, tokenize } from '../rewrite.js';
import { lowerSubsetToWgsl, type WgslShape } from './wgslLower.js';

/* -------------------------------------------------------------------------- */
/* Local shapes                                                                */
/* -------------------------------------------------------------------------- */

/** The argument bundle `emit()` hands to a registered target emitter. */
export type EmitterArgs = Parameters<TargetEmitter>[0];
/** What an emitter must return. */
export type EmitterResult = ReturnType<TargetEmitter>;

/* -------------------------------------------------------------------------- */
/* Type spelling per target                                                    */
/* -------------------------------------------------------------------------- */

/**
 * The single table that maps an IR type to each language's spelling. Uniform
 * declarations, literal values and the output-colour coercion all read from it, so a
 * new type is added in exactly one place.
 */
const TYPE_NAME: Record<TargetFormat, Record<ShaderType, string>> = {
  glsl: {
    float: 'float', vec2: 'vec2', vec3: 'vec3', vec4: 'vec4',
    mat2: 'mat2', mat3: 'mat3', int: 'int', bool: 'bool',
  },
  wgsl: {
    float: 'f32', vec2: 'vec2f', vec3: 'vec3f', vec4: 'vec4f',
    mat2: 'mat2x2f', mat3: 'mat3x3f', int: 'i32', bool: 'bool',
  },
  hlsl: {
    float: 'float', vec2: 'float2', vec3: 'float3', vec4: 'float4',
    mat2: 'float2x2', mat3: 'float3x3', int: 'int', bool: 'bool',
  },
  msl: {
    float: 'float', vec2: 'float2', vec3: 'float3', vec4: 'float4',
    mat2: 'float2x2', mat3: 'float3x3', int: 'int', bool: 'bool',
  },
  shaderlab: {
    float: 'float', vec2: 'float2', vec3: 'float3', vec4: 'float4',
    mat2: 'float2x2', mat3: 'float3x3', int: 'float', bool: 'float',
  },
  ue: {
    float: 'MaterialFloat', vec2: 'float2', vec3: 'float3', vec4: 'float4',
    mat2: 'float2x2', mat3: 'float3x3', int: 'int', bool: 'bool',
  },
};

/** Uniform member spelling *inside* a UE Custom node function body (never `MaterialFloat`). */
const UE_BODY_TYPE: Record<ShaderType, string> = {
  float: 'float', vec2: 'float2', vec3: 'float3', vec4: 'float4',
  mat2: 'float2x2', mat3: 'float3x3', int: 'int', bool: 'bool',
};

function typeName(target: TargetFormat, t: ShaderType): string {
  return TYPE_NAME[target][t];
}

/**
 * `literalOf()` spells literals in GLSL (`vec3(0.5, 0.5, 0.5)`). Every other target
 * only needs the constructor renamed, so that is all this does — one substitution,
 * no per-target literal grammar.
 */
export function targetLiteral(
  target: TargetFormat,
  type: ShaderType,
  value: number | boolean | number[],
): string {
  const s = literalOf(type, value);
  if (target === 'glsl') return s;
  return s.replace(
    /\b(vec2|vec3|vec4|mat2|mat3)\(/g,
    (_m, t: string) => `${TYPE_NAME[target][t as ShaderType]}(`,
  );
}

/* -------------------------------------------------------------------------- */
/* Token-aware text surgery                                                    */
/* -------------------------------------------------------------------------- */

function isMemberAccess(toks: ReturnType<typeof tokenize>, i: number): boolean {
  for (let j = i - 1; j >= 0; j--) {
    const p = toks[j]!;
    if (p.kind === 'space' || p.kind === 'comment') continue;
    return p.kind === 'punct' && p.text === '.';
  }
  return false;
}

/**
 * Replace whole identifiers, never substrings, and never inside comments.
 *
 * This is how host symbols and uniform names are remapped per target (`iTime` →
 * `U.iTime`, `u_scale` → `U.u_scale`, …). It runs *after* `rw()`, so the replacement
 * text is target-native and must not be rewritten again.
 */
export function substituteIdentifiers(src: string, map: Readonly<Record<string, string>>): string {
  const toks = tokenize(src);
  const out: string[] = [];
  for (let i = 0; i < toks.length; i++) {
    const t = toks[i]!;
    if (t.kind !== 'ident') {
      out.push(t.text);
      continue;
    }
    const rep = map[t.text];
    out.push(rep !== undefined && !isMemberAccess(toks, i) ? rep : t.text);
  }
  return out.join('');
}

/** Comments become spaces (newlines preserved) so offsets and line numbers stay valid. */
export function stripComments(src: string): string {
  let out = '';
  for (const t of tokenize(src)) {
    out += t.kind === 'comment' ? t.text.replace(/[^\n]/g, ' ') : t.text;
  }
  return out;
}

/* -------------------------------------------------------------------------- */
/* Library preparation                                                         */
/* -------------------------------------------------------------------------- */

/**
 * Helper *definitions* the rewriter lowers away.
 *
 * `rewrite.ts` expands `mat2_mul_`/`mat2_rot_` at their **call sites** — it is a token
 * transform and cannot rewrite a definition's signature. Left in place, the GLSL
 * definitions of these two would be mangled into nonsense in every target:
 *
 *     vec2 mat2_mul_(mat2 m, vec2 v) { … }   →  `vec2 (mat2 m * vec2 v) { … }`
 *     mat2 mat2_rot_(float a) { … }          →  `mat2 mat2(float a) { … }`
 *
 * so they are removed here. `mat2_rot_` call sites are additionally redirected to the
 * portable `rot2()` helper (see `PRE_REWRITE_FIXUPS`): the rewriter maps *call sites*
 * to the target's matrix **constructor**, and `mat2(a)` is a diagonal scale, not a
 * rotation — silently wrong code in all six targets. `mat2_mul_` call sites are inlined
 * correctly (`mul(m, v)` on HLSL, `(m * v)` elsewhere).
 */
const LOWERED_HELPER_DEFS = ['mat2_mul_', 'mat2_rot_'] as const;

/** Call-site redirects applied to library *and* node bodies before rewriting. */
const PRE_REWRITE_FIXUPS: Readonly<Record<string, string>> = { mat2_rot_: 'rot2' };

function removeFunctionDefinitions(src: string, names: readonly string[]): string {
  let out = src;
  for (const name of names) {
    const head = new RegExp(
      `^[ \\t]*(?:float|int|bool|vec2|vec3|vec4|mat2|mat3)[ \\t]+${name}[ \\t]*\\(`,
      'm',
    );
    const m = head.exec(out);
    if (!m) continue;
    const open = out.indexOf('{', m.index);
    if (open === -1) continue;
    let depth = 0;
    let close = -1;
    for (let i = open; i < out.length; i++) {
      if (out[i] === '{') depth++;
      else if (out[i] === '}') {
        depth--;
        if (depth === 0) {
          close = i;
          break;
        }
      }
    }
    if (close === -1) continue;
    const note = '// (one matrix helper definition was lowered per target here; see targets/README.md)';
    out = `${out.slice(0, m.index)}${note}${out.slice(close + 1)}`;
  }
  return out;
}

/** `OCTAVE_LIMIT` → the tier's literal, then the definitions the rewriter owns are dropped. */
export function libraryGlsl(doc: ShaderDoc): string {
  const limited = substituteIdentifiers(GLSL_LIBRARY, { OCTAVE_LIMIT: String(octavesFor(doc)) });
  return substituteIdentifiers(
    removeFunctionDefinitions(limited, LOWERED_HELPER_DEFS),
    PRE_REWRITE_FIXUPS,
  );
}

/** Node bodies, with the call-site fixups the rewriter cannot make for them. */
export function bodiesGlsl(args: EmitterArgs): string {
  return substituteIdentifiers(args.bodies.lines.join('\n'), PRE_REWRITE_FIXUPS);
}

/* -------------------------------------------------------------------------- */
/* Texture fetch lowering                                                      */
/* -------------------------------------------------------------------------- */

/**
 * `tex2D_` is a *one-argument* alias (the UV), because the restricted subset has no
 * samplers. Each target needs the real call shape, so the fetch is expanded here with
 * the texture/sampler this emitter declares.
 */
type SampleStyle = { fetch: string; build: (uv: string) => string };

const SAMPLE: Record<TargetFormat, SampleStyle> = {
  glsl: { fetch: 'texture', build: (uv) => `texture(iChannel0, ${uv})` },
  wgsl: { fetch: 'textureSample', build: (uv) => `textureSample(iChannel0, iChannel0Sampler, ${uv})` },
  hlsl: { fetch: 'tex2D', build: (uv) => `iChannel0.Sample(iChannel0Sampler, ${uv})` },
  msl: { fetch: 'sample', build: (uv) => `iChannel0.sample(iChannel0Sampler, ${uv})` },
  shaderlab: {
    fetch: 'tex2D',
    build: (uv) => `tex2D(_Channel0, ${uv} * _Channel0_ST.xy + _Channel0_ST.zw)`,
  },
  ue: { fetch: 'tex2D', build: (uv) => `Texture2DSample(iChannel0, iChannel0Sampler, ${uv})` },
};

export function expandFetchCalls(src: string, style: SampleStyle): string {
  const toks = tokenize(src);
  const out: string[] = [];
  for (let i = 0; i < toks.length; i++) {
    const t = toks[i]!;
    if (t.kind === 'ident' && t.text === style.fetch && !isMemberAccess(toks, i)) {
      let j = i + 1;
      while (j < toks.length && (toks[j]!.kind === 'space' || toks[j]!.kind === 'comment')) j++;
      if (toks[j]?.text === '(') {
        let depth = 0;
        let k = j;
        for (; k < toks.length; k++) {
          if (toks[k]!.text === '(') depth++;
          else if (toks[k]!.text === ')') {
            depth--;
            if (depth === 0) break;
          }
        }
        if (depth === 0) {
          const uv = toks
            .slice(j + 1, k)
            .map((x) => x.text)
            .join('')
            .trim();
          out.push(style.build(uv));
          i = k;
          continue;
        }
      }
    }
    out.push(t.text);
  }
  return out.join('');
}

/* -------------------------------------------------------------------------- */
/* Host symbols + uniforms → target identifiers                                */
/* -------------------------------------------------------------------------- */

/**
 * How each target sees the host symbols of `HOST_SYMBOLS` plus every IR uniform.
 *
 * The GLSL target is the only one where the host symbols keep their canonical names;
 * everything else either namespaces them (WGSL/MSL put them in the uniform struct),
 * redirects them at the engine (Unity, UE) or leaves them alone in a cbuffer (HLSL).
 */
function identifierMap(target: TargetFormat, uniforms: UniformDecl[]): Record<string, string> {
  const m: Record<string, string> = {};
  switch (target) {
    case 'glsl':
      // Host symbols are declared with their canonical names; uniforms likewise.
      break;
    case 'wgsl': {
      m.iResolution = 'U.iResolution';
      m.iTime = 'U.iTime';
      m.iTimeDelta = 'U.iTimeDelta';
      // `iFrame` is `i32` in the struct; the library never touches it, but a node may
      // use it as a float, so the *placeholder expansion* produces the conversion.
      m.iFrame = 'f32(U.iFrame)';
      m.iMouse = 'U.iMouse';
      // Surface symbols. `iNormal` / `iWorldPos` / `iViewDir` are `let` locals in the entry
      // point, not struct members — they are per-fragment, and a uniform cannot be. Only the
      // two the host owns per frame live in the struct.
      m.iCameraPos = 'U.iCameraPos';
      m.iLightDir = 'U.iLightDir';
      for (const u of uniforms) m[u.name] = `U.${u.name}`;
      break;
    }
    case 'hlsl':
      // cbuffer members are globals named exactly like the host symbols.
      break;
    case 'msl': {
      m.iResolution = 'U.iResolution';
      m.iTime = 'U.iTime';
      m.iTimeDelta = 'U.iTimeDelta';
      m.iFrame = 'float(U.iFrame)';
      m.iMouse = 'U.iMouse';
      m.iCameraPos = 'U.iCameraPos';
      m.iLightDir = 'U.iLightDir';
      for (const u of uniforms) m[u.name] = `U.${u.name}`;
      break;
    }
    case 'shaderlab': {
      // Unity has no global resolution/time/mouse/texel inputs of its own shape.
      m.iResolution = '_ScreenParams.xy';
      m.iTime = '_Time.y';
      m.iTimeDelta = 'unity_DeltaTime.x';
      m.iFrame = 'floor(_Time.w * 20.0)'; // _Time.w == t*3, so *20 gives a 60 fps counter
      m.iMouse = '_MousePos';
      m.iChannel0 = '_Channel0';
      // Surface symbols. `iNormal` / `iWorldPos` / `iViewDir` are locals the entry point
      // builds from `v2f`; only the two Unity already keeps per frame are remapped here.
      // `_WorldSpaceLightPos0` is the direction *towards* the light for a directional light,
      // which is exactly what the IR means by `iLightDir`.
      m.iCameraPos = '_WorldSpaceCameraPos';
      m.iLightDir = '_WorldSpaceLightPos0.xyz';
      for (const u of uniforms) m[u.name] = `_${u.name}`;
      break;
    }
    case 'ue': {
      m.iResolution = 'View.ViewSizeAndInvSize.xy';
      m.iTime = 'View.GameTime';
      m.iTimeDelta = 'View.DeltaTime';
      m.iFrame = 'float(View.FrameNumber)';
      m.iMouse = 'View.MousePosition';
      // Surface symbols are Custom-node input pins, exactly like the IR uniforms: the material
      // graph computes them (`VertexNormalWS`, `AbsoluteWorldPosition`, `CameraPositionWS`,
      // and a light vector node), so there is nothing to remap and no global to declare.
      // IR uniforms are Custom-node input pins, i.e. function parameters: no rename.
      break;
    }
  }
  return m;
}

/* -------------------------------------------------------------------------- */
/* Output colour                                                               */
/* -------------------------------------------------------------------------- */

const PARAM_KIND_TYPE: Record<string, ShaderType> = {
  float: 'float', int: 'int', bool: 'bool', vec2: 'vec2', vec3: 'vec3', vec4: 'vec4', color: 'vec4',
};

function paramTypeOf(v: ParamValue, fallback: ShaderType): ShaderType {
  return v.kind === 'param' ? v.type : PARAM_KIND_TYPE[v.kind] ?? fallback;
}

/** The `output` node's colour input, resolved to a GLSL expression plus its IR type. */
function rawOutputColor(
  args: EmitterArgs,
  diagnostics: Diagnostic[],
): { glsl: string; type: ShaderType } {
  const fallback = { glsl: 'vec4(0.0, 0.0, 0.0, 1.0)', type: 'vec4' as ShaderType };
  const outId = args.analysis.outputNode;
  if (!outId) {
    diagnostics.push({
      severity: 'error',
      code: 'EM001_NO_OUTPUT_NODE',
      message: 'Graph has no reachable `output` node; emitted a flat colour instead.',
      fix: 'Add an `output` node and wire its `color` input.',
    });
    return fallback;
  }
  const node = args.analysis.byId.get(outId)!;
  const schema: NodeSchema | undefined = args.catalog[node.kind];
  const port = schema?.inputs.find((p) => p.name === 'color') ?? schema?.inputs[0];
  if (!port) {
    diagnostics.push({
      severity: 'error',
      code: 'EM002_OUTPUT_HAS_NO_COLOR',
      message: `The \`output\` node schema (\`${node.kind}\`) declares no \`color\` input.`,
      fix: 'Give the output schema an input named `color`.',
    });
    return fallback;
  }

  const wire = node.inputs?.[port.name];
  if (wire && args.analysis.byId.has(wire.node)) {
    const src = args.analysis.byId.get(wire.node)!;
    const srcSchema = args.catalog[src.kind];
    const outName = wire.port ?? srcSchema.primaryOut ?? srcSchema.outputs[0]!.name;
    const outType = srcSchema.outputs.find((p) => p.name === outName)?.type ?? port.type;
    return { glsl: args.bodies.localName(wire.node, outName), type: outType };
  }

  const lit = node.params?.[port.name] ?? port.default;
  if (lit) return { glsl: paramExpr(lit), type: paramTypeOf(lit, port.type) };

  diagnostics.push({
    severity: 'error',
    code: 'EM003_OUTPUT_UNWIRED',
    message: `The \`output\` node's \`${port.name}\` input is neither wired nor given a literal.`,
    fix: `Wire a node into \`${outId}.${port.name}\` or set \`params.${port.name}\`.`,
  });
  return fallback;
}

/**
 * Widen whatever the graph produced into the `vec4` the epilogue writes out.
 * `render.transparent === false` is the IR's only statement about alpha, so it is
 * applied here: an opaque doc gets alpha forced to 1.0 even when the colour input is a
 * full vec4.
 */
function widenToVec4(
  expr: string,
  type: ShaderType,
  transparent: boolean,
  diagnostics: Diagnostic[],
): string {
  switch (type) {
    case 'vec4':
      return transparent ? expr : `vec4(${expr}.xyz, 1.0)`;
    case 'vec3':
      return `vec4(${expr}, 1.0)`;
    case 'vec2':
      return `vec4(${expr}, 0.0, 1.0)`;
    case 'float':
      return `vec4(${expr}, ${expr}, ${expr}, 1.0)`;
    case 'int':
    case 'bool':
      return `vec4(float(${expr}))`;
    default:
      diagnostics.push({
        severity: 'error',
        code: 'EM004_OUTPUT_COLOR_ARITY',
        message: `The output colour is a ${type}; there is no sensible vec4 widening for it.`,
        fix: 'Feed the output node a float/vec2/vec3/vec4 expression (a `color_ramp` or `mix` node).',
      });
      return 'vec4(1.0, 0.0, 1.0, 1.0)';
  }
}

/** Output colour rewritten into `target`, with the target's identifier mapping applied. */
function outputColor(args: EmitterArgs, target: TargetFormat, diagnostics: Diagnostic[]): string {
  const raw = rawOutputColor(args, diagnostics);
  const widened = widenToVec4(raw.glsl, raw.type, args.doc.render.transparent, diagnostics);
  const rewritten = args.rw(widened, `${target}: output colour`).source;
  return substituteIdentifiers(rewritten, identifierMap(target, args.uniforms));
}

/* -------------------------------------------------------------------------- */
/* Assembly                                                                    */
/* -------------------------------------------------------------------------- */

/** Join non-empty chunks with single newlines, guarantee one trailing newline. */
export function assemble(parts: (string | string[] | undefined | false)[]): string {
  const out: string[] = [];
  for (const p of parts) {
    if (p === undefined || p === false) continue;
    const s = Array.isArray(p) ? p.join('\n') : p;
    if (s.length === 0) continue;
    out.push(s.replace(/^\n+/, '').replace(/\s+$/, ''));
  }
  return `${out.join('\n\n')}\n`;
}

/**
 * Indent a lowered chunk and return it as lines, for embedding **inside a function body**.
 *
 * This exists because of where the node bodies belong. `emitBodies` produces straight-line
 * statements that read `uv` and `fragCoord`, which every target's entry point creates as
 * locals — so the bodies are part of the entry point, not module scope. Emitting them at
 * top level produces global initialisers that reference an undefined `uv`, which no
 * compiler on any of the six targets accepts.
 *
 * Blank lines are dropped so the generated body stays compact.
 */
function embed(chunk: string, indent = '  '): string[] {
  const text = Array.isArray(chunk) ? chunk.join('\n') : chunk;
  return text
    .replace(/\s+$/, '')
    .split('\n')
    .filter((l) => l.trim().length > 0)
    .map((l) => indent + l.replace(/^\s+/, ''));
}

/** Rewrite one chunk of restricted GLSL and remap its identifiers for `target`. */
function lower(
  args: EmitterArgs,
  target: TargetFormat,
  src: string,
  context: string,
  diagnostics?: Diagnostic[],
): string {
  const first = args.rw(src, context); // lint + rewrite; diagnostics are reported here, once
  let text = first.source;

  // --- fixpoint -----------------------------------------------------------------
  // `rewrite.ts`'s `lowerMatrixMul` inlines a helper call by splicing the *verbatim*
  // token text of its arguments, which is not itself run through the rename tables.
  // A perfectly ordinary node body such as the catalog's `transform`
  //
  //     mat2_mul_(rot2(a), (uv - vec2(0.5, 0.5)) * vec2(2.0, 2.0))
  //
  // therefore leaves `vec2` inside the inlined `mul(...)`/`(... * ...)` on HLSL, MSL and
  // WGSL. Re-running the (idempotent) rewriter reaches a fixpoint: target names are
  // never source names, so a second pass can only pick up text the first pass produced.
  // Re-running through `rewriteSource` rather than `args.rw` keeps the lint diagnostics
  // from being reported twice; only new *errors* are surfaced.
  for (let pass = 0; pass < 3; pass++) {
    const again = rewriteSource(text, target, `${context}: fixpoint`);
    if (again.source === text) break;
    text = again.source;
    if (diagnostics) {
      for (const d of again.diagnostics) if (d.severity === 'error') diagnostics.push(d);
    }
  }

  const withFetches = expandFetchCalls(text, SAMPLE[target]);
  return substituteIdentifiers(withFetches, identifierMap(target, args.uniforms));
}

/** The canonical uniform line for struct/cbuffer style targets. */
function memberLine(target: TargetFormat, u: UniformDecl, indent: string): string {
  const type = target === 'ue' ? UE_BODY_TYPE[u.type] : typeName(target, u.type);
  return `${indent}${type} ${u.name}; // default ${targetLiteral(target, u.type, u.value)}`;
}

function isGlslTypeName(s: string): boolean {
  return (TYPE_TOKENS as readonly string[]).includes(s);
}

function isAliasName(s: string): boolean {
  return (REWRITE_ALIASES as readonly string[]).includes(s);
}

/* ========================================================================== */
/* TARGET 1 — GLSL ES 3.00 (WebGL2)                                           */
/* ========================================================================== */

/**
 * The host requirement `render.feedback` creates, as header lines (empty when it is not set).
 *
 * A feedback document is *emittable everywhere* — its source is identical to one that samples a
 * static image — and that is exactly why the requirement has to be written down. If the host binds
 * anything other than the previous frame, the shader compiles, runs at full speed, and produces a
 * still image: no error, no warning, no clue. Every target's header already carries its
 * uncertainties; this is the same discipline applied to an integration step the emitter cannot do
 * on the user's behalf.
 */
function feedbackNotes(doc: ShaderDoc, comment = '//'): string[] {
  const transparency = doc.render.transparent === true ? [
    `${comment} TRANSPARENCY: output is straight (unpremultiplied) RGBA; alpha alone does not enable blending.`,
    `${comment} Host: enable blend ADD, color SRC_ALPHA / ONE_MINUS_SRC_ALPHA; alpha ONE / ONE_MINUS_SRC_ALPHA.`,
    `${comment} Keep depth testing, disable depth writes, draw opaque first and sort transparent back-to-front.`,
    `${comment} WebGL: blendFuncSeparate(SRC_ALPHA, ONE_MINUS_SRC_ALPHA, ONE, ONE_MINUS_SRC_ALPHA), depthMask(false).`,
    `${comment} WGSL/WebGPU: configure fragment.targets[].blend and depthStencil.depthWriteEnabled=false.`,
    `${comment} HLSL/D3D: configure render-target BlendState and DepthWriteMask=ZERO.`,
    `${comment} MSL/Metal: colorAttachments[].blendingEnabled=true, sourceRGBBlendFactor=sourceAlpha,`,
    `${comment} destinationRGBBlendFactor=oneMinusSourceAlpha, sourceAlphaBlendFactor=one,`,
    `${comment} destinationAlphaBlendFactor=oneMinusSourceAlpha; depthWriteEnabled=false.`,
    `${comment} UE: Material Blend Mode=Translucent; connect Custom RGB to color and A to Opacity.`,
    `${comment} ShaderLab: transparent Queue/RenderType, Blend and ZWrite Off are emitted below.`,
    `${comment} Feedback/state passes must overwrite their target with blending OFF; blend only the final presentation.`,
  ] : [];
  if (doc.render.feedback !== true) return transparency;
  return [...transparency,
    `${comment} FEEDBACK — this document reads the image channel as its **previous frame**`,
    `${comment} (render.feedback = true). The host must ping-pong: sample the last frame while`,
    `${comment} rendering into a second target, then swap them. Binding a static image here`,
    `${comment} compiles and runs and produces a still image, with nothing to point at.`,
  ];
}

/**
 * The only target that is expected to compile and run as-is in the browser.
 *
 * Y flip: none. GLSL's `gl_FragCoord` already has its origin at the bottom-left, which
 * is the IR's convention, so `uv = fragCoord / iResolution.xy` needs no correction.
 *
 * Header contract (consumed by `apps/web`): the first three lines are *exactly*
 *
 *     #version 300 es
 *     precision highp float;
 *     precision highp int;
 *
 * with nothing before them — no comment, no blank line — and line 4 is blank. A host
 * that supplies its own vertex shader can therefore split with
 * `src.split('\n').slice(0, 3)` and keep the remainder as the fragment body.
 */
const glslEmitter: TargetEmitter = (args) => {
  const diagnostics: Diagnostic[] = [];
  const surface = args.analysis.needs.surface;
  const lib = lower(args, 'glsl', libraryGlsl(args.doc), 'glsl: shared library', diagnostics);
  const bodies = embed(lower(args, 'glsl', bodiesGlsl(args), 'glsl: node bodies', diagnostics));
  const color = outputColor(args, 'glsl', diagnostics);

  const uniformLines = args.uniforms.map(
    (u) => `uniform ${typeName('glsl', u.type)} ${u.name}; // default ${literalOf(u.type, u.value)}`,
  );

  /*
   * Two entry points, one document each.
   *
   * A quad document is a picture: `uv` from `gl_FragCoord`. A **surface** document is a
   * shading function, and there is no `gl_FragCoord` on a mesh that could stand in for a
   * normal — so the source changes shape, the host has to supply a vertex stage, and `uv`
   * comes from the mesh instead of the screen.
   *
   * Only the entry point and the header differ. The library, the node bodies and the uniform
   * block are untouched, which is what keeps the quad path byte-identical.
   */
  const header = surface
    ? [
        '// AI4Shader — GLSL ES 3.00 **surface** fragment shader (WebGL2).',
        '// Uniform declaration order follows the IR `uniforms[]` array exactly.',
        ...feedbackNotes(args.doc),
        '//',
        '// VERTEX CONTRACT — the host must supply a vertex stage that writes:',
        '//   out vec3 vWorldPos;  // world-space fragment position',
        '//   out vec3 vNormal;    // world-space normal (normalised below)',
        '//   out vec2 vUv;        // mesh UV, IR orientation (origin bottom-left)',
        '// and set `iCameraPos` (world) and `iLightDir` (unit, surface -> light) as uniforms.',
        '// This file is the fragment stage only: compiling it against a fullscreen quad would',
        '// fail, because there is no normal on a quad.',
        '',
        'uniform vec3 iResolution;',
        'uniform float iTime;',
        'uniform float iTimeDelta;',
        'uniform int iFrame;',
        'uniform vec4 iMouse;',
        ...uniformLines,
        'uniform sampler2D iChannel0;',
        '// --- surface uniforms, set by the host per frame ---',
        'uniform vec3 iCameraPos; // world-space camera position',
        'uniform vec3 iLightDir;  // unit vector, surface -> light',
        '// --- interpolated vertex outputs ---',
        'in vec3 vWorldPos;',
        'in vec3 vNormal;',
        'in vec2 vUv;',
        '',
        'out vec4 fragColor;',
      ]
    : [
        '// AI4Shader — GLSL ES 3.00 fragment shader (WebGL2).',
        '// Uniform declaration order follows the IR `uniforms[]` array exactly.',
        ...feedbackNotes(args.doc),
        '',
        'uniform vec3 iResolution;',
        'uniform float iTime;',
        'uniform float iTimeDelta;',
        'uniform int iFrame;',
        'uniform vec4 iMouse;',
        ...uniformLines,
        'uniform sampler2D iChannel0;',
        '',
        'out vec4 fragColor;',
      ];

  const entry = surface
    ? [
        'void main() {',
        '  // GLSL UV and the IR both have a bottom-left origin: this target flips Y once, and',
        '  // only where the source is a texture (HLSL/MSL/ShaderLab), never here.',
        '  vec2 uv = vUv;',
        '  // `fragCoord` stays defined so a node that reads it still has a value, even though a',
        '  // surface shader has no screen position: it is the UV expressed back in pixels.',
        '  vec2 fragCoord = uv * iResolution.xy;',
        '  // ---- surface inputs ----',
        '  vec3 iWorldPos = vWorldPos;',
        '  vec3 iNormal = normalize(vNormal);',
        '  vec3 iViewDir = normalize(iCameraPos - vWorldPos);',
        '  // ---- graph (node bodies; straight-line, in topological order) ----',
        ...bodies.map((l) => `  ${l}`),
        '',
        `  fragColor = ${color};`,
        '}',
      ]
    : [
        'void main() {',
        '  // GLSL already has a bottom-left origin: this target never flips Y.',
        '  vec2 fragCoord = gl_FragCoord.xy;',
        '  vec2 uv = fragCoord / iResolution.xy;',
        '  // ---- graph (node bodies; straight-line, in topological order) ----',
        ...bodies.map((l) => `  ${l}`),
        '',
        `  fragColor = ${color};`,
        '}',
      ];

  const source = assemble([
    // --- header: fixed, contiguous, always first -----------------------------
    ['#version 300 es', 'precision highp float;', 'precision highp int;'],
    header,
    '// ---- shared helper library (restricted GLSL subset) ----',
    shimsFor('glsl'),
    lib,
    '// ---- entry point ----',
    entry,
  ]);

  return { source, diagnostics };
};

/* ========================================================================== */
/* TARGET 2 — WGSL (WebGPU)                                                   */
/* ========================================================================== */

/**
 * Lower one restricted-GLSL chunk for WGSL: rename, then restructure.
 *
 * The split of responsibility is deliberate. `args.rw` (+ the same fixpoint loop the
 * other targets use) owns every *token* rename and every lint diagnostic; `wgslLower`
 * owns every *structural* difference — signatures, `var` declarations, loop headers,
 * immutable parameters, and the scalar bounds GLSL lets you pass to a vector builtin —
 * and reports a `TGT0xx` error for anything it cannot lower.
 *
 * The texture fetch and the host/uniform identifier mapping stay here, after lowering,
 * because both are text-to-text rewrites of already-valid WGSL.
 */
function lowerWgsl(
  args: EmitterArgs,
  src: string,
  context: string,
  diagnostics: Diagnostic[],
  signatures?: Readonly<Record<string, WgslShape>>,
): { source: string; signatures: Record<string, WgslShape> } {
  // Shapes of everything the chunk cannot declare itself: the host symbols (whose struct
  // member types this emitter owns) and the IR uniforms. They let `wgslLower` prove an
  // argument's arity before broadcasting a scalar bound (GLSL allows `clamp(v3, 0.0, 1.0)`,
  // WGSL does not — see `wgslLower.ts` note 5).
  const symbols: Record<string, WgslShape> = {
    iResolution: 3,
    iTime: 1,
    iTimeDelta: 1,
    iFrame: 1,
    iMouse: 4,
  };
  for (const u of args.uniforms) {
    if (u.type === 'mat2' || u.type === 'mat3') continue;
    symbols[u.name] = TYPE_ARITY[u.type] as WgslShape;
  }

  const lowered = lowerSubsetToWgsl(src, {
    context,
    immutableNames: ['uv', ...HOST_SYMBOLS, ...args.uniforms.map((u) => u.name)],
    symbols,
    ...(signatures ? { signatures } : {}),
    rewrite: (text) => {
      const first = args.rw(text, context); // lint + rename; diagnostics are reported here, once
      let out = first.source;
      // Same fixpoint as `lower()`: `lowerMatrixMul` splices its arguments verbatim, so a
      // `vec2(0.5, 0.5)` inside `mat2_mul_(…)` survives the first pass un-renamed.
      for (let pass = 0; pass < 3; pass++) {
        const again = rewriteSource(out, 'wgsl', `${context}: fixpoint`);
        if (again.source === out) break;
        out = again.source;
        for (const d of again.diagnostics) if (d.severity === 'error') diagnostics.push(d);
      }
      return out;
    },
  });
  diagnostics.push(...lowered.diagnostics);

  const withFetches = expandFetchCalls(lowered.source, SAMPLE.wgsl);
  return {
    source: substituteIdentifiers(withFetches, identifierMap('wgsl', args.uniforms)),
    signatures: lowered.signatures,
  };
}

/**
 * Y flip: **here, and nowhere else in any target**. WebGPU's framebuffer origin is
 * top-left while the IR is bottom-left, so `fs_main` mirrors `pos.y` exactly once.
 *
 * `iFrame` is `i32` in `Uniforms`; the placeholder expansion turns it into
 * `f32(U.iFrame)` wherever a node uses it in float position.
 */
const wgslEmitter: TargetEmitter = (args) => {
  const diagnostics: Diagnostic[] = [];
  const surface = args.analysis.needs.surface;
  // The library is lowered first: its `signatures` tell the body pass the return shape of
  // every helper the node bodies call (needed to prove an arity before a scalar broadcast).
  const lib = lowerWgsl(args, libraryGlsl(args.doc), 'wgsl: shared library', diagnostics);
  const bodies = embed(
    lowerWgsl(args, bodiesGlsl(args), 'wgsl: node bodies', diagnostics, lib.signatures).source,
  );
  const color = outputColor(args, 'wgsl', diagnostics);
  const shims = shimsFor('wgsl');

  const members = [
    'iResolution: vec3f',
    'iTime: f32',
    'iTimeDelta: f32',
    'iFrame: i32',
    'iMouse: vec4f',
    ...args.uniforms.map((u) => `${u.name}: ${typeName('wgsl', u.type)}`),
    // Per-fragment host data, supplied each frame. `vec3f` is 16-byte aligned in the uniform
    // address space, so the workbench's layout reflector reads these offsets back out of the
    // struct rather than assuming they follow the previous member.
    ...(surface ? ['iCameraPos: vec3f', 'iLightDir: vec3f'] : []),
  ];

  const source = assemble([
    [
      surface ? '// AI4Shader — WebGPU **surface** shader pair (vertex + fragment).' : '// AI4Shader — WebGPU fragment shader.',
      '// Uniform declaration order follows the IR `uniforms[]` array exactly.',
      ...feedbackNotes(args.doc),
      '',
      `struct Uniforms { ${members.join(', ')} };`,
      '@group(0) @binding(0) var<uniform> U: Uniforms;',
      '@group(0) @binding(1) var iChannel0: texture_2d<f32>;',
      '@group(0) @binding(2) var iChannel0Sampler: sampler;',
      ...(surface
        ? [
            '',
            '// Host transform block, bound at binding 3. NOT part of `uniforms[]`: an MVP matrix',
            '// is a host concern, not a document tunable.',
            'struct Transform { world: mat4x4f, worldViewProj: mat4x4f };',
            '@group(0) @binding(3) var<uniform> X: Transform;',
            '',
            '// Vertex outputs, interpolated across the triangle.',
            'struct VSOut {',
            '  @builtin(position) position: vec4f,',
            '  @location(0) uv: vec2f,',
            '  @location(1) worldPos: vec3f,',
            '  @location(2) normal: vec3f,',
            '};',
          ]
        : []),
    ],
    '// ---- prelude shims (WGSL has no scalar mix/mod/atan2 of the GLSL shape) ----',
    shims,
    '// ---- shared helper library ----',
    lib.source,
    '// ---- entry point ----',
    surface
      ? [
          '@vertex',
          'fn vs_main(',
          '  @location(0) position: vec3f,',
          '  @location(1) normal: vec3f,',
          '  @location(2) uv: vec2f,',
          ') -> VSOut {',
          '  var out: VSOut;',
          '  let world = X.world * vec4f(position, 1.0);',
          '  out.worldPos = world.xyz;',
          '  // The upper 3x3 of a rigid transform is its own inverse transpose, so this is exact',
          '  // for rotation and translation and only approximate under non-uniform scale.',
          '  out.normal = (X.world * vec4f(normal, 0.0)).xyz;',
          '  out.uv = uv;',
          '  out.position = X.worldViewProj * vec4f(position, 1.0);',
          '  return out;',
          '}',
          '',
          '@fragment',
          'fn fs_main(in: VSOut) -> @location(0) vec4f {',
          '  // THE Y FLIP FOR THIS TARGET (exactly one). WebGPU texture V is top-left; the IR is',
          '  // bottom-left. No node body and no library function ever flips.',
          '  let uv = vec2f(in.uv.x, 1.0 - in.uv.y);',
          '  // A surface shader has no screen position; `fragCoord` is the mesh uv back in pixels',
          '  // so a node that reads it still has a value.',
          '  let fragCoord = uv * U.iResolution.xy;',
          '  let iWorldPos = in.worldPos;',
          '  let iNormal = normalize(in.normal);',
          '  let iViewDir = normalize(U.iCameraPos - in.worldPos);',
          '  // ---- graph (node bodies; straight-line, in topological order) ----',
          ...bodies.map((l) => `  ${l}`),
          '',
          `  return ${color};`,
          '}',
        ]
      : [
          '@fragment',
          'fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f {',
          '  // THE Y FLIP FOR THIS TARGET. WebGPU pixel origin is top-left; the IR is',
          '  // bottom-left. No node body and no library function ever flips.',
          '  let uv = vec2f(pos.x / U.iResolution.x, 1.0 - pos.y / U.iResolution.y);',
          '  // ---- graph (node bodies; straight-line, in topological order) ----',
          ...bodies.map((l) => `  ${l}`),
          '',
          `  return ${color};`,
          '}',
        ],
    [
      '// NOTE (see targets/README.md §WGSL): the helper library and the node bodies above',
      '// are lowered by `wgslLower.ts`, which turns the restricted-GLSL subset into WGSL',
      '// structure (`fn … -> T`, `var`, `for (var i: i32 = 0; …; i = i + 1)`) and copies',
      '// any parameter a body writes to into a `var`, since WGSL parameters are immutable.',
      '// A construct the pass cannot lower is reported as a TGT0xx error diagnostic rather',
      '// than emitted: a clean `diagnostics` array means nothing here was left unlowered.',
    ],
  ]);

  return { source, diagnostics };
};

/* ========================================================================== */
/* TARGET 3 — HLSL (D3D11/12 pixel + vertex shader pair)                      */
/* ========================================================================== */

/**
 * Y flip: once, in `PSMain`. `VSMain` emits a top-left-origin screen uv (the D3D
 * convention for `SV_POSITION`), so the pixel shader's single `1.0 - uv.y` restores
 * the IR's bottom-left orientation.
 *
 * ## The surface variant
 *
 * A surface document cannot keep the fullscreen triangle: there is no normal on a triangle
 * that covers the screen. So `VSMain` becomes a **mesh vertex shader** — it takes POSITION /
 * NORMAL / TEXCOORD0 and a `AI4ShaderTransform` cbuffer, and passes world position, world
 * normal and mesh uv through to the pixel stage.
 *
 * The transform block is the emitter's own, not `uniforms[]`: those are the document's
 * *tunables*, and a model-view-projection matrix is not something a user drags a slider for.
 * Keeping them in separate cbuffers also keeps the document's binding order stable, which the
 * sidecar asserts.
 */
const hlslEmitter: TargetEmitter = (args) => {
  const diagnostics: Diagnostic[] = [];
  const surface = args.analysis.needs.surface;
  const lib = lower(args, 'hlsl', libraryGlsl(args.doc), 'hlsl: shared library', diagnostics);
  const bodies = embed(lower(args, 'hlsl', bodiesGlsl(args), 'hlsl: node bodies', diagnostics));
  const color = outputColor(args, 'hlsl', diagnostics);

  const source = assemble([
    surface
      ? [
          '// AI4Shader — HLSL (Shader Model 5) **surface** shader pair.',
          '// VSMain transforms a mesh and passes world position / world normal / uv through;',
          '// PSMain evaluates the graph against them.',
          '// Uniform declaration order follows the IR `uniforms[]` array exactly.',
          ...feedbackNotes(args.doc),
          '',
          'cbuffer AI4ShaderParams : register(b0) {',
          '  float3 iResolution; float iTime; float iTimeDelta; int iFrame; float4 iMouse;',
          ...args.uniforms.map((u) => memberLine('hlsl', u, '  ')),
          '  float3 iCameraPos; float3 iLightDir;',
          '};',
          '// Host transform block. Not part of `uniforms[]`: an MVP matrix is not a document tunable.',
          'cbuffer AI4ShaderTransform : register(b1) {',
          '  float4x4 ai4sWorld;      // object -> world',
          '  float4x4 ai4sWorldViewProj;',
          '};',
          'Texture2D iChannel0 : register(t0);',
          'SamplerState iChannel0Sampler : register(s0);',
          'struct PSInput {',
          '  float4 position : SV_POSITION;',
          '  float2 uv : TEXCOORD0;',
          '  float3 worldPos : TEXCOORD1;',
          '  float3 normal : TEXCOORD2;',
          '};',
          'struct VSOutput {',
          '  float4 position : SV_POSITION;',
          '  float2 uv : TEXCOORD0;',
          '  float3 worldPos : TEXCOORD1;',
          '  float3 normal : TEXCOORD2;',
          '};',
        ]
      : [
          '// AI4Shader — HLSL (Shader Model 5). Compiles standalone: VSMain draws the',
          '// fullscreen triangle, PSMain evaluates the graph.',
          '// Uniform declaration order follows the IR `uniforms[]` array exactly.',
          ...feedbackNotes(args.doc),
          '',
          'cbuffer AI4ShaderParams : register(b0) {',
          '  float3 iResolution; float iTime; float iTimeDelta; int iFrame; float4 iMouse;',
          ...args.uniforms.map((u) => memberLine('hlsl', u, '  ')),
          '};',
          'Texture2D iChannel0 : register(t0);',
          'SamplerState iChannel0Sampler : register(s0);',
          'struct PSInput { float4 position : SV_POSITION; float2 uv : TEXCOORD0; };',
          'struct VSOutput { float4 position : SV_POSITION; float2 uv : TEXCOORD0; };',
        ],
    '// ---- shared helper library ----',
    shimsFor('hlsl'),
    lib,
    '// ---- entry points ----',
    surface
      ? [
          'VSOutput VSMain(float3 position : POSITION, float3 normal : NORMAL, float2 uv : TEXCOORD0) {',
          '  VSOutput o;',
          '  float4 world = mul(ai4sWorld, float4(position, 1.0));',
          '  o.worldPos = world.xyz;',
          '  // The inverse transpose is the correct normal matrix under non-uniform scale; for a',
          '  // rigid transform the upper 3x3 is its own inverse transpose, which is why this is',
          '  // only ever approximately right if a host scales non-uniformly. Stated, not hidden.',
          '  o.normal = mul((float3x3)ai4sWorld, normal);',
          '  o.uv = uv;',
          '  o.position = mul(ai4sWorldViewProj, float4(position, 1.0));',
          '  return o;',
          '}',
          '',
          'float4 PSMain(PSInput input) : SV_Target {',
          '  // THE Y FLIP FOR THIS TARGET (exactly one): D3D texture V is top-left, the IR is',
          '  // bottom-left.',
          '  float2 uv = float2(input.uv.x, 1.0 - input.uv.y);',
          '  // A surface shader has no screen position; `fragCoord` is the mesh uv back in pixels',
          '  // so that a node reading it still has a value.',
          '  float2 fragCoord = uv * iResolution.xy;',
          '  float3 iWorldPos = input.worldPos;',
          '  float3 iNormal = normalize(input.normal);',
          '  float3 iViewDir = normalize(iCameraPos - input.worldPos);',
          '  // ---- graph (node bodies; straight-line, in topological order) ----',
          ...bodies.map((l) => `  ${l}`),
          '',
          `  return ${color};`,
          '}',
        ]
      : [
          'VSOutput VSMain(uint id : SV_VertexID) {',
          '  VSOutput o;',
          '  // Fullscreen triangle: ids 0,1,2 -> uv (0,0) (2,0) (0,2).',
          '  float2 corner = float2(float((id << 1) & 2), float(id & 2));',
          '  // The vertex uv is TOP-left origin (D3D screen space); PSMain flips it back so',
          '  // the single IR-mandated Y flip stays in the fragment epilogue.',
          '  o.uv = float2(corner.x, 1.0 - corner.y);',
          '  o.position = float4(corner * 2.0 - 1.0, 0.0, 1.0);',
          '  return o;',
          '}',
          '',
          'float4 PSMain(PSInput input) : SV_Target {',
          '  // THE Y FLIP FOR THIS TARGET (exactly one).',
          '  float2 uv = float2(input.uv.x, 1.0 - input.uv.y);',
          '  // ---- graph (node bodies; straight-line, in topological order) ----',
          ...bodies.map((l) => `  ${l}`),
          '',
          `  return ${color};`,
          '}',
        ],
  ]);

  const sidecars = [
    {
      name: 'hlsl-bindings.json',
      content: `${JSON.stringify(
        {
          target: 'hlsl',
          entryPoints: { vertex: 'VSMain', fragment: 'PSMain' },
          profile: { vertex: 'vs_5_0', fragment: 'ps_5_0' },
          /** `quad` = the fullscreen triangle; `surface` = a mesh vertex stage. */
          geometry: surface ? 'surface' : 'quad',
          cbuffers: [
            {
              name: 'AI4ShaderParams',
              register: 'b0',
              layoutRule: 'hlsl-default-packing',
              members: [
                { name: 'iResolution', type: 'float3' },
                { name: 'iTime', type: 'float' },
                { name: 'iTimeDelta', type: 'float' },
                { name: 'iFrame', type: 'int' },
                { name: 'iMouse', type: 'float4' },
                ...args.uniforms.map((u) => ({ name: u.name, type: typeName('hlsl', u.type) })),
                ...(surface
                  ? [
                      { name: 'iCameraPos', type: 'float3', note: 'world-space camera position' },
                      { name: 'iLightDir', type: 'float3', note: 'unit, surface -> light' },
                    ]
                  : []),
              ],
            },
            // Declared only for a surface document: a quad has no model matrix to apply.
            ...(surface
              ? [
                  {
                    name: 'AI4ShaderTransform',
                    register: 'b1',
                    layoutRule: 'hlsl-default-packing',
                    note: 'Supplied by the host, NOT by the IR `uniforms[]` array.',
                    members: [
                      { name: 'ai4sWorld', type: 'float4x4', note: 'object -> world' },
                      { name: 'ai4sWorldViewProj', type: 'float4x4' },
                    ],
                  },
                ]
              : []),
          ],
          vertexInputs: surface
            ? [
                { semantic: 'POSITION', type: 'float3' },
                { semantic: 'NORMAL', type: 'float3' },
                { semantic: 'TEXCOORD0', type: 'float2', note: 'mesh uv' },
              ]
            : [{ semantic: 'SV_VertexID', type: 'uint' }],
          textures: [{ name: 'iChannel0', type: 'Texture2D', register: 't0' }],
          samplers: [{ name: 'iChannel0Sampler', type: 'SamplerState', register: 's0' }],
          uniforms: args.uniforms.map((u) => ({
            name: u.name,
            type: u.type,
            value: u.value,
            ...(u.min !== undefined ? { min: u.min } : {}),
            ...(u.max !== undefined ? { max: u.max } : {}),
          })),
          notes: [
            'Byte offsets are intentionally omitted: the cbuffer relies on HLSL default packing rules. Use packoffset if a host needs an explicit layout.',
            ...(surface
              ? [
                  'SURFACE shader: VSMain is a mesh vertex shader (POSITION/NORMAL/TEXCOORD0 in) and needs AI4ShaderTransform bound at b1.',
                  'The normal matrix is the upper 3x3 of ai4sWorld, which is only correct for a rigid transform. Under non-uniform scale, pass the inverse transpose instead.',
                ]
              : [
                  'Quad shader: VSMain takes no vertex buffer — it derives the fullscreen triangle from SV_VertexID.',
                ]),
          ],
        },
        null,
        2,
      )}\n`,
    },
  ];

  return { source, sidecars, diagnostics };
};

/* ========================================================================== */
/* TARGET 4 — Metal Shading Language                                          */
/* ========================================================================== */

/**
 * Y flip: once, in `fs_main`. Metal's `[[position]]` fragment input is in pixels with
 * a top-left origin, exactly like WebGPU.
 *
 * The surface variant adds a `vs_main` that transforms a mesh, and a `VSOut` to carry world
 * position and normal across the stage boundary. Metal's fragment `[[position]]` is a *pixel*
 * coordinate, so unlike GLSL the quad form stays fragment-only and the host supplies the
 * triangle — the same split every other target uses.
 */
const mslEmitter: TargetEmitter = (args) => {
  const diagnostics: Diagnostic[] = [];
  const surface = args.analysis.needs.surface;
  const lib = lower(args, 'msl', libraryGlsl(args.doc), 'msl: shared library', diagnostics);
  const bodies = embed(lower(args, 'msl', bodiesGlsl(args), 'msl: node bodies', diagnostics));
  const color = outputColor(args, 'msl', diagnostics);

  const source = assemble([
    [
      surface
        ? '// AI4Shader — Metal Shading Language **surface** function pair.'
        : '// AI4Shader — Metal Shading Language fragment function.',
      '// Uniform declaration order follows the IR `uniforms[]` array exactly.',
      '// The IR is int/float strict on this target: every int→float use is an explicit',
      '// constructor call, in the library and in the node bodies alike.',
      ...feedbackNotes(args.doc),
      '',
      '#include <metal_stdlib>',
      'using namespace metal;',
      '',
      'struct AI4ShaderUniforms {',
      '  float3 iResolution; float iTime; float iTimeDelta; int iFrame; float4 iMouse;',
      ...args.uniforms.map((u) => memberLine('msl', u, '  ')),
      ...(surface ? ['  float3 iCameraPos; float3 iLightDir;'] : []),
      '};',
      ...(surface
        ? [
            '// Host transform block at buffer 1. NOT part of `uniforms[]`: an MVP matrix is a',
            '// host concern, not a document tunable.',
            'struct AI4ShaderTransform { float4x4 world; float4x4 worldViewProj; };',
            'struct VSOut {',
            '  float4 position [[position]];',
            '  float2 uv;',
            '  float3 worldPos;',
            '  float3 normal;',
            '};',
          ]
        : ['struct FSIn { float4 position [[position]]; };']),
    ],
    '// ---- shared helper library ----',
    shimsFor('msl'),
    lib,
    '// ---- entry point ----',
    surface
      ? [
          'vertex VSOut vs_main(uint vid [[vertex_id]],',
          '                     const device float3* positions [[buffer(2)]],',
          '                     const device float3* normals [[buffer(3)]],',
          '                     const device float2* uvs [[buffer(4)]],',
          '                     constant AI4ShaderTransform& X [[buffer(1)]]) {',
          '  VSOut out;',
          '  float4 world = X.world * float4(positions[vid], 1.0);',
          '  out.worldPos = world.xyz;',
          '  // Upper 3x3 of a rigid transform: exact for rotation and translation, approximate',
          '  // under non-uniform scale. Stated in the sidecar rather than hidden.',
          '  out.normal = (X.world * float4(normals[vid], 0.0)).xyz;',
          '  out.uv = uvs[vid];',
          '  out.position = X.worldViewProj * float4(positions[vid], 1.0);',
          '  return out;',
          '}',
          '',
          'fragment float4 fs_main(VSOut in [[stage_in]],',
          '                        constant AI4ShaderUniforms& U [[buffer(0)]],',
          '                        texture2d<float> iChannel0 [[texture(0)]],',
          '                        sampler iChannel0Sampler [[sampler(0)]]) {',
          '  // THE Y FLIP FOR THIS TARGET (exactly one): Metal texture V is top-left, the IR is',
          '  // bottom-left.',
          '  float2 uv = float2(in.uv.x, 1.0 - in.uv.y);',
          '  // A surface shader has no screen position; `fragCoord` is the mesh uv back in pixels.',
          '  float2 fragCoord = uv * U.iResolution.xy;',
          '  float3 iWorldPos = in.worldPos;',
          '  float3 iNormal = normalize(in.normal);',
          '  float3 iViewDir = normalize(U.iCameraPos - in.worldPos);',
          '  // ---- graph (node bodies; straight-line, in topological order) ----',
          ...bodies.map((l) => `  ${l}`),
          '',
          `  return ${color};`,
          '}',
        ]
      : [
          'fragment float4 fs_main(FSIn in [[stage_in]],',
          '                        constant AI4ShaderUniforms& U [[buffer(0)]],',
          '                        texture2d<float> iChannel0 [[texture(0)]],',
          '                        sampler iChannel0Sampler [[sampler(0)]]) {',
          '  // THE Y FLIP FOR THIS TARGET (exactly one).',
          '  float2 uv = float2(in.position.x / U.iResolution.x, 1.0 - in.position.y / U.iResolution.y);',
          '  // ---- graph (node bodies; straight-line, in topological order) ----',
          ...bodies.map((l) => `  ${l}`),
          '',
          `  return ${color};`,
          '}',
        ],
  ]);

  const sidecars = [
    {
      name: 'ai4shader.metal.json',
      content: `${JSON.stringify(
        {
          target: 'msl',
          entryPoint: surface ? 'fs_main' : 'fs_main',
          vertexEntryPoint: surface ? 'vs_main' : null,
          geometry: surface ? 'surface' : 'quad',
          buffers: [
            {
              name: 'AI4ShaderUniforms',
              index: 0,
              members: [
                { name: 'iResolution', type: 'float3' },
                { name: 'iTime', type: 'float' },
                { name: 'iTimeDelta', type: 'float' },
                { name: 'iFrame', type: 'int' },
                { name: 'iMouse', type: 'float4' },
                ...args.uniforms.map((u) => ({ name: u.name, type: typeName('msl', u.type) })),
                ...(surface
                  ? [
                      { name: 'iCameraPos', type: 'float3', note: 'world-space camera position' },
                      { name: 'iLightDir', type: 'float3', note: 'unit, surface -> light' },
                    ]
                  : []),
              ],
            },
            // Only for a surface document: a quad has no model matrix to apply.
            ...(surface
              ? [
                  {
                    name: 'AI4ShaderTransform',
                    index: 1,
                    note: 'Supplied by the host, NOT by the IR `uniforms[]` array.',
                    members: [
                      { name: 'world', type: 'float4x4', note: 'object -> world' },
                      { name: 'worldViewProj', type: 'float4x4' },
                    ],
                  },
                  { name: 'ai4sPositions', index: 2, type: 'const device float3*' },
                  { name: 'ai4sNormals', index: 3, type: 'const device float3*' },
                  { name: 'ai4sUVs', index: 4, type: 'const device float2*' },
                ]
              : []),
          ],
          textures: [{ name: 'iChannel0', index: 0, type: 'texture2d<float>' }],
          samplers: [{ name: 'iChannel0Sampler', index: 0 }],
          uniforms: args.uniforms.map((u) => ({ name: u.name, type: u.type, value: u.value })),
          notes: [
            'Struct offsets follow the Metal buffer layout rules for this member order; the host must build the same struct, not a std140 block.',
            ...(surface
              ? [
                  'SURFACE shader: vs_main indexes the vertex buffers by [[vertex_id]] and needs AI4ShaderTransform at buffer 1.',
                  'The normal matrix is the upper 3x3 of `world`, which is only correct for a rigid transform.',
                ]
              : []),
          ],
        },
        null,
        2,
      )}\n`,
    },
  ];

  return { source, sidecars, diagnostics };
};

/* ========================================================================== */
/* TARGET 5 — Unity ShaderLab                                                 */
/* ========================================================================== */

function sanitizeShaderName(title: string): string {
  const clean = title.replace(/["\\\n\r]/g, ' ').replace(/\s+/g, ' ').trim();
  return clean.length ? clean : 'Untitled';
}

/** Unity property type + default per IR type. Unity has no int/bool property kinds. */
function unityProperty(
  u: UniformDecl,
): { kind: string; value: string; note?: string } {
  const single = Array.isArray(u.value) ? u.value[0]! : u.value;
  const num = (x: number | boolean): string => (typeof x === 'boolean' ? (x ? '1' : '0') : String(Number(x)));
  const vec4 = (arr: number[], pad: number): string => {
    const v = [arr[0] ?? pad, arr[1] ?? pad, arr[2] ?? pad, arr[3] ?? pad];
    return `(${v.map((n) => String(Number(n))).join(', ')})`;
  };
  switch (u.type) {
    case 'vec4':
      return { kind: 'Vector', value: vec4(Array.isArray(u.value) ? (u.value as number[]) : [Number(single)], 0) };
    case 'vec3':
      return { kind: 'Vector', value: vec4(Array.isArray(u.value) ? (u.value as number[]) : [Number(single)], 0) };
    case 'vec2':
      return { kind: 'Vector', value: vec4(Array.isArray(u.value) ? (u.value as number[]) : [Number(single)], 0) };
    case 'bool':
      return { kind: 'Toggle', value: num(single), note: 'Unity has no bool property kind; `Toggle` is emitted as specified and a Float 0/1 is the portable fallback.' };
    case 'int':
      return { kind: 'Float', value: num(single), note: 'Unity has no int property kind; the value is a Float and is truncated in the shader.' };
    case 'float':
      if (u.min !== undefined && u.max !== undefined) {
        return { kind: `Range(${u.min}, ${u.max})`, value: num(single) };
      }
      return { kind: 'Float', value: num(single) };
    default:
      return { kind: 'Float', value: '0', note: `Matrix uniforms have no ShaderLab property kind; declared as a global instead.` };
  }
}

/** URP is detected from the render spec, the spec category/tags and the author's notes. */
function mentionsUrp(doc: ShaderDoc): boolean {
  const hay = [
    JSON.stringify(doc.render),
    doc.spec.category,
    doc.spec.tags.join(' '),
    doc.spec.intent,
    doc.targetNotes?.shaderlab ?? '',
  ]
    .join(' ')
    .toLowerCase();
  return /\burp\b|universal\s+render/.test(hay);
}

/**
 * Y flip: none. The graph's `uv` comes from the mesh UV stream
 * (`appdata.uv` → `v2f.uv`), and Unity's UV origin is bottom-left for the default quad
 * and for every texture imported with the standard convention, which already matches
 * the IR. Nothing in this target mirrors an axis.
 */
const shaderlabEmitter: TargetEmitter = (args) => {
  const diagnostics: Diagnostic[] = [];
  const surface = args.analysis.needs.surface;
  const lib = lower(args, 'shaderlab', libraryGlsl(args.doc), 'shaderlab: shared library', diagnostics);
  const bodies = embed(lower(args, 'shaderlab', bodiesGlsl(args), 'shaderlab: node bodies', diagnostics));
  const color = outputColor(args, 'shaderlab', diagnostics);

  const urp = mentionsUrp(args.doc);
  const title = sanitizeShaderName(args.doc.title);
  const usesTexture = /\btex2D_\s*\(/.test(bodiesGlsl(args));

  // Unity has no Matrix property kind, so matrix uniforms are shader globals only —
  // putting them in `Properties` as a Float would be a lie the inspector would show.
  const properties = args.uniforms
    .filter((u) => u.type !== 'mat2' && u.type !== 'mat3')
    .map((u) => {
      const p = unityProperty(u);
      return `    _${u.name} ("${u.label ?? u.name}", ${p.kind}) = ${p.value}`;
    });
  const propertyNotes = args.uniforms
    .map((u) => ({
      name: u.name,
      note:
        u.type === 'mat2' || u.type === 'mat3'
          ? 'Matrix uniforms have no ShaderLab property kind; declared as a shader global only.'
          : unityProperty(u).note,
    }))
    .filter((x) => !!x.note);

  const declares = [
    ...args.uniforms.map((u) => {
      return `${typeName('shaderlab', u.type)} _${u.name}; // default ${literalOf(u.type, u.value)}`;
    }),
    '// Unity has no global mouse: the value is uploaded per-material.',
    'float4 _MousePos;',
    ...(usesTexture
      ? ['sampler2D _Channel0;', 'float4 _Channel0_ST; // Unity fills this in from the texture']
      : []),
  ];

  const include = urp
    ? '#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"'
    : '#include "UnityCG.cginc"';
  const clip = urp ? 'TransformObjectToHClip(v.vertex.xyz)' : 'UnityObjectToClipPos(v.vertex)';
  // Unity already has the two functions a surface shader needs, and they differ between the
  // built-in pipeline and URP. Using them rather than hand-rolling `mul(unity_ObjectToWorld,…)`
  // is what keeps the normal correct under a non-uniform scale: Unity's version applies the
  // inverse transpose, which the HLSL/MSL/WGSL emitters cannot do without a matrix the host
  // has not given them.
  const toWorld = urp ? 'TransformObjectToWorld(v.vertex.xyz)' : 'mul(unity_ObjectToWorld, v.vertex).xyz';
  const toWorldNormal = urp
    ? 'TransformObjectToWorldNormal(v.normal)'
    : 'UnityObjectToWorldNormal(v.normal)';

  const source = assemble([
    [
      `// AI4Shader — Unity ShaderLab target for "${title}".`,
      ...feedbackNotes(args.doc),
      '//',
      '// UNIFORM REMAPPING WAS APPLIED (token-level, by the emitter):',
      '//   iResolution -> _ScreenParams.xy',
      '//   iTime       -> _Time.y',
      '//   iTimeDelta  -> unity_DeltaTime.x',
      '//   iFrame      -> floor(_Time.w * 20.0)   (a 60 Hz frame counter derived from _Time.w)',
      '//   iMouse      -> _MousePos               (declared below; Unity has no global mouse)',
      usesTexture ? '//   iChannel0   -> _Channel0 + _Channel0_ST (texel fetch is ST-aware)' : '//   iChannel0   -> _Channel0 (unused by this graph)',
      '//   u_*         -> _u_*                    (ShaderLab property names)',
      ...(surface
        ? [
            '// SURFACE DOCUMENT — the extra remappings:',
            '//   iNormal     -> normalised i.worldNormal   (from UnityObjectToWorldNormal)',
            '//   iWorldPos   -> i.worldPos                 (from mul(unity_ObjectToWorld, v.vertex))',
            '//   iViewDir    -> normalize(_WorldSpaceCameraPos - i.worldPos)',
            '//   iCameraPos  -> _WorldSpaceCameraPos       (Unity built-in)',
            '//   iLightDir   -> _WorldSpaceLightPos0.xyz   (Unity built-in; needs LightMode)',
          ]
        : []),
      '//',
      '// NO UNITY EQUIVALENT (degraded, listed honestly rather than guessed at):',
      '//   * iMouse is not a global: a C# host must call Material.SetVector("_MousePos", ...).',
      '//     In the editor it stays (0,0,0,0) unless something drives it.',
      '//   * iTimeDelta depends on unity_DeltaTime, which is only populated by the',
      '//     built-in/URP camera loops; in a ScriptableRenderPass it may read 0.',
      '//   * iFrame is an approximation (60 Hz assumption), not a real frame index.',
      '//   * int and bool uniforms become Float/Toggle properties: Unity has no int or',
      '//     bool material property kind.',
      '//   * The vertex stage uses the mesh UV stream. Screen-space UVs would need a',
      '//     separate pipeline and are intentionally not faked here.',
    ],
    `Shader "AI4Shader/${title}" {`,
    ['  Properties {', ...properties, '  }'],
    '  SubShader {',
    [
      `    Tags { ${urp ? '"RenderPipeline"="UniversalPipeline" ' : ''}"RenderType"="${args.doc.render.transparent ? 'Transparent' : 'Opaque'}" "Queue"="${args.doc.render.transparent ? 'Transparent' : 'Geometry'}" }`,
      ...(args.doc.render.transparent ? ['    Blend SrcAlpha OneMinusSrcAlpha, One OneMinusSrcAlpha', '    ZWrite Off'] : []),
      '    Pass {',
      // `_WorldSpaceLightPos0` is only populated for a light-mode pass. A surface document
      // reads it, so the pass has to declare which light it wants; a quad document does not
      // and is deliberately left without the tag, since adding one changes how Unity batches it.
      ...(surface ? ['      Tags { "LightMode"="ForwardBase" }'] : []),
      '      HLSLPROGRAM',
      '      #pragma vertex vert',
      '      #pragma fragment frag',
      `      ${include}`,
      '',
      ...declares.map((d) => `      ${d}`),
      '',
      surface
        ? '      struct appdata { float4 vertex : POSITION; float3 normal : NORMAL; float2 uv : TEXCOORD0; };'
        : '      struct appdata { float4 vertex : POSITION; float2 uv : TEXCOORD0; };',
      surface
        ? '      struct v2f { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; float3 worldPos : TEXCOORD1; float3 worldNormal : TEXCOORD2; };'
        : '      struct v2f { float4 pos : SV_POSITION; float2 uv : TEXCOORD0; };',
      '',
      shimsFor('shaderlab'),
      lib,
      '',
      '      v2f vert(appdata v) {',
      '        v2f o;',
      `        o.pos = ${clip};`,
      '        o.uv = v.uv; // UV origin is bottom-left, matching the IR: no flip here.',
      ...(surface
        ? [
            `        o.worldPos = ${toWorld};`,
            `        o.worldNormal = ${toWorldNormal};`,
          ]
        : []),
      '        return o;',
      '      }',
      '',
      '      fixed4 frag(v2f i) : SV_Target {',
      '        float2 uv = i.uv;',
      ...(surface
        ? [
            '        // A surface shader has no screen position; `fragCoord` is the mesh uv back in',
            '        // pixels so a node reading it still has a value.',
            '        float2 fragCoord = uv * _ScreenParams.xy;',
            '        float3 iWorldPos = i.worldPos;',
            '        float3 iNormal = normalize(i.worldNormal);',
            '        float3 iViewDir = normalize(_WorldSpaceCameraPos - i.worldPos);',
          ]
        : []),
      '        // ---- graph (node bodies; straight-line, in topological order) ----',
      ...bodies.map((l) => `        ${l}`),
      '',
      `        return ${color};`,
      '      }',
      '      ENDHLSL',
      '    }',
    ],
    '  }',
    '  Fallback Off',
    '}',
  ]);

  /* --- sidecar: material property payload --------------------------------- */
  const yaml: string[] = [
    '# AI4Shader material payload (Unity).',
    '# Deterministic, generated — do not hand-edit; regenerate from the IR.',
    '# This is the property block a host adapter applies with',
    '# Material.SetFloat / SetVector / SetColor before the first draw.',
    `shader: "AI4Shader/${title}"`,
    'properties:',
  ];
  for (const u of args.uniforms) {
    const p = unityProperty(u);
    const isMatrix = u.type === 'mat2' || u.type === 'mat3';
    yaml.push(`  - name: _${u.name}`);
    yaml.push(`    uniform: ${u.name}`);
    yaml.push(`    kind: ${isMatrix ? 'Matrix' : p.kind.startsWith('Range') ? 'Range' : p.kind}`);
    yaml.push(`    label: "${u.label ?? u.name}"`);
    yaml.push(`    value: ${isMatrix ? 'null' : p.value}`);
    if (u.min !== undefined) yaml.push(`    min: ${u.min}`);
    if (u.max !== undefined) yaml.push(`    max: ${u.max}`);
    if (u.group) yaml.push(`    group: "${u.group}"`);
    if (u.origin) yaml.push(`    origin: "${u.origin.node}.${u.origin.param}"`);
  }
  if (propertyNotes.length) {
    yaml.push('notes:');
    for (const n of propertyNotes) yaml.push(`  - ${n.name}: "${n.note}"`);
  }
  if (surface) {
    yaml.push('surface:');
    yaml.push('  geometry: surface');
    yaml.push('  passTag: ForwardBase');
    yaml.push('  hostGlobals:');
    yaml.push('    - _WorldSpaceCameraPos: supplied by Unity per camera');
    yaml.push('    - _WorldSpaceLightPos0.xyz: supplied by Unity for the ForwardBase pass');
    yaml.push('  notes:');
    yaml.push('    - "iLightDir maps to _WorldSpaceLightPos0.xyz, which is the direction TOWARDS the light for a directional light. In a pass with no LightMode tag it is undefined."');
    yaml.push('    - "Normals come from UnityObjectToWorldNormal / TransformObjectToWorldNormal, so a non-uniformly scaled object still shades correctly."');
  } else {
    yaml.push('surface:');
    yaml.push('  geometry: quad');
  }

  return {
    source,
    sidecars: [{ name: 'Material.ai4shader.yaml', content: `${yaml.join('\n')}\n` }],
    diagnostics,
  };
};

/* ========================================================================== */
/* TARGET 6 — Unreal Engine material Custom node                              */
/* ========================================================================== */

/**
 * A UE Custom node takes no uniforms: every parameter is an input pin. So the IR's
 * uniforms become function arguments, and the emitted JSON sidecar is the exact pin
 * list a user has to create.
 *
 * Y flip: none in the math. `uv` is a pin fed from a TextureCoordinate node, and UE's
 * UV0 origin is bottom-left, so it already matches the IR.
 *
 * ## Surface documents cost nothing extra here
 *
 * This is the one target where a surface shader needs no new stage. A Custom node's inputs
 * are pins wired from the material graph, and UE already has nodes for every symbol the
 * surface group reads — `VertexNormalWS`, `AbsoluteWorldPosition`, `CameraPositionWS`, and
 * any light-vector node the user prefers. So the "entry point" is three more pins and three
 * more locals, and the graph on either side stays UE's business.
 *
 * That also makes this target the most honest about `iLightDir`: the IR treats it as a host
 * uniform, and here the user *chooses* what feeds it, which is what a toon shader's author
 * wants anyway.
 */
const ueEmitter: TargetEmitter = (args) => {
  const diagnostics: Diagnostic[] = [];
  const surface = args.analysis.needs.surface;
  const lib = lower(args, 'ue', libraryGlsl(args.doc), 'ue: shared library', diagnostics);
  const bodies = embed(lower(args, 'ue', bodiesGlsl(args), 'ue: node bodies', diagnostics));
  const color = outputColor(args, 'ue', diagnostics);

  const pinType = (t: ShaderType): string =>
    t === 'vec2' ? 'float2' : t === 'vec3' ? 'float3' : t === 'vec4' ? 'float4' : 'MaterialFloat';
  const jsonType = (t: ShaderType): string =>
    t === 'vec2' ? 'float2' : t === 'vec3' ? 'float3' : t === 'vec4' ? 'float4' : 'float';

  /** The surface pins, in the order the signature and the sidecar both use. */
  const surfacePins = surface
    ? [
        { name: 'InWorldPos', type: 'float3', label: 'World Position (AbsoluteWorldPosition)' },
        { name: 'InNormal', type: 'float3', label: 'World Normal (VertexNormalWS)' },
        { name: 'InCameraPos', type: 'float3', label: 'Camera Position (CameraPositionWS)' },
        { name: 'InLightDir', type: 'float3', label: 'Light Direction (unit, surface -> light)' },
      ]
    : [];

  const signature = [
    'float2 InUV',
    ...args.uniforms.map((u) => `${pinType(u.type)} ${u.name}`),
    ...surfacePins.map((p) => `${p.type} ${p.name}`),
  ];
  const callArgs = ['UV', ...args.uniforms.map((u) => u.name), ...surfacePins.map((p) => p.name)];

  const source = assemble([
    [
      '// AI4Shader — Unreal Engine 5 material Custom node.',
      // UE has no feedback precedent at all, so the note is if anything more important here: a
      // material graph samples whatever texture is wired into the channel pin, and "the previous
      // frame" is a RenderTarget the graph author has to create and ping-pong by hand.
      ...feedbackNotes(args.doc),
      '//',
      '// HOW THIS FILE IS USED: paste everything below into the Custom node\'s Code',
      '// field, add the pins from CustomNode.inputs.json, and set Output Type to',
      '// CMOT Float 4. The trailing call line is part of the pasted text.',
      '//',
      '// HOST SYMBOL REMAPPING WAS APPLIED:',
      '//   iResolution -> View.ViewSizeAndInvSize.xy',
      '//   iTime       -> View.GameTime',
      '//   iTimeDelta  -> View.DeltaTime',
      '//   iFrame      -> float(View.FrameNumber)',
      '//   iMouse      -> View.MousePosition',
      ...(surface
        ? [
            '//   iWorldPos   -> InWorldPos pin   (AbsoluteWorldPosition)',
            '//   iNormal     -> InNormal pin     (VertexNormalWS, normalised below)',
            '//   iCameraPos  -> InCameraPos pin  (CameraPositionWS)',
            '//   iViewDir    -> normalize(InCameraPos - InWorldPos)',
            '//   iLightDir   -> InLightDir pin   (any light-vector node; normalised below)',
          ]
        : []),
      '//',
      '// `View.GameTime` (and every other `View.*` member used here) is ONLY valid',
      '// inside a material function / material preview context, where the view uniform',
      '// buffer is bound. In a post-process or a custom pass this code must be given a',
      '// real input pin instead. `View.MousePosition` in particular is absent from some',
      '// engine versions: if it fails to compile, replace it with a scalar parameter.',
      '//',
      '// WHAT DEGRADES ON THIS TARGET:',
      '//   * No uniforms: every IR uniform is a pin, so the graph cannot be re-tuned',
      '//     without editing the material asset.',
      '//   * Texture nodes need a Texture Object pin AND a Sampler pin; the pin list in',
      '//     the sidecar covers float pins only, because UE pin types for textures are',
      '//     not expressible in the sidecar schema. `iChannel0`/`iChannel0Sampler` are',
      '//     referenced by name and must be added by hand if a texture node is used.',
      '//   * int/bool pins are floats: the implicit conversion is the host\'s business.',
      '//   * Constant loop bounds are unrolled by the material compiler; a large',
      '//     OCTAVE_LIMIT costs compile time and instruction count.',
    ],
    '// ---- shared helper library ----',
    shimsFor('ue'),
    lib,
    '// ---- graph + entry point ----',
    [
      `MaterialFloat4 ai4shader_main(${signature.join(', ')}) {`,
      '  // uv arrives as a pin (TextureCoordinate, bottom-left origin): no Y flip.',
      '  float2 uv = InUV;',
      ...(surface
        ? [
            '  // ---- surface inputs (pins) ----',
            '  // A surface shader has no screen position; `fragCoord` is the mesh uv back in',
            '  // pixels so a node reading it still has a value.',
            '  float2 fragCoord = uv * View.ViewSizeAndInvSize.xy;',
            '  float3 iWorldPos = InWorldPos;',
            '  float3 iNormal = normalize(InNormal);',
            '  float3 iViewDir = normalize(InCameraPos - InWorldPos);',
            '  float3 iLightDir = normalize(InLightDir);',
          ]
        : []),
      '  // ---- graph (node bodies; straight-line, in topological order) ----',
      ...bodies.map((l) => `  ${l}`),
      '',
      `  return ${color};`,
      '}',
    ],
    '// ---- Custom node output ----',
    [`return ai4shader_main(${callArgs.join(', ')});`],
  ]);

  /* --- sidecar: pin list --------------------------------------------------- */
  const inputs = [
    { name: 'UV', type: 'float2', defaultValue: [0, 0], label: 'UV (from a TextureCoordinate node)' },
    ...args.uniforms.map((u) => ({
      name: u.name,
      type: jsonType(u.type),
      defaultValue: u.value,
      label: u.label ?? u.name,
    })),
    // Surface pins last, so a quad document's pin list is byte-identical to what it was.
    ...(surface
      ? surfacePins.map((p) => ({
          name: p.name,
          type: jsonType('vec3' as ShaderType),
          defaultValue: [0, 0, 1],
          label: p.label,
        }))
      : []),
  ];

  /* --- sidecar: build recipe ---------------------------------------------- */
  const md: string[] = [
    `# Building this graph in Unreal Engine — ${args.doc.title}`,
    '',
    'Generated from the AI4Shader IR. Follow the steps in order; nothing here is optional.',
    '',
    '## 1. Build recipe',
    '',
    '1. Add a **Custom** node to the material graph (right-click → search `Custom`).',
    '2. Set **Output Type** to `CMOT Float 4`.',
    `3. Add ${inputs.length} input pin${inputs.length === 1 ? '' : 's'}, in this exact order and with these exact names:`,
    '',
    ...inputs.map((p, i) => `   ${i + 1}. \`${p.name}\` — type \`${p.type}\`, default \`${JSON.stringify(p.defaultValue)}\` (${p.label})`),
    '',
    '4. Paste the code from §3 into the node\'s **Code** field, replacing the default `return 0;`.',
    '5. Wire the pins: `UV` from a **TextureCoordinate** node (index 0); every `u_*` pin from a',
    '   **ScalarParameter**/**VectorParameter** so the value stays tunable outside the graph.',
    '6. Connect the Custom node\'s output to **Emissive Color** (and to **Opacity** only if the doc',
    '   is transparent).',
    '',
    '## 2. Pin list',
    '',
    '| # | name | type | default | label |',
    '|---|---|---|---|---|',
    ...inputs.map((p, i) => `| ${i + 1} | \`${p.name}\` | \`${p.type}\` | \`${JSON.stringify(p.defaultValue)}\` | ${p.label} |`),
    '',
    'The same list is machine-readable in `CustomNode.inputs.json`.',
    '',
    '## 3. Code to paste',
    '',
    '```hlsl',
    source.replace(/\n$/, ''),
    '```',
    '',
    '## 4. Degradations on this target',
    '',
    '- **No uniforms.** Every IR uniform becomes a pin; there is no runtime slider path.',
    '- **`View.GameTime` needs a view.** Valid in material functions and previews; in other',
    '  passes, drive the time from a pin instead.',
    '- **`View.MousePosition`** may not exist in every engine version — replace it with a pin.',
    '- **Texture nodes** need a Texture Object pin plus a Sampler pin, added by hand.',
    '- **Loops are unrolled** by the material compiler: high quality tiers cost compile time.',
    '',
  ];

  return {
    source,
    sidecars: [
      { name: 'CustomNode.inputs.json', content: `${JSON.stringify(inputs, null, 2)}\n` },
      { name: 'MaterialGraph.ai4shader.md', content: `${md.join('\n')}\n` },
    ],
    diagnostics,
  };
};

/* -------------------------------------------------------------------------- */
/* Registration                                                                */
/* -------------------------------------------------------------------------- */

registerTarget('glsl', glslEmitter);
registerTarget('wgsl', wgslEmitter);
registerTarget('hlsl', hlslEmitter);
registerTarget('msl', mslEmitter);
registerTarget('shaderlab', shaderlabEmitter);
registerTarget('ue', ueEmitter);

/** Every target this module registers. Used by tests and by the CLI's `--target all`. */
export const IMPLEMENTED_TARGETS: readonly TargetFormat[] = [
  'glsl', 'wgsl', 'hlsl', 'msl', 'shaderlab', 'ue',
];

/** Emitters by target, for hosts that want to call one directly. */
export const EMITTERS: Record<TargetFormat, TargetEmitter> = {
  glsl: glslEmitter,
  wgsl: wgslEmitter,
  hlsl: hlslEmitter,
  msl: mslEmitter,
  shaderlab: shaderlabEmitter,
  ue: ueEmitter,
};

/* -------------------------------------------------------------------------- */
/* Self-checks                                                                 */
/* -------------------------------------------------------------------------- */

/**
 * Structural validation of a non-GLSL target's output. This is not a compiler: it is
 * the set of properties this package *promises* about its own output, checked so a
 * regression fails a test instead of a user's build.
 *
 * Reports GLSL type names and `*_` aliases surviving outside comments (the leak the
 * token rewriter exists to prevent), plus a bare `texture(...)` call.
 */
export function checkTargetLeaks(src: string, target: TargetFormat): string[] {
  const problems: string[] = [];
  const toks = tokenize(src).filter((t) => t.kind !== 'space' && t.kind !== 'comment');
  for (let i = 0; i < toks.length; i++) {
    const t = toks[i]!;
    if (t.kind !== 'ident') continue;
    if (target !== 'glsl' && isGlslTypeName(t.text)) {
      problems.push(`GLSL type name \`${t.text}\` survived into ${target} at offset ${t.pos}`);
    }
    if (target === 'wgsl' && t.text === 'float') {
      problems.push(`GLSL type name \`float\` survived into wgsl at offset ${t.pos}`);
    }
    if (isAliasName(t.text)) {
      problems.push(`rewrite alias \`${t.text}\` survived into ${target} at offset ${t.pos}`);
    }
    if (t.text === 'texture' && toks[i + 1]?.text === '(') {
      // `[[texture(0)]]` (an MSL attribute) and `x.texture(...)` are not fetch calls.
      let j = i - 1;
      while (j >= 0 && (toks[j]!.kind === 'space' || toks[j]!.kind === 'comment')) j--;
      const prev = toks[j]?.text;
      if (prev !== '[' && prev !== '.' && prev !== '@') {
        problems.push(`bare \`texture(...)\` call survived into ${target} at offset ${t.pos}`);
      }
    }
  }
  // `texture(...)` is the *correct* call in GLSL; everywhere else it means the
  // one-argument alias was rewritten but never lowered into a real fetch.
  if (target === 'wgsl' && stripComments(src).includes('texture(')) {
    problems.push('bare `texture(` call survived into wgsl');
  }
  return problems;
}

/**
 * Structural check for WGSL — the strictest target, and the one with no compiler
 * available here. Asserts:
 *
 *   • braces balance (and never close before they open);
 *   • every `var` / `let` / `fn` declaration is well formed (`var<space> name: T;`,
 *     `let name = expr;`, `fn name(args) -> T {`);
 *   • no GLSL type names (`vec2/vec3/vec4/mat2/mat3/float`) outside comments;
 *   • no `fract_`/`mix_`/`mod_`/`atan2_`/`tex2D_`/`mat2_mul_`/`mat3_mul_` aliases;
 *   • no `texture(` call.
 *
 * Returns a list of human-readable problems; empty means "nothing this checker knows
 * how to object to". It does **not** mean the file compiles — see `targets/README.md`.
 */
export function checkWgslStructural(src: string): string[] {
  const problems: string[] = checkTargetLeaks(src, 'wgsl');
  const code = stripComments(src);

  // --- braces --------------------------------------------------------------
  let depth = 0;
  for (let i = 0; i < code.length; i++) {
    const c = code[i];
    if (c === '{') depth++;
    else if (c === '}') {
      depth--;
      if (depth < 0) {
        problems.push(`unbalanced \`}\` at offset ${i}`);
        depth = 0;
        break;
      }
    }
  }
  if (depth !== 0) problems.push(`braces do not balance: ${depth} unclosed \`{\``);

  // --- no preprocessor / stray punctuation ---------------------------------
  const hash = code.indexOf('#');
  if (hash !== -1) {
    problems.push(`\`#\` at offset ${hash}: WGSL has no preprocessor directives`);
  }

  // --- declarations --------------------------------------------------------
  const count = (re: RegExp): number => (code.match(re) ?? []).length;
  const varKeyword = /\bvar\b/g;
  const varDecl =
    /\bvar\s*(?:<[a-z_]+>\s*)?[A-Za-z_]\w*\s*:\s*[A-Za-z_]\w*(?:\s*<[^<>;]*>)?\s*(?:=[^;]*)?;/g;
  if (count(varKeyword) !== count(varDecl)) {
    problems.push(
      `malformed \`var\` declaration: ${count(varKeyword)} \`var\` keyword(s) but ${count(varDecl)} well-formed declaration(s)`,
    );
  }
  const letKeyword = /\blet\b/g;
  const letDecl =
    /\blet\s+[A-Za-z_]\w*\s*(?::\s*[A-Za-z_]\w*(?:\s*<[^<>;]*>)?\s*)?=[^;]*;/g;
  if (count(letKeyword) !== count(letDecl)) {
    problems.push(
      `malformed \`let\` declaration: ${count(letKeyword)} \`let\` keyword(s) but ${count(letDecl)} well-formed declaration(s)`,
    );
  }
  const fnKeyword = /\bfn\b/g;
  // The parameter list may contain attributes with their own parentheses
  // (`@builtin(position)`), and the return type may carry one too (`@location(0)`).
  const fnDecl =
    /\bfn\s+[A-Za-z_]\w*\s*\((?:[^()]|\([^()]*\))*\)\s*(?:->\s*[^;{]*?)?\s*\{/g;
  if (count(fnKeyword) !== count(fnDecl)) {
    problems.push(
      `malformed \`fn\` declaration: ${count(fnKeyword)} \`fn\` keyword(s) but ${count(fnDecl)} well-formed declaration(s)`,
    );
  }

  return problems;
}

/** Throwing form, for hosts that would rather fail loudly than inspect a list. */
export function assertWgslStructural(src: string): void {
  const problems = checkWgslStructural(src);
  if (problems.length) {
    throw new Error(`WGSL structural check failed:\n  - ${problems.join('\n  - ')}`);
  }
}
