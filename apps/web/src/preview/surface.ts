/**
 * Surface preview — the shader as a material on an object, in a lit scene
 * ======================================================================
 *
 * The flat preview answers "what does this pattern look like?". It is a fullscreen quad, and
 * a fullscreen quad is the worst possible way to judge a procedural shader: it hides
 * stretching, seams, scale, and every question about how the pattern behaves on a *surface*.
 * Liquid metal on a quad and liquid metal on a torus knot are different objects.
 *
 * This renderer draws the same pattern as a **material**: the emitted fragment shader is
 * rendered once per frame into an offscreen canvas by the existing backend, uploaded as a
 * texture, and sampled onto a lit mesh with a real environment, a key light and a ground
 * plane. The emitter is not touched at all — no IR change, no new target, no second entry
 * point — because the pattern is already a field, and a field can be mapped.
 *
 * ## Two material paths, and which one is in use
 *
 * **Pattern path (the default).** The emitted fragment shader is rendered into an offscreen
 * canvas by the existing backend, uploaded as a texture, and sampled onto the mesh. The pattern
 * is a field, and a field can be mapped — that works for any document, including one that only
 * ever reads `uv` and `iTime`.
 *
 * **Graph path.** A document whose reachable graph reads a *surface* input (`iNormal`,
 * `iWorldPos`, `iViewDir`, `iLightDir`) is not a picture at all, it is a shading function, and
 * baking it into a texture throws away the very thing it computes. For those documents the
 * emitted GLSL **surface** shader is compiled directly against a mesh vertex stage with real
 * interpolated varyings, so the graph is evaluated per fragment, in 3D. That is what makes a
 * toon ramp turn with the light instead of being a static stripe pasted onto the model.
 *
 * Which path is live is decided by `Emission.needs.surface` — the same flag that changes the
 * emitted entry point — and never by a heuristic in this file. A document that does not need a
 * surface keeps the byte-identical pattern path.
 *
 * What the pattern path costs, stated rather than hidden: the pattern's resolution is capped at
 * `PATTERN_SIZE` (512²), so a very high-frequency shader softens on a large object, and a
 * screen-space effect baked into the pattern (a vignette, say) is mapped onto the model rather
 * than recomputed. The panel says both.
 *
 * ## Exposure and tone mapping apply to the pattern path only
 *
 * The graph path hands the mesh shader's `fragColor` straight to the framebuffer. That is
 * deliberate: the graph is authored as a *display* colour — the same value the flat preview
 * shows — and running it through the material's filmic curve would make the 3D view disagree
 * with the 2D view for a reason the user cannot see. `uExposure` therefore scales the sky (and
 * the pattern path), and the panel says so while the graph path is active.
 *
 * ## The environment is analytic, and that is a choice
 *
 * There is no HDRI here and there cannot be one: no network, no binary assets, nothing a
 * reviewer could diff. So the environment is a gradient sky plus a bright sun disc, evaluated
 * in the shader and shared between the background pass and the material's reflection lookup.
 * That is enough for metal to have something to reflect — the entire reason a metalness slider
 * is worth having — and it is honest about being synthetic.
 */

import { buildMesh, type MeshData, type ShapeId } from './mesh.js';
import {
  deg2rad,
  lookAt,
  multiply,
  normalMatrix3,
  perspective,
  translation,
  type Mat4,
} from './mat4.js';
import {
  reflectUniforms,
  writeUniformByGlType,
  type ReflectedUniform,
  type UniformValue,
  type UniformValues,
} from './renderer.js';

/** Object only, or object standing on a ground plane. */
export type SurfaceMode = 'material' | 'scene';

/** How the pattern is projected onto the surface. */
export type MappingMode = 'triplanar' | 'uv';

export type MaterialSettings = {
  /** World units per pattern tile. Higher means the pattern repeats more. */
  density: number;
  metalness: number;
  roughness: number;
  /** 0 = neutral grey material, 1 = the pattern is the albedo. */
  patternMix: number;
  exposure: number;
  mapping: MappingMode;
  /** Blend opposing pattern edges for continuous repetition on closed meshes. */
  seamless: boolean;
};

export const DEFAULT_MATERIAL: MaterialSettings = {
  density: 1,
  metalness: 0.9,
  roughness: 0.28,
  patternMix: 1,
  exposure: 1,
  mapping: 'triplanar',
  seamless: true,
};

export type CaptureResult = { width: number; height: number; data: Uint8ClampedArray };

/** The pattern canvas is a texture, not a picture: 512² is plenty and keeps the upload cheap. */
export const PATTERN_SIZE = 512;

export type SurfaceChoice = {
  surface: SurfaceRenderer | null;
  /** Human-readable reason when the surface pass could not be built. */
  note?: string;
};

/**
 * A document handed to the 3D pass so the mesh can be shaded by the graph itself.
 *
 * Everything here comes from `Emission` verbatim. `source` is **not** modified in any way —
 * not one character — so what the mesh runs is what the export contains. That property is the
 * whole reason this path is worth having: it turns the 3D view into a check on the emitted
 * GLSL rather than a second implementation that could agree with the emitter while both are
 * wrong.
 */
export type GraphMaterial = {
  /** Complete GLSL ES 3.00 *surface* fragment source, exactly as `emit()` produced it. */
  source: string;
  /** Uniforms the source declares, in emission order. */
  uniforms: readonly { name: string; type: string; value: UniformValue }[];
  /** Current slider values, keyed by uniform name. */
  values: UniformValues;
};

/** The graph path's vertex stage, written to the contract the emitter documents. */
const GRAPH_VERT = `#version 300 es
precision highp float;

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUv;

uniform mat4 uModel;
uniform mat4 uViewProj;
uniform mat3 uNormalMat;

// The names matter: GLSL ES links varyings by name, and these three are the ones the
// emitted surface fragment shader declares as its vertex contract.
out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vUv;

void main() {
  vec4 world = uModel * vec4(aPos, 1.0);
  vWorldPos = world.xyz;
  vNormal = uNormalMat * aNormal;
  vUv = aUv;
  gl_Position = uViewProj * world;
}
`;

/* -------------------------------------------------------------------------- */
/* Shaders                                                                     */
/* -------------------------------------------------------------------------- */

const VERT = `#version 300 es
precision highp float;

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUv;

uniform mat4 uModel;
uniform mat4 uViewProj;
uniform mat3 uNormalMat;

out vec3 vWorld;
out vec3 vNormal;
out vec2 vUv;

void main() {
  vec4 world = uModel * vec4(aPos, 1.0);
  vWorld = world.xyz;
  vNormal = uNormalMat * aNormal;
  vUv = aUv;
  gl_Position = uViewProj * world;
}
`;

/**
 * The sky, drawn first as a fullscreen triangle.
 *
 * The view ray is rebuilt from three world-space basis vectors rather than from the clip
 * position alone. Using `normalize(vec3(ndc, 1.0))` would be shorter and wrong: that is a
 * *view-space* ray, so the sky would stay welded to the screen and the sun would not swing
 * past as the camera orbits — which is most of what makes the scene read as three-dimensional.
 *
 * Its colour function is duplicated in the material shader. GLSL has no shared prelude in this
 * project, and a background that disagreed with the reflections would look subtly wrong in a
 * way nobody could name.
 */
const SKY_VERT = `#version 300 es
precision highp float;
out vec2 vNdc;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vNdc = p * 2.0 - 1.0;
  gl_Position = vec4(vNdc, 1.0, 1.0);
}
`;

const SKY_FRAG = `#version 300 es
precision highp float;

in vec2 vNdc;
out vec4 outColor;

uniform vec3 uRayX;
uniform vec3 uRayY;
uniform vec3 uRayZ;
uniform vec3 uSunDir;
uniform float uExposure;

vec3 skyColor(vec3 d) {
  float up = clamp(d.y * 0.5 + 0.5, 0.0, 1.0);
  vec3 horizon = vec3(0.055, 0.062, 0.075);
  vec3 zenith  = vec3(0.115, 0.155, 0.225);
  vec3 below   = vec3(0.018, 0.020, 0.024);
  vec3 base = mix(below, mix(horizon, zenith, smoothstep(0.5, 1.0, up)), smoothstep(0.34, 0.52, up));
  float sun = pow(max(0.0, dot(d, normalize(uSunDir))), 320.0);
  return base + vec3(1.0, 0.95, 0.88) * sun * 8.0;
}

void main() {
  vec3 d = normalize(uRayX * vNdc.x + uRayY * vNdc.y + uRayZ);
  outColor = vec4(skyColor(d) * uExposure, 1.0);
}
`;

/**
 * The material.
 *
 * Deliberately a compact Cook-Torrance rather than a full PBR stack: one analytic key light, a
 * gradient environment for reflections, and Smith-Schlick visibility with a GGX distribution.
 * That is enough for the two things this preview has to show — that a metal reflects its
 * surroundings, and that roughness changes the size of the highlight — and every extra term is
 * another thing that could be wrong in a render nobody here can look at.
 *
 * The pattern is treated as **sRGB albedo** and linearised before lighting. Skipping that makes
 * every material washed out and makes the pattern look different from the flat preview for a
 * reason the user cannot see.
 */
const MATERIAL_FRAG = `#version 300 es
precision highp float;

in vec3 vWorld;
in vec3 vNormal;
in vec2 vUv;

uniform sampler2D uPattern;
uniform vec3 uCameraPos;
uniform vec3 uSunDir;
uniform float uDensity;
uniform float uMetalness;
uniform float uRoughness;
uniform float uPatternMix;
uniform float uExposure;
uniform int uMapping;
uniform bool uSeamless;
/** xy = footprint centre (xz in world), z = radius, w = strength. */
uniform vec4 uShadow;
/** 1.0 when this draw is the ground plane. */
uniform float uIsGround;

out vec4 outColor;

const float PI = 3.14159265359;

vec3 toLinear(vec3 c) {
  return pow(max(c, vec3(0.0)), vec3(2.2));
}

/**
 * Triplanar projection.
 *
 * A single planar projection smears the pattern wherever the surface is edge-on to it — on a
 * torus knot that is most of the object. Blending three axis-aligned projections by the normal
 * removes the smearing at the cost of two extra samples, and the sharpened weight
 * (pow to the 4th) keeps the blend narrow so the three projections do not visibly ghost over
 * each other on curved surfaces.
 *
 * The +0.5 offset is not cosmetic and neither is the sampler's REPEAT wrap — they are two halves
 * of one decision. Shipping with neither of them is the bug that pattern-mapping.test.ts exists
 * for, and it is worth stating in full because each half looked reasonable alone:
 *
 *   - the mesh is centred on the origin, so raw world coordinates are negative across half of
 *     every axis. Sampling at p.xy * scale therefore asks the texture for a texel outside the
 *     image, and under CLAMP_TO_EDGE the answer is the edge texel: a plausible colour, so nothing
 *     looked broken, but it carries no pattern information;
 *   - the visible result was only the part of the object that happened to land inside the unit
 *     square — the (+, +, +) corner of a cube and the positive half of the three faces meeting
 *     there, and on the ground plane the two unit-wide bands x in [0,1] and z in [0,1]: a cross of
 *     real pattern content in a field of four stretched corner texels.
 *
 * Measured, over the vertices of every shape at the default density, as the fraction left to the
 * sampler's edge clamp — the number pattern-mapping.test.ts pins at zero:
 *
 *     sphere 71%   torus 74%   knot 75%   cube 73%   ground 98%
 *
 * (With the offset but without the wrap those figures read 36 / 53 / 38 / 29 / 98, which is why
 * both halves are needed and why the fix is not "one or the other".)
 *
 * So: +0.5 moves the pattern's centre to the object's origin (a low density therefore shows the
 * whole pattern, once, centred, with no wrap at all), and REPEAT covers every coordinate that
 * still leaves the image — which at density 1 is most of them.
 *
 * NOTE for whoever edits this shader: it lives inside a JavaScript template literal in this file,
 * so a backtick in a GLSL comment terminates the literal and the TypeScript projector then fails
 * with a parse error pointing at whatever non-ASCII character follows. Quote GLSL names with
 * ordinary quotes here.
 */
// Two overlapping charts per axis. Each chart has zero weight at its own cut.
// Derivatives come from the unwrapped coordinates, never from fract at the cut.
vec3 samplePattern(vec2 uv) {
  vec2 dx = dFdx(uv);
  vec2 dy = dFdy(uv);
  if (!uSeamless) return textureGrad(uPattern, uv, dx, dy).rgb;
  vec2 q = fract(uv);
  vec2 shifted = fract(q + 0.5);
  vec2 w = smoothstep(vec2(0.0), vec2(0.25), q)
    * (1.0 - smoothstep(vec2(0.75), vec2(1.0), q));
  vec3 a = textureGrad(uPattern, q, dx, dy).rgb;
  vec3 b = textureGrad(uPattern, vec2(shifted.x, q.y), dx, dy).rgb;
  vec3 c = textureGrad(uPattern, vec2(q.x, shifted.y), dx, dy).rgb;
  vec3 d = textureGrad(uPattern, shifted, dx, dy).rgb;
  return mix(mix(d, c, w.x), mix(b, a, w.x), w.y);
}

vec3 sampleTriplanar(vec3 p, vec3 n, float scale) {
  vec3 w = pow(abs(n), vec3(4.0));
  w /= max(1e-4, w.x + w.y + w.z);
  vec3 x = samplePattern(p.zy * scale + 0.5);
  vec3 y = samplePattern(p.xz * scale + 0.5);
  vec3 z = samplePattern(p.xy * scale + 0.5);
  return x * w.x + y * w.y + z * w.z;
}

/**
 * The pattern, sampled for this fragment.
 *
 * The uMapping == 1 branch is the authored UV mapping: each face of the mesh carries its own
 * [0, 1] square, so density 1 lays the pattern out exactly once per face — the "decal" reading of
 * the same texture. Densities above 1 tile it, which is what the slider promises, and the REPEAT
 * wrap makes those tiles meet edge-to-edge instead of smearing the last row across the whole face.
 */
vec3 albedoAt(vec3 p, vec3 n) {
  // A closed UV surface must contain a whole number of periods at its 0/1 seam.
  float uvDensity = uSeamless ? max(1.0, floor(uDensity + 0.5)) : uDensity;
  vec3 pattern = uMapping == 1
    ? samplePattern(vUv * uvDensity)
    : sampleTriplanar(p, n, uDensity);
  // uPatternMix blends toward a neutral dielectric so the lighting itself can be judged
  // without the pattern arguing with it.
  return toLinear(mix(vec3(0.62), pattern, uPatternMix));
}

/** The environment: the sky, plus a dark lower half so a downward reflection is not sky. */
vec3 envColor(vec3 d) {
  float up = clamp(d.y * 0.5 + 0.5, 0.0, 1.0);
  vec3 horizon = vec3(0.055, 0.062, 0.075);
  vec3 zenith = vec3(0.115, 0.155, 0.225);
  vec3 below = vec3(0.018, 0.020, 0.024);
  vec3 base = mix(below, mix(horizon, zenith, smoothstep(0.5, 1.0, up)), smoothstep(0.34, 0.52, up));
  float sun = pow(max(0.0, dot(d, normalize(uSunDir))), 320.0);
  return base + vec3(1.0, 0.95, 0.88) * sun * 8.0;
}

void main() {
  vec3 n = normalize(vNormal);
  vec3 v = normalize(uCameraPos - vWorld);

  vec3 albedo = albedoAt(vWorld, n);
  float metal = clamp(uMetalness, 0.0, 1.0);
  float rough = clamp(uRoughness, 0.04, 1.0);

  vec3 f0 = mix(vec3(0.04), albedo, metal);
  vec3 diffuseColor = albedo * (1.0 - metal);

  vec3 l = normalize(uSunDir);
  vec3 h = normalize(l + v);
  float nol = max(dot(n, l), 0.0);
  float nov = max(dot(n, v), 1e-4);
  float noh = max(dot(n, h), 0.0);
  float voh = max(dot(v, h), 0.0);

  float a = rough * rough;
  float a2 = a * a;
  float denom = noh * noh * (a2 - 1.0) + 1.0;
  float distribution = a2 / max(1e-6, PI * denom * denom);

  float k = a * 0.5;
  float gv = nov / (nov * (1.0 - k) + k);
  float gl = nol / (nol * (1.0 - k) + k);

  vec3 fresnel = f0 + (1.0 - f0) * pow(1.0 - voh, 5.0);
  vec3 specular = distribution * gv * gl * fresnel / max(1e-4, 4.0 * nov * nol);

  vec3 sunColor = vec3(3.1, 2.95, 2.75);
  vec3 direct = (diffuseColor / PI + specular) * sunColor * nol;

  // Ambient: a diffuse half from the sky above, plus a reflection of the environment. Roughness
  // blurs a reflection; without a mip chain the cheapest honest approximation is to fade the
  // environment specular out as roughness rises, rather than pretend to filter it.
  vec3 reflected = reflect(-v, n);
  vec3 ambientDiffuse = envColor(n) * diffuseColor * 2.0;
  vec3 ambientSpecular = envColor(reflected) * f0 * (1.0 - rough * 0.75);

  vec3 color = direct + ambientDiffuse + ambientSpecular;

  if (uIsGround > 0.5) {
    // Contact shadow: a soft darkening of the floor near the object's footprint. A real shadow
    // map is a second pass and a second depth texture; this is a falloff, and the panel calls it
    // a falloff rather than passing it off as a shadow.
    float d = length(vWorld.xz - uShadow.xy);
    color *= mix(1.0, smoothstep(uShadow.z * 0.22, uShadow.z * 1.4, d), uShadow.w);
    // Distance fog, so the floor does not end in a hard edge where the mesh stops.
    float fog = 1.0 - exp(-length(vWorld.xz) * 0.16);
    color = mix(color, envColor(normalize(vec3(vWorld.x, 1.0, vWorld.z))) * 0.85, fog);
  }

  color *= uExposure;
  // ACES-ish filmic curve, then back to sRGB. Reinhard alone leaves the sun's reflection a flat
  // white disc with no colour in it.
  color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14);
  outColor = vec4(pow(clamp(color, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
}
`;

/* -------------------------------------------------------------------------- */
/* Renderer                                                                    */
/* -------------------------------------------------------------------------- */

type MeshBuffers = { vao: WebGLVertexArrayObject; count: number; radius: number; minY: number };

/**
 * Sun direction, shared by the sky pass and the material so the highlight matches the disc.
 *
 * Stored as a **unit** vector, and that is not cosmetic. The IR documents `iLightDir` as "unit,
 * surface → light", and a graph is entitled to write `dot(iNormal, iLightDir)` and treat the
 * result as a cosine. `[0.55, 0.72, 0.42]` is the readable direction and has length 0.99865 —
 * close enough that nobody would ever see it, which is exactly why it would survive: every
 * toon ramp in the 3D view would be a fraction of a percent dark, forever, for no stated
 * reason. The unit conversion happens once, here.
 */
const SUN = unit([0.55, 0.72, 0.42]);

function unit(v: readonly [number, number, number]): readonly [number, number, number] {
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}

function compileShader(
  gl: WebGL2RenderingContext,
  kind: number,
  src: string,
): { shader: WebGLShader } | { error: string } {
  const shader = gl.createShader(kind);
  if (!shader) return { error: 'createShader returned null' };
  gl.shaderSource(shader, src);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = (gl.getShaderInfoLog(shader) ?? '').trim();
    gl.deleteShader(shader);
    return { error: log || 'compile failed with no log' };
  }
  return { shader };
}

export class SurfaceRenderer {
  readonly backend = 'webgl2' as const;

  private readonly canvas: HTMLCanvasElement;
  private readonly gl: WebGL2RenderingContext;
  private readonly materialProgram: WebGLProgram;
  private readonly skyProgram: WebGLProgram;
  private readonly emptyVao: WebGLVertexArrayObject;
  private readonly patternTexture: WebGLTexture;

  private readonly meshes = new Map<ShapeId, MeshBuffers>();
  private readonly uniformCache = new Map<string, WebGLUniformLocation | null>();
  private readonly skyUniformCache = new Map<string, WebGLUniformLocation | null>();
  private readonly graphUniformCache = new Map<string, WebGLUniformLocation | null>();

  /** The canvas the pattern backend draws into. Owned by the host; may be null before wiring. */
  private pattern: HTMLCanvasElement | null = null;

  /* ---- graph path ---------------------------------------------------------- */

  /** Non-null only while the live document reads a surface input. */
  private graphProgram: WebGLProgram | null = null;
  private graph: GraphMaterial | null = null;
  private graphSlots = new Map<string, ReflectedUniform>();
  /** Names the last write could not bind. Surfaced by `describe()` and `graphUnknownUniforms`. */
  private graphUnknown: string[] = [];
  /**
   * The stand-in for `iChannel0`.
   *
   * A graph may sample the channel texture, and the 3D pass has no reference image to give it.
   * A single opaque white texel is the honest placeholder: white is the identity for the
   * multiply that most channel uses are, so the effect stays visible instead of turning black
   * and looking like a bug in the emitter.
   */
  private readonly whiteTexture: WebGLTexture;
  /** Shadertoy convention: xy = latest, zw = press point. */
  private pointer: [number, number, number, number] = [0, 0, 0, 0];
  private frame = 0;

  private shape: ShapeId = 'sphere';
  private mode: SurfaceMode = 'material';
  private material: MaterialSettings = { ...DEFAULT_MATERIAL };

  private width = 1;
  private height = 1;
  private yaw = 0.6;
  private pitch = 0.34;
  private distance = 3.4;
  private autoRotate = true;
  private lastT: number | null = null;

  private constructor(
    canvas: HTMLCanvasElement,
    gl: WebGL2RenderingContext,
    materialProgram: WebGLProgram,
    skyProgram: WebGLProgram,
  ) {
    this.canvas = canvas;
    this.gl = gl;
    this.materialProgram = materialProgram;
    this.skyProgram = skyProgram;
    this.width = Math.max(1, canvas.width);
    this.height = Math.max(1, canvas.height);

    const vao = gl.createVertexArray();
    const tex = gl.createTexture();
    if (!vao || !tex) throw new Error('createVertexArray/createTexture failed');
    this.emptyVao = vao;
    this.patternTexture = tex;

    const white = gl.createTexture();
    if (!white) throw new Error('createTexture failed');
    this.whiteTexture = white;
    gl.bindTexture(gl.TEXTURE_2D, white);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      1,
      1,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      new Uint8Array([255, 255, 255, 255]),
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    gl.bindTexture(gl.TEXTURE_2D, tex);
    /*
     * REPEAT, and this is a bug fix rather than a preference.
     *
     * The material samples the pattern in **world** coordinates, which leave `[0, 1]` on every
     * object that is centred on the origin — that is not an edge case, it is most of the surface.
     * Under CLAMP_TO_EDGE those fragments returned the edge texel, stretched over everything that
     * mapped there: the pattern showed up only on the part of the object inside the unit square
     * (one corner of a cube) and the ground plane read as a cross of pattern content in a field of
     * four flat corner texels. Nothing errored, which is why it survived.
     *
     * REPEAT is also what the density slider already promises in its own words ("world units per
     * pattern tile; higher means the pattern repeats more"). The earlier comment here argued the
     * opposite — that tiling would make the slider's meaning depend on the mapping mode — but a
     * clamped sampler made it depend on the mapping mode *and* silently discard most of the
     * object, which is strictly worse. Tiles meet edge-to-edge under REPEAT, including through the
     * bilinear filter at a tile boundary, which no in-shader `fract` can do.
     *
     * Measured damage of the clamped version, as the fraction of each shape's vertices handed to
     * the edge clamp: sphere 71%, torus 74%, knot 75%, cube 73%, ground 98%.
     */
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.bindTexture(gl.TEXTURE_2D, null);

    gl.enable(gl.DEPTH_TEST);
    gl.depthFunc(gl.LEQUAL);
    gl.enable(gl.CULL_FACE);
    gl.cullFace(gl.BACK);
  }

  /**
   * Build the surface pass, or explain why not.
   *
   * Returns a note instead of throwing when a shader in *this file* fails to compile: the host
   * would otherwise report it through the same path as "the browser has no WebGL2", which is
   * the wrong diagnosis for a typo in a string here.
   */
  static create(canvas: HTMLCanvasElement): SurfaceChoice {
    const gl = canvas.getContext('webgl2', {
      antialias: true,
      premultipliedAlpha: false,
      // Required: `captureFrame()` reads the drawing buffer after the frame, which is
      // undefined behaviour without it — the same reason the flat renderer sets it.
      preserveDrawingBuffer: true,
    });
    if (!gl) return { surface: null, note: 'WebGL2 不可用：浏览器未提供 webgl2 上下文' };
    const ctx = gl as WebGL2RenderingContext;

    const sources: Array<[string, number, string]> = [
      ['material.vert', ctx.VERTEX_SHADER, VERT],
      ['material.frag', ctx.FRAGMENT_SHADER, MATERIAL_FRAG],
      ['sky.vert', ctx.VERTEX_SHADER, SKY_VERT],
      ['sky.frag', ctx.FRAGMENT_SHADER, SKY_FRAG],
    ];
    const compiled = new Map<string, WebGLShader>();
    const failures: string[] = [];
    for (const [name, kind, src] of sources) {
      const result = compileShader(ctx, kind, src);
      if ('error' in result) failures.push(`${name}: ${result.error}`);
      else compiled.set(name, result.shader);
    }
    if (failures.length) {
      return { surface: null, note: `3D 材质着色器编译失败（这是本文件的 bug）：\n${failures.join('\n')}` };
    }

    const link = (vsName: string, fsName: string): WebGLProgram | string => {
      const p = ctx.createProgram();
      if (!p) return 'createProgram returned null';
      ctx.attachShader(p, compiled.get(vsName) as WebGLShader);
      ctx.attachShader(p, compiled.get(fsName) as WebGLShader);
      ctx.linkProgram(p);
      if (!ctx.getProgramParameter(p, ctx.LINK_STATUS)) {
        return (ctx.getProgramInfoLog(p) ?? '').trim() || 'link failed with no log';
      }
      return p;
    };

    const material = link('material.vert', 'material.frag');
    const sky = link('sky.vert', 'sky.frag');
    const problems: string[] = [];
    if (typeof material === 'string') problems.push(`material link: ${material}`);
    if (typeof sky === 'string') problems.push(`sky link: ${sky}`);
    if (problems.length) {
      return { surface: null, note: `3D 材质着色器链接失败（这是本文件的 bug）：\n${problems.join('\n')}` };
    }

    return {
      surface: new SurfaceRenderer(canvas, ctx, material as WebGLProgram, sky as WebGLProgram),
    };
  }

  /* ---- configuration ------------------------------------------------------ */

  /** Point the material at the canvas the pattern backend draws into. */
  setPattern(canvas: HTMLCanvasElement | null): void {
    this.pattern = canvas;
  }

  /**
   * Hand the 3D pass the emitted **surface** shader, or `null` to go back to the pattern path.
   *
   * Returns a note instead of throwing when the shader does not compile. A compile failure here
   * is a *real* finding, not an internal bug: the source is `emit()`'s output, so a driver that
   * rejects it has found something the structural checks and `glslangValidator` did not. The
   * host is expected to put the log in front of the user, which is why the message carries the
   * driver's own words and the line numbers rather than a summary.
   *
   * On failure the pass falls back to the pattern path rather than keeping the previous
   * program: continuing to draw a *stale revision* would be a worse lie than drawing the wrong
   * material — a stale material looks like the edit had no effect.
   */
  setGraphShader(next: GraphMaterial | null): string | null {
    if (!next) {
      this.disposeGraphProgram();
      return null;
    }

    // Values change on every slider tick; the source does not. Rebuilding a program for a
    // value change would put a link (milliseconds) on the parameter hot path.
    if (this.graphProgram && this.graph?.source === next.source) {
      this.graph = next;
      return null;
    }

    const gl = this.gl;
    const vs = compileShader(gl, gl.VERTEX_SHADER, GRAPH_VERT);
    if ('error' in vs) {
      this.disposeGraphProgram();
      return `3D 网格顶点着色器编译失败（这是 preview/surface.ts 的 bug）：\n${vs.error}`;
    }
    const fs = compileShader(gl, gl.FRAGMENT_SHADER, next.source);
    if ('error' in fs) {
      gl.deleteShader(vs.shader);
      this.disposeGraphProgram();
      return (
        '图形着色（surface）在 3D 视图里编译失败 —— 发射器产出的 GLSL 被真实驱动拒绝了。' +
        `平面预览与导出不受影响，3D 视图已退回贴图映射：\n${fs.error}`
      );
    }

    const program = gl.createProgram();
    if (!program) {
      gl.deleteShader(vs.shader);
      gl.deleteShader(fs.shader);
      this.disposeGraphProgram();
      return 'createProgram 返回 null，3D 图形着色不可用';
    }
    gl.attachShader(program, vs.shader);
    gl.attachShader(program, fs.shader);
    gl.linkProgram(program);
    const linked = gl.getProgramParameter(program, gl.LINK_STATUS) as boolean;
    // The shader objects themselves are not needed after a successful link, and holding them
    // would leak one pair per edit.
    gl.deleteShader(vs.shader);
    gl.deleteShader(fs.shader);
    if (!linked) {
      const log = (gl.getProgramInfoLog(program) ?? '').trim() || 'link failed with no log';
      gl.deleteProgram(program);
      this.disposeGraphProgram();
      return `3D 图形着色链接失败 —— 顶点输出与 surface 片元的插值变量对不上：\n${log}`;
    }

    this.disposeGraphProgram();
    this.graphProgram = program;
    this.graph = next;
    this.graphSlots = reflectUniforms(gl, program);
    return null;
  }

  /** Slider tick: update the graph material's values without touching the program. */
  setGraphValues(values: UniformValues): void {
    if (this.graph) this.graph = { ...this.graph, values };
  }

  /** Pointer position in *pixels* with a top-left origin, matching the flat preview. */
  setPointer(x: number, y: number, down: boolean): void {
    this.pointer[0] = x;
    this.pointer[1] = y;
    if (down) {
      this.pointer[2] = x;
      this.pointer[3] = y;
    }
  }

  /** True while the mesh is shaded by the graph instead of by the baked pattern. */
  get graphShaded(): boolean {
    return this.graphProgram !== null;
  }

  /**
   * True when the graph actually reads `iMouse`, as reported by the **driver**.
   *
   * `iMouse` is declared in every emitted source, so its presence in the header says nothing.
   * Reflection reports only the uniforms the compiler kept, which means this flag is true
   * exactly when some node in the graph feeds the pointer into the result. The host uses it to
   * decide what a drag means — wiping the glass, or moving the camera — and it is the only
   * signal available that distinguishes those without inspecting the graph a second time.
   */
  get pointerFedToGraph(): boolean {
    return this.graphProgram !== null && this.graphSlots.has('iMouse');
  }

  /** Emission-order uniform names the graph shader optimised away or never declared. */
  get graphUnknownUniforms(): readonly string[] {
    return this.graphUnknown;
  }

  private disposeGraphProgram(): void {
    if (this.graphProgram) this.gl.deleteProgram(this.graphProgram);
    this.graphProgram = null;
    this.graph = null;
    this.graphSlots.clear();
    this.graphUniformCache.clear();
    this.graphUnknown = [];
  }

  setShape(shape: ShapeId): void {
    this.shape = shape;
    this.distance = Math.max(1.7, this.meshFor(shape).radius * 3.1);
  }

  setMode(mode: SurfaceMode): void {
    this.mode = mode;
  }

  setMaterial(next: Partial<MaterialSettings>): void {
    this.material = { ...this.material, ...next };
  }

  get materialSettings(): MaterialSettings {
    return { ...this.material };
  }

  setAutoRotate(on: boolean): void {
    this.autoRotate = on;
  }

  get autoRotating(): boolean {
    return this.autoRotate;
  }

  /** Drag-to-orbit. `dx`/`dy` are in pixels. */
  orbit(dx: number, dy: number): void {
    this.yaw -= dx * 0.01;
    // Clamped short of the poles: at exactly ±90° the view direction is parallel to the
    // up-vector, `lookAt` has no valid basis to build, and the mesh disappears in a NaN.
    this.pitch = Math.max(-1.45, Math.min(1.45, this.pitch + dy * 0.01));
  }

  /** Wheel zoom. `factor` > 1 moves the camera back. */
  zoom(factor: number): void {
    const r = this.meshFor(this.shape).radius;
    this.distance = Math.max(r * 1.25, Math.min(r * 14, this.distance * factor));
  }

  /** Reset the camera to the framing `setShape` would have chosen. */
  resetCamera(): void {
    this.yaw = 0.6;
    this.pitch = 0.34;
    this.distance = Math.max(1.7, this.meshFor(this.shape).radius * 3.1);
  }

  /**
   * One-line description of what is on screen, for the info panel and status bar.
   *
   * The two paths are described differently on purpose. Saying "密度 1.00 · 金属度 0.90" while
   * the graph path ignores all three would be describing controls that do nothing.
   */
  describe(): string {
    const m = this.material;
    if (this.graphShaded) {
      const extra = this.graphUnknown.length
        ? ` · ${this.graphUnknown.length} 个 uniform 未生效`
        : '';
      return (
        `${this.shape} · ${this.mode === 'scene' ? '场景' : '材质'} · ` +
        `图着色（surface，逐像素求值，无贴图烘焙）${extra}`
      );
    }
    return (
      `${this.shape} · ${this.mode === 'scene' ? '场景' : '材质'} · ` +
      `${m.mapping === 'uv' ? 'UV 映射' : '三平面映射'} · 密度 ${m.density.toFixed(2)} · ` +
      (m.seamless ? `接缝融合${m.mapping === 'uv' ? `（实际重复 ${Math.max(1, Math.floor(m.density + 0.5))} 次）` : ''} · ` : '') +
      `金属度 ${m.metalness.toFixed(2)} · 粗糙度 ${m.roughness.toFixed(2)}`
    );
  }

  /* ---- meshes -------------------------------------------------------------- */

  private meshFor(shape: ShapeId): MeshBuffers {
    const hit = this.meshes.get(shape);
    if (hit) return hit;

    const gl = this.gl;
    const data: MeshData = buildMesh(shape);
    const vao = gl.createVertexArray();
    if (!vao) throw new Error('createVertexArray failed');

    gl.bindVertexArray(vao);
    const upload = (array: Float32Array, location: number, size: number): void => {
      const buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, array, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(location);
      gl.vertexAttribPointer(location, size, gl.FLOAT, false, 0, 0);
    };
    upload(data.positions, 0, 3);
    upload(data.normals, 1, 3);
    upload(data.uvs, 2, 2);

    const indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, data.indices, gl.STATIC_DRAW);
    gl.bindVertexArray(null);

    const entry: MeshBuffers = { vao, count: data.indices.length, radius: data.radius, minY: data.minY };
    this.meshes.set(shape, entry);
    return entry;
  }

  /* ---- uniforms ------------------------------------------------------------ */

  private uniform(name: string): WebGLUniformLocation | null {
    if (!this.uniformCache.has(name)) {
      this.uniformCache.set(name, this.gl.getUniformLocation(this.materialProgram, name));
    }
    return this.uniformCache.get(name) ?? null;
  }

  private skyUniform(name: string): WebGLUniformLocation | null {
    if (!this.skyUniformCache.has(name)) {
      this.skyUniformCache.set(name, this.gl.getUniformLocation(this.skyProgram, name));
    }
    return this.skyUniformCache.get(name) ?? null;
  }

  private graphUniform(name: string): WebGLUniformLocation | null {
    if (!this.graphUniformCache.has(name)) {
      this.graphUniformCache.set(
        name,
        this.graphProgram ? this.gl.getUniformLocation(this.graphProgram, name) : null,
      );
    }
    return this.graphUniformCache.get(name) ?? null;
  }

  /* ---- draw ---------------------------------------------------------------- */

  render(t: number): void {
    const gl = this.gl;
    const dt = this.lastT === null ? 0 : Math.min(0.1, t - this.lastT);
    this.lastT = t;
    this.frame += 1;
    if (this.autoRotate) this.yaw += dt * 0.35;

    gl.viewport(0, 0, this.width, this.height);
    gl.clearColor(0.012, 0.014, 0.018, 1);
    gl.clearDepth(1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    const object = this.meshFor(this.shape);
    const radius = object.radius;
    const eye: [number, number, number] = [
      Math.cos(this.pitch) * Math.sin(this.yaw) * this.distance,
      Math.sin(this.pitch) * this.distance,
      Math.cos(this.pitch) * Math.cos(this.yaw) * this.distance,
    ];
    const view = lookAt(eye, [0, 0, 0], [0, 1, 0]);
    const near = Math.max(0.05, radius * 0.1);
    const far = Math.max(near + 1, this.distance + radius * 4);
    const viewProj = multiply(perspective(deg2rad(42), this.width / this.height, near, far), view);

    this.uploadPattern();
    this.drawSky(eye);
    this.drawScene(viewProj, eye, object, t, dt);

    gl.bindVertexArray(null);
  }

  /**
   * Re-upload the pattern canvas.
   *
   * Every frame, because the pattern animates: a cached upload would freeze the material while
   * the flat preview kept moving, which reads as a broken feature rather than a slow one. The
   * pattern canvas is deliberately small (`PATTERN_SIZE`) so this is a sub-millisecond
   * `texImage2D` rather than a megabyte-scale stall.
   */
  private uploadPattern(): void {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.patternTexture);
    // A canvas has its origin at the top-left like every DOM surface; sampling it without this
    // flip mirrors the pattern vertically on the model.
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    if (this.pattern && this.pattern.width > 0 && this.pattern.height > 0) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, this.pattern);
    }
  }

  private drawSky(eye: readonly [number, number, number]): void {
    const gl = this.gl;
    const halfFov = Math.tan(deg2rad(42) / 2);
    const aspect = this.width / this.height;

    // World-space camera basis. `forward` points from the eye at the origin, and the three
    // vectors are pre-scaled so the fragment shader is one multiply-add per pixel.
    const len = Math.hypot(eye[0], eye[1], eye[2]) || 1;
    const fx = -eye[0] / len;
    const fy = -eye[1] / len;
    const fz = -eye[2] / len;
    /*
     * right = normalize(cross(forward, worldUp)) — and the sign matters.
     *
     * `cross(forward, up)` is `(-fz, 0, fx)`; writing `(fz, 0, -fx)` gives the *left* vector
     * instead, which mirrors the sky horizontally. Nothing about that looks like a bug in
     * isolation — a gradient sky is symmetric, and only the sun's disc would end up on the
     * wrong side of a scene that is otherwise correct. `mat4.test.ts` pins the same convention
     * for `lookAt`, which computes `cross(up, z)` with `z = -forward` and must agree with this.
     */
    let rx = -fz;
    let rz = fx;
    const rlen = Math.hypot(rx, 0, rz) || 1;
    const ry = 0;
    rx /= rlen;
    const nrz = rz / rlen;
    // up = cross(right, forward)
    const ux = ry * fz - nrz * fy;
    const uy = nrz * fx - rx * fz;
    const uz = rx * fy - ry * fx;

    gl.disable(gl.DEPTH_TEST);
    gl.depthMask(false);
    gl.useProgram(this.skyProgram);
    gl.uniform3f(this.skyUniform('uRayX'), rx * halfFov * aspect, ry * halfFov * aspect, nrz * halfFov * aspect);
    gl.uniform3f(this.skyUniform('uRayY'), ux * halfFov, uy * halfFov, uz * halfFov);
    gl.uniform3f(this.skyUniform('uRayZ'), fx, fy, fz);
    gl.uniform3f(this.skyUniform('uSunDir'), SUN[0], SUN[1], SUN[2]);
    gl.uniform1f(this.skyUniform('uExposure'), this.material.exposure);
    gl.bindVertexArray(this.emptyVao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.depthMask(true);
    gl.enable(gl.DEPTH_TEST);
  }

  private drawScene(
    viewProj: Mat4,
    eye: readonly [number, number, number],
    object: MeshBuffers,
    t: number,
    dt: number,
  ): void {
    const gl = this.gl;

    /*
     * Seating the object on the floor.
     *
     * `minY`, not `radius`. A sphere reaches `-radius` and a torus reaches only `-0.28`, so
     * using the bounding radius as the depth would leave the torus and the knot floating a
     * third of a unit in the air — which reads as a shadow bug, not as arithmetic.
     *
     * `lift = -minY` puts the object's lowest point exactly at `y = 0`, and the floor is then
     * placed a hair *below* zero. Placing the floor by the same `minY` expression would be the
     * tempting symmetry and is wrong: it would move the ground with the object instead of
     * meeting it, dropping the sphere a full radius through the floor.
     */
    const lift = this.mode === 'scene' ? -object.minY : 0;
    const groundY = -Math.max(object.radius * 0.004, 0.002);

    /** Uniform writers bound to the pattern program. */
    const patternUniforms = (): void => {
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, this.patternTexture);
      gl.uniform1i(this.uniform('uPattern'), 0);
      gl.uniform3f(this.uniform('uCameraPos'), eye[0], eye[1], eye[2]);
      gl.uniform3f(this.uniform('uSunDir'), SUN[0], SUN[1], SUN[2]);
      gl.uniform1f(this.uniform('uDensity'), this.material.density);
      gl.uniform1f(this.uniform('uMetalness'), this.material.metalness);
      gl.uniform1f(this.uniform('uRoughness'), this.material.roughness);
      gl.uniform1f(this.uniform('uPatternMix'), this.material.patternMix);
      gl.uniform1f(this.uniform('uExposure'), this.material.exposure);
      gl.uniform1i(this.uniform('uMapping'), this.material.mapping === 'uv' ? 1 : 0);
      gl.uniform1i(this.uniform('uSeamless'), this.material.seamless ? 1 : 0);
      gl.uniformMatrix4fv(this.uniform('uViewProj'), false, viewProj);
    };

    const drawPattern = (shape: ShapeId, model: Mat4, isGround: boolean, patternMix: number): void => {
      const entry = this.meshFor(shape);
      gl.useProgram(this.materialProgram);
      patternUniforms();
      gl.uniformMatrix4fv(this.uniform('uModel'), false, model);
      gl.uniformMatrix3fv(this.uniform('uNormalMat'), false, normalMatrix3(model));
      gl.uniform1f(this.uniform('uIsGround'), isGround ? 1 : 0);
      gl.uniform1f(this.uniform('uPatternMix'), patternMix);
      // Contact shadow: centred on the object's footprint, so it is a property of the *ground*
      // draw. Writing it here rather than at the call site is what keeps it from being set after
      // the ground has already been drawn — which looks like a shadow lagging one frame behind.
      gl.uniform4f(this.uniform('uShadow'), 0, 0, object.radius * 1.3, isGround ? 0.7 : 0);
      gl.bindVertexArray(entry.vao);
      gl.drawElements(gl.TRIANGLES, entry.count, gl.UNSIGNED_INT, 0);
    };

    const drawGraph = (shape: ShapeId, model: Mat4): void => {
      const entry = this.meshFor(shape);
      gl.useProgram(this.graphProgram);
      /*
       * The three transform uniforms belong to the *vertex* stage and their types are known
       * here, so they are written directly rather than through reflection. Going through the
       * reflected table would make the object silently vanish if a driver ever reported the
       * program's uniforms in an unexpected order — a failure that looks like a camera bug.
       */
      gl.uniformMatrix4fv(this.graphUniform('uViewProj'), false, viewProj);
      gl.uniformMatrix4fv(this.graphUniform('uModel'), false, model);
      gl.uniformMatrix3fv(this.graphUniform('uNormalMat'), false, normalMatrix3(model));
      this.writeGraphUniforms(eye, t, dt);
      gl.bindVertexArray(entry.vao);
      gl.drawElements(gl.TRIANGLES, entry.count, gl.UNSIGNED_INT, 0);
    };

    if (this.graphShaded) {
      /*
       * The floor keeps the *pattern* material, at `patternMix = 0`.
       *
       * Shading the ground with the graph would be the one-line version of this function and
       * the wrong picture: a toon ramp or a rim-light term evaluated on an infinite plane turns
       * the floor into a second copy of the object's lighting, which reads as a bug in the
       * graph. The graph is the object's shading; the floor stays a neutral grey material so
       * the object has something to stand on that does not argue with it.
       */
      if (this.mode === 'scene') {
        drawPattern('ground', translation(0, groundY, 0), true, 0);
      }
      drawGraph(this.shape, translation(0, lift, 0));
      return;
    }

    if (this.mode === 'scene') {
      drawPattern('ground', translation(0, groundY, 0), true, this.material.patternMix);
    }
    drawPattern(this.shape, translation(0, lift, 0), false, this.material.patternMix);
  }

  /**
   * Every host uniform the emitted surface shader expects, plus the document's own sliders.
   *
   * Written through the reflected type, so a document that declares `iResolution` as `vec3` and
   * one that declares it as `vec2` both get a correct write. Names the shader optimised away are
   * collected rather than ignored — but only the *document's* uniforms are reported. A host
   * symbol the compiler dropped (`iChannel0` in a graph that never samples it) is the normal
   * case, and counting it would turn a healthy document into "1 个 uniform 未生效".
   */
  private writeGraphUniforms(
    eye: readonly [number, number, number],
    t: number,
    dt: number,
  ): void {
    const gl = this.gl;
    const material = this.graph;
    if (!material) return;

    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, this.whiteTexture);

    const host: Array<[string, UniformValue]> = [
      ['iResolution', [this.width, this.height, 1]],
      ['iTime', t],
      ['iTimeDelta', dt],
      ['iFrame', this.frame],
      ['iMouse', [this.pointer[0], this.pointer[1], this.pointer[2], this.pointer[3]]],
      ['iCameraPos', [eye[0], eye[1], eye[2]]],
      // `SUN` is the direction *towards* the light, which is exactly what the IR means by
      // `iLightDir` — the pattern material normalises the same constant for its key light, so
      // the graph's highlight and the sky's sun disc agree by construction.
      ['iLightDir', [SUN[0], SUN[1], SUN[2]]],
      // Texture unit 1 holds the 1×1 white stand-in, not the pattern canvas: the pattern *is*
      // this graph, and feeding it back would be a feedback loop that renders as a smear.
      ['iChannel0', 1],
    ];
    for (const [name, value] of host) {
      const slot = this.graphSlots.get(name);
      if (slot) writeUniformByGlType(gl, slot.glType, slot.location, value);
    }

    const unknown: string[] = [];
    for (const decl of material.uniforms) {
      const value = material.values[decl.name];
      const slot = this.graphSlots.get(decl.name);
      if (value === undefined || !slot) {
        unknown.push(decl.name);
        continue;
      }
      writeUniformByGlType(gl, slot.glType, slot.location, value);
    }
    this.graphUnknown = unknown;
  }


  resize(w: number, h: number): void {
    const width = Math.max(1, Math.round(w));
    const height = Math.max(1, Math.round(h));
    if (width === this.width && height === this.height) return;
    this.width = width;
    this.height = height;
    this.canvas.width = width;
    this.canvas.height = height;
    this.gl.viewport(0, 0, width, height);
  }

  /** RGBA bytes, flipped to top-left origin so the compare panel can diff them. */
  captureFrame(): CaptureResult {
    const gl = this.gl;
    const w = this.width;
    const h = this.height;
    const raw = new Uint8Array(w * h * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, raw);
    const out = new Uint8ClampedArray(w * h * 4);
    const stride = w * 4;
    for (let y = 0; y < h; y++) {
      out.set(raw.subarray((h - 1 - y) * stride, (h - y) * stride), y * stride);
    }
    return { width: w, height: h, data: out };
  }

  dispose(): void {
    const gl = this.gl;
    for (const mesh of this.meshes.values()) gl.deleteVertexArray(mesh.vao);
    this.meshes.clear();
    gl.deleteTexture(this.patternTexture);
    gl.deleteTexture(this.whiteTexture);
    gl.deleteProgram(this.materialProgram);
    gl.deleteProgram(this.skyProgram);
    this.disposeGraphProgram();
    gl.deleteVertexArray(this.emptyVao);
    this.uniformCache.clear();
    this.skyUniformCache.clear();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }
}
