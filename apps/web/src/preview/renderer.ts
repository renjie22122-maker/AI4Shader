/**
 * Preview renderer contract
 * =========================
 *
 * Both real-GPU backends implement this. The interface is deliberately tiny and
 * *compile/uniform split* is the whole point (DESIGN.md §6.2):
 *
 *   compile()      expensive (5–50ms), only called when the graph or quality tier changes
 *   setUniforms()  free (<1ms), called on every slider tick
 *   render(t)      free-ish, called every frame
 *
 * A UI that calls compile() from a slider handler has thrown away the single most
 * important optimisation in the design. The parameter panel therefore only ever talks
 * to `setUniforms` + `render`.
 *
 * Compiled programs are keyed by a content hash of the source, so the same source is
 * compiled exactly once per session — including *failed* compiles (DESIGN.md §6.4),
 * because a broken document is usually dragged across several sliders before it is
 * fixed and re-compiling it every time is pure waste.
 */

import { stableHash } from '../ir.js';
// Types come from the authored sources: the projection is plain JavaScript, so it
// carries no type information. `import type` is erased, so this costs nothing at runtime.
import type { UniformDecl } from '../../../../packages/ir/src/index.js';

export type BackendId = 'webgl2' | 'webgpu';

export type CompileDiagnostic = {
  /** 1-based line in the *source that was handed to `compile()`*. */
  line?: number;
  column?: number;
  message: string;
  severity: 'error' | 'warning';
};

export type CompileResult = {
  ok: boolean;
  /** Raw compiler log, verbatim. Never paraphrase a driver message. */
  log: string;
  /** Parsed diagnostics with line numbers when the backend provides them. */
  diagnostics: CompileDiagnostic[];
  /** Wall-clock compile time in ms. Displayed in the UI — it is a selling point. */
  ms: number;
  /** True when this result came from the content-addressed cache instead of the driver. */
  cached?: boolean;
};

/** Values a slider (or the optimizer) can push into a live program. */
export type UniformValue = number | number[] | boolean;
export type UniformValues = Record<string, UniformValue>;

export interface PreviewRenderer {
  readonly backend: BackendId;
  /**
   * Compile from scratch. Returns real compiler diagnostics with line numbers.
   *
   * `options.feedback` comes from `render.feedback` and is a **host** instruction, not a shader
   * one: the source is identical either way. A backend that cannot provide a previous frame
   * ignores it — and must say so rather than silently binding a placeholder.
   */
  compile(
    source: string,
    uniforms: UniformDecl[],
    options?: { feedback?: boolean },
  ): Promise<CompileResult>;
  /**
   * Update uniform values WITHOUT recompiling.
   *
   * Both implementations return `false` when a name is unknown (the unpublished half of
   * this contract is typed `void` for signature compatibility); the returned flag is
   * surfaced to the UI through {@link unknownUniformsOf}.
   */
  setUniforms(values: UniformValues): void;
  /** Draw one frame at time t (seconds). */
  render(t: number): void;
  resize(w: number, h: number): void;
  dispose(): void;
}

/** A renderer that can hand back the pixels it just drew (used by the compare panel). */
export interface FrameCapture {
  /** RGBA, top-left origin (same orientation as `ImageData`), ready to diff. */
  captureFrame(): { width: number; height: number; data: Uint8ClampedArray };
}

/**
 * Does this object hand back the pixels it just drew?
 *
 * The parameter is `unknown` rather than `PreviewRenderer` because the workbench has two kinds
 * of thing that can be captured and only one of them compiles documents: the flat renderer, and
 * the 3D surface pass, which consumes the pattern as a texture and has no `compile()` at all.
 * Narrowing the input here is what lets the compare panel score *the picture on screen* in
 * either mode without pretending the surface pass is a shader compiler.
 */
export function supportsCapture(r: unknown): r is FrameCapture {
  return typeof (r as Partial<FrameCapture> | null | undefined)?.captureFrame === 'function';
}

/** Names the last `setUniforms()` call could not bind — usually a stale slider. */
export function unknownUniformsOf(r: PreviewRenderer): readonly string[] {
  return (r as Partial<{ unknownUniforms: readonly string[] }>).unknownUniforms ?? [];
}

/* -------------------------------------------------------------------------- */
/* Uniform reflection and writes                                               */
/* -------------------------------------------------------------------------- */

/** One driver-reported uniform: where to write it, and which GL type code it has. */
export type ReflectedUniform = {
  location: WebGLUniformLocation | null;
  glType: number;
};

/** Pads/truncates an incoming value to `n` components. */
export function uniformComponents(v: UniformValue, n: number): number[] {
  const src =
    typeof v === 'number' ? [v] : typeof v === 'boolean' ? [v ? 1 : 0] : Array.isArray(v) ? v : [0];
  const out: number[] = new Array<number>(n);
  for (let i = 0; i < n; i++) out[i] = Number(src[i] ?? 0);
  return out;
}

/**
 * Write one uniform **by the type the driver reported**, never by a type we assumed.
 *
 * This is the only place in the app that knows the mapping from a GL type code to a
 * `uniform*` call, and it is shared rather than copied. Two copies of this switch would
 * be two chances to disagree, and the failure mode is the worst kind: a `vec3` written
 * with `uniform1f` is not an error anywhere — WebGL silently writes the first component
 * and leaves the rest, so the shader runs and half the value is stale.
 *
 * @returns false when the type code is not something we can write (e.g. a struct).
 */
export function writeUniformByGlType(
  gl: WebGL2RenderingContext,
  glType: number,
  loc: WebGLUniformLocation | null,
  value: UniformValue,
): boolean {
  // Declared but unused: the driver optimised it out. Nothing to write, and not an error.
  if (!loc) return true;
  switch (glType) {
    case gl.FLOAT:
      gl.uniform1f(loc, uniformComponents(value, 1)[0] ?? 0);
      return true;
    case gl.FLOAT_VEC2:
      gl.uniform2fv(loc, uniformComponents(value, 2));
      return true;
    case gl.FLOAT_VEC3:
      gl.uniform3fv(loc, uniformComponents(value, 3));
      return true;
    case gl.FLOAT_VEC4:
      gl.uniform4fv(loc, uniformComponents(value, 4));
      return true;
    case gl.INT:
    case gl.BOOL:
    case gl.SAMPLER_2D:
    case gl.SAMPLER_CUBE:
    case gl.SAMPLER_2D_ARRAY:
    case gl.SAMPLER_3D:
      gl.uniform1i(loc, Math.round(uniformComponents(value, 1)[0] ?? 0));
      return true;
    case gl.INT_VEC2:
    case gl.BOOL_VEC2:
      gl.uniform2iv(loc, uniformComponents(value, 2).map(Math.round));
      return true;
    case gl.INT_VEC3:
    case gl.BOOL_VEC3:
      gl.uniform3iv(loc, uniformComponents(value, 3).map(Math.round));
      return true;
    case gl.INT_VEC4:
    case gl.BOOL_VEC4:
      gl.uniform4iv(loc, uniformComponents(value, 4).map(Math.round));
      return true;
    case gl.FLOAT_MAT2:
      gl.uniformMatrix2fv(loc, false, uniformComponents(value, 4));
      return true;
    case gl.FLOAT_MAT3:
      gl.uniformMatrix3fv(loc, false, uniformComponents(value, 9));
      return true;
    case gl.FLOAT_MAT4:
      gl.uniformMatrix4fv(loc, false, uniformComponents(value, 16));
      return true;
    default:
      return false;
  }
}

/**
 * Enumerate every *active* uniform of a linked program with its driver-reported type.
 *
 * Active, not declared: a uniform the compiler removed is not in this list, which is
 * exactly what makes `writeUniformByGlType` able to tell "unused" from "misspelled".
 */
export function reflectUniforms(
  gl: WebGL2RenderingContext,
  program: WebGLProgram,
): Map<string, ReflectedUniform> {
  const slots = new Map<string, ReflectedUniform>();
  const count = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS) as number;
  for (let i = 0; i < count; i++) {
    const info = gl.getActiveUniform(program, i);
    if (!info) continue;
    // Array uniforms report as `name[0]`; the base name is what a declaration uses.
    const name = info.name.replace(/\[0\]$/, '');
    slots.set(name, { location: gl.getUniformLocation(program, info.name), glType: info.type });
  }
  return slots;
}

/* -------------------------------------------------------------------------- */
/* Content-addressed program cache                                             */
/* -------------------------------------------------------------------------- */

/**
 * Content-addressed compile cache. Same source => same program, compiled once.
 *
 * A small FIFO cap keeps a long editing session from pinning every program it ever
 * built; the cap is generous enough that A/B-ing two revisions (DESIGN.md §6.4) always
 * hits, because two revisions are two entries.
 */
export class ProgramCache<T> {
  private readonly map = new Map<string, T>();

  constructor(private readonly maxEntries = 64) {}

  get(hash: string): T | undefined {
    return this.map.get(hash);
  }

  set(hash: string, v: T): void {
    if (this.map.has(hash)) this.map.delete(hash);
    this.map.set(hash, v);
    while (this.map.size > this.maxEntries) {
      const oldest = this.map.keys().next();
      if (oldest.done) break;
      this.map.delete(oldest.value);
    }
  }

  has(hash: string): boolean {
    return this.map.has(hash);
  }

  delete(hash: string): boolean {
    return this.map.delete(hash);
  }

  clear(): void {
    this.map.clear();
  }

  /** Snapshot of the keys, newest last. Used by `dispose()` to release GPU objects. */
  keys(): string[] {
    return [...this.map.keys()];
  }

  get size(): number {
    return this.map.size;
  }
}

/** Cache key for a shader source. Same function the IR uses, so keys never drift. */
export function hashSource(source: string): string {
  return stableHash(source);
}

/* -------------------------------------------------------------------------- */
/* Factory                                                                     */
/* -------------------------------------------------------------------------- */

export type RendererChoice = {
  renderer: PreviewRenderer | null;
  backend: BackendId;
  /** Human-readable reason when a backend was requested but is unavailable. */
  note?: string;
};

/** `navigator.gpu` presence check that does not require the WebGPU type definitions. */
export function webgpuSupported(): boolean {
  return typeof navigator !== 'undefined' && 'gpu' in navigator && navigator.gpu !== undefined;
}

/**
 * Build a renderer for the requested backend.
 *
 * Returns `null` (never throws) when the backend cannot be created — a missing
 * `navigator.gpu`, a browser that refuses a WebGL2 context, or a canvas that already
 * has a context of a different type. Callers fall back and *say so* in the UI.
 */
export async function createRenderer(
  canvas: HTMLCanvasElement,
  backend: BackendId = 'webgl2',
): Promise<RendererChoice> {
  if (backend === 'webgpu') {
    if (!webgpuSupported()) {
      return { renderer: null, backend: 'webgpu', note: 'WebGPU 不可用，已回退到 WebGL2' };
    }
    // Dynamic import keeps either backend out of the other's chunk and, more
    // importantly, out of the module-eval graph (no import cycle with this file).
    const { WebGPURenderer } = await import('./wgsl.js');
    const choice = await WebGPURenderer.create(canvas);
    if (!choice.renderer) {
      return { renderer: null, backend: 'webgpu', note: choice.note ?? 'WebGPU 初始化失败，已回退到 WebGL2' };
    }
    return { renderer: choice.renderer, backend: 'webgpu', note: choice.note };
  }

  const { WebGL2Renderer } = await import('./glsl.js');
  const renderer = WebGL2Renderer.create(canvas);
  if (!renderer) {
    return { renderer: null, backend: 'webgl2', note: 'WebGL2 不可用：浏览器未提供 webgl2 上下文' };
  }
  return { renderer, backend: 'webgl2' };
}
