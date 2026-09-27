/**
 * WebGL2 preview renderer (GLSL ES 3.00)
 * ======================================
 *
 * The generated GLSL is a *fragment* program: it declares `void main()` and derives its
 * coordinates from `gl_FragCoord`. The vertex stage is therefore ours, and it is a
 * single fullscreen triangle generated from `gl_VertexID` — no vertex buffer, no
 * attributes, no interleaved layout to keep in sync with the emitter.
 *
 * Two things here are load-bearing for the rest of the workbench:
 *
 *  1. **Line numbers survive.** `#version`/`precision` are hoisted, but the original
 *     source text is preserved verbatim right after them, so a compiler diagnostic that
 *     says line 143 really is line 143 of the source shown in the code panel (only a
 *     constant offset is applied when a line had to be synthesised).
 *  2. **`preserveDrawingBuffer: true`.** Without it `readPixels` after the frame is
 *     undefined behaviour, and the reference-image scoring feature has nothing to read.
 */

import {
  ProgramCache,
  hashSource,
  reflectUniforms,
  writeUniformByGlType,
  type ReflectedUniform,
} from './renderer.js';
import type {
  CompileDiagnostic,
  CompileResult,
  PreviewRenderer,
  UniformValue,
  UniformValues,
} from './renderer.js';
import type { UniformDecl } from '../../../../packages/ir/src/index.js';

/* -------------------------------------------------------------------------- */
/* Shader sources                                                              */
/* -------------------------------------------------------------------------- */

/**
 * Fullscreen triangle. Vertex 0/1/2 -> (0,0), (2,0), (0,2) in "p * 2 - 1" space, i.e.
 * (-1,-1), (3,-1), (-1,3): one oversized triangle covering the whole clip square, which
 * is cheaper than a quad and has no diagonal seam.
 */
export const VERTEX_SHADER_SOURCE = `#version 300 es
precision highp float;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

/**
 * The blit: copy one texture to the canvas, no colour management.
 *
 * Used by the feedback path, where the graph is drawn into an offscreen texture (so it can be
 * sampled next frame) and then has to reach the screen. Deliberately not a tonemap, not a gamma
 * step, not a `#version`-less shortcut — anything clever here would mean the preview showed
 * something the exported shader does not compute.
 *
 * It has its **own** vertex stage rather than reusing the graph's, and that is not duplication
 * for its own sake: the graph's vertex shader writes no varyings, and a fragment stage that
 * declares `in vec2 vUv` against it would compile and then sample at undefined coordinates.
 * GLSL links that without a word (see IMPLEMENTATION §6.30).
 */
export const BLIT_VERTEX_SOURCE = `#version 300 es
precision highp float;
out vec2 vUv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

export const BLIT_FRAGMENT_SOURCE = `#version 300 es
precision highp float;
uniform sampler2D uBlit;
in vec2 vUv;
out vec4 outColor;
void main() {
  outColor = texture(uBlit, vUv);
}
`;

/**
 * Split the emitted GLSL into its preprocessor/precision header and the rest.
 *
 * The header is the *contiguous leading run* of `#version` / `precision` / `#extension`
 * lines (blank lines allowed). Reassembling `header + '\n' + body` reproduces the input
 * byte for byte, which is what keeps compiler line numbers honest.
 */
export function splitGlsl(source: string): { header: string; body: string } {
  const lines = source.split('\n');
  const header: string[] = [];
  let i = 0;
  for (; i < lines.length; i++) {
    const t = (lines[i] ?? '').trim();
    if (t === '' || t.startsWith('#version') || t.startsWith('precision') || t.startsWith('#extension')) {
      header.push(lines[i] ?? '');
    } else {
      break;
    }
  }
  return { header: header.join('\n'), body: lines.slice(i).join('\n') };
}

/**
 * Produce the fragment shader that will actually be compiled, plus the line offset a
 * compiler diagnostic must have subtracted to point back into the emitted source.
 *
 * The emitted source already starts with `#version 300 es` in practice, in which case
 * this is the identity function and `lineOffset` is 0.
 */
export function fragmentSource(source: string): { code: string; lineOffset: number } {
  const { header } = splitGlsl(source);
  const prefix: string[] = [];
  // `#version` must be line 1, so it can only be synthesised in front of everything.
  if (!/^[^\S\n]*#version\b/m.test(header)) prefix.push('#version 300 es');
  if (!/^[^\S\n]*precision\b/m.test(header)) prefix.push('precision highp float;');
  if (prefix.length === 0) return { code: source, lineOffset: 0 };
  return { code: `${prefix.join('\n')}\n${source}`, lineOffset: prefix.length };
}

/* -------------------------------------------------------------------------- */
/* Log parsing                                                                 */
/* -------------------------------------------------------------------------- */

/**
 * GLSL info logs are line-oriented: `ERROR: 0:143: 'x' : undeclared identifier`.
 * The middle field is the source-string index (always 0 here), the last the line.
 */
const LOG_LINE = /^(ERROR|WARNING|INFO):\s*(?:\d+):(\d+):\s*(.*)$/;

export function parseGlslLog(log: string, lineOffset = 0): CompileDiagnostic[] {
  const out: CompileDiagnostic[] = [];
  for (const raw of log.split('\n')) {
    const line = raw.trim();
    if (!line) continue;
    const m = LOG_LINE.exec(line);
    if (m) {
      const severity = m[1] === 'ERROR' ? 'error' : 'warning';
      const reported = Number(m[2]);
      out.push({ line: Math.max(1, reported - lineOffset), message: m[3] ?? '', severity });
    } else if (/^(ERROR|WARNING)/i.test(line)) {
      out.push({ severity: /^ERROR/i.test(line) ? 'error' : 'warning', message: line });
    }
  }
  return out;
}

/* -------------------------------------------------------------------------- */
/* Renderer                                                                    */
/* -------------------------------------------------------------------------- */

type UniformSlot = ReflectedUniform;

type CachedProgram = {
  /** Exactly what `compile()` returned the first time — including failures (§6.4). */
  result: CompileResult;
  program: WebGLProgram | null;
  slots: Map<string, UniformSlot>;
};

export class WebGL2Renderer implements PreviewRenderer {
  readonly backend = 'webgl2' as const;

  private readonly canvas: HTMLCanvasElement;
  private readonly gl: WebGL2RenderingContext;
  private readonly cache = new ProgramCache<CachedProgram>();

  private current: CachedProgram | null = null;
  private bound: WebGLProgram | null = null;

  private width = 1;
  private height = 1;
  private frame = 0;
  private lastTime: number | null = null;
  private mouse: [number, number, number, number] = [0, 0, 0, 0];

  private unknown: string[] = [];

  /* ---- feedback state ------------------------------------------------------ */

  /** Whether `render.feedback` asked for the previous frame in the image channel. */
  private feedback = false;
  /** Ping-pong pair. `targets[front]` holds the previous frame; the other is being written. */
  private targets: ({ texture: WebGLTexture } | null)[] = [null, null];
  private targetSize: [number, number] = [0, 0];
  private fbo: WebGLFramebuffer | null = null;
  private front = 0;
  private blitProgram: WebGLProgram | null = null;

  private constructor(canvas: HTMLCanvasElement, gl: WebGL2RenderingContext) {
    this.canvas = canvas;
    this.gl = gl;
    this.width = Math.max(1, canvas.width);
    this.height = Math.max(1, canvas.height);
    gl.viewport(0, 0, this.width, this.height);
  }

  static create(canvas: HTMLCanvasElement): WebGL2Renderer | null {
    const gl = canvas.getContext('webgl2', {
      antialias: false,
      premultipliedAlpha: false,
      // Required: the reference-image comparison reads the framebuffer back.
      preserveDrawingBuffer: true,
    });
    if (!gl) return null;
    return new WebGL2Renderer(canvas, gl as WebGL2RenderingContext);
  }

  /* ---- compile ---------------------------------------------------------- */

  async compile(source: string, uniforms: UniformDecl[], options?: { feedback?: boolean }): Promise<CompileResult> {
    const t0 = performance.now();
    const { code, lineOffset } = fragmentSource(source);
    const key = hashSource(code);

    /*
     * Feedback is a property of the *document*, not of the source text, and it is applied before
     * anything is drawn. A new program means the accumulation is meaningless (it belongs to the
     * previous shader), so compiling resets it — the alternative is a trail from a document the
     * user has already edited away.
     */
    const wantsFeedback = options?.feedback === true;
    if (wantsFeedback !== this.feedback) {
      this.setFeedback(wantsFeedback);
    } else if (wantsFeedback) {
      this.resetAccumulation();
    }

    const hit = this.cache.get(key);
    if (hit) {
      this.current = hit;
      this.bound = null;
      this.unknown = [];
      return { ...hit.result, ms: round2(performance.now() - t0), cached: true };
    }

    const gl = this.gl;
    const logs: string[] = [];
    const diagnostics: CompileDiagnostic[] = [];

    const vs = this.buildShader(gl.VERTEX_SHADER, VERTEX_SHADER_SOURCE);
    const fs = this.buildShader(gl.FRAGMENT_SHADER, code);
    logs.push(`--- vertex ---\n${vs.log}`.trimEnd(), `--- fragment ---\n${fs.log}`.trimEnd());
    diagnostics.push(
      ...vs.diagnostics.map((d) => ({ ...d, message: `[vertex] ${d.message}` })),
      ...fs.diagnostics.map((d) =>
        d.line === undefined ? d : { ...d, line: Math.max(1, d.line - lineOffset) },
      ),
    );

    let program: WebGLProgram | null = null;
    if (vs.shader && fs.shader) {
      program = gl.createProgram();
      gl.attachShader(program, vs.shader);
      gl.attachShader(program, fs.shader);
      gl.linkProgram(program);
      const linkLog = (gl.getProgramInfoLog(program) ?? '').trim();
      logs.push(linkLog ? `--- link ---\n${linkLog}` : '--- link ---\n(no messages)');
      if (!gl.getProgramParameter(program, gl.LINK_STATUS) || linkLog.length > 0) {
        // A non-empty link log is not necessarily fatal, but it always deserves surfacing.
        diagnostics.push(
          ...parseGlslLog(linkLog).map((d) => ({ ...d, message: `[link] ${d.message}` })),
        );
      }
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        gl.deleteProgram(program);
        program = null;
      }
    }

    const ok = program !== null && diagnostics.every((d) => d.severity !== 'error');
    const result: CompileResult = {
      ok,
      log: logs.join('\n\n').trim(),
      diagnostics,
      ms: round2(performance.now() - t0),
      cached: false,
    };

    let slots = new Map<string, UniformSlot>();
    if (program) {
      slots = this.reflect(program);
      // Diagnostics for tunables the linker dropped: the slider exists, the shader
      // does not use it. Silent in most tools; here it is stated out loud.
      for (const u of uniforms) {
        if (!slots.has(u.name)) {
          diagnostics.push({
            severity: 'warning',
            message: `uniform \`${u.name}\` 未出现在链接后的程序中（可能被优化掉或未被引用）`,
          });
        }
      }
    }

    const entry: CachedProgram = { result, program, slots };
    this.cache.set(key, entry);
    this.current = entry;
    this.bound = null;
    this.unknown = [];
    if (vs.shader) gl.deleteShader(vs.shader);
    if (fs.shader) gl.deleteShader(fs.shader);

    if (program) {
      this.bind(program);
      // Seed the program with the document's current values so the first frame drawn
      // is already the document, not a shader full of zeroes.
      for (const u of uniforms) this.write(u.name, u.value as UniformValue);
    }
    return result;
  }

  private buildShader(
    kind: number,
    src: string,
  ): { shader: WebGLShader | null; log: string; diagnostics: CompileDiagnostic[] } {
    const gl = this.gl;
    const shader = gl.createShader(kind);
    if (!shader) return { shader: null, log: '(createShader failed)', diagnostics: [] };
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    const log = (gl.getShaderInfoLog(shader) ?? '').trim();
    const ok = gl.getShaderParameter(shader, gl.COMPILE_STATUS) as boolean;
    const diagnostics = parseGlslLog(log);
    if (!ok) {
      gl.deleteShader(shader);
      if (diagnostics.length === 0) {
        diagnostics.push({ severity: 'error', message: log || '着色器编译失败（无日志）', line: 1 });
      }
      return { shader: null, log: log || '(compile failed)', diagnostics };
    }
    return { shader, log: log || '(no messages)', diagnostics };
  }

  /** Enumerate every *active* uniform with its driver-reported type. */
  private reflect(program: WebGLProgram): Map<string, UniformSlot> {
    return reflectUniforms(this.gl, program);
  }

  /* ---- uniforms --------------------------------------------------------- */

  private bind(program: WebGLProgram): void {
    if (this.bound === program) return;
    this.gl.useProgram(program);
    this.bound = program;
  }

  setUniforms(values: UniformValues): boolean {
    const entry = this.current;
    if (!entry?.program) return false;
    this.bind(entry.program);
    const unknown: string[] = [];
    let allKnown = true;
    for (const [name, value] of Object.entries(values)) {
      if (!this.write(name, value)) {
        allKnown = false;
        unknown.push(name);
      }
    }
    this.unknown = unknown;
    return allKnown;
  }

  /** @returns false when the program has no such uniform at all. */
  private write(name: string, value: UniformValue): boolean {
    const slot = this.current?.slots.get(name);
    if (!slot) return false;
    return writeUniformByGlType(this.gl, slot.glType, slot.location, value);
  }

  /** Union of the names the last `setUniforms()` could not bind. */
  get unknownUniforms(): readonly string[] {
    return this.unknown;
  }

  setMouse(x: number, y: number, down: boolean): void {
    // Shadertoy convention: xy = latest position, zw = press position (z<0 while held).
    this.mouse[0] = x;
    this.mouse[1] = y;
    if (down) {
      this.mouse[2] = x;
      this.mouse[3] = y;
    }
  }

  /* ---- draw ------------------------------------------------------------- */

  render(t: number): void {
    const entry = this.current;
    if (!entry?.program) return;
    const gl = this.gl;
    this.bind(entry.program);

    if (this.feedback) this.ensureTargets();
    if (this.lastTime !== null && t < this.lastTime) this.resetAccumulation();
    const dt = this.lastTime === null ? 0 : Math.max(0, t - this.lastTime);
    this.lastTime = t;

    // Host symbols, set every frame. Written by reflected type, so a target that
    // declares iResolution as vec2 or vec3 both work.
    this.write('iResolution', [this.width, this.height]);
    this.write('iTime', t);
    this.write('iTimeDelta', dt);
    this.write('iFrame', this.frame);
    this.write('iMouse', [this.mouse[0], this.mouse[1], this.mouse[2], this.mouse[3]]);

    if (this.feedback) {
      this.drawWithFeedback();
      this.frame += 1;
      return;
    }

    gl.viewport(0, 0, this.width, this.height);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.frame += 1;
  }

  /* ---- feedback (previous frame in the image channel) --------------------- */

  /**
   * Turn `render.feedback` on or off. Off is the default and the historical behaviour.
   *
   * Called from `compile()` with the document's own flag, so a host that never asks for feedback
   * never allocates the buffers and never pays for the extra pass.
   */
  setFeedback(on: boolean): void {
    if (on === this.feedback) return;
    this.feedback = on;
    // Turning it off releases the pair rather than leaving two render-sized textures pinned to a
    // renderer the document no longer uses. Turning it on clears both, because black is the state
    // a feedback shader is written against.
    if (on) this.resetAccumulation();
    else this.disposeTargets();
  }

  get feedbackEnabled(): boolean {
    return this.feedback;
  }

  /**
   * Feed the previous frame in through `iChannel0` while drawing into a *different* texture.
   *
   * This is the whole feature, and the two-texture ping-pong is not an optimisation — it is the
   * only legal way to write it. Sampling the texture you are rendering into is undefined
   * behaviour in GL: drivers either return garbage, return black, or (on some tilers) silently
   * drop the draw. A test asserts the property directly: the texture bound to the channel must
   * not be the texture attached to the framebuffer.
   *
   * Why a second pass instead of drawing straight to the canvas: the canvas cannot be sampled, so
   * the accumulation buffer has to be offscreen. The blit at the end is one textured triangle and
   * is also what makes `captureFrame()` keep working — it still reads the canvas.
   */
  private drawWithFeedback(): void {
    const gl = this.gl;
    this.ensureTargets();
    const front = this.targets[this.front];
    const back = this.targets[1 - this.front];
    if (!front || !back || !this.fbo) {
      throw new Error('feedback buffers were not allocated');
    }

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, back.texture, 0);
    gl.viewport(0, 0, this.width, this.height);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, front.texture);
    this.write('iChannel0', 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Blit the frame we just wrote to the visible canvas, then make it the "previous" frame.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.width, this.height);
    this.bindBlit();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, back.texture);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.front = 1 - this.front;
  }

  /** Allocate (or re-allocate) the ping-pong pair at the current size. */
  private ensureTargets(): void {
    const gl = this.gl;
    if (this.targets[0] && this.targetSize[0] === this.width && this.targetSize[1] === this.height) {
      return;
    }
    this.disposeTargets();
    for (let i = 0; i < 2; i++) {
      const texture = gl.createTexture();
      if (!texture) throw new Error('createTexture failed for the feedback buffer');
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, this.width, this.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      // CLAMP + LINEAR: the shader samples this with its own uv, which may run slightly outside
      // [0,1] at the edges. REPEAT there would wrap the opposite edge into the frame, which on a
      // wipe reads as a streak that comes back from the other side.
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      this.targets[i] = { texture };
    }
    this.targetSize = [this.width, this.height];
    if (!this.fbo) this.fbo = gl.createFramebuffer();
    this.front = 0;
    this.resetAccumulation();
  }

  /**
   * Clear both buffers to **black**, which is the state a feedback shader has to be written
   * against: `mix(previous, current, k)` fades in from black over the first frames, exactly as it
   * does in every other host that offers a feedback buffer. Clearing to white instead would make
   * a "wipe the fog away" shader start already wiped, which looks like the shader is broken.
   */
  resetAccumulation(): void {
    this.frame = 0;
    this.lastTime = null;
    const gl = this.gl;
    if (!this.targets[0]) return;
    if (!this.fbo) this.fbo = gl.createFramebuffer();
    for (let i = 0; i < 2; i++) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this.targets[i]!.texture, 0);
      gl.viewport(0, 0, this.width, this.height);
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    this.front = 0;
  }

  private bindBlit(): void {
    if (!this.blitProgram) {
      const vs = this.buildShader(this.gl.VERTEX_SHADER, BLIT_VERTEX_SOURCE);
      const fs = this.buildShader(this.gl.FRAGMENT_SHADER, BLIT_FRAGMENT_SOURCE);
      if (!vs.shader || !fs.shader) {
        throw new Error(`blit program failed to build: ${vs.log} ${fs.log}`);
      }
      const program = this.gl.createProgram();
      if (!program) throw new Error('createProgram failed for the blit');
      this.gl.attachShader(program, vs.shader);
      this.gl.attachShader(program, fs.shader);
      this.gl.linkProgram(program);
      this.gl.deleteShader(vs.shader);
      this.gl.deleteShader(fs.shader);
      if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
        throw new Error(`blit program failed to link: ${this.gl.getProgramInfoLog(program) ?? ''}`);
      }
      this.blitProgram = program;
    }
    // `bind` caches the last program, so routing the blit through it keeps the cache honest.
    this.bind(this.blitProgram);
  }

  private disposeTargets(): void {
    for (const target of this.targets) {
      if (target) this.gl.deleteTexture(target.texture);
    }
    this.targets = [null, null];
    this.targetSize = [0, 0];
    this.front = 0;
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
    // A resize invalidates the buffers; `ensureTargets` reallocates on the next frame, and the
    // accumulation restarts — losing a trail on resize is far better than sampling a
    // differently-sized texture, which renders as garbage rather than as a resize.
    this.disposeTargets();
  }

  /** RGBA bytes, flipped to top-left origin so they line up with `ImageData`. */
  captureFrame(): { width: number; height: number; data: Uint8ClampedArray } {
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

  /** Program cache stats, surfaced in the UI to make the caching visible. */
  get cacheSize(): number {
    return this.cache.size;
  }

  dispose(): void {
    const gl = this.gl;
    for (const key of this.cache.keys()) {
      const entry = this.cache.get(key);
      if (entry?.program) gl.deleteProgram(entry.program);
    }
    this.cache.clear();
    this.current = null;
    this.bound = null;
    this.disposeTargets();
    if (this.fbo) gl.deleteFramebuffer(this.fbo);
    this.fbo = null;
    if (this.blitProgram) gl.deleteProgram(this.blitProgram);
    this.blitProgram = null;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }
}

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}
