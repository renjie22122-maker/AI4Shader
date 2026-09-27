/**
 * A recording WebGL2 context — for the handful of properties a headless suite can still prove
 *
 * ============================================================================
 *
 * There is no WebGL2 in this environment and no browser to borrow one from, so a test that
 * claimed to verify *rendering* would be lying. What this recorder verifies is the half that
 * actually breaks and that no compiler can see: the **host's use of the API**.
 *
 * The failures worth catching here are the ones that produce a plausible picture rather than an
 * error:
 *
 *   * a shader source that reaches the driver with something added or removed — "what you see is
 *     what you export" is a checked property, not a comment;
 *   * a feedback loop that samples the texture it is rendering into (undefined behaviour in GL:
 *     garbage, black, or a silently dropped draw, depending on the driver);
 *   * a ping-pong whose buffers are swapped at the wrong moment, which shows up only as an image
 *     that never advances;
 *   * an attachment that is never re-pointed after a resize, which renders at the old size.
 *
 * ## It throws on anything it does not implement
 *
 * Every element is wrapped in a proxy whose `get` throws for a method the recorder does not
 * model, naming it. That is the same rule as `dom-shim.ts` and for the same reason: a recorder
 * that silently accepts `gl.somethingNew()` cannot fail, and a check that cannot fail is
 * decoration. When this throws, the fix is to add the method **and an assertion about it** — the
 * error message says so.
 *
 * The GL constants are private values; only their identity matters, so they are arbitrary but
 * distinct, exactly like the real enums.
 */

export type Call = { name: string; args: unknown[] };

/** Parsed uniform declarations, used to model which uniforms a driver would keep. */
export function declaredUniforms(source: string): Map<string, string> {
  const out = new Map<string, string>();
  for (const m of source.matchAll(/^\s*uniform\s+(\w+)\s+(\w+)\s*(\[\s*\d+\s*\])?\s*;/gm)) {
    out.set(m[2] as string, m[1] as string);
  }
  return out;
}

/**
 * Is `name` used anywhere other than its own declaration?
 *
 * This is the crude stand-in for "the compiler kept it". It is deliberately conservative — a
 * name mentioned in a comment counts as used — because the property under test is *our* handling
 * of a uniform the driver dropped, and a simulation that drops too much would only make the test
 * stricter than the driver, never looser.
 */
export function isActive(source: string, name: string): boolean {
  const uses = source.split(new RegExp(`\\b${name}\\b`)).length - 1;
  const declarations = source.split(new RegExp(`uniform\\s+\\w+\\s+${name}\\b`)).length - 1;
  return uses > declarations;
}

export class RecordingGl {
  readonly calls: Call[] = [];
  /** Name -> last value written, across every program. */
  readonly written = new Map<string, unknown>();
  programsCreated = 0;
  programsDeleted = 0;

  /** Fragment sources the recorder is told to reject, keyed by a marker substring. */
  failOn: string | null = null;

  /* --- identity-tracked objects -------------------------------------------- */
  /** texture id -> size it was allocated at, or `[0,0]` for the 1×1 uploads. */
  readonly textureSizes = new Map<number, [number, number]>();
  /** framebuffer id -> the texture currently attached to COLOR_ATTACHMENT0. */
  readonly attachments = new Map<number, number | null>();

  private shaderSeq = 0;
  private readonly shaders = new Map<number, { kind: number; src: string }>();
  private readonly programs = new Map<number, { shaders: number[]; source: string }>();
  private programSeq = 0;
  private vaoSeq = 0;
  private bufferSeq = 0;
  private textureSeq = 0;
  private framebufferSeq = 0;

  /* --- constants (values are irrelevant; identity is not) ------------------ */
  readonly VERTEX_SHADER = 0x8b31;
  readonly FRAGMENT_SHADER = 0x8b30;
  readonly COMPILE_STATUS = 0x8b81;
  readonly LINK_STATUS = 0x8b82;
  readonly ACTIVE_UNIFORMS = 0x8b86;
  readonly FLOAT = 0x1406;
  readonly FLOAT_VEC2 = 0x8b50;
  readonly FLOAT_VEC3 = 0x8b51;
  readonly FLOAT_VEC4 = 0x8b52;
  readonly INT = 0x1404;
  readonly BOOL = 0x8b56;
  readonly SAMPLER_2D = 0x8b5e;
  readonly FLOAT_MAT3 = 0x8b5b;
  readonly FLOAT_MAT4 = 0x8b5c;
  readonly SAMPLER_CUBE = 0x8b60;
  readonly SAMPLER_3D = 0x8b5f;
  readonly SAMPLER_2D_ARRAY = 0x8b5d;
  readonly INT_VEC2 = 0x8b53;
  readonly INT_VEC3 = 0x8b54;
  readonly INT_VEC4 = 0x8b55;
  readonly BOOL_VEC2 = 0x8b57;
  readonly BOOL_VEC3 = 0x8b58;
  readonly BOOL_VEC4 = 0x8b59;
  readonly FLOAT_MAT2 = 0x8b5a;
  readonly TEXTURE_2D = 0x0de1;
  readonly TEXTURE0 = 0x84c0;
  readonly TEXTURE1 = 0x84c1;
  readonly RGBA = 0x1908;
  readonly UNSIGNED_BYTE = 0x1401;
  readonly UNSIGNED_INT = 0x1405;
  readonly TEXTURE_WRAP_S = 0x2802;
  readonly TEXTURE_WRAP_T = 0x2803;
  readonly TEXTURE_MIN_FILTER = 0x2801;
  readonly TEXTURE_MAG_FILTER = 0x2800;
  readonly CLAMP_TO_EDGE = 0x812f;
  readonly REPEAT = 0x2901;
  readonly LINEAR = 0x2601;
  readonly NEAREST = 0x2600;
  readonly DEPTH_TEST = 0x0b71;
  readonly CULL_FACE = 0x0b44;
  readonly BACK = 0x0405;
  readonly LEQUAL = 0x0203;
  readonly ARRAY_BUFFER = 0x8892;
  readonly ELEMENT_ARRAY_BUFFER = 0x8893;
  readonly STATIC_DRAW = 0x88e4;
  readonly TRIANGLES = 0x0004;
  readonly COLOR_BUFFER_BIT = 0x4000;
  readonly DEPTH_BUFFER_BIT = 0x0100;
  readonly FRAMEBUFFER = 0x8d40;
  readonly COLOR_ATTACHMENT0 = 0x8ce0;
  readonly FRAMEBUFFER_COMPLETE = 0x8cd5;
  readonly RENDERER = 0x1f01;
  readonly VERSION = 0x1f02;
  readonly SHADING_LANGUAGE_VERSION = 0x8b8c;
  readonly UNPACK_FLIP_Y_WEBGL = 0x9240;
  readonly MAX_TEXTURE_SIZE = 0x0d33;

  constructor() {
    return new Proxy(this, {
      get: (target, prop, receiver) => {
        if (typeof prop === 'symbol' || prop in target) {
          return Reflect.get(target, prop, receiver);
        }
        throw new Error(
          `RecordingGl has no ${String(prop)}: the code under test called a GL entry point this ` +
            `recorder does not model. Add it (and assert on it) rather than ignoring it.`,
        );
      },
    });
  }

  private record(name: string, ...args: unknown[]): void {
    this.calls.push({ name, args });
  }

  /* --- shaders ------------------------------------------------------------- */

  createShader(kind: number): number {
    const id = ++this.shaderSeq;
    this.shaders.set(id, { kind, src: '' });
    return id;
  }
  shaderSource(shader: number, src: string): void {
    const entry = this.shaders.get(shader);
    if (entry) entry.src = src;
  }
  compileShader(shader: number): void {
    this.record('compileShader', shader, this.shaders.get(shader)?.kind, this.shaders.get(shader)?.src);
  }
  getShaderParameter(shader: number, pname: number): boolean {
    if (pname !== this.COMPILE_STATUS) return false;
    const src = this.shaders.get(shader)?.src ?? '';
    return !(this.failOn && src.includes(this.failOn));
  }
  getShaderInfoLog(shader: number): string {
    const src = this.shaders.get(shader)?.src ?? '';
    if (this.failOn && src.includes(this.failOn)) {
      return `ERROR: 0:7: 'nonsense' : undeclared identifier (marker ${this.failOn})`;
    }
    return '';
  }
  deleteShader(shader: number): void {
    this.shaders.delete(shader);
  }

  /* --- programs ------------------------------------------------------------ */

  createProgram(): number {
    this.programsCreated += 1;
    const id = ++this.programSeq;
    this.programs.set(id, { shaders: [], source: '' });
    return id;
  }
  attachShader(program: number, shader: number): void {
    const entry = this.programs.get(program);
    if (entry) entry.shaders.push(shader);
  }
  linkProgram(program: number): void {
    const entry = this.programs.get(program);
    if (!entry) return;
    entry.source = entry.shaders.map((s) => this.shaders.get(s)?.src ?? '').join('\n');
    this.record('linkProgram', program);
  }
  getProgramParameter(program: number, pname: number): number | boolean {
    if (pname === this.LINK_STATUS) return true;
    if (pname === this.ACTIVE_UNIFORMS) {
      const source = this.programs.get(program)?.source ?? '';
      return [...declaredUniforms(source)].filter(([name]) => isActive(source, name)).length;
    }
    return 0;
  }
  getProgramInfoLog(): string {
    return '';
  }
  deleteProgram(program: number): void {
    this.programsDeleted += 1;
    this.programs.delete(program);
  }

  /** Source of the program created most recently. */
  get lastProgramSource(): string {
    return this.programs.get(this.programSeq)?.source ?? '';
  }

  getActiveUniform(program: number, index: number) {
    const source = this.programs.get(program)?.source ?? '';
    const active = [...declaredUniforms(source)]
      .filter(([name]) => isActive(source, name))
      .map(([name, type]) => ({ name, type }));
    const hit = active[index];
    return hit ? { name: hit.name, type: this.glTypeOf(hit.type), size: 1 } : null;
  }

  private glTypeOf(type: string): number {
    switch (type) {
      case 'float':
        return this.FLOAT;
      case 'vec2':
        return this.FLOAT_VEC2;
      case 'vec3':
        return this.FLOAT_VEC3;
      case 'vec4':
        return this.FLOAT_VEC4;
      case 'int':
        return this.INT;
      case 'bool':
        return this.BOOL;
      case 'mat3':
        return this.FLOAT_MAT3;
      case 'mat4':
        return this.FLOAT_MAT4;
      case 'sampler2D':
        return this.SAMPLER_2D;
      default:
        return this.FLOAT;
    }
  }

  getUniformLocation(program: number, name: string): string | null {
    const source = this.programs.get(program)?.source ?? '';
    return declaredUniforms(source).has(name) && isActive(source, name) ? `${program}:${name}` : null;
  }

  /* --- uniform writes ------------------------------------------------------ */

  private write(name: string, value: unknown): void {
    this.written.set(name, value);
    this.record('uniform', name, value);
  }
  private nameOf(loc: string | null): string {
    return typeof loc === 'string' ? loc.slice(loc.indexOf(':') + 1) : '';
  }
  uniform1f(loc: string | null, v: number): void {
    this.write(this.nameOf(loc), v);
  }
  uniform1i(loc: string | null, v: number): void {
    this.write(this.nameOf(loc), v);
  }
  uniform2fv(loc: string | null, v: number[]): void {
    this.write(this.nameOf(loc), [...v]);
  }
  uniform3fv(loc: string | null, v: number[]): void {
    this.write(this.nameOf(loc), [...v]);
  }
  uniform4fv(loc: string | null, v: number[]): void {
    this.write(this.nameOf(loc), [...v]);
  }
  uniform3f(loc: string | null, x: number, y: number, z: number): void {
    this.write(this.nameOf(loc), [x, y, z]);
  }
  uniform4f(loc: string | null, x: number, y: number, z: number, w: number): void {
    this.write(this.nameOf(loc), [x, y, z, w]);
  }
  uniformMatrix3fv(loc: string | null, _t: boolean, v: ArrayLike<number>): void {
    this.write(this.nameOf(loc), [...v]);
  }
  uniformMatrix4fv(loc: string | null, _t: boolean, v: ArrayLike<number>): void {
    this.write(this.nameOf(loc), [...v]);
  }

  /* --- textures and framebuffers ------------------------------------------- */

  createTexture(): number {
    const id = ++this.textureSeq;
    this.record('createTexture', id);
    return id;
  }
  deleteTexture(t: number): void {
    this.textureSizes.delete(t);
    this.record('deleteTexture', t);
  }
  bindTexture(target: number, texture: number | null): void {
    if (texture === null) this.boundTexture.delete(target);
    else this.boundTexture.set(target, texture);
    this.record('bindTexture', target, texture);
  }
  activeTexture(unit: number): void {
    this.record('activeTexture', unit);
  }
  texImage2D(...args: unknown[]): void {
    // Overloads: (target, level, ifmt, w, h, border, fmt, type, pixels) or the DOM upload form
    // (target, level, ifmt, fmt, type, source). Only the first carries a size, and a recorder
    // that guessed wrong would make "was this allocated at the new size?" unanswerable.
    const target = args[0] as number;
    if (args.length >= 9) {
      const [w, h] = [Number(args[3]), Number(args[4])];
      const bound = this.boundTexture.get(target);
      if (bound !== undefined) this.textureSizes.set(bound, [w, h]);
    }
    this.record('texImage2D', ...args);
  }
  texParameteri(target: number, pname: number, value: number): void {
    /*
     * The wrap mode is remembered *per texture*, because "which wrap mode" is only half the
     * question a test can ask: the other half is "on which texture". A recorder that kept one
     * global "last wrap mode" would happily report the flat renderer's clamped reference image
     * as the material's pattern wrap and pass a check it should fail.
     */
    const bound = this.boundTexture.get(target);
    if (bound !== undefined && (pname === this.TEXTURE_WRAP_S || pname === this.TEXTURE_WRAP_T)) {
      const entry = this.textureWrap.get(bound) ?? {};
      if (pname === this.TEXTURE_WRAP_S) entry.s = value;
      else entry.t = value;
      this.textureWrap.set(bound, entry);
    }
    this.record('texParameteri', target, pname, value);
  }
  pixelStorei(pname: number, value: number): void {
    this.record('pixelStorei', pname, value);
  }
  createFramebuffer(): number {
    const id = ++this.framebufferSeq;
    this.attachments.set(id, null);
    this.record('createFramebuffer', id);
    return id;
  }
  deleteFramebuffer(fb: number): void {
    this.attachments.delete(fb);
    this.record('deleteFramebuffer', fb);
  }
  bindFramebuffer(target: number, fb: number | null): void {
    this.boundFramebuffer = fb;
    this.record('bindFramebuffer', target, fb);
  }
  framebufferTexture2D(target: number, attachment: number, _textureTarget: number, texture: number | null, level: number): void {
    if (this.boundFramebuffer !== null) this.attachments.set(this.boundFramebuffer, texture);
    this.record('framebufferTexture2D', target, attachment, texture, level);
  }
  checkFramebufferStatus(): number {
    return this.FRAMEBUFFER_COMPLETE;
  }

  /** The texture currently bound to each texture target (unit 0 and beyond are not modelled). */
  private readonly boundTexture = new Map<number, number>();
  private boundFramebuffer: number | null = null;

  /** texture id -> the wrap modes configured on it, as far as they were set. */
  readonly textureWrap = new Map<number, { s?: number; t?: number }>();

  /** Which texture is bound to the unit a shader would sample. */
  get boundSampler(): number | null {
    return this.boundTexture.get(this.TEXTURE_2D) ?? null;
  }

  /** Which texture the current framebuffer is rendering into. */
  get renderTarget(): number | null {
    return this.boundFramebuffer === null ? null : (this.attachments.get(this.boundFramebuffer) ?? null);
  }

  /* --- state and draws ------------------------------------------------------ */

  createVertexArray(): number {
    this.record('createVertexArray');
    return ++this.vaoSeq;
  }
  deleteVertexArray(vao: number): void {
    this.record('deleteVertexArray', vao);
  }
  createBuffer(): number {
    this.record('createBuffer');
    return ++this.bufferSeq;
  }
  bindBuffer(target: number, buffer: number): void {
    this.record('bindBuffer', target, buffer);
  }
  bufferData(target: number, data: unknown, usage: number): void {
    this.record('bufferData', target, usage, ArrayBuffer.isView(data) ? data.length : 0);
  }
  enableVertexAttribArray(location: number): void {
    this.record('enableVertexAttribArray', location);
  }
  vertexAttribPointer(location: number, size: number, ...rest: unknown[]): void {
    this.record('vertexAttribPointer', location, size, ...rest);
  }
  viewport(x: number, y: number, w: number, h: number): void {
    this.record('viewport', x, y, w, h);
  }
  clearColor(r: number, g: number, b: number, a: number): void {
    this.record('clearColor', r, g, b, a);
  }
  clearDepth(d: number): void {
    this.record('clearDepth', d);
  }
  clear(mask: number): void {
    this.record('clear', mask);
  }
  enable(cap: number): void {
    this.record('enable', cap);
  }
  disable(cap: number): void {
    this.record('disable', cap);
  }
  depthFunc(fn: number): void {
    this.record('depthFunc', fn);
  }
  cullFace(mode: number): void {
    this.record('cullFace', mode);
  }
  depthMask(flag: boolean): void {
    this.record('depthMask', flag);
  }
  readPixels(...args: unknown[]): void {
    this.record('readPixels', ...args);
  }
  getParameter(pname: number): unknown {
    if (pname === this.RENDERER) return 'RecordingGl';
    if (pname === this.VERSION) return 'WebGL 2.0 (recording)';
    if (pname === this.SHADING_LANGUAGE_VERSION) return 'WebGL GLSL ES 3.00 (recording)';
    if (pname === this.MAX_TEXTURE_SIZE) return 4096;
    return null;
  }
  getError(): number {
    return 0;
  }
  useProgram(p: number): void {
    this.record('useProgram', p);
  }
  bindVertexArray(vao: unknown): void {
    this.record('bindVertexArray', vao);
  }
  drawElements(...args: unknown[]): void {
    // Same binding snapshot as `drawArrays`: a test about *what the mesh sampled* has to know
    // which texture and which wrap mode were live at the draw, not merely that a draw happened.
    this.record('drawElements', ...args, {
      sampler: this.boundSampler,
      target: this.renderTarget,
      framebuffer: this.boundFramebuffer,
      wrap: this.boundSampler === null ? undefined : this.textureWrap.get(this.boundSampler),
    });
  }
  /**
   * `drawArrays(mode, first, count)` — recorded together with what was bound at the time, which
   * is the only way a test can tell the graph draw from the blit.
   */
  drawArrays(...args: unknown[]): void {
    this.record('drawArrays', ...args, {
      sampler: this.boundSampler,
      target: this.renderTarget,
      framebuffer: this.boundFramebuffer,
    });
  }

  get drawCalls(): Call[] {
    return this.calls.filter((c) => c.name === 'drawElements' || c.name === 'drawArrays');
  }

  /** Draws of a *mesh*: the sky is a `drawArrays` over a generated triangle. */
  get meshDraws(): Call[] {
    return this.calls.filter((c) => c.name === 'drawElements');
  }

  /** Every shader source handed to `compileShader`, in order. */
  get compiledSources(): string[] {
    return this.calls.filter((c) => c.name === 'compileShader').map((c) => String(c.args[2] ?? ''));
  }

  /** The recorded binding state of every `drawArrays`, oldest first. */
  get trianglePasses(): { sampler: number | null; target: number | null; framebuffer: number | null }[] {
    return this.calls
      .filter((c) => c.name === 'drawArrays')
      .map((c) => c.args[3] as { sampler: number | null; target: number | null; framebuffer: number | null });
  }
}
