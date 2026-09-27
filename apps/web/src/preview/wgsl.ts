/**
 * WebGPU preview renderer (WGSL)
 * ==============================
 *
 * WGSL is the strictest of the six targets, so a browser WebGPU compile is the most
 * valuable validation signal the workbench can produce: real diagnostics with line AND
 * column, plus pipeline/bind-group validation through the error scopes.
 *
 * ── The uniform buffer problem, and how it is solved here ─────────────────────
 *
 * WGSL has no loose uniforms: everything lives in a struct behind one binding. The
 * emitter owns that struct, so the CPU-side byte layout must be derived from what the
 * emitter actually wrote. Rather than hard-coding offsets (which would silently break
 * the day a uniform is added), this renderer *reads the struct back out of the emitted
 * WGSL* and applies the WGSL uniform address-space layout rules:
 *
 *     scalar align 4 · vec2 align 8 · vec3/vec4 align 16 · struct align = max(members)
 *
 * Nothing in the shader text is rewritten or patched — it is parsed, not edited. If the
 * struct cannot be parsed, the renderer refuses to guess; see `layoutNote` and the
 * README's layout table for the documented fallback positions.
 *
 * ── Rendering ────────────────────────────────────────────────────────────────
 *
 * The emitted module declares `@fragment fn fs_main(...)`; the matching `@vertex`
 * entry point is supplied here (fullscreen triangle from `@builtin(vertex_index)`,
 * positions from a local array — no vertex buffer, no attributes). If pipeline or bind
 * group validation fails, `renderable` stays false and the UI falls back to WebGL2 with
 * an explicit message instead of drawing nothing.
 */

import { ProgramCache, hashSource } from './renderer.js';
import type {
  CompileDiagnostic,
  CompileResult,
  PreviewRenderer,
  UniformValue,
  UniformValues,
} from './renderer.js';
import type { UniformDecl } from '../../../../packages/ir/src/index.js';

/* -------------------------------------------------------------------------- */
/* WebGPU enum values (no @webgpu/types dependency)                            */
/* -------------------------------------------------------------------------- */

export const GPUBufferUsage = { MAP_READ: 0x0001, COPY_DST: 0x0008, UNIFORM: 0x0040 } as const;
export const GPUTextureUsage = {
  COPY_SRC: 0x0001,
  TEXTURE_BINDING: 0x0004,
  RENDER_ATTACHMENT: 0x0010,
} as const;

/* -------------------------------------------------------------------------- */
/* WGSL vertex entry point                                                     */
/* -------------------------------------------------------------------------- */

/**
 * Fullscreen triangle. Three vertices, clip-space (-1,-1) (3,-1) (-1,3).
 * The fragment stage receives `@builtin(position)` in *framebuffer* space (y down);
 * the emitter's epilogue performs the single Y flip the IR contract requires, so this
 * stage must not do one.
 */
export const WGSL_VERTEX_PRELUDE = `// --- supplied by the workbench (vertex stage) ---
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  var positions = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  return vec4f(positions[vi], 0.0, 1.0);
}
`;

/* -------------------------------------------------------------------------- */
/* WGSL reflection: bindings + uniform struct layout                           */
/* -------------------------------------------------------------------------- */

export type WgslBinding = {
  group: number;
  binding: number;
  varName: string;
  /** e.g. `Uniforms`, `texture_2d<f32>`, `sampler`. */
  type: string;
  kind: 'uniform' | 'texture' | 'sampler' | 'storage' | 'other';
};

export type WgslStructMember = {
  name: string;
  type: string;
  offset: number;
  size: number;
  align: number;
};

export type WgslStructLayout = {
  structName: string;
  members: WgslStructMember[];
  /** Size rounded up to the struct's alignment, as WGSL requires. */
  size: number;
};

export type WgslReflection = {
  bindings: WgslBinding[];
  uniforms?: WgslStructLayout;
  notes: string[];
};

/** WGSL type → (alignment, size). Only host-shareable types can appear in a uniform. */
const WGSL_TYPE_LAYOUT: Record<string, { align: number; size: number }> = {
  f32: { align: 4, size: 4 },
  i32: { align: 4, size: 4 },
  u32: { align: 4, size: 4 },
  f16: { align: 2, size: 2 },
  vec2f: { align: 8, size: 8 },
  vec2i: { align: 8, size: 8 },
  vec2u: { align: 8, size: 8 },
  vec3f: { align: 16, size: 12 },
  vec3i: { align: 16, size: 12 },
  vec3u: { align: 16, size: 12 },
  vec4f: { align: 16, size: 16 },
  vec4i: { align: 16, size: 16 },
  vec4u: { align: 16, size: 16 },
  mat2x2f: { align: 8, size: 16 },
  mat2x3f: { align: 16, size: 32 },
  mat3x2f: { align: 8, size: 24 },
  mat3x3f: { align: 16, size: 48 },
  mat4x4f: { align: 16, size: 64 },
};

/** `vec3<f32>` and `vec3f` are the same type spelled two ways. */
export function normalizeWgslType(type: string): string {
  const generic = /^vec([234])\s*<\s*(f32|i32|u32)\s*>$/.exec(type.replace(/\s+/g, ''));
  if (generic) {
    const suffix = generic[2] === 'f32' ? 'f' : generic[2] === 'i32' ? 'i' : 'u';
    return `vec${generic[1]}${suffix}`;
  }
  return type.replace(/\s+/g, '');
}

export function wgslTypeLayout(type: string): { align: number; size: number } | null {
  const norm = normalizeWgslType(type);
  const direct = WGSL_TYPE_LAYOUT[norm];
  if (direct) return direct;
  const arr = /^array<([^,]+),\s*(\d+)>$/.exec(norm);
  if (arr) {
    const inner = wgslTypeLayout(arr[1] ?? '');
    if (!inner) return null;
    // `array<T, N>` is only host-shareable when stride(T) == align(T) rounded to 16.
    const stride = Math.ceil(inner.size / 16) * 16;
    return { align: Math.max(16, inner.align), size: stride * Number(arr[2]) };
  }
  return null;
}

function roundUp(value: number, align: number): number {
  return Math.ceil(value / align) * align;
}

/** Split on a separator at bracket depth 0, so `array<f32, 4>` survives. */
function splitTopLevel(text: string, sep: string): string[] {
  const out: string[] = [];
  let depth = 0;
  let start = 0;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (c === '<' || c === '(' || c === '[') depth++;
    else if (c === '>' || c === ')' || c === ']') depth--;
    else if (c === sep && depth === 0) {
      out.push(text.slice(start, i));
      start = i + 1;
    }
  }
  out.push(text.slice(start));
  return out;
}

function stripWgslComments(src: string): string {
  return src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/[^\n]*/g, '');
}

/**
 * Parse a `struct Name { … }` declaration and lay it out per the WGSL uniform rules.
 * Returns `null` when the struct is absent or contains a type this renderer cannot
 * size — in which case the caller must not guess.
 */
export function parseWgslStruct(source: string, structName: string): WgslStructLayout | null {
  const clean = stripWgslComments(source);
  const head = new RegExp(`struct\\s+${structName}\\s*\\{`).exec(clean);
  if (!head) return null;
  let depth = 1;
  let end = -1;
  for (let i = head.index + head[0].length; i < clean.length; i++) {
    const c = clean[i];
    if (c === '{') depth++;
    else if (c === '}') {
      depth--;
      if (depth === 0) {
        end = i;
        break;
      }
    }
  }
  if (end < 0) return null;
  const body = clean.slice(head.index + head[0].length, end);

  const members: WgslStructMember[] = [];
  let offset = 0;
  let maxAlign = 1;
  for (const raw of splitTopLevel(body, ',')) {
    const text = raw.trim();
    if (!text) continue;
    const alignAttr = /@align\(\s*(\d+)\s*\)/.exec(text);
    const sizeAttr = /@size\(\s*(\d+)\s*\)/.exec(text);
    const decl = text.replace(/@[A-Za-z_]\w*(\([^)]*\))?/g, '').trim();
    const m = /^([A-Za-z_]\w*)\s*:\s*(.+)$/.exec(decl);
    if (!m) return null;
    const info = wgslTypeLayout(m[2] ?? '');
    if (!info) return null;
    const align = alignAttr ? Number(alignAttr[1]) : info.align;
    const size = sizeAttr ? Number(sizeAttr[1]) : info.size;
    offset = roundUp(offset, align);
    members.push({ name: m[1] ?? '', type: normalizeWgslType(m[2] ?? ''), offset, size, align });
    offset += size;
    maxAlign = Math.max(maxAlign, align);
  }
  return { structName, members, size: roundUp(offset, maxAlign) };
}

/**
 * Find every `@group(g) @binding(b) var…` declaration. Both attribute orders occur in
 * the wild, so both are matched.
 */
export function parseWgslBindings(source: string): WgslBinding[] {
  const clean = stripWgslComments(source);
  const out: WgslBinding[] = [];
  const patterns = [
    /@group\(\s*(\d+)\s*\)\s*@binding\(\s*(\d+)\s*\)\s*var(?:<([^>]*)>)?\s+([A-Za-z_]\w*)\s*:\s*([^;]+);/g,
    /@binding\(\s*(\d+)\s*\)\s*@group\(\s*(\d+)\s*\)\s*var(?:<([^>]*)>)?\s+([A-Za-z_]\w*)\s*:\s*([^;]+);/g,
  ];
  patterns.forEach((re, index) => {
    for (let m = re.exec(clean); m; m = re.exec(clean)) {
      const group = Number(index === 0 ? m[1] : m[2]);
      const binding = Number(index === 0 ? m[2] : m[1]);
      const addressSpace = (m[3] ?? '').trim();
      const varName = m[4] ?? '';
      const type = (m[5] ?? '').trim();
      let kind: WgslBinding['kind'] = 'other';
      if (addressSpace === 'uniform') kind = 'uniform';
      else if (type.startsWith('texture_')) kind = 'texture';
      else if (type.startsWith('sampler')) kind = 'sampler';
      else if (addressSpace === 'storage') kind = 'storage';
      out.push({ group, binding, varName, type, kind });
    }
  });
  return out.sort((a, b) => a.binding - b.binding);
}

/** Full reflection pass: bindings plus, when present, the uniform struct's layout. */
export function reflectWgsl(source: string): WgslReflection {
  const notes: string[] = [];
  const bindings = parseWgslBindings(source);
  const uniformBinding = bindings.find((b) => b.kind === 'uniform');
  let uniforms: WgslStructLayout | undefined;
  if (!uniformBinding) {
    notes.push('未找到 @group(0) @binding(0) var<uniform> 声明，无法绑定可调参数。');
  } else {
    const layout = parseWgslStruct(source, uniformBinding.type);
    if (layout) uniforms = layout;
    else notes.push(`无法解析 uniform 结构体 \`${uniformBinding.type}\`，可调参数不会被写入。`);
  }
  return { bindings, uniforms, notes };
}

/* -------------------------------------------------------------------------- */
/* Value writing                                                               */
/* -------------------------------------------------------------------------- */

function components(v: UniformValue, n: number): number[] {
  const src =
    typeof v === 'number' ? [v] : typeof v === 'boolean' ? [v ? 1 : 0] : Array.isArray(v) ? v : [0];
  const out: number[] = new Array<number>(n);
  for (let i = 0; i < n; i++) out[i] = Number(src[i] ?? 0);
  return out;
}

/** Write one uniform member into the staging buffer at its reflected offset. */
function writeMember(view: DataView, member: WgslStructMember, value: UniformValue): boolean {
  const o = member.offset;
  switch (member.type) {
    case 'f32':
      view.setFloat32(o, components(value, 1)[0] ?? 0, true);
      return true;
    case 'i32':
    case 'u32':
      view.setInt32(o, Math.round(components(value, 1)[0] ?? 0), true);
      return true;
    case 'vec2f':
    case 'vec2i':
    case 'vec2u': {
      const c = components(value, 2);
      for (let i = 0; i < 2; i++) view.setFloat32(o + i * 4, c[i] ?? 0, true);
      return true;
    }
    case 'vec3f':
    case 'vec3i':
    case 'vec3u': {
      const c = components(value, 3);
      for (let i = 0; i < 3; i++) view.setFloat32(o + i * 4, c[i] ?? 0, true);
      return true;
    }
    case 'vec4f':
    case 'vec4i':
    case 'vec4u': {
      const c = components(value, 4);
      for (let i = 0; i < 4; i++) view.setFloat32(o + i * 4, c[i] ?? 0, true);
      return true;
    }
    case 'mat2x2f': {
      const c = components(value, 4);
      for (let i = 0; i < 4; i++) view.setFloat32(o + i * 4, c[i] ?? 0, true);
      return true;
    }
    case 'mat3x3f': {
      const c = components(value, 9);
      for (let i = 0; i < 9; i++) view.setFloat32(o + i * 4, c[i] ?? 0, true);
      return true;
    }
    default:
      return false;
  }
}

/* -------------------------------------------------------------------------- */
/* Renderer                                                                    */
/* -------------------------------------------------------------------------- */

type CachedPipeline = {
  result: CompileResult;
  pipeline: GPURenderPipeline | null;
  bindGroup: GPUBindGroup | null;
  layout: WgslStructLayout | null;
  renderable: boolean;
};

export type WebGpuCreateResult = {
  renderer: WebGPURenderer | null;
  note?: string;
};

export class WebGPURenderer implements PreviewRenderer {
  readonly backend = 'webgpu' as const;

  private readonly canvas: HTMLCanvasElement;
  private readonly context: GPUCanvasContext;
  private readonly device: GPUDevice;
  private readonly format: GPUTextureFormat;
  private readonly cache = new ProgramCache<CachedPipeline>();

  private current: CachedPipeline | null = null;
  private uniformBuffer: GPUBuffer | null = null;
  private staging: ArrayBuffer = new ArrayBuffer(64);
  private stagingView = new DataView(this.staging);

  private width = 1;
  private height = 1;
  private frame = 0;
  private lastTime: number | null = null;
  private mouse: [number, number, number, number] = [0, 0, 0, 0];
  private unknown: string[] = [];
  /** Why WebGPU cannot draw, when it cannot. Surfaced verbatim in the UI. */
  private note = '';
  private adapterLabel = '';

  private constructor(
    canvas: HTMLCanvasElement,
    context: GPUCanvasContext,
    device: GPUDevice,
    format: GPUTextureFormat,
  ) {
    this.canvas = canvas;
    this.context = context;
    this.device = device;
    this.format = format;
    this.width = Math.max(1, canvas.width);
    this.height = Math.max(1, canvas.height);
    context.configure({ device, format, alphaMode: 'opaque' });
  }

  /** Never throws: a missing adapter/device is reported, and the caller falls back. */
  static async create(canvas: HTMLCanvasElement): Promise<WebGpuCreateResult> {
    const gpu = navigator.gpu;
    if (!gpu) return { renderer: null, note: 'WebGPU 不可用，已回退到 WebGL2' };
    try {
      const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) return { renderer: null, note: 'WebGPU 无可用适配器，已回退到 WebGL2' };
      const device = await adapter.requestDevice();
      const context = canvas.getContext('webgpu') as unknown as GPUCanvasContext | null;
      if (!context) return { renderer: null, note: 'canvas 无法创建 webgpu 上下文，已回退到 WebGL2' };
      const renderer = new WebGPURenderer(canvas, context, device, gpu.getPreferredCanvasFormat());
      const info = adapter.info;
      renderer.adapterLabel = info ? [info.vendor, info.architecture, info.device].filter(Boolean).join(' ') : '';
      return { renderer };
    } catch (err) {
      return { renderer: null, note: `WebGPU 初始化异常：${String(err)}；已回退到 WebGL2` };
    }
  }

  get adapterName(): string {
    return this.adapterLabel;
  }

  get layoutNote(): string {
    return this.note;
  }

  /** False while the last compile could not produce a drawable pipeline. */
  get renderable(): boolean {
    return this.current?.renderable === true;
  }

  get unknownUniforms(): readonly string[] {
    return this.unknown;
  }

  /* ---- compile ---------------------------------------------------------- */

  async compile(
    source: string,
    uniforms: UniformDecl[],
    options?: { feedback?: boolean },
  ): Promise<CompileResult> {
    const t0 = performance.now();
    /*
     * `render.feedback` is **not implemented on this backend**, and accepting the option is what
     * makes saying so possible. This path binds a 1×1 white texture to the image channel; a
     * document that asked for the previous frame would otherwise look like a shader that does
     * nothing, and the user would have no way to tell that the backend, not the document, is why.
     */
    const feedbackUnsupported = options?.feedback === true;
    const key = hashSource(source);
    const hit = this.cache.get(key);
    if (hit) {
      this.current = hit;
      if (!hit.renderable) this.note = this.note || '该 WGSL 之前未能通过管线校验（结果已缓存）';
      return { ...hit.result, ms: round2(performance.now() - t0), cached: true };
    }

    const device = this.device;
    const reflection = reflectWgsl(source);
    const preludeLines = WGSL_VERTEX_PRELUDE.split('\n').length - 1;
    const fullSource = `${WGSL_VERTEX_PRELUDE}\n${source}`;
    const diagnostics: CompileDiagnostic[] = [];

    device.pushErrorScope('validation');
    const module = device.createShaderModule({ code: fullSource, label: 'ai4shader-preview' });
    const info = await module.getCompilationInfo();
    for (const msg of info.messages) {
      diagnostics.push({
        severity: msg.type === 'error' ? 'error' : 'warning',
        // The module we compiled has our vertex prelude in front of it; report the
        // line number the user sees in the code panel, not the one WGSL saw.
        line: Math.max(1, msg.lineNum - preludeLines),
        column: msg.linePos,
        message: msg.message.trim(),
      });
    }

    const hasErrors = diagnostics.some((d) => d.severity === 'error');
    if (feedbackUnsupported) {
      diagnostics.push({
        severity: 'warning',
        message:
          'render.feedback = true，但 WebGPU 后端不实现反馈（iChannel0 绑定 1×1 白纹理）。' +
          '需要上一帧的效果请在 WebGL2 后端下预览；导出到宿主时由该宿主决定通道内容。',
      });
    }
    let pipeline: GPURenderPipeline | null = null;
    let bindGroup: GPUBindGroup | null = null;
    let renderable = false;

    if (!hasErrors) {
      try {
        pipeline = device.createRenderPipeline({
          label: 'ai4shader-preview',
          layout: 'auto',
          vertex: { module, entryPoint: 'vs_main' },
          fragment: { module, entryPoint: 'fs_main', targets: [{ format: this.format }] },
          primitive: { topology: 'triangle-list' },
        });
      } catch (err) {
        diagnostics.push({ severity: 'error', message: `创建渲染管线失败：${String(err)}` });
        pipeline = null;
      }

      if (pipeline) {
        const built = await this.buildBindGroup(pipeline, reflection, diagnostics);
        bindGroup = built.bindGroup;
        renderable = built.ok;
      }
    }

    const scopeError = await device.popErrorScope();
    if (scopeError) {
      diagnostics.push({ severity: 'error', message: `WebGPU 校验错误：${scopeError.message}` });
    }

    const ok = !hasErrors && renderable && !scopeError;
    let note = '';
    if (!ok && !hasErrors) {
      note = diagnostics.find((d) => d.severity === 'error')?.message ?? 'WebGPU 管线不可用';
    } else if (reflection.notes.length) {
      note = reflection.notes.join(' ');
    }

    const result: CompileResult = {
      ok,
      log: [
        `WebGPU / WGSL — 绑定: ${reflection.bindings.map((b) => `${b.binding}:${b.kind}`).join(', ') || '(none)'}`,
        reflection.uniforms
          ? `uniform 结构体 ${reflection.uniforms.structName}: ${reflection.uniforms.size} 字节\n` +
            reflection.uniforms.members
              .map((m) => `  +${String(m.offset).padStart(3, ' ')}  ${m.name}: ${m.type}  (size ${m.size}, align ${m.align})`)
              .join('\n')
          : 'uniform 结构体: 未解析',
        ...diagnostics.map((d) => `${d.severity === 'error' ? 'ERROR' : 'WARNING'}: ${d.line ?? '?'}:${d.column ?? '?'}: ${d.message}`),
      ].join('\n'),
      diagnostics,
      ms: round2(performance.now() - t0),
      cached: false,
    };

    const entry: CachedPipeline = { result, pipeline, bindGroup, layout: reflection.uniforms ?? null, renderable };
    this.cache.set(key, entry);
    this.current = entry;
    this.note = note;
    this.unknown = [];

    this.allocateBuffer(entry.layout);
    if (entry.renderable) for (const u of uniforms) this.write(u.name, u.value as UniformValue);
    return result;
  }

  /**
   * The pipeline is created with `layout: 'auto'`, so the bind group layout only
   * contains bindings the shader *statically uses*. Emitters commonly declare the
   * texture/sampler pair unconditionally, so entries are attempted from the widest
   * set downwards and the first combination that validates wins.
   */
  private async buildBindGroup(
    pipeline: GPURenderPipeline,
    reflection: WgslReflection,
    diagnostics: CompileDiagnostic[],
  ): Promise<{ ok: boolean; bindGroup: GPUBindGroup | null }> {
    const device = this.device;
    const layout = pipeline.getBindGroupLayout(0);
    const uniformBinding = reflection.bindings.find((b) => b.kind === 'uniform');
    const textureBinding = reflection.bindings.find((b) => b.kind === 'texture');
    const samplerBinding = reflection.bindings.find((b) => b.kind === 'sampler');
    if (!uniformBinding) return { ok: false, bindGroup: null };

    const all: GPUBindGroupEntry[] = [
      { binding: uniformBinding.binding, resource: { buffer: this.requireBuffer(8) } },
    ];
    if (textureBinding) {
      all.push({ binding: textureBinding.binding, resource: this.placeholderTexture().createView() });
    }
    if (samplerBinding) {
      all.push({
        binding: samplerBinding.binding,
        resource: device.createSampler({ magFilter: 'linear', minFilter: 'linear' }),
      });
    }

    const attempts: GPUBindGroupEntry[][] = [];
    if (all.length > 1) attempts.push(all, [all[0]!]);
    else attempts.push(all);

    for (const entries of attempts) {
      device.pushErrorScope('validation');
      let group: GPUBindGroup | null = null;
      try {
        group = device.createBindGroup({ label: 'ai4shader-preview', layout, entries });
      } catch {
        group = null;
      }
      const err = await device.popErrorScope();
      if (!err && group) return { ok: true, bindGroup: group };
      if (err) diagnostics.push({ severity: 'warning', message: `绑定组尝试失败（${entries.length} 项）：${err.message}` });
    }
    diagnostics.push({
      severity: 'error',
      message: 'WebGPU 绑定组无法创建：管线校验未通过，将回退到 WebGL2 渲染（WGSL 仍会做真实编译校验）。',
    });
    return { ok: false, bindGroup: null };
  }

  private placeholderTexture(): GPUTexture {
    const texture = this.device.createTexture({
      size: [1, 1, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    return texture;
  }

  private allocateBuffer(layout: WgslStructLayout | null): void {
    const size = Math.max(16, Math.ceil((layout?.size ?? 48) / 16) * 16);
    if (this.uniformBuffer && this.uniformBuffer.size >= size) {
      this.staging = new ArrayBuffer(size);
      this.stagingView = new DataView(this.staging);
      return;
    }
    this.uniformBuffer?.destroy();
    this.uniformBuffer = this.device.createBuffer({
      label: 'ai4shader-uniforms',
      size,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.staging = new ArrayBuffer(size);
    this.stagingView = new DataView(this.staging);
    this.mouse = [0, 0, 0, 0];
  }

  private requireBuffer(fallbackSize: number): GPUBuffer {
    if (!this.uniformBuffer) {
      this.uniformBuffer = this.device.createBuffer({
        label: 'ai4shader-uniforms',
        size: Math.max(16, Math.ceil(fallbackSize / 16) * 16),
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      });
    }
    return this.uniformBuffer;
  }

  /* ---- uniforms --------------------------------------------------------- */

  private write(name: string, value: UniformValue): boolean {
    const layout = this.current?.layout;
    if (!layout) return false;
    const member = layout.members.find((m) => m.name === name);
    if (!member) return false;
    return writeMember(this.stagingView, member, value);
  }

  setUniforms(values: UniformValues): boolean {
    if (!this.current?.layout) return false;
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

  setMouse(x: number, y: number, down: boolean): void {
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
    const buffer = this.uniformBuffer;
    if (!entry?.renderable || !entry.pipeline || !entry.bindGroup || !buffer) return;

    if (this.lastTime !== null && t < this.lastTime) {
      this.frame = 0;
      this.lastTime = null;
    }
    const dt = this.lastTime === null ? 0 : Math.max(0, t - this.lastTime);
    this.lastTime = t;

    if (entry.layout) {
      for (const [name, value] of [
        ['iResolution', [this.width, this.height]],
        ['iTime', t],
        ['iTimeDelta', dt],
        ['iFrame', this.frame],
        ['iMouse', [this.mouse[0], this.mouse[1], this.mouse[2], this.mouse[3]]],
      ] as [string, UniformValue][]) {
        this.write(name, value);
      }
    }

    this.device.queue.writeBuffer(buffer, 0, this.staging);
    this.frame += 1;

    const encoder = this.device.createCommandEncoder({ label: 'ai4shader-frame' });
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: this.context.getCurrentTexture().createView(),
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        },
      ],
    });
    pass.setPipeline(entry.pipeline);
    pass.setBindGroup(0, entry.bindGroup);
    pass.draw(3);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  resize(w: number, h: number): void {
    const width = Math.max(1, Math.round(w));
    const height = Math.max(1, Math.round(h));
    if (width === this.width && height === this.height) return;
    this.width = width;
    this.height = height;
    // The swap-chain follows the canvas element; no reconfigure is required.
    this.canvas.width = width;
    this.canvas.height = height;
  }

  get cacheSize(): number {
    return this.cache.size;
  }

  dispose(): void {
    this.uniformBuffer?.destroy();
    this.uniformBuffer = null;
    this.cache.clear();
    this.current = null;
    try {
      this.context.unconfigure();
    } catch {
      /* already gone */
    }
    this.device.destroy();
  }
}

function round2(n: number): number {
  return Math.round(n * 100) / 100;
}
