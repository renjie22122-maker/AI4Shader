/**
 * AI4Shader workbench — application shell
 * =======================================
 *
 * Owns the state, the wiring, and exactly one performance rule (DESIGN.md §6.2):
 *
 *     拖滑杆 = setUniforms + render        ——  绝不 emit()、绝不 compile()
 *     改结构 = emit + compile（并缓存）    ——  只有这里才会重新编译
 *
 * Everything else is a view: the parameter panel, the code view and the compare panel
 * are all driven from `state`, and every one of them talks back through a narrow host
 * interface rather than reaching into each other.
 *
 * The one thing this file deliberately does NOT do is mutate the document silently.
 * The optimizer's output is applied as a *preview* (sliders move, pixels change, the
 * document hash does not), and only the explicit 采纳 button turns a suggestion into a
 * `set_param` patch through the IR patcher — DESIGN.md §9.
 *
 * Loaded as a plain ES module by `index.html` (no bundler): the stylesheet is a
 * `<link>` in the document rather than an `import './style.css'`, because a CSS import
 * is a bundler feature and this file has to run in a browser as-is.
 */

// Values come from the projected IR through the typed facade (`./ir.js`); types come
// from the authored sources, because JavaScript carries no type information.
import { IR_VERSION, analyzeGraph, applyPatch, emit, hashDoc } from './ir.js';
import type {
  Emission,
  ParamValue,
  ShaderDoc,
  TargetFormat,
  UniformDecl,
} from '../../../packages/ir/src/index.js';

import { CodegenPanel } from './panel/codegen.js';
import { ParamPanel } from './panel/params.js';
import { AuthorPanel } from './panel/author.js';
import {
  createRenderer,
  supportsCapture,
  unknownUniformsOf,
  webgpuSupported,
  type BackendId,
  type CompileResult,
  type FrameCapture,
  type PreviewRenderer,
  type UniformValue,
} from './preview/renderer.js';
import {
  DEFAULT_MATERIAL,
  PATTERN_SIZE,
  SurfaceRenderer,
  type MappingMode,
  type MaterialSettings,
  type SurfaceMode,
} from './preview/surface.js';
import { OBJECT_SHAPES, SHAPE_LABELS, type ShapeId } from './preview/mesh.js';
import {
  type Attribution,
  type ScoreBreakdown,
} from './compare/score.js';
import { demoDocForCatalog } from './doc/demo.js';
import { asEmitCatalog, loadCatalog, type CatalogLoad } from './doc/catalog.js';
import type { FramePayload, WorkerRequest, WorkerResponse } from './worker/score.worker.js';

/* -------------------------------------------------------------------------- */
/* Constants                                                                   */
/* -------------------------------------------------------------------------- */

/** Resolution the reference image and every optimizer evaluation are measured at. */
const COMPARE_SIZE = 256;
const MAX_TIME = 60;
const STEP = 1 / 60;

const TIER_RES: Record<'draft' | 'interactive', number> = { draft: 256, interactive: 512 };
type Tier = 'draft' | 'interactive' | 'quality';
type ResChoice = 256 | 512 | 1024 | 'window';

/* -------------------------------------------------------------------------- */
/* Tiny DOM helpers                                                            */
/* -------------------------------------------------------------------------- */

function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function must<T extends HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing #${id} in index.html`);
  return node as T;
}

function button(label: string, onClick: () => void, className = ''): HTMLButtonElement {
  const b = h('button', className, label);
  b.type = 'button';
  b.addEventListener('click', onClick);
  return b;
}

/* -------------------------------------------------------------------------- */
/* State                                                                       */
/* -------------------------------------------------------------------------- */

export type PreviewMode = 'flat' | SurfaceMode;

type AppState = {
  catalog: CatalogLoad;
  doc: ShaderDoc;
  docKey: string;
  docOrigin: ShaderDoc;
  variant: string;

  renderer: PreviewRenderer | null;
  backend: BackendId;
  emission: Emission | null;
  uniforms: UniformDecl[];
  values: Record<string, UniformValue>;
  lastCompile: CompileResult | null;

  /**
   * Preview mode, and everything the 3D half of it needs.
   *
   * `renderer` always draws the **pattern** — the emitted fragment shader — whichever mode is
   * active. What changes is where it draws: the visible canvas in `flat`, a detached 512²
   * canvas in the surface modes, where `surface` then samples it as a texture and lights a
   * mesh with it. Keeping one field for "the thing that compiles the document" is what lets
   * the parameter panel, the code view and the optimizer stay unaware that a 3D mode exists.
   */
  previewMode: PreviewMode;
  surface: SurfaceRenderer | null;
  /** The offscreen pattern canvas, non-null exactly when `surface` is. */
  patternCanvas: HTMLCanvasElement | null;
  surfaceShape: ShapeId;
  surfaceMaterial: MaterialSettings;

  time: number;
  playing: boolean;
  tier: Tier;
  qualityRes: ResChoice;
  forcedRes: number | null;
  dirty: boolean;
  fps: number;

  reference: { data: Uint8ClampedArray; w: number; h: number } | null;
  breakdown: ScoreBreakdown | null;
  attribution: Attribution | null;
  suggestion: Record<string, number> | null;
  suggestionLoss: number | null;
  perEvalMs: number | null;
};

const canvasHost = must<HTMLElement>('preview').parentElement ?? document.body;

const state: AppState = {
  catalog: { catalog: {}, present: [], missing: [], sources: [], targets: [], categories: [], notes: [] },
  doc: null as unknown as ShaderDoc,
  docKey: '',
  docOrigin: null as unknown as ShaderDoc,
  variant: '',
  renderer: null,
  backend: 'webgl2',
  emission: null,
  uniforms: [],
  values: {},
  lastCompile: null,
  previewMode: 'flat',
  surface: null,
  patternCanvas: null,
  surfaceShape: 'sphere',
  surfaceMaterial: { ...DEFAULT_MATERIAL },
  time: 0,
  playing: true,
  tier: 'interactive',
  qualityRes: 512,
  forcedRes: null,
  dirty: true,
  fps: 0,
  reference: null,
  breakdown: null,
  attribution: null,
  suggestion: null,
  suggestionLoss: null,
  perEvalMs: null,
};

let params: ParamPanel;
let codegen: CodegenPanel;
let author: AuthorPanel;
let worker: Worker | null = null;
let runBtn: HTMLButtonElement;
let stopBtn: HTMLButtonElement;
let progressBar: HTMLElement;
let progressText: HTMLElement;
let optimizeBudget: HTMLInputElement;

/* -------------------------------------------------------------------------- */
/* Banners                                                                     */
/* -------------------------------------------------------------------------- */

function banner(kind: 'error' | 'warn' | 'info', text: string, dismissible = true): void {
  const box = h('div', `banner banner-${kind}`, text);
  if (dismissible) box.append(button('关闭', () => box.remove(), 'mini'));
  must('banners').append(box);
}

/* -------------------------------------------------------------------------- */
/* Renderer lifecycle                                                          */
/* -------------------------------------------------------------------------- */

function currentCanvas(): HTMLCanvasElement {
  return must<HTMLCanvasElement>('preview');
}

/**
 * A canvas can only ever hand out one kind of context, so switching backends means
 * replacing the element. Cheap, and it keeps `createRenderer` honest.
 */
function replaceCanvas(): HTMLCanvasElement {
  const old = currentCanvas();
  const fresh = h('canvas');
  fresh.id = 'preview';
  fresh.width = old.width;
  fresh.height = old.height;
  attachPointerHandlers(fresh);
  old.replaceWith(fresh);
  return fresh;
}

function attachPointerHandlers(canvas: HTMLCanvasElement): void {
  let dragging = false;
  let lastX = 0;
  let lastY = 0;

  const send = (ev: PointerEvent, down: boolean): void => {
    const r = state.renderer as Partial<{ setMouse(x: number, y: number, down: boolean): void }> | null;
    const rect = canvas.getBoundingClientRect();
    const x = ((ev.clientX - rect.left) / Math.max(1, rect.width)) * canvas.width;
    // IR convention is GLSL-like (origin bottom-left), so flip from pointer space.
    const y = canvas.height - ((ev.clientY - rect.top) / Math.max(1, rect.height)) * canvas.height;
    r?.setMouse?.(x, y, down);
    // The 3D pass gets the same pointer, in pixels, for graphs that read `iMouse` directly —
    // "wipe the fog off the glass" has to work on the object, not only on the fullscreen quad.
    state.surface?.setPointer(x, y, down);
  };

  /*
   * One handler, two meanings — and in surface mode, three.
   *
   * In `flat` the pointer is forwarded to the document as `iMouse`. In a surface mode a drag
   * orbits the camera. When the *graph itself* reads `iMouse` (which the driver confirms, see
   * `SurfaceRenderer.pointerFedToGraph`) a plain drag would otherwise do two things at once —
   * wipe the surface and swing the camera under the user's finger, which makes an interaction
   * impossible to aim. So the pointer wins the plain drag, and the camera moves to
   * Shift+drag / right-drag, which is stated in the material strip.
   */
  const wantsOrbit = (ev: PointerEvent, graphPointer: boolean): boolean =>
    graphPointer ? ev.shiftKey || (ev.buttons & 6) !== 0 : dragging;

  canvas.addEventListener('pointermove', (ev) => {
    const surface = state.surface;
    if (!surface) {
      send(ev, ev.buttons !== 0);
      return;
    }
    const graphPointer = surface.pointerFedToGraph;
    if (graphPointer) send(ev, (ev.buttons & 1) !== 0 && !ev.shiftKey);
    if (!wantsOrbit(ev, graphPointer)) {
      // Keep the anchor current while not orbiting, so switching to Shift mid-drag does not
      // jump the camera by the distance the pointer travelled before the switch.
      lastX = ev.clientX;
      lastY = ev.clientY;
      return;
    }
    surface.orbit(ev.clientX - lastX, ev.clientY - lastY);
    lastX = ev.clientX;
    lastY = ev.clientY;
    state.dirty = true;
  });
  canvas.addEventListener('pointerdown', (ev) => {
    if (state.surface) {
      if (state.surface.pointerFedToGraph) {
        send(ev, true);
        state.dirty = true;
        return;
      }
      dragging = true;
      lastX = ev.clientX;
      lastY = ev.clientY;
      return;
    }
    send(ev, true);
  });
  const release = (ev: PointerEvent): void => {
    if (state.surface) {
      dragging = false;
      if (state.surface.pointerFedToGraph) {
        send(ev, false);
        state.dirty = true;
      }
      return;
    }
    send(ev, false);
  };
  canvas.addEventListener('pointerup', release);
  canvas.addEventListener('pointercancel', release);
  canvas.addEventListener(
    'wheel',
    (ev: WheelEvent) => {
      const surface = state.surface;
      if (!surface) return;
      ev.preventDefault();
      surface.zoom(ev.deltaY > 0 ? 1.12 : 1 / 1.12);
      state.dirty = true;
    },
    { passive: false },
  );
}

/**
 * A detached canvas for the pattern when a surface mode is active.
 *
 * It is a *texture*, not a picture, so it does not need the display resolution: 512² is
 * uploaded every frame, and rendering the pattern at 1024² instead would quadruple both the
 * pattern pass and the upload for detail the material cannot resolve anyway.
 */
function makePatternCanvas(): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = PATTERN_SIZE;
  c.height = PATTERN_SIZE;
  return c;
}

async function setupRenderer(backend: BackendId): Promise<void> {
  state.renderer?.dispose();
  state.renderer = null;
  state.surface?.dispose();
  state.surface = null;
  state.patternCanvas = null;
  replaceCanvas();

  /*
   * Which canvas the pattern backend draws into is decided here and nowhere else.
   *
   *   flat      — the visible canvas, exactly as it always was.
   *   material  — `makePatternCanvas()`, with the visible canvas handed to the surface pass.
   *
   * The branch is on `previewMode`, so a mode change is a renderer rebuild. That is heavier
   * than it looks like it should be and it is still the right call: a canvas can only ever
   * hand out one kind of context, so "switch what the pixels mean" already meant "replace the
   * element" (see `replaceCanvas`). Making the mode part of the same rebuild keeps one path
   * instead of two.
   *
   * The mode is re-derived from the document first. `setDoc` is the other place that does this,
   * and both exist on purpose: a rebuild also happens without a document change (a backend
   * switch, a resize that replaces the canvas), and a surface document must not quietly land on
   * a fullscreen quad through one of those paths.
   */
  state.previewMode = previewModeFor(surfaceNeedsOf(state.doc), state.previewMode);
  const surfaceMode: SurfaceMode | null = state.previewMode === 'flat' ? null : state.previewMode;
  const patternTarget = surfaceMode ? makePatternCanvas() : currentCanvas();
  if (surfaceMode) state.patternCanvas = patternTarget;

  const choice = await createRenderer(patternTarget, backend);
  if (!choice.renderer) {
    if (backend === 'webgpu') {
      if (choice.note) banner('warn', choice.note);
      return setupRenderer('webgl2');
    }
    banner('error', choice.note ?? '无法创建渲染器：浏览器没有提供 WebGL2 上下文。');
    // Still fill the info card. Most of its rows — document, IR version, node and target
    // counts, doc hash — do not need a renderer at all, and leaving the card blank next to
    // an error banner makes a diagnosable problem look like a dead app.
    updateInfo();
    return;
  }

  state.renderer = choice.renderer;
  state.backend = choice.backend;
  if (choice.note) banner('info', choice.note, false);

  if (surfaceMode) {
    const built = SurfaceRenderer.create(currentCanvas());
    if (!built.surface) {
      /*
       * Drop to flat rather than leaving a blank canvas.
       *
       * The surface pass failing means a bug in `preview/surface.ts` (its shaders are string
       * constants compiled right here) or a browser without WebGL2 at all — and in both cases
       * the mode that is known to work should be the one on screen, with the reason stated.
       */
      banner('error', `${built.note ?? '3D 材质预览初始化失败'}\n已切回「平面」预览。`);
      state.previewMode = 'flat';
      state.patternCanvas = null;
      syncToolbar();
      return setupRenderer('webgl2');
    }
    state.surface = built.surface;
    state.surface.setPattern(state.patternCanvas);
    state.surface.setShape(state.surfaceShape);
    state.surface.setMode(surfaceMode);
    state.surface.setMaterial(state.surfaceMaterial);
    // Tied to the play button: pausing should freeze the material as well as the pattern,
    // because "paused" meaning "the camera keeps drifting" is a surprise.
    state.surface.setAutoRotate(state.playing);
  }

  await recompileAll();
  updateInfo();
}

/** True when the active renderer can actually draw (WebGPU can fall back to compile-only). */
function rendererDraws(): boolean {
  const r = state.renderer as Partial<{ renderable: boolean }> | null;
  return !!state.renderer && (r?.renderable ?? true);
}

/* -------------------------------------------------------------------------- */
/* Emit + compile (the expensive path)                                         */
/* -------------------------------------------------------------------------- */

function previewTarget(): TargetFormat {
  return state.backend === 'webgpu' ? 'wgsl' : 'glsl';
}

/** Re-run `emit` + `compile` for the preview target. Called ONLY on structural change. */
async function recompileAll(): Promise<void> {
  const renderer = state.renderer;
  if (!renderer) return;

  const target = previewTarget();
  let emission: Emission;
  try {
    emission = emit(state.doc, { catalog: asEmitCatalog(state.catalog.catalog), target });
  } catch (err) {
    setCompileStrip(null, `emit(${target}) 抛出异常：${err instanceof Error ? err.message : String(err)}`);
    return;
  }

  state.emission = emission;
  state.uniforms = emission.uniforms;
  for (const u of emission.uniforms) {
    if (!(u.name in state.values)) state.values[u.name] = u.value as UniformValue;
  }

  // Rebuild the sliders from the emission (this is where node provenance comes from).
  params.setUniforms(emission.uniforms);
  for (const [name, value] of Object.entries(state.values)) params.setValue(name, value);

  const result = await renderer.compile(emission.source, emission.uniforms, {
    // `render.feedback` is the document telling the host what to bind in the image channel.
    // Only the flat path implements it; the surface pass reports the stand-in it uses instead
    // (see `channelNote`).
    feedback: emission.needs.channel === true && state.doc.render.feedback === true,
  });
  state.lastCompile = result;
  setCompileStrip(result, null, emission);

  syncSurfaceMaterial(emission, target);

  if (state.backend === 'webgpu' && !rendererDraws()) {
    banner(
      'warn',
      'WebGPU 管线校验未通过或绑定组不可用：WGSL 已完成真实编译校验，但渲染回退到 WebGL2。',
    );
    await setupRenderer('webgl2');
    return;
  }

  renderNow();
  renderScoreUI();
  updateInfo();
}

/** Structural change: new document ⇒ new hash ⇒ code view re-emits lazily. */
async function setDoc(next: ShaderDoc, reason: string): Promise<void> {  state.doc = next;
  state.docKey = next.hash ?? hashDoc(next);
  codegen.setDoc(next, state.docKey);

  /*
   * A document that reads surface inputs cannot be shown on a fullscreen quad, and the failure
   * is silent: GLSL allows a fragment input with no matching vertex output, so the flat preview
   * *compiles* and then reads undefined varyings. The user sees a black or nonsense image with
   * "0 errors" in the strip — the worst kind of wrong. So the mode follows the document.
   *
   * The decision is taken here rather than inside `recompileAll` because changing the mode means
   * rebuilding the renderer (a canvas hands out one kind of context), which is exactly the path
   * a manual mode switch takes.
   */
  const better = previewModeFor(surfaceNeedsOf(next), state.previewMode);
  if (better !== state.previewMode) {
    state.previewMode = better;
    banner(
      'info',
      `这个文档读表面输入（法线 / 世界坐标 / 视线 / 光向），平面预览无法表达它 —— ` +
        `已切到「${better === 'scene' ? '场景' : '材质'}」预览。`,
      false,
    );
    syncToolbar();
    // `setupRenderer` ends with `recompileAll`, so the document is still emitted exactly once.
    await setupRenderer(state.backend);
  } else {
    await recompileAll();
  }
  setStatus(`${reason} · doc ${state.docKey.slice(0, 8)}`);
}

/**
 * The preview mode a document can actually be shown in.
 *
 * `flat` draws one fullscreen quad, which has no normal, no world position and no mesh UV — so
 * a surface document has nothing to shade there. The other two modes always work (they fall back
 * to the texture path if the driver rejects the emitted shader), which is why the promotion is
 * one-way and never downgrades a mode the user chose deliberately.
 *
 * Pure and exported: the interesting part is the decision, and a decision buried inside an async
 * renderer rebuild is a decision nothing can test.
 */
export function previewModeFor(needsSurface: boolean, current: PreviewMode): PreviewMode {
  return needsSurface && current === 'flat' ? 'material' : current;
}

/**
 * Does this document read the surface inputs?
 *
 * Asked through `analyzeGraph` — the same function `emit()` uses to decide the entry point —
 * rather than by scanning the graph for `normal` nodes here. Two notions of "is this a surface
 * shader" would eventually disagree, and the disagreement would show up as a document the
 * preview silently refuses to display.
 */
function surfaceNeedsOf(doc: ShaderDoc): boolean {
  try {
    return analyzeGraph(doc.graph, asEmitCatalog(state.catalog.catalog)).needs.surface;
  } catch {
    // A malformed graph is `emit`'s job to report, with diagnostics. Not knowing the answer here
    // must not stop the preview from trying.
    return false;
  }
}

/**
 * Decide how the 3D pass should shade the mesh: the graph itself, or the baked pattern.
 *
 * The switch is `Emission.needs.surface` — the emitter's own answer to "does this document read
 * an interpolated surface input". Nothing here inspects the graph looking for `normal` nodes;
 * two independent notions of "is this a surface shader" would eventually disagree, and the
 * disagreement would show up as a mesh shaded by a texture that was never meant to be one.
 *
 * The 3D pass is WebGL2-only, so a document previewed through WebGPU still needs the **GLSL**
 * projection for the mesh. That is a second `emit()` on a structural change — the same cost
 * class as the one already being paid, and never on a slider tick.
 */
function syncSurfaceMaterial(emission: Emission, target: TargetFormat): void {
  const surface = state.surface;
  if (!surface) return;

  if (!emission.needs.surface) {
    surface.setGraphShader(null);
    syncSurfaceControls();
    return;
  }

  const glsl =
    target === 'glsl'
      ? emission
      : emit(state.doc, { catalog: asEmitCatalog(state.catalog.catalog), target: 'glsl' });
  // A surface shader with errors cannot be trusted to compile, and handing it to the driver
  // would produce a log about *our* invalid source instead of the real diagnostic the flat
  // preview is already showing. Keep the pattern path and let the compile strip speak.
  if (glsl.diagnostics.some((d) => d.severity === 'error')) {
    surface.setGraphShader(null);
    syncSurfaceControls();
    return;
  }

  const note = surface.setGraphShader({
    source: glsl.source,
    uniforms: glsl.uniforms,
    values: state.values,
  });
  if (note) banner('warn', note);
  syncSurfaceControls();
}

/* -------------------------------------------------------------------------- */
/* Compile status strip                                                        */
/* -------------------------------------------------------------------------- */

/**
 * What `iChannel0` is bound to, or `null` when the document never reads it.
 *
 * Pure and exported because it is a *claim about the host*, and the three cases are genuinely
 * different: feedback (previous frame), a stand-in (white), and "the 3D pass cannot do feedback"
 * — which is a limitation of the material path, not of the document, and the honest thing is to
 * name it instead of letting the user conclude the shader is broken.
 */
export function channelNote(args: {
  needsChannel: boolean;
  feedback: boolean;
  surfacePath: boolean;
}): string | null {
  if (!args.needsChannel) return null;
  if (args.feedback) {
    return args.surfacePath
      ? 'iChannel0：3D 材质路径只提供白色占位（反馈仅支持平面预览）'
      : 'iChannel0：上一帧（反馈已启用）';
  }
  return 'iChannel0：白色占位（render.feedback 未开启）';
}

function setCompileStrip(result: CompileResult | null, hardError: string | null, emission?: Emission): void {
  const strip = must('compile-strip');
  strip.replaceChildren();
  strip.className = 'strip compile-strip';

  const dot = h('span', 'dot');
  const text = h('span', 'mono');
  if (hardError) {
    dot.classList.add('dot-bad');
    text.textContent = hardError;
    strip.append(dot, text);
    return;
  }
  if (!result) {
    dot.classList.add('dot-warn');
    text.textContent = '未编译';
    strip.append(dot, text);
    return;
  }

  const errors = result.diagnostics.filter((d) => d.severity === 'error');
  dot.classList.add(result.ok ? 'dot-ok' : 'dot-bad');
  text.textContent = result.ok
    ? `编译 ${result.ms.toFixed(1)}ms${result.cached ? '（缓存命中，未重复编译）' : ''}`
    : `编译失败 ${result.ms.toFixed(1)}ms · ${errors.length} 个错误`;
  strip.append(dot, text);

  if (emission) {
    strip.append(h('span', 'muted mono', `· ${emission.uniforms.length} uniform`));
    // Which binding the image channel got. Stated because it is invisible in the image and the
    // difference is enormous: the previous frame makes an accumulation shader work, a white
    // texel makes it look like a shader that does nothing.
    const channel = channelNote({
      needsChannel: emission.needs.channel === true,
      feedback: state.doc.render.feedback === true,
      surfacePath: state.surface !== null,
    });
    if (channel) strip.append(h('span', 'muted mono', `· ${channel}`));
  }
  const unknown = state.renderer ? unknownUniformsOf(state.renderer) : [];
  if (unknown.length) {
    strip.append(h('span', 'muted mono', `· 未绑定：${unknown.join(', ')}`));
  }

  if (result.diagnostics.length || result.log) {
    const details = h('details', 'diag-wrap');
    details.append(h('summary', undefined, `查看完整编译器输出与 ${result.diagnostics.length} 条诊断`));
    if (result.diagnostics.length) {
      const list = h('ul', 'diag-list');
      for (const d of result.diagnostics) {
        const li = h('li', `diag diag-${d.severity}`);
        const head = h('div', 'diag-head');
        head.append(h('code', undefined, d.severity === 'error' ? 'ERROR' : 'WARNING'));
        if (d.line !== undefined) {
          head.append(
            button(`line ${d.line}`, () => {
              codegen.show(previewTarget()).then(() => codegen.highlightLine(d.line ?? 1));
            }, 'chip'),
          );
        }
        li.append(head, h('div', 'diag-msg', d.message));
        list.append(li);
      }
      details.append(list);
    }
    details.append(h('pre', 'compile-log', result.log || '(no log)'));
    strip.append(details);
  }
}

/* -------------------------------------------------------------------------- */
/* Render loop                                                                 */
/* -------------------------------------------------------------------------- */

function resolveWindowSize(): number {
  const stage = canvasHost;
  const size = Math.max(128, Math.min(2048, Math.floor(stage.clientWidth) - 16));
  return size;
}

function effectiveResolution(): number {
  if (state.forcedRes) return state.forcedRes;
  if (state.tier === 'draft') return TIER_RES.draft;
  if (state.tier === 'interactive') return TIER_RES.interactive;
  return state.qualityRes === 'window' ? resolveWindowSize() : state.qualityRes;
}

function renderNow(): void {
  const r = state.renderer;
  if (!r) return;
  const size = effectiveResolution();
  if (state.surface) {
    /*
     * Two passes, and the order is the whole feature.
     *
   * The pattern pass draws the document into the detached canvas — an input, not a picture.
     * The surface pass then uploads it and lights a mesh with it at display resolution. So a
     * slider tick costs one extra 512² upload and one extra draw call, and nothing recompiles:
     * the PERF contract in `onParamChange` is untouched by this mode.
     */
    state.surface.resize(size, size);
    r.render(state.time);
    state.surface.render(state.time);
  } else {
    r.resize(size, size);
    r.render(state.time);
  }
  state.dirty = false;
}

let lastFrameAt = 0;
let fpsAccum = 0;
let fpsFrames = 0;

function loop(now: number): void {
  const dt = lastFrameAt ? Math.min(0.1, (now - lastFrameAt) / 1000) : 0;
  lastFrameAt = now;

  if (state.playing) {
    state.time = (state.time + dt) % MAX_TIME;
    state.dirty = true;
  }
  if (state.dirty) {
    renderNow();
    syncTimeUI();
  }

  fpsAccum += dt;
  fpsFrames += 1;
  if (fpsAccum >= 0.5) {
    state.fps = fpsFrames / fpsAccum;
    fpsAccum = 0;
    fpsFrames = 0;
    updateStatusLive();
  }
  requestAnimationFrame(loop);
}

/* -------------------------------------------------------------------------- */
/* Parameter hot path                                                          */
/* -------------------------------------------------------------------------- */

function onParamChange(name: string, value: UniformValue): void {
  state.values[name] = value;

  // PERF: 这是整个工作台最重要的一行。参数是命名 uniform，文档结构没变（docKey 不变），
  // PERF: 所有 program 依然有效 —— 只写一次 uniform 再画一帧即可。
  // PERF: 这里绝对不能出现 emit()/compile()：那会把「拖动滑杆」从 ~0.1ms 变成 5–50ms，
  // PERF: 也就是 DESIGN §6.1 表里唯一一格被标为「关键」的优化。
  state.renderer?.setUniforms({ [name]: value });
  // Same contract for the 3D pass when it is shading with the graph: a value map, no program.
  state.surface?.setGraphValues(state.values);
  renderNow();
}

let scoreTimer: number | null = null;

function scheduleScore(): void {
  if (!state.reference) return;
  if (scoreTimer !== null) window.clearTimeout(scoreTimer);
  scoreTimer = window.setTimeout(() => void requestScore(), 220);
}

function onInteractStart(): void {
  if (state.tier === 'quality') return;
  if (state.tier !== 'draft') {
    state.tier = 'draft';
    syncToolbar();
    renderNow();
  }
}

function onSettle(): void {
  if (state.tier === 'draft') {
    state.tier = 'interactive';
    syncToolbar();
    renderNow();
  }
  scheduleScore();
}

function resetUniforms(names: string[]): void {
  const source = new Map(state.uniforms.map((u) => [u.name, u.value]));
  const patch: Record<string, UniformValue> = {};
  for (const name of names) {
    const value = source.get(name);
    if (value === undefined) continue;
    patch[name] = value as UniformValue;
    params.setValue(name, value as UniformValue);
    state.values[name] = value as UniformValue;
  }
  // PERF: 重置同样是热路径：只回写 uniform，不重新发射、不重新编译。
  state.renderer?.setUniforms(patch);
  state.surface?.setGraphValues(state.values);
  renderNow();
  scheduleScore();
}

/* -------------------------------------------------------------------------- */
/* Code view                                                                   */
/* -------------------------------------------------------------------------- */

function buildCodegen(): void {
  codegen = new CodegenPanel({
    catalog: asEmitCatalog(state.catalog.catalog),
    onLineClick: () => {
      /* clicking a line is already the navigation; nothing else to do */
    },
  });
  must('codegen').append(codegen.el);
}

/* -------------------------------------------------------------------------- */
/* Toolbar                                                                     */
/* -------------------------------------------------------------------------- */

let backendSelect: HTMLSelectElement;
let resSelect: HTMLSelectElement;
let modeSelect: HTMLSelectElement;
let shapeSelect: HTMLSelectElement;
let tierButtons: HTMLButtonElement[] = [];
let playBtn: HTMLButtonElement;
let timeSlider: HTMLInputElement;
let timeLabel: HTMLElement;
let surfaceNote: HTMLElement;
let rotateBtn: HTMLButtonElement;
/** Material-strip groups that only mean something on the pattern path. */
let patternOnlyControls: HTMLElement[] = [];

/**
 * A labelled range input.
 *
   * The material controls are six sliders that all do the same four things, and the fourth —
   * pushing the value into the live renderer without recompiling — is the one that matters. A
 * helper is not just less code here; it is the only way to be sure no slider accidentally
 * takes the expensive path.
 */
function materialSlider(
  label: string,
  min: number,
  max: number,
  step: number,
  read: () => number,
  write: (v: number) => void,
  hint?: string,
): HTMLElement {
  const group = h('label', 'tool-group surface-slider');
  const caption = h('span', 'tool-label', label);
  const input = h('input');
  input.type = 'range';
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  input.value = String(read());
  // A hover hint, not a second label: "密度" is meaningless without knowing what it counts, and
  // the answer differs per mapping mode (world tiles vs per-face repeats).
  if (hint) group.title = hint;
  const readout = h('span', 'mono muted surface-value', read().toFixed(2));
  input.addEventListener('input', () => {
    const v = Number(input.value);
    write(v);
    readout.textContent = v.toFixed(2);
    state.dirty = true;
  });
  group.append(caption, input, readout);
  group.dataset.patternOnly = '';
  return group;
}

/**
 * Build the material strip: everything that describes *how the pattern is worn*.
 *
 * It is hidden in `flat` mode rather than disabled, because a greyed-out density slider on a
   * fullscreen quad would be a control with no meaning — and the mode selector right above it
 * already says why it is not there.
 */
function buildSurfaceStrip(): void {
  const strip = must('surface-strip');
  strip.replaceChildren();

  const apply = (patch: Partial<MaterialSettings>): void => {
    state.surfaceMaterial = { ...state.surfaceMaterial, ...patch };
    state.surface?.setMaterial(patch);
    if (state.surface) syncSurfaceControls();
    state.dirty = true;
  };

  const mapping = h('label', 'tool-group surface-slider');
  mapping.append(h('span', 'tool-label', '映射'));
  const mappingSelect = h('select');
  for (const [id, label, title] of [
    [
      'triplanar',
      '三平面',
      '按世界坐标从三个轴向投影后混合：曲面与薄管上不会拉丝。' +
        '图案以物体中心为中心、每 1/密度 个世界单位重复一次（密度 ≤ 0.5 时整个图案只铺一次，居中且无接缝）',
    ],
    [
      'uv',
      'UV',
      '按网格自带的 UV 采样：每个形状的接缝位置不同，适合检查 UV 布局。' +
        '密度 1 = 每个面正好铺一次；大于 1 会平铺',
    ],
  ] as const) {
    const opt = h('option');
    opt.value = id;
    opt.textContent = label;
    opt.title = title;
    mappingSelect.append(opt);
  }
  mappingSelect.value = state.surfaceMaterial.mapping;
  mappingSelect.addEventListener('change', () => {
    apply({ mapping: mappingSelect.value as MappingMode });
  });
  mapping.append(mappingSelect);
  mapping.dataset.patternOnly = '';

  const seamless = h('label', 'tool-group');
  const seamlessInput = h('input');
  seamlessInput.type = 'checkbox';
  seamlessInput.checked = state.surfaceMaterial.seamless;
  seamlessInput.addEventListener('change', () => apply({ seamless: seamlessInput.checked }));
  seamless.append(seamlessInput, h('span', 'tool-label', '接缝融合'));
  seamless.title = '融合图案边缘，消除重复采样的硬线；四个平移副本交叉融合，大尺度特征可能出现重影。UV 每片元 4 次、三平面 12 次纹理采样（关闭为 1/3 次）。UV 模式的重复次数取整。仅影响材质预览，导出源码不变。';
  seamless.dataset.patternOnly = '';

  strip.append(
    mapping,
    seamless,
    materialSlider(
      '密度',
      0.25,
      6,
      0.05,
      () => state.surfaceMaterial.density,
      (v) => apply({ density: v }),
      '三平面：每 1/密度 个世界单位重复一次（0.5 以下整个图案只铺一次）。UV：每个面重复 密度 次',
    ),
    materialSlider('金属度', 0, 1, 0.02, () => state.surfaceMaterial.metalness, (v) => apply({ metalness: v })),
    materialSlider('粗糙度', 0.04, 1, 0.01, () => state.surfaceMaterial.roughness, (v) => apply({ roughness: v })),
    materialSlider('图案强度', 0, 1, 0.02, () => state.surfaceMaterial.patternMix, (v) => apply({ patternMix: v })),
    materialSlider('曝光', 0.2, 2.5, 0.02, () => state.surfaceMaterial.exposure, (v) => apply({ exposure: v })),
  );

  const actionGroup = h('div', 'tool-group');
  rotateBtn = button('⟳ 自动旋转', () => {
    const on = !(state.surface?.autoRotating ?? false);
    state.surface?.setAutoRotate(on);
    rotateBtn.classList.toggle('active', on);
  });
  rotateBtn.classList.toggle('active', true);
  const resetBtn = button('重置视角', () => {
    state.surface?.resetCamera();
    state.dirty = true;
  });
  actionGroup.append(rotateBtn, resetBtn);
  strip.append(actionGroup);

  surfaceNote = h('div', 'surface-note muted mono');
  strip.append(surfaceNote);

  /*
   * Which of these controls can do anything right now.
   *
   * When the document is a surface shader the 3D pass runs the *graph*, and density, metalness,
   * roughness, mapping and pattern strength are properties of the texture path that no longer
   * exists. Leaving them live would be the worst available option: a slider that moves and
   * changes nothing reads as a broken app, and the user has no way to tell it apart from a
   * shader that ignores its parameters. They are disabled and the note says why.
   */
  patternOnlyControls = [...strip.querySelectorAll<HTMLElement>('[data-pattern-only]')];
  syncSurfaceControls();
}

/** Enable/disable the pattern-path-only controls for the material path actually in use. */
function syncSurfaceControls(): void {
  const graph = state.surface?.graphShaded ?? false;
  for (const group of patternOnlyControls) {
    group.classList.toggle('inactive', graph);
    for (const control of group.querySelectorAll('input, select')) {
      (control as HTMLInputElement | HTMLSelectElement).disabled = graph;
    }
  }
  if (surfaceNote) surfaceNote.textContent = surfaceNoteText();
}

/** What the 3D view is showing, and which controls are live, in one place. */
function surfaceNoteText(): string {
  const surface = state.surface;
  if (!surface) return '';
  if (!surface.graphShaded) return surface.describe();
  const unknown = surface.graphUnknownUniforms;
  const interaction = surface.pointerFedToGraph
    ? '\n该图读取 iMouse：拖动 = 直接操作指针（Shift+拖动 或 右键拖动 = 旋转视角）。'
    : '';
  return (
    `${surface.describe()}\n` +
    '密度/金属度/粗糙度/映射/图案强度只作用于贴图路径，已停用；曝光只作用于天空。' +
    (unknown.length ? `未生效的 uniform：${unknown.join('、')}` : '') +
    interaction
  );
}

function buildToolbar(): void {
  const bar = must('toolbar');

  const modeGroup = h('div', 'tool-group');
  modeGroup.append(h('span', 'tool-label', '预览'));
  modeSelect = h('select');
  for (const [id, label, title] of [
    ['flat', '平面', '全屏四边形：直接看图案本身'],
    ['material', '材质', '把图案当作材质贴到一个模型上，带光照与环境反射'],
    ['scene', '场景', '模型 + 地面，看它在环境里的样子'],
  ] as const) {
    const opt = h('option');
    opt.value = id;
    opt.textContent = label;
    opt.title = title;
    modeSelect.append(opt);
  }
  modeSelect.value = state.previewMode;
  modeSelect.addEventListener('change', () => {
    state.previewMode = modeSelect.value as PreviewMode;
    syncToolbar();
    // A mode change replaces the canvas and rebuilds the renderer, so it goes through the
    // same path a backend change does rather than a second, subtly different one.
    void setupRenderer(state.backend);
  });
  modeGroup.append(modeSelect);

  shapeSelect = h('select');
  for (const id of OBJECT_SHAPES) {
    const opt = h('option');
    opt.value = id;
    opt.textContent = SHAPE_LABELS[id];
    shapeSelect.append(opt);
  }
  shapeSelect.value = state.surfaceShape;
  shapeSelect.addEventListener('change', () => {
    state.surfaceShape = shapeSelect.value as ShapeId;
    state.surface?.setShape(state.surfaceShape);
    state.surface?.resetCamera();
    if (state.surface) syncSurfaceControls();
    state.dirty = true;
  });
  modeGroup.append(shapeSelect);

  const backendGroup = h('div', 'tool-group');
  backendGroup.append(h('span', 'tool-label', '后端'));
  backendSelect = h('select');
  for (const id of ['webgl2', 'webgpu'] as BackendId[]) {
    const opt = h('option');
    opt.value = id;
    opt.textContent = id === 'webgl2' ? 'WebGL2 (GLSL)' : 'WebGPU (WGSL)';
    if (id === 'webgpu' && !webgpuSupported()) {
      opt.disabled = true;
      opt.textContent += ' — 不可用';
    }
    backendSelect.append(opt);
  }
  backendSelect.value = state.backend;
  backendSelect.addEventListener('change', () => {
    void setupRenderer(backendSelect.value as BackendId);
  });
  backendGroup.append(backendSelect);

  const resGroup = h('div', 'tool-group');
  resGroup.append(h('span', 'tool-label', '分辨率'));
  resSelect = h('select');
  for (const choice of [256, 512, 1024, 'window'] as ResChoice[]) {
    const opt = h('option');
    opt.value = String(choice);
    opt.textContent = choice === 'window' ? '匹配窗口' : `${choice}×${choice}`;
    resSelect.append(opt);
  }
  resSelect.value = String(state.qualityRes);
  resSelect.addEventListener('change', () => {
    const raw = resSelect.value;
    state.qualityRes = raw === 'window' ? 'window' : (Number(raw) as 256 | 512 | 1024);
    state.tier = 'quality';
    syncToolbar();
    renderNow();
  });
  resGroup.append(resSelect);

  const tierGroup = h('div', 'tool-group');
  tierGroup.append(h('span', 'tool-label', '预览档'));
  tierButtons = (['draft', 'interactive', 'quality'] as Tier[]).map((tier) => {
    const label = tier === 'draft' ? '草稿' : tier === 'interactive' ? '交互' : '成品';
    const b = button(label, () => {
      state.tier = tier;
      syncToolbar();
      renderNow();
      if (tier === 'quality') void captureQualityStrip();
    });
    tierGroup.append(b);
    return b;
  });

  const timeGroup = h('div', 'tool-group');
  playBtn = button('⏸ 暂停', () => {
    state.playing = !state.playing;
    syncToolbar();
  });
  const stepBtn = button('⏭ 单帧', () => {
    state.playing = false;
    state.time = Math.min(MAX_TIME, state.time + STEP);
    state.dirty = true;
    syncToolbar();
  });
  const resetTimeBtn = button('重置时间', () => {
    state.time = 0;
    state.dirty = true;
    syncTimeUI();
  });
  timeSlider = h('input');
  timeSlider.type = 'range';
  timeSlider.min = '0';
  timeSlider.max = String(MAX_TIME);
  timeSlider.step = '0.01';
  timeSlider.value = '0';
  timeSlider.className = 'time-slider';
  timeSlider.addEventListener('input', () => {
    state.time = Number(timeSlider.value);
    state.playing = false;
    state.dirty = true;
    syncToolbar();
  });
  timeLabel = h('span', 'mono muted', '0.00s');
  timeGroup.append(playBtn, timeSlider, timeLabel, stepBtn, resetTimeBtn);

  const actionGroup = h('div', 'tool-group');
  actionGroup.append(
    openDocButton(),
    button('生成预览', () => {
      state.tier = 'quality';
      syncToolbar();
      renderNow();
      void captureQualityStrip();
    }),
    button('重新编译', () => {
      codegen.invalidate();
      state.values = {};
      void recompileAll();
    }),
    button('重置全部参数', () => resetUniforms(state.uniforms.map((u) => u.name))),
  );

  bar.append(modeGroup, backendGroup, resGroup, tierGroup, timeGroup, actionGroup);

  // Rebuilt only when it needs to exist, but always present in the DOM: `#surface-strip` is
  // declared in index.html so the boot test can assert the mount points from the document
  // rather than from a list kept here.
  buildSurfaceStrip();
  syncToolbar();
}

function syncToolbar(): void {
  if (backendSelect) backendSelect.value = state.backend;
  if (resSelect) resSelect.value = String(state.qualityRes);
  if (modeSelect) modeSelect.value = state.previewMode;
  if (shapeSelect) shapeSelect.value = state.surfaceShape;
  const labels: Tier[] = ['draft', 'interactive', 'quality'];
  tierButtons.forEach((b, i) => b.classList.toggle('active', labels[i] === state.tier));
  if (playBtn) playBtn.textContent = state.playing ? '⏸ 暂停' : '▶ 播放';
  const strip = must('surface-strip');
  strip.hidden = state.previewMode === 'flat';
  if (state.surface && surfaceNote) syncSurfaceControls();
  syncTimeUI();
}

function syncTimeUI(): void {
  if (!timeSlider) return;
  timeSlider.value = String(state.time);
  timeLabel.textContent = `${state.time.toFixed(2)}s`;
}

/** 成品档: four time points, captured as thumbnails (DESIGN.md §6.3). */
async function captureQualityStrip(): Promise<void> {
  const strip = must('quality-strip');
  strip.replaceChildren();
  const r = state.renderer;
  if (!r) return;
  const wasPlaying = state.playing;
  state.playing = false;
  const wasTime = state.time;
  for (const t of [0, 15, 30, 45]) {
    state.time = t;
    renderNow();
    const thumb = h('canvas');
    thumb.width = 96;
    thumb.height = 96;
    thumb.title = `t = ${t}s`;
    const ctx = thumb.getContext('2d');
    if (ctx) ctx.drawImage(currentCanvas(), 0, 0, thumb.width, thumb.height);
    strip.append(thumb);
  }
  state.time = wasTime;
  state.playing = wasPlaying;
  state.dirty = true;
  syncTimeUI();
}

/* -------------------------------------------------------------------------- */
/* Info panel                                                                  */
/* -------------------------------------------------------------------------- */

function buildInfo(): void {
  must('info').replaceChildren();
}

function updateInfo(): void {
  const dl = must('info');
  dl.replaceChildren();
  const row = (k: string, v: string): void => {
    dl.append(h('dt', undefined, k), h('dd', undefined, v));
  };
  row('文档', `${state.doc.title} (${state.doc.id})`);
  row('IR', `${state.doc.irVersion} · ${state.doc.graph.nodes.length} 节点 · ${state.uniforms.length} 参数`);
  row('docHash', state.docKey.slice(0, 12));
  row('演示变体', state.variant);
  row('预览模式', state.surface ? state.surface.describe() : '平面（全屏四边形）');
  row('catalog', `${state.catalog.present.length} 种节点${state.catalog.missing.length ? ` · 缺 ${state.catalog.missing.length}` : ''}`);
  row('目标发射器', state.catalog.targets.length ? state.catalog.targets.join(', ') : '未注册');
  row('渲染器', state.renderer ? `${state.renderer.backend}` : '未创建');
  const cacheSize = (state.renderer as Partial<{ cacheSize: number }> | null)?.cacheSize;
  if (typeof cacheSize === 'number') row('编译缓存', `${cacheSize} 个 program`);
  row('评分', state.breakdown ? `SSIM ${state.breakdown.ssim.toFixed(4)}` : '未评分');
  if (state.perEvalMs !== null) row('优化实测', `${state.perEvalMs.toFixed(1)} ms / 次评估`);
  for (const note of state.catalog.notes) row('提示', note);
}

function setStatus(text: string): void {
  const bar = must('statusbar');
  const span = h('span', undefined, text);
  bar.prepend(span);
  while (bar.childElementCount > 8) bar.lastElementChild?.remove();
}

let liveEl: HTMLElement | null = null;

function updateStatusLive(): void {
  if (!liveEl) {
    liveEl = h('span', 'live mono');
    must('statusbar').prepend(liveEl);
  }
  liveEl.textContent = [
    `${state.fps.toFixed(0)} fps`,
    `${effectiveResolution()}×${effectiveResolution()}`,
    `档位 ${state.tier}`,
    `t=${state.time.toFixed(2)}s`,
    state.renderer?.backend ?? '—',
  ].join('  ·  ');
}

/* -------------------------------------------------------------------------- */
/* Opening a document from disk                                                */
/* -------------------------------------------------------------------------- */

/**
 * "打开 .shader.json" — the workbench could generate documents and show one built-in demo, but
 * it could not open a file. That is a strange gap for a tool whose output is a portable
 * document: the CLI writes `examples/*.shader.json`, the agent writes documents, and neither
 * could be brought back into the preview to look at.
 *
 * The shape is not taken on trust. A JSON file that happens to be an array, or an old document
 * from a different IR version, is rejected with the reason named — the alternative is a
 * `TypeError: cannot read properties of undefined (reading 'nodes')` from somewhere deep in the
 * emitter, which reads like a bug in the workbench rather than a file that is not a document.
 */
function openDocButton(): HTMLButtonElement {
  const picker = h('input');
  picker.type = 'file';
  picker.accept = '.json,application/json';
  picker.style.display = 'none';
  picker.addEventListener('change', () => {
    const file = picker.files?.[0];
    if (file) void openDocFile(file);
    // Reset so choosing the same file twice fires `change` again.
    picker.value = '';
  });

  const button = h('button', '', '打开 .shader.json');
  button.title = '打开一个由 CLI 或作者面板写出的 ShaderDoc（例如 examples/toon-surface.shader.json）';
  button.addEventListener('click', () => picker.click());
  // The picker has to be in the document for a click to be dispatched to it in every browser.
  queueMicrotask(() => button.after(picker));
  return button;
}

async function openDocFile(file: File): Promise<void> {
  let text: string;
  try {
    text = await file.text();
  } catch (err) {
    banner('error', `读取 ${file.name} 失败：${err instanceof Error ? err.message : String(err)}`);
    return;
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (err) {
    banner('error', `${file.name} 不是合法 JSON：${err instanceof Error ? err.message : String(err)}`);
    return;
  }

  const problem = documentProblem(parsed);
  if (problem) {
    banner('error', `${file.name} 不是本工作台能打开的 ShaderDoc：${problem}`);
    return;
  }
  await setDoc(parsed as ShaderDoc, `打开 ${file.name}`);
}

/** `null` when this is plausibly a document this build can emit; otherwise what is wrong. */
export function documentProblem(value: unknown): string | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return '顶层不是对象';
  const doc = value as Partial<ShaderDoc>;
  if (typeof doc.irVersion !== 'string') return '缺少 irVersion';
  if (doc.irVersion !== IR_VERSION) {
    return `IR 版本是 ${doc.irVersion}，本工作台是 ${IR_VERSION}（先用 CLI 迁移，不要手改）`;
  }
  if (!doc.graph || !Array.isArray(doc.graph.nodes) || doc.graph.nodes.length === 0) {
    return 'graph.nodes 为空';
  }
  for (const node of doc.graph.nodes) {
    if (!node || typeof node.id !== 'string' || typeof node.kind !== 'string') {
      return 'graph.nodes 里有缺少 id/kind 的节点';
    }
  }
  return null;
}

/* -------------------------------------------------------------------------- */
/* Reference image + scoring                                                   */
/* -------------------------------------------------------------------------- */

function buildComparePanel(): void {
  const root = must('compare');

  const top = h('div', 'compare-top');
  const drop = h('div', 'ref-drop', '拖入参考图\n或点击选择');
  const file = h('input');
  file.type = 'file';
  file.accept = 'image/*';
  file.style.display = 'none';
  file.addEventListener('change', () => {
    const f = file.files?.[0];
    if (f) void loadReference(f);
  });
  drop.addEventListener('click', () => file.click());
  drop.addEventListener('dragover', (ev) => {
    ev.preventDefault();
    drop.classList.add('dragover');
  });
  drop.addEventListener('dragleave', () => drop.classList.remove('dragover'));
  drop.addEventListener('drop', (ev) => {
    ev.preventDefault();
    drop.classList.remove('dragover');
    const f = ev.dataTransfer?.files?.[0];
    if (f) void loadReference(f);
  });
  top.append(drop, file);

  const scoreBox = h('div');
  scoreBox.id = 'score-box';
  top.append(scoreBox);
  root.append(top);

  const optRow = h('div', 'optimize-row');
  const budget = h('input');
  budget.type = 'number';
  budget.min = '20';
  budget.max = '1000';
  budget.step = '10';
  budget.value = '200';
  budget.title = '评估次数上限（严格上限：每次评估 = 一次渲染 + 一次取像素）';
  optimizeBudget = budget;
  runBtn = button('自动优化参数', () => startOptimize(Number(budget.value)));
  runBtn.id = 'optimize-run';
  stopBtn = button('停止', () => stopOptimize());
  stopBtn.disabled = true;
  const progress = h('div', 'progress');
  progressBar = h('i');
  progress.append(progressBar);
  progressText = h('span', 'mono muted', '未开始');
  progressText.id = 'optimize-text';
  optRow.append(runBtn, stopBtn, budget, progress, progressText);
  root.append(optRow);

  const suggest = h('div');
  suggest.id = 'suggest-box';
  root.append(suggest);

  renderScoreUI();
}

async function loadReference(file: File): Promise<void> {
  try {
    const bitmap = await createImageBitmap(file);
    const canvas = h('canvas');
    canvas.width = COMPARE_SIZE;
    canvas.height = COMPARE_SIZE;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('2D canvas unavailable');
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, COMPARE_SIZE, COMPARE_SIZE);
    // Letterbox: preserve the reference's aspect instead of stretching it, so SSIM is
    // never measuring a resampling artefact.
    const scale = Math.min(COMPARE_SIZE / bitmap.width, COMPARE_SIZE / bitmap.height);
    const w = Math.max(1, Math.round(bitmap.width * scale));
    const hh = Math.max(1, Math.round(bitmap.height * scale));
    ctx.drawImage(bitmap, (COMPARE_SIZE - w) / 2, (COMPARE_SIZE - hh) / 2, w, hh);
    const img = ctx.getImageData(0, 0, COMPARE_SIZE, COMPARE_SIZE);
    state.reference = { data: img.data, w: COMPARE_SIZE, h: COMPARE_SIZE };
    bitmap.close();

    const drop = must('compare').querySelector('.ref-drop');
    if (drop) {
      drop.replaceChildren();
      const thumb = h('img');
      thumb.src = canvas.toDataURL('image/png');
      thumb.alt = '参考图';
      drop.append(thumb);
    }

    if (state.backend !== 'webgl2') {
      banner('info', '参考图评分需要逐帧读取像素（captureFrame），只有 WebGL2 渲染器提供该能力：已切回 WebGL2。', false);
      await setupRenderer('webgl2');
    }
    sendInit().catch((err) => banner('error', `compare worker 初始化失败：${String(err)}`));
    await requestScore();
  } catch (err) {
    banner('error', `读取参考图失败：${err instanceof Error ? err.message : String(err)}`);
  }
}

/**
 * Create the compare worker.
 *
 * The workbench has two ways to be loaded, and they need different worker URLs:
 *
 *   • **Zero-build (primary).** `scripts/serve.mjs` serves the repo and the browser loads
 *     `apps/web/dist/*.js`, so the worker is the projected sibling `score.worker.js` and
 *     the specifier is built at runtime — a literal `.ts` URL is a dead link here.
 *   • **Vite (optional, for anyone with npm).** `import.meta.url` ends in `.ts` in dev,
 *     and Vite can only bundle a worker it can see statically, so that branch uses Vite's
 *     own `?worker` form.
 *
 * The test is `endsWith('.ts')` rather than `endsWith('.js')` on purpose: anything that
 * is *not* recognisably the Vite path (a cache-busting query string, a proxy that rewrites
 * URLs) must still get a worker that exists, and the projected `.js` file always does.
 *
 * Known limitation: under `vite build` the emitted asset URL is a hashed `.js`, so the
 * Vite branch is not taken and the worker 404s. `vite dev` and the zero-build path — the
 * two ways this is actually run — both work. Recorded in README rather than hidden.
 */
async function createWorker(): Promise<Worker> {
  if (!import.meta.url.endsWith('.ts')) {
    const ext = '.js';
    return new Worker(new URL(`./worker/score.worker${ext}`, import.meta.url), { type: 'module' });
  }
  const mod = (await import('./worker/score.worker.ts?worker')) as { default: new () => Worker };
  return new mod.default();
}

let workerPromise: Promise<Worker> | null = null;

function ensureWorker(): Promise<Worker> {
  workerPromise ??= createWorker().then((w) => {
    worker = w;
    w.addEventListener('message', (ev: MessageEvent<WorkerResponse>) => handleWorkerMessage(ev.data));
      w.addEventListener('error', (ev) => banner('error', `compare worker 错误：${ev.message}`));
    return w;
  });
  return workerPromise;
}

async function sendInit(): Promise<void> {
  if (!state.reference) return;
  const w = await ensureWorker();
  w.postMessage({
    type: 'init',
    ref: { data: state.reference.data, w: state.reference.w, h: state.reference.h },
    uniforms: state.uniforms.map((u) => ({
      name: u.name,
      origin: u.origin,
      label: u.label,
      min: u.min,
      max: u.max,
      value: u.value,
    })),
  } satisfies WorkerRequest);
}

/** Render + read back one frame at the compare resolution. Only WebGL2 can do this. */
function captureForScoring(): FramePayload | null {
  /*
   * Measure what is actually on screen.
   *
   * In a surface mode the pattern is an *input* to the material, not the picture. Reading it
   * back would score the texture against the reference image and report a match for something
   * the user is not looking at — the failure would be invisible and the optimizer would tune
   * toward it. The surface pass is WebGL2 whatever the pattern backend is, so this also means
   * scoring works in a surface mode under the WebGPU backend, where the flat path cannot.
   */
  const target: (FrameCapture & object) | PreviewRenderer | null = state.surface ?? state.renderer;
  if (!target) return null;
  if (!supportsCapture(target)) {
    banner('warn', '当前渲染器不提供 captureFrame()：参考图评分与参数搜索需要 WebGL2。', false);
    return null;
  }
  renderNow();
  const frame = target.captureFrame();
  return { data: frame.data, w: frame.width, h: frame.height };
}

async function requestScore(): Promise<void> {
  if (!state.reference || !state.renderer) return;
  state.forcedRes = COMPARE_SIZE;
  const frame = captureForScoring();
  state.forcedRes = null;
  if (!frame) return;
  const w = await ensureWorker();
  w.postMessage({ type: 'score', frame } satisfies WorkerRequest);
}

function handleWorkerMessage(msg: WorkerResponse): void {
  switch (msg.type) {
    case 'ready':
      break;
    case 'score':
      state.breakdown = msg.result;
      state.attribution = msg.attribution;
      renderScoreUI();
      updateInfo();
      break;
    case 'render': {
      // One evaluation = one render + one postMessage round trip.
      const restore = state.forcedRes;
      state.forcedRes = COMPARE_SIZE;
      const frame = captureForScoring();
      state.forcedRes = restore;
      if (!frame) return;
      // `worker` is non-null here by construction: this branch only runs because that
      // worker sent us a message. Replying synchronously keeps the optimizer's
      // render→score round trip free of extra microtask hops.
      worker?.postMessage({ type: 'frame', id: msg.id, frame } satisfies WorkerRequest);
      break;
    }
    case 'progress': {
      state.perEvalMs = msg.perEvalMs;
      const total = Math.max(1, Number(optimizeBudget?.value ?? 200));
      progressBar.style.width = `${Math.min(100, (msg.evals / total) * 100)}%`;
        progressText.textContent = `评估 ${msg.evals}/${total} · 最优 loss ${msg.best.toFixed(4)} · ${msg.perEvalMs.toFixed(1)} ms/次`;
      break;
    }
    case 'result': {
      state.perEvalMs = msg.perEvalMs;
      state.suggestion = msg.result.values;
      state.suggestionLoss = msg.result.best;
      setStatus(`优化完成：${msg.result.evals} 次评估，${msg.elapsedMs.toFixed(0)}ms，${msg.perEvalMs.toFixed(1)} ms/次`);
      stopOptimizeUI(false);
      renderScoreUI();
      updateInfo();
      break;
    }
    case 'error':
      banner('error', msg.message);
      stopOptimizeUI(false);
      break;
  }
}

function stopOptimizeUI(running: boolean): void {
  if (runBtn) runBtn.disabled = running;
  if (stopBtn) stopBtn.disabled = !running;
  if (!running && progressBar) progressBar.style.width = '0%';
}

let playingBeforeOptimize = true;

function startOptimize(budget: number): void {
  if (!state.reference) {
    banner('warn', '先加载一张参考图，"自动优化参数" 才有目标可以拟合。');
    return;
  }
  const numeric = state.uniforms.filter((u) => u.type === 'float' || u.type === 'int');
  if (numeric.length === 0) {
    banner('warn', '当前文档没有数值型可调参数，搜索空间为空。');
    return;
  }
  // The loss must be a function of the parameters only: freeze the clock for the run,
  // otherwise the animation adds variance the optimizer would chase.
  playingBeforeOptimize = state.playing;
  state.playing = false;
  state.suggestion = null;
  stopOptimizeUI(true);
  void ensureWorker().then((w) => {
    w.postMessage({
      type: 'optimize',
      params: numeric.map((u) => ({
        name: u.name,
        min: typeof u.min === 'number' ? u.min : 0,
        max: typeof u.max === 'number' ? u.max : 1,
        value: Number(state.values[u.name] ?? u.value) || 0,
      })),
      budget: Math.max(4, Math.min(2000, Math.round(budget))),
    } satisfies WorkerRequest);
  });
}

function stopOptimize(): void {
  worker?.postMessage({ type: 'cancel' } satisfies WorkerRequest);
  state.playing = playingBeforeOptimize;
  stopOptimizeUI(false);
  setStatus('优化已停止（已保留当前最优）');
}

/* ---- score rendering --------------------------------------------------- */

function bar(value: number, max = 1): HTMLElement {
  const wrap = h('div', 'bar');
  const fill = h('i');
  fill.style.width = `${Math.max(0, Math.min(1, value / max)) * 100}%`;
  wrap.append(fill);
  return wrap;
}

function renderScoreUI(): void {
  const box = document.getElementById('score-box');
  if (!box) return;
  box.replaceChildren();

  if (!state.reference) {
      box.append(h('p', 'muted', '未加载参考图。拖一张图片到左侧方框，即可用真实像素对比评分。'));
    return;
  }
  const bd = state.breakdown;
  if (!bd) {
      box.append(h('p', 'muted', '评分中…'));
    return;
  }

  const grid = h('div', 'score-grid');
  const line = (label: string, value: string, node: HTMLElement): void => {
    grid.append(h('span', 'score-label', label), node, h('span', 'score-value', value));
  };

  line('SSIM 结构', bd.ssim.toFixed(4), bar(bd.ssim));
  line('直方图距离', bd.hist.toFixed(4), bar(bd.hist));
  line(
    '低频能量',
    `${bd.bands.low.toFixed(3)} / ${bd.bandsRef.low.toFixed(3)}`,
    bar(bd.bands.low, Math.max(1e-6, Math.max(bd.bands.low, bd.bandsRef.low))),
  );
  line(
    '中频能量',
    `${bd.bands.mid.toFixed(3)} / ${bd.bandsRef.mid.toFixed(3)}`,
    bar(bd.bands.mid, Math.max(1e-6, Math.max(bd.bands.mid, bd.bandsRef.mid))),
  );
  line(
    '高频能量',
    `${bd.bands.high.toFixed(3)} / ${bd.bandsRef.high.toFixed(3)}`,
    bar(bd.bands.high, Math.max(1e-6, Math.max(bd.bands.high, bd.bandsRef.high))),
  );
  if (bd.ssimLow !== undefined) line('低频 SSIM', bd.ssimLow.toFixed(4), bar(bd.ssimLow));
  grid.append(h('span', 'score-label', ''), h('span', 'muted', '柱状对比：渲染 / 参考'));
  box.append(grid);

  const att = state.attribution;
  if (att) {
    const verdict = h('div', 'verdict');
    verdict.append(h('div', undefined, att.verdict));
    if (att.hints.length) {
      const list = h('ul', 'hint-list');
      for (const hint of att.hints) {
        const item = h('li', 'hint-item');
        item.append(
          h('span', 'chip', `${hint.node}.${hint.param}`),
          h('span', 'hint-dir', hint.direction === 'increase' ? '↑ 调大' : '↓ 调小'),
          h('span', 'why', hint.why),
        );
        if (hint.suggest !== undefined) {
          const uniformName = findUniformByOrigin(hint.node, hint.param);
          const suggested = hint.suggest;
          item.append(
            button(`试 ${suggested}`, () => {
              if (!uniformName) return;
              params.setValue(uniformName, suggested);
              onParamChange(uniformName, suggested);
            }, 'mini'),
          );
        }
        list.append(item);
      }
      verdict.append(list);
    }
    box.append(verdict);
  }

  const suggestBox = document.getElementById('suggest-box');
  if (suggestBox) renderSuggestion(suggestBox);
}

function findUniformByOrigin(node: string, param: string): string | null {
  const hit = state.uniforms.find((u) => u.origin.node === node && u.origin.param === param);
  return hit?.name ?? null;
}

function renderSuggestion(box: HTMLElement): void {
  box.replaceChildren();
  const values = state.suggestion;
  if (!values) return;

  const title = h('div', 'verdict');
  title.append(
    h('div', undefined, `优化器建议值（loss ${state.suggestionLoss?.toFixed(4) ?? '—'}）—— 预览中，尚未写入文档`),
  );
  const table = h('table', 'suggest-table');
  const head = h('tr');
  head.append(h('th', undefined, 'uniform'), h('th', undefined, '当前'), h('th', undefined, '建议'));
  table.append(head);
  for (const [name, value] of Object.entries(values)) {
    const current = Number(state.values[name] ?? 0);
    if (Math.abs(current - value) < 1e-6) continue;
    const tr = h('tr');
    tr.append(
      h('td', undefined, name),
      h('td', undefined, current.toFixed(4)),
      h('td', undefined, value.toFixed(4)),
    );
    table.append(tr);
  }
  title.append(table);

  const actions = h('div', 'suggest-actions');
  actions.append(
    button('采纳（写入文档）', () => adoptSuggestion()),
    button('全部试一遍（仅预览）', () => {
      for (const [name, value] of Object.entries(values)) onParamChange(name, value);
    }),
    button('放弃', () => {
      state.suggestion = null;
      renderScoreUI();
    }),
  );
  title.append(actions);
  box.append(title);
}

/** 采纳 = a real `set_param` patch through the IR patcher (atomic, immutable, rehashed). */
function adoptSuggestion(): void {
  const values = state.suggestion;
  if (!values) return;
  const ops = Object.entries(values)
    .map(([name, value]) => {
      const decl = state.uniforms.find((u) => u.name === name);
      if (!decl) return null;
      return {
        op: 'set_param' as const,
        node: decl.origin.node,
        param: decl.origin.param,
        value: {
          kind: 'param' as const,
          name: decl.name,
          type: decl.type,
          value,
          min: decl.min,
          max: decl.max,
          step: decl.step,
          label: decl.label,
          group: decl.group,
        } satisfies ParamValue,
      reason: '参数优化器建议值（人工采纳）',
      };
    })
    .filter((op): op is NonNullable<typeof op> => op !== null);

  if (ops.length === 0) return;
  const result = applyPatch(
    state.doc,
      { summary: '采纳优化器建议值', ops, author: { kind: 'optimizer' } },
    state.catalog.catalog,
  );
  if (!result.ok) {
    banner('error', `采纳失败：${result.diagnostics.map((d) => `${d.code} ${d.message}`).join('；')}`);
    return;
  }
  state.suggestion = null;
  state.values = {};
  void setDoc(result.doc, `采纳优化器建议（${result.applied} 项）`);
}

/* -------------------------------------------------------------------------- */
/* Boot                                                                        */
/* -------------------------------------------------------------------------- */

async function boot(): Promise<void> {
  state.catalog = loadCatalog();

  const demo = demoDocForCatalog(state.catalog.catalog);
  state.doc = demo.doc;
  state.docOrigin = demo.doc;
  state.variant = demo.variant;
  state.docKey = demo.doc.hash ?? hashDoc(demo.doc);

  if (demo.issues.length) {
    banner(
      'error',
        `演示文档与 catalog 不匹配，预览可能无法工作：\n${demo.issues.slice(0, 8).join('\n')}\n` +
          '（修正 apps/web/src/doc/demo.ts 的端口名，不要改 catalog。）',
    );
  }
  if (state.catalog.sources.length === 0) {
    banner('warn', '未发现 packages/ir/src/catalog/：滑杆面板与代码视图会保持为空，直到 catalog 就位。');
  }
  for (const note of state.catalog.notes) banner('warn', note, false);

  buildToolbar();
  buildCodegen();
  buildInfo();
  buildComparePanel();
  attachPointerHandlers(currentCanvas());

  params = new ParamPanel(
    {
      onChange: onParamChange,
      onSettle,
      onInteractStart,
      onFocusOrigin: (node) => {
        const count = codegen.highlightNode(node);
        setStatus(count ? `已高亮 ${node} 生成的 ${count} 行` : `未在 ${previewTarget()} 源码中找到 ${node} 的行`);
      },
      onReset: (names) => resetUniforms(names),
      onResetAll: () => resetUniforms(state.uniforms.map((u) => u.name)),
    },
    {
      nodeKindOf: (id) => state.doc.graph.nodes.find((n) => n.id === id)?.kind,
      // `color` is a ParamValue kind, not a ShaderType: the document is the only place
      // that knows a vec4 uniform is meant to be a colour.
      isColor: (name) => {
        const decl = state.uniforms.find((u) => u.name === name);
        if (!decl) return false;
        const node = state.doc.graph.nodes.find((n) => n.id === decl.origin.node);
        return node?.params?.[decl.origin.param]?.kind === 'color';
      },
    },
  );
  must('params').append(params.el);

  codegen.setDoc(state.doc, state.docKey);

  /**
   * The authoring panel drives the agent loop.
   *
   * `applyDoc` reuses `setDoc`, so an agent-produced document goes through exactly the same
   * path as the built-in demo: re-emit, recompile, rebuild the sliders, refresh the code
   * view. There is no second pipeline for "documents that came from a model", which is what
   * keeps the preview honest about what it is showing.
   *
   * `setBusy` deliberately does **not** stop the render loop. The loop is
   * `requestAnimationFrame`-driven and the agent run is `await`ed, so a multi-second model
   * call cannot block a frame; freezing the preview would only make the app look hung. The
   * flag is used for the status line instead.
   */
  author = new AuthorPanel({
    applyDoc: async (next, reason) => {
      await setDoc(next, reason);
    },
    setStatus,
    setBusy: (busy) => {
      document.body.classList.toggle('author-busy', busy);
      if (busy) setStatus('agent 运行中…（预览仍在实时渲染）');
    },
    currentDoc: () => state.doc,
  });
  must('author').append(author.el);

  await setupRenderer('webgl2');
  syncToolbar();
    setStatus(`就绪 · ${state.variant} · ${state.catalog.present.length} 种节点 · ${state.catalog.targets.length} 个目标发射器`);
  if (state.catalog.targets.length === 0) {
    banner('warn', '目标发射器尚未注册（packages/ir/src/targets/）：代码视图会显示 EM000_UNKNOWN_TARGET 诊断，而不是空白。');
  }
  requestAnimationFrame(loop);
}

void boot();
