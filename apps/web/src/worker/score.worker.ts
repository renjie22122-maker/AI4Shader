/**
 * Scoring + parameter-search worker
 * =================================
 *
 * Everything that is pure CPU lives here so the render loop never stutters: SSIM,
 * histogram, band analysis, attribution, and both optimisers.
 *
 * ── The message flow, and why it is shaped like this ──────────────────────────
 *
 * A worker cannot touch the GL context, so the *main thread* stays the renderer and the
 * worker stays the measurer. One evaluation is therefore exactly:
 *
 *     worker  --{render, id, values}-->  main thread
 *     main    : setUniforms(values) → render(t) → captureFrame()
 *     main    --{frame, id, data}----->  worker
 *     worker  : compareFrames() → loss
 *
 * One evaluation = one render + one postMessage round trip, with the pixel buffer
 * *transferred* (not copied). Capturing at 256×256 keeps the whole loop well inside a
 * frame budget, and because parameter changes never recompile (DESIGN.md §6.2), the
 * per-evaluation cost is dominated by the round trip itself.
 *
 * Measured throughput on the development machine is reported in `apps/web/README.md`;
 * the UI shows the live number in the progress line.
 */

import {
  attribute,
  buildRoleTable,
  compareFrames,
  lossFrom,
  type Attribution,
  type ParamRef,
  type ParamRole,
  type ScoreBreakdown,
} from '../compare/score.js';
import {
  optimize,
  type OptimizeMethod,
  type OptimizeParam,
  type OptimizeProgress,
  type OptimizeResult,
} from '../optimize/search.js';

/* -------------------------------------------------------------------------- */
/* Protocol                                                                    */
/* -------------------------------------------------------------------------- */

export type FramePayload = { data: Uint8ClampedArray; w: number; h: number };

export type WorkerRequest =
  | { type: 'init'; ref: FramePayload; uniforms?: { name: string; origin: { node: string; param: string }; label?: string; min?: number; max?: number; value: number | boolean | number[] }[] }
  | { type: 'score'; frame: FramePayload }
  | { type: 'optimize'; params: OptimizeParam[]; budget: number; method?: OptimizeMethod }
  | { type: 'cancel' }
  | { type: 'frame'; id: number; frame: FramePayload };

export type WorkerResponse =
  | { type: 'ready' }
  | { type: 'score'; result: ScoreBreakdown; attribution: Attribution }
  | ({ type: 'progress' } & OptimizeProgress & { elapsedMs: number; perEvalMs: number })
  | { type: 'result'; result: OptimizeResult; elapsedMs: number; perEvalMs: number }
  | { type: 'render'; id: number; values: Record<string, number> }
  | { type: 'error'; message: string };

/** Structural stand-in for `DedicatedWorkerGlobalScope`, avoiding a lib.webworker clash. */
type WorkerScope = {
  postMessage(message: WorkerResponse, transfer?: Transferable[]): void;
};

const ctx = self as unknown as WorkerScope;

/* -------------------------------------------------------------------------- */
/* State                                                                       */
/* -------------------------------------------------------------------------- */

let reference: (FramePayload & { roles: Partial<Record<ParamRole, ParamRef>> }) | null = null;

let renderCounter = 0;
const pendingFrames = new Map<number, (frame: FramePayload) => void>();

let controller: AbortController | null = null;

/* -------------------------------------------------------------------------- */
/* Render round trip                                                           */
/* -------------------------------------------------------------------------- */

function requestFrame(values: Record<string, number>): Promise<FramePayload> {
  const id = ++renderCounter;
  return new Promise<FramePayload>((resolve) => {
    pendingFrames.set(id, resolve);
    ctx.postMessage({ type: 'render', id, values });
  });
}

/* -------------------------------------------------------------------------- */
/* Handlers                                                                    */
/* -------------------------------------------------------------------------- */

function handleInit(msg: Extract<WorkerRequest, { type: 'init' }>): void {
  reference = {
    ...msg.ref,
    roles: msg.uniforms ? buildRoleTable(msg.uniforms) : {},
  };
  ctx.postMessage({ type: 'ready' });
}

function handleScore(msg: Extract<WorkerRequest, { type: 'score' }>): void {
  if (!reference) {
    ctx.postMessage({ type: 'error', message: 'compare worker 未初始化参考图' });
    return;
  }
  const { data, w, h } = msg.frame;
  const result = compareFrames(reference.data, data, w, h);
  const attribution = attribute(result, { roles: reference.roles });
  ctx.postMessage({ type: 'score', result, attribution });
}

async function handleOptimize(msg: Extract<WorkerRequest, { type: 'optimize' }>): Promise<void> {
  if (!reference) {
    ctx.postMessage({ type: 'error', message: 'compare worker 未初始化参考图' });
    return;
  }
  controller?.abort();
  controller = new AbortController();
  const started = performance.now();

  const evaluate = async (values: Record<string, number>): Promise<number> => {
    const frame = await requestFrame(values);
    const breakdown = compareFrames(reference!.data, frame.data, frame.w, frame.h);
    return lossFrom(breakdown);
  };

  try {
    const result = await optimize({
      params: msg.params,
      evaluate,
      maxEvaluations: msg.budget,
      method: msg.method,
      signal: controller.signal,
      onProgress: (p: OptimizeProgress) => {
        const elapsedMs = performance.now() - started;
        ctx.postMessage({
          type: 'progress',
          ...p,
          elapsedMs,
          perEvalMs: p.evals > 0 ? elapsedMs / p.evals : 0,
        });
      },
    });
    const elapsedMs = performance.now() - started;
    ctx.postMessage({
      type: 'result',
      result,
      elapsedMs,
      perEvalMs: result.evals > 0 ? elapsedMs / result.evals : 0,
    });
  } catch (err) {
    ctx.postMessage({ type: 'error', message: `优化失败：${String(err)}` });
  }
}

/* -------------------------------------------------------------------------- */
/* Dispatch                                                                    */
/* -------------------------------------------------------------------------- */

self.addEventListener('message', (event: MessageEvent<WorkerRequest>) => {
  const msg = event.data;
  switch (msg.type) {
    case 'init':
      handleInit(msg);
      break;
    case 'score':
      handleScore(msg);
      break;
    case 'frame': {
      const resolve = pendingFrames.get(msg.id);
      if (resolve) {
        pendingFrames.delete(msg.id);
        resolve(msg.frame);
      }
      break;
    }
    case 'optimize':
      void handleOptimize(msg);
      break;
    case 'cancel':
      controller?.abort();
      controller = null;
      break;
  }
});
