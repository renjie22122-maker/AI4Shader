/**
 * Parameter search
 * ================
 *
 * Two black-box optimisers over the document's tunable uniforms, plus the hybrid the UI
 * actually runs (DESIGN.md §6.6):
 *
 *   1. {@link coordinateDescent} — coarse, greedy, embarrassingly parallel-friendly, and
 *      *very* good at the first 80% of the improvement when only a few knobs matter.
 *   2. {@link cmaesLite}         — a real (if small) CMA-ES that handles the correlation
 *      between knobs the coordinate pass cannot see.
 *
 * Both are written against the same three non-negotiables:
 *
 *   • `maxEvaluations` is a hard ceiling. Every candidate costs a render + a round trip,
 *     so overshooting the budget is a visible stall, not a rounding error.
 *   • `onProgress` fires after **every** evaluation, including rejected ones.
 *   • `signal` is honoured between evaluations; a stopped search returns the best point
 *     found so far rather than throwing.
 *
 * ── Why diagonal-only covariance (the simplification, stated plainly) ──────────
 *
 * Textbook CMA-ES maintains a full n×n covariance matrix (O(n²) memory, O(n³) to
 * decompose per generation). This implementation keeps a **diagonal** C = diag(D²):
 * the per-axis step sizes adapt, the correlations do not. That costs the ability to
 * follow a diagonal valley directly — it zig-zags instead — but it keeps the update
 * rules, the cumulative step-size control, and the rank-μ/rank-1 adaptation, all of
 * which is where CMA-ES's robustness over coordinate descent actually comes from.
 * With ~8 parameters and a ~200-evaluation budget, a full covariance would spend more
 * evaluations learning correlations than it could ever pay back. Documented rather than
 * hidden: this is CMA-ES with a diagonal model, not CMA-ES.
 */

export type OptimizeParam = {
  name: string;
  min: number;
  max: number;
  value: number;
};

/** Returns the loss for a candidate point. **Lower is better.** */
export type Evaluate = (values: Record<string, number>) => Promise<number>;

export type OptimizeProgress = {
  best: number;
  evals: number;
  values: Record<string, number>;
};

export type OptimizeOptions = {
  params: OptimizeParam[];
  evaluate: Evaluate;
  /** Finite evaluation ceiling; fractional budgets are rounded down. */
  maxEvaluations: number;
  onProgress: (p: OptimizeProgress) => void;
  signal?: AbortSignal;
  /** Deterministic sampling: same seed ⇒ same trajectory. Defaults to 1. */
  seed?: number;
};

export type OptimizeResult = {
  best: number;
  values: Record<string, number>;
  evals: number;
  /** Loss of the best candidate after each evaluation, oldest first. */
  history: number[];
};

export type OptimizeMethod = 'coordinate' | 'cmaes' | 'hybrid';

/* -------------------------------------------------------------------------- */
/* Budget bookkeeping                                                          */
/* -------------------------------------------------------------------------- */

function evaluationLimit(value: number): number {
  if (!Number.isFinite(value)) throw new RangeError('maxEvaluations must be finite');
  return Math.max(0, Math.floor(value));
}

/**
 * One place that owns "may I evaluate again?", the running best, and progress
 * reporting. Both algorithms go through it, so the budget contract cannot drift
 * between them.
 */
class Budget {
  readonly history: number[] = [];
  best = Number.POSITIVE_INFINITY;
  bestValues: Record<string, number>;
  evals = 0;
  private readonly limit: number;

  constructor(private readonly opts: OptimizeOptions) {
    this.limit = evaluationLimit(opts.maxEvaluations);
    this.bestValues = Object.fromEntries(opts.params.map((p) => [p.name, p.value]));
  }

  get exhausted(): boolean {
    return this.evals >= this.limit;
  }

  get aborted(): boolean {
    return this.opts.signal?.aborted === true;
  }

  /** @returns the loss, or `null` when the budget/abort stopped the evaluation. */
  async eval(values: Record<string, number>): Promise<number | null> {
    if (this.exhausted || this.aborted) return null;
    const loss = await this.opts.evaluate(values);
    this.evals += 1;
    if (loss < this.best) {
      this.best = loss;
      this.bestValues = { ...values };
    }
    this.history.push(this.best);
    this.opts.onProgress({ best: this.best, evals: this.evals, values: { ...this.bestValues } });
    return loss;
  }

  result(): OptimizeResult {
    return { best: this.best, values: this.bestValues, evals: this.evals, history: this.history };
  }
}

/* -------------------------------------------------------------------------- */
/* Deterministic RNG                                                           */
/* -------------------------------------------------------------------------- */

/** mulberry32 — 20 lines, seedable, and identical in every JS engine. */
function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box–Muller, one value per call (the second is cached). */
function makeGaussian(rng: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u = 0;
    let v = 0;
    let s = 0;
    do {
      u = rng() * 2 - 1;
      v = rng() * 2 - 1;
      s = u * u + v * v;
    } while (s === 0 || s >= 1);
    const f = Math.sqrt((-2 * Math.log(s)) / s);
    spare = v * f;
    return u * f;
  };
}

/* -------------------------------------------------------------------------- */
/* Parameter ↔ unit cube mapping                                                */
/* -------------------------------------------------------------------------- */

type Space = {
  names: string[];
  /** Normalized [0,1] coordinates of the current point. */
  toUnit(values: Record<string, number>): number[];
  toValues(unit: readonly number[]): Record<string, number>;
  /** Loss in real units between two normalized points, scaled by each axis's range. */
  spans: number[];
};

function makeSpace(params: OptimizeParam[]): Space {
  const names = params.map((p) => p.name);
  const spans = params.map((p) => Math.max(0, p.max - p.min));
  return {
    names,
    spans,
    toUnit(values) {
      return params.map((p, i) => {
        const v = values[p.name] ?? p.value;
        const u = (v - p.min) / (spans[i] || 1);
        return Number.isFinite(u) ? Math.min(1, Math.max(0, u)) : 0.5;
      });
    },
    toValues(unit) {
      const out: Record<string, number> = {};
      params.forEach((p, i) => {
        const u = Math.min(1, Math.max(0, unit[i] ?? 0.5));
        // Preserve narrow ranges and clamp floating-point endpoint roundoff.
        out[p.name] = Math.min(p.max, Math.max(p.min, p.min + u * (spans[i] ?? 0)));
      });
      return out;
    },
  };
}

/* -------------------------------------------------------------------------- */
/* Coordinate descent                                                          */
/* -------------------------------------------------------------------------- */

/**
 * Greedy pattern search: 5 sample points per axis, coarse-to-fine, halving the search
 * window on every full pass.
 *
 * One evaluation is spent on the starting point (so `history[0]` is always the
 * document's current quality), then each pass sweeps every axis. The window halves
 * after a pass that produced no improvement, which is what turns "5 points across the
 * whole range" into a converging search instead of a random walk.
 */
export async function coordinateDescent(opts: OptimizeOptions): Promise<OptimizeResult> {
  const budget = new Budget(opts);
  const space = makeSpace(opts.params);
  if (opts.params.length === 0) {
    await budget.eval({});
    return budget.result();
  }

  let current = space.toUnit(budget.bestValues);
  let currentLoss = await budget.eval(space.toValues(current));
  if (currentLoss === null) return budget.result();

  let window = 1;
  let stale = 0;
  while (!budget.exhausted && !budget.aborted && stale < 6) {
    let improved = false;
    for (let axis = 0; axis < opts.params.length && !budget.exhausted && !budget.aborted; axis++) {
      const base = current[axis] ?? 0.5;
      // 5 samples: the point itself is already evaluated, but re-evaluating it keeps
      // the sweep shape uniform and costs one render — cheap next to the clarity.
      const offsets = [-1, -0.5, 0, 0.5, 1];
      for (const o of offsets) {
        if (budget.exhausted || budget.aborted) break;
        const candidate = [...current];
        candidate[axis] = Math.min(1, Math.max(0, base + o * window * 0.5));
        const loss = await budget.eval(space.toValues(candidate));
        if (loss === null) break;
        if (loss < currentLoss - 1e-9) {
          current = candidate;
          currentLoss = loss;
          improved = true;
        }
      }
    }
    if (improved) stale = 0;
    else {
      stale += 1;
      window *= 0.5;
    }
  }
  return budget.result();
}

/* -------------------------------------------------------------------------- */
/* CMA-ES (diagonal)                                                           */
/* -------------------------------------------------------------------------- */

/**
 * A compact CMA-ES with per-axis (diagonal) step sizes.
 *
 * Population λ = min(16, 4 + ⌊3·ln n⌋) — the design doc's 16, shrunk for tiny n so a
 * 3-parameter document does not burn 16 renders per generation.
 *
 * State: mean `m` and per-axis std devs `D` (so C = diag(D²)), plus the two evolution
 * paths. Cumulative step-length control (CSA) adapts σ; the rank-μ and rank-1 updates
 * adapt D. Everything lives in the normalized unit cube, so params with wildly different
 * ranges (0.2..4 vs 0..1) do not fight over the step size.
 */
export async function cmaesLite(opts: OptimizeOptions): Promise<OptimizeResult> {
  const budget = new Budget(opts);
  const n = opts.params.length;
  if (n === 0) {
    await budget.eval({});
    return budget.result();
  }
  const space = makeSpace(opts.params);
  const rng = makeRng(opts.seed ?? 1);
  const gauss = makeGaussian(rng);

  const lambda = Math.max(4, Math.min(16, 4 + Math.floor(3 * Math.log(n))));
  const mu = Math.floor(lambda / 2);
  const rawWeights = Array.from({ length: mu }, (_, i) => Math.log(mu + 0.5) - Math.log(i + 1));
  const wSum = rawWeights.reduce((a, b) => a + b, 0);
  const weights = rawWeights.map((w) => w / wSum);
  const mueff = 1 / weights.reduce((a, w) => a + w * w, 0);

  // Standard CMA-ES constants (Hansen 2016, "The CMA Evolution Strategy: A Tutorial").
  const cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n);
  const cs = (mueff + 2) / (n + mueff + 5);
  const c1 = 2 / ((n + 1.3) * (n + 1.3) + mueff);
  const cmu = Math.min(1 - c1, (2 * (mueff - 2 + 1 / mueff)) / ((n + 2) * (n + 2) + mueff));
  const damps = 1 + 2 * Math.max(0, Math.sqrt((mueff - 1) / (n + 1)) - 1) + cs;
  const chiN = Math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n));

  let m = space.toUnit(budget.bestValues);
  let sigma = 0.3;
  const D = new Array<number>(n).fill(0.35);
  let pc = new Array<number>(n).fill(0);
  let ps = new Array<number>(n).fill(0);
  let generation = 0;

  const seedLoss = await budget.eval(space.toValues(m));
  if (seedLoss === null) return budget.result();
  let bestLoss = seedLoss;

  while (!budget.exhausted && !budget.aborted && generation < 60) {
    generation += 1;
    const population: { unit: number[]; loss: number }[] = [];

    for (let k = 0; k < lambda && !budget.exhausted && !budget.aborted; k++) {
      const unit = new Array<number>(n);
      for (let i = 0; i < n; i++) {
        unit[i] = Math.min(1, Math.max(0, (m[i] ?? 0.5) + sigma * (D[i] ?? 0.35) * gauss()));
      }
      const loss = await budget.eval(space.toValues(unit));
      if (loss === null) break;
      population.push({ unit, loss });
    }
    if (population.length === 0) break;

    population.sort((a, b) => a.loss - b.loss);
    const elite = population.slice(0, Math.min(mu, population.length));

    // Weighted recombination of the elite.
    const mNew = new Array<number>(n).fill(0);
    for (let i = 0; i < n; i++) {
      let acc = 0;
      for (let k = 0; k < elite.length; k++) acc += (weights[k] ?? 0) * (elite[k]?.unit[i] ?? 0);
      mNew[i] = acc;
    }

    // --- cumulative step-size control (CSA) --------------------------------
    const invD = D.map((d) => 1 / Math.max(1e-8, d));
    const y = mNew.map((v, i) => (v - (m[i] ?? 0)) / sigma);
    ps = ps.map((v, i) => (1 - cs) * v + Math.sqrt(cs * (2 - cs) * mueff) * (invD[i] ?? 1) * (y[i] ?? 0));
    const psNorm = Math.hypot(...ps);
    const hsig =
      psNorm / Math.sqrt(1 - Math.pow(1 - cs, 2 * (generation + 1))) / chiN < 1.4 + 2 / (n + 1) ? 1 : 0;
    pc = pc.map((v, i) => (1 - cc) * v + hsig * Math.sqrt(cc * (2 - cc) * mueff) * (y[i] ?? 0));

    // --- diagonal covariance adaptation (rank-1 + rank-μ) ------------------
    for (let i = 0; i < n; i++) {
      let rankMu = 0;
      for (let k = 0; k < elite.length; k++) {
        const z = ((elite[k]?.unit[i] ?? 0) - (m[i] ?? 0)) / sigma;
        rankMu += (weights[k] ?? 0) * z * z;
      }
      const d2 = (D[i] ?? 0.35) ** 2;
      const next = d2 + c1 * ((pc[i] ?? 0) ** 2 - d2) + cmu * (rankMu - d2);
      D[i] = Math.sqrt(Math.max(1e-8, next));
    }

    const newBest = elite[0]?.loss ?? bestLoss;
    const improved = newBest < bestLoss - 1e-9;
    // σ grows while the search keeps finding better points, shrinks when it stalls.
    sigma = Math.min(0.5, Math.max(0.01, sigma * Math.exp((cs / damps) * (psNorm / chiN - 1))));
    if (!improved && sigma <= 0.011) break;
    bestLoss = Math.min(bestLoss, newBest);
    m = mNew;
  }

  return budget.result();
}

/* -------------------------------------------------------------------------- */
/* Hybrid — what the UI runs                                                   */
/* -------------------------------------------------------------------------- */

/**
 * Coarse coordinate descent, then CMA-ES seeded from its result (DESIGN.md §6.6).
 *
 * The split is 40/60 by default: coordinate descent converges fast on the two or three
 * knobs that dominate most effects, and CMA-ES then does the fine, correlated work.
 * With a very small budget the coordinate pass takes everything, because CMA-ES needs
 * at least a few generations to be anything but expensive noise.
 */
export async function optimize(
  opts: OptimizeOptions & { method?: OptimizeMethod },
): Promise<OptimizeResult> {
  const method = opts.method ?? 'hybrid';
  if (method === 'coordinate') return coordinateDescent(opts);
  if (method === 'cmaes') return cmaesLite(opts);

  const total = evaluationLimit(opts.maxEvaluations);
  const coordinateBudget =
    opts.params.length <= 2 || total < 60 ? total : Math.max(20, Math.round(total * 0.4));

  const coarse = await coordinateDescent({ ...opts, maxEvaluations: coordinateBudget });
  const remaining = total - coarse.evals;
  if (remaining <= 0 || opts.signal?.aborted) return coarse;

  const seeded: OptimizeParam[] = opts.params.map((p) => ({
    ...p,
    value: coarse.values[p.name] ?? p.value,
  }));

  // The second stage reports progress with its own counters, so rebase them onto the
  // global count — the UI's progress bar is over the whole run, not one stage.
  const offset = coarse.evals;
  const refine = await cmaesLite({
    ...opts,
    params: seeded,
    maxEvaluations: remaining,
    onProgress: (p) =>
      opts.onProgress({
        best: Math.min(p.best, coarse.best),
        evals: offset + p.evals,
        values: p.best < coarse.best ? p.values : { ...coarse.values },
      }),
  });

  return {
    best: Math.min(coarse.best, refine.best),
    values: refine.best < coarse.best ? refine.values : coarse.values,
    evals: coarse.evals + refine.evals,
    history: [...coarse.history, ...refine.history.map((best) => Math.min(coarse.best, best))],
  };
}
