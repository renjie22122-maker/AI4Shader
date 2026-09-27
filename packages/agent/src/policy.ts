/**
 * AI4Shader — Round budgets and escalation policy
 * ===============================================
 *
 * Every stage of the loop has its own budget, and **exceeding a budget changes the
 * strategy rather than repeating the attempt**. That distinction is the whole point of this
 * file. The naive loop — "if it failed, ask again" — burns tokens reproducing the same
 * failure, and when it eventually succeeds it is by luck. A loop that escalates does
 * something structurally different at each level, so the set of things it can reach is
 * strictly larger.
 *
 * Reading `decideStrategy` top to bottom tells you everything the agent is allowed to do
 * when it is stuck.
 */

import type { Diagnostic, NodeKind, NodeSchema } from '@ai4shader/ir/types';
import type { CatalogLike } from './types.js';

export type Strategy =
  /** Deterministic, catalog-aware repair. No model call. The cheapest and best outcome. */
  | 'mechanical'
  /** Hand the diagnostics to the model with focused instructions. */
  | 'model-repair'
  /** Fall back to a simpler realisation of the same intent: drop optional features. */
  | 'simplify'
  /** Ask for the whole document again from the original intent. */
  | 'redraft'
  /** Report honestly and stop. */
  | 'give-up';

export type Policy = {
  /** Rounds per stage. A "round" is one full validate→repair cycle. */
  budget: {
    /** Deterministic repair passes. These are cheap, so a few are fine. */
    mechanical: number;
    /** Model repair passes. Expensive; keep this small. */
    modelRepair: number;
    /** Simplification attempts. At most one or two before redrafting is better. */
    simplify: number;
    /** Full redrafts. This is the last resort and it is costly. */
    redraft: number;
  };
  /**
   * Stop even if the budget is unspent when the diagnostic set has not improved for this
   * many consecutive rounds. Without it, a loop that alternates between two equally-bad
   * states spends its entire budget proving it is stuck.
   */
  plateauRounds: number;
  /** Hard ceiling on model calls per run, regardless of stage budgets. */
  maxModelCalls: number;
  /** Skip the model entirely for diagnostics the repairer understands. */
  preferMechanical: boolean;
  /**
   * Give up on a stage once this fraction of its budget is spent and progress is below
   * `minProgress`: the loop is not converging, and a fresh approach beats more rounds.
   */
  minProgress: number;
};

export const DEFAULT_POLICY: Policy = {
  budget: { mechanical: 3, modelRepair: 4, simplify: 2, redraft: 1 },
  plateauRounds: 2,
  maxModelCalls: 12,
  preferMechanical: true,
  minProgress: 0.5,
};

/** Per-stage state the loop threads through, so the decision is pure. */
export type StageState = {
  strategy: Strategy;
  /** Rounds already spent in this strategy. */
  rounds: number;
  /** Error-diagnostic count at the start, and now. Used to detect a plateau. */
  errorsAtStart: number;
  errorsNow: number;
  /** Consecutive rounds with no improvement in the error count. */
  plateau: number;
  /** Distinct diagnostic codes seen so far in this stage, in first-seen order. */
  codesSeen: string[];
};

export function initialStageState(errorsAtStart: number): StageState {
  return {
    strategy: 'mechanical',
    rounds: 0,
    errorsAtStart,
    errorsNow: errorsAtStart,
    plateau: 0,
    codesSeen: [],
  };
}

/** Record the outcome of a round and return the next state. */
export function advance(state: StageState, errorsNow: number, codes: string[], policy: Policy): StageState {
  const improved = errorsNow < state.errorsNow;
  const codesSeen = [...state.codesSeen];
  for (const c of codes) if (!codesSeen.includes(c)) codesSeen.push(c);
  return {
    ...state,
    rounds: state.rounds + 1,
    errorsNow,
    plateau: improved ? 0 : state.plateau + 1,
    codesSeen,
  };
}

/**
 * The escalation ladder.
 *
 * Order of checks matters and each one has a reason:
 *
 *  1. **Nothing wrong** -> done. (Callers normally check this, but the function is total.)
 *  2. **No improvement for `plateauRounds`** -> escalate immediately, regardless of budget
 *     left. Being stuck is not the same as being out of budget, and the correct response
 *     to being stuck is to change approach, not to spend the remaining rounds the same way.
 *  3. **A code that has already survived a repair of this strategy** -> escalate. This is
 *     the "same error twice" rule from the design doc; a second identical failure means the
 *     fix is wrong, not that it needs repeating.
 *  4. Otherwise, move within the ladder while the current strategy has budget.
 *  5. When every strategy is exhausted -> `give-up`, which the loop reports honestly rather
 *     than pretending the document is finished.
 */
export function decideStrategy(state: StageState, policy: Policy): Strategy {
  if (state.errorsNow === 0) return state.strategy;

  const { budget } = policy;

  if (state.plateau >= policy.plateauRounds) return next(state.strategy);

  switch (state.strategy) {
    case 'mechanical':
      if (state.rounds < budget.mechanical) return 'mechanical';
      return 'model-repair';
    case 'model-repair':
      if (state.rounds < budget.modelRepair) return 'model-repair';
      return 'simplify';
    case 'simplify':
      if (state.rounds < budget.simplify) return 'simplify';
      return 'redraft';
    case 'redraft':
      if (state.rounds < budget.redraft) return 'redraft';
      return 'give-up';
    case 'give-up':
      return 'give-up';
  }
}

function next(s: Strategy): Strategy {
  switch (s) {
    case 'mechanical':
      return 'model-repair';
    case 'model-repair':
      return 'simplify';
    case 'simplify':
      return 'redraft';
    default:
      return 'give-up';
  }
}

/** Does this strategy need the model? Drives the call budget and the offline story. */
export function needsModel(s: Strategy): boolean {
  return s === 'model-repair' || s === 'simplify' || s === 'redraft';
}

/* -------------------------------------------------------------------------- */
/* Diagnostic classification                                                   */
/* -------------------------------------------------------------------------- */

/**
 * Codes that no patch can address. Recognising these is what stops the loop from spending
 * its budget on a problem that is not in the document.
 *
 * Kept as a table with a reason per code, because "why can't we fix this?" is the question
 * a user asks when the agent gives up.
 *
 * ## What is *not* in this table
 *
 * `IR008_NO_OUTPUT` and `EM001_NO_OUTPUT_NODE` used to be, with the reasoning that a graph
 * with no sink "has to be built, which is a drafting job". A real deepseek-chat draft proved
 * that wrong: it produced six good nodes and forgot the sink, and the loop — reading this
 * table — declared the document unrecoverable after three model calls and emitted a flat
 * colour. Adding a sink is a mechanical edit (`repair.ts` → `addOutputNode`), and the table
 * is now reserved for problems that genuinely live outside the document: a kind the catalog
 * does not have, a target with no emitter, a host with no provider.
 *
 * The general lesson is worth more than the two deletions: **a document-level diagnostic is
 * not the same thing as an unactionable one.** The old table conflated "this complaint does
 * not name a node and a port" with "there is nothing to do about it".
 */
const UNRECOVERABLE: Record<string, string> = {
  IR002_UNKNOWN_KIND:
    'the requested node kind does not exist in the catalog; only a different design can fix it',
  EM000_UNKNOWN_TARGET: 'the target format is not implemented; no patch can add one at runtime',
  /*
   * A graph that reads a surface input, emitted to a target with no surface entry point.
   *
   * This one is worth spelling out because it is the first code in this table that a *valid,
   * correct* document can produce. Nothing is wrong with the graph; the emitter simply cannot
   * write it for that target yet. The model cannot fix it either — the entry point lives in
   * `src/targets/`, and the only document edit that would silence the diagnostic is deleting
   * the surface nodes, i.e. throwing away exactly what the user asked for.
   *
   * Without this entry the loop classified it as unrecognised, asked the model, watched the
   * model edit the document, and asked again — twelve model calls to reach the conclusion the
   * first round already had. A capability gap belongs in this table more than a graph error
   * does: it is by definition outside the document.
   */
  EM002_SURFACE_UNSUPPORTED:
    'this target has no surface entry point yet, so a shading graph cannot be written for it; ' +
    'the gap is in the emitter, not in the document',
  AG001_NO_PROVIDER: 'no provider was configured, so nothing can draft or repair',
  AG002_TARGETS_NOT_REGISTERED:
    'the target emitters were never imported; this is a host wiring problem, not a document problem',
  AG003_TARGET_UNAVAILABLE: 'the requested target has no registered emitter',
};

export function unrecoverableReason(code: string): string | undefined {
  return UNRECOVERABLE[code];
}

/** Errors block emission. Warnings and info do not, and must never drive repair rounds. */
export function errorsOf(diagnostics: readonly Diagnostic[]): Diagnostic[] {
  return diagnostics.filter((d) => d.severity === 'error');
}

/**
 * A stable identity for a diagnostic.
 *
 * Two diagnostics with the same code on the same node and port are *the same problem*, and
 * the loop uses this to tell "still broken" from "broke something else". Deliberately
 * excludes the message text: messages carry values (`expected vec4, got float`) that change
 * as the repairer makes progress, and treating those as new problems would hide real
 * progress and inflate the plateau counter.
 */
export function diagnosticKey(d: Diagnostic): string {
  return [d.code, d.at?.node ?? '-', d.at?.port ?? '-'].join('|');
}

export function countDistinctCodes(diagnostics: readonly Diagnostic[]): string[] {
  const out: string[] = [];
  for (const d of diagnostics) if (!out.includes(d.code)) out.push(d.code);
  return out;
}

/* -------------------------------------------------------------------------- */
/* Catalog helpers the repairer needs                                          */
/* -------------------------------------------------------------------------- */

/** All kinds in the catalog, in a stable order. */
export function catalogKinds(catalog: CatalogLike): NodeKind[] {
  return Object.keys(catalog).sort() as NodeKind[];
}

/**
 * A short, stable list of kinds that can satisfy a given output type, used by the repairer
 * to suggest a concrete replacement instead of an open-ended "use something else".
 */
export function kindsProducing(catalog: CatalogLike, type: string, exclude: string[] = []): NodeKind[] {
  return catalogKinds(catalog).filter((k) => {
    if (exclude.includes(k)) return false;
    const schema = catalog[k]!;
    return schema.outputs.some((p) => p.type === type);
  });
}

/**
 * The inverse: kinds that accept a given input type, used to find something to fill an
 * unwired required input.
 */
export function kindsAccepting(catalog: CatalogLike, type: string, exclude: string[] = []): NodeKind[] {
  return catalogKinds(catalog).filter((k) => {
    if (exclude.includes(k)) return false;
    const schema = catalog[k]!;
    return schema.inputs.some((p) => p.type === type);
  });
}
