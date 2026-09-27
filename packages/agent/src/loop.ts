/**
 * AI4Shader — The orchestration loop
 * =================================
 *
 * One function, `runAgent`, that takes a sentence and returns a document which emits
 * cleanly. Everything else in this package exists to serve it.
 *
 * ## Shape of the loop
 *
 * ```
 *   interpret  ──►  draft  ──►  ┌───────────── repair cycle ─────────────┐  ──►  emit
 *   (model)         (model)     │  emit → diagnose → repair → re-emit    │      (all targets)
 *                               │  escalating: mechanical → model →      │
 *                               │              simplify → redraft        │
 *                               └────────────────────────────────────────┘
 * ```
 *
 * The repair cycle is the substance. Two properties make it worth having:
 *
 *   1. **Mechanical repair runs before any model call.** A type mismatch is a lookup, not a
 *      conversation. On a healthy catalog this closes most failures with zero model calls
 *      beyond the initial draft, which is the difference between a tool that feels instant
 *      and one that feels like a slot machine.
 *   2. **Escalation changes strategy, never repeats it.** See `policy.ts`. Being stuck is
 *      detected by a plateau counter, independent of remaining budget, because the correct
 *      response to "that did not work" is a different approach — not the same approach with
 *      a fresh allocation.
 *
 * ## What the loop refuses to do
 *
 * It never returns a document it has not verified, and it never claims success it cannot
 * show. `AgentResult.clean` is true only when every requested target emitted with zero
 * error diagnostics. When it gives up it returns the *best* document it reached together
 * with `remaining`, `stopReason` and the full `trace` — an honest failure with a
 * reproducible log is worth more than a confident wrong answer.
 */

import type {
  Diagnostic,
  ShaderDoc,
  ShaderPatch,
  ShaderSpec,
  TargetFormat,
} from '@ai4shader/ir/types';
import {
  IR_VERSION,
  TARGET_EMITTERS,
  analyzeGraph,
  applyPatch,
  canonicalJson,
  createDoc,
  emit as defaultEmit,
  hashDoc,
  stableHash,
} from '@ai4shader/ir';
import { buildContext } from './context.js';
import {
  type Catalog,
  type CatalogLike,
  type PatchContext,
  type StageRecord,
  type StopReason,
  type AgentResult,
  type IntentInput,
  type IntentProvider,
  type PatchProvider,
  type LlmProvider,
} from './types.js';
import {
  DEFAULT_POLICY,
  type Policy,
  type Strategy,
  advance,
  countDistinctCodes,
  decideStrategy,
  diagnosticKey,
  errorsOf,
  initialStageState,
  needsModel,
} from './policy.js';
import { adviceFrom, planToPatches, repair } from './repair.js';
// The error type, not the provider: `loop.ts` never constructs an HTTP provider, it only reads
// what one told it about a failure so that `AG004_PROVIDER_FAILED` can carry advice that fits.
import { ProviderError } from './provider/http.js';

/* -------------------------------------------------------------------------- */
/* Options                                                                     */
/* -------------------------------------------------------------------------- */

export type AgentOptions = {
  /** What the user asked for. */
  prompt: string;
  /** Optional reference image (data URL); requires a vision-capable provider. */
  image?: string;
  /** Start from an existing document instead of drafting. */
  basedOn?: ShaderDoc;
  /** Existing documents apply the requested edit by default. Use repair for diagnostics only. */
  mode?: 'refine' | 'repair';
  /** One source-level requirement review after a clean draft; does not claim visual validation. */
  reviewRequirements?: boolean;
  /** Optional host compiler check. Errors enter the same repair loop as emitter diagnostics. */
  validate?: (doc: ShaderDoc, emissions: AgentResult['emissions']) => Promise<Diagnostic[]>;
  /**
   * The **complete** node catalog.
   *
   * Not `CatalogLike`: the loop emits, and `EmitOptions.catalog` requires every `NodeKind` to
   * be present. Accepting a partial catalog here would type-check at the call site and fail
   * at emit time with `IR002_UNKNOWN_KIND` for every kind the host forgot — a runtime error
   * standing in for a compile-time one. The repairer, which runs against a four-kind stub in
   * tests, still takes `CatalogLike`; the strictness is applied exactly where emitting
   * happens, which is what makes "this needs the full catalog" greppable.
   */
  catalog: Catalog;
  /**
   * The model. Either a full `LlmProvider` that also implements `interpret`/`propose`
   * (as `createStubProvider` and `createHttpProvider` do), or a pair. Omit entirely to run
   * in **mechanical-only** mode: draft from `basedOn`, repair deterministically, and stop
   * with honest diagnostics when a model is genuinely needed.
   */
  provider?: (LlmProvider & Partial<IntentProvider & PatchProvider>) | null;
  /** Override the budgets and escalation thresholds. */
  policy?: Policy;
  /** Which targets must emit cleanly. Defaults to the document's own `targets`. */
  requireTargets?: TargetFormat[];
  /** Which target drives the repair cycle. GLSL by default: it is the cheapest and the one
   * designed to run as-is, so it produces the most actionable diagnostics. */
  driverTarget?: TargetFormat;
  /** Injection point for tests and for hosts that register targets lazily. */
  emit?: typeof defaultEmit;
  signal?: AbortSignal;
  /** Called after every stage, for a live UI. Must not throw back into the loop. */
  onStage?: (record: StageRecord) => void;
};

/* -------------------------------------------------------------------------- */
/* The loop                                                                    */
/* -------------------------------------------------------------------------- */

export async function runAgent(options: AgentOptions): Promise<AgentResult> {
  const policy = options.policy ?? DEFAULT_POLICY;
  const catalog = options.catalog;
  const doEmit = options.emit ?? defaultEmit;
  const driver: TargetFormat = options.driverTarget ?? 'glsl';
  const trace: StageRecord[] = [];
  const usage = { calls: 0, promptTokens: 0, completionTokens: 0 };

  let doc = options.basedOn ?? null;
  /**
   * Which targets must emit cleanly.
   *
   * Resolved **lazily**, because at this point in a drafting run the document does not exist
   * yet: computing it eagerly meant `doc?.targets` was always `undefined`, so every run
   * silently required only GLSL — the six-target guarantee the whole project rests on would
   * have been quietly weakened to one target, and only for drafted documents.
   *
   * The same ordering trap applies to the document's *own* `targets`, so a freshly drafted
   * document is created with `defaultTargets()` and the requirement is read back afterwards.
   */
  const defaultTargets = (): TargetFormat[] => {
    const available = Object.keys(TARGET_EMITTERS) as TargetFormat[];
    // Every registered target, so a drafted document is held to the same standard as the
    // shipped example. Falls back to GLSL only when nothing is registered at all, which the
    // caller is told about separately via `AG002_TARGETS_NOT_REGISTERED`.
    return available.length ? available : ['glsl'];
  };

  const requiredTargets = (): TargetFormat[] =>
    options.requireTargets ?? doc?.targets ?? defaultTargets();

  const record = (r: StageRecord): void => {
    trace.push(r);
    try {
      options.onStage?.(r);
    } catch {
      // A UI callback must never be able to abort the run.
    }
  };

  const abort = (stopReason: StopReason, remaining: Diagnostic[], emissions: AgentResult['emissions']): AgentResult => ({
    doc: doc ?? createDoc({ title: fallbackTitle(options.prompt), targets: [...(options.requireTargets ?? defaultTargets())] }),
    emissions,
    stopReason,
    trace,
    remaining,
    clean: false,
    usage,
  });

  /* ---- 0. give the provider the catalog it will be judged against ---------- */
  /*
   * This one line is the difference between the online provider working and not.
   *
   * A patch provider that parses model replies validates every op against a catalog — is this
   * node kind real, does that port exist, does this wire type-check. `createHttpProvider`
   * defaults that catalog to `{}`, and **nothing called `setCatalog`**: not the CLI, not the
   * web facade, not the loop. So every `add_node` was checked against an empty catalog,
   * reported `AG012_UNKNOWN_KIND`, and dropped. A real deepseek-chat draft of six nodes
   * becomes an empty patch, the graph stays empty, and the run ends at the budget with
   * `IR008_NO_OUTPUT` — which reads exactly like a model that cannot follow instructions.
   *
   * The requirement was on the caller, it was invisible in the type of the failure, and *both*
   * hosts missed it. That is not two mistakes; it is one API that lets a host forget. The loop
   * is the component that owns the catalog it validates against, so the loop is what supplies
   * it, and no host can get it wrong by omission again.
   *
   * `setCatalog` stays optional on the provider type: the stub builds patches directly and has
   * no use for it.
   */
  options.provider?.setCatalog?.(catalog);
  if (options.signal?.aborted) return abort('aborted', [], {});

  /* ---- 1. interpret ------------------------------------------------------- */
  const intentProvider = pickIntentProvider(options.provider);
  if (!doc) {
    if (!intentProvider) {
      return abort(
        'provider-error',
        [
          {
            severity: 'error',
            code: 'AG001_NO_PROVIDER',
            message: 'No document was supplied and no provider is configured to draft one.',
            fix: 'Pass `basedOn` (a ShaderDoc) to refine an existing document, or configure a provider.',
          },
        ],
        {},
      );
    }
    const t0 = Date.now();
    let spec: ShaderSpec;
    try {
      if (usage.calls >= policy.maxModelCalls) return abort('budget', [], {});
      /*
       * Counted before the call, not after it.
       *
       * `usage.calls` is the number of requests this run *issued*, and the budget ceiling is
       * enforced against it — so a request that was sent and then failed has to be in the count.
       * Incrementing after the `await` (which is what this did) meant a run whose second call
       * failed reported `modelCalls=1`, which reads as "the model was never called" and sends the
       * reader looking for a configuration problem that the count itself had just disproved.
       */
      usage.calls += 1;
      spec = await intentProvider.interpret(
        {
          prompt: options.prompt,
          ...(options.image !== undefined ? { image: options.image } : {}),
          ...(options.basedOn !== undefined ? { basedOn: options.basedOn } : {}),
        },
        options.signal,
      );
    } catch (err) {
      return abort('provider-error', [providerFailure(err)], {});
    }
    doc = createDoc({
      title: titleFromSpec(spec, options.prompt),
      spec,
      // Explicitly the caller's requirement when they gave one, otherwise every registered
      // target — so a drafted document is held to the same six-format standard as the
      // shipped example rather than to whatever `createDoc` happens to default to.
      targets: [...(options.requireTargets ?? defaultTargets())],
      provenance: {
        model: options.provider?.name ?? 'none',
        steps: [{ at: new Date(0).toISOString(), actor: 'agent', summary: `interpreted: ${spec.intent}` }],
      },
    });
    record({
      stage: 'interpret',
      round: 0,
      summary: `intent: ${spec.intent} (${spec.category})`,
      diagnostics: [],
      ms: Date.now() - t0,
    });
  }

  /* ---- 2. draft ---------------------------------------------------------- */
  const patchProvider = pickPatchProvider(options.provider);
  /**
   * Diagnostics from a rejected draft patch, carried into the repair cycle.
   *
   * See the long comment at the rejection site: these are the only diagnostics that explain
   * *why* the draft failed, and without them an empty graph looks like an unfixable problem.
   */
  let draftRejection: Diagnostic[] = [];
  let rejectedDraft: string | undefined;
  {
    const t0 = Date.now();
    // A draft is only needed when the graph is empty. `basedOn` documents that already have
    // a graph skip straight to repair, which is what makes "refine this" cheap.
    const needsDraft = doc.graph.nodes.length === 0;
    if (needsDraft && patchProvider) {
      const before = hashDoc(doc);
      const ctx = buildContext({ stage: 'draft', doc, catalog, userIntent: options.prompt });
      try {
        if (options.signal?.aborted) return abort('aborted', [], {});
        if (usage.calls >= policy.maxModelCalls) return abort('budget', [], {});
        usage.calls += 1; // issued, not succeeded — see the note on the interpret call above
        const patch = await patchProvider.propose(ctx, options.signal);
        const applied = applyPatch(doc, patch, catalog);
        if (applied.ok) {
          doc = applied.doc;
        } else {
          // A rejected draft leaves an empty graph, and `applyPatch` already produced a
          // precise, localised diagnostic (`PT001_UNKNOWN_NODE @ghost`). Those are carried
          // into the repair cycle rather than dropped: an empty graph only generates
          // `IR008_NO_OUTPUT`, which says *nothing* about why the draft failed and is
          // classified as hopeless — so dropping them turned "the model referenced a node
          // that does not exist" into "the graph is hopeless", wasting the whole budget on a
          // problem the model could have fixed in one more round if it had been told.
          draftRejection = applied.diagnostics.filter((d) => d.severity === 'error');
          rejectedDraft = JSON.stringify(patch);
          record({
            stage: 'draft',
            round: 0,
            summary: `draft patch rejected: ${draftRejection.map((d) => d.code).join(', ') || 'unknown'}`,
            diagnostics: applied.diagnostics,
            hashBefore: before,
            hashAfter: hashDoc(doc),
            ms: Date.now() - t0,
          });
        }
      } catch (err) {
        return abort('provider-error', [providerFailure(err)], {});
      }
      if (!draftRejection.length) {
        record({
          stage: 'draft',
          round: 0,
          summary: `drafted ${doc.graph.nodes.length} nodes`,
          diagnostics: [],
          hashBefore: before,
          hashAfter: hashDoc(doc),
          ms: Date.now() - t0,
        });
      }
    } else if (needsDraft) {
      record({
        stage: 'draft',
        round: 0,
        summary: 'no provider: starting from an empty graph',
        diagnostics: [],
      });
    }
  }

  const assertDriversRegistered = (): Diagnostic | null => {
    if (TARGET_EMITTERS[driver]) return null;
    return {
      severity: 'error',
      code: 'AG002_TARGETS_NOT_REGISTERED',
      message: `No emitter is registered for \`${driver}\`.`,
      fix: "Import '@ai4shader/ir/targets' (it self-registers all six emitters) before calling runAgent, or pass `emit` explicitly.",
    };
  };
  {
    const problem = assertDriversRegistered();
    if (problem) return abort('provider-error', [problem], {});
  }

  /* ---- 3. repair cycle --------------------------------------------------- */
  let state = initialStageState(0);
  const emissions: AgentResult['emissions'] = {};
  /**
   * Diagnostics carried into the *first* round from a rejected draft patch.
   *
   * A rejected draft leaves an empty graph, and an empty graph emits one useless
   * `IR008_NO_OUTPUT` while the actionable complaint — `PT001_UNKNOWN_NODE @ghost` — is
   * sitting in the rejection. Seeding the first round with that complaint is what lets the
   * model fix a specific mistake instead of being told the graph is hopeless.
   *
   * Consumed once: from round 2 on, the emitter is the only source of truth, because the
   * emitter sees the whole document while a patch rejection only saw the patch. Keeping both
   * would double-report one failure in two vocabularies and defeat the plateau detector.
   */
  let seed: Diagnostic[] = [...draftRejection];
  /**
   * Document hash at the end of the previous round; drives the changed escape hatch.
   *
   * This was a node *count*, and a count is the wrong question. A repair round that rewires,
   * re-params or swaps a node's kind leaves the count identical while changing the document
   * completely — so the loop read a stale diagnostic, concluded nothing had happened, and
   * aborted as `unrecoverable` while holding a fix it had just received. The hash answers
   * "is this the same document it was?" which is what the escape hatch actually means.
   */
  let lastDocHash: string | null = null;
  let lastErrors: Diagnostic[] = [];
  let bestDoc = doc;
  let bestErrorCount = Number.POSITIVE_INFINITY;
  let editPending = Boolean(options.basedOn && options.mode !== 'repair' && patchProvider && options.prompt.trim());
  let reviewPending = Boolean(!options.basedOn && options.reviewRequirements && patchProvider);

  for (let round = 0; round < 64; round++) {
    if (options.signal?.aborted) return abort('aborted', lastErrors, emissions);

    const t0 = Date.now();
    const emission = doEmit(doc, { catalog, target: driver });
    lastErrors = errorsOf(emission.diagnostics);
    // A clean emission always wins over a seed: the seed described a patch, not the document
    // that resulted from it.
    if (seed.length && lastErrors.length) lastErrors = [...lastErrors, ...seed];
    seed = [];

    // Keep the best document seen, not the latest: an escalation can make things worse, and
    // returning a worse document than one already in hand would be a regression the user
    // would have to debug.
    if (lastErrors.length < bestErrorCount) {
      bestErrorCount = lastErrors.length;
      bestDoc = doc;
    }

    if (lastErrors.length === 0) {
      record({
        stage: 'emit',
        round,
        summary: `${driver}: clean (${emission.source.length} bytes)`,
        diagnostics: emission.diagnostics,
        ms: Date.now() - t0,
      });

      // The driver is clean; now prove it for every required target. A document that emits
      // GLSL but not WGSL is not finished, and finding that out here is much cheaper than
      // finding it out in the user's engine.
      const failures: Diagnostic[] = [];
      for (const target of requiredTargets()) {
        if (!TARGET_EMITTERS[target]) {
          failures.push({
            severity: 'error',
            code: 'AG003_TARGET_UNAVAILABLE',
            message: `Target \`${target}\` has no registered emitter.`,
            fix: "Import '@ai4shader/ir/targets' before calling runAgent.",
          });
          continue;
        }
        const e = doEmit(doc, { catalog, target });
        const errs = errorsOf(e.diagnostics);
        if (errs.length === 0) {
          emissions[target] = { source: e.source, diagnostics: e.diagnostics };
        } else {
          for (const d of errs) failures.push({ ...d, message: `[${target}] ${d.message}` });
        }
      }

      if (failures.length === 0 && options.validate) {
        try {
          failures.push(...errorsOf(await options.validate(doc, emissions)));
        } catch (error) {
          failures.push({ severity: 'error', code: 'AG004_HOST_VALIDATION', message: String(error) });
        }
      }

      // Compiler success says nothing about whether the requested edit was made.
      // Run the edit (or the optional requirement review) before accepting a clean graph.
      if (failures.length === 0 && (editPending || reviewPending)) {
        if (options.signal?.aborted) return abort('aborted', [], emissions);
        if (usage.calls >= policy.maxModelCalls) return abort('budget', [], emissions);
        const isEdit = editPending;
        editPending = false;
        reviewPending = false;
        const before = hashDoc(doc);
        const ctx = buildContext({ stage: 'refine', doc, catalog, userIntent: options.prompt,
          advice: isEdit ? [] : [
            'Source-level requirement review: check EVERY original requirement against the graph and custom bodies.',
            'Check independent layers, lifecycle timing, stable randomness, downward motion, normal/refraction and alpha.',
            'Fix missing implementable behavior. Record unsupported simulation or background inputs in spec.outOfScope.',
            'Do not claim to have seen rendered pixels. Return an empty patch only if no source changes are needed.',
          ],
        });
        try {
          usage.calls += 1;
          const patch = await patchProvider!.propose(ctx, options.signal);
          if (options.signal?.aborted) return abort('aborted', [], emissions);
          const res = applyPatch(doc, patch, catalog);
          record({ stage: 'refine', round, summary: `${isEdit ? 'requested edit' : 'requirement review'}: ${patch.summary}`,
            diagnostics: res.diagnostics, hashBefore: before, hashAfter: hashDoc(res.doc) });
          if (!res.ok) return abort('unrecoverable', errorsOf(res.diagnostics), emissions);
          if (isEdit && hashDoc(res.doc) === before) {
            return abort('unrecoverable', [{ severity: 'error', code: 'AG005_NO_EDIT',
              message: 'The model returned no effective change for the requested edit.',
              fix: 'Specify the layer or parameter to change, or retry the edit.' }], emissions);
          }
          doc = res.doc;
          if (hashDoc(doc) !== before) {
            // All emitted sources and the old best candidate are now stale.
            for (const target of Object.keys(emissions)) delete emissions[target as TargetFormat];
            bestDoc = doc;
            bestErrorCount = Number.POSITIVE_INFINITY;
            lastDocHash = null;
            continue;
          }
        } catch (error) {
          return abort(options.signal?.aborted ? 'aborted' : 'provider-error', [providerFailure(error)], emissions);
        }
      }

      if (failures.length === 0) {
        doc = { ...doc, hash: hashDoc(doc) };
        bestDoc = doc;
        return {
          doc,
          emissions,
          stopReason: 'clean',
          trace,
          remaining: [],
          clean: true,
          usage,
        };
      }
      lastErrors = failures;
      record({
        stage: 'emit',
        round,
        summary: `${failures.length} error(s) in target emission or host compilation`,
        diagnostics: failures,
        ms: Date.now() - t0,
      });
    }

    // --- decide what to do about it -----------------------------------------
    const plan = repair(doc, lastErrors, catalog);
    const strategy = decideStrategy(state, policy);

    if (strategy === 'give-up') {
      return abort('budget', lastErrors, emissions);
    }

    // Stop only when *nothing* progress-capable remains.
    //
    // The repairer has three outputs and each means something different:
    //   • `patch`    — a fix to apply now. Progress without a model call.
    //   • `advice`   — the problem is understood but needs judgement. The model can act on
    //                  this, so it is progress-capable and must NOT trigger a stop.
    //   • `hopeless` — nothing can be done from a diagnostic.
    // `plan.unknown` holds the diagnostics the repairer did not recognise at all; those are
    // progress-capable for the same reason advice is, because the model sees them verbatim.
    //
    // There is one more escape hatch, and it matters more than it looks: **a document that is
    // still being edited is not hopeless, whatever its current diagnostics say.** A model that
    // just added the missing `output` node produces the same `IR008_NO_OUTPUT` for one more
    // round, because the emitter ran before the model's fix existed. Without `changed` the
    // loop reads that stale diagnostic, declares the graph unreachable, and throws away the
    // fix it just received.
    //
    // ## Why `hopeless` outranks `changed`
    //
    // The escape hatch used to be checked *after* `hopeless`, which made it able to defeat a
    // verdict it has no business touching. A real run proved the cost: a toon document emitted
    // to a target with no surface entry point produced `EM002_SURFACE_UNSUPPORTED`, the
    // repairer classified it `hopeless` (correctly — no document edit can add an entry point),
    // and then the model dutifully edited the document for **ten more rounds**, because every
    // edit set `changed` and `changed` alone kept the loop alive. Twelve model calls to learn
    // something the first round already knew.
    //
    // `changed` exists for a *stale* diagnostic — one the emitter produced before a fix landed.
    // `hopeless` is not stale: it says the diagnostic cannot be resolved by editing the
    // document at all. Editing more cannot un-stale it, so the hatch does not apply, and the
    // honest answer is to stop and say so. `plan.unknown` and `advice` still outrank it, which
    // is what keeps the hatch doing its actual job.
    const patchable = planToPatches(plan.actions).length;
    const hasAdvice = plan.actions.some((a) => a.kind === 'advice');
    const allHopeless = plan.actions.length > 0 && plan.actions.every((a) => a.kind === 'hopeless');
    const changed = hashDoc(doc) !== lastDocHash;
    const progressCapable = patchable > 0 || hasAdvice || plan.unknown.length > 0;
    if (!progressCapable && (allHopeless || !changed)) {
      return abort('unrecoverable', lastErrors, emissions);
    }

    const before = hashDoc(doc);
    let appliedSomething = false;

    // Choosing `mechanical` when the plan contains no mechanical action would burn a round
    // doing nothing and then escalate anyway — and, because a no-op round does not improve
    // the error count, it would also inflate the plateau counter and can trip the "stuck"
    // rule after a single wasted pass. Escalating straight away is both faster and more
    // honest: the ladder's job is to pick a *capability*, and mechanical has none here.
    const effective: Strategy =
      strategy === 'mechanical' && patchable === 0 ? 'model-repair' : strategy;

    if (effective === 'mechanical') {
      for (const { patch } of planToPatches(plan.actions)) {
        const res = applyPatch(doc, patch, catalog);
        if (res.ok) {
          doc = res.doc;
          appliedSomething = true;
        }
        // A rejected mechanical patch is not fatal: the loop advances the plateau counter and
        // the next round escalates. Logging every rejection would drown the trace.
      }
    } else if (needsModel(effective)) {
      if (usage.calls >= policy.maxModelCalls) return abort('budget', lastErrors, emissions);
      if (!patchProvider) {
        // No model and mechanical repair is exhausted. Reporting the remaining diagnostics
        // with their `fix` fields is the most useful thing available.
        return abort('provider-error', lastErrors, emissions);
      }
      const stage: PatchContext['stage'] = effective === 'redraft' ? 'draft' : 'repair';
      const advice = [
        ...adviceFrom(plan.actions),
        ...strategyHint(effective, doc),
        ...(rejectedDraft ? [
          'The entire draft below was rejected atomically. Its nodes DO NOT exist. Resubmit a COMPLETE corrected patch, including add_node operations; do not merely set_body on absent nodes.',
          `Rejected draft JSON: ${rejectedDraft}`,
        ] : []),
      ];
      const ctx = buildContext({
        stage,
        doc: effective === 'redraft' ? { ...doc, graph: { nodes: [], edges: [] } } : doc,
        catalog,
        diagnostics: lastErrors,
        userIntent: options.prompt,
        advice,
      });
      try {
        usage.calls += 1; // issued, not succeeded — see the note on the interpret call above
        const patch = await patchProvider.propose(ctx, options.signal);
        const res = applyPatch(doc, patch, catalog);
        if (res.ok) {
          doc = res.doc;
          appliedSomething = true;
          rejectedDraft = undefined;
        } else {
          lastErrors = [...lastErrors, ...res.diagnostics.filter((d) => d.severity === 'error')];
        }
      } catch (err) {
        record({
          stage: 'repair',
          round,
          summary: `provider failed: ${err instanceof Error ? err.message : String(err)}`,
          diagnostics: lastErrors,
          hashBefore: before,
          ms: Date.now() - t0,
        });
        return abort('provider-error', lastErrors, emissions);
      }
    }

    record({
      stage: effective === 'mechanical' ? 'repair' : effective === 'redraft' ? 'draft' : 'refine',
      round,
      // Report what actually happened, not what was planned. An earlier version said
      // "applied N patch(es)" using the *plan's* length, which read as success even when
      // every patch had been rejected — the trace is the artefact a human debugs from, so it
      // has to distinguish "tried" from "changed".
      summary: appliedSomething
        ? `${effective}: document changed; ${lastErrors.length} error(s) before`
        : `${effective}: no change applied; ${lastErrors.length} error(s) remain`,
      diagnostics: lastErrors,
      hashBefore: before,
      hashAfter: hashDoc(doc),
      ms: Date.now() - t0,
    });

    lastDocHash = hashDoc(doc);
    state = advance(state, lastErrors.length, countDistinctCodes(lastErrors), policy);
  }

  return abort('budget', lastErrors, emissions);
}

/* -------------------------------------------------------------------------- */
/* Helpers                                                                     */
/* -------------------------------------------------------------------------- */

/** A provider may be one object serving both roles, or two separate ones. */
function pickIntentProvider(p: AgentOptions['provider']): IntentProvider | null {
  if (!p) return null;
  if (typeof (p as IntentProvider).interpret === 'function') return p as IntentProvider;
  return null;
}

function pickPatchProvider(p: AgentOptions['provider']): PatchProvider | null {
  if (!p) return null;
  if (typeof (p as PatchProvider).propose === 'function') return p as PatchProvider;
  return null;
}

/**
 * Extra instructions per escalation level.
 *
 * These are the *only* place the strategies differ in what they tell the model, and they
 * are deliberately blunt. "Change as little as possible" at `simplify` and "start again"
 * at `redraft` are opposite instructions, which is exactly why escalation works where
 * retrying does not.
 */
function strategyHint(strategy: Strategy, doc: ShaderDoc): string[] {
  switch (strategy) {
    case 'simplify':
      return [
        'Previous repairs did not converge. SIMPLIFY: reach the same visual intent with FEWER nodes. ' +
          'Reduce cost inside layers (e.g. octaves), but preserve every explicitly requested layer and lifecycle. ' +
          'Document unavoidable approximations in spec.outOfScope; do not silently remove requested behavior.',
      ];
    case 'redraft':
      return [
        `Previous repairs did not converge after several attempts, and the graph has been reset to empty. ` +
          `Build the effect again from the stated intent, using only node kinds and port names from the ` +
          `catalog. Use enough nodes to preserve the required independent layers and lifecycle. ` +
          `Document unsupported behavior in spec.outOfScope instead of replacing the effect with generic noise.`,
      ];
    default:
      return [
        'Fix ONLY the listed problems. Change as little as possible: do not restructure the graph, do not ' +
          'rename existing nodes, and do not touch uniforms that are not mentioned.',
      ];
  }
}

/**
 * The one diagnostic a failed model call produces.
 *
 * `message` carries the provider's own words verbatim; `fix` is where this file has to be careful,
 * because it is the only part the reader is told to act on. It used to say *"Check the provider
 * configuration (base URL, API key, model name) and retry"* for **every** failure — including the
 * ones that had already disproved exactly that advice. A failure that arrives as
 * `HTTP 200 … the response body could not be read` proves the URL, the key and the model name are
 * all correct, and answering it with "check the API key" costs the reader an afternoon.
 *
 * So the advice follows the *kind* of failure, which the HTTP provider classifies at the point
 * where the information exists:
 *
 *   transport  nothing arrived            -> configuration is unverified, but the network is first
 *   timeout    accepted, never finished   -> retry, or allow a longer ceiling
 *   read       answered, body lost        -> the world, not the configuration: retry
 *   http       4xx                        -> the provider's own answer is the diagnosis
 *   http       429/5xx                    -> transient; retry
 *   envelope   a reply we cannot read     -> truncated (retry) or a proxy answering instead
 *
 * A plain `Error` still gets configuration advice, because the only providers that throw one are
 * the local ones (a stub that ran out of scripted replies, a provider built with no key) where
 * that advice is right.
 */
function providerFailure(err: unknown): Diagnostic {
  const message = err instanceof Error ? err.message : String(err);
  return {
    severity: 'error',
    code: 'AG004_PROVIDER_FAILED',
    message: `The model call failed: ${message}`,
    fix: providerAdvice(err),
  };
}

function providerAdvice(err: unknown): string {
  if (!(err instanceof ProviderError)) {
    return 'Check the provider configuration (base URL, API key, model name) and retry.';
  }
  switch (err.kind) {
    case 'transport':
      return (
        'No response arrived, so the provider never saw the request: the API key, the model name ' +
        'and the body are all still unverified. Check that the host is reachable from this machine ' +
        '(and, from a browser, that it allows CORS), then retry.'
      );
    case 'timeout':
      return (
        'The provider accepted the request and the round trip did not finish in time — nothing here ' +
        'streams, so the ceiling has to cover generation as well as transfer. Retry, or raise ' +
        '`timeoutMs` for this provider.'
      );
    case 'read':
      return (
        'The provider answered and the body did not arrive, so the request itself was accepted: this ' +
        'is a transport failure, not a configuration one. Retry — the provider already retried it ' +
        'once or twice internally. If it is reproducible, look for a proxy, VPN or connection reset ' +
        'between this machine and the endpoint.'
      );
    case 'http':
      return err.retryable
        ? 'HTTP 429/5xx: the endpoint is transiently unavailable. Retry, and consider a smaller ' +
            '`maxTokens` or a less busy model if it is a 5xx from the model host itself.'
        : 'HTTP 4xx: the request was rejected, so sending it again unchanged fails identically. ' +
            'The provider response body above names the field it objected to — check the API key, ' +
            'the model name and the request body.';
    case 'envelope':
      return err.retryable
        ? 'The reply arrived incomplete. Retry, and raise `maxTokens` if the provider is cutting ' +
            'the answer off at its own output limit.'
        : 'HTTP 200 with a body that is not the expected envelope: usually a proxy, gateway or ' +
            'captive portal answering on the provider behalf. Check what the endpoint actually ' +
            'returns for this request.';
  }
}

function titleFromSpec(spec: ShaderSpec, fallback: string): string {
  const t = spec.intent.trim();
  if (!t) return fallbackTitle(fallback);
  return t.length > 48 ? `${t.slice(0, 45)}…` : t;
}

function fallbackTitle(prompt: string): string {
  const t = prompt.trim().replace(/\s+/g, ' ');
  if (!t) return 'untitled shader';
  return t.length > 48 ? `${t.slice(0, 45)}…` : t;
}

/* -------------------------------------------------------------------------- */
/* Trace utilities                                                             */
/* -------------------------------------------------------------------------- */

/**
 * A stable one-line summary of a run, for logs and issue reports.
 *
 * Exists because "the agent produced a bad shader" is unactionable, while "round 3, code
 * `IR031_MISSING_INPUT` on `n_ramp.t`, stopReason `budget`" is a bug report. Deterministic
 * and free of timings so it can be diffed across runs.
 */
export function summariseRun(result: AgentResult): string {
  const stages = result.trace.map((t) => `${t.stage}#${t.round}`).join(' -> ');
  const remaining = result.remaining.length
    ? Array.from(new Set(result.remaining.map((d) => d.code))).sort().join(',')
    : 'none';
  const hash = result.doc.hash ?? hashDoc(result.doc);
  return [
    `stop=${result.stopReason}`,
    `clean=${result.clean}`,
    `hash=${hash.slice(0, 8)}`,
    `modelCalls=${result.usage?.calls ?? 0}`,
    `remaining=[${remaining}]`,
    `trace=${stages}`,
  ].join(' ');
}

/** A content hash of a patch plan, so a repeated plan can be detected across rounds. */
export function planSignature(patches: readonly ShaderPatch[]): string {
  return stableHash(canonicalJson(patches));
}

/** The diagnostic keys outstanding in a result, for assertions and UI. */
export function remainingKeys(result: AgentResult): string[] {
  return Array.from(new Set(result.remaining.map(diagnosticKey))).sort();
}

/** `IR_VERSION` re-exported so callers can assert compatibility without importing the IR. */
export const AGENT_IR_VERSION = IR_VERSION;

/** Cheap check the CLI uses to fail fast with a better message than a missing-emitter error. */
export function targetsAvailable(): TargetFormat[] {
  return Object.keys(TARGET_EMITTERS) as TargetFormat[];
}

/**
 * Analyse without emitting — used by tests, and by a host that wants to know whether a
 * document is worth a model call before spending one.
 *
 * Takes the full `Catalog` for the same reason `AgentOptions.catalog` does: `analyzeGraph`
 * resolves port names and types through it, so a partial catalog produces `IR003`/`IR005`
 * for every port the host did not supply — errors about the *host*, reported as errors about
 * the document. Requiring the real thing at the type level makes that impossible.
 */
export function analyse(doc: ShaderDoc, catalog: Catalog): Diagnostic[] {
  return analyzeGraph(doc.graph, catalog).diagnostics;
}
