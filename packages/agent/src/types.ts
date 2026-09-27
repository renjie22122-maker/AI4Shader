/**
 * AI4Shader — Agent layer contracts
 * =================================
 *
 * The orchestration layer sits between "a human said something vague" and "the IR package
 * can act on it". It owns exactly one thing: a **budgeted loop** that turns intent into a
 * document that emits cleanly, and repairs its own failures along the way.
 *
 * ## The split that makes this testable
 *
 * The LLM is given two narrow, boring jobs and nothing else:
 *
 *   1. `IntentProvider`  — natural language (and later, images) -> `ShaderSpec`
 *   2. `PatchProvider`   — (current doc, diagnostics)          -> `ShaderPatch`
 *
 * Everything between those two calls is **deterministic, catalog-aware code**: validating,
 * emitting, classifying diagnostics, and turning them into patches. That is deliberate for
 * three reasons, in order of importance:
 *
 *   • **It is testable without a model.** A network call in the middle of a loop makes the
 *     loop untestable; a three-line `stubProvider` makes the whole thing deterministic.
 *   • **A type error does not need a language model.** `IR006_TYPE_MISMATCH` says exactly
 *     which port is wrong and what to insert. Asking a model to fix it costs a round trip
 *     and can make it worse; a lookup table costs nothing and cannot.
 *   • **The model cannot corrupt the document.** It emits patches, which are atomic and
 *     validated (`applyPatch`). The worst case is a rejected patch, never a broken doc.
 *
 * ## Round budgets, and why escalation is not "retry harder"
 *
 * Each stage has its own budget, and exceeding it triggers a **different strategy**, not
 * another attempt at the same one. Retrying an identical failed repair is the single most
 * common way an agent loop burns money and time, so the code makes it structurally
 * awkward: the stage counter and the strategy are chosen together in `policy.ts`.
 */

import type {
  Diagnostic,
  NodeKind,
  NodeSchema,
  ShaderDoc,
  ShaderPatch,
  ShaderSpec,
  TargetFormat,
} from '@ai4shader/ir/types';

/**
 * A few IR types are re-exported because the provider modules take them as arguments and
 * their callers import from this package, not from the IR. Doing it here — once, explicitly —
 * is what keeps "which module owns this type" answerable; the alternative was six modules
 * importing the same three names from `@ai4shader/ir/types` while two of them *re-exported*
 * them by accident, which type-checks as a local declaration that is not an export.
 */
export type { Diagnostic, ShaderDoc, ShaderPatch, ShaderSpec, TargetFormat };
export type { NodeKind, NodeSchema };
export type { PatchOp, ShaderGraph, ShaderNode, UniformDecl, ParamValue, Edge, ShaderType } from '@ai4shader/ir/types';

/**
 * The node catalog, as the agent layer needs it.
 *
 * Declared here, in the contract, and nowhere else: it was previously declared twice (in
 * `policy.ts` and in `provider/parse.ts`), which type-checks as two unrelated structural types
 * and would silently refuse a catalog that arrived through the other door.
 *
 * `Partial` because the *repairer and parser* legitimately work against a four-kind stub
 * catalog in tests. The emitter needs a total record, so `loop.ts` takes `Catalog` (all
 * kinds) — see its own note. Keeping the loose type here and the strict one there is
 * deliberate: it makes "this code needs the full catalog" a visible, greppable property
 * instead of a comment.
 */
export type CatalogLike = Partial<Record<NodeKind, NodeSchema>>;

/** The full catalog: every `NodeKind` present. Required wherever emitting happens. */
export type Catalog = Record<NodeKind, NodeSchema>;

/* -------------------------------------------------------------------------- */
/* LLM boundary                                                                */
/* -------------------------------------------------------------------------- */

/** A message in a provider-agnostic conversation. Kept minimal on purpose. */
export type LlmMessage = {
  role: 'system' | 'user' | 'assistant';
  content: string;
  /**
   * A base64 data URL for a reference image. Only ever set on a user message, and only
   * for providers that declare `vision: true`.
   */
  image?: string;
};

export type LlmRequest = {
  messages: LlmMessage[];
  /** Ask for JSON matching this shape. Providers that support it must honour it. */
  json?: boolean;
  /** Hard cap on the reply. Providers may clamp further. */
  maxTokens?: number;
  temperature?: number;
  /** Free-form label used for logging and cost accounting. */
  purpose: string;
};

export type LlmReply = {
  text: string;
  /** Which model actually answered, for `provenance`. */
  model?: string;
  usage?: { promptTokens?: number; completionTokens?: number };
  /** Wall-clock time, surfaced in the UI because it explains the latency. */
  ms?: number;
};

/**
 * The only thing the agent layer knows about a model.
 *
 * Implementations must be **pure with respect to the loop**: no *policy* retries hidden inside
 * (`complete` either answers or throws, and the loop decides what a throw means for the run), and
 * no mutation of the request. Retrying a *failed round* is a loop-level policy because the loop is
 * what knows the budget.
 *
 * The one exception is stated rather than implied, because it is what a reader would otherwise
 * mistake for a violation: the HTTP provider retries a **transport** failure in place (a network
 * error, a 429/5xx, an unreadable response body), at most twice, with a total backoff of 250 ms.
 * That is not the loop's retry — it is the same request, and it is bounded and invisible to the
 * budget on purpose: a request that never arrived is not a round, and charging the run for it
 * would make an intermittent connection look like a model that cannot do the task. `usage.calls`
 * therefore counts calls the **loop issued**, not HTTP attempts.
 */
export interface LlmProvider {
  readonly name: string;
  /** Whether `LlmMessage.image` will be honoured. */
  readonly vision: boolean;
  complete(req: LlmRequest, signal?: AbortSignal): Promise<LlmReply>;
}

/** Intent extraction: words (and optionally an image) -> a machine-checkable spec. */
export interface IntentProvider {
  interpret(input: IntentInput, signal?: AbortSignal): Promise<ShaderSpec>;
}

/**
 * Patch proposal.
 *
 * `context` is everything the model is allowed to see. It is assembled by `context.ts`
 * from the live document, and it is deliberately *small*: a node list, a uniform list, and
 * the diagnostics. Sending the whole IR (let alone the generated shader) would cost more,
 * be slower, and give the model more room to invent things that do not exist.
 */
export interface PatchProvider {
  propose(context: PatchContext, signal?: AbortSignal): Promise<ShaderPatch>;
  /**
   * Supply the catalog that replies are validated against.
   *
   * Optional, and the loop calls it before the first `propose` whenever a provider has it. It
   * belongs on this interface rather than only on the HTTP implementation because it is part
   * of *proposing*: a provider that parses model text has to judge whether a node kind is real
   * and whether a port exists, and it cannot do that without the catalog.
   *
   * Leaving it off this interface is what let both hosts forget to call it.
   * `createHttpProvider` silently defaulted to an empty catalog, every `add_node` failed
   * validation as `AG012_UNKNOWN_KIND`, and a real model's six-node draft arrived as an empty
   * patch — a failure that surfaces as `IR008_NO_OUTPUT`, i.e. looking exactly like a model
   * that cannot follow instructions. Providers that build patches directly (the stub) simply
   * do not implement it.
   */
  setCatalog?(catalog: CatalogLike): void;
}

export type IntentInput = {
  /** What the user typed. May be terse ("金属水面") or detailed. */
  prompt: string;
  /** Optional reference image as a data URL. */
  image?: string;
  /** Optional starting point: refine this document instead of drafting from nothing. */
  basedOn?: ShaderDoc;
  /** Constraints the caller insists on regardless of what the model proposes. */
  mustUseTargets?: TargetFormat[];
};

/* -------------------------------------------------------------------------- */
/* Loop context and results                                                     */
/* -------------------------------------------------------------------------- */

/**
 * The compact, model-facing view of a document.
 *
 * This is a *projection*, not the document: it carries the node ids and kinds, the
 * current uniform values with their ranges, and whatever the loop wants fixed. Nothing
 * here is authoritative — the doc is — and it exists so a prompt can be built without
 * shipping the whole graph.
 */
export type PatchContext = {
  /** Which stage asked. Changes the instructions the provider is given. */
  stage: 'draft' | 'repair' | 'refine';
  doc: ShaderDoc;
  /** Human-readable node inventory: `id (kind) — note`. */
  nodeLines: string[];
  /** Tunable uniforms with current value and range. */
  uniformLines: string[];
  /** What is wrong, if anything. Empty on the first draft. */
  diagnostics: Diagnostic[];
  /** The user's original words, when refining. */
  userIntent?: string;
};

/** Why the loop stopped. Always present, because "it stopped" is not an answer. */
export type StopReason =
  | 'clean' // emitted with no error diagnostics; the good ending
  | 'budget' // ran out of rounds at some stage
  | 'provider-error' // the model failed and could not be worked around
  | 'unrecoverable' // diagnostics that no patch in the catalog can address
  | 'aborted'; // the caller cancelled

export type StageName = 'interpret' | 'draft' | 'repair' | 'refine' | 'emit';

export type StageRecord = {
  stage: StageName;
  round: number;
  /** What happened, in one line, for the UI log. */
  summary: string;
  /** Diagnostics after the stage, already classified. */
  diagnostics: Diagnostic[];
  /** Set when the stage produced or changed a document. */
  hashBefore?: string;
  hashAfter?: string;
  ms?: number;
};

export type AgentResult = {
  /** The best document the loop reached. Never null: failures return the last good one. */
  doc: ShaderDoc;
  /** Emission per target, present only for documents that got that far. */
  emissions: Partial<Record<TargetFormat, { source: string; diagnostics: Diagnostic[] }>>;
  stopReason: StopReason;
  /** The full stage log. This is what the UI renders and what a bug report should contain. */
  trace: StageRecord[];
  /** Error diagnostics still outstanding when the loop stopped. Empty on `clean`. */
  remaining: Diagnostic[];
  /** All requested targets emitted without errors. Does not certify visual fidelity or physical behavior. */
  clean: boolean;
  usage?: { calls: number; promptTokens: number; completionTokens: number };
};

/* -------------------------------------------------------------------------- */
/* Repair                                                                       */
/* -------------------------------------------------------------------------- */

/**
 * What the deterministic repairer can do with a diagnostic.
 *
 * `patch` is the interesting case: it means the fix is known and does not need a model.
 * `advice` means the repairer recognises the problem but cannot fix it mechanically — the
 * text is handed to the provider as guidance, which is strictly better than passing the
 * raw diagnostic and hoping.
 * `hopeless` means no amount of patching will help (a missing node kind, an unreachable
 * graph), and the loop must escalate rather than spin.
 */
export type RepairAction =
  | { kind: 'patch'; patch: ShaderPatch; why: string }
  | { kind: 'advice'; advice: string; why: string }
  | { kind: 'hopeless'; reason: string };

export type Repair = {
  /** One repair per diagnostic, in the order they should be attempted. */
  actions: RepairAction[];
  /** Diagnostics the repairer does not understand at all. Always routed to the provider. */
  unknown: Diagnostic[];
};
