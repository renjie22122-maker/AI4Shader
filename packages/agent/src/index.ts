/**
 * AI4Shader — Agent layer public surface
 * ======================================
 *
 * `runAgent` is the whole orchestration loop behind one call. A caller supplies the user's
 * words and a provider, and gets back a document that emits cleanly, plus the trace of how
 * it got there.
 *
 * ```ts
 * // ⚠ REQUIRED FIRST. The six target emitters self-register via `registerTarget()`, and
 * // nothing the agent package imports pulls them in — the IR core deliberately has no
 * // dependency on any target. Without this line `runAgent` stops immediately with
 * // `provider-error` / `AG002_TARGETS_NOT_REGISTERED`, which is correct but surprising.
 * import '@ai4shader/ir/targets';
 *
 * import { CATALOG } from '@ai4shader/ir/catalog';
 * import { runAgent, createStubProvider } from '@ai4shader/agent';
 *
 * const result = await runAgent({
 *   prompt: '液态金属表面，要慢速流动的拉丝高光',
 *   catalog: CATALOG,
 *   provider: createStubProvider(), // no network, no key
 * });
 * if (result.clean) writeFileSync('out.shader.json', JSON.stringify(result.doc, null, 2));
 * else console.error(result.stopReason, result.remaining);
 * ```
 *
 * The loop is deterministic given a deterministic provider, which is what makes it
 * testable: `createStubProvider` replays scripted replies, so every branch of the state
 * machine — including the failure branches — is covered by the test suite without a model
 * or a network.
 */

export * from './types.js';
export { runAgent, summariseRun, targetsAvailable, analyse } from './loop.js';
export type { AgentOptions } from './loop.js';
export { repair, classify, planToPatches, adviceFrom, isSafeUniformName } from './repair.js';
export { buildContext, renderContext, nodeInventory, uniformInventory } from './context.js';
export type { PromptBundle } from './context.js';
export { createStubProvider, STUB_INTENTS } from './provider/stub.js';
export type { StubScript } from './provider/stub.js';
export { createHttpProvider, type HttpProviderOptions } from './provider/http.js';
export {
  parsePatch,
  parsePatchFor,
  parsePatchJson,
  parseSpec,
  parseIntentJson,
  extractJson,
  droppedOps,
  AG,
} from './provider/parse.js';
export { DEFAULT_POLICY, decideStrategy, errorsOf, diagnosticKey } from './policy.js';
export type { Policy, Strategy } from './policy.js';
export type { Catalog, CatalogLike, RepairAction, Repair } from './types.js';
