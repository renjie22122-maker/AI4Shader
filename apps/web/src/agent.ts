/**
 * Typed facade over the projected agent package
 * =============================================
 *
 * Same shape and the same single-cast trick as `ir.ts`, for the same reason: the browser
 * loads the **projected** JavaScript, which carries no types, and sprinkling casts across a
 * UI would hide the one place where the runtime/type gap actually exists.
 *
 * Two things about running the agent *in a browser* are worth stating plainly, because they
 * are properties of this host and not of the agent layer:
 *
 * 1. **The loop is synchronous-ish and can take seconds.** A real provider makes several
 *    network round trips. The panel therefore drives it with a progress callback and never
 *    blocks the render loop — the preview keeps animating while the agent thinks.
 *
 * 2. **An API key typed into a page is visible to that page.** There is no server to hold it.
 *    The panel keeps it in memory unless the user enables the remember-key checkbox.
 *    For anything but a personal machine, use the CLI with the key in the
 *    environment instead. This is a real limitation, not a disclaimer.
 */

import * as agentRuntime from '../../../packages/agent/dist/index.js';
import { CATALOG } from './ir.js';
import { createFragmentValidator } from './preview/validate.js';

import type { ShaderDoc } from '../../../packages/ir/src/index.js';

/** The authored agent module type — precise, and unavailable to the projection. */
type AgentModule = typeof import('../../../packages/agent/src/index.js');

const agent = agentRuntime as unknown as AgentModule;

export type {
  AgentResult,
  StageRecord,
  PatchContext,
  StopReason,
  LlmProvider,
  IntentProvider,
  PatchProvider,
} from '../../../packages/agent/src/index.js';

export const { runAgent, summariseRun, createStubProvider, createHttpProvider, DEFAULT_POLICY } = agent;

/**
 * The catalog the agent must use.
 *
 * Passed in explicitly rather than imported inside the agent: the agent layer deliberately
 * has no opinion about where node definitions come from, which is what lets it be tested
 * against a four-kind stub catalog.
 */
export const AGENT_CATALOG = CATALOG;

/** A one-line description of what a stop reason means, for the UI. */
export function describeStop(reason: string): string {
  switch (reason) {
    case 'clean':
      return '代码生成通过；画面和动态行为仍需预览确认';
    case 'budget':
      return '预算耗尽：换了几种策略仍未收敛，返回已达的最好版本';
    case 'provider-error':
      return '模型调用失败，或没有配置模型';
    case 'unrecoverable':
      return '本次修改未完成：请查看具体诊断';
    case 'aborted':
      return '已取消';
    default:
      return reason;
  }
}

/** Human-readable one-liner for a stage, used in the trace list. */
export function describeStage(stage: string): string {
  switch (stage) {
    case 'interpret':
      return '理解意图';
    case 'draft':
      return '起草图';
    case 'repair':
      return '确定性修复';
    case 'refine':
      return '模型修复';
    case 'emit':
      return '发射校验';
    default:
      return stage;
  }
}

/**
 * Providers the panel offers.
 *
 * `stub` is first because it is the one that always works: no key, no network, and it drafts
 * a complete six-target document. A user opening the workbench offline should still be able
 * to watch the whole loop run, and that is what makes the agent legible instead of magical.
 */
export type ProviderChoice = 'stub' | 'http';

export type AuthorRunOptions = {
  prompt: string;
  choice: ProviderChoice;
  basedOn?: ShaderDoc;
  /** Required for `http`; persistence is controlled by the caller's remember-key setting. */
  apiKey?: string;
  baseUrl?: string;
  model?: string;
  onStage?: (record: { stage: string; round: number; summary: string }) => void;
  signal?: AbortSignal;
};

/**
 * Run one authoring pass and return the result.
 *
 * Throws only for a *usage* problem (an `http` run with no key or no model). Every in-loop
 * failure comes back as a non-clean `AgentResult` with its diagnostics, because "the loop
 * gave up and here is why" is a result, not an exception.
 */
export async function authorOnce(opts: AuthorRunOptions): Promise<import('../../../packages/agent/src/index.js').AgentResult> {
  let provider: ReturnType<typeof createStubProvider> | ReturnType<typeof createHttpProvider>;

  if (opts.choice === 'stub') {
    provider = createStubProvider();
  } else {
    const apiKey = opts.apiKey?.trim();
    const model = opts.model?.trim();
    if (!apiKey) throw new Error('http provider 需要 API key');
    if (!model) throw new Error('http provider 需要 model 名称');
    provider = createHttpProvider({
      baseUrl: opts.baseUrl?.trim() || 'https://api.openai.com/v1',
      apiKey,
      model,
    });
  }

  const validator = opts.choice === 'http' ? createFragmentValidator() : null;
  try {
    return await runAgent({
      prompt: opts.prompt,
      catalog: AGENT_CATALOG,
      provider,
      reviewRequirements: opts.choice === 'http',
      ...(validator?.available ? { validate: validator.validate } : {}),
      ...(opts.basedOn ? { basedOn: opts.basedOn } : {}),
      ...(opts.onStage ? { onStage: opts.onStage } : {}),
      ...(opts.signal ? { signal: opts.signal } : {}),
    });
  } finally {
    validator?.dispose();
  }
}
