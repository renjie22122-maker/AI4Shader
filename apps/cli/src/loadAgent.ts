/**
 * `@ai4shader/agent` loading and provider selection for the CLI.
 * =============================================================
 *
 * Two jobs, both about the boundary between a command line and a model:
 *
 *   1. Load the agent package the same defensive way `loadIr.ts` loads the IR — dynamic
 *      imports with actionable errors, because a CLI must never die with
 *      `ERR_MODULE_NOT_FOUND` when the real problem is "you have not run the build".
 *   2. Decide **which provider** to use, and be explicit when there is none.
 *
 * ## Why provider selection lives here and not in the agent package
 *
 * The agent layer takes an `LlmProvider` and knows nothing about API keys, environment
 * variables or flags. That is what lets it be tested with a three-line stub. Everything
 * environment-shaped belongs at the edge, which is this file.
 *
 * ## Why "no provider" is a first-class outcome
 *
 * `author` and `refine` are useful without a model: refine can mechanically repair a
 * document, and both can report precise diagnostics. So a missing key is a *degraded* run,
 * not a usage error — the command says what it will and will not do, and the exit code
 * reflects whether the document actually emits. The one thing it must never do is silently
 * pretend a model was consulted.
 */

import { access } from 'node:fs/promises';
import { resolve } from 'node:path';

/* -------------------------------------------------------------------------- */
/* Minimal structural views of the agent package                               */
/* -------------------------------------------------------------------------- */

export type AgentProvider = {
  name: string;
  vision: boolean;
  complete(req: unknown, signal?: AbortSignal): Promise<{ text: string }>;
  interpret(input: unknown, signal?: AbortSignal): Promise<unknown>;
  propose(context: unknown, signal?: AbortSignal): Promise<unknown>;
};

export type AgentResult = {
  doc: unknown;
  emissions: Record<string, { source: string; diagnostics: Diagnostic[] }>;
  stopReason: string;
  trace: { stage: string; round: number; summary: string; diagnostics: Diagnostic[] }[];
  remaining: Diagnostic[];
  clean: boolean;
  usage?: { calls: number; promptTokens: number; completionTokens: number };
};

export type Diagnostic = { severity: string; code: string; message: string; fix?: string };

export type Agent = {
  runAgent(options: Record<string, unknown>): Promise<AgentResult>;
  createStubProvider(script?: unknown): AgentProvider;
  createHttpProvider(options: Record<string, unknown>): AgentProvider;
  summariseRun(result: AgentResult): string;
};

/* -------------------------------------------------------------------------- */
/* Loading                                                                     */
/* -------------------------------------------------------------------------- */

type Candidate = { specifier: string; probe?: string };

function agentCandidates(): Candidate[] {
  const here = new URL('.', import.meta.url);
  const fromDist = new URL('../../../packages/agent/dist/index.js', here).pathname;
  const fromSrc = new URL('../../../packages/agent/src/index.ts', here).pathname;
  return [
    { specifier: '@ai4shader/agent' },
    { specifier: fromDist, probe: fromDist },
    { specifier: fromSrc, probe: fromSrc },
  ];
}

async function fileExists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

async function firstImportable(candidates: Candidate[]): Promise<unknown | null> {
  for (const cand of candidates) {
    if (cand.probe) {
      const p = cand.probe.startsWith('/') && process.platform === 'win32' ? cand.probe.slice(1) : cand.probe;
      if (!(await fileExists(p))) continue;
    }
    try {
      return await import(cand.specifier);
    } catch {
      // Try the next candidate: a bare specifier fails in an unlinked checkout, an absolute
      // path fails when the build has not run. Neither failure is worth surfacing alone.
    }
  }
  return null;
}

/** Load the agent package, or throw the message a user can act on. */
export async function requireAgent(): Promise<Agent> {
  const mod = (await firstImportable(agentCandidates())) as Partial<Agent> | null;
  if (!mod?.runAgent || !mod.createStubProvider || !mod.createHttpProvider) {
    throw new Error(
      'cannot run the authoring commands without the agent package — run `npm run build` at the repo root\n' +
        '  looked for the `@ai4shader/agent` package and packages/agent/dist/index.js\n' +
        '  the agent layer needs the IR catalog and the target emitters to be built too',
    );
  }
  return mod as Agent;
}

/* -------------------------------------------------------------------------- */
/* Provider selection                                                          */
/* -------------------------------------------------------------------------- */

export type ProviderChoice =
  | { kind: 'stub' }
  | { kind: 'http'; baseUrl: string; apiKey: string; model: string; vision: boolean }
  | { kind: 'none'; why: string };

export type ProviderFlags = {
  /** `stub`, `http`, or undefined to infer from the environment. */
  provider?: string;
  apiKey?: string;
  baseUrl?: string;
  model?: string;
  vision?: boolean;
};

/** Default endpoint, so the common case needs no flags at all. Exported for the help text. */
export const DEFAULT_BASE_URL = 'https://api.openai.com/v1';

/**
 * Decide which provider to use, from flags first and the environment second.
 *
 * Environment variables are read here and nowhere else, so a reader can answer "where does
 * the API key come from?" in one grep. The order is flags > AI4SHADER_* > OPENAI_*.
 */
export function chooseProvider(flags: ProviderFlags, env: NodeJS.ProcessEnv = process.env): ProviderChoice {
  const explicit = flags.provider?.toLowerCase();

  if (explicit === 'stub') return { kind: 'stub' };
  if (explicit === 'none') {
    return { kind: 'none', why: 'requested with --provider none; only deterministic repair will run' };
  }

  const apiKey = flags.apiKey ?? env.AI4SHADER_API_KEY ?? env.OPENAI_API_KEY ?? '';
  const baseUrl = flags.baseUrl ?? env.AI4SHADER_BASE_URL ?? env.OPENAI_BASE_URL ?? DEFAULT_BASE_URL;
  const model = flags.model ?? env.AI4SHADER_MODEL ?? env.OPENAI_MODEL ?? '';

  if (explicit === 'http') {
    if (!apiKey) {
      return {
        kind: 'none',
        why: '--provider http needs an API key: pass --api-key, or set AI4SHADER_API_KEY / OPENAI_API_KEY',
      };
    }
    if (!model) {
      return {
        kind: 'none',
        why: '--provider http needs a model name: pass --model, or set AI4SHADER_MODEL / OPENAI_MODEL',
      };
    }
    return { kind: 'http', baseUrl, apiKey, model, vision: flags.vision ?? false };
  }

  // No explicit choice: use a real model only when one is fully configured, and say so.
  if (apiKey && model) return { kind: 'http', baseUrl, apiKey, model, vision: flags.vision ?? false };
  if (apiKey && !model) {
    return {
      kind: 'none',
      why:
        'an API key is set but no model name is: pass --model, or set AI4SHADER_MODEL / OPENAI_MODEL. ' +
        'Use `--provider stub` to run the deterministic offline provider instead',
    };
  }
  if (explicit === 'auto' || explicit === undefined) {
    return {
      kind: 'none',
      why:
        'no API key configured, so only deterministic repair will run. ' +
        'Pass `--provider stub` for the offline deterministic provider, or set AI4SHADER_API_KEY/OPENAI_API_KEY',
    };
  }
  return { kind: 'none', why: `unknown --provider \`${flags.provider}\`; expected stub, http or none` };
}

/** Turn a choice into a provider object, or `null` for the mechanical-only path. */
export function makeProvider(agent: Agent, choice: ProviderChoice): AgentProvider | null {
  switch (choice.kind) {
    case 'stub':
      return agent.createStubProvider();
    case 'http':
      return agent.createHttpProvider({
        baseUrl: choice.baseUrl,
        apiKey: choice.apiKey,
        model: choice.model,
        vision: choice.vision,
      });
    case 'none':
      return null;
  }
}

/** A short line describing the provider, for the run header. Never prints the key. */
export function describeProvider(choice: ProviderChoice): string {
  switch (choice.kind) {
    case 'stub':
      return 'stub (deterministic, offline)';
    case 'http':
      return `http ${choice.baseUrl} model=${choice.model}${choice.vision ? ' vision' : ''}`;
    case 'none':
      return `none — ${choice.why}`;
  }
}

/** Resolve a document path the way the other commands do (`-` means stdin). */
export function resolveDocPath(pathOrDash: string, cwd: string): string | null {
  return pathOrDash === '-' ? null : resolve(cwd, pathOrDash);
}
