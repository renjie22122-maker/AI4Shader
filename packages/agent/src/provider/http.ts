/**
 * AI4Shader — agent layer: the OpenAI-compatible HTTP provider
 * =============================================================
 *
 * The only part of the agent layer that talks to a network. Everything it does that is
 * not "make the request" comes from `prompt.ts` (what is sent) and `parse.ts` (what comes
 * back), so this file is deliberately thin: build the body, send it, read `choices[0]`,
 * hand the text to the parser.
 *
 * ## Three decisions worth stating out loud
 *
 * **1. Retries are transport-level only, and bounded.**
 * A network error, a 429 or a 5xx is the world being temporarily broken, so retrying is
 * cheap and right. A 4xx that is not 429 is *us* being wrong — a bad key, a bad model
 * name, a malformed body — and retrying it wastes the user's time on a request that can
 * only fail again. The two categories are separated explicitly in `isTransientStatus`,
 * and the backoff is capped at 250 ms **in total** so a test suite that exercises the
 * retry path cannot get slow. Hidden retries inside a provider would also break the
 * contract in `types.ts` ("no retries hidden inside — the loop is what knows the budget"),
 * which is why this is transport retry with a hard, small bound rather than a policy.
 *
 * "Transport-level" includes the *body*, not just the connection: a `200` whose bytes never
 * finished arriving is the same class of event as a connection that never opened, and this used
 * to treat them differently — the connection was retried twice, the body was not retried at all.
 * That asymmetry is what turned an intermittent mid-response drop into a hard failure reported as
 * `HTTP 200 … the response body could not be read`, with advice to check the API key that the
 * `200` had already proved correct.
 *
 * **2. The status and the provider's error body are surfaced verbatim.**
 * A swallowed 401 is the worst possible debugging experience: the user sees "the agent
 * failed" and has no way to learn that the key expired. So the thrown error carries the
 * status, the status text, the raw body (truncated only when enormous) and the request's
 * `purpose`, and it never paraphrases the provider's own message.
 *
 * **3. A patch that parses to zero surviving ops is NOT an error.**
 * `propose` returns the empty patch. The loop owns the budget, the diagnostics and the
 * decision of what an empty round means; throwing here would take that decision away and
 * turn a repairable round into a `provider-error` stop.
 *
 * ## Testability
 *
 * `fetchImpl` is injectable. That is not a testing hook bolted on — it is the only way
 * this file can be exercised at all in an environment with no network, and it makes the
 * request body itself assertable.
 */

import type { Diagnostic } from '@ai4shader/ir/types';

import type {
  IntentInput,
  IntentProvider,
  LlmMessage,
  LlmProvider,
  LlmReply,
  LlmRequest,
  PatchContext,
  PatchProvider,
  ShaderPatch,
  ShaderSpec,
} from '../types.js';
import { parseIntentJson, parsePatchJson, type CatalogLike } from './parse.js';
import { intentMessages, patchMessages } from './prompt.js';

/* -------------------------------------------------------------------------- */
/* Options                                                                     */
/* -------------------------------------------------------------------------- */

export type HttpProviderOptions = {
  /** Base URL *including* the version segment, e.g. `https://api.openai.com/v1`. */
  baseUrl: string;
  apiKey: string;
  model: string;
  /** Whether `IntentInput.image` is sent as an image part. */
  vision?: boolean;
  /** Injected for tests. Defaults to `globalThis.fetch`. */
  fetchImpl?: typeof fetch;
  /**
   * Retries for **retryable** failures only (network error, 429/5xx, an unreadable body).
   * Default 2.
   */
  maxRetries?: number;
  /** Request SSE by default; disable for gateways that only accept JSON responses. */
  stream?: boolean;
  /** Opt in for gateways supporting OpenAI's stream_options.include_usage. */
  includeUsage?: boolean;
  /**
   * Connection/JSON deadline in ms; SSE renews this idle deadline on each data chunk.
   * Default 120_000. `0` disables the timeout.
   */
  timeoutMs?: number;
  /**
   * The catalog `propose` validates replies against.
   *
   * Optional, and resolved at *call* time rather than construction time: a reply parsed
   * without a catalog can still be validated against `NODE_KINDS` (unknown kinds are
   * caught) but its ports cannot be checked, so callers should pass `CATALOG` here.
   * Existing nodes are always seeded from the proposal's current document.
   */
  catalog?: CatalogLike;
  /** Sent as `temperature` on every request unless the call overrides it. */
  temperature?: number;
  /** Sent as `max_tokens` on every request unless the call overrides it. */
  maxTokens?: number;
};

/**
 * The provider, widened with the two jobs and with the parse diagnostics.
 *
 * `interpret` and `propose` keep the contract's signatures exactly, so this object is
 * assignable to `IntentProvider` and `PatchProvider` and the loop can take it as either.
 * The diagnostics are exposed through `lastDiagnostics` (newest last) instead of through
 * the return type, because a parse that dropped ops must be *visible* without changing
 * the signature the loop depends on.
 */
export type HttpProvider = LlmProvider & IntentProvider & PatchProvider & {
  /** Parse diagnostics from each `interpret`/`propose` call, in call order. */
  readonly lastDiagnostics: Diagnostic[][];
  /** Override the catalog for subsequent `propose` calls. */
  setCatalog(catalog: CatalogLike): void;
};

/* -------------------------------------------------------------------------- */
/* Construction                                                                */
/* -------------------------------------------------------------------------- */

const DEFAULT_MAX_RETRIES = 2;

/**
 * Connection/JSON deadline; streaming bodies renew it as bytes arrive.
 *
 * It used to be 60 s, and that was the wrong number for this client: JSON fallback is non-streaming, so the
 * ceiling has to span the model's *generation* as well as the transfer. A large patch reply on a
 * slow link routinely took longer than a minute, and the abort landed **after** the provider had
 * already answered `200` — which is why the failure read as "HTTP 200, body unreadable" and why it
 * came and went. The ceiling is a hung-connection guard, not a quality-of-service target, so it is
 * generous; a caller that wants a tighter one passes `timeoutMs`.
 */
const DEFAULT_TIMEOUT_MS = 120_000;
/** Total backoff budget across all attempts. Small on purpose — see the header. */
const RETRY_BUDGET_MS = 250;
const MAX_ERROR_BODY = 2_000;

/**
 * What went wrong, in the one place the caller can act on it.
 *
 * The kinds exist because the *advice* differs, not because the messages do: a transport failure
 * means the provider never saw the request (so the key and the model are irrelevant), a read
 * failure means it answered and the bytes did not arrive (so retrying is the fix), and an envelope
 * failure means the bytes arrived and were not what we asked for (so retrying is pointless). The
 * old code collapsed all of these into one message that told the user to check their
 * configuration — which is actively wrong advice for two of the three.
 */
export type ProviderErrorKind = 'transport' | 'timeout' | 'read' | 'http' | 'envelope';

export class ProviderError extends Error {
  readonly kind: ProviderErrorKind;
  /**
   * `true` when the same request has a real chance of succeeding.
   *
   * `read` and `timeout` are retryable because nothing about the request was wrong — the bytes
   * simply did not arrive. `envelope` is not: the provider answered, and asking again produces the
   * same answer.
   */
  readonly retryable: boolean;

  constructor(kind: ProviderErrorKind, message: string, retryable: boolean, cause?: unknown) {
    super(message, cause === undefined ? undefined : { cause });
    this.name = 'ProviderError';
    this.kind = kind;
    this.retryable = retryable;
  }
}

/**
 * Build an OpenAI-compatible `/chat/completions` provider.
 *
 * ```ts
 * const provider = createHttpProvider({
 *   baseUrl: 'https://api.openai.com/v1',
 *   apiKey: process.env.OPENAI_API_KEY!,
 *   model: 'gpt-4o-mini',
 *   vision: true,
 *   catalog: CATALOG,
 * });
 * ```
 *
 * Throws at construction if `baseUrl`, `apiKey` or `model` is empty, and if no `fetch` is
 * available. Failing at construction beats failing on the first call: a missing key is a
 * configuration error, and the loop should not have to interpret it.
 */
export function createHttpProvider(opts: HttpProviderOptions): HttpProvider {
  const baseUrl = stripTrailingSlash(opts.baseUrl ?? '');
  if (!baseUrl) throw new Error('createHttpProvider: `baseUrl` is required (e.g. https://api.openai.com/v1).');
  if (!opts.apiKey) throw new Error('createHttpProvider: `apiKey` is required.');
  if (!opts.model) throw new Error('createHttpProvider: `model` is required.');

  const doFetch = opts.fetchImpl ?? globalThis.fetch;
  if (typeof doFetch !== 'function') {
    throw new Error(
      'createHttpProvider: no fetch available. Pass `fetchImpl` (older Node, or a sandboxed runtime).',
    );
  }

  const maxRetries = Math.max(0, opts.maxRetries ?? DEFAULT_MAX_RETRIES);
  const timeoutMs = Math.max(0, opts.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  const vision = opts.vision ?? false;

  let catalog: CatalogLike = opts.catalog ?? {};
  const lastDiagnostics: Diagnostic[][] = [];

  /** One round trip, with retries applied only to transient failures. */
  async function chat(req: LlmRequest, signal: AbortSignal | undefined): Promise<LlmReply> {
    const attemptLimit = maxRetries + 1;
    let retries = 0;
    let backoff = 0;
    let lastError: Error | undefined;

    for (let attempt = 1; attempt <= attemptLimit; attempt++) {
      let res: Response;
      /*
       * One abort controller per attempt, so `readBody` can tell "our own ceiling fired" from
       * "something else went wrong" — that distinction is the whole diagnosis for a `200` whose
       * body never arrived.
       */
      let timedOut = false;
      const ctrl = new AbortController();
      let timer: ReturnType<typeof setTimeout> | null = null;
      const activity = (): void => {
        if (timer) clearTimeout(timer);
        timer = timeoutMs > 0 ? setTimeout(() => {
              timedOut = true;
              ctrl.abort(new Error(`timeout after ${timeoutMs} ms`));
            }, timeoutMs) : null;
        (timer as unknown as { unref?: () => void } | null)?.unref?.();
      };
      activity();
      const attemptSignal = linkSignals(signal, ctrl.signal);
      const stopTimer = (): void => {
        if (timer) clearTimeout(timer);
      };

      try {
        try {
          res = await doFetch(`${baseUrl}/chat/completions`, {
            method: 'POST',
            headers: {
              'content-type': 'application/json',
              authorization: `Bearer ${opts.apiKey}`,
            },
            body: JSON.stringify(bodyOf(req, opts, vision)),
            signal: attemptSignal,
          });
        } catch (err) {
          // A caller-initiated abort is not a transport failure and must not be retried:
          // the loop asked us to stop.
          if (signal?.aborted) throw abortError(req, err);
          if (timedOut) {
            throw new ProviderError(
              'timeout',
              `LLM request failed (${req.purpose}): no response within ${timeoutMs} ms from ` +
                `${hostOf(baseUrl)} — nothing arrived at all, headers included.\n` +
                `  A host that answers in a browser but not here is usually blocked, proxied, or\n` +
                `  simply slower than the ceiling. Fix: retry, or raise \`timeoutMs\`.`,
              true,
              err,
            );
          }
          throw new ProviderError('transport', transportError(req, baseUrl, err), true, err);
        }

        if (!res.ok) {
          /*
           * The status is the error, so the body is only evidence — and evidence that fails to
           * load must not replace the finding. `errorBody` therefore never throws: a 502 from a
           * proxy routinely has *no* body at all, and reporting "empty body" instead of "HTTP 502"
           * would bury the one fact that matters.
           */
          const text = await errorBody(req, res, { timedOut: () => timedOut, timeoutMs, baseUrl });
          // 4xx that is not 429 is definitively our fault: no retry — see the header.
          throw httpError(req, res, text, maxRetries);
        }

        const reply = await replyFrom(req, res, { timedOut: () => timedOut, timeoutMs, baseUrl, activity });
        stopTimer();
        return reply;
      } catch (err) {
        /*
         * Retry anything the error itself calls retryable — which is where a body that never
         * finished arriving now lands. It used to propagate immediately, so an intermittent
         * mid-response drop (the reported "HTTP 200, body unreadable", which came and went) was a
         * hard failure while a *connection* failure was retried twice. The two are the same class
         * of event and deserve the same treatment.
         */
        stopTimer();
        if (signal?.aborted) throw err;
        lastError = err instanceof Error ? err : new Error(String(err));
        const retryable = err instanceof ProviderError ? err.retryable : false;
        if (!retryable || retries >= maxRetries || backoff >= RETRY_BUDGET_MS) throw lastError;
        retries++;
        backoff += await sleep(backoffFor(attempt), signal);
      }
    }

    throw lastError ?? new Error(`LLM request failed (${req.purpose}): retries exhausted.`);
  }

  const provider: HttpProvider = {
    name: `http:${opts.model}`,
    vision,
    lastDiagnostics,

    setCatalog(next: CatalogLike): void {
      catalog = next;
    },

    async complete(req: LlmRequest, signal?: AbortSignal): Promise<LlmReply> {
      return chat(req, signal);
    },

    async interpret(input: IntentInput, signal?: AbortSignal): Promise<ShaderSpec> {
      const req: LlmRequest = {
        messages: intentMessages(input, vision),
        json: true,
        purpose: 'intent',
        ...(opts.maxTokens !== undefined ? { maxTokens: opts.maxTokens } : {}),
        ...(opts.temperature !== undefined ? { temperature: opts.temperature } : {}),
      };
      const reply = await chat(req, signal);
      const parsed = parseIntentJson(reply.text);
      lastDiagnostics.push(parsed.diagnostics);
      return parsed.value;
    },

    async propose(context: PatchContext, signal?: AbortSignal): Promise<ShaderPatch> {
      const req: LlmRequest = {
        messages: patchMessages(context, catalog),
        json: true,
        purpose: `patch:${context.stage}`,
        ...(opts.maxTokens !== undefined ? { maxTokens: opts.maxTokens } : {}),
        ...(opts.temperature !== undefined ? { temperature: opts.temperature } : {}),
      };
      const reply = await chat(req, signal);
      const parsed = parsePatchJson(reply.text, catalog, context.doc);
      lastDiagnostics.push(parsed.diagnostics);
      // Zero surviving ops is a *valid* result: the loop has the budget and the
      // diagnostics, and it is the only thing that can decide what an empty round means.
      return parsed.value;
    },
  };

  return provider;
}

/* -------------------------------------------------------------------------- */
/* Request body                                                                */
/* -------------------------------------------------------------------------- */

/**
 * The OpenAI-compatible request body.
 *
 * `response_format: { type: 'json_object' }` is sent whenever `req.json` is set. It is the
 * difference between "the model usually emits JSON" and "the model emits JSON", and the
 * whole patch pipeline depends on the latter.
 */
function bodyOf(req: LlmRequest, opts: HttpProviderOptions, vision: boolean): Record<string, unknown> {
  const body: Record<string, unknown> = {
    model: opts.model,
    stream: opts.stream !== false,
    messages: req.messages.map((m) => encodeMessage(m, vision)),
  };
  if (req.json) body['response_format'] = { type: 'json_object' };
  if (opts.stream !== false && opts.includeUsage === true) body['stream_options'] = { include_usage: true };
  if (req.maxTokens !== undefined) body['max_tokens'] = req.maxTokens;
  if (req.temperature !== undefined) body['temperature'] = req.temperature;
  return body;
}

/**
 * One message, with the image part when there is one.
 *
 * A message with no image stays a plain string: some gateways reject the content-array
 * form for text-only messages, and there is no reason to send it.
 */
function encodeMessage(m: LlmMessage, vision: boolean): Record<string, unknown> {
  if (!m.image || !vision) return { role: m.role, content: m.content };
  return {
    role: m.role,
    content: [
      { type: 'text', text: m.content },
      { type: 'image_url', image_url: { url: m.image } },
    ],
  };
}

/* -------------------------------------------------------------------------- */
/* Response                                                                    */
/* -------------------------------------------------------------------------- */

/**
 * The state a body read needs to explain itself.
 *
 * `timedOut` is a **thunk**, and that is not a style choice: the ceiling fires *while* the body is
 * being read, so a boolean captured at call time is always `false` by the time the catch runs —
 * which silently reclassified every timeout as a generic read failure. The test that pins this is
 * "our own per-attempt ceiling firing mid-body is a timeout".
 */
type ReadState = { timedOut: () => boolean; timeoutMs: number; baseUrl: string; activity?: () => void };

/** Consume complete SSE events, including UTF-8/code points split across network chunks. */
async function readSse(res: Response, state: ReadState): Promise<LlmReply> {
  if (!res.body) throw new ProviderError('read', 'SSE response has no body', true);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  const reply: LlmReply = { text: '' };
  let buffer = '', data: string[] = [];
  let done = false, sawContent = false, bytes = 0;
  const dispatch = (): void => {
    if (!data.length) return;
    const payload = data.join('\n');
    data = [];
    if (payload.trim() === '[DONE]') { done = true; return; }
    let event: any;
    try { event = JSON.parse(payload); }
    catch { throw new ProviderError('envelope', 'Malformed JSON in SSE event', false); }
    if (!event || event.error) throw new ProviderError('envelope', 'SSE provider returned an error event', false);
    if (typeof event.model === 'string') reply.model = event.model;
    const choice = event.choices?.find((c: any) => c.index === 0 || c.index === undefined);
    if (choice?.finish_reason === 'length') throw new ProviderError('envelope', 'SSE output truncated at max_tokens; increase the output budget.', false);
    if (choice?.finish_reason === 'content_filter') throw new ProviderError('envelope', 'SSE output stopped by content_filter.', false);
    if (typeof choice?.delta?.content === 'string') {
      sawContent = true;
      reply.text += choice.delta.content;
    }
    if (event.usage) reply.usage = {
      ...(typeof event.usage.prompt_tokens === 'number' ? { promptTokens: event.usage.prompt_tokens } : {}),
      ...(typeof event.usage.completion_tokens === 'number' ? { completionTokens: event.usage.completion_tokens } : {}),
    };
  };
  try {
    while (!done) {
      const chunk = await reader.read();
      if (chunk.done) break;
      state.activity?.();
      bytes += chunk.value.byteLength;
      if (bytes > 16 * 1024 * 1024) throw new ProviderError('envelope', 'SSE response exceeds 16 MiB limit', false);
      buffer += decoder.decode(chunk.value, { stream: true });
      let newline: number;
      while (!done && (newline = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, newline).replace(/\r$/, '');
        buffer = buffer.slice(newline + 1);
        if (!line) dispatch();
        else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
      }
    }
    if (!done) throw new ProviderError('read', 'SSE ended before [DONE]; partial patch discarded.', true);
    if (!sawContent) throw new ProviderError('envelope', 'SSE reply contained no content.', false);
    return reply;
  } finally {
    try { await reader.cancel(); } catch { /* Preserve the original read failure. */ }
    reader.releaseLock();
  }
}

/** Pull JSON or SSE into the same complete LlmReply; partial patches never reach the loop. */
async function replyFrom(req: LlmRequest, res: Response, state: ReadState): Promise<LlmReply> {
  if (res.headers.get('content-type')?.includes('text/event-stream')) {
    let reply!: LlmReply;
    await readBody(req, res, state, async () => {
      reply = await readSse(res, state);
      return 'SSE complete';
    });
    return reply;
  }
  const raw = await readBody(req, res, state);
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch (err) {
    /*
     * The body arrived and is not JSON.
     *
     * Two shapes, and the difference is worth stating because it changes what the reader should
     * do: a *truncated* reply (the connection ended inside the JSON — the bytes are a prefix of a
     * real answer) versus a body that was never JSON to begin with (an HTML error page from a
     * proxy, a captive portal, a gateway that answered 200 with its own message). A prefix is worth
     * a retry; an error page is not, but it *is* worth reading, which is why the body is included.
     *
     * "Starts like JSON and does not end like JSON" is the test, rather than "does not end with a
     * brace" — an HTML page ends with `>` and would otherwise be classified as truncated and
     * retried three times against a proxy that will answer identically every time.
     */
    const trimmed = raw.trim();
    const looksTruncated =
      (trimmed.startsWith('{') || trimmed.startsWith('[')) && !/[}\]]$/.test(trimmed);
    throw new ProviderError(
      'envelope',
      `LLM returned a non-JSON envelope (${req.purpose}, HTTP ${res.status}, ${raw.length} chars` +
        `${looksTruncated ? ', and the text ends mid-structure — a truncated reply' : ''}): ` +
        `${truncate(raw)}\n` +
        `  content-type: ${res.headers?.get?.('content-type') ?? '(not exposed)'}\n` +
        `  Parse error: ${messageOf(err)}` +
        (looksTruncated
          ? `\n  A truncated reply usually means the connection was cut, or the provider hit its own\n` +
            `  output limit. Retrying is reasonable; raising \`maxTokens\` is the other fix.`
          : `\n  A 200 that is not JSON is usually a proxy or gateway answering on the provider's\n` +
            `  behalf. Check what ${hostOf(state.baseUrl)} actually returns for this request.`),
      looksTruncated,
      err,
    );
  }
  const body = parsed as {
    model?: unknown;
    choices?: { message?: { content?: unknown } }[];
    usage?: { prompt_tokens?: unknown; completion_tokens?: unknown };
  };
  const content = body?.choices?.[0]?.message?.content;
  if (typeof content !== 'string') {
    throw new ProviderError(
      'envelope',
      `LLM reply has no \`choices[0].message.content\` (${req.purpose}, HTTP ${res.status}): ${truncate(raw)}`,
      false,
    );
  }
  const reply: LlmReply = { text: content };
  if (typeof body.model === 'string') reply.model = body.model;
  const p = body.usage?.prompt_tokens;
  const c = body.usage?.completion_tokens;
  if (typeof p === 'number' || typeof c === 'number') {
    reply.usage = {
      ...(typeof p === 'number' ? { promptTokens: p } : {}),
      ...(typeof c === 'number' ? { completionTokens: c } : {}),
    };
  }
  return reply;
}

/**
 * Read the response body, and if that fails, say **why**.
 *
 * This is the fix for the failure that read `HTTP 200 … (the response body could not be read)`.
 * The old helper caught every error and returned that sentence as if it were the body, so the real
 * cause — an abort, a dropped connection — was thrown away, and the next step (`JSON.parse` of a
 * sentence) reported a *content* problem. The user was then told to check their API key, which was
 * never wrong: a `200` had already proved the key, the URL and the model were all fine.
 *
 * Three outcomes, all distinct:
 *
 *   * the bytes arrived → return them;
 *   * our own per-attempt ceiling fired → `timeout`, retryable, and the message names the ceiling
 *     and the fact that the provider had already answered (so the fix is a longer `timeoutMs` or a
 *     retry, not a different key);
 *   * anything else → `read`, retryable, carrying the error's own name and message.
 */
async function readBody(req: LlmRequest, res: Response, state: ReadState, consume?: () => Promise<string>): Promise<string> {
  try {
    const text = await (consume ? consume() : res.text());
    if (text.length === 0) {
      throw new ProviderError(
        'envelope',
        `LLM replied HTTP ${res.status} with an **empty body** (${req.purpose}).\n` +
          `  content-type: ${res.headers?.get?.('content-type') ?? '(not exposed)'}\n` +
          `  An empty 200 is not a reply: most often a proxy, a captive portal, or a gateway that\n` +
          `  swallowed the real response. Retrying is reasonable.`,
        true,
      );
    }
    return text;
  } catch (err) {
    if (err instanceof ProviderError) throw err;
    if (state.timedOut()) {
      throw new ProviderError(
        'timeout',
        `LLM request failed (${req.purpose}): the response body never finished arriving — the ` +
          `per-attempt timeout of ${state.timeoutMs} ms fired after ${hostOf(state.baseUrl)} had ` +
          `already answered HTTP ${res.status}.\n` +
          `  The provider saw the request and started replying, so the API key, the base URL and the\n` +
          `  model name are all correct. The usual cause is a reply that took longer to generate or\n` +
          `  transfer than the ceiling allows (JSON fallback is non-streaming, so the ceiling has to cover\n` +
          `  generation as well).\n` +
          `  Fix: retry, or raise \`timeoutMs\` for this provider (default ${DEFAULT_TIMEOUT_MS} ms).`,
        true,
        err,
      );
    }
    throw new ProviderError(
      'read',
      `LLM request failed (${req.purpose}): the response body could not be read after ` +
        `HTTP ${res.status} — ${messageOf(err)}\n` +
        `  The status line arrived and the body did not, so the request itself was accepted: this is\n` +
        `  a transport failure mid-response, not a configuration problem.\n` +
        `  In the order these actually happen:\n` +
        `    1. the connection was cut or stalled while the body was in flight (retry);\n` +
        `    2. a proxy, VPN or browser extension interfered with the response;\n` +
        `    3. the page was navigated or the run was cancelled — check for an abort that is not ours.`,
      true,
      err,
    );
  }
}

/* -------------------------------------------------------------------------- */
/* Errors                                                                      */
/* -------------------------------------------------------------------------- */

/**
 * The body of a *failed* response, where a missing body is not itself the story.
 *
 * Losing the provider's explanation is annoying; losing the status code because the explanation
 * failed to load is a real diagnosis bug. So this returns a placeholder and lets `httpError` carry
 * the status, which is the part the caller can act on.
 */
async function errorBody(req: LlmRequest, res: Response, state: ReadState): Promise<string> {
  try {
    return await readBody(req, res, state);
  } catch (err) {
    return `(the error body could not be read: ${messageOf(err)})`;
  }
}

/**
 * An HTTP failure, carrying everything needed to diagnose it without a second run.
 *
 * The kind is `http` and the retryability is the status's own: a 429 or a 5xx is the world being
 * briefly broken, a 401 is us being wrong. This used to be a plain `Error`, which after the retry
 * rewrite would have quietly stopped retrying 429/5xx — the retry test in the suite is what pins it.
 */
function httpError(req: LlmRequest, res: Response, body: string, maxRetries: number): Error {
  const transient = isTransientStatus(res.status);
  const verdict = transient
    ? `transient (429/5xx); retried up to ${maxRetries} time(s) with transport backoff`
    : 'not transient (4xx other than 429) — retrying would fail identically, so it was not retried';
  return new ProviderError(
    'http',
    `LLM request failed (${req.purpose}): HTTP ${res.status} ${res.statusText} — ${verdict}.\n` +
      `Provider response body:\n${truncate(body)}`,
    transient,
  );
}

function abortError(req: LlmRequest, cause: unknown): Error {
  return new Error(`LLM request aborted by the caller (${req.purpose}).`, { cause });
}

function messageOf(err: unknown): string {
  if (err instanceof Error) return err.name === 'TimeoutError' ? 'request timed out' : err.message;
  return String(err);
}

/**
 * A transport failure, explained.
 *
 * A browser reports every failure that is not an HTTP response as the same six words —
 * `Failed to fetch` — and a user reading that has no way to tell a blocked host from a CORS
 * refusal from a typo in the base URL. That ambiguity is not cosmetic: the most common cause
 * by far is that the endpoint is simply **not reachable from this machine**, which no amount
 * of re-typing the key will fix. This endpoint was written against a hardcoded
 * `https://api.openai.com/v1`, so a machine that cannot reach OpenAI produced a failure that
 * named neither the host nor the reason.
 *
 * So the message names the endpoint and the host, states the one thing that *is* certain
 * (no response arrived, therefore the provider never saw the request — which rules out the
 * key, the model name and the body), and lists the causes in the order they actually occur.
 *
 * The leading `network error` phrase is load-bearing: it is what the provider's own tests
 * match on, and what tells a reader at a glance which class of failure this is.
 */
function transportError(req: LlmRequest, baseUrl: string, cause: unknown): string {
  return (
    `LLM request failed (${req.purpose}): network error — ${messageOf(cause)}\n` +
    `  endpoint: ${baseUrl}/chat/completions\n` +
    `  host:     ${hostOf(baseUrl)}\n` +
    `  No response arrived at all, so the provider never saw this request. That rules out the\n` +
    `  API key, the model name and the request body. In the order these actually happen:\n` +
    `    1. the host is unreachable from this machine — offline, blocked, or proxy-only;\n` +
    `    2. a browser refused it for CORS: the endpoint must send Access-Control-Allow-Origin\n` +
    `       for this page's origin, and allow the Authorization header;\n` +
    `    3. the base URL has a typo, or a path the provider does not serve.\n` +
    `  Test the endpoint on its own, without the agent:\n` +
    `    curl -sS -o /dev/null -w '%{http_code}\\n' ${baseUrl}/models -H 'Authorization: Bearer <key>'`
  );
}

/** Host of a base URL, or the raw string when it will not parse. Never throws. */
function hostOf(baseUrl: string): string {
  try {
    return new URL(baseUrl).host;
  } catch {
    return baseUrl;
  }
}

/* -------------------------------------------------------------------------- */
/* Transport helpers                                                           */
/* -------------------------------------------------------------------------- */

/**
 * `true` for the statuses worth another attempt.
 *
 * Exported so the rule is testable on its own, and so a caller can log *why* a failed
 * request was or was not retried without re-deriving the predicate.
 */
export function isTransientStatus(status: number): boolean {
  return status === 429 || status >= 500;
}

/**
 * Backoff for attempt N, capped so the *total* across attempts stays inside
 * `RETRY_BUDGET_MS`. Whole milliseconds because a fractional timer is noise.
 */
function backoffFor(attempt: number): number {
  return Math.min(60 * attempt, RETRY_BUDGET_MS);
}

/** Sleep, but wake immediately (and resolve 0) if the caller aborts. */
function sleep(ms: number, signal: AbortSignal | undefined): Promise<number> {
  if (ms <= 0) return Promise.resolve(0);
  return new Promise((resolve) => {
    const onAbort = (): void => {
      clearTimeout(timer);
      resolve(0);
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve(ms);
    }, ms);
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

/**
 * A signal that fires when **either** the caller's signal or this attempt's ceiling does.
 *
 * `AbortSignal.any` is not available everywhere this package runs, so the composite is built by
 * hand. The caller's signal is forwarded, not merged into the ceiling controller, because the
 * ceiling controller's abort *is* the evidence that the timeout fired — a caller abort landing in
 * the same controller would be indistinguishable from it and would be misreported as a timeout.
 */
function linkSignals(signal: AbortSignal | undefined, ceiling: AbortSignal): AbortSignal {
  if (!signal) return ceiling;
  const ctrl = new AbortController();
  const forward = (from: AbortSignal): void => {
    if (from.aborted) ctrl.abort(from.reason);
    else from.addEventListener('abort', () => ctrl.abort(from.reason), { once: true });
  };
  forward(signal);
  forward(ceiling);
  return ctrl.signal;
}

function truncate(s: string): string {
  return s.length > MAX_ERROR_BODY ? `${s.slice(0, MAX_ERROR_BODY)}… (${s.length} chars total)` : s;
}

function stripTrailingSlash(s: string): string {
  return s.trim().replace(/\/+$/, '');
}
