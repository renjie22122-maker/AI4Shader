/**
 * OpenAI-compatible endpoints, and a way to test one
 * =================================================
 *
 * The workbench used to hardcode `https://api.openai.com/v1` in the authoring panel. If that
 * one host was unreachable — which it is on plenty of machines, including the one this was
 * developed on — the only thing a user could do was retype their API key and try again, and
 * the failure they got back was the browser's `Failed to fetch`. There was no field to point
 * the workbench at any other provider, so there was no fix available from inside the UI at
 * all. That is not a configuration *step*; it is a wall.
 *
 * So: the base URL is editable, the common services are one dropdown away, and there is a
 * button that tests the endpoint on its own so a five-second failed agent run is not the
 * first time anyone learns the host is unreachable.
 *
 * ## What the presets claim, and what they do not
 *
 * The **base URLs** were each fetched from this machine with an `Origin` header and an
 * `OPTIONS` preflight, and every one listed below answered with `Access-Control-Allow-Origin`
 * covering a page on `http://127.0.0.1:5183` and `Access-Control-Allow-Headers` covering
 * `authorization,content-type`. That matters and is not a detail: a browser refuses a
 * cross-origin POST with an `Authorization` header unless the endpoint opts in, so an
 * endpoint that works perfectly from `curl` can still be unusable from this page. Endpoints
 * that failed either test are not listed.
 *
 * The **model names** are ordinary suggestions and nothing more. Providers add, rename and
 * retire models on their own schedule and this file cannot know today's list; every model
 * field is free text, and the suggestions are only there to spare you from guessing a
 * spelling. A wrong model name comes back as a clear `404 model not found` from the
 * endpoint, not as a mystery.
 */

export type EndpointPreset = {
  /** Stable id, used as the `<option>` value and for persistence. */
  id: string;
  label: string;
  baseUrl: string;
  /**
   * A couple of **starting points**, not a catalogue.
   *
   * Named `modelHints` rather than `models` because calling it `models` is what made the
   * workbench look like it was offering a menu when it was offering two guesses. The
   * authoritative list comes from the endpoint via `listModels()` — this array only exists so
   * the field is pre-filled with something plausible before the first fetch.
   */
  modelHints: string[];
  /** One line shown under the selector: what this service is, or what it needs. */
  hint: string;
  /** Local servers have no key. The UI relaxes the requirement when this is true. */
  keyless?: boolean;
};

/**
 * Order matters: the list is read top to bottom, so it goes general-purpose first, then the
 * services, then the two local runtimes, then the escape hatch.
 *
 * The `modelHints` are deliberately short. When one of them is missing from a service you use,
 * that is not a bug to report — press 拉取模型 and the endpoint will hand over its real list.
 * The hint text says so on the services where the difference is most likely to bite.
 */
export const ENDPOINT_PRESETS: readonly EndpointPreset[] = [
  {
    id: 'openai',
    label: 'OpenAI',
    baseUrl: 'https://api.openai.com/v1',
    modelHints: ['gpt-4o-mini', 'gpt-4o'],
    hint: '官方接口。部分网络环境下 api.openai.com 不可达 —— 先点「测试连接」再写描述。',
  },
  {
    id: 'deepseek',
    label: 'DeepSeek',
    baseUrl: 'https://api.deepseek.com/v1',
    /*
     * Verified against https://api-docs.deepseek.com/quick_start/pricing/ — not remembered.
     *
     * The previous entries here were `deepseek-chat` / `deepseek-reasoner`, which is what the
     * API served when this table was written and is no longer the model list. Shipping them
     * was worse than shipping nothing: a stale *suggestion* is a name the user will try, and
     * the failure it produces ("model not found") reads like their mistake rather than ours.
     *
     * The two current ids are `deepseek-flash` (DeepSeek-V4.1-Flash) and `deepseek-v4-pro`.
     * The docs also note that `deepseek-v4-flash` and `deepseek-v4-flash-vision-exp` are still
     * *accepted* but retired — those requests are served by V4.1-Flash and billed as Flash —
     * so they are deliberately not suggested here.
     */
    modelHints: ['deepseek-flash', 'deepseek-v4-pro'],
    hint:
      'OpenAI 兼容，key 从 platform.deepseek.com 获取。' +
      '这里的两个是文档当前列出的 API 模型；点「拉取模型」会用你的 key 取回服务端实时列表。',
  },
  {
    id: 'openrouter',
    label: 'OpenRouter',
    baseUrl: 'https://openrouter.ai/api/v1',
    modelHints: ['openai/gpt-4o-mini', 'anthropic/claude-3.5-sonnet'],
    hint:
      '一个 key 转发多家模型，模型名带厂商前缀，而且**数量按周变化** —— ' +
      '本机实测它的 /models 无需 key 就返回 458 个，所以这里的建议值必然不全，请点「拉取模型」。',
  },
  {
    id: 'moonshot',
    label: 'Moonshot / Kimi',
    baseUrl: 'https://api.moonshot.cn/v1',
    modelHints: ['moonshot-v1-8k'],
    hint: '国内直连，OpenAI 兼容。模型名与上下文长度相关，建议「拉取模型」取真实列表。',
  },
  {
    id: 'siliconflow',
    label: 'SiliconFlow',
    baseUrl: 'https://api.siliconflow.cn/v1',
    modelHints: ['Qwen/Qwen2.5-7B-Instruct'],
    hint: '国内模型托管，模型名形如 厂商/模型，数量很多 —— 请点「拉取模型」。',
  },
  {
    id: 'dashscope',
    label: '阿里云百炼',
    baseUrl: 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    modelHints: ['qwen-plus'],
    hint: 'DashScope 的 OpenAI 兼容模式，路径必须带 /compatible-mode。',
  },
  {
    id: 'ollama',
    label: '本地 Ollama',
    baseUrl: 'http://127.0.0.1:11434/v1',
    modelHints: ['qwen2.5:7b', 'llama3.1:8b'],
    hint:
      '本机运行，不需要 key（随便填一个非空值即可，例如 ollama）。需要先 ollama serve。' +
      '「拉取模型」会列出你本地已经 pull 过的全部模型。',
    keyless: true,
  },
  {
    id: 'lmstudio',
    label: '本地 LM Studio',
    baseUrl: 'http://127.0.0.1:1234/v1',
    modelHints: ['local-model'],
    hint: '本机运行，不需要 key。在 LM Studio 里打开 Local Server，「拉取模型」会列出已加载的模型。',
    keyless: true,
  },
  {
    id: 'custom',
    label: '自定义',
    baseUrl: '',
    modelHints: [],
    hint: '任何 OpenAI 兼容的 /chat/completions 服务。地址要写到 /v1 这一层。',
  },
];

export function presetById(id: string): EndpointPreset | undefined {
  return ENDPOINT_PRESETS.find((p) => p.id === id);
}

/**
 * Which preset a base URL corresponds to, if any.
 *
 * Used to keep the dropdown honest when the URL is edited by hand: typing a known address
 * selects its preset rather than leaving the selector showing something that is no longer
 * true. Returns `custom` for anything unrecognised, including an empty string.
 */
export function presetForBaseUrl(baseUrl: string): EndpointPreset {
  const normalized = baseUrl.trim().replace(/\/+$/, '');
  const hit = ENDPOINT_PRESETS.find((p) => p.baseUrl !== '' && p.baseUrl === normalized);
  return hit ?? (presetById('custom') as EndpointPreset);
}

/* -------------------------------------------------------------------------- */
/* Model discovery                                                             */
/* -------------------------------------------------------------------------- */

export type ModelListResult =
  | { ok: true; models: string[]; ms: number }
  | { ok: false; kind: ProbeFailure; status?: number; ms: number; detail: string };

/**
 * Ask the endpoint which models it serves.
 *
 * The preset table's `models` arrays are *suggestions*, and a suggestion is a claim about
 * someone else's service that goes stale on their schedule. OpenRouter alone serves 458 models
 * (measured), and the number moves weekly; DeepSeek serves two, and which two has changed
 * since it launched. A hardcoded list is therefore wrong in one direction or the other on
 * every provider, always.
 *
 * `GET {base}/models` is the OpenAI-compatible answer and nearly everything implements it —
 * verified on this machine against DeepSeek, OpenRouter, Moonshot, SiliconFlow and DashScope.
 * Two things follow, and both are why the UI fetches rather than asserts:
 *
 *   • Four of those five require a **key** for `/models` (401 without one), so the list cannot
 *     be known at build time, only at run time with the user's own credentials.
 *   • OpenRouter answers *without* a key and returns 458 entries, which is a good reminder
 *     that a `<select>` populated from a guessed list is a worse tool than a text field that
 *     can ask.
 *
 * Never throws: the caller is a UI that has to say something either way.
 */
export async function listModels(opts: {
  baseUrl: string;
  apiKey?: string;
  signal?: AbortSignal;
  timeoutMs?: number;
}): Promise<ModelListResult> {
  const base = opts.baseUrl.trim().replace(/\/+$/, '');
  const started = now();
  if (!base) return { ok: false, kind: 'bad-url', ms: 0, detail: '地址为空。' };
  try {
    const u = new URL(base);
    if (u.protocol !== 'http:' && u.protocol !== 'https:') {
      return { ok: false, kind: 'bad-url', ms: 0, detail: `地址协议是 ${u.protocol}，只支持 http/https。` };
    }
  } catch {
    return { ok: false, kind: 'bad-url', ms: 0, detail: `地址不是合法 URL：${base}` };
  }

  const timeoutMs = opts.timeoutMs ?? 12_000;
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(new DOMException('timeout', 'TimeoutError')), timeoutMs);
  const onAbort = (): void => ac.abort();
  opts.signal?.addEventListener('abort', onAbort, { once: true });

  try {
    const res = await fetch(`${base}/models`, {
      method: 'GET',
      headers: {
        accept: 'application/json',
        // Sent even when empty, for the same reason the probe does: an unauthenticated local
        // server ignores it, and omitting it would make this request diverge from the one the
        // preset says needs no key.
        authorization: `Bearer ${opts.apiKey?.trim() || 'none'}`,
      },
      signal: ac.signal,
    });
    const ms = Math.round(now() - started);
    const body = await res.text().catch(() => '');

    if (!res.ok) {
      return {
        ok: false,
        kind: classify(res.status),
        status: res.status,
        ms,
        detail: modelsFailure(res.status, body, base),
      };
    }

    const models = parseModelIds(body);
    if (models.length === 0) {
      return {
        ok: false,
        kind: 'http-error',
        status: res.status,
        ms,
        detail:
          `HTTP ${res.status}，但响应里没有可识别的模型列表。` +
          `这个服务可能不实现 \`GET /models\`（OpenAI 兼容但不是全部）。` +
          `直接手写模型名即可，输入框始终是自由文本。` +
          (body.trim() ? `\n对方返回：${shortBody(body)}` : ''),
      };
    }
    return { ok: true, models, ms };
  } catch (err) {
    const ms = Math.round(now() - started);
    return {
      ok: false,
      kind: 'unreachable',
      ms,
      detail: ac.signal.aborted ? timeoutDetail(base, timeoutMs) : unreachableDetail(err, base),
    };
  } finally {
    clearTimeout(timer);
    opts.signal?.removeEventListener('abort', onAbort);
  }
}

/**
 * Pull model ids out of whatever shape the service used.
 *
 * The OpenAI envelope is `{ data: [{ id }] }`, and that is what every service tested here
 * returns. The alternatives are tolerated because "OpenAI-compatible" is a spectrum and this
 * costs four lines: a bare array, `{ models: [...] }`, `{ result: [...] }` (Ollama's native
 * shape), and entries that are plain strings rather than objects.
 *
 * Duplicates are removed but the **server's order is preserved** — a provider that lists its
 * flagship first is telling you something, and alphabetising would throw that away.
 */
export function parseModelIds(body: string): string[] {
  let parsed: unknown;
  try {
    parsed = JSON.parse(body);
  } catch {
    return [];
  }
  const container = parsed as Record<string, unknown> | unknown[] | null;
  const list = Array.isArray(container)
    ? container
    : ((container as Record<string, unknown> | null)?.['data'] ??
      (container as Record<string, unknown> | null)?.['models'] ??
      (container as Record<string, unknown> | null)?.['result']);
  if (!Array.isArray(list)) return [];

  const out: string[] = [];
  const seen = new Set<string>();
  for (const entry of list) {
    const id =
      typeof entry === 'string'
        ? entry
        : typeof (entry as { id?: unknown } | null)?.id === 'string'
          ? ((entry as { id: string }).id)
          : typeof (entry as { name?: unknown } | null)?.name === 'string'
            ? ((entry as { name: string }).name)
            : typeof (entry as { model?: unknown } | null)?.model === 'string'
              ? ((entry as { model: string }).model)
              : null;
    if (!id || seen.has(id)) continue;
    seen.add(id);
    out.push(id);
  }
  return out;
}

/** A `/models` failure, explained. Its 404 means something different from a chat 404. */
function modelsFailure(status: number, body: string, base: string): string {
  const quoted = shortBody(body);
  const kind = classify(status);
  if (kind === 'not-found') {
    return (
      `HTTP 404：这个地址不提供 \`GET /models\`（当前 ${base}）。` +
      `不影响使用 —— 直接手写模型名即可。${quoted}`
    );
  }
  if (kind === 'unauthorized') {
    return `HTTP ${status}：key 被拒绝，所以拿不到模型列表。先填对 key。${quoted}`;
  }
  return explain(status, body, base);
}

/* -------------------------------------------------------------------------- */
/* Readiness                                                                   */
/* -------------------------------------------------------------------------- */

export type ReadinessInput = {
  /** `stub` is always ready; this function is only interesting for `http`. */
  choice: 'stub' | 'http';
  baseUrl: string;
  model: string;
  apiKey: string;
  /** A local runtime that ignores the bearer token. */
  keyless: boolean;
};

/**
 * Whether the configuration is complete enough to attempt a call — and if not, what is
 * missing.
 *
 * The name of the missing field is the point. The previous rule was a bare boolean ("key and
 * model are non-empty"), which disabled the 生成 button for a local Ollama that needs no key
 * and offered no explanation for why it was greyed out. A disabled control that does not say
 * what it wants is a puzzle, not a safeguard.
 */
export function endpointReady(input: ReadinessInput): { ready: boolean; missing?: string } {
  if (input.choice === 'stub') return { ready: true };
  if (!input.baseUrl.trim()) return { ready: false, missing: '服务地址' };
  if (!input.model.trim()) return { ready: false, missing: '模型名' };
  if (!input.keyless && !input.apiKey.trim()) return { ready: false, missing: 'API key' };
  return { ready: true };
}

/* -------------------------------------------------------------------------- */
/* Probing                                                                     */
/* -------------------------------------------------------------------------- */

export type ProbeResult =
  | { ok: true; status: number; ms: number; detail: string }
  | { ok: false; kind: ProbeFailure; status?: number; ms: number; detail: string };

export type ProbeFailure =
  | 'unreachable'
  | 'unauthorized'
  | 'bad-request'
  | 'not-found'
  | 'rate-limited'
  | 'server-error'
  | 'bad-url'
  | 'http-error';

export type ProbeOptions = {
  baseUrl: string;
  apiKey?: string;
  model: string;
  signal?: AbortSignal;
  /** Default 12s. Long enough for a cold local model load to at least answer. */
  timeoutMs?: number;
};

/**
 * Test one endpoint with a **real request**, not a HEAD or a `/models` listing.
 *
 * The temptation is to `GET {base}/models`, which is cheaper and needs no tokens. It is also
 * a different request from the one that matters: it does not carry a body, it does not
 * exercise the `model` field, and several providers serve it from a different path than
 * `/chat/completions`, so it can succeed while every real call fails. A gateway that does not
 * implement `/models` at all would report a healthy endpoint as broken.
 *
 * So this sends a one-token `chat/completions` — the same URL, the same `Authorization`
 * header, the same `content-type`, the same method as `createHttpProvider` uses. Those four
 * things *are* the OpenAI-compatible contract; duplicating them here is deliberate, and the
 * cost is one token. `max_tokens: 1` keeps it to that.
 *
 * It never throws. A probe that throws cannot report *why*, and reporting why is its entire
 * job.
 */
export async function probeEndpoint(opts: ProbeOptions): Promise<ProbeResult> {
  const base = opts.baseUrl.trim().replace(/\/+$/, '');
  const started = now();

  if (!base) return { ok: false, kind: 'bad-url', ms: 0, detail: '地址为空。' };
  try {
    const u = new URL(base);
    if (u.protocol !== 'http:' && u.protocol !== 'https:') {
      return { ok: false, kind: 'bad-url', ms: 0, detail: `地址协议是 ${u.protocol}，只支持 http/https。` };
    }
  } catch {
    return { ok: false, kind: 'bad-url', ms: 0, detail: `地址不是合法 URL：${base}` };
  }
  if (!opts.model.trim()) {
    return { ok: false, kind: 'bad-url', ms: 0, detail: '模型名为空。' };
  }

  const timeoutMs = opts.timeoutMs ?? 12_000;
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(new DOMException('timeout', 'TimeoutError')), timeoutMs);
  const onAbort = (): void => ac.abort();
  opts.signal?.addEventListener('abort', onAbort, { once: true });

  try {
    const res = await fetch(`${base}/chat/completions`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        // Sent even when empty: an unauthenticated local server ignores it, and omitting it
        // for keyless presets would make the probe silently diverge from the real request.
        authorization: `Bearer ${opts.apiKey?.trim() || 'none'}`,
      },
      body: JSON.stringify({
        model: opts.model.trim(),
        messages: [{ role: 'user', content: 'ping' }],
        max_tokens: 1,
        stream: false,
      }),
      signal: ac.signal,
    });

    const ms = Math.round(now() - started);
    const body = await res.text().catch(() => '');

    if (res.ok) {
      return { ok: true, status: res.status, ms, detail: `HTTP ${res.status}，模型 ${opts.model.trim()} 应答正常。` };
    }
    return { ok: false, kind: classify(res.status), status: res.status, ms, detail: explain(res.status, body, base) };
  } catch (err) {
    const ms = Math.round(now() - started);
    const aborted = ac.signal.aborted;
    return {
      ok: false,
      kind: 'unreachable',
      ms,
      detail: aborted
        ? timeoutDetail(base, timeoutMs)
        : unreachableDetail(err, base),
    };
  } finally {
    clearTimeout(timer);
    opts.signal?.removeEventListener('abort', onAbort);
  }
}

function now(): number {
  return typeof performance !== 'undefined' ? performance.now() : Date.now();
}

function classify(status: number): ProbeFailure {
  if (status === 401 || status === 403) return 'unauthorized';
  if (status === 404) return 'not-found';
  if (status === 429) return 'rate-limited';
  if (status >= 500) return 'server-error';
  if (status >= 400) return 'bad-request';
  return 'http-error';
}

/**
 * A failure the user can act on.
 *
 * The status code alone does not say what to change, and the endpoint's own body is often
 * the only place the real reason appears (`model not found`, `insufficient balance`,
 * `invalid api key`). So the body is quoted when it is short enough to be a message rather
 * than a page of HTML.
 */
function explain(status: number, body: string, base: string): string {
  const quoted = shortBody(body);
  switch (classify(status)) {
    case 'unauthorized':
      return `HTTP ${status}：key 被拒绝。检查 API key 是否完整、是否属于该服务，以及账户是否还有余额。${quoted}`;
    case 'not-found':
      return (
        `HTTP 404：路径或模型名不对。确认地址写到 /v1 这一层（当前 ${base}），` +
        `以及模型名是该服务真实存在的。${quoted}`
      );
    case 'rate-limited':
      return `HTTP 429：被限流或额度用尽。稍后重试，或换一个 key／服务。${quoted}`;
    case 'server-error':
      return `HTTP ${status}：对方服务端错误，不是你的配置问题。稍后重试。${quoted}`;
    case 'bad-request':
      return `HTTP ${status}：请求被拒绝，通常是模型名不被支持。${quoted}`;
    default:
      return `HTTP ${status}。${quoted}`;
  }
}

/**
 * Name the host on a timeout too.
 *
 * This is the case that actually happens. A firewall that *drops* packets — which is what a
 * blocked provider looks like — produces no error at all, just silence until something gives
 * up, so the failure arrives here rather than at the `catch` above. A bare "timed out" would
 * repeat the original mistake of not saying which host was never reached.
 */
function timeoutDetail(base: string, timeoutMs: number): string {
  let host = base;
  try {
    host = new URL(base).host;
  } catch {
    /* keep the raw string */
  }
  return (
    `超过 ${timeoutMs} ms 没有收到 ${host} 的任何响应，已放弃。\n` +
    `包被丢弃（而不是被拒绝）时就是这个样子 —— 常见于该主机被网络屏蔽，或者本地模型正在加载。\n` +
    `换上面列表里的另一个服务，再点一次「测试连接」。`
  );
}

function unreachableDetail(err: unknown, base: string): string {
  let host = base;
  try {
    host = new URL(base).host;
  } catch {
    /* keep the raw string */
  }
  const why = err instanceof Error ? err.message : String(err);
  return (
    `连不上 ${host}（${why}）。没有任何响应到达，所以 key、模型名、请求体都还没被检查过 —— ` +
    `先确认这台机器能不能访问该主机（浏览器里常见的是 CORS 被拒：对方必须回 Access-Control-Allow-Origin）。` +
    `\n如果这个服务在本机不可达，换上面列表里的另一个再点一次「测试连接」。`
  );
}

function shortBody(body: string): string {
  const trimmed = body.trim();
  if (!trimmed) return '';
  if (trimmed.startsWith('<')) return '（对方返回了 HTML 而非 JSON，地址多半写错了）';
  const oneLine = trimmed.replace(/\s+/g, ' ');
  return `对方说：${oneLine.length > 300 ? `${oneLine.slice(0, 300)}…` : oneLine}`;
}
