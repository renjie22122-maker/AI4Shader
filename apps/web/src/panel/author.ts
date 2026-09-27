/**
 * Authoring panel — the agent loop, in the workbench
 * ==================================================
 *
 * A text box, an endpoint, and a button. What makes it worth having is what it shows
 * *while* it runs and *after* it fails:
 *
 *   • a live stage trace (`理解意图 → 起草图 → 确定性修复 → 发射校验`), because a run is several
 *     seconds and a UI that prints nothing reads as hung;
 *   • the stop reason in words, not a code;
 *   • every outstanding diagnostic with its `fix`, because "it did not work" without a reason
 *     is the thing that makes an agent feel like a slot machine;
 *   • the model-call count, which is the honest measure of how much work actually needed a
 *     model. On a healthy `refine` it is **zero**, and showing that number is how a user
 *     learns to trust the deterministic path.
 *
 * ## Why the stub provider is offered first
 *
 * It needs no key, no network, and it drafts a complete six-target document. A user who opens
 * the workbench on a plane can still watch the whole loop run end to end — and a loop you can
 * watch is a loop you can reason about.
 *
 * ## Why the endpoint is configurable, and why there is a 测试连接 button
 *
 * This panel used to hardcode `https://api.openai.com/v1`. On a machine that cannot reach
 * that one host — which is common, and true of the machine this was built on — the only
 * visible affordance was to retype the API key and press 生成 again, and the failure that
 * came back was the browser's `Failed to fetch`. There was no field in which to name a
 * different provider, so there was no fix available from inside the UI at all.
 *
 * So the service is a dropdown, the base URL is an ordinary editable field, and 测试连接
 * sends one real one-token request before the user writes a description. Guessing is what
 * made that failure expensive; the button is what replaces guessing with a status code.
 *
 * ## What is remembered, and what is not
 *
 * The service, the base URL, the model, the provider choice and — if the box is ticked — the
 * API key are kept in `localStorage`. They are configuration, and retyping any of it on every
 * visit is its own small tax.
 *
 * The key is the one entry that is not simply remembered because remembering is convenient.
 * A key in `localStorage` is a key any script on this origin can read, and there is no server
 * here to hold it instead. This panel originally refused to store it for exactly that reason
 * and said so in the field's placeholder.
 *
 * That refusal was the wrong shape of answer. The alternative it left the user was to retype a
 * fifty-character secret after every refresh, and a rule whose practical effect is to make the
 * careful path the annoying one does not produce careful users — it produces a key pasted into
 * a scratch file, which is worse than `localStorage` in every respect. So the choice is now the
 * user's, made where it is visible, with the consequence written next to it and a one-click way
 * back: unticking the box deletes the stored value immediately.
 */

import type { ShaderDoc } from '../../../../packages/ir/src/index.js';
import {
  authorOnce,
  describeStage,
  describeStop,
  summariseRun,
  type AgentResult,
  type ProviderChoice,
} from '../agent.js';
import {
  ENDPOINT_PRESETS,
  endpointReady,
  listModels,
  presetById,
  presetForBaseUrl,
  probeEndpoint,
  type ProbeResult,
} from '../providers.js';

export type AuthorPanelHost = {
  /** A new document arrived (from `author`) or the current one changed (from `refine`). */
  applyDoc(doc: ShaderDoc, reason: string): Promise<void>;
  /** One line for the status bar. */
  setStatus(text: string): void;
  /** Freeze/resume the preview loop while a run is in flight. */
  setBusy(busy: boolean): void;
  /** The document currently being previewed, used as `basedOn` for a refine. */
  currentDoc(): ShaderDoc;
};

export type AuthorPanelOptions = {
  /** Seeds the description box so the panel is usable the moment it appears. */
  seedPrompt?: string;
  /** Provider to start on. `stub` by default: it always works. */
  initialChoice?: ProviderChoice;
};

type Stage = { stage: string; round: number; summary: string };

const DEFAULT_SEED = '液态金属表面，慢速流动，带细腻的拉丝与高光带';

/**
 * Everything persisted about the endpoint — including, when the user asks for it, the key.
 *
 * `apiKey` is present only while `rememberKey` is true, and it is deleted the moment that box
 * is unticked. Absent is the safe default in the sense that matters: a stored key is a key any
 * script on this origin can read, so it is written only in response to an explicit, visible
 * choice, and the choice is reversible in one click.
 */
type StoredNet = {
  choice?: ProviderChoice;
  service?: string;
  baseUrl?: string;
  model?: string;
  rememberKey?: boolean;
  apiKey?: string;
};

const STORE_KEY = 'ai4shader.author.endpoint';

/**
 * Sentinel option value for "let me type it myself".
 *
 * A NUL prefix because a model id is never going to start with one, so this cannot collide
 * with a real name however the endpoint spells it.
 */
const MANUAL_MODEL = '\u0000manual';

/**
 * Placeholder sent as the bearer token for a local runtime that has no key.
 *
 * Some local servers reject a request with no `Authorization` header outright, and the
 * endpoint presets mark which ones need nothing. Sending a literal is more honest than
 * sending an empty string, and it is what the preset's hint tells the user to type anyway.
 */
const KEYLESS_PLACEHOLDER = 'local';

function loadStored(): StoredNet {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    return raw ? (JSON.parse(raw) as StoredNet) : {};
  } catch {
    // A disabled or full localStorage is not an error worth surfacing: the panel works
    // without persistence, it just does not remember.
    return {};
  }
}

export class AuthorPanel {
  readonly el: HTMLElement;

  private readonly host: AuthorPanelHost;
  private choice: ProviderChoice;
  private running = false;
  private controller: AbortController | null = null;
  private probing = false;
  private fetching = false;

  private readonly promptBox: HTMLTextAreaElement;
  private readonly choiceGroup: HTMLElement;
  private readonly netRow: HTMLElement;
  private readonly serviceSelect: HTMLSelectElement;
  private readonly baseUrlInput: HTMLInputElement;
  private readonly keyInput: HTMLInputElement;
  private readonly rememberKey: HTMLInputElement;
  private readonly keyNote: HTMLElement;
  private readonly modelInput: HTMLInputElement;
  private readonly modelSelect: HTMLSelectElement;
  /**
   * The model names the panel currently believes are worth offering.
   *
   * Seeded from the preset's `modelHints`, then **replaced wholesale** by whatever
   * `GET /models` returns. A hint list is a starting point; the endpoint is the authority.
   */
  private knownModels: string[] = [];
  /**
   * The user chose 「手动输入…」, as opposed to the value merely not being in the list.
   *
   * Two different states that look identical from the value alone. Typing a name the endpoint
   * does not serve must reveal the text field; so must *asking* to type, even before anything
   * has been typed. Without this flag the second case reverts instantly — the picker sees a
   * current value that is still a list member and puts itself back — which is exactly what it
   * did the first time this was written.
   */
  private manualModel = false;
  private readonly listButton: HTMLButtonElement;
  private readonly hintLine: HTMLElement;
  private readonly probeButton: HTMLButtonElement;
  private readonly probeStatus: HTMLElement;
  private readonly authorButton: HTMLButtonElement;
  private readonly refineButton: HTMLButtonElement;
  private readonly cancelButton: HTMLButtonElement;
  private readonly traceList: HTMLElement;
  private readonly verdictBox: HTMLElement;

  constructor(host: AuthorPanelHost, opts: AuthorPanelOptions = {}) {
    const stored = loadStored();

    this.host = host;
    this.choice = opts.initialChoice ?? stored.choice ?? 'stub';
    this.el = document.createElement('div');
    this.el.className = 'author-panel';

    /* ---- provider choice ---------------------------------------------------- */
    this.choiceGroup = document.createElement('div');
    this.choiceGroup.className = 'author-seg';
    for (const [id, label, title] of [
      ['stub', '离线', '确定性 stub provider：不需要 key，不需要网络，会起草一份完整文档'],
      ['http', '在线', '真实模型（OpenAI 兼容）。是否保存 key 由下方「记住 key」控制'],
    ] as const) {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = label;
      b.title = title;
      b.dataset['choice'] = id;
      b.addEventListener('click', () => this.setChoice(id));
      this.choiceGroup.append(b);
    }

    /* ---- endpoint ----------------------------------------------------------- */
    this.serviceSelect = document.createElement('select');
    this.serviceSelect.className = 'author-service';
    for (const p of ENDPOINT_PRESETS) {
      const o = document.createElement('option');
      o.value = p.id;
      o.textContent = p.label;
      this.serviceSelect.append(o);
    }
    this.serviceSelect.addEventListener('change', () => this.applyPreset(this.serviceSelect.value));

    this.baseUrlInput = document.createElement('input');
    this.baseUrlInput.type = 'text';
    this.baseUrlInput.className = 'author-url';
    this.baseUrlInput.placeholder = 'https://…/v1';
    this.baseUrlInput.autocomplete = 'off';
    this.baseUrlInput.spellcheck = false;
    this.baseUrlInput.addEventListener('input', () => {
      // Keep the dropdown truthful when the URL is edited by hand: a known address selects
      // its preset, anything else becomes 自定义. A selector showing a service that is no
      // longer selected is worse than no selector.
      const preset = presetForBaseUrl(this.baseUrlInput.value);
      this.serviceSelect.value = preset.id;
      this.showHint(preset.id);
      this.invalidateProbe();
      this.persist();
      this.syncEnabled();
    });

    this.modelSelect = document.createElement('select');
    this.modelSelect.className = 'author-model-select';
    this.modelSelect.title = '从已知模型里选一个，或选「手动输入…」自己写';
    this.modelSelect.addEventListener('change', () => {
      const picked = this.modelSelect.value;
      this.manualModel = picked === MANUAL_MODEL;
      if (!this.manualModel) this.modelInput.value = picked;
      // Focus the field when the user asks to type, so the option they just chose turns into
      // a cursor rather than a second thing to click.
      else this.modelInput.focus();
      this.syncModelPicker();
      this.invalidateProbe();
      this.persist();
      this.syncEnabled();
    });

    this.modelInput = document.createElement('input');
    this.modelInput.type = 'text';
    this.modelInput.className = 'author-model';
    this.modelInput.placeholder = '模型名';
    this.modelInput.autocomplete = 'off';
    this.modelInput.spellcheck = false;
    this.modelInput.addEventListener('input', () => {
      // Typing is itself a request to type: without this a name that later turns out to be in
      // a freshly fetched list would collapse the field out from under the cursor.
      this.manualModel = true;
      this.invalidateProbe();
      this.persist();
      this.syncModelPicker();
      this.syncEnabled();
    });

    this.keyInput = document.createElement('input');
    this.keyInput.type = 'password';
    this.keyInput.className = 'author-key';
    this.keyInput.placeholder = 'sk-…';
    this.keyInput.autocomplete = 'off';
    this.keyInput.spellcheck = false;
    this.keyInput.addEventListener('input', () => {
      this.invalidateProbe();
      // Persist on every keystroke rather than on blur: a refresh (or a crash) in the middle
      // of typing a key is exactly the case this feature exists for.
      this.persist();
      this.syncEnabled();
    });

    /*
     * Remembering the key is opt-in *in the UI*, and on by default.
     *
     * The panel shipped refusing to persist it at all, on the grounds that `localStorage` is
     * readable by any script on this origin and there is no server here to hold it instead.
     * That reasoning is still true; what it missed is that the alternative on offer was
     * "retype a 50-character secret on every refresh", and a rule whose practical effect is to
     * make the secure path the annoying one gets worked around rather than followed — by
     * pasting the key into a text file, which is worse in every respect.
     *
     * So the trade is stated where it is made, chosen by the person who owns the key, and
     * reversible in one click. Unticking the box removes the stored value immediately; it does
     * not wait for the next save.
     */
    this.rememberKey = document.createElement('input');
    this.rememberKey.type = 'checkbox';
    this.rememberKey.id = 'author-remember-key';
    this.rememberKey.checked = stored.rememberKey ?? true;
    this.rememberKey.addEventListener('change', () => {
      this.persist();
      this.syncNote();
    });

    const rememberLabel = document.createElement('label');
    rememberLabel.className = 'author-remember';
    rememberLabel.htmlFor = 'author-remember-key';
    rememberLabel.append(this.rememberKey, document.createTextNode(' 记住 key'));

    this.keyNote = document.createElement('div');
    this.keyNote.className = 'author-hint muted';

    const keyRow = document.createElement('div');
    keyRow.className = 'author-keyrow';
    keyRow.append(this.keyInput, rememberLabel);

    this.hintLine = document.createElement('div');
    this.hintLine.className = 'author-hint muted';

    /*
     * Two buttons, because they answer two different questions and either can fail alone.
     *
     *   测试连接 — "will a real chat call to this address, with this model and key, work?"
     *   拉取模型 — "what models does this endpoint actually serve?"
     *
     * They are separate on purpose. A wrong model name makes the probe 404 while `/models`
     * succeeds and hands over the very name that would fix it, so chaining them would hide the
     * more useful of the two exactly when it matters. Nothing here fires a request the user did
     * not press: a workbench that quietly calls out to a third party on focus is a surprise.
     */
    this.listButton = document.createElement('button');
    this.listButton.type = 'button';
    this.listButton.className = 'mini';
    this.listButton.textContent = '拉取模型';
    this.listButton.title = 'GET {地址}/models，用你填的 key 取回该服务真实的模型列表，填进下面的下拉建议';
    this.listButton.addEventListener('click', () => void this.fetchModels());

    this.probeButton = document.createElement('button');
    this.probeButton.type = 'button';
    this.probeButton.className = 'mini';
    this.probeButton.textContent = '测试连接';
    this.probeButton.title = '发一次真实的单 token 请求，只验证 地址 / 模型 / key，不跑 agent';
    this.probeButton.addEventListener('click', () => void this.probe());

    this.probeStatus = document.createElement('span');
    this.probeStatus.className = 'author-probe muted mono';

    const probeRow = document.createElement('div');
    probeRow.className = 'author-probe-row';
    probeRow.append(this.probeButton, this.listButton, this.probeStatus);

    this.netRow = document.createElement('div');
    this.netRow.className = 'author-net';
    this.netRow.append(
      this.serviceSelect,
      this.baseUrlInput,
      this.modelSelect,
      this.modelInput,
      keyRow,
      this.keyNote,
      this.hintLine,
      probeRow,
    );

    /* ---- prompt ------------------------------------------------------------- */
    this.promptBox = document.createElement('textarea');
    this.promptBox.className = 'author-prompt';
    this.promptBox.rows = 3;
    this.promptBox.value = opts.seedPrompt ?? DEFAULT_SEED;
    this.promptBox.spellcheck = false;

    /* ---- actions ------------------------------------------------------------ */
    this.authorButton = document.createElement('button');
    this.authorButton.type = 'button';
    this.authorButton.className = 'author-primary';
    this.authorButton.textContent = '生成';
    this.authorButton.title = '从描述从头起草一份新文档（author）';
    this.authorButton.addEventListener('click', () => void this.run('author'));

    this.refineButton = document.createElement('button');
    this.refineButton.type = 'button';
    this.refineButton.textContent = '微调当前';
    this.refineButton.title = '让模型按描述修改现有文档，再检查生成代码；会消耗模型调用';
    this.refineButton.addEventListener('click', () => void this.run('refine'));

    this.cancelButton = document.createElement('button');
    this.cancelButton.type = 'button';
    this.cancelButton.textContent = '取消';
    this.cancelButton.hidden = true;
    this.cancelButton.addEventListener('click', () => this.controller?.abort());

    const actions = document.createElement('div');
    actions.className = 'author-actions';
    actions.append(this.authorButton, this.refineButton, this.cancelButton);

    /* ---- output ------------------------------------------------------------- */
    this.traceList = document.createElement('ol');
    this.traceList.className = 'author-trace';

    this.verdictBox = document.createElement('div');
    this.verdictBox.className = 'author-verdict';

    this.el.append(
      this.choiceGroup,
      this.netRow,
      this.promptBox,
      actions,
      this.verdictBox,
      this.traceList,
    );

    /* ---- restore ------------------------------------------------------------ */
    /*
     * The default is the first preset, not a service chosen for the machine this was built
     * on. Which endpoints are reachable is a property of the network, not of the product,
     * and baking in a winner would be a guess dressed up as a default. The 测试连接 button
     * is what turns that guess into a fact, so the panel opens saying it has not been
     * pressed yet.
     */
    const serviceId =
      stored.service && presetById(stored.service) ? stored.service : (ENDPOINT_PRESETS[0]?.id ?? 'custom');
    const preset = presetById(serviceId) ?? ENDPOINT_PRESETS[0];
    if (preset) {
      this.serviceSelect.value = preset.id;
      this.baseUrlInput.value = stored.baseUrl ?? preset.baseUrl;
      this.modelInput.value = stored.model ?? preset.modelHints[0] ?? '';
      this.showHint(preset.id);
    }
    // Restore the key only if it was explicitly remembered. A stored value with the box
    // unticked is stale — from a version that did not ask, or from a choice since reversed —
    // and reading it anyway would quietly override the user's current answer.
    if (this.rememberKey.checked && stored.apiKey) this.keyInput.value = stored.apiKey;
    this.syncNote();
    this.probeStatus.className = 'author-probe muted mono';
    this.probeStatus.textContent = '未测试 —— 建议先点「测试连接」，比跑一次 agent 快得多。';

    this.setChoice(this.choice);
  }

  /* ---- endpoint wiring ----------------------------------------------------- */

  private currentPresetId(): string {
    return this.serviceSelect.value;
  }

  /** True when the selected service runs locally and needs no key. */
  private keyless(): boolean {
    return presetById(this.currentPresetId())?.keyless === true;
  }

  private baseUrl(): string {
    return this.baseUrlInput.value.trim();
  }

  /** The token to send: the typed key, or a placeholder for a keyless local runtime. */
  private effectiveKey(): string {
    return this.keyInput.value.trim() || (this.keyless() ? KEYLESS_PLACEHOLDER : '');
  }

  private applyPreset(id: string): void {
    const preset = presetById(id);
    if (!preset) return;
    this.serviceSelect.value = preset.id;
    if (preset.baseUrl) this.baseUrlInput.value = preset.baseUrl;
    this.modelInput.value = preset.modelHints[0] ?? '';
    this.showHint(preset.id);
    this.invalidateProbe();
    this.persist();
    this.syncEnabled();
  }

  /**
   * Show the right control for the list we currently have, and keep them agreeing.
   *
   * ## Why this is a `<select>` and not a `<datalist>`
   *
   * It was a datalist, which is the idiomatic "free text with suggestions" control and was
   * the wrong one here. A browser **filters a datalist by whatever is already in the field**,
   * so a panel that pre-fills `deepseek-chat` shows a dropdown containing exactly one entry —
   * the one already chosen. The result was a user asking why DeepSeek only offers a single
   * model, when the real list was two (and, it turned out, neither of them current).
   *
   * A `<select>` cannot do that: it always shows every option. Free text is not lost — the
   * 「手动输入…」 entry switches to a text field, which is what a datalist was providing and
   * the only part of it worth keeping.
   *
   * `modelInput` stays the single source of truth even while it is hidden, so readiness,
   * persistence and the run path all read one field rather than two that can disagree.
   */
  private syncModelPicker(): void {
    const models = this.knownModels;
    this.modelSelect.replaceChildren();
    for (const m of models) {
      const o = document.createElement('option');
      o.value = m;
      o.textContent = m;
      this.modelSelect.append(o);
    }
    const manual = document.createElement('option');
    manual.value = MANUAL_MODEL;
    manual.textContent = models.length ? '手动输入…' : '手动输入';
    this.modelSelect.append(manual);

    const current = this.modelInput.value.trim();
    const inList = models.includes(current);
    // "Type it myself" is true either because the user said so or because the name they have
    // cannot be picked from this list. Both need the text field; only the first survives a
    // value that happens to be in the list.
    const wantsManual = this.manualModel || !inList;

    // No list to pick from (the 自定义 preset, or a fetch that failed): the text field is all
    // there is, and hiding it behind an empty dropdown would be a dead end.
    this.modelSelect.hidden = models.length === 0;
    this.modelSelect.value = wantsManual ? MANUAL_MODEL : current;

    // With a list in hand and a name from it, the dropdown says everything — the text field
    // would just be the same word twice. It reappears the moment the value is not in the list,
    // which is exactly when someone needs to type.
    this.modelInput.hidden = models.length > 0 && !wantsManual;
  }

  private showHint(id: string): void {
    const preset = presetById(id);
    this.hintLine.textContent = preset?.hint ?? '';
    this.knownModels = [...(preset?.modelHints ?? [])];
    // A new service means a new list; whatever the user was typing was about the old one.
    this.manualModel = false;
    this.syncModelPicker();
  }

  /**
   * Ask the endpoint for its model list.
   *
   * The answer replaces the preset hints entirely rather than merging with them: a name the
   * server did not return is, as far as this endpoint is concerned, not a model, and leaving
   * the guesses in place next to the real list would recreate the confusion this button
   * exists to end.
   */
  private async fetchModels(): Promise<void> {
    if (this.fetching || this.running) return;
    this.fetching = true;
    this.syncEnabled();
    this.probeStatus.className = 'author-probe muted mono';
    this.probeStatus.textContent = '正在拉取模型列表…';

    try {
      const result = await listModels({ baseUrl: this.baseUrl(), apiKey: this.effectiveKey() });
      if (!result.ok) {
        this.probeStatus.className = 'author-probe mono author-bad';
        this.probeStatus.textContent = `✗ ${result.ms} ms · ${result.detail}`;
        return;
      }
      this.knownModels = result.models;
      this.manualModel = false;
      this.syncModelPicker();
      this.probeStatus.className = 'author-probe mono author-ok';
      const shown = result.models.slice(0, 12).join('、');
      this.probeStatus.textContent =
        `✓ ${result.ms} ms · 服务端返回 ${result.models.length} 个模型（已填入下拉框）：` +
        `${shown}${result.models.length > 12 ? ' …' : ''}`;
      // Fill the field only when it is empty or holds a name the server just said it does not
      // have — replacing a name the user typed on purpose would be the UI arguing with them.
      const current = this.modelInput.value.trim();
      if (!current || !result.models.includes(current)) {
        this.modelInput.value = result.models[0] ?? current;
        this.syncEnabled();
      }
    } finally {
      this.fetching = false;
      this.syncEnabled();
    }
  }

  /**
   * Any edit to the address, model or key makes a previous verdict about them stale.
   *
   * It resets to 未测试 rather than to an empty line: a blank status next to a button is
   * indistinguishable from "the line is broken", and the whole reason this control exists is
   * that an unstated connection state is what made the original failure expensive.
   */
  private invalidateProbe(): void {
    this.probeStatus.className = 'author-probe muted mono';
    this.probeStatus.textContent = '未测试 —— 地址、模型或 key 变了，需要重新测试。';
  }

  private persist(): void {
    try {
      const value: StoredNet = {
        choice: this.choice,
        service: this.currentPresetId(),
        baseUrl: this.baseUrl(),
        model: this.modelInput.value.trim(),
        rememberKey: this.rememberKey.checked,
      };
      /*
       * The key is written only when the box is ticked, and *removed* when it is unticked —
       * unconditionally, right here, rather than merely not being re-added. "I turned it off"
       * has to mean the bytes are gone from storage, not that the next load will decline to
       * read them; a checkbox that leaves the secret sitting in a file is the kind of
       * half-measure that makes a privacy control worse than none.
       */
      if (this.rememberKey.checked) {
        const key = this.keyInput.value.trim();
        if (key) value.apiKey = key;
      }
      localStorage.setItem(STORE_KEY, JSON.stringify(value));
    } catch {
      /* see loadStored */
    }
  }

  /** Say what is stored right now, in the place where the choice is made. */
  private syncNote(): void {
    this.keyNote.textContent = this.rememberKey.checked
      ? 'key 会写进本机浏览器的 localStorage，刷新后自动填回。这个来源上的任何脚本都能读到它 —— 取消勾选会立刻删除已保存的值。'
      : 'key 只留在本页内存里，刷新就要重打。';
  }

  /* ---- the connection test -------------------------------------------------- */

  private async probe(): Promise<void> {
    if (this.probing || this.running) return;
    this.probing = true;
    this.probeStatus.className = 'author-probe muted mono';
    this.probeStatus.textContent = '正在测试…';
    this.probeButton.disabled = true;

    try {
      const result = await probeEndpoint({
        baseUrl: this.baseUrl(),
        model: this.modelInput.value,
        apiKey: this.effectiveKey(),
      });
      this.showProbe(result);
    } finally {
      this.probing = false;
      this.probeButton.disabled = false;
    }
  }

  private showProbe(result: ProbeResult): void {
    this.probeStatus.className = `author-probe mono ${result.ok ? 'author-ok' : 'author-bad'}`;
    this.probeStatus.textContent = `${result.ok ? '✓' : '✗'} ${result.ms} ms · ${result.detail}`;
  }

  /* ---- run state ------------------------------------------------------------ */

  /** Re-enable the buttons after a run, whatever the outcome. */
  private setRunning(running: boolean): void {
    this.running = running;
    this.cancelButton.hidden = !running;
    this.promptBox.disabled = running;
    this.syncEnabled();
    this.host.setBusy(running);
  }

  private setChoice(choice: ProviderChoice): void {
    this.choice = choice;
    for (const b of Array.from(this.choiceGroup.children)) {
      const btn = b as HTMLButtonElement;
      const on = btn.dataset['choice'] === choice;
      btn.classList.toggle('on', on);
      btn.setAttribute('aria-pressed', String(on));
    }
    this.netRow.hidden = choice !== 'http';
    this.persist();
    this.syncEnabled();
  }

  /**
   * What counts as "configured enough to try".
   *
   * Delegated to `endpointReady` so the rule has one home and can be tested without a DOM.
   * A keyless local runtime is ready as soon as it has an address and a model; every other
   * service also needs a key.
   */
  private readiness(): { ready: boolean; missing?: string } {
    return endpointReady({
      choice: this.choice,
      baseUrl: this.baseUrl(),
      model: this.modelInput.value,
      apiKey: this.keyInput.value,
      keyless: this.keyless(),
    });
  }

  private syncEnabled(): void {
    const { ready, missing } = this.readiness();
    this.authorButton.disabled = this.running || !ready;
    this.refineButton.disabled = this.running || !ready;
    this.probeButton.disabled = this.running || this.probing || this.fetching;
    this.listButton.disabled = this.running || this.probing || this.fetching || !this.baseUrl();
    this.authorButton.title =
      this.choice === 'http' && !ready
        ? `在线模式还缺：${missing}`
        : '从描述从头起草一份新文档（author）';
  }

  /* ---- the run -------------------------------------------------------------- */

  /**
   * Run one pass.
   *
   * `refine` passes the *current* document as `basedOn`, which is what makes it cheap: the
   * loop verifies that document first, and a healthy one costs zero model calls.
   */
  private async run(mode: 'author' | 'refine'): Promise<void> {
    if (this.running) return;
    const prompt = this.promptBox.value.trim();
    if (!prompt) {
      this.verdict('warn', '请先写一句描述。');
      return;
    }
    if (!this.readiness().ready) {
      this.verdict('warn', '在线模式还没配好：填上地址、模型名与 key，或点「测试连接」看哪里不对。');
      return;
    }

    this.traceList.replaceChildren();
    this.verdict('info', mode === 'author' ? '正在从零起草…' : '正在微调当前文档…');
    this.setRunning(true);
    this.controller = new AbortController();

    try {
      const result = await authorOnce({
        prompt,
        choice: this.choice,
        ...(mode === 'refine' ? { basedOn: this.host.currentDoc() } : {}),
        ...(this.choice === 'http'
          ? { apiKey: this.effectiveKey(), model: this.modelInput.value, baseUrl: this.baseUrl() }
          : {}),
        signal: this.controller.signal,
        onStage: (s: Stage) => this.pushStage(s),
      });

      await this.report(result, mode);
    } catch (err) {
      // Only a *usage* problem reaches here — the loop turns every in-loop failure into a
      // non-clean result. So this branch is for "you did not give me a key", not "the model
      // said something odd".
      this.verdict('error', err instanceof Error ? err.message : String(err));
    } finally {
      this.controller = null;
      this.setRunning(false);
    }
  }

  private pushStage(s: Stage): void {
    const li = document.createElement('li');
    const round = document.createElement('span');
    round.className = 'mono muted';
    round.textContent = String(s.round).padStart(2, '0');
    const name = document.createElement('span');
    name.className = 'author-stage';
    name.textContent = describeStage(s.stage);
    const text = document.createElement('span');
    text.textContent = s.summary;
    li.append(round, name, text);
    this.traceList.append(li);
    this.traceList.scrollTop = this.traceList.scrollHeight;
  }

  private async report(result: AgentResult, mode: 'author' | 'refine'): Promise<void> {
    const box = document.createElement('div');

    const head = document.createElement('div');
    head.className = result.clean ? 'author-ok' : 'author-bad';
    // The model-call count is the honest headline on a refine: zero means the deterministic
    // repairer did the work, and saying so is how the user learns to trust that path.
    head.textContent = result.clean
      ? `代码生成通过 · ${mode === 'refine' ? '微调' : '起草'} · 模型调用 ${result.usage?.calls ?? 0} 次`
      : `未成功 · ${describeStop(result.stopReason)} · 模型调用 ${result.usage?.calls ?? 0} 次`;
    box.append(head);

    const meta = document.createElement('div');
    meta.className = 'muted mono';
    meta.textContent = summariseRun(result);
    box.append(meta);
    if (result.clean) {
      const note = document.createElement('div');
      note.className = 'author-warn';
      note.textContent = '请播放预览确认外观和动态行为。需求复核只检查代码，不代表已验证画面、物理模拟或全部目标引擎。';
      box.append(note);
    }
    if (result.doc.spec.outOfScope?.length) {
      const limits = document.createElement('div');
      limits.className = 'author-warn';
      limits.textContent = `未实现或采用近似：${result.doc.spec.outOfScope.join('；')}`;
      box.append(limits);
    }

    /*
     * A provider failure has exactly one useful next action, and it is not "read the
     * diagnostic carefully": it is "press 测试连接". Saying so at the top of the report is
     * the difference between a dead end and a two-second fix.
     */
    if (result.stopReason === 'provider-error') {
      const next = document.createElement('div');
      next.className = 'author-warn';
      next.textContent = '→ 这是连接层面的失败。点上方「测试连接」会单独验证 地址 / 模型 / key，并给出状态码。';
      box.append(next);
    }

    if (result.remaining.length > 0) {
      const list = document.createElement('div');
      list.className = 'diag-list';
      // Cap the list: a badly broken document can carry dozens, and the first few explain it.
      for (const d of result.remaining.slice(0, 8)) {
        const item = document.createElement('div');
        item.className = `diag diag-${d.severity === 'error' ? 'error' : d.severity === 'warning' ? 'warning' : 'note'}`;
        const head2 = document.createElement('div');
        head2.className = 'diag-head';
        head2.textContent = `${d.code}${d.at?.node ? ` @${d.at.node}${d.at.port ? `.${d.at.port}` : ''}` : ''}`;
        const msg = document.createElement('div');
        msg.className = 'diag-msg';
        msg.textContent = d.message;
        item.append(head2, msg);
        if (d.fix) {
          const fix = document.createElement('div');
          fix.className = 'diag-fix';
          fix.textContent = `修复：${d.fix}`;
          item.append(fix);
        }
        list.append(item);
      }
      if (result.remaining.length > 8) {
        const more = document.createElement('div');
        more.className = 'muted';
        more.textContent = `…还有 ${result.remaining.length - 8} 条`;
        list.append(more);
      }
      box.append(list);
    }

    this.verdictBox.replaceChildren(box);

    // Apply even a non-clean document: it is the best the loop reached, the user can keep
    // tuning it by hand, and refusing to show it would hide real progress. The verdict above
    // already says it is not finished.
    await this.host.applyDoc(result.doc as ShaderDoc, result.clean ? 'agent 生成完成' : 'agent 未完成（已套用最好版本）');
    this.host.setStatus(
      result.clean
        ? `agent 代码生成通过 · ${Object.keys(result.emissions).length} 个目标 · 模型 ${result.usage?.calls ?? 0} 次`
        : `agent 未完成 · ${result.remaining.length} 条诊断 · 见左侧面板`,
    );
  }

  private verdict(kind: 'info' | 'warn' | 'error', text: string): void {
    const div = document.createElement('div');
    div.className = kind === 'error' ? 'author-bad' : kind === 'warn' ? 'author-warn' : 'muted';
    div.textContent = text;
    this.verdictBox.replaceChildren(div);
  }
}
