/**
 * A DOM small enough to boot the workbench, strict enough to be worth trusting
 * ===========================================================================
 *
 * `apps/web/dist/main.js` is the one module in this repository that nothing has ever
 * *executed* in CI. Every other path has a real gate: the IR has 102 tests, the CLI
 * end-to-end suite spawns the real binary, and `python/tests/test_emitted_artifacts.py`
 * feeds the emitters' output to `glslangValidator` and `dxc`. The workbench had
 * "syntactically valid" and "every URL is 200" — which is why a broken page could survive
 * a fully green board, and why "为什么用不了" was a question that had to be answered by
 * guessing instead of by running something.
 *
 * So this file exists to run it. It is not a browser and does not pretend to be: there is
 * no layout, no CSS cascade, no GPU. What it does provide is enough of the DOM for
 * `boot()` to reach the end, which is exactly the span that a blank page lives in —
 * module evaluation, the `must()` contract with `index.html`, catalog loading, target
 * emitter registration, panel construction, and one turn of the render loop.
 *
 * ## Why unknown property access throws instead of returning `undefined`
 *
 * The tempting shim is a `Proxy` that answers `undefined` to everything, and it is worse
 * than useless: `el.style.setProperty(...)` becomes `TypeError: cannot read properties of
 * undefined`, which reads like an application bug, and `if (el.somethingUnimplemented)`
 * silently takes the wrong branch and the test still passes. A shim that cannot fail
 * cannot verify anything.
 *
 * Every element here is therefore wrapped in a proxy whose `get` **throws** for a property
 * the shim does not implement, naming the property and the tag. The failure mode becomes
 * "the shim is missing `setPointerCapture`", which is a two-minute fix, instead of a stack
 * trace inside application code that looks like a regression.
 *
 * The exceptions are `Symbol.*` and `then`, which must stay quiet: the first because the
 * language probes symbols on everything, the second because one `await` on an element
 * would otherwise deadlock against a throwing trap.
 *
 * ## The `index.html` contract is read, not restated
 *
 * `main.ts` resolves its mount points through `must(id)`, which throws on a missing id, and
 * `#preview` is resolved at *module scope* — so a rename in `index.html` blanks the page
 * before a single line of the app runs. The id list below is scraped from the real
 * `index.html` rather than copied, so this test fails when that contract is broken instead
 * of quietly testing a list that no longer matches the document.
 */

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

type Listener = (ev: unknown) => void;

const here = dirname(fileURLToPath(import.meta.url));

/** Properties the proxy must answer quietly, because the language itself probes them. */
const QUIET = new Set(['then', 'toJSON', 'constructor', 'prototype', 'nodeType', 'nodeName']);

export class FakeClassList {
  private readonly names = new Set<string>();

  add(...c: string[]): void {
    for (const n of c) if (n) this.names.add(n);
  }
  remove(...c: string[]): void {
    for (const n of c) this.names.delete(n);
  }
  contains(c: string): boolean {
    return this.names.has(c);
  }
  toggle(c: string, force?: boolean): boolean {
    const on = force === undefined ? !this.names.has(c) : force;
    if (on) this.names.add(c);
    else this.names.delete(c);
    return on;
  }
  get length(): number {
    return this.names.size;
  }
  item(i: number): string | null {
    return [...this.names][i] ?? null;
  }
  toString(): string {
    return [...this.names].join(' ');
  }
  /** Used by the `className` setter; replaces rather than merges, as the DOM does. */
  reset(className: string): void {
    this.names.clear();
    for (const n of className.split(/\s+/)) if (n) this.names.add(n);
  }
}

/* -------------------------------------------------------------------------- */
/* Selector matching                                                           */
/* -------------------------------------------------------------------------- */

/**
 * A compound selector: `tag`, `.class`, `#id` and `[attr="value"]`, in any combination.
 *
 * Deliberately not a CSS engine. These three forms plus the descendant combinator and the
 * comma group are the entire selector vocabulary `apps/web/src` uses — `grep` for
 * `querySelector` and you get four call sites, all of them `.class` or `tag[attr="v"]`.
 * Anything richer would be untested generality, and anything less would not run the code.
 */
function matches(el: FakeElement, compound: string): boolean {
  const parts = compound.match(/^[a-zA-Z0-9-]+|#[^.[\]]+|\.[^.[\]]+|\[[^\]]+\]/g);
  if (!parts) return false;
  for (const p of parts) {
    if (p.startsWith('#')) {
      if (el.id !== p.slice(1)) return false;
    } else if (p.startsWith('.')) {
      if (!el.classList.contains(p.slice(1))) return false;
    } else if (p.startsWith('[')) {
      const m = /^\[([\w-]+)(?:=["']?([^"'\]]*)["']?)?\]$/.exec(p);
      if (!m) return false;
      const value = el.getAttribute(m[1] as string);
      if (value === null) return false;
      if (m[2] !== undefined && value !== m[2]) return false;
    } else if (p !== '*' && el.tagName !== p.toUpperCase()) {
      return false;
    }
  }
  return true;
}

/** Descendant-combinator match, right to left: `a b` means "b inside a". */
function matchesChain(el: FakeElement, chain: string[]): boolean {
  const last = chain[chain.length - 1] as string;
  if (!matches(el, last)) return false;
  let node = el.parentElement;
  for (let i = chain.length - 2; i >= 0; i--) {
    const want = chain[i] as string;
    while (node && !matches(node, want)) node = node.parentElement;
    if (!node) return false;
    node = node.parentElement;
  }
  return true;
}

/* -------------------------------------------------------------------------- */
/* Elements                                                                    */
/* -------------------------------------------------------------------------- */

export class FakeElement {
  readonly tagName: string;
  readonly style = makeStyle();
  readonly dataset: Record<string, string> = {};
  readonly classList = new FakeClassList();
  readonly children: FakeElement[] = [];
  readonly attributes = new Map<string, string>();
  readonly listeners = new Map<string, Listener[]>();

  parentElement: FakeElement | null = null;
  id = '';
  hidden = false;
  disabled = false;
  value = '';
  checked = false;
  type = '';
  src = '';
  alt = '';
  title = '';
  placeholder = '';
  spellcheck = true;
  rows = 0;
  scrollTop = 0;
  scrollHeight = 0;
  clientWidth = 800;
  clientHeight = 600;
  offsetWidth = 800;
  offsetHeight = 600;
  width = 0;
  height = 0;
  /** `<input type="file">` — never populated, because nothing here can pick a file. */
  files: FakeElement[] | null = null;

  private text = '';

  constructor(tagName: string) {
    this.tagName = tagName.toUpperCase();
  }

  /* ---- text ---------------------------------------------------------------- */

  get textContent(): string {
    /*
     * Own text first, then the children's.
     *
     * The real DOM flattens text nodes and elements into one document-ordered list, and
     * modelling that faithfully would mean making text a node type. Every use in this app
     * writes its text before appending anything (`h('div', cls, text)` then `append(btn)`),
     * so leading-then-children is the faithful order here — and it is load-bearing: a
     * banner is `text + 关闭按钮`, and reading only the children would report a banner's
     * content as the word "关闭".
     */
    return this.text + this.children.map((c) => c.textContent).join('');
  }

  set textContent(v: string) {
    this.text = v ?? '';
    for (const c of this.children) c.parentElement = null;
    this.children.length = 0;
  }

  /**
   * `innerHTML` is stored, not parsed.
   *
   * The one setter in the codebase is the code view's syntax-highlighted block, and the
   * one reader is a `querySelector('.code-line[data-line="N"]')` used for scroll-to-line.
   * Parsing HTML to satisfy that would mean writing an HTML parser; leaving it empty is
   * honest and the query simply finds nothing, which is a code path the view already
   * handles (it is the "this target has no line for that node" branch).
   */
  innerHTML = '';

  /* ---- attributes ---------------------------------------------------------- */

  get className(): string {
    return this.classList.toString();
  }

  set className(v: string) {
    this.classList.reset(v ?? '');
  }

  setAttribute(name: string, value: string): void {
    this.attributes.set(name, String(value));
    if (name === 'id') this.id = String(value);
    if (name === 'class') this.classList.reset(String(value));
    if (name.startsWith('data-')) this.dataset[name.slice(5)] = String(value);
  }

  getAttribute(name: string): string | null {
    if (name === 'id') return this.id || null;
    if (name === 'class') return this.className || null;
    return this.attributes.get(name) ?? null;
  }

  removeAttribute(name: string): void {
    this.attributes.delete(name);
  }

  hasAttribute(name: string): boolean {
    return this.getAttribute(name) !== null;
  }

  styleSet(prop: string, value: string): void {
    this.style.set(prop, value);
  }

  /* ---- tree ---------------------------------------------------------------- */

  append(...nodes: unknown[]): void {
    for (const n of nodes) {
      if (typeof n === 'string') {
        this.text += n;
        continue;
      }
      const el = n as FakeElement;
      el.parentElement = this;
      this.children.push(el);
    }
  }

  appendChild(n: FakeElement): FakeElement {
    this.append(n);
    return n;
  }

  prepend(...nodes: FakeElement[]): void {
    for (const n of nodes) {
      n.parentElement = this;
      this.children.unshift(n);
    }
  }

  /**
   * Insert `nodes` immediately after this element.
   *
   * Used by the "打开 .shader.json" button, which parks its hidden `<input type=file>` next to
   * itself: a click can only be forwarded to a picker that is in the document, and the shim
   * gets the real semantics rather than a no-op so the placement is actually observable.
   */
  after(...nodes: FakeElement[]): void {
    const p = this.parentElement;
    if (!p) return;
    const at = p.children.indexOf(this);
    if (at < 0) return;
    for (const n of nodes) n.parentElement = p;
    p.children.splice(at + 1, 0, ...nodes);
  }

  replaceChildren(...nodes: FakeElement[]): void {
    for (const c of this.children) c.parentElement = null;
    this.children.length = 0;
    this.text = '';
    this.append(...nodes);
  }

  remove(): void {
    const p = this.parentElement;
    if (!p) return;
    const i = p.children.indexOf(this);
    if (i >= 0) p.children.splice(i, 1);
    this.parentElement = null;
  }

  /**
   * Swap this element for `nodes` in place.
   *
   * `setupRenderer` does this on every backend change — a canvas can only ever hand out
   * one kind of context, so switching means replacing the element rather than reusing it.
   */
  replaceWith(...nodes: FakeElement[]): void {
    const p = this.parentElement;
    if (!p) return;
    const i = p.children.indexOf(this);
    if (i < 0) return;
    for (const n of nodes) n.parentElement = p;
    p.children.splice(i, 1, ...nodes);
    this.parentElement = null;
  }

  get childElementCount(): number {
    return this.children.length;
  }

  get firstElementChild(): FakeElement | null {
    return this.children[0] ?? null;
  }

  get lastElementChild(): FakeElement | null {
    return this.children[this.children.length - 1] ?? null;
  }

  removeChild(n: FakeElement): FakeElement {
    n.remove();
    return n;
  }

  /** Depth-first walk of this subtree, self included. */
  *walk(): Generator<FakeElement> {
    yield this;
    for (const c of this.children) yield* c.walk();
  }

  querySelector<T = FakeElement>(selector: string): T | null {
    const found = this.querySelectorAll<T>(selector);
    return found.length ? (found[0] as T) : null;
  }

  querySelectorAll<T = FakeElement>(selector: string): T[] {
    const groups = selector.split(',').map((s) => s.trim().split(/\s+/));
    const out: FakeElement[] = [];
    for (const el of this.walk()) {
      if (el === this) continue;
      if (groups.some((chain) => matchesChain(el, chain)) && !out.includes(el)) out.push(el);
    }
    return out as T[];
  }

  getBoundingClientRect(): Record<string, number> {
    return { left: 0, top: 0, right: this.width, bottom: this.height, width: this.width, height: this.height, x: 0, y: 0 };
  }

  /* ---- events -------------------------------------------------------------- */

  addEventListener(type: string, fn: Listener): void {
    const list = this.listeners.get(type) ?? [];
    list.push(fn);
    this.listeners.set(type, list);
  }

  removeEventListener(type: string, fn: Listener): void {
    const list = this.listeners.get(type);
    if (list) this.listeners.set(type, list.filter((f) => f !== fn));
  }

  /** Real dispatch, so a test can actually click 生成 and watch the agent run. */
  dispatchEvent(ev: { type: string }): boolean {
    for (const fn of this.listeners.get(ev.type) ?? []) fn(ev);
    return true;
  }

  click(): void {
    if (this.disabled) return;
    this.dispatchEvent({ type: 'click' });
  }

  focus(): void {}
  blur(): void {}
  scrollIntoView(): void {}
  setPointerCapture(): void {}
  releasePointerCapture(): void {}

  /* ---- canvas -------------------------------------------------------------- */

  /**
   * No GPU, so no context.
   *
   * Returning `null` is not a cop-out: `WebGL2Renderer.create` and `createRenderer` are
   * both written to return `null` rather than throw when a context is unavailable, and
   * `setupRenderer` turns that into a visible banner. Exercising that branch is worth
   * more than a fake GL that would report a compile success nothing verified.
   */
  getContext(_kind: string): null {
    return null;
  }

  toDataURL(): string {
    return 'data:image/png;base64,';
  }
}

/* -------------------------------------------------------------------------- */
/* Installation                                                                */
/* -------------------------------------------------------------------------- */

/**
 * A CSSStyleDeclaration stand-in.
 *
 * A plain object rather than a `Map`, because every use in the app is property assignment
 * (`file.style.display = 'none'`, `progressBar.style.width = '50%'`) and a Map would
 * silently swallow all of them into a JS property named `width` that nothing can read.
 * Keeping them as ordinary properties means a later assertion can *check* the width.
 */
export type FakeStyle = Record<string, string> & {
  setProperty(prop: string, value: string): void;
  getPropertyValue(prop: string): string;
  removeProperty(prop: string): void;
};

function makeStyle(): FakeStyle {
  const style = {
    setProperty(prop: string, value: string): void {
      style[prop] = value;
    },
    getPropertyValue(prop: string): string {
      return style[prop] ?? '';
    },
    removeProperty(prop: string): void {
      delete style[prop];
    },
  } as FakeStyle;
  return style;
}

/**
 * A working `localStorage`, not a stub.
 *
 * The panel's whole point in using storage is that a value survives a reload, and a stub that
 * discards writes renders that untestable — the one behaviour a user is most likely to notice
 * when it breaks. A `Map` behind the same API costs four lines and makes "does it remember?"
 * an assertion instead of a hope.
 */
export type FakeStorage = {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
  clear(): void;
  /** Everything currently stored, for assertions. */
  snapshot(): Record<string, string>;
};

function makeStorage(): FakeStorage {
  const map = new Map<string, string>();
  return {
    getItem: (k) => map.get(k) ?? null,
    setItem: (k, v) => void map.set(k, String(v)),
    removeItem: (k) => void map.delete(k),
    clear: () => map.clear(),
    snapshot: () => Object.fromEntries(map),
  };
}

/** The single store every `installDom()` hands out — see the note at the installation site. */
let sharedStorage: FakeStorage | null = null;

type Shimmable = { createElement(tag: string): FakeElement; getElementById(id: string): FakeElement | null; body: FakeElement };

export type Shim = {
  document: Shimmable;
  /** Elements by id, so a test can reach a panel the app mounted. */
  byId(id: string): FakeElement | null;
  /** The `requestAnimationFrame` callbacks queued so far. */
  frames: Array<(t: number) => void>;
  /** Run one queued animation frame; returns false when none is pending. */
  step(time?: number): boolean;
  /** Everything `console.error`/`console.warn` saw, for asserting a clean boot. */
  consoleErrors: string[];
  /** Uncaught errors and rejections seen during the run. */
  failures: string[];
  /** The live `localStorage`, so a test can assert what was persisted — or clear it. */
  storage: FakeStorage;
};

const ID_RE = /\bid="([^"]+)"/g;

/** The mount points `main.ts` requires, read out of the real document. */
export function idsInIndexHtml(htmlPath = join(here, '..', 'index.html')): string[] {
  const html = readFileSync(htmlPath, 'utf8');
  return [...html.matchAll(ID_RE)].map((m) => m[1] as string);
}

/**
 * Wrap an element so unimplemented property access is a loud, attributable failure.
 *
 * See the file header for why this is a feature and not a shortcut.
 */
function strict(el: FakeElement): FakeElement {
  return new Proxy(el, {
    get(target, prop, recv) {
      if (typeof prop === 'symbol' || QUIET.has(prop)) return Reflect.get(target, prop, recv);
      if (prop in target) return Reflect.get(target, prop, recv);
      throw new Error(
        `dom-shim: <${target.tagName.toLowerCase()}${target.id ? `#${target.id}` : ''}> ` +
          `has no "${prop}" in this shim. Add it to apps/web/test/dom-shim.ts — the real ` +
          `browser has it, so a failure here means the shim is incomplete, not the app.`,
      );
    },
  });
}

export function installDom(): Shim {
  const frames: Array<(t: number) => void> = [];
  const consoleErrors: string[] = [];
  const failures: string[] = [];

  const registry = new Map<string, FakeElement>();
  const document = {
    body: strict(new FakeElement('body')),
    documentElement: strict(new FakeElement('html')),
    head: strict(new FakeElement('head')),
    createElement(tag: string): FakeElement {
      return strict(new FakeElement(tag));
    },
    createElementNS(_ns: string, tag: string): FakeElement {
      return strict(new FakeElement(tag));
    },
    createTextNode(text: string): FakeElement {
      const el = strict(new FakeElement('#text'));
      el.textContent = text;
      return el;
    },
    createDocumentFragment(): FakeElement {
      return strict(new FakeElement('#fragment'));
    },
    getElementById(id: string): FakeElement | null {
      // Ids minted by the app (`#score-box`, `#suggest-box`) are found by walking the tree;
      // the ones declared in `index.html` are pre-registered below.
      for (const el of document.body.walk()) if (el.id === id) return el;
      return registry.get(id) ?? null;
    },
    querySelector(sel: string): FakeElement | null {
      return document.body.querySelector(sel);
    },
    querySelectorAll(sel: string): FakeElement[] {
      return document.body.querySelectorAll(sel);
    },
    addEventListener(): void {},
    removeEventListener(): void {},
  };
  (document.body as FakeElement).parentElement = document.documentElement as unknown as FakeElement;

  /**
   * Mount points from `index.html`.
   *
   * `#preview` is the important one: `main.ts` resolves it at module scope, so the element
   * must exist before `dist/main.js` is even imported. It gets a real parent, because the
   * module reads `.parentElement` and falls back to `document.body`.
   */
  const declared = new Set(idsInIndexHtml());
  for (const id of declared) {
    const tag = id === 'preview' ? 'canvas' : id === 'topbar' || id === 'statusbar' ? 'footer' : 'div';
    const el = strict(new FakeElement(tag));
    el.id = id;
    registry.set(id, el);
    if (id === 'preview') {
      const stage = strict(new FakeElement('div'));
      stage.className = 'stage';
      stage.append(el);
      (document.body as FakeElement).append(stage);
    } else {
      (document.body as FakeElement).append(el);
    }
  }

  const g = globalThis as unknown as Record<string, unknown>;

  /**
   * Install a global.
   *
   * `Object.defineProperty` rather than assignment because a few of these — `navigator`,
   * `performance`, `crypto` — already exist on Node's `globalThis` as accessor properties,
   * and assigning to a getter-only global throws in strict mode (an ES module always is).
   * A raw assignment here would fail with a message about the *shim*, which is the least
   * useful thing it could say.
   */
  const define = (key: string, value: unknown): void => {
    try {
      Object.defineProperty(globalThis, key, { value, writable: true, configurable: true });
    } catch {
      g[key] = value;
    }
  };

  define('document', document);
  define('window', g);
  define('devicePixelRatio', 1);
  define('innerWidth', 1280);
  define('innerHeight', 800);
  define('self', g);
  define('addEventListener', (): void => {});
  define('removeEventListener', (): void => {});
  define('requestAnimationFrame', (fn: (t: number) => void): number => frames.push(fn));
  define('cancelAnimationFrame', (): void => {});
  define('matchMedia', () => ({
    matches: false,
    addEventListener() {},
    removeEventListener() {},
  }));
  define('getComputedStyle', () => makeStyle());
  define('ResizeObserver', class {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
  });
  define('IntersectionObserver', class {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
  });
  define('ImageData', class {
    constructor(
      readonly data: Uint8ClampedArray,
      readonly width: number,
      readonly height: number,
    ) {}
  });
  define('createImageBitmap', (): never => {
    throw new Error('dom-shim: createImageBitmap is not implemented (no image decoding here)');
  });
  define('Worker', class {
    postMessage(): void {}
    addEventListener(): void {}
    terminate(): void {}
  });

  /*
   * One store per process, like one store per origin.
   *
   * Each `installDom()` used to mint its own `localStorage`, so a second suite's install
   * silently replaced the global while the first suite kept asserting against its own object.
   * The tests then disagreed about what the panel had written — a fail-ure that looks like a
   * persistence bug and is actually the harness modelling two browsers in one process.
   * Reusing the instance is both the fix and the more faithful emulation; suites that need a
   * clean slate call `storage.clear()`, exactly as a test would in a real browser.
   */
  const storage = sharedStorage ?? makeStorage();
  sharedStorage = storage;
  define('localStorage', storage);

  const realError = console.error.bind(console);
  const realWarn = console.warn.bind(console);
  console.error = (...a: unknown[]): void => {
    consoleErrors.push(a.map(String).join(' '));
    realError(...a);
  };
  console.warn = (...a: unknown[]): void => {
    consoleErrors.push(a.map(String).join(' '));
    realWarn(...a);
  };

  process.on('unhandledRejection', (r) => failures.push(`unhandledRejection: ${String(r)}`));
  process.on('uncaughtException', (e) => failures.push(`uncaughtException: ${String(e)}`));

  return {
    document: document as unknown as Shimmable,
    byId: (id) => document.getElementById(id),
    frames,
    step(time = 16): boolean {
      const fn = frames.shift();
      if (!fn) return false;
      fn(time);
      return true;
    },
    consoleErrors,
    failures,
    storage,
  };
}
