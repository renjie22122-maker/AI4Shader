/**
 * Generated-code view
 * ===================
 *
 * Six targets, one document. Emission is **lazy and cached**:
 *
 *   • a target is emitted the first time its tab is opened, never before — opening the
 *     workbench costs one `emit()` call, not six;
 *   • results are cached under `(target, docHash)`, so flipping between tabs, or
 *     dragging a slider (which does not change `docHash`), costs nothing;
 *   • a document edit produces a new `docHash`, so the cache misses exactly once per
 *     real change.
 *
 * The highlighting is hand-written and dependency-free. **Every token is HTML-escaped
 * before it is concatenated into markup** — shader code is full of `a < b`, `<<`, and
 * `->`, and one unescaped `<` silently swallows the rest of the document.
 */

import { emit } from '../ir.js';
// Types resolve against the authored sources; the projection carries none. Erased at runtime.
import type {
  Diagnostic,
  Emission,
  NodeKind,
  NodeSchema,
  ShaderDoc,
  TargetFormat,
} from '../../../../packages/ir/src/index.js';

export type TargetTab = {
  target: TargetFormat;
  label: string;
  /** File name the CLI would write, shown as a hint. */
  file: string;
};

export const TARGET_TABS: TargetTab[] = [
  { target: 'glsl', label: 'GLSL', file: 'shader.frag' },
  { target: 'wgsl', label: 'WGSL', file: 'shader.wgsl' },
  { target: 'hlsl', label: 'HLSL', file: 'shader.hlsl' },
  { target: 'msl', label: 'MSL', file: 'shader.metal' },
  { target: 'shaderlab', label: 'ShaderLab', file: 'Shader.shader' },
  { target: 'ue', label: 'UE', file: 'CustomNode.hlsl' },
];

/* -------------------------------------------------------------------------- */
/* Highlighting                                                                */
/* -------------------------------------------------------------------------- */

export function escapeHtml(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

/** Keywords across all six targets. Union on purpose: one table, no per-language drift. */
const KEYWORDS = new Set([
  'if', 'else', 'for', 'while', 'return', 'break', 'continue', 'discard', 'switch', 'case', 'default',
  'struct', 'fn', 'let', 'var', 'const', 'constexpr', 'static', 'uniform', 'in', 'out', 'inout',
  'varying', 'attribute', 'layout', 'precision', 'highp', 'mediump', 'lowp', 'buffer', 'cbuffer',
  'register', 'packoffset', 'true', 'false', 'loop', 'alias', 'override', 'enable', 'requires',
  'Properties', 'SubShader', 'Pass', 'CGPROGRAM', 'ENDCG', 'HLSLPROGRAM', 'ENDHLSL', 'Shader',
  'Tags', 'Fallback', 'pragma', 'vertex', 'fragment', 'kernel', 'half', 'fixed',
]);

const TYPES = new Set([
  'void', 'float', 'int', 'uint', 'bool', 'double', 'half',
  'vec2', 'vec3', 'vec4', 'mat2', 'mat3', 'mat4', 'ivec2', 'ivec3', 'ivec4', 'bvec2', 'bvec3', 'bvec4',
  'f32', 'i32', 'u32', 'f16', 'vec2f', 'vec3f', 'vec4f', 'vec2i', 'vec3i', 'vec4i',
  'mat2x2f', 'mat3x3f', 'mat4x4f', 'mat2x2', 'mat3x3', 'mat4x4',
  'float2', 'float3', 'float4', 'float2x2', 'float3x3', 'float4x4',
  'int2', 'int3', 'int4', 'uint2', 'uint3', 'uint4', 'sampler2D', 'texture2d', 'SamplerState',
]);

const TOKEN_RE = new RegExp(
  [
    '(\\/\\/[^\\n]*)', // 1 line comment
    '(\\/\\*[\\s\\S]*?\\*\\/)', // 2 block comment (single line)
    '(`(?:[^`\\\\]|\\\\.)*`|"(?:[^"\\\\]|\\\\.)*")', // 3 string
    '(#[A-Za-z_][\\w]*)', // 4 preprocessor
    '(\\b\\d+\\.?\\d*(?:[eE][+-]?\\d+)?[fFuUlL]?\\b|\\.\\d+(?:[eE][+-]?\\d+)?\\b)', // 5 number
    '([A-Za-z_]\\w*)', // 6 identifier
    '([^A-Za-z_\\d]+)', // 7 everything else
  ].join('|'),
  'g',
);

export type HighlightState = { inBlock: boolean };

/**
 * Highlight one line of source and return HTML.
 *
 * The only place a `<` from the source can reach the DOM is through `escapeHtml`, which
 * is applied to **every** token — including the "everything else" bucket, which is where
 * `a < b` and `<<` land.
 */
export function highlightLine(line: string, state: HighlightState = { inBlock: false }): string {
  if (state.inBlock) {
    const end = line.indexOf('*/');
    if (end === -1) return `<span class="hl-comment">${escapeHtml(line)}</span>`;
    state.inBlock = false;
    return (
      `<span class="hl-comment">${escapeHtml(line.slice(0, end + 2))}</span>` +
      highlightLine(line.slice(end + 2), state)
    );
  }

  let out = '';
  let last = 0;
  TOKEN_RE.lastIndex = 0;
  for (let m = TOKEN_RE.exec(line); m; m = TOKEN_RE.exec(line)) {
    if (m.index > last) out += escapeHtml(line.slice(last, m.index));
    last = m.index + m[0].length;
    const text = m[0];
    if (m[1] !== undefined) out += `<span class="hl-comment">${escapeHtml(text)}</span>`;
    else if (m[2] !== undefined) out += `<span class="hl-comment">${escapeHtml(text)}</span>`;
    else if (m[3] !== undefined) out += `<span class="hl-string">${escapeHtml(text)}</span>`;
    else if (m[4] !== undefined) out += `<span class="hl-pre">${escapeHtml(text)}</span>`;
    else if (m[5] !== undefined) out += `<span class="hl-number">${escapeHtml(text)}</span>`;
    else if (m[6] !== undefined) {
      if (KEYWORDS.has(text)) out += `<span class="hl-keyword">${escapeHtml(text)}</span>`;
      else if (TYPES.has(text)) out += `<span class="hl-type">${escapeHtml(text)}</span>`;
      else if (text.startsWith('L_')) out += `<span class="hl-local">${escapeHtml(text)}</span>`;
      else if (text.startsWith('u_') || text.startsWith('i') && /^i[A-Z]/.test(text)) {
        out += `<span class="hl-uniform">${escapeHtml(text)}</span>`;
      } else if (/^[A-Za-z_]\w*\s*\($/.test(text)) out += `<span class="hl-fn">${escapeHtml(text)}</span>`;
      else out += escapeHtml(text);
      // An unterminated block comment opens multi-line state.
      if (line.slice(m.index).startsWith('/*') && !line.slice(m.index).includes('*/')) state.inBlock = true;
    } else out += escapeHtml(text);
  }
  if (last < line.length) out += escapeHtml(line.slice(last));
  return out;
}

/* -------------------------------------------------------------------------- */
/* Panel                                                                       */
/* -------------------------------------------------------------------------- */

export type CodegenOptions = {
  catalog: Record<NodeKind, NodeSchema>;
  /** Called when the user clicks a diagnostic that carries a line number. */
  onLineClick?: (target: TargetFormat, line: number) => void;
};

type CacheEntry = { emission: Emission; lines: string[] };

export class CodegenPanel {
  readonly el: HTMLElement;

  private readonly opts: CodegenOptions;
  private readonly cache = new Map<string, CacheEntry>();
  private readonly tabs = new Map<TargetFormat, HTMLButtonElement>();
  private readonly infoEl: HTMLElement;
  private readonly codeEl: HTMLElement;
  private readonly diagEl: HTMLElement;
  private readonly copyBtn: HTMLButtonElement;

  private doc: ShaderDoc | null = null;
  private docKey = '';
  private active: TargetFormat = 'glsl';
  private lastError = '';

  constructor(opts: CodegenOptions) {
    this.opts = opts;
    this.el = document.createElement('div');
    this.el.className = 'codegen';

    const bar = document.createElement('div');
    bar.className = 'codegen-tabs';
    for (const tab of TARGET_TABS) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'tab';
      btn.textContent = tab.label;
      btn.title = tab.file;
      btn.addEventListener('click', () => void this.show(tab.target));
      this.tabs.set(tab.target, btn);
      bar.append(btn);
    }
    this.el.append(bar);

    const toolbar = document.createElement('div');
    toolbar.className = 'codegen-toolbar';
    this.infoEl = document.createElement('span');
    this.infoEl.className = 'codegen-info muted';
    this.copyBtn = document.createElement('button');
    this.copyBtn.type = 'button';
    this.copyBtn.className = 'mini';
    this.copyBtn.textContent = '复制';
    this.copyBtn.addEventListener('click', () => void this.copy());
    toolbar.append(this.infoEl, this.copyBtn);
    this.el.append(toolbar);

    this.diagEl = document.createElement('div');
    this.diagEl.className = 'codegen-diags';
    this.el.append(this.diagEl);

    this.codeEl = document.createElement('div');
    this.codeEl.className = 'codegen-code';
    this.el.append(this.codeEl);
  }

  /**
   * Point the panel at a document. Does NOT emit anything: the visible tab is emitted on
   * demand, and only if this doc hash has not been seen for that target.
   */
  setDoc(doc: ShaderDoc, docKey: string): void {
    this.doc = doc;
    this.docKey = docKey;
    void this.show(this.active);
  }

  /** Currently visible target. */
  get target(): TargetFormat {
    return this.active;
  }

  /** The cached emission for the visible tab, if it has been produced. */
  currentEmission(): Emission | undefined {
    return this.cache.get(`${this.active}:${this.docKey}`)?.emission;
  }

  async show(target: TargetFormat): Promise<void> {
    this.active = target;
    for (const [t, btn] of this.tabs) btn.classList.toggle('active', t === target);
    if (!this.doc) return;

    const entry = this.getOrEmit(target);
    if (!entry) {
      this.codeEl.replaceChildren();
      const err = document.createElement('pre');
      err.className = 'code-error';
      err.textContent = this.lastError;
      this.codeEl.append(err);
      this.infoEl.textContent = '发射失败';
      return;
    }

    const { emission } = entry;
    this.infoEl.textContent =
      `${TARGET_TABS.find((t) => t.target === target)?.file ?? ''} · hash ${emission.hash.slice(0, 8)} · ` +
      `${emission.uniforms.length} 个 uniform · ${entry.lines.length} 行`;
    this.renderDiagnostics(emission);
    this.renderCode(entry.lines);
  }

  private getOrEmit(target: TargetFormat): CacheEntry | null {
    const key = `${target}:${this.docKey}`;
    const hit = this.cache.get(key);
    if (hit) return hit;
    if (!this.doc) return null;
    try {
      const emission = emit(this.doc, { catalog: this.opts.catalog, target });
      const entry: CacheEntry = { emission, lines: emission.source.split('\n') };
      this.cache.set(key, entry);
      return entry;
    } catch (err) {
      this.lastError = `emit(${target}) 抛出异常：${err instanceof Error ? err.message : String(err)}`;
      return null;
    }
  }

  /** Force a re-emit on the next `show()` (used by the 重新编译 button). */
  invalidate(): void {
    this.cache.clear();
  }

  /* ---- rendering -------------------------------------------------------- */

  private renderCode(lines: string[]): void {
    const state: HighlightState = { inBlock: false };
    const html = lines
      .map((line, i) => {
        const text = highlightLine(line, state);
        const num = String(i + 1);
        return `<div class="code-line" data-line="${num}"><span class="ln">${num}</span><span class="lc">${text || ' '}</span></div>`;
      })
      .join('');
    // Safe: every interpolation above went through escapeHtml().
    this.codeEl.innerHTML = `<div class="code-block">${html}</div>`;
  }

  private renderDiagnostics(emission: Emission): void {
    this.diagEl.replaceChildren();
    const notes = this.doc?.targetNotes?.[emission.target];
    if (notes) {
      const note = document.createElement('p');
      note.className = 'diag diag-note';
      note.textContent = `目标说明：${notes}`;
      this.diagEl.append(note);
    }
    if (emission.diagnostics.length === 0) {
      const ok = document.createElement('p');
      ok.className = 'diag diag-ok';
      ok.textContent = '发射无诊断。';
      this.diagEl.append(ok);
      return;
    }
    const list = document.createElement('ul');
    list.className = 'diag-list';
    for (const d of emission.diagnostics) list.append(this.renderDiagnostic(d));
    this.diagEl.append(list);
  }

  private renderDiagnostic(d: Diagnostic): HTMLElement {
    const li = document.createElement('li');
    li.className = `diag diag-${d.severity}`;
    const head = document.createElement('div');
    head.className = 'diag-head';
    const code = document.createElement('code');
    code.textContent = d.code;
    head.append(code);
    if (d.at?.node) {
      const at = document.createElement('button');
      at.type = 'button';
      at.className = 'chip';
      at.textContent = d.at.port ? `${d.at.node}.${d.at.port}` : d.at.node;
      at.addEventListener('click', () => this.highlightNode(d.at?.node ?? ''));
      head.append(at);
    }
    if (typeof d.at?.line === 'number') {
      const ln = document.createElement('button');
      ln.type = 'button';
      ln.className = 'chip';
      ln.textContent = `line ${d.at.line}`;
      ln.addEventListener('click', () => this.highlightLine(d.at?.line ?? 1));
      head.append(ln);
    }
    const msg = document.createElement('div');
    msg.className = 'diag-msg';
    msg.textContent = d.message;
    li.append(head, msg);
    if (d.fix) {
      const fix = document.createElement('div');
      fix.className = 'diag-fix';
      fix.textContent = `修复建议：${d.fix}`;
      li.append(fix);
    }
    return li;
  }

  /* ---- navigation ------------------------------------------------------- */

  /**
   * Highlight every line that defines or uses `L_<node>_<port>`.
   *
   * The emitter names each node output exactly that way in all six targets (the
   * rewriter only renames *types* and the `foo_` aliases), so a plain substring search
   * over the emitted source is an honest node→line map — and it is the feature that
   * makes the generated code checkable instead of a wall of text.
   */
  highlightNode(nodeId: string): number {
    const needle = `L_${nodeId.replace(/[^A-Za-z0-9_]/g, '_')}_`;
    return this.highlightMatching((line) => line.includes(needle));
  }

  /** Highlight one 1-based line (compile diagnostics land here). */
  highlightLine(line: number): void {
    this.clearHighlights();
    const el = this.codeEl.querySelector<HTMLElement>(`.code-line[data-line="${line}"]`);
    if (!el) return;
    el.classList.add('hl-line');
    el.scrollIntoView({ block: 'center', behavior: 'smooth' });
    this.opts.onLineClick?.(this.active, line);
  }

  private highlightMatching(predicate: (line: string) => boolean): number {
    this.clearHighlights();
    const entry = this.cache.get(`${this.active}:${this.docKey}`);
    if (!entry) return 0;
    const matched: HTMLElement[] = [];
    const nodes = this.codeEl.querySelectorAll<HTMLElement>('.code-line');
    entry.lines.forEach((line, i) => {
      if (!predicate(line)) return;
      const el = nodes[i];
      if (!el) return;
      el.classList.add('hl-node');
      matched.push(el);
    });
    matched[0]?.scrollIntoView({ block: 'center', behavior: 'smooth' });
    return matched.length;
  }

  clearHighlights(): void {
    for (const el of this.codeEl.querySelectorAll('.hl-node, .hl-line')) {
      el.classList.remove('hl-node', 'hl-line');
    }
  }

  private async copy(): Promise<void> {
    const emission = this.currentEmission();
    if (!emission) return;
    try {
      await navigator.clipboard.writeText(emission.source);
      this.copyBtn.textContent = '已复制';
    } catch {
      this.copyBtn.textContent = '复制失败';
    }
    window.setTimeout(() => {
      this.copyBtn.textContent = '复制';
    }, 1200);
  }
}
