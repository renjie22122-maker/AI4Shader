/**
 * Parameter panel
 * ===============
 *
 * The panel is a *pure view over `UniformDecl[]`*: nothing here knows what a "liquid
 * metal" is, and adding a node to the catalog adds sliders with no UI work. That is the
 * payoff of parameters being first-class in the IR (DESIGN.md §3.4).
 *
 * The one rule that matters: **every interaction goes through the hot path**.
 * A slider tick calls `host.onChange(name, value)`, which does `setUniforms` + `render`
 * and never `emit()`/`compile()`. See the `// PERF:` note at that call site in `main.ts`.
 *
 * Three details that make it usable rather than merely present:
 *   • the internal uniform name is shown in monospace next to the label, because it is
 *     exactly the string the CLI takes (`--set u_noiseScale=2`);
 *   • every slider carries a chip naming the node input it came from, and clicking it
 *     highlights the generated lines for that node;
 *   • the number box and the slider are two views of one value — editing either one
 *     moves the other, and both are clamped into the declared range.
 */

import type { UniformDecl, ShaderType } from '../../../../packages/ir/src/index.js';
import type { UniformValue } from '../preview/renderer.js';

export type ParamPanelHost = {
  /** Hot path: apply the value to the live program and redraw. Never re-emits. */
  onChange(name: string, value: UniformValue): void;
  /** Fired 150ms after the last change — the signal to leave draft resolution. */
  onSettle(): void;
  /** A slider drag started (switch to the draft tier, freeze time). */
  onInteractStart(): void;
  /** Click on the provenance chip: highlight this node's lines in the code view. */
  onFocusOrigin(node: string, param: string, uniform: string): void;
  /** Per-group 重置. */
  onReset(names: string[]): void;
  /** Global 重置. */
  onResetAll(): void;
};

export type ParamPanelOptions = {
  /** node id -> node kind, used for the chip tooltip. */
  nodeKindOf?: (nodeId: string) => string | undefined;
  /**
   * True when this uniform came from a `kind:'color'` literal.
   *
   * `UniformDecl` carries a `ShaderType` (`color` is lowered to `vec4`), so the
   * distinction only exists in the document. The host answers it from the graph, which
   * is what lets a colour uniform get a colour picker and a plain vec4 not.
   */
  isColor?: (name: string) => boolean;
};

type Control = {
  decl: UniformDecl;
  /** Push an external value (reset, optimizer suggestion) into the widgets. */
  set(value: UniformValue): void;
  /** Current widget value. */
  get(): UniformValue;
};

const UNGROUPED = '未分组';

/* -------------------------------------------------------------------------- */
/* Colour helpers                                                              */
/* -------------------------------------------------------------------------- */

/** sRGB transfer function; the IR is linear everywhere (types.ts constraint 6). */
export function linearToSrgb(c: number): number {
  const x = Math.min(1, Math.max(0, c));
  return x <= 0.0031308 ? x * 12.92 : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
}

export function srgbToLinear(c: number): number {
  const x = Math.min(1, Math.max(0, c));
  return x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
}

const to255 = (c: number): number => Math.round(linearToSrgb(c) * 255);
const from255 = (v: number): number => srgbToLinear(v / 255);

function rgbToHex(rgb: readonly number[]): string {
  const h = (i: number): string => to255(rgb[i] ?? 0).toString(16).padStart(2, '0');
  return `#${h(0)}${h(1)}${h(2)}`;
}

function hexToLinear(hex: string): [number, number, number] {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  if (!m) return [0, 0, 0];
  const n = parseInt(m[1] ?? '000000', 16);
  return [from255((n >> 16) & 255), from255((n >> 8) & 255), from255(n & 255)];
}

/* -------------------------------------------------------------------------- */
/* Range inference                                                             */
/* -------------------------------------------------------------------------- */

/**
 * A slider needs a range. The catalog is supposed to declare one (DESIGN.md §9), but a
 * document may bind a uniform without min/max, and a panel that renders a zero-width
 * slider is worse than useless. Fallbacks: `value*0.25 .. value*4`, or `0..1` at zero.
 */
export function inferRange(decl: { min?: number; max?: number; step?: number; value: number | boolean | number[] }): {
  min: number;
  max: number;
  step: number;
} {
  const v = typeof decl.value === 'number' ? decl.value : 0;
  let min = decl.min;
  let max = decl.max;
  if (min === undefined || max === undefined) {
    if (v === 0) {
      min = 0;
      max = 1;
    } else if (v > 0) {
      min = v * 0.25;
      max = v * 4;
    } else {
      min = v * 4;
      max = v * 0.25;
    }
  }
  const span = max - min;
  return { min, max, step: decl.step ?? Math.max(span / 200, 1e-4) };
}

function arityOf(type: ShaderType): number {
  switch (type) {
    case 'vec2':
      return 2;
    case 'vec3':
      return 3;
    case 'vec4':
      return 4;
    default:
      return 1;
  }
}

/* -------------------------------------------------------------------------- */
/* Panel                                                                       */
/* -------------------------------------------------------------------------- */

export class ParamPanel {
  readonly el: HTMLElement;

  private readonly host: ParamPanelHost;
  private readonly opts: ParamPanelOptions;
  private readonly controls = new Map<string, Control>();
  private readonly values = new Map<string, UniformValue>();
  private settleTimer: number | null = null;

  constructor(host: ParamPanelHost, opts: ParamPanelOptions = {}) {
    this.host = host;
    this.opts = opts;
    this.el = document.createElement('div');
    this.el.className = 'param-panel';
  }

  /** Rebuild the whole panel from a new uniform list (graph changed → new uniforms). */
  setUniforms(uniforms: UniformDecl[]): void {
    this.controls.clear();
    this.values.clear();
    this.el.replaceChildren();

    if (uniforms.length === 0) {
      const empty = document.createElement('p');
      empty.className = 'muted';
      empty.textContent = '当前文档没有可调参数（kind:"param" 的输入会出现在这里）。';
      this.el.append(empty);
      return;
    }

    const groups = new Map<string, UniformDecl[]>();
    for (const u of uniforms) {
      const key = u.group && u.group.trim() ? u.group : UNGROUPED;
      const list = groups.get(key);
      if (list) list.push(u);
      else groups.set(key, [u]);
    }

    for (const [group, decls] of groups) {
      this.el.append(this.buildGroup(group, decls));
    }
  }

  /** Current values, as the optimizer and the reset logic see them. */
  currentValues(): Record<string, UniformValue> {
    const out: Record<string, UniformValue> = {};
    for (const [name, value] of this.values) out[name] = value;
    return out;
  }

  /** Push an external value into the widgets (optimizer preview, reset, restore). */
  setValue(name: string, value: UniformValue): void {
    this.values.set(name, value);
    this.controls.get(name)?.set(value);
  }

  /** True when the panel currently owns a control for this uniform. */
  has(name: string): boolean {
    return this.controls.has(name);
  }

  /* ---- group ------------------------------------------------------------ */

  private buildGroup(group: string, decls: UniformDecl[]): HTMLElement {
    const details = document.createElement('details');
    details.className = 'param-group';
    details.open = true;

    const summary = document.createElement('summary');
    const title = document.createElement('span');
    title.textContent = group;
    const count = document.createElement('span');
    count.className = 'muted';
    count.textContent = `${decls.length}`;
    const reset = document.createElement('button');
    reset.type = 'button';
    reset.className = 'mini';
    reset.textContent = '重置';
    reset.title = `把「${group}」恢复为文档原始值`;
    reset.addEventListener('click', (ev) => {
      ev.preventDefault();
      ev.stopPropagation();
      this.host.onReset(decls.map((d) => d.name));
    });
    summary.append(title, count, reset);
    details.append(summary);

    for (const decl of decls) details.append(this.buildRow(decl));
    return details;
  }

  /* ---- one row ---------------------------------------------------------- */

  private buildRow(decl: UniformDecl): HTMLElement {
    const row = document.createElement('div');
    row.className = 'param-row';
    row.dataset['uniform'] = decl.name;

    const head = document.createElement('div');
    head.className = 'param-head';

    const label = document.createElement('span');
    label.className = 'param-label';
    label.textContent = decl.label ?? decl.name;
    label.title = decl.label ? decl.name : `${decl.name}`;

    const name = document.createElement('code');
    name.className = 'param-name';
    name.textContent = decl.name;
    name.title = `CLI: --set ${decl.name}=…`;

    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'chip';
    const kind = this.opts.nodeKindOf?.(decl.origin.node);
    chip.textContent = `${decl.origin.node}.${decl.origin.param}`;
    chip.title = `来自节点 ${decl.origin.node}${kind ? ` (${kind})` : ''} 的 ${decl.origin.param} 输入 — 点击在源码中定位`;
    chip.addEventListener('click', () => this.host.onFocusOrigin(decl.origin.node, decl.origin.param, decl.name));

    head.append(label, name, chip);
    row.append(head);

    const control = this.buildControl(decl, row, head);
    this.controls.set(decl.name, control);
    this.values.set(decl.name, control.get());
    return row;
  }

  private buildControl(decl: UniformDecl, row: HTMLElement, head: HTMLElement): Control {
    const type = decl.type;
    if (type === 'bool') {
      const wrap = document.createElement('label');
      wrap.className = 'param-bool';
      const box = document.createElement('input');
      box.type = 'checkbox';
      box.checked = decl.value === true;
      box.addEventListener('change', () => {
        this.values.set(decl.name, box.checked);
        this.touch();
      });
      const text = document.createElement('span');
      text.textContent = box.checked ? '开' : '关';
      box.addEventListener('change', () => {
        text.textContent = box.checked ? '开' : '关';
      });
      wrap.append(box, text);
      row.append(wrap);
      return {
        decl,
        get: () => box.checked,
        set: (v) => {
          box.checked = v === true || v === 1;
          text.textContent = box.checked ? '开' : '关';
        },
      };
    }

    if (type === 'float' || type === 'int') {
      const { min, max, step } = inferRange(decl);
      const value = typeof decl.value === 'number' ? decl.value : 0;
      const isInt = type === 'int';
      const slider = rangeInput(isInt ? { min: Math.round(min), max: Math.round(max), step: Math.max(1, Math.round(step)) } : { min, max, step }, value);
      const number = numberInput({ min, max, step: isInt ? 1 : step }, value);
      wirePair(slider, number, (v) => {
        const out = isInt ? Math.round(v) : v;
        this.values.set(decl.name, out);
        this.touch();
      });
      const line = document.createElement('div');
      line.className = 'param-line';
      line.append(slider, number);
      row.append(line);
      return {
        decl,
        get: () => Number(slider.value),
        set: (v) => {
          const n = typeof v === 'number' ? v : Array.isArray(v) ? Number(v[0] ?? 0) : v ? 1 : 0;
          slider.value = String(isInt ? Math.round(n) : n);
          number.value = slider.value;
        },
      };
    }

    /* vec2 / vec3 / vec4 / color — one sub-slider per component. */
    const arity = arityOf(type);
    const initial = Array.isArray(decl.value) ? decl.value : [0, 0, 0, 0];
    const parts: number[] = Array.from({ length: arity }, (_, i) => Number(initial[i] ?? 0));
    const sliders: HTMLInputElement[] = [];
    const numbers: HTMLInputElement[] = [];

    const sub = document.createElement('div');
    sub.className = 'param-vec';

    let picker: HTMLInputElement | null = null;
    const isColor = this.opts.isColor?.(decl.name) ?? false;
    const components = isColor ? ['R', 'G', 'B', 'A'] : ['x', 'y', 'z', 'w'];
    for (let i = 0; i < arity; i++) {
      const rowEl = document.createElement('div');
      rowEl.className = 'param-sub';
      const tag = document.createElement('span');
      tag.className = 'param-comp';
      tag.textContent = components[i] ?? String(i);
      const per = inferRange({ min: decl.min, max: decl.max, step: decl.step, value: parts[i] ?? 0 });
      const slider = rangeInput(per, parts[i] ?? 0);
      const number = numberInput(per, parts[i] ?? 0);
      wirePair(slider, number, (v) => {
        parts[i] = v;
        this.values.set(decl.name, [...parts]);
        this.touch();
        // Keep the swatch in sync while the R/G/B sliders are dragged.
        if (picker && i < 3) picker.value = rgbToHex(parts);
      });
      sliders.push(slider);
      numbers.push(number);
      rowEl.append(tag, slider, number);
      sub.append(rowEl);
    }

    if (isColor) {
      picker = document.createElement('input');
      picker.type = 'color';
      picker.className = 'param-color';
      picker.value = rgbToHex(parts);
      picker.title = '以 sRGB 拾色；写入 IR 前会转回线性 RGB（IR 内部一律线性）';
      picker.addEventListener('input', () => {
        const linear = hexToLinear(picker?.value ?? '#000000');
        for (let i = 0; i < 3 && i < arity; i++) {
          parts[i] = linear[i] ?? 0;
          sliders[i]!.value = String(parts[i]);
          numbers[i]!.value = String(Math.round((parts[i] ?? 0) * 1000) / 1000);
        }
        this.values.set(decl.name, [...parts]);
        this.touch();
      });
      head.append(picker);
    }

    row.append(sub);
    return {
      decl,
      get: () => [...parts],
      set: (v) => {
        const arr = Array.isArray(v) ? v : [v];
        for (let i = 0; i < arity; i++) {
          parts[i] = Number(arr[i] ?? 0);
          if (sliders[i]) sliders[i]!.value = String(parts[i]);
          if (numbers[i]) numbers[i]!.value = String(parts[i]);
        }
        if (picker) picker.value = rgbToHex(parts);
      },
    };
  }

  /** One interaction finished: report the value and (re)arm the settle timer. */
  private touch(): void {
    this.host.onInteractStart();
    if (this.settleTimer !== null) window.clearTimeout(this.settleTimer);
    // 150ms is the DESIGN.md §6.3 debounce: draft while dragging, higher tier after.
    this.settleTimer = window.setTimeout(() => {
      this.settleTimer = null;
      this.host.onSettle();
    }, 150);
  }

  destroy(): void {
    if (this.settleTimer !== null) window.clearTimeout(this.settleTimer);
    this.el.replaceChildren();
  }
}

/* -------------------------------------------------------------------------- */
/* Input helpers                                                               */
/* -------------------------------------------------------------------------- */

function rangeInput(range: { min: number; max: number; step: number }, value: number): HTMLInputElement {
  const el = document.createElement('input');
  el.type = 'range';
  el.min = String(range.min);
  el.max = String(range.max);
  el.step = String(range.step);
  el.value = String(value);
  el.className = 'param-slider';
  return el;
}

function numberInput(range: { min: number; max: number; step: number }, value: number): HTMLInputElement {
  const el = document.createElement('input');
  el.type = 'number';
  el.min = String(range.min);
  el.max = String(range.max);
  el.step = String(range.step);
  el.value = String(Math.round(value * 1000) / 1000);
  el.className = 'param-number';
  return el;
}

/**
 * Slider and number box are two views of one value.
 *
 * `input` is used (not `change`) so a drag feels instant; the host re-arms its own
 * 150ms timer for the resolution upgrade, which is why nothing is debounced here.
 */
function wirePair(slider: HTMLInputElement, number: HTMLInputElement, apply: (v: number) => void): void {
  const clamp = (v: number): number => {
    const min = Number(slider.min);
    const max = Number(slider.max);
    if (!Number.isFinite(v)) return min;
    return Math.min(max, Math.max(min, v));
  };
  slider.addEventListener('input', () => {
    const v = clamp(Number(slider.value));
    number.value = String(Math.round(v * 1000) / 1000);
    apply(v);
  });
  number.addEventListener('input', () => {
    const v = clamp(Number(number.value));
    slider.value = String(v);
    apply(v);
  });
  number.addEventListener('change', () => {
    const v = clamp(Number(number.value));
    number.value = String(Math.round(v * 1000) / 1000);
    slider.value = String(v);
    apply(v);
  });
}
