/**
 * Built-in demo document — "液态金属" (liquid metal)
 * =================================================
 *
 * Hand-authored, deliberately: it is the document the workbench opens with, so it must
 * exercise the features that matter (domain warp → fbm detail → colour ramp → grade,
 * ten tunable uniforms, a real time input) without depending on an LLM.
 *
 * It is intentionally *parallel* to `packages/ir/src/examples/liquidMetal.ts`: node ids,
 * wiring and uniform names match that reference document so the CLI example and the
 * browser preview describe the same effect. The browser copy differs in exactly two
 * ways, both additive:
 *
 *   • `n_flow` — a second, time-driven domain warp, so the preview animates. The
 *     reference document's `domain_warp.time` is bound to the `u_speed` uniform (a
 *     phase offset), which is a static value; without a time source the time scrubber
 *     would be decorative.
 *   • `n_hue` — an HSV hue shift, which is where `u_metalHueShift` lives.
 *
 * ── The catalog is a separate workstream ─────────────────────────────────────
 *
 * Port names here are written against the catalog *specification*. {@link assertDemoDoc}
 * walks the document against the real catalog at startup and throws a precise error
 * naming the exact node, port and expected type — the UI turns that into a banner
 * instead of a blank screen, and {@link demoDocForCatalog} degrades to the closest
 * variant the catalog can actually express (dropping the time-driven warp and/or the
 * hue node) rather than failing outright.
 */

import { IR_VERSION, canConnect } from '../ir.js';
import type {
  NodeKind,
  NodeSchema,
  ParamValue,
  ShaderDoc,
  ShaderNode,
  ShaderType,
  WireRef,
} from '../../../../packages/ir/src/index.js';

/* -------------------------------------------------------------------------- */
/* Param helpers                                                               */
/* -------------------------------------------------------------------------- */

type NumRange = { min: number; max: number; step: number };

function param(
  name: string,
  value: number,
  range: NumRange,
  label: string,
  group: string,
): ParamValue {
  return {
    kind: 'param',
    name,
    type: 'float',
    value,
    min: range.min,
    max: range.max,
    step: range.step,
    label,
    group,
  };
}

const f = (value: number, range?: NumRange): ParamValue =>
  range ? { kind: 'float', value, min: range.min, max: range.max, step: range.step } : { kind: 'float', value };

const color = (r: number, g: number, b: number, a = 1): ParamValue => ({ kind: 'color', value: [r, g, b, a] });
const vec3 = (r: number, g: number, b: number): ParamValue => ({ kind: 'vec3', value: [r, g, b] });

const src = (node: string, port?: string): WireRef => (port ? { node, port } : { node });

/* -------------------------------------------------------------------------- */
/* Uniform catalogue (the sliders)                                             */
/* -------------------------------------------------------------------------- */

/** Groups, in panel order: 变形 / 纹理 / 动画 / 调色. */
export const DEMO_GROUPS = ['变形', '纹理', '动画', '调色'] as const;

/** Every name here is also a valid `ai4shader emit --set <name>=<value>` argument. */
export const DEMO_UNIFORM_NAMES = [
  'u_warpStrength',
  'u_warpScale',
  'u_detailScale',
  'u_octaves',
  'u_gain',
  'u_ridged',
  'u_speed',
  'u_contrast',
  'u_saturation',
  'u_metalHueShift',
] as const;

/* -------------------------------------------------------------------------- */
/* Document                                                                    */
/* -------------------------------------------------------------------------- */

export type DemoOptions = {
  /** Include the `time` node + second warp stage (requires a `time` node in the catalog). */
  animated?: boolean;
  /** Include the `hsv` hue-shift node (requires `hsv` in the catalog). */
  hueShift?: boolean;
};

export function createDemoDoc(opts: DemoOptions = {}): ShaderDoc {
  const animated = opts.animated ?? true;
  const hueShift = opts.hueShift ?? true;

  const nodes: ShaderNode[] = [];

  /* n_uv — normalised screen coordinates (IR convention: origin bottom-left). */
  nodes.push({
    id: 'n_uv',
    kind: 'uv',
    pos: { x: 0, y: 0 },
    note: '归一化屏幕坐标。原点在左下，与 GLSL 一致；每个目标在入口点各翻转一次。',
  });

  if (animated) {
    nodes.push({
      id: 'n_time',
      kind: 'time',
      pos: { x: 0, y: 180 },
      note: '宿主时间 iTime（秒）。接到域变形的 time 输入上，是动画的唯一来源。',
    });
  }

  /* n_warp — the primary domain warp: this node is most of the "flowing" quality. */
  nodes.push({
    id: 'n_warp',
    kind: 'domain_warp',
    pos: { x: 260, y: 120 },
    params: {
      strength: param('u_warpStrength', 0.45, { min: 0, max: 1.5, step: 0.01 }, '流动强度', '变形'),
      scale: param('u_warpScale', 0.9, { min: 0.2, max: 4, step: 0.05 }, '流动尺度', '变形'),
      // In the reference document this is the only time input; here the animation comes
      // from n_flow, so u_speed is the flow phase (a static offset you can dial).
      time: param('u_speed', 0.25, { min: 0, max: 2, step: 0.01 }, '流动速度', '动画'),
    },
    inputs: { uv: src('n_uv', 'uv') },
    note: '域变形：用噪声推开 uv。受力输入是 strength/scale/time 三个参数。',
  });

  if (animated) {
    /* n_flow — a second, time-driven warp stage. Small strength, so it reads as the
     * surface slowly creeping rather than as a second pattern. */
    nodes.push({
      id: 'n_flow',
      kind: 'domain_warp',
      pos: { x: 400, y: 240 },
      params: {
        strength: f(0.18, { min: 0, max: 1, step: 0.01 }),
        scale: f(1.6, { min: 0.2, max: 4, step: 0.05 }),
      },
      inputs: { uv: src('n_warp', 'uv'), time: src('n_time') },
      note: '第二级域变形，时间直接来自 iTime —— 这一级让画面动起来，且不改动第一级的形态。',
    });
  }

  /* n_detail — fine fbm sampled in the *warped* space, so the grain follows the flow. */
  nodes.push({
    id: 'n_detail',
    kind: 'fbm',
    pos: { x: 560, y: 120 },
    params: {
      scale: param('u_detailScale', 6, { min: 1, max: 24, step: 0.5 }, '纹理密度', '纹理'),
      octaves: param('u_octaves', 4, { min: 1, max: 8, step: 1 }, '细节层数', '纹理'),
      lacunarity: f(2, { min: 1.5, max: 3, step: 0.05 }),
      gain: param('u_gain', 0.5, { min: 0.2, max: 0.8, step: 0.01 }, '细节衰减', '纹理'),
      ridged: param('u_ridged', 0.75, { min: 0, max: 1, step: 0.01 }, '折痕强度', '纹理'),
    },
    inputs: { uv: src(animated ? 'n_flow' : 'n_warp', 'uv') },
    note: '细节噪声。采样在变形后的空间里，所以纹理跟着流线走而不是贴在屏幕上。',
  });

  /* n_ramp — three-stop metal ramp driven by the fbm value. */
  nodes.push({
    id: 'n_ramp',
    kind: 'color_ramp',
    pos: { x: 820, y: 120 },
    params: {
      colorA: color(0.02, 0.03, 0.05),
      colorB: color(0.35, 0.38, 0.45),
      colorC: color(0.95, 0.97, 1.0),
    },
    inputs: { t: src('n_detail', 'value') },
    note: '三段色标：暗部近黑偏蓝，中间调枪灰，亮部是冷高光。',
  });

  /* n_hue — HSV hue shift, where u_metalHueShift lives. */
  const colorSource = hueShift ? 'n_hue' : 'n_ramp';
  if (hueShift) {
    nodes.push({
      id: 'n_hue',
      kind: 'hsv',
      pos: { x: 960, y: 240 },
      params: {
        hue_shift: param('u_metalHueShift', 0, { min: -0.5, max: 0.5, step: 0.005 }, '金属色相偏移', '调色'),
        saturation: f(1, { min: 0, max: 2, step: 0.01 }),
        value: f(1, { min: 0, max: 2, step: 0.01 }),
      },
      inputs: { color: src('n_ramp', 'color') },
      note: '色相偏移：金属的冷暖全在这一个数字上，±0.5 = 半个色环。',
    });
  }

  /* n_grade — exposure / contrast / saturation / gamma / tint, in that order. */
  nodes.push({
    id: 'n_grade',
    kind: 'grade',
    pos: { x: 1120, y: 120 },
    params: {
      // `exposure` is in stops in the catalog (`pow(2, exposure)`), so 0 is neutral.
      exposure: f(0, { min: -2, max: 2, step: 0.01 }),
      contrast: param('u_contrast', 1.35, { min: 0.5, max: 3, step: 0.01 }, '对比度', '调色'),
      saturation: param('u_saturation', 0.35, { min: 0, max: 2, step: 0.01 }, '饱和度', '调色'),
      gamma: f(1, { min: 0.4, max: 2.4, step: 0.01 }),
      tint: vec3(0.86, 0.92, 1.0),
    },
    inputs: { color: src(colorSource, 'color') },
    note: '低饱和度 + 冷色 tint 是「金属」而不是「彩色塑料」的关键。',
  });

  nodes.push({
    id: 'n_out',
    kind: 'output',
    pos: { x: 1380, y: 120 },
    inputs: { color: src('n_grade', 'color') },
  });

  return {
    irVersion: IR_VERSION,
    id: 'liquid-metal',
    title: '液态金属',
    spec: {
      intent: '流动的液态金属表面，带细腻的拉丝与高光带',
      category: 'procedural-surface',
      tags: ['金属', '流体', '噪声', '程序化'],
      features: [
        '表面有连续流动的变形，不是静态图案',
        '存在沿流线方向的细密拉丝纹理',
        '有明暗过渡的高光带，金属质感',
        '整体色调偏冷，可在 ±0.5 色相范围内调整',
      ],
      antiFeatures: ['blocky 的方格状噪声', '硬边的黑白条纹', '明显的重复图案'],
      outOfScope: ['不产生凹凸/法线信息，输出是纯 2D 颜色，不能直接驱动 3D 光照'],
    },
    stage: 'fragment',
    render: {
      resolution: [512, 512],
      targetFps: 60,
      referenceGpu: '集显 / 移动端 GPU',
      qualityTier: 'medium',
      maxIterations: 6,
      transparent: false,
    },
    targets: ['glsl', 'wgsl', 'hlsl', 'msl', 'shaderlab', 'ue'],
    graph: {
      nodes,
      // Left empty on purpose: the patcher's `normalizeEdges()` derives the edge list
      // from `nodes[].inputs`, so wires are declared in exactly one place.
      edges: [],
    },
    provenance: {
      model: 'hand-authored',
      steps: [
        { at: '2024-01-01T00:00:00.000Z', actor: 'human', summary: '手写演示文档：域变形驱动的液态金属' },
      ],
    },
  };
}

/* -------------------------------------------------------------------------- */
/* Validation against the real catalog                                         */
/* -------------------------------------------------------------------------- */

/** Literal kind → the type it produces (mirrors `checkParamAgainstType` in the emitter). */
const LITERAL_TYPE: Record<string, ShaderType> = {
  float: 'float',
  int: 'int',
  bool: 'bool',
  vec2: 'vec2',
  vec3: 'vec3',
  vec4: 'vec4',
  color: 'vec4',
  param: 'float',
};

export type DemoIssue = {
  node: string;
  kind: string;
  port?: string;
  code: string;
  message: string;
};

/**
 * Walk a document against a catalog and report **every** problem with the exact node,
 * input port and expected type.
 *
 * This runs at startup because the alternative — an unknown kind surfacing as a blank
 * canvas and one `IR002` line — is a terrible first impression, and because a demo
 * document that silently drifts from the catalog is worse than no demo at all.
 */
export function collectDemoIssues(
  catalog: Partial<Record<NodeKind, NodeSchema>>,
  doc: ShaderDoc = createDemoDoc(),
): DemoIssue[] {
  const issues: DemoIssue[] = [];
  const byId = new Map(doc.graph.nodes.map((n) => [n.id, n]));

  for (const node of doc.graph.nodes) {
    const schema = catalog[node.kind];
    if (!schema) {
      issues.push({
        node: node.id,
        kind: node.kind,
        code: 'IR002_UNKNOWN_KIND',
        message: `节点 ${node.id} 的 kind \`${node.kind}\` 不在 catalog 中（catalog 现有 ${Object.keys(catalog).length} 种）`,
      });
      continue;
    }

    for (const [portName, value] of Object.entries(node.params ?? {})) {
      const port = schema.inputs.find((p) => p.name === portName);
      if (!port) {
        issues.push({
          node: node.id,
          kind: node.kind,
          port: portName,
          code: 'IR003_UNKNOWN_PORT',
          message: `节点 ${node.id} (${node.kind}) 没有输入端口 \`${portName}\`；合法端口：${schema.inputs.map((p) => p.name).join(', ') || '(无)'}`,
        });
        continue;
      }
      const actual =
        value.kind === 'param' ? (value as { type: ShaderType }).type : LITERAL_TYPE[value.kind] ?? 'float';
      if (!canConnect(actual, port.type)) {
        issues.push({
          node: node.id,
          kind: node.kind,
          port: portName,
          code: 'IR013_LITERAL_TYPE',
          message: `节点 ${node.id}.${portName} 期望 ${port.type}，但给了 ${actual}`,
        });
      }
      if (value.kind === 'param' && !/^[A-Za-z_]\w*$/.test(value.name)) {
        issues.push({
          node: node.id,
          kind: node.kind,
          port: portName,
          code: 'IR015_BAD_UNIFORM_NAME',
          message: `uniform 名 \`${value.name}\` 不是合法标识符`,
        });
      }
    }

    for (const [portName, wire] of Object.entries(node.inputs ?? {})) {
      const port = schema.inputs.find((p) => p.name === portName);
      if (!port) {
        issues.push({
          node: node.id,
          kind: node.kind,
          port: portName,
          code: 'IR003_UNKNOWN_PORT',
          message: `节点 ${node.id} (${node.kind}) 没有输入端口 \`${portName}\`；合法端口：${schema.inputs.map((p) => p.name).join(', ') || '(无)'}`,
        });
        continue;
      }
      const source = byId.get(wire.node);
      if (!source) {
        issues.push({
          node: node.id,
          kind: node.kind,
          port: portName,
          code: 'IR004_DANGLING_WIRE',
          message: `节点 ${node.id}.${portName} 连到了不存在的节点 \`${wire.node}\``,
        });
        continue;
      }
      const sourceSchema = catalog[source.kind];
      if (!sourceSchema) continue; // already reported on the source node
      const outName = wire.port ?? sourceSchema.primaryOut ?? sourceSchema.outputs[0]?.name;
      const outPort = sourceSchema.outputs.find((p) => p.name === outName);
      if (!outPort) {
        issues.push({
          node: node.id,
          kind: node.kind,
          port: portName,
          code: 'IR005_UNKNOWN_OUTPUT',
          message: `节点 ${source.id} (${source.kind}) 没有输出端口 \`${outName}\`；合法输出：${sourceSchema.outputs.map((p) => p.name).join(', ')}`,
        });
        continue;
      }
      if (!canConnect(outPort.type, port.type)) {
        issues.push({
          node: node.id,
          kind: node.kind,
          port: portName,
          code: 'IR006_TYPE_MISMATCH',
          message: `${source.id}.${outName} 是 ${outPort.type}，但 ${node.id}.${portName} 期望 ${port.type}`,
        });
      }
    }

    // Inputs that are neither wired nor given a literal and have no schema default:
    // the emitter reports IR031 for these, so the demo must not rely on them.
    for (const port of schema.inputs) {
      const wired = node.inputs?.[port.name] !== undefined;
      const literal = node.params?.[port.name] !== undefined;
      if (!wired && !literal && !port.default) {
        issues.push({
          node: node.id,
          kind: node.kind,
          port: port.name,
          code: 'IR031_MISSING_INPUT',
          message: `节点 ${node.id} 的输入 \`${port.name}\` (${port.type}) 既没有连线也没有字面量，schema 也没有默认值`,
        });
      }
    }
  }

  return issues;
}

/**
 * Throwing form, called at startup.
 * @throws Error whose message lists every mismatch, so the UI banner is actionable.
 */
export function assertDemoDoc(
  catalog: Partial<Record<NodeKind, NodeSchema>>,
  doc: ShaderDoc = createDemoDoc(),
): void {
  const issues = collectDemoIssues(catalog, doc);
  if (issues.length === 0) return;
  const head = issues.slice(0, 12).map((i) => `  • [${i.code}] ${i.message}`);
  const more = issues.length > 12 ? `\n  … 另有 ${issues.length - 12} 处` : '';
  throw new Error(
    `演示文档与 catalog 不匹配（${issues.length} 处）：\n${head.join('\n')}${more}\n` +
      '请修正 apps/web/src/doc/demo.ts 的端口名，而不是修改 catalog。',
  );
}

export type DemoResolution = {
  doc: ShaderDoc;
  /** Which variant was usable, for the status line. */
  variant: string;
  /** Non-empty when no variant validated — the UI shows these as a banner. */
  issues: string[];
};

/**
 * Pick the richest demo variant the catalog can actually express.
 *
 * Degradation order: animated+hue → static+hue → animated → static. The static variant
 * is exactly the reference document's shape, so a catalog that supports the CLI example
 * always supports at least this.
 */
export function demoDocForCatalog(
  catalog: Partial<Record<NodeKind, NodeSchema>>,
): DemoResolution {
  const variants: { name: string; opts: DemoOptions }[] = [
    { name: '动画 + 色相', opts: { animated: true, hueShift: true } },
    { name: '静态 + 色相', opts: { animated: false, hueShift: true } },
    { name: '动画（无色相）', opts: { animated: true, hueShift: false } },
    { name: '静态（基准形态）', opts: { animated: false, hueShift: false } },
  ];

  let firstIssues: string[] = [];
  for (const variant of variants) {
    const doc = createDemoDoc(variant.opts);
    const issues = collectDemoIssues(catalog, doc);
    if (issues.length === 0) return { doc, variant: variant.name, issues: [] };
    if (firstIssues.length === 0) {
      firstIssues = issues.map((i) => `[${i.code}] ${i.message}`);
    }
  }

  return { doc: createDemoDoc({ animated: false, hueShift: false }), variant: '未验证', issues: firstIssues };
}
