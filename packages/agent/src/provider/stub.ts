/**
 * AI4Shader — agent layer: the scripted stub provider
 * ====================================================
 *
 * One stub, two jobs: it satisfies `LlmProvider` **and** carries `interpret`/`propose`, so
 * the loop can be driven with no network, no API key and no clock. That is not a testing
 * convenience bolted on at the end — it is the reason the provider boundary in `types.ts`
 * is drawn where it is. A network call in the middle of an orchestration loop makes the
 * loop untestable; a deterministic replay makes every branch of the state machine,
 * including the failure branches, reachable from a test.
 *
 * ## Determinism is the contract
 *
 * No randomness, no timers, no I/O, no `Date`. The same script produces byte-identical
 * requests on every run, which is what lets a test assert on *what the model was asked*
 * (`calls`) rather than only on what came back. Prompt regressions are caught by exactly
 * that assertion — a prompt change that silently drops the catalog inventory from the
 * patch turn shows up as a changed request string, not as a mysterious model failure.
 *
 * ## The default script is a real fixture, not a placeholder
 *
 * With no script, `createStubProvider()` drafts the liquid-metal document from nothing:
 * `interpret` returns `STUB_INTENTS.liquidMetal` and `propose` returns a patch that builds
 * the whole graph with the *real* catalog's node kinds and port names. The test suite
 * emits the resulting document for all six targets and requires zero error diagnostics —
 * so if a port is renamed in `packages/ir/src/catalog/nodes.ts`, this stub stops being a
 * valid fixture and the test fails. That failure is the point: it is the canary for the
 * one failure mode the IR design exists to prevent.
 */

import {
  type IntentInput,
  type IntentProvider,
  type LlmProvider,
  type LlmReply,
  type LlmRequest,
  type PatchContext,
  type PatchOp,
  type PatchProvider,
  type ShaderNode,
  type ShaderPatch,
  type ShaderSpec,
} from '../types.js';
import { patchMessages, intentMessages } from './prompt.js';
// The stub renders real prompts for its `calls` record, and `patchUserPrompt` needs a
// catalog. Importing the real one is not a dependency smell here: the default script is
// checked against this exact catalog by the end-to-end test, so the two cannot drift
// without the test failing.
import { CATALOG } from '@ai4shader/ir/catalog';

/* -------------------------------------------------------------------------- */
/* Script                                                                      */
/* -------------------------------------------------------------------------- */

export type StubScript = {
  /** One reply per `interpret` call, in order. */
  intents?: ShaderSpec[];
  /** One reply per `propose` call, in order. The last one repeats when exhausted. */
  patches?: ShaderPatch[];
  /** Throw on call N to exercise the provider-error branch. 0-based. */
  failAtCall?: number;
};

/**
 * The provider the loop actually consumes: `LlmProvider` for transport plus the two
 * narrow jobs. `calls` is the record of what was asked, and it is the only introspection
 * a test needs — the stub deliberately exposes no "canned answers" lookup, because a test
 * that asserts on the answer it just configured asserts nothing.
 */
export type StubProvider = LlmProvider & IntentProvider & PatchProvider & {
  /** Every request this provider was given, in call order. Same objects it received. */
  readonly calls: LlmRequest[];
};

/* -------------------------------------------------------------------------- */
/* Fixtures: hand-written intents                                              */
/* -------------------------------------------------------------------------- */

/** The fixture slugs `STUB_INTENTS` is keyed by. */
export type StubIntentSlug = 'liquidMetal' | 'plasma' | 'waterCaustics';

/**
 * Hand-written `ShaderSpec` fixtures, keyed by a short slug.
 *
 * These are **fixtures, not model output**: they are written the way a good intent reply
 * looks, so the tests that consume them test the loop rather than a model's mood. The
 * `features` are all observable on purpose — that is the field a critic checks the
 * rendered image against, and a feature nobody can see disables that check.
 */
export const STUB_INTENTS: Record<StubIntentSlug, ShaderSpec> = {
  liquidMetal: {
    intent: '流动的液态金属表面，带细腻的拉丝与高光带',
    category: 'procedural-surface',
    tags: ['金属', '流体', '噪声', '程序化', '无缝'],
    features: [
      '表面有连续流动的变形，不是静态图案',
      '存在沿流线方向的细密拉丝纹理',
      '有明暗过渡的高光带，金属质感',
      '整体色调偏冷',
      '无可见的平铺接缝或网格状重复',
    ],
    antiFeatures: ['blocky 的方格状噪声', '硬边的黑白条纹', '明显的重复图案'],
    outOfScope: [
      '不产生凹凸/法线信息，输出是纯 2D 颜色，不能直接驱动 3D 光照',
      '不含环境反射；要做真实反射需要额外的 cubemap 输入通道',
    ],
  },

  plasma: {
    intent: '缓慢翻滚的等离子火焰，紫红色为主，带冷色芯部',
    category: 'abstract-background',
    tags: ['等离子', '火焰', '发光', '抽象'],
    features: [
      '有缓慢翻滚的团块状结构，随时间变形',
      '亮部偏紫红，芯部偏冷白',
      '存在多个尺度的细节叠加，不是单一频率的噪声',
      '整体亮度有起伏，像在呼吸',
    ],
    antiFeatures: ['锐利的几何边', '高频闪烁'],
    outOfScope: ['不含泛光(bloom)后处理；亮度只到 1.0，超出部分被截断'],
  },

  waterCaustics: {
    intent: '清澈水面下的焦散光斑，缓慢移动，偏青蓝色',
    category: 'stylized-water',
    tags: ['水', '焦散', '青蓝', '程序化'],
    features: [
      '有网状交织的亮线，像水面聚焦的光斑',
      '亮线随时间缓慢漂移',
      '暗部偏深青，亮部接近白色',
      '有远近层次：近处光斑更大更疏',
    ],
    antiFeatures: ['静止不动的图案', '均匀的噪声雾'],
    outOfScope: ['不含折射/视差；水下物体不会因水面而扭曲'],
  },
};

/* -------------------------------------------------------------------------- */
/* The default script                                                          */
/* -------------------------------------------------------------------------- */

/** Node ids the default patch builds, in the order it builds them. */
export const LIQUID_METAL_NODES = ['n_uv', 'n_warp', 'n_detail', 'n_ramp', 'n_grade', 'n_out'] as const;

/**
 * The graph the default patch builds:
 *
 *     n_uv ─► n_warp ─► n_detail ─► n_ramp ─► n_grade ─► n_out
 *             (transform)  (fbm)     (ramp)    (grade)
 *
 * `n_warp` is a `transform`, and that is a deliberate reading of the requirement that
 * `transform` is what produces centred uv. Multi-output nodes *are* supported now — `time`,
 * `polar`, `voronoi` and `sdf` each expose several ports — but the scalar-only rule means a
 * node's ports must be scalars, and centred uv is a `vec2`. So "centred uv" cannot be a second
 * output of `uv` even in principle, and it has to be a node. In a six-node graph that node is
 * `n_warp`: the alternative — a seventh node — spends a node to say one thing the graph
 * already says, and the id is a label, not a contract.
 *
 * The flow still comes from the noise being sampled in a *scaled* space, so the surface
 * reads as brushed metal rather than as flat grain.
 */
function liquidMetalPatch(): ShaderPatch {
  const node = (
    id: string,
    kind: ShaderNode['kind'],
    pos: { x: number; y: number },
    note: string,
    params?: ShaderNode['params'],
  ): ShaderNode => ({
    id,
    kind,
    pos,
    ...(params ? { params } : {}),
    note,
  });

  const nodes: ShaderNode[] = [
    node('n_uv', 'uv', { x: 0, y: 0 }, '归一化屏幕坐标，原点左下（GLSL 约定）。'),

    node(
      'n_warp',
      'transform',
      { x: 160, y: 0 },
      '把 uv 从 0..1 映射到 -0.5..0.5 的居中坐标。居中必须由 `transform` 表达：`uv` 节点只有一个输出端口。',
      { scale: { kind: 'vec2', value: [2, 2] }, offset: { kind: 'vec2', value: [-0.5, -0.5] }, rotate: { kind: 'float', value: 0 } },
    ),

    node(
      'n_detail',
      'fbm',
      { x: 420, y: 0 },
      '细节噪声，采样在居中并被缩放的空间里，所以纹理有方向感而不是贴在屏幕上。ridged 推向折痕，是"金属"而非"塑料"的关键。',
      {
        scale: { kind: 'float', value: 3.5, min: 1, max: 24, step: 0.5 },
        octaves: { kind: 'float', value: 5, min: 1, max: 8, step: 1 },
        lacunarity: { kind: 'float', value: 2, min: 1.5, max: 3, step: 0.05 },
        gain: { kind: 'float', value: 0.5, min: 0.2, max: 0.8, step: 0.01 },
        ridged: { kind: 'float', value: 0.75, min: 0, max: 1, step: 0.01 },
      },
    ),

    node(
      'n_ramp',
      'color_ramp',
      { x: 680, y: 0 },
      '三段色标：暗部近黑偏蓝，中间调枪灰，亮部接近纯白的冷高光。',
      {
        colorA: { kind: 'color', value: [0.02, 0.03, 0.05, 1] },
        colorB: { kind: 'color', value: [0.35, 0.38, 0.45, 1] },
        colorC: { kind: 'color', value: [0.95, 0.97, 1, 1] },
      },
    ),

    node(
      'n_grade',
      'grade',
      { x: 940, y: 0 },
      '低饱和度 + 冷色 tint，是"金属"而不是"彩色塑料"的关键。一个图里只用一次，放在末端。',
      {
        exposure: { kind: 'float', value: 0, min: -2, max: 2, step: 0.01 },
        contrast: { kind: 'float', value: 1.35, min: 0.5, max: 3, step: 0.01 },
        saturation: { kind: 'float', value: 0.35, min: 0, max: 2, step: 0.01 },
        gamma: { kind: 'float', value: 1, min: 0.1, max: 3, step: 0.01 },
        tint: { kind: 'vec3', value: [0.86, 0.92, 1] },
      },
    ),

    node('n_out', 'output', { x: 1200, y: 0 }, '最终图像。一个图里恰好一个 output 节点。'),
  ];

  // Port names below are the catalog's, verbatim. `connect` is written out for every wire
  // so the patch reads as the graph it builds; `add_node.inputs` is deliberately NOT used,
  // because a wire declared inside `add_node` cannot reference a node that has not been
  // added yet — the patcher rejects it as a dangling wire.
  const wires: [string, string, string, string][] = [
    ['n_uv', 'uv', 'n_warp', 'uv'],
    ['n_warp', 'uv', 'n_detail', 'uv'],
    ['n_detail', 'value', 'n_ramp', 't'],
    ['n_ramp', 'color', 'n_grade', 'color'],
    ['n_grade', 'color', 'n_out', 'color'],
  ];

  const ops: PatchOp[] = [
    ...nodes.map((n): PatchOp => ({
      op: 'add_node',
      node: n,
      reason: n.note ?? `add ${n.id}`,
    })),
    ...wires.map(([fromId, fromPort, toId, toPort]): PatchOp => ({
      op: 'connect',
      from: { node: fromId, port: fromPort },
      to: { node: toId, port: toPort },
      reason: `${fromId}.${fromPort} -> ${toId}.${toPort}`,
    })),
    // Nine tunables, all of them values a human would actually drag. Every `name` is a
    // legal identifier in all six targets and `u_`-prefixed, so a later `expose_param` on
    // the same port can never collide with a driver-reserved word.
    expose('n_detail', 'scale', 'u_detailScale', { min: 1, max: 24, step: 0.5, label: '纹理密度', group: '纹理' }),
    expose('n_detail', 'octaves', 'u_octaves', { min: 1, max: 8, step: 1, label: '细节层数', group: '纹理' }),
    expose('n_detail', 'gain', 'u_gain', { min: 0.2, max: 0.8, step: 0.01, label: '细节衰减', group: '纹理' }),
    expose('n_detail', 'ridged', 'u_ridged', { min: 0, max: 1, step: 0.01, label: '折痕强度', group: '纹理' }),
    expose('n_grade', 'contrast', 'u_contrast', { min: 0.5, max: 3, step: 0.01, label: '对比度', group: '调色' }),
    expose('n_grade', 'saturation', 'u_saturation', { min: 0, max: 2, step: 0.01, label: '饱和度', group: '调色' }),
    expose('n_grade', 'exposure', 'u_exposure', { min: -2, max: 2, step: 0.01, label: '曝光', group: '调色' }),
    expose('n_warp', 'scale', 'u_uvScale', { min: 1, max: 4, step: 0.05, label: '坐标缩放', group: '变形' }),
  ];

  return {
    summary: '搭建液态金属图：居中坐标 → 细节噪声 → 三段色标 → 调色 → 输出，并暴露 8 个可调参数',
    ops,
    author: { kind: 'llm', model: 'stub', round: 0 },
  };
}

function expose(
  node: string,
  param: string,
  name: string,
  opts: { min?: number; max?: number; step?: number; label?: string; group?: string },
): PatchOp {
  return {
    op: 'expose_param',
    node,
    param,
    name,
    ...opts,
    reason: `把 ${node}.${param} 变成可调滑块 ${name}`,
  };
}

/**
 * The default reply sequence. Kept as a function rather than a constant so each provider
 * instance gets its own arrays — a test that mutates a returned patch cannot reach into
 * another test's script.
 */
function defaultScript(): Required<Pick<StubScript, 'intents' | 'patches'>> {
  return {
    intents: [STUB_INTENTS.liquidMetal],
    patches: [liquidMetalPatch()],
    // No further patches: see `propose` — running out means "nothing more to say", not a
    // repeat. Repeating the last patch would make a stuck loop look like progress.
  };
}

/* -------------------------------------------------------------------------- */
/* The provider                                                                */
/* -------------------------------------------------------------------------- */

export type StubOptions = {
  /** Which fixture `interpret` returns when the script's intents run out. */
  fallbackIntent?: StubIntentSlug;
  /** `LlmProvider.name`; appears in traces and cost accounting. */
  name?: string;
  /** Whether `LlmMessage.image` is honoured. The stub never looks at the image. */
  vision?: boolean;
};

/**
 * Build a deterministic, scripted provider.
 *
 * Behaviour, in full:
 *
 *   • `interpret` returns `script.intents[n]`; past the end it returns the fallback
 *     fixture (`STUB_INTENTS.liquidMetal` by default). A spec is always returned —
 *     "I have no spec" is not a state the loop can act on.
 *   • `propose` returns `script.patches[n]`; past the end it returns an **empty-ops
 *     patch**, which is how the loop learns the stub has nothing left to say. The last
 *     patch repeating is available for a caller that wants it: list it twice.
 *   • Every call is recorded in `calls` before the reply is computed, so a `failAtCall`
 *     throw still leaves the request visible to the test that provoked it.
 *   • `failAtCall === n` throws on the (n+1)-th call across **both** methods, counting
 *     from 0, and the error message names the call and its `purpose`.
 *   • `LlmReply.text` is the JSON of the spec/patch, so a test can feed it straight back
 *     through `parseIntentJson` / `parsePatchJson` and exercise the real parsing path.
 */
export function createStubProvider(script?: StubScript, opts: StubOptions = {}): StubProvider {
  const fallbackSlug = opts.fallbackIntent ?? 'liquidMetal';
  const fallbackSpec = STUB_INTENTS[fallbackSlug];
  const defaults = defaultScript();

  const intents = (script?.intents ?? defaults.intents).map(cloneSpec);
  const patches = (script?.patches ?? defaults.patches).map(clonePatch);
  const failAtCall = script?.failAtCall;
  const calls: LlmRequest[] = [];

  let intentIndex = 0;
  let patchIndex = 0;

  /** Shared bookkeeping for both jobs: record, maybe fail, then reply. */
  const begin = (req: LlmRequest, signal: AbortSignal | undefined): void => {
    const call = calls.length;
    calls.push(req);
    if (signal?.aborted) {
      throw new Error(`stub provider: aborted before call ${call} (${req.purpose})`);
    }
    if (failAtCall !== undefined && call === failAtCall) {
      throw new Error(`stub provider: scripted failure at call ${call} (${req.purpose})`);
    }
  };

  const finish = (req: LlmRequest, json: string): LlmReply => ({
    text: json,
    model: opts.name ?? 'stub',
    usage: { promptTokens: countPromptTokens(req), completionTokens: json.length },
    // Always zero: a wall-clock measurement here would make `calls` non-deterministic to
    // compare and would put a timer on the loop's critical path for no information.
    ms: 0,
  });

  const provider: StubProvider = {
    name: opts.name ?? 'stub',
    vision: opts.vision ?? false,
    calls,

    async complete(req: LlmRequest, signal?: AbortSignal): Promise<LlmReply> {
      begin(req, signal);
      // `complete` is the transport method: it does not interpret the request, so it
      // replies with the next scripted spec/patch chosen by `purpose`. A caller using the
      // stub as a plain `LlmProvider` gets the same bytes `interpret`/`propose` would.
      const json = req.purpose.startsWith('intent')
        ? JSON.stringify(nextIntent())
        : JSON.stringify(nextPatch());
      return finish(req, json);
    },

    async interpret(input: IntentInput, signal?: AbortSignal): Promise<ShaderSpec> {
      const req: LlmRequest = {
        messages: intentMessages(input, provider.vision),
        json: true,
        purpose: 'intent',
      };
      begin(req, signal);
      return nextIntent();
    },

    async propose(context: PatchContext, signal?: AbortSignal): Promise<ShaderPatch> {
      const req: LlmRequest = {
        // The prompt is rendered, not faked: whatever `patchUserPrompt` grows, `calls`
        // shows it, so a test can assert on the exact bytes a model would have seen.
        messages: patchMessages(context, CATALOG),
        json: true,
        purpose: `patch:${context.stage}`,
      };
      begin(req, signal);
      return nextPatch();
    },
  };

  function nextIntent(): ShaderSpec {
    const spec = intents[intentIndex];
    if (spec) {
      intentIndex++;
      return cloneSpec(spec);
    }
    // Past the end: the same fixture again. A spec is always available because the loop
    // has no branch for "no intent" — refusing would abort a run over a fixture detail.
    return cloneSpec(fallbackSpec);
  }

  function nextPatch(): ShaderPatch {
    const patch = patches[patchIndex];
    if (patch) {
      patchIndex++;
      return clonePatch(patch);
    }
    return {
      summary: 'stub script exhausted: no further ops',
      ops: [],
      author: { kind: 'llm', model: opts.name ?? 'stub' },
    };
  }

  return provider;
}

/* -------------------------------------------------------------------------- */
/* Cloning — the stub never hands out its own state                            */
/* -------------------------------------------------------------------------- */

/**
 * Every reply is a deep copy.
 *
 * A script is a fixture, and a fixture that a caller can mutate by editing the value it
 * received is not reusable across tests: one test's `patch.ops.push(...)` would silently
 * change what the next test is given. `structuredClone` is the right tool — the whole IR
 * is JSON by design (see `types.ts`, constraint 2), so it round-trips exactly.
 */
function cloneSpec(spec: ShaderSpec): ShaderSpec {
  return structuredClone(spec);
}

function clonePatch(patch: ShaderPatch): ShaderPatch {
  return structuredClone(patch);
}

/* -------------------------------------------------------------------------- */
/* Misc                                                                        */
/* -------------------------------------------------------------------------- */

/**
 * A deterministic stand-in for a token count.
 *
 * Deliberately not a real tokenizer: the loop only needs *a* number that is stable across
 * runs so cost assertions are reproducible. `chars / 4` is the usual English approximation
 * and is honest about being one.
 */
function countPromptTokens(req: LlmRequest): number {
  let chars = 0;
  for (const m of req.messages) chars += m.content.length;
  return Math.ceil(chars / 4);
}
