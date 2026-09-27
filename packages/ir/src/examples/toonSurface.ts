/**
 * AI4Shader — reference example: "三渲二（卡通着色）表面" (toon / cel shading)
 * =====================================================================
 *
 * The second reference document, and the one that answers a different question than
 * `liquidMetal.ts` does. That document is a **picture**: it reads `uv` and writes a colour, and
 * its whole world is the screen. This one is a **shading function**: it reads the surface
 * normal, the world position, the view direction and the light direction, and it computes
 * lighting. Nothing about it can be evaluated on a fullscreen quad — there is no normal on a
 * quad — which is exactly why it is here.
 *
 * It exists to be three things:
 *
 *   1. **The surface contract's behaviour spec.** `npm run emit:demo` must produce six clean
 *      files, and on GLSL/HLSL the result must survive a real compiler. A refactor that breaks
 *      the `needs.surface` flag, the interpolated varyings, or any target's second entry point
 *      breaks this document first.
 *   2. **A tutorial in layered shading.** Read the node comments top to bottom and you have the
 *      shape of almost every stylised shader: a quantised light term, a silhouette term, and a
 *      positional term, each mixed in on top of the last.
 *   3. **A single source shared by the CLI example, the Python compiler tests and the
 *      workbench.** The checked-in `examples/toon-surface.shader.json` is generated from this
 *      file (`npm run build:demo`), so the two can never drift.
 *
 * ── Graph ────────────────────────────────────────────────────────────────────────
 *
 *     n_normal ─┬─► n_nl   ──► n_shade ──► n_base ──┐
 *               │   (N·L)      (阈值)      (三段色标) │
 *     n_light ──┘                                    ├─► n_rim_mix ──► n_grad_mix ──► n_out
 *     n_normal ─┬─► n_nv ──► n_rim ──► n_rim_inv ──► n_rim_amt ──┘         ▲
 *     n_view ───┘   (N·V)    (阈值)     (1-x)        (×强度)              │
 *     n_world ──► n_height ──► n_grad ─────────────────────────────────────┘
 *
 *   Layer 1  明暗分层   N·L thresholded through a three-stop ramp — the hard terminator that
 *                       makes it read as cel shading rather than as smooth falloff.
 *   Layer 2  边缘光     V·N inverted and thresholded — a bright band at the silhouette.
 *   Layer 3  高度渐变   world-space Y through a ramp, so the model's top and bottom differ.
 *
 * ── Why these values ─────────────────────────────────────────────────────────────
 *
 *   shadeLow 0.06 / shadeHigh 0.34   The transition band is the *style*. Too narrow (< 0.1
 *                                    wide) and it aliases into a jagged terminator on a
 *                                    low-poly mesh; too wide (> 0.6) and it stops reading
 *                                    as toon at all.
 *   rimStart 0.35 / rimEnd 0.95      A rim that only appears in the last few degrees of
 *                                    grazing angle disappears on a low-poly silhouette.
 *   rimStrength 0.85                 Above ~1.3 the rim floods the whole object and the
 *                                    form is lost.
 *   gradLow -0.8 / gradHigh 0.9      The default sphere spans about ±1 in Y, so these place
 *                                    the gradient across the object rather than off it.
 *
 * The rim is built as `1 - smoothstep(...)` with **edge0 < edge1**, not by swapping the edges.
 * GLSL leaves a reversed `smoothstep` undefined; "undefined" on real drivers means "works on
 * mine", which is the kind of bug that ships.
 */

import { IR_VERSION, type ParamValue, type ShaderDoc } from '../types.js';

export const TOON_SURFACE_ID = 'toon-surface';

/** A float uniform with a slider range — the shape every tunable in this document uses. */
function knob(
  name: string,
  value: number,
  min: number,
  max: number,
  step: number,
  label: string,
  group: string,
): ParamValue {
  return { kind: 'param', name, type: 'float', value, min, max, step, label, group };
}

/** Build the reference document. Pure function, no I/O, deterministic. */
export function createToonSurfaceDoc(): ShaderDoc {
  return {
    irVersion: IR_VERSION,
    id: TOON_SURFACE_ID,
    title: '三渲二（卡通着色）表面',
    spec: {
      intent: '读表面法线与光向的三渲二（卡通）着色：硬边明暗分界、轮廓边缘光、随高度变化的固有色',
      category: 'surface-shading',
      tags: ['三渲二', '卡通', '表面着色', '边缘光', '多层叠加'],
      features: [
        '明暗之间是可控宽度的硬过渡，不是连续渐变',
        '轮廓处有一圈亮边（边缘光），强度可调',
        '固有色随世界高度变化，模型上下端色相不同',
        '三个图层各自独立可调，互不干扰',
      ],
      antiFeatures: ['平滑的写实光照过渡', '纯屏幕空间坐标产生的假立体感', '与光照方向无关的固定贴图'],
      outOfScope: [
        '不做描边（outline）：描边需要背面膨胀或后处理，不是一次前向着色能表达的',
        '不含阴影贴图，明暗分界只由 N·L 决定，看不到自身遮挡',
      ],
    },
    stage: 'fragment',
    render: {
      resolution: [512, 512],
      targetFps: 60,
      referenceGpu: '集显 / 移动端 GPU',
      qualityTier: 'medium',
      maxIterations: 1,
      transparent: false,
    },
    targets: ['glsl', 'wgsl', 'hlsl', 'msl', 'shaderlab', 'ue'],
    targetNotes: {
      glsl: '这是 surface 片元着色器：需要宿主提供写入 vWorldPos / vNormal / vUv 的顶点阶段，见文件头注释。工作台的 3D 预览就是这样的宿主。',
      hlsl: 'VSMain + PSMain 成对发射，可直接编译进管线。',
      wgsl: '顶点与片元在同一个文件里，变换块绑定在 binding 3。',
      msl: '在非 macOS 环境无法本地编译验证。',
      shaderlab: '表面输入映射到 Unity 的 worldNormal / worldPos / _WorldSpaceCameraPos / _WorldSpaceLightPos0。',
      ue: '表面输入来自 Custom 节点的四个输入引脚，需要在材质图里接好。',
    },
    graph: {
      nodes: [
        /* ---- the four host inputs ------------------------------------------ */
        {
          id: 'n_normal',
          kind: 'normal',
          pos: { x: 0, y: 0 },
          note: '世界空间单位法线。四边形着色器没有这个量 —— 这正是本示例存在的理由。',
        },
        {
          id: 'n_light',
          kind: 'light_dir',
          pos: { x: 0, y: 120 },
          note: '由表面指向光源的单位向量，由宿主提供（IR 契约里 iLightDir 就是单位向量）。',
        },
        {
          id: 'n_view',
          kind: 'view_dir',
          pos: { x: 0, y: 240 },
          note: '由表面指向相机的单位向量，宿主用 iCameraPos 与插值出的世界坐标算出来。',
        },
        {
          id: 'n_world',
          kind: 'world_pos',
          pos: { x: 0, y: 360 },
          note: '插值出的世界坐标。第三个图层用它做上下渐变。',
        },

        /* ---- layer 1: the toon ramp ---------------------------------------- */
        {
          id: 'n_nl',
          kind: 'dot',
          pos: { x: 260, y: 60 },
          inputs: { a: { node: 'n_normal', port: 'normal' }, b: { node: 'n_light', port: 'direction' } },
          note: 'N·L —— 唯一决定明暗的量。没有阴影贴图时，它也是明暗分界的全部依据。',
        },
        {
          id: 'n_shade',
          kind: 'smoothstep',
          pos: { x: 460, y: 60 },
          params: {
            edge0: knob('u_shadeLow', 0.06, -1, 1, 0.005, '暗部边界', '明暗分层'),
            edge1: knob('u_shadeHigh', 0.34, -1, 1, 0.005, '亮部边界', '明暗分层'),
          },
          inputs: { value: { node: 'n_nl', port: 'value' } },
          note: '把 N·L 压成一条窄带：带内是过渡，带外是平的。两个边界之间的宽度就是"有多卡通"。',
        },
        {
          id: 'n_base',
          kind: 'color_ramp',
          pos: { x: 660, y: 60 },
          params: {
            colorA: { kind: 'color', value: [0.09, 0.1, 0.19, 1] },
            colorB: { kind: 'color', value: [0.62, 0.5, 0.62, 1] },
            colorC: { kind: 'color', value: [0.98, 0.95, 0.88, 1] },
          },
          inputs: { t: { node: 'n_shade', port: 'value' } },
          note: '三段固有色：阴影偏冷紫，中间调中性，亮部暖白。卡通着色的"贵"感几乎全在这三个颜色里。',
        },

        /* ---- layer 2: rim light -------------------------------------------- */
        {
          id: 'n_nv',
          kind: 'dot',
          pos: { x: 260, y: 220 },
          inputs: { a: { node: 'n_normal', port: 'normal' }, b: { node: 'n_view', port: 'direction' } },
          note: 'N·V：正对相机时接近 1，轮廓处趋近 0。',
        },
        {
          id: 'n_rim',
          kind: 'smoothstep',
          pos: { x: 460, y: 220 },
          params: {
            edge0: knob('u_rimStart', 0.35, 0, 1, 0.005, '边缘光起点', '边缘光'),
            edge1: knob('u_rimEnd', 0.95, 0, 1, 0.005, '边缘光终点', '边缘光'),
          },
          inputs: { value: { node: 'n_nv', port: 'value' } },
          note: '正对相机的地方为 1，所以下面还要取反。edge0 < edge1 是必须的：反过来的 smoothstep 在 GLSL 里未定义。',
        },
        {
          id: 'n_rim_inv',
          kind: 'math',
          pos: { x: 660, y: 220 },
          params: {
            op: { kind: 'float', value: 1, min: 0, max: 14, step: 1 },
            a: { kind: 'float', value: 1 },
          },
          inputs: { b: { node: 'n_rim', port: 'value' } },
          note: 'op=1 是减法：1 - smoothstep(...)，于是轮廓处为 1、正面为 0。',
        },
        {
          id: 'n_rim_amt',
          kind: 'math',
          pos: { x: 860, y: 220 },
          params: {
            op: { kind: 'float', value: 2, min: 0, max: 14, step: 1 },
            b: knob('u_rimStrength', 0.85, 0, 2, 0.01, '边缘光强度', '边缘光'),
          },
          inputs: { a: { node: 'n_rim_inv', port: 'value' } },
          note: 'op=2 是乘法：把边缘光的强度做成一个滑杆。`mix` 内部会 clamp，所以超过 1 只是全亮。',
        },
        {
          id: 'n_rim_mix',
          kind: 'mix',
          pos: { x: 1060, y: 140 },
          params: { b: { kind: 'color', value: [0.35, 0.75, 1, 1] } },
          inputs: { a: { node: 'n_base', port: 'color' }, t: { node: 'n_rim_amt', port: 'value' } },
          note: '冷色边缘光叠在明暗分层之上 —— 第一层与第二层的合成点。',
        },

        /* ---- layer 3: height gradient -------------------------------------- */
        {
          id: 'n_height',
          kind: 'split_vec4',
          pos: { x: 260, y: 400 },
          inputs: { value: { node: 'n_world', port: 'position' } },
          note: '取世界坐标的 y。向量拆分节点存在的理由就在这：连线规则只允许加宽，一个 vec4 进不了 float 输入。',
        },
        {
          id: 'n_grad',
          kind: 'smoothstep',
          pos: { x: 460, y: 400 },
          params: {
            edge0: knob('u_gradLow', -0.8, -4, 4, 0.01, '渐变下界', '高度渐变'),
            edge1: knob('u_gradHigh', 0.9, -4, 4, 0.01, '渐变上界', '高度渐变'),
          },
          inputs: { value: { node: 'n_height', port: 'y' } },
          note: '世界空间的高度渐变，而不是屏幕空间 —— 所以相机绕着模型转的时候，它不会跟着动。',
        },
        {
          id: 'n_grad_mix',
          kind: 'mix',
          pos: { x: 1260, y: 300 },
          params: { b: { kind: 'color', value: [1.0, 0.86, 0.62, 1] } },
          inputs: { a: { node: 'n_rim_mix', port: 'value' }, t: { node: 'n_grad', port: 'value' } },
          note: '第三层叠上去：顶部偏暖，底部保持前两层的结果。',
        },
        {
          id: 'n_out',
          kind: 'output',
          pos: { x: 1460, y: 240 },
          inputs: { color: { node: 'n_grad_mix', port: 'value' } },
        },
      ],
      edges: [],
    },
    provenance: {
      model: 'hand-authored',
      steps: [
        {
          at: '2024-01-02T00:00:00.000Z',
          actor: 'human',
          summary: '手写参考示例：读取表面法线的卡通着色（三层叠加：明暗分层 + 边缘光 + 世界高度渐变）',
        },
      ],
    },
  };
}

/**
 * The document as checked in. `edges` is intentionally left empty here — the patcher's
 * `normalizeEdges()` derives it from `nodes[].inputs` at load time, so there is exactly
 * one authoritative place wires are declared and no chance of the two disagreeing.
 */
export const TOON_SURFACE_DOC: ShaderDoc = createToonSurfaceDoc();
