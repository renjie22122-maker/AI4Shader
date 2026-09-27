/**
 * AI4Shader — reference example: "液态金属" (liquid metal)
 * =======================================================
 *
 * This is the canonical worked example, and it is deliberately hand-authored rather
 * than generated. It exists to be three things at once:
 *
 *   1. **A behaviour spec.** `npm run emit:demo` must produce six clean files from
 *      this. If a refactor breaks this document, it broke the pipeline.
 *   2. **A tutorial.** Read the node comments below top to bottom and you have read
 *      a compressed course in procedural 2D shading: domain warp -> fbm detail ->
 *      multi-stop ramp -> grade.
 *   3. **A single source shared by the CLI example and the browser workbench.** The
 *      checked-in `examples/liquid-metal.shader.json` is generated from this file
 *      (`npm run build:demo`), so the two can never drift.
 *
 * Every tunable is exposed as a named uniform, which is the point: the effect has
 * exactly nine knobs, all reachable from the GUI sliders, the CLI's `--set`, and the
 * optimizer's search space, without touching the graph.
 *
 * ── Graph ──────────────────────────────────────────────────────────────────────
 *
 *     n_uv ──► n_warp ──► n_detail ──► n_ramp ──► n_grade ──► n_out
 *                 ▲                       ▲
 *            n_warp_noise            (3 colour stops)
 *
 *   n_uv          normalised screen coords (IR convention: origin bottom-left)
 *   n_warp_noise  low-frequency fbm that drives the domain distortion
 *   n_warp        pushes uv around by that noise -> the "flowing" quality
 *   n_detail      fine-grained fbm sampled in the *warped* space -> surface grain
 *   n_ramp        maps the 0..1 detail value through steel -> gunmetal -> highlight
 *   n_grade       exposure / contrast / saturation / gamma / tint, in that order
 *   n_out         fragment result
 *
 * ── Why these values ───────────────────────────────────────────────────────────
 *
 *   warpScale 0.9   below ~0.5 the warp is a flat shift; above ~3 it stops reading
 *                   as flow and starts reading as static.
 *   octaves 4       enough ridge structure to look metallic, cheap enough for a
 *                   256x256 draft preview inside one frame.
 *   ridged 0.75     pushes fbm toward creases; 1.0 is all creases and reads as
 *                   crumpled foil rather than poured metal.
 *   contrast 1.35   the ramp is already low-contrast by construction; this is what
 *                   makes the highlight read as a specular band.
 */

import { IR_VERSION, type ShaderDoc } from '../types.js';

export const LIQUID_METAL_ID = 'liquid-metal';

/** Build the reference document. Pure function, no I/O, deterministic. */
export function createLiquidMetalDoc(): ShaderDoc {
  return {
    irVersion: IR_VERSION,
    id: LIQUID_METAL_ID,
    title: '液态金属',
    spec: {
      intent: '流动的液态金属表面，带细腻的拉丝与高光带',
      category: 'procedural-surface',
      tags: ['金属', '流体', '噪声', '程序化', '无缝'],
      features: [
        '表面有连续流动的变形，不是静态图案',
        '存在沿流线方向的细密拉丝纹理',
        '有明暗过渡的高光带，金属质感',
        '整体色调偏冷，可在 ±0.5 色相范围内调整',
        '无可见的平铺接缝或网格状重复',
      ],
      antiFeatures: [
        'blocky 的方格状噪声',
        '硬边的黑白条纹',
        '明显的重复图案',
      ],
      outOfScope: [
        '不产生凹凸/法线信息，输出是纯 2D 颜色，不能直接驱动 3D 光照',
        '不含环境反射；要做真实反射需要额外的 cubemap 输入通道',
      ],
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
    targetNotes: {
      ue: 'UE 的 Custom 节点不能声明任意 uniform，全部 9 个参数会变成节点输入引脚；`iMouse` 在材质里没有等价物，已移除。',
      shaderlab: 'Unity 没有全局鼠标位置，`iMouse` 映射为显式 uniform `_MousePos`，需要在脚本里手动写入。',
      msl: '在非 macOS 环境无法本地编译验证，已通过 SPIR-V 交叉检查。',
    },
    graph: {
      nodes: [
        {
          id: 'n_uv',
          kind: 'uv',
          pos: { x: 0, y: 0 },
          note: '归一化屏幕坐标。原点在左下，与 GLSL 一致；每个目标在入口点各翻转一次。',
        },
        {
          id: 'n_centered',
          kind: 'transform',
          pos: { x: 130, y: -120 },
          params: {
            scale: { kind: 'vec2', value: [2, 2] },
            offset: { kind: 'vec2', value: [-0.5, -0.5] },
            rotate: { kind: 'float', value: 0 },
          },
          inputs: { uv: { node: 'n_uv', port: 'uv' } },
          note: '把 uv 从 0..1 映射到 -0.5..0.5 的居中坐标。这是"居中"的标准做法：uv 节点只提供一个输出，所以居中由 transform 表达，而不是多输出端口。',
        },
        {
          id: 'n_warp',
          kind: 'domain_warp',
          pos: { x: 260, y: 120 },
          params: {
            strength: { kind: 'param', name: 'u_warpStrength', type: 'float', value: 0.45, min: 0.0, max: 1.5, step: 0.01, label: '流动强度', group: '变形' },
            scale: { kind: 'param', name: 'u_warpScale', type: 'float', value: 0.9, min: 0.2, max: 4.0, step: 0.05, label: '流动尺度', group: '变形' },
            time: { kind: 'param', name: 'u_speed', type: 'float', value: 0.25, min: 0.0, max: 2.0, step: 0.01, label: '流动速度', group: '动画' },
            // Tiled so the document keeps its promise: its `features` claim "no visible tiling
            // seam", and a raw noise lattice cannot honour that — `hash21(floor(p))` depends on
            // the absolute cell, so uv square N+1 does not agree with square 1. The wrap costs one
            // `mod_` per corner and changes nothing visible until an engine repeats the texture.
            tile: { kind: 'float', value: 1 },
          },
          inputs: {
            uv: { node: 'n_centered', port: 'uv' },
          },
          note:
            '域变形：用噪声场推开 uv。这一个节点贡献了绝大部分"流动感"。' +
            '注意它是自包含的——噪声必须按坐标求值才能产生空间变化，所以把某个噪声节点的"单个值"接进来是没意义的，' +
            'IR 里也没有这个输入。要调制变形强度请用 `strength` 参数，或换 `scale`。（早期版本的示例文档在这里挂了一个孤立节点，是错的。）',
        },
        {
          id: 'n_detail',
          kind: 'fbm',
          pos: { x: 540, y: 120 },
          params: {
            scale: { kind: 'param', name: 'u_detailScale', type: 'float', value: 6.0, min: 1.0, max: 24.0, step: 0.5, label: '纹理密度', group: '纹理' },
            octaves: { kind: 'param', name: 'u_octaves', type: 'float', value: 4.0, min: 1, max: 8, step: 1, label: '细节层数', group: '纹理' },
            lacunarity: { kind: 'float', value: 2.0, min: 1.5, max: 3.0, step: 0.05 },
            gain: { kind: 'param', name: 'u_gain', type: 'float', value: 0.5, min: 0.2, max: 0.8, step: 0.01, label: '细节衰减', group: '纹理' },
            ridged: { kind: 'param', name: 'u_ridged', type: 'float', value: 0.75, min: 0.0, max: 1.0, step: 0.01, label: '折痕强度', group: '纹理' },
            tile: { kind: 'float', value: 1 },
          },
          inputs: { uv: { node: 'n_warp', port: 'uv' } },
          note: '细节噪声，采样在变形后的空间里，所以纹理跟着流线走而不是贴在屏幕上。',
        },
        {
          id: 'n_ramp',
          kind: 'color_ramp',
          pos: { x: 820, y: 120 },
          params: {
            colorA: { kind: 'color', value: [0.02, 0.03, 0.05, 1] },
            colorB: { kind: 'color', value: [0.35, 0.38, 0.45, 1] },
            colorC: { kind: 'color', value: [0.95, 0.97, 1.0, 1] },
          },
          inputs: { t: { node: 'n_detail', port: 'value' } },
          note: '三段色标：暗部近黑偏蓝，中间调枪灰，亮部接近纯白的冷高光。',
        },
        {
          id: 'n_grade',
          kind: 'grade',
          pos: { x: 1100, y: 120 },
          params: {
            exposure: { kind: 'float', value: 1.0, min: 0.2, max: 3.0, step: 0.01 },
            contrast: { kind: 'param', name: 'u_contrast', type: 'float', value: 1.35, min: 0.5, max: 3.0, step: 0.01, label: '对比度', group: '调色' },
            saturation: { kind: 'param', name: 'u_saturation', type: 'float', value: 0.35, min: 0.0, max: 2.0, step: 0.01, label: '饱和度', group: '调色' },
            gamma: { kind: 'float', value: 1.0, min: 0.4, max: 2.4, step: 0.01 },
            tint: { kind: 'vec3', value: [0.86, 0.92, 1.0] },
          },
          inputs: { color: { node: 'n_ramp', port: 'color' } },
          note: '低饱和度 + 冷色 tint 是"金属"而非"彩色塑料"的关键。',
        },
        {
          id: 'n_out',
          kind: 'output',
          pos: { x: 1380, y: 120 },
          inputs: { color: { node: 'n_grade', port: 'color' } },
        },
      ],
      edges: [],
    },
    provenance: {
      model: 'hand-authored',
      steps: [
        {
          at: '2024-01-01T00:00:00.000Z',
          actor: 'human',
          summary: '手写参考示例：域变形驱动的金属表面',
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
export const LIQUID_METAL_DOC: ShaderDoc = createLiquidMetalDoc();
