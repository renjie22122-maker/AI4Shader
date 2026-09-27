/**
 * AI4Shader — reference example: "荷塘" (lotus pond)
 * =================================================
 *
 * The fourth reference document and the largest: 40 nodes, eleven tunables, five layers stacked
 * in one pass. It exists to answer a question the other three cannot — **can this thing build a
 * scene, or only an effect?** A 7-node noise ramp proves the pipeline runs; a layered scene with
 * water, caustics, a reflection band, two lily pads, a lotus flower and haze proves the *catalog*
 * is wide enough to compose with, which is the claim a user tests the moment they ask for
 * something they have seen in a game.
 *
 * ── Layers, bottom to top ───────────────────────────────────────────────────────
 *
 *   1. 水面      polar radius -> ripple rings; fbm + voronoi edges -> caustics; a ramp for depth
 *   2. 倒影      a vertical gradient thresholded into a band, tinted warm-to-cool and broken up
 *                by the caustic edges so it is not a flat wash
 *   3. 睡莲叶    two `sdf` circles under different transforms, each with its own ramp and mask
 *   4. 荷花      `kaleidoscope` gives the petal repetition for free; the flower's core is a
 *                third `sdf` on top
 *   5. 氛围      a screen-space haze, then a grade
 *
 * ── Three things this document taught the catalog ───────────────────────────────
 *
 *  * **A vertical ramp is one node, not three.** The first draft wired `uv` straight into a float
 *    input, which is a type error the diagnostics named precisely; the right answer is the
 *    `gradient` node with `angle = π/2`, which also means the ramp does not care about aspect.
 *  * **Placement is `transform`.** Each element gets its own translated/rotated copy of the
 *    centred coordinates — that is what makes two lily pads two lily pads instead of one.
 *  * **Compositing is `mix` with a mask.** Every layer is `mix(under, over, mask)`, and the mask
 *    is the sdf's own output — which is why `sdf` exposes `mask` as well as `distance`.
 *
 * ── Why these values ────────────────────────────────────────────────────────────
 *
 *   rippleDensity 26   below ~10 the rings read as separate waves rather than as water; above
 *                      ~50 they alias at 512².
 *   causticScale 7     voronoi cells have to be small enough to read as light dancing on a
 *                      surface and large enough that the edges are not noise.
 *   petals 7           odd numbers avoid the mirror symmetry that makes a kaleidoscope look
 *                      manufactured.
 */

import { IR_VERSION, type ParamValue, type ShaderDoc } from '../types.js';

export const LOTUS_POND_ID = 'lotus-pond';

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

const f = (value: number): ParamValue => ({ kind: 'float', value });
const v2 = (x: number, y: number): ParamValue => ({ kind: 'vec2', value: [x, y] });
const color = (r: number, g: number, b: number): ParamValue => ({ kind: 'color', value: [r, g, b, 1] });
const wire = (node: string, port: string) => ({ node, port });

/** Build the reference document. Pure function, no I/O, deterministic. */
export function createLotusPondDoc(): ShaderDoc {
  return {
    irVersion: IR_VERSION,
    id: LOTUS_POND_ID,
    title: '荷塘',
    spec: {
      intent: '分层叠加的荷塘：水面（涟漪 + 水光）、天空倒影、睡莲叶、荷花、水雾与调色',
      category: 'layered-scene',
      tags: ['荷塘', '多层', '水面', '场景'],
      features: [
        '水面有随时间移动的细密涟漪与水光',
        '上半部有天空倒影的暖→冷渐变',
        '水面上有多片睡莲叶，位置与朝向不同',
        '有一朵花瓣数可调的荷花',
        '整体有轻微水雾与调色',
      ],
      antiFeatures: ['规则的棋盘格', '硬边的纯色块'],
      outOfScope: [
        '不含鱼与昆虫 —— 需要动画对象与运动轨迹，不是一层噪声能替代的',
        '不含真实反射（无 cubemap），倒影是渐变色带加扰动',
      ],
    },
    stage: 'fragment',
    render: {
      resolution: [512, 512],
      targetFps: 60,
      referenceGpu: '集显 / 移动端 GPU',
      qualityTier: 'medium',
      maxIterations: 4,
      transparent: false,
    },
    targets: ['glsl', 'wgsl', 'hlsl', 'msl', 'shaderlab', 'ue'],
    targetNotes: {
      msl: '在非 macOS 环境无法本地编译验证。',
      shaderlab: 'iMouse 映射为 _MousePos；本示例不读鼠标，无额外要求。',
      ue: '全部 11 个参数变成 Custom 节点的输入引脚。',
    },
    provenance: {
      model: 'hand-authored',
      steps: [
        {
          at: '2024-01-04T00:00:00.000Z',
          actor: 'human',
          summary: '手写参考示例：五层叠加的荷塘场景，用来验证目录的表达力上限',
        },
      ],
    },
    graph: {
      nodes: [
        /* ---- coordinates --------------------------------------------------- */
        { id: 'n_uv', kind: 'uv', pos: { x: 0, y: 0 } },
        {
          id: 'n_center',
          kind: 'transform',
          pos: { x: 0, y: 120 },
          params: { scale: v2(2, 2), offset: v2(-1, -1), rotate: f(0) },
          inputs: { uv: wire('n_uv', 'uv') },
          note: '居中到 -1..1：后面所有元素的摆放都在这个坐标系里做。',
        },
        {
          id: 'n_time',
          kind: 'time',
          pos: { x: 0, y: 240 },
          params: { speed: knob('u_speed', 0.35, 0, 2, 0.01, '水流速度', '水面') },
        },

        /* ---- layer 1: the water body ---------------------------------------- */
        {
          id: 'n_ripple_src',
          kind: 'polar',
          pos: { x: 200, y: 0 },
          params: { center: v2(0.2, 0.1) },
          inputs: { uv: wire('n_center', 'uv') },
          note: '极坐标给出到波源的距离，涟漪就是它的函数。',
        },
        {
          id: 'n_ripple_freq',
          kind: 'math',
          pos: { x: 400, y: 0 },
          params: { op: f(0), b: knob('u_rippleDensity', 26, 4, 80, 1, '涟漪密度', '水面') },
          inputs: { a: wire('n_ripple_src', 'radius') },
        },
        {
          id: 'n_ripple_wave',
          kind: 'math',
          pos: { x: 600, y: 0 },
          params: { op: f(1) },
          inputs: { a: wire('n_ripple_freq', 'value'), b: wire('n_time', 'time') },
          note: '减去时间：整圈涟漪向外走。',
        },
        {
          id: 'n_ripple',
          kind: 'math',
          pos: { x: 800, y: 0 },
          params: { op: f(9) },
          inputs: { a: wire('n_ripple_wave', 'value') },
          note: 'op=9 是 sin。',
        },
        { id: 'n_half', kind: 'const_float', pos: { x: 800, y: 120 }, params: { value: f(0.5) } },
        {
          id: 'n_ripple01',
          kind: 'math',
          pos: { x: 1000, y: 0 },
          params: { op: f(2) },
          inputs: { a: wire('n_ripple', 'value'), b: wire('n_half', 'value') },
          note: 'sin 的输出是 -1..1，乘 0.5 之后喂给需要 0..1 的色标。',
        },
        {
          id: 'n_caustic',
          kind: 'voronoi',
          pos: { x: 200, y: 140 },
          params: { scale: knob('u_causticScale', 7, 1, 24, 0.5, '水光尺度', '水面'), randomness: f(0.9) },
          inputs: { uv: wire('n_center', 'uv') },
          note: 'voronoi 的 edges 是细胞边界 —— 水光的经典来源。',
        },
        {
          id: 'n_surface_noise',
          kind: 'fbm',
          pos: { x: 200, y: 280 },
          params: {
            scale: knob('u_waterScale', 3.5, 0.5, 12, 0.1, '水面纹理密度', '水面'),
            octaves: f(4),
            lacunarity: f(2),
            gain: f(0.5),
            ridged: f(0.2),
          },
          inputs: { uv: wire('n_center', 'uv') },
        },
        {
          id: 'n_water_mix',
          kind: 'math',
          pos: { x: 400, y: 220 },
          params: { op: f(0) },
          inputs: { a: wire('n_surface_noise', 'value'), b: wire('n_ripple01', 'value') },
        },
        {
          id: 'n_water_tone',
          kind: 'color_ramp',
          pos: { x: 600, y: 220 },
          params: { colorA: color(0.02, 0.07, 0.09), colorB: color(0.06, 0.22, 0.24), colorC: color(0.35, 0.62, 0.6) },
          inputs: { t: wire('n_water_mix', 'value') },
          note: '水色：深处近黑的蓝绿 → 浅处的青。',
        },

        /* ---- layer 2: the reflection band ----------------------------------- */
        {
          id: 'n_reflect_ramp',
          kind: 'gradient',
          pos: { x: 600, y: 360 },
          params: { angle: f(1.5707), scale: f(1) },
          inputs: { uv: wire('n_center', 'uv') },
          note: '竖直渐变用 gradient + 90°，一个节点；拿 uv 的 y 分量要额外拆一次向量。',
        },
        {
          id: 'n_reflect_mask',
          kind: 'smoothstep',
          pos: { x: 800, y: 360 },
          params: {
            edge0: knob('u_reflectLow', 0.05, -1, 1, 0.01, '反射下界', '倒影'),
            edge1: knob('u_reflectHigh', 0.75, -1, 1, 0.01, '反射上界', '倒影'),
          },
          inputs: { value: wire('n_reflect_ramp', 'value') },
        },
        {
          id: 'n_reflect_tint',
          kind: 'color_ramp',
          pos: { x: 800, y: 480 },
          params: { colorA: color(1, 0.72, 0.55), colorB: color(0.75, 0.55, 0.75), colorC: color(0.45, 0.6, 0.9) },
          inputs: { t: wire('n_reflect_ramp', 'value') },
          note: '黄昏的暖 → 天空的冷，颜色本身随高度变化。',
        },
        {
          id: 'n_reflect_detail',
          kind: 'mix',
          pos: { x: 1000, y: 300 },
          params: { b: color(1, 0.95, 0.9) },
          inputs: { a: wire('n_water_tone', 'color'), t: wire('n_caustic', 'edges') },
          note: '用水光的亮边打散倒影，免得它是一块平的色带。',
        },
        {
          id: 'n_reflect_tinted',
          kind: 'mix',
          pos: { x: 1200, y: 300 },
          inputs: { a: wire('n_reflect_detail', 'value'), b: wire('n_reflect_tint', 'color'), t: wire('n_reflect_mask', 'value') },
        },

        /* ---- layer 3: lily pads --------------------------------------------- */
        {
          id: 'n_pad_uv',
          kind: 'transform',
          pos: { x: 1200, y: 120 },
          params: {
            scale: v2(1, 1),
            offset: v2(-0.45, 0.3),
            rotate: knob('u_padRotate', 0.4, -3.15, 3.15, 0.01, '叶片朝向', '睡莲'),
          },
          inputs: { uv: wire('n_center', 'uv') },
          note: '一块叶子 = 一套自己的坐标系 + 一个 sdf。摆放靠 transform，不靠改 sdf 的参数。',
        },
        {
          id: 'n_pad_sdf',
          kind: 'sdf',
          pos: { x: 1400, y: 120 },
          params: { shape: f(0), size: v2(0.34, 0.3), radius: f(0.18), smoothness: f(0.02) },
          inputs: { uv: wire('n_pad_uv', 'uv') },
        },
        {
          id: 'n_pad_color',
          kind: 'color_ramp',
          pos: { x: 1400, y: 240 },
          params: { colorA: color(0.06, 0.2, 0.1), colorB: color(0.16, 0.4, 0.18), colorC: color(0.35, 0.62, 0.3) },
          inputs: { t: wire('n_pad_sdf', 'distance') },
          note: '用 distance 上色：叶心深、叶缘亮，天然有一点体积感。',
        },
        {
          id: 'n_pad_mask',
          kind: 'antialias',
          pos: { x: 1600, y: 180 },
          inputs: { value: wire('n_pad_sdf', 'distance') },
          note:
            '边缘用 antialias 而不是 sdf 自带的 mask：后者的宽度是 uv 单位里的固定值，远了太利、近了太糊；' +
            'antialias 用 fwidth(distance) 把过渡宽度绑到**一个像素**上，缩放时边缘才是稳定的。',
        },
        {
          id: 'n_pad',
          kind: 'mix',
          pos: { x: 1800, y: 300 },
          inputs: { a: wire('n_reflect_tinted', 'value'), b: wire('n_pad_color', 'color'), t: wire('n_pad_mask', 'value') },
        },
        {
          id: 'n_pad2_uv',
          kind: 'transform',
          pos: { x: 1200, y: 60 },
          params: { scale: v2(0.7, 0.7), offset: v2(0.35, -0.28), rotate: f(-1.1) },
          inputs: { uv: wire('n_center', 'uv') },
        },
        {
          id: 'n_pad2_sdf',
          kind: 'sdf',
          pos: { x: 1400, y: 0 },
          params: { shape: f(0), size: v2(0.3, 0.26), radius: f(0.16), smoothness: f(0.02) },
          inputs: { uv: wire('n_pad2_uv', 'uv') },
        },
        {
          id: 'n_pad2_rgb',
          kind: 'hsv',
          pos: { x: 1600, y: 120 },
          params: { hue_shift: f(0.04), saturation: f(1.05), value: f(0.85) },
          inputs: { color: wire('n_pad_color', 'color') },
          note: '第二片叶子换个色相，免得两片看起来是复制粘贴。',
        },
        {
          id: 'n_pad2_mask',
          kind: 'antialias',
          pos: { x: 1600, y: 40 },
          inputs: { value: wire('n_pad2_sdf', 'distance') },
        },
        {
          id: 'n_pad2',
          kind: 'mix',
          pos: { x: 2000, y: 240 },
          inputs: { a: wire('n_pad', 'value'), b: wire('n_pad2_rgb', 'color'), t: wire('n_pad2_mask', 'value') },
        },

        /* ---- layer 4: the lotus flower -------------------------------------- */
        {
          id: 'n_flower_uv',
          kind: 'transform',
          pos: { x: 1600, y: 480 },
          params: { scale: v2(1, 1), offset: v2(-0.6, -0.55), rotate: f(0) },
          inputs: { uv: wire('n_center', 'uv') },
        },
        {
          id: 'n_petals',
          kind: 'kaleidoscope',
          pos: { x: 1800, y: 480 },
          params: { segments: knob('u_petals', 7, 3, 16, 1, '花瓣数', '荷花') },
          inputs: { uv: wire('n_flower_uv', 'uv') },
        },
        {
          id: 'n_petal_sdf',
          kind: 'sdf',
          pos: { x: 2000, y: 480 },
          params: { shape: f(4), size: v2(0.12, 0.5), radius: f(0.05), smoothness: f(0.03) },
          inputs: { uv: wire('n_petals', 'uv') },
          note: 'star 形状放在万花筒坐标里，一片花瓣自动变成一圈花瓣。',
        },
        {
          id: 'n_petal_color',
          kind: 'color_ramp',
          pos: { x: 2000, y: 360 },
          params: { colorA: color(0.95, 0.72, 0.82), colorB: color(0.92, 0.35, 0.55), colorC: color(0.6, 0.1, 0.3) },
          inputs: { t: wire('n_petal_sdf', 'distance') },
        },
        {
          id: 'n_petal_mask',
          kind: 'antialias',
          pos: { x: 2200, y: 480 },
          inputs: { value: wire('n_petal_sdf', 'distance') },
          note: '花瓣是最容易出锯齿的地方：细、密、又靠近画面中心。',
        },
        {
          id: 'n_flower',
          kind: 'mix',
          pos: { x: 2400, y: 420 },
          inputs: { a: wire('n_pad2', 'value'), b: wire('n_petal_color', 'color'), t: wire('n_petal_mask', 'value') },
        },
        {
          id: 'n_flower_core',
          kind: 'sdf',
          pos: { x: 2000, y: 600 },
          params: { shape: f(0), size: v2(0.08, 0.08), radius: f(0.04), smoothness: f(0.02) },
          inputs: { uv: wire('n_flower_uv', 'uv') },
        },
        {
          id: 'n_core_color',
          kind: 'color_ramp',
          pos: { x: 2200, y: 600 },
          params: { colorA: color(0.85, 0.75, 0.2), colorB: color(0.98, 0.9, 0.45), colorC: color(1, 0.98, 0.8) },
          inputs: { t: wire('n_flower_core', 'distance') },
        },
        {
          id: 'n_core_mask',
          kind: 'antialias',
          pos: { x: 2400, y: 600 },
          inputs: { value: wire('n_flower_core', 'distance') },
        },
        {
          id: 'n_flower_full',
          kind: 'mix',
          pos: { x: 2600, y: 480 },
          inputs: { a: wire('n_flower', 'value'), b: wire('n_core_color', 'color'), t: wire('n_core_mask', 'value') },
        },

        /* ---- layer 5: haze and grade ---------------------------------------- */
        {
          id: 'n_haze',
          kind: 'gradient',
          pos: { x: 2400, y: 240 },
          params: { angle: f(1.5707), scale: f(1) },
          inputs: { uv: wire('n_uv', 'uv') },
          note: '雾用的是屏幕空间的 uv（不是居中坐标）：它是一层空气，不跟着构图走。',
        },
        {
          id: 'n_haze_amt',
          kind: 'math',
          pos: { x: 2600, y: 240 },
          params: { op: f(2), b: knob('u_haze', 0.22, 0, 2, 0.01, '水雾', '氛围') },
          inputs: { a: wire('n_haze', 'value') },
        },
        {
          id: 'n_haze_mix',
          kind: 'mix',
          pos: { x: 2800, y: 360 },
          params: { b: color(1, 0.85, 0.7) },
          inputs: { a: wire('n_flower_full', 'value'), t: wire('n_haze_amt', 'value') },
        },
        {
          id: 'n_grade',
          kind: 'grade',
          pos: { x: 3000, y: 360 },
          params: {
            exposure: f(1.05),
            contrast: knob('u_contrast', 1.15, 0.5, 3, 0.01, '对比度', '调色'),
            saturation: knob('u_saturation', 1.1, 0, 2, 0.01, '饱和度', '调色'),
            gamma: f(1),
            tint: { kind: 'vec3', value: [0.98, 1, 0.98] },
          },
          inputs: { color: wire('n_haze_mix', 'value') },
        },
        {
          id: 'n_out',
          kind: 'output',
          pos: { x: 3200, y: 360 },
          inputs: { color: wire('n_grade', 'color') },
        },
      ],
      edges: [],
    },
  };
}

/**
 * The document as checked in. `edges` is intentionally left empty here — the patcher's
 * `normalizeEdges()` derives it from `nodes[].inputs` at load time, so there is exactly one
 * authoritative place wires are declared and no chance of the two disagreeing.
 */
export const LOTUS_POND_DOC: ShaderDoc = createLotusPondDoc();
