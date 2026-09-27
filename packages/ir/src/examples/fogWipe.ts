/**
 * AI4Shader — reference example: "水雾玻璃（可擦）" (wipable fogged glass)
 * =====================================================================
 *
 * The third reference document, and the first one that needs **memory**. The other two compute
 * `f(uv, time)`: hand them a pixel and they answer without ever asking what the last frame looked
 * like. This one cannot work that way. "Wipe the fog off the glass" is only meaningful if the
 * wiped part *stays* wiped, and a function with no memory has nowhere to keep that — a wipe made
 * of `iMouse` alone is a hole that follows the finger and heals behind it, which is what a
 * stateless shader can honestly do and is not what anyone means by wiping.
 *
 * So the document declares `render.feedback: true`, and the host feeds the **previous frame** back
 * in through the image channel. That is the whole feature: the channel is not an input image here,
 * it is a register.
 *
 * ── The trick: accumulate clearance in the alpha channel ─────────────────────────
 *
 * The obvious formulation — `mix(previous, current, k)` — accumulates *colour*, which means the
 * whole image drifts: fog fades everywhere, not just where the finger went. What is wanted is a
 * value that only ever increases where the finger has been, and `max` gives exactly that:
 *
 *     clearance = max(previous_clearance, touches_this_frame)
 *
 * `clearance` lives in the output's **alpha** (`make_vec4(..., w: clearance)`), so the next frame
 * reads it back out of the channel with `split_vec4(...).w`. Using a component as state is the
 * standard idiom for a single-buffer feedback effect, and it is why `make_vec4` — a node that
 * exists only to put four scalars together — is load-bearing here rather than decorative.
 *
 * ── Graph (28 nodes, four tunables) ─────────────────────────────────────────────
 *
 *   state    n_prev(texture) ─► n_prev_split ──────────────► .w ─┐
 *   pointer  n_mouse ─► n_mouse_xy ─┐                            │
 *            n_res   ─► n_res_xy   ─┴─► ÷ ─► n_pointer ─┐        │
 *                                                       ▼        ▼
 *            n_uv ─► n_uv_xy ─┐                    n_dist ─► n_brush ─► 1-x ─┐
 *                            └─► make_vec4 ─────────┘                        │
 *   fog      n_fog_noise ─► n_fog (ramp) ─┐                          n_clearance = max
 *   scene    n_scene_noise ─► n_scene ────┴─► mix ──► make_vec4(rgb, w: clearance) ──► output
 *
 * ── What the host must do ───────────────────────────────────────────────────────
 *
 * Ping-pong: sample the previous frame while rendering into a second target, then swap. Bind a
 * static image instead and this shader compiles, runs at full speed, and produces a still frame —
 * which is why every emitted target carries that requirement in its own header
 * (`feedbackNotes` in `targets/_shared.ts`), and why the workbench's flat preview implements it.
 * The 3D material preview does not: "previous frame" of a moving mesh is a screen-space
 * accumulation, a different feature, and the panel says so rather than pretending.
 *
 * ── Why these values ────────────────────────────────────────────────────────────
 *
 *   brush 0.06 / brushSoft 0.14   A hard-edged brush reads as a rubber stamp; a soft one needs
 *                                 to be wide enough that the accumulation does not band. The
 *                                 softness is also what hides the per-frame sampling: the wipe
 *                                 only looks smooth because overlapping frames overlap.
 *   fogScale 9, octaves 5         Dense enough that the fog reads as droplets rather than as a
 *                                 gradient, cheap enough for a 512² preview.
 */

import { IR_VERSION, type ParamValue, type ShaderDoc } from '../types.js';

export const FOG_WIPE_ID = 'fog-wipe';

/** A float uniform with a slider range. */
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

/** Build the reference document. Pure function, no I/O, deterministic. */
export function createFogWipeDoc(): ShaderDoc {
  return {
    irVersion: IR_VERSION,
    id: FOG_WIPE_ID,
    title: '水雾玻璃（可擦）',
    spec: {
      intent: '蒙着水雾的玻璃：手指划过的地方被擦干净，并且一直保持干净，露出后面的景物',
      category: 'interactive-feedback',
      tags: ['水雾', '交互', '反馈', '擦除'],
      features: [
        '画面初始被一层细密水雾覆盖',
        '指针划过处的水雾被擦开，露出后面的景物',
        '擦开的地方不会自己恢复，越擦越透',
        '雾气有细密的纹理，不是均匀的灰',
      ],
      antiFeatures: ['擦除痕迹会随时间淡出', '笔刷边缘是硬边方块'],
      outOfScope: [
        '不含水滴沿玻璃流下的物理模拟 —— 那需要重力方向的累积场，是另一个效果',
        '不含指纹/油污等真实材质，水雾是程序化噪声',
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
      // The whole point of the document: without this the host binds a static image and the
      // wipe is a hole that follows the pointer instead of a mark that stays.
      feedback: true,
    },
    targets: ['glsl', 'wgsl', 'hlsl', 'msl', 'shaderlab', 'ue'],
    targetNotes: {
      glsl: 'readback 语义：iChannel0 必须是「上一帧」。工作台的平面预览已经这么绑（ping-pong + 离屏 FBO）；3D 材质预览只提供白色占位。',
      hlsl: '宿主需要两个 RT 轮换：本帧渲染到 B 的同时把 A 作为 iChannel0 采样，然后交换。',
      wgsl: '同上；WebGPU 后端需要两个 texture + 两个 bind group 交替。工作台目前未实现。',
      msl: '同上；非 macOS 环境无法本地编译验证。',
      shaderlab: '需要脚本层维护两张 RenderTexture 并每帧交换 _Channel0 —— Unity 不会自动做这件事。',
      ue: '材质图里把上一帧的 RenderTarget 接到通道引脚，并用 SceneCapture 每帧回写；UE 没有内建的反馈缓冲。',
    },
    provenance: {
      model: 'hand-authored',
      steps: [
        {
          at: '2024-01-03T00:00:00.000Z',
          actor: 'human',
          summary: '手写参考示例：用 render.feedback 把「擦开的水雾」累积在 alpha 通道里',
        },
      ],
    },
    graph: {
      nodes: [
        /* ---- coordinates and inputs ---------------------------------------- */
        { id: 'n_uv', kind: 'uv', pos: { x: 0, y: 0 }, note: '归一化屏幕坐标，原点左下。' },
        { id: 'n_res', kind: 'resolution', pos: { x: 0, y: 120 }, note: '像素尺寸：指针位置是像素，要除以它才能和 uv 同量纲。' },
        { id: 'n_mouse', kind: 'mouse', pos: { x: 0, y: 240 }, note: 'xy = 当前指针（像素），zw = 按下位置。IR 约定原点左下。' },

        /* ---- the state ------------------------------------------------------ */
        {
          id: 'n_prev',
          kind: 'texture',
          pos: { x: 260, y: 0 },
          inputs: { uv: { node: 'n_uv', port: 'uv' } },
          note:
            '读图像通道。在 render.feedback 文档里，这个通道装的是**上一帧自己**，' +
            '所以这一条线是整个效果的记忆：没有它，擦除只能是一帧的痕迹。',
        },
        {
          id: 'n_prev_split',
          kind: 'split_vec4',
          pos: { x: 460, y: 0 },
          inputs: { value: { node: 'n_prev', port: 'color' } },
          note: '取出 .w —— 那就是上一帧累积下来的「已擦净程度」。',
        },

        /* ---- where the pointer is, in uv ------------------------------------ */
        {
          id: 'n_mouse_xy',
          kind: 'split_vec4',
          pos: { x: 260, y: 240 },
          inputs: { value: { node: 'n_mouse', port: 'mouse' } },
          note: 'vec4 进不了任何标量输入 —— 这正是拆解节点存在的理由。',
        },
        {
          id: 'n_res_xy',
          kind: 'split_vec2',
          pos: { x: 260, y: 140 },
          inputs: { value: { node: 'n_res', port: 'resolution' } },
        },
        {
          id: 'n_px',
          kind: 'math',
          pos: { x: 460, y: 200 },
          params: { op: f(3) },
          inputs: { a: { node: 'n_mouse_xy', port: 'x' }, b: { node: 'n_res_xy', port: 'x' } },
          note: 'op=3 是除法。像素 → uv。',
        },
        {
          id: 'n_py',
          kind: 'math',
          pos: { x: 460, y: 300 },
          params: { op: f(3) },
          inputs: { a: { node: 'n_mouse_xy', port: 'y' }, b: { node: 'n_res_xy', port: 'y' } },
        },
        {
          id: 'n_pointer',
          kind: 'make_vec2',
          pos: { x: 660, y: 250 },
          inputs: { x: { node: 'n_px', port: 'value' }, y: { node: 'n_py', port: 'value' } },
        },

        /* ---- distance from this pixel to the pointer ------------------------ */
        {
          id: 'n_uv_xy',
          kind: 'split_vec2',
          pos: { x: 660, y: 0 },
          inputs: { value: { node: 'n_uv', port: 'uv' } },
          note: 'make_vec4 收四个标量，所以 vec2 必须先拆开 —— 文档里写清楚了这条规则，这里照着做。',
        },
        {
          id: 'n_ptr_xy',
          kind: 'split_vec2',
          pos: { x: 660, y: 140 },
          inputs: { value: { node: 'n_pointer', port: 'value' } },
        },
        {
          id: 'n_uv4',
          kind: 'make_vec4',
          pos: { x: 860, y: 0 },
          // w = 0 是「方向」的约定值（默认是 1，那是「点」）。distance 只读 .xyz，但把意图写下来
          // 才不会在某天换成一个读 w 的算子时悄悄出错。
          params: { z: f(0), w: f(0) },
          inputs: { x: { node: 'n_uv_xy', port: 'x' }, y: { node: 'n_uv_xy', port: 'y' } },
        },
        {
          id: 'n_ptr4',
          kind: 'make_vec4',
          pos: { x: 860, y: 120 },
          params: { z: f(0), w: f(0) },
          inputs: { x: { node: 'n_ptr_xy', port: 'x' }, y: { node: 'n_ptr_xy', port: 'y' } },
        },
        {
          id: 'n_dist',
          kind: 'vec_math',
          pos: { x: 1060, y: 60 },
          params: { op: f(2) },
          inputs: { a: { node: 'n_uv4', port: 'value' }, b: { node: 'n_ptr4', port: 'value' } },
          note: 'op=2 是 distance。它读 .xyz 并忽略 w，和 dot 一致 —— 这才是四个分量里 w 的正确用法。',
        },
        {
          id: 'n_dist_x',
          kind: 'split_vec4',
          pos: { x: 1260, y: 60 },
          inputs: { value: { node: 'n_dist', port: 'value' } },
          note: 'distance 把标量广播到四个分量，取任意一个都行。',
        },
        {
          id: 'n_brush',
          kind: 'smoothstep',
          pos: { x: 1460, y: 60 },
          params: {
            edge0: knob('u_brush', 0.06, 0.005, 0.4, 0.005, '笔刷半径', '擦除'),
            edge1: knob('u_brushSoft', 0.14, 0.01, 0.6, 0.005, '笔刷边缘', '擦除'),
          },
          inputs: { value: { node: 'n_dist_x', port: 'x' } },
          note: '距离越远值越大（1 = 完全没碰到）。edge0 < edge1 是必须的。',
        },
        { id: 'n_one', kind: 'const_float', pos: { x: 1460, y: 200 }, params: { value: f(1) } },
        {
          id: 'n_touch',
          kind: 'math',
          pos: { x: 1660, y: 160 },
          params: { op: f(1) },
          inputs: { a: { node: 'n_one', port: 'value' }, b: { node: 'n_brush', port: 'value' } },
          note: 'op=1 是减法：1 - smoothstep(...) → 笔刷中心为 1，远处为 0。',
        },
        {
          id: 'n_clearance',
          kind: 'math',
          pos: { x: 1860, y: 160 },
          params: { op: f(6) },
          inputs: { a: { node: 'n_prev_split', port: 'w' }, b: { node: 'n_touch', port: 'value' } },
          note:
            'op=6 是 max —— **整个效果的机制**。取最大值意味着擦净程度只增不减：' +
            '手指离开后痕迹留在那里，而不是跟着手指走、随手离开而复原。' +
            '换成加法会过曝，换成 mix 会让整幅画慢慢变透。',
        },

        /* ---- what is behind the glass --------------------------------------- */
        {
          id: 'n_scene_noise',
          kind: 'fbm',
          pos: { x: 260, y: 420 },
          params: {
            scale: knob('u_sceneScale', 4, 1, 16, 0.5, '背景纹理密度', '背景'),
            octaves: f(4),
            lacunarity: f(2),
            gain: f(0.5),
            ridged: f(0.3),
          },
          inputs: { uv: { node: 'n_uv', port: 'uv' } },
        },
        {
          id: 'n_scene',
          kind: 'color_ramp',
          pos: { x: 460, y: 420 },
          params: { colorA: color(0.03, 0.05, 0.08), colorB: color(0.12, 0.2, 0.3), colorC: color(0.5, 0.6, 0.7) },
          inputs: { t: { node: 'n_scene_noise', port: 'value' } },
          note: '窗外的东西：冷色、低对比，因为它是被雾挡住的背景，不该抢戏。',
        },

        /* ---- the fog --------------------------------------------------------- */
        {
          id: 'n_fog_noise',
          kind: 'fbm',
          pos: { x: 660, y: 520 },
          params: {
            scale: knob('u_fogScale', 9, 1, 30, 0.5, '雾气密度', '雾气'),
            octaves: f(5),
            lacunarity: f(2.1),
            gain: f(0.55),
            ridged: f(0.45),
          },
          inputs: { uv: { node: 'n_uv', port: 'uv' } },
          note: 'ridged 0.45 让雾看起来是细密水珠而不是一团平滑的灰。',
        },
        {
          id: 'n_fog',
          kind: 'color_ramp',
          pos: { x: 860, y: 520 },
          params: { colorA: color(0.55, 0.6, 0.65), colorB: color(0.86, 0.9, 0.94), colorC: color(0.98, 0.99, 1) },
          inputs: { t: { node: 'n_fog_noise', port: 'value' } },
        },

        /* ---- compose --------------------------------------------------------- */
        {
          id: 'n_scene_under',
          kind: 'mix',
          pos: { x: 660, y: 320 },
          inputs: { a: { node: 'n_fog', port: 'color' }, b: { node: 'n_scene', port: 'color' }, t: { node: 'n_clearance', port: 'value' } },
          note: 't = 已擦净程度：0 处是雾，1 处是完全露出的背景。',
        },
        {
          id: 'n_out_rgb',
          kind: 'split_vec4',
          pos: { x: 860, y: 320 },
          inputs: { value: { node: 'n_scene_under', port: 'value' } },
        },
        {
          id: 'n_out',
          kind: 'make_vec4',
          pos: { x: 1060, y: 320 },
          inputs: {
            x: { node: 'n_out_rgb', port: 'x' },
            y: { node: 'n_out_rgb', port: 'y' },
            z: { node: 'n_out_rgb', port: 'z' },
            w: { node: 'n_clearance', port: 'value' },
          },
          note: '把状态写回 alpha —— 下一帧的 n_prev_split.w 读的就是这个值。',
        },
        {
          id: 'n_result',
          kind: 'output',
          pos: { x: 1260, y: 320 },
          inputs: { color: { node: 'n_out', port: 'value' } },
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
export const FOG_WIPE_DOC: ShaderDoc = createFogWipeDoc();
