# AI4Shader — 设计方案

> 一个把「想要的效果」变成「六种格式的可编译 shader」的 AI 工作流，并且生成结果可以被连续微调、秒级预览。

---

## 0. 一句话概括

用户给一句话描述 / 一张参考图 / 一段效果视频，AI4Shader 产出**一棵与语言无关的 Shader IR**，再由确定性编译器把它投影成 GLSL / WGSL / HLSL / MSL / ShaderLab / UE 六种格式，并在浏览器里用真实 GPU 实时预览；用户拖个滑杆或在对话框里说「噪声再大一点」，**100ms 内**就能看到新结果，且所有格式同时更新、始终语义一致。

核心赌注只有一个：

> **不要让 LLM 直接写 shader 代码，让 LLM 写「图」，让编译器写代码。**

---

## 1. 问题定义

### 1.1 直接让 LLM 逐语言写 shader，为什么不行

| 失败模式 | 具体表现 |
|---|---|
| **语义漂移** | GLSL 版和 HLSL 版是两次独立生成，噪声频率、边界处理、颜色空间必然对不上。"工具里好看、Unity 里不一样"是这类系统的头号 bug。 |
| **无法微调** | "把这个漩涡转快一点" → 必须重新生成整段代码 → 可能连带改坏别的地方，且不可复现。用户要的是转 0.3 秒看到结果，不是等 20 秒拿到一份新代码。 |
| **不可验证** | LLM 声称"写好了"，但没人能证明它能编译。没有编译器的反馈，模型只是在生成看起来像 shader 的文本。 |
| **无法评分** | "像不像参考图"是一个可测量的量（SSIM / LPIPS），但逐语言生成的流程里没有稳定基线可以做这个测量。 |
| **成本与延迟** | 6 种格式 × 每次微调 = 6N 次 LLM 调用。微调本该是 O(1) 的本地操作。 |

### 1.2 本方案的回答

把系统切成两半，中间用一份**有类型、可序列化、与目标语言无关的图**隔开：

```
  模糊的、多模态的、需要理解意图的那一半       确定的、可验证的、需要正确性的那一半
 ┌──────────────────────────────────────┐   ┌──────────────────────────────────────┐
 │  文本 / 图片 / 视频  →  Shader IR     │   │  Shader IR  →  6 种格式源码  →  像素  │
 │  LLM 负责（创造性、容错性）            │   │  编译器负责（确定性、可复现）          │
 └──────────────────────────────────────┘   └──────────────────────────────────────┘
                          ↑                                    ↓
                          └──────── 参考图对比评分反馈 ─────────┘
```

- LLM 只在**左侧**工作，输出的是图的**小补丁**，不是代码。
- 右侧是纯函数：`emit(doc, target) → source`，同样的输入永远得到逐字节相同的输出。
- 微调 = 改图上的一个数字 = 重跑右侧 = 6 种格式同时更新。

---

## 2. 架构总览

八层，自下而上：

```
┌───────────────────────────────────────────────────────────────────────────┐
│ L8  交互层    Chat 对话 / 节点画布 / 滑杆面板 / 版本树 / 评分面板 / 导出向导  │
├───────────────────────────────────────────────────────────────────────────┤
│ L7  编排层    Planner → IRDrafter → Validator → Renderer → Critic → Patcher │
│               LangGraph 风格状态机，带重试预算与失败模式分类                  │
├───────────────────────────────────────────────────────────────────────────┤
│ L6  评分层    参考图 vs 渲染图：SSIM / LPIPS / 直方图 / FFT 频带 / CLIP 语义  │
│               失败归因 → 反推到具体节点 → 生成补丁建议                       │
├───────────────────────────────────────────────────────────────────────────┤
│ L5  优化层    参数自动搜索（CMA-ES / 贝叶斯）· 预设 · 版本 A/B · 性能预算裁剪  │
├───────────────────────────────────────────────────────────────────────────┤
│ L4  预览层    WebGL2(GLSL) + WebGPU(WGSL) 实机编译 · 增量重建 · 编译缓存      │
│               多分辨率 · 时间 scrub · 参数热替换（不重新编译）               │
├───────────────────────────────────────────────────────────────────────────┤
│ L3  校验层    ① 静态：图/类型/范围  ② 结构：受限子集 lint                    │
│               ③ 真编译：glslang(WASM) / DXC / naga / Unity / UE            │
│               ④ 运行时：链接 + 首帧渲染                                      │
├───────────────────────────────────────────────────────────────────────────┤
│ L2  代码生成  IR → 拓扑序 → 类型检查 → uniform 收集 → 占位符展开              │
│               → 跨语言重写引擎 → 各目标装配 → 6 份源码 + 边车文件             │
├───────────────────────────────────────────────────────────────────────────┤
│ L1  Shader IR 有向无环图 · 43 种节点 · 类型化端口 · 一等公民参数 · 内容哈希    │
└───────────────────────────────────────────────────────────────────────────┘
```

**依赖方向严格向下。** L8 只跟 L7 说话，L2 不知道 L8 存在，L1 不 import 任何东西（连第三方库都不 import）。这保证了 IR 可以被 CLI、CI、Unity Editor 插件、Python 离线服务同时复用。

---

## 3. 核心设计决策

### 3.1 决策一：Shader IR 是唯一真源

```ts
type ShaderDoc = {
  irVersion: '1.0.0';
  id: string;
  hash: string;            // 内容哈希 = 编译缓存 key + golden 图 key + 版本树节点 id
  spec: ShaderSpec;        // 「想要什么」：意图、特征清单、反特征清单
  graph: ShaderGraph;      // 「怎么实现」：节点 + 连线
  render: RenderSpec;      // 分辨率、质量档、迭代上限、是否透明
  targets: TargetFormat[];
  provenance: {...};       // 模型、种子、父版本、编辑历史
};
```

三条不可协商的约束：

1. **与目标语言无关。** 节点里不出现任何语言名、uniform block、纹理绑定、精度限定符。这些全是 lowering 的事。
2. **纯 JSON、可哈希。** 它能被 diff、被 patch、被版本化、被缓存。`hash` 是编译产物和 golden 图的 key，这直接让"编译缓存""A/B 对比""回滚"三件事变成免费功能。
3. **确定性。** `emit(doc, target)` 是纯函数。没有 `Date.now()`，没有 `Math.random()`，所有生成标识符都由节点 id 派生。同样的 IR 永远产出逐字节相同的 shader —— 这是能安心做缓存和回归测试的前提。

### 3.2 决策二：LLM 出补丁，不出文档

LLM 从不整体重写 IR。它输出一串小操作：

```json
{
  "summary": "增强漩涡对比度并把噪声频率提高一倍",
  "ops": [
    { "op": "set_param", "node": "n_fbm_1", "param": "scale",
      "value": { "kind": "param", "name": "u_noiseScale", "type": "float", "value": 8.0, "min": 0.5, "max": 32 },
      "reason": "参考图纹理更细密" },
    { "op": "expose_param", "node": "n_grade_1", "param": "contrast",
      "name": "u_contrast", "min": 0.5, "max": 3.0, "label": "对比度" }
  ]
}
```

好处：

- **原子性**：补丁要么整体生效，要么整体拒绝。不存在"改坏了一半"的文档。
- **可审查**：每条 op 带 `reason`，UI 里就是一行人类可读的 diff。
- **可回滚**：版本树是一个 DAG，每个节点存 hash，回滚 = 切指针。
- **失败代价低**：补丁被拒绝 → 重试一次；而不是拿到一份语法错误的 200 行代码要从头 debug。
- **可被非 LLM 使用**：参数优化器（第 5 节）产出的是同一种补丁。人的滑杆操作也是。三种来源共用一条管线，UI 和日志不用分情况处理。

### 3.3 决策三：单一受限 GLSL 子集 + 跨语言重写引擎

这是整个多格式能力的支点，也是这个方案里最有技术含量的一块。

**问题**：为 6 种语言各写一套 AST 发射器 = 6 份会各自漂移的实现。噪声函数在 6 个地方被重新实现，就是"6 个平台的画面不一样"的根源。

**做法**：内容只写一次，用一份**故意收窄的 GLSL 子集**（`packages/ir/src/glslSubset.ts`）：

```glsl
// 允许：float/vec2/vec3/vec4/mat2/mat3 字面量与构造器、+ - * / 比较 三元、
//       单分量或四分量 swizzle、if/else、常量边界的 for、白名单函数、
//       编译器提供的共享函数库（hash/noise/fbm/voronoi/sdf/color/palette/tonemap）
// 禁止：# 预处理、precision、layout、sampler、texture()、discard、struct、
//       数组、位运算、uint、while/switch、非 const 循环边界、隐式 int→float
```

收窄是为了让**重写是可靠的**：子集小到可以用 token 级变换证明正确，不需要写 5 个解析器。

跨语言差异集中在一张表里（`rewrite.ts`）：

| 作者写法 | GLSL | WGSL | HLSL / UE | MSL |
|---|---|---|---|---|
| `vec3` | `vec3` | `vec3f` | `float3` | `float3` |
| `fract_` | `fract` | `fract` | `frac` | `fract` |
| `mod_` | `mod` | `wf_mod`（垫片） | `fmod` | `fmod` |
| `mix_` | `mix` | `wf_mix`（垫片） | `lerp` | `mix` |
| `atan2_(y,x)` | `atan(y,x)` | `atan2(vec2f(y,x))` | `atan2(y,x)` | `atan2(y,x)` |
| `inversesqrt_` | `inversesqrt` | `inverseSqrt` | `rsqrt` | `rsqrt` |
| `tex2D_` | `texture` | `textureSample` | `tex2D` | `sample` |
| `mat2_mul_(m,v)` | `m * v` | `m * v` | **`mul(m, v)`** | `m * v` |
| `dFdx_` | `dFdx` | `dpdx` | `ddx` | `dfdx` |

最后一行 `mat2_mul_` 说明了为什么不能只做字符串替换：HLSL 的 `*` 对矩阵是逐分量乘，语义完全不对；而 GLSL/WGSL/MSL 是列主序，`m * v` 才对。重写器对这类函数做**参数级**处理（解析实参列表，重组），而不是改名。

**重写器是全函数**：不认识的 token 原样透传；认识的但无法正确 lower 的，报 `RW0xx` 诊断而不是生成"看起来合理"的错误代码。

**共享函数库是同一份 GLSL 源码**，所以浏览器预览里的噪声和 Unity 打包后的噪声是同一个函数逐行重写来的。这条约束消灭了最大的一类跨平台 bug。

### 3.4 决策四：参数是一等公民

IR 里的任何数字输入都有两种形态：

```ts
// 内联字面量 —— 固定值，不暴露
{ kind: 'float', value: 4.0 }

// 绑定到命名 uniform —— 变成滑杆，可被优化器搜索，可存进预设
{ kind: 'param', name: 'u_noiseScale', type: 'float', value: 4.0,
  min: 0.5, max: 32, step: 0.1, label: '噪声密度', group: '噪声' }
```

`expose_param` / `inline_param` 两个补丁操作在两种形态间自由切换。这一个设计同时喂饱了：

- **GUI**：遍历所有 `kind:'param'` 就是自动生成的滑杆面板，带正确的 min/max/step 和中文标签。
- **优化器**：搜索空间 = 所有 uniform 的 `[min,max]`，不需要问 LLM。
- **预设系统**：预设 = `{uniform 名 → 值}` 的字典，与 IR 结构解耦。
- **性能削减**：质量档位改的是 `OCTAVE_LIMIT` 这类编译期常量，一个档位一次重新发射。

### 3.5 决策五：预览必须走真实 GPU

不用软渲染器、不用近似。浏览器里就是真的 WebGL2 / WebGPU：

- GLSL 走 `WebGL2RenderingContext`，用 `gl.getShaderInfoLog()` 拿真实编译错误（带行号）。
- WGSL 走 `navigator.gpu`，用 `device.pushErrorScope('validation')` + `shaderModule.getCompilationInfo()` 拿**带行号列号的 WGSL 诊断**。WGSL 是最严格的目标，能在这里编译通过是很有价值的信号。
- 离线/CI 时用 `@webgpu/glslang`(WASM) 校验 GLSL/HLSL → SPIR-V，[naga](https://www.npmjs.com/package/@forgeax/engine-naga) / [wgslender](https://github.com/HugoDaniel/wgslender) 校验 WGSL，无需 GPU。
- Unity 与 UE 无法在浏览器里真编译，这一点**明确降级并告知用户**（见 §5.4），不用"看起来成功了"糊弄。

**实施后的补充：预览有两种"真实"，而它们不是同一件事。** 光栅化一张全屏四边形能证明"着色器能编译、能出图"，
但证明不了"这个效果在表面上是什么样" —— 一个读 `iNormal` 的文档在四边形上没有法线可言。所以预览分两条材质路径，
由发射器的 `needs.surface` 决定：四边形文档渲染成贴图再贴到受光网格上（图案是一个场，场可以被映射）；
**表面文档把发射出的 surface 着色器逐字节交给驱动**，配上写入 `vWorldPos`/`vNormal`/`vUv` 的网格顶点阶段，
在每个片元上真求值。这条路径的副产物比功能本身更值钱：预览开始**真的消费发射器的输出**，
于是"发射器与宿主对插值契约的理解不一致"这类错误第一次有了观测点（见 `IMPLEMENTATION.md` §6.28/§6.30）。

**还有第三种"真实"：无状态是这套 IR 的默认，而用户不会一直待在里面。** "手指抹开水雾玻璃"、"拖尾"、
"越擦越透"都需要**跨帧状态**，而 `f(uv, time)` 没有地方存它。文档可以用 `render.feedback` 声明
"把上一帧喂回图像通道"，宿主据此做双缓冲乒乓 —— 源码一个字节不变，因为变的只是宿主绑什么。
这个标志刻意放在**文档级**而不是从图里推断：采样一张静态图和采样自己上一帧在图上完全同形，
猜错的代价是每一个既有的通道用户悄悄变成反馈回路。反过来说，这也是 IR 的一条边界：
**能被一次前向着色表达的效果，这套系统都能做；需要多 pass 状态的效果，目前只能靠宿主提供的那一个反馈通道**
（见 `IMPLEMENTATION.md` §6.36 与 §10）。

---

## 4. Agent 工作流

### 4.1 状态机

```
                        ┌──────────────┐
   文本/图片/视频/代码 ──▶│ S1 意图解析   │  VLM + LLM → ShaderSpec
                        └──────┬───────┘   {intent, category, tags,
                               │            features[], antiFeatures[],
                               ▼            outOfScope[]}
                        ┌──────────────┐
                        │ S2 检索       │  按 category 取 few-shot：
                        └──────┬───────┘  相似 IR 片段 + 节点用法示例
                               ▼
                        ┌──────────────┐
                        │ S3 起草 IR    │  输出 NodeSchema 合法的图
                        └──────┬───────┘  （不是代码！）
                               ▼
                   ┌───────────────────────┐
                   │ S4 静态校验            │◀──────────────┐
                   │ 类型/端口/环/缺失输入   │               │
                   └───────────┬───────────┘               │
                        通过   │   失败：结构化诊断           │
                               ▼        （带 fix 字段）      │
                   ┌───────────────────────┐               │
                   │ S5 多目标发射 + 真编译 │               │
                   │ WebGL2 / WebGPU /     │               │
                   │ glslang / naga        │               │
                   └───────────┬───────────┘               │
                        通过   │   失败：编译日志            │
                               ▼        （带行号→节点映射）   │
                   ┌───────────────────────┐               │
                   │ S6 渲染 golden        │               │
                   │ 512×512 · 4 个时间点   │               │
                   └───────────┬───────────┘               │
                               ▼                           │
                   ┌───────────────────────┐               │
                   │ S7 Critic 评分         │               │
                   │ 视觉：SSIM/LPIPS      │               │
                   │ 语义：CLIP 特征对齐    │──── 需要改进 ──┘
                   │ 规格：features 逐条核  │   产出 Patch
                   │ 性能：帧时间预算       │
                   └───────────┬───────────┘
                        达标   │
                               ▼
                   ┌───────────────────────┐
                   │ S8 收尾               │  导出包 + 参数清单
                   │ 多格式 bundle + 文档   │  + 各目标限制说明
                   └───────────────────────┘
```

### 4.2 循环预算与失败升级

不能让 agent 无限打转。三级预算：

| 阶段 | 最大轮次 | 超限后的动作 |
|---|---|---|
| S4 静态修复 | 3 | 回退到 S3 重新起草，并把失败历史喂回去 |
| S5 编译修复 | 5 | 若同一错误码重复出现 ≥2 次，判定为"结构性问题"，放弃局部修补，降级到更简单的实现（如 `raymarch` → `fbm` 叠加），并在 UI 明确标注降级 |
| S7 视觉迭代 | 8 | 输出当前最优版本 + 诚实报告：哪些 `features` 没达成、差在哪、建议怎么手工调整 |

**关键点**：每一级都用**不同的策略**兜底，而不是同一策略重试。重复重试同一个失败的修补是这类 agent 最常见的浪费。

### 4.3 Critic 的失败归因（这是让它真的好用的部分）

"SSIM 0.61，不够好"对模型毫无帮助。评分层必须把差异**归因到具体节点**：

| 观测量 | 归因结论 | 反推的补丁 |
|---|---|---|
| 低频结构差异大（下采样后 SSIM 低） | 主体形状/布局不对 | 改 `sdf`/`domain_warp` 的 `scale`、`size` |
| 高频细节差异大（FFT 高频带能量差 > 2×） | 噪声频率/octave 不匹配 | 改 `scale`、`octaves`、`lacunarity` |
| 亮度直方图整体偏移 | 曝光/色调映射 | 改 `grade.exposure`、插入 `tonemap_aces` |
| 饱和度差（HSV 空间 S 通道均值差） | 色彩过淡/过艳 | 改 `grade.saturation` 或 `hsv.saturation` |
| 色相分布差异 | 配色方案不对 | 改 `color_ramp` 的三个色标 / `palette` 参数 |
| 边缘过度锐利或模糊（梯度幅值分布） | `smoothness`/`smoothstep` 边宽不对 | 改 `sdf.smoothness` |
| 帧间差异过小（时间维度 SSIM ≈ 1） | 动画没生效 | 检查 `time` 是否接进了 `domain_warp`/`noise` |

每一行都编码成一个 `if` 规则，产出**具体的 `set_param` 补丁**。这比让 LLM 看两张图猜要可靠得多，而且成本几乎为零。

---

## 5. 多格式代码生成

### 5.1 生成管线

```
ShaderDoc
   │
   ├─ 1. 图校验          analyzeGraph：端口存在性、类型兼容、环检测、死节点
   ├─ 2. 确定性拓扑排序   Kahn，同层按声明顺序 → 输出稳定
   ├─ 3. uniform 收集    去重 + 类型冲突检测 + 稳定顺序
   ├─ 4. 占位符展开      节点模板 + 已解析输入表达式 + OCTAVE_LIMIT
   ├─ 5. 跨语言重写      受限 GLSL → 目标语言（库 + 节点体 + 入口）
   └─ 6. 装配             prologue + shim + 库 + 主体 + 入口 + 边车文件
```

第 4 步有个刻意的设计：**每个节点的每个输出都变成一个局部变量**（`L_n_fbm_1_value`）。这带来三件事：生成的代码人能读；shader 调试器的行号有意义；重写器永远只见到简单语句，不用处理嵌套表达式。

### 5.2 六个目标的产物

| 目标 | 产物 | 编译方式 | 支持度 |
|---|---|---|---|
| **GLSL** | `shader.frag`（`#version 300 es`） | 浏览器 WebGL2 真编译 | ★★★★★ 完整 |
| **WGSL** | `shader.wgsl` | 浏览器 WebGPU 真编译 | ★★★★★ 完整 |
| **HLSL** | `shader.hlsl`（含 `VSMain`/`PSMain`） | DXC（WASM 或本地 `dxc.exe`） | ★★★★★ 完整 |
| **MSL** | `shader.metal` + 绑定索引 JSON | macOS `xcrun metal`；其他平台走 DXC→SPIR-V→SPIRV-Cross | ★★★★☆ 生成可靠，非 macOS 无法本地编译验证 |
| **ShaderLab** | `Shader.shader` + `.mat` payload | Unity Editor 批处理（`-executeMethod`） | ★★★★☆ 结构完整，必须 Unity 才能最终验证 |
| **UE** | HLSL 片段 + `CustomNode.inputs.json` + 搭建配方 | UE 无法对 Custom 节点做无头编译 | ★★★☆☆ 明确降级 |

### 5.3 目标间的语义对齐

三处必须显式处理的差异，全部集中在重写器和各目标入口点里，节点体内一处都不许出现：

1. **Y 轴方向**。GLSL 帧缓冲原点在左下，WGSL/HLSL/MSL/D3D 在左上。约定：**IR 内部一律按 GLSL 习惯（原点左下）**，每个目标在自己的入口点里翻转恰好一次。
   - WGSL: `let uv = vec2f(pos.x / W, 1.0 - pos.y / H);`
2. **矩阵乘法顺序**。见 §3.3，用 `mat2_mul_` 这类 helper，而不是裸 `*`。
3. **uniform 宿主映射**（UE 最麻烦，因为 Custom 节点不能声明任意 uniform）：
   - `iResolution` → UE `View.ViewSizeAndInvSize.xy`，Unity `_ScreenParams.xy`
   - `iTime` → UE `View.GameTime`，Unity `_Time.y`
   - `iMouse` → Unity/UE 都没有全局鼠标，**声明成一个显式 uniform** 并在注释里说明
   - IR 里的自定义 uniform → UE 的 Custom 节点**输入引脚**（附 `CustomNode.inputs.json` 引脚清单）

### 5.4 诚实的降级声明

系统的原则是：**做不到就明说，绝不用"看起来成功了"糊弄**。

- UE 目标不能无头编译 → 系统输出"结构已生成，需在 UE 内验证"，且不在 CI 里报绿。
- 不可表达的需求（例如"要一张手绘贴图"）→ 写进 `spec.outOfScope`，在 UI 明确提示，而不是硬凑一个不像的 shader。
- 某目标不支持某特性（例如 WGSL 某些上下文无导数）→ `targetNotes` 里写明，UI 在导出时弹出提示。

---

## 6. 微调与秒级预览

### 6.1 为什么能快

把"改一个数字"的开销拆开看：

| 步骤 | 耗时 | 优化手段 |
|---|---|---|
| 改 IR（打补丁） | < 1ms | 纯对象操作 |
| 重新发射源码 | ~1–3ms | 确定性纯函数，无 I/O |
| **编译 shader** | 5–50ms | **关键：uniform 变化不触发重编译** |
| 渲染一帧（低分辨率） | 1–8ms | 微调时降到 256×256 |
| 上传显示 | < 16ms | 直接画到 canvas |
| **合计（滑杆拖动）** | **~10–20ms / 帧** | 走在 60fps 里，感觉是"实时"而非"提交后等待" |
| 合计（结构变化，需重编译） | 30–120ms | 仍然是一次点击内的响应 |

### 6.2 参数热替换（最重要的一条优化）

因为参数是**命名 uniform**，拖动滑杆时：

```
不重新编译 shader ──▶ 只调用 gl.uniform1f(loc, v) ──▶ 立刻重绘
```

只有**图结构变化**（加节点、改连线、切换质量档）才走"重新发射 + 重新编译"。这一条把 95% 的微调操作从"几十毫秒的编译"降到"零编译"。

WebGPU 那边同理：`uniform` buffer 用 `queue.writeBuffer()` 更新，pipeline 和 bind group 都不动。

### 6.3 三级预览策略

| 模式 | 分辨率 | 触发 | 目的 |
|---|---|---|---|
| **草稿** | 256×256，时间冻结 | 拖动滑杆 / 编辑节点时 | 结构感，进 60fps |
| **交互** | 512×512，动画运行 | 停手 150ms 后 | 看动态效果 |
| **成品** | 用户选定（可 4K），4 个时间点 | 点"生成预览" / 导出时 | 评分 + 分享 |

配 150ms 防抖 + `requestIdleCallback`：用户连续拖滑杆时只跑草稿档，停手后自动升档。

### 6.4 编译缓存

`emit()` 返回的 `hash` 是所有编译产物的 key：

```
cache[hash(source)] = { program: WebGLProgram, uniforms: Map<name, location>, log }
```

- 同一份源码永远只编译一次，即使来回切换版本。
- 版本树上任意两个节点可以瞬间 A/B（两个 program 都在缓存里）。
- 编译失败也缓存（key 里带上错误日志），避免对同一份坏代码反复触发编译。

### 6.5 微调的四种入口，一套管线

```
① 滑杆拖动    ─┐
② 对话框说话  ─┤─▶ ShaderPatch ─▶ applyPatch（原子）─▶ 新 ShaderDoc ─▶ 发射 ─▶ 预览
③ 参数优化器  ─┤                     │
④ 预设切换    ─┘                     └─▶ 版本树追加节点（hash 链）
```

四条路产出同一种对象，所以撤销/重做、日志、A/B、导出全都只用实现一次。

### 6.6 参数自动优化器

用户框选参考图的一块区域，点"自动匹配"：

```
目标：argmin_θ  L(render(θ), reference)
L = 1 − SSIM + λ₁·LPIPS + λ₂·|直方图差| + λ₃·|高频能量比 − 1|

搜索空间 θ = 所有 kind:'param' 的 uniform，各自 [min,max] 区间
算法：先坐标下降粗搜（每维 5 个采样点，贪心）
      → 再 CMA-ES 精搜（种群 16，≤ 40 代）
每次评估 = 改 uniform（零编译） + 渲染 256×256 + 算 SSIM
单次评估 ≈ 12ms  →  40 代 × 16 个体 ≈ 8 秒
```

Web Worker 里跑，不卡 UI。进度条实时显示当前最优的 SSIM 和参数值。**因为参数不触发重编译，这个搜索才可能在浏览器里 8 秒跑完** —— 这是 §3.4 那个设计决策的复利。

### 6.7 对话框微调

用户在对话里说"漩涡再快一点，颜色偏冷一些"，走的还是 S7 → Patch：

- LLM 看到的是：当前 `ShaderDoc` 的**紧凑摘要**（节点清单 + uniform 清单 + 当前值 + `spec.features`），不是 200 行代码。上下文小、便宜、快。
- 输出是 `set_param` / `expose_param` / 少量 `add_node`。
- 如果用户的要求隐含新结构（"加一层描边"），LLM 产出 `add_node` + `connect` 序列，仍然原子生效。

---

## 7. 项目结构

```
AI4Shader/
├── packages/ir/                    ← 核心，零依赖，可被任何宿主复用
│   ├── src/types.ts                IR 契约（所有类型的唯一出处）
│   ├── src/glslSubset.ts           受限 GLSL 子集规则 + 共享函数库
│   ├── src/rewrite.ts              跨语言重写引擎 + 保留字防护
│   ├── src/emitter.ts              发射核心：校验/拓扑/类型/uniform/展开
│   ├── src/catalog/nodes.ts        43 种节点定义
│   ├── src/patcher.ts              补丁应用（原子、不可变、可哈希）
│   ├── src/targets/                6 个目标发射器 + 各语言 lowering 契约
│   └── test/                       node:test 测试
├── apps/cli/                       ai4shader validate|emit|params|tune
├── apps/web/                       预览工作台（Vite + TS，无框架依赖）
│   ├── src/main.ts                 应用外壳与状态
│   ├── src/preview/glsl.ts         WebGL2 渲染器
│   ├── src/preview/wgsl.ts         WebGPU 渲染器
│   ├── src/panel/params.ts         滑杆面板（由 IR 自动生成）
│   ├── src/panel/codegen.ts        六格式源码视图
│   ├── src/compare/score.ts        SSIM / 直方图 / 频带评分
│   ├── src/optimize/search.ts      坐标下降 + CMA-ES
│   └── index.html
├── python/                         离线批处理与图像相似度
│   ├── ai4shader/validate.py       glslang/dxc/naga 调用封装
│   ├── ai4shader/score.py          LPIPS / SSIM / CLIP 评分
│   └── ai4shader/batch.py          批量生成与回归测试
├── platforms/
│   ├── unity/Editor/               Unity Editor 插件（编译验证 + 导入）
│   └── unreal/                     UE 插件（Custom 节点自动搭建）
└── docs/
    ├── DESIGN.md                   本文档
    └── IMPLEMENTATION.md           实现现状与已验证/未验证清单
```

---

## 8. 技术栈选型理由

| 选择 | 理由 | 放弃了什么 |
|---|---|---|
| Web 优先预览 | 唯一能"零安装、真实 GPU、秒级"三者兼得的宿主 | 需要浏览器；离线批处理靠 Python 补齐 |
| 无框架前端 | 预览工作台核心是 canvas + 滑杆，框架会带来构建复杂度而无收益 | 大规模 UI 开发体验 |
| TypeScript strict, ESM | IR 要被 CLI/浏览器/Node 同时消费 | — |
| `@webgpu/glslang` (WASM) | 浏览器内真编译 GLSL/HLSL，无需服务器 | ~4MB WASM，懒加载 |
| Python 做离线评分 | LPIPS/CLIP 的成熟实现都在 PyTorch | 引入第二种语言（但职责边界清晰：Python 只做"看图打分"和"批量驱动"） |
| IR 零依赖 | 能被 Unity C# 插件通过 JSON 直接消费 | 不能用 zod 等做运行时校验（自己写，反而更可控） |

---

## 9. 已知风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| **WGSL 比 GLSL 严格得多** | 重写后的 WGSL 可能编译不过 | WGSL 是最强校验信号：拿浏览器 WebGPU 当编译器，错误带行列号；行号映射回节点后让 LLM 定向修复。已在发射器里加结构性检查兜底。 |
| **目录覆盖不了复杂需求** | 用户想要的效果图里没有 | ① 组合节点（`domain_warp` + `fbm` 能表达很多）；② 超范围写入 `spec.outOfScope` 明确告知；③ `custom_glsl` 逃生舱（**本轮已实现**：受限子集代码挂在 `node.body` 上，经 `lintSubset` 校验后由同一个 token rewriter 降到六个目标，LLM 侧用 `set_body` 补丁写入 —— 见 `IMPLEMENTATION.md` §6.26/§6.27）。**实施后发现最缺的不是"更多效果节点"，而是类型系统的另一半**：连接规则只加宽，所以一个 `vec4` 生产者接不进任何 `vec2` 输入 —— 已用 `split_vec2`/`split_vec4`/`make_vec2`/`make_vec4` 补齐（`IMPLEMENTATION.md` §6.25）；目录现为 **43 种** |
| **LLM 编造不存在的函数/节点** | 图校验失败 | 这正是 IR 方案的价值：校验发生在静态层，错误是"没有 `foo` 这个输入"而不是一行晦涩的编译错误；且 `PN001` 类诊断自带合法选项清单 |
| **图像相似度 ≠ 用户满意** | 优化器过拟合到参考图 | 评分是**多信号**的（结构/频率/色彩/语义），且从不自动覆盖用户版本 —— 优化结果永远作为新分支出现在版本树里，由用户决定是否采纳 |
| **滑杆范围设置不当** | 工具"感觉坏了" | 每个节点的 `min/max/step` 是 catalog 里必填的一部分，且有测试保证默认值落在范围内 |
| **生成代码不可读** | 用户不信任工具 | 每个节点输出一个命名局部变量；导出时附 `README.md` 说明每个节点生成了哪几行；源码视图带节点高亮 |
| **UE/Unity 无法自动验证** | 用户拿到不能用的代码 | 明确降级标注 + 提供 Editor 插件做真实验证（路线图 P2） |

---

## 10. 路线图

### P0 — 可跑通原型（本次交付）

- [x] Shader IR 契约 + 31 节点目录
- [x] 跨语言重写引擎 + 受限 GLSL 子集 + 共享函数库
- [x] 6 目标确定性发射器
- [x] 补丁系统（原子、不可变、内容哈希、版本树基础）
- [x] CLI：`validate` / `emit` / `params` / `tune`
- [x] Web 工作台：WebGL2 + WebGPU 预览、自动滑杆面板、六格式源码视图、参数搜索评分（代码完成；本机无法启动浏览器，尚未实测）
- [x] 参考图对比评分（SSIM / 直方图 / 频带）
- [x] 演示文档与一个可直接跑的示例 shader

**验证标准**：`npm test` 全绿；`ai4shader emit demo.shader.json --target all` 产出 6 份文件；浏览器打开工作台能看到动画并拖动滑杆实时变化。

### P1 — 智能闭环（1–2 周）

- [ ] 接入真实 LLM（文本 → IR 起草、对话微调、Critic 归因）
- [ ] VLM 参考图理解（多模态 → `ShaderSpec.features`）
- [ ] S4–S7 自动修复循环与三级预算
- [ ] LPIPS / CLIP 评分（Python 服务）
- [ ] CMA-ES 参数优化器
- [ ] 版本树 UI + A/B 对比

### P2 — 平台落地（3–4 周）

- [ ] Unity Editor 插件：`-executeMethod` 无头编译验证、`.mat` 一键导入
- [ ] UE 插件：读取 `CustomNode.inputs.json` 自动搭建材质图
- [ ] MSL 本地编译验证（macOS CI）
- [ ] 视频参考输入（关键帧提取 + 时间轴对齐）
- [ ] Shader 变体管理（LOD / 质量档批量导出）

### P3 — 规模化（1–2 月）

- [ ] IR 片段检索库（按 `spec.category` + 图相似度取 few-shot）
- [x] `custom_glsl` 逃生舱：用户/LM 贴受限子集代码，lint 后进图并仍能多目标投影（**已完成**，见 `IMPLEMENTATION.md` §6.26/§6.27）
- [ ] golden 图回归测试（IR hash → 参考图，防跨平台漂移）
- [ ] 团队协作：IR 版本的 PR 流程、diff 评审
- [ ] 性能预算驱动的自动降级（按目标 GPU 帧时间反推 `qualityTier`）

---

## 11. 为什么这个架构能成立（总结）

四个设计决策互相支撑，缺一个都不成立：

1. **IR 做唯一真源** → 六种格式天然一致，微调只需改一处。
2. **LLM 出补丁不出文档** → 失败廉价、可回滚、可审查，且能同时服务滑杆和优化器。
3. **单一受限子集 + 重写引擎** → 内容写一次，六语言投影，"工具里好看、引擎里不一样"从架构上不可能发生。
4. **参数是命名 uniform** → 微调不触发重编译，于是滑杆能实时、浏览器里能跑 8 秒的参数搜索。

第 3 条带来一致性，第 4 条带来速度，第 1 条让前两条有意义，第 2 条让整个系统可以安全地被一个会犯错的模型驱动。

---

## 附：相关工具与参考

- [@webgpu/glslang](https://app.unpkg.com/@webgpu/glslang@0.0.15/files/README.md) — 浏览器内 GLSL/HLSL → SPIR-V 校验
- [wgslender](https://github.com/HugoDaniel/wgslender) — 无 WebGPU 上下文的 WGSL 校验/lint
- [naga](https://www.npmjs.com/package/@forgeax/engine-naga) — WGSL/SPIR-V 多后端着色器翻译器
- [DirectX Shader Compiler](https://chromium.googlesource.com/external/github.com/microsoft/DirectXShaderCompiler/) — HLSL → DXIL/SPIR-V
- [The Book of Shaders](https://thebookofshaders.com/) — 节点目录里 noise/fbm/domain-warp 的语义来源
- [Inigo Quilez: palette](https://iquilezles.org/articles/palettes/) — `palette()` 余弦配色
- [IQ: 2D distance functions](https://iquilezles.org/articles/distfunctions2d/) — `sd_*` 函数族
