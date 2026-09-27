# AI4Shader

把「我想要这个效果」变成**六种格式的可编译 shader**，并且能连续微调、秒级预览。

给一句话描述、一张参考图、或一段效果视频，系统产出一棵与语言无关的 **Shader IR**，再由确定性编译器把它投影成
**GLSL / WGSL / HLSL / MSL / ShaderLab / UE**，在浏览器里用真实 GPU 预览；拖个滑杆或在对话框里说「噪声再大一点」，
100ms 内看到新结果，且六种格式同时更新、语义始终一致。

```
- [当前状态](docs/IMPLEMENTATION.md)  ← 先读这份：做成了什么、验证到什么程度、哪里没做
- [设计方案](docs/DESIGN.md)          ← 架构、Agent 编排、为什么这样设计
- [接口契约](docs/AGENT_CONTRACT.md)  ← 文件归属、模块解析、冻结 API
- [节点目录](docs/CATALOG.md)         ← 43 种节点的语义与端口
- [目标后端契约](packages/ir/src/targets/README.md) ← 各语言的 lowering 规则与已有限制
```

---

## 核心想法

> **不要让 LLM 直接写 shader 代码，让 LLM 写「图」，让编译器写代码。**

系统被切成两半，中间用一份**有类型、可序列化、与目标语言无关的图**（Shader IR）隔开：

```
   模糊的、需要理解意图的那一半              确定的、需要正确性的那一半
 ┌───────────────────────────────┐      ┌─────────────────────────────────┐
 │ 文本/图片/视频  →  Shader IR   │      │  Shader IR → 6 种格式 → 真实像素 │
 │ LLM 负责（创造性、容错）        │      │ 编译器负责（确定性、可复现）      │
 └───────────────────────────────┘      └─────────────────────────────────┘
              ↑                                        ↓
              └──────── 参考图对比评分，归因到具体节点 ───┘
```

四个互相支撑的设计决策：

1. **IR 是唯一真源** → 六种格式天然一致；微调只改一处。
2. **LLM 出「补丁」不出文档** → 失败廉价、可原子回滚、可审查，且滑杆和参数优化器能共用同一条管线。
3. **单一受限 GLSL 子集 + 跨语言重写引擎** → 内容只写一次，六语言投影；"工具里好看、引擎里不一样"在架构上不可能发生。
4. **参数是命名 uniform** → 微调不触发重新编译，于是滑杆能实时、浏览器里能跑得动参数搜索。

---

## 快速开始

需要 **Node ≥ 22.6**（推荐 24+）。其余零依赖。

```bash
node scripts/setup-links.mjs      # 建立 workspace 链接（等价于 npm install 的那一步）
npm run check                     # 投影 + import 自检 + 全部测试
```

### 生成六种格式

```bash
npm run validate:demo             # 校验示例文档
npm run emit:demo                 # 输出 6 个目标 + 边车文件到 dist/demo/

# 同一个动作，换一个“表面着色”文档：读法线/世界坐标/视线/光向，六个目标各自换入口点
node apps/cli/dist/cli.js emit examples/toon-surface.shader.json --target all --out dist/toon --force
```

产物（`dist/demo/shader/`）：

| 文件 | 目标 |
|---|---|
| `shader.frag` | GLSL ES 3.00（WebGL2） |
| `shader.wgsl` | WGSL（WebGPU） |
| `shader.hlsl` + `hlsl-bindings.json` | HLSL（SM5+，含全屏三角顶点着色器） |
| `shader.metal` + `ai4shader.metal.json` | Metal Shading Language |
| `Shader.shader` + `Material.ai4shader.yaml` | Unity ShaderLab + 材质属性 |
| `ue/CustomNode.hlsl` + `ue/CustomNode.inputs.json` + `ue/MaterialGraph.ai4shader.md` | UE 材质 Custom 节点 + 引脚清单 + 搭建配方 |
| `params.json` / `README.md` | 全部可调参数 / 各目标限制说明 |

### 打开预览工作台

```bash
npm run serve        # http://127.0.0.1:5183/apps/web/index.html
```

工作台提供：真实 WebGL2/WebGPU 编译（驱动原始日志 + 编译耗时）、由 IR 自动生成的滑杆面板、
六格式源码视图、参考图对比评分（SSIM / 直方图 / 频带归因）、参数自动搜索。

预览有三种模式：**平面**（全屏四边形，看图案本身）、**材质**（把文档当材质贴到模型上，带光照与环境）、
**场景**（再加地面）。当文档读的是**表面输入**（法线 / 世界坐标 / 视线 / 光向）时，材质与场景走的是
**图着色**路径 —— 发射出的 surface 着色器直接在每个片元上求值，而不是把图案烘成贴图；此时面板会说明
哪几个控件属于贴图路径（它们会被禁用，而不是拖了没反应）。

**无缝平铺**由噪声节点自己的 `tile` 开关提供（默认关）：打开后格子的查表按周期取模，几何保持不折叠，
所以格子 `period` 的格点就是格子 0 的格点平移一个周期 —— 接缝处**精确相等**，而不是"两边看起来差不多"。
`liquid-metal` 就开着它，因为它的 `features` 里承诺了"无可见的平铺接缝"，而原始噪声格子做不到这一点：
`hash21(floor(p))` 依赖绝对格子，在预览的一张 uv 方格上永远看不出来，一平铺就露馅。

**需要记忆的效果**（擦开就不复原的水雾、拖尾、任何形式的累积）走的是**反馈**路径：文档声明
`render.feedback: true`，平面预览就用双缓冲乒乓把**上一帧**喂回图像通道，于是 `mix(上一帧, 当前, k)`
真的能累积。着色器源码一个字节都没变 —— 变的只是宿主往通道里绑什么 —— 所以这个声明必须是文档级的：
从图里猜会把每一个把通道当静态图用的老文档悄悄变成反馈回路。四个示例可以直接打开看：

| 示例 | 是什么 | 关键点 |
|---|---|---|
| `examples/liquid-metal.shader.json` | 液态金属 | 7 节点，四边形图案，9 个可调 uniform，**可无缝平铺** |
| `examples/toon-surface.shader.json` | 三渲二卡通着色 | 16 节点，读法线/光向，三层叠加（明暗分层 + 边缘光 + 高度渐变） |
| `examples/fog-wipe.shader.json` | 可擦的水雾玻璃 | 28 节点，**需要反馈**：把"已擦净程度"累积在 alpha 里，用 `max` 让它只增不减 |
| `examples/lotus-pond.shader.json` | 荷塘 | 43 节点 / 11 个 uniform / 五层（水面 + 倒影 + 睡莲叶 + 荷花 + 水雾），边缘用 `antialias` 稳定 |

> 想看的话点工具栏的 **打开 .shader.json**，选上面任意一份（文档由 `packages/ir/src/examples/*.ts`
> 生成，可用 `npm run build:demo` 重生成）。三渲二会自动切到材质预览；可擦水雾请在**平面**模式下
> 用鼠标拖动 —— 拖动是"擦"，因为它读 `iMouse`，那时候转视角是 Shift+拖动。

> 静态服务器**必须服务仓库根目录**，因为 `index.html` 的 import map 用的是绝对路径。
> 如果你装了 npm，也可以用 `npx vite` 走打包路径（端口 5183）。

### 用 CLI 做事

```bash
node apps/cli/dist/cli.js validate examples/liquid-metal.shader.json
node apps/cli/dist/cli.js emit <doc> --target glsl --stdout
node apps/cli/dist/cli.js params <doc> --set u_contrast=2.0     # 微调单个参数
node apps/cli/dist/cli.js tune   <doc> --param u_detailScale=1..24 --steps 5
```

### 让 Agent 自己写

```bash
# 一句话 → 六种格式的完整 bundle。用离线 stub provider：不需要 API key，不需要网络。
node apps/cli/dist/cli.js author "液态金属表面，慢速流动的拉丝高光" \
  --provider stub --doc-out out.shader.json --out dist/authored

# 改一个已有文档。配置在线 provider 后会实际调用模型修改，再校验生成代码。
node apps/cli/dist/cli.js refine out.shader.json "对比度再高一点" --out dist/refined

# 接真实模型（OpenAI 兼容）
export AI4SHADER_API_KEY=sk-...
node apps/cli/dist/cli.js author "冰洞里的蓝色晶体" --model gpt-4o-mini --out dist/cave
```

`author` 会打印完整的阶段轨迹（`interpret → draft → emit → repair…`），失败时给出诊断、`stopReason`
与 `remaining`，从不声称一个它没验证过的结果。

在线起草还会执行一次需求源码复核；浏览器在线模式把真实 WebGL2 片元编译错误送回修复流程。
“代码生成通过”不代表画面或物理行为已经验收。未实现和采用近似的要求会显示在作者面板。
`runAgent({ basedOn, provider, prompt })` 会执行修改请求；仅需确定性修复时使用 `mode: 'repair'`。
透明与跨帧反馈可通过 `set_render` 的 `transparent` / `feedback` 布尔字段设置。
多层 `custom_glsl` 节点各自有局部作用域，可以安全复用临时变量名。

雨窗这类复杂效果的实测与当前限制见 [在线雨窗回归记录](docs/RAIN_GLASS_TEST.md)。

**修复器会改文档，而不是删东西。** 一条类型不对的连线过去会被 `disconnect` 掉：那个端口有默认值，
所以文档依然合法、运行依然报成功，而用户刚接上的效果凭空消失。现在两个方向都改成**用节点重建值** ——
`vec4 → float` 插一个 `split_vec4` 取 `.x`，`vec2 → vec4` 插 `split_vec2` + `make_vec4` ——
仍然不消耗模型调用，只是修完的画面是用户想要的那个。

> **一个必须知道的坑**：任何直接调用 `runAgent` 的宿主都必须先 `import '@ai4shader/ir/targets'`。
> 六个发射器是自注册的，而 IR 核心刻意不依赖任何目标语言代码 —— 少了这一行会立刻以
> `AG002_TARGETS_NOT_REGISTERED` 停止（这是正确行为，但很意外）。CLI 已经处理好了。

---

## 仓库结构

```
packages/ir/                 核心，零依赖，可被任何宿主复用
  src/types.ts               Shader IR 契约 —— 所有类型的唯一出处
  src/glslSubset.ts          受限 GLSL 子集规则 + 共享函数库
  src/rewrite.ts             跨语言重写引擎（所有语言差异只许出现在这里）
  src/emitter.ts             发射核心：校验 / 拓扑 / 类型 / uniform / 展开
  src/catalog/nodes.ts       43 种节点定义
  src/patcher.ts             补丁应用：原子、不可变、内容哈希
  src/examples/              两份参考文档的 TS 真源（四边形 / surface），examples/*.json 由它们生成
  src/targets/               六个目标发射器 + 各语言 lowering 契约
packages/agent/              Agent 闭环：受预算状态机 + 确定性修复器 + 可插拔 LLM Provider
  src/policy.ts              预算、升级阶梯、诊断分类
  src/repair.ts              诊断码 → 具体补丁（不经过模型）
  src/context.ts             模型可见的上下文投影
  src/loop.ts                闭环本身
  src/provider/              stub（离线确定性）/ http（OpenAI 兼容）/ 解析 / 提示词
apps/cli/                    validate | emit | params | tune | author | refine
apps/web/                    预览工作台（无框架、零构建）
scripts/                     零构建管线与验证工具
examples/                    由 TS 真源生成的示例文档（永不手改）
docs/                        设计方案、实现现状、接口契约、节点目录
```

### 为什么是零构建

`src/*.ts` 是**唯一真源**（带类型，人能读，`tsc --noEmit` 能检查）；`scripts/project-ts.ts` 用
**Node 内置的 `stripTypeScriptTypes`** 把它投影成 `dist/*.js` —— Node 和浏览器加载的是同一份产物。
没有打包器、没有 200MB 依赖树，一个 checkout 加两条命令就能看到像素。

---

## 本机验证状态（重要）

以下为 2026-09-27 的本机复验结果；可用浏览器与真实 DeepSeek，npm 不在 PATH 时使用 `node scripts/run-tests.mjs`：

- **一条命令跑全部**：`npm test` → **613 个测试通过**
  （`packages/ir` 129 · `packages/agent` 121 · `apps/cli` 32 真 e2e · `apps/web` 89 · `python/` 242）。
  完整测试结束后自动运行文档计数门禁；也可执行 `npm run check:docs`。它读取本次实跑结果，
  核对 README 与 IMPLEMENTATION 的当前计数；源码变更后旧证据失效，历史快照不参与比较。
  Node 四套零跳过；Python 那套有 **1 个条件跳过**（只有 Pillow 真的产出隔行 PNG 才跑那条断言，本机没产出）。
  另有 `npm run build`（投影 **47** 个模块）、`npm run check:imports`（**92** 条 import 全部解析）、
  `tsc --noEmit -p apps/web/tsconfig.json` exit 0（TypeScript 装在 `apps/web/node_modules`，只覆盖 `apps/web/src`）。
- **GLSL 与 HLSL 都已被真实编译器编译通过，四个示例、三种形状**：
  四边形文档 `glslangValidator -S frag shader.frag` → **exit 0**，`dxc -T ps_6_0 -E PSMain shader.hlsl` → **exit 0，零 error 零 warning**；
  三渲二示例（`examples/toon-surface.shader.json`）另外编过了 `dxc -T vs_6_0 -E VSMain` ——
  表面文档发射的是**一对**入口点，只编片元那半等于没验证插值契约。可擦水雾（`fog-wipe`）与荷塘（`lotus-pond`，39 节点五层）
  同样两个入口各编一次。**编译器抓到过真东西**：`vec4(0.0)` 这种标量 splat 在 GLSL/WGSL/MSL 合法、在 HLSL 是硬错误，
  它当时正躺在一个已发布的节点体里，通过了全部结构性检查 —— 现在它是 `lintSubset` 的 `RW013`，不再需要编译器来发现。
  这是整个项目最重要的一条验证 —— 结构性检查只能证明输出**长得像**目标语言，只有编译器能证明它**就是**。
  两个编译器都是后来才找到的：`glslangValidator` 藏在 Android SDK 的 Vulkan 工具里（不在 PATH 上），
  而 dxc 此前的"崩溃"其实是**编码问题**（本机代码页 936，它无法解码没有 BOM 的非 ASCII 源文件）——
  加上 BOM 后立刻给出了真实诊断，并暴露了一个 `atan2` 缺陷（HLSL 的 `atan2` 只有标量重载）。
  `python/tests/test_emitted_artifacts.py` 把这条链路固化成常驻测试，含一个反例：
  故意写错的 shader 必须被拒绝，否则说明校验器根本没在工作。
- **surface 文档在 3D 预览里跑的是发射出来的着色器本身**。`needs.surface` 为真时，`emit()` 的 GLSL
  源码被**逐字节**交给驱动，配一个写入 `vWorldPos`/`vNormal`/`vUv` 的网格顶点阶段，图在每个片元上真求值 ——
  所以 `dot(iNormal, iLightDir)` 这样的项在预览里真实存在，卡通明暗分界会随光转动。四边形文档仍走
  "渲染成贴图再贴到网格上"的老路。发射器与预览之间的宿主契约（源码逐字节相同、插值变量逐名相同、
  uniform 按驱动反射写入、滑杆不触发重编译、编译失败退回贴图路径并显示驱动原话）由 7 条单测钉住。
- **工作台自己的着色器现在也被真实编译器编过**。3D pass 的材质 / 天空 / 顶点阶段是手写 GLSL，
  之前只被"录制型 GL"看过（它记录调用，不编译着色器）。`python/tests/test_preview_shaders.py`
  从 `apps/web/dist/preview/surface.js`（浏览器真正加载的那份）里取出五个阶段，逐个交给
  `glslangValidator -S vert|frag`，全部 exit 0。**为什么值得单独一条**：这些着色器写在
  TypeScript 的模板字符串里，一个反引号就能提前结束字面量（本轮踩了两次，报错指向别处）。
- **贴图映射的正确性是被量出来的，不是被描述出来的**。图案贴到网格上时用的是**世界坐标**，
  而物体以原点为中心 —— 于是有一半坐标是负的。此前采样器是 `CLAMP_TO_EDGE` 且没有偏移，
  负坐标不报错、不回绕，而是返回**边缘像素**并被拉伸铺满整片区域：图案只出现在坐标恰好落进
  单位正方形的部分（立方体上是 `(+,+,+)` 那个角与相邻三个面的正半边；地面上是 `x∈[0,1]` 与
  `z∈[0,1]` 两条一单位宽的带子 —— 也就是用户报的"只有一个十字有图案"）。
  `apps/web/test/pattern-mapping.test.ts` 从**着色器源码本身**解析出投影与偏移、从**录制到的 GL 状态**
  读出 wrap 模式，再对每个形状的每个顶点算一遍，断言"没有任何顶点被交给边缘钳制"。
  修复前的实测比例：球 71% · 环面 74% · 纽结 75% · 立方体 73% · 地面 98%。
- **未实测**：WGSL / MSL / ShaderLab / UE 在各自真实编译器上的编译（本机无 `naga`/`tint`、非 macOS、无 Unity/UE），
  以及浏览器工作台里**第一帧像素**（无法启动浏览器；DOM 与渲染循环已在 `apps/web/test/boot.test.ts` 里跑通）。
  UE 那边能查的部分已经查了：引脚清单被断言必须带四个表面输入且每个 uniform 都有对应引脚 —— 这是"搭建配方完整"，
  不是"UE 能编译"。逐条原因、风险等级与补齐命令见 **[docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md)** 第 4、5 节。
- **WGSL 的正确性**：`targets/wgslLower.ts` 降级 pass（函数签名、不可变参数、`var` 声明、循环头、
  `textureSample` 三参数）通过了结构性检查，但**"合法"这句话仍然没有编译器背书** ——
  它从"已确认不合法"变成了"未验证"，这两者必须分清。
- 所有格式的**离线评分**与**编译器校验**都在 `python/`：SSIM 为 numpy 实现，
  **LPIPS 与 CLIP 用的是本机已缓存的真实权重**（`capabilities()` 会如实报告哪些真跑了、哪些降级了）。
- 生成文件的**顶部注释里逐条写明了该目标的不确定项** —— 不确定性放在产物里，而不是藏在一个没人读的 README 里。

如果你在有网、有浏览器、有编译器的机器上：`npm run verify` 会发射**两份**真实文档（四边形 + 三渲二 surface），
在无头浏览器里各自编译并绘制，把驱动的原始日志、uniform 绑定数、亮度统计和渲染帧一起产出。
**"编译通过但画面全平"会被判定为失败** —— 能编译不等于画对了。surface 那一页的顶点阶段是按发射出的片元着色器
自己的 `in` 声明合成的，所以它证明的是"片元阶段在真实驱动上能编译并出像素"，而不是"网格满足插值契约"
（后者由 `apps/web/test/surface-graph.test.ts` 逐名比对两个阶段来钉住）。

---

## 许可

MIT

材质预览默认开启 `seamless` 接缝融合：UV 4 次、三平面 12 次采样（关闭为 1/3 次），大尺度特征可能重影。可关闭对比，导出源码不含此预览处理。透明产物写明宿主混合要求，ShaderLab 直接生成透明渲染状态。在线 HTTP 默认使用 SSE，保留 JSON 回退；完整契约见 [AGENT_CONTRACT](docs/AGENT_CONTRACT.md)，实测边界见 [RAIN_GLASS_TEST](docs/RAIN_GLASS_TEST.md)。
