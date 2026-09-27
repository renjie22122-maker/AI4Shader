# AI4Shader — 实现现状
> `docs/DESIGN.md` 说明**应该**怎么做；这份文档说明**实际做成了什么、验证到什么程度、哪里还没做**。
>
> 最后更新：本次交付。所有结论都来自本机实际运行，不是设计意图的复述。

---

## 1. 一句话结论

IR、六目标代码生成、CLI、零构建浏览器工作台、补丁系统全部打通并有测试覆盖；**GLSL 输出经过结构化校验、并作为唯一"设计上就要直接运行"的目标**。但**这台机器上无法运行任何 shader 编译器，也无法启动浏览器**，所以六种格式的真实编译验证没有完成 —— 第 4 节逐条说明，不含糊。

---

## 2. 本机环境的硬约束（决定了下面所有取舍）

| 约束 | 实测结果 | 对交付的影响 |
|---|---|---|
| **没有 npm / pnpm** | `npm`/`pnpm` 不在 PATH；`corepack enable` 被沙箱拒绝（EPERM） | 不能 `npm install`，不能用 Vite / tsc / vitest / 任何依赖 |
| **没有网络** | `registry.npmjs.org` 探测失败 | 不能装任何包，不能用 `@webgpu/glslang` WASM |
| **不能启动浏览器** | Chrome/Edge 存在，但启动即 `FATAL: platform_channel.cc: Check failed: Access is denied` | 无法做真实 WebGL2/WebGPU 编译与截图验证 |
| **有 `dxc.exe`**（Windows SDK 10.0.22621） | 修正：先前的"崩溃"结论是错的 —— `0x80070459` 是**编码失败**（本机代码页 936，dxc 无法解码无 BOM 的非 ASCII 源），不是编译器问题。加 BOM 后正常给出诊断，并暴露了一个真实的 `atan2` 缺陷（已修） | **HLSL 现已完整编译通过**（exit 0，零 error 零 warning）。另有一个坑：宿主代码页为 936 时，任何非 ASCII 源都必须带 BOM |
| **有 `glslangValidator`**（Android SDK 的 Vulkan 工具，不在 PATH 上） | `glslangValidator -S frag` 对生成的 GLSL 返回 exit 0 | **GLSL 完整编译通过**。发现方式：`python/ai4shader/validate.py` 会搜索 Android SDK 根目录 |
| **有 Node v24.21.0** | 自带 `stripTypeScriptTypes`、`node --test` | 构成了整套零构建管线的基础 |

这些约束不是借口，是本机事实。**设计上它们都不成立**：正常环境里 `npm install` + 浏览器 + WASM 编译器即可补齐第 4 节的未验证项。第 5 节给出具体补齐方式。

---

## 3. 实际交付的内容

### 3.1 核心：`packages/ir`

| 文件 | 内容 | 状态 |
|---|---|---|
| `src/types.ts` | Shader IR 契约（`ShaderDoc`/`ShaderGraph`/`ParamValue`/`Diagnostic`/`ShaderPatch`…） | YES 冻结 |
| `src/glslSubset.ts` | 受限 GLSL 子集规则 + 共享函数库（hash/noise/fbm/voronoi/SDF/color/tonemap） | YES |
| `src/rewrite.ts` | 跨语言 token 重写器 + 各目标重命名表 + WGSL/HLSL/MSL 标量垫片 | YES |
| `src/emitter.ts` | 图校验、确定性拓扑排序、类型检查、uniform 收集、占位符展开、多输出解构 | YES 已修 3 个 bug，见 §6 |
| `src/catalog/nodes.ts` | **31** 种节点（不是 33 —— 原设计文档的计数有误，已更正） | YES |
| `src/patcher.ts` | 原子补丁应用、不可变、内容哈希、版本分叉 | YES |
| `src/targets/_shared.ts` | 六个目标发射器 | YES 已修关键 bug，见 §6 |
| `src/examples/liquidMetal.ts` | 手写参考示例（7 节点 / 9 个可调 uniform） | YES |

### 3.2 零构建管线（本环境的产物，但值得保留）

| 文件 | 作用 |
|---|---|
| `scripts/project-ts.ts` | 用 **Node 内置 `stripTypeScriptTypes`** 把 `src/*.ts` 投影成 `dist/*.js`。`src` 是唯一真源，`dist` 是产物，永不可手改。 |
| `scripts/serve.mjs` | 无依赖静态服务器（默认 5183）。**必须服务仓库根目录**，因为 `index.html` 的 import map 用的是绝对路径。 |
| `scripts/check-imports.mjs` | 解析 `dist/` 里每一条相对 import，把"改错路径 → 白屏"变成构建期错误。 |
| `scripts/verify-shaders.mjs` | 生成真实 GPU 验证页（见 §4.2 的诚实说明）。 |

**为什么这个管线值得留：** 它把"工具链"从 200MB 的依赖树压缩成一个 40 行的脚本，而且 `dist/*.js` 同时能被 Node 和浏览器加载 —— 同一份产物，两种宿主。这是真实收益，不只是环境妥协。

### 3.3 应用

| 路径 | 内容 | 状态 |
|---|---|---|
| `apps/cli` | `validate` / `emit` / `params` / `tune` | YES 实测可用 |
| `apps/web` | WebGL2 + WebGPU 预览、自动滑杆面板、六格式源码视图、SSIM/直方图/频带评分、参数搜索 | PARTIAL 代码完成、**未在浏览器中运行过**（§4.2） |
| `packages/agent` | Agent 闭环：意图 → IR → 校验 → 发射 → 诊断 → 补丁修复的受预算状态机；可插拔 LLM Provider；离线确定性 stub | YES 闭环 + provider 层本机实测（89/89 测试）；真实 LLM 调用未发生（§4.2） |

### 3.4 Agent 闭环（本机实测）

一句话中文需求 → 六种格式，**零 API key、零网络、零浏览器**：

```
provider: stub | vision: false

stop=clean clean=true modelCalls=2 remaining=[none]
trace: interpret#0 -> draft#0 -> emit#0

emissions:
  glsl         9800 bytes  errors=0
  wgsl        11926 bytes  errors=0
  hlsl        10929 bytes  errors=0
  msl         10549 bytes  errors=0
  shaderlab   12302 bytes  errors=0
  ue          11618 bytes  errors=0

nodes: n_uv:uv, n_warp:transform, n_detail:fbm, n_ramp:color_ramp, n_grade:grade, n_out:output
uniforms: u_uvScale, u_gain, u_octaves, u_ridged, u_detailScale, u_contrast, u_exposure, u_saturation
```

**关键设计：LLM 只负责两件窄事**（`interpret` 文本→Spec、`propose` 上下文→补丁），其余全是确定性、
可目录感知的代码。三个理由按重要性排序：① **没有模型也能测试**——网络调用会让闭环不可测，而三行 stub
就让每条分支都能确定性覆盖；② **类型错误不需要语言模型**——`IR006` 已经指名了哪个端口错了，查表是
零成本且不会改坏别的东西；③ **模型无法损坏文档**——它出的是补丁，原子且被校验，最坏情况是被拒绝。

**升级阶梯**（`policy.ts`）：`mechanical → model-repair → simplify → redraft → give-up`。
超预算时换的是**策略**而不是重试同一个——"同样的失败重试一次"是这类 agent 最常见的浪费，
所以代码让它结构上难以发生。卡住由 plateau 计数独立检测，与剩余预算无关。

**实测的几条性质**（`packages/agent/test/loop.test.ts`，23 个测试全绿）：

| 性质 | 证据 |
|---|---|
| 干净起草只需 2 次模型调用 | interpret + propose，之后机械修复零调用 |
| **repair 诊断模式可零模型调用** | 文档里有个 vec2→vec4 错接，闭环自己摘掉那根线并达到 clean，`usage.calls === 0`，provider 一次都没被问 |
| 被拒的补丁不会被误判为"图不可救" | `applyPatch` 原子拒绝后，`PT001_UNKNOWN_NODE @ghost` 被种进第一轮修复上下文，模型看到的是"你引用了不存在的节点"而不是"没有输出节点" |
| 机械修复能处理的不占用模型轮次 | 若计划里没有机械动作就立即升级，不空转一轮（空转还会虚增 plateau 计数） |
| 失败一定给出诚实结论 | 永远返回一份文档 + `remaining` + `stopReason` + 完整 `trace`；`clean` 只在**所有**目标与已配置的宿主校验零 error 时为真，不表示视觉验收通过 |
| 确定性 | 同一 provider 两次运行得到相同 hash 与相同 trace 形状 |

**本机没做的**：真实 LLM 调用（无网络/无 key）。`HttpProvider`（OpenAI 兼容）代码已写但**从未发出过
一个 HTTP 请求**——它的测试用注入的假 `fetch`，所以测的是解析、重试策略与错误透传，不是与真实 API
的兼容性。**这条限制在 §6.49 那轮变得更重要而不是更轻**：`HTTP 200 + 响应体读不出来` 的**根因**
（60 秒上限落在大补丁生成的中途）是从用户的现象与代码路径推出来的，本机无法复现一个"慢到超过上限"
的真实端点；被测试钉住的是**分类与建议**（读体失败要重试、要报 `read`/`timeout`、不许让读者去查密钥），
以及"上限触发"这条分支本身（假 `fetch` 在信号 abort 时让读体 promise 拒绝，与真实栈同形）。
所以：**"下次遇到同类失败时给出的诊断是对的" 已验证，"这个失败不再发生" 未验证** —— 后者取决于
真实链路的延迟，只能用 `timeoutMs` 与重试去覆盖。

**CLI 已接通**（`author` / `refine` 两个子命令）。实测六条命令全部 exit 0：

| 命令 | 结果 |
|---|---|
| `validate` | 0 诊断 PASS |
| `emit --target all` | 6 目标 13 文件 |
| `params --set u_contrast=2.2` | hash 正确更新 |
| `tune --param … --steps 3 --emit` | 3 个变体 |
| `author "…" --provider stub --doc-out … --out …` | `stop=clean modelCalls=2`，写出文档 + 六格式 bundle |
| `refine <doc> "…" --provider stub --out …` | **`modelCalls=0`** |

**一个必须知道的陷阱**：任何直接调用 `runAgent` 的宿主都必须先 `import '@ai4shader/ir/targets'`——
六个发射器是自注册的，而 IR 核心刻意不依赖任何目标语言代码。少了这一行会立刻以
`AG002_TARGETS_NOT_REGISTERED` 停止（正确但意外），错误消息里已指名修法，`index.ts` 的用法示例
顶部也标了 ⚠。CLI 已经处理好了。

### 3.5 Web 工作台内的 Agent（本机实测）

工作台左侧新增「让 Agent 写」面板：描述框 + provider 切换（**离线 stub 优先**）+ 生成/微调按钮，
运行时显示实时阶段轨迹，失败时列出每条诊断及其 `fix`，并把**模型调用次数**作为标题数字——
干净文档上的 `refine` 是 **0**，把这个数字显示出来，用户才会学会信任确定性路径。

面板走的是工作台既有的 `setDoc()`，所以 agent 产出的文档和内置示例经过**完全相同**的
重新发射/重新编译/滑杆重建/代码视图路径。没有第二条"来自模型的文档"管线。

`apps/web/src/agent.ts` 是与 `ir.ts` 同样的单次转型门面；实测（用工作台自己的 facade 模块）：

```
catalog kinds visible to the panel : 31
providerReady(http) without key    : false  (按钮禁用，不会尝试未签名的请求)
stage callbacks                    : interpret#0 -> draft#0 -> emit#0
emissions                          : glsl, hlsl, msl, shaderlab, ue, wgsl   全部零 error
refine modelCalls                  : 0
```

**未验证**：这些代码从未在浏览器里跑过（本机无法启动浏览器，§4.2）。面板的每一次 DOM 操作、
`onStage` 回调的渲染、以及 `setBusy` 对状态栏的影响，都只经过 Node 侧的模块加载与逻辑调用验证。

### 3.6 多输出端口（本机实测）

`emitBodies` 的多输出解构此前对**任何** schema 都无效，所以目录里每个节点只有 1 个输出端口。
修复后（§6.7）重新加回了丢失的端口，并实测确认**每个端口都取到正确的分量**：

```
voronoi  in[uv,scale,randomness] -> out[f1,f2,edges,cell]
time     in[speed]               -> out[time,sin,cos]
polar    in[uv,center]           -> out[angle,radius]
sdf      in[uv,shape,size,radius,smoothness] -> out[distance,mask]
```

实测发射（`voronoi` 第 3、4 个端口，`time` 的第 2、3 个，`sdf` 的两个，全部可达且零 error）：

```glsl
float L_p_v_edges    = tmp_p_v.z;
float L_p_v_cell     = tmp_p_v.w;
float L_p_t_sin      = tmp_p_t.y;
float L_p_s_mask     = tmp_p_s.y;
```

`uv` 与 `tile` **仍然只有一个输出端口**，这是发射器契约的直接后果而非遗漏：它们的两个输出都是
`vec2`，而契约要求多输出节点返回一个 `vec4` 且每个端口只能是标量。把 `vec2` 端口合法化需要
重新设计那个契约（例如允许返回 `mat2x2`），不是加个端口就能解决的。`catalog.test.ts` 里有一条
测试钉住这个事实，所以将来有人改动契约时会被迫注意到它们，而不是让它们被悄悄留在原地。

---

## 4. 验证状态（逐项诚实说明）

图例：YES 本机实测通过 ｜ PARTIAL 部分验证 ｜ NO 本机无法验证

### 4.1 YES 已实测通过

| 项目 | 证据 |
|---|---|
| 单元测试 | **613 个全绿**：`packages/ir` 129 · `packages/agent` 121 · `apps/cli` 32 · `apps/web` 89 · `python/` 242。Node 四套**零跳过**；Python 一套有 **1 个条件跳过** —— `test_imageio.py` 只在 Pillow 真的产出隔行 PNG 时才断言隔行结构，本机没产出。这是环境能力探测，不是被掩盖的失败 |
| 浏览器真实编译回流 | 在线模式注入 `validate(doc, emissions)`，GLSL 驱动错误 `AG030_GLSL_COMPILE` 携带日志/源码回到修复轮；每次修改重新验证。仅片元编译，非链接、视觉或其它引擎验收；WebGL2 不可用时不注入此钩子。 |
| **通用表达力：43 种节点** | 除表面分组外，本轮补齐了通用 shader 反复需要而目录里缺失的原语：`split_vec2` / `split_vec4`（把向量拆成标量）、`make_vec2` / `make_vec4`（用标量拼出向量）、`vec_math`（向量版 `math`：normalize / length / distance / reflect / fract / floor / abs / pow / max / min / sign / clamp / cross）。**这条修掉了一个硬伤**：`mouse` 输出的是孤零零一个 `vec4`，而连接规则只允许加宽，所以它**进不了任何空间输入** —— "在指针位置画东西"不是难，是不可能，而且那条线会被 repairer 当作类型不匹配**静默删掉**。现在 `mouse → split_vec4 → ÷resolution → make_vec2 → noise.uv` 这条链六个目标全部零 error 发射，GLSL 与 HLSL 均被真实编译器编译通过（exit 0、零 warning） |
| **表面着色（三渲二）全目标可用** | IR 从 31 种节点扩到 **36 种**（新增 `surface` 组：`normal` / `world_pos` / `view_dir` / `light_dir` / `dot`；向量原语组随后把它推到 42，见上一行），**六个目标全部实现表面入口点**：GLSL 用插值 `in` 变量，HLSL/WGSL/MSL 额外发射网格顶点着色器与变换块，ShaderLab 用 Unity 的 `UnityObjectToWorldNormal`/`_WorldSpaceLightPos0`，UE 用 Custom 节点的四个输入引脚。标志位按**可达节点**计算，删掉最后一个 `normal` 会逐字节回到四边形形态 |
| **GLSL / HLSL 表面 shader 被真实编译器验证** | 四边形文档：`glslangValidator -S frag` → **exit 0**；`dxc -T ps_6_0 -E PSMain` → **exit 0，零 error 零 warning**。**surface 文档（三渲二示例）两阶段都编过**：`glslangValidator -S frag` → exit 0；`dxc -T vs_6_0 -E VSMain` 与 `-T ps_6_0 -E PSMain` → **双双 exit 0、零 warning**（`dxc` 的 `implicit truncation` 是语义警告，断言到 0）。还有一条更早的验证：预览的网格顶点阶段与发射出的 surface 片元阶段用 `glslangValidator -l` 真的**链接**过，exit 0。**但链接通过并不证明插值契约成立** —— 把 `vNormal` 一致改名后链接器照样 exit 0，见 §6.30；WGSL 走 `checkWgslStructural`（本机无 naga/tint）；MSL / ShaderLab / UE 本机无编译器，只能人工阅读 —— 这三项的编译**仍未验证**，UE 的四个表面引脚改为结构性断言（见 §4.2） |
| 测试门禁 | `npm test` **一条命令跑全部五套**（Node 四套 + Python 一套）exit 0；外加 `npm run build`（投影 **47** 个模块）、`npm run check:imports`（按**浏览器**的解析规则）、`npm run check:served`（自己起服务器，沿 HTTP 走真实模块图：全部 200 + JavaScript MIME type + import map 生效） |
| 文档数字门禁 | 完整 `run-tests.mjs` 从实跑输出记录各套通过数及源码指纹，随后运行 `check-docs.mjs` 对照 README 与 §4.1 的总数、分套数、模块数、导入数和相对/裸导入明细。不扫描 test() 文本猜数量；源码变化使旧证据失效，缺 Python/局部运行不更新证据。`--self-test` 用错误计数检验拦截。历史实测快照保留。 |
| **工作台真的能启动** | `apps/web/test/boot.test.ts`：装一个最小 DOM，import 真实的 `apps/web/dist/main.js`，断言 `boot()` 跑到底 —— 每个挂载点被填上、Agent 面板可用、无 WebGL2 时挂 banner 而非抛异常、渲染循环转一帧后仍在排队、**点「生成」跑完 Agent 闭环并落到六目标文档**、代码视图六个 tab。**在这条之前，「工作台是白的」与「工作台是好的」在所有自动化检查眼里是同一个结果** |
| **3D 材质/场景预览** | 两条材质路径，由 `Emission.needs.surface` 决定，不是启发式：**贴图路径**（图案由后端画进 512² 离屏画布，再作为纹理贴到受光网格上）用于四边形文档；**图着色路径**用于 surface 文档 —— `emit()` 产出的 GLSL surface 源码**逐字节**交给驱动，配上写入 `vWorldPos`/`vNormal`/`vUv` 的网格顶点阶段，图在**每个片元**上真求值。于是 `dot(iNormal, iLightDir)` 这类项在预览里真实存在，卡通明暗分界会随光照转动，而不是一条固定的色带。宿主 uniform（`iCameraPos`/`iLightDir`/`iResolution`/`iTime`/`iMouse`/`iChannel0` 与文档自己的滑杆）按**驱动反射出的类型**写入；图真的读 `iMouse` 时（由反射说了算，不是再扫一遍图）普通拖动改为操作指针、Shift/右键拖动才是转相机；编译失败则退回贴图路径并把驱动原话显示出来，且**不留旧版本**。几何与相机是纯函数并被单测：绕序、near/far 映射、`lookAt` 基正交、`multiply` 顺序、`rotationY` 行列式 = +1 —— **这些错误都不抛异常，只会渲染出一张看着像那么回事的图**，而本环境没有浏览器可看 |
| **surface 宿主契约被 7 条单测钉住** | `apps/web/test/surface-graph.test.ts` 用一个**录制型 GL**（未实现的方法一律抛错，所以新调用不会被忽略）验证的是发射器与预览之间的宿主契约：①交给驱动的片元源码与 `emit()` 输出**逐字节相同**；②顶点阶段 `out` 的集合与片元阶段 `in` 的集合**逐名逐类型相同**（把 `vWorldPos` 改名后只有这条测试变红 —— `glslangValidator -l` 会照常 exit 0，见 §6.30）；③文档声明的每个 uniform 都按反射类型写入，且 `iLightDir` 真的是单位向量（这条当场抓到 §6.29）；④滑杆只改值、**不新建也不链接** program；⑤被驱动拒绝时退回贴图路径、报告驱动原话、且旧版本不残留；⑥四边形文档永远不会被送进图着色路径 |
| **贴图映射契约被量出来（2 条单测）** | `apps/web/test/pattern-mapping.test.ts` 不靠肉眼看像素，也不靠复制一遍算术：它从**材质着色器自己的源码**里解析出三平面投影用的轴与偏移，从**录制到的 GL 状态**里读出图案纹理的 wrap 模式，再对每个形状（球/环面/纽结/立方体/地面）的**每个顶点**算一遍采样坐标，断言**没有任何顶点被交给采样器的边缘钳制**。修复前的实测比例（被钳制的顶点占比）：球 71% · 环面 74% · 纽结 75% · 立方体 73% · 地面 98% —— 也就是"图案只出现在物体的一角"和"地面只有一个十字"（§6.48）。**这条测试刻意做成能测"错的映射"**：偏移量缺失时默认为 0 而不是拒绝运行，所以它能给旧代码给出上面那组数字，而不是在一开始就报"解析失败" |
| **预览自己的五个手写着色器被真实编译器编过** | `python/tests/test_preview_shaders.py`：材质片元、天空顶点/片元、图案顶点、图着色顶点 —— 从 `apps/web/dist/preview/surface.js`（浏览器真正加载的投影产物）里取出，逐个交给 `glslangValidator -S vert|frag`，全部 **exit 0**。此前这五个阶段只被录制型 GL "编译"过（它按标记字符串返回 `COMPILE_STATUS`）—— 对宿主契约是对的，对 GLSL 等于没查。**为什么值得单列**：这些着色器写在 TypeScript 模板字符串里，一个反引号就能提前结束字面量，而报错会指向别处（本轮踩了两次，见 §6 末尾的工具事故） |
| **三渲二示例：六个目标 + 双阶段真实编译** | `examples/toon-surface.shader.json`（16 节点 / 7 个 uniform / `needs.surface = true`）由 `packages/ir/src/examples/toonSurface.ts` 经 `npm run build:demo` 生成，三层叠加（明暗分层 + 边缘光 + 世界高度渐变）。六个目标发射**零 error 零 warning**；GLSL 被 `glslangValidator -S frag` 编译通过；HLSL 的 **`VSMain` 与 `PSMain` 各按自己的 profile** 被 `dxc` 编译通过且**零 warning**。这三条现在是 `python/tests/test_emitted_artifacts.py` 里的自动化断言（含"两个入口都要找到、profile 要跟着入口走"），不再是手工记录 |
| **打开 `.shader.json`** | 工作台此前只能生成文档或看内置示例，**打不开文件** —— 而它的产物就是文档。新增「打开 .shader.json」按钮：读入、按 `irVersion`/`graph.nodes` 预检（不合格的文件被**点名**拒绝，而不是让 `TypeError: cannot read properties of undefined` 从发射器深处冒出来）、然后走与生成文档完全相同的 `setDoc` 路径 |
| **反馈：让无状态着色器有记忆（`render.feedback`）** | 「手指抹开水雾玻璃」在无状态着色器里根本做不到 —— 用 `iMouse` 拼出来的只是一个跟着手指走的洞，手一挪就复原。文档现在可以声明 `render.feedback: true`，宿主把**上一帧**喂回图像通道，于是 `mix(上一帧, 当前, k)` 才真的能累积。**着色器源码一个字节都没变**（采样通道的还是那个 `texture` 节点），变的是宿主绑定什么 —— 这就是为什么它必须是一个**文档级**标志而不是从图里猜：采样静态图和采样自己上一帧在图上完全同形。它不能被猜出来，猜错的后果是每个既有通道用户悄悄变成反馈回路 |
| **反馈的工作台实现（平面预览）** | WebGL2 后端做**双缓冲乒乓** + 离屏 FBO：图渲染到 B 的同时把 A 作为 `iChannel0` 采样，然后 blit 到画布并交换。这不是优化而是唯一合法的写法 —— 采样正在渲染的纹理在 GL 里是未定义行为（驱动可能给垃圾、给黑、或静默丢帧）。6 条单测用录制型 GL 钉住：**采样的纹理不能是正在渲染的那张**、两个缓冲严格交替、resize 会按新尺寸重新分配、关掉标志会释放缓冲且不分配、重新编译会清空累积（否则用户改完文档会看到上一个着色器的残影慢慢淡出）。WebGPU 后端**不实现**，并在编译结果里说明；3D 材质路径同样只提供白色占位 —— 移动网格的「上一帧」是屏幕空间累积，是另一个功能，面板照实说 |
| **四个参考示例，三种形状** | `liquid-metal`（7 节点，四边形图案）· `toon-surface`（16 节点，surface 着色，读法线/光向）· `fog-wipe`（28 节点，**需要反馈**：把已擦净程度累积在 alpha 里，用 `max` 让它只增不减）· `lotus-pond`（**43 节点 / 11 个 uniform / 五层**：水面 + 倒影 + 睡莲叶 + 荷花 + 水雾，边缘用 `antialias` 而不是固定宽度的 `smoothstep`）。全部由 `packages/ir/src/examples/*.ts` 经 `npm run build:demo` 生成，六个目标零 error 零 warning，并且都进了 Python 侧的**真实编译器**断言（GLSL 过 `glslangValidator`，HLSL 两个入口各过 `dxc`）。荷塘那份是**表达能力**的探针：它没有用任何新原语，只是把目录里的东西叠起来 —— 如果目录不再能组合，它会先失败，而且失败会指向具体节点 |
| **能力矩阵：8 种功能 × 6 个目标** | `packages/ir/test/matrix.test.ts` 把目标里那句"任何需求"写成了一张表：每个功能一份小而真实的文档，乘以六个目标，断言**零 error 且零 warning**。八行是 surface / channel / feedback / **feedback+surface** / derivatives / escape-hatch / quality-scaled / tileable。其中 `feedback+surface` 那一行值得单独说：它是 IR 里合法、却**没有任何宿主能显示**的组合（平面路径跑不了 surface 着色器，3D 路径不实现反馈），所以它必须能发射 —— 导出到 Unity/UE 才是用户拿走的东西 —— 而预览的限制是预览的限制，工作台照实说。另外四条测试分别钉住：surface 行真的换了入口点（其余行没有）、channel 行真的声明了通道、逃生舱在每个目标上都被**降级**（GLSL 保留 `mod(`、HLSL 必须出现 `wf_mod(` 且不能有 `mod_`）、以及 `REWRITE_ALIASES` 里**每一个别名**都在六个目标上被解析掉 —— 导数那一族最尖：四个目标四种拼写（`dFdx` / `dpdx` / `ddx` / `dfdx`），"解析掉了"和"解析成了对的名字"分别断言。这张表是发现 §6.43 的路径：写第 8 行时随手用了 `rot2(a) * v`，于是去问编译器 |
| **逃生舱里的宿主符号也算能力** | 一段 `custom_glsl` 片段可以直接读宿主符号，包括只有表面入口点才声明的那五个。读了它们，文档就自动是**表面文档**（`needs.surface` 现在同时由 schema 与 body 决定，见 §6.46）—— 于是六个目标都会给出插值顶点输出与相机 uniform，片段里的 `dot(normalize(iNormal), normalize(iLightDir))` 真的成立。这条把逃生舱的表达力又放宽一格：目录没有的表面着色，一段片段也能写。匹配用词边界，作者自己的 `iNormalised` 局部量不会误触发（有测试） |
| **导数宽度抗锯齿（`antialias`）** | 程序化着色器最常见的质量抱怨是"图案一缩小就闪、边缘锯齿很硬"，标准答案是让阈值宽度跟着**一个屏幕像素**走。新增 `antialias` 节点：`smoothstep(-w, w, value - edge)`，`w = fwidth(value) * amount`。它同时给 `needs.derivatives` 这个从第一版就存在、却**没有任何节点声明、也没有任何发射器读**的标志配上了唯一的用户（见 §6.41）。用在 `sdf.distance` 上比 `sdf.mask` 更好：后者的宽度是 uv 单位里的固定值，远了太利、近了太糊。**为什么这里是安全的**：导数只在统一控制流里有定义，而受限子集禁止数据相关分支 —— 主体里每个 `if` 都是 uniform 参数比较，整个 quad 一起走。荷塘示例的三处硬 mask 已经改用它，GLSL 与 HLSL 均编译通过（`fwidth` 在六个目标同名可用） |
| **死端口不再沉默（`IR037_INERT_INPUT`）** | 目录里每一种节点**声明过的输入**都被拿去和它的 body 对照（当时是 42 种，`antialias` 是那之后加的，现在是 43），凡是"声明了但从不读"的端口，只要被接了线就报警。目前只有 `shape.uv` 一个（catalog 明说它为了管线对称性而被忽略）。规则**从 body 派生**，所以未来带死端口的节点一上线就被覆盖；`custom_glsl` 按 kind 豁免（每实例 body，只用两个槽是常态）并且豁免本身有测试。发现这条的方式是写探针脚本而不是读代码 —— 42 种节点 × 平均 4 个输入，靠眼睛是看不完的 |
| **`RW013_SCALAR_SPLAT_CONSTRUCTOR`** | 一条从**编译器**学来的规则：`vec4(0.0)` 在 GLSL/WGSL/MSL 里是标量 splat，在 HLSL 里是**硬错误**（`too few elements in vector initialization`）。重写器只改类型名、不动参数表，所以这是"六分之五能用"的典型。它已经真的发生过一次：`vec_math` 的声明行写成 `vec4(0.0)`，通过了全部结构性检查，只在把 `dxc` 指向产物时才暴露。现在 `lintSubset` 直接拒绝**单个数字字面量**的向量构造，并给出要写成的完整形式 —— 只针对字面量，所以 `vec4(v)`（拷贝）与 `vec4(a,b,c,d)` 不受影响，规则是精确的而不是保守的 |
| **无缝可平铺噪声（`tile`）** | 噪声在数学上**不是**周期的：`hash21(floor(p))` 是绝对格子的函数，所以 uv 方格 N+1 和第 1 格对不上，贴图被引擎平铺的那一刻就会出现接缝 —— 而"给我一张无缝/可平铺的贴图"是这类工具最常被要求的事情之一，此前**做不到**。现在 `noise` / `fbm` / `voronoi` / `domain_warp` / `raymarch` 各有一个 `tile` 输入（0/1，默认 0），打开后格子的**查表**按周期取模，几何保持不折叠 —— 这一点是关键：格子 `period` 的格点就是格子 0 的格点平移一个周期，所以接缝处的插值**精确相等**，而不只是"两边看起来差不多"。`fbm` 的周期还要随八度一起放大（第 `i` 个八度采样在 `p * lacunarity^i`），否则只有第一个八度平铺。**实现只有一份**：`noise2`/`gnoise2`/`fbm2`/`voronoi2` 都是传 `on = 0` 的一行包装，两份实现永远不可能漂移；`on = 0` 是精确的而不是近似的（`mix_(i, mod_(i, per), 0.0)` 恒等于 `i`） |

| tsc（`apps/web`） | `tsc --noEmit -p apps/web/tsconfig.json` exit 0。**只覆盖 `src/`**：`test/` 需要 `@types/node`，本 checkout 没装（见 §4.2） |
| **GLSL + HLSL 被真实编译器编译通过** | `glslangValidator -S frag` → **exit 0**；`dxc -T ps_6_0 -E PSMain` → **exit 0，零 error 零 warning**。这两条是本轮才发现可能的 —— 见 §4.2 |
| 类型投影 | 47 个模块全部投影成功，且每个产物在写盘前都通过 `data:` URL 真实 `import` 过一次 |
| import 完整性 | **92 条 import** 全部检查：78 条相对路径解析到真实文件，14 条裸标识符（5 条经 import map 落到包入口，9 条是 `apps/cli` 的 Node 内建 —— 那条树不进浏览器） |
| 包解析 | `import '@ai4shader/ir'` → 56 个导出；`@ai4shader/ir/catalog` → 43 种节点 |
| **六目标发射** | 示例文档对全部 6 个目标发射，**error 诊断数 = 0**，warning 仅 `IR011`/`RW010` 两类良性的 |
| **Agent 闭环** | 中文一句话 → 6 节点 → 8 个可调 uniform → 六格式零 error，`modelCalls=2`，无 key 无网络 |
| CLI 六条命令 | `validate`/`emit`/`params`/`tune`/`author`/`refine` 全部 exit 0（手工逐条执行验证） |
| 确定性 | 同一 IR 连续发射逐字节一致；同一 provider 两次闭环得到相同 hash（测试覆盖） |
| 跨目标泄漏 | 5 个非 GLSL 目标中不残留 GLSL 类型名或 `_` 别名（测试覆盖） |
| 质量档生效 | `low`/`ultra` 产生不同的循环边界，6 个目标一致（测试覆盖） |
| Y 翻转唯一性 | 每目标恰好翻转一次（测试覆盖） |
| 扇入不误判为环 | 一个节点接进同一消费者多个输入时不再误报 `IR007`（回归测试覆盖） |
| 示例文档 | 7 节点 / 9 uniform，`analyzeGraph` 零诊断 |

**发射产物实测片段**（`dist/demo/shader/shader.frag`，真实输出，人工审阅过）：
```glsl
void main() {
  vec2 fragCoord = gl_FragCoord.xy;
  vec2 uv = fragCoord / iResolution.xy;
  // ---- graph (node bodies; straight-line, in topological order) ----
  vec2 L_n_uv_uv = uv;
  vec2 L_n_centered_uv = (rot2(0.0) * (L_n_uv_uv - vec2(0.5,0.5)) * vec2(2.0,2.0)) + vec2(-0.5,-0.5) + vec2(0.5,0.5);
  vec2 p_n_warp = L_n_centered_uv * u_warpScale + vec2(u_speed, u_speed * 0.5);
  vec2 q_n_warp = vec2(noise2(p_n_warp), noise2(p_n_warp + vec2(13.7, 7.1))) - vec2(0.5, 0.5);
  vec2 L_n_warp_uv = L_n_centered_uv + q_n_warp * u_warpStrength;
  float L_n_detail_value = fbm2(L_n_warp_uv * u_detailScale, int(u_octaves), 2.0, u_gain, u_ridged);
  vec4 ab_n_ramp = mix(vec4(0.02,0.03,0.05,1.0), vec4(0.35,0.38,0.45,1.0), smoothstep(0.0, 0.5, L_n_detail_value));
  ...
  fragColor = vec4(L_n_grade_color.xyz, 1.0);
}
```
这段是**先手工审阅确认语义正确、再作为验收基准**的：变量命名、类型、`int(u_octaves)` 的显式转换、`fbm2` 的新签名都对。

### 4.2 NO 本机无法验证（必须知道）

| 项目 | 为什么无法验证 | 风险等级 |
|---|---|---|
| **GLSL 编译** | **已验证通过**。本机 `glslangValidator` 藏在 Android SDK 的 Vulkan 工具里（`%LOCALAPPDATA%\Android\Sdk\emulator\lib64\vulkan\glslangValidator.exe`，11.7.0）—— 之前一直"未找到"，是因为只在 PATH 和 Vulkan/Windows SDK 里找过。`glslangValidator -S frag shader.frag` 对生成的 GLSL 返回 **exit 0**。`python/ai4shader/validate.py` 的搜索路径里本来就包含 Android SDK，所以这一层现在能真正跑起来 | 低。**"GLSL 能编译"这句话现在有编译器背书了** |
| **GLSL 在浏览器里出图** | 仍然做不到：Chrome 启动即被沙箱拒绝（named pipe）。`scripts/verify-shaders.mjs` 已写好 —— 在有浏览器的机器上一条命令即可产出编译日志与渲染截图 | 低。"能编译"已由 glslang 确认，剩下的只是"像素对不对"，而那是渲染管线的问题，不是着色器的问题 |
| **WGSL 合法性** | 本机无 `naga`/`tint`，无浏览器，**所以没有编译器验证过**。已实现 `targets/wgslLower.ts` 降级 pass：函数签名（`fn f(p: f32) -> f32`）、不可变参数（`var p_v = p;` 复制后再改）、局部声明（`var`）、循环头（`for (var i = …; i = i + 1)`）、`int/float` 显式转换、`textureSample` 的三参数形式。结构性检查（括号配平、无 C 风格声明、无参数赋值）由 `test/wgsl.test.ts` 覆盖 | 中。**从"已确认不合法"降为"未验证"** —— 这一步的价值就在于前者是缺陷、后者是待办，但**不能因此声称它合法** |
| **HLSL 编译** | **已验证通过**，且现在覆盖 surface 文档的**两个阶段**。此前记的"dxc 遇到含函数体的输入就崩溃（`0x80070459`，无诊断）"是**错的**，两处都错：`0x80004005` 是 dxc 正常的"编译失败"退出码（会带 `行列` 诊断）；`0x80070459` 是**文件编码失败**，不是崩溃 —— 本机 ANSI 代码页是 **936**，dxc 1.6 无法解码**没有 BOM 的非 ASCII** 源文件，而每个产物头部注释里都有一个 UTF-8 破折号。加 BOM 后立刻变成正常诊断，并暴露一个**真实缺陷**：`atan2(float2(y,x))` —— HLSL 的 `atan2` 只有标量重载（GLSL/WGSL 都是向量版）。修好后四边形文档 `dxc -T ps_6_0 -E PSMain` **exit 0、零 error 零 warning**；surface 文档 `-T vs_6_0 -E VSMain` 与 `-T ps_6_0 -E PSMain` 同样双双 exit 0、零 warning（`python/ai4shader/validate.py` 会对非 ASCII 源写 BOM 并在 `notes` 里记录；`dxc_profile()` 按入口点选 profile，因为 `-T ps_6_0 -E VSMain` 是 dxc 报错，不是对顶点阶段的验证） | 低。**"HLSL 能编译"现在也有编译器背书了，且是两个阶段** |
| **两处"不是缺陷"的澄清** | ① dxc 的"崩溃"其实是编码问题，被误记为环境限制并写进了上一版文档；② `mod_→fmod` 的截断语义差异是真的（见 §6.4），但 HLSL 的 `wf_mod`/`wf_mix` 只有标量重载，导致向量调用被**静默截断**（dxc 报 `implicit truncation of vector type`）。已补齐 `float2/3/4` 重载，让每个调用点都是精确匹配 | — |
| **反馈（`render.feedback`）只在平面预览里实现** | WebGPU 后端与 3D 材质路径都不实现，且都**明说**：WGSL 的编译结果里带一条 warning，材质面板的通道说明写成"3D 材质路径只提供白色占位（反馈仅支持平面预览）"。移动网格的"上一帧"是屏幕空间累积（要先渲染到 FBO 再合成），是另一个功能而不是同一件事 | 中。**能做到的已经做到并验证；做不到的被指名，而不是悄悄给一张白纹理** |
| **MSL 编译** | 非 macOS，无 `xcrun metal` | 中 |
| **ShaderLab 编译** | 无 Unity | 中。已按 URP/Built-in 两条路径分别生成并标注不确定项 |
| **UE Custom 节点** | 无 UE，且 UE 本身不支持对 Custom 节点做无头编译。**能查的部分已经查了**：`CustomNode.inputs.json` 的引脚清单被断言必须包含四个表面输入（`InWorldPos`/`InNormal`/`InCameraPos`/`InLightDir`，都是 `float3`，都带"该接哪个 UE 节点"的 label），且文档里每个 uniform 都必须有对应引脚 —— 这是**搭建配方完整**的证明，**不是**"UE 能编译"的证明 | 中（编译未验证）→ 低（配方正确性已验证）。已明确降级并输出引脚清单 + 搭建配方 |
| **浏览器工作台** | 启动路径**已验证**（`apps/web/test/boot.test.ts`，见 §4.1）：模块图、面板构建、Agent 闭环、渲染循环、预览模式切换都跑通了。剩下的缺口收窄到 **GPU 那一层**：测试里 `canvas.getContext()` 返回 `null`，所以它验证的正是降级分支，"第一帧的像素对不对" 仍然没人看过 | 中 → **低**。工作台从"整体未验证"变成"除像素以外已验证" |
| **3D 预览的画面本身** | 同上：材质 pass 的着色器、光照、地面与阴影衰减**没有在真实 GPU 上跑过一次**。已做的是：几何与相机的纯函数被单测（绕序、投影深度范围、相机基、矩阵复合顺序），着色器编译失败会**指名报错并退回平面模式**而不是留一块空画布，图案/材质两趟的顺序有代码断言；图着色路径另外被 7 条单测钉住宿主契约（源码逐字节、插值变量逐名、uniform 按反射类型、滑杆不重编译、失败退回、四边形不进图路径）；手写的五个阶段另外被 `glslangValidator` 真编译过（`python/tests/test_preview_shaders.py`）。**贴图映射**这一层也不再只是"看起来对"：`pattern-mapping.test.ts` 从着色器源码与录制到的 GL 状态还原出映射，对每个顶点断言"没有任何顶点被交给采样器的边缘钳制" —— 它当场抓到的是 §6.48 那个"图案只出现在一角 / 地面只有一个十字"的真缺陷。**仍然没人看过一帧像素**：录制型 GL 记录的是调用，不是图像 | 中。**"能编译、几何正确、宿主契约正确、映射被量过、失败可见" 已成立，"好不好看" 未验证** —— 与 GLSL 那条同一性质：缺的是像素，不是逻辑。**注意 §6.48 的教训**：映射错误不会报错，只会画出一张"像那么回事"的图，所以这一层的验证必须靠**算术**（顶点采样坐标是否越界），不能靠"看着还行" |
| **`@types/node` 未安装** | 本 checkout 的 `apps/web/node_modules/@types/` 下只有 `estree`，无网络装不了。后果：`packages/ir/tsconfig.json`（`include` 了 `test/**`）与 `apps/cli/tsconfig.json`（写了 `"types": ["node"]`）的 `tsc` 在本机**必然报错**，与代码质量无关；`apps/web` 因为只 `include` `src/` 所以仍然全绿 | 低，但**必须说清楚**：本机"tsc 全绿"只对 `apps/web/src` 成立，不是全仓 |
| **CLI 的 e2e 测试套件** | 该套件断言的是**进程边界**（子进程 stdout、退出码、落盘文件），需要带管道 stdio 的 `spawn`，本沙箱以 `EPERM` 拒绝（`spawn EPERM` 已在 §2 记录为由沙箱文档化的边界） | 低。CLI 本身已由手工端到端验证；`scripts/run-tests.mjs` 会启动时探测该能力，并把这类套件报成 `SKIPPED-BY-ENVIRONMENT` 而**不是**失败 —— "跑不了"和"跑挂了"绝不允许混淆 |

### 4.3 关于"未验证"的诚实声明

发射器生成的每个 `.wgsl`/`.hlsl`/`.metal`/`.shader` 文件**顶部都带一段注释**，逐条列出该目标的不确定项与已知限制。这是刻意的：把不确定性写进产物本身，而不是留在一个没人读的 README 里。

---

## 5. 补齐验证的清单（在正常环境里）

```bash
# 一次性
npm install                                     # 建立 workspace 软链（本机是手工 junction）

# 1) GLSL 真实编译 + 出图（最关键的一项）
npm run build
node scripts/serve.mjs &
node scripts/verify-shaders.mjs                 # 需要 Chrome/Edge 可启动

# 2) WGSL 合法性：装 naga 或 tint，直接喂 shader.wgsl
cargo install naga-cli && naga dist/demo/shader/shader.wgsl
# 或浏览器路径：应用内 WebGPU 后端 + getCompilationInfo()，行列号已经接好了

# 3) HLSL（本机已通过：四边形文档 exit 0、零 error 零 warning）
dxc -T ps_6_0 -E PSMain dist/demo/shader/shader.hlsl
#    表面文档是两个入口点，profile 必须跟着入口走（本机同样已通过、零 warning）
node apps/cli/dist/cli.js emit examples/toon-surface.shader.json --target hlsl --out dist/toon --force
dxc -T vs_6_0 -E VSMain <bundle>/shader.hlsl
dxc -T ps_6_0 -E PSMain <bundle>/shader.hlsl
#    另外三份示例同样已通过（feedback 与 43 节点的场景文档各两个入口）
node apps/cli/dist/cli.js emit examples/fog-wipe.shader.json    --target hlsl --out dist/fog  --force
node apps/cli/dist/cli.js emit examples/lotus-pond.shader.json  --target hlsl --out dist/pond --force

# 4) MSL（macOS）
xcrun -sdk macosx metal -c dist/demo/shader/shader.metal

# 5) ShaderLab：Unity 批处理
Unity -batchmode -quit -projectPath <proj> -executeMethod AI4Shader.Editor.Validate

# 6) 类型检查
npx tsc -p packages/ir/tsconfig.json --noEmit
```

`scripts/verify-shaders.mjs` 已经把这些串起来：它发射**两个**真实文档（四边形示例 + 三渲二 surface 示例，见 §4.1 的四个示例）、把 GLSL 嵌进两个自包含页面、在无头 Chrome 里编译+绘制、把驱动的原始 `getShaderInfoLog`、uniform 绑定数、亮度统计与渲染帧一起输出。**"COMPILED 但画面全平"会被判定为失败** —— 能编译不等于画对了。

surface 那一页的顶点阶段是**脚本按发射出的片元着色器自己的 `in` 声明生成**的（`vWorldPos`/`vNormal`/`vUv` 由全屏三角形的坐标合成），所以它证明的是"发射出的片元阶段在真实驱动上能编译并画出像素"，**不是**"网格能满足插值契约" —— 后者由 `apps/web/test/surface-graph.test.ts` 逐名比对两个阶段的声明来钉住（因为 `glslangValidator -l` 对改名后的 varying 照样 exit 0，见 §6.30）。本机的两个阶段已用 `glslangValidator -l` 交叉验证过 exit 0，但**浏览器那一步在本沙箱无法执行**（`spawn EPERM`）。

---

## 6. 本次修复的真实 bug（由并行开发者报告 + 我复核）

这些不是重构，是会导致画面错误的缺陷：

| # | 位置 | 症状 | 修法 |
|---|---|---|---|
| 1 | `targets/_shared.ts`（6 个发射器） | **节点体被放在 `main()` 外面**，生成的 GLSL/WGSL/HLSL/MSL 全都无法编译 —— 全局初始化引用了不存在的 `uv` | 新增 `embed()`，把节点体缩进后嵌入各目标入口点内部 |
| 2 | `rewrite.ts` `lowerMatrixMul` | 内联实参时不再走重命名，`mat2_mul_(rot2(a), (uv - vec2(0.5,0.5)) * scale)` 会把 `vec2(` 泄漏进 HLSL/WGSL/MSL | 对两个实参递归 `rewriteSource` 后再内联 |
| 3 | `rewrite.ts` `mat2_rot_` | 被重命名成 `mat2` 构造器 → **旋转变成等比缩放**。两者在图像上完全不同 | 重命名为库函数 `rot2` |
| 4 | `rewrite.ts` `mod_` | 映射到 `fmod`，而 `fmod` **向零截断**、GLSL `mod` **向下取整**。`hsv2rgb` 的色相可为负 → 只有 2/6 个目标出现色相断层 | HLSL/MSL 改用 `wf_mod`（floor 形式）垫片 |
| 5 | `rewrite.ts` `lintSubset` | 循环正则匹配"以空格连接的 token"，`i ++` 永远匹配不上 → **每个正常 for 循环都误报** | 去掉空格连接，并允许负初值与 `<=` |
| 6 | `emitter.ts` `coerce()` | float→vec2 生成 `vec2(x, x, x)` —— 三个分量，GLSL 编译错误 | 按目标元数精确生成 |
| 7 | `emitter.ts` 多输出解构 | 对**任何** schema 都无效：每个端口都别名到整个容器 | 约定多输出节点返回 `vec4`，按 `.x/.y/.z/.w` 取分量；超过 4 个或非标量端口报 `IR033` |
| 8 | `emitter.ts` | `outputs: []` 直接 `TypeError` | 为空时作为语句发射 |
| 9 | `glslSubset.ts` | 模板字符串里有未转义反引号，导致**整个包无法 import** | 转义（由目录开发者修复） |
| 10 | `examples/liquidMetal.ts` | 引用了不存在的 `n_uv.centered` 端口 | `uv` 单输出；居中改用 `transform` 节点表达 |
| 11 | `emitter.ts` `analyzeGraph` 拓扑排序 | **最严重的一个**：入度按「线」计数、出边去重按 `from->to` 对，于是「一个节点接进同一个消费者的多个输入」时入度永远归不了零 → **误报 `IR007_CYCLE` 并丢掉该节点及其下游**，产出一个能编译但图整个消失的 shader（静默、彻底失效）。Kahn 算法要求的是边的**多重集**而非边集 | 去掉 `seenEdge` 去重；补回归测试（验证扇入不误报、真环仍能检出、且全量发射里有该节点） |
| 12 | `rewrite.ts` `rewriteNumber` | hex 检查写在 `wgsl` 提前返回**之后**，于是在该目标上不可达：`0x10` 静默变成非法的 `0x10.0`，而专门用来警告它的 `RW002` 从不触发。**把诊断放在 `return` 之后 = 静默删除它** | 检查前移，并把 hex 归一化为十进制；补 6 目标回归测试 |
| 13 | `scripts/project-ts.ts` | 文件头承诺「每个投影产物在写盘前都通过 data: URL 真实加载」，但 `validateModule` 从未被调用 —— 承诺是假的 | 在 `main()` 里按序调用（26 个模块全部通过加载），并提供 `--no-validate` 显式退出 |
| 14 | `packages/agent/dist/*.js` + `apps/web/index.html` | **工作台在浏览器里是空壳**：`packages/agent/dist` 有 5 处 `import … from '@ai4shader/ir'` 裸说明符，Node 经 `node_modules/@ai4shader` 软链解析得了，**浏览器直接拒绝整个模块图**（`Failed to resolve module specifier "@ai4shader/ir"`）→ `dist/main.js` 一行都没执行，页面只剩静态 HTML，每个面板都是空的。引入路径很清楚：Agent 面板（`src/panel/author.ts`）是本轮才接到 `main.ts` 上的，它是**第一个**从浏览器图里 `import` `packages/agent/dist` 的东西（此前没有任何 web 模块引用它），于是把裸说明符第一次拖进了浏览器路径。**所有 Node 侧检查全绿**：投影成功、`check-imports` 只查相对说明符、启动测试借用 Node 的解析器。更糟的是 `check-imports.mjs` 的结尾**用一段文字声明了「产物里不该有裸说明符」这条不变量，然后只建议你手动 grep** —— 不变量从未被检查过 | ①`index.html` 加**精确** import map（两条：`@ai4shader/ir`、`@ai4shader/ir/catalog`；不能写前缀映射，import map 不做扩展名解析）；②`check-imports.mjs` 改为**强制**：裸说明符无条目 / 条目指向不存在文件 / 条目无人使用，三者都是失败；③新增 `test/register-hooks.mjs`，让 Node **按浏览器的方式**解析裸说明符，于是启动测试不再替浏览器背书；④新增 `scripts/check-served.mjs`，自己起服务器沿 HTTP 走一遍真实模块图（200 + JavaScript MIME type + 说明符可解析 + map 在 module script 之前） |

第 3、4 条尤其值得记：它们**不会**导致编译失败，只会让画面悄悄不对，而且只影响部分目标。这正是本方案存在的理由。

第 14 条是另一类：它**不会让任何检查变红**。一段声称某条不变量成立的注释，比没有注释更危险 —— 因为所有人都会去那里确认那件已经坏掉的事。

| # | 位置 | 症状 | 修法 |
|---|---|---|---|
| 15 | `apps/web/src/panel/author.ts` | **在线模式在部分网络下没有任何可用出路**：base URL 被写死成 `https://api.openai.com/v1`，界面上**没有**任何地方能改成别的服务。本机实测 `api.openai.com:443` 连不上（DNS 返回的是一个黑洞地址，TCP 超时），而 `api.deepseek.com` / `openrouter.ai` / `api.moonshot.cn` / `api.siliconflow.cn` / `dashscope.aliyuncs.com` 全部可达且**都返回了覆盖本页来源的 CORS 头**。于是用户能做的只有重打一遍 key 再点一次「生成」，拿到的是浏览器那句 `Failed to fetch` —— 既不说是哪个主机，也不说为什么。**"配错了"与"根本改不了"是两回事** | ①服务变成下拉框（预设的地址都实测过可达 + CORS，未通过的不列入）；②地址可手改，改到已知地址会自动选中对应服务；③`测试连接` 发一次**真实的单 token `chat/completions`**（同地址、同 `Authorization` 头、同 `content-type`），把 401/404/超时分别说清楚 —— 不用 `GET /models` 是因为那是另一个请求，会在真实调用必然失败时报"一切正常"；④就绪判定会指出**缺哪一项**（原来一个布尔值让本地 Ollama 永远点不动「生成」且不说原因）；⑤服务/地址/模型/离线还是在线持久化，**key 不落盘** |
| 16 | `packages/agent/src/loop.ts` + `provider/http.ts` + `types.ts` | **在线 provider 从来没能真正起草过任何东西** —— 本轮最有分量的一个缺陷。`createHttpProvider` 会解析模型回复并**按 catalog 校验每一个 op**（这个 kind 存在吗、这个端口有吗、这条线类型对吗），而它的 catalog 默认是 `{}`，`setCatalog()` **在接口上声明了、实现了，却没有任何地方调用过**：CLI 没调、web 门面没调、loop 也没调。于是模型发来的每一个 `add_node` 都被判为 `AG012_UNKNOWN_KIND` 并丢弃，**整份草稿到达时是空的**，图永远是空的，一路烧到预算上限，最后剩下 `IR008_NO_OUTPUT` —— 读起来就像"模型不听话"。用户就是这样撞上的：deepseek-chat 起草了六个节点，界面上什么都没发生 | 三处一起改：①`PatchProvider` 接口新增可选的 `setCatalog?()` —— 它本来就属于"propose"这件事，一个要解析模型文本的 provider 没有 catalog 就无法判断端口是否存在；②`createHttpProvider` 保留默认值，但不再是唯一来源；③**loop 在第一次 propose 之前把 catalog 交给 provider**。放在 loop 是因为它是唯一同时拥有 catalog 与 provider 的组件 —— **需求落在调用方、失败形态又完全不像"忘了配置"，两个宿主都漏了。那不是两次失误，是一个允许宿主忘记的 API** |
| 17 | `packages/agent/src/policy.ts` + `repair.ts` + `loop.ts` | 第 16 条造成的空图，又被**错误地归类为"不可修复"**：`IR008_NO_OUTPUT` / `EM001_NO_OUTPUT_NODE` 写在 `UNRECOVERABLE` 表里，理由是"没有输出节点就得重新画图，那是起草的活"。**一个真实的 six-node 草稿证明这个理由不成立** —— 加一个 sink 是一次纯机械编辑。附带第二个缺陷：`changed` 逃生舱比较的是**节点个数**，而一轮改线/改参数/换 kind 的重修不会改变个数，于是 loop 读着过期的诊断、断定"什么都没发生"、在手里正握着一个修复的时候中止 | ①新增 `addOutputNode()`：加 `output` 并接上**最深的、没有被任何节点消费的**那个源（同级按 id 定序，保证两次运行产生同一个补丁）；②从 `UNRECOVERABLE` 里删掉这两个码 —— 该表只留给真正在文档之外的问题（不存在的 kind、没有发射器的目标、没配 provider）。**教训比删除本身更重要：文档级诊断 ≠ 无法处理的诊断**，旧表把"这条抱怨没有指名节点和端口"当成了"无事可做"；③`changed` 改用 `hashDoc` 比较；④**空图必须走 advice 而不是 patch** —— 给空文档加一个孤零零的 `output` 会发射一片纯色、发射器会说 clean、而被拒绝草稿的诊断会被"clean 覆盖 seed"的规则丢掉，于是用户在看着黑屏的时候被告知成功了。这条守卫是测试逼出来的，不是想出来的 |
| 18 | `apps/web/src/providers.ts` + `panel/author.ts` | **把猜测当成了菜单**：预设表给每个服务硬编码 2–3 个"模型名"，界面上看起来就是"只能从这几个里选"。用户直接问"为什么不能选 DeepSeek 所有的模型"。这是两个错误叠在一起：①**枚举别人的模型清单这件事在构建期不可能做对** —— 本机实测 OpenRouter 的 `/models` 无需 key 就返回 **458** 个，按周变化；而 DeepSeek / Moonshot / SiliconFlow / 百炼的 `/models` **都要 key**（401），所以真实列表只能用用户自己的凭据在运行时问出来；②字段名叫 `models`，读起来就是"模型清单"而不是"起步值" | ①新增 `listModels()` 走 `GET {base}/models`，容忍 4 种返回形状（OpenAI 信封 / Ollama `models[].name` / 裸数组 / `result`），去重但**保留服务端顺序**；②面板加 **拉取模型** 按钮，取回后**整体替换**建议值，并显示"服务端返回 N 个"；③`models` 改名 `modelHints`，注释写明它是起步值不是目录；④两个按钮**故意不合并**：模型名写错时测试连接 404 而拉取模型会成功并把正确名字递过来，串在一起会在最需要时藏起更有用的那个；⑤手写的名字若不在服务端列表里，输入框**保留**它 —— 界面不该和用户争 |
| 19 | `apps/web/src/panel/author.ts` | **一条正确但端错了的规则**：面板原本**拒绝**持久化 key，理由写得很清楚 —— `localStorage` 里的 key 是这个来源上任何脚本都能读的，而这里没有服务器能替你保管。这个理由至今成立，但它漏算了替代方案是什么：**每次刷新重打一个 50 字符的密钥**。一条实际效果是让谨慎路径变得烦人的规则不会让人更谨慎，只会让人把 key 粘进草稿文件 —— 那比 `localStorage` 在任何方面都更糟。用户直接要求"至少存本地" | ①加 **记住 key** 复选框，**默认勾选**，状态与代价写在做出选择的那一行旁边；②取消勾选**立刻删除**已存储的值（不是"下次不读"—— 一个把密钥留在磁盘上的开关，是比没有开关更坏的半成品）；③`rememberKey:false` 时**忽略**存储里遗留的 key，否则旧状态会静默压过当前选择；④把 shim 的 `localStorage` 从空实现换成真的 `Map`，于是"刷新后还在吗"从指望变成了断言 —— `apps/web/test/author-panel.test.ts` 构造**两次**面板，第二次构造就是一次刷新 |
| 20 | `apps/web/src/panel/author.ts` | **控件本身在藏东西**：模型字段用的是 `<input list=…>` + `<datalist>`，而**浏览器会按输入框里已有的文字过滤 datalist**。面板一进来就预填了模型名，于是点开下拉只看到那一个名字 —— 用户的原话是"为什么只能选择 deepseek-chat"。真实列表当时有两个，控件把另一个也藏了。**一个内置过滤的控件 + 一个预填的值 = 一个看起来只有一项的菜单** | 换成真正的 `<select>`（永远显示全部）+ 末尾「手动输入…」保住自由文本；没有列表时（「自定义」预设、或拉取失败）只显示输入框，绝不给一个空下拉框。附带一个只有测试才会发现的坑：选择「手动输入…」会**立刻弹回**，因为"用户要求手输"和"这个值不在列表里"是两种状态、从值上看起来一模一样 —— 加一个 `manualModel` 标志区分 |
| 21 | `apps/web/src/providers.ts` | **过期的建议比没有建议更糟**：DeepSeek 那一行写的是 `deepseek-chat` / `deepseek-reasoner`，那是这张表写下时的模型。用户问"没有 v4.1 / flash / pro 吗" —— 查了官方文档才发现现在的 API 模型是 **`deepseek-flash`（DeepSeek-V4.1-Flash）** 与 **`deepseek-v4-pro`**，旧的两个已经不在文档里了。硬编码的名字一旦过期，就会变成**用户去试的一个名字**，而它返回的 "model not found" 读起来像是用户输错了 | 按 [官方 Models & Pricing](https://api-docs.deepseek.com/quick_start/pricing/) 修正，并在注释里写清楚**是查证来的而不是回忆来的**、以及为什么 `deepseek-v4-flash` 这类"仍被接受但已退役"的别名故意不列入。这条同时是第 18 条的反面教材：提示值可以是起步点，**但必须是核对过的起步点** |
| 22 | `packages/ir`（契约）+ `src/targets/_shared.ts` | **三渲二在这个 IR 里不是"难"，是不可表达**。当时那 31 种节点里**零个**表面/光照词汇：没有 `normal`、没有 `light`，`math` 只有标量、没有点积。整张图是 `f(uv,time)→color`，而三渲二的定义是 `quantize(N·L)` —— **两个不同的定义域**。后果不止于预览：六个目标发射出的也全是全屏四边形，**拿到 Unity/UE 里贴不到模型上**。用户的原话是"目前对于三渲二完全错误" | ①`HOST_SYMBOLS` 增加 `iNormal` / `iWorldPos` / `iViewDir` / `iLightDir` / `iCameraPos`，新增 `surface` 节点分组（36 种）；②`NodeSchema.needs.surface` —— 这是**契约变更而非提示**：一旦可达节点声明它，每个目标的入口点形态就变了（插值顶点输出取代 `gl_FragCoord`，`uv` 来自网格而非屏幕）；③标志位按**可达集**计算，删掉最后一个 `normal` 必须让发射逐字节回到四边形形态；④**`SURFACE_CAPABLE` 是闸门而不是备注** —— 尚未支持的目标报错并返回空 source，因为让它们照常写出一份引用了未声明符号的"成品"才是真正的伤害 |

**第 22 条值得单独记一句**：我在这个特性上**走错了两次方向**（先想在预览里加卡通着色，再想靠加模型/颜色解决）。两次都错在同一个地方 —— **把渲染器能假装的东西，当成了用户能写出来的东西**。ramp 形状、色阶数、描边、高光，这些**就是三渲二 shader 本身**；由我的代码固定的那一刻，用户就失去了它。正确的修法只能在 IR 里，因为那才是"用户写的东西"所在的那一层。

| # | 位置 | 症状 | 修法 |
|---|---|---|---|
| 23 | `packages/agent/src/loop.ts` + `policy.ts` | **一个无法修复的诊断，烧掉了 12 次模型调用**。真实运行：三渲二文档发射到尚无表面入口的目标 → `EM002_SURFACE_UNSUPPORTED` → trace 显示 `emit→refine` 重复 10 轮直到预算耗尽。两处叠加：①`EM002` 没进 `UNRECOVERABLE` 表，于是 repairer 认不出它，loop 把它当成"模型或许能修"；②更关键的是**停止条件里 `changed` 排在 `hopeless` 之后** —— 第 0 轮的 `changed` 恒为真（没有上一轮的 hash 可比），模型每轮改文档又让它继续为真，于是一个 repairer 已经判定"文档之外的问题"的诊断，可以无限期压住终止判断 | ①`EM002` 进 `UNRECOVERABLE`（**它是表里第一个"正确文档也会产生"的码** —— 图没问题，是发射器还写不出来，能力缺口按定义就在文档之外）；②停止条件改为 `!progressCapable && (allHopeless || !changed)`：`changed` 是为**过期诊断**准备的逃生舱，而 `hopeless` 不是过期，是"改文档也解决不了"，多改几轮不会让它变新鲜。补两条测试：一条用"每轮都真的改文档"的 provider 断言 `calls === 0`，一条钉住分类本身 |
| 24 | `packages/ir/src/rewrite.ts` | **HLSL 上两个目标悄悄偏色**。GLSL 的 `mix(e1,e2,e3)` 对**向量** `e3` 是按分量混合，`mod(x,y)` 的 `y` 也可以是向量；而 HLSL/MSL 垫片只声明了标量形式，于是向量实参**被截断成第一个分量**。代价不是理论上的：共享库的 `srgb2linear` 读作 `mix_(lo, hi, step(vec3(0.04045), c))` —— GLSL 里每个通道按自己的阈值转换，HLSL 里三个通道全用**红色通道**的阈值。**六个目标里有两个偏色，且没有任何编译错误**。`dxc -Wconversion` 其实一直在每个产物里警告 `implicit truncation of vector type` | ①补齐 `wf_mix` 的向量因子重载与 `wf_mod` 的向量除数重载（HLSL 的 `lerp` 本来就接受两种，垫片只需说出来）；②**Python 侧的 dxc 检查改为断言 `warning_count == 0`** —— 它此前只断言 `error_count == 0`，把编译器一直在说的话丢掉了；③补一条 TS 测试直接断言**垫片家族在两个实参位置上都完整**（按调用点断言不可能：出问题的那个实参本身是个 `step(...)` 调用，正则不知道它的类型）—— 把检查放在缺失真正发生的地方 |

**第 24 条的方法论**：这条是被**真实编译器**抓到的，不是被推理或测试抓到的。它同时暴露了一个更值得记的失败模式 —— **"断言没有错误"不等于"断言正确"**。`dxc` 从始至终都在报警告，而检查里只数了 error。

| # | 位置 | 症状 | 修法 |
|---|---|---|---|
| 25 | `packages/ir/src/catalog/nodes.ts` + `types.ts` | **类型系统缺了半边，于是"在指针位置画东西"根本无法表达**。连接规则只允许**加宽**（float → vecN），没有隐式收窄、也没有连线上的 swizzle，所以一个 `vec4` 生产者只能接进 `vec4` 输入。而 `mouse` 输出的正是孤零零一个 `vec4`（xy 指针、zw 按下位置）—— 它接不进 `transform.offset`（vec2）、接不进 `noise.uv`（vec2），**接不进任何空间输入**。更糟的是那条线会被 `fixTypeMismatch` 判定为"源比目标宽"然后**删掉**：效果消失，运行报成功。用户最早的"手指抹在水雾玻璃上做不出来"就是这个 | ①`split_vec2` / `split_vec4`（向量拆标量）、`make_vec2` / `make_vec4`（标量拼向量）—— 把类型系统的缺失半边**表达为节点**，而不是改连接规则，这样六个目标的降级逻辑一行都不用动；②`vec_math`：向量版 `math`（normalize / length / distance / reflect / fract / floor / abs / pow / max / min / sign / clamp / cross），因为原来的 `math` 只有标量、`vec_op` 只有 7 个逐分量算子；③`dot` 的默认值从 `v3()` 修正为 `v4()`，并新增 `v4()` builder —— 方向量的 `w` 是 0，用 `color()` 当默认值数值上能用但读起来是错的。**验证方式**：把 `mouse → split_vec4 → ÷resolution → make_vec2 → noise.uv → output` 真的搭出来，六个目标零 error，GLSL 与 HLSL 真实编译 exit 0、零 warning |

| 26 | `packages/ir/src/glslSubset.ts` + `rewrite.ts` | **一条只对着"已有输入"执行的规则不算规则**。受限子集禁止 `mix(`/`mod(`/`fract(`/`atan(` 等内建拼写（必须用别名：HLSL 叫 `lerp`，`fmod` 截断而 GLSL 的 `mod` 向下取整）。但这条规则**只在 `catalog.test.ts` 里被检查** —— 一个扫描**已发布节点体**的单元测试。它成立的前提是"节点体是测试能枚举的常量"。`custom_glsl` 一上线，节点体就能在运行时到达，前提就没了：一段写 `mix(a,b,t)` 的片段在 GLSL 里合法、在 HLSL 里**编译失败**，零诊断 | 把规则搬进生产线：新增 `BANNED_BUILTIN_SPELLINGS`（拼写 → 别名 + 理由），`lintSubset` 报告 `RW012_BUILTIN_SPELLING`，于是它对**每一个** body 生效，无论来自目录还是运行时。`catalog.test.ts` 的禁令表改为**从生产常量派生**而不是再抄一份 —— 两份必须一致的清单，失败方式恰恰是最有意思的那种：有人给 rewriter 加了别名，生产检查学会了，测试清单没有，目录就悄悄发出去一个 linter 会拒绝的 body |
| 27 | `packages/ir/src/catalog/nodes.ts` | **`custom_glsl` 第一版的端口说明是假的**。注释与 summary 都写着"四个 vec4 输入，float/vec2/vec3 都能接进来并被 splat"。**错**：`canConnect` 只把 float/int/bool 加宽到向量，**vec2 不会加宽成 vec4**。于是 `uv` 根本接不进这个节点 —— 每个需要坐标的自定义效果都得先 `split_vec2` → `make_vec4`。文档说一套、类型系统做另一套，而发现方式是把它真接一遍：`PT014_TYPE_MISMATCH: n_uv.uv is vec2 but n_custom.a expects vec4` | 给 `custom_glsl` **一个自己的 `uv: vec2` 端口**（节点的职责就是"某个坐标上的效果"，把端口写明白省掉每个作者同样的三步舞），并把 summary 改成真话："标量会加宽进 vec4 端口；vec2/vec3 必须先过 `make_vec4`"。同一条修正也进了给模型的 prompt 指南 —— 说明书和类型系统不一致时，模型会照说明书写 |

| 28 | `apps/web/src/preview/surface.ts` | **3D 预览从来没有运行过发射出来的着色器 —— "三渲二完全错误"的第二个原因**。§6.22 补上了 IR 的表面输入与六个目标的表面入口点，但预览这边仍把文档当成一张**贴图**：图案由后端画进 512² 画布、再贴到受光网格上。于是一个写着 `dot(iNormal, iLightDir)` 的文档在预览里**根本没有这一项** —— 用户看到的是一条固定色带，而不是随光照转动的明暗分界，而且他会合理地认为"这工具做不出三渲二"。**发射器已经会写正确的东西，是宿主没有用它** | 按 `Emission.needs.surface` 分两条材质路径：为真时把 `emit()` 的 GLSL surface 源码**逐字节**编译到一个写入 `vWorldPos`/`vNormal`/`vUv` 的网格顶点阶段上，图在每片元真求值；宿主 uniform 按**驱动反射**写（`iCameraPos`、`iLightDir`、`iResolution`、`iTime`、`iMouse`、`iChannel0` 1×1 白纹理、以及文档自己的滑杆），图真的读 `iMouse` 时普通拖动改为操作指针（Shift/右键拖动才是转相机），图案路径的三个贴图参数在此时**禁用并说明原因**（能拖但没反应的滑杆比隐藏它更糟）；编译失败退回贴图路径、报告驱动原话、且**不留旧版本**（留着旧版本会让人以为改动没生效）。验证见 §4.1 那 7 条单测 |
| 29 | `apps/web/src/preview/surface.ts` | **宿主喂给图的 `iLightDir` 不是单位向量**。`SUN = [0.55, 0.72, 0.42]` 的长度是 **0.99865**，而 IR 与发射出的头注释都写明 `iLightDir` 是"单位向量，表面→光"。贴图材质自己 `normalize(uSunDir)`，所以这条偏差在视觉上不存在；图着色路径把原始常量直接交给 `iLightDir`，于是 `dot(iNormal, iLightDir)` 不是余弦 —— 每一张卡通图的明暗分界都系统性偏暗 0.13%。**这种量级没人会看见，所以它会一直留在那里**，而且一旦有人把 SUN 改成明显非单位的向量，卡通明暗会整体偏移而没有任何线索。发现方式是给新写的那条宿主契约测试加了"`iLightDir` 必须是单位向量" | 常量定义时就归一化一次（`unit([0.55,0.72,0.42])`），天光、贴图材质与图共用同一个向量 |
| 30 | 验证方法本身的边界：`glslangValidator -l` | **顶点与片元之间的插值变量对不上，链接器不会报错**。把预览顶点阶段里的 `vNormal` **一致地**改名（声明与赋值都改）后，`glslangValidator -l mesh.vert surface.frag` 依然 **exit 0**。也就是说"两个阶段能链接"**不能**证明插值契约成立：GLSL ES 允许片元输入没有对应的顶点输出（读到未定义值），这正是那种"编译通过、画面随机"的失败。这条不是产品缺陷，记在这里是因为它决定了"靠编译器验证"能走多远 | 契约改由单测钉住：把两侧的 `out`/`in` 声明集合**逐名逐类型**比对（顺序无关）。负向对照做过：只改一侧的名字，**只有那条测试变红**，编译器全程沉默 |
| 31 | `python/ai4shader/validate.py` | **"HLSL 编译通过"只覆盖了片元阶段**。`discover_entry()` 回答的是"片元入口叫什么"，返回 `PSMain`；但 surface 文档的 HLSL 里是 `VSMain` + `PSMain` 两个入口，而在本轮之前**没有任何测试把顶点阶段交给过编译器** —— 插值契约（`worldPos` / `normal` / `uv`）恰恰住在顶点阶段。另外 `-T` 是硬编码的 `ps_6_0`，用它去编 `VSMain` 不是"验证顶点阶段"，是 dxc 报错 | 新增 `entry_points()` 返回文件里**全部**入口（四边形文档同样有两个：它的 `VSMain` 用 `SV_VertexID` 画全屏三角形），新增 `dxc_profile()` 按入口点选 `-T`（`VSMain → vs_6_0`）。Python 测试对每个入口各编译一次，断言 `error_count == 0` **且** `warning_count == 0`，并另外断言 surface 文档的顶点阶段确实**消费网格属性**（`POSITION` 语义）而不是回退成了全屏三角形 —— 否则"有两个入口"这件事在四边形文档上也成立，测试会为错误的原因通过 |
| 32 | `packages/agent/src/provider/prompt.ts` | **能力上线了，禁令还留在提示词里**。intent 提示词第 2 条写着"只描述 2D 片元着色器能做的事：颜色、图案、运动、遮罩、调色板。**不要几何、不要光照模型、不要网格、不要场景反射**"，理由是"**图里没有 3D pass**"。这条理由在 §6.22 之后就不成立了，而它造成的后果比缺失更糟：模型被明确告知光照是不可能的，于是被要求做"三渲二"时，它会用**屏幕坐标**拼一个"看起来像卡通着色的图案" —— 在四边形预览里勉强像，放到表面上就完全错，而且不随光、不随相机。这正是用户报的"三渲二完全错误"在提示词层的残留 | 重写第 2 条：说明图**会**读表面法线/世界坐标/视线/光向，把"不可能"精确限定为几何、额外网格、阴影贴图、场景反射；补一个三渲二示例回复（每个 feature 都是"表面如何迎光"的陈述）；新增 `SURFACE_GUIDE` 给补丁阶段 —— 五个表面节点、两条真实构建链（明暗分层 / 边缘光）、以及两个后果（文档变成 surface 着色器、`uv` 来自网格）。测试两条：旧的禁止措辞必须消失、表面输入与后果必须出现 |
| 33 | `apps/web/src/main.ts` | **surface 文档落进平面预览会"编译通过"**。GLSL ES 允许片元输入没有匹配的顶点输出（读到未定义值），所以把三渲二文档放在平面模式里，着色器**编译成功**、strip 显示 0 error、画面是黑的或噪声 —— 与"生成器坏了"完全无法区分。而平面模式恰恰是默认模式 | `previewModeFor(needsSurface, current)`（纯函数、被导出、被单测）让预览模式**跟着文档走**：surface 文档自动切到「材质」并说明原因，`setDoc` 与 `setupRenderer` 两个入口都执行它（后端切换会重建渲染器但文档没变，那条路也必须走）。只降不升的切换刻意不做：用户选的 `scene` 不会被改回去 |
| 34 | `packages/agent/src/repair.ts` | **同一个端口收到两个相反的修复，删除那个最后执行**。一个端口类型不对会被报告**两次**：`IR006_TYPE_MISMATCH`（"`n_uv.uv` 是 vec2 而 `n_o.color` 要 vec4"）和 `IR032_NO_COERCION`（"没有已知转换"）。计划的去重键是 `(code,node,port)`，于是两条都进了计划，而两个 handler 早已分道扬镳：`IR006` 插入拆分/拼装节点，`IR032` 走 `dropWire`。**先插入、后删除**，删除生效、插入作废 —— 循环报告 `clean`、0 次模型调用，而效果消失了。**"机械修复"如果不是幂等的、可交换的，它就是在和未来的自己打架** | ①`IR032` 改走同一个 `fixTypeMismatch`：它描述的就是同一个缺陷，两个码必须给同一个答案；②去重键改成**站点** `node|port`（节点级诊断仍带 code）—— 一个端口只能挂一条线，所以端口就是修复的单位。回归测试直接断言 `repair()` 对这个双码输入只产出**一个** patch，且其中不含 `disconnect` |
| 35 | `packages/agent/src/repair.ts` | **"类型不匹配"的修复方式曾经是删除连线**。§6.25 把类型系统的缺失半边补成了节点，但修复器不知道：`fixTypeMismatch` 对"源比目标宽"的答案是 `disconnect`。而它删的那个端口**有默认值**，所以文档依然合法、运行依然报成功、用户刚接上的效果**凭空消失**。另一侧也没好到哪去：`vec2 → vec4` 被判成"加宽"，修复器返回**一段说明文字**，于是花一次模型调用请别人做同一件它自己能做的事。**自动修复的价值在于它敢改文档；只敢删东西的修复器把"修复"退化成"降级"，还把证据一起删了** | 两个方向都改成**用节点重建值**：`vec4 → float` 插 `split_vec4` 取 `.x`；`vec4 → vec2` 插 `split_vec4` + `make_vec2`；`vec2 → float` 插 `split_vec2` 取 `.x`；`vec2 → vec4` 插 `split_vec2` + `make_vec4`（未接的 z/w 用目录默认值，不是垃圾）。节点 id 由 `makeNodeId(kind, taken)` 生成，所以同一文档修两次得到同一个补丁 —— 循环的变化检测依赖这一点。不可表达的情形（只有 `vec3` 源）保留为 advice 并说明，因为目录里没有任何节点输出 `vec3`，也没有 `split_vec3`。验证：三个循环级测试（两条方向各一条 + 一条断言"修复后连线仍在、且经过插入的节点"）与一条针对双码的站点去重测试 |

| 36 | `packages/ir`（契约）+ `apps/web/src/preview/glsl.ts` | **无状态着色器做不到"抹开水雾"这件事，而工具从没说过**。用户最早要的效果是"手指抹在水雾玻璃上"：用 `iMouse` 拼出来的擦除只是**跟着手指走的洞**，手一挪就复原 —— 因为 `f(uv, time)` 没有任何地方存放"刚才擦过哪里"。目录里能表达累积吗？能（`mix` + 通道），但通道在预览里永远是 1×1 白纹理，而且**没有任何东西告诉宿主该绑什么**：采样静态图和采样自己上一帧在图上完全同形，所以"要不要反馈"是一个**图之外**的事实 | ①`RenderSpec.feedback?: boolean` —— 文档级声明，源码一个字节不变（这正是它必须显式声明的原因：从图里猜会把每一个既有的通道用户悄悄变成反馈回路）；②`NodeSchema.needs.channel` + `Emission.needs.channel` —— 宿主由此知道**是否有人读通道**，并在 `feedback` 为真却无人读时报 `IR036_FEEDBACK_UNUSED`（宿主为此分配两张纹理、加一趟 pass，画面却不会有任何变化，从外面看和"反馈坏了"一模一样）；③预览实现双缓冲乒乓 + 离屏 FBO，采样与渲染目标**永不重合**；④每个目标的产物头部都写明宿主必须做乒乓（`feedbackNotes`），因为 Unity/UE/Metal 的宿主读的是那个文件，不是这份文档 |
| 37 | `packages/ir/src/rewrite.ts` + `src/catalog/nodes.ts` | **一条注释把"HLSL 会拒绝的写法"说成了"没问题"**。重写器把 `vec4` 改名成 `float4` 却不动参数表，而 `float4(0.0)` 在 HLSL 里是硬错误（`too few elements in vector initialization`）—— GLSL/WGSL/MSL 都接受标量 splat。源码里那条注释恰好讨论过这件事，结论是"MSL 不允许……其实允许，所以不用处理"，把 HLSL 整段漏了。**它不是纸面问题**：`vec_math` 的声明行当时正写成 `vec4(0.0)`，通过了全部结构性检查（括号配平、子集 lint、六目标零诊断），只在把 `dxc` 指向产物时才现形 —— 而那是我自己在这一轮改 `vec_math` 时写下的 | ①`vec_math` 的声明行改成四分量显式形式，并在注释里写明为什么不能偷懒；②新增 `RW013_SCALAR_SPLAT_CONSTRUCTOR`：`lintSubset` 拒绝**单个数字字面量**的向量构造，`fix` 直接给出 `vec4(0.0, 0.0, 0.0, 0.0)`。只针对字面量，所以 `vec4(v)`（拷贝）与 `vec4(a,b,c,d)` 不受影响 —— 规则是精确的而不是保守的；③`patcher.test.ts` 断言 `set_body` 会拒绝它、且长形式与拷贝形式都能通过 |
| 38 | `packages/ir/src/catalog/nodes.ts` | **同一个方向量，走两条路会得到两个不同的向量**。这个 IR 的 `w` 分量是**类型标签**：0 = 方向，1 = 点（`v4()` 与 `color()` 的区别就在这）。`dot` 一直遵守（`dot(a.xyz, b.xyz)`），`vec_math` 的 `length`/`distance`/`reflect`/`cross` 也遵守 —— 只有 `normalize` 不遵守，它归一化四个分量。而 `make_vec4` 的 `w` 默认是 **1**（点），所以 `make_vec4(0.6, 0.8)` 走 `vec_math` 归一化得到 `(0.51, 0.68, 0, 0.85)`，走 `dot`/`reflect` 得到 `(0.6, 0.8, 0)`。两者都是合法 GLSL，没有任何诊断，下游每一个依赖法线的项都轻微偏掉。发现方式是搭"擦水雾"示例时要把 uv 和指针转成 vec4 去算 distance，顺手核对了 `w` 的语义 | ①`normalize` 改成 `vec4(normalize(${a}.xyz), 0.0)`，与同节点其它算子一致；②声明行原本是 `normalize(${a})` —— 一次与 `op` 无关、结果被丢掉的四分量归一化，改成 `vec4(0,0,0,0)` 并说明每个分支都会赋值；③summary 写清楚哪几个算子是三维语义（读 `.xyz`）、哪些是逐分量（含 `w`）；④`catalog.test.ts` 断言五处都读 `.xyz`、声明行不做实事、并且 `dot` 仍然忽略 `w` —— 把两条会漂移的规则钉在一起 |

| 39 | `packages/ir/src/glslSubset.ts` + `catalog/nodes.ts` | **示例文档承诺了一件工具做不到的事**：`liquid-metal` 的 `spec.features` 里写着"无可见的平铺接缝或网格状重复"，而噪声场**不是周期的** —— `hash21(floor(p))` 依赖绝对格子，uv 方格 N+1 与第 1 格对不上。在预览里（一张 uv 方格）永远看不出来，把贴图丢进引擎一平铺就露馅。这是最坏的一类文档缺陷：**承诺写在文件里，缺陷只在别人的工程里出现**。同时它也是一个真实的表达能力缺口 —— "无缝/可平铺贴图"是这类工具最常被要求的东西之一 | ①库新增 `wrap2` 与四个周期化实现 `noise2t` / `gnoise2t` / `fbm2t` / `voronoi2t`：只把**查表**的格子坐标按周期取模，几何保持不折叠（这样接缝处插值精确相等，而不是"两边差不多"）；②`noise`/`fbm`/`voronoi`/`domain_warp`/`raymarch` 各加一个 `tile` 输入（默认 0，所以既有文档一个字节不变）；③`fbm2t` 的周期随八度一起放大，否则只有第一个八度平铺；④**实现只有一份**：`noise2` 等都是传 `on = 0` 的一行包装，`on = 0` 时 `mix_(i, mod_(i, per), 0.0)` 恒等于 `i`，精确无副作用；⑤把 `liquid-metal` 的两个噪声节点打开 `tile`，让那句承诺变成真的；⑥`catalog.test.ts` 4c 断言：凡采样噪声的节点都必须有 `tile`、必须调 `*t` 版本、周期必须来自该节点自己的 scale/frequency，且默认必须是 0 |
| 40 | `packages/ir/src/glslSubset.ts`（函数顺序） | **把包装函数写在被包装函数之前，六个目标全都发射成功，只有真编译器不答应**。GLSL 与 MSL 要求函数**先声明后调用**；我为了把 `fbm2` 改成一行包装（消除两份实现漂移的可能），顺手把包装放在了 `fbm2t` 前面。发射器照常产出、子集 linter 照常通过、`Emission` 的诊断为零 —— 因为这些都是**结构**检查，没有一个会去问"这个符号在调用点可见吗"。抓到它的是 `glslangValidator`，在第 41 条 Python 断言里，一句话：`'fbm2' : no matching overloaded function found` | 把四个周期化实现都放在各自包装之前（并在库里写明顺序是有语义的、而 TypeScript 不检查这一点）。**这条的价值不在于修好了一个顺序**，而在于它是"结构性检查"与"真编译器"之间差别的又一次实证：前者能证明形状，后者才能证明可见性、解析与重载 |

| 41 | `packages/ir/src/types.ts` + `catalog/nodes.ts` + `glslSubset.ts` | **一个声明了却没有用户的能力标志**。`NodeSchema.needs.derivatives` 从第一版就在类型里，`Emission.needs.derivatives` 也一路算下来，而**没有任何节点声明它、也没有任何发射器读它** —— 一整条从类型到发射器的路径，两端都是死的。更糟的是它挡住了真正的问题：程序化着色器最常见的质量抱怨（图案一缩小就闪、锯齿很硬）的答案就是导数宽度抗锯齿，而工具里没有这个原语，作者只能用固定宽度的 `smoothstep` —— 离远了太利、离近了太糊 | ①新增 `antialias` 节点：`smoothstep(-w, w, value - edge)`，`w = fwidth(value) * amount`，过渡宽度永远是一个屏幕像素；②`fwidth` 进 `PORTABLE_FUNCTIONS`（六个目标同名可用）；③`catalog.test.ts` 的规则从"不许有人用导数"改成**对称**的"声明标志与调用导数必须一致"；④荷塘示例的三处硬 mask（两片睡莲叶、花瓣、花心）改用 `antialias(sdf.distance)`，既演示也真的更稳。**为什么这里用导数是安全的**：导数只在统一控制流里有定义，而受限子集禁止数据相关分支 —— 主体里所有 `if` 都是 uniform 参数比较，整个 quad 走同一条路。这是子集换来的保证，不是运气 |
| 42 | `packages/ir/src/emitter.ts` | **接进一个节点根本不读的端口，什么都不会发生，也不会有任何提示**。`shape.uv` 就是这样一个端口：catalog 的 summary 明说它为了管线对称性而被接受、并且被忽略。文档写了，但**诊断层不知道** —— 用户接上去、画面没变、无从判断这是设计还是 bug，和"连线被静默删掉"是同一类失败。发现方式是写了一个探针脚本，把 42 种节点里**每个声明过的输入**拿去和它的 body 对照，列出所有"声明了但从不读"的端口 | ①新增 `IR037_INERT_INPUT` 警告：某个端口被接了线，而该节点的 body 从不读它；②**从 body 派生而不是维护一张名单** —— 未来任何带死端口的节点一上线就被覆盖，而一个还在声明、但 body 不再读的端口会立刻开始报警；③`custom_glsl` 按 kind 豁免（它的 body 是每实例的，只用两个槽是常态），豁免本身也有测试；④WGSL 合规 fixture 里那条 `n_shape.uv` 连线改为不接（它本来就不该接）—— 白名单那条路更省事，但会让 fixture 自带一条永远不起作用的线，还把发现它的检查静音掉 |
| 43 | `packages/ir/src/rewrite.ts` | **HLSL 上的旋转是镜像的**。`mat2_mul_(m, v)` 被降级成 `mul(m, v)`，而 HLSL 的矩阵构造是**行主序**：同一个 `float2x2(c, -s, s, c)` 在 HLSL 里是 GLSL 那个矩阵的**转置**，所以 `mul(m, v)` 得到的是 `Mᵀ·v`，每一次旋转方向都反过来。**证据来自编译器本身**，不是记忆：把 `mul(float2x2(c,-s,s,c), float2(1,0))` 交给 `dxc -Fc` 反汇编，输出是 `(c, s)`；而 GLSL 的 `mat2(c,-s,s,c) * vec2(1,0)` 是 `(c, -s)`，`mul(vec2(1,0), m)` 也是 `(c, -s)`。原来的注释写着"HLSL 矩阵默认列主序存储、`*` 对矩阵是逐分量"，**前半句是错的**（HLSL 默认行主序），后半句对；于是"必须用 `mul`"这个结论对、"参数顺序"这一步错。四个示例里 `rotate` 恰好都是 0 或接近 0，所以仓库里**没有任何东西能显示它** —— 症状只会出现在用户的 Unity 里："同样的文档，图案在引擎里转反了" | ①HLSL 家族改成 `mul(向量, 矩阵)`（参数交换），并列上上面那条反汇编证据；②新增 `RW014_MATRIX_PRODUCT`：body 里出现"矩阵值的 `*`"（左边是 `rot2`/`scale2`/`mat2_rot_` 调用、或本 body 声明的 `mat2/mat3` 局部量）就报错并给出 `mat2_mul_` 的写法 —— 因为 HLSL **根本不接受** `matrix * vector`（`dxc`: "cannot convert from 'float2' to 'float2x2'"），这类 body 会在六个目标里的两个上编译失败；③测试断言参数顺序本身，并明确写出"`mul(m, v)` 就是转置" |
| 44 | `python/tests/test_emitted_artifacts.py`（验证覆盖） | **逃生舱从来没有被真实编译器编译过**。`custom_glsl` 的 body 是运行时文本，而所有编译测试用的都是示例文档 —— 没有一个示例用逃生舱（它本来就是最后手段），于是六个目标里唯一"文本不可枚举"的路径，恰好是唯一没有编译器背书的路径。补上之后立刻现形：一段写 `rot2(0.5) * v` 的片段在 GLSL/WGSL/MSL 上编译通过、在 HLSL 上**编译失败**（见 §6.43），而它在 IR 层是零诊断的 | 新增两条 Python 测试：①一段把**全部**别名（`mod_`/`mix_`/`fract_`/`atan2_`/`mat2_mul_`）都用上的片段，六个目标各发射一次并断言别名没有残留（先剥注释 —— 库的散文里就写着这些别名）；②同一文档交给 `glslangValidator` 与 `dxc`（两个入口、零 warning）。这条测试是**发现**第 43 条的路径，而不只是它的回归测试 |
| 45 | `packages/ir/src/emitter.ts` + `packages/ir/test/patcher.test.ts`（测试隔离） | **一个测试文件的模块级全局写入，让另外两个文件里的 16 个测试失败**。`patcher.test.ts` 在模块作用域调用 `registerTarget('glsl', stub)`，用假发射器替掉真的；`--test-isolation=none` 让所有套件共用一个进程，于是"替掉"影响的是接下来跑的一切。它一直是坏的，只是**取决于测试文件的执行顺序**，直到我新增一个文件才翻面：`targets.test.ts` 与 `wgsl.test.ts` 开始报"the quad entry must still read gl_FragCoord" —— 一条指向发射器、而不是指向肇事者的错误信息。我先尝试用 `before`/`after` 恢复，失败（钩子与并发文件的关系比想象的松） | 给 `EmitOptions` 加一个**真正的注入点** `emitter?: TargetEmitter`：`emit` 优先用它，而不是全局注册表。于是测试（以及任何想试自己 lower 实现的宿主）不必再换全局 —— "换掉再放回去的全局，就是一个会被别的东西看到中间态的全局"。`patcher.test.ts` 改为每次调用注入 stub，`registerTarget` 在该文件里彻底消失 |
| 46 | `packages/ir/src/emitter.ts`（`analyzeGraph`） | **逃生舱里的宿主符号不在能力统计之内**。`needs.surface` 只看 `NodeSchema.needs`，而 `custom_glsl` 的 body 是**自由文本**：一段写 `dot(normalize(iNormal), normalize(iLightDir))` 的片段里，`iNormal` 是只有**表面入口点**才声明的量，可是没有任何节点声明 `needs.surface`，于是发射器照常输出**四边形头**——里面根本没有 `iNormal`。结果是**零诊断 + 编译失败**：文件看起来是完整的，只有引擎会说不。发现方式是逐个宿主符号写一段"读它"的片段，再交给 `glslangValidator`：5 个（`iNormal`/`iWorldPos`/`iViewDir`/`iLightDir`/`iCameraPos`）全部 exit 2，其余 6 个（`iTime`/`iTimeDelta`/`iFrame`/`iMouse`/`iResolution`/`iChannel0`）exit 0 —— 两类符号的差别正好就是四边形头与表面头的差别 | ①把 body 也算进**可达集的能力面**：任何节点的 `body`（今天只有 `custom_glsl` 有）里出现 `SURFACE_HOST_SYMBOLS` 之一，就置 `needs.surface`，于是六个目标自动给出表面入口点，符号真实存在；②`SURFACE_HOST_SYMBOLS` 作为**具名常量**导出（`iCameraPos` 归在这一组而不是屏幕组：四边形宿主从不设置它，声明出来只会给着色器一个恒为 0 的、看起来很正常的 uniform）；③匹配用词边界，作者自己的 `iNormalised` 局部量不会误触发 —— 这一条有测试；④Python 侧补一条：这样的文档在六个目标上发射，并在 `glslangValidator` 下真正编译通过。**顺带一个副作用是好的**：一段片段现在可以绕开目录直接写着色（`dot(normalize(iNormal), normalize(iLightDir))`），逃生舱的表达力又宽了一格 |
| 47 | `packages/agent/src/provider/prompt.ts` | **提示词落后于目录两轮**。`antialias` 与 `tile` 都已经在目录里，而提示词从没提过它们的**用途** —— 目录清单只列出"有这个节点/这个输入"，而请求永远不会自己说出它们的名字：没有人会说"请用导数宽度阈值"，他们说的是"缩小时图案在闪"、"贴到 Unity 里有一道缝"。另外逃生舱的别名清单少了 `dFdx_`/`dFdy_`/`mat3_mul_`，矩阵乘积规则完全没写（而 `RW014` 现在会**拒绝**写错的身体，于是模型白丢一轮），body 能读宿主符号这件事也没说 | ①新增 `CRAFT_GUIDE`（每条程序化着色器迟早会遇到的两件事：没有闪烁、没有接缝），把 `antialias`/`fwidth` 与 `tile` 连同**症状**一起写清楚，并列出哪些生成器带 `tile`；②把 `w` 是类型标签这条规则写进同一节；③`CUSTOM_BODY_GUIDE` 补齐别名清单、加入"绝不用 `*` 写矩阵乘积"（连理由一起）与"body 可以读宿主符号、读了表面五个就自动变成表面文档"；④测试从 `REWRITE_ALIASES` **派生**别名断言，所以清单与实现不可能各自漂移；⑤为让 agent 包能 import `REWRITE_ALIASES`，`@ai4shader/ir` 的 `exports` 增加 `./glslSubset`，并顺手修正 `AGENT_CONTRACT.md` 里那份**写错了的** export 映射（文档写 `src/*.ts`，实际是 `dist/*.js`，而且源码里的相对 import 都带 `.js` 后缀，`src/` 根本不能直接加载） |
| 48 | `apps/web/src/preview/surface.ts`（材质采样） | **图案只出现在物体的一角，地面只有一个十字**。用户报的正是这两句：「场景的屏幕只有一个十字有着色器纹理」「立方体出现只有一个角和这个角的相邻半个面有 shader」。原因不是几何、不是相机、也不是着色器编译 —— 是**映射**：三平面投影用**世界坐标**采样（`texture(uPattern, p.xy * scale)`），而网格以原点为中心，于是**每个轴都有一半坐标是负的**；同时纹理是 `CLAMP_TO_EDGE`。负坐标既不报错也不回绕，而是返回**边缘像素**并被双线性过滤拉伸铺满整片区域 —— 一个"看起来像那么回事的颜色"，所以没有任何东西显得坏了。可见的只有坐标**恰好落进单位正方形**的那部分：立方体上是 `(+,+,+)` 那个角与相邻三个面的正半边（场景模式下物体被抬到 `y∈[0,1.2]`，于是每面正好是"一半"），地面上是 `x∈[0,1]` 与 `z∈[0,1]` 两条一单位宽的带子 —— 两条带子并起来就是那个十字，四条象限各自被压成四个角的像素色。**实测被钳制的顶点比例：球 71% · 环面 74% · 纽结 75% · 立方体 73% · 地面 98%** | 两半缺一不可，所以两边都改：①投影加 `+ 0.5`，让图案中心落在物体原点（于是低密度时整个图案**一次、居中、完全不回绕**地铺在物体上）；②采样器改 `REPEAT`（密度滑杆本来就写着"每世界单位重复几次"，而钳制让它的含义还取决于映射模式，并且悄悄丢掉了大半个物体）；③`pattern-mapping.test.ts` 两条断言把这一对契约钉死 —— 从着色器源码解析投影与偏移、从录制到的 GL 状态读 wrap 模式，再对每个顶点算一遍"是否被交给边缘钳制"，必须为 0。**"带偏移但不回绕"也测过**（36/53/38/29/98）—— 所以修复不是"二者取一" |
| 49 | `packages/agent/src/provider/http.ts` | **`HTTP 200` + "响应体读不出来"，然后建议你检查 API 密钥**。用户报的原始信息是 `AG004_PROVIDER_FAILED … LLM returned a non-JSON envelope (patch:draft, HTTP 200): (the response body could not be read)`。这里有三层错，每一层都值得单独说：①**原因被丢掉** —— 读体失败的 catch 把任何异常都替换成一句 `(the response body could not be read)` 当作"响应体"返回，于是下一步 `JSON.parse` 去解析这句话，报的是**内容**问题（"返回的不是 JSON 信封"），真正的原因（abort、连接被切）已经不存在了；②**读体失败不重试，而连接失败重试两次** —— 同一类事件被当成两类对待，而报告里的现象"时有时无"正是间歇性传输故障的典型特征；③**建议是错的** —— `providerFailure()` 对所有失败都给同一句"检查 base URL / API key / 模型名"，可 `200` 已经证明这三样全对。**根因判断**：60 秒的每次尝试上限覆盖了整次往返（这里没有任何流式，所以上限必须容纳**生成**时间），大补丁回复在慢链路上超过一分钟后 **abort 落在 provider 已经答了 `200` 之后** —— 所以症状是"200 且读不出来"，而且时好时坏 | ①`ProviderError` 带 `kind`（`transport`/`timeout`/`read`/`http`/`envelope`）与 `retryable`，分类发生在**信息还存在的地方**；②新增 `readBody()`：读到了就返回；**是我们自己的上限触发**就报 `timeout` 并说明"provider 已经答了 `HTTP 200`，所以 key/URL/模型都对，只是比上限慢"，默认上限 60s → **120s**（它是挂起连接的护栏，不是服务质量目标）；其他情况报 `read`，带上底层错误自己的名字与消息；③`chat()` 的重试改成按 `retryable` 判断，于是**读体失败和连接失败享受同一条重试预算**；④`AG004` 的 `fix` 按 kind 分支（`read`/`timeout` 明确写"这是传输失败，不是配置问题；重试"）；⑤`loop.ts` 侧断言四条 kind 各自给出对的建议、且不再出现"检查 API key" |
| 50 | `packages/agent/src/provider/http.ts`（两处，由第 49 条的新测试当场抓出） | ①**超时标志是按值捕获的**。`readBody` 需要回答"是我们自己的上限把它掐掉的吗"，而 `timedOut` 是一个在 `chat()` 里 `let` 出来的布尔量；作为对象字段传进去时**在调用那一刻就被复制**，而超时发生在**读体进行中** —— 于是每次超时都被归成普通的 `read`，第 49 条辛苦建立的诊断当场失效。测试「our own per-attempt ceiling firing mid-body is a timeout」在修好之前拿到的 `kind` 是 `'read'`。②**"截断"的判据把 HTML 错误页也算了进去**。原来判据是"结尾不是 `}` 或 `]`"，而 `</html>` 的结尾是 `>` —— 于是一个反向代理的错误页被判成"回复被切断"，对着一个每次都会同样回答的代理**重试三次**。改成"**开头像 JSON** 而结尾不像"：代理页以 `<` 开头，截断的回复以 `{`/`[` 开头 | ①`timedOut` 改成 **thunk**（`() => timedOut`），并把"为什么必须是 thunk"写进类型定义旁边 —— 这类 bug 的特点是**代码看起来完全正确**；②判据改成 `startsWith('{')||startsWith('[')` 且不以 `}`/`]` 结尾；③两条新测试分别钉住"上限触发必须报 timeout 而且不许让读者去查密钥"和"截断要重试、代理页不许重试" |
| 51 | `packages/agent/src/loop.ts` | **`modelCalls=1`，而实际发出去了两个请求**。用户的报错面板写着 `modelCalls=1` 加 `trace=interpret#0` —— interpret 成功、`patch:draft` 失败，而计数器只记了成功那一次：`usage.calls += 1` 写在 `await` **之后**。后果不是数字难看，而是**它把读者引向错误的方向**："模型调用 1 次"读起来像"模型根本没被叫过"，于是去找配置问题 —— 而计数器刚刚才证明配置没问题。同时 `maxModelCalls` 预算是按这个计数判定的，所以失败的那次调用**不占预算**，一个反复失败的 provider 可以在预算不变的情况下无限重试 | `usage.calls += 1` 移到 `await` **之前**（三处：interpret、draft、repair），语义明确为"**发出的请求数**"，不是"成功的请求数"；`loop.test.ts` 断言"第一个调用就抛异常的运行必须报 `modelCalls=1`"。顺带说明白：HTTP provider 内部的传输级重试（最多 2 次）不改变这个计数 —— 循环发起的**一次调用**仍然是一次 |
| 52 | emitter / wgslLower | 多层 custom_glsl 的同名局部变量泄漏，agent clean 而驱动失败 | 节点块作用域，输出提升；覆盖单输出及多个标量输出，WGSL 递归转换块；真实 glslang/dxc 回归 |
| 53 | HTTP patch parser | 缺当前文档，已有节点 set_body 被丢弃 | 解析以 context.doc 为初始状态，HTTP 响应到 applyPatch 回归 |
| 54 | agent loop | 微调未执行、空补丁或被拒补丁被误报完成 | refine 执行请求，repair 保留诊断模式；空修改非 clean；新草稿单次需求复核与 outOfScope |
| 55 | GLSL/WGSL preview | 初帧不为零、时间倒退污染反馈 | iFrame 从 0 起，倒退时清空积累，回归验证 |
| 56 | target hosts | alpha 被保留但宿主不混合 | 六目标产物含 straight-alpha 混合/深度/排序及反馈 pass 例外；ShaderLab 生成 Blend、ZWrite Off、Transparent Queue |
| 57 | HTTP SSE | 非流式大回复在 HTTP 200 后中断 | 默认 SSE，增量 UTF-8/事件解码、空闲超时续期、缺 DONE 丢弃并按预算重试；取消不重试；JSON 网关回退；线上结论见 RAIN_GLASS_TEST |
| 58 | material preview | 非周期图案上模型时出现硬接缝 | seamless 四图融合，fract 前导数；UV 整数周期；默认 UV 4 / 三平面 12 次采样，大尺度重影代价写入 UI/文档；不改变导出 |

另记一条**测试基础设施**的教训：`installDom()` 原本每次调用都新造一个 `localStorage`，于是两个套件各自持有一个对象、而全局只留最后一个 —— 断言说"没写进去"，实际是**测试自己在一个进程里模拟了两个浏览器**。改成整个进程共用一个（正如一个来源只有一个），需要干净状态的套件调 `storage.clear()`。

另记一条**工具事故**（同一类事故在 §6.48 那轮又发生了一次，所以这里补上第二次的经过）：用 PowerShell 的 `Get-Content -Raw` + `Set-Content` 去改 `main.ts`，读按 CP936 解码、写按 CP936 编码，把 UTF-8 里 CP936 无法往返的字节对压成了一个 `?` —— 76 行、约 110 处，中文注释和**用户可见的字符串**都被吃掉了字符，连换行都被吞掉导致几行被焊接在一起。恢复办法是：`apps/web/dist/main.js` 是损坏**之前**投影出来的，而投影只删注释、**字符串字面量逐字节保留**，于是拿它当基准交叉校验修好的文件 —— 139 条非 ASCII 串全部命中，0 缺失。

**第二次（`surface.ts`，拿同一个脚本去删三个 `+ 0.5` 做反向验证）**：损坏的形状一模一样（45 行、非 ASCII 变成 U+FFFD、两处换行被吞），但**上面那个恢复办法这次不可用** —— 发现时 `project-ts` 已经跑过一遍，而它对投影失败的模块会**删掉旧的 `dist` 产物**，那份"损坏前的投影"已经不存在了。恢复用的是另一条路：ASCII 骨架完好，所以写一个**按行号打补丁的脚本**，把 45 行逐一换成正确文本（多行替换也允许，正好覆盖被焊接的两行），并且**每个被替换的行都必须真的含有损坏标记**，否则脚本拒绝运行 —— 这样一张过期的表不可能悄悄改写健康的代码。**两条教训**：①源码绝不走 PowerShell 文本读写（第二次仍是因为想省一步）；②损坏一旦发生，**先别再跑投影**，`dist/` 里的旧产物是唯一的交叉校验基准。


---

## 7. 可复现的命令

```powershell
# 建立 workspace 软链（正常环境用 npm install）
New-Item -ItemType Directory -Force node_modules\@ai4shader
New-Item -ItemType Junction node_modules\@ai4shader\ir  packages\ir
New-Item -ItemType Junction node_modules\@ai4shader\cli apps\cli
New-Item -ItemType Junction node_modules\@ai4shader\web apps\web

# 投影 + 自检
node --experimental-strip-types --experimental-transform-types scripts/project-ts.ts
node scripts/check-imports.mjs

# 测试
cd packages\ir
node --test --test-isolation=none --experimental-strip-types --import ./test/register-hooks.mjs test/*.test.ts
cd ..\..

# 示例文档：由 TS 真源生成，永不手改
node scripts/build-demo.ts

# CLI
node apps/cli/dist/cli.js validate examples/liquid-metal.shader.json
node apps/cli/dist/cli.js emit examples/liquid-metal.shader.json --target all --out dist/demo --force
node apps/cli/dist/cli.js params examples/liquid-metal.shader.json --set u_contrast=2.0

# 工作台（需要能启动浏览器）
node scripts/serve.mjs
#   → http://127.0.0.1:5183/apps/web/index.html
```

`npm` 脚本等价物（`npm test` / `npm run build` / `npm run emit:demo`）已在根 `package.json` 配好，在正常环境里可直接用。

---

## 8. 与 `DESIGN.md` 的差异（设计意图 vs 实际）

| 设计文档说的 | 实际 | 原因 |
|---|---|---|
| 33 种节点 | **42 种** | 原文计数错误（`NODE_KINDS` 当时是 31）；此后新增 `surface` 组 5 种与向量原语组 5 种 |
| `uv.centered` / `voronoi.f2` 等多输出端口 | 已加回：`time.sin/cos`、`polar.angle/radius`、`voronoi.f1/f2/edges/cell`、`sdf.distance/mask` | 多输出解构此前是坏的（§6.7），修好后端口才能加回（§3.6）。**`uv.centered` 与 `tile.cell` 仍未加回**：它们的两个输出都是 `vec2`，而契约要求多输出节点返回一个 `vec4` 且端口必须是标量。合法化需要改契约，不是加端口 |
| 浏览器内 WASM 编译校验（`@webgpu/glslang` / `naga`） | 未接入 | 无网络，装不了 |
| `domain_warp` 接受噪声节点输入 | **不接受**，噪声在节点内部按坐标求值 | 接进来的"单个噪声值"在整幅画面上是常量，无法产生空间变化 —— 这个输入在设计上就是错的。已在节点文档与示例注释里说明 |
| CMA-ES 参数优化器 | 代码完成（对角协方差 CMA-ES），未在浏览器中实测 | 同上 |
| Unity / UE 插件 | 未实现 | 需要对应引擎；发射器已产出它们需要的全部边车文件 |

---

## 9. 下一步优先级

1. **用真实编译器校验 WGSL** —— 降级 pass（`targets/wgslLower.ts`）已实现并通过结构性检查（§4.2），
   但**没有任何编译器说过它合法**。在有 `naga`/`tint` 或浏览器的机器上跑一次即可把它从"未验证"
   变成"已验证"，这是投入产出比最高的一步。
   背景：原先的问题是 `rewrite.ts` 只重命名类型与函数名、不做语法重构，所以生成的库与节点体仍是
   C 风格 GLSL（`f32 hash11(f32 p) {`、`vec2f i = floor(p);`、`for (int k = 0; …)`、以及给不可变的
   参数赋值）。受限子集的语法是封闭的，所以修法是在 `targets/` 下加一个专门的降级 pass，
   而不是继续扩充重命名表 —— 这也是当初把子集收窄的回报之一。
2. **在有浏览器的机器上跑 `verify-shaders.mjs`** —— 这一项的含义本轮又变了：GLSL 与 HLSL 已经由真实编译器验证（§4.1），
   所以它要证明的不再是"能不能编译"，而是**第一帧像素**。而这件事现在有三层分量：发射出的 surface
   着色器已经真的驱动 3D 网格（§6.28）、反馈路径的乒乓逻辑只在录制型 GL 里验证过（§4.1）、
   而"擦开的水雾到底像不像"只有人眼能判断。一条命令能同时验证编译、uniform 绑定与出图。
3. **多输出端口加回目录** —— `time.sin/cos`、`voronoi.f2/edges`、`sdf.distance`、`polar.radius` 等；
   解构已经修好（§6.7），是纯目录工作。`math` 节点目前用 9–14 号算子代为提供 sin/cos。
4. **接入真实 LLM** —— `patcher.ts` 的补丁格式已经是最终形态，接上即可形成 §4 的自动闭环。
5. **Python 侧 LPIPS/CLIP 评分** —— 提升 Critic 的归因质量。

---

## 10. 这次交付没有做的事（明确列出）

### 设计待办：水滴 simulation host 与场景颜色输入

当前 `feedback` 只有上一帧 RGBA8，不能同时承载独立的质量、速度、位置、轨迹和背景。
后续先设计宿主能力与 IR pass 图：明确状态纹理格式/精度、初始化、固定时间步长、
双缓冲依赖与重置；再定义重力、阻力、质量守恒的合并、碰撞驱动的加速/改道及分裂。
背景折射另外需要显式 scene-color/depth 输入和透明排序，不能占用历史状态纹理冒充背景。
验收应覆盖不同帧率下运动一致性、质量守恒、轨迹寿命、重置和多目标宿主绑定。
这是尚未实现的设计条目，不属于现有 rain shader 的已实现功能；不要在状态布局未定义前
增加一个声称支持物理碰撞的节点。

| 没做 | 为什么 |
|---|---|
| 真实 LLM 接入（文本 → IR、对话微调、Critic 归因） | 文本生成与微调现已接入并使用 DeepSeek 实测；SSE 完整复跑见 RAIN_GLASS_TEST.md。视觉 Critic 仍未实现。**接口已就位**：`ShaderPatch` 格式、`applyPatch` 的原子语义、诊断的 `fix` 字段都是为了这个循环设计的 |
| Unity / UE 编辑器插件 | 需要对应引擎。发射器已产出它们要的全部边车文件（引脚清单、材质属性、搭建配方） |
| Python 离线评分服务（LPIPS / CLIP） | 无 Python 依赖可用。浏览器侧已有 SSIM / 直方图 / 频带归因，且归因规则（§4.3）已实现为可执行规则 |
| ~~`custom_glsl` 逃生舱~~ | **本轮已实现**（见 §6.26/§6.27）：`custom_glsl` 节点在 `node.body` 上携带受限子集代码，经 `lintSubset` 校验、由同一个 token rewriter 降到六个目标，`set_body` 补丁操作写入。超出目录表达力的需求现在有出路，不再只能写进 `spec.outOfScope` |
| **多 pass 状态**（除那一个反馈通道之外） | 一次前向着色加上"上一帧"这一个寄存器，覆盖了累积类效果（擦除、拖尾、阻尼反馈）。但**真正的多 pass**——把结果写到若干张中间缓冲、在不同分辨率上做模糊/降采样、用上一 pass 的深度做遮挡——需要的是 pass 图而不是节点图，那是另一套 IR。当前的说法是诚实的：`render.feedback` 只有一个通道，`outOfScope` 里写明做不到的效果（例如荷塘示例里"水滴沿玻璃流下"需要重力方向的累积场） |
| golden 图回归测试 | 依赖能真实渲染的环境 |
| 视频参考输入 | 依赖 P1 的多模态接入 |
