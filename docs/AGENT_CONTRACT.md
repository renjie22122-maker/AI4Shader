# 接口契约 — 并行开发时的文件归属与稳定 API

> 这份文件存在的唯一目的：让多个 agent（或多人）能同时改这个仓库而不互相踩。
> **改这个文件需要同时改所有相关模块**，所以任何修改都必须在 PR 描述里说明。

---

## 1. 文件归属

| 路径 | 归属 | 规则 |
|---|---|---|
| `packages/ir/src/types.ts` | **冻结** | IR 契约。改动会影响全部 6 个目标 + GUI + CLI + 平台插件。只允许**追加**可选字段；删改必须升 `IR_VERSION`。 |
| `packages/ir/src/glslSubset.ts` | **冻结** | 受限子集规则 + 共享函数库。改函数库等于改所有目标的输出，必须跑全量 golden 回归。 |
| `packages/ir/src/rewrite.ts` | **冻结** | 跨语言重写引擎。所有语言差异只许出现在这里。 |
| `packages/ir/src/emitter.ts` | **冻结** | 发射核心。 |
| `packages/ir/src/index.ts` | 共享（已冻结） | 纯 re-export，不含逻辑。 |
| `packages/ir/src/catalog/` | catalog 负责人 | 节点定义。**任何人不得在别处硬编码端口名。** |
| `packages/ir/src/targets/` | targets 负责人 | 6 个发射器 + lowering 契约。 |
| `packages/ir/src/patcher.ts` | patcher 负责人 | 补丁应用。 |
| `apps/cli/` | patcher 负责人 | 命令行。 |
| `apps/web/` | web 负责人 | 浏览器工作台。 |
| `docs/` | 所有人（追加） | `DESIGN.md` 是设计意图，`IMPLEMENTATION.md` 是实现现状。 |
| `examples/` | 所有人（追加） | 示例文档。 |

**"冻结"= 只能通过一次显式的、说明理由的编辑来改。** 如果你认为冻结文件里有 bug，报告它并给出最小复现，不要顺手改掉——其他三个模块正在基于当前行为写测试。

---

## 2. 模块解析（必须精确一致）

三种宿主必须以**完全相同**的方式解析包路径，否则浏览器里能跑、CLI 里挂掉。

```jsonc
// packages/ir/package.json
{
  "name": "@ai4shader/ir",
  "type": "module",
  // The *actual* map points at the projection (`dist/*.js`), because that is what Node and the
  // browser load. It used to be documented as `src/*.ts`, which was never true of the shipped
  // file: the sources are `.ts` and every relative import inside them carries a `.js` suffix
  // (line below), so `src/` is not directly loadable.
  "exports": {
    ".":            "./dist/index.js",
    "./catalog":    "./dist/catalog/index.js",
    "./targets":    "./dist/targets/index.js",
    "./patcher":    "./dist/patcher.js",
    "./types":      "./dist/types.js",
    // The restricted subset's rules and alias table are a public contract, not an internal
    // detail: the agent's prompt quotes the alias list, `lintSubset` enforces the banned
    // spellings, and both must agree with `REWRITE_ALIASES`. A host that wants to check a body
    // before submitting it needs this too.
    "./glslSubset": "./dist/glslSubset.js"
  }
}
```

- **Node CLI**：直接 `import { emit } from '@ai4shader/ir'`，靠 npm workspaces 软链 + `exports` 解析。运行需要 `--experimental-strip-types`（Node ≥ 22.6）。
- **Vite 浏览器**：`vite.config.ts` 里配 `resolve.alias`，把 `@ai4shader/ir` 指到 `../../packages/ir/src/index.ts`，`@ai4shader/ir/catalog` 指到 `../../packages/ir/src/catalog/index.ts`。不要依赖 Vite 自动处理 workspace `exports`。
- 相对 import **必须带 `.js` 后缀**（`./types.js`），因为 `moduleResolution: nodenext` 和浏览器都按真实文件名解析。

### 两个聚合文件的契约

`packages/ir/src/catalog/index.ts` — 只做 re-export，且必须同时提供这两个名字：

```ts
export { CATALOG, CATEGORIES } from './nodes.js';
```

`packages/ir/src/targets/index.ts` — 导入所有目标模块以触发 `registerTarget` 副作用，然后 re-export：

```ts
import './glsl.js'; import './wgsl.js'; import './hlsl.js';
import './msl.js';  import './shaderlab.js'; import './ue.js';
export { emit, registerTarget } from '../emitter.js';
export type { Emission, EmitOptions } from '../emitter.js';
```

**`emit()` 只在 `targets/` 被 import 之后才可用** —— 发射器通过 `registerTarget()` 自注册。任何调用 `emit()` 的宿主都必须先 import `@ai4shader/ir/targets`（或 `@ai4shader/ir/catalog` 之外再加一次）。这是刻意的：IR 包本身不依赖任何目标语言代码路径。

---

## 3. 已冻结 API（各模块必须提供这些精确的名字）

### `packages/ir/src/types.ts`（已实现）

`IR_VERSION`, `NODE_KINDS`, `TYPE_ARITY`, `HOST_SYMBOLS`, `ShaderType`, `NodeKind`, `ShaderNode`, `WireRef`, `ParamValue`, `UniformDecl`, `ShaderGraph`, `Edge`, `ShaderDoc`, `ShaderSpec`, `RenderSpec`, `QualityTier`, `TargetFormat`, `ShaderStage`, `PortSpec`, `NodeSchema`, `Severity`, `Diagnostic`, `PatchOp`, `ShaderPatch`

### `packages/ir/src/emitter.ts`（已实现）

`emit(doc, opts): Emission`, `registerTarget(target, emitter)`, `TARGET_EMITTERS`, `analyzeGraph`, `collectUniforms`, `emitBodies`, `expandTemplate`, `literalOf`, `paramExpr`, `stableHash`, `octavesFor`, `canConnect`, `sanitizeId`, `splitReturn`
类型：`Emission`, `EmitOptions`, `TargetEmitter`, `GraphAnalysis`, `EmittedBody`, `ExpandContext`

```ts
type Emission = {
  target: TargetFormat;
  source: string;
  sidecars?: { name: string; content: string }[];
  diagnostics: Diagnostic[];
  uniforms: UniformDecl[];
  symbols: { name: string; type: ShaderType; origin?: string }[];
  hash: string;
};
```

> **注意**：`Emission` 没有 `ok` 字段。调用方用 `diagnostics.some(d => d.severity === 'error')` 判断成功与否。加 `ok` 字段是不必要的冗余，而且会和 `diagnostics` 有分歧的可能。**不要加。**

### `packages/ir/src/rewrite.ts`（已实现）

`tokenize`, `rewriteSource`, `shimsFor`, `lintSubset`, `isReservedAnywhere`, `safeIdentifier`, `WGSL_SHIMS`, `EMPTY_SHIMS`
类型：`Token`, `TokenKind`, `RewriteResult`

### `packages/ir/src/catalog/index.ts`（catalog 负责人提供）

`CATALOG: Record<NodeKind, NodeSchema>`, `CATEGORIES`

### `packages/ir/src/patcher.ts`（patcher 负责人提供）

`makeNodeId`, `addNode`, `connect`, `disconnect`, `removeNode`, `setParam`, `exposeParam`, `inlineParam`, `normalizeEdges`, `wire`, `applyPatch`, `canonicalJson`, `createDoc`, `hashDoc`, `forkDoc`
类型：`PatchResult`

### `packages/ir/src/targets/index.ts`（targets 负责人提供）

re-export `emit`, `registerTarget`（见上）

---

## 4. 渲染器契约（`apps/web` 与未来平台插件共用）

任何"把生成的 shader 画出来"的宿主都实现这个接口。WebGL2 和 WebGPU 各一份实现；将来 Unity/UE 插件对应各自的原生实现，但签名保持一致，这样评分与优化逻辑可以跨宿主复用。

```ts
interface PreviewRenderer {
  readonly backend: 'webgl2' | 'webgpu';
  compile(source: string, uniforms: UniformDecl[]): Promise<CompileResult>;
  /** 只更新 uniform 值，不重新编译。这是 60fps 滑杆的关键。 */
  setUniforms(values: Record<string, number | number[] | boolean>): void;
  render(t: number): void;
  resize(w: number, h: number): void;
  dispose(): void;
}

type CompileResult = {
  ok: boolean;
  log: string;                    // 驱动原始日志，逐字保留
  diagnostics: { line?: number; column?: number; message: string; severity: 'error' | 'warning' }[];
  ms: number;                     // 编译耗时，UI 要显示
};
```

**性能契约（不可违反）**：
- `setUniforms` 之后必须能在**同一帧内**重绘，且**不得**触发 `compile()`。
- 滑杆交互路径上不允许调用 `emit()`。只有图结构变化、质量档变化、目标切换才重新发射。
- 编辑器里任何"拖动时卡顿"的问题，第一嫌疑都是误在热路径上调了 `emit()` 或 `compile()`。

---

## 5. 诊断码命名空间

不同阶段用不同前缀，便于 agent 按类别做失败归因和重试策略：

| 前缀 | 阶段 | 例子 |
|---|---|---|
| `IR0xx` | 图与类型校验 | `IR006_TYPE_MISMATCH`, `IR007_CYCLE` |
| `RW0xx` | 受限子集 lint 与重写 | `RW001_BANNED_TOKEN`, `RW003_MATMUL_ARITY` |
| `PT0xx` | 补丁应用 | `PT001_UNKNOWN_OP` |
| `EM0xx` | 发射器注册与装配 | `EM000_UNKNOWN_TARGET` |
| `C0xx` | 真实编译器回报（GLSL/WGSL/DXC/Metal） | 由驱动日志解析而来 |
| `R0xx` | 渲染期（链接失败、绑定缺失、首帧为空） | `R403_BANDING` 类视觉问题 |
| `S0xx` | 评分与归因 | `S001_LOW_FREQ_MISMATCH` |

**规则**：诊断的 `fix` 字段必须给出**可执行的下一步**（推荐某个具体输入名或操作），而不是"请检查你的代码"。这个字段是 LLM 修复循环唯一真正消费的东西。

---

## 6. 版本兼容策略

- `IR_VERSION` 在 `types.ts` 里。任何破坏性变更（删字段、改语义、改默认值语义）都要升 minor 并写迁移函数。
- `ShaderDoc.hash` 只覆盖**影响代码生成的内容**：`spec` / `stage` / `graph` / `render` / `targets` / `irVersion`。**不覆盖** `id` / `title` / `createdAt` / `updatedAt` / `provenance`。否则改个标题就会让所有编译缓存失效。
- 加节点种类、加目标格式、加诊断码都是**兼容**变更，不需要升版本。
- `catalog` 里给某个输入改默认值是**兼容**变更，但会让旧文档的 hash 变吗？不会—— 已落盘的文档里默认值是显式展开的（`emitBodies` 在缺失时用 `port.default` 补），所以改动默认值只影响新文档。这是刻意的：老文档的可复现性优先于"改默认值能追溯生效"。

---

## 7. 提交前的最低验证

```powershell
npm test                                   # 全部工作区测试
node --experimental-strip-types apps/cli/src/cli.ts validate examples/liquid-metal.shader.json
node --experimental-strip-types apps/cli/src/cli.ts emit examples/liquid-metal.shader.json --target all --out dist/demo
cd apps/web; npx tsc --noEmit; npx vite build
```

四条都要过。`emit --target all` 的输出里不允许出现任何 `severity: 'error'` 诊断。
# 宿主校验与在线传输补充（2026-09-27）

`AgentOptions.validate?: (doc, emissions) => Promise<Diagnostic[]>` 是可选的宿主校验钩子。
每次各目标发射零错误后调用；错误并入修复上下文，补丁应用后重新发射和校验。
浏览器在线模式注入 `createFragmentValidator()`，用独立 WebGL2 context 编译 GLSL
片元源码，驱动日志与对应源码行进入诊断。它不检查链接、视觉效果或其它引擎运行。
WebGL2 不可用时跳过；因此 `clean` 只代表已执行检查通过，不能表示视觉验收通过。

`mode: 'refine'` 执行有 provider 和用户请求的现有文档修改；`mode: 'repair'` 只修诊断。
有效请求的空修改及被拒补丁不能算完成。`reviewRequirements: true` 为新草稿增加一次
源码需求复核，必须保留原始需求和明确的 `spec.outOfScope`，不能声称看过画面。
HTTP 的 patch 解析以 `context.doc` 为初始状态，使已有节点的 `set_body`/连接能够保留。

诊断使用完整字符串作为身份，不能只按数字前缀归类：

| 完整码 | 含义 |
|---|---|
| `AG001_NO_JSON` / `AG001_NO_PROVIDER` | 分别为回复缺 JSON / 未配置 provider |
| `AG002_BAD_JSON` / `AG002_TARGETS_NOT_REGISTERED` | 分别为 JSON 无法解析 / 目标未注册 |
| `AG003_TRAILING_COMMA` / `AG003_TARGET_UNAVAILABLE` | 分别为容错移除尾逗号 / 目标不可用 |
| `AG005_SPEC_FIELD` | 意图规格字段不合法，与 AG005_NO_EDIT 不同 |
| `AG004_NOT_OBJECT` | JSON 顶层不是对象 |
| `AG004_PROVIDER_FAILED` | 模型调用失败，修复建议按 ProviderError.kind 分类 |
| `AG004_HOST_VALIDATION` | 宿主校验钩子抛出异常 |
| `AG005_NO_EDIT` | 用户请求没有产生有效修改 |
| `AG006_SPEC_NO_FEATURES` | 意图规格没有特征 |
| `AG030_GLSL_COMPILE` | 浏览器真实 GLSL 驱动编译失败（原名 AG006_GLSL_COMPILE，已避开撞号） |

HTTP 默认发送 `stream: true`，按 SSE 事件增量消费 UTF-8 body；只有 `[DONE]` 完成的
正文才交给 JSON/patch 解析。断流/缺 DONE 丢弃部分回复，并遵循既有传输重试预算；
格式错误、服务端错误事件和 length 截断不能作为成功补丁。调用方取消不重试。
默认 120 秒超时对 SSE 是无数据超时，每个网络数据块续期；非流式 JSON 回退仍为
整次请求上限。`stream: false` 支持 JSON-only 网关；服务端即使返回 JSON 也可正常解析。
传输重试最多两次，仍计为一次循环模型调用；流式消费不等于 UI 逐 token 展示。
SSE 不能保证代理或服务永不掉线，线上通过必须另有实测记录。

`HttpProviderOptions.includeUsage: true` 在流式请求中加入
`stream_options: { include_usage: true }`，用于需要显式索取 token usage 的网关。
默认关闭以兼容拒绝未知字段的服务；`stream: false` 时不发送该字段。没有返回 usage
表示未知，不能当成零消耗。旧诊断保留完整码以兼容既有消费者；新码不得只复用数字前缀。
