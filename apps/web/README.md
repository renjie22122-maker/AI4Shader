# AI4Shader 工作台 (`@ai4shader/web`)

零框架、**零构建**的浏览器工作台：真实 GPU 编译预览 + 由 IR 自动生成的滑杆面板 +
六格式源码视图 + 参考图评分与参数搜索。

---

## 运行（零构建路径，首选）

本机没有 npm、没有 Vite、也不需要它们。工作台由三个文件支撑：

| 文件 | 作用 |
|---|---|
| `apps/web/src/**/*.ts` | **唯一真源**（类型在这里，人读的也是它） |
| `apps/web/dist/**/*.js` | `scripts/project-ts.ts` 用 Node 自带的类型擦除投影出来的产物，浏览器真正加载的东西 |
| `apps/web/index.html` | 静态外壳，`<script type="module" src="./dist/main.js">` |

```powershell
# 1) 投影 TS -> JS（改完 src 就要重跑；会重建 dist）
node --experimental-strip-types --experimental-transform-types scripts/project-ts.ts
#    根 package.json 里也有：npm run build

# 2) 起静态服务器（必须服务【仓库根】，原因见下）
node scripts/serve.mjs
#    根 package.json 里也有：npm run serve

# 3) 打开
#    http://127.0.0.1:5183/                        ← 直接开这个，会 302 到下面那条
#    http://127.0.0.1:5183/apps/web/index.html     ← 真实地址
```

> **`/` 会重定向到工作台。** 页面之所以住在这么长的路径上，是因为它每一层 import 都是相对
> 路径、要跨出 `apps/web` 走到 `packages/ir/dist`，所以服务器必须服务仓库根（下一节）。
> 让人手打整条路径是更差的交换 —— 打开 `http://127.0.0.1:5183/` 看到一个光秃秃的 404，
> 读起来就是「服务器坏了」。这个误会本项目已经付过一次代价，所以现在根路径直接 302，
> 404 页面也会写明你可能想去的地址。

### 为什么服务器必须服务仓库根

`apps/web/dist/main.js` 通过相对路径引用 IR：

```js
import { emit } from '../../../packages/ir/dist/index.js';
```

这条路径从 `apps/web/dist/` 出发正好走到仓库根，所以文档根必须是仓库根。把服务器指向
`apps/web/` 会让每一条 IR 导入 404。`scripts/serve.mjs` 默认就是仓库根，并在你改 `--root`
时打印警告。

### import map：为什么必须有，以及它曾经害过谁

工作台**有**一个 import map，写在 `index.html` 里，只有两条：

```json
{ "@ai4shader/ir":         "/packages/ir/dist/index.js",
  "@ai4shader/ir/catalog": "/packages/ir/dist/catalog/index.js" }
```

这里原先写的是「工作台**不使用 import map**：所有说明符都是真实的相对路径，Node 与浏览器
解析方式一致」。**这句话的后半句是假的**，而它为一次真实的空白页买了单：

`packages/agent/dist/*.js` 里有 5 处 `import … from '@ai4shader/ir'`（裸说明符）。Node 通过
`node_modules/@ai4shader` 软链解析得了，浏览器不行 —— 它会直接拒绝**整个模块图**：

```
Failed to resolve module specifier "@ai4shader/ir".
Relative references must start with either "/", "./", or "../".
```

后果是 `dist/main.js` **一行都没执行**：页面把静态 HTML 渲染出来（卡片标题都在），
每个面板都是空的。而 Node 侧的每一项检查都是绿的 —— 投影成功、`check-imports` 只检查相对
说明符、启动测试借用 Node 的解析器。**「Node 能解析」被当成了「浏览器能解析」。**

所以现在：

- 这两条是**精确**匹配，不是前缀映射。import map **不做扩展名解析**，写
  `"@ai4shader/ir/": "/packages/ir/dist/"` 会把 `@ai4shader/ir/types` 指向一个没有扩展名的
  文件 `types` —— 404。逐条列出子路径，才能让检查逐条验证。
- `scripts/check-imports.mjs` 会在三种情况下**失败**：裸说明符没有对应条目、条目指向不存在的
  文件、条目没有任何人 import（第二套解析系统就是这样烂掉的）。
- `scripts/check-served.mjs` 会自己起服务器，**沿 HTTP 走一遍真实模块图**，用页面自己声明的
  map 解析每个说明符。它填的是另外两项都够不着的那一层：`.js` 被当成 `text/plain` 发出去，
  浏览器同样拒绝执行，而磁盘上的字节完全正确。
- `apps/web/test/boot.test.ts` 通过 `test/register-hooks.mjs` **让 Node 按浏览器的方式解析**：
  裸说明符只能走 import map，没条目就带着浏览器那句原文报错。测试不能一边借用 Node 的解析器，
  一边声称自己代表浏览器 —— 这条钩子就是把这个差值抹掉。

前两条 specifier 之外的东西仍然全是真实相对路径，`dist/*.js` 依旧既能被浏览器加载、也能被
`node -e "import('./apps/web/dist/doc/demo.js')"` 加载。

> 端口 5183 是本工作台的端口；3080 属于另一个应用，别动。

### 可选：有 npm 的环境

`vite.config.ts` / `vite.config.offline.mjs` / `tools/vite-sandbox.mjs` 保留着，供装了 npm 的
人使用（`npx vite` 走 `src/main.ts`，会重新打包）。零构建路径不需要它们，README 之后提到
「Vite 路径」时指的都是这条可选路线。

一个已知限制：`vite build` 产物的 URL 是带哈希的 `.js`，工作线程的运行时分支判定为
「非 Vite」而去找 `score.worker.js`（构建产物里没有这个文件），于是参考图面板的 worker 会
404。`vite dev` 与零构建路径都正常 —— 这两条才是实际会走的路线。限制写在这里，而不是藏起来。

---

## 怎么用（三步）

打开页面后，左栏第一张卡片就是 Agent 面板：

1. **写一句中文描述**（默认已经填好「液态金属表面，慢速流动，带细腻的拉丝与高光带」）。
2. **选 provider**：
   - **离线**（默认）—— 确定性 stub，不需要 key、不需要网络，会起草一份完整的六目标文档。
     它的用途是让你能**完整看一遍闭环**：理解意图 → 起草图 → 确定性修复 → 发射校验，
     以及每一步的模型调用次数。
   - **在线** —— 真实模型（OpenAI 兼容）。见下面的「配置在线服务」。
3. **点「生成」** → 文档被套用到工作台，右侧代码视图、中间预览、左栏滑杆全部跟着换掉。
   点 **「微调当前」** 是在现有文档上改进；一份已经能干净发射的文档**不会消耗模型调用**，
   面板会把「模型调用 0 次」显示出来 —— 这是让你判断哪部分真的需要模型。

### 配置在线服务

选「在线」后出现四个控件：**服务 / 地址 / 模型 / key**，加 **测试连接** 与 **拉取模型** 两个按钮。

- **服务**是一个下拉框，列出的每一项都已经实测过两件事：网络可达，且**允许浏览器跨域调用**
  （`Access-Control-Allow-Origin` 覆盖本页来源、`Access-Control-Allow-Headers` 覆盖
  `authorization`）。第二件事不是细节：一个用 `curl` 完全正常的端点，如果没开 CORS，
  浏览器照样会拒发这个带 `Authorization` 的 POST。测不过的服务没有列进来。
  目前有 OpenAI / DeepSeek / OpenRouter / Moonshot(Kimi) / SiliconFlow / 阿里云百炼 /
  本地 Ollama / 本地 LM Studio / 自定义。
- 选中一项会**自动填好地址与一个起步模型名**，所以不需要知道任何 URL。地址仍然可以手改；
  手改成一个已知地址会自动选中对应服务，改成别的一律变成「自定义」—— 下拉框不会显示一个
  已经不成立的服务名。
- **模型是一个下拉框**，列出所有已知模型，最后一项是「手动输入…」。这不是 `<datalist>` ——
  `<datalist>` 会**按输入框里已有的文字过滤自己**，所以一个预填了模型名的面板，下拉里只会显示
  那一个名字。第一版就是这么写的，用户看到的现象是"为什么只能选 deepseek-chat"。
  `<select>` 永远显示全部，而「手动输入…」保留了自由文本这条退路（`/models` 不是每个网关都实现，
  「自定义」预设也没有任何提示值 —— 那时只显示输入框，不会给一个空下拉框）。
- **拉取模型**发 `GET {地址}/models`，用你填的 key 把**服务端真实的模型列表**取回来，**整体替换**
  下拉框里的内容，并显示"服务端返回 N 个"。这是唯一权威的来源：硬编码的列表在任何服务上都必然错一头。
  本机实测：OpenRouter 的 `/models` **无需 key** 就返回 **458** 个模型，而 DeepSeek / Moonshot /
  SiliconFlow / 百炼的 `/models` **都要 key**（401）—— 所以这份列表在构建期根本不可能知道，
  只能运行时用你自己的凭据去问。取回后如果你手写的名字不在列表里，输入框会保留它 —— 界面不该和你争。
- **测试连接**发的是**一次真实的单 token `chat/completions` 请求** —— 和正式调用同样的地址、
  同样的 `Authorization` 头、同样的 `content-type`、同样的方法。不用 `GET /models` 是因为
  那是另一个请求：它不带 body、不校验模型名，有些网关还不实现它，于是会在真实调用必然失败时
  报「一切正常」。两个按钮**故意不合并**：模型名写错时，测试连接会 404，而拉取模型会成功并
  把正确的名字递给你 —— 串在一起恰恰会在最需要它的时候把更有用的那个藏起来。
- **key 默认会被记住**：勾选「记住 key」时它写进本机浏览器的 `localStorage`，刷新后自动填回。
  旁边直接写着代价：**这个来源上的任何脚本都能读到它**。取消勾选会**立刻删除**已保存的值
  （不是"下次不读"，是现在就没了），key 随即只留在本页内存里。
  服务 / 地址 / 模型 / 离线还是在线 这四项始终会被记住 —— 它们是配置，每次重打一遍是自己给自己加的税。

  > 这个面板原本**拒绝**保存 key，理由就是上面那句"任何脚本都能读到"。那个理由至今成立，
  > 但它漏算了一件事：留给用户的替代方案是"每次刷新重打一个 50 字符的密钥"，而**一条实际效果
  > 是让谨慎路径变得烦人的规则，不会让人变谨慎 —— 它会让人把 key 粘进一个草稿文件**，
  > 那比 `localStorage` 在任何方面都更糟。所以现在这个取舍由 key 的主人自己决定，
  > 写在做出选择的地方，并且一键可逆。

> **在线服务配错时，失败会指名道姓。** 之前这个面板把 `https://api.openai.com/v1` 写死在代码里，
> 于是在一个连不上这个主机的机器上（很常见，本项目的开发机就是），界面上**没有任何地方**能填别的
> 服务，也没有任何办法从 UI 里修好它 —— 能做的只有重打一遍 key 再点一次「生成」，而拿到的是浏览器
> 那句 `Failed to fetch`。那不是一个配置步骤，那是一堵墙。
>
> 现在：地址可改、常见服务一个下拉框、测试连接把你原本要跑五秒 agent 才知道的事变成一秒的状态码。

其余部分：

- **中间**：真实 GPU 预览、工具栏、参考图匹配（拖一张图进去 → SSIM / 直方图 / 频带归因 →
  可执行的调参建议 + 参数搜索）。
- **左栏滑杆**：由 IR 自动生成，不是手写的。拖动 = 只写 uniform 再画一帧，**不重新发射、
  不重新编译**（`main.ts` 里那条 `// PERF:` 注释说的就是这件事）。
- **右栏**：六个目标的源码（GLSL / WGSL / HLSL / MSL / ShaderLab / UE），按需发射 + 内容哈希缓存。

### 预览模式：平面 / 材质 / 场景

工具栏第一组就是它。**平面**是全屏四边形，看得是图案本身；**材质**把同一份图案当作材质贴在
一个受光的模型上；**场景**再加上地面与接触阴影。

- **形状**：球体 / 环面 / 环面纽结 / 立方体。四个形状各自压不同的失败：球体上的接缝与极点、
  环面上的 UV 换向、纽结上任何投影拉丝、立方体上的硬边法线。地面是场景的地板，不作为可选对象
  —— 那会让相机正对着一张平面的边缘取景。
- **拖拽旋转 / 滚轮缩放**；播放按钮同时冻结图案与自动旋转（"暂停"却让相机继续飘是个意外）。
- **材质条**：映射（三平面 / UV）、密度、金属度、粗糙度、图案强度、曝光。六个滑杆走的都是热路径
  —— 只写 uniform，不重新编译。

它**不改发射器**。图案仍然由原来的后端渲染，只是画进一块 512² 的离屏画布，再作为纹理贴到网格上：

```
平面：  [GLSL/WGSL 后端] ──────────────────────────────► 可见画布
材质：  [GLSL/WGSL 后端] ──► 512² 离屏画布 ──► 纹理 ──► [材质 pass, WebGL2] ──► 可见画布
```

这样做换来的是：IR 契约、六个目标、`emit()` 全都不动，而且 WebGL2 与 WebGPU 两条后端都能用
（材质 pass 只吃像素）。代价写在面板上，不藏起来：

- 图案分辨率上限是 512²，所以极高频率的 shader 在大模型上会变软；
- 图案里如果有屏幕空间效果（比如暗角），它会被**贴**到模型上而不是重新计算；
- 环境光是解析式的（渐变天空 + 太阳），没有 HDRI —— 无网络、无二进制资源。够金属有东西可反射，
  但它是合成的，面板上就是这么写的；
- 地面上的暗斑是**接触阴影的衰减近似**，不是阴影贴图。

参考图评分在材质/场景模式下**测的是屏幕上的画面**，不是图案 —— 图案是材质的输入，拿输入去和
参考图比对会报出一个用户看不到的"匹配"。顺带一个副作用：材质 pass 固定是 WebGL2，所以在
WebGPU 后端下也能评分（平面模式下不行）。

**需要 WebGL2**。没有 WebGL2 时页面不会白屏：会挂一条红色 banner 说明原因，其余面板照常可用
（信息卡片、代码视图、Agent 面板都不依赖 GPU）。切到材质/场景而 WebGL2 不可用时同样会明说，
并留在能看到的状态，而不是留一块空画布。参考图评分与参数搜索**必须** WebGL2 —— 只有
`WebGL2Renderer` 提供 `captureFrame()`。

---

## 已实现并且已验证 / 未验证

验证环境里 **无法运行浏览器**（见文末「验证记录」），所以下面严格区分「已验证」与「未验证」。

### 已验证（Node 侧，可复现）

- `tsc --noEmit` 全绿（`apps/web/node_modules` 里的 TypeScript 5.9.3）。**只覆盖 `src/`**：
  `test/` 需要 `@types/node`，而本 checkout 里根本没装它（详见「验证记录」）。
- **工作台真的能启动**：`apps/web/test/boot.test.ts` 装一个最小 DOM，`import` 真实的
  `apps/web/dist/main.js`，然后断言 `boot()` 跑到底：`index.html` 里每个挂载点都被填上、
  Agent 面板可用、没有 WebGL2 时会挂 banner 而不是抛异常、渲染循环转一帧后仍在排队、
  **点「生成」能真的跑完 Agent 闭环并落到六目标文档**、代码视图有六个 tab、工具栏有
  平面/材质/场景与四个形状、材质条在平面模式下是隐藏的、切到材质模式而 WebGL2 不可用时
  **会新增一条指名原因的 banner 且渲染循环还活着**。
  这条测试补的正是「其它检查全绿但页面是白的」那一格 —— 在这之前，"模块能解析" 和
  "URL 返回 200" 都分辨不出这两种情况。
  配套的 `test/register-hooks.mjs` 让 Node **按浏览器的方式解析**裸说明符（只能走 import map），
  所以这条测试不会再借用 Node 的解析器去替浏览器背书。
- **3D 预览的几何与相机是纯函数，且被单测过**（`test/mesh.test.ts` 8 条、`test/mat4.test.ts` 8 条）。
  这不是形式主义：网格绕序反了、投影矩阵转置了、相机基向量取反了，**这三件事都不会抛异常，
  都会渲染出一张"看起来像那么回事"的图**，而这个环境里没有浏览器可以看。所以断言的是定义本身 ——
  每个三角形的几何法线必须与它顶点的法线同向（绕序检查）、near 面必须映射到 -1 且 far 面到 +1、
  `lookAt` 的基必须正交且看向 -Z、`multiply` 必须按 `投影 × 视图 × 模型` 复合、
  `rotationY` 的行列式必须是 +1（-1 是镜像，网格会里外翻而顶点一个都没动）。
  写这些测试时它们各自抓到过一个真 bug：纽结的 UV 跑到了 2.0，以及环面/纽结整圈绕序是反的。
- **按浏览器的方式解析全部 import**：`node scripts/check-imports.mjs` 检查 81 条 import，
  其中 5 条裸说明符必须精确命中 import map 里的条目、条目必须指向存在的文件、且不能有没人用的
  条目。
- **服务器发出去的那份页面能加载**：`node scripts/check-served.mjs` 自己起 `serve.mjs`，
  沿 HTTP 抓 33 个模块，逐个检查 200 + JavaScript MIME type + 说明符可解析（用页面自己声明的
  map），并确认 import map 出现在 module script **之前**。
- `scripts/project-ts.ts` 投影 14/14 个 web 模块成功。
- `node scripts/check-imports.mjs`：所有相对导入解析成功。
- 演示文档 vs 真实 catalog：**`assertDemoDoc` PASS**，变体「动画 + 色相」，0 处不匹配。
- 六目标发射全部 **0 error / 0 warning**（GLSL 10.5KB、WGSL 11.3KB、HLSL 11.7KB、
  MSL 11.3KB、ShaderLab 13.2KB、UE 12.4KB），每个目标 10 个 uniform。
- 评分链路：`compareFrames` + `lossFrom` 在 256×256 上实测 **3.0 ms/次**（Node，纯 CPU），
  归因函数能产出可执行建议（例：`n_detail.scale ↓` + 「结构位置不匹配」+ 建议值）。

### 未验证（需要浏览器 / GPU）

- **页面是否真的渲染出像素**：沙箱禁止 Chrome 启动（Mojo 需要命名管道），
  详见文末。启动路径已由上面那条测试覆盖，但「第一帧长什么样」仍然没人看过 ——
  `canvas.getContext()` 在测试里返回 `null`，走的正是降级分支。
- WebGL2 真机编译诊断与行号、`captureFrame` 读回。
- WebGPU 的 uniform 布局是否与发射器逐字节一致（表在下面，靠 `getCompilationInfo()` 与
  自动布局的绑定组校验在运行时自证；解析不出来时会明确报错而不是猜）。
- 滑杆热路径的实测帧时间（代码里没有任何 `emit()`/`compile()`，只有 `setUniforms` + `render`，
  见 `main.ts` 的 `// PERF:` 注释，但没有帧率数字）。
- **在线 provider 的真实 HTTP 调用**：请求体的构造、解析、重试、错误处理都有单元测试
  （`packages/agent/test/provider.test.ts`，用一个假的 `fetch`），浏览器里的真实调用则从来没有
  发生过 —— 无网络凭据。**这条缝里曾经藏着一个大缺陷**：单元测试直接构造 provider *并传了
  catalog*，所以谁也看不见"两个宿主都忘了传 catalog"，于是模型发来的每个 `add_node` 都被判为
  未知 kind 并丢弃、整份草稿到达时是空的。现在 `apps/cli/test/author-http.test.ts` 用**真的
  HTTP 服务端 + 真的 CLI 子进程**覆盖了整条链（interpret → draft → parse → repair → emit → 落盘），
  而 loop 会自己把 catalog 交给任何需要它的 provider。

### 诚实的桩 / 降级

- **参考图评分与参数搜索需要 WebGL2**：只有 `WebGL2Renderer` 提供 `captureFrame()`
  （`preserveDrawingBuffer` + `gl.readPixels`）。在 WebGPU 后端下加载参考图会明确提示并切回
  WebGL2。
- **ShaderLab / UE 不做真机编译**：浏览器里没有 Unity/UE。代码视图里会显示
  `targetNotes` 的降级说明，不假装成功。
- **优化器的搜索空间**是数值型 uniform 的 `[min,max]`；没有 min/max 的按
  `value*0.25..value*4`（值为 0 时 `0..1`）兜底，面板与搜索共用同一套推断。
- 优化结果**只作为预览**：滑杆会动、像素会变、`docHash` 不变；只有点「采纳」才会经
  `applyPatch` 写入文档（DESIGN.md §9）。

---

## 文件map

| 文件 | 职责 |
|---|---|
| `src/main.ts` | 应用外壳与状态；唯一的热路径调用点（带 `// PERF:` 注释） |
| `src/ir.ts` | 投影 IR 的**类型化门面**：值来自 `packages/ir/dist`，类型来自 `packages/ir/src` |
| `src/agent.ts` | 投影 agent 包的类型化门面 + `authorOnce()` + provider 选择 |
| `src/providers.ts` | OpenAI 兼容服务的预设表（地址已实测可达 + CORS）、就绪判定 `endpointReady`、真实单 token 请求的 `probeEndpoint` |
| `src/panel/author.ts` | Agent 面板：描述框、离线/在线 provider、服务/地址/模型/key、测试连接、阶段轨迹、停止原因、诊断与 `fix` |
| `src/preview/mesh.ts` | 程序化网格（球体/环面/纽结/立方体/地面）：纯函数、无资源文件、有绕序与不变量测试 |
| `src/preview/mat4.ts` | 轨道相机所需的最小矩阵库（列主序，与 `uniformMatrix4fv` 一致） |
| `src/preview/surface.ts` | 3D 材质 pass：把图案当纹理，Cook-Torrance + 解析环境，轨道相机、地面、接触阴影衰减 |
| `src/preview/renderer.ts` | `PreviewRenderer` 契约 + `ProgramCache` + 工厂 |
| `src/preview/glsl.ts` | WebGL2 渲染器（`gl_VertexID` 全屏三角形、真实编译日志与行号） |
| `src/preview/wgsl.ts` | WebGPU 渲染器（WGSL 结构体反射 → CPU 布局、错误作用域校验） |
| `src/preview/webgpu-types.d.ts` | 手写的 WebGPU 环境类型（项目零依赖，不引入 `@webgpu/types`） |
| `src/panel/params.ts` | 自动滑杆面板（分组、数值框双向同步、节点出处 chip） |
| `src/panel/codegen.ts` | 六目标源码视图（按需发射 + 内容哈希缓存 + 手写高亮） |
| `src/compare/score.ts` | SSIM / 直方图 / 频带 / 归因（纯函数） |
| `src/optimize/search.ts` | 坐标下降 + 对角 CMA-ES + 混合策略 |
| `src/worker/score.worker.ts` | 评分与搜索的工作线程（一次评估 = 一次渲染 + 一次消息往返） |
| `src/doc/demo.ts` | 内置「液态金属」演示文档 + `assertDemoDoc` 运行时校验 |
| `src/doc/catalog.ts` | catalog 装载与运行时完整性检查 |
| `src/style.css` | 全部样式（`index.html` 用 `<link>` 引入） |
| `src/env.d.ts` | 仅为**可选** Vite 路径声明的 `node:url` |

---

## 性能策略（DESIGN.md §6 的落地）

1. **参数热替换**：滑杆 → `renderer.setUniforms({...})` → `render(t)`。
   不 `emit()`、不 `compile()`。见 `main.ts` 的 `onParamChange`，那里有一条 `// PERF:` 注释
   说明为什么这一行不能改。
2. **三级预览**：草稿 256（拖动时）/ 交互 512（停手 150ms 后）/ 成品（用户选定分辨率 + 4 个
   时间点缩略图）。
3. **编译缓存**：`ProgramCache<T>` 以源码内容哈希为键，**失败结果也缓存**（同一份坏代码不会
   被反复送去编译），A/B 两个版本各占一条记录。

---

## WebGPU uniform 布局

WGSL 没有裸 uniform，可调参数必须落在发射器写的结构体里。渲染器**不硬编码偏移**，而是把
`struct` 从发射出的 WGSL 里**读回来**，按 WGSL uniform 地址空间规则重算：

| 成员 | 类型 | 对齐 | 偏移 | 大小 |
|---|---|---|---|---|
| `iResolution` | `vec3f` | 16 | 0 | 12 |
| `iTime` | `f32` | 4 | 16 | 4 |
| `iTimeDelta` | `f32` | 4 | 20 | 4 |
| `iFrame` | `i32` | 4 | 24 | 4 |
| `iMouse` | `vec4f` | 16 | 32 | 16 |
| （可调 uniform，按 `Emission.uniforms` 顺序） | `f32`/`vec2f`/`vec3f`/`vec4f` | 4/8/16/16 | 依次 16 字节对齐 | 4/8/12/16 |

- 结构体总大小按最大对齐（16）向上取整，缓冲区同样按 16 字节分配。
- `vec3f` 16 字节对齐正是上表 `iResolution` 占 12 字节却让 `iTime` 落在 16 的原因 ——
  CPU 侧照抄这条规则，不靠猜。
- 若结构体解析失败、或某个 `Emission.uniforms` 里的名字不在结构体中，渲染器**不猜**：
  它把原因写进 `layoutNote` 与编译日志，UI 显示为「仅编译校验」并回退到 WebGL2 渲染，
  而不是画出一片黑。
- 绑定组按着色器里实际声明的 `@group(0) @binding(n)` 构造，并逐级重试（`layout: 'auto'` 只
  包含着色器静态使用的绑定），失败时给出校验错误原文。

---

## 参数搜索的实测成本

| 环节 | 成本 | 说明 |
|---|---|---|
| `compareFrames` + `lossFrom`（256×256） | **3.0 ms**（实测，Node，纯 CPU） | SSIM（8×8 窗）+ 64 bin 直方图 + 3 级金字塔 + 颜色统计 |
| 渲染 256×256 | < 1 ms（估计） | 参数变化不触发重编译 |
| `captureFrame()` (`readPixels`) | ~1–3 ms（估计） | 256 KB 回读 |
| `postMessage` 往返 | ~0.1 ms（估计） | 一次评估一次往返 |

→ **单次评估约 5–7 ms（估计）**，200 次评估约 1.0–1.4 s。只有 3.0 ms 是实测的，其余受限于
本环境无法运行浏览器；UI 在优化过程中会实时显示**实测**的 ms/次。

---

## 验证记录（本机沙箱）

```
$ node apps/web/node_modules/typescript/bin/tsc --noEmit -p apps/web/tsconfig.json
(无输出，退出码 0)          # 只覆盖 src/，原因见下

$ node --experimental-transform-types scripts/project-ts.ts
✓ @ai4shader/ir: 12/12 modules -> packages\ir\dist
✓ @ai4shader/agent: 10/10 modules -> packages\agent\dist
✓ @ai4shader/cli: 3/3 modules -> apps\cli\dist
✓ @ai4shader/web: 14/14 modules -> apps\web\dist

$ node scripts/check-imports.mjs
✓ packages/ir/dist: 12 modules, 24 relative + 0 bare imports checked
✓ packages/agent/dist: 10 modules, 17 relative + 5 bare imports checked
✓ apps/cli/dist: 3 modules, 2 relative + 9 bare imports checked (Node-only tree, not map-governed)
✓ apps/web/dist: 14 modules, 24 relative + 0 bare imports checked

81 imports checked; 5 bare specifiers resolve through the import
map in apps/web/index.html. 9 more are Node builtins in apps/cli/dist, which no browser loads.

$ node --test --test-isolation=none --import ./test/register-hooks.mjs apps/web/test/*.test.ts
（boot 12 · mesh 8 · mat4 8 —— 共 28 条全绿）

$ node scripts/check-served.mjs
✓ served page: 37 modules fetched from http://127.0.0.1:5199, all 200 with a JavaScript MIME type
✓ import map declared before the module script; every bare specifier resolves through it
ℹ pass 7  fail 0

$ node -e "import('./apps/web/dist/doc/catalog.js').then(...)"
present: 31  missing: 0  targets: glsl,wgsl,hlsl,msl,shaderlab,ue
```

**`tsc` 只覆盖 `src/`，而这是一个缺口，不是选择。** `apps/web/package.json` 没有声明
`@types/node`，本 checkout 里也没有装它（`apps/web/node_modules/@types/` 下只有 `estree`），
所以任何 `node:test` / `node:fs` 代码都过不了检查。同一原因让
`packages/ir/tsconfig.json`（它 `include` 了 `test/**`）与 `apps/cli/tsconfig.json`
（它写了 `"types": ["node"]`）在本机**必然报错**，与代码质量无关。要在正常环境里补上这一格：
`npm i -D @types/node -w @ai4shader/web`，然后把 `"test"` 加进 `apps/web/tsconfig.json` 的
`include`。在那之前，`apps/web/test/` 是被**运行**的，不是被检查的。

**浏览器验证做不到，原因是沙箱而不是代码**：本机 Chrome 一启动就

```
FATAL:mojo\public\cpp\platform\platform_channel.cc:108] Check failed: . : Access is denied. (0x5)
```

Chrome 的 Mojo IPC 需要命名管道，而当前沙箱拒绝创建；Playwright 同样在
`--remote-debugging-pipe` 上 `spawn EPERM`。因此「页面渲染出非空像素」这一条**没有被验证**。
`apps/web/test/boot.test.ts` 是把这条缺口收窄到「只剩 GPU 那一层」的办法：它用最小 DOM 跑
真实的 `dist/main.js`，证明启动路径、面板构建、Agent 闭环都是通的，而把
`canvas.getContext()` 留在 `null` —— 于是它验证的正是降级分支，而不是假装有 GPU。

这也解释了为什么这个测试值得存在：**在它之前，「工作台是白的」和「工作台是好的」在
所有自动化检查眼里长得一模一样。**
