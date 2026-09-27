# Target lowering contract

Six emitters, one framework. This document is what a contributor needs before touching
`_shared.ts`: what each language's entry point looks like, which symbols the emitter
declares, what the host must bind, **where the Y flip happens**, and — just as important —
**what is not supported and why**.

Everything here is enforced by `packages/ir/test/targets.test.ts`. If you change an
emitter, that suite is the contract's executable half.

---

## 1. Assembly

Every target produces its source from the same five slots, in this order:

| slot | content | produced by |
|---|---|---|
| 1. prologue | version/pragma, uniform block, samplers, I/O structs | the emitter, written in the target language |
| 2. shim block | `rw(shimsFor(target))` — WGSL only today | `rewrite.ts` |
| 3. library | `rw(GLSL_LIBRARY)` with `OCTAVE_LIMIT` already substituted | `rewrite.ts` |
| 4. bodies | `rw(bodies.lines.join('\n'))` — the graph, straight-line | `rewrite.ts` |
| 5. epilogue | the entry point: computes `uv`, evaluates the graph, writes the fragment | the emitter |

`assemble()` joins non-empty slots with blank lines. Prologues and epilogues are **never**
passed through `rw()`: they are already target-native, and several of them legitimately
contain tokens the restricted subset bans (`uniform`, `struct`, `gl_FragCoord`,
`sampler2D`, `SV_Target`, …). Rewriting them would emit `RW001_BANNED_TOKEN` errors for
code the emitter itself wrote.

Three transformations are applied *around* `rw()`; they are the only places a target may
touch a token stream:

1. **pre-rewrite fixups** (`PRE_REWRITE_FIXUPS`) — `mat2_rot_` call sites become `rot2()`
   calls. `rewrite.ts` lowers `mat2_rot_` to the target's matrix **constructor**, and
   `mat2(a)` is a diagonal scale, not a rotation: silently wrong in all six targets.
   `rot2()` is defined in the shared library and is correct everywhere.
2. **texture-fetch expansion** (`expandFetchCalls`) — the subset's `tex2D_(uv)` alias has
   one argument because the subset has no samplers; each target expands it into its real
   fetch call with the texture/sampler the emitter declared.
3. **identifier remapping** (`substituteIdentifiers`) — host symbols and uniform names,
   token-aware, comment-safe, member-access-safe. Runs *after* `rw()` so the replacement
   text is not rewritten again.

`lower()` also runs the rewriter to a **fixpoint** (at most three passes, normally two).
`rewrite.ts`'s `lowerMatrixMul` inlines a helper call by splicing the *verbatim* token
text of its arguments, so a type constructor inside those arguments is not renamed by the
first pass:

```glsl
// catalog body (`transform`):
return mat2_mul_(rot2(a), (uv - vec2(0.5, 0.5)) * vec2(2.0, 2.0));
// first pass on hlsl — `vec2` survives *inside* the inlined arguments:
return mul(rot2(a),  (uv - vec2(0.5, 0.5)) * vec2(2.0, 2.0));
```

Re-running the rewriter is safe because target names (`float2`, `vec2f`, `f32`, `frac`, …)
are never source names, so a second pass can only pick up text the first pass produced.
The extra passes go through `rewriteSource()` directly, not through `args.rw()`, so lint
diagnostics are not reported twice; only new `error` diagnostics are surfaced. This is a
workaround for a `rewrite.ts` defect (§8, item 2) — the proper fix is to run the spliced
arguments through the rename tables inside `lowerMatrixMul`.

### Determinism

`emit(doc, {target})` twice ⇒ byte-identical `source` and sidecars. Rules for new code:

* no `Date.now()`, no `Math.random()`, no `crypto`;
* never `Object.keys()`/`for…in` over an object when the result reaches the output —
  iterate `uniforms[]`, `bodies.lines`, `catalog[..].inputs`, or an explicitly ordered
  array literal;
* every generated identifier derives from a node id or a uniform name;
* sidecars are plain `JSON.stringify(value, null, 2)` over arrays in IR order.

---

## 2. Uniform declarations

IR uniform order (`emission.uniforms`, from `collectUniforms`) is preserved **exactly** in
every target. Only the placement differs:

| target | where | example |
|---|---|---|
| `glsl` | `uniform` globals | `uniform float u_scale; // default 3.0` |
| `wgsl` | members of `struct Uniforms`, after the five host fields | `u_scale: f32,` |
| `hlsl` | members of `cbuffer AI4ShaderParams : register(b0)` | `float u_scale; // default 3.0` |
| `msl` | members of `struct AI4ShaderUniforms` (`[[buffer(0)]]`) | `float u_scale; // default 3.0` |
| `shaderlab` | ShaderLab `Properties` + `_`-prefixed globals | `_u_scale ("Scale", Range(0.5, 8)) = 3` |
| `ue` | **function parameters** = Custom node input pins | `MaterialFloat u_scale` |

Defaults are rendered with `literalOf()` (renamed per target by `targetLiteral()`). They
are emitted as **trailing comments**, not as initialisers, except in the ShaderLab
property block — which is a real, engine-visible default. Two reasons: GLSL ES 3.00
uniform initialisers are not portable across drivers, and WGSL/HLSL struct–cbuffer
members have no initialiser syntax at all. The host is expected to upload every uniform
listed in `emission.uniforms` before the first draw.

`collectUniforms()` only scans `node.params`. A catalog port whose `default` is a
`{kind:'param'}` value is therefore *referenced* by the emitted body but never
*declared*: port defaults must be literals, uniforms must be node params. (Reported as a
catalog/emitter contract note; see §9.)

---

## 3. Coordinate convention and the Y flip

The IR is GLSL-like: origin **bottom-left**, `uv ∈ [0,1]`, RGBA, no sRGB. A target whose
framebuffer origin is top-left must flip **exactly once**, in its entry-point epilogue.
Never in a node body, never in the library, never twice.

| target | flip | where |
|---|---|---|
| `glsl` | none | `gl_FragCoord` is already bottom-left |
| `wgsl` | **yes** | `fs_main`: `1.0 - pos.y / U.iResolution.y` |
| `hlsl` | yes, once | `PSMain`: `1.0 - input.uv.y`. `VSMain` deliberately emits a **top-left** screen uv, so the mandated PS flip restores bottom-left. |
| `msl` | **yes** | `fs_main`: `1.0 - in.position.y / U.iResolution.y` |
| `shaderlab` | none | `uv` comes from the mesh UV stream (`appdata.uv`), whose origin is bottom-left in Unity |
| `ue` | none | `uv` is an input pin fed by a TextureCoordinate node (bottom-left UV0) |

The tests assert the flip appears exactly once per target (`1.0 - pos.y`,
`1.0 - input.uv.y`, `1.0 - in.position.y`) and does not appear at all in GLSL, ShaderLab
or UE.

---

## 4. Per-target reference

### 4.1 `glsl` — GLSL ES 3.00 / WebGL2

This is the only target expected to compile and run as-is.

**Header split contract (consumed by `apps/web`).** The first three lines are *exactly*:

```
#version 300 es
precision highp float;
precision highp int;
```

Nothing precedes line 1 — no comment, no blank line, no BOM. Line 4 is blank and the
remainder is a complete fragment body starting with `uniform vec3 iResolution;`. The
guaranteed split rule is:

```ts
const lines = source.split('\n');
const header = lines.slice(0, 3).join('\n');  // #version + 2 precision lines
const body = lines.slice(4).join('\n');       // fragment shader body
```

A host that supplies its own vertex shader concatenates `header` + its own body-level
code, or prepends these three lines to `body`.

**Declares:** `iResolution` (`vec3`), `iTime`, `iTimeDelta` (`float`), `iFrame` (`int`),
`iMouse` (`vec4`), one `uniform` per IR uniform, `iChannel0` (`sampler2D`), and
`out vec4 fragColor;`.

**Entry point:** `void main()`. `uv = gl_FragCoord.xy / iResolution.xy`.

**Not supported / caveats:**

* No vertex stage: the IR is fragment-only by construction (`ShaderStage`).
* Unused host uniforms are optimised away by the driver, so
  `getUniformLocation('iChannel0')` legitimately returns `null` for a graph with no
  texture node. The runtime must tolerate missing locations.
* Uniform defaults are comments, not initialisers (see §2) — upload everything.

### 4.2 `wgsl` — WebGPU (lowered by `wgslLower.ts`, no longer "token-rename only")

**Declares** (mandatory block, emitted verbatim):

```wgsl
struct Uniforms { iResolution: vec3f, iTime: f32, iTimeDelta: f32, iFrame: i32, iMouse: vec4f, <uniforms…> };
@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var iChannel0: texture_2d<f32>;
@group(0) @binding(2) var iChannel0Sampler: sampler;
```

**Entry point:** `@fragment fn fs_main(@builtin(position) pos: vec4f) -> @location(0) vec4f`.
The Y flip lives here and nowhere else.

**Host remapping:** every host symbol and uniform is namespaced into the struct
(`U.iTime`, `U.u_scale`). `iFrame` is `i32` in the struct, so its placeholder expansion
produces `f32(U.iFrame)` wherever a node uses it in float position; the library's
hash/noise code never reads it.

**What is valid:** the whole module — prologue, shim block, shared helper library, node
bodies and `fs_main` — subject to the unverified list below. `checkWgslStructural()`
verifies braces, `var`/`let`/`fn` declaration form, absence of `#`, absence of GLSL type
names and `*_` aliases, and absence of `texture(`.

**Who does what.** `rewrite.ts` still performs only the *token* renames (`float` → `f32`,
`vec2` → `vec2f`, `mix_` → `wf_mix`, …). Everything WGSL does differently in *syntax or
arity* is done by `targets/wgslLower.ts`, which `wgslEmitter` runs over the library and the
node bodies:

1. **Function signatures** — `f32 hash11(f32 p) { … }` → `fn hash11(p: f32) -> f32 { … }`;
   `void` loses the arrow. Parameters are typed and in `name: type` order.
2. **Local declarations** — `vec2f i = floor(p);` → `var i: vec2f = floor(p);`. The type is
   spelled out because `checkWgslStructural()` requires the `var name: T` form.
3. **Loop headers** — `for (i32 i = 0; i < 5; i++)` → `for (var i: i32 = 0; i < 5; i = i + 1)`.
   Bounds must be literals; anything else is a `TGT003` error rather than a guess at `i32`
   vs `f32`.
4. **Immutable parameters.** WGSL parameters are `let`-like, so every parameter a body
   writes — directly (`p = …`), by component (`p.x = …`) or by `p++` — is copied into a
   `var p_v: T = p;` at the top of that body, and only that body is renamed to use the copy.
5. **Integer literals.** The literal argument of `i32(…)` and the initialiser/bound of an
   integer loop stay integral (`i32(5)`, `i < 8`); every other literal keeps its decimal
   point. This repairs what `rewriteNumber` cannot know (it appends `.0` everywhere).
6. **The scalar shims are inlined.** `wf_mix`/`wf_mod` are declared scalar-only and WGSL has
   no overloading, but the library calls them with `vec2`/`vec3`/`vec4` arguments, so each
   call site becomes `a + (b - a) * t` / `a - b * floor(a / b)` (arguments parenthesised
   when they are not atomic). `wf_atan2` is scalar-only and stays a call.
7. **Scalar bounds of vector builtins.** GLSL declares `clamp(genType, float, float)`,
   `step(float, genType)` and friends; WGSL's table does not (gpuweb/gpuweb#4815), so
   `clamp(v3, 0.0, 1.0)` is lowered to `clamp(v3, vec3f(0.0), vec3f(1.0))` — but only when a
   conservative shape analysis can *prove* the vector's arity. An unprovable arity is left
   untouched (see the residual risks below).

Anything the pass cannot lower is reported as an `error` diagnostic in the `TGT0xx` range
with a `fix` that names the construct (`?:`, non-literal loop bounds, arrays/bitwise/`%`,
a braceless `if`, assignment to `uv`/a uniform, …). A diagnostic is the correct outcome;
plausible-looking wrong WGSL is not.

**Still unverified — no WGSL compiler exists in this repository.** The module passes
`checkWgslStructural()`, `structuralBalanceProblems()`, a call-arity check against every
`fn` it defines, "no parameter is assigned" and "no local declared twice" checks, and the
catalog-wide assertions in `test/wgsl.test.ts`. What only a real `tint`/`naga` compile can
settle is listed in the module header of `src/targets/wgslLower.ts` (note 5) and the test
file: the scalar-broadcast proof, `vecNf(scalar)` constructor legality, `textureSample`
uniformity, and every *type* the structural checks do not model.

### 4.3 `hlsl` — D3D11/12, self-contained

**Declares:** `cbuffer AI4ShaderParams : register(b0)` (`iResolution`, `iTime`,
`iTimeDelta`, `iFrame`, `iMouse`, then the IR uniforms), `Texture2D iChannel0 : register(t0)`,
`SamplerState iChannel0Sampler : register(s0)`, `PSInput`/`VSOutput`.

**Entry points:** `PSMain(PSInput) : SV_Target` and `VSMain(uint id : SV_VertexID)`, which
draws the fullscreen triangle from vertex ids alone — the file compiles without any
engine header. With 3 vertices and no vertex buffer bound, ids 0/1/2 give the uv corners
`(0,0) (2,0) (0,2)`.

**Sidecar:** `hlsl-bindings.json` — register assignments (`b0`, `t0`, `s0`), the cbuffer
member list in declaration order, entry-point names/profiles and the uniform defaults.
Byte offsets are intentionally **not** emitted: the cbuffer relies on HLSL default packing
rules, and guessing std140-style offsets would be worse than omitting them. Add
`packoffset` if a host needs a fixed layout.

**Not supported / caveats:**

* `tex2D_(uv)` expands to `iChannel0.Sample(iChannel0Sampler, uv)` — the texture and
  sampler must be bound even for a graph that does not use them.
* `iChannel0` being unbound in a "no texture" graph is the host's problem: HLSL will
  still demand a resource at the declared register.

### 4.4 `msl` — Metal

**Declares:** `#include <metal_stdlib>`, `using namespace metal;`,
`struct AI4ShaderUniforms { … }` bound as `constant AI4ShaderUniforms& U [[buffer(0)]]`,
`texture2d<float> iChannel0 [[texture(0)]]`, `sampler iChannel0Sampler [[sampler(0)]]`,
`struct FSIn { float4 position [[position]]; }`.

**Entry point:** `fragment float4 fs_main(FSIn in [[stage_in]], …)`. Y flip once, from
`in.position`.

**int/float strictness:** Metal has no implicit int↔float conversion, so every int→float
use is an explicit constructor call — in the library (`float(i)`), in coerced wires, and
in this emitter's identifier mapping (`iFrame` → `float(U.iFrame)`).

**Sidecar:** `ai4shader.metal.json` — buffer/texture/sampler indices and the struct member
order the host must reproduce. Build the struct from *these* members; do not assume a
std140 block.

**Not supported / caveats:**

* Struct offsets follow Metal's buffer layout rules for this member order
  (`float3, float, float, int, float4, …`). The host must build the same struct, not a
  std140-style block.
* Float matrix constructors in the shared library (`rot2`/`scale2` build a `float2x2` from
  four scalars) rely on Metal accepting the flat-scalar matrix constructor. That is
  **unverified here** — no Metal compiler was available. If it is rejected, `rot2`/`scale2`
  need a `float2x2(float2, float2)` form, which is a library edit, not an emitter edit.

### 4.5 `shaderlab` — Unity

Emits a complete `.shader`: `Shader "AI4Shader/<title>"`, `Properties`, one `SubShader`
with `Tags`, one `Pass` with `HLSLPROGRAM`, `#pragma vertex vert`, `#pragma fragment frag`
and either `#include "UnityCG.cginc"` (built-in RP) or
`#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"` (URP,
detected from `render`, `spec.category`, `spec.tags` or a `targetNotes.shaderlab` note).
URP also swaps `UnityObjectToClipPos` for `TransformObjectToHClip` and adds
`"RenderPipeline"="UniversalPipeline"`.

**Host symbol remapping** (applied token-wise, documented in a header comment in the
generated file):

| IR | Unity |
|---|---|
| `iResolution` | `_ScreenParams.xy` |
| `iTime` | `_Time.y` |
| `iTimeDelta` | `unity_DeltaTime.x` |
| `iFrame` | `floor(_Time.w * 20.0)` — `_Time.w == t*3`, so ×20 gives a 60 Hz counter |
| `iMouse` | `_MousePos`, a declared `float4` (Unity has no global mouse) |
| `iChannel0` | `_Channel0` + `_Channel0_ST`-aware `tex2D(_Channel0, uv * _Channel0_ST.xy + _Channel0_ST.zw)` |
| `u_scale` | `_u_scale` (ShaderLab property name) |

**Entry point:** `fixed4 frag(v2f i) : SV_Target`, with `vert` doing
`UnityObjectToClipPos(v.vertex)`. `uv` is the mesh UV: no Y flip (bottom-left already).

**Sidecar:** `Material.ai4shader.yaml` — one entry per uniform with `kind`, `label`,
`value`/`min`/`max`, and the original uniform name, ready for a host adapter calling
`Material.SetFloat`/`SetVector`.

**Not supported / caveats (also written into the generated file):**

* **No Unity equivalent for the mouse**: `_MousePos` stays zero unless a C# host drives it.
* **`iFrame` is an approximation**, not a real frame index (Unity exposes no frame count to
  shaders).
* **`unity_DeltaTime` may read 0** outside the built-in/URP camera loops.
* **No int or bool property kinds**: `int` becomes `Float`, `bool` becomes `Toggle`
  (as specified). If a Unity version rejects `Toggle`, use `Float` 0/1.
* **Matrix uniforms have no property kind** and are declared as plain globals.
* **The vertex stage uses the mesh UV stream.** Screen-space UVs would need a separate
  pipeline and are not faked.

### 4.6 `ue` — Unreal Engine material Custom node

A Custom node cannot receive uniforms: every parameter must be an input pin. So each IR
uniform becomes a named `ai4shader_main` argument, and the sidecars are the pin list and
the build recipe.

**Entry point:** `MaterialFloat4 ai4shader_main(float2 InUV, <pin per uniform>)`, followed
by a trailing `return ai4shader_main(UV, …);` so the pasted text is complete on its own
(UE compiles function definitions followed by a final `return`).

**Sidecars:**

* `CustomNode.inputs.json` — an ordered array of `{name, type, defaultValue, label}`,
  `type ∈ {"float","float2","float3","float4"}`; the exact pin list, `UV` first.
* `MaterialGraph.ai4shader.md` — the numbered build recipe (Custom node → `CMOT Float 4`
  → pins → paste → wire) plus the literal code block to paste.

**Host symbol remapping:** `iResolution` → `View.ViewSizeAndInvSize.xy`, `iTime` →
`View.GameTime`, `iTimeDelta` → `View.DeltaTime`, `iFrame` → `float(View.FrameNumber)`,
`iMouse` → `View.MousePosition`. This is stated in a header comment in the generated file,
together with the warning that **`View.GameTime` is only valid inside a material function
or preview context** where the view uniform buffer is bound.

**Not supported / caveats:**

* **No uniforms at all** — nothing can be re-tuned without editing the material asset.
* **`View.MousePosition` may not exist** in every engine version; replace it with a pin.
* **Texture nodes** need a Texture Object pin *and* a Sampler pin. The sidecar schema only
  expresses float pins, so `iChannel0`/`iChannel0Sampler` are referenced by name and must
  be added by hand.
* **int/bool pins are floats**; the implicit conversion is the host's business.
* **Loops are unrolled** by the material compiler: a large `OCTAVE_LIMIT` costs compile
  time and instruction slots.
* Nothing here was compiled against a real UE installation — treat the emitted HLSL as
  reviewed-by-construction, not verified.

---

## 5. Self-checks available to hosts

```ts
import { checkTargetLeaks, checkWgslStructural, assertWgslStructural } from '@ai4shader/ir/targets/_shared.js';

checkTargetLeaks(src, 'hlsl')   // [] = no GLSL type name / *_ alias / bare texture( outside comments
checkWgslStructural(src)        // [] = braces, var/let/fn form, no '#', no leaks, no texture(
assertWgslStructural(src)       // throwing form
```

`checkTargetLeaks` ignores comments and member accesses, and knows that `[[texture(0)]]`
is an MSL attribute rather than a fetch call. It is a *promise checker*, not a compiler:
an empty array means "nothing this package promises was violated", not "a driver will
accept this".

---

## 6. Adding a target

1. Write `const myEmitter: TargetEmitter = (args) => { … }` in `_shared.ts`, using
   `libraryGlsl(args.doc)`, `bodiesGlsl(args)`, `lower(args, target, src, context)`,
   `assemble()` and `targetLiteral()` so it inherits determinism and the leak guarantees.
2. Add the target to `TYPE_NAME` (the one place type spelling lives).
3. Add its host-symbol mapping to `identifierMap()` and its fetch shape to `SAMPLE`.
4. `registerTarget('mytarget', myEmitter)` at module scope, and add it to
   `IMPLEMENTED_TARGETS` / `EMITTERS`.
5. Extend `checkTargetLeaks` if the language has a construct that looks like a leak
   (see the MSL `[[texture(0)]]` case).
6. Add the target to the test fixture loops — every test iterates `IMPLEMENTED_TARGETS`,
   so a new target is covered by determinism, leak, uniform-order, octave, Y-flip and
   diagnostic checks for free.
7. Document it here: entry point, declarations, Y flip, unsupported features.

---

## 7. Running the tests

```powershell
cd packages/ir
npm test          # node --test --test-isolation=none --import ./test/register-hooks.mjs test/targets.test.ts
```

Two environment notes:

* `test/register-hooks.mjs` installs a resolver hook mapping `./x.js` → `./x.ts`. The
  package is authored for `moduleResolution: nodenext`, so every relative import carries
  a `.js` extension — tsc, Vite and bundlers rewrite that, **Node does not**, and the test
  run would otherwise die with `Cannot find module '.../src/types.js'`.
* `--test-isolation=none` runs the file in-process instead of spawning a child per test
  file. That is required in sandboxes that deny piped child processes; `npm run
  test:isolated` is the same suite with the default (child-per-file) isolation.

`npm run typecheck` needs `typescript` and `@types/node`, which are declared in
`devDependencies` but are not installed in a bare checkout.

---

## 8. Known cross-target divergences (owned by `rewrite.ts`, not by this directory)

These are real semantic differences that no emitter can fix locally. They are recorded
here so nobody "fixes" them by patching generated output.

1. **`mod_` → `fmod` on HLSL/MSL.** GLSL `mod` floors, C `fmod` truncates. They differ for
   negative operands, and `hsv2rgb()` feeds `mod_` a hue that can be negative — so a
   negative hue produces a different colour on HLSL/MSL than on GLSL/WGSL. The WGSL path
   already has the correct `wf_mod` shim; HLSL/MSL should emit `x - y*floor(x/y)` (or get
   the same shim). Not fixable from `targets/`.
2. **`lowerMatrixMul` splices unrewritten argument text.** A type constructor inside a
   `mat2_mul_`/`mat3_mul_` argument keeps its GLSL spelling on every target — the real
   catalog's `transform` node hits this (`mat2_mul_(rot2(r), (uv - vec2(0.5,0.5)) * scale)`).
   Emitters work around it with the fixpoint rewrite (§1); the proper fix is one line in
   `rewrite.ts` (run `argA`/`argB` through the same rename loop, or re-tokenise the
   result). It also leaves a double space in the inlined form (`(m *  v)`).
3. **`mat2_rot_` → matrix constructor.** `rewrite.ts` maps the alias to `mat2`/`float2x2`
   construction, which is a diagonal scale, not a rotation. Emitters work around it by
   redirecting call sites to `rot2()` before rewriting (see §1), but the alias table in
   `rewrite.ts` is still wrong for any caller that bypasses this framework.
4. **Integer literals on WGSL** come out of `rewriteSource()` as `5.0`, because it cannot
   know the expected type. `targets/wgslLower.ts` rewrites the integer positions it can
   prove — `i32(<literal>)` and the init/bound of an integer loop — back to `5`, so this is
   no longer a wgsl-target defect (see §4.2, item 5).
5. **`lintSubset` loop-shape warning is a false positive** for `for (int j = -1; j <= 1; j++)`:
   the regex only accepts `= <digits>` and `<`. `voronoi2` therefore emits three
   `RW010_LOOP_SHAPE` warnings on every emission. They are warnings, never errors, but
   they make "no diagnostics" impossible.
6. **Catalog port defaults that are `{kind:'param'}`** are referenced by the emitted body
   but never collected as uniforms (`collectUniforms` only scans `node.params`), so the
   shader references an undeclared name. Catalog port defaults must be literals.

## Matrix construction is not portable, and the alias is why

`mat2(a, b, c, d)` is **column-major** in GLSL, WGSL and MSL: `(a, b)` is the first *column*.
In HLSL it is **row-major**, so the same four values build the **transpose**. That turns into
three separate facts, all verified against the compiler rather than recalled:

| | GLSL / WGSL / MSL | HLSL / UE |
|---|---|---|
| `mat2(c, -s, s, c)` | first column is `(c, -s)` | first row is `(c, -s)` — the transpose |
| `matrix * vector` | the product | **does not compile** (`cannot convert from 'float2' to 'float2x2'`) |
| `mul(matrix, vector)` | n/a | compiles, and is the transpose of the GLSL result |
| `mul(vector, matrix)` | n/a | **equals** the GLSL result |

So a body never writes a matrix product directly. It writes `mat2_mul_(matrix, vector)`, and the
rewriter emits `(m * v)` where matrices are column-major and `mul(v, m)` on the HLSL family. The
argument swap is the whole point: without it every rotation in a document comes out mirrored on
two of six targets, which is invisible in any document whose rotation happens to be zero —
including all four examples, until this was found.

`lintSubset` enforces the alias (`RW014_MATRIX_PRODUCT`): a `*` with a `rot2`/`scale2`/
`mat2_rot_` call or a `mat2`/`mat3` local on either side is an error, with the alias named in the
fix. The evidence for the numbers above: `dxc -Fc` on
`mul(float2x2(cos,-sin,sin,cos), float2(1,0))` disassembles to `(cos, sin)`, while
`mul(float2(1,0), m)` disassembles to `(cos, -sin)`.
