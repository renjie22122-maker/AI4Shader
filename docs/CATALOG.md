# Node catalog — contract and current state

Owner: catalog agent (`packages/ir/src/catalog/`). Append-only for everyone else.
Machine-checked by `packages/ir/test/catalog.test.ts` (23 tests).

```powershell
cd packages/ir
node --test --test-isolation=none --experimental-strip-types --import ./test/register-hooks.mjs test/catalog.test.ts
```

## What the catalog is

`CATALOG: Record<NodeKind, NodeSchema>` covers **all 43 kinds in `NODE_KINDS`** (the number is
whatever `NODE_KINDS` says — it is the source of truth; earlier revisions of this line said 31
and then 33, both wrong at the time), plus `CATEGORIES` for the palette (7 sections that
partition the kinds exactly once). Port names, defaults and slider ranges exist only here.

## Rules a body must satisfy (all machine-checked)

| Rule | Why |
|---|---|
| ends in exactly one `return <expr>;` | `splitReturn` turns it into the port assignment |
| every `${x}` is a declared input, `OCTAVE_LIMIT`, or `node_id` | `expandTemplate` resolves exactly those |
| float literals carry a decimal point (`2.0`, `float(i)` for int→float) | GLSL ES 3.00 rejects implicit int→float |
| only portable fns / `_` aliases / `GLSL_LIBRARY` helpers, with matching arity | one authored source for 6 targets |
| no `#`, `precision`, `layout`, `uniform`, samplers, arrays, `while`/`switch`, `discard`, raw `break;` | outside the restricted subset |
| `for (int i = 0; i < ${OCTAVE_LIMIT}; i++)` only, and only with `needs.qualityScaled` | tier-scaled cost, uniform lowering |
| at most 4 output ports, **all scalar**, and a multi-output body returns `vec4(...)` | see "multi-output" below |
| **every local is namespaced `<name>_${node_id}`** | see "scoping" below |
| every input has a default with `min`/`max`/`step` | a dropped node must render and be tunable |
| a non-first `primaryOut` is named in the node's summary | the port table is not the only place people look (test 2e) |

## Multi-output nodes and the `vec4` contract

`emitBodies` destructures multi-output nodes: the body returns **one `vec4`** and the declared
ports are its `.x/.y/.z/.w` **in declaration order**. The guard in the emitter is

```ts
const portsOk = outs.length <= 4 && outs.every(p => TYPE_ARITY[p.type] === 1);
```

so ports must be **scalar** (`float`/`int`/`bool`), and anything else is diagnosed as
`IR033_BAD_MULTI_OUTPUT` and emitted through a degenerate fallback rather than silently.
Emission is `vec4 tmp_<id> = <expr>;` followed by one `float L_<id>_<port> = tmp_<id>.<comp>;`
per port, which is why a later port (`.y`, `.z`, `.w`) is what proves destructuring works:
wiring `.x` cannot distinguish the fixed path from the old broken one (test 11b).

Four kinds use it; the table is pinned by tests 1b, 2c and 11c:

| kind | ports (`.x` → `.w`) | body returns | notes |
|---|---|---|---|
| `time` | `time:float`, `sin:float`, `cos:float` | `vec4(t, sin(t), cos(t), 0.0)` | `.w` unused, 3 ports |
| `polar` | `angle:float`, `radius:float` | `vec4(angle, radius, 0.0, 0.0)` | `angle` keeps the 0..1 (0.5 = +x) convention it had as the packed `polar.x` |
| `voronoi` | `f1:float`, `f2:float`, `edges:float`, `cell:float` | `vec4(v.x, v.y, v.y - v.x, v.z)` | the only 4-port node; `edges = f2 - f1`, `cell = v.z` |
| `sdf` | `distance:float`, `mask:float` | `vec4(d, mask, 0.0, 0.0)` | declared `distance`-first, but `primaryOut` stays `mask` |

`primaryOut` is the port a bare `{"node": "id"}` wire resolves to, and it **need not be the
first declared port** — `sdf` uses that so existing graphs keep reading the mask signal they
always read. Backward compatibility of the *primary signal* is the rule: `polar`'s primary
went from the packed `polar: vec2` to `angle: float` (the same number its `.x` held), and
`time.time`, `voronoi.f1` and `sdf.mask` are unchanged. A wire that named the old port name
(`polar`) is a schema error (`IR005_UNKNOWN_OUTPUT`), not a silent mis-wire.

### The two kinds that stay single-output (`uv`, `tile`)

Both need a **pair of vec2 ports**, which the contract above cannot express:

* `uv` wants `uv: vec2` + `centered: vec2`
* `tile` wants `uv: vec2` (in-cell) + `cell: vec2` (cell id)

A vec2 port would have to alias two components of the returned `vec4`, and no single
lowering of that exists across all six targets — so this is a **known limitation of the
emitter contract, not an oversight**, asserted by test 1d with the reason, and stated in
each node's `summary` where a graph author will read it. Promoting one half to two scalars
does not help while the other half stays a vec2. Recipes that cover the missing half:

* `uv.centered` → `transform` with scale (2,2) and offset (-0.5,-0.5)
* `tile.cell` → `voronoi` (`cell` port) for a per-cell random id; `checker` for hard cells

## Scoping: why locals carry `${node_id}`

`emitBodies` concatenates every node's statements into **one flat list**, and a body cannot
open a block it also has to close (its `return` is the last statement). So all bodies share
one function scope: two `noise` nodes would both declare `vec2 p` and the shader would fail
with "redefinition of 'p'". Every declaration therefore reads `p_${node_id}`. Verified: with
un-namespaced locals the all-kinds graph reports `p x2, v x2, g x2, …`; as shipped, zero
duplicates (test 10, plus test 2b as a static guard).

`${node_id}` is substituted verbatim (not through `sanitizeId`), so node ids must be GLSL
identifiers — `patcher.makeNodeId` produces exactly that (`fbm`, `fbm_2`, …).

## Known limitations of the catalog (and the frozen files behind them)

1. **`uv` and `tile` cannot expose their second coordinate space.** They need a pair of
   `vec2` ports (`uv`+`centered`, `uv`+`cell`) and multi-output ports must be scalar — see
   "Multi-output nodes" above. Their `summary` carries the in-graph recipe, and test 1d
   fails if either ever grows a port without the emitter contract having changed first.
   Every *other* port the design called for is back: `time.sin/cos`, `polar.radius`,
   `sdf.distance`, `voronoi.f2/edges/cell`.
2. **`shape.uv` is inert.** It is declared because the node brief requires it; the summary
   says so, and test 2f fails if any *other* input is declared but unread.
3. **`tex2D_(uv)`** needs the host channel (`iChannel0`); `targets/` lowers it per target
   via `expandFetchCalls`.
4. **`output` carries one passthrough port `color`.** It stays single-output: a second
   port would have to be scalar and there is no second scalar to expose — and with
   `outputs: []`, `emitBodies` emits the returned expression as a bare statement.

## Frozen-file findings (reported, not worked around)

| File | Finding |
|---|---|
| `glslSubset.ts` | **was not parseable**: unescaped backticks inside the `GLSL_LIBRARY` template literal (lines 154/156/287) → `ERR_INVALID_TYPESCRIPT_SYNTAX` for every importer. Escaped them (`\``); rendered GLSL text unchanged. |
| `emitter.ts` | multi-output destructuring **was** invalid for every schema and is now fixed (see "Multi-output nodes"): ports are the components of one returned `vec4`, at most four, scalars only. The remaining gap is composite ports — a `vec2` port would need a lowering that does not exist in all six targets, which is what keeps `uv` and `tile` single-output. |
| `emitter.ts` | `OCTAVE_LIMIT` inside `GLSL_LIBRARY` is never expanded (only node bodies go through `expandTemplate`); each target must substitute the tier budget itself — `targets/_shared.ts` does. |
| `rewrite.ts` | `lintSubset`'s loop-shape regex is tested against tokens joined with spaces, so `i ++` can never match `\w\+\+` → `RW010_LOOP_SHAPE` fires on every well-formed `for` loop (library included). Warning-only, so the catalog asserts "no errors, and no warnings other than `RW010*`". |
| `rewrite.ts` | `mix_` maps to the scalar-only WGSL shim `wf_mix`; `mix_(vec4, vec4, vec4)` (used by `rgb2hsv`) has no WGSL overload. target-side concern. |
| `targets/_shared.ts` | all six emitters place the node bodies **outside** `void main()`, so every `L_…`/body local is a global initializer that reads `uv`/`iFrame` — invalid GLSL/WGSL/HLSL/MSL. Bodies must be emitted inside the entry point, before the output-colour expression. |

Two findings reported by this catalog were fixed upstream while this change was prepared —
re-checked, not assumed:

* `coerce()` no longer emits `vec2(x, x, x)`; `float → vecN` now splats N times (probed by
  wiring a `float` output into `checker.count`, which is legal and emits `vec2(x, x)`).
* `outputs: []` no longer crashes `emitBodies`; it emits the returned expression as a
  statement (the comment in the emitter says so, and `output` still declares `color`).

## Ports and defaults (quick reference)

| kind | in | out |
|---|---|---|
| uv | — | `uv: vec2` |
| time | speed | `time: float`, `sin: float`, `cos: float` |
| mouse | — | `mouse: vec4` |
| resolution | — | `resolution: vec2` |
| const_color | value: vec4 | `color: vec4` |
| const_float | value: float | `value: float` |
| gradient | uv, angle, scale | `value: float` |
| texture | uv, mix_amount | `color: vec4` |
| transform | uv, scale, rotate, offset | `uv: vec2` |
| tile | uv, count | `uv: vec2` |
| kaleidoscope | uv, segments | `uv: vec2` |
| polar | uv, center | `angle: float`, `radius: float` |
| domain_warp | uv, strength, scale, time | `uv: vec2` |
| noise | uv, scale, offset, mode | `value: float` |
| fbm | uv, scale, octaves, lacunarity, gain, ridged | `value: float` |
| voronoi | uv, scale, randomness | `f1: float`, `f2: float`, `edges: float`, `cell: float` |
| stripe | uv, angle, frequency, width | `value: float` |
| checker | uv, count | `value: float` |
| sdf | uv, shape, size, radius, smoothness | `distance: float`, `mask: float` |
| raymarch | uv, height, frequency, steps, time | `shade: float` |
| math | a, b, op | `value: float` |
| vec_op | a, b, op | `value: vec4` |
| remap | value, inMin, inMax, outMin, outMax | `value: float` |
| smoothstep | edge0, edge1, value | `value: float` |
| antialias | value, edge, amount | `value: float` (threshold whose width follows the derivative) |
| mix | a, b, t | `value: vec4` |
| shape | uv (inert), mask, color, background | `value: vec4` |
| color_ramp | t, colorA, colorB, colorC | `color: vec4` |
| hsv | color, hue_shift, saturation, value | `color: vec4` |
| grade | color, exposure, contrast, saturation, gamma, tint | `color: vec4` |
| blend | a, b, mode, opacity | `value: vec4` |
| output | color | `color: vec4` |

**Surface group (5)** — every one of these declares `needs.surface: true`, and the flag is what
changes every target's entry point (see below):

| kind | in | out |
|---|---|---|
| normal | — | `normal: vec4` (unit, world space — `vec4(iNormal, 0.0)`) |
| world_pos | — | `position: vec4` (world space — `vec4(iWorldPos, 1.0)`) |
| view_dir | — | `direction: vec4` (unit, surface → camera) |
| light_dir | — | `direction: vec4` (unit, surface → light) |
| dot | a: vec4, b: vec4 | `value: float` (`dot(a.xyz, b.xyz)`) |

**Vector primitives (5)** — the other half of the type system. The connection rule only
*widens* (`float → vecN`); it never narrows and there is no swizzle on a wire, so before these
kinds a `vec4` producer could not reach a `vec2` input at all — `mouse` was unusable, and the
wire that tried it was silently deleted by the repairer as a type mismatch.

| kind | in | out |
|---|---|---|
| split_vec2 | value: vec2 | `x: float`, `y: float` |
| split_vec4 | value: vec4 | `x: float`, `y: float`, `z: float`, `w: float` |
| make_vec2 | x: float, y: float | `value: vec2` |
| make_vec4 | x, y, z, w: float | `value: vec4` |
| vec_math | a: vec4, b: vec4, op | `value: vec4` (normalize / length / distance / reflect / fract / floor / abs / pow / max / min / sign / clamp / cross) |

**Escape hatch (1)**:

| kind | in | out |
|---|---|---|
| custom_glsl | uv: vec2, a, b, c, d: vec4 | `value: vec4` (body from `node.body`, validated by `lintSubset`) |

`custom_glsl` is the one kind whose body is **per instance** rather than a catalog constant:
the text lives on `node.body`, is written in the same restricted GLSL subset as every other
body, and is lowered by the same token rewriter. Two things it will not do: a scalar widens
into a `vec4` port but a `vec2`/`vec3` must go through `make_vec4` first, and the subset does
not check *scope* — naming a helper that does not exist is a compile error per target, not a
diagnostic.

**Why the surface group is a separate category rather than five more generators:** a quad
shader is a *picture* (`uv` from `gl_FragCoord`) and a surface shader is a *shading function*
(interpolated `vWorldPos` / `vNormal` / `vUv`). Reachability of any node with
`needs.surface: true` switches the emitted entry point, the vertex stage, the host uniforms and
the 3D preview's material path — so the flag is computed over the **reachable** set, and
deleting the last surface node returns the document byte-for-byte to the quad form.

**The other capability flag: `needs.channel`.** Exactly one kind declares it — `texture`, the
node that samples the host image channel — and test 9b derives the rule from the bodies rather
than listing it, so a second channel-reading kind cannot be added without the flag. Unlike
`surface` it is **not** a change to the emitted source: what the channel *contains* is the host's
decision, and the source is identical either way. It is reported because a host has to bind
something, and because `render.feedback` — the document asking for the **previous frame** in that
channel, which is what lets a wiped fog stay wiped — is only meaningful when a reachable node
reads it. Asking for it otherwise is `IR036_FEEDBACK_UNUSED`: the host allocates two textures and
an extra pass per frame for a picture that cannot change.

**`w` is a tag, not a component.** A vec4 in this IR carries its kind in `w`: 0 for a direction
(`v4()`), 1 for a point (`color()`). `dot`, and `vec_math`'s `normalize` / `length` / `distance` /
`reflect` / `cross`, read `.xyz` and ignore it; the component-wise ops (`fract`, `floor`, `abs`,
`pow`, `max`, `min`, `sign`, `clamp`, and everything in `vec_op`) include it. This matters because
`make_vec4` defaults `w` to **1**: a "direction" built without saying otherwise is a point, and
normalising it four-dimensionally gives a different vector from normalising it in three — both
valid GLSL, no diagnostic, every downstream term slightly off. Test 9c pins both halves.

**The noise generators can tile.** `noise`, `fbm`, `voronoi`, `domain_warp` and `raymarch`
each take a `tile` input (0/1, default 0). Off, nothing changes. On, the cell *lookup* wraps
modulo the period while the geometry stays unwrapped, so the grid point of cell `period` is the
grid point of cell 0 translated by one period — the field is continuous across the seam, not
merely similar on both sides. `fbm`'s period scales with the octave (`per = per * lacunarity`),
because octave `i` samples in a lattice `lacunarity^i` times denser; without that, only the first
octave tiles and the seam hides under the low frequencies. The period always comes from the
node's own `scale`/`frequency`, which is the only quantity that knows how many cells fit in one
uv unit. There is **one** lattice implementation: `noise2`/`gnoise2`/`fbm2`/`voronoi2` are
one-line wrappers passing `on = 0`, and that path is exact (`mix_(i, mod_(i, per), 0.0) === i`).

**`antialias` is the only derivative reader, and it is what `needs.derivatives` was reserved
for.** It thresholds `value` at `edge` with a transition width of `fwidth(value) * amount`, so
the edge is one screen pixel wide at any zoom or resolution — the difference between a pattern
that shimmers and one that holds still. Two things make it safe here: `fwidth` is available in
the fragment stage of all six targets under that exact name, and a derivative is only *defined*
in uniform control flow — which the restricted subset guarantees, because it bans
data-dependent branching, so every branch in a body is a uniform comparison the whole quad takes
together. Use it on `sdf.distance` instead of `sdf.mask` when the shape is small or the camera
moves: `mask`'s edge width is a fixed number of uv units, which is too sharp far away and too
soft up close.

**A wired port the body never reads is reported (`IR037_INERT_INPUT`).** `shape.uv` is the one
such port in the catalog — accepted "for pipeline symmetry" and ignored — and wiring it does
nothing, silently. The check is derived from each body, not from a list, so a future node with a
dead port is covered the moment it ships and a body that stops reading a port it still declares
starts warning immediately. `custom_glsl` is exempt: its body is per instance, so a snippet that
uses two of its slots is normal.

**No scalar splat in a vector constructor.** `vec4(0.0)` is valid GLSL, WGSL and MSL and a hard
error in HLSL, so the subset rejects it (`RW013_SCALAR_SPLAT_CONSTRUCTOR`) and the fix names the
four-argument form. `vec4(v)` (a copy) and `vec4(a, b, c, d)` are unaffected — the rule is about a
bare numeric literal, which can never be a vector. One shipped body had it, passed every
structural check, and was caught by `dxc`.

Enum inputs are float selectors (slider-friendly): `noise.mode` 0 value / 1 gradient /
2 turbulence; `sdf.shape` 0 circle / 1 box / 2 hexagon / 3 triangle / 4 star / 5 ring;
`math.op` 0 add … 8 atan2, 9 sin, 10 cos, 11 abs, 12 sqrt, 13 frac, 14 sign;
`vec_op.op` 0 add / 1 sub / 2 mul / 3 div / 4 screen / 5 overlay / 6 difference;
`vec_math.op` 0 normalize / 1 length / 2 distance / 3 reflect / 4 fract / 5 floor / 6 abs /
7 pow / 8 max / 9 min / 10 sign / 11 clamp / 12 cross;
`blend.mode` 0 normal / 1 add / 2 multiply / 3 screen / 4 overlay.

In the `out` column a comma-separated list is a **multi-output** node, in `.x/.y/.z/.w`
order; the first name is not always the primary one (`sdf`'s primary is `mask`, its second
port). Ports and defaults are defined once, in `src/catalog/nodes.ts` — never hard-code a
port name anywhere else (`docs/AGENT_CONTRACT.md` §1).

## Test infrastructure

`catalog.test.ts` is 23 tests: rules 1–9 are static checks over the catalog, test 10 emits a
graph containing **every** kind once (plus a second instance of the local-heavy kinds) and
asserts zero error diagnostics, no duplicate local, and the correct `vec4 tmp_<id>` /
`L_<id>_<port> = tmp_<id>.<comp>` pair for every multi-output kind; tests 11a–11c wire each
output port — including every non-`.x` port — into `output` and check the component it reads.

`node --test` needs two things in this repo/sandbox:

* `--import ./test/register-hooks.mjs` — `test/ts-resolve.mjs` maps a relative `./x.js`
  specifier onto `./x.ts` (Node's type stripping does not, and §2 of AGENT_CONTRACT.md
  requires `.js` specifiers).
* `--test-isolation=none` — the default process-per-file isolation spawns a child with
  piped stdio, which the restricted sandbox denies (`spawn EPERM`). Note this shares module
  state between test files: `patcher.test.ts`'s "real emit() pipeline" test then sees the
  emitters registered by `targets/index.ts` and fails. `npm run test:isolated` is the
  process-per-file variant for a normal machine.
