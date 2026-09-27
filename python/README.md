# AI4Shader — offline Python layer

Two jobs that belong in Python rather than in the browser or in Node:

1. **Real compiler validation.** `glslangValidator`, `dxc`, `naga`, `xcrun metal`
   are command-line tools. Wrapping them here gives the Node CLI and CI a single
   "validate every target" command whose verdict is a real compiler's, not a
   structural guess.
2. **Image-similarity scoring and failure attribution.** `apps/web/src/compare/score.ts`
   has a crude in-browser SSIM; this is the reference implementation, and it adds
   the part that makes a score *actionable*: `DESIGN.md` §4.3's attribution table
   as executable rules.

Nothing in this package makes a network call. Ever. See
[No network](#no-network).

---

## Quick start

There is **no install step** (see [`pyproject.toml`](pyproject.toml)). Run from
this directory:

```powershell
python -m ai4shader compilers                 # which real compilers exist here
python -m ai4shader capabilities              # which metrics can run, and why
python -m ai4shader capabilities --probe      # ...and actually load LPIPS/CLIP
python -m ai4shader validate ..\dist\demo\shader\shader.frag
python -m ai4shader validate ..\dist\demo\shader\shader.hlsl --tool dxc
python -m ai4shader score render.png reference.jpg --size 512 --lpips
python -m ai4shader attribute render.png reference.jpg --catalog ..\examples\liquid-metal.shader.json
python -m ai4shader batch ..\examples\liquid-metal.shader.json --targets all --out report.json
python -m pytest tests/
```

Exit codes are chosen so CI cannot mistake "no compiler" for "shader is fine":

| code | meaning |
|---|---|
| `0` | the assertion the command makes is true |
| `1` | a real failure — a compiler rejected the source, or a batch has failures |
| `2` | usage error |
| `3` | **no verdict exists** — the compiler is absent, or `--tool` named one that is not installed |
| `4` | `--require-verified` was given and something was unverified |

---

## What actually runs on this machine

Measured on the development machine (Windows, Anaconda Python 3.11.5, code page 936).

### Compilers — `ai4shader compilers`

| tool | found | path |
|---|---|---|
| `glslangValidator` | **yes** | `%LOCALAPPDATA%\Android\Sdk\emulator\lib64\vulkan\glslangValidator.exe` (Glslang 11.7.0) |
| `dxc` | **yes** | `C:\Program Files (x86)\Windows Kits\10\bin\10.0.22621.0\x64\dxc.exe` (dxcompiler 1.6.2112.16) |
| `naga` | no | — |
| `xcrun metal` | no | not macOS |
| `unity` | no | — |

Neither `glslangValidator` nor `dxc` is on `PATH`; they ship inside SDKs.
`shutil.which` alone finds nothing, so `validate._candidate_dirs()` also probes the
Vulkan SDK, the Windows Kits version directories, the Android SDK emulator bundle
and `~/.cargo/bin`, plus the `AI4SHADER_GLSLANG` / `AI4SHADER_DXC` / `AI4SHADER_NAGA`
/ `AI4SHADER_METAL` overrides and `AI4SHADER_TOOL_PATHS`. Reporting "no compiler"
on a machine that has one is its own kind of false negative.

### Metrics — `ai4shader capabilities`

| metric | status | notes |
|---|---|---|
| `ssim`, `histogram_distance`, `band_energy`, `gradient_energy`, `color_stats` | **real** | numpy only, no optional deps |
| `lpips` | **REAL — ran** | `lpips.LPIPS(net='alex')` loaded from cached weights |
| `clip_similarity` | **REAL — ran** | OpenAI CLIP `ViT-B/32`, sha256 verified |

**The real learned metrics ran. Neither fell back.** The weights were already
cached locally:

```
lpips linear weights : C:\Users\<user>\anaconda3\Lib\site-packages\lpips\weights\v0.1\alex.pth
alexnet backbone     : C:\Users\<user>\.cache\torch\hub\checkpoints\alexnet-owt-7be5be79.pth   (244 MB)
CLIP ViT-B/32        : C:\Users\<user>\.cache\clip\ViT-B-32.pt                                (354 MB)
```

The CLIP checkpoint's sha256 is checked against the value published in OpenAI's
`_MODELS` table (`40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af`)
before use, so a truncated or substituted file is reported unavailable rather
than silently producing a meaningless embedding.

### First-call timing (measured, 256×256 inputs)

| call | first | subsequent |
|---|---|---|
| `lpips(a, b)` | ~5.4 s to load (244 MB checkpoint + torch import), then **103 ms** for the first forward pass | **29 ms/call** |
| `clip_similarity(a, b)` | **135 ms** (after torch is loaded) | **108 ms/call** |
| `capabilities()` (no probe) | 244 ms — dominated by the CLIP sha256 over 354 MB | memoised on `(path, size, mtime)`; ~0 ms after |
| `ssim` / `hist` / `band_energy` on 512×512 | — | a few ms |

**This matters for the optimiser.** `DESIGN.md` §6.6 budgets ~12 ms per
evaluation for a parameter sweep. SSIM fits that budget; LPIPS at 29 ms does not
— and the one-off 5.4 s model load is paid inside the first evaluation unless the
model is constructed up front. If §6.6's optimiser is ever driven from Python,
call `capabilities(probe=True)` (or `score.lpips(img, img)`) once at start-up to
warm the model, and treat LPIPS as a term to evaluate every *N*th candidate
rather than every one.

`torch` is never imported at `score.py` module scope; `capabilities()` without
`--probe` imports nothing heavy, and `tests/test_metrics_optional.py` asserts
that in a subprocess.

---

## What is **not** verified here

Said plainly, per `DESIGN.md` §5.4.

- **WGSL is still unverified.** No `naga`/`tint` exists on this machine, so
  `validate` returns `ok=None` for `shader.wgsl`. This is unchanged from
  `docs/IMPLEMENTATION.md` §4.2 — the tooling to close it is now in place, the
  tool is not.
- **MSL is still unverified.** Not macOS, no `xcrun metal`. The documented
  fallback (`DXC → SPIR-V → SPIRV-Cross`) is not implemented; `spirv-cross` is
  also absent.
- **ShaderLab is still unverified.** No Unity.
- **UE is unverifiable by construction.** `DESIGN.md` §5.4 and
  `packages/ir/src/targets/README.md` §4.6 both say a Custom node cannot be
  headless-compiled. `validate` always returns `ok=None` for it.
- **The GPU round trip is untested.** Everything here reads PNG/JPEG/PPM from
  disk. Nothing in this package captures a frame from a real GPU, so the
  metrics have only ever been exercised on synthetic images and emitted
  artefacts — **not on an actual render/reference pair**. The numbers are
  correct; the pipeline that produces the render has not been exercised.
- **`batch` has only been run against the one shipped example document**
  (`examples/liquid-metal.shader.json`). Its document-shape assumptions are
  tolerant of failure (a missing document becomes a `cli-missing` row, never an
  exception) but have not been tested against a second document shape.
- **JPEG loading is untested in the stdlib path** — by design, JPEG requires
  Pillow, and Pillow is present here, so the "no Pillow" JPEG error path is
  tested only by assertion on the message, not by running without Pillow.
- **16-bit PNG colour types 2/3/4/6** are decoded by the fallback but only
  grey 16-bit and 8-bit palette are covered by tests.
- **Adam7-interlaced PNG** raises `NotImplementedError` rather than decoding.
  Handled explicitly, but not tested against a real interlaced file (Pillow
  here would not produce one).

---

## Findings that came out of building this

These are real defects and environment traps discovered by running actual
compilers and actual metrics, not by reading code.

### 1. `score.ts`'s `high` frequency band is broken

`apps/web/src/compare/score.ts` computes:

```ts
const up1 = upsample(d2.data, d2.w, d2.h, w, h);   // d2 is 1/4 scale, target is FULL scale
high = rmsOfDifference(difference(p0, up1));
```

`upsample()` maps `y -> min(sh-1, y>>1)`. Going from 1/4 scale to full scale
needs **two** doublings; `y>>1` performs one, so for `y >= 2*sh` the index clamps
to `sh-1`. Measured directly (`python/dev/verify_band_bug.py`): the bottom-right
quadrant of `up1` contains **one distinct value** (std exactly 0), and `high` is
~0.294 for a 256×256 image whose luma standard deviation is ~0.29 — i.e. `high`
*is* global contrast, not fine detail.

The consequence, measured:

| perturbation | `score.ts` `high` ratio | correct `high` ratio |
|---|---|---|
| box blur, radius 2 | 0.989 | **0.221** |
| box blur, radius 4 | 0.985 | **0.192** |
| white noise, σ = 0.05 | 1.003 | **1.503** |
| white noise, σ = 0.16 | 1.010 | **3.362** |

The browser's high band is essentially blind to the thing it is named after. That
breaks `DESIGN.md` §4.3's second attribution row ("high-frequency band energy
ratio far from 1"), and it corrupts the λ₃ term of the §6.6 optimiser loss —
a search driven by `lossFrom()` will be pushed to reduce *global contrast*
rather than to fix noise frequency.

The fix is one variable: pass `d1` (1/2 scale), not `d2`, to `upsample`. This
module implements the correct statistic by default and exposes
`band_energy(..., ts_compat=True)` to reproduce the browser's current numbers
exactly, so the two can be cross-checked while the TypeScript is fixed.

### 2. `dxc`'s "crash" is an encoding bug, and there is a real HLSL error behind it

`docs/IMPLEMENTATION.md` §4.2 records:

> `dxc.exe` 存在且在最小 shader 上工作，但一遇到含函数体的输入就崩溃（`0x80070459`，无诊断）

That is not what happens. Two separate things are going on.

**(a) `0x80070459` is a file-encoding failure, not a crash and not a shader
error.** `dxc` 1.6 on a host whose ANSI code page is CJK (code page **936** here)
cannot decode a source file that contains a non-ASCII byte and **no byte-order
mark**. The emitted artefacts all carry a UTF-8 em dash (`—`) in their header
comment, so this fires on every real file. With a BOM, `dxc` emits normal
`line:column` diagnostics.

Reproduced and bisected (`python/tests/test_validate.py::
test_the_opaque_dxc_encoding_failure_is_diagnosed_not_reported_as_a_shader_bug`):

| source | result |
|---|---|
| ASCII | `rc=0` |
| UTF-8 em dash, no BOM | **`rc=1`, `dxc failed : error code 0x80070459`, no diagnostics** |
| UTF-8 em dash, with BOM | `rc=0` |
| UTF-8 CJK characters, no BOM | `rc=0` (the byte sequence happens to be valid GBK) |
| GBK-encoded em dash | `rc=0` |
| `-encoding utf8` flag | **does not help** |

`validate_source` therefore writes HLSL temp files with a UTF-8 BOM whenever the
source is not pure ASCII, and records that in `notes`. That converts an opaque
failure into real diagnostics. This is a workaround for a `dxc` limitation, not a
source edit — but **the emitters should stop putting non-ASCII in generated code
comments**, which is the actual fix.

**(b) With the BOM, `dxc` reveals a genuine HLSL bug.** The real emitted
`shader.hlsl` from `examples/liquid-metal.shader.json` does not compile:

```
shader.hlsl:26:43: error: use of undeclared identifier 'atan2'
float wf_atan2(float y, float x) { return atan2(float2(y, x)); }
                                          ^
```

HLSL's `atan2` is scalar-only; GLSL's is not. The shared library in
`packages/ir/src/glslSubset.ts` passes it a `float2`, which is valid GLSL and
valid WGSL but **not valid HLSL**. `dxc` also reports 24
`-Wconversion` warnings (`implicit truncation of vector type`) in the same file,
all from `wf_mix`/`wf_mod` calls with mixed vector/scalar arguments.

Also worth correcting: `0x80004005` (`E_FAIL`) is `dxc`'s **normal**
exit code for "compilation failed" — it is not a crash. Confirmed against
deliberately broken shaders, which produce proper diagnostics and `E_FAIL`.

### 3. GLSL now has a real verdict

`docs/IMPLEMENTATION.md` §4.2's top item — "GLSL 在真实 WebGL2 上编译并出图 …
没有编译器说过它 OK" — is partly closed. `glslangValidator` compiles the emitted
`shader.frag` from the shipped example with **exit 0 and no diagnostics**
(35 ms). That is not a WebGL2 driver, and it does not render a frame, but a real
GLSL compiler has now accepted the source.

### 4. `DESIGN.md` §4.3 asks for a node that does not exist

The exposure row says: "改 `grade.exposure`、插入 `tonemap_aces`". The catalog
(`packages/ir/src/catalog/nodes.ts`, 35 declared kinds) contains **no tonemap
node**. `attribute` therefore emits a second, lower-confidence hint recording
that the suggested fix is not expressible today, instead of naming a node kind
the emitter would reject. See interpretation note 3 in `attribute.py`.

---

## Where `DESIGN.md` §4.3 was ambiguous

The table is a design sketch, not a specification, and three rows needed an
interpretation. All three are documented in `attribute.py`'s module docstring and
pinned by tests.

1. **`sdf.size` / `domain_warp.scale` are named without a direction.**
   `sdf.size` is a `vec2` (there is no scalar "increase"), and `scale` means
   opposite things on the two nodes it appears on — a larger `fbm.scale` means
   *finer* noise, a larger `domain_warp.scale` means a *finer* warp. **Chosen
   reading:** derive the direction from the measured low-band energy ratio
   (`< 0.85×` → increase, `> 1.25×` → decrease), and when the coarse energy
   *matches* the reference — so the structure is misplaced rather than
   mis-energised — report `direction=None` and say so, rather than pick a
   direction from nothing. This differs from `score.ts`, which always picks one.

2. **`sdf.smoothness` is inverted relative to its name.** In the catalog the mask
   is `1.0 - smoothstep(0.0, smoothness, d)`: `smoothness` is the smoothstep
   **width**, so a *larger* value gives a *softer* edge and a *lower* gradient.
   "Edges too sharp" therefore means `smoothness` should **increase**. The
   opposite reading is the intuitive one and would send the optimiser the wrong
   way, so it is asserted explicitly in `tests/test_attribute.py`.

   `score.ts` routes this observation to `contrast`/`detailScale` instead and
   never mentions `smoothness` at all, so the two implementations disagree here.
   That is a deliberate divergence: the design table names `sdf.smoothness`, and
   this module follows the design table.

3. **`color_ramp` stops have no scalar direction.** The hue row says "改
   `color_ramp` 的三个色标 / `palette` 参数". A colour stop is RGBA; "change the
   three stops" is not an increase or a decrease. **Chosen reading:** prefer
   `hsv.hue_shift`, which *is* a scalar in turns and takes a real direction; fall
   back to the ramp only when no hue-offset parameter exists, and then report
   `direction=None` with the reason in `why`.

Two further judgement calls, because two rules genuinely fire on the same
observation:

- **Blur vs. added noise.** A blur and added grain both move the gradient RMS and
  the high band. The discriminator is the **mid band**: a low-pass (blur) leaves
  it intact, a frequency-content change (grain, octave count) moves it too.
  Threshold `T_MID_MOVED = 0.15`, measured — a box blur moves the mid band by
  0.03–0.09, added white noise at σ 0.08/0.16 moves it by 0.15/0.42. When both
  moved, the frequency rule wins and the edge rule is attenuated ×0.35.
- **Saturation is a confounded observable.** `S = (max-min)/max` grows with
  per-channel spread, so adding *achromatic* noise raises measured HSV
  saturation. When a broadband frequency mismatch is present, the saturation rule
  is attenuated ×0.30 so it cannot outrank the rule that explains the difference.

### Direction is a property of the parameter, not of the observation

A rule observes an **effect** ("the edges need to be harder"); a parameter has a
**polarity** (which way of turning it produces that effect). Conflating the two
produces confidently wrong advice, because the same observation needs *opposite*
directions from two legitimate answers:

| parameter | effect of *increasing* it | so "make edges harder" means |
|---|---|---|
| `sdf.smoothness` | smoothstep **width** → softer edge, lower gradient | **decrease** |
| `grade.contrast` | scales the gradient → steeper transitions | **increase** |

`attribute.POLARITY` records this per rule and per candidate `(kind, param)`, as
`+1` / `-1` / `None`, and a parameter whose polarity is unknown gets
`direction=None` rather than a guess. `domain_warp.scale` is the worked example:
it sets the warp's *spatial frequency*, and whether a finer warp raises or lowers
coarse-structure contrast depends on the noise underneath it, so no direction is
claimed. `test_direction_is_a_property_of_the_parameter_not_the_observation`
drives both parameters from the same blur and asserts they come out opposite.

---

## Design decisions worth knowing

### Letterboxing, never stretching

A reference image and a render are almost never the same size, and stretching them
to match silently corrupts every metric — SSIM drops because structure moved, band
energy shifts because the effective sampling rate changed, and the attribution
then blames the shader for a resize artefact. `load(..., size=...)` letterboxes by
default, and `score()` **refuses** mismatched shapes rather than resizing.

Padding is filled with **each image's own mean colour** rather than black. Black
padding is identical in both images, so it contributes SSIM ≈ 1.0 over the bar
area and *inflates* the score. Mean-colour padding carries each image's own tone,
so the bars differ whenever the images differ and cannot be mistaken for
agreement. `pad="black"` is available for conventional letterboxing.

### Resizing never uses PIL

Even when Pillow is present. Two reasons: a score computed on a machine with
Pillow must equal the score computed without it, and downscaling uses an exact
**area average** rather than bilinear point sampling, so a 4K reference reduced to
512×512 does not alias into fake high-frequency detail — which matters, because
measuring high-frequency detail honestly is the whole point of the band metric.

### The stdlib codecs are reachable on a machine that has Pillow

`load`/`save` take `prefer="auto" | "pil" | "stdlib"`. Every codec test runs
twice, once per path. A fallback that is never executed is a fallback that does
not work, and this is the only way to exercise it here.

### Python and TypeScript agree on SSIM by construction

`score.ts` computes SSIM on 0..255 luma with `C1 = (0.01·255)²`; this module uses
0..1 with `C1 = 0.01²`. SSIM is invariant under that common rescaling, so the two
are the *same number*, not merely close.
`tests/test_score.py::test_ssim_matches_the_typescript_formula` transcribes the
TypeScript expression literally and checks it. The histogram binning is likewise
proved identical on all 256 levels rather than argued in a comment.

`ScoreBreakdown.to_dict()` emits the **exact camelCase key names** of the
TypeScript type, so a JSON blob produced here can be handed to the browser or the
Node CLI with no translation table.

### An absent compiler is never a pass

`ValidationResult.ok` is three-valued: `True` / `False` / `None`. `None` means
**no verdict exists** — absent tool, timeout, or a crash with no diagnostics — and
the CLI turns it into exit code `3`, distinct from `1`. A CI job can therefore
tell "the shader is wrong" from "this machine cannot tell". `--tool <name>`
asserts that a *specific* compiler ran, so a job can demand `dxc` rather than
accepting whichever one happened to be installed.

### `lpips` returns `None`, never a substitute

`lpips()` returns `None` when the weights are absent, and `capabilities()` says
why. A fabricated similarity is worse than no similarity, because it silently
drives the optimiser in a direction the data does not support.

There is **no automatic fallback inside `lpips()`**. A dependency-free
multi-scale distance does exist, as `perceptual_fallback()`, deliberately under
its own name so nothing can ever report it as an LPIPS value. If you need a
degradation path, branch on `capabilities()` and call it explicitly.

### `no_network`

Model loading runs inside `score.no_network()`, which replaces
`socket.socket.connect` / `connect_ex` / `create_connection` with a function that
raises. If a loader decides it needs to download something, it fails loudly here
instead of silently reaching the internet. `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` are set with `setdefault` at import as a second layer.
Two tests exercise the guard directly.

---

## Module map

| file | what it does |
|---|---|
| `ai4shader/imageio.py` | load/save without a hard Pillow dependency; pure-stdlib PNG (`zlib`+`struct`) and PPM codecs; letterbox/crop/stretch |
| `ai4shader/score.py` | `ssim`, `histogram_distance`, `band_energy`, `color_stats`, `gradient_energy`, `score`, `loss`, `lpips`, `clip_similarity`, `capabilities` |
| `ai4shader/attribute.py` | `DESIGN.md` §4.3 as literal rules; `catalog_from_document` resolves real node ids and uniform names |
| `ai4shader/validate.py` | subprocess wrappers around `glslangValidator` / `dxc` / `naga` / `xcrun metal`; tool discovery; diagnostic parsing |
| `ai4shader/batch.py` | drives the Node CLI over many documents, then validates each output; diffable JSON report |
| `ai4shader/cli.py` | `python -m ai4shader ...` |
| `dev/` | threshold-measurement and probe scripts; **not** part of the test suite |

### Reading a `ScoreBreakdown`

Units, ranges, and which direction is better:

| field | meaning | range | better |
|---|---|---|---|
| `ssim` | global SSIM over Rec.709 luma, non-overlapping 8×8 blocks | −1 … 1 | higher |
| `hist` | 1 − mean per-channel histogram intersection, 64 bins | 0 … 1 | lower |
| `bands` / `bandsRef` | `{low, mid, high}` RMS energy, `0..1` luma scale | ≥ 0 | ratio, not value |
| `ssimLow` | SSIM on 1/4-scale luma — isolates layout from grain | −1 … 1 | higher |
| `meanLuma` | mean luma of each image | 0 … 1 | — |
| `saturation` | mean HSV saturation | 0 … 1 | — |
| `hueMean` | saturation-weighted circular mean hue, **turns** | 0 … 1 (circular!) | — |
| `gradient` | RMS central-difference gradient magnitude | ≥ 0 | — |
| `lumaOffset` | signed `render − reference` mean luma | −1 … 1 | 0 |
| `lpips` (separate) | learned perceptual distance | ≥ 0 | lower |
| `clipSimilarity` (separate) | CLIP embedding cosine | −1 … 1 | higher |

`hueMean` is a direction on a circle — compare it with a wrapped difference, never
`abs(a − b)`. `clipSimilarity` is **not** a percentage: unrelated images routinely
score 0.6–0.9, so read it only as a relative signal between candidate renders.

---

## Tests

```powershell
python -m pytest tests/
```

**197 passed, 1 skipped** on the development machine. The skip is
`test_stdlib_png_rejects_interlaced_with_a_clear_error` — the installed Pillow
does not produce an Adam7-interlaced PNG, so the guard cannot be exercised here.
The test reports that rather than passing vacuously.

`tests/` contains the suite for the modules above. One file,
`tests/test_emitted_artifacts.py` (8 tests), was contributed in parallel by
another developer working in this workspace; it drives the whole pipeline
(`examples/liquid-metal.shader.json` → `apps/cli emit` → `validate.validate_file`
→ a verdict attributed to a named tool) and is included in the count above.

The suite covers, among other things:

- `ssim(a, a) == 1.0` within 1e-6, symmetry, and **monotonicity** in brightness
  shift, bounded strictly between identical and unrelated.
- `ssim` equal to a literal transcription of the TypeScript formula.
- PPM and PNG round trips lossless within 8-bit quantisation, on both the stdlib
  and Pillow codec paths, in both cross-directions.
- Histogram binning identical to the TypeScript rule on all 256 levels.
- `band_energy` of white noise ≫ a smooth gradient; the `high` band responds to
  blur and grain; `ts_compat=True` reproduces the browser's bug.
- **The attribution discrimination test:** synthetic pairs differing in exactly
  one way (blur ×2, brightness shift ×2, saturation ×2, added high-frequency noise
  ×2, shifted shape ×2, hue rotation ×2) each routed to the correct rule, plus
  direction assertions for five of them.
- Every rule's `(node kind, input)` pair checked against the real
  `packages/ir/src/catalog/nodes.ts`.
- `validate_source` on a deliberately broken shader returns `ok=False` **iff** a
  compiler is available, and `ok=None` with a reason otherwise — asserted, never
  skipped.
- The `dxc` encoding trap characterised end to end, including reproducing
  `0x80070459` with no BOM.
- Real LPIPS: `lpips(a,a) == 0`, monotonic under perturbation, and *not* secretly
  SSIM. The `None` path is tested by **injecting** unavailability, so the
  degradation is covered without pretending the weights are missing.
- The batch report is byte-identical across runs and contains no absolute paths
  or timestamps.

## No network

There is no network code in this package. No download, no telemetry, no fetch.
The only subprocesses are the shader compilers and the Node CLI. Model loading is
wrapped in `score.no_network()`, and `tests/test_metrics_optional.py` opens a real
connection attempt inside that guard to prove it raises.
