/**
 * AI4Shader — Node catalog
 * ========================
 *
 * The 31 kinds in `NODE_KINDS` (types.ts), each with its ports, its literal defaults and
 * the restricted-GLSL body the emitter expands. This file is the ONLY place a port name,
 * a default value or a slider range may be written down.
 *
 * ── HOW A BODY IS USED ─────────────────────────────────────────────────────────
 *
 * `glsl` is a statement body ending in exactly one `return <expr>;`. At emit time
 * (`expandTemplate` + `splitReturn` + `emitBodies` in emitter.ts):
 *
 *   1. every `${inputName}` becomes the already-rendered expression for that input
 *      (a local such as `L_n_fbm_1_value`, a uniform name, or a literal like `vec2(0.3, 0.3)`),
 *   2. `${OCTAVE_LIMIT}` becomes the current quality tier's loop bound,
 *   3. the statements before the `return` are emitted verbatim, and the returned
 *      expression is assigned to a `L_<node>_<port>` local of the declared output type.
 *
 * ── HOUSE RULES (machine-checked in test/catalog.test.ts) ──────────────────────
 *
 *   • every `${...}` names a declared input, or is `OCTAVE_LIMIT`, or is `node_id`
 *     (`expandTemplate` substitutes the node's own id; see the scoping rule below);
 *   • exactly one `return`, and it is the last statement;
 *   • float literals ALWAYS carry a decimal point (`2.0`, never `2`); ints become floats
 *     via `float(i)`, never implicitly;
 *   • only the portable functions, the `_`-suffixed aliases and the helpers declared in
 *     `GLSL_LIBRARY` may be called — no invented helpers;
 *   • no `#`, `precision`, `layout`, `uniform`, samplers, `texture(...)` (use `tex2D_`),
 *     `discard`, arrays, structs, `while`/`do`/`switch`, raw `break;`;
 *   • `for (int i = 0; i < ${OCTAVE_LIMIT}; i++)` is the only loop shape, and only for
 *     nodes flagged `needs.qualityScaled`;
 *   • comments inside a body must not contain a backtick or the text `${`, and must not
 *     contain the word `return` (splitReturn scans for the last occurrence of it);
 *   • **a multi-output body returns a `vec4`** — see the note below;
 *   • **every local a body declares is namespaced with `${node_id}`** — see the
 *     scoping note below.
 *
 * ── SCOPING: WHY LOCALS CARRY `${node_id}` ────────────────────────────────────
 *
 * `emitBodies` concatenates every node's statements into ONE flat list, and the targets
 * splice that list into a single function body (the epilogue reads the `L_<node>_<port>`
 * locals from the same scope). Nothing wraps a node body in `{ }` — and it could not:
 * the body's `return` is the last statement, so a body cannot close a block it opened.
 *
 * Consequence: two nodes of the same kind would both declare `vec2 p` in that one scope
 * and the shader would fail to compile with "redefinition of 'p'". Locals inside a body's
 * own `if`/`for` blocks are already scoped, but they are namespaced too, so the rule stays
 * uniform and reviewable: every declaration reads `<name>_${node_id}`.
 *
 * `${node_id}` is substituted verbatim (not through `sanitizeId`), so node ids must
 * already be GLSL identifiers — which is exactly what `patcher.makeNodeId` produces
 * (`fbm`, `fbm_2`, ...) and what the patcher accepts for an explicit `node.id`.
 *
 * ── MULTI-OUTPUT NODES: THE vec4 CONTRACT ─────────────────────────────────────
 *
 * A node may now declare up to four outputs, and `emitBodies` destructures them
 * properly. The contract it enforces (`IR033_BAD_MULTI_OUTPUT`) is:
 *
 *   • `outputs.length <= 4`, and the ports are taken in declaration order as the
 *     `.x/.y/.z/.w` components of the body's return value;
 *   • `TYPE_ARITY[port.type] === 1` for every port — **scalars only**
 *     (`float`/`int`/`bool`). A `vec2` port would have to alias two components, which
 *     the token rewriter cannot lower identically to all six targets;
 *   • therefore such a body must `return vec4(...)`, not a scalar.
 *
 * A body still declares exactly ONE `return`, as its last statement; the tuple is built
 * in that return (locals first, if the maths needs them). Emission is
 * `vec4 tmp_<id> = <returned expr>;` followed by one `L_<id>_<port> = tmp_<id>.<comp>;`
 * per port, so a port keeps its own `float`/`int` local and every wire is typed.
 *
 * `primaryOut` is the port a bare `{"node": "id"}` wire resolves to. It need not be the
 * first declared port — `sdf` declares `distance` first but keeps `mask` primary, because
 * that is the port graphs already wire — and `catalog.test.ts` checks it names a real port.
 *
 * ── WHY `uv` AND `tile` STILL HAVE ONE OUTPUT ─────────────────────────────────
 *
 * Both need a **pair of vec2 outputs** (`uv` + `centered`, `uv` + `cell`), and the
 * scalar-only rule above cannot express that: `TYPE_ARITY[port.type] === 1` rejects any
 * `vec2` port, and promoting one vec2 to two scalars is not enough while the other stays
 * a vec2. `time`/`polar`/`voronoi`/`sdf` got back exactly the ports that *are* scalar
 * (sin/cos, radius, f2/edges/cell, distance), and only these two remain single-output.
 * `catalog.test.ts` test 1d pins that, so a future change making vec2 ports legal is
 * forced to notice them rather than leaving them silently behind. In the meantime the
 * summaries carry the in-graph recipe for the missing half (`transform` scale (2,2)
 * offset (-0.5,-0.5) for `centered`; `voronoi` for a per-cell id).
 */

import type { NodeKind, NodeSchema, ParamValue } from '../types.js';

/* -------------------------------------------------------------------------- */
/* Small builders — keep the defaults readable and the ranges impossible to   */
/* typo. Every builder returns a plain ParamValue (the IR stays JSON).         */
/* -------------------------------------------------------------------------- */

/**
 * Float slider: value + the range that drives the tuning UI.
 *
 * Note what is deliberately absent: a `label`. A literal `ParamValue` has no label field and
 * nothing reads one, so any label threaded through here would be an unreachable second copy
 * of the caption the user actually sees — which comes from `PortSpec.label`, where the
 * semantic name of an input belongs. These builders used to accept and forward one, which
 * type-checked as four errors the moment anyone ran `tsc` over the catalog, and was dead code
 * before that.
 */
const f = (value: number, min: number, max: number, step = 0.01): ParamValue => ({
  kind: 'float',
  value,
  min,
  max,
  step,
});

const v2 = (x: number, y: number): ParamValue => ({ kind: 'vec2', value: [x, y] });
const v3 = (x: number, y: number, z: number): ParamValue => ({
  kind: 'vec3',
  value: [x, y, z],
});
/**
 * A vec4 literal for a *direction* or a *position*, not a colour.
 *
 * `w = 0` by default because that is what a direction needs: `normal`, `view_dir` and
 * `light_dir` all carry `w = 0` so a plain `dot` of two of them is the 3-component dot
 * product. A position passes `w = 1`. Using `color()` for these would work numerically and
 * read wrong — the next person would assume a colour and be surprised by the alpha.
 */
const v4 = (x: number, y: number, z: number, w = 0): ParamValue => ({
  kind: 'vec4',
  value: [x, y, z, w],
});
const color = (r: number, g: number, b: number, a = 1): ParamValue => ({
  kind: 'color',
  value: [r, g, b, a],
});

/* -------------------------------------------------------------------------- */
/* Catalog                                                                     */
/* -------------------------------------------------------------------------- */

export const CATALOG: Record<NodeKind, NodeSchema> = {
  /* ------------------------------------------------------------------ */
  /* Generators — no graph inputs; they read the host or a literal.      */
  /* ------------------------------------------------------------------ */

  uv: {
    kind: 'uv',
    title: 'UV',
    category: 'generator',
    summary:
      'Normalised screen coordinates in 0..1, origin bottom-left. Body returns the epilogue-declared `uv` local directly. ' +
      'centered (-1..1) = `transform` with scale (2,2) and offset (-0.5,-0.5); fragCoord = `uv` * `resolution`. ' +
      'It is NOT a second port: `centered` is a vec2 and multi-output ports must be scalar (see the header note).',    inputs: [],
    outputs: [{ name: 'uv', type: 'vec2', label: 'uv' }],
    primaryOut: 'uv',
    // COUPLING: `uv` is declared by the target epilogue *before* the node bodies run, so
    // it must resolve to the name `uv`. It cannot be a placeholder: this node has no
    // inputs, and expandTemplate would report an unknown-placeholder error.
    glsl: `
      return uv;
    `,
  },

  time: {
    kind: 'time',
    title: 'Time',
    category: 'generator',
    summary:
      'Animated seconds * speed, from the host clock, with its sine and cosine alongside. Ports: `time`, `sin`, `cos`. ' +
      'An oscillator no longer needs a `math` node per axis — wire `sin` for the 0-centred wave, `time` for the raw clock.',
    inputs: [{ name: 'speed', type: 'float', label: 'Speed', default: f(1, 0, 4, 0.01) }],
    outputs: [
      { name: 'time', type: 'float', label: 'Time' },
      { name: 'sin', type: 'float', label: 'Sin' },
      { name: 'cos', type: 'float', label: 'Cos' },
    ],
    primaryOut: 'time',
    // MULTI-OUTPUT: .x = time, .y = sin(time), .z = cos(time), .w unused (0.0).
    glsl: `
      float t_\${node_id} = iTime * \${speed};
      return vec4(t_\${node_id}, sin(t_\${node_id}), cos(t_\${node_id}), 0.0);
    `,
  },

  mouse: {
    kind: 'mouse',
    title: 'Mouse',
    category: 'generator',
    summary: 'Pointer state from the host: xy = current position (pixels), zw = click position.',
    inputs: [],
    outputs: [{ name: 'mouse', type: 'vec4', label: 'Mouse' }],
    primaryOut: 'mouse',
    glsl: `
      return iMouse;
    `,
  },

  resolution: {
    kind: 'resolution',
    title: 'Resolution',
    category: 'generator',
    summary: 'Viewport size in pixels. Divide fragCoord by it to get uv; multiply uv by it to get pixels.',
    inputs: [],
    outputs: [{ name: 'resolution', type: 'vec2', label: 'Resolution' }],
    primaryOut: 'resolution',
    glsl: `
      return iResolution.xy;
    `,
  },

  const_color: {
    kind: 'const_color',
    title: 'Color',
    category: 'generator',
    summary: 'A flat RGBA constant — the usual source for tint, background and palette stops. Expose it as a uniform to tune it.',
    inputs: [{ name: 'value', type: 'vec4', label: 'Color', default: color(0.25, 0.55, 1, 1) }],
    outputs: [{ name: 'color', type: 'vec4', label: 'Color' }],
    primaryOut: 'color',
    glsl: `
      return \${value};
    `,
  },
  /* ------------------------------------------------------------------ */
  /* Surface inputs                                                      */
  /* ------------------------------------------------------------------ */

  /*
   * These five are what makes a **shading** graph possible rather than only a pattern.
   *
   * Everything else in the catalog describes a picture: a function of screen coordinates and
   * time. That is enough for a background, a texture or a mask, and it is not enough for
   * cel shading, because the defining operation of cel shading is `quantize(N · L)` — and
   * neither `N` nor `L` exists in a picture.
   *
   * So they are generators whose value comes from the *host*, in exactly the way `iTime` and
   * `iMouse` already did. What is new is that they cannot be supplied by a fullscreen quad,
   * so a graph that reads one changes the shape of every emitted shader (see
   * `NodeSchema.needs.surface`).
   *
   * All four output `vec4` with `w = 0` for directions, `w = 1` for positions. Vec4 rather
   * than vec3 is not arbitrary: the connection rule only widens, so a vec3 output could not
   * reach `vec_op`, `mix`, `blend` or `dot`, all of which take vec4 — it could reach nothing
   * at all, which is exactly the trap `mouse` fell into.
   */

  normal: {
    kind: 'normal',
    title: 'Surface Normal',
    category: 'surface',
    summary:
      'Unit surface normal in world space (xyz; w = 0). The `N` of every lighting term — ' +
      'wire it into `dot` with `light_dir` for a diffuse ramp. Makes the shader a surface shader.',
    inputs: [],
    outputs: [{ name: 'normal', type: 'vec4', label: 'Normal' }],
    primaryOut: 'normal',
    needs: { surface: true },
    glsl: `
      return vec4(iNormal, 0.0);
    `,
  },

  world_pos: {
    kind: 'world_pos',
    title: 'World Position',
    category: 'surface',
    summary:
      'Fragment position in world space (xyz; w = 1). Use it for triplanar texturing, distance fog ' +
      'or a gradient anchored to the object rather than to the screen.',
    inputs: [],
    outputs: [{ name: 'position', type: 'vec4', label: 'Position' }],
    primaryOut: 'position',
    needs: { surface: true },
    glsl: `
      return vec4(iWorldPos, 1.0);
    `,
  },

  view_dir: {
    kind: 'view_dir',
    title: 'View Direction',
    category: 'surface',
    summary:
      'Unit vector from the fragment towards the camera (xyz; w = 0). `dot(normal, view_dir)` is ' +
      'the facing term: 1 head-on, 0 at the silhouette — the input to rim light and fresnel.',
    inputs: [],
    outputs: [{ name: 'direction', type: 'vec4', label: 'Direction' }],
    primaryOut: 'direction',
    needs: { surface: true },
    glsl: `
      return vec4(iViewDir, 0.0);
    `,
  },

  light_dir: {
    kind: 'light_dir',
    title: 'Light Direction',
    category: 'surface',
    summary:
      'Unit vector from the fragment towards the key light (xyz; w = 0). A host uniform, so the ' +
      'workbench exposes it as a control and an engine sets it per scene. `dot(normal, light_dir)` ' +
      'is the raw 0..1 diffuse term a toon ramp quantises.',
    inputs: [],
    outputs: [{ name: 'direction', type: 'vec4', label: 'Direction' }],
    primaryOut: 'direction',
    needs: { surface: true },
    glsl: `
      return vec4(iLightDir, 0.0);
    `,
  },

  dot: {
    kind: 'dot',
    title: 'Dot Product',
    category: 'surface',
    summary:
      'dot(a.xyz, b.xyz) — the scalar projection of one direction onto another. `dot(normal, light_dir)` ' +
      'gives the diffuse term; `dot(normal, view_dir)` gives the facing term. Ignore the w components.',
    inputs: [
      { name: 'a', type: 'vec4', label: 'A', default: v4(0, 0, 1) },
      { name: 'b', type: 'vec4', label: 'B', default: v4(0, 1, 0) },
    ],
    outputs: [{ name: 'value', type: 'float', label: 'Value' }],
    primaryOut: 'value',
    glsl: `
      return dot(\${a}.xyz, \${b}.xyz);
    `,
  },

  /* ------------------------------------------------------------------ */
  /* Vector construction and decomposition                               */
  /* ------------------------------------------------------------------ */

  /*
   * Why these three exist, and why their absence was a real limitation rather than a gap in
   * convenience.
   *
   * The connection rule only *widens* (float -> vecN). There is no implicit narrowing and no
   * swizzle on a wire, so before these nodes a `vec4` producer could only ever feed a `vec4`
   * input. The `mouse` node outputs a single `vec4` (xy = pointer, zw = press) and therefore
   * could not reach **any** spatial input at all: not `transform.offset` (vec2), not
   * `noise.uv` (vec2), nothing. Drawing at the pointer was not hard, it was impossible, and
   * the wire that tried was silently dropped by the repairer as a type mismatch.
   *
   * `split_vec4` down to scalars and `make_vec2` back up is the missing half of the type
   * system expressed as two nodes rather than as a change to the connection rule. That keeps
   * the rule — and every target's lowering — exactly as it was.
   */

  split_vec4: {
    kind: 'split_vec4',
    title: 'Split Vector',
    category: 'math',
    summary:
      'Breaks a vec4 into its four scalar components. The way to get numbers out of `mouse`, ' +
      '`normal` or `world_pos`: feed a component into `math`/`smoothstep`, or rebuild a ' +
      'different-shaped vector with `make_vec2`.',
    inputs: [{ name: 'value', type: 'vec4', label: 'Vector', default: v4(1, 0, 0, 1) }],
    outputs: [
      { name: 'x', type: 'float', label: 'X' },
      { name: 'y', type: 'float', label: 'Y' },
      { name: 'z', type: 'float', label: 'Z' },
      { name: 'w', type: 'float', label: 'W' },
    ],
    primaryOut: 'x',
    // The multi-output contract: one vec4 whose components become the ports, in order.
    glsl: `
      return \${value};
    `,
  },

  split_vec2: {
    kind: 'split_vec2',
    title: 'Split Vec2',
    category: 'math',
    summary:
      'Breaks a vec2 into its two scalar components. The vec2 counterpart of `split_vec4`, and ' +
      'the way to get numbers out of `uv` or `resolution` — e.g. divide a pointer position by ' +
      'the viewport size to reach uv space.',
    inputs: [{ name: 'value', type: 'vec2', label: 'Vector', default: v2(0.5, 0.5) }],
    outputs: [
      { name: 'x', type: 'float', label: 'X' },
      { name: 'y', type: 'float', label: 'Y' },
    ],
    primaryOut: 'x',
    // A vec4 whose first two components carry the ports, in declaration order — the same
    // convention `polar` and `sdf` use for their non-vec4 results.
    glsl: `
      return vec4(\${value}.x, \${value}.y, 0.0, 0.0);
    `,
  },

  make_vec2: {
    kind: 'make_vec2',
    title: 'Make Vec2',
    category: 'math',
    summary:
      'Builds a vec2 from two scalars — a coordinate out of two numbers. Pair it with ' +
      '`split_vec4` to reshape a vec4 into the vec2 that a domain or noise node wants.',
    inputs: [
      { name: 'x', type: 'float', label: 'X', default: f(0.5, -8, 8, 0.001) },
      { name: 'y', type: 'float', label: 'Y', default: f(0.5, -8, 8, 0.001) },
    ],
    outputs: [{ name: 'value', type: 'vec2', label: 'Value' }],
    primaryOut: 'value',
    glsl: `
      return vec2(\${x}, \${y});
    `,
  },

  make_vec4: {
    kind: 'make_vec4',
    title: 'Make Vec4',
    category: 'math',
    summary:
      'Builds a vec4 from four scalars — the inverse of `split_vec4`. The usual way to assemble ' +
      'a computed colour, or a direction from three separately-derived numbers.',
    inputs: [
      { name: 'x', type: 'float', label: 'X', default: f(0, 0, 1, 0.001) },
      { name: 'y', type: 'float', label: 'Y', default: f(0, 0, 1, 0.001) },
      { name: 'z', type: 'float', label: 'Z', default: f(0, 0, 1, 0.001) },
      { name: 'w', type: 'float', label: 'W', default: f(1, 0, 1, 0.001) },
    ],
    outputs: [{ name: 'value', type: 'vec4', label: 'Value' }],
    primaryOut: 'value',
    glsl: `
      return vec4(\${x}, \${y}, \${z}, \${w});
    `,
  },

  /* ------------------------------------------------------------------ */
  /* Vector arithmetic                                                   */
  /* ------------------------------------------------------------------ */

  vec_math: {
    kind: 'vec_math',
    title: 'Vector Math',
    category: 'math',
    summary:
      'Vector counterpart of `math`, selected by `op`: 0 normalize, 1 length, 2 distance, ' +
      '3 reflect, 4 fract, 5 floor, 6 abs, 7 pow(|a|,b), 8 max, 9 min, 10 sign, 11 clamp(a,0,1), ' +
      '12 cross. Returns a vec4 in every case, so it chains into `mix`/`blend`/`vec_op`; the ' +
      'scalar-valued ops (length, distance) are broadcast across all four components. ' +
      'The ops whose mathematical meaning lives in three dimensions — normalize, length, ' +
      'distance, reflect, cross — read `.xyz` and ignore `w`, because a vec4 in this IR carries ' +
      'its kind in `w` (0 for a direction, 1 for a point) and `dot` has always done the same. ' +
      'The rest are component-wise and include `w`.',
    inputs: [
      { name: 'a', type: 'vec4', label: 'A', default: v4(0, 0, 1) },
      { name: 'b', type: 'vec4', label: 'B', default: v4(0, 1, 0) },
      { name: 'op', type: 'float', label: 'Op', default: f(0, 0, 12, 1) },
    ],
    outputs: [{ name: 'value', type: 'vec4', label: 'Value' }],
    primaryOut: 'value',
    glsl: `
      // Declaration only — every branch below assigns. It used to initialise to \`normalize(\${a})\`,
      // which is the one op that is *not* component-wise (see the summary): a stray full-vec4
      // normalisation of the first input, executed on every call whatever the op, for no result.
      // Written with four arguments on purpose: \`vec4(0.0)\` is a scalar splat, which GLSL accepts
      // and **HLSL does not** (\`dxc\`: "too few elements in vector initialization").
      vec4 r_\${node_id} = vec4(0.0, 0.0, 0.0, 0.0);
      if (\${op} < 0.5) {
        // \`.xyz\`, like reflect/cross/dot: \`w\` is the direction-vs-point tag, not a component.
        r_\${node_id} = vec4(normalize(\${a}.xyz), 0.0);
      } else if (\${op} < 1.5) {
        float l_\${node_id} = length(\${a}.xyz);
        r_\${node_id} = vec4(l_\${node_id}, l_\${node_id}, l_\${node_id}, l_\${node_id});
      } else if (\${op} < 2.5) {
        float d_\${node_id} = distance(\${a}.xyz, \${b}.xyz);
        r_\${node_id} = vec4(d_\${node_id}, d_\${node_id}, d_\${node_id}, d_\${node_id});
      } else if (\${op} < 3.5) {
        r_\${node_id} = vec4(reflect(\${a}.xyz, normalize(\${b}.xyz)), 0.0);
      } else if (\${op} < 4.5) {
        r_\${node_id} = fract_(\${a});
      } else if (\${op} < 5.5) {
        r_\${node_id} = floor(\${a});
      } else if (\${op} < 6.5) {
        r_\${node_id} = abs(\${a});
      } else if (\${op} < 7.5) {
        r_\${node_id} = pow(abs(\${a}), \${b});
      } else if (\${op} < 8.5) {
        r_\${node_id} = max(\${a}, \${b});
      } else if (\${op} < 9.5) {
        r_\${node_id} = min(\${a}, \${b});
      } else if (\${op} < 10.5) {
        r_\${node_id} = sign(\${a});
      } else if (\${op} < 11.5) {
        r_\${node_id} = clamp(\${a}, vec4(0.0, 0.0, 0.0, 0.0), vec4(1.0, 1.0, 1.0, 1.0));
      } else {
        r_\${node_id} = vec4(cross(\${a}.xyz, \${b}.xyz), 0.0);
      }
      return r_\${node_id};
    `,
  },

  /* ------------------------------------------------------------------ */
  /* Escape hatch                                                        */
  /* ------------------------------------------------------------------ */

  /**
   * The one node whose body is not in this file.
   *
   * A catalog cannot be complete: any list of effects is a list of the effects someone already
   * thought of, and the whole point of this project is that the user names the one nobody did.
   * `custom_glsl` is the acknowledgement, and it was designed for from the start — DESIGN.md
   * lists it as P3 and this file's header calls the alternative "硬凑" — but it was never built,
   * so anything outside the 31 original kinds ended up in `spec.outOfScope` with an apology.
   *
   * ## Why it is safe to let code into the graph
   *
   * Because the code is written in the **restricted GLSL subset** the catalog bodies already
   * use, and it goes through the **same pipeline**: `lintSubset` rejects the banned spellings,
   * `expandTemplate` substitutes the `${a}`…`${d}` placeholders, and `rewriteSource` renames
   * types and `mix_`/`mod_`/`fract_` per target and lowers matrix products. Nothing about the
   * six-target guarantee is weakened — a snippet is lowered by exactly the machinery that
   * lowers `fbm` or `voronoi`, because as far as that machinery is concerned it *is* a body.
   *
   * ## The port list, and the mistake the first version of it made
   *
   * A schema is per-*kind*, so the ports cannot be declared per instance. The first version
   * declared four `vec4` inputs and claimed in this comment that "a float, vec2 or vec3 wires
   * in and is splatted". **That was false.** `canConnect` widens `float`/`int`/`bool` to a
   * vector — it does *not* widen `vec2` to `vec4`, so `uv` could not reach the node at all and
   * every custom effect that needed a coordinate would have had to go `split_vec2` →
   * `make_vec4` first. The summary said one thing and the type system did another.
   *
   * So there is a `uv` port of its own. The node's job is "an effect at a coordinate", and
   * naming that port explicitly costs one line and saves every author the same three-node
   * dance. The four `vec4` ports carry colours, directions and assembled values; a scalar
   * widens into them, and a vec2/vec3 needs `make_vec4` — which is now stated rather than
   * assumed.
   *
   * The snippet must `return` a vec4 (the ports read `.x/.y/.z/.w`), and it may call any
   * function in the shared library — `noise2`, `fbm2`, `voronoi2`, `hsv2rgb`, `luma`,
   * `sd_circle` … — by the names that library uses, since those names are already lowered for
   * every target.
   */
  custom_glsl: {
    kind: 'custom_glsl',
    title: 'Custom GLSL',
    category: 'math',
    summary:
      'ESCAPE HATCH: a body you write yourself, in the same restricted GLSL subset the built-in ' +
      'nodes use. A `uv` coordinate, four vec4 inputs (a,b,c,d) and one vec4 output; reference ' +
      'them as ${uv} ${a}…${d} and end with `return vec4(...)`. Use it when no combination of ' +
      'the other kinds can express the effect — it is lowered to all six targets by the same ' +
      'rewriter, so it is not a per-target special case. May call the shared library (noise2, ' +
      'fbm2, voronoi2, hsv2rgb, luma, sd_circle, …). A scalar widens into a vec4 port; a vec2 ' +
      'or vec3 must go through `make_vec4` first, because the connection rule only widens from ' +
      'a scalar. **Until a body is written the node returns `a` unchanged, so the other inputs ' +
      'do not affect the result** — and that state is reported as `IR034_CUSTOM_BODY_EMPTY`, ' +
      'because a pass-through is not the effect anyone asked for.',
    inputs: [
      { name: 'uv', type: 'vec2', label: 'UV', default: v2(0.5, 0.5) },
      { name: 'a', type: 'vec4', label: 'A', default: color(0, 0, 0, 1) },
      { name: 'b', type: 'vec4', label: 'B', default: color(1, 1, 1, 1) },
      { name: 'c', type: 'vec4', label: 'C', default: color(0.5, 0.5, 0.5, 1) },
      { name: 'd', type: 'vec4', label: 'D', default: color(0, 1, 0, 1) },
    ],
    outputs: [{ name: 'value', type: 'vec4', label: 'Value' }],
    primaryOut: 'value',
    // A placeholder that is *valid on its own*: a `custom_glsl` node added without a body must
    // still emit and compile, so the default is a pass-through rather than an empty string.
    // `node.body` normally replaces this; the fallback exists so that a half-built graph is a
    // graph, and the difference shows up as "nothing happened" rather than as a syntax error.
    glsl: `
      return \${a};
    `,
  },

  const_float: {
    kind: 'const_float',
    title: 'Value',
    category: 'generator',
    summary: 'A flat scalar constant — the usual source for masks, thresholds and blend factors.',
    inputs: [{ name: 'value', type: 'float', label: 'Value', default: f(0.5, -4, 4, 0.001) }],
    outputs: [{ name: 'value', type: 'float', label: 'Value' }],
    primaryOut: 'value',
    glsl: `
      return \${value};
    `,
  },

  gradient: {
    kind: 'gradient',
    title: 'Linear Gradient',
    category: 'generator',
    summary: 'A repeating linear ramp: fract(dot(uv, direction(angle)) * scale). 0..1, one band per unit of scale.',
    inputs: [
      { name: 'uv', type: 'vec2', label: 'UV', default: v2(0.5, 0.5) },
      { name: 'angle', type: 'float', label: 'Angle', default: f(0, 0, 6.2832, 0.001) },
      { name: 'scale', type: 'float', label: 'Scale', default: f(4, 0.1, 64, 0.1) },
    ],
    outputs: [{ name: 'value', type: 'float', label: 'Value' }],
    primaryOut: 'value',
    glsl: `
      return fract_(dot(\${uv}, vec2(cos(\${angle}), sin(\${angle}))) * \${scale});
    `,
  },

  texture: {
    kind: 'texture',
    title: 'Texture',
    category: 'generator',
    summary:
      'Samples the host image channel (iChannel0 in the GLSL/WebGL2 host; the target wires its own binding) at uv, ' +
      'scaled by mix_amount. RGBA, linear light. Target support: needs a bound channel; check targetNotes when exporting.',
    inputs: [
      { name: 'uv', type: 'vec2', label: 'UV', default: v2(0.5, 0.5) },
      { name: 'mix_amount', type: 'float', label: 'Amount', default: f(1, 0, 1, 0.01) },
    ],
    outputs: [{ name: 'color', type: 'vec4', label: 'Color' }],
    primaryOut: 'color',
    // The one node that reads the host image channel, which is why the flag exists: a host has
    // to bind *something* there, and `render.feedback` is only meaningful when this node is
    // reachable. Declared here rather than sniffed from the body text so the reachability
    // analysis and the binding requirement can never disagree.
    needs: { channel: true },
    glsl: `
      return tex2D_(\${uv}) * \${mix_amount};
    `,
  },

  /* ------------------------------------------------------------------ */
  /* Domain — remap the coordinate space. All of them take and return uv. */
  /* ------------------------------------------------------------------ */

  transform: {
    kind: 'transform',
    title: 'Transform',
    category: 'domain',
    summary:
      'Rotate (about uv 0.5,0.5), scale and offset uv. Recipes: centered coords = scale (2,2) offset (-0.5,-0.5); ' +
      'zoom = scale > 1; mirror = negative scale.',
    inputs: [
      { name: 'uv', type: 'vec2', label: 'UV', default: v2(0.5, 0.5) },
      { name: 'scale', type: 'vec2', label: 'Scale', default: v2(1, 1) },
      { name: 'rotate', type: 'float', label: 'Rotate', default: f(0, 0, 6.2832, 0.001) },
      { name: 'offset', type: 'vec2', label: 'Offset', default: v2(0, 0) },
    ],
    outputs: [{ name: 'uv', type: 'vec2', label: 'UV' }],
    primaryOut: 'uv',
    needs: { matrix: true },
    glsl: `
      return mat2_mul_(rot2(\${rotate}), (\${uv} - vec2(0.5, 0.5)) * \${scale}) + \${offset} + vec2(0.5, 0.5);
    `,
  },

  tile: {
    kind: 'tile',
    title: 'Tile',
    category: 'domain',
    summary:
      'Repeats the space: uv * count wrapped into one cell (0..1). The `cell` id (floor(uv * count)) is not a port: it is a ' +
      'vec2 and multi-output ports must be scalar (see the header note), so use `voronoi` for per-cell randomness, or ' +
      'expose count and read the pattern in one cell.',
    inputs: [
      { name: 'uv', type: 'vec2', label: 'UV', default: v2(0.5, 0.5) },
      { name: 'count', type: 'vec2', label: 'Count', default: v2(4, 4) },
    ],
    outputs: [{ name: 'uv', type: 'vec2', label: 'UV (in cell)' }],
    primaryOut: 'uv',
    glsl: `
      return fract_(\${uv} * \${count});
    `,
  },

  kaleidoscope: {
    kind: 'kaleidoscope',
    title: 'Kaleidoscope',
    category: 'domain',
    summary: 'Folds the space around uv 0.5,0.5 into `segments` mirrored wedges — the classic snowflake/mandala symmetry.',
    inputs: [
      { name: 'uv', type: 'vec2', label: 'UV', default: v2(0.5, 0.5) },
      { name: 'segments', type: 'float', label: 'Segments', default: f(6, 2, 32, 1) },
    ],
    outputs: [{ name: 'uv', type: 'vec2', label: 'UV' }],
    primaryOut: 'uv',
    glsl: `
      vec2 p_\${node_id} = \${uv} - vec2(0.5, 0.5);
      float r_\${node_id} = length(p_\${node_id});
      float a_\${node_id} = atan2_(p_\${node_id}.y, p_\${node_id}.x);
      float seg_\${node_id} = 6.2831853 / max(\${segments}, 2.0);
      a_\${node_id} = abs(mod_(a_\${node_id} + seg_\${node_id} * 0.5, seg_\${node_id}) - seg_\${node_id} * 0.5);
      return vec2(cos(a_\${node_id}), sin(a_\${node_id})) * r_\${node_id} + vec2(0.5, 0.5);
    `,
  },

  polar: {
    kind: 'polar',
    title: 'Polar',
    category: 'domain',
    summary:
      'Cartesian to polar, both components as ports: `angle` 0..1 (0.5 = the +x axis, the old packed x) and `radius` in uv ' +
      'units (the old packed y). Feed `angle` into `gradient`/`stripe` for angular effects and `radius` into `sdf` (ring) or ' +
      '`remap` for radial falloff — the pair no longer has to be split with a swizzle node.',
    inputs: [
      { name: 'uv', type: 'vec2', label: 'UV', default: v2(0.5, 0.5) },
      { name: 'center', type: 'vec2', label: 'Center', default: v2(0.5, 0.5) },
    ],
    outputs: [
      { name: 'angle', type: 'float', label: 'Angle (0..1)' },
      { name: 'radius', type: 'float', label: 'Radius' },
    ],
    primaryOut: 'angle',
    // MULTI-OUTPUT: .x = angle (unchanged 0..1 convention), .y = radius, .zw unused.
    glsl: `
      vec2 p_\${node_id} = \${uv} - \${center};
      float a_\${node_id} = atan2_(p_\${node_id}.y, p_\${node_id}.x) / 6.2831853 + 0.5;
      float r_\${node_id} = length(p_\${node_id});
      return vec4(a_\${node_id}, r_\${node_id}, 0.0, 0.0);
    `,
  },

  domain_warp: {
    kind: 'domain_warp',
    title: 'Domain Warp',
    category: 'domain',
    summary:
      'Pushes uv along a smooth 2D noise field (Inigo Quilez style warp) — the single cheapest way to make anything look organic. ' +
      'Wire `time` into `time` to make the field drift.',
    inputs: [
      { name: 'uv', type: 'vec2', label: 'UV', default: v2(0.5, 0.5) },
      { name: 'strength', type: 'float', label: 'Strength', default: f(0.15, 0, 0.5, 0.001) },
      { name: 'scale', type: 'float', label: 'Scale', default: f(3, 0.1, 64, 0.1) },
      { name: 'time', type: 'float', label: 'Time', default: f(0, 0, 60, 0.01) },
      { name: 'tile', type: 'float', label: 'Tileable', default: f(0, 0, 1, 1) },
    ],
    outputs: [{ name: 'uv', type: 'vec2', label: 'UV' }],
    primaryOut: 'uv',
    glsl: `
      vec2 p_\${node_id} = \${uv} * \${scale} + vec2(\${time}, \${time} * 0.5);
      vec2 per_\${node_id} = vec2(\${scale}, \${scale});
      vec2 q_\${node_id} = vec2(noise2t(p_\${node_id}, per_\${node_id}, \${tile}), noise2t(p_\${node_id} + vec2(13.7, 7.1), per_\${node_id}, \${tile})) - vec2(0.5, 0.5);
      return \${uv} + q_\${node_id} * \${strength};
    `,
  },

  /* ------------------------------------------------------------------ */
  /* Noise / pattern — all scalars, all 0..1 unless noted.               */
  /* ------------------------------------------------------------------ */

  noise: {
    kind: 'noise',
    title: 'Noise',
    category: 'noise',
    summary:
      'Single-octave noise in 0..1. mode 0 = value noise (noise2), 1 = gradient/Perlin (gnoise2), ' +
      '2 = turbulence (|2n-1|, i.e. ridged at one octave). Selects branchlessly with step + mix.',
    inputs: [
      { name: 'uv', type: 'vec2', label: 'UV', default: v2(0.5, 0.5) },
      { name: 'scale', type: 'float', label: 'Scale', default: f(8, 0.1, 64, 0.1) },
      { name: 'offset', type: 'vec2', label: 'Offset', default: v2(0, 0) },
      { name: 'mode', type: 'float', label: 'Mode', default: f(0, 0, 2, 1) },
      { name: 'tile', type: 'float', label: 'Tileable', default: f(0, 0, 1, 1) },
    ],
    outputs: [{ name: 'value', type: 'float', label: 'Value' }],
    primaryOut: 'value',
    glsl: `
      vec2 p_\${node_id} = \${uv} * \${scale} + \${offset};
      vec2 per_\${node_id} = vec2(\${scale}, \${scale});
      float v_\${node_id} = noise2t(p_\${node_id}, per_\${node_id}, \${tile});
      float g_\${node_id} = gnoise2t(p_\${node_id}, per_\${node_id}, \${tile});
      float tb_\${node_id} = abs(noise2t(p_\${node_id}, per_\${node_id}, \${tile}) * 2.0 - 1.0);
      float ig_\${node_id} = step(0.5, \${mode}) * (1.0 - step(1.5, \${mode}));
      float it_\${node_id} = step(1.5, \${mode});
      float n_\${node_id} = mix_(v_\${node_id}, g_\${node_id}, ig_\${node_id});
      return mix_(n_\${node_id}, tb_\${node_id}, it_\${node_id});
    `,
  },

  fbm: {
    kind: 'fbm',
    title: 'FBM',
    category: 'noise',
    summary:
      'Fractal Brownian motion (shared fbm2 helper) in 0..1: octaves of value noise with lacunarity/gain, ' +
      'optionally ridged. Octave count is clamped to the quality tier, so the low tier is cheap by construction.',
    inputs: [
      { name: 'uv', type: 'vec2', label: 'UV', default: v2(0.5, 0.5) },
      { name: 'scale', type: 'float', label: 'Scale', default: f(4, 0.1, 64, 0.1) },
      { name: 'octaves', type: 'float', label: 'Octaves', default: f(5, 1, 8, 1) },
      { name: 'lacunarity', type: 'float', label: 'Lacunarity', default: f(2, 1, 4, 0.01) },
      { name: 'gain', type: 'float', label: 'Gain', default: f(0.5, 0, 1, 0.01) },
      { name: 'ridged', type: 'float', label: 'Ridged', default: f(0, 0, 1, 0.01) },
      { name: 'tile', type: 'float', label: 'Tileable', default: f(0, 0, 1, 1) },
    ],
    outputs: [{ name: 'value', type: 'float', label: 'Value' }],
    primaryOut: 'value',
    needs: { qualityScaled: true },
    glsl: `
      return fbm2t(\${uv} * \${scale}, int(\${octaves}), \${lacunarity}, \${gain}, \${ridged}, vec2(\${scale}, \${scale}), \${tile});
    `,
  },

  voronoi: {
    kind: 'voronoi',
    title: 'Voronoi',
    category: 'noise',
    summary:
      'Worley from the shared voronoi2 helper, all four of its channels as ports: f1 = distance to the nearest jittered ' +
      'cell centre (0..~1.4), f2 = distance to the second nearest, edges = f2 - f1 (thin near cell borders — the classic ' +
      'crack/foam mask) and cell = a per-cell random id in 0..1 (wire it into `color_ramp` or `hsv` for stained glass).',
    inputs: [
      { name: 'uv', type: 'vec2', label: 'UV', default: v2(0.5, 0.5) },
      { name: 'scale', type: 'float', label: 'Scale', default: f(6, 0.1, 64, 0.1) },
      { name: 'randomness', type: 'float', label: 'Randomness', default: f(1, 0, 1, 0.01) },
      { name: 'tile', type: 'float', label: 'Tileable', default: f(0, 0, 1, 1) },
    ],
    outputs: [
      { name: 'f1', type: 'float', label: 'F1' },
      { name: 'f2', type: 'float', label: 'F2' },
      { name: 'edges', type: 'float', label: 'Edges (F2-F1)' },
      { name: 'cell', type: 'float', label: 'Cell ID' },
    ],
    primaryOut: 'f1',
    // MULTI-OUTPUT: .x = v.x (F1), .y = v.y (F2), .z = v.y - v.x (edges), .w = v.z (cell id).
    // All four are scalars, so this is the one node that fills every component.
    glsl: `
      vec3 v_\${node_id} = voronoi2t(\${uv} * \${scale}, \${randomness}, vec2(\${scale}, \${scale}), \${tile});
      return vec4(v_\${node_id}.x, v_\${node_id}.y, v_\${node_id}.y - v_\${node_id}.x, v_\${node_id}.z);
    `,
  },

  stripe: {
    kind: 'stripe',
    title: 'Stripes',
    category: 'noise',
    summary:
      'Soft-edged repeating stripes in 0..1, rotated by `angle` at `frequency` bands per uv unit. ' +
      'width is the filled fraction of each band (0.5 = half filled), so 1.0 - width gives the inverse stripes.',
    inputs: [
      { name: 'uv', type: 'vec2', label: 'UV', default: v2(0.5, 0.5) },
      { name: 'angle', type: 'float', label: 'Angle', default: f(0, 0, 6.2832, 0.001) },
      { name: 'frequency', type: 'float', label: 'Frequency', default: f(8, 0.1, 64, 0.1) },
      { name: 'width', type: 'float', label: 'Width', default: f(0.5, 0.01, 1, 0.01) },
    ],
    outputs: [{ name: 'value', type: 'float', label: 'Value' }],
    primaryOut: 'value',
    glsl: `
      vec2 dir_\${node_id} = vec2(cos(\${angle}), sin(\${angle}));
      float t_\${node_id} = dot(\${uv}, dir_\${node_id}) * \${frequency};
      float band_\${node_id} = abs(fract_(t_\${node_id}) - 0.5) * 2.0;
      return smoothstep(1.0 - clamp(\${width}, 0.001, 1.0), 1.0, band_\${node_id});
    `,
  },

  checker: {
    kind: 'checker',
    title: 'Checker',
    category: 'noise',
    summary: 'Checkerboard: exactly 0.0 or 1.0 per cell, from floor(uv * count). Wire count from a float to keep it square.',
    inputs: [
      { name: 'uv', type: 'vec2', label: 'UV', default: v2(0.5, 0.5) },
      { name: 'count', type: 'vec2', label: 'Count', default: v2(8, 8) },
    ],
    outputs: [{ name: 'value', type: 'float', label: 'Value' }],
    primaryOut: 'value',
    glsl: `
      vec2 c_\${node_id} = floor(\${uv} * \${count});
      return mod_(c_\${node_id}.x + c_\${node_id}.y, 2.0);
    `,
  },

  sdf: {
    kind: 'sdf',
    title: 'Shape Mask (SDF)',
    category: 'noise',
    summary:
      'Signed distance to a primitive plus the mask built from it: `mask` = 1 inside, 0 outside, smoothness = edge width in ' +
      'uv units; `distance` = the raw signed field (negative inside), for outlines, glow falloff or a `remap` into any ' +
      'other band. `mask` stays the primary port, so existing graphs wire it unchanged. shape selects ' +
      'circle/box/hexagon/triangle/star/ring with < comparisons (no switch).',
    inputs: [
      { name: 'uv', type: 'vec2', label: 'UV', default: v2(0.5, 0.5) },
      { name: 'shape', type: 'float', label: 'Shape', default: f(0, 0, 5, 1) },
      { name: 'size', type: 'vec2', label: 'Size / Thickness', default: v2(0.3, 0.3) },
      { name: 'radius', type: 'float', label: 'Radius', default: f(0.3, 0, 1, 0.001) },
      { name: 'smoothness', type: 'float', label: 'Smoothness', default: f(0.01, 0.001, 1, 0.001) },
    ],
    // Declared in the task's `.x/.y` order (distance, mask); `primaryOut` is what keeps a
    // bare wire pointing at `mask`, which is the port every existing graph consumes.
    outputs: [
      { name: 'distance', type: 'float', label: 'Distance' },
      { name: 'mask', type: 'float', label: 'Mask' },
    ],
    primaryOut: 'mask',
    // MULTI-OUTPUT: .x = d, .y = mask, .zw unused. Both ports come from the one distance
    // field, so the mask is bit-identical to the single-output body this replaced.
    glsl: `
      vec2 p_\${node_id} = \${uv} - vec2(0.5, 0.5);
      vec2 sz_\${node_id} = \${size};
      float d_\${node_id} = sd_circle(p_\${node_id}, \${radius});
      if (\${shape} < 0.5) {
        d_\${node_id} = sd_circle(p_\${node_id}, \${radius});
      } else if (\${shape} < 1.5) {
        d_\${node_id} = sd_box(p_\${node_id}, sz_\${node_id});
      } else if (\${shape} < 2.5) {
        d_\${node_id} = sd_hexagon(p_\${node_id}, \${radius});
      } else if (\${shape} < 3.5) {
        d_\${node_id} = sd_triangle(p_\${node_id}, \${radius});
      } else if (\${shape} < 4.5) {
        d_\${node_id} = sd_star(p_\${node_id}, \${radius}, 5.0);
      } else {
        d_\${node_id} = sd_ring(p_\${node_id}, \${radius}, max(sz_\${node_id}.x, 0.001));
      }
      float m_\${node_id} = 1.0 - smoothstep(0.0, max(\${smoothness}, 0.0001), d_\${node_id});
      return vec4(d_\${node_id}, m_\${node_id}, 0.0, 0.0);
    `,
  },

  raymarch: {
    kind: 'raymarch',
    title: 'Heightfield March',
    category: 'noise',
    summary:
      'Quality-scaled octave march over a noise heightfield, shaded into 0..1. ' +
      'The loop bound is the tier budget and `steps` masks iterations, so cost scales with render.qualityTier.',
    inputs: [
      { name: 'uv', type: 'vec2', label: 'UV', default: v2(0.5, 0.5) },
      { name: 'height', type: 'float', label: 'Height', default: f(0.5, 0, 1, 0.01) },
      { name: 'frequency', type: 'float', label: 'Frequency', default: f(4, 0.1, 32, 0.1) },
      { name: 'steps', type: 'float', label: 'Steps', default: f(6, 1, 8, 1) },
      { name: 'time', type: 'float', label: 'Time', default: f(0, 0, 60, 0.01) },
      { name: 'tile', type: 'float', label: 'Tileable', default: f(0, 0, 1, 1) },
    ],
    outputs: [{ name: 'shade', type: 'float', label: 'Shade' }],
    primaryOut: 'shade',
    needs: { qualityScaled: true },
    glsl: `
      vec2 q_\${node_id} = \${uv} * \${frequency} + vec2(0.0, \${time} * 0.5);
      vec2 per_\${node_id} = vec2(\${frequency}, \${frequency});
      float acc_\${node_id} = 0.0;
      float w_\${node_id} = 0.0;
      float amp_\${node_id} = 1.0;
      for (int i = 0; i < \${OCTAVE_LIMIT}; i++) {
        float live_\${node_id} = step(float(i) + 0.5, \${steps});
        float h_\${node_id} = noise2t(q_\${node_id}, per_\${node_id}, \${tile}) * \${height};
        acc_\${node_id} = acc_\${node_id} + h_\${node_id} * amp_\${node_id} * live_\${node_id};
        w_\${node_id} = w_\${node_id} + amp_\${node_id} * live_\${node_id};
        q_\${node_id} = q_\${node_id} * 2.0;
        per_\${node_id} = per_\${node_id} * 2.0;
        amp_\${node_id} = amp_\${node_id} * 0.5;
      }
      return clamp(acc_\${node_id} / max(w_\${node_id}, 0.0001) * 2.0, 0.0, 1.0);
    `,
  },

  /* ------------------------------------------------------------------ */
  /* Math                                                                */
  /* ------------------------------------------------------------------ */

  math: {
    kind: 'math',
    title: 'Math',
    category: 'math',
    summary:
      'Binary scalar op selected by `op`: 0 add, 1 sub, 2 mul, 3 div, 4 pow(|a|,b)*sign(a), 5 min, 6 max, 7 mod, 8 atan2(a,b), ' +
      '9 sin(a), 10 cos(a), 11 abs(a), 12 sqrt(|a|), 13 fract(a), 14 sign(a). ' +
      'Ops 9-14 exist because sin/cos/abs are otherwise unreachable from the graph; division and mod use a guarded divisor so b=0 cannot produce inf/NaN.',
    inputs: [
      { name: 'a', type: 'float', label: 'A', default: f(1, -4, 4, 0.001) },
      { name: 'b', type: 'float', label: 'B', default: f(0.5, -4, 4, 0.001) },
      { name: 'op', type: 'float', label: 'Op', default: f(0, 0, 14, 1) },
    ],
    outputs: [{ name: 'value', type: 'float', label: 'Value' }],
    primaryOut: 'value',
    glsl: `
      float aa_\${node_id} = \${a};
      float bb_\${node_id} = \${b};
      float bd_\${node_id} = bb_\${node_id};
      if (abs(bd_\${node_id}) < 0.0001) {
        bd_\${node_id} = 0.0001;
      }
      float r_\${node_id} = aa_\${node_id} + bb_\${node_id};
      if (\${op} < 0.5) {
        r_\${node_id} = aa_\${node_id} + bb_\${node_id};
      } else if (\${op} < 1.5) {
        r_\${node_id} = aa_\${node_id} - bb_\${node_id};
      } else if (\${op} < 2.5) {
        r_\${node_id} = aa_\${node_id} * bb_\${node_id};
      } else if (\${op} < 3.5) {
        r_\${node_id} = aa_\${node_id} / bd_\${node_id};
      } else if (\${op} < 4.5) {
        r_\${node_id} = pow(abs(aa_\${node_id}), bb_\${node_id}) * sign(aa_\${node_id});
      } else if (\${op} < 5.5) {
        r_\${node_id} = min(aa_\${node_id}, bb_\${node_id});
      } else if (\${op} < 6.5) {
        r_\${node_id} = max(aa_\${node_id}, bb_\${node_id});
      } else if (\${op} < 7.5) {
        r_\${node_id} = mod_(aa_\${node_id}, bd_\${node_id});
      } else if (\${op} < 8.5) {
        r_\${node_id} = atan2_(aa_\${node_id}, bb_\${node_id});
      } else if (\${op} < 9.5) {
        r_\${node_id} = sin(aa_\${node_id});
      } else if (\${op} < 10.5) {
        r_\${node_id} = cos(aa_\${node_id});
      } else if (\${op} < 11.5) {
        r_\${node_id} = abs(aa_\${node_id});
      } else if (\${op} < 12.5) {
        r_\${node_id} = sqrt(abs(aa_\${node_id}));
      } else if (\${op} < 13.5) {
        r_\${node_id} = fract_(aa_\${node_id});
      } else {
        r_\${node_id} = sign(aa_\${node_id});
      }
      return r_\${node_id};
    `,
  },

  vec_op: {
    kind: 'vec_op',
    title: 'Combine (vec4)',
    category: 'math',
    summary:
      'Component-wise RGBA op: 0 add, 1 subtract, 2 multiply, 3 divide (guarded), 4 screen, 5 overlay, 6 difference. ' +
      'The workhorse for combining colours and masks — opacity is left to `blend`/`mix`.',
    inputs: [
      { name: 'a', type: 'vec4', label: 'A', default: color(0, 0, 0, 1) },
      { name: 'b', type: 'vec4', label: 'B', default: color(1, 1, 1, 1) },
      { name: 'op', type: 'float', label: 'Op', default: f(0, 0, 6, 1) },
    ],
    outputs: [{ name: 'value', type: 'vec4', label: 'Value' }],
    primaryOut: 'value',
    glsl: `
      vec4 one_\${node_id} = vec4(1.0, 1.0, 1.0, 1.0);
      vec4 r_\${node_id} = \${a} + \${b};
      if (\${op} < 0.5) {
        r_\${node_id} = \${a} + \${b};
      } else if (\${op} < 1.5) {
        r_\${node_id} = \${a} - \${b};
      } else if (\${op} < 2.5) {
        r_\${node_id} = \${a} * \${b};
      } else if (\${op} < 3.5) {
        r_\${node_id} = \${a} / max(\${b}, vec4(0.0001, 0.0001, 0.0001, 0.0001));
      } else if (\${op} < 4.5) {
        r_\${node_id} = one_\${node_id} - (one_\${node_id} - \${a}) * (one_\${node_id} - \${b});
      } else if (\${op} < 5.5) {
        vec4 lo_\${node_id} = 2.0 * \${a} * \${b};
        vec4 hi_\${node_id} = one_\${node_id} - 2.0 * (one_\${node_id} - \${a}) * (one_\${node_id} - \${b});
        r_\${node_id} = lo_\${node_id} + (hi_\${node_id} - lo_\${node_id}) * step(vec4(0.5, 0.5, 0.5, 0.5), \${a});
      } else {
        r_\${node_id} = abs(\${a} - \${b});
      }
      return r_\${node_id};
    `,
  },

  remap: {
    kind: 'remap',
    title: 'Remap',
    category: 'math',
    summary:
      'Linear remap of `value` from the in range to the out range. Swap outMin/outMax to invert; a decreasing in range is ' +
      'clamped by the divisor guard, so use outMin > outMax for inversion instead.',
    inputs: [
      { name: 'value', type: 'float', label: 'Value', default: f(0.5, -4, 4, 0.001) },
      { name: 'inMin', type: 'float', label: 'In Min', default: f(0, -8, 8, 0.01) },
      { name: 'inMax', type: 'float', label: 'In Max', default: f(1, -8, 8, 0.01) },
      { name: 'outMin', type: 'float', label: 'Out Min', default: f(0, -8, 8, 0.01) },
      { name: 'outMax', type: 'float', label: 'Out Max', default: f(1, -8, 8, 0.01) },
    ],
    outputs: [{ name: 'value', type: 'float', label: 'Value' }],
    primaryOut: 'value',
    glsl: `
      return \${outMin} + (\${value} - \${inMin}) / max(\${inMax} - \${inMin}, 0.0001) * (\${outMax} - \${outMin});
    `,
  },

  smoothstep: {
    kind: 'smoothstep',
    title: 'Smoothstep',
    category: 'math',
    summary:
      'Smooth 0..1 ramp as `value` crosses edge0 -> edge1 (the builtin argument order: edges first, value last). ' +
      'This is also how a soft mask is made from a distance.',
    inputs: [
      { name: 'edge0', type: 'float', label: 'Edge 0', default: f(0, -1, 1, 0.001) },
      { name: 'edge1', type: 'float', label: 'Edge 1', default: f(1, -1, 1, 0.001) },
      { name: 'value', type: 'float', label: 'Value', default: f(0.5, -4, 4, 0.001) },
    ],
    outputs: [{ name: 'value', type: 'float', label: 'Value' }],
    primaryOut: 'value',
    glsl: `
      return smoothstep(\${edge0}, \${edge1}, \${value});
    `,
  },

  antialias: {
    kind: 'antialias',
    title: 'Anti-alias',
    category: 'math',
    summary:
      'Threshold `value` at `edge` with a transition width taken from its own screen-space ' +
      'derivative: `smoothstep(-w, w, value - edge)` where `w = fwidth(value) * amount`. This is ' +
      'the difference between a pattern that shimmers and crawls as the camera moves or the ' +
      'resolution drops and one that holds still. Use it wherever a hard threshold would alias: ' +
      'on `sdf.distance` (a far better mask than `sdf.mask`, whose fixed edge width is too sharp ' +
      'far away and too soft up close), on `stripe`/`checker` values, or on any noise after a ' +
      '`remap`. `amount` > 1 trades sharpness for stability; the width is never zero, so a ' +
      'degenerate derivative cannot produce a hard edge.',
    inputs: [
      { name: 'value', type: 'float', label: 'Value', default: f(0.5, -4, 4, 0.001) },
      { name: 'edge', type: 'float', label: 'Edge', default: f(0.5, -4, 4, 0.001) },
      { name: 'amount', type: 'float', label: 'Width', default: f(1, 0, 4, 0.01) },
    ],
    outputs: [{ name: 'value', type: 'float', label: 'Value' }],
    primaryOut: 'value',
    /*
     * The one node that reads a derivative, and the reason the flag exists.
     *
     * `fwidth` is per-pixel quad state, so it is only meaningful in a fragment stage — which is
     * every stage that ever evaluates a node body here (the vertex stages the surface documents
     * emit do nothing but interpolate). It is also only *defined* when the value is computed in
     * uniform control flow: a derivative inside a divergent branch is undefined in GLSL and in
     * HLSL alike. The restricted subset is what makes this safe — it bans data-dependent control
     * flow, so the only branches in a body are on uniform parameters, which the whole quad takes
     * together. That is a real guarantee, not a hope, and it is why this node can exist at all.
     */
    needs: { derivatives: true },
    glsl: `
      float d_\${node_id} = \${value} - \${edge};
      float w_\${node_id} = max(fwidth(d_\${node_id}) * \${amount}, 0.0001);
      return smoothstep(-w_\${node_id}, w_\${node_id}, d_\${node_id});
    `,
  },

  mix: {
    kind: 'mix',
    title: 'Mix',
    category: 'math',
    summary: 'Linear interpolation between two RGBA values by t (clamped to 0..1). a at t=0, b at t=1.',
    inputs: [
      { name: 'a', type: 'vec4', label: 'A', default: color(0, 0, 0, 1) },
      { name: 'b', type: 'vec4', label: 'B', default: color(1, 1, 1, 1) },
      { name: 't', type: 'float', label: 'T', default: f(0.5, 0, 1, 0.001) },
    ],
    outputs: [{ name: 'value', type: 'vec4', label: 'Value' }],
    primaryOut: 'value',
    glsl: `
      return mix_(\${a}, \${b}, clamp(\${t}, 0.0, 1.0));
    `,
  },

  /* ------------------------------------------------------------------ */
  /* Color                                                               */
  /* ------------------------------------------------------------------ */

  shape: {
    kind: 'shape',
    title: 'Composite by Mask',
    category: 'color',
    summary:
      'Paints `color` over `background` where mask is 1 — the standard "cut a shape out" step (wire `sdf.mask` in). ' +
      'The `uv` input is accepted for pipeline symmetry and does not affect the result in v1.',
    inputs: [
      { name: 'uv', type: 'vec2', label: 'UV', default: v2(0.5, 0.5) },
      { name: 'mask', type: 'float', label: 'Mask', default: f(1, 0, 1, 0.001) },
      { name: 'color', type: 'vec4', label: 'Color', default: color(1, 0.45, 0.15, 1) },
      { name: 'background', type: 'vec4', label: 'Background', default: color(0.05, 0.05, 0.08, 1) },
    ],
    outputs: [{ name: 'value', type: 'vec4', label: 'Value' }],
    primaryOut: 'value',
    glsl: `
      return mix_(\${background}, \${color}, clamp(\${mask}, 0.0, 1.0));
    `,
  },

  color_ramp: {
    kind: 'color_ramp',
    title: 'Color Ramp',
    category: 'color',
    summary:
      'Three-stop ramp A -> B -> C driven by t (0..1), blended with two smoothstep stages so the mid stop lands at t = 0.5. ' +
      'Wire a noise/mask into t for palette-mapped looks.',
    inputs: [
      { name: 't', type: 'float', label: 'T', default: f(0.5, 0, 1, 0.001) },
      { name: 'colorA', type: 'vec4', label: 'Stop A', default: color(0.02, 0.02, 0.12, 1) },
      { name: 'colorB', type: 'vec4', label: 'Stop B', default: color(0.85, 0.15, 0.45, 1) },
      { name: 'colorC', type: 'vec4', label: 'Stop C', default: color(1, 0.9, 0.55, 1) },
    ],
    outputs: [{ name: 'color', type: 'vec4', label: 'Color' }],
    primaryOut: 'color',
    glsl: `
      vec4 ab_\${node_id} = mix_(\${colorA}, \${colorB}, smoothstep(0.0, 0.5, \${t}));
      vec4 bc_\${node_id} = mix_(\${colorB}, \${colorC}, smoothstep(0.5, 1.0, \${t}));
      return mix_(ab_\${node_id}, bc_\${node_id}, step(0.5, \${t}));
    `,
  },

  hsv: {
    kind: 'hsv',
    title: 'HSV Adjust',
    category: 'color',
    summary:
      'Round-trips through rgb2hsv/hsv2rgb to shift hue (wraps), scale saturation and scale value, keeping alpha. ' +
      'hue_shift is in turns: 0.5 is the complement, 0.3333 is a third of the wheel.',
    inputs: [
      { name: 'color', type: 'vec4', label: 'Color', default: color(0.25, 0.55, 1, 1) },
      { name: 'hue_shift', type: 'float', label: 'Hue Shift', default: f(0, -1, 1, 0.001) },
      { name: 'saturation', type: 'float', label: 'Saturation', default: f(1, 0, 2, 0.01) },
      { name: 'value', type: 'float', label: 'Value', default: f(1, 0, 2, 0.01) },
    ],
    outputs: [{ name: 'color', type: 'vec4', label: 'Color' }],
    primaryOut: 'color',
    glsl: `
      vec3 c_\${node_id} = rgb2hsv(\${color}.xyz);
      c_\${node_id}.x = fract_(c_\${node_id}.x + \${hue_shift});
      c_\${node_id}.y = clamp(c_\${node_id}.y * \${saturation}, 0.0, 1.0);
      c_\${node_id}.z = clamp(c_\${node_id}.z * \${value}, 0.0, 1.0);
      return vec4(hsv2rgb(c_\${node_id}), \${color}.w);
    `,
  },

  grade: {
    kind: 'grade',
    title: 'Grade',
    category: 'color',
    summary:
      'Photographic grade in a fixed order: exposure (stops) -> contrast about mid-grey 0.5 -> saturation about luma -> ' +
      'gamma -> tint. Alpha is preserved. Use it once at the end of the graph, not per-branch.',
    inputs: [
      { name: 'color', type: 'vec4', label: 'Color', default: color(0.5, 0.5, 0.5, 1) },
      { name: 'exposure', type: 'float', label: 'Exposure', default: f(0, -2, 2, 0.01) },
      { name: 'contrast', type: 'float', label: 'Contrast', default: f(1, 0, 3, 0.01) },
      { name: 'saturation', type: 'float', label: 'Saturation', default: f(1, 0, 3, 0.01) },
      { name: 'gamma', type: 'float', label: 'Gamma', default: f(1, 0.1, 3, 0.01) },
      { name: 'tint', type: 'vec3', label: 'Tint', default: v3(1, 1, 1) },
    ],
    outputs: [{ name: 'color', type: 'vec4', label: 'Color' }],
    primaryOut: 'color',
    glsl: `
      vec3 c_\${node_id} = \${color}.xyz * pow(2.0, \${exposure});
      c_\${node_id} = (c_\${node_id} - vec3(0.5, 0.5, 0.5)) * \${contrast} + vec3(0.5, 0.5, 0.5);
      float l_\${node_id} = luma(c_\${node_id});
      c_\${node_id} = mix_(vec3(l_\${node_id}, l_\${node_id}, l_\${node_id}), c_\${node_id}, \${saturation});
      float ig_\${node_id} = 1.0 / max(\${gamma}, 0.0001);
      c_\${node_id} = pow(max(c_\${node_id}, vec3(0.0, 0.0, 0.0)), vec3(ig_\${node_id}, ig_\${node_id}, ig_\${node_id}));
      c_\${node_id} = c_\${node_id} * \${tint};
      return vec4(c_\${node_id}, \${color}.w);
    `,
  },

  blend: {
    kind: 'blend',
    title: 'Blend',
    category: 'color',
    summary:
      'Paints b over a with opacity: mode 0 normal, 1 add, 2 multiply, 3 screen, 4 overlay. ' +
      'Opacity is the blend amount (0 = a, 1 = full b), applied after the mode, component-wise including alpha.',
    inputs: [
      { name: 'a', type: 'vec4', label: 'A (base)', default: color(0.05, 0.05, 0.08, 1) },
      { name: 'b', type: 'vec4', label: 'B (top)', default: color(1, 0.5, 0.2, 1) },
      { name: 'mode', type: 'float', label: 'Mode', default: f(0, 0, 4, 1) },
      { name: 'opacity', type: 'float', label: 'Opacity', default: f(1, 0, 1, 0.001) },
    ],
    outputs: [{ name: 'value', type: 'vec4', label: 'Value' }],
    primaryOut: 'value',
    glsl: `
      vec4 one_\${node_id} = vec4(1.0, 1.0, 1.0, 1.0);
      float k_\${node_id} = clamp(\${opacity}, 0.0, 1.0);
      vec4 r_\${node_id} = mix_(\${a}, \${b}, k_\${node_id});
      if (\${mode} < 0.5) {
        r_\${node_id} = mix_(\${a}, \${b}, k_\${node_id});
      } else if (\${mode} < 1.5) {
        r_\${node_id} = mix_(\${a}, \${a} + \${b}, k_\${node_id});
      } else if (\${mode} < 2.5) {
        r_\${node_id} = mix_(\${a}, \${a} * \${b}, k_\${node_id});
      } else if (\${mode} < 3.5) {
        r_\${node_id} = mix_(\${a}, one_\${node_id} - (one_\${node_id} - \${a}) * (one_\${node_id} - \${b}), k_\${node_id});
      } else {
        vec4 lo_\${node_id} = 2.0 * \${a} * \${b};
        vec4 hi_\${node_id} = one_\${node_id} - 2.0 * (one_\${node_id} - \${a}) * (one_\${node_id} - \${b});
        r_\${node_id} = mix_(\${a}, lo_\${node_id} + (hi_\${node_id} - lo_\${node_id}) * step(vec4(0.5, 0.5, 0.5, 0.5), \${b}), k_\${node_id});
      }
      return r_\${node_id};
    `,
  },

  /* ------------------------------------------------------------------ */
  /* Output                                                              */
  /* ------------------------------------------------------------------ */

  output: {
    kind: 'output',
    title: 'Output',
    category: 'output',
    summary:
      'Final image. The target returns the emitted local for this node (L_<id>_color) as the fragment colour. ' +
      'Exactly one output node per graph; analyzeGraph takes the first and warns about the rest. ' +
      'It carries one passthrough port named `color` because emitBodies cannot express a portless node (see the header note).',
    inputs: [{ name: 'color', type: 'vec4', label: 'Color', default: color(0, 0, 0, 1) }],
    outputs: [{ name: 'color', type: 'vec4', label: 'Color' }],
    primaryOut: 'color',
    glsl: `
      return \${color};
    `,
  },
};

/* -------------------------------------------------------------------------- */
/* Palette grouping                                                            */
/* -------------------------------------------------------------------------- */

export type CatalogCategory = {
  /** Matches `NodeSchema['category']`. */
  id: NodeSchema['category'];
  /** Palette section heading. */
  title: string;
  /** One line explaining what belongs in this section, shown as the section tooltip. */
  summary: string;
  /** Kinds in this section, in palette order. Partitions NODE_KINDS exactly. */
  kinds: NodeKind[];
};

export const CATEGORIES: CatalogCategory[] = [
  {
    id: 'generator',
    title: 'Generators',
    summary: 'Sources: host inputs (uv, time, mouse, resolution) and constants.',
    kinds: ['uv', 'time', 'mouse', 'resolution', 'const_color', 'const_float', 'gradient', 'texture'],
  },
  {
    id: 'surface',
    title: 'Surface',
    summary:
      'Inputs that only exist on a 3D surface: normal, world position, view and light direction. ' +
      'Reading any of these turns the document into a shading graph and changes every emitted ' +
      'shader to take interpolated vertex outputs. This is where cel/toon shading starts.',
    kinds: ['normal', 'world_pos', 'view_dir', 'light_dir', 'dot'],
  },
  {
    id: 'domain',
    title: 'Domain',
    summary: 'Remap the coordinate space before any pattern is evaluated.',
    kinds: ['transform', 'tile', 'kaleidoscope', 'polar', 'domain_warp'],
  },
  {
    id: 'noise',
    title: 'Pattern & Noise',
    summary: 'Procedural noise, repeating patterns and distance fields.',
    kinds: ['noise', 'fbm', 'voronoi', 'stripe', 'checker', 'sdf', 'raymarch'],
  },
  {
    id: 'math',
    title: 'Math',
    summary: 'Scalar and vector arithmetic, ranges and interpolation.',
    kinds: ['math', 'vec_op', 'remap', 'smoothstep', 'mix', 'antialias', 'vec_math', 'split_vec2', 'split_vec4', 'make_vec2', 'make_vec4', 'custom_glsl'],
  },
  {
    id: 'color',
    title: 'Color',
    summary: 'Compositing, palettes, grading and blending.',
    kinds: ['shape', 'color_ramp', 'hsv', 'grade', 'blend'],
  },
  {
    id: 'output',
    title: 'Output',
    summary: 'The single terminal node of a graph.',
    kinds: ['output'],
  },
];
