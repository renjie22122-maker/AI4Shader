/**
 * AI4Shader — agent layer: prompt construction
 * =============================================
 *
 * The prompts are the product. Everything else in this package exists to make them
 * cheap, correct and reproducible; a prompt that is vague here costs a round trip there.
 *
 * ## The three rules this file is built around
 *
 *   1. **The catalog is generated, never summarised by hand.** `patchUserPrompt` walks
 *      the `NodeSchema` objects it is handed and prints their real port names and types.
 *      A hand-written "here are the node kinds" blob drifts the moment a port is renamed,
 *      and the failure mode is a model confidently emitting `uv.centered` — a port that
 *      was removed. Generating it means the prompt cannot disagree with the validator.
 *
 *   2. **One JSON object, nothing else.** Both system prompts open with that rule and
 *      restate the exact shape, because "explain your reasoning and then give JSON" is
 *      the single most common way a structured-output call turns into a parse error.
 *
 *   3. **The document is projected, never dumped.** `PatchContext` carries `nodeLines`
 *      and `uniformLines` precisely so the whole IR — to say nothing of the generated
 *      shader — never reaches the model. Shipping the document would cost more, be
 *      slower, and give the model more room to invent node ids that do not exist.
 *
 * ## Why the rules are written in the imperative, with the reason attached
 *
 * Every rule below is followed by *why it exists*, because a model that understands the
 * constraint generalises it to the case the rule did not anticipate ("don't invent a port
 * name — the emitter has no code for it" generalises; "don't use `centered`" does not).
 */

import type { NodeKind, NodeSchema } from '@ai4shader/ir/types';

import type { CatalogLike, IntentInput, LlmMessage, PatchContext } from '../types.js';

/**
 * Re-exported so `prompt.ts` callers keep one name for it. The definition lives in
 * `types.ts` — this file previously declared its own copy, which type-checks as an unrelated
 * structural type and would refuse a catalog that arrived from anywhere else.
 */
export type { CatalogLike };

/* -------------------------------------------------------------------------- */
/* Shared fragments                                                            */
/* -------------------------------------------------------------------------- */

/**
 * The patch wire format, shown to the model verbatim.
 *
 * Kept in one place because the *shape* of a patch is the one thing the prompt must
 * never paraphrase: a model that gets `{ op, node, param, value }` approximately right
 * produces something the parser drops, and the round is wasted.
 */
const PATCH_SHAPE = `{
  "summary": "one line describing the whole change; it becomes the undo label",
  "ops": [
    { "op": "add_node",     "node": { "id": "n_scale", "kind": "const_float",
                                      "params": { "value": { "kind": "float", "value": 3.0, "min": 1, "max": 12, "step": 0.1 } } },
                            "reason": "why this node exists" },
    { "op": "connect",      "from": { "node": "n_uv",  "port": "uv" },
                            "to":   { "node": "n_fbm", "port": "uv" },        "reason": "…" },
    { "op": "set_param",    "node": "n_fbm", "param": "scale",
                            "value": { "kind": "float", "value": 6.0 },       "reason": "…" },
    { "op": "expose_param", "node": "n_fbm", "param": "scale", "name": "u_detailScale",
                            "min": 1, "max": 24, "step": 0.5,
                            "label": "纹理密度", "group": "纹理",                "reason": "…" },
    { "op": "disconnect",   "to": { "node": "n_fbm", "port": "uv" },          "reason": "…" },
    { "op": "remove_node",  "node": "n_old",                                  "reason": "…" },
    { "op": "set_body",     "node": "n_custom", "body": "return vec4(\${a}.xyz * \${b}.x, 1.0);",
                                                                              "reason": "…" },
    { "op": "inline_param", "node": "n_fbm", "param": "gain",                 "reason": "…" },
    { "op": "set_render",   "path": "qualityTier", "value": "medium",         "reason": "…" },
    { "op": "set_render",   "path": "transparent", "value": true,             "reason": "preserve output alpha" },
    { "op": "set_render",   "path": "feedback", "value": false,               "reason": "enable only with a defined previous-frame state layout" },
    { "op": "set_spec",     "spec": { "features": ["…"] },                    "reason": "…" }
  ]
}`;

/**
 * The escape hatch, spelled out.
 *
 * A capability the model does not know about is a capability it will never use, and the whole
 * reason `custom_glsl` exists is that a catalog cannot cover every effect a user will name.
 * So it is described here with the same care as the patch shape: when to reach for it, what
 * the ports are, which language the snippet is written in, and the one thing it cannot
 * promise. Left to inference, a model either ignores it (and writes a worse approximation out
 * of the built-in nodes) or writes real GLSL that the rewriter cannot lower.
 */
const CUSTOM_BODY_GUIDE = `ESCAPE HATCH — "custom_glsl", for effects the other kinds cannot express:
  Use it when you can see the effect in your head and no combination of the catalog reaches it
  (a named phenomenon, an unusual operator, a specific formula). Do NOT reach for it first: a
  graph of built-in nodes is inspectable, tunable and repairable, and this is none of those.
  Its ports are fixed: a \`uv\` coordinate (vec2), four vec4 inputs a,b,c,d, and one vec4
  output. Reference them as \${uv} \${a} \${b} \${c} \${d}; end with a single \`return vec4(...);\`.
  A scalar widens into a vec4 port, but a vec2/vec3 does NOT — put it through \`make_vec4\` first.
  The body is written in the SAME restricted GLSL subset as the built-in nodes — not free GLSL:
    aliases   mix_ mod_ fract_ tex2D_ atan2_ inversesqrt_ dFdx_ dFdy_ mat2_mul_ mat3_mul_
    banned    precision/layout/samplers/discard/#define/struct/arrays; use the aliases
    loops     for (int i = 0; i < CONST; i++) only
    vectors   spell every scalar component: vec2(s,s), vec3(s,s,s), vec4(s,s,s,s).
              Never use vec2(s), vec3(s), vec4(s): scalar-splat constructors are rejected.
              Use ascending smoothstep edges; for a reverse fade use 1.0-smoothstep(low,high,x).
    helpers   the shared library is available by name: noise2, fbm2, voronoi2, hsv2rgb, luma,
              sd_circle, sd_box, rot2, scale2 … plus \`fwidth\` for derivative-width edges.
              Anything else is a compile error per target, which is the one thing this node
              cannot check for you.
  **Never write a matrix product with \`*\`.** \`mat2(c, -s, s, c)\` is column-major in GLSL and
  ROW-major in HLSL, so the same values build the transpose there, and HLSL rejects
  \`matrix * vector\` outright. Write \`mat2_mul_(matrix, vector)\`: the rewriter emits the plain
  product where matrices are column-major and \`mul(vector, matrix)\` on HLSL. A body containing
  \`rot2(a) * v\` is refused (RW014) rather than emitted broken.
  The body may also read any host symbol by name — \`iTime\`, \`iMouse\`, \`iResolution\`, \`iFrame\`,
  \`iTimeDelta\`, \`iChannel0\`, and the surface family \`iNormal\` / \`iWorldPos\` / \`iViewDir\` /
  \`iLightDir\` / \`iCameraPos\`. Naming one of the surface five is how a snippet shades a surface
  directly (a \`dot(normalize(iNormal), normalize(iLightDir))\` per-fragment), and it makes the whole
  document a surface document automatically — same consequences as wiring a \`normal\` node.
  Set it with \`set_body\` (it may only target a \`custom_glsl\` node), and remember the vec4
  ports are vec4: use \`.x/.y/.z\` or add a \`split_vec4\` when you need scalars back out.`;


/**
 * The surface group, which is the newest capability and therefore the one a model will not
 * reach for on its own.
 *
 * The failure this prevents is specific and was observed: asked for 三渲二 (cel shading), a
 * model builds a *picture* of a cel-shaded sphere out of screen-space gradients, because that is
 * the only kind of effect it believes the graph can express. The result looks approximately
 * right in the flat preview and is exactly wrong everywhere else — it does not respond to the
 * light, it does not move with the camera, and it collapses the moment the document is put on an
 * actual surface. Naming the four inputs and what makes a document become a surface shader is
 * what turns that into a shading function.
 */
const SURFACE_GUIDE = `SURFACE SHADING — when the effect is about how a surface meets the light:
  Five kinds read the host's surface inputs, and they are ordinary nodes otherwise:
    normal      -> vec4(xyz = unit world normal, w = 0)
    world_pos   -> vec4(xyz = world position,    w = 1)
    view_dir    -> vec4(xyz = unit surface->camera, w = 0)
    light_dir   -> vec4(xyz = unit surface->light,  w = 0)
    dot         -> float   (dot(a.xyz, b.xyz); feed it two surface vectors)
  Reach for them for: 三渲二 / cel shading (a hard terminator), rim or fresnel edges, matcap-like
  shading, height or position gradients in world space, anything that must turn with the light
  or the camera. Do NOT use them to fake a 3D-looking picture out of screen coordinates — that
  is a different, worse thing, and the user asked for the first one.
  Building one is a chain, not a single node: \`normal\` + \`light_dir\` -> \`dot\` -> \`smoothstep\`
  (the terminator) -> \`color_ramp\` (the palette). A rim is \`normal\` + \`view_dir\` -> \`dot\` ->
  \`smoothstep\` -> \`math\` (op=1 subtract from 1, then op=2 multiply by a strength uniform) -> \`mix\`.
  Two consequences worth knowing before you choose this:
    * The document becomes a **surface shader**. Every target then emits a different entry point
      (GLSL needs a host vertex stage, HLSL/WGSL/MSL emit a vertex+fragment pair, Unity maps to
      its own globals, UE gets four extra Custom-node pins), and \`uv\` comes from the mesh rather
      than from the screen. The 3D preview switches to the material view automatically.
    * \`light_dir\` is the *only* light. There is no shadow map and no second light, so an effect
      that needs occlusion or multi-light blending cannot be built — put it in the spec's
      \`outOfScope\` instead of approximating it.`;

/**
 * The two quality problems every procedural shader eventually has, and the two inputs that fix
 * them.
 *
 * Both are cases where the catalog already had the answer and the model had no reason to look for
 * it: `antialias` is one kind among forty-odd, and `tile` is one input among hundreds in the
 * inventory. What is missing from an inventory is *when* — and "when" is what a model needs,
 * because the request that calls for them never names them. Nobody asks for "derivative-width
 * thresholding"; they ask why the pattern shimmers, or why the texture shows a seam in Unity.
 */
const CRAFT_GUIDE = `TWO THINGS EVERY PROCEDURAL SHADER NEEDS SOONER OR LATER:
  1. NO SHIMMER — a hard threshold aliases: the edges crawl and sparkle as the camera moves or the
     resolution drops, because one pixel covers many cells of the pattern. Threshold through
     \`antialias\` instead of \`smoothstep\` with a fixed width: its transition width is taken from
     the value's own screen-space derivative (\`fwidth\`), so the edge is one pixel wide at any
     zoom or resolution. Reach for it on \`sdf.distance\` (small or thin shapes alias worst), on
     \`stripe\`/\`checker\` values, and on any high-frequency noise after a \`remap\`. Its \`amount\`
     widens the transition when the pattern is still unstable; the width is never zero, so it
     cannot degenerate into a hard edge.
  2. NO SEAM — a noise field is NOT periodic: the same uv square next to itself does not match,
     so a texture built from one seams the moment an engine repeats it. Every generator that
     samples noise (\`noise\`, \`fbm\`, \`voronoi\`, \`domain_warp\`, \`raymarch\`) has a \`tile\` input;
     set \`tile = 1\` when the request mentions a seamless / tileable texture, a material that
     will be repeated, or a pattern that fills a surface. It is free visually and costs one
     \`mod\` per cell.
  Also true of every vec4 here: its \`w\` is a TAG, not a component — 0 for a direction (\`v4()\`),
  1 for a point (\`color()\`). \`dot\`, and \`vec_math\`'s normalize/length/distance/reflect/cross,
  read \`.xyz\` and ignore it; the component-wise ops include it. So a direction built with
  \`make_vec4(x, y)\` is a *point* unless you set \`w = 0\`.`;

/** The `ParamValue` shapes, which are the other thing a model reliably gets wrong. */
const PARAM_VALUE_SHAPE = `A literal is ALWAYS an object with a "kind" tag — never a bare number:
  { "kind": "float", "value": 6.0, "min": 1, "max": 24, "step": 0.5 }   // min/max/step optional
  { "kind": "int",   "value": 4,   "min": 1, "max": 8 }
  { "kind": "bool",  "value": true }
  { "kind": "vec2",  "value": [2.0, 2.0] }
  { "kind": "vec3",  "value": [1.0, 1.0, 1.0] }
  { "kind": "vec4",  "value": [0.1, 0.1, 0.1, 1.0] }
  { "kind": "color", "value": [0.9, 0.95, 1.0, 1.0] }
"color" is a vec4 with a colour picker in the UI; use it for any RGBA input.`;

const DYNAMIC_LAYERS_GUIDE = `LAYERED DYNAMIC EFFECTS (rain on glass, condensation, trails, dissolves):
  Preserve EVERY requested behavior, including negative requirements and timing. Do not collapse
  stationary fog, impact droplets, falling beads and residual trails into one animated noise field.
  Build separate named branches/custom_glsl nodes for independent layers, then compose once.
  For rain: fog is a uniform stationary optical veil (constant color/alpha, NOT FBM/cloud patches);
  impact droplets have deterministic random
  position/size per event, a fast birth, a fixed position during life, then fade. Hash a cell/event
  ID, NEVER continuous time for position. Avoid visible grids: jitter, occupancy, varied lifetime.
  Falling beads are sparse and smaller than the screen: y moves monotonically DOWN (uv.y decreases
  in the GLSL quad host), lateral changes are bounded and smooth. A trail stays at past positions,
  is thin, and decays with AGE since passage; it must not travel with the bead or persist forever.
  Derive optical normals from the droplet height field; blend clear centers and soft edges with
  the fog. Keep actual alpha in output.w and set_render transparent=true when transparency is asked.
  Alpha alone cannot refract a background. This host supplies NO scene-color input by default.
  No automatic particle collision/merge/split simulator or multi-pass state layout exists here.
  A deterministic analytic path can approximate motion, but MUST NOT be described as physical
  collision, mass conservation or collision-driven acceleration. Put such unmet requirements in
  spec.outOfScope, in the user's language. Do not silently replace them with random sideways motion.
  render.feedback=true supplies the previous RGBA8 output as iChannel0 in the FLAT WebGL2 preview.
  It is one color/state buffer, not an independent background+height+velocity texture set. Values
  clamp to [0,1]. Define channel meanings and initialize on iFrame==0; use iTimeDelta for decay.
  Do not use alpha simultaneously for a hidden state variable and optical transparency.
  Exported engines need their own feedback host. Prefer analytic finite-lived trails if feedback
  would corrupt transparency. A single output graph may have many layers; do not invent passes.
  Before returning, check every requested layer and lifecycle against the source, and list exact
  approximations in outOfScope. Do not claim visual verification from source inspection alone.`;

/* -------------------------------------------------------------------------- */
/* systemPrompt                                                                */
/* -------------------------------------------------------------------------- */

/**
 * The system prompt for one of the two jobs.
 *
 * The two are separate prompts and not one prompt with a mode flag because the jobs have
 * contradictory instincts: the intent call wants the model to *infer* (fill in what the
 * user did not say), and the patch call wants it to *invent nothing* (every name it
 * writes must already exist). Mixing those instructions in one prompt is how a patch call
 * ends up "helpfully" adding a node kind that does not exist.
 */
export function systemPrompt(kind: 'intent' | 'patch'): string {
  return kind === 'intent' ? intentSystemPrompt() : patchSystemPrompt();
}

function intentSystemPrompt(): string {
  return [
    // Rule 1 is the same in both prompts and is load-bearing: a reply with prose around
    // the JSON is recoverable (the parser scans for the first balanced object) but a reply
    // that *replaces* the JSON with prose is not, and asking for JSON-only is what keeps
    // the recovery path rare.
    'You convert a user request for a 2D procedural shader into a single JSON object.',
    'Reply with ONE JSON object and nothing else: no prose, no markdown fence, no comments.',
    '',
    'The object describes the *intent*, not the graph. Do not name nodes, ports, node kinds,',
    'uniforms or shading languages — a later stage turns this into a graph, and inventing',
    'graph details here only gives it something wrong to copy.',
    '',
    'Shape (all four fields required):',
    '{',
    '  "intent":   "one line saying what the shader is for",',
    '  "category": "short slug, e.g. procedural-surface, stylized-water, abstract-background",',
    '  "tags":     ["3-6 short keywords, in the language of the request"],',
    '  "features": ["one observable requirement per item; preserve ALL requested layers, lifecycle and interactions"],',
    '  "antiFeatures":  ["things the user explicitly does not want"],   // omit if none',
    '  "outOfScope":    ["things you deliberately did not model"]        // omit if none',
    '}',
    '',
    'Rules, and why each one exists:',
    '',
    '1. `features` must be checkable by looking at the image — "swirls rotate over time",',
    '   "the highlight band is cooler than the mid-tones". Not "looks good", not "uses fbm".',
    '   Include temporal requirements that need a sequence of frames to verify. Do not reduce',
    '   a detailed request to 3-6 generic features or silently omit difficult requirements.',
    '',
    '2. Describe what a **shader** can do — colour, pattern, motion, mask, palette, and, when',
    '   the effect is about how a surface meets the light, its *shading*: a hard terminator, a',
    '   rim, a fresnel edge, a height gradient. Ask yourself whether the effect would look the',
    '   same on a flat wall as on a sphere: if not, say so in those terms ("明暗分界随光照移动",',
    '   "轮廓处有亮边"). Do not promise geometry, extra meshes, shadow maps or reflections of a',
    '   scene — those need data the shader is not given and a renderer this tool does not drive.',
    '   WHY: the graph reads the surface normal, world position, view direction and light',
    '   direction when the document needs them, so a shading request is buildable and its',
    '   features are checkable in the 3D preview. A lighting model that needs *more* than those',
    '   four inputs is not, and promising one produces a document nobody can build.',
    '',
    '3. Anything the user asked for that you cannot express in 2D, put in `outOfScope` with',
    '   a plain-language reason. Do not silently drop it and do not pretend to deliver it.',
    '   WHY: the user is told what was not built; a silent drop looks like a bug in the tool.',
    '',
    '4. Keep the user\'s own language for `intent` and `tags` when the request is not in',
    '   English. WHY: these strings are shown back to the user verbatim.',
    '',
    '5. `antiFeatures` is a negative prompt that is checked against the result. Use it when',
    '   the user says "not X", "avoid X", "不要 X".',
    '',
    'Example reply for "液态金属表面，要慢速流动的拉丝高光":',
    '{"intent":"流动的液态金属表面，带细腻的拉丝与高光带","category":"procedural-surface",',
    ' "tags":["金属","流体","噪声","程序化"],',
    ' "features":["表面有连续流动的变形，不是静态图案","存在沿流线方向的细密拉丝纹理",',
    '             "有明暗过渡的高光带，金属质感","整体色调偏冷"],',
    ' "antiFeatures":["blocky 的方格状噪声","硬边的黑白条纹"],',
    ' "outOfScope":["不产生凹凸/法线信息，输出是纯 2D 颜色"]}',
    '',
    'Example reply for "三渲二卡通着色，轮廓带一圈亮边":',
    '{"intent":"卡通着色的表面：明暗之间是硬过渡，轮廓有亮边","category":"surface-shading",',
    ' "tags":["三渲二","卡通","表面着色","边缘光"],',
    ' "features":["明暗之间是可控宽度的硬过渡，不是连续渐变","明暗分界随光照方向移动",',
    '             "轮廓处有一圈亮边","固有色随模型高度变化"],',
    ' "outOfScope":["不做描边（outline）","不含阴影贴图，看不到自身遮挡"]}',
    '   Note what makes this answerable: every feature is a statement about a *surface* meeting',
    '   light, which is exactly what the surface inputs describe.',
  ].join('\n');
}

function patchSystemPrompt(): string {
  return [
    'You edit a shader graph by emitting a single JSON patch object.',
    'Reply with ONE JSON object and nothing else: no prose, no markdown fence, no comments.',
    '',
    '--- 1. The only allowed output shape ---',
    '',
    PATCH_SHAPE,
    '',
    PARAM_VALUE_SHAPE,
    '',
    CUSTOM_BODY_GUIDE,
    '',
    SURFACE_GUIDE,
    '',
    CRAFT_GUIDE,
    '',
    DYNAMIC_LAYERS_GUIDE,
    '',
    '--- 2. Hard rules ---',
    '',
    'R1. Emit ONLY JSON matching the shape above. You are a structured-output endpoint, not',
    '    a chat partner: an explanation outside the JSON makes the reply unparseable, and the',
    '    `reason` field on every op is where your reasoning belongs.',
    '',
    'R2. Use ONLY node kinds and port names that appear in the CATALOG section below. Never',
    '    invent a kind, an input port or an output port, and never guess a name because it',
    '    "should" exist. WHY: the emitter has a hand-written code template per port; a port',
    '    that is not in the catalog has no code, so the document cannot be emitted at all.',
    '    If the catalog lacks what you need, build it from the nodes it does have, or say so',
    '    in `summary` and emit the ops you can justify.',
    '',
    'R3. Node ids match [A-Za-z_][A-Za-z0-9_]* — letters, digits and underscore, not starting',
    '    with a digit. Prefix new ids with `n_`. WHY: ids become GLSL/WGSL/HLSL identifiers,',
    '    so `my-node` or `2nd_fbm` is a shader compile error in all six targets.',
    '',
    'R4. Every `add_node`, `connect`, `set_param`, `expose_param`, `inline_param` and',
    '    `remove_node` op may only name a node that is either in the CURRENT DOCUMENT below',
    '    or added earlier in this same patch. Wires point from an `add_node` to an `add_node`.',
    '    WHY: a wire to a node nobody created is a dangling reference, not a no-op.',
    '',
    'R5. Shader code is allowed ONLY in custom_glsl node.body or a set_body operation,',
    '    using the restricted subset above. Never put expressions in literal parameter values',
    '    or emit a complete shader outside the JSON patch. Use custom_glsl for named effects',
    '    whose required behavior cannot be expressed adequately by the built-in nodes.',
    '',
    'R6. ONE patch per call. Do not emit two JSON objects, do not emit an array of patches,',
    '    and do not ask a follow-up question — the caller has no way to answer one.',
    '',
    'R7. Explain every op in its `reason`: what it does and which part of the intent it',
    '    serves. WHY: `reason` becomes the diff annotation the user reads when reviewing or',
    '    reverting your change, and it is the only place a reviewer learns your intent.',
    '',
    'R8. When repairing, change as little as possible: fix exactly the diagnostics you were',
    '    given and leave working parts of the graph alone. WHY: an unnecessary rewrite of a',
    '    working branch costs a round trip and can turn one broken wire into three.',
    '',
    'R9. Prefer making values tunable over hard-coding them. `expose_param` with a `min`, a',
    '    `max` and a `label` turns a literal into a slider; that is what lets a human or an',
    '    optimizer tune the result without another model call.',
    '',
    'R10. Prefer `connect` over duplicating a node. Two nodes computing the same value cost',
    '     a second evaluation and drift apart the moment one is tuned.',
    '',
    '--- 3. CATALOG: every kind you may use ---',
    '',
    'Format: `kind [category] "title"`, then `inputs:` and `outputs:` as `name: type`.',
    'A port list of `(none)` means the node takes no inputs; it is a generator.',
    'An empty `inputs` list is not an error — generators such as `uv` and `time` have none.',
    '',
  ].join('\n');
}

/* -------------------------------------------------------------------------- */
/* intentUserPrompt                                                            */
/* -------------------------------------------------------------------------- */

/** The user turn for the intent call. */
export function intentUserPrompt(input: IntentInput): string {
  const lines: string[] = ['User request:', input.prompt.trim() || '(empty)'];

  if (input.image) {
    // The image rides as a separate `LlmMessage.image` for providers that declare
    // `vision`. Saying so in the text as well means a text-only provider still knows a
    // reference existed, instead of silently producing a spec for half the request.
    lines.push(
      '',
      'A reference image is attached (see the image part of this message). Describe what is',
      'in it as `features` — colours, structure, motion — not as "matches the image".',
    );
  }

  if (input.basedOn) {
    const doc = input.basedOn;
    lines.push(
      '',
      'This is a REFINEMENT of an existing shader, not a new one. Its current declared intent:',
      `  intent:   ${doc.spec.intent || '(none)'}`,
      `  category: ${doc.spec.category}`,
      `  tags:     ${doc.spec.tags.join(', ') || '(none)'}`,
      `  features: ${doc.spec.features.join(' | ') || '(none)'}`,
      '',
      'Carry those forward unless the request contradicts them, and describe the *result*',
      'after the change — not the change itself. WHY: this spec is compared against the',
      'rendered image, so a spec that describes a diff cannot be checked against anything.',
    );
  }

  if (input.mustUseTargets?.length) {
    lines.push(
      '',
      `The result will be exported to these targets, so nothing may be target-specific: ${input.mustUseTargets.join(', ')}.`,
    );
  }

  return lines.join('\n');
}

/* -------------------------------------------------------------------------- */
/* patchUserPrompt                                                             */
/* -------------------------------------------------------------------------- */

/**
 * The user turn for the patch call: the compact projection of the document, the
 * diagnostics, and the stage-specific instruction.
 *
 * Budget note: this renders `nodeLines`, `uniformLines` and `diagnostics` only. It never
 * touches `ShaderDoc.graph` — see rule 3 in the file header.
 */
export function patchUserPrompt(context: PatchContext, catalog: CatalogLike): string {
  const lines: string[] = [];

  lines.push(`--- STAGE: ${context.stage} ---`);
  lines.push('');

  // The catalog goes first for the same reason a manual's index goes first: the model
  // writes the ops immediately after reading the last thing it saw, and the last thing
  // it sees before the document should be the list of legal names.
  lines.push(renderCatalog(catalog, context));
  lines.push('');

  lines.push('--- CURRENT DOCUMENT ---');
  lines.push(renderDocument(context));
  lines.push('');

  if (context.diagnostics.length) {
    lines.push('--- DIAGNOSTICS TO FIX ---');
    lines.push(renderDiagnostics(context));
    lines.push('');
  }

  lines.push('--- TASK ---');
  lines.push(stageInstruction(context));

  return lines.join('\n');
}

/** Stage-specific instruction. Short, imperative, and different per stage. */
function stageInstruction(context: PatchContext): string {
  switch (context.stage) {
    case 'draft':
      return [
        'The document is empty or nearly empty. Build the graph that realizes the intent:',
        'add the nodes, wire them so the signal reaches the single `output` node, and expose',
        'the handful of values a human would want to tune as named uniforms.',
        '',
        'A graph that emits needs, at minimum: a source of coordinates (`uv`), at least one',
        'node that turns them into a colour (vec4), and an `output` node whose `color` input',
        'is fed by that vec4. Check the types as you wire: a float output cannot feed a vec4',
        'input except through the supported scalar-splat conversion.',
        '',
        'Reply with the patch JSON only.',
      ].join('\n');

    case 'repair':
      return [
        'Repair the document so that every error diagnostic above is gone.',
        '',
        'Change AS LITTLE AS POSSIBLE. Each diagnostic names a node and a port; fix that and',
        'leave the rest of the graph exactly as it is. If a diagnostic is a duplicate of one',
        'you already fixed, fix it once. Do not restructure, do not rename ids that are',
        'already correct, and do not remove a node that is merely unused unless it is the',
        'cause of the diagnostic.',
        '',
        'Use the `fix` field of each diagnostic — it names the operation that resolves it.',
        'Reply with the patch JSON only.',
      ].join('\n');

    case 'refine':
      return [
        'The document already emits cleanly. Adjust it toward the user\'s request above',
        'without breaking it: prefer `set_param` / `expose_param` over structural changes,',
        'and when structure is needed, add to the graph rather than rebuilding it.',
        '',
        'Reply with the patch JSON only.',
      ].join('\n');

    default:
      return 'Reply with the patch JSON only.';
  }
}

/* -------------------------------------------------------------------------- */
/* Rendering: catalog                                                          */
/* -------------------------------------------------------------------------- */

/**
 * The catalog inventory, generated from the `NodeSchema` objects themselves.
 *
 * Two tiers, because a prompt is a budget:
 *
 *   • **The kinds this stage is likely to need** get the full record — title, summary and
 *     the port names with their types. That is the information R2 depends on.
 *   • **Everything else** gets one line: `kind [category] (inputs -> outputs)`. Still the
 *     real port names, ~15 tokens instead of ~70.
 *
 * Nothing is ever *omitted*: a kind the model cannot see is a kind it will re-invent under
 * a different name. The `ALL node kinds` line is the authoritative whitelist and is
 * deliberately redundant with the per-kind lines, because it is the one line a model
 * copies a kind name from.
 */
function renderCatalog(catalog: CatalogLike, context: PatchContext): string {
  const relevant = relevantKinds(context);
  const all = (Object.values(catalog).filter(Boolean) as NodeSchema[])
    .sort((a, b) => (a.kind < b.kind ? -1 : a.kind > b.kind ? 1 : 0));

  const first = all.filter((s) => relevant.has(s.kind));
  const rest = all.filter((s) => !relevant.has(s.kind));

  const out: string[] = [];
  out.push(`ALL node kinds (this list is exhaustive — anything not here does not exist):`);
  out.push(`  ${all.map((s) => s.kind).join(', ') || '(the catalog is empty — nothing can be added)'}`);
  out.push('');

  if (first.length) {
    out.push('Most relevant to this stage, with their exact ports:');
    for (const s of first) out.push(renderSchema(s));
    out.push('');
  }
  if (rest.length) {
    out.push('The rest of the catalog, same format, terser (also allowed):');
    for (const s of rest) out.push(renderSchemaLine(s));
  }
  return out.join('\n');
}

/**
 * Which kinds to surface first.
 *
 * This is a *prompt layout* decision, not a filter: the model sees the whole catalog
 * either way. Ordering by what the stage needs measurably reduces how often a draft
 * reaches for an exotic node before it has an `output`.
 */
function relevantKinds(context: PatchContext): Set<NodeKind> {
  const kinds = new Set<NodeKind>(['uv', 'output', 'color_ramp', 'grade', 'fbm', 'domain_warp', 'transform', 'custom_glsl', 'mix', 'make_vec4', 'split_vec4']);
  for (const line of context.nodeLines) {
    // `id (kind) — note`; the kind is whatever sits inside the first parentheses.
    const m = /\(([^)]+)\)/.exec(line);
    if (m) kinds.add(m[1]!.trim() as NodeKind);
  }
  if (context.stage === 'repair') {
    for (const d of context.diagnostics) {
      // `fix` strings name the node that resolves a mismatch ("Insert a `vec_op` node…").
      const fromFix = /`([a-z_]+)` node/.exec(d.fix ?? '');
      if (fromFix) kinds.add(fromFix[1] as NodeKind);
    }
  }
  return kinds;
}

/** One schema in full: title, summary, ports with types. */
function renderSchema(s: NodeSchema): string {
  const shown = s.summary.replace(/\s+/g, ' ').trim();
  return `• ${s.kind} [${s.category}] "${s.title}"\n    inputs:  ${portLine(s.inputs)}\n    outputs: ${portLine(s.outputs)}\n    ${shown}`;
}

/** One schema in one line. Same port names, no prose. */
function renderSchemaLine(s: NodeSchema): string {
  return `• ${s.kind} [${s.category}] " ${s.title} " | in: ${portLine(s.inputs)} | out: ${portLine(s.outputs)}`;
}

function portLine(ports: readonly { name: string; type: string }[]): string {
  return ports.map((p) => `${p.name}: ${p.type}`).join(', ') || '(none)';
}

/* -------------------------------------------------------------------------- */
/* Rendering: document and diagnostics                                         */
/* -------------------------------------------------------------------------- */

/**
 * The document, from the projection only.
 *
 * `nodeLines` and `uniformLines` are produced by `context.ts` from the live document; if
 * a caller hands over empty arrays the honest answer is "(empty)", not a dump of the IR.
 */
function renderDocument(context: PatchContext): string {
  const out: string[] = [];
  out.push(`spec: ${JSON.stringify(context.doc.spec)}`);
  out.push(`render: ${JSON.stringify(context.doc.render)}`);
  out.push(`targets: ${context.doc.targets.join(', ')}`);

  out.push(`nodes (${context.nodeLines.length}):`);
  out.push(...(context.nodeLines.length ? context.nodeLines.map((l) => `  ${l}`) : ['  (none)']));
  for (const node of context.doc.graph.nodes) {
    if (node.kind === 'custom_glsl' && node.body !== undefined) {
      out.push(`custom_glsl ${node.id} body (current source):`, node.body);
    }
  }

  out.push('');
  out.push(`tunable uniforms (${context.uniformLines.length}):`);
  out.push(...(context.uniformLines.length ? context.uniformLines.map((l) => `  ${l}`) : ['  (none)']));

  if (context.userIntent) {
    out.push('');
    out.push('The user asked for:');
    out.push(`  ${context.userIntent.trim()}`);
  }

  return out.join('\n');
}

/**
 * Diagnostics, verbatim, `fix` included.
 *
 * `fix` is the field the repair loop actually consumes, so it is printed on its own line
 * rather than folded into the message: a model that reads "Use `u_detailScale`" produces a
 * patch that applies, and one that reads only "reserved name" produces another guess.
 */
function renderDiagnostics(context: PatchContext): string {
  return context.diagnostics
    .map((d, i) => {
      const where = d.at
        ? [d.at.node ? `node=${d.at.node}` : '', d.at.port ? `port=${d.at.port}` : ''].filter(Boolean).join(' ')
        : '';
      const head = `  ${i + 1}. [${d.severity}] ${d.code}${where ? ` (${where})` : ''}: ${d.message}`;
      return d.fix ? `${head}\n     fix: ${d.fix}` : head;
    })
    .join('\n');
}

/* -------------------------------------------------------------------------- */
/* Message assembly                                                            */
/* -------------------------------------------------------------------------- */

/**
 * The two-message conversation both providers send.
 *
 * Images are attached as a separate message carrying `LlmMessage.image`; text-only
 * providers never see the image part, and `vision: false` providers are told in the text
 * that a reference existed (see `intentUserPrompt`). The shape is exported so a provider
 * cannot drift from it.
 */
export function intentMessages(input: IntentInput, vision: boolean): LlmMessage[] {
  const messages: LlmMessage[] = [
    { role: 'system', content: systemPrompt('intent') },
    { role: 'user', content: intentUserPrompt(input) },
  ];
  if (input.image && vision) messages.push({ role: 'user', content: 'Reference image:', image: input.image });
  return messages;
}

/** The patch conversation. One user turn; the catalog and the document live in it. */
export function patchMessages(context: PatchContext, catalog: CatalogLike): LlmMessage[] {
  return [
    { role: 'system', content: systemPrompt('patch') },
    { role: 'user', content: patchUserPrompt(context, catalog) },
  ];
}
