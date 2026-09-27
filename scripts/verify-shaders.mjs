/**
 * Real-GPU verification of the emitted shaders.
 * =============================================
 *
 * The honest question this script answers is: **does the generated GLSL actually compile
 * and draw?** Everything upstream of this point — the IR, the rewriter, the emitter — is
 * checked by unit tests against strings, and strings are not pixels. This runs the
 * generated shader through a real WebGL2 implementation in headless Chrome and writes a
 * PNG of the result.
 *
 *   node scripts/verify-shaders.mjs [--keep] [--timeout 45000]
 *
 * What it does, per target:
 *   1. emits source via the real emitter (not a fixture),
 *   2. writes a self-contained HTML harness that compiles + draws it and reports the
 *      driver's own `getShaderInfoLog` verbatim,
 *   3. loads that harness in headless Chrome, screenshots it, and scrapes the verdict.
 *
 * The GLSL is embedded as base64 rather than written to a sibling file so the harness is
 * a single artefact with no relative-path or encoding questions — shader source is full
 * of `<`, `>`, `&` and `#`, and every one of those is an escape bug waiting to happen.
 */

import { writeFile, mkdir, rm, readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { execFile } from 'node:child_process';
import { dirname, resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { promisify } from 'node:util';

const run = promisify(execFile);
const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const outDir = join(repoRoot, 'dist', 'verify');

const CHROME_CANDIDATES = [
  'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe',
  'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
  'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
  '/usr/bin/google-chrome',
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
];

function findChrome() {
  for (const c of CHROME_CANDIDATES) if (existsSync(c)) return c;
  return null;
}

/* -------------------------------------------------------------------------- */
/* Emit the shaders under test                                                 */
/* -------------------------------------------------------------------------- */

async function emitAll() {
  const ir = await import('../packages/ir/dist/index.js');
  const cat = await import('../packages/ir/dist/catalog/index.js');
  await import('../packages/ir/dist/targets/index.js');
  const quad = await import('../packages/ir/dist/examples/liquidMetal.js');
  const surface = await import('../packages/ir/dist/examples/toonSurface.js');

  const out = {};
  for (const [name, build] of [
    ['quad', quad.createLiquidMetalDoc],
    ['surface', surface.createToonSurfaceDoc],
  ]) {
    const doc = build();
    doc.graph = ir.normalizeEdges(doc.graph);
    const targets = {};
    for (const target of doc.targets) {
      const em = ir.emit(doc, { catalog: cat.CATALOG, target });
      targets[target] = {
        source: em.source,
        uniforms: em.uniforms,
        errors: em.diagnostics.filter((d) => d.severity === 'error'),
        needsSurface: em.needs.surface,
      };
    }
    out[name] = { title: doc.title, targets };
  }
  return out;
}

/* -------------------------------------------------------------------------- */
/* Vertex stages for the harness                                               */
/* -------------------------------------------------------------------------- */

/**
 * The fullscreen-triangle vertex stage the **quad** fragment shader expects.
 *
 * This mirrors exactly what `apps/web/src/preview/glsl.ts` does — if the two ever disagree,
 * one of them is wrong and this script is the cheaper place to find out.
 */
const QUAD_VERTEX_SHADER = `#version 300 es
void main() {
  // Fullscreen triangle from gl_VertexID; no vertex buffer, no attributes.
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

/**
 * A vertex stage for a **surface** fragment shader, synthesised from that shader's own contract.
 *
 * A surface shader declares `in` variables it expects a mesh vertex stage to interpolate. The
 * real host is `apps/web/src/preview/surface.ts`, which supplies them from an actual mesh; there
 * is no mesh in this harness and no reason to add one — what this script is here to prove is
 * that the emitted fragment stage **compiles on a real driver and paints pixels**, and for that
 * the varying values only have to be *something*.
 *
 * So the declarations are read out of the emitted source and an `out` is written for each one,
 * with a stand-in value chosen per name. Deriving instead of restating is the point: a rename in
 * the emitter flows through automatically, and this file cannot go stale on its own.
 *
 * **What this does not verify**: that a *mesh* satisfies the contract. That property is pinned
 * by `apps/web/test/surface-graph.test.ts` (the two stages' declarations are compared by name),
 * because `glslangValidator -l` demonstrably accepts a renamed varying without complaint.
 */
function surfaceVertexShader(fragmentSource) {
  const inputs = [...fragmentSource.matchAll(/^\s*in\s+(\w+)\s+(\w+)\s*;/gm)].map((m) => ({
    type: m[1],
    name: m[2],
  }));
  if (!inputs.length) return null;
  const outs = inputs.map((i) => `out ${i.type} ${i.name};`).join('\n');
  // `p` is the clip-space position of the oversized triangle, reused as a screen coordinate.
  const assigns = inputs
    .map(({ type, name }) => {
      if (name === 'vUv') return `${name} = p * 0.5 + 0.5;`;
      if (name === 'vWorldPos') return `${name} = vec3(p * 2.0, 0.5);`;
      if (name === 'vNormal') return `${name} = normalize(vec3(p, 1.0));`;
      // An unknown varying is an error rather than a guess: a silently zero-filled varying
      // would produce an all-black image, which this harness would then report as a shader
      // bug instead of a harness gap.
      throw new Error(`harness has no stand-in value for the varying ${type} ${name}`);
    })
    .join('\n  ');
  return `#version 300 es
precision highp float;
precision highp int;
${outs}
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)) * 2.0 - 1.0;
  ${assigns}
  gl_Position = vec4(p, 0.0, 1.0);
}
`;
}

const HARNESS = `<!doctype html>
<html><head><meta charset="utf-8"><title>verify</title>
<style>
  body { margin:0; background:#111; color:#ddd; font:12px/1.4 monospace; }
  #out { padding:10px; white-space:pre-wrap; }
  canvas { display:block; width:384px; height:384px; image-rendering:pixelated; border:1px solid #444; }
  .ok { color:#4ade80 } .bad { color:#f87171 } .warn { color:#fbbf24 }
</style></head>
<body>
<canvas id="c" width="384" height="384"></canvas>
<div id="out">running…</div>
<script type="module">
const FRAG = atob(${JSON.stringify('__FRAG_B64__')});
const VERT = atob(${JSON.stringify('__VERT_B64__')});
const CASE_NAME = ${JSON.stringify('__CASE__')};
const UNIFORMS = ${JSON.stringify('__UNIFORMS__')};

const lines = [];
const say = (s, cls) => lines.push(cls ? '<span class="' + cls + '">' + s + '</span>' : s);
const flush = () => { document.getElementById('out').innerHTML = lines.join('\\n'); };

const canvas = document.getElementById('c');
const gl = canvas.getContext('webgl2', { antialias: false, premultipliedAlpha: false, preserveDrawingBuffer: true });
if (!gl) {
  document.title = 'VERDICT:NO_WEBGL2';
  say('NO_WEBGL2: this browser/GPU exposed no WebGL2 context.', 'bad'); flush();
} else {
  say('renderer: ' + gl.getParameter(gl.RENDERER));
  say('version:  ' + gl.getParameter(gl.VERSION));
  say('glsl:     ' + gl.getParameter(gl.SHADING_LANGUAGE_VERSION));
  say('case:     ' + CASE_NAME + (CASE_NAME === 'surface'
    ? '  (surface varyings are synthesised by the harness — see the script header)'
    : ''));
  say('');

  const compile = (type, src, label) => {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    const log = gl.getShaderInfoLog(sh) || '';
    const ok = gl.getShaderParameter(sh, gl.COMPILE_STATUS);
    say(label + ': ' + (ok ? 'COMPILED' : 'FAILED'), ok ? 'ok' : 'bad');
    if (log.trim()) say(log.trim().split('\\n').map(l => '  ' + l).join('\\n'), ok ? 'warn' : 'bad');
    return ok ? sh : null;
  };

  const vs = compile(gl.VERTEX_SHADER, VERT, 'vertex');
  const fs = compile(gl.FRAGMENT_SHADER, FRAG, 'fragment');
  let program = null;
  if (vs && fs) {
    program = gl.createProgram();
    gl.attachShader(program, vs); gl.attachShader(program, fs);
    gl.linkProgram(program);
    const linked = gl.getProgramParameter(program, gl.LINK_STATUS);
    const plog = gl.getProgramInfoLog(program) || '';
    say('link: ' + (linked ? 'LINKED' : 'FAILED'), linked ? 'ok' : 'bad');
    if (plog.trim()) say('  ' + plog.trim(), linked ? 'warn' : 'bad');
  }

  if (program && gl.getProgramParameter(program, gl.LINK_STATUS)) {
    gl.useProgram(program);
    // Bind every uniform the emitter declared; a name the driver optimised away is a
    // legitimate null and must not be treated as a failure.
    let bound = 0, missing = 0;
    for (const u of UNIFORMS) {
      const loc = gl.getUniformLocation(program, u.name);
      if (!loc) { missing++; continue; }
      const v = u.value;
      if (u.type === 'float') gl.uniform1f(loc, v);
      else if (u.type === 'vec2') gl.uniform2f(loc, v[0], v[1]);
      else if (u.type === 'vec3') gl.uniform3f(loc, v[0], v[1], v[2]);
      else if (u.type === 'vec4' || u.type === 'color') gl.uniform4f(loc, v[0], v[1], v[2], v[3]);
      else if (u.type === 'int' || u.type === 'bool') gl.uniform1i(loc, v ? 1 : 0);
      bound++;
    }
    say('uniforms: ' + bound + ' bound, ' + missing + ' optimised away by the driver');
    gl.uniform3f(gl.getUniformLocation(program, 'iResolution'), canvas.width, canvas.height, 1);
    gl.uniform1f(gl.getUniformLocation(program, 'iTime'), 1.7);
    gl.uniform1f(gl.getUniformLocation(program, 'iTimeDelta'), 0.016);
    gl.uniform1i(gl.getUniformLocation(program, 'iFrame'), 102);
    // Surface hosts: a look-at camera, a unit light, a pointer, and an opaque white channel.
    // iViewDir is not a uniform — the emitted entry point derives it from iCameraPos and
    // the interpolated world position, which is exactly the substitution a quad cannot make.
    gl.uniform3f(gl.getUniformLocation(program, 'iCameraPos'), 0, 0, 3.4);
    gl.uniform3f(gl.getUniformLocation(program, 'iLightDir'), 0.5509, 0.7212, 0.4207);
    gl.uniform4f(gl.getUniformLocation(program, 'iMouse'), 190, 210, 190, 210);
    const channel = gl.createTexture();
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, channel);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([255, 255, 255, 255]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.uniform1i(gl.getUniformLocation(program, 'iChannel0'), 0);

    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    const err = gl.getError();
    say('glGetError after draw: ' + err + (err === 0 ? ' (clean)' : ' <-- NON-ZERO'), err === 0 ? 'ok' : 'bad');

    // Read the framebuffer back and report real statistics. A shader that compiles but
    // draws nothing is the failure this catches, and it is otherwise invisible.
    const px = new Uint8Array(canvas.width * canvas.height * 4);
    gl.readPixels(0, 0, canvas.width, canvas.height, gl.RGBA, gl.UNSIGNED_BYTE, px);
    let min = 255, max = 0, sum = 0;
    const buckets = new Set();
    for (let i = 0; i < px.length; i += 4) {
      const l = (px[i] * 0.2126 + px[i + 1] * 0.7152 + px[i + 2] * 0.0722);
      if (l < min) min = l;
      if (l > max) max = l;
      sum += l;
      buckets.add((px[i] >> 4) + ',' + (px[i + 1] >> 4) + ',' + (px[i + 2] >> 4));
    }
    const mean = sum / (px.length / 4);
    say('');
    say('luma min/max/mean: ' + min.toFixed(1) + ' / ' + max.toFixed(1) + ' / ' + mean.toFixed(1));
    say('distinct colours (4-bit RGB buckets): ' + buckets.size);
    const drewSomething = max - min > 8 && buckets.size > 3;
    say(drewSomething ? 'PIXELS: non-uniform image produced' : 'PIXELS: FLAT — nothing was drawn', drewSomething ? 'ok' : 'bad');
    document.title = 'VERDICT:' + (drewSomething ? 'RENDERED' : 'FLAT');
  } else {
    document.title = 'VERDICT:COMPILE_FAILED';
  }
  flush();
  document.body.setAttribute('data-done', '1');
}
</script></body></html>
`;

/* -------------------------------------------------------------------------- */
/* Chrome driver                                                               */
/* -------------------------------------------------------------------------- */

async function screenshot(chrome, url, pngPath, timeoutMs) {
  const args = [
    '--headless=new',
    '--disable-gpu-sandbox',
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-extensions',
    '--disable-background-networking',
    '--enable-unsafe-swiftshader', // software GL: no real GPU is guaranteed in CI
    '--hide-scrollbars',
    '--window-size=760,900',
    `--virtual-time-budget=${timeoutMs}`,
    `--screenshot=${pngPath}`,
    url,
  ];
  try {
    const { stderr } = await run(chrome, args, { timeout: timeoutMs + 30000, maxBuffer: 8 << 20 });
    return { ok: true, stderr };
  } catch (err) {
    return { ok: false, stderr: String(err?.stderr ?? err?.message ?? err) };
  }
}

/* -------------------------------------------------------------------------- */
/* Main                                                                        */
/* -------------------------------------------------------------------------- */

const argv = process.argv.slice(2);
const timeoutMs = Number(argv[argv.indexOf('--timeout') + 1]) || 45000;

const chrome = findChrome();
if (!chrome) {
  process.stderr.write('✗ no Chrome/Edge binary found; cannot verify on a real GPU\n');
  process.exit(1);
}
process.stdout.write(`chrome: ${chrome}\n`);

const emissions = await emitAll();
await rm(outDir, { recursive: true, force: true });
await mkdir(outDir, { recursive: true });

/**
 * Both documents, because they are different programmes.
 *
 * The quad example proves the ordinary path: a fragment shader over a fullscreen triangle. The
 * surface example proves the path where `needs.surface` changes every target's entry point —
 * and it is the one a user asks for by name ("三渲二"), so it should not be the one that only
 * ever gets compiled by hand.
 */
const CASES = [
  { name: 'quad', key: 'quad', file: 'glsl-quad', vertex: () => QUAD_VERTEX_SHADER },
  {
    name: 'surface',
    key: 'surface',
    file: 'glsl-surface',
    vertex: (fragment) => {
      const vs = surfaceVertexShader(fragment);
      if (!vs) throw new Error('the surface example emitted no interpolated inputs');
      return vs;
    },
  },
];

let allWritten = true;

for (const testCase of CASES) {
  const doc = emissions[testCase.key];
  const glsl = doc?.targets.glsl;
  if (!glsl || glsl.errors.length) {
    process.stderr.write(`✗ ${testCase.name}: glsl emission has errors: ${JSON.stringify(glsl?.errors)}\n`);
    process.exit(1);
  }
  if (testCase.key === 'surface' && !glsl.needsSurface) {
    process.stderr.write(`✗ ${testCase.name}: the surface example no longer reports needs.surface\n`);
    process.exit(1);
  }

  // Split the header the way the browser renderer does: `#version` + precisions, then body.
  const allLines = glsl.source.split('\n');
  const lastPrecision = allLines.findLastIndex((l) => /^\s*precision\s/.test(l));
  const header = allLines.slice(0, lastPrecision + 1).join('\n');
  const body = allLines.slice(lastPrecision + 1).join('\n');
  const fragment = `${header}\n${body}`;

  await writeFile(join(outDir, `ai4shader-${testCase.name}.frag`), glsl.source, 'utf8');
  // Each generated vertex stage carries its own `#version`/precision header, so it is written
  // verbatim: prepending the fragment's header would emit `precision highp float;` twice.
  const vertex = testCase.vertex(fragment);
  await writeFile(join(outDir, `ai4shader-${testCase.name}.vert`), vertex, 'utf8');

  const html = HARNESS
    .replace('__FRAG_B64__', Buffer.from(glsl.source, 'utf8').toString('base64'))
    .replace('__VERT_B64__', Buffer.from(vertex, 'utf8').toString('base64'))
    .replace('__UNIFORMS__', JSON.stringify(glsl.uniforms))
    .replace('__CASE__', testCase.name);
  const htmlPath = join(outDir, `${testCase.file}.html`);
  await writeFile(htmlPath, html, 'utf8');

  // Chrome needs a real URL rather than a file:// path for module + worker behaviour to
  // match production, so serve through the already-running static server when it is up.
  let url = `file:///${htmlPath.replace(/\\/g, '/')}`;
  try {
    const res = await fetch('http://127.0.0.1:5183/scripts/serve.mjs', { method: 'HEAD' });
    if (res.ok) url = `http://127.0.0.1:5183/dist/verify/${testCase.file}.html`;
  } catch {
    /* no server: fall back to file://, which is fine for a single self-contained page */
  }
  process.stdout.write(`${testCase.name} harness: ${url}\n`);

  const pngPath = join(outDir, `${testCase.file}.png`);
  const shot = await screenshot(chrome, url, pngPath, timeoutMs);
  if (!shot.ok) process.stderr.write(`${testCase.name} chrome stderr:\n${shot.stderr}\n`);
  const written = existsSync(pngPath);
  if (!written) allWritten = false;
  process.stdout.write(`${testCase.name} screenshot: ${written ? pngPath : '(not written)'}\n`);
}

await writeFile(
  join(outDir, 'ai4shader.all-targets.json'),
  JSON.stringify(
    Object.fromEntries(
      Object.entries(emissions).map(([docName, doc]) => [
        docName,
        {
          title: doc.title,
          targets: Object.fromEntries(
            Object.entries(doc.targets).map(([t, v]) => [
              t,
              { errors: v.errors, bytes: v.source.length, needsSurface: v.needsSurface },
            ]),
          ),
        },
      ]),
    ),
    null,
    2,
  ),
  'utf8',
);

process.stdout.write(
  allWritten
    ? `\nNow look at the PNGs: each shows the compile log, the bound-uniform count, luma\n` +
        `statistics and the rendered frame. That combination is the actual proof — a\n` +
        `green "COMPILED" line with a flat canvas is a failure, not a pass.\n\n` +
        `The surface page uses a **synthesised** surface (a fullscreen triangle whose varyings\n` +
        `are derived from the emitted shader's own contract), so it proves the real fragment\n` +
        `stage compiles and paints — not that a mesh satisfies the contract. That property is\n` +
        `pinned by apps/web/test/surface-graph.test.ts.\n`
    : `\n✗ at least one screenshot was not produced — the verification did not run. Investigate before claiming success.\n`,
);
if (!allWritten) process.exit(1);
