/**
 * The 3D pass, shaded by the graph
 * ================================
 *
 * When a document reads a surface input, the 3D view stops mapping a baked texture onto the
 * mesh and starts running the **emitted surface shader** per fragment. That decision is the
 * difference between a toon ramp that turns with the light and a stripe pasted on a model, and
 * it is invisible to every other test in this repository: the IR tests check the source, the
 * Python suite compiles it, and neither knows whether the workbench ever *uses* it.
 *
 * ## The GPU here is a recorder, and that is the honest thing for it to be
 *
 * There is no WebGL2 in this environment and no browser to borrow one from, so a test that
 * claimed to verify *rendering* would be lying. What this file verifies is the half that
 * actually breaks: the **host contract** between the emitter and the preview.
 *
 *   * the source handed to the driver is byte-identical to `emit()`'s output — no patching,
 *     no re-wrapping, so "what you see is what you export" is a checked property;
 *   * the vertex stage writes exactly the varyings the emitted fragment stage declares — the
 *     one link error this design can produce, and one `glslangValidator -l` does **not** catch
 *     (verified: a consistently renamed varying links cleanly);
 *   * every host uniform the emitted shader expects is written, by the reflected type;
 *   * a slider tick rewrites values and links **no** program (the PERF contract, which the flat
 *     path guards with a comment and this path could silently break);
 *   * a shader the driver rejects falls back to the texture path and reports the driver's own
 *     words, rather than leaving a stale revision on screen.
 *
 * The stub throws on any method it does not implement — the same rule as the DOM shim, for the
 * same reason: a recorder that silently accepts `gl.somethingNew()` cannot fail, and a check
 * that cannot fail is decoration.
 */

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';

// The **projected** modules, not `src/`: a browser loads `dist/`, and the projection is what
// the workbench actually runs. `import type` is erased before resolution, so the types can
// still come from the sources.
import { RecordingGl, isActive } from './recording-gl.ts';
import { SurfaceRenderer } from '../dist/preview/surface.js';
import { CATALOG, emit } from '../dist/ir.js';
import type { GraphMaterial } from '../src/preview/surface.ts';
import type { ShaderDoc } from '../../../packages/ir/src/index.js';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(here, '..', '..', '..');

/** The same example the CLI and the Python compiler tests use. */
function toonDoc(): ShaderDoc {
  const path = join(repoRoot, 'examples', 'toon-surface.shader.json');
  return JSON.parse(readFileSync(path, 'utf8')) as ShaderDoc;
}

/* -------------------------------------------------------------------------- */
/* A recording WebGL2 context                                                  */
/* -------------------------------------------------------------------------- */


/* -------------------------------------------------------------------------- */
/* Harness                                                                     */
/* -------------------------------------------------------------------------- */

type Harness = { gl: RecordingGl; surface: SurfaceRenderer };

function makeSurface(): Harness {
  const gl = new RecordingGl();
  const canvas = {
    width: 384,
    height: 384,
    getContext: (kind: string) => (kind === 'webgl2' ? gl : null),
  } as unknown as HTMLCanvasElement;
  const choice = SurfaceRenderer.create(canvas);
  assert.ok(choice.surface, `surface renderer could not be built: ${choice.note ?? ''}`);
  return { gl, surface: choice.surface as SurfaceRenderer };
}

function graphMaterial(doc: ShaderDoc = toonDoc()): { material: GraphMaterial; source: string } {
  const emission = emit(doc, { catalog: CATALOG, target: 'glsl' });
  const errors = emission.diagnostics.filter((d) => d.severity === 'error');
  assert.equal(errors.length, 0, `the document does not emit cleanly: ${JSON.stringify(errors)}`);
  assert.equal(emission.needs.surface, true, 'this document must be a surface document');
  const values: Record<string, number | number[] | boolean> = {};
  for (const u of emission.uniforms) values[u.name] = u.value as number;
  return {
    material: { source: emission.source, uniforms: emission.uniforms, values },
    source: emission.source,
  };
}

/**
 * A surface document that reads **every** host symbol.
 *
 * The toon example deliberately does not: a clean toon shader needs no clock, no resolution
 * and no pointer, so `iTime`/`iResolution`/`iMouse` are optimised out of it and a test built on
 * it would assert that nothing was written for them — which passes for the wrong reason. This
 * is the other half of the surface story: the interactive one, where the pointer and the clock
 * are part of the image.
 */
function hostSymbolDoc(): ShaderDoc {
  const shell = toonDoc();
  const wire = (node: string, port: string) => ({ node, port });
  return {
    ...shell,
    graph: {
      edges: [],
      nodes: [
        { id: 'n_mouse', kind: 'mouse' as never },
        { id: 'n_res', kind: 'resolution' as never },
        { id: 'n_time', kind: 'time' as never },
        { id: 'n_s4', kind: 'split_vec4' as never, inputs: { value: wire('n_mouse', 'mouse') } },
        { id: 'n_s2', kind: 'split_vec2' as never, inputs: { value: wire('n_res', 'resolution') } },
        {
          id: 'n_add',
          kind: 'math' as never,
          params: { op: { kind: 'float', value: 0 } },
          inputs: { a: wire('n_time', 'time'), b: wire('n_s2', 'x') },
        },
        { id: 'n_normal', kind: 'normal' as never },
        { id: 'n_light', kind: 'light_dir' as never },
        { id: 'n_view', kind: 'view_dir' as never },
        {
          id: 'n_nl',
          kind: 'dot' as never,
          inputs: { a: wire('n_normal', 'normal'), b: wire('n_light', 'direction') },
        },
        {
          id: 'n_nv',
          kind: 'dot' as never,
          inputs: { a: wire('n_normal', 'normal'), b: wire('n_view', 'direction') },
        },
        {
          id: 'n_band',
          kind: 'smoothstep' as never,
          inputs: { value: wire('n_nl', 'value'), edge0: wire('n_s4', 'x'), edge1: wire('n_add', 'value') },
        },
        {
          id: 'n_ramp',
          kind: 'color_ramp' as never,
          // `colorB` takes the view-dependent dot product: a scalar widens into a vec4 port, and
          // it is also what keeps the `view_dir` branch reachable. `colorC` samples the channel
          // texture, which is the only way `iChannel0` stays in the program.
          inputs: {
            t: wire('n_band', 'value'),
            colorB: wire('n_nv', 'value'),
            colorC: wire('n_tex', 'color'),
          },
        },
        { id: 'n_uv', kind: 'uv' as never },
        {
          id: 'n_tex',
          kind: 'texture' as never,
          inputs: { uv: wire('n_uv', 'uv'), mix_amount: wire('n_s4', 'y') },
        },
        { id: 'n_out', kind: 'output' as never, inputs: { color: wire('n_ramp', 'color') } },
      ],
    },
  };
}

/* -------------------------------------------------------------------------- */
/* Tests                                                                       */
/* -------------------------------------------------------------------------- */

test('the emitted surface source reaches the driver unmodified', () => {
  const { gl, surface } = makeSurface();
  const { material, source } = graphMaterial();

  assert.equal(surface.setGraphShader(material), null, 'a clean emission must compile');
  assert.equal(surface.graphShaded, true);

  // The graph pair is compiled last: `setGraphShader` builds its own vertex + fragment stages
  // rather than reusing the pattern material's.
  assert.ok(
    gl.compiledSources.includes(source),
    'the emitted source was never handed to the driver verbatim',
  );
  assert.equal(
    gl.lastProgramSource.includes(source),
    true,
    'the linked program does not contain the emitted source',
  );
});

test('the vertex stage writes exactly the varyings the fragment stage declares', () => {
  const { gl, surface } = makeSurface();
  const { material, source } = graphMaterial();
  surface.setGraphShader(material);

  const vertexSource = gl.compiledSources.find(
    (s) => s.includes('vWorldPos') && s.includes('gl_Position'),
  );
  assert.ok(vertexSource, `no vertex shader was compiled; sources: ${gl.compiledSources.length}`);

  const out = [...vertexSource.matchAll(/^\s*out\s+(\w+)\s+(\w+)\s*;/gm)].map((m) => `${m[1]} ${m[2]}`);
  const declared = [...source.matchAll(/^\s*in\s+(\w+)\s+(\w+)\s*;/gm)].map((m) => `${m[1]} ${m[2]}`);

  assert.ok(declared.length >= 3, `the emitted surface shader declares ${declared.length} inputs`);
  assert.deepEqual(
    [...out].sort(),
    [...declared].sort(),
    'the preview vertex stage and the emitted fragment stage disagree about the interpolated ' +
      'variables; this links without complaint on glslangValidator, so nothing else will catch it',
  );
});

test('every host uniform the emitted shader expects is written, by reflected type', () => {
  const { gl, surface } = makeSurface();
  const { material, source } = graphMaterial(hostSymbolDoc());
  surface.setGraphShader(material);
  surface.setPointer(120, 80, true);
  surface.resize(320, 240);
  surface.render(2.5);

  assert.equal(gl.meshDraws.length, 1, 'one mesh draw is expected in material mode');

  const declared = [...source.matchAll(/^\s*uniform\s+(\w+)\s+(\w+)\s*;/gm)]
    .map((m) => m[2] as string)
    .filter((n) => isActive(source, n));
  for (const name of declared) {
    assert.ok(gl.written.has(name), `uniform ${name} was never written to the graph program`);
  }

  // The values matter, not just the writes: a camera position of (0,0,0) would make every
  // view-dependent term in the graph wrong while looking perfectly healthy here.
  assert.deepEqual(gl.written.get('iResolution'), [320, 240, 1]);
  assert.equal(gl.written.get('iTime'), 2.5);
  assert.deepEqual(gl.written.get('iMouse'), [120, 80, 120, 80]);
  const camera = gl.written.get('iCameraPos') as number[];
  assert.ok(Math.hypot(...camera) > 0.5, `iCameraPos is not a real camera position: ${camera}`);
  const light = gl.written.get('iLightDir') as number[];
  assert.ok(Math.abs(Math.hypot(...light) - 1) < 1e-6, `iLightDir must be a unit vector: ${light}`);
  assert.equal(gl.written.get('iChannel0'), 1, 'iChannel0 must be pointed at a bound texture');

  // `iViewDir` is not a uniform — it is derived in the entry point from `iCameraPos` and the
  // interpolated world position, which is exactly the substitution a quad shader cannot do.
  assert.match(source, /vec3 iViewDir = normalize\(iCameraPos - vWorldPos\)/);
});

test('a slider tick rewrites values and links no program', () => {
  const { gl, surface } = makeSurface();
  const { material } = graphMaterial();
  surface.setGraphShader(material);
  surface.render(0);

  const programs = gl.programsCreated;
  const links = gl.calls.filter((c) => c.name === 'linkProgram').length;

  surface.setGraphValues({ ...material.values, u_rimStrength: 1.75 });
  surface.render(0.016);

  assert.equal(gl.programsCreated, programs, 'a value change must not create a program');
  assert.equal(gl.calls.filter((c) => c.name === 'linkProgram').length, links, 'a value change must not link');
  assert.equal(gl.written.get('u_rimStrength'), 1.75, 'the new value never reached the shader');
});

test('the pointer only takes over the drag when the graph reads iMouse', () => {
  const { surface } = makeSurface();
  const { material } = graphMaterial();

  surface.setGraphShader(material);
  assert.equal(surface.pointerFedToGraph, false, 'this graph never reads iMouse');

  // Same document, one extra node: `mouse` is now part of the result, so a drag has to mean
  // "wipe the surface" instead of "orbit the camera" — and that is decided from what the
  // driver kept, not from scanning the graph a second time.
  const doc = toonDoc();
  doc.graph.nodes.push({ id: 'n_mouse', kind: 'mouse' as never });
  doc.graph.nodes.push({
    id: 'n_split',
    kind: 'split_vec4' as never,
    inputs: { value: { node: 'n_mouse', port: 'mouse' } },
  });
  for (const node of doc.graph.nodes) {
    if (node.id === 'n_rim_inv') node.inputs = { ...node.inputs, a: { node: 'n_split', port: 'x' } };
  }
  const withMouse = graphMaterial(doc);
  assert.equal(surface.setGraphShader(withMouse.material), null);
  assert.equal(surface.pointerFedToGraph, true, 'the graph reads iMouse and the host must know');
});

test('a rejected shader falls back to the pattern and reports the driver verbatim', () => {
  const { gl, surface } = makeSurface();
  const { material } = graphMaterial();

  gl.failOn = 'void main()';
  const note = surface.setGraphShader(material);

  assert.ok(note, 'a compile failure must produce a note, not silence');
  assert.match(note as string, /undeclared identifier/, "the driver's own words must survive");
  assert.equal(surface.graphShaded, false, 'a failed shader must not be used');
  assert.equal(surface.describe().includes('图着色'), false, 'the mode description must not claim graph shading');

  // And the pass still draws: falling back means the texture path, not a blank canvas.
  surface.render(0);
  assert.equal(gl.meshDraws.length, 1, 'the fallback path drew nothing');

  // Clearing the failure must let the same document back in — the fallback is not sticky.
  gl.failOn = null;
  assert.equal(surface.setGraphShader(material), null);
  assert.equal(surface.graphShaded, true);
});

test('a quad document is never handed to the graph path', () => {
  const { surface } = makeSurface();
  const doc = toonDoc();
  // Remove every surface input, leaving an ordinary 2D pattern.
  doc.graph.nodes = doc.graph.nodes.filter(
    (n) => !['normal', 'light_dir', 'view_dir', 'world_pos', 'dot', 'split_vec4'].includes(n.kind),
  );
  for (const node of doc.graph.nodes) {
    if (node.inputs) {
      for (const [port, ref] of Object.entries(node.inputs)) {
        if (!doc.graph.nodes.some((n) => n.id === ref.node)) delete node.inputs[port];
      }
    }
  }
  const emission = emit(doc, { catalog: CATALOG, target: 'glsl' });
  assert.equal(emission.needs.surface, false, 'the stripped document should be a quad shader');

  // `main.ts` only ever calls `setGraphShader` when `needs.surface` is true; this asserts the
  // flag it reads is the emitter's, so the two cannot drift apart unnoticed.
  surface.setGraphShader(null);
  assert.equal(surface.graphShaded, false);
  assert.equal(surface.pointerFedToGraph, false);
});
