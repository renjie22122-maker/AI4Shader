/**
 * Where the pattern lands on the object — and what happens where the mapping runs out
 * ================================================================================
 *
 * The pattern path bakes the document into a texture and maps that texture onto a mesh. The
 * mapping looks like four lines of arithmetic and it is the single easiest thing in the preview
 * to get wrong in a way that produces a *picture* rather than an error — which is exactly what
 * happened, and what this file exists to prevent from happening again.
 *
 * ## The failure this file is built around
 *
 * The 3D pass sampled the pattern in **raw world coordinates** (`texture(uPattern, p.xy * scale)`)
 * into a texture configured **CLAMP_TO_EDGE**. Both halves are individually defensible; together
 * they are a bug, and the bug is silent:
 *
 *   * an object is centred on the origin, so about half of every axis is *negative*;
 *   * a coordinate outside `[0, 1]` under CLAMP does not wrap and does not fail — it returns the
 *     edge texel, stretched over the whole area that maps there;
 *   * so the pattern was only ever visible on the part of the surface whose world coordinates
 *     happened to land inside the unit square. On a cube that is the `(+, +, +)` corner and the
 *     positive half of each of the three faces meeting there; on the ground plane (8 units wide,
 *     mapped 1 unit per tile) it is the two unit-wide bands `x ∈ [0,1]` and `z ∈ [0,1]` — a cross
 *     of real pattern content standing in a field of four stretched corner texels.
 *
 * That is the reported symptom, verbatim: "only a cross has the shader texture", "the cube shows
 * only one corner and half of the adjacent faces". Nothing was mis-compiled and nothing threw; the
 * object was simply sampling a texel that carried no information.
 *
 * ## What is verified here, and what is not
 *
 * There is no GPU in this environment, so this file does **not** verify pixels. It verifies the two
 * halves of the mapping contract, using the *values the code itself produced* rather than a copy of
 * its arithmetic:
 *
 *   * the wrap mode the pattern texture is configured with, read from the recorded GL calls at the
 *     moment of the mesh draw;
 *   * the projection and the offset, parsed out of the material's fragment source, so a change to
 *     the shader changes what is measured rather than silently decoupling from it;
 *   * and then the consequence, computed over every vertex of every shape: **no vertex may be left
 *     to the sampler's edge clamp.** A mapping that clamps is the defect above.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { RecordingGl } from './recording-gl.ts';
import { SurfaceRenderer } from '../dist/preview/surface.js';
import { SHAPE_IDS, buildMesh } from '../dist/preview/mesh.js';

/* -------------------------------------------------------------------------- */
/* Harness                                                                     */
/* -------------------------------------------------------------------------- */

function makeSurface(): { gl: RecordingGl; surface: SurfaceRenderer } {
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

/** The material's fragment source, as handed to the driver. */
function materialSource(gl: RecordingGl): string {
  const src = gl.compiledSources.find((s) => s.includes('sampleTriplanar'));
  assert.ok(src, 'the pattern material was never compiled; the recorder saw no such source');
  return src;
}

test('seam blending defaults on and can be disabled without rebuilding the material', () => {
  const { gl, surface } = makeSurface();
  surface.render(0);
  assert.equal(gl.written.get('uSeamless'), 1);
  const links = gl.calls.filter(c => c.name === 'linkProgram').length;
  surface.setMaterial({ seamless: false, mapping: 'uv', density: 1.25 });
  surface.render(0);
  assert.equal(gl.written.get('uSeamless'), 0);
  assert.equal(gl.written.get('uMapping'), 1);
  assert.equal(gl.written.get('uDensity'), 1.25);
  assert.equal(gl.calls.filter(c => c.name === 'linkProgram').length, links);
  surface.dispose();
});

/**
 * The projection the shader actually performs, read out of its own source.
 *
 * Parsing the source rather than hardcoding `'zy'`, `'xz'`, `'xy'` and `0.5` is the difference
 * between a test that checks the code and a test that checks a second copy of the code: if the
 * shader's axes or offset change, the numbers below change with them.
 */
type Projection = { axis: 'x' | 'y' | 'z'; u: 'x' | 'y' | 'z'; v: 'x' | 'y' | 'z'; offset: number };

function parseTriplanar(source: string): Projection[] {
  const body = /vec3\s+sampleTriplanar\s*\([^)]*\)\s*\{([\s\S]*?)\n\}/.exec(source);
  assert.ok(body, 'the material no longer has a `sampleTriplanar` function to inspect');

  const out: Projection[] = [];
  // The offset is *optional* here, and defaults to zero. That is deliberate: the coverage test
  // below has to be able to measure a mapping that has no offset and report what it costs, rather
  // than refusing to run on exactly the version of the code it exists to catch.
  const re =
    /vec3\s+([xyz])\s*=\s*samplePattern\(\s*p\.([xyzw]{2})\s*\*\s*scale\s*(?:\+\s*([0-9.]+)\s*)?\)/g;
  for (const m of (body[1] as string).matchAll(re)) {
    const pair = m[2] as string;
    out.push({
      axis: m[1] as 'x' | 'y' | 'z',
      u: pair[0] as 'x' | 'y' | 'z',
      v: pair[1] as 'x' | 'y' | 'z',
      offset: m[3] === undefined ? 0 : Number(m[3]),
    });
  }
  assert.equal(
    out.length,
    3,
    'the triplanar projection is not three texture() calls of the form ' +
      '`p.<axes> * scale [+ <offset>]`; this test measures the mapping through exactly that shape ' +
      `of expression. Source was:\n${body[1] as string}`,
  );
  return out;
}

/** Would the sampler have to clamp this coordinate to stay inside the image? */
function clamps(coord: number): boolean {
  return coord < 0 || coord > 1;
}

/* -------------------------------------------------------------------------- */
/* The mapping                                                                 */
/* -------------------------------------------------------------------------- */

test('the material centres and wraps the pattern instead of clamping it', () => {
  const { gl, surface } = makeSurface();
  surface.render(0);

  const draw = gl.meshDraws[0];
  assert.ok(draw, 'the pattern path drew no mesh at all');
  const state = draw.args[4] as { sampler: number | null; wrap?: { s?: number; t?: number } };

  // The two halves of the contract. Either one alone is not enough: a centred projection into a
  // clamped sampler still smears (the sphere reaches ±1), and a repeating sampler with an
  // off-centre projection puts a tile seam through the middle of the model.
  const projections = parseTriplanar(materialSource(gl));
  for (const p of projections) {
    assert.ok(
      p.offset > 0,
      `the ${p.axis}-projection samples the pattern at p.${p.u}${p.v} * scale with no offset, so ` +
        `the pattern's origin sits at the object's origin and three quarters of a unit cube ` +
        `samples outside the image`,
    );
  }
  assert.deepEqual(
    state.wrap,
    { s: gl.REPEAT, t: gl.REPEAT },
    'the pattern texture must wrap: world coordinates leave the unit square on every object, ' +
      'and CLAMP_TO_EDGE answers that with a stretched edge texel rather than the pattern',
  );
});

test('no vertex of any shape is left to the sampler edge clamp', () => {
  const { gl, surface } = makeSurface();
  surface.render(0);
  const projections = parseTriplanar(materialSource(gl));
  const state = gl.meshDraws[0]?.args[4] as { wrap?: { s?: number; t?: number } } | undefined;
  const wraps = state?.wrap?.s === gl.REPEAT && state?.wrap?.t === gl.REPEAT;
  const density = 1; // `DEFAULT_MATERIAL.density`: the framing the workbench opens with.

  let total = 0;
  const clamped = new Map<string, number>();

  for (const shape of SHAPE_IDS) {
    const mesh = buildMesh(shape);
    let hit = 0;
    for (let i = 0; i < mesh.positions.length; i += 3) {
      const pos = {
        x: mesh.positions[i] as number,
        y: mesh.positions[i + 1] as number,
        z: mesh.positions[i + 2] as number,
      };
      const n = {
        x: Math.abs(mesh.normals[i] as number),
        y: Math.abs(mesh.normals[i + 1] as number),
        z: Math.abs(mesh.normals[i + 2] as number),
      };
      // The shader weights the three projections by |n|^4 and normalises, so away from an edge
      // one of them carries the sample on its own. On an exact 45° edge the blend is a mix, and
      // requiring *every* contributing projection to be in range is the stricter reading.
      const weights = { x: n.x ** 4, y: n.y ** 4, z: n.z ** 4 };
      const dominant = (['x', 'y', 'z'] as const).reduce((a, b) => (weights[b] > weights[a] ? b : a));

      const projection = projections.find((p) => p.axis === dominant);
      assert.ok(projection, `the shader has no ${dominant}-projection`);
      const u = pos[projection.u] * density + projection.offset;
      const v = pos[projection.v] * density + projection.offset;
      total += 1;
      if (!wraps && (clamps(u) || clamps(v))) hit += 1;
    }
    if (hit) clamped.set(shape, hit / (mesh.positions.length / 3));
  }

  assert.equal(
    clamped.size,
    0,
    `these shapes hand part of their surface to the sampler's edge clamp, so that part shows one ` +
      `stretched texel instead of the pattern (fraction of vertices affected): ` +
      `${[...clamped].map(([s, f]) => `${s} ${(f * 100).toFixed(0)}%`).join(', ')}`,
  );
});
