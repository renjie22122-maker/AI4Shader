/**
 * Mesh invariants
 * ===============
 *
 * These generators exist so that the 3D preview has something to draw without a binary asset
 * nobody can review. That only pays off if the arithmetic is actually right, and "actually
 * right" here has one failure mode that is both likely and invisible: **winding**.
 *
 * A mesh wound the wrong way renders identically to a correct one on a wireframe, renders
 * *black* under lighting with back-face culling, and produces that result in a browser this
 * environment cannot start. Getting it wrong is a one-character edit and finding it by hand
 * costs a debugging session in a tool nobody can see. So the first test below is the one that
 * matters: for every triangle, the geometric normal implied by the winding must agree with
 * the vertex normals the surface was built from. A flipped quad fails it immediately, and it
 * fails for any shape, convex or not, without assuming anything about the topology.
 *
 * The rest are the ordinary bounds checks — indices in range, unit normals, finite floats,
 * UVs inside [0,1] — which catch the other half: a generator that emits a lattice but
 * forgets to index it, or a UV that leaves the texture's domain and silently wraps.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  SHAPE_IDS,
  SHAPE_LABELS,
  buildMesh,
  cube,
  ground,
  sphere,
  torus,
  torusKnot,
  type MeshData,
  type ShapeId,
} from '../src/preview/mesh.ts';

function positionOf(m: MeshData, i: number): [number, number, number] {
  return [m.positions[i * 3] ?? 0, m.positions[i * 3 + 1] ?? 0, m.positions[i * 3 + 2] ?? 0];
}

function normalOf(m: MeshData, i: number): [number, number, number] {
  return [m.normals[i * 3] ?? 0, m.normals[i * 3 + 1] ?? 0, m.normals[i * 3 + 2] ?? 0];
}

function sub(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function cross(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function dot(a: [number, number, number], b: [number, number, number]): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function length(a: [number, number, number]): number {
  return Math.hypot(a[0], a[1], a[2]);
}

const CASES: Array<[ShapeId, MeshData]> = [
  ['sphere', sphere()],
  ['torus', torus()],
  ['knot', torusKnot()],
  ['cube', cube()],
  ['ground', ground()],
];

test('every shape is indexed, non-empty, and covers a sane triangle count', () => {
  for (const [id, m] of CASES) {
    assert.ok(m.positions.length > 0, `${id}: no vertices`);
    assert.equal(m.indices.length % 3, 0, `${id}: index count is not a multiple of 3`);
    assert.ok(m.indices.length >= 3, `${id}: no triangles`);
    assert.equal(m.positions.length, m.normals.length, `${id}: normals are not per-vertex`);
    assert.equal(m.positions.length / 3, m.uvs.length / 2, `${id}: uvs are not per-vertex`);
  }
});

test('winding agrees with the vertex normals — the inside-out check', () => {
  for (const [id, m] of CASES) {
    let agree = 0;
    let disagree = 0;
    let degenerate = 0;

    for (let t = 0; t < m.indices.length; t += 3) {
      const ia = m.indices[t] ?? 0;
      const ib = m.indices[t + 1] ?? 0;
      const ic = m.indices[t + 2] ?? 0;
      const a = positionOf(m, ia);
      const b = positionOf(m, ib);
      const c = positionOf(m, ic);

      // The normal the *winding* implies, by the right-hand rule.
      const face = cross(sub(b, a), sub(c, a));
      const area = length(face);
      if (area < 1e-12) {
        degenerate++;
        continue;
      }

      // Average the three vertex normals the surface was built with.
      const na = normalOf(m, ia);
      const nb = normalOf(m, ib);
      const nc = normalOf(m, ic);
      const vn: [number, number, number] = [
        (na[0] + nb[0] + nc[0]) / 3,
        (na[1] + nb[1] + nc[1]) / 3,
        (na[2] + nb[2] + nc[2]) / 3,
      ];

      if (dot(face, vn) > 0) agree++;
      else disagree++;
    }

    assert.ok(agree > 0, `${id}: no triangle had a usable normal`);
    assert.equal(
      disagree,
      0,
      `${id}: ${disagree} of ${agree + disagree} triangles are wound the wrong way — ` +
        `they would render black under back-face culling. Swap the quad order in mesh.ts.`,
    );
    // Degenerate triangles are allowed only where the shape genuinely has them (a sphere's
    // poles are skipped, so a well-built mesh should have none at all).
    assert.equal(degenerate, 0, `${id}: ${degenerate} zero-area triangles`);
  }
});

test('indices stay inside the vertex arrays', () => {
  for (const [id, m] of CASES) {
    const count = m.positions.length / 3;
    for (const i of m.indices) {
      assert.ok(i >= 0 && i < count, `${id}: index ${i} is out of range (vertex count ${count})`);
    }
  }
});

test('positions and normals are finite, and normals are unit length', () => {
  for (const [id, m] of CASES) {
    for (let i = 0; i < m.positions.length; i++) {
      assert.ok(Number.isFinite(m.positions[i]), `${id}: position[${i}] is not finite`);
    }
    for (let v = 0; v < m.normals.length / 3; v++) {
      const n = normalOf(m, v);
      const len = length(n);
      assert.ok(
        Math.abs(len - 1) < 1e-5,
        `${id}: normal of vertex ${v} has length ${len}, expected 1 — ` +
          `a non-unit normal makes every lighting term wrong by that factor`,
      );
    }
  }
});

test('UVs stay inside [0, 1]', () => {
  for (const [id, m] of CASES) {
    for (let i = 0; i < m.uvs.length; i++) {
      const value = m.uvs[i] ?? 0;
      assert.ok(
        value >= 0 && value <= 1,
        `${id}: uv[${i}] = ${value} is outside [0, 1]; the material would sample outside the pattern`,
      );
    }
  }
});

test('every shape is centred near the origin and reports a usable radius', () => {
  for (const [id, m] of CASES) {
    assert.ok(m.radius > 0, `${id}: radius is ${m.radius}`);
    let maxAbsY = 0;
    for (let v = 0; v < m.positions.length / 3; v++) {
      const p = positionOf(m, v);
      maxAbsY = Math.max(maxAbsY, Math.abs(p[1]));
    }
    // The camera frames the mesh with `radius`, so a shape that sits far off-origin would
    // be framed badly even though every other invariant held.
    assert.ok(maxAbsY <= m.radius * 1.001, `${id}: geometry extends beyond its own radius`);
  }
  // The ground is deliberately flat at y = 0: it is a floor, not an object.
  const g = ground();
  for (let v = 0; v < g.positions.length / 3; v++) {
    assert.equal(positionOf(g, v)[1], 0, 'the ground plane must lie exactly on y = 0');
  }
});

test('determinism: the same shape is byte-identical every time', () => {
  // The preview rebuilds a mesh on every mode change; a generator that varied run to run
  // would make the mesh cache useless and the screenshots unreproducible.
  for (const id of SHAPE_IDS) {
    const a = buildMesh(id);
    const b = buildMesh(id);
    assert.deepEqual(Array.from(a.positions), Array.from(b.positions), `${id}: positions differ`);
    assert.deepEqual(Array.from(a.indices), Array.from(b.indices), `${id}: indices differ`);
  }
});

test('buildMesh and the labels cover exactly SHAPE_IDS', () => {
  assert.deepEqual([...SHAPE_IDS].sort(), ['cube', 'ground', 'knot', 'sphere', 'torus']);
  for (const id of SHAPE_IDS) {
    assert.equal(typeof SHAPE_LABELS[id], 'string');
    assert.ok(SHAPE_LABELS[id].length > 0, `${id} has no label`);
    assert.equal(buildMesh(id).indices.length > 0, true, `${id} built nothing`);
  }
});
