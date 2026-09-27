/**
 * Procedural meshes
 * =================
 *
 * Every shape the 3D preview can wear, generated from a formula rather than loaded from a
 * file. That is not a stylistic preference: this project has no bundler, no `npm install`,
 * and no network, so a `.glb` would be a binary asset nobody could verify, review or diff.
 * A sphere is forty lines of arithmetic that a test can check for unit normals and in-range
 * indices, and it is the same forty lines on every machine.
 *
 * ## What the shapes are for
 *
 * A procedural shader is a *field*, and a flat quad is the worst possible way to judge one:
 * it hides stretching, seams and scale. The set is chosen so that each shape stresses a
 * different failure:
 *
 *   sphere  — a smooth closed surface. Any seam or pole artefact is obvious on it.
 *   torus   — a surface with a hole and non-trivial curvature; UV wrapping shows up twice.
 *   knot    — a thin tube bent through itself. The hardest shape for a projected pattern,
 *             which is exactly why it is the one that tells you the mapping is right.
 *   cube    — flat faces meeting at 90°, the case where a bad normal or a bad tangent frame
 *             becomes a hard edge instead of a soft gradient.
 *   ground  — not a shape to look at but to stand on: it turns "an object" into "a scene",
 *             and gives the material something to reflect and shadow onto.
 *
 * ## Conventions
 *
 * Right-handed, Y up, model centred on the origin and normalised to roughly a unit radius,
 * so one camera distance frames every shape. Winding is counter-clockwise when seen from
 * outside, which is what the renderer's back-face culling assumes.
 *
 * Normals are computed analytically where the surface has a closed form (a sphere's normal
 * *is* its position) and from face accumulation otherwise. Analytic normals are not just
 * faster: accumulating face normals around a pole produces the shading artefact you would
 * then spend an hour blaming on the shader.
 */

/** Interleaved-free vertex buffers plus indices. Plain typed arrays, ready for `bufferData`. */
export type MeshData = {
  positions: Float32Array;
  normals: Float32Array;
  uvs: Float32Array;
  indices: Uint32Array;
  /** Bounding radius from the origin. The renderer frames the camera with this. */
  radius: number;
  /**
   * Lowest vertex, i.e. how far the shape reaches below the origin.
   *
   * Separate from `radius` because they answer different questions and conflating them is a
   * visible bug: the scene seats an object on the floor with `y = -minY`, and using `-radius`
   * instead puts a torus (radius 1.0, but only 0.28 deep) a third of a metre in the air. It is
   * the kind of error that looks like a lighting problem.
   */
  minY: number;
};

export type ShapeId = 'sphere' | 'torus' | 'knot' | 'cube' | 'ground';

export const SHAPE_IDS: readonly ShapeId[] = ['sphere', 'torus', 'knot', 'cube', 'ground'];

/** Incremental builder, so each generator reads as the surface it describes. */
class MeshBuilder {
  private readonly pos: number[] = [];
  private readonly nrm: number[] = [];
  private readonly uv: number[] = [];
  private readonly idx: number[] = [];

  vertex(px: number, py: number, pz: number, nx: number, ny: number, nz: number, u: number, v: number): number {
    const i = this.pos.length / 3;
    this.pos.push(px, py, pz);
    this.nrm.push(nx, ny, nz);
    this.uv.push(u, v);
    return i;
  }

  tri(a: number, b: number, c: number): void {
    this.idx.push(a, b, c);
  }

  /** Two triangles for the quad `a b c d` wound counter-clockwise. */
  quad(a: number, b: number, c: number, d: number): void {
    this.idx.push(a, b, c, a, c, d);
  }

  /** Index the next vertex will get. Lets a generator record a lattice's origin up front. */
  get vertexCount(): number {
    return this.pos.length / 3;
  }

  build(): MeshData {
    const positions = new Float32Array(this.pos);
    const normals = new Float32Array(this.nrm);
    let radius = 0;
    let minY = Infinity;
    for (let i = 0; i < positions.length; i += 3) {
      const x = positions[i] ?? 0;
      const y = positions[i + 1] ?? 0;
      const z = positions[i + 2] ?? 0;
      radius = Math.max(radius, Math.hypot(x, y, z));
      minY = Math.min(minY, y);
    }
    return {
      positions,
      normals,
      uvs: new Float32Array(this.uv),
      indices: new Uint32Array(this.idx),
      radius,
      minY: Number.isFinite(minY) ? minY : 0,
    };
  }
}

/** Grid vertex index for a `(rings+1) x (segments+1)` lattice. */
function gridIndex(i: number, j: number, rowLength: number): number {
  return i * rowLength + j;
}

/**
 * A UV sphere.
 *
 * The poles are real vertices rather than a shared degenerate ring, so the normals there are
 * exact (`(0, ±1, 0)`) instead of an average of quads that have no area. The seam column is
 * duplicated with `u = 1` so the texture does not wrap backwards across the last quad.
 */
export function sphere(segments = 64, rings = 32): MeshData {
  const b = new MeshBuilder();
  const rows = rings + 1;
  const cols = segments + 1;
  for (let i = 0; i < rows; i++) {
    const v = i / rings;
    const phi = v * Math.PI;
    const sp = Math.sin(phi);
    const cp = Math.cos(phi);
    for (let j = 0; j < cols; j++) {
      const u = j / segments;
      const theta = u * Math.PI * 2;
      const x = sp * Math.cos(theta);
      const y = cp;
      const z = sp * Math.sin(theta);
      b.vertex(x, y, z, x, y, z, u, 1 - v);
    }
  }
  for (let i = 0; i < rings; i++) {
    for (let j = 0; j < segments; j++) {
      const a = gridIndex(i, j, cols);
      const c = gridIndex(i, j + 1, cols);
      const d = gridIndex(i + 1, j + 1, cols);
      const e = gridIndex(i + 1, j, cols);
      // Skip the zero-area triangles at the two poles.
      if (i !== 0) b.tri(a, c, d);
      if (i !== rings - 1) b.tri(a, d, e);
    }
  }
  return b.build();
}

/**
 * A torus of major radius `R` and tube radius `r`.
 *
 * The normal is exact: the outward direction at a surface point is the vector from the
 * centre of the tube's cross-section, which is why this needs no normal accumulation.
 */
export function torus(segments = 128, sides = 48, R = 0.72, r = 0.28): MeshData {
  const b = new MeshBuilder();
  const ringCount = segments + 1;
  for (let i = 0; i <= segments; i++) {
    const u = i / segments;
    const phi = u * Math.PI * 2;
    const cosPhi = Math.cos(phi);
    const sinPhi = Math.sin(phi);
    for (let j = 0; j <= sides; j++) {
      const v = j / sides;
      const theta = v * Math.PI * 2;
      const cosTheta = Math.cos(theta);
      const sinTheta = Math.sin(theta);
      const nx = cosTheta * cosPhi;
      const ny = sinTheta;
      const nz = cosTheta * sinPhi;
      b.vertex(R * cosPhi + r * nx, r * ny, R * sinPhi + r * nz, nx, ny, nz, u, v);
    }
  }
  const cols = sides + 1;
  for (let i = 0; i < segments; i++) {
    for (let j = 0; j < sides; j++) {
      /*
       * `a d c b`, not `a b c d`.
       *
       * Walking the lattice in index order (i, then j) traverses this quad *clockwise* when
       * seen from outside, because the surface's outward normal is `N x B` rather than
       * `B x N` — the tube is parameterised with the ring going one way and the ring's
       * direction the other. Getting this wrong is invisible on a wireframe and black and
       * baffling on a lit render, so `mesh.test.ts` asserts that every face normal agrees
       * with the vertex normals it was built from.
       */
      b.quad(
        gridIndex(i, j, cols),
        gridIndex(i, j + 1, cols),
        gridIndex(i + 1, j + 1, cols),
        gridIndex(i + 1, j, cols),
      );
    }
  }
  return b.build();
}

/** A point on the (p, q) torus knot curve, matching the usual parametrisation. */
function knotPoint(t: number, p: number, q: number, radius: number): [number, number, number] {
  const quOverP = (q / p) * t;
  const cs = Math.cos(quOverP);
  return [
    radius * (2 + cs) * 0.5 * Math.cos(t),
    radius * (2 + cs) * 0.5 * Math.sin(t),
    radius * Math.sin(quOverP) * 0.5,
  ];
}

/**
 * A (p, q) torus knot swept into a tube.
 *
 * The frame is built from the curve tangent plus a reference up-vector, which is enough for
 * a knot whose curvature never brings the tangent parallel to up. A full parallel-transport
 * frame would be the general answer; it would also be forty more lines that change nothing
 * visible on this shape, and this file is meant to be readable.
 */
export function torusKnot(
  p = 2,
  q = 3,
  tubularSegments = 320,
  radialSegments = 24,
  radius = 0.62,
  tube = 0.17,
): MeshData {
  const b = new MeshBuilder();
  const cols = radialSegments + 1;
  const eps = 1e-4;

  for (let i = 0; i <= tubularSegments; i++) {
    const u = i / tubularSegments;
    const t = u * p * Math.PI * 2;
    const c = knotPoint(t, p, q, radius);
    const ahead = knotPoint(t + eps, p, q, radius);
    const behind = knotPoint(t - eps, p, q, radius);

    // Tangent by central difference: a knot has no closed-form normal, but it has a
    // derivative, and differencing it symmetrically keeps the frame stable at the ends.
    let tx = ahead[0] - behind[0];
    let ty = ahead[1] - behind[1];
    let tz = ahead[2] - behind[2];
    const tlen = Math.hypot(tx, ty, tz) || 1;
    tx /= tlen;
    ty /= tlen;
    tz /= tlen;

    // Normal = tangent x up, then binormal = tangent x normal.
    let nx = ty * 0 - tz * 1;
    let ny = tz * 0 - tx * 0;
    let nz = tx * 1 - ty * 0;
    let nlen = Math.hypot(nx, ny, nz);
    if (nlen < 1e-6) {
      // Degenerate only if the tangent is vertical; pick another reference axis.
      nx = ty * 0 - tz * 0;
      ny = tz * 1 - tx * 0;
      nz = tx * 0 - ty * 1;
      nlen = Math.hypot(nx, ny, nz) || 1;
    }
    nx /= nlen;
    ny /= nlen;
    nz /= nlen;

    const bx = ty * nz - tz * ny;
    const by = tz * nx - tx * nz;
    const bz = tx * ny - ty * nx;

    for (let j = 0; j <= radialSegments; j++) {
      const v = j / radialSegments;
      const theta = v * Math.PI * 2;
      const ct = Math.cos(theta);
      const st = Math.sin(theta);
      const ox = ct * nx + st * bx;
      const oy = ct * ny + st * by;
      const oz = ct * nz + st * bz;
      /*
       * `u`, not `u * p`.
       *
       * A (p, q) knot closes after `p` turns of the curve parameter, so `u * p` is the
       * tempting way to keep the texture's aspect even along the tube — and it is what this
       * did first. It also pushes `u` to 2.0 for the (2, 3) knot, i.e. outside the domain
       * the material samples, so the last stretch of the tube would read from whatever the
       * sampler does outside `[0, 1]` (clamp, usually, which smears the seam). One repeat
       * along the whole curve is equally seamless and stays inside the unit square.
       */
      b.vertex(c[0] + tube * ox, c[1] + tube * oy, c[2] + tube * oz, ox, oy, oz, u, v);
    }
  }

  for (let i = 0; i < tubularSegments; i++) {
    for (let j = 0; j < radialSegments; j++) {
      // Same reversal as the torus, and for the same reason: `B = T x N` means walking the
      // lattice in index order winds the quad clockwise from outside.
      b.quad(
        gridIndex(i, j, cols),
        gridIndex(i, j + 1, cols),
        gridIndex(i + 1, j + 1, cols),
        gridIndex(i + 1, j, cols),
      );
    }
  }
  return b.build();
}

/**
 * An axis-aligned cube with subdivided faces.
 *
 * Subdivision is not decoration: the material samples the pattern per fragment, so a
 * two-triangle face would show a visible diagonal in any flat-shaded term and would make a
 * displacement-style effect impossible.
 */
export function cube(size = 1.2, segments = 24): MeshData {
  const b = new MeshBuilder();
  const half = size / 2;
  // normal, then the two in-plane axes that keep the winding counter-clockwise outside.
  const faces: Array<[[number, number, number], [number, number, number], [number, number, number]]> = [
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
    [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
    [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
    [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
  ];

  for (const [n, right, up] of faces) {
    // Record where this face's lattice starts, then emit it. `origin` has to be read before
    // the loop: `vertexCount` is the index the *next* vertex will receive.
    const origin = b.vertexCount;
    for (let i = 0; i <= segments; i++) {
      for (let j = 0; j <= segments; j++) {
        const u = i / segments;
        const v = j / segments;
        const su = (u - 0.5) * size;
        const sv = (v - 0.5) * size;
        const x = n[0] * half + right[0] * su + up[0] * sv;
        const y = n[1] * half + right[1] * su + up[1] * sv;
        const z = n[2] * half + right[2] * su + up[2] * sv;
        b.vertex(x, y, z, n[0], n[1], n[2], u, v);
      }
    }
    const cols = segments + 1;
    for (let i = 0; i < segments; i++) {
      for (let j = 0; j < segments; j++) {
        b.quad(
          origin + gridIndex(i, j, cols),
          origin + gridIndex(i + 1, j, cols),
          origin + gridIndex(i + 1, j + 1, cols),
          origin + gridIndex(i, j + 1, cols),
        );
      }
    }
  }
  return b.build();
}

/**
 * A ground plane at `y = 0`, facing up.
 *
 * `segments` exists so that a large plane can carry a screen-space-ish pattern without the
 * material's per-fragment evaluation happening across two enormous triangles, which some
 * drivers interpolate differently enough to matter at grazing angles.
 */
export function ground(size = 8, segments = 32): MeshData {
  const b = new MeshBuilder();
  const cols = segments + 1;
  const half = size / 2;
  for (let i = 0; i <= segments; i++) {
    for (let j = 0; j <= segments; j++) {
      const u = i / segments;
      const v = j / segments;
      b.vertex((u - 0.5) * size, 0, (v - 0.5) * size, 0, 1, 0, u, v);
    }
  }
  for (let i = 0; i < segments; i++) {
    for (let j = 0; j < segments; j++) {
      b.quad(
        gridIndex(i, j, cols),
        gridIndex(i, j + 1, cols),
        gridIndex(i + 1, j + 1, cols),
        gridIndex(i + 1, j, cols),
      );
    }
  }
  return b.build();
}

/** Build by id, with the defaults each shape was tuned to. */
export function buildMesh(id: ShapeId): MeshData {
  switch (id) {
    case 'sphere':
      return sphere();
    case 'torus':
      return torus();
    case 'knot':
      return torusKnot();
    case 'cube':
      return cube();
    case 'ground':
      return ground();
  }
}

/** Human label for the UI. Kept next to the shapes so a new shape cannot be added silently. */
export const SHAPE_LABELS: Record<ShapeId, string> = {
  sphere: '球体',
  torus: '环面',
  knot: '环面纽结',
  cube: '立方体',
  ground: '地面',
};

/**
 * The shapes offered as an *object* to preview.
 *
 * `ground` is absent on purpose: it is the floor a scene is built on, not something to look
 * at, and offering it as an object would produce a camera framing a plane edge-on. It stays
 * in `SHAPE_IDS` because it is a real mesh with real invariants to test.
 */
export const OBJECT_SHAPES: readonly ShapeId[] = ['sphere', 'torus', 'knot', 'cube'];
