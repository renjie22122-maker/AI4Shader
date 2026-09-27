/**
 * Camera math invariants
 * ======================
 *
 * A wrong matrix here does not throw, does not warn, and does not look obviously wrong: it
 * renders a picture of something. A transposed projection still shows a shape; a `lookAt`
 * with a non-orthonormal basis still shows a shape, gently sheared; a multiply in the wrong
 * order produces orbit controls that feel inverted, which is a bug people fix in the wrong
 * file. None of that is reachable from a screenshot this environment cannot take anyway.
 *
 * So the properties are asserted directly, and they are the properties that define the
 * matrices rather than the numbers they happen to produce: the near plane maps to -1 and the
 * far plane to +1 (the definition of an OpenGL projection), the view basis is orthonormal and
 * looks down -Z (the definition of a right-handed view), composition is associative and
 * order-sensitive (the definition of a matrix product), and `normalMatrix3` is the rotation
 * it claims to be (so that normals rotate with the surface instead of against it).
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  deg2rad,
  identity,
  lookAt,
  multiply,
  normalMatrix3,
  perspective,
  rotationY,
  transformPoint,
  translation,
  type Mat4,
  type Vec3,
} from '../src/preview/mat4.ts';

const EPS = 1e-5;

function close(actual: number, expected: number, what: string, eps = EPS): void {
  assert.ok(
    Math.abs(actual - expected) <= eps,
    `${what}: expected ${expected}, got ${actual} (tolerance ${eps})`,
  );
}

/** `m * v` for a column vector, ignoring the perspective divide. */
function mulVec(m: Mat4, v: readonly [number, number, number, number]): [number, number, number, number] {
  const out: [number, number, number, number] = [0, 0, 0, 0];
  for (let row = 0; row < 4; row++) {
    let sum = 0;
    for (let k = 0; k < 4; k++) sum += (m[k * 4 + row] ?? 0) * v[k]!;
    out[row] = sum;
  }
  return out;
}

test('identity leaves a point where it was', () => {
  const p = transformPoint(identity(), [1, -2, 3]);
  close(p[0], 1, 'x');
  close(p[1], -2, 'y');
  close(p[2], 3, 'z');
});

test('multiply composes right-to-left, as the renderer uses it', () => {
  // (T * Ry) applied to a point must translate first and rotate second — i.e. the rotation
  // happens in the translated frame. Getting this backwards is the classic "orbit feels
  // inverted" bug.
  const t = translation(5, 0, 0);
  const r = rotationY(Math.PI / 2); // +X -> -Z
  const combined = multiply(t, r);
  const p = transformPoint(combined, [1, 0, 0]);
  // rotate (1,0,0) -> (0,0,-1), then translate by (5,0,0) -> (5,0,-1)
  close(p[0], 5, 'x');
  close(p[1], 0, 'y');
  close(p[2], -1, 'z');

  const reversed = transformPoint(multiply(r, t), [1, 0, 0]);
  // translate first -> (6,0,0), then rotate -> (0,0,-6)
  close(reversed[0], 0, 'reversed x');
  close(reversed[2], -6, 'reversed z');
});

test('multiply is associative and identity is neutral', () => {
  const a = rotationY(0.7);
  const b = translation(1, 2, 3);
  const c = rotationY(-0.3);
  const left = multiply(multiply(a, b), c);
  const right = multiply(a, multiply(b, c));
  for (let i = 0; i < 16; i++) close(left[i] ?? 0, right[i] ?? 0, `element ${i}`);

  const id = identity();
  for (let i = 0; i < 16; i++) {
    close(multiply(a, id)[i] ?? 0, a[i] ?? 0, `a*I element ${i}`);
    close(multiply(id, a)[i] ?? 0, a[i] ?? 0, `I*a element ${i}`);
  }
});

test('perspective maps the near plane to -1 and the far plane to +1', () => {
  // This is the *definition* of an OpenGL-style projection and the single property that a
  // transposed or sign-flipped matrix fails. It is also what makes the depth test work.
  const near = 0.1;
  const far = 100;
  const p = perspective(deg2rad(50), 16 / 9, near, far);

  const onNear = mulVec(p, [0, 0, -near, 1]);
  close(onNear[2] / onNear[3], -1, 'near plane depth');

  const onFar = mulVec(p, [0, 0, -far, 1]);
  close(onFar[2] / onFar[3], 1, 'far plane depth');

  // `w` must be -z: the divide has to come out positive for a point in front of the camera.
  const front = mulVec(p, [0, 0, -5, 1]);
  assert.ok(front[3] > 0, `w should be positive in front of the camera, got ${front[3]}`);
  close(front[3], 5, 'w = -z');
});

test('perspective narrows with a smaller field of view and respects aspect', () => {
  const wide = perspective(deg2rad(90), 1, 0.1, 10);
  const narrow = perspective(deg2rad(30), 1, 0.1, 10);
  // tan(45) = 1, tan(15) ~ 0.268: the narrower the fov, the larger the focal term.
  close(wide[5] ?? 0, 1, 'wide fov focal length');
  assert.ok((narrow[5] ?? 0) > (wide[5] ?? 0), 'a narrower fov must magnify');

  const wideAspect = perspective(deg2rad(60), 2, 0.1, 10);
  close((wideAspect[0] ?? 0) / (wideAspect[5] ?? 0), 0.5, 'x scale is y scale / aspect');
});

test('lookAt produces an orthonormal basis looking down -Z', () => {
  const eye: Vec3 = [3, 2, 5];
  const target: Vec3 = [0, 0, 0];
  const m = lookAt(eye, target, [0, 1, 0]);

  // Rows 0..2 of the rotation part are the camera basis vectors.
  const x: Vec3 = [m[0] ?? 0, m[4] ?? 0, m[8] ?? 0];
  const y: Vec3 = [m[1] ?? 0, m[5] ?? 0, m[9] ?? 0];
  const z: Vec3 = [m[2] ?? 0, m[6] ?? 0, m[10] ?? 0];

  for (const [name, v] of [['x', x], ['y', y], ['z', z]] as const) {
    close(Math.hypot(v[0], v[1], v[2]), 1, `basis ${name} is unit length`);
  }
  const dt = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  close(dt(x, y), 0, 'x . y');
  close(dt(y, z), 0, 'y . z');
  close(dt(x, z), 0, 'x . z');

  // The origin must land on the view axis at minus the eye distance.
  const p = transformPoint(m, target);
  close(p[0], 0, 'target x on the axis');
  close(p[1], 0, 'target y on the axis');
  close(p[2], -Math.hypot(eye[0], eye[1], eye[2]), 'target sits at -distance along -Z');
});

test('normalMatrix3 extracts the rotation, not the translation', () => {
  const r = rotationY(0.9);
  const model = multiply(translation(10, -4, 2), r);
  const n = normalMatrix3(model);

  // A normal along +X rotated by 0.9 rad about Y is (cos, 0, -sin); translation must not
  // leak in, or the lighting would slide across the surface as the object spins.
  const expected: [number, number, number] = [Math.cos(0.9), 0, -Math.sin(0.9)];
  const got: [number, number, number] = [
    (n[0] ?? 0) * 1 + (n[3] ?? 0) * 0 + (n[6] ?? 0) * 0,
    (n[1] ?? 0) * 1 + (n[4] ?? 0) * 0 + (n[7] ?? 0) * 0,
    (n[2] ?? 0) * 1 + (n[5] ?? 0) * 0 + (n[8] ?? 0) * 0,
  ];
  for (let i = 0; i < 3; i++) close(got[i]!, expected[i]!, `rotated normal component ${i}`);
});

test('rotationY is a proper rotation: it preserves length and det = +1', () => {
  const m = rotationY(1.234);
  const p = transformPoint(m, [1, 2, 3]);
  close(Math.hypot(p[0], p[1], p[2]), Math.hypot(1, 2, 3), 'length preserved');

  const det =
    (m[0] ?? 0) * ((m[5] ?? 0) * (m[10] ?? 0) - (m[6] ?? 0) * (m[9] ?? 0)) -
    (m[4] ?? 0) * ((m[1] ?? 0) * (m[10] ?? 0) - (m[2] ?? 0) * (m[9] ?? 0)) +
    (m[8] ?? 0) * ((m[1] ?? 0) * (m[6] ?? 0) - (m[2] ?? 0) * (m[5] ?? 0));
  // det = -1 would be a reflection: the mesh would render inside-out while every vertex
  // stayed exactly on the surface, which is a very confusing thing to debug.
  close(det, 1, 'determinant');
});
