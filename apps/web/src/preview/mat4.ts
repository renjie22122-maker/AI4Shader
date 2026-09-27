/**
 * Just enough matrix math for one orbiting camera
 * ==============================================
 *
 * Four matrices and a multiply. The project has no dependencies and no `npm install`, so the
 * alternative to these ninety lines is vendoring a linear-algebra library — for a preview
 * that draws one mesh from one camera.
 *
 * ## Column-major, and why that is not a detail
 *
 * WebGL's `uniformMatrix4fv(loc, transpose, value)` expects **column-major** storage, which
 * is why `transpose` is almost always `false` and why a matrix laid out row-major here would
 * produce a render that looks *almost* right — a shape that is subtly sheared rather than
 * obviously broken. That is the worst kind of bug to find without a browser. Every function
 * below stores `m[col * 4 + row]`, the convention `glMatrix` uses, so any reference result
 * from a familiar library can be compared element for element.
 *
 * ## Tested, because the failure is silent
 *
 * `mat4.test.ts` checks the properties that actually matter and that a screenshot cannot:
 * that `perspective` maps the near plane to -1 and the far plane to +1 in clip space, that
 * `lookAt` produces an orthonormal basis looking down -Z, and that `multiply` composes in the
 * order the renderer uses (`projection * view * model`). A transposed matrix would still
 * produce a picture of something.
 */

export type Mat4 = Float32Array;
export type Vec3 = readonly [number, number, number];

export function identity(): Mat4 {
  const m = new Float32Array(16);
  m[0] = 1;
  m[5] = 1;
  m[10] = 1;
  m[15] = 1;
  return m;
}

/**
 * `a * b`, in that order.
 *
 * The order is the whole contract: the renderer wants `projection * view * model`, so that a
 * point is transformed by the model first and by the projection last. Writing this the other
 * way round produces a camera that appears to swing the object in the wrong direction, which
 * reads as "the orbit control is inverted" and gets fixed in the wrong file.
 */
export function multiply(a: Mat4, b: Mat4): Mat4 {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col++) {
    for (let row = 0; row < 4; row++) {
      let sum = 0;
      for (let k = 0; k < 4; k++) sum += (a[k * 4 + row] ?? 0) * (b[col * 4 + k] ?? 0);
      out[col * 4 + row] = sum;
    }
  }
  return out;
}

/**
 * Right-handed perspective projection into the OpenGL clip range, `z ∈ [-1, 1]`.
 *
 * The `-1` in the `[2][3]` slot (column 2, row 3) is what makes the result right-handed: the
 * camera looks down -Z, so points in front of it have negative view-space z and the `w`
 * component must be `-z` for the perspective divide to come out positive.
 */
export function perspective(fovYRadians: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovYRadians / 2);
  const nf = 1 / (near - far);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) * nf;
  m[11] = -1;
  m[14] = 2 * far * near * nf;
  return m;
}

function normalize(v: Vec3): Vec3 {
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/**
 * A view matrix, built by hand rather than by inverting a camera transform.
 *
 * The basis is orthonormal by construction (`z = normalize(eye - target)`, `x = normalize(up
 * × z)`, `y = z × x`), so the inverse of the rotation is its transpose and no general matrix
 * inverse is needed. `up` being parallel to the view direction — the camera looking straight
 * down — would make `x` degenerate; the caller clamps pitch before that happens, and the
 * `|| 1` guards keep a NaN from propagating if it ever does.
 */
export function lookAt(eye: Vec3, target: Vec3, up: Vec3): Mat4 {
  const z = normalize([eye[0] - target[0], eye[1] - target[1], eye[2] - target[2]]);
  const x = normalize(cross(up, z));
  const y = cross(z, x);
  const m = new Float32Array(16);
  m[0] = x[0];
  m[1] = y[0];
  m[2] = z[0];
  m[4] = x[1];
  m[5] = y[1];
  m[6] = z[1];
  m[8] = x[2];
  m[9] = y[2];
  m[10] = z[2];
  m[12] = -dot(x, eye);
  m[13] = -dot(y, eye);
  m[14] = -dot(z, eye);
  m[15] = 1;
  return m;
}

/** Rotation about Y, used to spin the model. */
export function rotationY(angle: number): Mat4 {
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  const m = identity();
  m[0] = c;
  m[2] = -s;
  m[8] = s;
  m[10] = c;
  return m;
}

/** Translation, used to lift a shape off the ground in scene mode. */
export function translation(x: number, y: number, z: number): Mat4 {
  const m = identity();
  m[12] = x;
  m[13] = y;
  m[14] = z;
  return m;
}

/**
 * The upper-left 3×3 of a model matrix, as a 9-element column-major array.
 *
 * Correct for a rigid transform — a rotation and a translation — because the inverse
 * transpose of a rotation is the rotation. It is **not** correct under non-uniform scale,
 * where normals genuinely need the inverse transpose; the preview never scales a mesh (it
 * frames the camera instead), and doing the full inverse here would be code that exists to
 * handle a case that cannot occur.
 */
export function normalMatrix3(model: Mat4): Float32Array {
  const out = new Float32Array(9);
  out[0] = model[0] ?? 0;
  out[1] = model[1] ?? 0;
  out[2] = model[2] ?? 0;
  out[3] = model[4] ?? 0;
  out[4] = model[5] ?? 0;
  out[5] = model[6] ?? 0;
  out[6] = model[8] ?? 0;
  out[7] = model[9] ?? 0;
  out[8] = model[10] ?? 0;
  return out;
}

/** Transform a point by a matrix, with the perspective divide. */
export function transformPoint(m: Mat4, p: Vec3): [number, number, number] {
  const x = p[0];
  const y = p[1];
  const z = p[2];
  const w = (m[3] ?? 0) * x + (m[7] ?? 0) * y + (m[11] ?? 0) * z + (m[15] ?? 0);
  const iw = w === 0 ? 1 : 1 / w;
  return [
    ((m[0] ?? 0) * x + (m[4] ?? 0) * y + (m[8] ?? 0) * z + (m[12] ?? 0)) * iw,
    ((m[1] ?? 0) * x + (m[5] ?? 0) * y + (m[9] ?? 0) * z + (m[13] ?? 0)) * iw,
    ((m[2] ?? 0) * x + (m[6] ?? 0) * y + (m[10] ?? 0) * z + (m[14] ?? 0)) * iw,
  ];
}

export const deg2rad = (deg: number): number => (deg * Math.PI) / 180;
