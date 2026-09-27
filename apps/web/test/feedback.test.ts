/**
 * The feedback path — a stateless shader with a memory
 * ====================================================
 *
 * `render.feedback` asks the host to put the **previous frame** into the image channel. That one
 * binding is what turns "a shader that draws the current frame" into "a shader that can accumulate":
 * trails, smears, a fog that stays wiped where a finger went, damped feedback, smoke that drifts.
 * Without it every frame starts from nothing, which is why the same document looks like a shader
 * that does nothing.
 *
 * ## What is actually at risk
 *
 * Three failures, none of which announces itself:
 *
 *  1. **Sampling the render target.** GL forbids reading a texture you are writing. Drivers
 *     respond with garbage, a black frame, or a silently dropped draw — and the code looks
 *     symmetric and reasonable either way. The ping-pong exists only to avoid this, so this file
 *     asserts the property directly: the texture bound to the channel must not be the texture
 *     attached to the framebuffer.
 *  2. **Swapping at the wrong moment.** Swap before the draw and the accumulation reads the frame
 *     being written; swap twice and it never advances. Both produce a *still* image rather than an
 *     error.
 *  3. **A resize that keeps the old attachment.** The next frame samples a differently-sized
 *     texture. On a driver that tolerates it the picture is garbage; on one that does not it is
 *     black.
 *
 * The recorder in `recording-gl.ts` models framebuffer attachments and the bound texture
 * precisely so these are checkable here, where there is no GPU. What it cannot check is what the
 * pixels look like — stated plainly rather than implied by a green test.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import { RecordingGl } from './recording-gl.ts';
import { BLIT_FRAGMENT_SOURCE, WebGL2Renderer } from '../dist/preview/glsl.js';

/** A tiny quad shader that samples the channel — the smallest thing that needs feedback. */
const CHANNEL_SHADER = `#version 300 es
precision highp float;
uniform vec3 iResolution;
uniform float iTime;
uniform sampler2D iChannel0;
out vec4 fragColor;
void main() {
  vec2 uv = gl_FragCoord.xy / iResolution.xy;
  vec4 previous = texture(iChannel0, uv);
  fragColor = mix(previous, vec4(uv, 0.5, 1.0), 0.1);
}
`;

const UNIFORMS = [
  { name: 'u_dummy', type: 'float' as const, value: 0.5, origin: { node: 'n', param: 'p' } },
];

function makeRenderer(): { gl: RecordingGl; renderer: WebGL2Renderer } {
  const gl = new RecordingGl();
  const canvas = {
    width: 64,
    height: 64,
    getContext: (kind: string) => (kind === 'webgl2' ? gl : null),
  } as unknown as HTMLCanvasElement;
  const renderer = WebGL2Renderer.create(canvas);
  assert.ok(renderer, 'the recording context must be accepted');
  return { gl, renderer };
}

test('feedback initialization sees frame zero and time rewind resets history', async () => {
  const { gl, renderer } = makeRenderer();
  const source = CHANNEL_SHADER.replace('uniform float iTime;', 'uniform float iTime;\nuniform float iTimeDelta;\nuniform int iFrame;')
    .replace('vec4 previous = texture(iChannel0, uv);', 'vec4 previous = texture(iChannel0, uv) + vec4(float(iFrame) + iTimeDelta);');
  await renderer.compile(source, UNIFORMS, { feedback: true });
  renderer.render(1);
  assert.equal(gl.written.get('iFrame'), 0);
  renderer.render(1.1);
  assert.equal(gl.written.get('iFrame'), 1);
  assert.ok(Number(gl.written.get('iTimeDelta')) > 0);
  renderer.render(0);
  assert.equal(gl.written.get('iFrame'), 0);
  assert.equal(gl.written.get('iTimeDelta'), 0);
  renderer.dispose();
});

test('the previous frame is sampled from a texture that is not being rendered into', async () => {
  const { gl, renderer } = makeRenderer();
  await renderer.compile(CHANNEL_SHADER, UNIFORMS, { feedback: true });
  renderer.resize(64, 64);
  renderer.render(1 / 60);

  const [graphPass, blitPass] = gl.trianglePasses;
  assert.ok(graphPass && blitPass, `expected two passes, got ${gl.trianglePasses.length}`);

  // The graph draws into an offscreen texture...
  assert.notEqual(graphPass.target, null, 'the graph must draw into an offscreen target');
  assert.equal(graphPass.sampler, graphPass.target === null ? null : graphPass.sampler);
  // ...and the texture it samples is the *other* one. This is the assertion the whole feature
  // hangs on: `sampler === target` is a feedback loop, which is undefined behaviour.
  assert.notEqual(
    graphPass.sampler,
    graphPass.target,
    'the graph sampled the texture it was rendering into — that is a feedback loop, not feedback',
  );

  // The blit reads what the graph just wrote and goes to the screen.
  assert.equal(blitPass.framebuffer, null, 'the blit must target the canvas');
  assert.equal(blitPass.sampler, graphPass.target, 'the blit must read the frame that was just drawn');
  assert.ok(
    gl.compiledSources.includes(BLIT_FRAGMENT_SOURCE),
    'the blit program was never compiled',
  );
});

test('the ping-pong alternates, so the accumulation advances instead of standing still', async () => {
  const { gl, renderer } = makeRenderer();
  await renderer.compile(CHANNEL_SHADER, UNIFORMS, { feedback: true });
  renderer.resize(32, 32);

  const seen: (number | null)[] = [];
  for (let i = 0; i < 4; i++) {
    renderer.render(i / 60);
    const pass = gl.trianglePasses.at(-2); // the graph pass of this frame; the last is the blit
    assert.ok(pass);
    seen.push(pass.sampler);
  }
  // Two distinct buffers in strict alternation. A single repeated value means the swap is missing
  // (the shader would accumulate into the same texture and read stale data); an unordered pair
  // means it happens twice per frame.
  assert.equal(new Set(seen).size, 2, `expected two buffers, saw ${JSON.stringify(seen)}`);
  assert.notEqual(seen[0], seen[1], 'the buffers must alternate every frame');
  assert.equal(seen[0], seen[2], 'the ping-pong must return to the first buffer on frame 3');
  assert.equal(seen[1], seen[3], 'the ping-pong must return to the second buffer on frame 4');
});

test('a resize reallocates the pair at the new size', async () => {
  const { gl, renderer } = makeRenderer();
  await renderer.compile(CHANNEL_SHADER, UNIFORMS, { feedback: true });
  renderer.resize(48, 32);
  renderer.render(0);

  const sizes = [...gl.textureSizes.entries()].filter(([, size]) => size[0] !== 0);
  assert.ok(sizes.length >= 2, `expected two sized textures, got ${JSON.stringify(sizes)}`);
  for (const [id, size] of sizes) {
    assert.deepEqual(size, [48, 32], `texture ${id} was allocated at ${size.join('x')}`);
  }

  // And a later resize must not leave the old attachment in place: sampling a stale-sized
  // texture is garbage rather than an error, which is exactly the failure worth pinning.
  const before = new Set(gl.textureSizes.keys());
  renderer.resize(16, 16);
  renderer.render(1 / 60);
  const newest = [...gl.textureSizes.entries()].filter(([id]) => !before.has(id));
  assert.ok(newest.length >= 2, 'a resize must allocate a new pair');
  for (const [, size] of newest) assert.deepEqual(size, [16, 16]);
});

test('without the flag nothing is allocated and the graph draws straight to the canvas', async () => {
  const { gl, renderer } = makeRenderer();
  await renderer.compile(CHANNEL_SHADER, UNIFORMS, { feedback: false });
  renderer.render(0);

  assert.equal(gl.trianglePasses.length, 1, 'exactly one pass: no blit, no offscreen target');
  assert.equal(gl.trianglePasses[0]!.framebuffer, null, 'the default path draws to the canvas');
  assert.equal(renderer.feedbackEnabled, false);
  assert.equal(gl.textureSizes.size, 0, 'no feedback buffers may be allocated when unused');
});

test('turning feedback off releases the buffers', async () => {
  const { gl, renderer } = makeRenderer();
  await renderer.compile(CHANNEL_SHADER, UNIFORMS, { feedback: true });
  renderer.render(0);
  const allocated = [...gl.textureSizes.keys()];
  assert.equal(allocated.length, 2);

  await renderer.compile(CHANNEL_SHADER, UNIFORMS, { feedback: false });
  for (const id of allocated) {
    assert.equal(gl.textureSizes.has(id), false, `texture ${id} leaked when feedback was turned off`);
  }
  assert.equal(renderer.feedbackEnabled, false);
});

test('a fresh compile restarts the accumulation', async () => {
  /*
   * Accumulated state belongs to the *program that produced it*. Carrying it into a new one means
   * the user edits the document and sees a ghost of the previous shader fading out — which reads
   * as a rendering bug rather than as a design decision.
   *
   * Observed through the clear: `resetAccumulation` clears both buffers, and the recorder sees a
   * clear while an offscreen framebuffer is bound.
   */
  const { gl, renderer } = makeRenderer();
  await renderer.compile(CHANNEL_SHADER, UNIFORMS, { feedback: true });
  renderer.render(0);
  const clearsBefore = gl.calls.filter((c) => c.name === 'clear').length;

  await renderer.compile(CHANNEL_SHADER.replace('0.1', '0.2'), UNIFORMS, { feedback: true });
  const clearsAfter = gl.calls.filter((c) => c.name === 'clear').length;
  assert.ok(clearsAfter > clearsBefore, 'recompiling must clear the accumulation buffers');
});
