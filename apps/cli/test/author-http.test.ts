/**
 * `author` against a real HTTP provider — the loop, the repairer and the emitter end to end.
 *
 * Everything else in this repository tests the agent loop with a *scripted* provider, which
 * is the right way to pin the loop's own behaviour but skips a whole class of failure: a real
 * model answers with real, slightly-wrong JSON, and the interesting question is what the
 * pipeline does about it. This file closes that gap with a fake OpenAI-compatible endpoint —
 * a real socket, real HTTP, the real `createHttpProvider`, the real loop, the real emitter and
 * the real CLI process.
 *
 * It exists because of a concrete report. A `deepseek-chat` draft built a graph of six good
 * nodes and forgot the `output` node; the loop read `IR008_NO_OUTPUT`, consulted its
 * unrecoverable table, and gave up after three model calls — for a defect that is one
 * `add_node` and one `connect`. The repairer now handles it, and this test is what keeps that
 * true on the path a user actually takes rather than only under a scripted provider.
 *
 * ## The fake server answers in sequence
 *
 *   call 1  a valid `ShaderSpec` (the `interpret` stage)
 *   call 2  a patch that builds three connected nodes and no sink (the `draft` stage)
 *   call 3+ a no-op patch, so an unfixed loop would keep asking and eventually give up
 *
 * A pass therefore requires the run to succeed on the *second* reply. If the sink repair
 * regresses, the third reply is a no-op and the assertions on the written document fail.
 */

import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { closeSync, openSync } from 'node:fs';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { createServer, type Server } from 'node:http';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import test from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const CLI_DIR = resolve(HERE, '..');
const CLI = join(CLI_DIR, 'src/cli.ts');
const HOOKS = join(CLI_DIR, '.node/register-hooks.mjs');
const REPO = resolve(CLI_DIR, '../..');
const NODE_FLAGS = ['--experimental-strip-types', '--no-warnings', '--import', pathToFileURL(HOOKS).href];

const SPEC = {
  spec: {
    intent: '流动的液态金属表面，带细腻的拉丝与高光带',
    category: 'procedural-surface',
    tags: ['金属', '流体', '噪声'],
    features: ['连续流动的变形', '细密拉丝', '高光带', '冷色调'],
  },
};

/** A real-shaped draft: three connected nodes, and no sink. */
const DRAFT_WITHOUT_OUTPUT = {
  summary: 'build a noise field and tint it',
  ops: [
    { op: 'add_node', node: { id: 'n_uv', kind: 'uv' } },
    { op: 'add_node', node: { id: 'n_noise', kind: 'fbm' } },
    { op: 'connect', from: { node: 'n_uv', port: 'uv' }, to: { node: 'n_noise', port: 'uv' } },
    { op: 'add_node', node: { id: 'n_tint', kind: 'const_color' } },
    { op: 'connect', from: { node: 'n_noise', port: 'value' }, to: { node: 'n_tint', port: 'color' } },
  ],
};

type Fake = { server: Server; port: number; calls: () => number; close: () => Promise<void> };

/** The OpenAI-compatible envelope, with `content` carrying a JSON reply. */
async function fakeEndpoint(): Promise<Fake> {
  let calls = 0;
  const server = createServer((req, res) => {
    let body = '';
    req.on('data', (c) => (body += c));
    req.on('end', () => {
      calls++;
      const payload = calls === 1 ? SPEC : calls === 2 ? DRAFT_WITHOUT_OUTPUT : { summary: 'no-op', ops: [] };
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(
        JSON.stringify({
          model: 'fake-model',
          choices: [{ message: { role: 'assistant', content: JSON.stringify(payload) } }],
          usage: { prompt_tokens: 1, completion_tokens: 1 },
        }),
      );
    });
  });
  // Port 0: the kernel picks a free one. A fixed port would collide with a developer's own
  // server (or with a parallel run) and fail as a mystery connection error.
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', r));
  const address = server.address();
  const port = typeof address === 'object' && address ? address.port : 0;
  return {
    server,
    port,
    calls: () => calls,
    close: () => new Promise<void>((r) => server.close(() => r())),
  };
}

type Run = { code: number; stdout: string; stderr: string };

/**
 * Spawn the CLI and read its output back through **files**, not pipes.
 *
 * A confined sandbox denies `spawn` with any piped stdio (`EPERM`), and the failure looks like
 * the code under test is broken. Same technique, and same reasoning, as `cli.test.ts`.
 */
async function runCli(args: string[], tag: string): Promise<Run> {
  const outFile = join(tmpdir(), `ai4shader-http-${process.pid}-${tag}.out`);
  const errFile = join(tmpdir(), `ai4shader-http-${process.pid}-${tag}.err`);
  const outFd = openSync(outFile, 'w');
  const errFd = openSync(errFile, 'w');
  try {
    const code = await new Promise<number>((res) => {
      const child = spawn(process.execPath, [...NODE_FLAGS, 'apps/cli/src/cli.ts', ...args], {
        cwd: REPO,
        stdio: ['ignore', outFd, errFd],
      });
      child.on('close', (c) => res(c ?? 1));
      child.on('error', () => res(-1));
    });
    return {
      code,
      stdout: await readFile(outFile, 'utf8').catch(() => ''),
      stderr: await readFile(errFile, 'utf8').catch(() => ''),
    };
  } finally {
    closeSync(outFd);
    closeSync(errFd);
  }
}

test('author: a model draft with no output node is recovered, not abandoned', async (t) => {
  const fake = await fakeEndpoint();
  const dir = await mkdtemp(join(tmpdir(), 'ai4shader-http-'));
  t.after(async () => {
    await fake.close();
    await rm(dir, { recursive: true, force: true });
  });

  const run = await runCli(
    [
      'author',
      '液态金属表面，慢速流动',
      '--provider',
      'http',
      '--base-url',
      `http://127.0.0.1:${fake.port}/v1`,
      '--api-key',
      'test-key',
      '--model',
      'fake-model',
      '--doc-out',
      join(dir, 'authored.shader.json'),
      '--json',
    ],
    'author-nosink',
  );

  assert.equal(
    run.code,
    0,
    `author exited ${run.code}\n--- stdout ---\n${run.stdout}\n--- stderr ---\n${run.stderr}`,
  );

  const doc = JSON.parse(await readFile(join(dir, 'authored.shader.json'), 'utf8'));
  const kinds = doc.graph.nodes.map((n: { kind: string }) => n.kind);
  const out = doc.graph.nodes.find((n: { kind: string }) => n.kind === 'output');

  assert.ok(out, `the repair must have added an output node; kinds were: ${kinds.join(', ')}`);
  assert.ok(
    out.inputs?.color,
    'the added output must be wired to a source, not left as a flat default — a black frame ' +
      'reported as success is worse than an honest failure',
  );
  assert.ok(
    doc.graph.nodes.some((n: { id: string }) => n.id === out.inputs.color.node),
    `the wire must point at a node that exists, got ${JSON.stringify(out.inputs.color)}`,
  );

  // The run must have finished on the draft reply (plus the interpret that preceded it).
  // A regression here means the loop fell through to the no-op replies and burned its budget,
  // which is exactly the shape of the original report.
  assert.ok(
    fake.calls() <= 3,
    `expected at most 3 model calls (interpret + draft + one repair), the endpoint saw ${fake.calls()}`,
  );
});
