/**
 * `apps/cli` — end-to-end tests.
 *
 * Every test here **shells out** to the real CLI with `node --experimental-strip-types`,
 * because the contract being tested is the process boundary: stdout, stderr, exit codes and
 * the files that land on disk. Calling `main()` in-process would test a different program.
 *
 * The tests do *not* depend on `packages/ir/src/catalog` or `.../targets` existing — those
 * are owned by other agents and were not present when this file was written. Each test that
 * genuinely needs the real emitters is named "(requires catalog)" and `t.skip()`s itself
 * when the directory is missing, so the suite is green now and exercises the real path
 * automatically once those modules land.
 *
 * Run with:  node --test --import ./apps/cli/.node/register-hooks.mjs apps/cli/test/cli.test.ts
 *       or:  npm test --workspace @ai4shader/cli      (from the repo root)
 *
 * NOTE ON ISOLATION. `node --test` runs each test *file* in a child process by default, and
 * a restricted sandbox that forbids `child_process.spawn` with piped stdio (EPERM on
 * `spawn`) cannot do that. `--experimental-test-isolation=none` runs the file in-process and
 * makes this suite runnable there — but note that with isolation disabled Node's test runner
 * treats its positional arguments as *files*, not glob patterns, so point it at
 * `test/cli.test.ts` rather than at the `test/` directory.
 */

import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { closeSync, openSync } from 'node:fs';
import { cp, mkdtemp, mkdir, readFile, rm, writeFile, access } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, isAbsolute, join, relative, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { makeBrokenDoc, makeDoc, makeStructuralDoc } from './fixtures.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const CLI_DIR = resolve(HERE, '..');
const CLI = join(CLI_DIR, 'src/cli.ts');
const HOOKS = join(CLI_DIR, '.node/register-hooks.mjs');
const REPO = resolve(CLI_DIR, '../..');
const IR_SRC = join(REPO, 'packages/ir/src');

/**
 * Flags for every child process.
 *
 * The hook and the entry point are passed as **`file://` URLs, not paths**, for two
 * independent reasons that both bite on Windows:
 *
 *   • `--import` resolves a relative specifier against the *child's* cwd, and most tests here
 *     deliberately run the CLI from an unrelated temporary directory. A relative hook path
 *     therefore fails exactly where the suite is trying to prove cwd-independence.
 *   • An absolute Windows path (`D:\…`) is not a valid ESM specifier; Node's loader rejects
 *     it with `ERR_UNSUPPORTED_ESM_URL_SCHEME` ("absolute paths must be valid file:// URLs").
 *
 * Passing URLs fixes both at once and is also what the tests are supposed to be measuring:
 * the CLI's behaviour, not the harness's.
 */
const NODE_FLAGS = ['--experimental-strip-types', '--no-warnings', '--import', pathToFileURL(HOOKS).href];

/* -------------------------------------------------------------------------- */
/* Harness                                                                     */
/* -------------------------------------------------------------------------- */

type Run = { code: number; stdout: string; stderr: string };

/**
 * Spawn a command and capture its output **through files rather than pipes**.
 *
 * Why not `stdio: ['pipe','pipe','pipe']`, which is the obvious way to do this: a confined
 * sandbox (and some hardened CI runners) denies `spawn` with any piped stdio, because a pipe
 * is a named pipe and opening one is the thing being restricted. The failure is `EPERM`
 * thrown synchronously, so it looks like the *code under test* is broken when in fact the
 * harness is.
 *
 * Verified on this machine: `inherit` and `ignore` both succeed, `pipe` (in any combination,
 * including stdout-only) throws `EPERM`. Redirecting the child's stdout/stderr to real files
 * and reading them back afterwards needs no pipe at all, so the contract — *assert on the
 * process boundary: its exit code, its stdout, its stderr* — is preserved exactly, and the
 * suite runs in both environments instead of being skipped in one of them.
 *
 * `stdin` is only wired up when a test actually needs it; otherwise the child gets `ignore`,
 * which is both honest (nothing is being fed in) and one less inherited descriptor to reason
 * about.
 */
async function capture(
  file: string,
  args: string[],
  opts: { cwd?: string; stdin?: string; env?: Record<string, string> } = {},
): Promise<Run> {
  const tag = `${process.pid}-${Math.random().toString(36).slice(2, 8)}`;
  const outFile = join(tmpdir(), `ai4shader-run-${tag}.out`);
  const errFile = join(tmpdir(), `ai4shader-run-${tag}.err`);
  const inFile = opts.stdin !== undefined ? join(tmpdir(), `ai4shader-run-${tag}.in`) : null;
  if (inFile) await writeFile(inFile, opts.stdin!, 'utf8');

  const outFd = openSync(outFile, 'w');
  const errFd = openSync(errFile, 'w');
  const inFd = inFile ? openSync(inFile, 'r') : 'ignore';

  try {
    const code = await new Promise<number>((res, rej) => {
      const cwd = opts.cwd ?? REPO;
      // The entry point is passed as a path **relative to the child's cwd**, and only
      // converted to a `file://` URL when it cannot be made relative (a different drive on
      // Windows). Two constraints meet here:
      //   • An absolute Windows path (`D:\…`) is not a valid ESM specifier — Node's loader
      //     rejects it with `ERR_UNSUPPORTED_ESM_URL_SCHEME`.
      //   • A `file://` URL solves that, but as the *entry-point* argument Node resolves it
      //     against the cwd, producing `…\Temp\file:\D:\…` and `ERR_MODULE_NOT_FOUND`.
      // A relative path satisfies both, which is also what a user typing the command would
      // do — so the tests exercise the ordinary invocation rather than a harness-only form.
      const absolute = resolve(file);
      // `relative` returns an absolute path when the two are on different drives (C: temp
      // vs D: repo, which is the normal layout here), so that case must be detected rather
      // than falling through to the URL form — a `file://` URL as the *entry point* is
      // resolved against the cwd by Node and produces `…\Temp\file:\D:\…`.
      const rel = relative(cwd, absolute);
      const entry = rel && !isAbsolute(rel) && !rel.startsWith('..') ? rel : absolute;
      let child;
      try {
        child = spawn(process.execPath, [...NODE_FLAGS, entry, ...args], {
          cwd,
          stdio: [inFd, outFd, errFd],
          // `env` is merged, not replaced, so the child still finds PATH etc.; passing a
          // variable as an empty string is how a test removes ambient configuration (a
          // developer with OPENAI_API_KEY exported would otherwise silently take the
          // network path).
          ...(opts.env ? { env: { ...process.env, ...opts.env } } : {}),
        });
      } catch (err) {
        rej(err);
        return;
      }
      child.on('error', rej);
      child.on('close', (c) => res(c ?? -1));
    });

    return {
      code,
      stdout: await readFile(outFile, 'utf8'),
      stderr: await readFile(errFile, 'utf8'),
    };
  } finally {
    closeSync(outFd);
    closeSync(errFd);
    if (typeof inFd === 'number') closeSync(inFd);
    await rm(outFile, { force: true });
    await rm(errFile, { force: true });
    if (inFile) await rm(inFile, { force: true });
  }
}

/**
 * Spawn the CLI. `cwd` is a parameter on purpose: "every command must work when run from
 * any cwd" is a requirement, so most tests run from a temporary directory unrelated to the
 * repo.
 */
function runCli(
  args: string[],
  opts: { cwd?: string; stdin?: string; env?: Record<string, string> } = {},
): Promise<Run> {
  return capture(CLI, args, opts);
}

const exists = async (p: string): Promise<boolean> => {
  try {
    await access(p);
    return true;
  } catch {
    return false;
  }
};

/** A scratch directory per test: no test can see another's generated files. */
async function scratch(t: { after: (fn: () => Promise<void> | void) => void }): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'ai4shader-cli-'));
  t.after(async () => {
    await rm(dir, { recursive: true, force: true });
  });
  return dir;
}

async function writeJson(dir: string, name: string, value: unknown): Promise<string> {
  const file = join(dir, name);
  await mkdir(dirname(file), { recursive: true });
  await writeFile(file, `${JSON.stringify(value, null, 2)}\n`, 'utf8');
  return file;
}

/** The real catalog lands here; every `(requires catalog)` test keys off this path. */
const CATALOG_PRESENT = await exists(join(IR_SRC, 'catalog/index.ts'));
const TARGETS_PRESENT = await exists(join(IR_SRC, 'targets/index.ts'));
const REAL_IR = CATALOG_PRESENT && TARGETS_PRESENT;

/** Does the real catalog actually define every kind a document uses? */
async function catalogCovers(kinds: string[]): Promise<boolean> {
  if (!CATALOG_PRESENT) return false;
  const mod = (await import(pathToFileURL(join(IR_SRC, 'catalog/index.ts')).href)) as {
    CATALOG?: Record<string, unknown>;
  };
  return kinds.every((k) => mod.CATALOG?.[k] !== undefined);
}

/* -------------------------------------------------------------------------- */
/* --help                                                                      */
/* -------------------------------------------------------------------------- */

test('--help prints usage and exits 0, from any cwd', async (t) => {
  const dir = await scratch(t);
  for (const args of [[], ['--help'], ['-h'], ['help']]) {
    const r = await runCli(args, { cwd: dir });
    assert.equal(r.code, 0, `\`ai4shader ${args.join(' ')}\` should exit 0, got ${r.code}\n${r.stderr}`);
    assert.match(r.stdout, /ai4shader/);
    assert.match(r.stdout, /validate/);
    assert.match(r.stdout, /emit/);
    assert.match(r.stdout, /params/);
    assert.match(r.stdout, /tune/);
    assert.match(r.stdout, /EXIT CODES/);
  }

  // Every subcommand documents itself, and the help mentions its own flags.
  const subs: [string, RegExp[]][] = [
    ['validate', [/ai4shader validate/, /--json/, /EXIT/]],
    ['emit', [/ai4shader emit/, /--target/, /shader\.frag/, /params\.json/, /--force/, /NEVER SILENTLY OVERWRITTEN/]],
    ['params', [/ai4shader params/, /--set/, /PATCHER/]],
    ['tune', [/ai4shader tune/, /--param/, /--steps/, /256/, /sweep\.json/]],
  ];
  for (const [cmd, patterns] of subs) {
    const r = await runCli([cmd, '--help'], { cwd: dir });
    assert.equal(r.code, 0, `\`${cmd} --help\` exited ${r.code}`);
    for (const p of patterns) assert.match(r.stdout, p, `\`${cmd} --help\` is missing ${p}`);
  }
  // The short form works too.
  assert.equal((await runCli(['validate', '-h'], { cwd: dir })).code, 0);
});

test('an unknown command is a usage error (exit 2)', async () => {
  const r = await runCli(['frobnicate']);
  assert.equal(r.code, 2);
  assert.match(r.stderr, /unknown command/);
  assert.match(r.stderr, /USAGE/);
});

test('--version prints a version and exits 0', async () => {
  const r = await runCli(['--version']);
  assert.equal(r.code, 0);
  assert.match(r.stdout.trim(), /^\d+\.\d+\.\d+$/);
});

/* -------------------------------------------------------------------------- */
/* validate                                                                    */
/* -------------------------------------------------------------------------- */

test('validate: a structurally sound document exits 0 and prints the summary table', async (t) => {
  const dir = await scratch(t);
  const doc = join(dir, 'doc.shader.json');
  await writeFile(doc, JSON.stringify(makeStructuralDoc(), null, 2), 'utf8');

  const r = await runCli(['validate', doc], { cwd: dir });
  assert.equal(r.code, 0, `expected 0, got ${r.code}\n${r.stdout}\n${r.stderr}`);
  // The four numbers the brief requires.
  assert.match(r.stdout, /SUMMARY/);
  assert.match(r.stdout, /nodes\s+1/);
  assert.match(r.stdout, /uniforms\s+0/);
  assert.match(r.stdout, /depth\s+1/);
  assert.match(r.stdout, /PASS/);
});

test('validate: an error diagnostic exits 1 and groups by severity', async (t) => {
  const dir = await scratch(t);
  const doc = join(dir, 'broken.shader.json');
  await writeFile(doc, JSON.stringify(makeBrokenDoc(), null, 2), 'utf8');

  const r = await runCli(['validate', doc], { cwd: dir });
  assert.equal(r.code, 1, `expected 1\n${r.stdout}`);
  assert.match(r.stdout, /error/);
  assert.match(r.stdout, /IR004_DANGLING_WIRE/);
  assert.match(r.stdout, /fix:/, 'a diagnostic must print its actionable fix');
  assert.match(r.stdout, /FAIL 1 error/);
});

test('validate --json prints parseable output with the same verdict', async (t) => {
  const dir = await scratch(t);
  const good = join(dir, 'good.shader.json');
  const bad = join(dir, 'bad.shader.json');
  await writeFile(good, JSON.stringify(makeStructuralDoc(), null, 2), 'utf8');
  await writeFile(bad, JSON.stringify(makeBrokenDoc(), null, 2), 'utf8');

  const ok = await runCli(['validate', good, '--json'], { cwd: dir });
  assert.equal(ok.code, 0);
  const okJson = JSON.parse(ok.stdout) as {
    ok: boolean;
    summary: { nodes: number; uniforms: number; depth: number };
    counts: { error: number };
  };
  assert.equal(okJson.ok, true);
  assert.equal(okJson.summary.nodes, 1);
  assert.equal(okJson.counts.error, 0);

  const fail = await runCli(['validate', bad, '--json'], { cwd: dir });
  assert.equal(fail.code, 1);
  const failJson = JSON.parse(fail.stdout) as { ok: boolean; diagnostics: { code: string; severity: string }[] };
  assert.equal(failJson.ok, false);
  assert.ok(failJson.diagnostics.some((d) => d.code === 'IR004_DANGLING_WIRE' && d.severity === 'error'));
});

test('validate: reads the document from stdin when the path is `-`', async (t) => {
  const dir = await scratch(t);
  const r = await runCli(['validate', '-', '--json'], { cwd: dir, stdin: JSON.stringify(makeStructuralDoc()) });
  assert.equal(r.code, 0, r.stderr);
  const parsed = JSON.parse(r.stdout) as { file: string; ok: boolean };
  assert.equal(parsed.file, '<stdin>');
  assert.equal(parsed.ok, true);
});

test('validate: bad input is a usage error, not a validation failure', async (t) => {
  const dir = await scratch(t);
  const missing = await runCli(['validate', join(dir, 'nope.json')], { cwd: dir });
  assert.equal(missing.code, 2, 'a missing file is the caller\'s mistake');
  assert.match(missing.stderr, /no such file/);

  const notJson = join(dir, 'x.json');
  await writeFile(notJson, '{ not json', 'utf8');
  const broken = await runCli(['validate', notJson], { cwd: dir });
  assert.equal(broken.code, 2);
  assert.match(broken.stderr, /not valid JSON/);

  const array = join(dir, 'arr.json');
  await writeFile(array, '[]', 'utf8');
  assert.equal((await runCli(['validate', array], { cwd: dir })).code, 2);

  const noPath = await runCli(['validate'], { cwd: dir });
  assert.equal(noPath.code, 2);
  assert.match(noPath.stderr, /needs a document path/);

  const twoPaths = await runCli(['validate', 'a.json', 'b.json'], { cwd: dir });
  assert.equal(twoPaths.code, 2);
  assert.match(twoPaths.stderr, /takes one document/);
});

test('validate: works from an unrelated cwd with no node_modules anywhere', async (t) => {
  // The temp dir is outside the repo, so nothing resolves through the workspace.
  const dir = await scratch(t);
  const doc = join(dir, 'doc.shader.json');
  await writeFile(doc, JSON.stringify(makeStructuralDoc(), null, 2), 'utf8');
  const r = await runCli(['validate', doc], { cwd: dir });
  assert.equal(r.code, 0, r.stderr);
  assert.match(r.stdout, /PASS/);
});

test('validate: reports missing catalog/targets as a note, not a crash', async (t) => {
  // A *copy* of the CLI under a workspace root that has no `packages/ir`: this is the
  // "another agent has not landed the catalog yet" state, which must degrade to a
  // structural check with a loud note rather than crashing at import time.
  const dir = await scratch(t);
  const copy = join(dir, 'apps/cli');
  await cp(CLI_DIR, copy, { recursive: true, filter: (src) => !src.includes('node_modules') });
  const doc = await writeJson(dir, 'doc.shader.json', makeStructuralDoc());

  const run = (args: string[]): Promise<Run> => capture(join(copy, 'src/cli.ts'), args, { cwd: dir });

  const json = await run(['validate', doc, '--json']);
  assert.equal(json.code, 0, `a structural-only validate must still succeed\n${json.stderr}`);
  const parsed = JSON.parse(json.stdout) as { catalog: { complete: boolean; kinds: number } };
  assert.equal(parsed.catalog.complete, false);
  assert.equal(parsed.catalog.kinds, 1, 'the fallback catalog knows exactly `output`');

  const human = await run(['validate', doc]);
  assert.match(human.stderr, /catalog\/targets are not built yet — run `npm run build` in packages\/ir/);

  // `emit` genuinely cannot work without the emitters, and must say why.
  const emit = await run(['emit', doc, '--target', 'glsl', '--out', join(dir, 'o')]);
  assert.equal(emit.code, 1, 'a missing catalog is not a usage error');
  assert.match(emit.stderr, /cannot emit without the node catalog and the target emitters/);
});

/* -------------------------------------------------------------------------- */
/* emit                                                                        */
/* -------------------------------------------------------------------------- */

test('emit: writes the bundle layout', async (t) => {
  const dir = await scratch(t);
  const doc = await writeJson(dir, 'doc.shader.json', {
    ...makeDoc({ title: 'Bundle Layout Test' }),
    targets: ['wgsl'],
  });
  const out = join(dir, 'bundle');

  const r = await runCli(['emit', doc, '--target', 'all', '--out', out], { cwd: dir });

  // `all` asks for six targets. Which of them have an emitter depends on whether the catalog
  // and the target modules are loadable in this process, so the exit code is *not* asserted
  // here — the `(requires catalog)` tests below cover "all six work" and skip themselves when
  // they cannot. What this test owns is the **bundle layout**, which must be identical either
  // way: that is the contract `emit` makes to whatever consumes its output.
  const slugDir = join(out, 'bundle-layout-test');
  for (const f of ['shader.wgsl', 'params.json', 'README.md']) {
    assert.ok(await exists(join(slugDir, f)), `expected ${f} in the bundle\n${r.stdout}`);
  }
  if (REAL_IR) assert.equal(r.code, 0, `with the real catalog all six must emit\n${r.stdout}`);

  const readme = await readFile(join(slugDir, 'README.md'), 'utf8');
  assert.match(readme, /^# Bundle Layout Test/m);
  assert.match(readme, /a soft procedural noise field/, 'README carries spec.intent');
  assert.match(readme, /continuous/, 'README lists spec.features');
  assert.match(readme, /## How to build each target/);
  assert.match(readme, /## Parameters/);

  const params = JSON.parse(await readFile(join(slugDir, 'params.json'), 'utf8')) as Record<string, unknown>[];
  assert.ok(Array.isArray(params));
  for (const key of ['name', 'type', 'value', 'min', 'max', 'step', 'label', 'group']) {
    assert.ok(key in params[0]!, `params.json entry is missing \`${key}\``);
  }
});

test('emit --stdout writes the source to stdout and nothing to disk', async (t) => {
  const dir = await scratch(t);
  const doc = await writeJson(dir, 'doc.shader.json', makeDoc({ title: 'Stdout Test' }));
  const r = await runCli(['emit', doc, '--target', 'wgsl', '--stdout'], { cwd: dir });
  assert.equal(r.code, 0, r.stderr);
  // Assert it is *a WGSL module*, not the text of any particular emitter: with the real
  // targets loaded this is the real lowering, and a stub-specific string would pin the test
  // to whichever emitter happened to be registered.
  assert.match(r.stdout, /@fragment|fn \w+\s*\(/, `expected WGSL on stdout, got:\n${r.stdout.slice(0, 300)}`);
  assert.equal(await exists(join(dir, 'stdout-test')), false);

  // `--stdout` with `all` is a usage error: six sources on one stream is not a thing.
  const multi = await runCli(['emit', doc, '--target', 'all', '--stdout'], { cwd: dir });
  assert.equal(multi.code, 2);
  assert.match(multi.stderr, /cannot be combined/);
});

test('emit: never overwrites a differing file without --force', async (t) => {
  const dir = await scratch(t);
  const doc = await writeJson(dir, 'doc.shader.json', makeDoc({ title: 'Overwrite Test' }));
  const out = join(dir, 'out');
  const target = join(out, 'overwrite-test/shader.wgsl');

  const first = await runCli(['emit', doc, '--target', 'wgsl', '--out', out], { cwd: dir });
  assert.equal(first.code, 0, first.stderr);
  const generated = await readFile(target, 'utf8');

  // A human edits the generated shader.
  await writeFile(target, '// hand-edited, do not lose\n', 'utf8');

  const second = await runCli(['emit', doc, '--target', 'wgsl', '--out', out], { cwd: dir });
  assert.equal(second.code, 0, second.stderr);
  assert.match(second.stdout, /~/, 'a differing file must be marked with ~');
  assert.match(second.stdout, /\.new/);
  assert.equal(await readFile(target, 'utf8'), '// hand-edited, do not lose\n', 'the hand edit was destroyed');
  assert.equal(await readFile(`${target}.new`, 'utf8'), generated, 'the new content must be beside it');

  // --force overwrites in place.
  const forced = await runCli(['emit', doc, '--target', 'wgsl', '--out', out, '--force'], { cwd: dir });
  assert.equal(forced.code, 0, forced.stderr);
  assert.equal(await readFile(target, 'utf8'), generated);

  // Re-emitting identical content is reported as unchanged, and writes nothing new.
  const third = await runCli(['emit', doc, '--target', 'wgsl', '--out', out], { cwd: dir });
  assert.equal(third.code, 0);
  assert.match(third.stdout, /unchanged/);
});

test('emit: usage errors', async (t) => {
  const dir = await scratch(t);
  const doc = await writeJson(dir, 'doc.shader.json', makeDoc());

  const noTarget = await runCli(['emit', doc, '--out', join(dir, 'o')], { cwd: dir });
  assert.equal(noTarget.code, 2);
  assert.match(noTarget.stderr, /--target/);

  const badTarget = await runCli(['emit', doc, '--target', 'cobol', '--out', join(dir, 'o')], { cwd: dir });
  assert.equal(badTarget.code, 2);
  assert.match(badTarget.stderr, /unknown target/);

  const noOut = await runCli(['emit', doc, '--target', 'wgsl'], { cwd: dir });
  assert.equal(noOut.code, 2);
  assert.match(noOut.stderr, /--out/);
});

test('emit --target glsl (requires catalog)', async (t) => {
  if (!REAL_IR) {
    t.skip('packages/ir/src/catalog and .../targets have not landed yet');
    return;
  }
  const dir = await scratch(t);
  const doc = await writeJson(dir, 'doc.shader.json', { ...makeDoc({ title: 'Real Glsl' }), targets: ['glsl'] });
  const out = join(dir, 'out');
  const r = await runCli(['emit', doc, '--target', 'glsl', '--out', out], { cwd: dir });
  assert.equal(r.code, 0, `${r.stdout}\n${r.stderr}`);
  assert.ok(await exists(join(out, 'real-glsl/shader.frag')));
});

test('emit --target all (requires catalog): every target lands in the documented layout', async (t) => {
  if (!REAL_IR || !(await catalogCovers(['uv', 'fbm', 'output']))) {
    t.skip('the real catalog/targets have not landed yet (or do not cover this fixture)');
    return;
  }
  const dir = await scratch(t);
  const doc = await writeJson(dir, 'doc.shader.json', makeDoc({ title: 'Full Bundle' }));
  const out = join(dir, 'out');
  const r = await runCli(['emit', doc, '--target', 'all', '--out', out], { cwd: dir });
  // The contract (§7) is explicit: `--target all` must not produce an error diagnostic.
  assert.equal(r.code, 0, `${r.stdout}\n${r.stderr}`);

  const slugDir = join(out, 'full-bundle');
  for (const f of [
    'shader.frag',
    'shader.wgsl',
    'shader.hlsl',
    'shader.metal',
    'Shader.shader',
    'ue/CustomNode.hlsl',
    'params.json',
    'README.md',
  ]) {
    assert.ok(await exists(join(slugDir, f)), `missing ${f}\n${r.stdout}`);
  }
});

/* -------------------------------------------------------------------------- */
/* params                                                                      */
/* -------------------------------------------------------------------------- */

test('params: lists, overrides through the patcher, and round-trips the hash', async (t) => {
  if (!(await catalogCovers(['uv', 'fbm', 'output']))) {
    t.skip('the real catalog has not landed yet');
    return;
  }
  const dir = await scratch(t);
  const before = { ...makeDoc({ title: 'Params Test' }), targets: ['wgsl'] };
  const doc = await writeJson(dir, 'doc.shader.json', before);

  const listed = await runCli(['params', doc, '--json'], { cwd: dir });
  assert.equal(listed.code, 0, listed.stderr);
  const listing = JSON.parse(listed.stdout) as { hash: string; uniforms: { name: string }[] };
  assert.equal(listing.hash, before.hash ?? null);

  // `--set` must go through the patcher: the hash has to move.
  const tuned = join(dir, 'tuned.json');
  const set = await runCli(['params', doc, '--set', 'u_scale=12', '--out', tuned], { cwd: dir });
  assert.equal(set.code, 0, set.stderr);
  assert.match(set.stdout, /applied 1 override through the patcher/);

  const after = JSON.parse(await readFile(tuned, 'utf8')) as {
    hash: string;
    provenance: { steps: { summary: string; actor: string }[] };
    graph: { nodes: { id: string; params?: Record<string, { value: unknown }> }[] };
  };
  assert.notEqual(after.hash, before.hash, 'the hash must change when a uniform value changes');
  assert.equal(after.provenance.steps.at(-1)!.summary, 'params --set u_scale');
  assert.equal(after.provenance.steps.at(-1)!.actor, 'human');
  assert.equal(after.graph.nodes.find((n) => n.id === 'fbm_1')!.params!.scale!.value, 12);
});

test('params: rejects an unknown uniform and an out-of-range value (exit 2)', async (t) => {
  if (!(await catalogCovers(['uv', 'fbm', 'output']))) {
    t.skip('the real catalog has not landed yet');
    return;
  }
  const dir = await scratch(t);
  const doc = await writeJson(dir, 'doc.shader.json', makeDoc());

  const unknown = await runCli(['params', doc, '--set', 'u_nope=1'], { cwd: dir });
  assert.equal(unknown.code, 2);
  assert.match(unknown.stderr, /no uniform named/);
  assert.match(unknown.stderr, /Available:/, 'the error must say what *is* available');

  const range = await runCli(['params', doc, '--set', 'u_scale=9999'], { cwd: dir });
  assert.equal(range.code, 2);
  assert.match(range.stderr, /above its maximum/);

  const malformed = await runCli(['params', doc, '--set', 'u_scale'], { cwd: dir });
  assert.equal(malformed.code, 2);
  assert.match(malformed.stderr, /expects <name>=<value>/);

  const badType = await runCli(['params', doc, '--set', 'u_scale=hello'], { cwd: dir });
  assert.equal(badType.code, 2);
  assert.match(badType.stderr, /not a number/);
});

test('params: a document with no uniforms lists nothing and still exits 0', async (t) => {
  const dir = await scratch(t);
  const doc = await writeJson(dir, 'doc.shader.json', makeStructuralDoc());
  const r = await runCli(['params', doc], { cwd: dir });
  assert.equal(r.code, 0, r.stderr);
  assert.match(r.stdout, /no uniforms/);
});

/* -------------------------------------------------------------------------- */
/* tune                                                                        */
/* -------------------------------------------------------------------------- */

test('tune: a deterministic Cartesian sweep with value-derived variant ids', async (t) => {
  if (!(await catalogCovers(['uv', 'fbm', 'output']))) {
    t.skip('the real catalog has not landed yet');
    return;
  }
  const dir = await scratch(t);
  const doc = await writeJson(dir, 'doc.shader.json', makeDoc({ title: 'Tune Test' }));
  const out = join(dir, 'sweep');

  const r = await runCli(
    ['tune', doc, '--param', 'u_scale=0.5..8', '--param', 'u_octaves=2..6', '--steps', '3', '--out', out],
    { cwd: dir },
  );
  assert.equal(r.code, 0, `${r.stdout}\n${r.stderr}`);
  assert.match(r.stdout, /variants 9/);

  const sweep = JSON.parse(await readFile(join(out, 'sweep.json'), 'utf8')) as {
    baseHash: string;
    steps: number;
    cap: number;
    params: { name: string; lo: number; hi: number }[];
    variants: { id: string; index: number; dir: string; values: Record<string, number>; docHash: string }[];
  };
  assert.equal(sweep.steps, 3);
  assert.equal(sweep.cap, 256);
  assert.equal(sweep.variants.length, 9);
  assert.deepEqual(sweep.params.map((p) => p.name), ['u_scale', 'u_octaves']);

  // Every variant id is derived from its own values — not from the index.
  const scaleVals = [...new Set(sweep.variants.map((v) => v.values['u_scale']!))].sort((a, b) => a - b);
  assert.deepEqual(scaleVals, [0.5, 4.25, 8], 'even sampling from lo to hi, endpoints included');
  for (const v of sweep.variants) {
    // The id lists the axes **sorted by name**, which is what makes it stable and diffable
    // across runs; `u_octaves` sorts before `u_scale`.
    assert.equal(
      v.id,
      `u_octaves=${v.values['u_octaves']!.toFixed(4)}__u_scale=${v.values['u_scale']!.toFixed(4)}`,
    );
    assert.equal(v.dir, `sweep-${v.index}`);
  }
  const ids = sweep.variants.map((v) => v.id);
  assert.equal(new Set(ids).size, ids.length, 'variant ids must be unique');
  assert.equal(new Set(sweep.variants.map((v) => v.docHash)).size, 9, 'each grid point is a different document');

  // No shaders without --emit.
  assert.equal(await exists(join(out, 'sweep-1')), false);

  // ... and a second run of the same command produces the same sweep.json byte for byte.
  const again = await runCli(
    ['tune', doc, '--param', 'u_scale=0.5..8', '--param', 'u_octaves=2..6', '--steps', '3', '--out', out],
    { cwd: dir },
  );
  assert.equal(again.code, 0, again.stderr);
  assert.match(again.stdout, /unchanged/, 'an identical sweep must not be reported as a divergence');
});

test('tune --emit writes one directory per variant and records diagnostics', async (t) => {
  if (!(await catalogCovers(['uv', 'fbm', 'output']))) {
    t.skip('the real catalog has not landed yet');
    return;
  }
  const dir = await scratch(t);
  const doc = await writeJson(dir, 'doc.shader.json', { ...makeDoc({ title: 'Tune Emit' }), targets: ['wgsl'] });
  const out = join(dir, 'sweep');

  const r = await runCli(
    ['tune', doc, '--param', 'u_scale=1..4', '--steps', '3', '--out', out, '--emit', '--target', 'wgsl'],
    { cwd: dir },
  );
  assert.equal(r.code, 0, `${r.stdout}\n${r.stderr}`);
  const sweep = JSON.parse(await readFile(join(out, 'sweep.json'), 'utf8')) as {
    variants: { index: number; id: string; dir: string; files: string[]; ok: boolean }[];
  };
  assert.equal(sweep.variants.length, 3);
  for (const v of sweep.variants) {
    assert.ok(await exists(join(out, v.dir)), `missing ${v.dir}`);
    assert.equal(v.files.length, 1);
    assert.match(v.files[0]!, new RegExp(`^${v.dir}/shader\\.wgsl$`));
  }
});

test('tune: refuses a sweep above the 256-variant cap, with the count', async (t) => {
  if (!(await catalogCovers(['uv', 'fbm', 'output']))) {
    t.skip('the real catalog has not landed yet');
    return;
  }
  const dir = await scratch(t);
  const doc = await writeJson(dir, 'doc.shader.json', makeDoc());
  const r = await runCli(
    ['tune', doc, '--param', 'u_scale=1..4', '--param', 'u_octaves=1..8', '--param', 'u_gain=0..1', '--steps', '7', '--out', join(dir, 's')],
    { cwd: dir },
  );
  assert.equal(r.code, 2, r.stdout);
  assert.match(r.stderr, /343 variants/, 'the error must state the actual count');
  assert.match(r.stderr, /256/);
  assert.equal(await exists(join(dir, 's/sweep.json')), false, 'nothing may be written when the sweep is refused');
});

test('tune: usage errors', async (t) => {
  if (!(await catalogCovers(['uv', 'fbm', 'output']))) {
    t.skip('the real catalog has not landed yet');
    return;
  }
  const dir = await scratch(t);
  const doc = await writeJson(dir, 'doc.shader.json', makeDoc());

  const noParams = await runCli(['tune', doc, '--steps', '3'], { cwd: dir });
  assert.equal(noParams.code, 2);
  assert.match(noParams.stderr, /at least one --param/);

  const badSteps = await runCli(['tune', doc, '--param', 'u_scale=1..4', '--steps', '0'], { cwd: dir });
  assert.equal(badSteps.code, 2);
  assert.match(badSteps.stderr, /positive integer/);

  const inverted = await runCli(['tune', doc, '--param', 'u_scale=8..1'], { cwd: dir });
  assert.equal(inverted.code, 2);
  assert.match(inverted.stderr, /inverted/);

  const unknown = await runCli(['tune', doc, '--param', 'u_nope=1..4'], { cwd: dir });
  assert.equal(unknown.code, 2);
  assert.match(unknown.stderr, /no uniform named/);
});

test('tune: repeated --param for one uniform builds an explicit grid (no Cartesian blowup)', async (t) => {
  if (!(await catalogCovers(['uv', 'fbm', 'output']))) {
    t.skip('the real catalog has not landed yet');
    return;
  }
  const dir = await scratch(t);
  const doc = await writeJson(dir, 'doc.shader.json', makeDoc({ title: 'Explicit Grid' }));
  const out = join(dir, 'sweep');
  const r = await runCli(
    ['tune', doc, '--param', 'u_scale=0.5', '--param', 'u_scale=2', '--param', 'u_scale=8', '--steps', '4', '--out', out],
    { cwd: dir },
  );
  assert.equal(r.code, 0, r.stderr);
  const sweep = JSON.parse(await readFile(join(out, 'sweep.json'), 'utf8')) as {
    variants: { values: Record<string, number> }[];
  };
  assert.deepEqual(
    sweep.variants.map((v) => v.values['u_scale']),
    [0.5, 2, 8],
    'three explicit values must give three variants, not 4^3',
  );
});

/* -------------------------------------------------------------------------- */
/* cwd independence                                                            */
/* -------------------------------------------------------------------------- */

test('every command works from an unrelated cwd (absolute paths in, relative out)', async (t) => {
  if (!(await catalogCovers(['uv', 'fbm', 'output']))) {
    t.skip('the real catalog has not landed yet');
    return;
  }
  const home = await scratch(t);
  const elsewhere = join(home, 'deep/nested/place');
  await mkdir(elsewhere, { recursive: true });
  const doc = await writeJson(home, 'doc.shader.json', makeDoc({ title: 'Cwd Test' }));

  // Relative input path, resolved against the *cwd*, which is `elsewhere`.
  await writeFile(join(elsewhere, 'local.shader.json'), JSON.stringify(makeDoc()), 'utf8');
  const local = await runCli(['validate', 'local.shader.json'], { cwd: elsewhere });
  assert.equal(local.code, 0, local.stderr);

  // Relative --out is resolved against the cwd too.
  const emit = await runCli(['emit', doc, '--target', 'wgsl', '--out', 'rel-out'], { cwd: elsewhere });
  assert.equal(emit.code, 0, emit.stderr);
  assert.ok(await exists(join(elsewhere, 'rel-out/cwd-test/shader.wgsl')), 'relative --out must land under the cwd');

  // And a `-` document piped in from anywhere.
  const piped = await runCli(['validate', '-'], { cwd: elsewhere, stdin: JSON.stringify(makeDoc()) });
  assert.equal(piped.code, 0, piped.stderr);
});

/* -------------------------------------------------------------------------- */
/* Determinism                                                                 */
/* -------------------------------------------------------------------------- */

test('the same document emits byte-identical output on every run', async (t) => {
  const dir = await scratch(t);
  const doc = await writeJson(dir, 'doc.shader.json', makeDoc({ title: 'Determinism' }));
  const one = join(dir, 'one');
  const two = join(dir, 'two');
  await runCli(['emit', doc, '--target', 'wgsl', '--out', one], { cwd: dir });
  await runCli(['emit', doc, '--target', 'wgsl', '--out', two], { cwd: dir });
  assert.equal(
    await readFile(join(one, 'determinism/shader.wgsl'), 'utf8'),
    await readFile(join(two, 'determinism/shader.wgsl'), 'utf8'),
  );
});

/* -------------------------------------------------------------------------- */
/* author / refine — the agent loop, driven from the command line               */
/* -------------------------------------------------------------------------- */

type AuthorJson = {
  stopReason: string;
  clean: boolean;
  modelCalls: number;
  remaining: { code: string }[];
  trace: { stage: string; round: number; summary: string }[];
  doc: { graph: { nodes: { id: string; kind: string }[] }; targets: string[] };
};

/**
 * `--provider stub` is the whole reason these tests can exist: the loop is exercised from the
 * command line, with no network and no API key, and produces a real six-target bundle. If the
 * stub ever stops drafting a working document this suite fails, which is exactly the signal
 * wanted — the stub is a fixture the documentation points users at, not a placeholder.
 */
test('author drafts a document and a full bundle using the offline stub provider (requires catalog)', async (t) => {
  if (!REAL_IR) return t.skip('catalog/targets not present');

  const dir = await scratch(t);
  const out = join(dir, 'bundle');
  const docOut = join(dir, 'authored.shader.json');

  const run = await runCli(
    ['author', 'molten metal, slow flow', '--provider', 'stub', '--doc-out', docOut, '--out', out, '--json'],
    { cwd: dir },
  );

  assert.equal(run.code, 0, `author failed:\n${run.stderr}\n${run.stdout}`);
  const result = JSON.parse(run.stdout) as AuthorJson;

  assert.equal(result.clean, true, 'the stub must draft a document that emits cleanly');
  assert.equal(result.stopReason, 'clean');
  assert.deepEqual(result.remaining, []);
  // Two calls: interpret + one draft. A higher number would mean the loop asked the model to
  // fix something the deterministic repairer can see for itself.
  assert.ok(result.modelCalls <= 3, `expected a cheap draft, got ${result.modelCalls} model calls`);
  assert.ok(result.doc.graph.nodes.length >= 4, 'a drafted graph should have real content');
  assert.deepEqual(
    result.doc.targets.slice().sort(),
    ['glsl', 'hlsl', 'msl', 'shaderlab', 'ue', 'wgsl'],
    'a drafted document must be held to all six targets',
  );

  // The document and every target's source must exist on disk, not merely be reported.
  const written = JSON.parse(await readFile(docOut, 'utf8')) as { graph: { nodes: unknown[] } };
  assert.ok(written.graph.nodes.length >= 4);
  for (const file of ['shader/shader.frag', 'shader/shader.wgsl', 'shader/shader.hlsl', 'shader/shader.metal']) {
    assert.ok(await exists(join(out, file)), `${file} was reported but not written`);
  }

  // Determinism: the same stub must give the same content hash, or the project's cache keys
  // and golden images would be meaningless.
  const again = await runCli(['author', 'molten metal, slow flow', '--provider', 'stub', '--json'], { cwd: dir });
  const second = JSON.parse(again.stdout) as AuthorJson & { doc: { hash?: string } };
  const first = JSON.parse(run.stdout) as AuthorJson & { doc: { hash?: string } };
  assert.equal(second.doc.hash, first.doc.hash, 'the loop must be deterministic for a stub provider');
});

test('author with no provider reports what is missing instead of inventing a graph', async (t) => {
  const dir = await scratch(t);
  const run = await runCli(['author', 'anything at all', '--provider', 'none', '--json'], { cwd: dir });

  // Exit 1 is the contract for "diagnostics outstanding": the command did its job by saying so.
  assert.equal(run.code, 1);
  const result = JSON.parse(run.stdout) as AuthorJson;
  assert.equal(result.clean, false);
  assert.equal(result.modelCalls, 0);
  assert.ok(
    result.remaining.some((d) => d.code === 'AG001_NO_PROVIDER'),
    `expected AG001_NO_PROVIDER, got ${JSON.stringify(result.remaining)}`,
  );
  // The human-readable form must explain the fix, not just the fault.
  assert.match(run.stderr, /no API key|provider stub|deterministic repair/);
});

test('refine without a provider performs only deterministic repair', async (t) => {
  if (!REAL_IR) return t.skip('catalog/targets not present');

  const dir = await scratch(t);
  // Build a valid document first, so the test does not depend on a fixture staying valid.
  const authored = join(dir, 'base.shader.json');
  const made = await runCli(
    ['author', 'molten metal', '--provider', 'stub', '--doc-out', authored, '--json'],
    { cwd: dir },
  );
  assert.equal(made.code, 0, `setup author failed:\n${made.stderr}`);

  const out = join(dir, 'refined');
  const run = await runCli(['refine', authored, 'repair diagnostics', '--out', out, '--json', '--force'], {
    cwd: dir,
  });
  assert.equal(run.code, 0, `refine failed:\n${run.stderr}\n${run.stdout}`);

  const result = JSON.parse(run.stdout) as AuthorJson;
  assert.equal(result.clean, true);
  assert.equal(
    result.modelCalls,
    0,
    'deterministic repair without a provider must not issue model calls',
  );
  assert.ok(await exists(join(out, 'shader/shader.frag')), 'refine --out must write a bundle');
});

test('refine refuses a missing instruction rather than silently doing nothing', async (t) => {
  const dir = await scratch(t);
  const doc = await writeJson(dir, 'doc.shader.json', makeStructuralDoc());
  const run = await runCli(['refine', doc, '--provider', 'none'], { cwd: dir });
  assert.equal(run.code, 2, 'a missing instruction is a usage error');
  assert.match(run.stderr, /needs an instruction/);
});

test('author rejects an unresolvable provider configuration with an actionable message', async (t) => {
  const dir = await scratch(t);
  const run = await runCli(['author', 'x', '--provider', 'http', '--model', 'gpt-4o-mini', '--json'], {
    cwd: dir,
    // Explicitly no key in the environment, whatever the developer's shell happens to hold.
    env: { AI4SHADER_API_KEY: '', OPENAI_API_KEY: '' },
  });
  // No key means no provider, which means the loop reports AG001 rather than attempting a
  // request it cannot sign. The point is that it never pretends to have called a model.
  const result = JSON.parse(run.stdout) as AuthorJson;
  assert.equal(result.modelCalls, 0);
  assert.ok(result.remaining.length > 0);
  assert.match(run.stderr, /API key/);
});
