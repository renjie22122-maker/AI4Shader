/**
 * AI4Shader — test runner
 * =======================
 *
 * `npm test` in a normal checkout, and the same thing without npm here. Wraps
 * `node --test` with the two flags this project actually needs, and — more importantly —
 * decides them from the running Node version instead of hardcoding them, because the
 * flags changed under us:
 *
 *   Node 22.6 – 23.5   needs `--experimental-strip-types`
 *   Node >= 23.6       strips types by default; passing the flag is merely redundant
 *   Node >= 22.7       needs `--experimental-transform-types` for `enum`/`namespace`
 *
 * A hardcoded flag list that is wrong for the installed Node produces "bad option", which
 * reads like a broken project rather than a version mismatch.
 *
 *   node scripts/run-tests.mjs [--isolated] [pkgDir ...]
 *
 * `--isolated` uses the default process-per-file runner. The default here is
 * `--test-isolation=none` because a confined sandbox can deny the runner's piped child
 * spawn; a normal machine should prefer `--isolated`, which is why it exists.
 */

import { spawn, spawnSync } from 'node:child_process';
import { readdirSync, existsSync, mkdtempSync, openSync, closeSync, readFileSync, rmSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { sourceFingerprint, checkDocs } from './check-docs.mjs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const argv = process.argv.slice(2);
const isolated = argv.includes('--isolated');
const explicit = argv.filter((a) => !a.startsWith('--'));

const [major, minor] = process.versions.node.split('.').map(Number);
const flags = [];
if (major < 23 || (major === 23 && minor < 6)) flags.push('--experimental-strip-types');
if (major > 22 || (major === 22 && minor >= 7)) flags.push('--experimental-transform-types');
if (!isolated) flags.push('--test-isolation=none');

/** Directories that look like testable workspaces. */
function findTestDirs() {
  if (explicit.length) return explicit.map((p) => resolve(repoRoot, p));
  const out = [];
  for (const group of ['packages', 'apps']) {
    const base = join(repoRoot, group);
    if (!existsSync(base)) continue;
    for (const name of readdirSync(base)) {
      const dir = join(base, name);
      if (existsSync(join(dir, 'test'))) out.push(dir);
    }
  }
  return out;
}

const dirs = findTestDirs();
if (!dirs.length) {
  process.stderr.write('no workspace with a test/ directory found\n');
  process.exit(2);
}

process.stdout.write(`node ${process.versions.node}  flags: ${flags.join(' ') || '(none)'}\n\n`);

/**
 * Can this process spawn a child with **piped** stdio?
 *
 * Reported for diagnostics only — it no longer gates any suite. It used to hold back the CLI
 * end-to-end suite, on the theory that asserting on a subprocess's stdout requires a pipe. It
 * does not: redirecting the child's stdout/stderr to temporary files and reading them back is
 * equivalent for every assertion that suite makes, needs no pipe, and is what
 * `apps/cli/test/cli.test.ts` now does. The result was 31 tests that had never once executed
 * here, all of them now passing — a reminder that "cannot run in this environment" deserves a
 * check before it becomes a standing excuse.
 *
 * The probe stays because knowing the mode matters when a *new* suite fails: `EPERM` on spawn
 * with pipes is an environment boundary, while the same error in a file-capture harness is a
 * bug.
 */
async function probeSpawn() {
  if (!existsSync(process.execPath)) return { ok: false, why: 'no node binary' };
  return new Promise((res) => {
    let child;
    try {
      child = spawn(process.execPath, ['-e', 'process.exit(0)'], { stdio: ['ignore', 'pipe', 'pipe'] });
    } catch (err) {
      // A denied spawn can throw synchronously rather than emitting `error`, depending
      // on the platform and the policy — handle both or the probe itself becomes the crash.
      res({ ok: false, why: `${err.code ?? 'error'}: ${err.message}` });
      return;
    }
    child.on('error', (err) => res({ ok: false, why: `${err.code ?? 'error'}: ${err.message}` }));
    child.on('close', (code) => res({ ok: code === 0, why: code === 0 ? '' : `exit ${code}` }));
  });
}

/** Files whose failures are attributable to the environment, not to the code. */
const NEEDS_SUBPROCESS = [];

const spawnProbe = await probeSpawn();
if (!spawnProbe.ok) {
  process.stdout.write(
    `NOTE  spawning with *piped* stdio is not permitted here (${spawnProbe.why}).\n` +
      `      No suite is held back for it: the CLI end-to-end tests capture output through\n` +
      `      temporary files, which needs no pipe. A spawn EPERM inside a file-capture\n` +
      `      harness is therefore a bug, not an environment limit.\n\n`,
  );
}

let failed = 0;
let skippedByEnv = 0;
const counts = {};
const fingerprint = sourceFingerprint();
async function capture(command, args, cwd) {
  const temp = mkdtempSync(join(tmpdir(), 'ai4shader-tests-'));
  const file = join(temp, 'output.txt');
  const fd = openSync(file, 'w');
  try {
    const code = await new Promise(resolve => {
      const child = spawn(command, args, { cwd, stdio: ['ignore', fd, fd], env: { ...process.env, PYTHONIOENCODING: 'utf-8' } });
      child.on('close', c => resolve(c ?? 1));
      child.on('error', error => { console.error(error.message); resolve(1); });
    });
    const output = readFileSync(file, 'utf8');
    process.stdout.write(output);
    return { code, output };
  } finally { closeSync(fd); rmSync(temp, { recursive: true, force: true }); }
}
for (const dir of dirs) {
  const testDir = join(dir, 'test');
  const allFiles = readdirSync(testDir).filter((f) => f.endsWith('.test.ts'));
  const files = [];
  const held = [];
  for (const f of allFiles) {
    if (!spawnProbe.ok && NEEDS_SUBPROCESS.some((re) => re.test(f))) held.push(f);
    else files.push(f);
  }
  if (held.length) {
    skippedByEnv += held.length;
    process.stdout.write(`SKIP  ${held.join(', ')} — requires subprocess capture\n`);
  }
  if (!files.length) continue;
  const relFiles = files.map((f) => join('test', f));

  // A resolver hook is needed because the sources use `.js` specifiers that must find
  // `.ts` files under type stripping. Workspaces that provide one get it; workspaces
  // without one run bare.
  const hooks = existsSync(join(testDir, 'register-hooks.mjs'))
    ? ['--import', './test/register-hooks.mjs']
    : [];

  const args = ['--test', '--test-reporter=spec', ...flags, ...hooks, ...relFiles];
  process.stdout.write(`== ${dir.replace(repoRoot + '\\', '').replace(repoRoot + '/', '')} ==\n`);

  const { code, output } = await capture(process.execPath, args, dir);
  const match = /^.*?pass (\d+)\s*$/m.exec(output);
  if (!match) failed++;
  else counts[dir.slice(repoRoot.length + 1).replaceAll('\\', '/')] = Number(match[1]);
  if (code !== 0) failed++;
  process.stdout.write('\n');
}

/* -------------------------------------------------------------------------- */
/* Python                                                                      */
/* -------------------------------------------------------------------------- */

/**
 * Run the Python layer's suite too, when it and a Python interpreter are both present.
 *
 * It is part of the same gate on purpose. `python/` owns the jobs that cannot be done in Node
 * — wrapping real shader compilers, and the image metrics the optimizer searches against — and
 * a suite that is only ever run by hand is a suite that is only ever green on the day it was
 * written. The most valuable test in it compiles the emitters' real output with
 * `glslangValidator`, which is a cross-language assertion: it can only fail when the Node side
 * regresses, and nothing on the Node side would notice.
 *
 * A missing interpreter is reported, not treated as a failure: this repository's Node half has
 * no Python dependency, and failing the gate for an absent optional tool would be the same
 * mistake as reporting a missing compiler as a broken shader.
 */
const pythonDir = join(repoRoot, 'python');
if (existsSync(join(pythonDir, 'tests'))) {
  const candidates = process.platform === 'win32' ? ['python', 'py'] : ['python3', 'python'];
  const runner = candidates.find((c) => {
    try {
      return spawnSync(c, ['-c', 'import pytest'], { stdio: 'ignore' }).status === 0;
    } catch {
      return false;
    }
  });

  if (!runner) {
    process.stdout.write(
      `SKIP  python/tests — no interpreter with pytest on PATH (tried: ${candidates.join(', ')})\n` +
        `      The Python layer owns the real-compiler validation and the image metrics; run it\n` +
        `      with \`cd python && python -m pytest tests\` wherever it is available.\n\n`,
    );
    skippedByEnv++;
  } else {
    process.stdout.write(`== python (${runner} -m pytest) ==\n`);
    const { code, output } = await capture(runner, ['-m', 'pytest', 'tests', '-q', '--no-header'], pythonDir);
    const match = /(\d+) passed/.exec(output);
    if (!match) failed++;
    else counts.python = Number(match[1]);
    if (code !== 0) failed++;
    process.stdout.write('\n');
  }
}

if (failed) {
  process.stderr.write(`${failed} workspace(s) failed\n`);
  process.exit(1);
}
process.stdout.write(
  `all runnable suites passed${skippedByEnv ? ` (${skippedByEnv} held back by the environment, see NOTE above)` : ''}\n`,
);
if (!explicit.length && !skippedByEnv) {
  if (sourceFingerprint() !== fingerprint) throw new Error('Sources changed during tests; rerun before recording evidence.');
  const path = join(repoRoot, 'dist/verification/test-counts.json');
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify({ fingerprint, counts }, null, 2) + '\n', 'utf8');
  checkDocs();
} else console.log('Documentation count gate requires a complete run including Python; no new evidence recorded.');
