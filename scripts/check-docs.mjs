/** Compare current claims with a completed test run, never count test() text. */
import { readFileSync, readdirSync, existsSync, mkdtempSync, openSync, closeSync, rmSync } from 'node:fs';
import { resolve, join } from 'node:path';
import { tmpdir } from 'node:os';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';

export const root = fileURLToPath(new URL('../', import.meta.url));
export function sourceFingerprint() {
  const hash = createHash('sha256');
  function walk(dir) {
    for (const entry of readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
      if (['node_modules', 'dist', '.git', '__pycache__', '.pytest_cache'].includes(entry.name)) continue;
      const path = join(dir, entry.name);
      if (entry.isDirectory()) walk(path);
      else if (/\.(ts|mjs|py|json)$/.test(entry.name)) { hash.update(path.slice(root.length)); hash.update(readFileSync(path)); }
    }
  }
  for (const dir of ['packages', 'apps', 'python', 'scripts']) walk(join(root, dir));
  hash.update(readFileSync(join(root, 'package.json')));
  return hash.digest('hex');
}

export function checkClaims(readme, implementation, counts, modules, imports) {
  const errors = [];
  const total = Object.values(counts).reduce((a, b) => a + b, 0);
  const row = implementation.split('\n').find(l => l.startsWith('| 单元测试 |')) ?? '';
  for (const [name, text] of [['README', readme], ['§4.1', row]]) {
    if (!new RegExp(`\\*\\*${total} 个`).test(text)) errors.push(`${name}: total should be ${total}`);
    for (const [suite, n] of Object.entries(counts)) {
      const label = suite === 'python' ? 'python/' : suite;
      if (!new RegExp('`' + label + '` ' + n + '(?![0-9])').test(text)) errors.push(`${name}: ${label} should be ${n}`);
    }
  }
  for (const [name, text] of [['README', readme], ['IMPLEMENTATION', implementation]]) {
    const moduleClaims = [...text.matchAll(/投影 \*\*(\d+)\*\*|类型投影 \| (\d+)/g)];
    if (!moduleClaims.length || moduleClaims.some(m => Number(m[1] ?? m[2]) !== modules)) errors.push(`${name}: modules should be ${modules}`);
    const claims = [...text.matchAll(/\*\*(\d+)\*\* 条 import|(\d+) 条 import/g)];
    if (!claims.length || claims.some(m => Number(m[1] ?? m[2]) !== imports)) errors.push(`${name}: imports should be ${imports}`);
  }
  return errors;
}

export function checkDocs() {
  const reportFile = join(root, 'dist/verification/test-counts.json');
  if (!existsSync(reportFile)) throw new Error('Run node scripts/run-tests.mjs first; completed test evidence is missing.');
  const report = JSON.parse(readFileSync(reportFile, 'utf8'));
  if (report.fingerprint !== sourceFingerprint()) throw new Error('Test evidence is stale: run the full test suite again.');
  if (Object.keys(report.counts).sort().join() !== ['apps/cli', 'apps/web', 'packages/agent', 'packages/ir', 'python'].join()) throw new Error('Incomplete test evidence.');
  // Reuse the authoritative import checker, capturing raw bytes rather than a shell text pipe.
  const dir = mkdtempSync(join(tmpdir(), 'ai4shader-docs-'));
  let output;
  try {
    const file = join(dir, 'imports.txt'), fd = openSync(file, 'w');
    let result;
    try { result = spawnSync(process.execPath, ['scripts/check-imports.mjs'], { cwd: root, stdio: ['ignore', fd, fd] }); }
    finally { closeSync(fd); }
    output = readFileSync(file, 'utf8');
    if (result.status !== 0) throw new Error(result.error?.message ?? output);
  } finally { rmSync(dir, { recursive: true, force: true }); }
  const modules = [...output.matchAll(/: (\d+) modules,/g)].reduce((n, m) => n + Number(m[1]), 0);
  const imports = Number(/(\d+) imports checked;/.exec(output)?.[1]);
  if (!modules || !imports) throw new Error('Import checker evidence format changed.');
  const readme = readFileSync(join(root, 'README.md'), 'utf8');
  const implementation = readFileSync(join(root, 'docs/IMPLEMENTATION.md'), 'utf8');
  const errors = checkClaims(readme, implementation, report.counts, modules, imports);
  const relative = [...output.matchAll(/, (\d+) relative \+/g)].reduce((n,m) => n + Number(m[1]), 0);
  const detail = implementation.split('\n').find(l => l.startsWith('| import 完整性 |')) ?? '';
  if (!detail.includes(`${relative} 条相对路径`) || !detail.includes(`${imports-relative} 条裸标识符`)) errors.push('IMPLEMENTATION: relative/bare import breakdown differs');
  if (process.argv.includes('--self-test')) {
    assert.ok(checkClaims(readme.replace(/\*\*\d+ 个测试通过/, '**999999 个测试通过'), implementation, report.counts, modules, imports).length);
    assert.ok(checkClaims(readme, implementation, report.counts, modules + 1, imports).length);
    assert.ok(checkClaims(readme, implementation, report.counts, modules, imports + 1).length);
    assert.ok(checkClaims(readme, implementation, { ...report.counts, 'packages/agent': report.counts['packages/agent'] + 1 }, modules, imports).length);
  }
  if (errors.length) throw new Error(errors.join('\n'));
  console.log(`docs verified: ${Object.values(report.counts).reduce((a,b)=>a+b,0)} tests, ${modules} modules, ${imports} imports`);
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { checkDocs(); } catch (error) { console.error(error.message); process.exitCode = 1; }
}
