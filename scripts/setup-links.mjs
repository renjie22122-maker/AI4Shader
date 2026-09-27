/**
 * AI4Shader — workspace link setup
 * ================================
 *
 * `import '@ai4shader/ir'` only resolves if something creates `node_modules/@ai4shader/ir`.
 * `npm install` does that for a workspace; this does it without npm, which matters because
 * the zero-build path is supposed to work from a bare checkout.
 *
 *   node scripts/setup-links.mjs [--force]
 *
 * Junctions on Windows, symlinks elsewhere. Junction creation on Windows does not require
 * the developer-mode privilege that symlinks do, which is the whole reason for the
 * distinction. Running this twice is a no-op; `--force` recreates the links.
 */

import { symlinkSync, existsSync, lstatSync, rmSync, mkdirSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const scopeDir = join(repoRoot, 'node_modules', '@ai4shader');
const force = process.argv.includes('--force');
const isWindows = process.platform === 'win32';

/** name -> package directory, mirroring the `workspaces` globs in the root package.json. */
const WORKSPACES = [
  ['ir', 'packages/ir'],
  ['agent', 'packages/agent'],
  ['cli', 'apps/cli'],
  ['web', 'apps/web'],
];

mkdirSync(scopeDir, { recursive: true });

let made = 0;
let skipped = 0;
const failures = [];

for (const [name, rel] of WORKSPACES) {
  const target = join(repoRoot, rel);
  const link = join(scopeDir, name);

  if (!existsSync(target)) {
    skipped++;
    process.stdout.write(`· @ai4shader/${name}: ${rel} does not exist yet, skipped\n`);
    continue;
  }

  const existing = existsSync(link) || (() => {
    try {
      lstatSync(link);
      return true;
    } catch {
      return false;
    }
  })();

  if (existing) {
    if (!force) {
      skipped++;
      process.stdout.write(`· @ai4shader/${name}: already linked\n`);
      continue;
    }
    rmSync(link, { recursive: true, force: true });
  }

  try {
    symlinkSync(target, link, isWindows ? 'junction' : 'dir');
    made++;
    process.stdout.write(`ok @ai4shader/${name} -> ${relative(repoRoot, target)}\n`);
  } catch (err) {
    failures.push(`@ai4shader/${name}: ${err instanceof Error ? err.message : String(err)}`);
  }
}

if (failures.length) {
  process.stderr.write(
    `\nFAILED to create ${failures.length} link(s):\n` +
      failures.map((f) => `  ${f}`).join('\n') +
      `\n\nOn Windows, creating a *symlink* needs Developer Mode or elevation. This script\n` +
      `uses a junction instead, which does not — so a failure here usually means the target\n` +
      `path is on a different volume than the repository.\n`,
  );
  process.exit(1);
}

process.stdout.write(
  `\n${made} link(s) created, ${skipped} skipped.\n` +
    `Verify with:  node -e "import('@ai4shader/ir').then(m => console.log(Object.keys(m).length))"\n` +
    `(it prints the export count; a module-not-found error means the links or the build are missing)\n`,
);
