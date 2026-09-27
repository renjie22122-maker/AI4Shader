/**
 * AI4Shader — TypeScript -> JavaScript projection
 * ===============================================
 *
 * ## Why this exists
 *
 * This project's runtime has no bundler, no `npm install`, and no `tsc`. That is not an
 * accident of the environment, it is a deliberate constraint that turned out to be a
 * feature:
 *
 *   • The IR package is **pure computation with zero dependencies**, so it runs
 *     unmodified under Node's native type stripping AND in a browser, as long as the
 *     type syntax is gone.
 *   • The browser workbench must open from a static file server with no build step,
 *     because a shader tool that needs a 200 MB toolchain before it shows you a gradient
 *     is a tool nobody tries.
 *
 * So: **`src/*.ts` is the authored source of truth** — it carries the types, it is what
 * a human reads, and it is what a future `tsc --noEmit` would check. This script
 * projects it to `dist/*.js`, which is what Node and the browser actually load. Relative
 * imports keep their `.js` specifiers, which resolve correctly in both hosts because the
 * emitted tree mirrors the source tree exactly.
 *
 * ## Why the projection uses Node's own stripper
 *
 * An earlier revision of this file contained a hand-written ~600-line type eraser: a
 * tokenizer, a bracket-role classifier, a left-context declarator analyser and a set of
 * structural regexes. It was correct for the constructs in this repo after a great deal
 * of debugging, and it was the wrong tool. `node:module`'s `stripTypeScriptTypes` is the
 * same implementation that `node --experimental-strip-types` uses to run `.ts` files, so
 * using it means the projection and the runtime agree **by construction** rather than by
 * coincidence — there is exactly one TypeScript front end in the loop, and it is not
 * ours.
 *
 *   node --experimental-strip-types --experimental-transform-types scripts/project-ts.ts
 *
 * The `--experimental-transform-types` flag is what makes `enum` and `namespace`
 * transformable instead of an error; `mode: 'strip'` keeps import specifiers untouched
 * (TypeScript's own `verbatimModuleSyntax` semantics), which is what a browser needs.
 *
 * ## Verification, not assertion
 *
 * Type erasure that silently produces broken JavaScript is worse than no build at all,
 * so every projected module is loaded through a `data:` URL before it is written. That
 * catches syntax errors AND missing-export errors, which a parse-only check would miss.
 */

import { readFile, writeFile, mkdir, readdir, rm } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

export type ProjectResult = { code: string };

/**
 * Erase the types from one module.
 *
 * `mode: 'transform'` rather than `'strip'`, because `strip` rejects TypeScript
 * *parameter properties* (`constructor(private readonly x: T)`) — a construct that is
 * genuinely useful in the preview renderers and that no browser can execute. Transform
 * mode rewrites them into real field assignments. The module specifiers, which are the
 * only other thing transform mode may touch, are written with explicit `.js` extensions
 * throughout this repo, so they survive untouched — which `validateModule` below checks
 * empirically rather than trusting.
 */
export function project(src: string): ProjectResult {
  const code = stripTypeScriptTypes(src, { mode: 'transform', sourceMap: false });
  return { code };
}

/**
 * Load a projected module through a `data:` URL.
 *
 * Catches syntax errors and missing bindings. A module whose top-level code legitimately
 * throws (because it needs a DOM, say) is not a failure — the browser modules do exactly
 * that when imported under Node.
 *
 * Note what this does **not** catch: a relative import pointing at a file that was never
 * written. Importing through a `data:` URL gives the module a `data:` base, so `./x.js`
 * cannot resolve, while a *bare* specifier like `node:fs` still does — which makes the two
 * indistinguishable from the error text alone. That class of breakage is caught instead by
 * `checkRelativeDeps` below, statically, against the files this run actually wrote.
 */
export async function validateModule(code: string, label: string): Promise<void> {
  const url = 'data:text/javascript;base64,' + Buffer.from(code, 'utf8').toString('base64');
  try {
    await import(url);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    if (/SyntaxError|does not provide an export|Invalid or unexpected|is not defined|Cannot find package/.test(msg)) {
      throw new Error(`${label}: projected module failed to load — ${msg}`);
    }
  }
}

/**
 * Verify that every **relative** import in the emitted tree resolves to a file that this
 * run wrote (or that already exists).
 *
 * This is the check that catches the failure mode `validateModule` structurally cannot: a
 * module that imports a sibling which was never emitted, because the source file is missing
 * or the project's file list drifted. Without it the projection reports success and the
 * browser reports a blank page, which is the worst possible division of labour.
 *
 * Bare specifiers are ignored by design — in the browser those are the host's problem, and
 * the zero-build path has none of them (see `scripts/check-imports.mjs`).
 */
export function checkRelativeDeps(
  written: { dest: string; code: string }[],
  repoRoot: string,
): string[] {
  const problems: string[] = [];
  const known = new Set(written.map((w) => resolve(w.dest)));
  const IMPORT_RE = /(?:from\s+|import\s*\(\s*)['"](\.[^'"]+)['"]/g;

  for (const { dest, code } of written) {
    for (const m of code.matchAll(IMPORT_RE)) {
      const spec = m[1]!;
      if (spec.includes('?')) continue; // bundler-only request, unreachable here
      const target = resolve(dirname(dest), spec);
      if (known.has(target) || existsSync(target)) continue;
      problems.push(
        `${relative(repoRoot, dest)} imports ${spec}\n      -> ${relative(repoRoot, target)} does not exist`,
      );
    }
  }
  return problems;
}

/* -------------------------------------------------------------------------- */
/* CLI                                                                         */
/* -------------------------------------------------------------------------- */

async function walk(dir: string, out: string[] = []): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true });
  for (const e of entries) {
    const full = join(dir, e.name);
    if (e.isDirectory()) {
      if (e.name === 'node_modules' || e.name === 'dist') continue;
      await walk(full, out);
    } else if (e.name.endsWith('.ts') && !e.name.endsWith('.d.ts')) {
      out.push(full);
    }
  }
  return out;
}

const TARGETS = [
  { name: '@ai4shader/ir', pkg: 'packages/ir' },
  { name: '@ai4shader/agent', pkg: 'packages/agent' },
  { name: '@ai4shader/cli', pkg: 'apps/cli' },
  { name: '@ai4shader/web', pkg: 'apps/web', /** Browser sources are loaded from `dist` too. */ srcDir: 'src', outDir: 'dist' },
];

async function main(): Promise<void> {
  const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
  // `--no-validate` exists for the rare case of projecting a module that intentionally
  // depends on a host global which only exists in the browser. It is opt-*out* on
  // purpose: the check is the safety net, so skipping it must be a deliberate act.
  const validate = !process.argv.includes('--no-validate');
  let total = 0;
  const failures: string[] = [];

  for (const t of TARGETS) {
    const srcDir = join(repoRoot, t.pkg, t.srcDir ?? 'src');
    const outDir = join(repoRoot, t.pkg, t.outDir ?? 'dist');
    if (!existsSync(srcDir)) {
      process.stdout.write(`· ${t.name}: no ${relative(repoRoot, srcDir)} yet, skipped\n`);
      continue;
    }
    await rm(outDir, { recursive: true, force: true });
    const files = (await walk(srcDir)).sort();
    let failed = 0;
    const written: { dest: string; code: string }[] = [];
    for (const file of files) {
      const rel = relative(srcDir, file);
      const dest = join(outDir, rel.replace(/\.ts$/, '.js'));
      await mkdir(dirname(dest), { recursive: true });
      const label = `${t.name}/${rel}`;
      try {
        const { code } = project(await readFile(file, 'utf8'));
        // Load the projected module before writing it. Type erasure that produces
        // JavaScript which *parses* but has a dangling binding is the failure mode this
        // catches, and it only shows up when the browser hits it — which is the worst
        // possible place to find out. A module whose top-level code legitimately throws
        // (because it needs a DOM) is not a failure; `validateModule` distinguishes them.
        if (validate) await validateModule(code, label);
        await writeFile(dest, code, 'utf8');
        written.push({ dest, code });
        total++;
      } catch (err) {
        failed++;
        failures.push(`${label}: ${err instanceof Error ? err.message : String(err)}`);
      }
    }

    // Now that the whole tree for this target is known, check that the tree is *closed*:
    // every relative import points at a file that exists in it. A module importing a
    // sibling that was never emitted is the one breakage `validateModule` cannot see, and
    // it turns into a blank page in the browser rather than a build error.
    for (const p of checkRelativeDeps(written, repoRoot)) {
      failed++;
      failures.push(`${t.name}: unresolved relative import\n      ${p}`);
    }

    process.stdout.write(
      `✓ ${t.name}: ${files.length - failed}/${files.length} modules -> ${relative(repoRoot, outDir)}\n`,
    );
  }

  if (failures.length) {
    process.stderr.write(`\n✗ ${failures.length} module(s) failed to project:\n`);
    for (const f of failures) process.stderr.write(`\n${f}\n`);
    process.exit(1);
  }
  process.stdout.write(
    `\n${total} module(s) projected. \`src/*.ts\` is the source of truth; never edit \`dist/*.js\`.\n`,
  );
}

// Run when invoked directly, or when `--run` is passed (which the npm script does).
const invokedPath = process.argv[1] ? resolve(process.argv[1]) : '';
const invokedDirectly = invokedPath === fileURLToPath(import.meta.url);
if (invokedDirectly || process.argv.includes('--run')) {
  await main();
}
