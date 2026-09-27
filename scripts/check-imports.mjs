/**
 * Resolve every import in an emitted tree the way the *browser* would, and report the ones
 * it could not.
 *
 * This exists because the zero-build path has no bundler to catch a bad specifier: a typo
 * in an import is not a build error, it is a blank page with one red line in a console most
 * people will not open. Running this over `dist/` turns that class of bug into a build-time
 * failure.
 *
 *   node scripts/check-imports.mjs [dir ...]     # defaults to the emitted trees
 *
 * ## Why bare specifiers are now a hard failure
 *
 * This script used to check only relative specifiers and then *print a paragraph* saying
 * that bare ones are a design smell the browser cannot resolve, suggesting an `rg` command
 * to look for them by hand. That paragraph was the bug: the invariant was never true.
 * `packages/agent/dist/*.js` shipped seven `import … from '@ai4shader/ir'`-style lines, Node
 * resolved them happily through the `node_modules/@ai4shader` links, every Node-side check
 * stayed green — and a browser refused the whole module graph with
 *
 *     Failed to resolve module specifier "@ai4shader/ir".
 *
 * `dist/main.js` never executed; the page rendered its static HTML with every panel empty.
 * A check that *describes* an invariant it does not enforce is worse than no check, because
 * it is where everyone looks to confirm the thing that is broken.
 *
 * So the invariant is enforced now:
 *
 *   • **Browser trees** (`apps/web/dist`, `packages/ir/dist`, `packages/agent/dist`) are
 *     served to a browser, so a bare specifier there is legal only with an **exact** entry
 *     in the import map declared in `apps/web/index.html`, and that entry must point at a
 *     file that exists.
 *   • **`apps/cli/dist` is Node-only** and is not map-governed. It imports `node:fs`,
 *     `node:path` and friends, which are bare by definition and unresolvable in a browser
 *     *by design* — mapping them would be nonsense. It is also the one tree that is
 *     executed for real by a test suite (31 end-to-end tests spawn the CLI), so a broken
 *     specifier there already fails loudly.
 *
 * "Exact" matters: import maps do no extension resolution, so a prefix mapping
 * (`"@ai4shader/ir/": "…/dist/"`) would send `@ai4shader/ir/types` to a file named `types`
 * with no extension — a 404. Listing every subpath explicitly is what lets this check verify
 * each one. Unused map entries are a failure too: a second resolution system rots by keeping
 * entries nothing imports while losing the one that mattered.
 *
 * ## Specifiers are read from *statements*, not from string literals
 *
 * The extraction is line-anchored rather than a bare `from '…'` scan. A scan that looks for
 * `from` followed by a quoted string also matches ordinary code — `const q = raw['…']`, a
 * message that happens to contain the word `from` — and this script's first version did
 * exactly that, inventing eleven "unresolvable imports" out of a single array access. The
 * projection does not reformat, so every real import starts its own line (`import x from`,
 * `} from`, `import '`, `export … from`), and a dynamic `import('…')` is the one form that
 * can sit anywhere.
 */

import { readdirSync, readFileSync, existsSync, statSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';

/**
 * Trees a browser loads, and therefore trees whose bare specifiers must be in the map.
 *
 * Membership is about "is this served to a browser", not "is this nice to have": a single
 * unresolvable specifier anywhere in the reachable graph fails the whole page, not just the
 * module that contains it.
 */
const BROWSER_TREES = new Set(['packages/ir/dist', 'packages/agent/dist', 'apps/web/dist']);

/** The document whose import map governs the browser path. */
const INDEX_HTML = 'apps/web/index.html';

const roots = process.argv.slice(2);
const targets = roots.length
  ? roots
  : ['packages/ir/dist', 'packages/agent/dist', 'apps/cli/dist', 'apps/web/dist'];

/** Recursively list files under `dir`. */
function walk(dir, out = []) {
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const full = join(dir, e.name);
    if (e.isDirectory()) walk(full, out);
    else out.push(full);
  }
  return out;
}

/**
 * Read the import map out of the real `index.html`.
 *
 * Parsed from the document rather than duplicated here on purpose: a copy would let the
 * check and the page disagree, which is the exact failure mode this whole script is about.
 */
function readImportMap(htmlPath) {
  if (!existsSync(htmlPath)) return null;
  const html = readFileSync(htmlPath, 'utf8');
  const m = /<script\s+type="importmap"\s*>([\s\S]*?)<\/script>/.exec(html);
  if (!m) return null;
  try {
    return JSON.parse(m[1]).imports ?? {};
  } catch (err) {
    process.stderr.write(`✗ ${htmlPath}: the import map is not valid JSON — ${err.message}\n`);
    process.exit(1);
  }
}

/* ---- specifier extraction ------------------------------------------------- */

const LINE_FORMS = [
  /^[ \t]*import\s+['"]([^'"]+)['"]/, // side-effect import
  /^[ \t]*(?:import|export)\b[^'"]*\bfrom\s*['"]([^'"]+)['"]/, // single-line import/export
  /^[ \t]*\}?\s*from\s*['"]([^'"]+)['"]/, // the `} from '…'` line of a multi-line import
];
const DYNAMIC_RE = /\bimport\s*\(\s*['"]([^'"]+)['"]\s*\)/g;

/** Every module specifier in `src`, in source order. */
function specifiersOf(src) {
  const found = [];
  for (const line of src.split('\n')) {
    for (const re of LINE_FORMS) {
      const m = re.exec(line);
      if (m) {
        found.push(m[1]);
        break;
      }
    }
    for (const m of line.matchAll(DYNAMIC_RE)) found.push(m[1]);
  }
  return found;
}

/* ---- run ------------------------------------------------------------------ */

const importMap = readImportMap(INDEX_HTML);
if (importMap === null) {
  process.stderr.write(
    `✗ no import map found in ${INDEX_HTML}.\n` +
      `  The browser path needs one: packages/agent/dist imports '@ai4shader/ir' by bare\n` +
      `  specifier, and a browser cannot resolve that without a map.\n`,
  );
  process.exit(1);
}

let checked = 0;
let unresolved = 0;
let bareBrowser = 0;
let bareNodeOnly = 0;
const problems = [];
const usedBare = new Set();

for (const root of targets) {
  if (!existsSync(root) || !statSync(root).isDirectory()) {
    process.stdout.write(`· ${root}: not present, skipped\n`);
    continue;
  }
  const browserTree = BROWSER_TREES.has(root.replace(/\\/g, '/'));
  const files = walk(root).filter((f) => f.endsWith('.js') || f.endsWith('.mjs'));
  let local = 0;
  let bare = 0;

  for (const file of files) {
    for (const spec of specifiersOf(readFileSync(file, 'utf8'))) {
      // A specifier carrying a query string (`./x.ts?worker`) is a *bundler* request —
      // Vite's worker-suffix syntax — and is unreachable in the zero-build path by
      // construction, so it is not a missing file. Skipping it here keeps this check
      // meaningful instead of training people to ignore its output.
      if (spec.includes('?')) continue;
      checked++;

      /* ---- relative or rooted: resolvable in Node *and* in a browser -------- */
      if (spec.startsWith('.') || spec.startsWith('/')) {
        local++;
        const target = spec.startsWith('/') ? resolve(spec.slice(1)) : resolve(dirname(file), spec);
        if (!existsSync(target)) {
          unresolved++;
          problems.push(`${file}\n    imports ${spec}\n    -> ${target} does not exist`);
        }
        continue;
      }

      /* ---- bare specifier --------------------------------------------------- */
      bare++;
      if (!browserTree) {
        bareNodeOnly++;
        continue;
      }
      bareBrowser++;
      usedBare.add(spec);

      const mapped = importMap[spec];
      if (mapped === undefined) {
        unresolved++;
        problems.push(
          `${file}\n    imports ${spec}   (bare)\n` +
            `    -> no exact entry for "${spec}" in the import map in ${INDEX_HTML}\n` +
            `       a browser fails the WHOLE module graph: ` +
            `Failed to resolve module specifier "${spec}"`,
        );
        continue;
      }
      if (!mapped.startsWith('/')) {
        unresolved++;
        problems.push(
          `${file}\n    imports ${spec}\n` +
            `    -> the map sends it to ${mapped}, which is not root-relative; ` +
            `scripts/serve.mjs serves the repository root, so it must start with "/"`,
        );
        continue;
      }
      if (!existsSync(resolve(mapped.slice(1)))) {
        unresolved++;
        problems.push(
          `${file}\n    imports ${spec}\n    -> the map sends it to ${mapped}, which does not exist`,
        );
      }
    }
  }

  process.stdout.write(
    `✓ ${root}: ${files.length} modules, ${local} relative + ${bare} bare imports checked` +
      `${bare && !browserTree ? ' (Node-only tree, not map-governed)' : ''}\n`,
  );
}

const unused = Object.keys(importMap).filter((k) => !usedBare.has(k));

if (unresolved) {
  process.stderr.write(`\n✗ ${unresolved} of ${checked} imports would not resolve in a browser:\n\n`);
  for (const p of problems) process.stderr.write(`${p}\n\n`);
  process.exit(1);
}

if (unused.length) {
  process.stderr.write(
    `\n✗ ${unused.length} import-map entr${unused.length === 1 ? 'y is' : 'ies are'} unused:\n` +
      unused.map((k) => `    ${k} -> ${importMap[k]}\n`).join('') +
      `  Nothing imports ${unused.length === 1 ? 'it' : 'them'}, and a second resolution\n` +
      `  system rots by keeping entries that no longer matter while losing the one that does.\n` +
      `  Remove ${unused.length === 1 ? 'it' : 'them'} from ${INDEX_HTML}.\n`,
  );
  process.exit(1);
}

process.stdout.write(
  `\n${checked} imports checked; ${bareBrowser} bare specifiers resolve through the import\n` +
    `map in ${INDEX_HTML}.${bareNodeOnly ? ` ${bareNodeOnly} more are Node builtins in apps/cli/dist, which no browser loads.` : ''}\n`,
);
