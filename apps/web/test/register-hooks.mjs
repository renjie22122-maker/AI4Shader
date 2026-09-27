/**
 * Make Node resolve module specifiers the way a *browser* resolves them.
 * ================================================================
 *
 * `apps/web/test/boot.test.ts` exists to answer one question — "would this page work in a
 * browser?" — and for a while it answered it wrong. Node resolves bare specifiers through
 * `node_modules`; a browser does not, and fails the **entire** module graph instead. So
 * `packages/agent/dist/*.js` importing `@ai4shader/ir` was invisible to every Node-side
 * check while being fatal in the only environment that matters:
 *
 *     Failed to resolve module specifier "@ai4shader/ir".
 *     Relative references must start with either "/", "./", or "../".
 *
 * `dist/main.js` never executed and the page rendered its static shell with every panel
 * empty — while the smoke test, the projection check, the import check and the type checker
 * all reported success.
 *
 * The lesson is not "add a check for bare specifiers"; it is that **a test cannot borrow
 * Node's resolver and then claim to speak for a browser.** So this hook removes the
 * difference. For the duration of the test run:
 *
 *   • a bare specifier resolves **only** through the import map declared in the real
 *     `apps/web/index.html`, exactly as a browser would;
 *   • a bare specifier with no entry is a hard error carrying the browser's own wording,
 *     so the failure names the actual cause instead of surfacing later as an odd `undefined`;
 *   • `node:` builtins stay available, because the test process is still Node and the DOM
 *     shim genuinely needs `node:test` and `node:fs`. Those cannot appear in the browser
 *     graph, and `scripts/check-imports.mjs` is what proves they do not.
 *
 * The map is read from the document rather than restated here. A copy would let the map and
 * the check drift, and drift between a declaration and its enforcement is precisely the bug
 * this file was written to end.
 */

import { readFileSync } from 'node:fs';
import { registerHooks } from 'node:module';
import { dirname, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..', '..');

/** The document whose import map governs the browser path. */
const INDEX_HTML = resolve(repoRoot, 'apps', 'web', 'index.html');

function loadImportMap() {
  const html = readFileSync(INDEX_HTML, 'utf8');
  const m = /<script\s+type="importmap"\s*>([\s\S]*?)<\/script>/.exec(html);
  if (!m) {
    throw new Error(
      `${INDEX_HTML} has no <script type="importmap">. The browser path needs one: ` +
        `packages/agent/dist imports '@ai4shader/ir' by bare specifier.`,
    );
  }
  return JSON.parse(m[1]).imports ?? {};
}

const importMap = loadImportMap();

/** `node:` builtins plus the handful of bare builtin names, which are Node-only anyway. */
const isBuiltin = (spec) =>
  spec.startsWith('node:') || spec === 'fs' || spec === 'path' || spec === 'url' || spec === 'util';

/** Anything carrying a URL scheme — `file:`, `node:`, `data:` — is not a bare specifier. */
const hasScheme = (spec) => /^[a-zA-Z][a-zA-Z0-9+.-]*:/.test(spec);

registerHooks({
  resolve(specifier, context, nextResolve) {
    // Relative, absolute, builtin and already-resolved URL specifiers behave identically in
    // both worlds. (The test runner hands the entry point over as a `file://` URL.)
    if (
      specifier.startsWith('.') ||
      specifier.startsWith('/') ||
      isBuiltin(specifier) ||
      hasScheme(specifier)
    ) {
      return nextResolve(specifier, context);
    }

    /*
     * A `?query` suffix is a bundler request (Vite's `?worker`), not a module a browser
     * would ever be handed on this path. Let Node's own resolver deal with it so that the
     * Vite branch of `createWorker()` is not what fails here.
     */
    if (specifier.includes('?')) return nextResolve(specifier, context);

    // Exact match only. Import maps do no extension resolution and no prefix matching
    // unless a trailing-slash key is declared, and none is — see index.html for why.
    const mapped = importMap[specifier];
    if (mapped === undefined) {
      throw new Error(
        `Failed to resolve module specifier "${specifier}".\n` +
          `  imported by ${context.parentURL ?? '(entry)'}\n` +
          `  A browser cannot resolve a bare specifier without an import map, and ` +
          `apps/web/index.html has no entry for "${specifier}".\n` +
          `  This is the failure that leaves the workbench a static shell — add the entry, ` +
          `or make the import relative.`,
      );
    }
    if (!mapped.startsWith('/')) {
      throw new Error(
        `The import map in apps/web/index.html sends "${specifier}" to "${mapped}", which is ` +
          `not root-relative. scripts/serve.mjs serves the repository root, so every entry ` +
          `must start with "/".`,
      );
    }
    return nextResolve(pathToFileURL(resolve(repoRoot, mapped.slice(1))).href, context);
  },
});
