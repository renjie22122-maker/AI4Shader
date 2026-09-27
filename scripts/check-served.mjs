/**
 * Does the page the server actually hands out load in a browser?
 * =============================================================
 *
 * `check-imports.mjs` reads `dist/` off disk with browser-accurate resolution, and
 * `apps/web/test/boot.test.ts` executes the real module graph. Both are necessary and
 * neither is sufficient, because both skip the layer in between: **HTTP**. A static server
 * gets to break a page in ways the filesystem never will —
 *
 *   • a `.js` file served as `text/plain` is refused by the browser's module loader
 *     ("Expected a JavaScript module script but the server responded with a MIME type of…")
 *     even though the bytes are perfect;
 *   • a 404 on one module fails the whole graph, not that module;
 *   • an import map that is served differently from the one on disk (rewritten, cached,
 *     truncated) resolves nothing.
 *
 * So this starts `scripts/serve.mjs` itself, fetches the page, and walks the module graph
 * over real HTTP: every specifier resolved the way a browser resolves it, including through
 * the import map the page itself declares. It is the closest thing to "open the page" that
 * this environment allows, since no browser can start here.
 *
 * The server is spawned with `stdio: 'ignore'` on purpose. A confined sandbox denies
 * `spawn` with *piped* stdio (`EPERM`) but permits `ignore`, and this script needs none of
 * the child's output — it talks to it over the socket, which is the thing under test anyway.
 *
 *   node scripts/check-served.mjs [--port 5199] [--keep]
 */

import { spawn } from 'node:child_process';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const argv = process.argv.slice(2);
const portArg = argv.indexOf('--port');
const PORT = portArg >= 0 ? Number(argv[portArg + 1]) : 5199;
const KEEP = argv.includes('--keep');
const BASE = `http://127.0.0.1:${PORT}`;
const ENTRY_PAGE = '/apps/web/index.html';

/* -------------------------------------------------------------------------- */
/* Specifier extraction — must match scripts/check-imports.mjs exactly          */
/* -------------------------------------------------------------------------- */

/**
 * Statements, not string literals.
 *
 * A plain `from '…'` scan also matches ordinary code (`const q = raw['…']`) and invents
 * imports that were never there. The projection does not reformat, so every real import
 * starts its own line; a dynamic `import('…')` is the one form that can sit anywhere.
 */
const LINE_FORMS = [
  /^[ \t]*import\s+['"]([^'"]+)['"]/,
  /^[ \t]*(?:import|export)\b[^'"]*\bfrom\s*['"]([^'"]+)['"]/,
  /^[ \t]*\}?\s*from\s*['"]([^'"]+)['"]/,
];
const DYNAMIC_RE = /\bimport\s*\(\s*['"]([^'"]+)['"]\s*\)/g;

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

/* -------------------------------------------------------------------------- */
/* Server lifecycle                                                            */
/* -------------------------------------------------------------------------- */

const child = KEEP
  ? null
  : spawn(process.execPath, ['scripts/serve.mjs', '--port', String(PORT)], {
      cwd: repoRoot,
      stdio: 'ignore',
    });

/** Poll until the server answers, so nothing here depends on a fixed sleep. */
async function waitForServer(timeoutMs = 10_000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const res = await fetch(BASE + ENTRY_PAGE, { redirect: 'manual' });
      if (res.status > 0) return true;
    } catch {
      /* not listening yet */
    }
    await new Promise((r) => setTimeout(r, 100));
  }
  return false;
}

function stop() {
  if (child && !child.killed) child.kill();
}
process.on('exit', stop);
process.on('SIGINT', () => {
  stop();
  process.exit(130);
});

if (!KEEP) {
  const up = await waitForServer();
  if (!up) {
    stop();
    process.stderr.write(
      `✗ scripts/serve.mjs did not come up on port ${PORT} within 10s.\n` +
        `  Either the port is already in use (try --port <other>) or the server itself is\n` +
        `  broken — start it by hand to see why: node scripts/serve.mjs --port ${PORT}\n`,
    );
    process.exit(1);
  }
}

/* -------------------------------------------------------------------------- */
/* The walk                                                                    */
/* -------------------------------------------------------------------------- */

const failures = [];
const pageRes = await fetch(BASE + ENTRY_PAGE);
if (!pageRes.ok) {
  process.stderr.write(`✗ ${ENTRY_PAGE} -> HTTP ${pageRes.status}\n`);
  stop();
  process.exit(1);
}
const page = await pageRes.text();

const mapMatch = /<script\s+type="importmap"\s*>([\s\S]*?)<\/script>/.exec(page);
if (!mapMatch) {
  process.stderr.write(
    `✗ the served ${ENTRY_PAGE} declares no import map.\n` +
      `  packages/agent/dist imports '@ai4shader/ir' by bare specifier, which a browser\n` +
      `  cannot resolve. The whole module graph is refused and the page is a static shell.\n`,
  );
  stop();
  process.exit(1);
}
let importMap;
try {
  importMap = JSON.parse(mapMatch[1]).imports ?? {};
} catch (err) {
  process.stderr.write(`✗ the served import map is not valid JSON — ${err.message}\n`);
  stop();
  process.exit(1);
}

// The map must be declared before the module script, or the browser ignores it.
const mapAt = page.indexOf('type="importmap"');
const moduleAt = page.indexOf('type="module"');
if (moduleAt < mapAt) {
  failures.push(
    `the import map appears AFTER the first module script in ${ENTRY_PAGE}; ` +
      `a browser ignores a map that arrives late, and every bare specifier then fails`,
  );
}

const entry = /<script\s+type="module"\s+src="([^"]+)"/.exec(page)?.[1];
if (!entry) {
  process.stderr.write(`✗ the served ${ENTRY_PAGE} has no <script type="module"> entry point\n`);
  stop();
  process.exit(1);
}

const seen = new Map();
const queue = [{ ref: '/apps/web/' + entry.replace(/^\.\//, ''), via: ENTRY_PAGE }];

while (queue.length) {
  const { ref, via } = queue.shift();
  if (seen.has(ref)) continue;
  seen.set(ref, via);

  let res;
  try {
    res = await fetch(BASE + ref);
  } catch (err) {
    failures.push(`${ref}\n    -> fetch failed: ${err.message}   (imported by ${via})`);
    continue;
  }
  const ctype = res.headers.get('content-type') ?? '(none)';
  if (!res.ok) {
    failures.push(`${ref}\n    -> HTTP ${res.status}   (imported by ${via})`);
    continue;
  }
  if (!/javascript/.test(ctype)) {
    failures.push(
      `${ref}\n    -> content-type is ${ctype}, so the browser's module loader refuses it\n` +
        `       even though the bytes are fine   (imported by ${via})`,
    );
    continue;
  }

  for (const spec of specifiersOf(await res.text())) {
    // A `?query` specifier is a bundler request (Vite's `?worker`), unreachable on this
    // path by construction. Same exemption as scripts/check-imports.mjs.
    if (spec.includes('?')) continue;

    if (spec.startsWith('./') || spec.startsWith('../')) {
      queue.push({ ref: new URL(spec, new URL(ref, BASE)).pathname, via: ref });
    } else if (spec.startsWith('/')) {
      queue.push({ ref: spec, via: ref });
    } else {
      const mapped = importMap[spec];
      if (mapped === undefined) {
        failures.push(
          `${ref}\n    imports ${spec}   (bare)\n` +
            `    -> no entry in the import map the page declares\n` +
            `       a browser fails the WHOLE module graph: ` +
            `Failed to resolve module specifier "${spec}"`,
        );
      } else {
        queue.push({ ref: mapped, via: ref });
      }
    }
  }
}

if (failures.length) {
  process.stderr.write(
    `\n✗ ${failures.length} problem(s): the page served at ${BASE}${ENTRY_PAGE} would NOT load in a browser.\n\n`,
  );
  for (const f of failures) process.stderr.write(`  ${f}\n\n`);
  stop();
  process.exit(1);
}

process.stdout.write(
  `✓ served page: ${seen.size} modules fetched from ${BASE}, all 200 with a JavaScript MIME type\n` +
    `✓ import map declared before the module script; every bare specifier resolves through it\n` +
    `\nThe served page ${ENTRY_PAGE} would load in a browser.\n`,
);
stop();
