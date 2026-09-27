/**
 * AI4Shader — zero-dependency static file server
 * =============================================
 *
 * The workbench is designed to open with no build step, which means it needs *a* server
 * (ES modules cannot be loaded over `file://` — the browser blocks them as cross-origin).
 * This is that server: ~120 lines, no dependencies, and deliberately boring, because the
 * whole point of the zero-build path is that nothing stands between a checkout and a
 * rendered pixel.
 *
 *   node scripts/serve.mjs            # http://127.0.0.1:5183/apps/web/index.html
 *   node scripts/serve.mjs --port 8080
 *
 * ## Why it serves the REPO ROOT
 *
 * Every import in the workbench is a *relative* specifier that reaches out of `apps/web`
 * into `packages/ir/dist` (for example `../../../../packages/ir/dist/index.js`). Those
 * paths only resolve if the request path preserves the repository layout, so the server
 * root has to be the repository root.
 *
 * ## The import map this server is paired with
 *
 * This header used to claim there was "deliberately no import map and no bare specifier
 * anywhere in the browser path". The second half was false: `packages/agent/dist/*.js`
 * imports `@ai4shader/ir` and `@ai4shader/ir/catalog` by bare specifier, which Node resolves
 * through the `node_modules/@ai4shader` links and a browser refuses outright. One refused
 * specifier fails the *entire* module graph, so `dist/main.js` never ran and the page
 * rendered its static shell with every panel empty.
 *
 * `apps/web/index.html` now declares an explicit import map for those two specifiers, and
 * `scripts/check-served.mjs` walks the graph this server hands out — over HTTP, resolving
 * through the map the page itself declares — so the served page is verified to load rather
 * than assumed to. The map is not a second resolution system layered on top of a working
 * one; it is the mechanism the browser was always going to need, written down and checked.
 *
 * `--root` exists for pointing at a subdirectory, but doing so breaks every cross-package
 * import, so `main` prints a loud warning rather than letting you debug a blank page.
 *
 * ## Safety
 *
 * Path traversal is rejected by resolving the request against the root and verifying the
 * result is still inside it. This is a local development tool, but "it is only local" is
 * how directory-traversal bugs get shipped, and the check is three lines.
 */

import { createServer } from 'node:http';
import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { extname, join, normalize, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));

const MIME = new Map(Object.entries({
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.gif': 'image/gif',
  '.ico': 'image/x-icon',
  '.woff2': 'font/woff2',
  '.txt': 'text/plain; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
  '.wgsl': 'text/plain; charset=utf-8',
  '.glsl': 'text/plain; charset=utf-8',
  '.frag': 'text/plain; charset=utf-8',
  '.metal': 'text/plain; charset=utf-8',
  '.hlsl': 'text/plain; charset=utf-8',
  '.shader': 'text/plain; charset=utf-8',
}));

function parseArgs(argv) {
  const out = { port: 5183, root: REPO_ROOT, host: '127.0.0.1' };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--port' || a === '-p') out.port = Number(argv[++i]);
    else if (a === '--root') out.root = resolve(argv[++i]);
    else if (a === '--host') out.host = argv[++i];
    else if (a === '--help' || a === '-h') out.help = true;
  }
  return out;
}

/** Resolve a URL path to a file inside `root`, or null if it escapes or is missing. */
async function resolveFile(root, urlPath) {
  let pathname = decodeURIComponent(urlPath.split('?')[0].split('#')[0]);
  if (pathname.endsWith('/')) pathname += 'index.html';
  const candidate = normalize(join(root, pathname));
  if (candidate !== root && !candidate.startsWith(root + sep)) return null;
  try {
    const st = await stat(candidate);
    if (st.isDirectory()) {
      const idx = join(candidate, 'index.html');
      const stIdx = await stat(idx).catch(() => null);
      return stIdx?.isFile() ? idx : null;
    }
    return st.isFile() ? candidate : null;
  } catch {
    return null;
  }
}

const args = parseArgs(process.argv.slice(2));
if (args.help) {
  process.stdout.write(
    'Usage: node scripts/serve.mjs [--port 5183] [--host 127.0.0.1] [--root <dir>]\n\n' +
      'Serves the repository root so that the workbench import map resolves.\n' +
      `Default root: ${REPO_ROOT}\n`,
  );
  process.exit(0);
}

if (!Number.isInteger(args.port) || args.port <= 0 || args.port > 65535) {
  process.stderr.write(`✗ invalid --port\n`);
  process.exit(2);
}
if (args.root !== REPO_ROOT) {
  process.stderr.write(
    `WARNING --root is not the repository root.\n` +
      `  The workbench imports across packages by relative path\n` +
      `  (../../../../packages/ir/dist/index.js), so the document root must be\n` +
      `  ${REPO_ROOT} for it to load. Serving anyway.\n`,
  );
}

/** Where the workbench lives, relative to the repo root this server insists on serving. */
const WORKBENCH = '/apps/web/index.html';

const server = createServer(async (req, res) => {
  const raw = req.url ?? '/';
  const url = raw.split('?')[0].split('#')[0];
  if (req.method !== 'GET' && req.method !== 'HEAD') {
    res.writeHead(405, { allow: 'GET, HEAD' }).end('method not allowed');
    return;
  }

  /*
   * `/` and `/apps/web` redirect to the workbench.
   *
   * The page lives at a long path because every import in it is relative and crosses out of
   * `apps/web` into `packages/ir/dist`, which is what forces the document root to be the repo
   * root. Requiring people to type that whole path is a worse trade than one redirect: opening
   * `http://127.0.0.1:5183/` and getting a bare 404 reads as "the server is broken", and that
   * is a diagnosis this project has already cost someone once.
   */
  if (url === '/' || url === '/index.html' || url === '/apps/web' || url === '/apps/web/') {
    res.writeHead(302, { location: WORKBENCH, 'cache-control': 'no-store' }).end();
    return;
  }

  const file = await resolveFile(args.root, url);
  if (!file) {
    // A 404 that names the likely cause costs nothing and saves a lot of guessing.
    const hint = url.startsWith('/apps/web') && !url.includes('.')
      ? ` (did you mean ${WORKBENCH}?)`
      : '';
    res.writeHead(404, { 'content-type': 'text/plain; charset=utf-8' });
    res.end(`404 ${url}${hint}\n`);
    return;
  }

  const type = MIME.get(extname(file).toLowerCase()) ?? 'application/octet-stream';
  // No caching: this is a development server and a stale module is a wasted debugging
  // session. (The shader compile cache in the app is content-addressed and unaffected.)
  res.writeHead(200, { 'content-type': type, 'cache-control': 'no-store' });
  if (req.method === 'HEAD') {
    res.end();
    return;
  }
  createReadStream(file).pipe(res);
});

server.listen(args.port, args.host, () => {
  const url = `http://${args.host}:${args.port}/apps/web/index.html`;
  process.stdout.write(
    `AI4Shader static server\n` +
      `  root  ${args.root}\n` +
      `  工作台 ${url}\n` +
      `  示例   http://${args.host}:${args.port}/examples/liquid-metal.shader.json\n\n` +
      `按 Ctrl+C 停止。\n`,
  );
});
