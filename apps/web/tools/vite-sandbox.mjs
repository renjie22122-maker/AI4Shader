/**
 * Vite launcher for sandboxes that deny child processes with piped stdio
 * =====================================================================
 *
 * Vite's only mandatory child process on Windows is a single probe:
 *
 *     exec("net use", cb)      // WindowsSafeRealPathSync: detect mapped network drives
 *
 * That probe is optional — Vite's own callback starts with `if (error) return;`, i.e. a
 * failing probe means "no network drives, use the plain realpath". But `exec()` builds
 * its pipes synchronously, so in a sandbox that refuses them the EPERM is thrown *before*
 * the callback can run, escapes `optimizeSafeRealPathSync()`, and kills the build with a
 * stack trace that has nothing to do with the project.
 *
 * This launcher installs a wrapper around `child_process.exec` that converts such a
 * synchronous throw into the ordinary callback error Vite already handles. Nothing else
 * is patched, and nothing inside `node_modules` is modified.
 *
 * Usage (see `npm run build:sandbox`):
 *
 *     node tools/vite-sandbox.mjs build --config vite.config.offline.mjs --configLoader native
 *
 * `createRequire` is used deliberately: importing `node:child_process` as ESM here would
 * create the module facade *before* the patch, and Vite would keep the original binding.
 */

import { createRequire } from 'node:module';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const require = createRequire(import.meta.url);
const childProcess = require('node:child_process');

const originalExec = childProcess.exec;

childProcess.exec = function sandboxTolerantExec(...args) {
  try {
    return originalExec.apply(this, args);
  } catch (err) {
    const callback = args.find((a) => typeof a === 'function');
    if (!callback) throw err;
    queueMicrotask(() => callback(err, '', ''));
    // Minimal stand-in for a ChildProcess: Vite only needs the callback to fire.
    const stub = { on: () => stub, once: () => stub, kill: () => true, pid: 0 };
    return stub;
  }
};

// The CLI reads process.argv, so the arguments passed to this launcher are forwarded
// verbatim. `vite/bin/vite.js` is not in the package's `exports` map, so the dist entry
// is resolved through the package itself and imported by absolute path.
const viteEntry = require.resolve('vite');
await import(pathToFileURL(path.join(path.dirname(viteEntry), 'cli.js')).href);
