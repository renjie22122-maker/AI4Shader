/**
 * esbuild-free Vite configuration — sandbox / offline fallback
 * ===========================================================
 *
 * `vite.config.ts` is the canonical configuration. This file is the *same* settings,
 * expressed so that Vite can use them **without spawning esbuild**, which matters in
 * two situations:
 *
 *   1. A hardened/offline CI sandbox that blocks child processes with piped stdio
 *      (esbuild's JS API talks to `esbuild.exe` over pipes; when that spawn is denied,
 *      `vite build` dies with `spawn EPERM` before it ever reads a source file).
 *   2. Anywhere `esbuild` cannot be installed for the host platform.
 *
 * Run it with the native config loader (no esbuild bundling of the config either):
 *
 *     vite build --config vite.config.offline.mjs --configLoader native
 *
 * TypeScript is transpiled **in process** by the `typescript` compiler API through the
 * plugin below, and minification/target-lowering (the other two esbuild consumers) are
 * turned off. Output is byte-for-byte equivalent modulo formatting.
 */

import { fileURLToPath } from 'node:url';
import ts from 'typescript';

const ir = (rel) => fileURLToPath(new URL(`../../packages/ir/src/${rel}`, import.meta.url));

/** Strip types with the TypeScript compiler API. No child processes, no pipes. */
function typescriptInProcess() {
  return {
    name: 'ai4shader:typescript-in-process',
    enforce: 'pre',
    transform(code, id) {
      const file = id.split('?')[0] ?? id;
      if (!/\.(ts|tsx|mts|cts)$/.test(file) || file.endsWith('.d.ts')) return null;
      const out = ts.transpileModule(code, {
        fileName: file,
        compilerOptions: {
          target: ts.ScriptTarget.ES2022,
          module: ts.ModuleKind.ESNext,
          moduleResolution: ts.ModuleResolutionKind.Bundler,
          jsx: ts.JsxEmit.Preserve,
          isolatedModules: true,
          useDefineForClassFields: true,
          sourceMap: true,
        },
      });
      return { code: out.outputText, map: out.sourceMapText ?? null };
    },
  };
}

export default {
  server: { port: 5183, strictPort: false },
  esbuild: false,
  resolve: {
    alias: [
      { find: '@ai4shader/ir/catalog', replacement: ir('catalog/index.ts') },
      { find: '@ai4shader/ir/patcher', replacement: ir('patcher.ts') },
      { find: /^@ai4shader\/ir$/, replacement: ir('index.ts') },
    ],
  },
  plugins: [typescriptInProcess()],
  build: {
    // esbuild is disabled, so Vite must not ask it to lower syntax or minify.
    target: 'esnext',
    minify: false,
    cssMinify: false,
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: true,
    chunkSizeWarningLimit: 1500,
  },
  worker: {
    format: 'es',
    // Workers are bundled by a *separate* rollup environment seeded from
    // `worker.plugins`, so a transform plugin that is only in `plugins` silently does
    // not apply to `score.worker.ts`. Register it twice on purpose.
    plugins: () => [typescriptInProcess()],
  },
};
