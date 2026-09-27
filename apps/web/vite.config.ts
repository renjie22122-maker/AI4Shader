import { defineConfig } from 'vite';
import { fileURLToPath } from 'node:url';

/**
 * The IR package is consumed straight from source (no build step, no npm link):
 * the browser build resolves the same `.ts` files the CLI and the tests use, so the
 * preview and the CLI can never disagree about what a document means.
 *
 * NOTE: these targets may not exist yet while the IR package is being written by
 * other agents — an alias that is never imported is never resolved, so `vite build`
 * still succeeds. Runtime feature detection lives in `src/doc/catalog.ts`.
 */
const ir = (rel: string): string =>
  fileURLToPath(new URL(`../../packages/ir/src/${rel}`, import.meta.url));

export default defineConfig({
  // Also declared under the `vite` key of package.json (port 3080 belongs to another app).
  server: { port: 5183, strictPort: false },
  resolve: {
    alias: [
      { find: '@ai4shader/ir/catalog', replacement: ir('catalog/index.ts') },
      { find: '@ai4shader/ir/patcher', replacement: ir('patcher.ts') },
      { find: /^@ai4shader\/ir$/, replacement: ir('index.ts') },
    ],
  },
  build: {
    target: 'es2022',
    outDir: 'dist',
    emptyOutDir: true,
    sourcemap: true,
    chunkSizeWarningLimit: 1500,
  },
  worker: { format: 'es' },
});
