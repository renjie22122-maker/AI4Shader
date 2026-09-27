/**
 * Node → `.js` → `.ts` resolution hook.
 *
 * The package is authored for `moduleResolution: nodenext`, so every relative import
 * carries a `.js` extension (`./types.js`) — but the file on disk is `types.ts`. tsc,
 * Vite and bundlers all rewrite that; **Node does not** (there is no `.js` → `.ts`
 * fallback in the type-stripping loader). Without this hook
 * `node --test test/targets.test.ts` dies with
 * `Cannot find module '.../src/types.js'`.
 *
 * The hook only fires when the `.js` file genuinely does not exist, so a real build
 * output next to the sources keeps working.
 */
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

export async function resolve(specifier, context, nextResolve) {
  if (specifier.endsWith('.js') && (specifier.startsWith('./') || specifier.startsWith('../'))) {
    try {
      const url = new URL(specifier, context.parentURL);
      const tsUrl = new URL(`${url.href.slice(0, -3)}.ts`);
      if (existsSync(fileURLToPath(tsUrl))) {
        return { url: tsUrl.href, shortCircuit: true };
      }
    } catch {
      /* fall through to the default resolver */
    }
  }
  return nextResolve(specifier, context);
}
