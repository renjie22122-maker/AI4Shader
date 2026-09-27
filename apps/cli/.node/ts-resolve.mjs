import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

export async function resolve(specifier, context, nextResolve) {
  if (specifier.endsWith('.js') && (specifier.startsWith('./') || specifier.startsWith('../'))) {
    try {
      const url = new URL(specifier, context.parentURL);
      const tsUrl = new URL(url.href.slice(0, -3) + '.ts');
      // Only when the `.js` is genuinely absent, so a real build output next to the sources
      // keeps working.
      if (existsSync(fileURLToPath(tsUrl))) {
        return { url: tsUrl.href, shortCircuit: true };
      }
    } catch {
      /* not a file URL, or the .ts sibling does not exist — fall through */
    }
  }
  return nextResolve(specifier, context);
}
