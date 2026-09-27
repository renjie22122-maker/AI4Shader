// Registers the `.js` -> `.ts` resolver hook. Used by every `node` invocation in this
// package: see `apps/cli/.node/ts-resolve.mjs` for why it is needed.
import { register } from 'node:module';

register('./ts-resolve.mjs', import.meta.url);
