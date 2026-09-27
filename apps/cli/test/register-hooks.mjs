/**
 * Registers the `.js` → `.ts` resolver hook (see `ts-resolve.mjs`) for the test run.
 *
 *     node --test --experimental-strip-types --import ./test/register-hooks.mjs \
 *       test/targets.test.ts
 *
 * Kept as a separate file because `module.register()` must run before the module graph
 * of the test file is resolved.
 */
import { register } from 'node:module';

register('./ts-resolve.mjs', import.meta.url);
