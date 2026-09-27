/**
 * `@ai4shader/ir` — public surface of the core package.
 *
 * Deliberately narrow. Everything a host needs to (a) construct a document,
 * (b) analyse it, (c) emit source for any target, and (d) patch it, is re-exported
 * here. Anything not here is an implementation detail and may move.
 *
 * This file is the package's only aggregator: adding an export here changes the
 * public API for the CLI, the browser workbench, the Unity plugin's JSON bridge and
 * the Python batch driver at once. Add with intent.
 */

export * from './types.js';
export * from './emitter.js';
export * from './rewrite.js';
export * from './glslSubset.js';
export * from './patcher.js';
