/**
 * Typed facade over the projected IR
 * ==================================
 *
 * The workbench runs the **projected** IR (`packages/ir/dist/*.js`, produced by
 * `scripts/project-ts.ts`) because that is what a browser can load, but JavaScript
 * carries no type information: a checker reading the projection sees `kind: string`
 * where the real type is `kind: NodeKind`, and `string | undefined` where the source
 * has a non-null assertion.
 *
 * Rather than sprinkle casts across every call site, this module reconciles the two
 * exactly once:
 *
 *   • values come from `dist` — the bytes the browser and Node execute;
 *   • the *type* of those values comes from `src` via `typeof import(...)`, which is a
 *     type-only construct and is erased by the projection.
 *
 * One documented double cast, in one file, and every other module gets precisely typed
 * `emit`, `hashDoc`, `applyPatch`, `CATALOG`, … with no casts at all.
 *
 * Types themselves are imported directly from `packages/ir/src/index.js` by the modules
 * that need them — an `import type` is erased, so it has no runtime cost and no bearing
 * on which file the browser loads.
 */

import * as irRuntime from '../../../packages/ir/dist/index.js';
import * as catalogRuntime from '../../../packages/ir/dist/catalog/index.js';

// Side effect only: each target emitter registers itself through `registerTarget()` at
// module scope. Without this import every `emit()` returns `EM000_UNKNOWN_TARGET`.
import '../../../packages/ir/dist/targets/index.js';

import type { NodeKind, NodeSchema } from '../../../packages/ir/src/index.js';

/** The authored IR module type — precise, and unavailable to the projection. */
type IrModule = typeof import('../../../packages/ir/src/index.js');

const ir = irRuntime as unknown as IrModule;

export const { emit, hashDoc, applyPatch, stableHash, canConnect, IR_VERSION, TARGET_EMITTERS, analyzeGraph } = ir;

/**
 * The catalog is its own module (content lives outside the engine) so it is re-typed
 * here as well. The projection widens literal types (`kind: string`), which is exactly
 * the gap these casts bridge — hence `unknown` in the middle of each one.
 *
 * `CATEGORIES` is typed structurally instead of with the catalog's own `CatalogCategory`:
 * importing that type would pull the entire 900-line catalog module into the checker's
 * program, and the workbench only ever reads three fields of it.
 */
export type CatalogCategorySummary = {
  id: string;
  title: string;
  summary: string;
  kinds: readonly string[];
};

export const CATALOG = catalogRuntime.CATALOG as unknown as Record<NodeKind, NodeSchema>;
export const CATEGORIES = catalogRuntime.CATEGORIES as unknown as readonly CatalogCategorySummary[];
export const NODE_KINDS = ir.NODE_KINDS;
