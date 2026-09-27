/**
 * Catalog loading
 * ===============
 *
 * The catalog and the six target emitters are **static imports**. There is no bundler in
 * the loop and no `npm install`: `apps/web/src/*.ts` is projected to `apps/web/dist/*.js`
 * by `scripts/project-ts.ts` (Node's own type stripper), and the browser loads those
 * files directly. That only works if every specifier is a real, resolvable path — which
 * is why the IR is imported by its projected location (`packages/ir/dist/...`) rather
 * than by the bare specifier `@ai4shader/ir`.
 *
 * Importing the targets module is a **side-effect import**: each emitter calls
 * `registerTarget()` at module scope, which is what makes `emit()` able to do anything
 * at all. If that import is ever dropped, every target starts returning
 * `EM000_UNKNOWN_TARGET` — so the check below treats an empty `TARGET_EMITTERS` as a
 * condition to report loudly rather than a state to tolerate.
 *
 * The one thing this module still guards is *content*: a catalog that is present but
 * missing kinds must produce a visible banner, not a blank canvas. That check now reads
 * the imported object at runtime instead of relying on a conditional import.
 */

import {
  CATALOG,
  CATEGORIES,
  NODE_KINDS,
  TARGET_EMITTERS,
  type CatalogCategorySummary,
} from '../ir.js';

import type { NodeKind, NodeSchema, TargetFormat } from '../../../../packages/ir/src/index.js';

export type CatalogLoad = {
  /** What the catalog provides. In practice complete; `emit()` reports missing kinds. */
  catalog: Partial<Record<NodeKind, NodeSchema>>;
  present: NodeKind[];
  missing: NodeKind[];
  /** Module paths that contributed, for the info panel. */
  sources: string[];
  /** Target emitters that registered themselves on import. */
  targets: TargetFormat[];
  /** Palette grouping from the catalog, for the info panel. */
  categories: readonly CatalogCategorySummary[];
  notes: string[];
};

const CATALOG_MODULE = 'packages/ir/dist/catalog/index.js';
const TARGETS_MODULE = 'packages/ir/dist/targets/index.js';

export function loadCatalog(): CatalogLoad {
  const notes: string[] = [];
  const catalog: Partial<Record<NodeKind, NodeSchema>> = CATALOG;

  const present = NODE_KINDS.filter((k) => catalog[k] !== undefined);
  const missing = NODE_KINDS.filter((k) => catalog[k] === undefined);

  if (present.length === 0) {
    notes.push(`${CATALOG_MODULE} 没有导出任何节点定义：滑杆面板与代码视图会保持为空。`);
  } else if (missing.length > 0) {
    notes.push(`catalog 缺少 ${missing.length} 种节点：${missing.join(', ')}`);
  }

  const targets = Object.keys(TARGET_EMITTERS) as TargetFormat[];
  if (targets.length === 0) {
    notes.push(
      `${TARGETS_MODULE} 未注册任何目标发射器：emit() 会返回 EM000_UNKNOWN_TARGET，代码视图会显示该诊断而不是空白。`,
    );
  }

  return {
    catalog,
    present,
    missing,
    sources: present.length ? [CATALOG_MODULE] : [],
    targets,
    categories: CATEGORIES,
    notes,
  };
}

/**
 * Widen a (possibly partial) catalog to the full record `emit()` asks for.
 *
 * Safe by construction: a kind that is genuinely absent produces a precise
 * `IR002_UNKNOWN_KIND` diagnostic from the emitter the moment a document uses it — which
 * is a better error than anything this function could invent. The cast only silences the
 * type checker about a gap the runtime already reports.
 */
export function asEmitCatalog(
  catalog: Partial<Record<NodeKind, NodeSchema>>,
): Record<NodeKind, NodeSchema> {
  return catalog as Record<NodeKind, NodeSchema>;
}
