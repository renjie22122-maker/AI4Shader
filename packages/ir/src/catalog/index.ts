/**
 * `@ai4shader/ir/catalog` — the node catalog aggregator.
 *
 * Pure re-export, no logic: hosts (CLI, browser workbench, Vite alias
 * `@ai4shader/ir/catalog`) resolve this file and get the frozen API:
 *
 *   CATALOG     — every kind in `NODE_KINDS`, keyed by kind
 *   CATEGORIES  — the same kinds grouped for the palette
 *
 * Port names, defaults and slider ranges live ONLY in `./nodes.js`; never
 * hard-code a port name anywhere else (see docs/AGENT_CONTRACT.md §1).
 */

export { CATALOG, CATEGORIES } from './nodes.js';
export type { CatalogCategory } from './nodes.js';
