/**
 * Ambient declarations.
 *
 * `apps/web/src` is browser code and is projected to `dist/*.js` by
 * `scripts/project-ts.ts`; nothing under `src/` may import from `node:*`. The single
 * declaration below exists only for the **optional** Vite path (`vite.config.ts`, for
 * anyone who has npm installed), which runs in Node and needs `fileURLToPath`.
 *
 * The primary, zero-install path needs none of this: the browser loads
 * `packages/ir/dist/*.js` and `apps/web/dist/*.js` as plain ES modules.
 */

declare module 'node:url' {
  /** Convert a `file:` URL (string or `URL`) to a platform-native path. */
  export function fileURLToPath(url: string | URL): string;
}
