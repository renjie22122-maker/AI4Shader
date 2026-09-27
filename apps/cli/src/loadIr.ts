/**
 * The CLI's single dependency on the rest of the workspace.
 *
 * Two things are loaded, and they are loaded *lazily on purpose*:
 *
 *   1. `@ai4shader/ir/catalog` — `CATALOG`, the node schemas. Without it no port name,
 *      port type or default can be checked, so `validate` / `emit` / `params` / `tune`
 *      would all be guessing.
 *   2. `@ai4shader/ir/targets` — imported for its **side effect**: every target emitter
 *      self-registers through `registerTarget()` at module-eval time, so `emit()` returns
 *      `EM000_UNKNOWN_TARGET` until that module has been evaluated
 *      (`docs/AGENT_CONTRACT.md` §2). Importing only the catalog is the classic way to get
 *      a CLI that "works" and emits nothing.
 *
 * Both directories are owned by other agents and may not exist yet. A static
 * `import { CATALOG } from '@ai4shader/ir/catalog'` would make that a hard crash at module
 * load with a bare `ERR_MODULE_NOT_FOUND`; resolution therefore happens inside a function
 * where it can become a diagnostic a human can act on. The specifiers are the real ones —
 * there is no stub fallback.
 */

import { access } from 'node:fs/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join, resolve } from 'node:path';

/* -------------------------------------------------------------------------- */
/* Structural types                                                            */
/* -------------------------------------------------------------------------- */

/**
 * The IR shapes this CLI needs, declared structurally rather than imported.
 *
 * `import type { ShaderDoc } from '@ai4shader/ir'` looks free — TypeScript erases it — but
 * Node resolves a module's *entire* import graph before it strips types, so a type-only
 * import of a bare specifier is still a hard `ERR_MODULE_NOT_FOUND` when the workspace is
 * not linked. Everything the CLI knows about the IR is therefore declared here, from the
 * frozen contract in `packages/ir/src/types.ts` and `docs/AGENT_CONTRACT.md` §3. The real
 * modules are reached through `loadIr()` below, inside a function, where a failure can be
 * reported instead of thrown at import time.
 *
 * These are deliberately *loose* (`unknown`-tolerant) where a wrong guess would only cost a
 * type error in this package: the compiler enforces the real shapes.
 */

export type ShaderType = 'float' | 'vec2' | 'vec3' | 'vec4' | 'mat2' | 'mat3' | 'int' | 'bool';

export type NodeKind = string;

export type PortSpec = { name: string; type: ShaderType; default?: ParamValue; label?: string };

export type NodeSchema = {
  kind: NodeKind;
  title: string;
  category: string;
  summary: string;
  inputs: PortSpec[];
  outputs: PortSpec[];
  primaryOut?: string;
  glsl: string;
};

export type WireRef = { node: string; port?: string };

export type ParamValue =
  | { kind: 'float'; value: number; min?: number; max?: number; step?: number }
  | { kind: 'int'; value: number; min?: number; max?: number }
  | { kind: 'bool'; value: boolean }
  | { kind: 'vec2'; value: [number, number] }
  | { kind: 'vec3'; value: [number, number, number] }
  | { kind: 'vec4'; value: [number, number, number, number] }
  | { kind: 'color'; value: [number, number, number, number] }
  | {
      kind: 'param';
      name: string;
      type: ShaderType;
      value: number | boolean | number[];
      min?: number;
      max?: number;
      step?: number;
      label?: string;
      group?: string;
    };

export type UniformDecl = {
  name: string;
  type: ShaderType;
  value: number | boolean | number[];
  min?: number;
  max?: number;
  step?: number;
  label?: string;
  group?: string;
  origin: { node: string; param: string };
};

export type ShaderNode = { id: string; kind: NodeKind; params?: Record<string, ParamValue>; inputs?: Record<string, WireRef> };

export type TargetFormat = 'glsl' | 'wgsl' | 'hlsl' | 'msl' | 'shaderlab' | 'ue';

export type Diagnostic = {
  severity: 'error' | 'warning' | 'info';
  code: string;
  message: string;
  at?: { node?: string; port?: string; line?: number; column?: number };
  fix?: string;
};

export type ShaderStage = 'fragment';

export type ShaderDoc = {
  irVersion: string;
  id: string;
  hash?: string;
  title: string;
  createdAt?: string;
  updatedAt?: string;
  spec: { intent: string; category: string; tags: string[]; features: string[]; antiFeatures?: string[]; outOfScope?: string[] };
  stage: string;
  graph: { nodes: ShaderNode[]; edges: { from: WireRef & { port: string }; to: { node: string; port: string } }[] };
  render: { resolution: [number, number]; qualityTier: string; maxIterations: number; transparent: boolean };
  targets: TargetFormat[];
  targetNotes?: Partial<Record<TargetFormat, string>>;
  provenance?: { model?: string; seed?: number; parents?: string[]; steps?: { at: string; actor: string; summary: string }[] };
};

/** The node catalog, keyed by kind. */
export type Catalog = Record<NodeKind, NodeSchema>;

export type Ir = {
  catalog: Catalog;
  /** True when the real catalog *and* the real target emitters were loaded. */
  complete: boolean;
  /** Human-readable note to print when `complete` is false. */
  note?: string;
};

/**
 * A catalog just rich enough to report graph-*structural* problems (duplicate ids, cycles,
 * a missing output node, dangling wires) without the real node definitions.
 *
 * Used only when `packages/ir/src/catalog` is absent, i.e. during parallel development or
 * in a broken checkout. In that mode the CLI still does the work that needs no port
 * information — `params` can list and override uniforms, `validate` reports structure — and
 * it says so loudly on stderr rather than pretending an unverifiable document is clean.
 */
const STRUCTURAL_CATALOG = {
  output: {
    kind: 'output',
    title: 'Image output',
    category: 'output',
    summary: 'Final colour written to the framebuffer.',
    inputs: [{ name: 'color', type: 'vec4' }],
    outputs: [{ name: 'color', type: 'vec4' }],
    primaryOut: 'color',
    glsl: 'return ${color};',
  },
} as unknown as Catalog;
const MISSING_HINT = [
  'catalog/targets are not built yet — run `npm run build` in packages/ir',
  'or install the workspace links with `npm install` at the repo root',
].join(', ');

/** Where the two modules would live in an unbuilt checkout of this monorepo. */
export function irSourceDir(): string {
  // <root>/apps/cli/src/loadIr.ts -> <root>/packages/ir/src
  return resolve(dirname(fileURLToPath(import.meta.url)), '../../../packages/ir/src');
}

async function fileExists(path: string): Promise<boolean> {
  try {
    await access(path);
    return true;
  } catch {
    return false;
  }
}

type Candidate = { specifier: string; probe?: string };

/**
 * Try each candidate in order and return the first that imports.
 *
 * A module that exists but *throws* while evaluating is a real bug in that module and is
 * re-thrown: only "cannot find module" / "unknown file extension" fall through to the next
 * candidate, or a syntax error in the catalog would be reported as "catalog missing".
 */
async function firstImportable(candidates: Candidate[]): Promise<{ mod: unknown; specifier: string } | null> {
  for (const { specifier, probe } of candidates) {
    if (probe !== undefined && !(await fileExists(probe))) continue;
    try {
      return { mod: await import(specifier), specifier };
    } catch (err) {
      const code = (err as { code?: string }).code;
      if (code !== 'ERR_MODULE_NOT_FOUND' && code !== 'ERR_UNKNOWN_FILE_EXTENSION') throw err;
    }
  }
  return null;
}

/**
 * Candidate specifiers for a module, **bare package first**.
 *
 * The order is not cosmetic. `TARGET_EMITTERS` is a module-level map, so registration is only
 * visible to code that reached the *same module instance*. Two different specifiers for the
 * same file can produce two instances:
 *
 *   • `@ai4shader/ir/targets` resolves through `node_modules` to the built `dist/*.js`;
 *   • a `file://` URL to `packages/ir/src/targets/index.ts` loads the TypeScript source.
 *
 * Those are different files, so the process ends up with two registries: the CLI registers
 * into the source one and `@ai4shader/agent` — which imports the bare specifier — looks in the
 * dist one and finds nothing. The symptom is `AG002_TARGETS_NOT_REGISTERED` while every
 * emitter is present and correct, which is a genuinely misleading error.
 *
 * Preferring the bare specifier makes the CLI and the agent agree by construction. The source
 * path stays as a fallback for an unbuilt checkout, where the bare specifier cannot resolve at
 * all; in that mode nothing is consistent, but "degraded with a loud note" is what the rest of
 * this module already promises.
 */
function candidatesFor(subpath: 'catalog/index' | 'targets/index'): Candidate[] {
  const src = join(irSourceDir(), `${subpath}.ts`);
  return [
    { specifier: `@ai4shader/ir/${subpath.split('/')[0]}` },
    { specifier: pathToFileURL(src).href, probe: src },
  ];
}

/** `@ai4shader/ir/catalog`, falling back to the source path in an unbuilt checkout. */
function catalogCandidates(): Candidate[] {
  return candidatesFor('catalog/index');
}

/** `@ai4shader/ir/targets`, falling back to the source path in an unbuilt checkout. */
function targetCandidates(): Candidate[] {
  return candidatesFor('targets/index');
}

/**
 * Load the catalog and register every target emitter.
 *
 * Never throws for a missing module: the caller decides whether the missing half is fatal
 * (`emit`, `tune`) or merely degrades the answer (`validate`, `params`).
 */
export async function loadIr(): Promise<Ir> {
  const catalogMod = await firstImportable(catalogCandidates());
  if (!catalogMod) {
    return { catalog: STRUCTURAL_CATALOG, complete: false, note: `structural-only catalog: ${MISSING_HINT}` };
  }

  const catalog = (catalogMod.mod as { CATALOG?: Catalog }).CATALOG;
  if (!catalog || typeof catalog !== 'object') {
    throw new Error(
      `\`${catalogMod.specifier}\` loaded but exports no \`CATALOG\` (AGENT_CONTRACT.md §3: the catalog module must re-export it from \`./nodes.js\`).`,
    );
  }

  const targetMod = await firstImportable(targetCandidates());
  if (!targetMod) {
    return {
      catalog,
      complete: false,
      note: `no target emitters: ${MISSING_HINT}. \`emit\` would report EM000_UNKNOWN_TARGET for every target.`,
    };
  }

  return { catalog, complete: true };
}

/** Load the catalog and emitters, or throw the actionable message that `emit` deserves. */
export async function requireIr(): Promise<Ir> {
  const ir = await loadIr();
  if (!ir.complete) {
    throw new Error(
      `cannot emit without the node catalog and the target emitters — ${MISSING_HINT}\n` +
        `  looked in: ${irSourceDir()}\n` +
        `  and for the \`@ai4shader/ir\` package (exports "./catalog", "./targets")`,
    );
  }
  return ir;
}
