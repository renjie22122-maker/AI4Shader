/**
 * Regenerate the checked-in example documents from their TypeScript sources of truth.
 *
 * The checked-in examples exist so that a new user can run
 *   `ai4shader emit examples/liquid-metal.shader.json --target all`
 * without a build step. But they are *generated*, never hand-edited: two copies of the
 * same document is exactly the drift this project exists to eliminate, so the
 * TypeScript source and the JSON are kept in sync mechanically instead of by discipline.
 *
 *   npm run build          # projects src/*.ts -> dist/*.js  (must run first)
 *   npm run build:demo     # this script
 *
 * It imports from `dist/`, not `src/`, because Node resolves `./x.js` against real files
 * and the authored sources are `.ts`. Running the projection first is not a workaround —
 * it is the same artifact the CLI and the browser load, so the examples are validated
 * against exactly what ships.
 *
 * Two documents, because one document cannot be both shapes:
 *
 *   liquid-metal    a **quad** shader — it reads `uv` and writes a colour.
 *   toon-surface    a **surface** shader — it reads the interpolated normal / world position /
 *                   view direction and computes lighting. Every target emits a different entry
 *                   point for it, which is why the Python suite compiles its HLSL *twice*
 *                   (`VSMain` against `vs_6_0`, `PSMain` against `ps_6_0`).
 *
 * The script re-derives `graph.edges` from `nodes[].inputs` via `normalizeEdges()` and
 * fails loudly if an example references a node kind or input port that no longer
 * exists — a stale example is worse than no example.
 */

import { writeFile, mkdir } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

import { createLiquidMetalDoc } from '../packages/ir/dist/examples/liquidMetal.js';
import { createToonSurfaceDoc } from '../packages/ir/dist/examples/toonSurface.js';
import { createFogWipeDoc } from '../packages/ir/dist/examples/fogWipe.js';
import { createLotusPondDoc } from '../packages/ir/dist/examples/lotusPond.js';
import { normalizeEdges, canonicalJson, hashDoc } from '../packages/ir/dist/patcher.js';
import { CATALOG } from '../packages/ir/dist/catalog/index.js';
import { analyzeGraph, emit } from '../packages/ir/dist/emitter.js';
// Side effect only: each target emitter registers itself through `registerTarget()` at module
// scope. Without this, `emit()` reports EM000_UNKNOWN_TARGET for every target and the check
// below would call that a broken example.
import '../packages/ir/dist/targets/index.js';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..');

const EXAMPLES = [
  { file: 'liquid-metal.shader.json', build: createLiquidMetalDoc, surface: false, channel: false },
  { file: 'toon-surface.shader.json', build: createToonSurfaceDoc, surface: true, channel: false },
  // The one that needs a *host* feature rather than a language one: `render.feedback` asks for the
  // previous frame in the image channel. The `channel` expectation below is what keeps the flag
  // and the graph from drifting apart — `feedback: true` on a document that never samples the
  // channel is an IR036 warning, and this script would refuse to write it.
  { file: 'fog-wipe.shader.json', build: createFogWipeDoc, surface: false, channel: true },
  // The scene document: 40 nodes, five layers. Nothing about it is a new *capability* — it is the
  // same primitives composed — which is exactly why it is worth checking in: it is the evidence
  // that the catalog composes, and the thing to point at when someone asks whether the tool can
  // build a scene rather than an effect.
  { file: 'lotus-pond.shader.json', build: createLotusPondDoc, surface: false, channel: false },
];

let failed = false;

for (const example of EXAMPLES) {
  const outPath = resolve(repoRoot, 'examples', example.file);
  const doc = example.build();
  doc.graph = normalizeEdges(doc.graph);
  doc.hash = hashDoc(doc);

  // Validate against the live catalog before writing. A generated example that does not
  // load is a trap for the next person, so this is a hard failure rather than a warning.
  const analysis = analyzeGraph(doc.graph, CATALOG);
  const errors = analysis.diagnostics.filter((d) => d.severity === 'error');
  if (errors.length) {
    process.stderr.write(`✗ ${example.file} is invalid against the current catalog:\n`);
    for (const e of errors) {
      process.stderr.write(`  [${e.code}] ${e.message}\n`);
      if (e.fix) process.stderr.write(`      fix: ${e.fix}\n`);
    }
    failed = true;
    continue;
  }
  const warnings = analysis.diagnostics.filter((d) => d.severity === 'warning');
  if (warnings.length) {
    process.stderr.write(`⚠ ${example.file}: ${warnings.length} warning(s)\n`);
    for (const w of warnings) process.stderr.write(`  [${w.code}] ${w.message}\n`);
  }

  /*
   * The shape of the document is a claim, and a claim that can be checked here rather than
   * discovered by whoever next opens the file. A quad example that quietly grew a surface
   * input would invalidate the Python suite's assumption that `shader.frag` compiles as a
   * standalone program; a surface example that lost one would silently stop testing the
   * second entry point.
   */
  const glsl = emit(doc, { catalog: CATALOG, target: 'glsl' });
  const emitErrors = glsl.diagnostics.filter((d) => d.severity === 'error');
  if (emitErrors.length) {
    process.stderr.write(`✗ ${example.file} does not emit:\n`);
    for (const e of emitErrors) process.stderr.write(`  [${e.code}] ${e.message}\n`);
    failed = true;
    continue;
  }
  if (glsl.needs.surface !== example.surface) {
    process.stderr.write(
      `✗ ${example.file}: expected needs.surface=${example.surface}, got ${glsl.needs.surface}\n`,
    );
    failed = true;
    continue;
  }
  if (glsl.needs.channel !== example.channel) {
    process.stderr.write(
      `✗ ${example.file}: expected needs.channel=${example.channel}, got ${glsl.needs.channel}` +
        `${example.channel ? ' — render.feedback would be a request nothing can answer' : ''}\n`,
    );
    failed = true;
    continue;
  }

  await mkdir(dirname(outPath), { recursive: true });
  // Canonical (key-sorted) JSON: a regenerated file produces an empty git diff when
  // nothing semantically changed. Pretty-printing is for humans reading the example.
  const json = JSON.stringify(JSON.parse(canonicalJson(doc)), null, 2) + '\n';
  await writeFile(outPath, json, 'utf8');

  const uniformCount = new Set(
    doc.graph.nodes.flatMap((n) =>
      Object.values(n.params ?? {})
        .filter((p) => p.kind === 'param')
        .map((p) => (p as { name: string }).name),
    ),
  ).size;

  process.stdout.write(
    `✓ wrote ${outPath}\n` +
      `  nodes: ${doc.graph.nodes.length}   uniforms: ${uniformCount}   ` +
      `surface: ${glsl.needs.surface}   channel: ${glsl.needs.channel}   hash: ${doc.hash}\n`,
  );
}

if (failed) process.exit(1);
