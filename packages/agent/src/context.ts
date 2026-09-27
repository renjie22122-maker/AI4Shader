/**
 * AI4Shader — Model-facing context projection
 * ==========================================
 *
 * Everything the model is allowed to see, assembled from the live document.
 *
 * The temptation is to send the whole `ShaderDoc` as JSON — it is the source of truth, it
 * already exists, and it costs nothing to `JSON.stringify`. That is the wrong trade for
 * three compounding reasons:
 *
 *   • **Cost and latency scale with the prompt.** A real document is several kB of nested
 *     JSON; a projection is a dozen lines. The loop makes many calls, so this multiplies.
 *   • **Every field is an invitation.** Given `render.maxIterations` and `qualityTier`, a
 *     model will sometimes "helpfully" change them. Given only what a task needs, it cannot.
 *   • **The projection is a contract.** Lines like `n_warp (domain_warp)` are checkable
 *     against the catalog, so a hallucination is visible at a glance — to the model while
 *     writing, and to a human reading the trace afterwards.
 *
 * The projection is **lossy on purpose**. If a future task needs a field, it gets added
 * here deliberately, which is the point: the model's view of the document becomes an
 * explicit, reviewable decision instead of an accident of serialisation.
 */

import type { Diagnostic, ShaderDoc, UniformDecl } from '@ai4shader/ir/types';
import { collectUniforms } from '@ai4shader/ir';
import type { CatalogLike, PatchContext } from './types.js';

/** Everything handed to a provider in one call, already rendered. */
export type PromptBundle = {
  context: PatchContext;
  system: string;
  user: string;
};

/**
 * Build the context for a provider call.
 *
 * @param stage what is being asked for — the instructions differ, the data does not
 */
export function buildContext(args: {
  stage: PatchContext['stage'];
  doc: ShaderDoc;
  catalog: CatalogLike;
  diagnostics?: readonly Diagnostic[];
  userIntent?: string;
  /** Extra guidance from the repairer, e.g. fixes it recognised but cannot apply. */
  advice?: readonly string[];
}): PatchContext {
  const { stage, doc, catalog, diagnostics = [], userIntent, advice = [] } = args;

  return {
    stage,
    doc,
    nodeLines: nodeInventory(doc, catalog),
    uniformLines: uniformInventory(doc),
    // Advice rides along inside the diagnostic list the model sees. Both are "things to
    // address", and separating them would make the prompt describe two problems when the
    // user perceives one.
    diagnostics: [...diagnostics, ...advice.map(adviceToDiagnostic)],
    ...(userIntent !== undefined ? { userIntent } : {}),
  };
}

/**
 * One line per node: `id (kind) -> out:port:type; in:port=source`.
 *
 * Shows the wiring inline rather than as a separate edge list, because the model needs to
 * answer "is this input already connected?" and a single line answers it without a join.
 */
export function nodeInventory(doc: ShaderDoc, catalog: CatalogLike): string[] {
  return doc.graph.nodes.map((n) => {
    const schema = catalog[n.kind];
    const outs = schema
      ? schema.outputs.map((p) => `${p.name}:${p.type}`).join(' ')
      : '?';
    const ins = Object.entries(n.inputs ?? {})
      .map(([port, wire]) => `${port}=${wire.node}${wire.port ? `.${wire.port}` : ''}`)
      .join(' ');
    const params = Object.entries(n.params ?? {})
      .filter(([, v]) => v.kind !== 'param')
      .map(([k, v]) => `${k}=${literalSummary(v)}`)
      .join(' ');
    const bits = [`${n.id} (${n.kind})`, `out[${outs}]`];
    if (ins) bits.push(`in[${ins}]`);
    if (params) bits.push(`lit[${params}]`);
    if (n.note) bits.push(`// ${n.note}`);
    return bits.join(' ');
  });
}

/**
 * One line per tunable: `name type=value [min..max] label — origin`.
 *
 * The origin (`node.port`) matters: it is what lets the model reason about *which* part of
 * the effect a slider controls, and it is what the UI uses to highlight the node.
 */
export function uniformInventory(doc: ShaderDoc): string[] {
  const { uniforms } = collectUniforms(doc.graph, doc.graph.nodes.map((n) => n.id), new Map(doc.graph.nodes.map((n) => [n.id, n])));
  return uniforms.map(describeUniform);
}

function describeUniform(u: UniformDecl): string {
  const value = Array.isArray(u.value) ? `[${u.value.join(', ')}]` : String(u.value);
  const range =
    u.min !== undefined && u.max !== undefined ? ` range[${u.min}..${u.max}]` : '';
  const label = u.label ? ` "${u.label}"` : '';
  const group = u.group ? ` group:${u.group}` : '';
  return `${u.name} ${u.type}=${value}${range}${label}${group} — ${u.origin.node}.${u.origin.param}`;
}

function literalSummary(v: { kind: string; value?: unknown; name?: string }): string {
  if (v.kind === 'param') return `param:${v.name}`;
  const value = v.value;
  return Array.isArray(value) ? `[${value.join(',')}]` : String(value);
}

/** Repairs the repairer recognised but could not apply are surfaced as advisory items. */
function adviceToDiagnostic(advice: string): Diagnostic {
  return {
    severity: 'info',
    code: 'AG100_REPAIR_ADVICE',
    message: advice,
    fix: 'Apply this guidance; it comes from the deterministic repairer, not from a model.',
  };
}

/* -------------------------------------------------------------------------- */
/* Prompt rendering                                                            */
/* -------------------------------------------------------------------------- */

/** Render a context as the user-turn text. Deterministic: same context -> same string. */
export function renderContext(context: PatchContext): string {
  const parts: string[] = [];

  if (context.userIntent) {
    parts.push(`## What the user asked for`, context.userIntent, '');
  }

  parts.push(
    `## Current document`,
    `title: ${context.doc.title}`,
    `stage: ${context.doc.stage}`,
    `quality: ${context.doc.render.qualityTier}  maxIterations: ${context.doc.render.maxIterations}  resolution: ${context.doc.render.resolution.join('x')}`,
    `targets: ${context.doc.targets.join(', ')}`,
    `transparent: ${context.doc.render.transparent}  feedback: ${context.doc.render.feedback === true}`,
    '',
    `### Intent`,
    `intent: ${context.doc.spec.intent}`,
    `category: ${context.doc.spec.category}`,
    context.doc.spec.features.length ? `features:\n${context.doc.spec.features.map((f) => `  - ${f}`).join('\n')}` : 'features: (none)',
    context.doc.spec.antiFeatures?.length
      ? `anti-features (must NOT appear):\n${context.doc.spec.antiFeatures.map((f) => `  - ${f}`).join('\n')}`
      : '',
    '',
    `### Nodes (${context.nodeLines.length})`,
    ...(context.nodeLines.length ? context.nodeLines.map((l) => `  ${l}`) : ['  (none yet)']),
    ...context.doc.graph.nodes.filter(n => n.kind === 'custom_glsl' && n.body !== undefined)
      .map(n => `custom_glsl ${n.id} body:\n${n.body}`),
    '',
    `### Tunable uniforms (${context.uniformLines.length})`,
    ...(context.uniformLines.length ? context.uniformLines.map((l) => `  ${l}`) : ['  (none yet)']),
  );

  if (context.diagnostics.length) {
    parts.push(
      '',
      `### Problems to fix (${context.diagnostics.length})`,
      ...context.diagnostics.map(
        (d) => `  [${d.severity}] ${d.code}: ${d.message}${d.fix ? `\n      fix: ${d.fix}` : ''}`,
      ),
    );
  }

  return parts.filter((p) => p !== '').join('\n');
}

/**
 * Turn a context into the three pieces a provider needs, given a system prompt.
 * Kept separate from the prompt text itself so the wording lives in one place per provider.
 */
export function bundle(context: PatchContext, system: string): PromptBundle {
  return { context, system, user: renderContext(context) };
}
