/**
 * AI4Shader — Deterministic, catalog-aware repair
 * ==============================================
 *
 * Turns diagnostics into patches. **No model is involved**, and that is the point.
 *
 * A `IR006_TYPE_MISMATCH` already states which port is wrong and what the right type is; a
 * `IR031_MISSING_INPUT` names the node and the port. Both are answerable by a lookup over
 * the catalog. Routing them to a language model costs a round trip, adds latency the user
 * feels, and introduces the possibility of the model "fixing" something that was not
 * broken. So the repairer runs first, and the model only ever sees what the repairer
 * genuinely cannot do.
 *
 * ## The three outcomes
 *
 *   `patch`     — the fix is known. Apply it and re-validate. Free.
 *   `advice`    — the problem is recognised but not mechanically fixable. The guidance text
 *                 is handed to the model, which is strictly better than handing it the raw
 *                 diagnostic and hoping it infers the same thing.
 *   `hopeless`  — no patch can help (a missing node kind, an unreachable graph). The loop
 *                 must escalate instead of spinning. Recognising these is what stops the
 *                 loop burning its budget on a problem that is not in the document.
 *
 * ## Why every patch here is the smallest one that helps
 *
 * The repairer is called repeatedly, and a patch that changes three things makes it
 * impossible to attribute the next diagnostic to any of them. So each action touches one
 * node or one wire, and the `why` field records the diagnostic code that motivated it —
 * which is what the trace in the UI shows.
 */

import type {
  Diagnostic,
  NodeKind,
  NodeSchema,
  ParamValue,
  ShaderDoc,
  ShaderNode,
  ShaderPatch,
  ShaderType,
  PatchOp,
  WireRef,
} from '@ai4shader/ir/types';
import { isReservedAnywhere, makeNodeId, safeIdentifier } from '@ai4shader/ir';
import { unrecoverableReason } from './policy.js';
import type { CatalogLike, Repair, RepairAction } from './types.js';

/* -------------------------------------------------------------------------- */
/* Entry point                                                                 */
/* -------------------------------------------------------------------------- */

/**
 * Build the repair plan for a set of diagnostics.
 *
 * Returns actions in the order they should be attempted, plus the diagnostics the repairer
 * does not understand. Actions are *not* applied here: the loop applies them so that a
 * rejected patch is attributed to the round that produced it.
 */
export function repair(doc: ShaderDoc, diagnostics: readonly Diagnostic[], catalog: CatalogLike): Repair {
  const actions: RepairAction[] = [];
  const unknown: Diagnostic[] = [];

  /*
   * One repair per **site**, not per (site, code).
   *
   * A single defect is routinely reported under two codes — a port that receives the wrong type
   * produces both `IR006_TYPE_MISMATCH` and `IR032_NO_COERCION` — and two repairs aimed at one
   * port do not compose: they fight. The version of this key that included the code produced a
   * plan that inserted a joiner *and* deleted the wire, and because the deletion was applied
   * last it won, leaving a document that emitted cleanly with the effect quietly missing.
   *
   * The site is the unit of repair because a port can only hold one wire. Node-level
   * diagnostics (no port) keep the code in their key, so a cycle and a dead-node report on the
   * same node are both planned.
   */
  const seen = new Set<string>();
  const queue = diagnostics.filter((d) => d.severity === 'error');

  // A pathological document (an LLM redraft gone wrong) can carry hundreds of diagnostics.
  // Planning for all of them is wasted work: the first handful of fixes usually eliminate
  // the rest, because most errors are downstream of a few missing wires.
  const MAX_ACTIONS = 12;
  let considered = 0;

  for (const d of queue) {
    const node = d.at?.node ?? '-';
    const port = d.at?.port;
    const key = port ? `${node}|${port}` : `${d.code}|${node}|-`;
    if (seen.has(key)) continue;
    seen.add(key);

    const hopeless = unrecoverableReason(d.code);
    if (hopeless) {
      actions.push({ kind: 'hopeless', reason: `${d.code}: ${hopeless}` });
      continue;
    }

    if (considered >= MAX_ACTIONS) {
      unknown.push(d);
      continue;
    }
    considered++;

    const action = repairOne(doc, d, catalog);
    if (action) actions.push(action);
    else unknown.push(d);
  }

  return { actions, unknown };
}

/** Back-compat alias: `classify` was the original name for the same entry point. */
export const classify = repair;

/* -------------------------------------------------------------------------- */
/* Per-code handlers                                                          */
/* -------------------------------------------------------------------------- */

function repairOne(doc: ShaderDoc, d: Diagnostic, catalog: CatalogLike): RepairAction | null {
  switch (d.code) {
    /* ---- wiring problems: the fix is to remove or move the wire --------------- */
    case 'IR003_UNKNOWN_PORT':
    case 'IR004_DANGLING_WIRE':
    case 'IR005_UNKNOWN_OUTPUT':
      return dropWire(d);

    case 'IR032_NO_COERCION':
      // The same defect as `IR006_TYPE_MISMATCH`, reported a second time at the same port: "no
      // known conversion from vec2 to vec4" *is* the mismatch. It used to be answered with
      // `dropWire`, which meant one port received two opposing repairs in the same round —
      // insert the joiner, then delete the wire — and the deletion, applied last, won. The
      // document came out "clean" with the effect gone, which is the exact failure the
      // constructive repair was written to end. Same handler, one repair, no race.
      return fixTypeMismatch(doc, d, catalog);

    case 'IR006_TYPE_MISMATCH':
      return fixTypeMismatch(doc, d, catalog);

    case 'IR012_PARAM_TYPE':
    case 'IR013_LITERAL_TYPE':
      return fixParamType(doc, d, catalog);

    case 'IR031_MISSING_INPUT':
      return fillMissingInput(doc, d, catalog);

    case 'IR007_CYCLE':
      return breakCycle(doc, d);

    case 'IR020_UNIFORM_TYPE_CONFLICT':
      return renameUniform(doc, d);

    case 'IR033_BAD_MULTI_OUTPUT':
      return {
        kind: 'advice',
        advice:
          `Node ${d.at?.node ?? '?'} declares multiple output ports, which the emitter ` +
          `supports only as a single vec4 with at most four scalar ports. Replace it with ` +
          `one node per output, or pick a catalog node that has a single output.`,
        why: 'the multi-output contract is a schema-level constraint, not a graph edit',
      };

    /* ---- dead weight: removing it cannot change the image --------------------- */
    case 'IR011_DEAD_NODES':
      return removeDeadNodes(d);

    /* ---- a rejected patch: the complaint is precise, the fix needs judgement ------ */
    //
    // These arrive from `applyPatch` when a *proposed* patch was refused, which is a
    // different situation from a document that does not emit. The diagnostic names the
    // offending op, so the repairer cannot act — but it must NOT classify them as hopeless,
    // or the loop stops before giving the model the one piece of information it needs.
    case 'PT001_UNKNOWN_NODE':
    case 'PT016_DANGLING_WIRE':
      return {
        kind: 'advice',
        advice:
          `${d.message} Fix the referenced node: either add the missing node with ` +
          `\`add_node\`, or point the wire at a node that exists. Do not reference ids that ` +
          `are not in the node list above.`,
        why: 'the patch named a node that does not exist; only the patch author knows what was meant',
      };

    case 'PT014_TYPE_MISMATCH':
      return {
        kind: 'advice',
        advice:
          `${d.message} Point the wire at an output whose type matches the input, or satisfy ` +
          `the input with a literal/uniform of the required type instead of a wire.`,
        why: 'the patch wired incompatible types; the intended source is the author\'s to choose',
      };

    case 'PT011_NOT_ATOMIC':
      return {
        kind: 'advice',
        advice:
          `Patch rejected in full because of the op named above. Emit a corrected patch ` +
          `containing ONLY legal ops; a patch is all-or-nothing, so one bad op discards the rest.`,
        why: 'the rejection is the wrapper around a specific op failure, reported separately',
      };

    /* ---- a graph with no sink: add one, and wire the best source into it ------ */
    case 'IR008_NO_OUTPUT':
    case 'EM001_NO_OUTPUT_NODE':
      return addOutputNode(doc, d, catalog);

    /* ---- everything else is the model's problem ------------------------------- */
    default:
      return null;
  }
}

/**
 * No `output` node: add one, and wire the most plausible source into it.
 *
 * This used to be classified `hopeless`, with a comment arguing that "no edit derived from a
 * diagnostic can invent one — the graph has to be built, which is a drafting job". That
 * reasoning was wrong, and a real DeepSeek draft paid for it: the model produced a perfectly
 * good graph of six nodes and simply forgot the sink. The loop called it `unrecoverable`
 * after three model calls and handed back a flat colour.
 *
 * Adding a sink is not a drafting job. It is: pick the catalog's `output` kind, give it a
 * fresh id, and wire the best available signal into its input — the same shape of edit
 * `fillMissingInput` has always done for every other unwired port. The `output` node is only
 * special in that its absence produces a *document-level* diagnostic instead of a
 * node-and-port one, and the old code mistook "the diagnostic does not name a port" for
 * "there is nothing to do".
 *
 * ## Choosing the source
 *
 * The graph cannot say which node was *meant* to be the final image, so this picks the most
 * defensible answer rather than a clever one:
 *
 *   1. only **sinks** — a node nothing else consumes. A node that feeds another node is
 *      mid-graph by construction, and promoting it to the output would silently change the
 *      image the author was building;
 *   2. among those, the one with the **greatest depth** in the graph, because that is the
 *      end of the longest chain of work;
 *   3. ties broken by id, so two runs over the same document produce the same patch. A
 *      repairer that picks differently each run makes a trace impossible to reproduce.
 *
 * Ties in *type* are resolved in favour of the port type `output` actually wants (vec4),
 * then wider-is-better. When the graph has no sink at all — an empty document, or a pure
 * chain that already ends somewhere — the node is still added, with its own default, because
 * a document with an `output` and a flat colour is something the user can see, tune and
 * extend, and a document with no `output` is a dead end that emits nothing.
 */
function addOutputNode(doc: ShaderDoc, d: Diagnostic, catalog: CatalogLike): RepairAction | null {
  const schema = catalog['output'];
  if (!schema) {
    return {
      kind: 'hopeless',
      reason: `${d.code}: the catalog has no \`output\` kind, so no patch can supply one`,
    };
  }
  const input = schema.inputs[0];
  if (!input) {
    return {
      kind: 'hopeless',
      reason: `${d.code}: the catalog's \`output\` kind declares no input port to wire into`,
    };
  }

  /*
   * An empty graph is a failed draft, not a missing sink.
   *
   * This guard is load-bearing and was found by a test, not by reasoning. A draft patch that
   * is rejected atomically — the model wired `output.color` from a node it never added —
   * leaves the document with *nothing in it*. Adding a lone `output` to that makes the
   * emitter report success on a flat colour, the seeded `PT001_UNKNOWN_NODE` is discarded
   * because "a clean emission wins over a seed", and the user is told the run succeeded while
   * looking at a black frame. The loop's oldest and best-tested behaviour is to carry that
   * rejection back to the model, and the sink repair must not paper over it.
   *
   * So the repair applies only when there is a graph to attach a sink to. Otherwise this is
   * `advice`: progress-capable, so the model is asked again, and honest when there is no
   * model to ask.
   */
  if (!doc.graph.nodes.some((n) => n.kind !== 'output')) {
    return {
      kind: 'advice',
      advice:
        `${d.code} is the symptom, not the problem: the document has no nodes to emit. ` +
        `Build the graph first — start from \`uv\`, add the nodes that produce the look, and ` +
        `finish by wiring one of them into an \`output\` node's \`color\` input.`,
      why: 'adding a sink to an empty graph emits a flat colour that would be reported as success',
    };
  }

  const taken = doc.graph.nodes.map((n) => n.id);
  const id = makeNodeId('output', taken);

  // `applyPatch` fills every declared input from the catalog, so `{ id, kind }` is a complete
  // node here — the same shape `add_node` takes from a model-authored patch.
  const op: PatchOp = {
    op: 'add_node',
    node: { id, kind: 'output' } as ShaderNode,
    reason: d.message,
  };

  const source = pickOutputSource(doc, catalog, input.type);
  if (!source) {
    return {
      kind: 'patch',
      why: `add \`${id}\` with no source: the graph has no sink to wire into it (${d.code})`,
      patch: {
        summary: 'Add the missing output node',
        ops: [op],
      },
    };
  }

  const wire: PatchOp = {
    op: 'connect',
    from: { node: source.id, port: source.port },
    to: { node: id, port: input.name },
    reason: `${d.code}: no output node; wired the deepest unconsumed node`,
  };

  return {
    kind: 'patch',
    why: `add \`${id}\` and wire ${source.id}.${source.port} into it (${d.code})`,
    patch: {
      summary: `Add the missing output node and wire ${source.id}.${source.port}`,
      ops: [op, wire],
    },
  };
}

/**
 * The node most likely to be the intended final image: the deepest unconsumed one.
 *
 * Exported for the test that pins the choice, because "which node did it pick?" is the whole
 * substance of this repair and a wrong answer is silent — it renders, it just renders the
 * wrong thing.
 */
export function pickOutputSource(
  doc: ShaderDoc,
  catalog: CatalogLike,
  wanted: ShaderType,
): { id: string; port: string } | null {
  const consumed = new Set<string>();
  for (const node of doc.graph.nodes) {
    for (const wire of Object.values(node.inputs ?? {})) consumed.add(wire.node);
  }

  /** Longest path from any root to this node, by repeated relaxation over the edge list. */
  const depth = new Map<string, number>();
  for (const node of doc.graph.nodes) depth.set(node.id, 0);
  for (let pass = 0; pass < doc.graph.nodes.length; pass++) {
    let moved = false;
    for (const node of doc.graph.nodes) {
      for (const wire of Object.values(node.inputs ?? {})) {
        const from = depth.get(wire.node) ?? 0;
        if ((depth.get(node.id) ?? 0) < from + 1) {
          depth.set(node.id, from + 1);
          moved = true;
        }
      }
    }
    if (!moved) break;
  }

  const candidates = doc.graph.nodes
    .filter((n) => !consumed.has(n.id) && n.kind !== 'output')
    .map((n) => {
      const s = catalog[n.kind];
      if (!s) return null;
      const wantedArity = arity(wanted);
      // Prefer an exact type match, then the widest output that still fits. A float can be
      // read as a colour but a vec4 cannot be squeezed into a float input, so "widest that
      // fits" is the safe direction when nothing matches exactly.
      const fits = s.outputs.filter((p) => arity(p.type) <= wantedArity);
      const ports = (fits.length ? fits : s.outputs).slice().sort((a, b) => {
        const rank = (t: ShaderType): number => (t === wanted ? 0 : 1);
        return rank(a.type) - rank(b.type) || arity(b.type) - arity(a.type);
      });
      const port = ports[0];
      if (!port) return null;
      return { id: n.id, port: port.name, depth: depth.get(n.id) ?? 0 };
    })
    .filter((x): x is { id: string; port: string; depth: number } => x !== null)
    .sort((a, b) => b.depth - a.depth || a.id.localeCompare(b.id));

  const pick = candidates[0];
  return pick ? { id: pick.id, port: pick.port } : null;
}

/** A wire that cannot resolve: the only safe edit is to remove it. */
function dropWire(d: Diagnostic): RepairAction | null {
  const node = d.at?.node;
  const port = d.at?.port;
  if (!node || !port) {
    return {
      kind: 'advice',
      advice: `A wiring error (${d.code}) could not be attributed to a specific input port. ${d.message}`,
      why: 'the diagnostic lacks a node/port, so no single wire can be identified',
    };
  }
  return {
    kind: 'patch',
    why: `drop the unresolvable wire into ${node}.${port} (${d.code})`,
    patch: {
      summary: `Remove the wire into ${node}.${port}: ${d.code}`,
      ops: [{ op: 'disconnect', to: { node, port }, reason: d.message }],
    },
  };
}

/**
 * A type mismatch has two possible fixes and they are not equivalent:
 *
 *   • If the source already produces something wider than the input wants, the input is
 *     simply wrong — the graph author aimed a vec4 at a float. Dropping the wire lets the
 *     input default, which is the intended "leave this at its default" reading.
 *   • If the source is narrower (float -> vec4), the usual intent is for the value to be
 *     *used*, not discarded. Substituting a source that produces the right type preserves
 *     the author's intent, and because a candidate's port name is known from the catalog, the
 *     substitution is a concrete, checkable edit rather than a guess. (Several nodes now
 *     expose more than one output; a candidate is only usable if *some* port has the wanted
 *     type, which is what `kindsProducing` checks.)
 *
 * Anything else is reported as advice: inventing a conversion node would require the
 * catalog to expose one, and it does not.
 */
function fixTypeMismatch(doc: ShaderDoc, d: Diagnostic, catalog: CatalogLike): RepairAction | null {
  const nodeId = d.at?.node;
  const portName = d.at?.port;
  if (!nodeId || !portName) return null;

  const node = doc.graph.nodes.find((n) => n.id === nodeId);
  const schema = node ? catalog[node.kind] : undefined;
  const inputPort = schema?.inputs.find((p) => p.name === portName);
  const wire = node?.inputs?.[portName];
  if (!inputPort || !wire) return null;

  const sourceNode = doc.graph.nodes.find((n) => n.id === wire.node);
  const sourceSchema = sourceNode ? catalog[sourceNode.kind] : undefined;
  const sourcePort = sourceSchema?.outputs.find((p) => p.name === (wire.port ?? sourceSchema.primaryOut));

  const wanted = inputPort.type;
  const have = sourcePort?.type;

  /*
   * Re-express the value through nodes, first, in **both** directions.
   *
   * The rule this replaces was "narrower -> wider: advise; wider -> narrower: delete the wire",
   * and each half was wrong in its own way. Deleting the wire is the worse one (the port falls
   * back to its default, so the document still emits cleanly and the effect silently disappears),
   * but advising on `vec2 -> vec4` was not much better: `canConnect` widens only scalars, so a
   * vec2 aimed at a vec4 is a mismatch the *catalog can build* — `split_vec2` then `make_vec4` —
   * and a repairer that answers a buildable edit with prose has just spent a model call to have
   * someone else do the same work worse.
   *
   * Both go through `insertNarrowing` because they are the same edit seen from two sides: take
   * the value apart, then put it back together as the wanted type. The names come from the
   * catalog, so `applyPatch` checks the result like any other patch — including rejecting it
   * atomically if the graph is cyclic, which is exactly what happened the first time this ran.
   */
  const rebuilt = insertNarrowing(doc, catalog, d, { nodeId, portName, wire, have, wanted });
  if (rebuilt) return rebuilt;

  // Nothing insertable: only `vec3` sources land here (no node outputs a vec3, and there is no
  // `split_vec3`), so this is honest advice rather than a guess.
  if (have && arity(have) < arity(wanted)) {
    const candidates = Object.keys(catalog)
      .sort()
      .filter((k): k is NodeKind => {
        const s = catalog[k as NodeKind];
        return !!s && s.outputs.some((p) => p.type === wanted) && !s.inputs.some((p) => p.type === have);
      });
    const pick = candidates[0];
    if (pick) {
      const schemaPick = catalog[pick]!;
      const outPort = schemaPick.outputs.find((p) => p.type === wanted)!;
      return {
        kind: 'advice',
        advice:
          `Node ${nodeId}.${portName} needs ${wanted} but ${wire.node} produces ${have}. ` +
          `Either wire a ${wanted}-producing node (for example \`${pick}\`, output ` +
          `\`${outPort.name}\`) into it, or insert a node that widens ${have} to ${wanted}.`,
        why: 'a widening change needs a catalog node that the repairer cannot choose for the user',
      };
    }
  }

  return {
    kind: 'advice',
    advice:
      `${nodeId}.${portName} needs ${wanted} but ${wire.node} produces ${have ?? '?'}, and the ` +
      `catalog has no node that takes ${have ?? '?'} apart into ${wanted}. Rework the chain so ` +
      `the wide value is not needed here, or wire a ${wanted}-producing node instead.`,
    why: 'dropping the wire would leave the port on its default, which looks like a working document',
  };
}

/**
 * Insert the splitter/joiner nodes that turn `have` into `wanted`, or `null` if the catalog
 * cannot express it.
 *
 * The reachable cases, all four of them, and the chain each one builds:
 *
 *   vec4 -> float   split_vec4(value) -> .x
 *   vec4 -> vec2    split_vec4(value) -> make_vec2(x, y)
 *   vec2 -> float   split_vec2(value) -> .x
 *   vec2 -> vec4    split_vec2(value) -> make_vec4(x, y)      (z, w keep their defaults)
 *
 * `vec3` appears in none of them on purpose: no node in the catalog *outputs* a vec3, so the
 * vec3 sources of a mismatch cannot arise, and there is no `split_vec3` to build one with.
 * Guessing at an unreachable case is how a repairer acquires behaviour nobody can test.
 *
 * The inserted ids come from `makeNodeId(kind, taken)` — `split_vec4`, `split_vec4_2`, … — so
 * repairing the same document twice produces the same patch, and the ids are valid GLSL
 * identifiers (they become `L_<id>_<port>` locals). The loop's change detection depends on the
 * determinism: a repair that renames its own nodes every round never converges.
 */
function insertNarrowing(
  doc: ShaderDoc,
  catalog: CatalogLike,
  d: Diagnostic,
  at: { nodeId: string; portName: string; wire: WireRef; have?: ShaderType; wanted: ShaderType },
): RepairAction | null {
  const { nodeId, portName, wire, have, wanted } = at;
  if (have !== 'vec4' && have !== 'vec2') return null;
  if (wanted !== 'float' && wanted !== 'vec2' && wanted !== 'vec4') return null;
  if (!catalog.split_vec4 || !catalog.split_vec2 || !catalog.make_vec2 || !catalog.make_vec4) {
    return null;
  }

  const taken = doc.graph.nodes.map((n) => n.id);
  const splitKind: NodeKind = have === 'vec4' ? 'split_vec4' : 'split_vec2';
  // `makeNodeId` takes the *kind*, so the id it returns is a valid GLSL identifier rather than
  // a sentence: ids are emitted as `L_<id>_<port>` locals. The kind is also what makes the id
  // deterministic, which the loop's change detection depends on.
  const splitId = makeNodeId(splitKind, taken);
  taken.push(splitId);

  const ops: PatchOp[] = [
    {
      op: 'add_node',
      node: { id: splitId, kind: splitKind } as ShaderNode,
      reason: `${d.code}: take ${have} apart so ${nodeId}.${portName} can receive ${wanted}`,
    },
    {
      op: 'connect',
      from: { node: wire.node, port: wire.port },
      to: { node: splitId, port: 'value' },
      reason: `${d.code}: the wide value is preserved, not dropped`,
    },
  ];

  if (wanted === 'float') {
    ops.push({
      op: 'connect',
      from: { node: splitId, port: 'x' },
      to: { node: nodeId, port: portName },
      reason: `${d.code}: first component of ${have}`,
    });
    return {
      kind: 'patch',
      why: `split ${have} into a float instead of deleting the wire (${d.code})`,
      patch: { summary: `Insert ${splitKind} before ${nodeId}.${portName}`, ops },
    };
  }

  // vec4 -> vec2 rebuilds a vec2; vec2 -> vec4 rebuilds a vec4. Both go through
  // `make_vecN`, whose unwired inputs keep the catalog defaults (0 for xyz, 1 for w), so the
  // result is the original value with the extra components zero-filled rather than garbage.
  const makeKind: NodeKind = wanted === 'vec2' ? 'make_vec2' : 'make_vec4';
  const makeId = makeNodeId(makeKind, taken);
  ops.push(
    {
      op: 'add_node',
      node: { id: makeId, kind: makeKind } as ShaderNode,
      reason: `${d.code}: rebuild ${wanted} from the components`,
    },
    {
      op: 'connect',
      from: { node: splitId, port: 'x' },
      to: { node: makeId, port: 'x' },
      reason: `${d.code}`,
    },
    {
      op: 'connect',
      from: { node: splitId, port: 'y' },
      to: { node: makeId, port: 'y' },
      reason: `${d.code}`,
    },
    {
      op: 'connect',
      from: { node: makeId, port: 'value' },
      to: { node: nodeId, port: portName },
      reason: `${d.code}: the rebuilt ${wanted}`,
    },
  );
  return {
    kind: 'patch',
    why: `rebuild ${have} as ${wanted} instead of deleting the wire (${d.code})`,
    patch: { summary: `Insert ${splitKind} + ${makeKind} before ${nodeId}.${portName}`, ops },
  };
}

/** A literal or uniform whose type contradicts the port: fold it back to the port's type. */
function fixParamType(doc: ShaderDoc, d: Diagnostic, catalog: CatalogLike): RepairAction | null {
  const nodeId = d.at?.node;
  const portName = d.at?.port;
  if (!nodeId || !portName) return null;

  const node = doc.graph.nodes.find((n) => n.id === nodeId);
  const schema = node ? catalog[node.kind] : undefined;
  const port = schema?.inputs.find((p) => p.name === portName);
  if (!port) return null;

  const current = node?.params?.[portName];
  // Fold a mismatched uniform back to a plain literal of the correct type. This is the one
  // case where the fix is unambiguous: the *value* is known and only its spelling is wrong.
  if (current?.kind === 'param') {
    const value = coerceValue(current.value, port.type);
    if (value !== null) {
      return {
        kind: 'patch',
        why: `replace the ${current.type} uniform \`${current.name}\` with a ${port.type} literal (${d.code})`,
        patch: {
          summary: `Inline ${current.name} as a ${port.type} literal`,
          ops: [
            {
              op: 'set_param',
              node: nodeId,
              param: portName,
              value,
              reason: d.message,
            },
          ],
        },
      };
    }
  }

  return {
    kind: 'advice',
    advice:
      `Node ${nodeId}.${portName} expects ${port.type} but is given a different type. ` +
      `Set \`params.${portName}\` to a ${port.type} value, or wire a ${port.type}-producing output into it.`,
    why: 'the correct value is not derivable from the diagnostic',
  };
}

/** An unwired required input: wire it from a generator, or report what it needs. */
function fillMissingInput(doc: ShaderDoc, d: Diagnostic, catalog: CatalogLike): RepairAction | null {
  const nodeId = d.at?.node;
  const portName = d.at?.port;
  if (!nodeId || !portName) return null;

  const node = doc.graph.nodes.find((n) => n.id === nodeId);
  const schema = node ? catalog[node.kind] : undefined;
  const port = schema?.inputs.find((p) => p.name === portName);
  if (!port) return null;

  // Prefer an already-present node that produces the wanted type: adding a node to satisfy
  // an input usually duplicates something the graph already has.
  const existing = doc.graph.nodes
    .filter((n) => n.id !== nodeId)
    .map((n) => ({ n, s: catalog[n.kind] }))
    .filter((x): x is { n: (typeof doc.graph.nodes)[number]; s: NodeSchema } => !!x.s)
    .map((x) => ({ id: x.n.id, port: x.s.outputs.find((p) => p.type === port.type) }))
    .filter((x): x is { id: string; port: { name: string; type: ShaderType } } => !!x.port)
    .sort((a, b) => a.id.localeCompare(b.id));

  const pick = existing[0];
  if (pick) {
    return {
      kind: 'patch',
      why: `wire ${pick.id}.${pick.port.name} into the unconnected ${nodeId}.${portName} (${d.code})`,
      patch: {
        summary: `Connect ${nodeId}.${portName}`,
        ops: [
          {
            op: 'connect',
            from: { node: pick.id, port: pick.port.name },
            to: { node: nodeId, port: portName },
            reason: d.message,
          },
        ],
      },
    };
  }

  // Nothing in the graph can supply it. Name a concrete kind that could, so the model (or
  // the user) has somewhere to go rather than a dead end.
  const candidates = Object.keys(catalog)
    .sort()
    .filter((k) => catalog[k as NodeKind]?.outputs.some((p) => p.type === port.type));
  return {
    kind: 'advice',
    advice:
      `Node ${nodeId}.${portName} (${port.type}) has no input. Wire something that produces ` +
      `${port.type} into it` +
      (candidates.length ? ` — for example \`${candidates[0]}\`.` : '.'),
    why: 'the graph contains no node that can supply this type',
  };
}

/**
 * Break a cycle by removing one wire inside it.
 *
 * Only edges whose both endpoints are named in the diagnostic are considered, which keeps
 * the edit inside the reported cycle. The edge chosen is the lexicographically smallest
 * `to.node`/`to.port` pair so the repair is deterministic — a repairer that picks a
 * different edge on each run makes the loop's behaviour impossible to reproduce from a
 * trace.
 */
function breakCycle(doc: ShaderDoc, d: Diagnostic): RepairAction | null {
  const message = d.message;
  const inCycle = doc.graph.nodes.map((n) => n.id).filter((id) => message.includes(id));
  if (inCycle.length < 2) {
    return {
      kind: 'hopeless',
      reason: `IR007_CYCLE could not be localised to specific nodes: ${message}`,
    };
  }

  const candidates: { node: string; port: string }[] = [];
  for (const id of doc.graph.nodes.map((n) => n.id)) {
    if (!inCycle.includes(id)) continue;
    const node = doc.graph.nodes.find((n) => n.id === id)!;
    for (const [port, wire] of Object.entries(node.inputs ?? {})) {
      if (inCycle.includes(wire.node)) candidates.push({ node: id, port });
    }
  }
  if (!candidates.length) {
    return { kind: 'hopeless', reason: `No edge inside the reported cycle: ${message}` };
  }
  candidates.sort((a, b) => `${a.node}.${a.port}`.localeCompare(`${b.node}.${b.port}`));
  const edge = candidates[0]!;
  return {
    kind: 'patch',
    why: `remove the edge ${edge.node}.${edge.port} to break the cycle (${d.code})`,
    patch: {
      summary: `Break the cycle by disconnecting ${edge.node}.${edge.port}`,
      ops: [{ op: 'disconnect', to: edge, reason: message }],
    },
  };
}

/**
 * Two uniforms share a name with different types. Renaming is mechanical, and renaming the
 * *second* one is what keeps the first node's shader output unchanged.
 */
function renameUniform(doc: ShaderDoc, d: Diagnostic): RepairAction | null {
  const nodeId = d.at?.node;
  const portName = d.at?.port;
  if (!nodeId || !portName) return null;
  const node = doc.graph.nodes.find((n) => n.id === nodeId);
  const param = node?.params?.[portName];
  if (param?.kind !== 'param') return null;

  const taken = new Set<string>();
  for (const n of doc.graph.nodes) {
    for (const v of Object.values(n.params ?? {})) if (v.kind === 'param') taken.add(v.name);
  }
  const preferred = `${param.name}_${sanitiseSuffix(nodeId)}`;
  const fresh = safeIdentifier(preferred, taken);

  const newParam: ParamValue = { ...param, name: fresh };
  return {
    kind: 'patch',
    why: `rename the conflicting uniform \`${param.name}\` to \`${fresh}\` (${d.code})`,
    patch: {
      summary: `Rename uniform ${param.name} -> ${fresh}`,
      ops: [{ op: 'set_param', node: nodeId, param: portName, value: newParam, reason: d.message }],
    },
  };
}

/** Dead nodes are not emitted, so removing them changes nothing visible. */
function removeDeadNodes(d: Diagnostic): RepairAction | null {
  // `IR011` lists the ids in its message; there is no structured field for them, so the
  // ids are recovered by matching against the document. A diagnostic format with an
  // `at.nodes` array would be better, but the message is stable and tested.
  return {
    kind: 'advice',
    advice: `${d.message} ${d.fix ?? ''}`.trim(),
    why: 'dead nodes are already excluded from the emitted shader; removing them is cosmetic',
  };
}

/* -------------------------------------------------------------------------- */
/* Helpers                                                                     */
/* -------------------------------------------------------------------------- */

const ARITY: Record<string, number> = {
  float: 1, int: 1, bool: 1, vec2: 2, vec3: 3, vec4: 4, mat2: 4, mat3: 9,
};

function arity(t: ShaderType): number {
  return ARITY[t] ?? 0;
}

/** Build a literal of `to` from an existing value, or null when that is not possible. */
function coerceValue(value: number | boolean | number[], to: ShaderType): ParamValue | null {
  const asArray = Array.isArray(value) ? value : [Number(value)];
  const n = arity(to);
  if (n === 0 && to !== 'bool') return null;
  if (to === 'bool') return { kind: 'bool', value: Boolean(Array.isArray(value) ? value[0] : value) };
  if (to === 'int') return { kind: 'int', value: Math.round(asArray[0] ?? 0) };
  if (to === 'float') return { kind: 'float', value: asArray[0] ?? 0 };
  const filled = Array.from({ length: n }, (_, i) => asArray[i] ?? asArray[0] ?? 0);
  if (to === 'vec2') return { kind: 'vec2', value: [filled[0]!, filled[1]!] };
  if (to === 'vec3') return { kind: 'vec3', value: [filled[0]!, filled[1]!, filled[2]!] };
  if (to === 'vec4') return { kind: 'vec4', value: [filled[0]!, filled[1]!, filled[2]!, filled[3]!] };
  return null;
}

function sanitiseSuffix(id: string): string {
  return id.replace(/[^A-Za-z0-9_]/g, '_').replace(/^_+/, '') || 'alt';
}

/* -------------------------------------------------------------------------- */
/* Patch assembly                                                              */
/* -------------------------------------------------------------------------- */

/**
 * Fold a repair plan into as few patches as can be applied together.
 *
 * Batching matters: applying five one-op patches costs five `applyPatch` calls, five
 * re-analyses and five re-hashes. Applying one five-op patch is one of each. But the
 * batching is **conservative** — an action is only merged when its ops can be proven not
 * to interact, and `hopeless` actions are never merged into anything.
 *
 * Deliberately returns patches in plan order rather than one merged patch: `applyPatch` is
 * atomic, so merging everything means one bad op rejects every good fix alongside it. The
 * loop applies these in sequence and stops at the first rejection.
 */
export function planToPatches(actions: readonly RepairAction[]): { patch: ShaderPatch; why: string }[] {
  const out: { patch: ShaderPatch; why: string }[] = [];
  for (const a of actions) {
    if (a.kind !== 'patch') continue;
    if (a.patch.ops.length === 0) continue;
    out.push({ patch: a.patch, why: a.why });
  }
  return out;
}

/** Diagnostics the repairer recognised but could not turn into an edit. Fed to the model. */
export function adviceFrom(actions: readonly RepairAction[]): string[] {
  return actions.filter((a) => a.kind === 'advice').map((a) => a.advice);
}

/** Ops a repair plan would apply, for the trace. */
export function opsOf(actions: readonly RepairAction[]): PatchOp[] {
  return actions.filter((a) => a.kind === 'patch').flatMap((a) => a.patch.ops);
}

/** True when the only thing wrong is something no patch can reach. */
export function isUnrecoverable(actions: readonly RepairAction[]): boolean {
  const real = actions.filter((a) => a.kind !== 'hopeless');
  return actions.length > 0 && real.length === 0;
}

/** Names that a generated uniform must not collide with. Used by callers that mint them. */
export function isSafeUniformName(name: string): boolean {
  return /^u_[A-Za-z_][A-Za-z0-9_]*$/.test(name) && !isReservedAnywhere(name);
}
