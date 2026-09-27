#!/usr/bin/env node
/**
 * ai4shader — command line interface
 * ==================================
 *
 * Four commands, all of them usable by a human and by a machine:
 *
 *   validate  structural + type validation, diagnostics grouped by severity
 *   emit      IR -> shader source for one target, or a full bundle directory
 *   params    list / override uniform values (the fine-tuning loop's interface)
 *   tune      deterministic Cartesian sweep of named uniforms
 *
 * Design rules, all of them load-bearing for the agent loops that call this:
 *
 *   • ZERO runtime dependencies. `node:util`'s `parseArgs` is the parser; colours are a
 *     dozen hand-rolled escapes that switch themselves off when stdout is not a TTY (or
 *     when `NO_COLOR` is set). A CLI that needs a package manager to start cannot be the
 *     bootstrap step of a build.
 *   • EXIT CODES ARE THE API. 0 ok, 1 validation/emit errors, 2 usage error. Anything
 *     that reads this in CI branches on the number, never on the message.
 *   • `--set` GOES THROUGH THE PATCHER. `params --set u_gain=0.4` builds a `ShaderPatch`
 *     and applies it, so `hash` is recomputed, provenance gains a step, and the value is
 *     type-checked exactly as an LLM-authored op would be. Writing into the JSON directly
 *     would make the CLI a second, unvalidated writer of the IR.
 *   • NOTHING IS OVERWRITTEN SILENTLY. People hand-edit generated shaders. A differing
 *     existing file produces `<name>.new` plus a `~` marker; `--force` opts in.
 *   • CWD-INDEPENDENT. Every path is resolved against the document's own directory or an
 *     absolute path; nothing depends on where the process happens to be.
 */

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, isAbsolute, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseArgs } from 'node:util';

import type {
  Catalog,
  Diagnostic,
  ShaderDoc,
  ShaderStage,
  TargetFormat,
  UniformDecl,
} from './loadIr.js';
import { loadIr, requireIr } from './loadIr.js';
import {
  chooseProvider,
  DEFAULT_BASE_URL,
  describeProvider,
  makeProvider,
  requireAgent,
  type AgentProvider,
  type AgentResult,
} from './loadAgent.js';

/* -------------------------------------------------------------------------- */
/* Exit codes                                                                  */
/* -------------------------------------------------------------------------- */

/** 0 ok · 1 the document or the emission is bad · 2 the *invocation* is bad. */
const EXIT = { OK: 0, PROBLEM: 1, USAGE: 2 } as const;

/** Thrown for anything the user can fix by retyping the command. */
class UsageError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'UsageError';
  }
}

/** Thrown for a well-formed command that cannot be carried out. */
class CliError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'CliError';
  }
}

/* -------------------------------------------------------------------------- */
/* Output                                                                      */
/* -------------------------------------------------------------------------- */

const useColor = (): boolean =>
  Boolean(process.stdout.isTTY) && process.env['NO_COLOR'] === undefined && process.env['TERM'] !== 'dumb';

const c = {
  bold: (s: string) => (useColor() ? `\u001b[1m${s}\u001b[22m` : s),
  dim: (s: string) => (useColor() ? `\u001b[2m${s}\u001b[22m` : s),
  red: (s: string) => (useColor() ? `\u001b[31m${s}\u001b[39m` : s),
  green: (s: string) => (useColor() ? `\u001b[32m${s}\u001b[39m` : s),
  yellow: (s: string) => (useColor() ? `\u001b[33m${s}\u001b[39m` : s),
  cyan: (s: string) => (useColor() ? `\u001b[36m${s}\u001b[39m` : s),
};

const out = (s = ''): void => void process.stdout.write(`${s}\n`);
const err = (s = ''): void => void process.stderr.write(`${s}\n`);

/* -------------------------------------------------------------------------- */
/* Help                                                                        */
/* -------------------------------------------------------------------------- */

const USAGE = `${c.bold('ai4shader')} — validate, emit and tune AI4Shader IR documents

${c.bold('USAGE')}
  ai4shader <command> [options]

${c.bold('COMMANDS')}
  validate <doc>            Check structure, ports, types and uniforms
  emit <doc>                Emit shader source for one target, or a whole bundle
  params <doc>              List uniform values, or override them
  tune <doc>                Deterministic sweep over named uniform ranges
  author <prompt>           Draft a shader from a description, via the agent loop
  refine <doc> <prompt>     Improve an existing document, via the agent loop

${c.bold('GLOBAL')}
  <doc>                     Path to a \`.shader.json\` document, or \`-\` to read stdin
  -h, --help                Help for the command (or for ai4shader itself)
  --json                    Machine-readable output on stdout

${c.bold('EXIT CODES')}
  0  success
  1  validation or emission errors (diagnostics are printed)
  2  usage error (bad flags, missing file, unreadable document)

${c.bold('EXAMPLES')}
  ai4shader validate examples/liquid-metal.shader.json
  ai4shader emit examples/liquid-metal.shader.json --target all --out dist/demo
  ai4shader emit doc.shader.json --target wgsl --stdout
  ai4shader params doc.shader.json --set u_scale=2.5
  ai4shader tune doc.shader.json --param u_scale=0.1..8 --param u_gain=0..1 --steps 5 --out dist/sweep
  ai4shader author "molten metal, slow flow" --provider stub --out dist/authored
  ai4shader author "ice cave" --model gpt-4o-mini --doc-out cave.shader.json --out dist/cave
  ai4shader refine doc.shader.json "more contrast" --out dist/refined

Run \`ai4shader <command> --help\` for the options of one command.`;

const HELP: Record<string, string> = {
  validate: `${c.bold('ai4shader validate')} <doc.shader.json> [--json]

Validate a document: node kinds, port names, wire types, cycles, uniforms, output, and
loop/quality limits. Diagnostics are grouped by severity.

${c.bold('OPTIONS')}
  --json        Print \`{ file, ok, summary, diagnostics }\` instead of the human report
  --catalog     Also require the node catalog (fail if \`packages/ir/src/catalog\` is missing)
  -h, --help    Show this help

${c.bold('EXIT')}
  0  no error-severity diagnostics (warnings are printed and do not fail)
  1  at least one error
  2  usage error

${c.bold('EXAMPLES')}
  ai4shader validate examples/liquid-metal.shader.json
  cat doc.shader.json | ai4shader validate -
  ai4shader validate doc.shader.json --json | jq '.diagnostics[].code'`,

  emit: `${c.bold('ai4shader emit')} <doc.shader.json> --target <target> [--out <dir>] [--stdout]

Emit shader source. The IR is the single source of truth; every target is a projection.

${c.bold('OPTIONS')}
  --target <t>   glsl | wgsl | hlsl | msl | shaderlab | ue | all   (required)
  --out <dir>    Write files under <dir>/<slug>/. Required unless --stdout.
  --stdout       Print the source to stdout (one target only)
  --force        Overwrite files that differ from what would be generated
  --json         Print a machine-readable emission report
  -h, --help     Show this help

${c.bold('BUNDLE LAYOUT (--target all)')}
  <out>/<slug>/shader.frag        glsl
  <out>/<slug>/shader.wgsl        wgsl
  <out>/<slug>/shader.hlsl        hlsl
  <out>/<slug>/shader.metal       msl
  <out>/<slug>/Shader.shader      unity / shaderlab
  <out>/<slug>/ue/CustomNode.hlsl ue, plus whatever sidecars the emitter returns
  <out>/<slug>/params.json        uniforms: name,type,value,min,max,step,label,group
  <out>/<slug>/README.md          intent, features, per-target build notes + limitations

${c.bold('NEVER SILENTLY OVERWRITTEN')}
  If a file already exists with different content, the new content is written next to it
  as \`<name>.new\` and the line is marked \`~\`. Pass --force to overwrite. Generated
  shaders are meant to be hand-edited, and losing that edit is not recoverable.

${c.bold('EXIT')}
  0  every requested target emitted with no error-severity diagnostic
  1  an emission produced an error diagnostic (a rejected document)
  2  usage error

${c.bold('EXAMPLES')}
  ai4shader emit doc.shader.json --target glsl --stdout
  ai4shader emit doc.shader.json --target all --out dist/demo
  ai4shader emit doc.shader.json --target wgsl --out dist --force`,

  params: `${c.bold('ai4shader params')} <doc.shader.json> [--set <name>=<value> …] [--out <file>]

List the document's uniforms and their current values, or override values. This is the
machine interface the fine-tuning loop and CI use.

${c.bold('OPTIONS')}
  --set <n>=<v>   Override a uniform. Repeatable. Value forms:
                    u_gain=0.5            scalar (float / int / bool)
                    u_tint=1,0,0,1        vecN / colour, comma separated
                    u_on=true             booleans accept true/false/1/0/yes/no
  --out <file>    Write the patched document here (pretty-printed JSON)
  --json          Print the machine-readable listing (or the patched document summary)
  -h, --help      Show this help

${c.bold('ROUND-TRIPS THROUGH THE PATCHER')}
  Every --set becomes a \`set_param\` patch op, so the value is type-checked against the
  port, applied atomically, \`hash\` is recomputed and a provenance step is appended. If
  any override fails, *none* is applied and the command exits 1.

${c.bold('EXIT')}
  0  listed, or every override applied
  1  an override was rejected (unknown uniform, wrong type, out of range)
  2  usage error

${c.bold('EXAMPLES')}
  ai4shader params doc.shader.json
  ai4shader params doc.shader.json --json
  ai4shader params doc.shader.json --set u_scale=3.5 --set u_tint=1,0.3,0.2,1 --out tuned.json`,

  tune: `${c.bold('ai4shader tune')} <doc.shader.json> --param <name>=<lo>..<hi> … --steps <n> [--out <dir>]

Deterministic sweep: sample every listed uniform evenly across its range and emit the
Cartesian product. This is the batch half of tuning; the browser tool does the interactive
half.

${c.bold('OPTIONS')}
  --param <n>=<lo>..<hi>  Uniform range. Repeatable. Repeat the *same name* to build a
                          non-uniform grid for that one uniform:
                            --param u_scale=0.5..2 --param u_scale=8 --param u_scale=32
  --steps <n>             Samples per uniform range (default 5). Must be ≥ 1.
  --target <t>            Target to emit per variant (default: the document's first, or glsl)
  --out <dir>             Output directory (default \`dist/sweep\`). Required for --emit.
  --emit                  Actually emit each variant into <out>/sweep-<n>/
  --force                 Overwrite differing files when emitting
  --json                  Print the sweep manifest to stdout as well
  -h, --help              Show this help

${c.bold('DETERMINISM')}
  Sample values are \`lo + (hi-lo)*i/(steps-1)\` in IEEE double arithmetic and are written
  with 4 decimals. Each variant id is built from its own assignments, sorted by name —
  \`u_gain=0.0000__u_scale=2.0000\` — so two runs produce the same ids, the same directory
  names and a diffable \`sweep.json\`. Variant <n> is always the same point of the grid.

${c.bold('CAP')}
  At most 256 combinations (steps ^ params). Above that the command fails with the exact
  count rather than producing a directory nobody will look at.

${c.bold('OUTPUT')}
  <out>/sweep.json     every variant: id, index, values, doc hash, directory, diagnostics
  <out>/sweep-<n>/     the emitted files for variant n (with --emit)

${c.bold('EXIT')}
  0  the sweep was produced (per-variant emission diagnostics are recorded in sweep.json)
  1  a variant failed to emit
  2  usage error

${c.bold('EXAMPLES')}
  ai4shader tune doc.shader.json --param u_scale=0.1..8 --param u_gain=0..1 --steps 5 --out dist/sweep
  ai4shader tune doc.shader.json --param u_iter=4..64 --steps 4 --out dist/sweep --emit --target glsl`,

  author: `${c.bold('ai4shader author')} <prompt> [options]

Runs the agent loop: interpret the description into a ShaderSpec, draft a graph, validate it,
emit every target, and repair its own failures. Prints the stage trace, so a slow or failed run
is legible rather than mysterious.

${c.bold('OPTIONS')}
  <prompt>                  The description, or --prompt <text>. Positional words are joined.
  --provider <stub|http|none|auto>
                            Default: auto — uses http only when a key AND a model are configured,
                            otherwise runs deterministic repair only and says so.
                            \`stub\` is the offline deterministic provider: no network, no key, and
                            it drafts a complete working document, which makes it the right way to
                            try the loop or to write a regression test.
  --model <name>            Model name for the http provider.
  --api-key <key>           Or AI4SHADER_API_KEY / OPENAI_API_KEY.
  --base-url <url>          Or AI4SHADER_BASE_URL / OPENAI_BASE_URL. Default ${DEFAULT_BASE_URL}.
  --vision                  Declare that the model accepts images.
  --target <t|all>          Which targets must emit cleanly. Default: all registered targets.
  --doc-out <file>          Write the resulting IR document here.
  --out <dir>               Write the multi-format bundle here (same layout as \`emit\`).
  --force                   Overwrite differing files in place instead of writing \`.new\`.
  --json                    Machine-readable result on stdout.
  -h, --help

${c.bold('EXIT CODES')}
  0  every required target emitted with zero errors
  1  the loop stopped with diagnostics outstanding (they are printed with their fixes)
  2  usage error

${c.bold('EXAMPLES')}
  ai4shader author "molten metal, slow flow" --provider stub --out dist/authored
  ai4shader author "ice cave with blue crystals" --doc-out cave.shader.json --out dist/cave
  ai4shader author "plasma globe" --provider none --json`,

  refine: `${c.bold('ai4shader refine')} <doc.shader.json> <prompt> [options]

Same loop as \`author\`, starting from an existing document instead of a blank one. The document
is verified first, so a document that already emits cleanly costs **zero model calls** — the
deterministic repairer handles anything it recognises before a model is consulted.

${c.bold('OPTIONS')}
  <doc.shader.json>         The document to improve (\`-\` reads stdin).
  <prompt>                  What to change, or --prompt <text>.
  Everything \`author\` accepts, plus:
  --doc-out <file>          Defaults to overwriting the input document.

${c.bold('EXIT CODES')}
  0  every required target emitted with zero errors
  1  diagnostics outstanding
  2  usage error

${c.bold('EXAMPLES')}
  ai4shader refine doc.shader.json "more contrast" --out dist/refined
  ai4shader refine doc.shader.json "slower" --provider none --json`,
};

/* -------------------------------------------------------------------------- */
/* Arg parsing                                                                 */
/* -------------------------------------------------------------------------- */

const COMMON = { help: { type: 'boolean', short: 'h' } } as const;

const OPTIONS = {
  validate: {
    ...COMMON,
    json: { type: 'boolean' },
    catalog: { type: 'boolean' },
  },
  emit: {
    ...COMMON,
    target: { type: 'string' },
    out: { type: 'string' },
    stdout: { type: 'boolean' },
    force: { type: 'boolean' },
    json: { type: 'boolean' },
  },
  params: {
    ...COMMON,
    set: { type: 'string', multiple: true },
    out: { type: 'string' },
    json: { type: 'boolean' },
  },
  tune: {
    ...COMMON,
    param: { type: 'string', multiple: true },
    steps: { type: 'string' },
    target: { type: 'string' },
    out: { type: 'string' },
    emit: { type: 'boolean' },
    force: { type: 'boolean' },
    json: { type: 'boolean' },
  },
  author: {
    ...COMMON,
    prompt: { type: 'string' },
    provider: { type: 'string' },
    'api-key': { type: 'string' },
    'base-url': { type: 'string' },
    model: { type: 'string' },
    vision: { type: 'boolean' },
    target: { type: 'string' },
    out: { type: 'string' },
    'doc-out': { type: 'string' },
    force: { type: 'boolean' },
    json: { type: 'boolean' },
  },
  refine: {
    ...COMMON,
    prompt: { type: 'string' },
    provider: { type: 'string' },
    'api-key': { type: 'string' },
    'base-url': { type: 'string' },
    model: { type: 'string' },
    vision: { type: 'boolean' },
    target: { type: 'string' },
    out: { type: 'string' },
    'doc-out': { type: 'string' },
    force: { type: 'boolean' },
    json: { type: 'boolean' },
  },
} as const;

type CommandName = keyof typeof OPTIONS;

type Parsed = {
  values: Record<string, string | boolean | string[] | undefined>;
  positionals: string[];
};

/** `parseArgs` in strict mode already rejects unknown flags and missing values. */
function parse(command: CommandName, argv: string[]): Parsed {
  try {
    const { values, positionals } = parseArgs({
      args: argv,
      options: OPTIONS[command] as never,
      allowPositionals: true,
      strict: true,
    });
    return { values: values as Parsed['values'], positionals };
  } catch (e) {
    throw new UsageError(`${(e as Error).message}\n\nRun \`ai4shader ${command} --help\`.`);
  }
}

const str = (v: unknown): string | undefined => (typeof v === 'string' ? v : undefined);
const flag = (v: unknown): boolean => v === true;
const list = (v: unknown): string[] => (Array.isArray(v) ? (v as string[]) : []);

/** Exactly one positional, or a usage error that says what was expected. */
function docArg(command: CommandName, positionals: string[]): string {
  if (positionals.length === 0) throw new UsageError(`\`${command}\` needs a document path (or \`-\` for stdin)`);
  if (positionals.length > 1) {
    throw new UsageError(`\`${command}\` takes one document, got ${positionals.length}: ${positionals.join(', ')}`);
  }
  return positionals[0]!;
}

/* -------------------------------------------------------------------------- */
/* Document IO                                                                 */
/* -------------------------------------------------------------------------- */

/** `-` reads stdin, so the CLI composes: `… | ai4shader validate -`. */
async function readDoc(path: string, cwd: string): Promise<{ doc: ShaderDoc; file: string }> {
  let text: string;
  let file: string;
  if (path === '-') {
    file = '<stdin>';
    text = await readStdin();
  } else {
    file = isAbsolute(path) ? path : resolve(cwd, path);
    try {
      text = await readFile(file, 'utf8');
    } catch (e) {
      const code = (e as { code?: string }).code;
      if (code === 'ENOENT') throw new UsageError(`no such file: ${file}`);
      if (code === 'EISDIR') throw new UsageError(`that is a directory, not a document: ${file}`);
      throw new CliError(`cannot read ${file}: ${(e as Error).message}`);
    }
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (e) {
    // A usage error (exit 2), not a `CliError` (exit 1). The distinction the exit codes draw
    // is *whose mistake it is*: 1 means "the document has diagnostics", 2 means "you called
    // this wrong". A file that is not JSON at all never became a document, so there are no
    // diagnostics to report and nothing for a caller to inspect — it is the same class of
    // mistake as naming a file that does not exist, which this function already treats as a
    // usage error two lines above. Reporting 1 here told a script "the shader is bad" when
    // the truth was "that is not a JSON file", and the two want different handling.
    throw new UsageError(`${file} is not valid JSON: ${(e as Error).message}`);
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new UsageError(`${file} must contain a JSON object, got ${Array.isArray(parsed) ? 'an array' : typeof parsed}`);
  }
  return { doc: parsed as ShaderDoc, file };
}

function readStdin(): Promise<string> {
  return new Promise((res, rej) => {
    const chunks: Buffer[] = [];
    process.stdin.on('data', (chunk: Buffer) => chunks.push(chunk));
    process.stdin.on('end', () => res(Buffer.concat(chunks).toString('utf8')));
    process.stdin.on('error', rej);
  });
}

/** Pretty-print with a trailing newline: these files are read by humans and by git. */
const serialize = (doc: ShaderDoc): string => `${JSON.stringify(doc, null, 2)}\n`;

async function writeDocFile(path: string, doc: ShaderDoc, cwd: string): Promise<string> {
  const file = isAbsolute(path) ? path : resolve(cwd, path);
  await mkdir(dirname(file), { recursive: true });
  await writeFile(file, serialize(doc), 'utf8');
  return file;
}

/* -------------------------------------------------------------------------- */
/* The IR modules, loaded on demand                                            */
/* -------------------------------------------------------------------------- */

/**
 * Import the first candidate that resolves.
 *
 * The workspace source path is tried first so the CLI runs in a plain checkout with no
 * `node_modules`; the package specifiers are the fallback for an installed copy. A module
 * that exists but throws is re-thrown — only "not found" moves on.
 *
 * Do *not* write this as `import(a).catch(() => import(b))`: the fallback then runs inside
 * the catch callback, so its own failure escapes the surrounding try/catch and surfaces as
 * an unhandled "unexpected error".
 */
async function importFirst<T>(candidates: string[]): Promise<T | null> {
  for (const spec of candidates) {
    try {
      return (await import(spec)) as T;
    } catch (e) {
      const code = (e as { code?: string }).code;
      const message = String((e as Error).message ?? '');
      // A bare package specifier that is not linked reports `ERR_MODULE_NOT_FOUND`; a
      // `file:` URL to a missing path reports it too. Both mean "try the next candidate".
      if (code !== 'ERR_MODULE_NOT_FOUND' && code !== 'ERR_UNKNOWN_FILE_EXTENSION') throw e;
      void message;
    }
  }
  return null;
}

/** `<root>/packages/ir/src/<file>` — the source path of an unbuilt checkout. */
function irSourceUrl(file: string): string {
  return new URL(`../../../packages/ir/src/${file}`, import.meta.url).href;
}

type Patcher = {
  applyPatch: (
    doc: ShaderDoc,
    patch: unknown,
    catalog: Catalog,
  ) => { doc: ShaderDoc; applied: number; diagnostics: Diagnostic[]; ok: boolean };
  hashDoc: (doc: ShaderDoc) => string;
};

let patcherPromise: Promise<Patcher> | null = null;
async function patcher(): Promise<Patcher> {
  patcherPromise ??= (async () => {
    // Same ordering rule as `emitterModule()` below: the package specifier first, so the
    // patcher and the emitter come from one module instance. `hashDoc` must be the same
    // function the emitter's cache keys are derived from, or "the hash changed" stops meaning
    // "the content changed".
    const mod = await importFirst<Partial<Patcher>>(['@ai4shader/ir/patcher', '@ai4shader/ir', irSourceUrl('patcher.ts')]);
    if (typeof mod?.applyPatch !== 'function' || typeof mod.hashDoc !== 'function') {
      throw new CliError(
        'cannot load the IR patcher (@ai4shader/ir/patcher) — run `npm install` at the repo root, or check that packages/ir/src/patcher.ts is present',
      );
    }
    return mod as Patcher;
  })();
  return patcherPromise;
}

/** `analyzeGraph`, from the same module graph as everything else. */
async function emitterModule(): Promise<{
  emit: (doc: ShaderDoc, opts: { catalog: Catalog; target: TargetFormat }) => {
    source: string;
    sidecars?: { name: string; content: string }[];
    diagnostics: Diagnostic[];
    uniforms: UniformDecl[];
    hash: string;
  };
  analyzeGraph: (graph: ShaderDoc['graph'], catalog: Catalog) => { diagnostics: Diagnostic[]; order: string[] };
}> {
  const mod = await importFirst<{
    emit?: unknown;
    analyzeGraph?: unknown;
  }>([
    // Bare specifier FIRST, and this order is load-bearing rather than cosmetic.
    // `TARGET_EMITTERS` is a module-level map, so `emit()` can only see a target that was
    // registered on the *same module instance*. A `file://` URL to `src/emitter.ts` loads the
    // TypeScript source, which is a different file from the built `dist/emitter.js` that
    // `@ai4shader/ir/targets` imports — so the loaders register into one registry and `emit`
    // reads the other, and every target reports `EM000_UNKNOWN_TARGET` while all six emitters
    // are present and correct. Preferring the package specifier makes the whole process share
    // one instance; the source path remains as a fallback for an unbuilt checkout.
    '@ai4shader/ir',
    irSourceUrl('emitter.ts'),
  ]);
  if (typeof mod?.emit !== 'function' || typeof mod.analyzeGraph !== 'function') {
    throw new CliError(
      'cannot load the IR emitter core (@ai4shader/ir) — run `npm install` at the repo root, or check that packages/ir/src/emitter.ts is present',
    );
  }
  return mod as never;
}

/* -------------------------------------------------------------------------- */
/* Diagnostics                                                                 */
/* -------------------------------------------------------------------------- */

const SEVERITY_ORDER = ['error', 'warning', 'info'] as const;

function sortDiagnostics(ds: readonly Diagnostic[]): Diagnostic[] {
  const rank = new Map(SEVERITY_ORDER.map((s, i) => [s, i] as const));
  return [...ds].sort((a, b) => {
    const bySeverity = (rank.get(a.severity) ?? 9) - (rank.get(b.severity) ?? 9);
    if (bySeverity !== 0) return bySeverity;
    const nodeA = a.at?.node ?? '';
    const nodeB = b.at?.node ?? '';
    if (nodeA !== nodeB) return nodeA < nodeB ? -1 : 1;
    return a.code < b.code ? -1 : a.code > b.code ? 1 : 0;
  });
}

const hasErrors = (ds: readonly Diagnostic[]): boolean => ds.some((d) => d.severity === 'error');
const countOf = (ds: readonly Diagnostic[], severity: string): number => ds.filter((d) => d.severity === severity).length;

function severityTag(severity: string): string {
  if (severity === 'error') return c.red('error  ');
  if (severity === 'warning') return c.yellow('warning');
  return c.cyan('info   ');
}

/** Diagnostics grouped by severity, worst first, each with its actionable `fix`. */
function printDiagnostics(ds: readonly Diagnostic[], indent = '  '): void {
  if (ds.length === 0) {
    out(`${indent}${c.green('✓')} no diagnostics`);
    return;
  }
  const sorted = sortDiagnostics(ds);
  for (const severity of SEVERITY_ORDER) {
    const group = sorted.filter((d) => d.severity === severity);
    if (group.length === 0) continue;
    out(`${indent}${severityTag(severity)} ${c.dim(`(${group.length})`)}`);
    for (const d of group) {
      const where = d.at?.node ? c.dim(` [${d.at.node}${d.at.port ? `.${d.at.port}` : ''}]`) : d.at?.line ? c.dim(` [line ${d.at.line}]`) : '';
      out(`${indent}  ${c.bold(d.code)}${where}`);
      out(`${indent}    ${d.message}`);
      if (d.fix) out(`${indent}    ${c.dim('fix:')} ${d.fix}`);
    }
  }
}

function diagnosticCounts(ds: readonly Diagnostic[]): { error: number; warning: number; info: number } {
  return { error: countOf(ds, 'error'), warning: countOf(ds, 'warning'), info: countOf(ds, 'info') };
}

/**
 * Structural validation, used only when `emitter.ts` could not be loaded.
 *
 * `validate` is the one command that must work in a checkout where the workspace is not
 * linked: it is the command you run *to find out* that something is wrong. This covers what
 * can be checked from the catalog alone — duplicate ids, unknown kinds, unknown ports,
 * dangling and mistyped wires, cycles, a missing output node. It mirrors `analyzeGraph`'s
 * codes so a document gets the same verdict either way; on a linked workspace
 * `analyzeGraph` is used instead and this never runs.
 */
function structuralAnalysis(doc: ShaderDoc, catalog: Catalog): { diagnostics: Diagnostic[]; order: string[] } {
  const diagnostics: Diagnostic[] = [];
  const byId = new Map<string, ShaderDoc['graph']['nodes'][number]>();
  for (const n of doc.graph.nodes) {
    if (byId.has(n.id)) {
      diagnostics.push({
        severity: 'error',
        code: 'IR001_DUPLICATE_ID',
        message: `Duplicate node id \`${n.id}\`.`,
        at: { node: n.id },
        fix: 'Node ids must be unique; regenerate with addNode().',
      });
      continue;
    }
    if (!catalog[n.kind]) {
      diagnostics.push({
        severity: 'error',
        code: 'IR002_UNKNOWN_KIND',
        message: `Unknown node kind \`${n.kind}\` on node \`${n.id}\`.`,
        at: { node: n.id },
        fix: `Known kinds: ${Object.keys(catalog).join(', ')}`,
      });
      continue;
    }
    byId.set(n.id, n);
  }

  for (const n of byId.values()) {
    const schema = catalog[n.kind]!;
    for (const key of Object.keys(n.params ?? {})) {
      if (schema.inputs.some((p) => p.name === key)) continue;
      diagnostics.push({
        severity: 'error',
        code: 'IR003_UNKNOWN_PORT',
        message: `Node \`${n.id}\` (${n.kind}) has no input \`${key}\`.`,
        at: { node: n.id, port: key },
        fix: `Valid inputs: ${schema.inputs.map((p) => p.name).join(', ') || '(none)'}`,
      });
    }
    for (const [key, wire] of Object.entries(n.inputs ?? {})) {
      const port = schema.inputs.find((p) => p.name === key);
      if (!port) {
        diagnostics.push({
          severity: 'error',
          code: 'IR003_UNKNOWN_PORT',
          message: `Node \`${n.id}\` (${n.kind}) has no input \`${key}\`.`,
          at: { node: n.id, port: key },
          fix: `Valid inputs: ${schema.inputs.map((p) => p.name).join(', ') || '(none)'}`,
        });
        continue;
      }
      const src = byId.get(wire.node);
      if (!src) {
        diagnostics.push({
          severity: 'error',
          code: 'IR004_DANGLING_WIRE',
          message: `Node \`${n.id}\` input \`${key}\` reads from missing node \`${wire.node}\`.`,
          at: { node: n.id, port: key },
          fix: 'Remove the wire or re-create the source node.',
        });
        continue;
      }
      const srcSchema = catalog[src.kind]!;
      const outName = wire.port ?? srcSchema.primaryOut ?? srcSchema.outputs[0]?.name;
      const out = srcSchema.outputs.find((p) => p.name === outName);
      if (!out) {
        diagnostics.push({
          severity: 'error',
          code: 'IR005_UNKNOWN_OUTPUT',
          message: `Node \`${wire.node}\` (${src.kind}) has no output \`${outName}\`.`,
          at: { node: n.id, port: key },
          fix: `Valid outputs: ${srcSchema.outputs.map((p) => p.name).join(', ')}`,
        });
      }
    }
  }

  // Cycle detection: Kahn's algorithm, reporting the nodes that never reach in-degree 0.
  const indeg = new Map<string, number>();
  const adj = new Map<string, string[]>();
  for (const id of byId.keys()) {
    indeg.set(id, 0);
    adj.set(id, []);
  }
  for (const n of byId.values()) {
    for (const wire of Object.values(n.inputs ?? {})) {
      if (!byId.has(wire.node)) continue;
      adj.get(wire.node)!.push(n.id);
      indeg.set(n.id, (indeg.get(n.id) ?? 0) + 1);
    }
  }
  const queue = doc.graph.nodes.map((n) => n.id).filter((id) => byId.has(id) && indeg.get(id) === 0);
  const order: string[] = [];
  const seen = new Set<string>();
  while (queue.length) {
    const id = queue.shift()!;
    order.push(id);
    for (const next of adj.get(id) ?? []) {
      const k = `${id}->${next}`;
      if (seen.has(k)) continue;
      seen.add(k);
      const d = (indeg.get(next) ?? 0) - 1;
      indeg.set(next, d);
      if (d === 0) queue.push(next);
    }
  }
  if (order.length !== byId.size) {
    diagnostics.push({
      severity: 'error',
      code: 'IR007_CYCLE',
      message: `Graph contains a cycle among: ${[...byId.keys()].filter((id) => !order.includes(id)).join(', ')}.`,
      fix: 'Break the cycle — the IR is a DAG by definition.',
    });
  }

  const outputs = doc.graph.nodes.filter((n) => n.kind === 'output' && byId.has(n.id));
  if (outputs.length === 0) {
    diagnostics.push({
      severity: 'error',
      code: 'IR008_NO_OUTPUT',
      message: 'Graph has no `output` node.',
      fix: 'Add an `output` node and wire a vec4 into its `color` input.',
    });
  }

  return { diagnostics, order };
}

/* -------------------------------------------------------------------------- */
/* Graph helpers (read-only)                                                   */
/* -------------------------------------------------------------------------- */

/** Longest wire path ending at any node, computed on the raw `inputs` refs. */
function graphDepth(doc: ShaderDoc): number {
  const byId = new Map(doc.graph.nodes.map((n) => [n.id, n] as const));
  const memo = new Map<string, number>();
  const visiting = new Set<string>();

  const depthOf = (id: string): number => {
    const cached = memo.get(id);
    if (cached !== undefined) return cached;
    if (visiting.has(id)) return 0; // a cycle: `analyzeGraph` reports it, do not hang here
    visiting.add(id);
    let best = 1;
    for (const wire of Object.values(byId.get(id)?.inputs ?? {})) {
      if (!byId.has(wire.node)) continue;
      best = Math.max(best, depthOf(wire.node) + 1);
    }
    visiting.delete(id);
    memo.set(id, best);
    return best;
  };

  let depth = 0;
  for (const node of doc.graph.nodes) depth = Math.max(depth, depthOf(node.id));
  return depth;
}

type UniformRow = UniformDecl & { kind: string; node: string; param: string };

type UniformValue = Extract<NonNullable<ShaderDoc['graph']['nodes'][number]['params']>[string], { kind: 'param' }>;

/** Walk the graph for `kind: 'param'` bindings; the authority on what a uniform is. */
function collectUniformRows(doc: ShaderDoc): UniformRow[] {
  const rows: UniformRow[] = [];
  const seen = new Set<string>();
  for (const node of doc.graph.nodes) {
    for (const [param, value] of Object.entries(node.params ?? {})) {
      if (value.kind !== 'param') continue;
      if (seen.has(value.name)) continue;
      seen.add(value.name);
      rows.push({
        kind: value.type,
        name: value.name,
        type: value.type,
        value: value.value,
        ...(value.min !== undefined ? { min: value.min } : {}),
        ...(value.max !== undefined ? { max: value.max } : {}),
        ...(value.step !== undefined ? { step: value.step } : {}),
        ...(value.label !== undefined ? { label: value.label } : {}),
        ...(value.group !== undefined ? { group: value.group } : {}),
        node: node.id,
        param,
        origin: { node: node.id, param },
      });
    }
  }
  return rows;
}

const docTargets = (doc: ShaderDoc): TargetFormat[] =>
  Array.isArray(doc.targets) && doc.targets.length > 0 ? doc.targets : ['glsl'];

/* -------------------------------------------------------------------------- */
/* Slug + file layout                                                          */
/* -------------------------------------------------------------------------- */

/** Filesystem-safe name for a document. Deterministic: same title, same directory. */
export function slugify(title: string): string {
  const s = title
    .normalize('NFKD')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  if (s.length === 0) return 'shader';
  if (s.length <= 48) return s;
  // Long descriptive titles are common ("Liquid Metal — animated liquid metal with a cold
  // rim"). Truncate at a word boundary rather than mid-word, so the directory stays usable
  // in a shell and a diff still reads.
  const cut = s.slice(0, 48);
  const trimmed = cut.slice(0, Math.max(cut.lastIndexOf('-'), 24)).replace(/-+$/, '');
  return trimmed.length > 0 ? trimmed : cut;
}

const ALL_TARGETS: TargetFormat[] = ['glsl', 'wgsl', 'hlsl', 'msl', 'shaderlab', 'ue'];

/** Where a target's primary source goes inside the bundle directory. */
function targetFilePath(target: TargetFormat, slug: string): string {
  switch (target) {
    case 'glsl':
      return `${slug}/shader.frag`;
    case 'wgsl':
      return `${slug}/shader.wgsl`;
    case 'hlsl':
      return `${slug}/shader.hlsl`;
    case 'msl':
      return `${slug}/shader.metal`;
    case 'shaderlab':
      return `${slug}/Shader.shader`;
    case 'ue':
      return `${slug}/ue/CustomNode.hlsl`;
  }
}

/** Sidecar paths are relative to the file the sidecar belongs to. */
function sidecarPath(target: TargetFormat, slug: string, name: string): string {
  const dir = target === 'ue' ? `${slug}/ue` : slug;
  return name.includes('/') ? `${dir}/${name}` : `${dir}/${name}`;
}

/* -------------------------------------------------------------------------- */
/* File writer that refuses to lose a hand edit                                */
/* -------------------------------------------------------------------------- */

type WriteOutcome = 'written' | 'unchanged' | 'diverged';

async function writeGuarded(file: string, content: string, opts: { force: boolean }): Promise<WriteOutcome> {
  await mkdir(dirname(file), { recursive: true });
  let existing: string | null = null;
  try {
    existing = await readFile(file, 'utf8');
  } catch {
    existing = null;
  }
  if (existing === content) return 'unchanged';
  if (existing !== null && !opts.force) {
    // Someone (or something) already wrote a different file here. Generated shaders are
    // meant to be edited by hand, so the new content goes beside it and the caller is told.
    await writeFile(`${file}.new`, content, 'utf8');
    return 'diverged';
  }
  await writeFile(file, content, 'utf8');
  return 'written';
}

function reportWrite(displayPath: string, outcome: WriteOutcome): void {
  if (outcome === 'written') out(`  ${c.green('+')} ${displayPath}`);
  else if (outcome === 'unchanged') out(`  ${c.dim('=')} ${c.dim(displayPath)} ${c.dim('(unchanged)')}`);
  else out(`  ${c.yellow('~')} ${displayPath} ${c.yellow('differs — wrote')} ${displayPath}.new`);
}

/* -------------------------------------------------------------------------- */
/* validate                                                                    */
/* -------------------------------------------------------------------------- */

async function cmdValidate(argv: string[], cwd: string): Promise<number> {
  const { values, positionals } = parse('validate', argv);
  if (flag(values['help'])) {
    out(HELP['validate']!);
    return EXIT.OK;
  }
  const docPath = docArg('validate', positionals);
  const json = flag(values['json']);

  const { doc, file } = await readDoc(docPath, cwd);
  const ir = await loadIr();
  if (!json) {
    if (ir.complete) out(c.dim(`catalog: ${Object.keys(ir.catalog).length} node kinds`));
    else err(`${c.yellow('warning')} ${ir.note}`);
  }

  // `analyzeGraph` is the same analysis `emit()` runs, so `validate` cannot disagree with
  // `emit` about whether a document is good. When the emitter core is not loadable (an
  // unlinked checkout), the local structural pass stands in — with the same diagnostic
  // codes — so `validate` still answers the question it exists to answer.
  const emitter = await importFirst<{
    analyzeGraph?: (graph: ShaderDoc['graph'], catalog: Catalog) => { diagnostics: Diagnostic[]; order: string[] };
  }>([irSourceUrl('emitter.ts'), '@ai4shader/ir']);
  const analysis = emitter?.analyzeGraph
    ? emitter.analyzeGraph(doc.graph, ir.catalog)
    : structuralAnalysis(doc, ir.catalog);

  const diagnostics = sortDiagnostics(analysis.diagnostics);
  const nodes = doc.graph.nodes.length;
  const uniforms = collectUniformRows(doc);
  const depth = graphDepth(doc);
  const counts = diagnosticCounts(diagnostics);
  const ok = counts.error === 0;

  if (json) {
    out(
      JSON.stringify(
        {
          file,
          ok,
          summary: { nodes, edges: doc.graph.edges.length, uniforms: uniforms.length, depth },
          counts,
          catalog: { complete: ir.complete, kinds: Object.keys(ir.catalog).length },
          hash: doc.hash ?? null,
          diagnostics,
        },
        null,
        2,
      ),
    );
    return ok ? EXIT.OK : EXIT.PROBLEM;
  }

  out(`${c.bold(file)}`);
  out(`  ${c.dim('title')}   ${doc.title ?? '(untitled)'}`);
  if (doc.spec?.intent) out(`  ${c.dim('intent')}  ${doc.spec.intent}`);
  const stage = doc.stage ?? 'fragment';
  out(`  ${c.dim('stage')}   ${stage}   ${c.dim('quality')} ${doc.render?.qualityTier ?? '?'}   ${c.dim('maxIterations')} ${doc.render?.maxIterations ?? '?'}`);
  out();
  out(c.bold('  SUMMARY'));
  out(`  ${'nodes'.padEnd(10)} ${nodes}`);
  out(`  ${'edges'.padEnd(10)} ${doc.graph.edges.length}`);
  out(`  ${'uniforms'.padEnd(10)} ${uniforms.length}`);
  out(`  ${'depth'.padEnd(10)} ${depth}`);
  out(`  ${'targets'.padEnd(10)} ${docTargets(doc).join(', ')}`);
  if (doc.hash) out(`  ${'hash'.padEnd(10)} ${doc.hash}`);
  out();
  out(c.bold('  DIAGNOSTICS'));
  printDiagnostics(diagnostics);
  out();
  out(
    ok
      ? `${c.green(c.bold('  PASS'))} ${c.dim(`(${counts.warning} warning${counts.warning === 1 ? '' : 's'}, ${counts.info} info)`)}`
      : `${c.red(c.bold('  FAIL'))} ${counts.error} error${counts.error === 1 ? '' : 's'}`,
  );
  return ok ? EXIT.OK : EXIT.PROBLEM;
}

/* -------------------------------------------------------------------------- */
/* emit                                                                        */
/* -------------------------------------------------------------------------- */

type EmitOutcome = {
  target: TargetFormat;
  ok: boolean;
  hash: string;
  source: string;
  sidecars: { name: string; content: string }[];
  diagnostics: Diagnostic[];
  uniforms: UniformDecl[];
};

async function emitTargets(doc: ShaderDoc, targets: TargetFormat[], catalog: Catalog): Promise<EmitOutcome[]> {
  // `loadIr()`/`requireIr()` have already evaluated `@ai4shader/ir/targets`, which is what
  // registers every emitter (AGENT_CONTRACT.md §2). `emit` itself is target-independent.
  const { emit } = await emitterModule();

  return targets.map((target) => {
    const emission = emit(doc, { catalog, target });
    return {
      target,
      // `Emission` has no `ok` field by design (AGENT_CONTRACT.md §3): success is the
      // absence of an error-severity diagnostic, not whether `source` happens to be empty.
      ok: !hasErrors(emission.diagnostics),
      hash: emission.hash,
      source: emission.source,
      sidecars: emission.sidecars ?? [],
      diagnostics: emission.diagnostics,
      uniforms: emission.uniforms ?? [],
    };
  });
}

function readmeFor(doc: ShaderDoc, slug: string, results: EmitOutcome[], built: string[]): string {
  const lines: string[] = [];
  lines.push(`# ${doc.title || slug}`);
  lines.push('');
  if (doc.spec?.intent) {
    lines.push(`> ${doc.spec.intent}`);
    lines.push('');
  }
  lines.push(`Generated from \`${slug}\` by \`ai4shader emit --target all\`. **Do not edit the`);
  lines.push('generated shaders here and expect the change to survive regeneration** — edit the');
  lines.push('IR document and re-emit. Files that already differ are never overwritten silently:');
  lines.push('the new content lands next to them as `<name>.new`.');
  lines.push('');

  lines.push('## Features');
  lines.push('');
  const features = doc.spec?.features ?? [];
  if (features.length === 0) lines.push('_None declared._');
  else for (const f of features) lines.push(`- ${f}`);
  lines.push('');
  if (doc.spec?.antiFeatures?.length) {
    lines.push('### Explicitly out of scope');
    lines.push('');
    for (const f of doc.spec.antiFeatures) lines.push(`- ${f}`);
    lines.push('');
  }
  if (doc.spec?.outOfScope?.length) {
    lines.push('### Not expressible in this IR');
    lines.push('');
    for (const f of doc.spec.outOfScope) lines.push(`- ${f}`);
    lines.push('');
  }

  lines.push('## Rendering contract');
  lines.push('');
  lines.push('| | |');
  lines.push('|---|---|');
  lines.push(`| stage | \`${doc.stage ?? 'fragment'}\` |`);
  lines.push(`| reference resolution | ${doc.render?.resolution?.join(' × ') ?? '?'} |`);
  lines.push(`| quality tier | \`${doc.render?.qualityTier ?? '?'}\` |`);
  lines.push(`| max loop iterations | ${doc.render?.maxIterations ?? '?'} |`);
  lines.push(`| transparency | ${doc.render?.transparent ? 'yes' : 'no'} |`);  lines.push(`| IR hash | \`${doc.hash ?? '(unhashed)'}\` |`);
  lines.push('');
  lines.push('All targets share one coordinate convention: **GLSL-like**, origin bottom-left, uv in');
  lines.push('[0,1], linear RGBA (no sRGB) inside the graph. Each adapter flips Y exactly once, in');
  lines.push('its entry-point epilogue.');
  lines.push('');

  lines.push('## How to build each target');
  lines.push('');
  for (const r of results) {
    const file = targetFilePath(r.target, slug).replace(`${slug}/`, '');
    lines.push(`### \`${r.target}\` — \`${file}\``);
    lines.push('');
    lines.push(BUILD_NOTES[r.target]);
    lines.push('');
  }

  const notes = doc.targetNotes ?? {};
  const noted = ALL_TARGETS.filter((t) => Boolean(notes[t]));
  if (noted.length > 0) {
    lines.push('## Per-target limitations');
    lines.push('');
    for (const t of noted) lines.push(`- **${t}** — ${notes[t]}`);
    lines.push('');
  }

  lines.push('## Parameters');
  lines.push('');
  const uniforms = results.find((r) => r.uniforms.length > 0)?.uniforms ?? [];
  if (uniforms.length === 0) {
    lines.push('_This shader has no exposed uniforms: every value is baked into the graph._');
  } else {
    lines.push('| name | type | value | range | label | group |');
    lines.push('|---|---|---|---|---|---|');
    for (const u of uniforms) {
      const range = u.min !== undefined || u.max !== undefined ? `${u.min ?? '−∞'} … ${u.max ?? '∞'}` : '—';
      lines.push(`| \`${u.name}\` | \`${u.type}\` | \`${formatValue(u.value)}\` | ${range} | ${u.label ?? '—'} | ${u.group ?? '—'} |`);
    }
    lines.push('');
    lines.push('Machine-readable form: [`params.json`](params.json). Override one value with');
    lines.push('`ai4shader params <doc> --set u_name=value`, or sweep a range with `ai4shader tune`.');
  }
  lines.push('');

  lines.push('## Files');
  lines.push('');
  // Relative to the bundle directory, which is where this README lives.
  for (const p of built) lines.push(`- \`${p.replace(`${slug}/`, '')}\``);
  lines.push('- `README.md` (this file)');
  lines.push('');
  return lines.join('\n');
}

const BUILD_NOTES: Record<TargetFormat, string> = {
  glsl:
    'Drop into a WebGL2 fragment shader host. Compile with `#version 300 es` and the host\n' +
    'uniform declarations (`iResolution`, `iTime`, …) from `HOST_SYMBOLS`.',
  wgsl: 'Drop into a WebGPU render pipeline. The WGSL shims needed by the shared library are\nemitted above the entry point.',
  hlsl: 'Compile with DXC (`dxc -T ps_6_0 -E main`). Matrices are lowered to `mul()` in the\nmatrix helper, so column-major semantics are preserved.',
  msl: 'Compile with `xcrun -sdk macosx metal -c shader.metal`. `mat*_mul_` helpers are lowered\nfor MSL row/column layout.',
  shaderlab:
    'Open the Unity project, drop `Shader.shader` into `Assets/`, and assign it to a material.\n' +
    'The material inspector exposes every uniform in `params.json` that carries a `label`.',
  ue:
    'Custom HLSL node for a UE material: paste `ue/CustomNode.hlsl` into a Custom node and wire\n' +
    'the inputs in the order listed in the node header. Sidecars in the same directory describe\n' +
    'the pin layout.',
};

function formatValue(v: number | boolean | number[]): string {
  if (Array.isArray(v)) return `(${v.map((x) => String(x)).join(', ')})`;
  return String(v);
}

function paramsJson(uniforms: readonly UniformDecl[]): string {
  const rows = uniforms.map((u) => ({
    name: u.name,
    type: u.type,
    value: u.value,
    min: u.min ?? null,
    max: u.max ?? null,
    step: u.step ?? null,
    label: u.label ?? null,
    group: u.group ?? null,
    origin: u.origin,
  }));
  return `${JSON.stringify(rows, null, 2)}\n`;
}

async function cmdEmit(argv: string[], cwd: string): Promise<number> {
  const { values, positionals } = parse('emit', argv);
  if (flag(values['help'])) {
    out(HELP['emit']!);
    return EXIT.OK;
  }
  const docPath = docArg('emit', positionals);
  const json = flag(values['json']);
  const force = flag(values['force']);
  const toStdout = flag(values['stdout']);

  const rawTarget = str(values['target']);
  if (!rawTarget) throw new UsageError('`emit` needs --target <glsl|wgsl|hlsl|msl|shaderlab|ue|all>');
  const targets: TargetFormat[] = rawTarget === 'all' ? [...ALL_TARGETS] : [asTarget(rawTarget)];

  const outDirArg = str(values['out']);
  if (!toStdout && !outDirArg) throw new UsageError('`emit` needs --out <dir>, or --stdout for a single target');
  if (toStdout && targets.length > 1) throw new UsageError('--stdout cannot be combined with --target all (that is 6 files)');
  // Resolved here so the bundle writer receives a concrete directory; the nullable form is
  // only needed to express "neither flag was given", which the guard above already rejected.
  const bundleDir = outDirArg ? (isAbsolute(outDirArg) ? outDirArg : resolve(cwd, outDirArg)) : null;

  const { doc, file } = await readDoc(docPath, cwd);
  const { catalog } = await requireIr();
  const results = await emitTargets(doc, targets, catalog);

  if (toStdout) {
    const only = results[0]!;
    process.stdout.write(only.source.endsWith('\n') ? only.source : `${only.source}\n`);
    if (!only.ok) {
      err('');
      printDiagnostics(only.diagnostics, '  ');
    }
    return only.ok ? EXIT.OK : EXIT.PROBLEM;
  }
  if (!bundleDir) throw new UsageError('`emit` needs --out <dir>, or --stdout for a single target');

  return writeBundle({ doc, file, outDir: bundleDir, targets, results, force, json });
}

/**
 * Write an emission bundle to disk and report it.
 *
 * Extracted from `cmdEmit` so that `author` and `refine` produce *byte-identical* bundles to
 * `emit`. The alternative — a second writer for the agent commands — is the classic way two
 * code paths that are supposed to agree drift apart, and the drift would show up as a
 * different `params.json` or a missing sidecar rather than as an error.
 */
async function writeBundle(args: {
  doc: ShaderDoc;
  file: string;
  outDir: string;
  targets: TargetFormat[];
  results: EmitOutcome[];
  force: boolean;
  json: boolean;
  /**
   * Suppress the human-readable report while still writing and still returning the exit code.
   *
   * Needed because `--json` promises *exactly one* JSON document on stdout: a caller doing
   * `JSON.parse(stdout)` breaks on any preceding progress line. Silence is the only correct
   * behaviour, but it must not extend to skipping the work — that was the bug this replaces.
   */
  quiet?: boolean;
}): Promise<number> {
  const { doc, file, outDir, targets, results, force, json, quiet } = args;
  const slug = slugify(doc.title ?? 'shader');
  const built: string[] = [];
  const written: { path: string; outcome: WriteOutcome }[] = [];

  const record = async (relPath: string, content: string): Promise<void> => {
    const abs = join(outDir, relPath);
    const outcome = await writeGuarded(abs, content, { force });
    written.push({ path: relPath, outcome });
    const rel = join(slug, relPath);
    if (!built.includes(rel)) built.push(rel);
  };

  for (const r of results) {
    await record(targetFilePath(r.target, slug), r.source);
    for (const sc of r.sidecars) await record(sidecarPath(r.target, slug, sc.name), sc.content);
  }

  // Uniforms are target-independent by construction, so the first non-empty list is *the*
  // list; `params.json` is the file `ai4shader params --set` and the tuning loop read.
  const uniforms = results.find((r) => r.uniforms.length > 0)?.uniforms ?? [];
  await record(`${slug}/params.json`, paramsJson(uniforms));
  await record(`${slug}/README.md`, readmeFor(doc, slug, results, built));

  const failures = results.filter((r) => !r.ok);
  const diverged = written.filter((w) => w.outcome === 'diverged');

  // `quiet` is checked first: the caller has already committed to emitting exactly one JSON
  // document, so neither this report nor the human-readable one below may appear.
  if (quiet) return failures.length === 0 ? EXIT.OK : EXIT.PROBLEM;

  if (json) {
    out(
      JSON.stringify(
        {
          file,
          out: outDir,
          slug,
          targets: results.map((r) => ({
            target: r.target,
            ok: r.ok,
            hash: r.hash,
            bytes: r.source.length,
            sidecars: r.sidecars.map((s) => s.name),
            diagnostics: r.diagnostics,
          })),
          files: written.map((w) => ({ path: w.path, outcome: w.outcome })),
          uniforms: uniforms.length,
        },
        null,
        2,
      ),
    );
    return failures.length === 0 ? EXIT.OK : EXIT.PROBLEM;
  }

  out(`${c.bold(file)} ${c.dim('→')} ${outDir}`);
  out(`${c.dim('slug')} ${slug}   ${c.dim('targets')} ${targets.join(', ')}   ${c.dim('hash')} ${doc.hash ?? '(unhashed)'}`);
  out();
  for (const w of written) reportWrite(w.path, w.outcome);
  out();

  for (const r of results) {
    if (!r.ok) {
      out(`${c.red(c.bold(`  ${r.target} FAILED`))}`);
      printDiagnostics(r.diagnostics);
    }
  }
  if (diverged.length > 0) {
    out(
      `${c.yellow('note')} ${diverged.length} file${diverged.length === 1 ? '' : 's'} differed and ${
        diverged.length === 1 ? 'was' : 'were'
      } written as \`.new\`. Pass --force to overwrite in place.`,
    );
    out();
  }
  out(
    failures.length === 0
      ? `${c.green(c.bold('  OK'))} ${results.length} target${results.length === 1 ? '' : 's'}, ${written.length} file${written.length === 1 ? '' : 's'}`
      : `${c.red(c.bold('  FAILED'))} ${failures.length}/${results.length} targets`,
  );
  return failures.length === 0 ? EXIT.OK : EXIT.PROBLEM;
}

/* -------------------------------------------------------------------------- */
/* author / refine — the agent loop                                            */
/* -------------------------------------------------------------------------- */

/**
 * Shared body of `author` and `refine`.
 *
 * They differ in exactly one respect: `author` starts from the prompt alone, `refine` starts
 * from an existing document. Everything else — provider selection, budgets, reporting, bundle
 * writing — is identical, so they share one implementation. Two near-identical commands would
 * be two places for the reporting to drift.
 */
async function runAgentCommand(
  command: 'author' | 'refine',
  argv: string[],
  cwd: string,
): Promise<number> {
  const { values, positionals } = parse(command, argv);
  if (flag(values['help'])) {
    out(HELP[command]!);
    return EXIT.OK;
  }

  const json = flag(values['json']);
  // Positional layout differs between the two: `author <prompt…>` and
  // `refine <doc> <prompt…>`. For `refine` the document is `positionals[0]`, so the
  // instruction is everything *after* it — joining all positionals (the obvious thing) hands
  // the document path to the model as part of the request.
  const words = command === 'author' ? positionals : positionals.slice(1);
  const prompt = str(values['prompt']) ?? words.join(' ').trim();
  if (!prompt) {
    throw new UsageError(
      command === 'author'
        ? '`author` needs a description: `ai4shader author "molten metal, slow flow"` or --prompt <text>'
        : '`refine` needs an instruction: `ai4shader refine doc.shader.json "more contrast"` or --prompt <text>',
    );
  }

  // `docArg` insists on exactly one positional, but `refine` legitimately has more (the
  // document plus the instruction), so it is handed only the first. Passing the whole list
  // produced "`refine` takes one document, got 2" for the documented usage.
  const basedOn = command === 'refine' ? await readDoc(docArg('refine', positionals.slice(0, 1)), cwd) : null;

  const { catalog } = await requireIr();
  const agent = await requireAgent();

  const choice = chooseProvider({
    ...(str(values['provider']) !== undefined ? { provider: str(values['provider'])! } : {}),
    ...(str(values['api-key']) !== undefined ? { apiKey: str(values['api-key'])! } : {}),
    ...(str(values['base-url']) !== undefined ? { baseUrl: str(values['base-url'])! } : {}),
    ...(str(values['model']) !== undefined ? { model: str(values['model'])! } : {}),
    ...(values['vision'] !== undefined ? { vision: flag(values['vision']) } : {}),
  });
  const provider = makeProvider(agent, choice);

  if (choice.kind === 'none') {
    // Say it up front rather than letting the user infer it from a puzzling result. Silence
    // here would let someone believe a model was consulted and judge the output accordingly.
    err(`${c.yellow('note')} ${describeProvider(choice)}`);
    if (command === 'author' && !basedOn) {
      err(
        `${c.yellow('note')} \`author\` cannot draft a graph without a provider — ` +
          `it will report what is missing instead of inventing one. Try \`--provider stub\` for an offline demo.`,
      );
    }
    err('');
  }

  // The stage callback is what makes a slow run legible: a model call can take seconds, and a
  // command that prints nothing for 30 seconds reads as hung.
  const showProgress = !json;
  const onStage = showProgress
    ? (record: { stage: string; round: number; summary: string }): void => {
        out(`${c.dim(`  ${String(record.round).padStart(2)} ${record.stage.padEnd(9)}`)} ${record.summary}`);
      }
    : undefined;

  const targetArg = str(values['target']);
  const requireTargets =
    targetArg === undefined
      ? undefined
      : targetArg === 'all'
        ? [...ALL_TARGETS]
        : [asTarget(targetArg)];

  const result = (await agent.runAgent({
    prompt,
    catalog,
    provider,
    reviewRequirements: choice.kind === 'http',
    ...(basedOn ? { basedOn: basedOn.doc } : {}),
    ...(requireTargets ? { requireTargets } : {}),
    ...(onStage ? { onStage } : {}),
  })) as AgentResult;

  const doc = result.doc as ShaderDoc;

  /**
   * Write the requested artefacts **before** deciding how to report.
   *
   * `--json` used to return here, before anything was written, which meant
   * `author --json --doc-out out.shader.json` printed a document and left the filesystem
   * untouched. The exit code said success and the file was not there: the worst shape of bug,
   * because a script around it sees a clean run and then fails on the missing file, far from
   * the cause.
   *
   * `--json` is an *output format*, not a mode. Every other command already treats it that
   * way — `emit --json` still writes the bundle, `params --json --out` still writes the
   * document — so this brings the two agent commands in line with the rest of the CLI.
   */
  const outDirArg = str(values['out']);
  const outDir = outDirArg ? (isAbsolute(outDirArg) ? outDirArg : resolve(cwd, outDirArg)) : null;
  const docOut = str(values['doc-out']) ?? (basedOn ? basedOn.file : undefined);

  const written: string[] = [];
  if (docOut) {
    const target = basedOn && !str(values['doc-out']) ? basedOn.file : resolve(cwd, docOut);
    written.push(await writeDocFile(target, doc, cwd));
  }

  let bundleExit: number | null = null;
  if (outDir && Object.keys(result.emissions).length > 0) {
    const { catalog: fullCatalog } = await requireIr();
    const targetList = requireTargets ?? (doc.targets as TargetFormat[]) ?? [...ALL_TARGETS];
    const results = await emitTargets(doc, targetList, fullCatalog);
    bundleExit = await writeBundle({
      doc,
      file: docOut ?? '(authored)',
      outDir,
      targets: targetList,
      results,
      force: flag(values['force']),
      // The bundle's own report is suppressed under `--json`: there must be exactly one JSON
      // document on stdout, and this command already emits one.
      json: false,
      quiet: json,
    });
  }

  if (json) {
    out(
      JSON.stringify(
        {
          stopReason: result.stopReason,
          clean: result.clean,
          modelCalls: result.usage?.calls ?? 0,
          remaining: result.remaining,
          trace: result.trace.map((t) => ({ stage: t.stage, round: t.round, summary: t.summary })),
          // Report what actually landed on disk. A machine consumer needs to know where the
          // artefacts are, and whether writing them succeeded.
          files: written,
          outDir,
          ...(bundleExit !== null ? { bundleOk: bundleExit === EXIT.OK } : {}),
          doc,
        },
        null,
        2,
      ),
    );
    // A failed bundle is a real failure even when the loop itself succeeded, so it must move
    // the exit code.
    if (bundleExit !== null && bundleExit !== EXIT.OK) return bundleExit;
    return result.clean ? EXIT.OK : EXIT.PROBLEM;
  }

  out();
  out(`${c.bold(agent.summariseRun(result))}`);

  if (result.remaining.length > 0) {
    out();
    out(`${c.red(c.bold('  STILL BROKEN'))} ${result.remaining.length} error(s):`);
    printDiagnostics(result.remaining);
  }

  // Write the document before the bundle: the document is the artefact worth keeping even if
  // a target fails to emit, and a user who has to re-run an expensive model call because the
  // output directory was missing would rightly be annoyed.
  if (written.length) {
    out();
    for (const p of written) out(`${c.dim('document')} ${p}`);
  }

  if (bundleExit !== null) {
    return bundleExit;
  }

  if (!docOut && !outDir) {
    out();
    out(
      `${c.yellow('note')} nothing was written: pass --out <dir> for a shader bundle, ` +
        `--doc-out <file> for the IR document, or both.`,
    );
  }

  return result.clean ? EXIT.OK : EXIT.PROBLEM;
}

function asTarget(value: string): TargetFormat {
  if ((ALL_TARGETS as string[]).includes(value)) return value as TargetFormat;
  throw new UsageError(`unknown target \`${value}\` — expected one of: ${ALL_TARGETS.join(', ')}, all`);
}

/* -------------------------------------------------------------------------- */
/* params                                                                      */
/* -------------------------------------------------------------------------- */

type SetOverride = { name: string; text: string };

function parseSetArgs(raw: string[]): SetOverride[] {
  return raw.map((entry) => {
    const eq = entry.indexOf('=');
    if (eq <= 0) throw new UsageError(`--set expects <name>=<value>, got \`${entry}\``);
    const name = entry.slice(0, eq).trim();
    const text = entry.slice(eq + 1).trim();
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) throw new UsageError(`\`${name}\` is not a valid uniform name`);
    return { name, text };
  });
}

/** Parse a CLI value into the `ParamValue` the uniform's declared type demands. */
function overrideFor(row: UniformRow, text: string): Extract<UniformValue, { kind: 'param' }> {
  const nums = (arity: number): number[] => {
    const parts = text.split(',').map((p) => p.trim());
    if (parts.length !== arity) {
      throw new UsageError(`\`${row.name}\` is ${row.type}: expected ${arity} comma-separated numbers, got ${parts.length}`);
    }
    return parts.map((p) => {
      const n = Number(p);
      if (!Number.isFinite(n)) throw new UsageError(`\`${row.name}\`: \`${p}\` is not a number`);
      return n;
    });
  };
  const scalar = (): number => {
    const n = Number(text);
    if (!Number.isFinite(n)) throw new UsageError(`\`${row.name}\`: \`${text}\` is not a number`);
    return n;
  };

  let value: number | boolean | number[];
  switch (row.type) {
    case 'float':
      value = scalar();
      break;
    case 'int':
      value = Math.round(scalar());
      break;
    case 'bool': {
      const t = text.toLowerCase();
      if (!['true', 'false', '1', '0', 'yes', 'no'].includes(t)) {
        throw new UsageError(`\`${row.name}\` is bool: \`${text}\` is not true/false/1/0/yes/no`);
      }
      value = t === 'true' || t === '1' || t === 'yes';
      break;
    }
    case 'vec2':
      value = nums(2);
      break;
    case 'vec3':
      value = nums(3);
      break;
    default:
      value = nums(4);
      break;
  }

  // An out-of-range value is rejected, not clamped. A uniform's declared range is the range
  // its slider can produce, so `--set u_gain=35` is almost always a typo (or a decimal point
  // lost in a shell pipeline) — clamping it to 1 would silently give the caller a shader
  // they did not ask for, and the caller is usually a script that will not read the warning.
  if (typeof value === 'number') {
    if (row.min !== undefined && value < row.min) {
      throw new UsageError(`\`${row.name}\` = ${value} is below its minimum ${row.min} (range ${row.min}..${row.max ?? '∞'})`);
    }
    if (row.max !== undefined && value > row.max) {
      throw new UsageError(`\`${row.name}\` = ${value} is above its maximum ${row.max} (range ${row.min ?? '−∞'}..${row.max})`);
    }
  }

  return {
    kind: 'param',
    name: row.name,
    type: row.type,
    value,
    ...(row.min !== undefined ? { min: row.min } : {}),
    ...(row.max !== undefined ? { max: row.max } : {}),
    ...(row.step !== undefined ? { step: row.step } : {}),
    ...(row.label !== undefined ? { label: row.label } : {}),
    ...(row.group !== undefined ? { group: row.group } : {}),
  };
}

function printUniformTable(rows: readonly UniformRow[], hash: string | undefined): void {
  if (rows.length === 0) {
    out(c.dim('  (no uniforms — every value is baked into the graph)'));
    return;
  }
  const headers = ['name', 'type', 'value', 'min', 'max', 'step', 'label', 'group', 'origin'];
  const cells = rows.map((r) => [
    r.name,
    r.type,
    formatValue(r.value),
    r.min !== undefined ? String(r.min) : '—',
    r.max !== undefined ? String(r.max) : '—',
    r.step !== undefined ? String(r.step) : '—',
    r.label ?? '—',
    r.group ?? '—',
    `${r.node}.${r.param}`,
  ]);
  const widths = headers.map((h, i) => Math.max(h.length, ...cells.map((row) => row[i]!.length)));
  const line = (row: string[]): string => row.map((cell, i) => cell.padEnd(widths[i]!)).join('  ').trimEnd();
  out(`  ${c.bold(line(headers))}`);
  out(`  ${c.dim(widths.map((w) => '─'.repeat(w)).join('  '))}`);
  for (const row of cells) out(`  ${line(row)}`);
  if (hash) out(`\n  ${c.dim('hash')} ${hash}`);
}

async function cmdParams(argv: string[], cwd: string): Promise<number> {
  const { values, positionals } = parse('params', argv);
  if (flag(values['help'])) {
    out(HELP['params']!);
    return EXIT.OK;
  }
  const docPath = docArg('params', positionals);
  const json = flag(values['json']);
  const overrides = parseSetArgs(list(values['set']));
  const outArg = str(values['out']);

  const { doc, file } = await readDoc(docPath, cwd);
  const ir = await loadIr();
  if (!ir.complete && !json) err(`${c.yellow('warning')} ${ir.note}`);

  let result = doc;
  let applied = 0;
  let diagnostics: Diagnostic[] = [];
  let patchedHash: string | undefined;

  if (overrides.length > 0) {
    const rows = collectUniformRows(doc);
    const byName = new Map(rows.map((r) => [r.name, r] as const));
    const ops = overrides.map(({ name, text }) => {
      const row = byName.get(name);
      if (!row) {
        const known = rows.length > 0 ? rows.map((r) => r.name).join(', ') : '(none)';
        throw new UsageError(`no uniform named \`${name}\` in ${file}. Available: ${known}`);
      }
      return { op: 'set_param' as const, node: row.node, param: row.param, value: overrideFor(row, text), reason: 'ai4shader params --set' };
    });

    const patch = { summary: `params --set ${overrides.map((o) => o.name).join(' ')}`, author: { kind: 'human' as const }, ops };
    const { applyPatch } = await patcher();
    const res = applyPatch(doc, patch, ir.catalog);
    diagnostics = res.diagnostics;
    if (!res.ok) {
      err(`${c.red(c.bold('rejected'))} the patch was not applied; the document is unchanged.`);
      printDiagnostics(diagnostics, '  ');
      return EXIT.PROBLEM;
    }
    result = res.doc;
    applied = res.applied;
    patchedHash = result.hash;
  }

  const rows = collectUniformRows(result);
  let writtenFile: string | null = null;
  if (outArg) writtenFile = await writeDocFile(outArg, result, cwd);

  if (json) {
    out(
      JSON.stringify(
        {
          file,
          applied,
          hash: result.hash ?? null,
          previousHash: doc.hash ?? null,
          wrote: writtenFile,
          uniforms: rows.map((r) => ({
            name: r.name,
            type: r.type,
            value: r.value,
            min: r.min ?? null,
            max: r.max ?? null,
            step: r.step ?? null,
            label: r.label ?? null,
            group: r.group ?? null,
            origin: { node: r.node, param: r.param },
          })),
          diagnostics,
        },
        null,
        2,
      ),
    );
    return EXIT.OK;
  }

  out(`${c.bold(file)}`);
  out();
  printUniformTable(rows, result.hash);
  if (overrides.length > 0) {
    out();
    out(`  ${c.green('✓')} applied ${applied} override${applied === 1 ? '' : 's'} through the patcher`);
    for (const o of overrides) out(`    ${c.dim(o.name)} = ${c.cyan(o.text)}`);
    out(`  ${c.dim('hash')} ${doc.hash ?? '(none)'} ${c.dim('→')} ${result.hash ?? '(none)'}`);
    const warnings = diagnostics.filter((d) => d.severity !== 'info');
    if (warnings.length > 0) {
      out();
      printDiagnostics(warnings, '  ');
    }
  }
  if (writtenFile) out(`\n  ${c.green('+')} ${writtenFile}`);
  return EXIT.OK;
}

/* -------------------------------------------------------------------------- */
/* tune                                                                        */
/* -------------------------------------------------------------------------- */

const SWEEP_CAP = 256;

type RangeSpec = { name: string; lo: number; hi: number; explicit: boolean };

type Variant = {
  id: string;
  index: number;
  dir: string;
  values: Record<string, number>;
  passed: boolean;
};

/** `u_scale=0.1..8` -> a range; `u_scale=8` -> a single explicit value. */
function parseRange(entry: string): RangeSpec {
  const eq = entry.indexOf('=');
  if (eq <= 0) throw new UsageError(`--param expects <name>=<lo>..<hi>, got \`${entry}\``);
  const name = entry.slice(0, eq).trim();
  const body = entry.slice(eq + 1).trim();
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) throw new UsageError(`\`${name}\` is not a valid uniform name`);
  const dots = body.indexOf('..');
  if (dots === -1) {
    const v = Number(body);
    if (!Number.isFinite(v)) throw new UsageError(`--param ${name}: \`${body}\` is neither a number nor <lo>..<hi>`);
    return { name, lo: v, hi: v, explicit: true };
  }
  const lo = Number(body.slice(0, dots));
  const hi = Number(body.slice(dots + 2));
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
    throw new UsageError(`--param ${name}: both ends of \`${body}\` must be numbers`);
  }
  if (lo > hi) throw new UsageError(`--param ${name}: range is inverted (${lo} > ${hi})`);
  return { name, lo, hi, explicit: false };
}

/** `lo + (hi-lo)*i/(steps-1)`; a single value stays a single value. */
function sample(lo: number, hi: number, steps: number): number[] {
  if (lo === hi) return [lo];
  if (steps === 1) return [lo];
  return Array.from({ length: steps }, (_, i) => lo + ((hi - lo) * i) / (steps - 1));
}

const fmt4 = (n: number): string => n.toFixed(4);

/** Cartesian product, first spec varying slowest so the grid order is conventional. */
function buildVariants(specs: RangeSpec[], steps: number, slug: string): Variant[] {
  const axes: { name: string; values: number[] }[] = [];
  // Repeating a name means "these are the samples I want for it", not "a second axis".
  const order: string[] = [];
  const grouped = new Map<string, number[]>();
  for (const spec of specs) {
    if (!grouped.has(spec.name)) {
      grouped.set(spec.name, []);
      order.push(spec.name);
    }
    const values = grouped.get(spec.name)!;
    // Deduplicate so `--param x=0.5..2 --param x=0.5` does not produce the same point twice.
    for (const v of sample(spec.lo, spec.hi, steps)) if (!values.includes(v)) values.push(v);
  }
  for (const name of order) axes.push({ name, values: grouped.get(name)!.sort((a, b) => a - b) });

  const total = axes.reduce((acc, axis) => acc * axis.values.length, 1);
  if (total > SWEEP_CAP) {
    const shape = axes.map((a) => `${a.name}×${a.values.length}`).join(' × ');
    throw new UsageError(
      `sweep would produce ${total} variants (${shape}), above the ${SWEEP_CAP} cap.\n` +
        `  Reduce --steps, drop a --param, or give one uniform explicit values instead of a range.`,
    );
  }

  const outVariants: Variant[] = [];
  for (let i = 0; i < total; i++) {
    const values: Record<string, number> = {};
    let rest = i;
    // Last axis varies fastest.
    for (let a = axes.length - 1; a >= 0; a--) {
      const axis = axes[a]!;
      values[axis.name] = axis.values[rest % axis.values.length]!;
      rest = Math.floor(rest / axis.values.length);
    }
    // The id is built from the values, sorted by name, so it is stable across runs and
    // diffable in a regression report. The index alone would renumber on any edit.
    const id = axes
      .map((a) => a.name)
      .sort()
      .map((n) => `${n}=${fmt4(values[n]!)}`)
      .join('__');
    outVariants.push({
      id,
      index: i + 1,
      dir: `sweep-${i + 1}`,
      values,
      passed: false,
    });
  }
  void slug;
  return outVariants;
}

async function cmdTune(argv: string[], cwd: string): Promise<number> {
  const { values, positionals } = parse('tune', argv);
  if (flag(values['help'])) {
    out(HELP['tune']!);
    return EXIT.OK;
  }
  const docPath = docArg('tune', positionals);
  const json = flag(values['json']);
  const doEmit = flag(values['emit']);
  const force = flag(values['force']);

  const rawParams = list(values['param']);
  if (rawParams.length === 0) {
    const count = collectUniformRows((await readDoc(docPath, cwd)).doc).length;
    throw new UsageError(
      `\`tune\` needs at least one --param <name>=<lo>..<hi>` +
        (count > 0 ? ` (the document has ${count} uniform${count === 1 ? '' : 's'}; see \`ai4shader params ${docPath}\`)` : ''),
    );
  }
  const stepsRaw = str(values['steps']);
  const steps = stepsRaw === undefined ? 5 : Number(stepsRaw);
  if (!Number.isInteger(steps) || steps < 1) throw new UsageError(`--steps must be a positive integer, got \`${stepsRaw}\``);

  const specs = rawParams.map(parseRange);
  const { doc, file } = await readDoc(docPath, cwd);
  const outDirArg = str(values['out']);
  const outDir = outDirArg ? (isAbsolute(outDirArg) ? outDirArg : resolve(cwd, outDirArg)) : resolve(cwd, 'dist/sweep');

  if (doEmit) await requireIr(); // fail before creating anything if emitters are missing

  const slug = slugify(doc.title ?? 'shader');
  const target: TargetFormat = str(values['target']) ? asTarget(str(values['target'])!) : docTargets(doc)[0]!;
  const variants = buildVariants(specs, steps, slug);

  // Which node+port owns each swept uniform — needed to build the patch ops.
  const rows = collectUniformRows(doc);
  const byName = new Map(rows.map((r) => [r.name, r] as const));
  for (const spec of specs) {
    if (!byName.has(spec.name)) {
      const known = rows.length > 0 ? rows.map((r) => r.name).join(', ') : '(none)';
      throw new UsageError(`no uniform named \`${spec.name}\` in ${file}. Available: ${known}`);
    }
  }

  const { applyPatch, hashDoc } = await patcher();
  const ir = doEmit ? await requireIr() : await loadIr();
  const baseHash = doc.hash ?? hashDoc(doc);

  out(`${c.bold(file)}`);
  out(
    `  ${c.dim('sweep')} ${specs.map((s) => (s.lo === s.hi ? `${s.name}=${s.lo}` : `${s.name}=${s.lo}..${s.hi}`)).join('  ')}` +
      `   ${c.dim('steps')} ${steps}   ${c.dim('variants')} ${variants.length}`,
  );
  out(`  ${c.dim('out')} ${outDir}   ${c.dim('emit')} ${doEmit ? target : 'no (pass --emit)'}`);
  out();

  const manifest: Record<string, unknown>[] = [];
  let emitted = 0;
  let failed = 0;

  for (const variant of variants) {
    const ops = Object.keys(variant.values)
      .sort()
      .map((name) => {
        const row = byName.get(name)!;
        return {
          op: 'set_param' as const,
          node: row.node,
          param: row.param,
          value: {
            kind: 'param' as const,
            name: row.name,
            type: row.type,
            value: variant.values[name]!,
            ...(row.min !== undefined ? { min: row.min } : {}),
            ...(row.max !== undefined ? { max: row.max } : {}),
            ...(row.step !== undefined ? { step: row.step } : {}),
            ...(row.label !== undefined ? { label: row.label } : {}),
            ...(row.group !== undefined ? { group: row.group } : {}),
          },
        };
      });

    const res = applyPatch(doc, { summary: `sweep ${variant.id}`, author: { kind: 'optimizer' as const }, ops }, ir.catalog);
    if (!res.ok) {
      // A rejected override is a usage problem, not a shader problem: report and stop.
      err(`${c.red(c.bold('rejected'))} variant ${variant.index} (${variant.id})`);
      printDiagnostics(res.diagnostics, '  ');
      return EXIT.USAGE;
    }

    let files: string[] = [];
    let diagnostics: Diagnostic[] = [];
    let ok = true;
    if (doEmit) {
      const results = await emitTargets(res.doc, [target], ir.catalog);
      const result = results[0]!;
      ok = result.ok;
      diagnostics = result.diagnostics;
      const written: string[] = [];
      const write = async (rel: string, content: string): Promise<void> => {
        const outcome = await writeGuarded(join(outDir, variant.dir, rel), content, { force });
        written.push(outcome === 'diverged' ? `${variant.dir}/${rel}.new` : `${variant.dir}/${rel}`);
      };
      await write(targetFilePath(target, slug).replace(`${slug}/`, ''), result.source);
      for (const sc of result.sidecars) await write(sc.name, sc.content);
      files = written;
      if (ok) emitted++;
      else failed++;
      out(`  ${ok ? c.green('✓') : c.red('✗')} ${String(variant.index).padStart(3)} ${variant.id}`);
    } else {
      out(`  ${c.dim('·')} ${String(variant.index).padStart(3)} ${variant.id} ${c.dim(res.doc.hash ?? '')}`);
    }
    variant.passed = ok;

    manifest.push({
      id: variant.id,
      index: variant.index,
      dir: variant.dir,
      values: variant.values,
      docHash: res.doc.hash ?? null,
      target: doEmit ? target : null,
      files,
      ok,
      diagnostics,
    });
  }

  const sweep = {
    doc: file,
    slug,
    baseHash,
    steps,
    target: doEmit ? target : null,
    cap: SWEEP_CAP,
    // The machine interface a CI regression job consumes: parameter *ranges* first, then
    // every variant with a value-derived id, so two runs can be diffed by id.
    params: specs.map((s) => ({ name: s.name, lo: s.lo, hi: s.hi, explicit: s.explicit })),
    variants: manifest,
  };
  const sweepPath = join(outDir, 'sweep.json');
  await mkdir(outDir, { recursive: true });
  const sweepJson = `${JSON.stringify(sweep, null, 2)}\n`;
  const outcome = await writeGuarded(sweepPath, sweepJson, { force });

  out();
  reportWrite('sweep.json', outcome);
  if (json) {
    out();
    process.stdout.write(sweepJson);
    return failed === 0 ? EXIT.OK : EXIT.PROBLEM;
  }
  out(
    doEmit
      ? `${c.green(c.bold('  DONE'))} ${variants.length} variants, ${emitted} emitted${failed > 0 ? `, ${c.red(`${failed} failed`)}` : ''}`
      : `${c.green(c.bold('  DONE'))} ${variants.length} variants planned (add --emit to write shaders)`,
  );
  return failed === 0 ? EXIT.OK : EXIT.PROBLEM;
}

/* -------------------------------------------------------------------------- */
/* Entry point                                                                 */
/* -------------------------------------------------------------------------- */

export async function main(argv: string[], cwd = process.cwd()): Promise<number> {
  const [command, ...rest] = argv;
  if (command === undefined || command === '--help' || command === '-h' || command === 'help') {
    out(USAGE);
    return EXIT.OK;
  }
  if (command === '--version' || command === '-v') {
    out('0.1.0');
    return EXIT.OK;
  }
  if (!(command in OPTIONS)) {
    err(`${c.red('error')} unknown command \`${command}\``);
    err('');
    err(USAGE);
    return EXIT.USAGE;
  }

  const name = command as CommandName;
  try {
    switch (name) {
      case 'validate':
        return await cmdValidate(rest, cwd);
      case 'emit':
        return await cmdEmit(rest, cwd);
      case 'params':
        return await cmdParams(rest, cwd);
      case 'tune':
        return await cmdTune(rest, cwd);
      case 'author':
        return await runAgentCommand('author', rest, cwd);
      case 'refine':
        return await runAgentCommand('refine', rest, cwd);
    }
  } catch (e) {
    if (e instanceof UsageError) {
      err(`${c.red('usage error')} ${e.message}`);
      return EXIT.USAGE;
    }
    if (e instanceof CliError) {
      err(`${c.red('error')} ${e.message}`);
      return EXIT.PROBLEM;
    }
    err(`${c.red('unexpected error')} ${(e as Error).stack ?? String(e)}`);
    return EXIT.PROBLEM;
  }
}

/**
 * Run when executed, stay silent when imported (the tests import `main` directly and by
 * spawning `node src/cli.ts`, and a module that calls `process.exit` on import cannot be
 * tested).
 */
const invokedDirectly =
  process.argv[1] !== undefined && import.meta.url === pathToFileURL(resolve(process.argv[1])).href;
if (invokedDirectly) {
  process.exitCode = await main(process.argv.slice(2));
}

/* Re-exported for the tests and for embedding the CLI in another tool. */
export { cmdValidate, cmdEmit, cmdParams, cmdTune, EXIT, OPTIONS };
