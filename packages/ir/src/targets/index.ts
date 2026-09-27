/**
 * AI4Shader — `@ai4shader/ir/targets`
 * ====================================
 *
 * Aggregation entry point. Emitters self-register through `registerTarget()` at module
 * scope, so importing this module is what makes `emit()` able to do anything at all:
 *
 *     import { emit } from '@ai4shader/ir/targets';
 *
 * The six emitters live in one module (`_shared.ts`) because they share the assembly
 * framework, the type table and the texture-fetch lowering — splitting them would mean
 * six copies of the same prologue logic, which is exactly the drift this package exists
 * to prevent. The single side-effect import below therefore registers **all six**
 * targets; the exported names are the ones the CLI and the Vite workbench resolve:
 * `emit`, `registerTarget`, `Emission`, `EmitOptions`.
 */

import './_shared.js';

export { emit, registerTarget } from '../emitter.js';
export type { Emission, EmitOptions } from '../emitter.js';
