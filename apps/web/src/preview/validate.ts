import type { Diagnostic, ShaderDoc } from '../../../../packages/ir/src/index.js';
import type { AgentResult } from '../../../../packages/agent/src/index.js';

/** A private context validates generated fragments without disturbing the visible preview. */
export function createFragmentValidator() {
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2');
  const cache = new Map<string, Diagnostic[]>();
  return {
    available: gl !== null,
    async validate(_doc: ShaderDoc, emissions: AgentResult['emissions']): Promise<Diagnostic[]> {
      if (!gl || !emissions.glsl) return [];
      if (gl.isContextLost()) throw new Error('WebGL2 validation context was lost');
      const source = emissions.glsl.source;
      const hit = cache.get(source);
      if (hit) return hit;
      const shader = gl.createShader(gl.FRAGMENT_SHADER);
      if (!shader) throw new Error('Unable to allocate a shader for validation');
      try {
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        const ok = gl.getShaderParameter(shader, gl.COMPILE_STATUS);
        const log = gl.getShaderInfoLog(shader) ?? '';
        const lines = source.split('\n');
        const excerpts = [...log.matchAll(/ERROR:\s*\d+:(\d+):/g)].slice(0, 6).map((match) => {
          const line = Number(match[1]);
          return lines.slice(Math.max(0, line - 3), line + 2)
            .map((text, i) => `${Math.max(1, line - 2) + i}: ${text}`).join('\n');
        });
        const result: Diagnostic[] = ok ? [] : [{
          severity: 'error', code: 'AG030_GLSL_COMPILE',
          message: `WebGL2 fragment compiler:\n${log}\n${excerpts.join('\n')}`,
          fix: 'Fix the custom_glsl body using the exact compiler error and source excerpt. Preserve all effect layers.',
        }];
        cache.set(source, result);
        return result;
      } finally {
        gl.deleteShader(shader);
      }
    },
    dispose() {
      cache.clear();
      gl?.getExtension('WEBGL_lose_context')?.loseContext();
    },
  };
}
