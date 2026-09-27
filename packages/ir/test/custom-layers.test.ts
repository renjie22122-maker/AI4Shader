import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createDoc, emit, normalizeEdges } from '../src/index.js';
import { CATALOG } from '../src/catalog/index.js';
import '../src/targets/index.js';

// Reduced from the actual DeepSeek rainy-glass draft: both layers declared hi and nrm.
export function layeredDoc() {
  return createDoc({ targets: ['glsl', 'wgsl', 'hlsl', 'msl', 'shaderlab', 'ue'],
    graph: normalizeEdges({ nodes: [
      { id: 'drops', kind: 'custom_glsl', body: 'float hi = 0.3;\nvec2 nrm = vec2(0.1, 0.2);\nreturn vec4(nrm, hi, 0.4);' },
      { id: 'trails', kind: 'custom_glsl', inputs: { a: { node: 'drops', port: 'value' } },
        body: 'float hi = 0.6;\nvec2 nrm = vec2(0.3, 0.4);\nreturn ${a} + vec4(nrm, hi, 0.1);' },
      { id: 'out', kind: 'output', inputs: { color: { node: 'trails', port: 'value' } } },
    ], edges: [] }),
  });
}

test('independent custom layers have private locals and graph-visible outputs in every target', () => {
  const doc = layeredDoc();
  for (const target of doc.targets) {
    const result = emit(doc, { catalog: CATALOG, target });
    assert.deepEqual(result.diagnostics.filter(d => d.severity === 'error'), [], target);
    assert.match(result.source, /\{ \/\/ custom_glsl drops/);
    assert.match(result.source, /\{ \/\/ custom_glsl trails/);
    assert.match(result.source, /L_trails_value\s*=/);
  }
});

test('transparent artifacts carry host blending instructions and Unity render state', () => {
  for (const transparent of [false, true]) {
    const doc = layeredDoc();
    doc.render.transparent = transparent;
    for (const target of doc.targets) {
      const result = emit(doc, { catalog: CATALOG, target });
      assert.deepEqual(result.diagnostics.filter(d => d.severity === 'error'), []);
      assert.equal(result.source.includes('TRANSPARENCY:'), transparent, target);
      if (target === 'shaderlab') {
        assert.equal(result.source.includes('Blend SrcAlpha OneMinusSrcAlpha'), transparent);
        assert.equal(result.source.includes('ZWrite Off'), transparent);
        assert.equal(result.source.includes('"Queue"="Transparent"'), transparent);
      }
    }
  }
});

test('custom layers with multiple scalar ports keep temporaries private too', () => {
  const catalog = { ...CATALOG, custom_glsl: { ...CATALOG.custom_glsl,
    outputs: [{ name: 'value', type: 'float' as const, label: 'Value' }, { name: 'extra', type: 'float' as const, label: 'Extra' }],
  } };
  for (const target of layeredDoc().targets) {
    const result = emit(layeredDoc(), { catalog, target });
    assert.deepEqual(result.diagnostics.filter(d => d.severity === 'error'), [], target);
    assert.match(result.source, /\{ \/\/ custom_glsl drops/);
    assert.match(result.source, /\{ \/\/ custom_glsl trails/);
    assert.match(result.source, /L_trails_extra\s*=/);
  }
});
