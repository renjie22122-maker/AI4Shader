// Scratch: verify the host-blending notes a `transparent` document now carries.
import { createDoc, emit, normalizeEdges } from '../packages/ir/dist/index.js';
import { CATALOG } from '../packages/ir/dist/catalog/index.js';
import '../packages/ir/dist/targets/index.js';

function docFor(transparent) {
  return createDoc({
    targets: ['glsl', 'wgsl', 'hlsl', 'msl', 'shaderlab', 'ue'],
    render: { transparent },
    graph: normalizeEdges({
      nodes: [
        { id: 'uv', kind: 'uv' },
        { id: 'n', kind: 'fbm', inputs: { uv: { node: 'uv', port: 'uv' } } },
        { id: 'out', kind: 'output', inputs: { color: { node: 'n', port: 'value' } } },
      ],
      edges: [],
    }),
  });
}

for (const transparent of [true, false]) {
  for (const target of ['glsl', 'wgsl', 'hlsl', 'msl', 'shaderlab', 'ue']) {
    const r = emit(docFor(transparent), { catalog: CATALOG, target });
    const hits = r.source.split('\n').filter((l) => /blend|Blend|ZWrite|Queue|alpha|Alpha/.test(l));
    console.log(`\n===== transparent:${transparent}  ${target}  (errors ${r.diagnostics.filter((d) => d.severity === 'error').length}) =====`);
    console.log(hits.map((l) => l.trim()).slice(0, 7).join('\n') || '(no mention of blending / alpha)');
  }
}
