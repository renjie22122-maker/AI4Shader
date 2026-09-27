/**
 * Endpoint presets, readiness and model-list parsing
 * =================================================
 *
 * The pure half of `providers.ts`. The network half (`probeEndpoint`, `listModels`) is
 * exercised against a real socket elsewhere — a fake OpenAI endpoint in
 * `apps/cli/test/author-http.test.ts` — because a test that stubs `fetch` proves the request
 * was *shaped* right, not that a real server accepts it.
 *
 * `parseModelIds` is here because it is the one place where a wrong answer is silent. Every
 * service returns the OpenAI `{ data: [{ id }] }` envelope, but "OpenAI-compatible" is a
 * spectrum and Ollama's own shape is `{ models: [{ name }] }`. A parser that returned `[]` for
 * a shape it did not recognise would be indistinguishable, in the UI, from a service that has
 * no models — so the shapes are pinned, including the ones that must fail closed.
 *
 * The preset checks are not decoration either. `presetForBaseUrl` keeps the dropdown honest
 * when someone types an address by hand, and `endpointReady` decides whether 生成 is clickable
 * — the old version of which disabled the button for a local Ollama that needs no key and
 * never said why.
 */

import assert from 'node:assert/strict';
import { test } from 'node:test';

import {
  ENDPOINT_PRESETS,
  endpointReady,
  parseModelIds,
  presetById,
  presetForBaseUrl,
} from '../src/providers.ts';

/* -------------------------------------------------------------------------- */
/* parseModelIds                                                               */
/* -------------------------------------------------------------------------- */

test('parseModelIds reads the OpenAI envelope', () => {
  assert.deepEqual(parseModelIds('{"object":"list","data":[{"id":"a"},{"id":"b"}]}'), ['a', 'b']);
});

test('parseModelIds tolerates the other shapes a compatible service may use', () => {
  // Ollama's native shape. `{ result: [...] }` is what several gateways wrap with.
  assert.deepEqual(parseModelIds('{"models":[{"name":"llama3.1:8b"}]}'), ['llama3.1:8b']);
  assert.deepEqual(parseModelIds('{"result":["m1","m2"]}'), ['m1', 'm2']);
  assert.deepEqual(parseModelIds('["x","y"]'), ['x', 'y']);
  assert.deepEqual(parseModelIds('{"data":[{"model":"gpt-x"}]}'), ['gpt-x']);
});

test('parseModelIds preserves the server order and de-duplicates', () => {
  // Order is information: a provider that lists its flagship first is saying something, and
  // alphabetising would throw that away. Duplicates are a real occurrence across gateways.
  const body = '{"data":[{"id":"z"},{"id":"a"},{"id":"z"},{"nope":1},"b"]}';
  assert.deepEqual(parseModelIds(body), ['z', 'a', 'b']);
});

test('parseModelIds fails closed on anything it cannot read', () => {
  // An empty list must mean "no models", never "a shape I did not handle" — the caller
  // distinguishes those two and reports them differently.
  assert.deepEqual(parseModelIds('<html>not json</html>'), []);
  assert.deepEqual(parseModelIds('{"data":[]}'), []);
  assert.deepEqual(parseModelIds('{"data":"a string"}'), []);
  assert.deepEqual(parseModelIds('null'), []);
  assert.deepEqual(parseModelIds('{"data":[{"noId":true},42,null]}'), []);
});

/* -------------------------------------------------------------------------- */
/* Presets                                                                     */
/* -------------------------------------------------------------------------- */

test('every preset is complete and self-consistent', () => {
  const ids = new Set<string>();
  for (const p of ENDPOINT_PRESETS) {
    assert.ok(p.id.length > 0, 'a preset has no id');
    assert.ok(!ids.has(p.id), `duplicate preset id: ${p.id}`);
    ids.add(p.id);
    assert.ok(p.label.length > 0, `${p.id} has no label`);
    assert.ok(p.hint.length > 0, `${p.id} has no hint — the hint is the only thing that says what it needs`);
    // Every preset except the escape hatch must be reachable without the user inventing a URL.
    if (p.id !== 'custom') {
      assert.match(p.baseUrl, /^https?:\/\//, `${p.id} has no usable base URL`);
    }
    // Suggestions are starting points, never a promised catalogue. If a preset claimed more
    // than a handful it would be pretending to be the server's list, which is the confusion
    // 拉取模型 exists to end.
    assert.ok(p.modelHints.length <= 3, `${p.id} carries too many model hints to be honest about`);
  }
  assert.ok(presetById('custom'), 'there must be an escape hatch for endpoints not listed');
});

test('presetForBaseUrl maps a hand-typed address back to its service', () => {
  // Trailing slashes are the common hand-typing difference and must not defeat the lookup.
  assert.equal(presetForBaseUrl('https://api.deepseek.com/v1').id, 'deepseek');
  assert.equal(presetForBaseUrl('https://api.deepseek.com/v1/').id, 'deepseek');
  assert.equal(presetForBaseUrl('  https://api.deepseek.com/v1  ').id, 'deepseek');
  // Anything unrecognised — including an empty field — becomes 自定义 rather than silently
  // keeping a service name that is no longer true.
  assert.equal(presetForBaseUrl('https://my-gateway.internal/v1').id, 'custom');
  assert.equal(presetForBaseUrl('').id, 'custom');
});

/* -------------------------------------------------------------------------- */
/* Readiness                                                                   */
/* -------------------------------------------------------------------------- */

test('endpointReady names the missing field instead of just refusing', () => {
  const base = { choice: 'http' as const, baseUrl: '', model: '', apiKey: '', keyless: false };
  assert.deepEqual(endpointReady({ ...base, choice: 'stub' }), { ready: true });
  assert.deepEqual(endpointReady(base), { ready: false, missing: '服务地址' });
  assert.deepEqual(endpointReady({ ...base, baseUrl: 'https://x/v1' }), { ready: false, missing: '模型名' });
  assert.deepEqual(endpointReady({ ...base, baseUrl: 'https://x/v1', model: 'm' }), {
    ready: false,
    missing: 'API key',
  });
  assert.deepEqual(endpointReady({ ...base, baseUrl: 'https://x/v1', model: 'm', apiKey: 'k' }), {
    ready: true,
  });
});

test('a keyless local runtime is ready without a key', () => {
  // The old rule was a bare boolean — "key and model are non-empty" — which left 生成
  // permanently disabled for a local Ollama and gave no reason. A disabled control that does
  // not say what it wants is a puzzle, not a safeguard.
  assert.deepEqual(
    endpointReady({
      choice: 'http',
      baseUrl: 'http://127.0.0.1:11434/v1',
      model: 'qwen2.5:7b',
      apiKey: '',
      keyless: true,
    }),
    { ready: true },
  );
  // Keyless still needs an address and a model: "no key required" is not "anything goes".
  assert.equal(endpointReady({ choice: 'http', baseUrl: '', model: 'm', apiKey: '', keyless: true }).ready, false);
  assert.equal(
    endpointReady({ choice: 'http', baseUrl: 'http://x/v1', model: '', apiKey: '', keyless: true }).ready,
    false,
  );
});
