import assert from 'node:assert/strict';
import { test } from 'node:test';
import { optimize, type OptimizeMethod, type OptimizeOptions } from '../src/optimize/search.ts';

function options(overrides: Partial<OptimizeOptions> = {}): OptimizeOptions {
  return {
    params: [{ name: 'x', min: 0, max: 1, value: 0.5 }],
    evaluate: async ({ x }) => (x - 0.3) ** 2,
    maxEvaluations: 80,
    onProgress: () => {},
    ...overrides,
  };
}

for (const method of ['coordinate', 'cmaes', 'hybrid'] as OptimizeMethod[]) {
  test(`${method}: fixed and sub-micro ranges stay inside their bounds`, async () => {
    const params = [
      { name: 'fixed', min: 2, max: 2, value: 2 },
      { name: 'tiny', min: 0.00000011, max: 0.00000019, value: 0.00000015 },
    ];
    await optimize({ ...options({ params, evaluate: async (values) => {
      for (const p of params) assert.ok(values[p.name] >= p.min && values[p.name] <= p.max,
        `${p.name} escaped its range: ${values[p.name]}`);
      return values.tiny;
    } }), method });
  });

  test(`${method}: history records the running best, including across hybrid stages`, async () => {
    let calls = 0;
    const result = await optimize({ ...options({
      params: ['x', 'y', 'z'].map((name) => ({ name, min: 0, max: 1, value: 0.5 })),
      evaluate: async () => ++calls,
    }), method });
    assert.ok(calls > 32);
    assert.deepEqual(result.history, Array(calls).fill(1));
    assert.equal(result.best, 1);
  });

  test(`${method}: a fractional evaluation budget cannot be exceeded`, async () => {
    const result = await optimize({ ...options({ maxEvaluations: 2.5 }), method });
    assert.equal(result.evals, 2);
  });

  test(`${method}: a progress callback cannot corrupt the best point`, async () => {
    const result = await optimize({ ...options({
      maxEvaluations: 1,
      onProgress: (p) => { p.values.x = 100; },
    }), method });
    assert.equal(result.values.x, 0.5);
  });

  test(`${method}: cancellation stops evaluations and preserves the best point`, async () => {
    const controller = new AbortController();
    const result = await optimize({ ...options({
      signal: controller.signal,
      onProgress: () => controller.abort(),
    }), method });
    assert.equal(result.evals, 1);
    assert.equal(result.values.x, 0.5);
  });

  test(`${method}: zero budget and prior cancellation do not evaluate`, async () => {
    for (const overrides of [{ maxEvaluations: 0 }, { signal: AbortSignal.abort() }]) {
      const result = await optimize({ ...options({
        ...overrides,
        evaluate: async () => { assert.fail('evaluation must not run'); },
      }), method });
      assert.equal(result.evals, 0);
      assert.deepEqual(result.history, []);
    }
  });

  test(`${method}: non-finite budgets fail before evaluation`, async () => {
    for (const maxEvaluations of [NaN, Infinity, -Infinity]) {
      await assert.rejects(optimize({ ...options({
        maxEvaluations,
        evaluate: async () => { assert.fail('evaluation must not run'); },
      }), method }), /maxEvaluations must be finite/);
    }
  });

  test(`${method}: search improves a quadratic objective reproducibly`, async () => {
    const first = await optimize({ ...options(), method });
    const second = await optimize({ ...options(), method });
    assert.ok(first.best < 0.001);
    assert.ok(first.evals <= 80);
    assert.deepEqual(first, second);
  });
}

test('hybrid: second-stage progress cannot mutate the coarse result', async () => {
  let calls = 0;
  const result = await optimize({ ...options({
    params: ['x', 'y', 'z'].map((name) => ({ name, min: 0, max: 1, value: 0.5 })),
    evaluate: async () => ++calls,
    onProgress: (p) => { p.values.x = 100; },
  }) });
  assert.ok(calls > 32);
  assert.equal(result.values.x, 0.5);
});
