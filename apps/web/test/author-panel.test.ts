/**
 * What the authoring panel remembers across a reload
 * =================================================
 *
 * The panel keeps its endpoint configuration in `localStorage` so a refresh does not mean
 * retyping it. That is the kind of behaviour which is invisible when it works and infuriating
 * when it does not, and it cannot be checked from `boot.test.ts`: that suite imports the app
 * once, so there is no second load to compare against.
 *
 * So this builds the panel twice against the same storage and asserts the round trip. The
 * second `AuthorPanel` is what a reload *is* — the constructor reads storage and nothing else
 * — which makes it a faithful stand-in for one rather than a simulation of one.
 *
 * The key gets its own tests because it is the entry with a real trade-off attached: storing
 * it is convenient and it is also readable by any script on this origin. The behaviour that
 * has to hold is that the user's answer is obeyed in both directions — stored when asked,
 * **deleted** when the answer changes — because a privacy switch that leaves the bytes behind
 * is worse than not offering one.
 */

import assert from 'node:assert/strict';
import { after, before, test } from 'node:test';

import { installDom, type FakeElement, type Shim } from './dom-shim.ts';
// Types from the authored source; the *values* come from the projection, because that is what
// the browser loads and what these tests are about. The same split `src/*.ts` uses.
import type { AuthorPanel as AuthorPanelClass, AuthorPanelHost } from '../src/panel/author.ts';

type Panel = InstanceType<typeof AuthorPanelClass>;

let shim: Shim;
let AuthorPanel: typeof AuthorPanelClass;

before(async () => {
  shim = installDom();
  const mod = (await import('../dist/panel/author.js')) as unknown as {
    AuthorPanel: typeof AuthorPanelClass;
  };
  AuthorPanel = mod.AuthorPanel;
});

after(() => {
  for (const f of shim.failures) console.error(`[author-panel.test] leaked ${f}`);
});

/** The host the panel talks back to; none of these tests exercise it. */
function host(): AuthorPanelHost {
  return {
    applyDoc: async () => {},
    setStatus: () => {},
    setBusy: () => {},
    currentDoc: () => ({}) as never,
  };
}

/** Build a panel, as a page load would. */
function open(): Panel {
  return new AuthorPanel(host());
}

function field(panel: Panel, selector: string): FakeElement {
  const el = panel.el.querySelector<FakeElement>(selector);
  assert.ok(el, `the panel has no ${selector}`);
  return el;
}

/** Type into a field the way a user does: set the value, then fire `input`. */
function type(el: FakeElement, value: string): void {
  el.value = value;
  el.dispatchEvent({ type: 'input' });
}

function click(el: FakeElement): void {
  el.dispatchEvent({ type: 'click' });
}

/** Everything the panel stores, keyed as it writes it. */
function stored(): Record<string, unknown> {
  const raw = shim.storage.getItem('ai4shader.author.endpoint');
  return raw ? (JSON.parse(raw) as Record<string, unknown>) : {};
}

const KEY = 'sk-a-very-secret-value-0123456789';

test('endpoint configuration survives a reload', () => {
  shim.storage.clear();

  const first = open();
  const service = field(first, 'select.author-service');
  service.value = 'deepseek';
  service.dispatchEvent({ type: 'change' });
  type(field(first, 'input.author-model'), 'deepseek-reasoner');
  click(field(first, '.author-seg button')); // the 在线 button
  const offline = first.el.querySelectorAll<FakeElement>('.author-seg button')[1];
  assert.ok(offline);
  click(offline);

  const second = open();
  assert.equal(field(second, 'select.author-service').value, 'deepseek');
  assert.equal(field(second, 'input.author-url').value, 'https://api.deepseek.com/v1');
  assert.equal(field(second, 'input.author-model').value, 'deepseek-reasoner');
  // The provider choice is configuration too: coming back on 离线 when 在线 was selected would
  // make the remembered key look like it had not been remembered.
  const choice = second.el.querySelectorAll<FakeElement>('.author-seg button');
  assert.equal(choice[1]?.classList.contains('on'), true, 'the online/offline choice was not restored');
});

test('the key is remembered when the box is ticked — the default', () => {
  shim.storage.clear();

  const first = open();
  const remember = field(first, 'input#author-remember-key');
  assert.equal(remember.checked, true, 'remembering should be the default: the user asked for it');
  type(field(first, 'input.author-key'), KEY);

  assert.equal(stored()['apiKey'], KEY, 'the key was not written to storage');

  const second = open();
  assert.equal(field(second, 'input.author-key').value, KEY, 'the key did not come back after a reload');
});

test('unticking the box deletes the stored key immediately', () => {
  shim.storage.clear();

  const first = open();
  const remember = field(first, 'input#author-remember-key');
  type(field(first, 'input.author-key'), KEY);
  assert.equal(stored()['apiKey'], KEY);

  remember.checked = false;
  remember.dispatchEvent({ type: 'change' });

  // Not "will not be read next time" — gone from storage now. A switch that leaves the secret
  // on disk is a promise the implementation does not keep.
  assert.equal('apiKey' in stored(), false, 'unticking left the key in storage');
  assert.equal(stored()['rememberKey'], false);

  // And a fresh load must not resurrect it from anywhere.
  const second = open();
  assert.equal(field(second, 'input.author-key').value, '', 'an unticked box still restored the key');
});

test('a stored key is ignored when the box says not to remember it', () => {
  // The stale-state case: a value left by a previous choice must not silently override the
  // current one. Storage is hand-written here because that is exactly the state a downgrade or
  // a partially-applied change would leave behind.
  shim.storage.clear();
  shim.storage.setItem(
    'ai4shader.author.endpoint',
    JSON.stringify({ service: 'deepseek', rememberKey: false, apiKey: KEY }),
  );

  const panel = open();
  assert.equal(field(panel, 'input#author-remember-key').checked, false);
  assert.equal(field(panel, 'input.author-key').value, '', 'a stale key was restored anyway');
});

test('the note next to the field states which of the two is in force', () => {
  shim.storage.clear();
  const panel = open();
  const note = field(panel, '.author-keyrow + .author-hint, .author-hint');
  const remember = field(panel, 'input#author-remember-key');

  // The trade-off belongs where the choice is made, not in a README. A checkbox whose
  // consequence is documented somewhere else is a checkbox nobody can answer.
  assert.match(note.textContent, /localStorage/, `the remembered case must say where it goes: "${note.textContent}"`);
  assert.match(note.textContent, /任何脚本/, `it must also say who can read it: "${note.textContent}"`);

  remember.checked = false;
  remember.dispatchEvent({ type: 'change' });
  assert.match(note.textContent, /内存/, `the not-remembered case must say so: "${note.textContent}"`);
});

/* -------------------------------------------------------------------------- */
/* The model picker                                                            */
/* -------------------------------------------------------------------------- */

function selectService(panel: Panel, id: string): void {
  const service = field(panel, 'select.author-service');
  service.value = id;
  service.dispatchEvent({ type: 'change' });
}

test('the picker lists every known model, not only the one already selected', () => {
  /*
   * The reported bug, as an assertion.
   *
   * This control used to be an `<input list=…>` with a `<datalist>`. Browsers **filter a
   * datalist by the field's current contents**, so a panel that pre-fills a model name offers
   * a dropdown containing exactly that one name. The user's report was "why can DeepSeek only
   * choose deepseek-chat" — the list was longer, and the widget was hiding all of it.
   */
  shim.storage.clear();
  const panel = open();
  selectService(panel, 'deepseek');

  const picker = field(panel, 'select.author-model-select');
  const options = picker.children.map((o) => o.value).filter((v) => !v.startsWith('\u0000'));
  assert.ok(
    options.length >= 2,
    `the picker must offer the whole list, found ${JSON.stringify(options)}`,
  );
  assert.ok(options.includes('deepseek-flash'), `expected the current DeepSeek id, got ${JSON.stringify(options)}`);
  assert.ok(options.includes('deepseek-v4-pro'), `expected the current DeepSeek id, got ${JSON.stringify(options)}`);
  // The names this table used to ship are gone from the API. A stale *suggestion* is worse
  // than none: it is a name the user will try, and "model not found" reads like their mistake.
  assert.ok(!options.includes('deepseek-chat'), 'a retired model id is still being suggested');
  assert.equal(picker.value, 'deepseek-flash', 'the first hint should be selected');
});

test('拉取模型 replaces the hints with whatever the endpoint returns', async () => {
  shim.storage.clear();
  const panel = open();
  selectService(panel, 'deepseek');

  const realFetch = globalThis.fetch;
  globalThis.fetch = (async () => ({
    ok: true,
    status: 200,
    text: async () =>
      JSON.stringify({ object: 'list', data: [{ id: 'zzz-new' }, { id: 'aaa-old' }, { id: 'zzz-new' }] }),
  })) as unknown as typeof globalThis.fetch;

  try {
    const buttons = panel.el.querySelectorAll<FakeElement>('.author-probe-row button');
    const fetchBtn = buttons[1];
    assert.ok(fetchBtn, 'there is no 拉取模型 button');
    click(fetchBtn);
    const status = field(panel, '.author-probe');
    const deadline = Date.now() + 5000;
    while (Date.now() < deadline && !status.textContent.startsWith('✓') && !status.textContent.startsWith('✗')) {
      await new Promise((r) => setTimeout(r, 10));
    }
    assert.match(status.textContent, /^✓/, `the fetch should have succeeded: ${status.textContent}`);

    const picker = field(panel, 'select.author-model-select');
    const options = picker.children.map((o) => o.value).filter((v) => !v.startsWith('\u0000'));
    // Server order preserved, de-duplicated — a provider that lists its flagship first is
    // saying something, and alphabetising would throw that away.
    assert.deepEqual(options, ['zzz-new', 'aaa-old'], `the server list did not replace the hints: ${options}`);
    // The field had a name the server does not serve, so it must have moved to the server's
    // first entry rather than leaving a name that cannot work.
    assert.equal(field(panel, 'input.author-model').value, 'zzz-new');
  } finally {
    globalThis.fetch = realFetch;
  }
});

test('there is always a way to type a model name by hand', () => {
  // The picker must never become a cage: `/models` is not implemented by every gateway, and
  // the 自定义 preset ships no hints at all.
  shim.storage.clear();
  const panel = open();
  selectService(panel, 'deepseek');

  const picker = field(panel, 'select.author-model-select');
  const manual = picker.children.find((o) => o.value.startsWith('\u0000'));
  assert.ok(manual, 'the picker has no 手动输入 entry');
  assert.match(manual.textContent, /手动输入/);

  picker.value = manual.value;
  picker.dispatchEvent({ type: 'change' });
  const text = field(panel, 'input.author-model');
  assert.equal(text.hidden, false, 'choosing 手动输入 must reveal the text field');

  type(text, 'my-own-model-name');
  assert.equal(text.value, 'my-own-model-name');
  assert.equal(picker.value, manual.value, 'a typed name must leave the picker on 手动输入');

  // And a service with no hints at all shows only the text field, never an empty dropdown.
  selectService(panel, 'custom');
  assert.equal(field(panel, 'select.author-model-select').hidden, true, 'an empty picker must be hidden');
  assert.equal(field(panel, 'input.author-model').hidden, false, 'the text field must still be there');
});
