import { readFileSync, writeFileSync } from 'node:fs';

const files = process.argv.slice(2);
if (!files.length) {
  console.error('usage: node scripts/strip-emoji.mjs <file...>');
  process.exit(2);
}

// Status glyphs in docs read as noise in plain-text tools and terminals that do not
// render them, so the docs use words. Anything else non-ASCII is left alone (the docs
// are deliberately bilingual and full of CJK).
const REPLACEMENTS = [
  [/\u2705/g, 'YES'],
  [/\u26A0\uFE0F?/g, 'PARTIAL'],
  [/\u274C/g, 'NO'],
  [/\u2717/g, 'FAIL'],
  [/\u2713/g, 'ok'],
  [/\u2B50/g, '*'],
  [/\uD83D\uDD25/g, ''],
];

for (const f of files) {
  const before = readFileSync(f, 'utf8');
  let after = before;
  for (const [re, to] of REPLACEMENTS) after = after.replace(re, to);
  after = after.replace(/[ \t]+$/gm, '');
  if (after === before) {
    console.log(`· ${f}: unchanged`);
    continue;
  }
  writeFileSync(f, after, 'utf8');
  console.log(`ok ${f}: normalised`);
}
