import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { checkVersion } from './check-version.mjs';

test('version check accepts aligned metadata and rejects version/log drift', () => {
  const root = mkdtempSync(path.join(tmpdir(), 'aimem-version-'));
  try {
    mkdirSync(path.join(root, 'docs'));
    const write = (name, value) => writeFileSync(path.join(root, name), value);
    write('package.json', JSON.stringify({ version: '0.2.0' }));
    const lock = { version: '0.2.0', packages: { '': { version: '0.2.0' } } };
    const log = '## 0.2.0 - 2026-11-09\n\nSession: `test-session`\nType: minor\n';
    write('package-lock.json', JSON.stringify(lock));
    write('docs/SESSION_LOG.md', log);
    assert.equal(checkVersion(root), '0.2.0');
    for (const broken of [
      { ...lock, version: '0.1.0' },
      { ...lock, packages: { '': { version: '0.1.0' } } }
    ]) {
      write('package-lock.json', JSON.stringify(broken));
      assert.throws(() => checkVersion(root), /must match/);
    }
    write('package-lock.json', JSON.stringify(lock));
    for (const broken of [
      log.replace('0.2.0', '0.1.0'), log.replace('Session: `test-session`', ''),
      log.replace('Type: minor', ''), ''
    ]) {
      write('docs/SESSION_LOG.md', broken);
      assert.throws(() => checkVersion(root));
    }
    write('docs/SESSION_LOG.md', log);
    write('package.json', JSON.stringify({ version: 'invalid' }));
    assert.throws(() => checkVersion(root), /MAJOR.MINOR.PATCH/);
  } finally {
    rmSync(root, { recursive: true });
  }
});
