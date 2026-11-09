import { readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

export function checkVersion(root) {
  const read = name => readFileSync(path.join(root, name), 'utf8');
  const manifest = JSON.parse(read('package.json'));
  const lock = JSON.parse(read('package-lock.json'));
  const log = read('docs/SESSION_LOG.md');
  const entry = log.match(/^## (\d+\.\d+\.\d+) - (\d{4}-\d{2}-\d{2})$/m);
  if (!/^\d+\.\d+\.\d+$/.test(manifest.version)) {
    throw new Error('Application version must use MAJOR.MINOR.PATCH');
  }
  if (lock.version !== manifest.version || lock.packages?.['']?.version !== manifest.version) {
    throw new Error('package.json and both root package-lock.json versions must match');
  }
  if (!entry || entry[1] !== manifest.version) {
    throw new Error('Newest docs/SESSION_LOG.md entry must match the application version');
  }
  const body = log.slice(entry.index + entry[0].length).split(/^## /m)[0];
  if (!/^Session: `[^`\n]+`$/m.test(body) || !/^Type: (patch|minor|major)$/m.test(body)) {
    throw new Error('Newest session entry requires Session: `id` and Type: patch|minor|major');
  }
  return manifest.version;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    console.log(`Version ${checkVersion(path.resolve(fileURLToPath(new URL('..', import.meta.url))))}: manifests and session log agree`);
  } catch (error) {
    console.error(error.message);
    process.exitCode = 1;
  }
}
