# Application Versioning

Use `MAJOR.MINOR.PATCH`. `package.json` is the authoritative application
version; both root versions in `package-lock.json` must agree.

| Change | Increment | Example |
|---|---|---|
| Bug fix, maintenance, documentation-only checkpoint | Patch | 0.2.0 -> 0.2.1 |
| New backward-compatible capability | Minor; reset patch | 0.2.1 -> 0.3.0 |
| Breaking supported contract, or explicitly approved first stable release | Major; reset minor and patch | 0.3.0 -> 1.0.0 |

Use the highest applicable change in a checkpoint. Do not increment for every
chat message, memory refresh, or individual edit. Increment once per delivered
code/documentation checkpoint before commit/sync/push; changes in one session
can have multiple checkpoints. Memory-only capture commits need no version
bump. Do not bump again merely to push an already versioned checkpoint.
Major does not mean merely "a large feature." The prototype remains pre-1.0
until stability and supported contracts are explicitly agreed.

## Checkpoint procedure

1. Inspect all changes since the last versioned checkpoint; choose the highest
   applicable increment.
2. Run `npm version patch --no-git-tag-version`, substituting `minor` or `major`
   as appropriate. This updates both manifests without committing or tagging.
3. Prepend an entry to [SESSION_LOG.md](SESSION_LOG.md) using the exact new
   version, date, session identifier, increment type, changes, validation, and
   limitations. Link relevant prior commits and historical notes where useful.
   Record actual validation, not planned or assumed results.
4. Run `npm run version:check`, relevant tests, and `npm run build`.
5. Commit manifests, session log, and changes together. A recommended message
   is `feat(v0.3.0): ...` or `fix(v0.2.1): ...`. The log entry is in that
   commit; it does not need its own commit hash, avoiding a self-reference.
6. When commit/push is requested, push the checkpoint normally. For an
   explicitly designated release, create an annotated `vX.Y.Z` tag pointing
   at its verified commit and push that tag too. Do not fabricate historical
   tags or publish a GitHub Release without a release request.

`npm run build` automatically checks the manifest versions against the latest
session-log entry and requires its session/type metadata. This catches drift;
it cannot decide the semantic impact of a change, verify the truth of notes,
or force a human/agent to select the correct bump.

Application release versions are separate from extraction rulesets, lab report
schemas, model generations, and individual neural run IDs. Do not renumber
those just to match the application release.

## Historical reconciliation

The package remained at 0.1.0 while commit `4c2e089` mentioned v0.1.2 in
historical MemLab documentation. There were no Git release tags. Do not rewrite
that history or claim those versions were consistently released.

0.2.0 establishes the first synchronized application/session-log checkpoint,
including the new layer lab and neural features since the old package version.
Historical [MemLab session notes](../MemLab/session_notes.md) remain unchanged;
their old layer definitions are historical discussion, not current rules.
