# Versioned Session Log

Newest application checkpoint first. Entries map delivered versions to the
session that produced them; historical conversation memory remains separate.
See [versioning policy](VERSIONING.md) for bump and release rules.

## 0.2.0 - 2026-11-09

Session: `28c01cf3-de47-499d-a9f1-4db6b3099c04`
Type: minor

### Changes

- Reconciled the stale 0.1.0 package version with the feature-bearing current
  application. Established patch/minor/major rules and build-time manifest/log
  checks; future checkpoint commits must update this log and the version.
- Added a muted bottom-right memory-graph version badge sourced from the
  application manifest when the graph is generated.
- Integrated Layer 0 possibilities and explicitly agreed Layer 1 exclusions,
  exact-text extraction with provenance, alongside existing approved training.
- Added two extraction panels and four chat generation comparisons, including
  captured-seed overlays and initial consonant repair. Available outputs remain
  visible when an empty mode prevents an equal-length comparison.
- Added isolated from-scratch neural character MLP training, new-text approval,
  persisted run checkpoints, and weight/loss/before-after progress reporting.
- Preserved promoted memory pairings and individual links; established-parent
  galaxies use solid shell connectors, while Under evaluation promotions do not.
- Recorded backlog topics including memory consolidation and thought progression.

### Earlier checkpoints in this session

- `706f71a`: seed layers, extraction/generation lab, galaxy relationships.
- `1577f05`: neural weight lab and zero-common-output display fix.
- These commits were pushed without synchronized application version bumps;
  this entry reconciles them rather than retroactively relabeling them.

### Validation and limitations

- Previous feature checkpoint: 61 targeted Python tests and production build
  passed; neural training/report persistence and saved chat-lab rendering were
  checked in the browser.
- Versioning checks/tests and production build are validated in this checkpoint.
- Graph-generation fixture includes its application manifest and verifies the
  rendered badge; all 61 targeted Python tests pass with that integration.
- Synthetic neural run: 653 trainable parameters, 300 steps, training loss
  1.5904 -> 0.0106, repeating-pattern held-out loss 1.5908 -> 0.0205.
  This verifies mechanics, not independent generalization.
- Neural training remains isolated from ordinary chat and historical memories.
  Layer-assisted neural training is not implemented. Local model checkpoints
  remain ignored and are not included in Git release history.
- The unrelated pre-existing full-suite test failure documented during this
  session was not fixed; targeted results do not claim a clean full suite.
