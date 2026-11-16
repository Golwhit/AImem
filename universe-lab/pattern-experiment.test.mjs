import test from 'node:test';
import assert from 'node:assert/strict';
import { runPatternExperiment } from './pattern-experiment.js';

test('candidates freeze disjoint groups after one internal round', () => {
  for (const seed of [0, 1, 20261115, 0xffffffff]) {
    const result = runPatternExperiment(seed, 23);
    const members = result.groups.flatMap((group) => group.members);
    assert.deepEqual([...members].sort((a, b) => a - b), Array.from({ length: 23 }, (_, i) => i + 1));
    assert.equal(result.attempts[0].actor, 1);
    assert.equal(result.attempts[0].result, 'alone');
    for (const group of result.groups) {
      assert.ok(group.members.length >= 1 && group.members.length <= 3);
      const groupBonds = result.bonds.filter((bond) => group.members.includes(bond.quark));
      assert.ok(groupBonds.every((bond) => group.members.includes(bond.partner)));
      if (!group.complete) {
        assert.ok(group.members.length < 3);
        assert.equal(group.internal.length, 0);
        continue;
      }
      assert.equal(group.members.length, 3);
      assert.deepEqual(group.internal.map((attempt) => attempt.actor), group.members);
      assert.ok(group.internal.every((attempt) => group.members.includes(attempt.partner) && attempt.actor !== attempt.partner));
      assert.equal(groupBonds.length, 2 + group.internal.filter((attempt) => attempt.result === 'formed').length);
      const finishedAt = result.attempts.indexOf(group.internal.at(-1));
      assert.ok(result.attempts.slice(finishedAt + 1).every((attempt) =>
        !group.members.includes(attempt.actor) && !group.members.includes(attempt.partner)));
    }
    assert.ok(result.bonds.every((bond) => bond.quark !== 0 && bond.partner !== 0));
  }
});

test('initial run is reproducible and includes both chains and triangles', () => {
  const first = runPatternExperiment(20261115, 23);
  assert.deepEqual(first, runPatternExperiment(20261115, 23));
  const counts = first.groups.filter((group) => group.complete).map((group) =>
    first.bonds.filter((bond) => group.members.includes(bond.quark)).length);
  assert.ok(counts.includes(2));
  assert.ok(counts.includes(3));
  assert.notDeepEqual(first.bonds, runPatternExperiment(20261116, 23).bonds);
});

test('invalid experiment settings are rejected', () => {
  assert.throws(() => runPatternExperiment(-1, 23), RangeError);
  assert.throws(() => runPatternExperiment(1, 0), RangeError);
  assert.throws(() => runPatternExperiment(1, 101), RangeError);
});
