export const bondColors = { '01': '#67e8d5', '10': '#c5a5ff', '00': '#84aaff', '11': '#ffb46f', both: '#ffdc73' };

function randomSource(seed) {
  let state = seed >>> 0;
  return (limit) => {
    const bucket = Math.floor(0x100000000 / limit);
    const ceiling = bucket * limit;
    let value;
    do {
      state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
      value = state;
    } while (value >= ceiling);
    return Math.floor(value / bucket);
  };
}

export function runPatternExperiment(seed, popouts) {
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff ||
      !Number.isInteger(popouts) || popouts < 1 || popouts > 100) {
    throw new RangeError('Choose a 32-bit seed and 1 to 100 pop-outs.');
  }
  const random = randomSource(seed);
  const quarks = [{ id: 0, bit: 1, seeder: true }];
  const bonds = [];
  const candidates = [];
  const attempts = [];
  const frozen = new Set();
  const outcomes = ['outgoing', 'incoming', 'both', 'free'];

  const component = (id) => {
    const members = new Set([id]);
    let changed = true;
    while (changed) {
      changed = false;
      for (const bond of bonds) {
        if (members.has(bond.quark) && !members.has(bond.partner)) {
          members.add(bond.partner);
          changed = true;
        } else if (members.has(bond.partner) && !members.has(bond.quark)) {
          members.add(bond.quark);
          changed = true;
        }
      }
    }
    return [...members].sort((a, b) => a - b);
  };

  const choose = (actor, partners, phase) => {
    if (!partners.length) {
      const attempt = { actor, partner: null, outcome: 'alone', result: 'alone', phase };
      attempts.push(attempt);
      return attempt;
    }
    const partner = partners[random(partners.length)];
    const outcome = outcomes[random(outcomes.length)];
    const exists = bonds.some((bond) => (bond.quark === actor && bond.partner === partner) ||
      (bond.quark === partner && bond.partner === actor));
    const result = exists ? 'existing' : outcome === 'free' ? 'free' : 'formed';
    const attempt = { actor, partner, outcome, result, phase };
    attempts.push(attempt);
    if (result === 'formed') {
      bonds.push({ quark: actor, partner, outcome, outgoing: outcome === 'both' ? 0.25 : outcome === 'outgoing' ? 0.5 : 0,
        incoming: outcome === 'both' ? 0.25 : outcome === 'incoming' ? 0.5 : 0 });
    }
    return attempt;
  };

  const freeze = (members, phase) => {
    members.forEach((id) => frozen.add(id));
    const internal = members.map((actor) => choose(actor, members.filter((id) => id !== actor), 'internal'));
    candidates.push({ members, phase, internal });
  };

  const tryExternal = (actor, phase) => {
    if (frozen.has(actor)) return;
    const group = component(actor);
    const partners = quarks.filter((quark) => !quark.seeder && quark.id !== actor && !frozen.has(quark.id))
      .filter((quark) => group.includes(quark.id) || group.length + component(quark.id).length <= 3)
      .map((quark) => quark.id);
    const attempt = choose(actor, partners, phase);
    if (attempt.result === 'formed') {
      const merged = component(actor);
      if (merged.length === 3) freeze(merged, phase);
    }
  };

  for (let index = 0; index < popouts; index += 1) {
    for (const phase of ['predict', 'reveal', 'measure', 'reweight']) {
      let newest = null;
      if (phase === 'reveal') {
        newest = quarks.length;
        quarks.push({ id: newest, bit: index % 2, seeder: false });
        tryExternal(newest, phase);
      }
      for (const quark of quarks) {
        if (!quark.seeder && quark.id !== newest) tryExternal(quark.id, phase);
      }
    }
  }
  const groups = candidates.map((candidate) => ({ members: candidate.members, complete: true, internal: candidate.internal }));
  const grouped = new Set(groups.flatMap((group) => group.members));
  for (const quark of quarks) {
    if (quark.seeder || grouped.has(quark.id)) continue;
    const members = component(quark.id);
    members.forEach((id) => grouped.add(id));
    groups.push({ members, complete: false, internal: [] });
  }
  return { seed, quarks, bonds, groups, attempts };
}
