import { useEffect, useRef, useState } from 'react';
import PatternLab from './PatternLab';
import { bondColors } from './pattern-experiment';
import './universe-lab.css';

const number = (value) => Number(value).toFixed(4);
const names = { predict: 'Predict', reveal: 'Reveal', measure: 'Measure error', reweight: 'Reweight' };
const runBatchSteps = 50;
const runBatchIntervalMs = 1000;

function describeBond(bond) {
  if (!bond) return 'No initial bond choice was recorded for this older quark.';
  if (bond.outcome === 'first') return 'The first pop-out initially floats freely.';
  const partner = `#${bond.partner + 1}`;
  if (bond.outcome === 'free') return `Chose ${partner} and initially made no bond.`;
  if (bond.outcome === 'both') return `Connected both ways with ${partner}, strength 0.25 each.`;
  if (bond.outcome === 'outgoing') return `New quark → ${partner}, strength 0.5.`;
  return `${partner} → new quark, strength 0.5.`;
}

function describeAttempt(attempt) {
  const actor = `#${attempt.quark + 1}`;
  if (attempt.result === 'alone') return `${actor}: no eligible partner to choose.`;
  const partner = `#${attempt.partner + 1}`;
  const outcomes = { outgoing: `${actor} → ${partner} (0.5)`, incoming: `${partner} → ${actor} (0.5)`,
    both: `${actor} ↔ ${partner} (0.25 each)`, free: 'no bond' };
  const result = { existing: 'existing bond kept', formed: 'bond formed', free: 'no bond formed' };
  return `${attempt.scope === 'internal' ? 'Internal: ' : ''}${actor} chose ${partner}: ${outcomes[attempt.outcome]}; ${result[attempt.result]}.`;
}

function attemptSummary(event) {
  const attempts = event?.pattern_attempts || event?.attempts;
  if (!attempts) return '';
  const formed = attempts.filter((attempt) => attempt.result === 'formed').length;
  return ` Recorded attempts: ${attempts.length}; new bonds from those attempts: ${formed}.`;
}

function AttemptDetails({ attempts }) {
  const [open, setOpen] = useState(false);
  return <details onToggle={(event) => setOpen(event.currentTarget.open)}>
    <summary>Connection attempts ({attempts.length})</summary>
    {open && <ul>{attempts.map((attempt, index) => <li key={`${attempt.quark}-${index}`}>{describeAttempt(attempt)}</li>)}</ul>}
  </details>;
}

function describe(event) {
  switch (event.stage) {
    case 'spark': return 'Spark: the first 1 appears.';
    case 'fuel': return `The experimenter added ${event.added} energy.`;
    case 'predict': return `Saw ${event.previous}; guessed ${event.guess} (${number(event.probability)} chance of 1).`;
    case 'reveal': return `Batch ${event.batch}, pair ${event.pair}: the seeder revealed ${event.bit}.${event.bond ? ` ${describeBond(event.bond)}` : ''}`;
    case 'measure': return `Error: ${number(event.error)}; the guess was ${event.correct ? 'correct' : 'incorrect'}.`;
    case 'reweight': return `Bias ${number(event.before.bias)} → ${number(event.after.bias)}; link ${number(event.before.link)} → ${number(event.after.link)}.`;
    default: return `Unknown event: ${event.stage}`;
  }
}

function StepRunControls({ canStep, busy, running, onStep, onRun, onPause }) {
  return <div className="universe-controls">
    <button type="button" disabled={!canStep || busy || running} onClick={onStep}>Step once</button>
    {running
      ? <button type="button" onClick={onPause}>Pause after this batch</button>
      : <button type="button" disabled={!canStep || busy} onClick={onRun}>Run until energy ends</button>}
  </div>;
}

function UniverseGraph({ events, recordedBonds }) {
  const mapRef = useRef(null);
  const [follow, setFollow] = useState(true);
  const bits = events.filter((event) => event.stage === 'spark' || event.stage === 'reveal');
  const positions = bits.map((_, index) => {
    if (index === 0) return { x: 0, y: 0 };
    const noise = Math.sin(index * 127.1) * 43758.5453;
    const variation = noise - Math.floor(noise) - 0.5;
    const angle = index * Math.PI * (3 - Math.sqrt(5)) + variation * 0.24 / Math.sqrt(index);
    const distance = 26 * Math.sqrt(index) + variation * 5;
    return { x: distance * Math.cos(angle), y: distance * Math.sin(angle) };
  });
  const extent = Math.max(325, Math.ceil(26 * Math.sqrt(bits.length) + 32));
  const width = Math.max(775, extent * 2);
  const height = extent * 2;
  useEffect(() => {
    if (follow && mapRef.current && positions.length) {
      const latest = positions[positions.length - 1];
      mapRef.current.scrollLeft = latest.x + width / 2 - mapRef.current.clientWidth / 2;
      mapRef.current.scrollTop = latest.y + height / 2 - mapRef.current.clientHeight / 2;
    }
  }, [bits.length, follow, width, height]);
  if (!bits.length) return <div className="universe-empty">Nothing has appeared yet. Spark creates the first 1.</div>;
  const bonds = recordedBonds.map((bond) => {
    const from = bond.outcome === 'incoming' ? bond.partner : bond.quark;
    const to = bond.outcome === 'incoming' ? bond.quark : bond.partner;
    const color = bond.outcome === 'both' ? 'both' : `${bits[from].bit}${bits[to].bit}`;
    return { from, to, color, bond };
  });
  const patterns = {};
  bonds.forEach(({ from, to, bond }) => {
    const pair = bond.outcome === 'both'
      ? `${Math.min(bits[from].bit, bits[to].bit)} ↔ ${Math.max(bits[from].bit, bits[to].bit)}`
      : `${bits[from].bit} → ${bits[to].bit}`;
    patterns[pair] = (patterns[pair] || 0) + 1;
  });
  return <figure className="universe-figure">
    <label className="universe-follow"><input type="checkbox" checked={follow}
      onChange={(event) => setFollow(event.target.checked)} /> Follow the newest quark</label>
    <div className="universe-map" ref={mapRef}>
    <svg viewBox={`${-width / 2} ${-height / 2} ${width} ${height}`} width={width} height={height}
      role="img" aria-label={`${bits.length} individual quarks and ${bonds.length} recorded bonds`}>
      {bonds.map(({ from, to, color, bond }) => {
        const start = positions[from];
        const end = positions[to];
        const distance = Math.hypot(end.x - start.x, end.y - start.y);
        const dx = (end.x - start.x) / distance;
        const dy = (end.y - start.y) / distance;
        const strength = bond.outcome === 'both' ? bond.outgoing : Math.max(bond.outgoing, bond.incoming);
        return <line key={`bond-${Math.min(from, to)}-${Math.max(from, to)}`}
          x1={start.x + dx * 10} y1={start.y + dy * 10} x2={end.x - dx * 10} y2={end.y - dy * 10}
          stroke={bondColors[color]} strokeWidth={2 * strength}>
          <title>{bond.outcome === 'both'
            ? `#${from + 1} ↔ #${to + 1}: strength 0.25 each`
            : `#${from + 1} → #${to + 1}: strength 0.5`}</title>
        </line>;
      })}
      {bits.map((event, index) => <g key={index}>
        <title>{event.stage === 'spark' ? '#1: the original seeder' : `#${index + 1}: ${event.bit}. ${describeBond(event.bond)}`}</title>
        <circle cx={positions[index].x} cy={positions[index].y} r="9"
          fill={event.stage === 'spark' ? '#b8622c' : event.bit ? '#15727b' : '#354a79'}
          stroke={event.stage === 'spark' ? '#ffd28a' : event.bit ? '#78e5dc' : '#9daeff'} strokeWidth="1.5" />
        <text x={positions[index].x} y={positions[index].y + 3.5} textAnchor="middle" fill="#fff" fontSize="10">{event.bit}</text>
        <text x={positions[index].x} y={positions[index].y + 18} textAnchor="middle" fill="#bdcbe0" fontSize="8">#{index + 1}</text>
      </g>)}
    </svg>
    </div>
    <figcaption>Each circle is one quark. Hover over a quark or line for its identity and recorded choice.
      Positions are decorative, not physical movement; crossing lines are not extra bonds.
      Line width reflects strength in one direction: 0.5 is thicker than 0.25.
      Earlier timeline lines were not bonds.</figcaption>
    <p>Current two-quark bond patterns (not letters): {Object.entries(patterns).length
      ? Object.entries(patterns).map(([pattern, count]) => `${pattern}: ${count}`).join(' · ')
      : 'none yet'}.</p>
  </figure>;
}

export default function UniverseLab() {
  const [data, setData] = useState(null);
  const [fuel, setFuel] = useState('24');
  const [busy, setBusy] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [experimentName, setExperimentName] = useState('Quark experiment');
  const [selectedId, setSelectedId] = useState('');
  const busyRef = useRef(false);
  const runningRef = useRef(false);
  const mountedRef = useRef(true);
  const dataRef = useRef(null);

  const request = async (action, payload = {}) => {
    const response = await fetch('/api/universe-lab', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action, ...payload, ...(action !== 'read' && dataRef.current?.revision
        ? { expected_revision: dataRef.current.revision } : {}) }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
    if (mountedRef.current) {
      dataRef.current = result;
      setData(result);
    }
    return result;
  };

  useEffect(() => {
    mountedRef.current = true;
    request('read').catch((failure) => { if (mountedRef.current) setError(failure.message); });
    return () => { mountedRef.current = false; runningRef.current = false; };
  }, []);

  const perform = async (action, payload = {}) => {
    if (busyRef.current || runningRef.current) return;
    busyRef.current = true;
    setBusy(true);
    setError('');
    setNotice('');
    try {
      await request(action, payload);
      const messages = { save: 'Experiment saved. You can continue this run.',
        reset: 'The universe is empty and ready for a new spark.',
        reload: 'Saved experiment loaded with its weights, energy, phase, and history.',
        delete: 'Saved experiment deleted. The current run was not changed.',
        read: 'Current universe refreshed.' };
      if (mountedRef.current && messages[action]) setNotice(messages[action]);
    }
    catch (failure) { if (mountedRef.current) setError(failure.message); }
    finally { busyRef.current = false; if (mountedRef.current) setBusy(false); }
  };

  const run = async () => {
    if (busyRef.current || runningRef.current || !data?.state.universe) return;
    runningRef.current = true;
    setRunning(true);
    setError('');
    setNotice('');
    let world = data.state.universe;
    try {
      while (runningRef.current && (world.energy > 0 || world.phase !== 'predict')) {
        const started = performance.now();
        world = (await request('step', { steps: runBatchSteps })).state.universe;
        const remaining = runBatchIntervalMs - (performance.now() - started);
        if (runningRef.current && (world.energy > 0 || world.phase !== 'predict') && remaining > 0) {
          await new Promise((resolve) => setTimeout(resolve, remaining));
        }
      }
    } catch (failure) {
      if (mountedRef.current) setError(failure.message);
    } finally {
      runningRef.current = false;
      if (mountedRef.current) setRunning(false);
    }
  };

  const world = data?.state.universe;
  const events = world?.events || [];
  const canStep = world && (world.energy > 0 || world.phase !== 'predict');
  const fuelValue = Number(fuel);
  const validFuel = Number.isInteger(fuelValue) && fuelValue >= 1 && fuelValue <= (data?.max_fuel || 1000);
  const ready = data?.api_version === 2 && Boolean(data.revision) && !busy && !running;
  const validName = experimentName.trim().length > 0 && experimentName.trim().length <= 80;
  const selected = data?.experiments?.find((experiment) => experiment.id === selectedId);
  const controls = { canStep, busy, running, onStep: () => void perform('step'),
    onRun: () => void run(), onPause: () => { runningRef.current = false; } };
  const replace = (action, disposition) => {
    if (disposition === 'discard' && !window.confirm('Discard the current run? Its unsaved progress will be lost.')) return;
    void perform(action, { id: selectedId, disposition, confirm: true, name: experimentName });
  };
  return <section className="universe-lab hero-card">
    <p className="universe-eyebrow">A fresh experiment · quark stage</p>
    <h2>Universe Lab</h2>
    <p>Before the spark there is no bit in this experiment. The spark creates the first <strong>1</strong>.
      Then our seeder makes <code>01 | 0101 | 010101 | …</code>, one pair more in each batch.</p>
    <p className="universe-note">This is a small rule we designed, not a claim that a real universe came from nothing.
      The seeder supplies the answers; the learner predicts them. This lab cannot change chat weights or memories.</p>
    <form className="universe-controls" onSubmit={(event) => {
      event.preventDefault();
      void perform(world ? 'fuel' : 'spark', { fuel: fuelValue, ...(!world ? { bond_mode: 'three-quark' } : {}) });
    }}>
      <label>{world ? 'Energy to add' : 'Starting energy'}
        <input aria-label={world ? 'Energy to add' : 'Starting energy'} type="number" min="1"
          max={data?.max_fuel || 1000} step="1" value={fuel} onChange={(event) => setFuel(event.target.value)} />
      </label>
      <button type="submit" disabled={!data || !validFuel || busy || running}>
        {world ? 'Add fuel' : 'Spark the universe'}
      </button>
      <StepRunControls {...controls} />
    </form>
    <p>Fuel can be changed for each addition, from 1 to {data?.max_fuel || 1000}.
      The spark costs one; each prediction and its reveal, measurement, and update cost one more together.</p>
    {error && <p className="universe-error" role="alert">{error}</p>}
    {notice && <p className="universe-current" role="status">{notice}</p>}
    {!data && !error && <p role="status">Reading the isolated universe…</p>}
    {data && <>
      <div className="universe-status" aria-live="polite">
        <span><strong>Energy:</strong> {world ? world.energy : 'not created'}</span>
        <span><strong>Next step:</strong> {world ? names[world.phase] : 'Spark'}</span>
        <span><strong>Seeder batch:</strong> {world ? world.batch : 'not started'}</span>
        <span><strong>Previous bit:</strong> {world ? world.previous : 'none'}</span>
        <span><strong>Weights:</strong> {world ? `bias ${number(world.weights.bias)}, link ${number(world.weights.link)}` : 'not created'}</span>
      </div>
      <p className="universe-current" role="status">{events.length ? describe(events[events.length - 1]) : 'Nothing exists in the experiment yet.'}</p>
      <section className="universe-panel">
        <h3>Keep or start over</h3>
        <p>Adding fuel continues this run with its current weights. Saving keeps a separate local copy.
          Reset starts with nothing; loading restores the saved weights, energy, current step, and history.</p>
        <div className="universe-controls universe-archives">
          <label>Experiment name
            <input type="text" maxLength="80" value={experimentName}
              onChange={(event) => setExperimentName(event.target.value)} />
          </label>
          <button type="button" disabled={!ready || !world || !validName}
            onClick={() => void perform('save', { name: experimentName })}>Save current run</button>
          <button type="button" disabled={!ready || !world || !validName}
            onClick={() => replace('reset', 'save')}>Save and reset</button>
          <button type="button" disabled={!ready || !world}
            onClick={() => replace('reset', 'discard')}>Discard and reset</button>
        </div>
        <div className="universe-controls universe-archives">
          <label>Saved experiment
            <select value={selectedId} onChange={(event) => setSelectedId(event.target.value)}>
              <option value="">Choose a saved experiment</option>
              {(data.experiments || []).map((experiment) => <option key={experiment.id} value={experiment.id}>
                {experiment.name} · {new Date(experiment.saved_at).toLocaleString()}
              </option>)}
            </select>
          </label>
          {world ? <>
            <button type="button" disabled={!ready || !selected || !validName}
              onClick={() => replace('reload', 'save')}>Save current and load</button>
            <button type="button" disabled={!ready || !selected}
              onClick={() => replace('reload', 'discard')}>Discard current and load</button>
          </> : <button type="button" disabled={!ready || !selected}
            onClick={() => replace('reload', 'none')}>Load selected</button>}
          <button type="button" disabled={!ready || !selected} onClick={() => {
            if (window.confirm(`Delete the saved experiment "${selected.name}"? This cannot be undone.`)) {
              void perform('delete', { id: selected.id, confirm: true });
            }
          }}>Delete selected save</button>
          <button type="button" disabled={busy || running} onClick={() => void perform('read')}>Refresh current state</button>
        </div>
        {data.api_version !== 2 && <p>Restart the app server to enable saving and loading experiments.</p>}
        <p>Saved copies stay on this computer and are not included in Git. Deleting a copy does not delete the current run.</p>
      </section>
      <div className="universe-phases" aria-label="Prediction and learning stages">
        {Object.entries(names).map(([key, label], index) => <div key={key}
          className={world?.phase === key ? 'universe-phase universe-phase-next' : 'universe-phase'}
          aria-current={world?.phase === key ? 'step' : undefined}>
          <span>{index + 1}</span> {label}{world?.phase === key && ' · next'}
        </div>)}
      </div>
      <section className="universe-panel">
        <div className="universe-panel-heading">
          <h3>Quarks and their bonds</h3>
          <StepRunControls {...controls} />
        </div>
        {world?.bond_mode === 'three-quark'
          ? <p>Unfrozen pop-outs choose eligible partners, never the seeder. Groups cannot exceed three.
            When three connect, each tries once within the group and the candidate freezes.
            Each direction, both directions, or no bond has a 25% chance; existing bonds stay unchanged.</p>
          : <p>The first pop-out starts free. On each step, every pop-out chooses another pop-out,
            never the seeder; each direction, both directions, or no bond has a 25% chance.
            One-way strength is 0.5; two-way strength is 0.25 each. Existing bonds stay unchanged.
            Bonds do not affect predictions yet.</p>}
        <p className="universe-connection-update" role="status">
          {events.length ? describe(events[events.length - 1]) : 'Nothing exists in the experiment yet.'}
          {attemptSummary(events[events.length - 1])}
          {' '}Next: {world ? names[world.phase] : 'Spark'}.
        </p>
        <div className="universe-bond-legend" aria-label="Bond colors">
          <span><i style={{ background: '#67e8d5' }} />0 → 1</span>
          <span><i style={{ background: '#c5a5ff' }} />1 → 0</span>
          <span><i style={{ background: '#84aaff' }} />0 → 0</span>
          <span><i style={{ background: '#ffb46f' }} />1 → 1</span>
          <span><i style={{ background: '#ffdc73' }} />Both ways</span>
        </div>
        {Array.isArray(data.bonds) && (world?.bond_mode !== 'three-quark' || Array.isArray(data.groups))
          ? world?.bond_mode === 'three-quark'
            ? <PatternLab events={events} bonds={data.bonds} groups={data.groups} />
            : <UniverseGraph events={events} recordedBonds={data.bonds} />
          : <p role="alert">Refresh current state to load the updated bond graph.</p>}
      </section>
      <div className="universe-grid">
        <section className="universe-panel">
          <h3>What just happened?</h3>
          <p>Each prediction sees the previous bit, not the next answer. Error is answer minus the predicted chance of 1.
            The two weights change only during <strong>Reweight</strong>.</p>
          <ol className="universe-events">{events.slice().reverse().map((event, index) =>
            <li key={events.length - index}><span>Step {events.length - index}</span> {describe(event)}
              <small> · energy {event.energy}</small>
              {(event.pattern_attempts || event.attempts)?.length > 0 &&
                <AttemptDetails attempts={event.pattern_attempts || event.attempts} />}</li>)}</ol>
          {!events.length && <p>No events yet.</p>}
        </section>
        <section className="universe-panel universe-code">
          <h3>The actual Python, explained</h3>
          <p>These are the real lines used by this experiment. Text after <code>#</code> explains each line.</p>
          <pre><code>{data.source.map((line, index) => `${String(index + 1).padStart(3, ' ')}  ${line}`).join('\n')}</code></pre>
        </section>
      </div>
      <p className="universe-note">The learner uses a tiny two-weight prediction rule; it does not yet generate letters, words,
        thoughts, sentences, or memories. Those are future stages, not claims about what this experiment can do.</p>
    </>}
  </section>;
}
