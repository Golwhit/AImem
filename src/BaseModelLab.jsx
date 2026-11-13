import { useEffect, useRef, useState } from 'react';
import './base-model-lab.css';

const format = (value) => Number(value).toFixed(6);
const tokenLabel = (label) => label === ' ' ? 'space' : label;

function Values({ title, values, columns = 8 }) {
  const maximum = Math.max(...values.map(Math.abs), 0.000001);
  return <section className="lab-values">
    <h4>{title}</h4>
    <div className="lab-cells" style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}>
      {values.map((value, index) => <span key={index} title={`Index ${index}: ${value}`}
        style={{ background: `rgba(${value < 0 ? '225,100,100' : '67,169,211'},${0.1 + 0.6 * Math.abs(value) / maximum})` }}>
        <small>{index}</small>{value.toFixed(3)}
      </span>)}
    </div>
    <small>All values shown. Blue: positive; red: negative. Intensity is relative within this grid.</small>
  </section>;
}

function Probabilities({ trace, after = false }) {
  const values = trace[after ? 'after' : 'before'];
  const top = values.probabilities.reduce((best, value, index, probabilities) => value > probabilities[best] ? index : best, 0);
  return <><p>Highest-probability training slot: <strong>{tokenLabel(trace.tokens[top].label)}</strong>.
    Target: <strong>{tokenLabel(trace.tokens[trace.target].label)}</strong>.
    {' '}Reserved slots participate in this training calculation; normal text sampling excludes their IDs 0 and 1.</p>
    <div className="lab-scroll"><table className="lab-table">
    <caption>{after ? 'After update' : 'Before update'}: actual output distribution (including reserved tokens)</caption>
    <thead><tr><th>Token</th><th>Raw score</th><th>Probability</th></tr></thead>
    <tbody>{trace.tokens.map((token) => <tr key={token.id} className={token.id === trace.target ? 'lab-target' : ''}>
      <th>{tokenLabel(token.label)} {token.id === trace.target ? '(target)' : ''}</th>
      <td>{format(values.logits[token.id])}</td>
      <td><div className="lab-probability"><span style={{ width: `${values.probabilities[token.id] * 100}%` }} /></div>
        {(values.probabilities[token.id] * 100).toFixed(3)}%</td>
    </tr>)}</tbody>
  </table></div></>;
}

function Network({ trace, stage }) {
  const values = trace[stage === 'recheck' ? 'after' : 'before'];
  const columns = [
    { name: 'Context IDs', x: 65, labels: trace.context.map((token) => tokenLabel(trace.tokens[token].label)), active: stage === 'example' },
    { name: 'Embeddings', x: 240, labels: ['8 values', '8 values', '8 values', '8 values'], active: stage === 'embedding' },
    { name: 'Hidden', x: 415, labels: values.hidden.slice(0, 4).map((v) => v.toFixed(3)), active: ['hidden', 'gradient'].includes(stage) },
    { name: 'Output', x: 590, labels: trace.tokens.slice(0, 4).map((t) => tokenLabel(t.label)), active: ['logits', 'probability', 'loss', 'recheck', 'update'].includes(stage) },
  ];
  return <figure className="lab-network">
    <svg viewBox="0 0 680 280" role="img" aria-label="Four context tokens to four embedding rows, sixteen hidden neurons, and output tokens">
      {columns.slice(0, -1).map((column, i) => column.labels.flatMap((_, a) => columns[i + 1].labels.map((__, b) =>
        <line key={`${i}-${a}-${b}`} x1={column.x + 36} y1={65 + a * 48} x2={columns[i + 1].x - 36} y2={65 + b * 48}
          stroke={stage === 'gradient' ? '#e5a56e' : '#405578'} strokeWidth="1" />)))}
      {columns.map((column) => <g key={column.name}>
        <text x={column.x} y="25" textAnchor="middle" fill="#dbeafe">{column.name}</text>
        {column.labels.map((label, index) => <g key={index}>
          <rect x={column.x - 39} y={48 + index * 48} width="78" height="34" rx="8"
            fill={column.active ? '#155e75' : '#1e293b'} stroke={column.active ? '#67e8f9' : '#64748b'} />
          <text x={column.x} y={70 + index * 48} textAnchor="middle" fill="#f1f5f9" fontSize="11">{label}</text>
        </g>)}
      </g>)}
      <text x="340" y="265" textAnchor="middle" fill="#b9c9de" fontSize="12">
        4 × 8 embedding inputs → 16 hidden neurons → {trace.tokens.length} output slots
      </text>
    </svg>
    <figcaption>Schematic: only four hidden/output nodes are drawn; connections are illustrative, not weight magnitudes.
      Exact values appear below. Orange connections mark the backward-error stage.</figcaption>
  </figure>;
}

function WeightedSum({ trace, output = false }) {
  const index = output ? trace.target : 0;
  const inputs = output ? trace.before.hidden : trace.before.embedding;
  const group = trace.weights.find((entry) => entry.name === (output ? 'output' : 'hidden'));
  const bias = trace.weights.find((entry) => entry.name === (output ? 'output_bias' : 'hidden_bias')).before[index];
  const width = output ? trace.tokens.length : trace.hidden_size;
  const terms = inputs.map((value, i) => ({ value, weight: group.before[i * width + index], product: value * group.before[i * width + index] }));
  const total = bias + terms.reduce((sum, term) => sum + term.product, 0);
  return <section>
    <p>{output ? `Target output ${tokenLabel(trace.tokens[index].label)}` : 'Hidden neuron 0'}:
      {' '}bias {format(bias)} + weighted sum = {format(total)}
      {!output && ` → tanh = ${format(Math.tanh(total))}`}</p>
    <details><summary>Inspect all {inputs.length} multiplications</summary>
      <div className="lab-scroll"><table className="lab-table">
        <thead><tr><th>Input index</th><th>Input</th><th>Connection weight</th><th>Product</th></tr></thead>
        <tbody>{terms.map((term, i) => <tr key={i}><th>{i}</th>
          <td>{format(term.value)}</td><td>{format(term.weight)}</td><td>{format(term.product)}</td></tr>)}</tbody>
      </table></div>
    </details>
  </section>;
}

function WeightInspector({ trace, initializing = false }) {
  const [group, setGroup] = useState('output_bias');
  const [offset, setOffset] = useState(0);
  const values = trace.weights.find((entry) => entry.name === group);
  const limit = 16;
  const start = Math.min(offset, Math.max(0, values.count - 1));
  return <section>
    <label>Parameter group <select value={group} onChange={(event) => { setGroup(event.target.value); setOffset(0); }}>
      {trace.weights.map((entry) => <option key={entry.name}>{entry.name}</option>)}
    </select></label>
    <p>{values.count} values; {initializing ? 'seeded initialization' : `${values.changed} changed in this update`}.</p>
    <div className="lab-scroll"><table className="lab-table">
      <thead><tr><th>Index</th>{initializing ? <th>Initial value</th> : <><th>Before</th><th>Gradient</th><th>− rate × gradient</th><th>After</th></>}</tr></thead>
      <tbody>{values.before.slice(start, start + limit).map((before, i) => {
        const index = start + i;
        return <tr key={index}><th>{index}</th>{initializing
          ? <td>{format(trace.initial_weights[group][index])}</td>
          : <><td>{format(before)}</td><td>{format(values.gradient[index])}</td>
            <td>{format(-trace.rate * values.gradient[index])}</td><td>{format(values.after[index])}</td></>}</tr>;
      })}</tbody>
    </table></div>
    <div className="lab-controls">
      <button disabled={start === 0} onClick={() => setOffset(Math.max(0, start - limit))}>Previous values</button>
      <span>{start + 1}–{Math.min(start + limit, values.count)} of {values.count}</span>
      <button disabled={start + limit >= values.count} onClick={() => setOffset(start + limit)}>More values</button>
    </div>
  </section>;
}

function LearningCurve({ trace }) {
  const values = trace.curve.map((row) => row.corpus_loss);
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const points = values.map((value, index) => `${35 + index * 530 / Math.max(1, values.length - 1)},${110 - (value - minimum) * 80 / Math.max(0.001, maximum - minimum)}`).join(' ');
  return <figure className="lab-curve">
    <svg viewBox="0 0 600 145" role="img" aria-label="Average training-string loss across completed SGD updates; lower is better">
      <line x1="35" y1="115" x2="565" y2="115" stroke="#64748b" />
      <polyline points={points} fill="none" stroke="#67e8f9" strokeWidth="3" />
      <text x="35" y="138" fill="#cbd5e1">0 updates</text>
      <text x="460" y="138" fill="#cbd5e1">{trace.updates} updates</text>
      <text x="35" y="18" fill="#cbd5e1">Loss range {format(minimum)}–{format(maximum)} (scaled axis)</text>
    </svg>
    <figcaption>Average loss on the same tiny training string, not an unseen test.
      Lower is better; a downward change does not establish language understanding.</figcaption>
    <p>Start: {format(values[0])} → now: {format(values[values.length - 1])}</p>
  </figure>;
}

function StageVisual({ trace, stage }) {
  const label = (id) => tokenLabel(trace.tokens[id].label);
  switch (stage) {
    case 'alphabet':
      return <><p>Text: <code>{JSON.stringify(trace.text)}</code></p><div className="lab-token-list">
        {trace.tokens.map((token) => <span key={token.id}>{tokenLabel(token.label)} → ID {token.id}</span>)}
      </div><p>Boundary pads missing history. Unknown represents characters outside the alphabet.
        These are neural reserved tokens, not L0/L1 recognition.</p></>;
    case 'initialize':
      return <><p>{trace.parameters} trainable numbers; context {trace.context_size}, embedding {trace.embedding_size}, hidden {trace.hidden_size}.
        Initialization is always seed {trace.seed}, before any of the displayed updates.</p><WeightInspector trace={trace} initializing /></>;
    case 'example':
      return <><p>Update {trace.updates}: source character position {trace.example_position + 1} of {trace.examples_per_pass}.
        {' '}Pass {Math.floor((trace.updates - 1) / trace.examples_per_pass) + 1}.</p>
        <p>Source: <code>{trace.text.slice(0, trace.example_position)}<mark>{trace.text[trace.example_position]}</mark>{trace.text.slice(trace.example_position + 1)}</code></p>
        <div className="lab-token-list">{trace.context.map((id, i) => <span key={i}>{label(id)} [{id}]</span>)}
          <strong>→ target: {label(trace.target)} [{trace.target}]</strong></div></>;
    case 'embedding':
      return <Values title="32 flattened embedding inputs, in context order" values={trace.before.embedding} />;
    case 'hidden':
      return <><Values title="16 actual tanh activations" values={trace.before.hidden} />
        <WeightedSum trace={trace} />
        <p>Activations are temporary; connection weights persist through training.</p></>;
    case 'logits':
      return <><WeightedSum trace={trace} output /><Probabilities trace={trace} /></>;
    case 'probability':
      return <Probabilities trace={trace} />;
    case 'loss':
      return <div className="lab-equation">Target {label(trace.target)}: −log({format(trace.before.probabilities[trace.target])})
        {' '}= <strong>{format(trace.loss_before)}</strong><p>A confident wrong prediction gives a larger loss.
          No weight is changed by measuring loss.</p></div>;
    case 'gradient':
      return <><Values title="Output bias gradients: probability minus target indicator" values={trace.weights.find((entry) => entry.name === 'output_bias').gradient} columns={4} />
        <p>The correct target has 1 subtracted from its probability; other outputs keep their probability.
          Backpropagation also calculates gradients for hidden connections and shared embedding rows.</p>
        <WeightInspector trace={trace} /></>;
    case 'update':
      return <><p>Every scalar uses: <code>after = before − {trace.rate} × gradient</code>.
        Zero gradients leave weights unchanged.</p><WeightInspector trace={trace} /></>;
    case 'recheck':
      return <><p>Same example loss: <strong>{format(trace.loss_before)} → {format(trace.loss_after)}</strong>.
        {' '}Target probability: {(100 * trace.before.probabilities[trace.target]).toFixed(3)}%
        {' '}→ {(100 * trace.after.probabilities[trace.target]).toFixed(3)}%.</p>
        <Probabilities trace={trace} after /><LearningCurve trace={trace} /></>;
    default:
      return null;
  }
}

export default function BaseModelLab() {
  const [text, setText] = useState('ab ab ab ');
  const [rate, setRate] = useState('0.1');
  const [trace, setTrace] = useState(null);
  const [stageIndex, setStageIndex] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const controller = useRef(null);

  const load = async (updates = 1, reset = false) => {
    controller.current?.abort();
    const request = new AbortController();
    controller.current = request;
    setLoading(true);
    setError('');
    try {
      const response = await fetch('/api/base-model-lab/trace', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        signal: request.signal, body: JSON.stringify({ text, rate: Number(rate), updates }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      setTrace(data);
      setStageIndex(reset ? 0 : 2);
    } catch (failure) {
      if (failure.name !== 'AbortError') setError(`Lab trace unavailable: ${failure.message}`);
    } finally {
      if (!request.signal.aborted) setLoading(false);
    }
  };

  useEffect(() => {
    void load(1, true);
    return () => { controller.current?.abort(); };
  }, []);

  const lesson = trace?.lessons[stageIndex];
  const dirty = trace && (text !== trace.text || Number(rate) !== trace.rate);
  return <section className="base-lab hero-card">
    <p className="eyebrow">From scratch • base only</p>
    <h2>Base Model Lab</h2>
    <p>Build → predict → measure error → learn. Real Python calculations, not simulated animation.
      This disposable model cannot change active chat, its checkpoint, or your memories.</p>
    <p>This walkthrough teaches the preserved original character base, not the working New Base
      with word/punctuation features and character fallback.</p>
    <form onSubmit={(event) => { event.preventDefault(); void load(1, true); }}>
      <label>Tiny training text (printable ASCII, up to 128 characters)
        <input aria-label="Tiny training text" maxLength={128} value={text} onChange={(event) => setText(event.target.value)} />
      </label>
      <label>Learning rate
        <input aria-label="Lab learning rate" type="number" min="0.001" max="1" step="0.001" value={rate} onChange={(event) => setRate(event.target.value)} />
      </label>
      <button disabled={loading} type="submit">{loading ? 'Calculating…' : 'Build / reset teaching model'}</button>
    </form>
    {error && <p className="banner banner-error" role="alert">{error}</p>}
    {dirty && <p role="status">Inputs changed. Build/reset to apply them; the trace below still uses {JSON.stringify(trace.text)} and rate {trace.rate}.</p>}
    {trace && lesson && <>
      <p className="lab-isolation">Base only: no L0/L1, no pretrained weights. Fixed seed {trace.seed}.
        {' '}Inspecting update {trace.updates}/32; context {trace.context_size}, embeddings {trace.embedding_size}, hidden neurons {trace.hidden_size}.</p>
      <nav className="lab-steps" aria-label="Calculation steps">{trace.lessons.map((entry, index) =>
        <button key={entry.id} disabled={loading} aria-current={index === stageIndex ? 'step' : undefined}
          onClick={() => setStageIndex(index)}>{entry.title}</button>)}</nav>
      <div className="lab-controls">
        <button disabled={loading || stageIndex === 0} onClick={() => setStageIndex(stageIndex - 1)}>Back a step</button>
        <span aria-live="polite">Step {stageIndex + 1} of {trace.lessons.length}</span>
        <button disabled={loading || stageIndex === trace.lessons.length - 1} onClick={() => setStageIndex(stageIndex + 1)}>Next step</button>
      </div>
      <h3>{lesson.title}</h3>
      <p>{lesson.explanation}</p>
      <div className="lab-teaching-space">
        <section className="lab-visual" aria-label="Calculation visualization">
          <Network trace={trace} stage={lesson.id} />
          <StageVisual key={`${lesson.id}-${trace.updates}`} trace={trace} stage={lesson.id} />
        </section>
        <section className="lab-short-code">
          <h4>Small code representation</h4>
          <p>Conceptual sketch; the exact executable Python is below.</p>
          <pre><code>{lesson.compact}</code></pre>
        </section>
      </div>
      <section className="lab-source">
        <h4>Actual Python, explained line by line</h4>
        <p><code>{lesson.source.file} · NeuralModel.{lesson.source.method}</code>.
          Highlighted lines relate to this step. Source is read directly from the running backend;
          explanations beside it are teaching annotations, not replacement code.</p>
        <div className="lab-source-lines">{lesson.source.lines.map((line) => <div key={line.number}
          className={`lab-source-line ${line.highlight ? 'lab-source-highlight' : ''}`}>
          <pre><span className="lab-line-number">{line.number}</span><code>{line.code}</code></pre>
          {line.note && <p>{line.note}</p>}
        </div>)}</div>
      </section>
      <div className="lab-controls">
        <button disabled={loading || dirty || trace.updates >= 32} onClick={() => void load(trace.updates + 1)}>
          Train next character example
        </button>
        <span>One update = one next-character example, not one sentence or one pass.</span>
      </div>
      <details><summary>What is really happening?</summary><p>{trace.policy}</p>
        <p>Each request replays the same seeded training prefix in memory.
          Back/Next reveals stages of the calculated update; it does not train live chat.
          “Train next” extends that prefix by one update. Reset returns to the first update.
          Initialization always shows the original untrained weights; other calculations show the selected update.</p>
        <p>Core source SHA256: <code>{trace.core_sha256}</code></p>
      </details>
    </>}
  </section>;
}
