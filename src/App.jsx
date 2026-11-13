import { useEffect, useRef, useState } from 'react';
import BaseModelLab from './BaseModelLab';

export default function App() {
  const [activeTab, setActiveTab] = useState(() => {
    const view = new URLSearchParams(window.location.search).get('view');
    return ['generation', 'base-lab'].includes(view) ? view : 'chat';
  });
  const [sessionId] = useState(() => {
    const existing = sessionStorage.getItem('memory-chat-session');
    if (existing) return existing;
    const created = crypto.randomUUID();
    sessionStorage.setItem('memory-chat-session', created);
    return created;
  });

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Push Bridge POC</p>
          <h1>Memory + AI Chat</h1>
          <p className="subtitle">Local memory</p>
        </div>
      </header>

      <nav className="tabs">
        <button className={activeTab === 'chat' ? 'tab active' : 'tab'} onClick={() => setActiveTab('chat')}>Chat</button>
        <button className={activeTab === 'generation' ? 'tab active' : 'tab'} onClick={() => setActiveTab('generation')}>Generation</button>
        <button className={activeTab === 'base-lab' ? 'tab active' : 'tab'} onClick={() => setActiveTab('base-lab')}>Base Model Lab</button>
        <button className={activeTab === 'graph' ? 'tab active' : 'tab'} onClick={() => setActiveTab('graph')}>Memory Graph</button>
      </nav>

      <main style={{ maxWidth: activeTab === 'base-lab' ? 1200 : 800, margin: '0 auto', padding: '0 16px' }}>
        {activeTab === 'base-lab' && <BaseModelLab />}
        {activeTab === 'chat' && <ChatPanel sessionId={sessionId} />}
        {activeTab === 'generation' && <><NeuralPanel /><GeneratorPanel /></>}
        {activeTab === 'graph' && (
          <iframe src="/graph" style={{ width: '100%', height: '75vh', border: 'none', background: '#0f0f1a' }} title="Memory Graph" />
        )}
      </main>
    </div>
  );
}

function ChatPanel({ sessionId }) {
  const [messages, setMessages] = useState([]);
  const [generatorStatus, setGeneratorStatus] = useState(null);
  const [chatModel, setChatModel] = useState(null);
  const [progressError, setProgressError] = useState('');
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [includeLab, setIncludeLab] = useState(false);
  const bottomRef = useRef(null);

  const refreshProgress = async () => {
    try {
      const [response, modelResponse] = await Promise.all([fetch('/api/generator'), fetch('/api/chat-model')]);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      if (!modelResponse.ok) throw new Error(`Chat model HTTP ${modelResponse.status}`);
      setGeneratorStatus(await response.json());
      setChatModel(await modelResponse.json());
      setProgressError('');
    } catch (err) {
      setProgressError(`Progress unavailable: ${err.message}`);
    }
  };

  useEffect(() => {
    void refreshProgress();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const send = async () => {
    const text = input;
    if (!text.trim() || loading) return;

    setMessages((prev) => [...prev, { role: 'user', text }]);
    setInput('');
    setLoading(true);
    setError('');

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, sessionId, include_in_generator: true, include_lab: includeLab })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      setMessages((prev) => [...prev, {
        role: 'assistant',
        text: data.reply,
        mode: data.mode,
        chatModel: data.chat_model,
        generation: data.generation,
        trainingSources: data.training_sources,
        generatorLearning: data.generator_learning,
        responseId: data.response_memory_id,
        selection: data.selection,
        lab: data.lab,
        labError: data.lab_error,
      }]);
      void refreshProgress();
    } catch (err) {
      setError(err.message);
      setMessages((prev) => prev.slice(0, -1));
    } finally {
      setLoading(false);
    }
  };

  const rateResponse = async (responseId, rating, approveForScoring, correction = '', trainNeuralCorrection = false) => {
    setMessages((previous) => previous.map((message) => message.responseId === responseId ? { ...message, ratingPending: true } : message));
    setError('');
    try {
      const response = await fetch('/api/response-feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ response_id: responseId, rating, correction, approve_for_scoring: approveForScoring,
          approve_correction: approveForScoring && Boolean(correction.trim()), train_neural_correction: trainNeuralCorrection })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      setMessages((previous) => previous.map((message) => message.responseId === responseId ? { ...message, rating, scorer: data.scorer,
        correctionSaved: Boolean(correction.trim()), correctionTrained: data.neural_training?.trained || data.scorer?.correction_training?.trained,
        neuralCorrectionTrained: data.neural_training?.trained } : message));
      void refreshProgress();
    } catch (err) {
      setError(err.message);
    } finally {
      setMessages((previous) => previous.map((message) => message.responseId === responseId ? { ...message, ratingPending: false } : message));
    }
  };

  return (
    <section className="hero-card" style={{ marginTop: 24 }}>
      <div>
        <p className="eyebrow">AI Chat</p>
        <h2>Memory and local generation</h2>
      </div>
      {chatModel?.ready && <p><strong>Active chat: {chatModel.label || 'Historical layered neural model (L0/L1)'}</strong>
        {' '}— {chatModel.parameters.toLocaleString()} parameters, seed {chatModel.seed},
        {' '}{chatModel.training_round ? `training round ${chatModel.training_round}` : `${chatModel.passes} passes`}; run {chatModel.run_id}.
        {' '}Training history: {chatModel.source?.training}.
        {' '}{chatModel.note}
        {chatModel.comparison_baseline && <> Original comparison control:
          {' '}{chatModel.comparison_baseline.parameters.toLocaleString()} parameters,
          {' '}seed {chatModel.comparison_baseline.seed}, {chatModel.comparison_baseline.passes} passes.</>}
        {' '}Optional model-lab comparisons below use the separate legacy generator.</p>}

      <div className="generator-status" aria-live="polite">
        {generatorStatus ? (
          <>
            <span>{chatModel?.ready ? 'Legacy generator (not active chat)' : 'Generation'} {generatorStatus.generation}</span>
            <span>{generatorStatus.document_count} training sources</span>
            <span>{generatorStatus.training_characters.toLocaleString()} characters</span>
          </>
        ) : <span>Loading progress...</span>}
        <span>{messages.filter((message) => message.rating === 'useful').length} relevant this session</span>
        <span>{messages.filter((message) => message.rating === 'needs_correction').length} need correction this session</span>
        <a href="/?view=generation">Compare generations</a>
      </div>
      {progressError && <p className="banner banner-error">{progressError}</p>}
      {!chatModel?.ready && generatorStatus?.seed_extraction && (
        <p className="message-meta">
          Seed layers run alongside training: {generatorStatus.seed_extraction.counts.alive} alive,
          {' '}{generatorStatus.seed_extraction.counts.dead} dead, and {generatorStatus.seed_extraction.counts.maybe} review occurrences
          {' '}across {generatorStatus.seed_extraction.document_count} sources. Existing training is unchanged.
        </p>
      )}
      {!chatModel?.ready && generatorStatus?.scorer && (
        <section aria-label="Response learning progress" style={{ borderBottom: '1px solid #334155', paddingBottom: 12 }}>
          <div className="generator-status">
            <span>Response scorer generation {generatorStatus.scorer.generation}</span>
            <span>{generatorStatus.scorer.examples} reviewed pairs</span>
            <span>{generatorStatus.scorer.queries} distinct prompts</span>
            <span>{generatorStatus.scorer.training?.status || 'Awaiting ratings'}</span>
          </div>
          <label style={{ display: 'block', fontSize: 12 }}>Reviewed pairs: {generatorStatus.scorer.examples} / {generatorStatus.scorer.minimum_examples}
            <progress style={{ width: '100%' }} aria-label="Reviewed pairs toward validation minimum" value={Math.min(generatorStatus.scorer.examples, generatorStatus.scorer.minimum_examples)} max={generatorStatus.scorer.minimum_examples} />
          </label>
          <label style={{ display: 'block', fontSize: 12 }}>Distinct prompts: {generatorStatus.scorer.queries} / {generatorStatus.scorer.minimum_queries}
            <progress style={{ width: '100%' }} aria-label="Distinct prompts toward validation minimum" value={Math.min(generatorStatus.scorer.queries, generatorStatus.scorer.minimum_queries)} max={generatorStatus.scorer.minimum_queries} />
          </label>
          {generatorStatus.scorer.training?.reason && <small className="message-meta">{generatorStatus.scorer.training.reason}</small>}
          {generatorStatus.scorer.training?.candidate_loss != null && <small className="message-meta">Held-out error: {generatorStatus.scorer.training.baseline_loss.toFixed(4)} → {generatorStatus.scorer.training.candidate_loss.toFixed(4)}</small>}
        </section>
      )}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 360, overflowY: 'auto', margin: '12px 0' }}>
        {messages.length === 0 && (
          <p style={{ color: '#aaa', textAlign: 'center', padding: 16 }}>No messages yet. Start a conversation!</p>
        )}
        {messages.map((m, i) => (
          <div key={i} style={{ alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start', maxWidth: '80%', padding: '8px 12px', borderRadius: 12, background: m.role === 'user' ? '#3b82f6' : '#2a2a3a', color: '#eee', fontSize: 14 }}>
            <div>{m.text}</div>
            {m.lab && <ModelLab lab={m.lab} />}
            {m.labError && <p className="banner banner-error">Comparison unavailable: {m.labError}. Memory capture and ordinary chat continued.</p>}
            {m.mode === 'local-character-ngram' && (
              <small className="message-meta">
                Experimental local character model, generation {m.generation}.
                {m.trainingSources?.length ? ` Sources: ${m.trainingSources.map((source) => source.title).join(', ')}.` : ''}
              </small>
            )}
            {['local-layered-neural', 'local-base-neural', 'local-conversation-neural'].includes(m.mode) && <small className="message-meta">
              {m.mode === 'local-conversation-neural' ? 'Persistent conversation learner (reviewed teaching and feedback)'
                : m.mode === 'local-base-neural' ? 'Experimental layer-free New Base' : 'Historical L0/L1 neural'} output; not a grounded answer or scripture.
              {' '}Run {m.chatModel?.run_id}, seed {m.chatModel?.seed},
              {' '}{m.chatModel?.training_round ? `training round ${m.chatModel.training_round}` : `${m.chatModel?.passes} passes`}.
              {m.mode === 'local-conversation-neural'
                ? ' Unmarked replies do not train. Relevant or a submitted correction trains the active weights.'
                : ' Prompts and feedback are saved for review, not neural weight updates.'}
            </small>}
            {m.generatorLearning?.reason === 'fixed_neural_checkpoint' && <small className="message-meta">
              Generating this reply did not train weights; the prompt was saved to memory.
            </small>}
            {m.generatorLearning?.added && <small className="message-meta">Your prompt was added for future generation {m.generatorLearning.generation}.</small>}
            {m.generatorLearning?.reason === 'already_present' && <small className="message-meta">Your prompt is already in the generator corpus.</small>}
            {m.generatorLearning?.reason === 'too_short' && <small className="message-meta">Prompt skipped: at least four characters are needed for generator training.</small>}
            {m.generatorLearning?.reason === 'error' && <small className="message-meta">Prompt was saved to memory but not generator training: {m.generatorLearning.message}</small>}
            {m.selection && <small className="message-meta">{Number.isFinite(m.selection.score) ? `Selected from ${m.selection.candidate_count} candidates. Scorer generation ${m.selection.generation}, selection score ${m.selection.score.toFixed(2)}.` : 'No generated candidate was available; showing the memory fallback.'}</small>}
            {m.responseId && (
              <div role="group" aria-label="Response relevance" style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 8 }}>
                <button type="button" title="Approve this exact answer and train immediately" aria-pressed={m.rating === 'useful'} disabled={m.ratingPending} onClick={() => rateResponse(m.responseId, 'useful', m.mode === 'local-character-ngram')}>{m.ratingPending && m.rating !== 'needs_correction' ? 'Training…' : 'Relevant'}</button>
                <button type="button" title="Review this pair and train response selection" aria-pressed={m.rating === 'needs_correction'} disabled={m.ratingPending} onClick={() => rateResponse(m.responseId, 'needs_correction', m.mode === 'local-character-ngram')}>Needs correction{m.mode === 'local-character-ngram' ? ' and train' : ''}</button>
                {m.rating && <small className="message-meta" role="status">{m.neuralCorrectionTrained
                  ? 'Approved answer trained, verified and active in chat.'
                  : m.scorer ? `Local preference updated. Scorer generation ${m.scorer.generation}: ${m.scorer.training?.status}.`
                    : m.mode === 'local-conversation-neural' ? 'No weight update until you submit a correction.' : 'Feedback saved for review.'}</small>}
              </div>
            )}
            {m.rating === 'needs_correction' && (
              <div style={{ marginTop: 8 }}>
                <label style={{ display: 'block', fontSize: 12 }}>Your corrected response
                  <textarea rows={3} maxLength={m.mode === 'local-conversation-neural' ? 95 : 2000} value={m.correction || ''} style={{ width: '100%', marginTop: 6, resize: 'vertical' }} onChange={(event) => {
                    const correction = event.target.value;
                    setMessages((previous) => previous.map((message) => message.responseId === m.responseId ? { ...message, correction, correctionSaved: false } : message));
                  }} />
                </label>
                <button type="button" disabled={m.ratingPending || !m.correction?.trim() || (m.mode === 'local-character-ngram' && m.correction.trim().length < 4)} onClick={async () => {
                  await rateResponse(m.responseId, 'needs_correction', m.mode === 'local-character-ngram', m.correction,
                    m.mode === 'local-conversation-neural');
                }}>{m.ratingPending ? 'Training correction…' : m.mode === 'local-conversation-neural' ? 'Train correction now'
                  : m.mode === 'local-character-ngram' ? 'Approve correction and train' : 'Save correction for review'}</button>
                {m.mode === 'local-conversation-neural' && <small className="message-meta">Submitting approves this exact prompt and correction for immediate weight training. One line, up to 95 characters.</small>}
                {m.correctionSaved && <small className="message-meta" role="status">{m.neuralCorrectionTrained
                  ? 'Correction learned, verified and active in chat. Included in future training rounds.'
                  : m.correctionTrained ? 'Your correction trained the generator and positive response selection.' : 'Correction saved for review; not added to generator training text.'}</small>}
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div style={{ alignSelf: 'flex-start', padding: '8px 12px', borderRadius: 12, background: '#2a2a3a', color: '#aaa', fontSize: 14 }}>
            Thinking…
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {error && <p className="banner banner-error" style={{ margin: '8px 0' }}>{error}</p>}

      <div className="response-row">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && send()}
          placeholder="Enter a prompt..."
          maxLength={2000}
          disabled={loading}
        />
        <button onClick={send} disabled={loading} className="primary">
          {loading ? '…' : 'Send'}
        </button>
      </div>
      <label className="message-meta">
        <input type="checkbox" checked={includeLab} onChange={(event) => setIncludeLab(event.target.checked)} />
        {' '}{chatModel?.ready
          ? 'Opt into the historical layer lab (retired; does not train the working base).'
          : 'Compare extraction and generation before training this prompt (local lab).'}
      </label>
    </section>
  );
}

function ModelLab({ lab }) {
  const extractionModes = [
    ['old_model_transitions', 'Old model extraction'],
    ['combined', 'Layers extraction (Layer 0 + Layer 1)'],
  ];
  return (
    <details className="model-lab" open>
      <summary>Model lab · pre-training generation {lab.generation}</summary>
      <h3>Extraction comparison</h3>
      <p className="message-meta">
        Exact input: {JSON.stringify(lab.prompt)}.
        Old-model extraction shows normalized four-character contexts and next characters.
        Layers extraction shows raw 1–3 character patterns, source offsets, and eligibility.
        Unknowns remain for review; dead patterns are separate. Longer exclusions cannot be detected within these windows.
      </p>
      <div className="lab-grid">
        {extractionModes.map(([key, title]) => (
          <section key={key} className="evaluation-run">
            <h4>{title}</h4>
            <p className="message-meta">
              {Object.entries(lab.extraction[key].reduce((counts, item) => {
                const status = item.status || 'captured';
                counts[status] = (counts[status] || 0) + (item.offsets?.length ?? item.count);
                return counts;
              }, {})).map(([status, count]) => `${status}: ${count}`).join(' · ')} occurrences.
            </p>
            {key === 'old_model_transitions' ? (
              <pre className="lab-extraction">{lab.extraction[key].map(item =>
                `${JSON.stringify(item.context)} → ${JSON.stringify(item.next)} × ${item.count}`
              ).join('\n') || '(No transitions)'}</pre>
            ) : (
              ['alive', 'maybe', 'dead'].map(status => (
                <div key={status}>
                  <h5>{status === 'maybe' ? 'Unknown / needs review' : status === 'alive' ? 'Alive captures' : 'Known dead (not for review)'}</h5>
                  <pre className="lab-extraction">{lab.extraction[key].filter(item => item.status === status).map(item =>
                    `${JSON.stringify(item.text)} × ${item.offsets.length} · offsets ${item.offsets.join(', ')} · eligible: ${item.layer_zero_eligible}${item.development ? ` · development: ${[...new Set(item.development.map(value => value.state))].join(', ')}` : ''}`
                  ).join('\n') || '(None)'}</pre>
                </div>
              ))
            )}
          </section>
        ))}
      </div>
      {lab.development && <DevelopmentReview lab={lab} />}
      {lab.layer_two && <section aria-label="Layer 2 attraction evidence">
        <h3>Layer 2 · Attraction evidence (not training)</h3>
        <p className="message-meta">
          Exact overlapping 2–4-character patterns from this prompt only. Scores are not permission.
          “Always” means observed in this source, not a universal rule. Source ends count as alternatives.
        </p>
        <h4>Observed-always pairs</h4>
        <pre className="lab-extraction">{JSON.stringify(
          lab.layer_two.pairs.filter(pair => pair.always_followed_in_source || pair.always_preceded_in_source),
          null, 2
        )}</pre>
        <details><summary>All attraction evidence, counts, offsets and uncertainty</summary>
          <pre className="lab-extraction">{JSON.stringify(lab.layer_two, null, 2)}</pre>
        </details>
        <p className="message-meta">
          Forward: next character given the left character. Backward: preceding character given the right.
          Unknown pairs have no evidence, not zero permission. Statistical bounds assume independent
          observations; repeated or overlapping text weakens that interpretation.
        </p>
      </section>}
      <h3>Generation comparison</h3>
      <p className="message-meta">
        Same {lab.character_budget}-character maximum per mode.{' '}
        {lab.common_display_characters > 0
          ? `Equal-length comparison: ${lab.common_display_characters} characters each, cropped to the shortest actual output without padding.`
          : 'Equal-length comparison unavailable because at least one mode produced no text. Available full outputs are shown below at their actual lengths; these are not an equal-length comparison.'}
        {' '}Full outputs and shortfalls are shown below.
        {lab.development
          ? ' Developmental layers generation uses foundation/admitted blocks, whole-join Layer 1 checks, then Layer 2 attraction. Admission is not training approval or sentence correctness.'
          : ' Layers-only is a captured-seed block sampler, not a trained model or proof of sentence validity.'}
        {lab.layer_two_generation && ' The two new Layer 2 modes can generate without a living seed: experimental outside Layer 0 admission, with known Layer 1 exclusions enforced. They use this prompt only, not persistent training.'}
      </p>
      <div className="lab-grid">
        {[
          ['old', 'Old model alone'],
          ['layers', lab.development ? 'Developmental layers: blocks → exclusions → attraction' : 'Layers 0/1 alone (rule output)'],
          ['combined', 'Layers 0/1 first → old model continuation'],
          ...(lab.generation_comparison.old_first ? [['old_first', lab.generation_comparison.old_first.context_mode === 'aligned_capture_overlay' ? 'Old model + aligned layer overlay' : 'Old model first → layers continuation (historical)']] : []),
          ...(lab.generation_comparison.layer_two ? [
            ['layer_two', 'Layer 2 alone + Layer 1 exclusions (experimental)'],
            ['old_layer_two', 'Old model + Layer 2 next-character scoring (experimental)'],
          ] : []),
        ].map(([key, title]) => {
          const output = lab.generation_comparison[key];
          return (
            <section key={key} className="evaluation-run">
              <h4>{title}</h4>
              <pre className="evaluation-sample">{
                lab.common_display_characters > 0
                  ? output.display_text
                  : output.text || `(No output: ${output.reason || 'this mode produced no text'})`
              }</pre>
              <small className="message-meta">
                Actual: {output.characters}/{lab.character_budget} characters; shortfall: {output.shortfall}.
                {output.reason && ` ${output.reason}.`}
              </small>
              {key === 'layers' && output.context_mode === 'developmental_blocks' && <details>
                <summary>Developmental block choices and composition checks</summary>
                <pre className="lab-extraction">{JSON.stringify({
                  policy: output.policy, assessment: output.assessment, trace: output.trace,
                }, null, 2)}</pre>
              </details>}
              {output.attraction_metrics && <p className="message-meta">
                {output.identical_to_old ? 'Identical to old output.' : 'Different from old output.'}
                {' '}Prompt-fit loss: {output.attraction_metrics.prompt_attraction_nll?.toFixed(4) ?? 'not scored (empty)'}.
                {' '}Supported next characters: {output.attraction_metrics.supported_next_characters}/{output.attraction_metrics.scored_characters}.
                {' '}Layer 1 violation: {output.attraction_metrics.layer_one_violation || 'none detected'}.
                {' '}These are descriptive checks, not held-out improvement or sentence correctness.
              </p>}
              {(key === 'layer_two' || key === 'old_layer_two') && <details>
                <summary>Layer 2 sampling and excluded candidates</summary>
                <pre className="lab-extraction">{JSON.stringify({policy: output.policy, trace: output.trace}, null, 2)}</pre>
              </details>}
              {key === 'combined' && <p className="message-meta">
                Protected opening: {JSON.stringify(output.protected_prefix)}.
                Context: {output.context_mode || 'unavailable'}. The continuation is not certified by the layers.
              </p>}
              {key === 'old_first' && output.context_mode === 'aligned_capture_overlay' && <>
                <p className="message-meta">
                  Old response remains the base. {output.replacements.length} aligned corrections; no appended text.
                  Matching lowercase letters are corrected using living captures from this input.
                  A dead initial consonant followed by a space can also be replaced by a captured living response-start A or I.
                  This does not certify the whole response.
                  {' '}This is a narrow edit pass, not reversed model execution; no supported match means identical output is expected.
                </p>
                <h5>Layer replacements and supporting source offsets</h5>
                <pre className="lab-extraction">{output.replacements.length ? JSON.stringify(output.replacements, null, 2) : '(No aligned corrections)'}</pre>
              </>}
              {key === 'old_first' && output.context_mode !== 'aligned_capture_overlay' && <>
                <p className="message-meta">
                  Preserved old-model opening: {Array.from(output.protected_prefix).length} characters.
                  The equal-length crop may show only that opening; the appended rule output is shown separately below.
                  Captured blocks are not certified as a valid continuation or sentence.
                </p>
                <h5>Layers continuation (exact spaces)</h5>
                <pre className="evaluation-sample">{output.continuation || '(No continuation)'}</pre>
              </>}
              <details><summary>Full output (exact spaces)</summary><pre className="evaluation-sample">{output.text || '(No output)'}</pre></details>
            </section>
          );
        })}
      </div>
      <small className="message-meta">
        <a href={`/api/model-lab/${lab.record_id}`} download={`model-lab-${lab.record_id}.json`}>Download exact comparison</a>.
        {' '}Rules: {lab.ruleset}. No lab output is automatically used for training.
      </small>
    </details>
  );
}

function DevelopmentReview({ lab }) {
  const [offset, setOffset] = useState('0');
  const [size, setSize] = useState('2');
  const [action, setAction] = useState('defer');
  const [rationale, setRationale] = useState('');
  const [acknowledged, setAcknowledged] = useState(false);
  const [status, setStatus] = useState(lab.development.registry);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const preview = Array.from(lab.prompt).slice(Number(offset), Number(offset) + Number(size)).join('');

  useEffect(() => {
    fetch('/api/development').then(async response => {
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      setStatus(data);
    }).catch(err => setError(err.message));
  }, []);

  const decide = async event => {
    event.preventDefault();
    setBusy(true);
    setError('');
    setNotice('');
    try {
      const response = await fetch('/api/development/decisions', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          record_id: lab.record_id, offset: Number(offset), size: Number(size),
          action, rationale, acknowledge_lab_only: acknowledged,
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      setStatus(data);
      setNotice('Review saved. Run a new comparison to use the current registry; this saved report is unchanged.');
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };
  return <section aria-label="Developmental pattern review">
    <h3>Development: capture → review → test → admit</h3>
    <p className="message-meta">Input: {lab.development.input_assessment.state}.
      {' '}{lab.development.input_assessment.reason}.
      Incomplete fragments are developing; complete unknowns are unreviewed.
      Neither means dead, admitted, or trained. Rules apply only to the English seed experiment.</p>
    <details><summary>Review a building block for lab generation</summary>
      <p className="message-meta">
        Select 1–4 exact characters from this saved prompt. Admission tests the observed boundary
        and surrounding joins. Start-only and interior-only placement are retained.
        Every later composition is checked again. Rationale is a user declaration, not legal clearance.
      </p>
      <form className="generator-form" onSubmit={decide}>
        <label>Source offset (zero-based characters)<input type="number" min="0" max={Array.from(lab.prompt).length - 1} required value={offset} onChange={event => setOffset(event.target.value)} /></label>
        <label>Pattern length<input type="number" min="1" max="4" required value={size} onChange={event => setSize(event.target.value)} /></label>
        <p>Exact selection: <code>{JSON.stringify(preview)}</code></p>
        <label>Decision<select value={action} onChange={event => setAction(event.target.value)}>
          <option value="defer">Defer / needs more evidence</option>
          <option value="admit">Admit for developmental lab generation</option>
          <option value="revoke">Revoke admission</option>
        </select></label>
        <label>Review rationale<textarea required maxLength={1000} value={rationale} onChange={event => setRationale(event.target.value)} /></label>
        <label><input required type="checkbox" checked={acknowledged} onChange={event => setAcknowledged(event.target.checked)} />
          I reviewed the exact source. This decision permits lab generation only, not training, and does not change Layer 0.
        </label>
        <button disabled={busy || !acknowledged} type="submit">{busy ? 'Saving review…' : 'Save developmental review'}</button>
      </form>
      {notice && <p className="message-meta">{notice}</p>}
      <h4>Current review decisions (not this report’s historical snapshot)</h4>
      <pre className="lab-extraction">{JSON.stringify(status, null, 2)}</pre>
    </details>
    {error && <p className="banner banner-error">{error}</p>}
  </section>;
}

function NeuralComparison({ comparison, rates = false }) {
  const title = rates ? 'Learning rate comparison' : 'Character context comparison';
  const label = model => rates ? `Rate ${model.learning_rate}` : `${model.context}-character context`;
  const key = model => rates ? model.learning_rate : model.context;
  return <section className={rates ? 'rate-comparison' : 'context-comparison'}>
    <h3>{title}</h3>
    <p>Status: {comparison.status}. Each model: {comparison.steps_per_model.toLocaleString()} target
      examples, batch size {comparison.batch_size}.
      {rates ? ' Context: 4 characters.' : ` Learning rate: ${comparison.learning_rate}.`}
      {' '}Final test evaluations: {comparison.final_test_evaluations}.</p>
    {comparison.error && <p className="banner banner-error">{comparison.error}</p>}
    <table>
      <thead><tr><th>{rates ? 'Learning rate' : 'Context'}</th><th>Parameters</th><th>Steps</th><th>Training loss</th><th>Validation loss</th></tr></thead>
      <tbody>{comparison.models.map(model => {
        const last = model.history[model.history.length - 1];
        return <tr key={key(model)}>
          <td>{rates ? model.learning_rate : `${model.context} characters`}</td><td>{model.parameter_count.toLocaleString()}</td>
          <td>{model.steps}</td><td>{last?.training_loss.toFixed(4) ?? 'pending'}</td>
          <td>{last?.held_out_loss.toFixed(4) ?? 'pending'}</td>
        </tr>;
      })}</tbody>
    </table>
    <p>{comparison.evaluation_note}</p>
    {comparison.models.map(model => <details key={key(model)}>
      <summary>{label(model)}: samples and loss history</summary>
      {model.samples.map(sample => <div key={sample.prompt}>
        <p>Continuation after <code>{JSON.stringify(sample.prompt)}</code>:</p>
        <pre className="generator-sample">{sample.text}</pre>
      </div>)}
      <pre>{JSON.stringify(model.history, null, 2)}</pre>
    </details>)}
    <details><summary>Exact {rates ? 'learning rate' : 'context'} comparison report</summary><pre>{JSON.stringify(comparison, null, 2)}</pre></details>
  </section>;
}

function LayeredEfficiency({ report }) {
  const loss = metric => metric ? (metric.loss === null ? `infinite (${metric.blocked_targets} blocked)` : metric.loss.toFixed(4)) : 'pending';
  return <section className="layered-efficiency">
    <h3>Full Layer 0/1/2 learning and effort</h3>
    <p>Status: {report.status}. Context {report.context}; learning rate {report.learning_rate};
      {' '}{report.steps_per_neural_arm.toLocaleString()} target draws per neural arm.
      {' '}Baseline target validation loss: {report.target_validation_loss?.toFixed(4) ?? 'pending'}.
      {' '}Final test evaluations: {report.final_test_evaluations}.</p>
    {report.error && <p className="banner banner-error">{report.error}</p>}
    <table>
      <thead><tr><th>Path</th><th>Neural parameters</th><th>Updates</th><th>Training loss</th><th>Validation loss</th><th>Validation coverage</th></tr></thead>
      <tbody>{report.models.map(model => {
        const last = model.history[model.history.length - 1];
        return <tr key={model.name}>
          <td>{model.name}</td><td>{model.parameter_count.toLocaleString()}</td><td>{model.gradient_updates}</td>
          <td>{loss(last?.training)}</td><td>{loss(last?.validation)}</td>
          <td>{last ? `${(last.validation.coverage * 100).toFixed(2)}%` : 'pending'}</td>
        </tr>;
      })}</tbody>
    </table>
    <h4>Work to first measured match</h4>
    <p>Every path reads all {report.training_characters} training characters. Zero gradient updates is
      not zero learning work: layer construction and source observations still count.
      Timing on this shared machine is diagnostic, not a general speed benchmark.</p>
    <table>
      <thead><tr><th>Path</th><th>Updates at match</th><th>Preparation + updates (seconds)</th><th>CPU seconds at match</th><th>End-to-end at match (seconds)</th></tr></thead>
      <tbody>{report.models.map(model => <tr key={model.name}>
        <td>{model.name}</td><td>{model.first_match?.gradient_updates ?? 'not reached'}</td>
        <td>{model.first_match?.training_work_seconds.toFixed(3) ?? '—'}</td>
        <td>{model.first_match?.training_work_cpu_seconds.toFixed(3) ?? '—'}</td>
        <td>{model.first_match?.end_to_end_seconds.toFixed(3) ?? '—'}</td>
      </tr>)}</tbody>
    </table>
    <p>{report.evaluation_note}</p>
    {report.models.map(model => <details key={model.name}>
      <summary>{model.name}: samples, observations, conflicts and history</summary>
      <p>{model.candidate_count} unconfirmed candidates; {model.layer_zero_window_observations} Layer 0
        window observations; {model.layer_two_context_observations} Layer 2 context observations.
        {' '}{model.skipped_rule_conflicts} sampled targets blocked by rules.
        {' '}{model.layer_state_json_bytes.toLocaleString()} serialized bytes of layer evidence.</p>
      {model.samples.map(item => <div key={item.prompt}>
        <p>After <code>{JSON.stringify(item.prompt)}</code>; exclusion: {item.layer_one_violation ?? 'none detected'}.</p>
        <pre className="generator-sample">{item.text}</pre>
      </div>)}
      <pre>{JSON.stringify(model.history, null, 2)}</pre>
    </details>)}
    <details><summary>Exact policy and efficiency report</summary><pre>{JSON.stringify(report, null, 2)}</pre></details>
  </section>;
}

function LayerProof({ report }) {
  const fmt = (value, digits = 4) => value === null || value === undefined ? '—' : value.toFixed(digits);
  const summary = report.summary;
  const fresh = report.fresh_evaluation;
  const verdicts = fresh?.verdicts;
  const yes = value => value === true ? 'met' : value === false ? 'not met' : '—';
  const key = fraction => Number.isInteger(fraction) ? fraction.toFixed(1) : String(fraction);
  return <section className="layered-efficiency">
    <h3>Can the layers be proven? Size curve, seeds, effort, rules and fresh test</h3>
    <p>Status: {report.status}. {report.result_count ?? 0} runs: {report.arms?.length} paths ×
      {' '}{report.fractions?.length} training sizes × {report.seeds?.length} seeds.
      {' '}Baseline (neural, 100% of Acts 1) mean validation: {fmt(summary?.baseline_100_mean_validation)}.
      {' '}Primary layered path (cheapest within 0.02 nats of the best on validation, chosen before fresh data):
      {' '}{report.preregistration?.primary_arm ?? 'not yet registered'}.</p>
    {report.error && <p className="banner banner-error">{report.error}</p>}
    {summary && <table>
      <thead><tr><th>Path</th>{report.fractions.map(f => <th key={f}>{f * 100}% data: validation mean (range)</th>)}
        <th>Training operations at 100%</th><th>Training CPU s at 100%</th><th>Rule breaks / 1,000 generated</th>
        {fresh && <th>Fresh Acts 4–5 loss</th>}</tr></thead>
      <tbody>{Object.entries(summary.arms).map(([name, rows]) => <tr key={name}>
        <td>{name}</td>
        {report.fractions.map(f => {
          const v = rows[key(f)]?.validation;
          return <td key={f}>{v ? `${fmt(v.mean)} (${fmt(v.min, 3)}–${fmt(v.max, 3)})` : 'infinite/blocked'}</td>;
        })}
        <td>{Math.round(rows['1.0']?.training_operations.mean ?? 0).toLocaleString()}</td>
        <td>{fmt(rows['1.0']?.training_cpu_seconds.mean, 3)}</td>
        <td>{fmt(rows['1.0']?.rule_violations_per_1000.mean, 2)}</td>
        {fresh && <td>{fmt(verdicts?.fresh_loss?.[name]?.['1.0'])}</td>}
      </tr>)}</tbody>
    </table>}
    {report.preregistration && <details open={!fresh}>
      <summary>Pre-registered success criteria (frozen {report.preregistration.created_at})</summary>
      <ul>{Object.entries(report.preregistration.criteria).map(([key, text]) => <li key={key}><strong>{key}</strong>: {text}
        {verdicts && verdicts[key] !== undefined && <> — <em>{typeof verdicts[key] === 'object' ? JSON.stringify(verdicts[key]) : yes(verdicts[key])}</em></>}
      </li>)}</ul>
    </details>}
    <p>Operations are formula-based estimates (each multiply-add, table update/read or rule lookup counts as 1).
      Counts-only paths are deterministic, so their seed repeats only change generation sampling.
      Generated text is not scripture or the World English Bible. Lower loss is not understanding.</p>
    {report.examples?.map(item => <details key={item.arm}>
      <summary>{item.arm}: sample (seed 1, 100% data)</summary>
      {item.samples.map(sample => <pre key={sample.prompt} className="generator-sample">{sample.prompt}{sample.text}</pre>)}
    </details>)}
    <details><summary>Exact proof report</summary><pre>{JSON.stringify(report, null, 2)}</pre></details>
  </section>;
}

function OnlyStates({ report }) {
  const rows = {};
  report.results.forEach(item => {
    const key = `${item.arm}${item.penalty === null ? '' : ` (off-path weight ${item.penalty})`}`;
    rows[key] = { ...(rows[key] || {}), [item.fraction]: item.validation_loss };
  });
  const fmt = value => value === null || value === undefined ? 'infinite' : value.toFixed(4);
  return <section className="layered-efficiency">
    <h3>Layer 0 only states (one-way configurations)</h3>
    <p>Status: {report.status}. Earned from Acts 1: {report.discovery.forward.count} forward
      ({report.discovery.forward.mutated} with mutation variants) and {report.discovery.backward.count} backward
      only states, configurations of 2–{report.max_context + 1} characters. Unconfirmed candidates; they guide but never forbid.
      Acts 2 is reused development data, not a fresh test.</p>
    <table>
      <thead><tr><th>Path</th>{report.fractions.map(f => <th key={f}>{f * 100}% of Acts 1</th>)}</tr></thead>
      <tbody>{Object.entries(rows).map(([name, values]) => <tr key={name}>
        <td>{name}</td>{report.fractions.map(f => <td key={f}>{fmt(values[f])}</td>)}
      </tr>)}</tbody>
    </table>
    <details><summary>How often only states held on Acts 2, by configuration length</summary>
      <pre>{JSON.stringify(report.hold_rates_on_validation, null, 2)}</pre></details>
    <details><summary>Exact only-state report</summary><pre>{JSON.stringify(report, null, 2)}</pre></details>
  </section>;
}

function WordOnlyStates({ report }) {
  const rows = {};
  report.results.forEach(item => {
    const name = item.support === null || item.support === undefined ? item.arm
      : `${item.arm} (support ${item.support}, off-path weight ${item.penalty})`;
    rows[name] = { ...(rows[name] || {}), [String(item.fraction)]: item.validation_loss };
  });
  const key = f => String(f).includes('.') ? String(f) : `${f}.0`;
  const fmt = value => value === null || value === undefined ? '—' : value.toFixed(4);
  const words = list => list.map(o => `${o.word}→${o.path}${Object.keys(o.variants).length ? ' (mutated)' : ''}`).join(', ');
  return <section className="layered-efficiency">
    <h3>Word-level only states (one-way and opposite)</h3>
    <p>Status: {report.status}. A word only state says one word has always been followed (forward) or preceded
      (opposite) by exactly one word in Acts 1. Unconfirmed candidates; they guide but never forbid.
      Acts 2 is reused development data, not a fresh test.</p>
    {Object.entries(report.discovery).map(([support, d]) => <p key={support}>Support {support}: {d.forward.count} forward
      ({words(d.forward.strongest.slice(0, 4))}); {d.backward.count} opposite.</p>)}
    <table>
      <thead><tr><th>Path</th>{report.fractions.map(f => <th key={f}>{f * 100}% of Acts 1</th>)}</tr></thead>
      <tbody>{Object.entries(rows).map(([name, values]) => <tr key={name}>
        <td>{name}</td>{report.fractions.map(f => <td key={f}>{fmt(values[key(f)] ?? values[String(f)])}</td>)}
      </tr>)}</tbody>
    </table>
    <details><summary>How often word only states held on Acts 2</summary>
      <pre>{JSON.stringify(report.hold_rates_on_validation, null, 2)}</pre></details>
    <details><summary>Exact word only-state report</summary><pre>{JSON.stringify(report, null, 2)}</pre></details>
  </section>;
}

function FixedPatterns({ report }) {
  const pct = value => value === null || value === undefined ? '—' : `${(value * 100).toFixed(1)}%`;
  const reading = (label, part) => part && <tr key={label}><td>{label}</td>
    <td>{part.discovery.patterns}</td><td>{pct(part.reading.coverage)}</td>
    <td>{pct(part.reading.character_accuracy)}</td><td>{pct(part.reading.decisions_saved_fraction)}</td></tr>;
  const locked = report.locked?.status === 'completed' ? report.locked : null;
  const shown = locked || report.development;
  return <section className="layered-efficiency">
    <h3>Fixed patterns (only chains) for fast reading and generation</h3>
    <p>A fixed pattern is a 1–9 character trigger that, every time it appeared at least {report.support} times in
      approved Acts text, was followed by exactly the same characters. Reading skips the characters a pattern
      supplies; generation writes the whole continuation as one decision. Patterns are earned evidence from one
      book, not proof that a sequence can only exist one way.</p>
    <table>
      <thead><tr><th>Check</th><th>Patterns</th><th>Correctly supplied</th><th>Correct when claimed</th><th>Decisions saved</th></tr></thead>
      <tbody>
        {reading('Development: learn Acts 1, 2, 4–26 → Acts 27', report.development)}
        {reading('Locked once: learn Acts 1, 2, 4–27 → Acts 28', locked)}
      </tbody>
    </table>
    {locked && <p>Pre-registered verdicts: {Object.entries(locked.verdicts).map(([name, v]) =>
      `${name} ${v.met ? 'met' : 'not met'} (${pct(v.value)} vs ${pct(v.target)})`).join('; ')}.</p>}
    {report.preregistration?.revision && <p>Revision: {report.preregistration.revision}</p>}
    {shown && <p>Most useful patterns: {shown.reading.top_patterns.slice(0, 10)
      .map(item => `${item.trigger.replace('\n', '¶')}→${item.continuation}`).join(', ')}</p>}
    {shown && <details><summary>Generation samples (counts plus patterns; not scripture)</summary>
      {shown.generation.map(item => <p key={item.prompt}>{pct(item.decisions_saved_fraction)} decisions saved,
        {' '}{item.macros} patterns used: <code>{item.text.slice(0, 160)}</code></p>)}</details>}
    <details><summary>Exact fixed-pattern report</summary><pre>{JSON.stringify(report, null, 2)}</pre></details>
  </section>;
}

function StrictLayers({ report }) {
  const pct = value => `${(value * 100).toFixed(1)}%`;
  const r = report.strict.reading;
  return <section className="layered-efficiency">
    <h3>Strict layer (if it sees this, it is this) and light pattern layer</h3>
    <p>Strict: a known word, or a phrase seen at least {report.strict.discovery.phrase_support} times, is claimed only after
      the whole unit is confirmed in the text, so it never guesses a word from its start. Light: the earlier
      94–95% patterns only add weight to counts. Development check on Acts {report.held_chapter}; no fresh claim.</p>
    <table><tbody>
      <tr><td>Units learned</td><td>{report.strict.discovery.words} words, {report.strict.discovery.phrases} phrases</td></tr>
      <tr><td>Strict wrong claims</td><td>{r.strict_wrong_claims} of {r.strict_claims} (by construction)</td></tr>
      <tr><td>Letters recognized whole</td><td>{pct(r.letter_coverage)}</td></tr>
      <tr><td>Reading decisions saved</td><td>{pct(r.decisions_saved_fraction)}</td></tr>
      <tr><td>Unknown words handed to other layers</td><td>{r.unknown_words}</td></tr>
    </tbody></table>
    <table>
      <thead><tr><th>Light layer weight</th><th>Loss on Acts {report.held_chapter}</th><th>Pattern guesses right / wrong</th></tr></thead>
      <tbody>{report.light.results.map(row => <tr key={String(row.weight)}><td>{String(row.weight)}</td>
        <td>{row.validation_loss.toFixed(4)}</td><td>{row.fired ? `${row.fired.right} / ${row.fired.wrong || 0}` : '—'}</td></tr>)}</tbody>
    </table>
    <details><summary>Generation samples (word counts plus strict spelling; not scripture)</summary>
      {report.strict.generation.map(item => <p key={item.prompt}>{pct(item.decisions_saved_fraction)} decisions saved,
        {' '}{item.misspelled_words} misspelled: <code>{item.text.slice(0, 160)}</code></p>)}</details>
    <details><summary>Exact strict-layer report</summary><pre>{JSON.stringify(report, null, 2)}</pre></details>
  </section>;
}

function SkeletonLayer({ report }) {
  const rate = value => value.toExponential(1);
  const rows = [['Development books', report.development], ['Locked books (once)', report.locked]]
    .filter(([, run]) => run && run.by_support && run.by_support[String(report.support)]);
  const locked = report.locked && report.locked.verdicts;
  const examples = rows.length ? rows[rows.length - 1][1].by_support[String(report.support)].discovery.strict_examples.hidden1 : [];
  return <section className="layered-efficiency">
    <h3>Strict skeletons (letters hidden or missing, only one word fits)</h3>
    <p>A skeleton such as <code>th*t</code> is strict only when exactly one word in the Layer 0 list of possible words
      (the SCOWL word list, its regular forms, and the learning text) fits it, the word was seen at least {report.support} times,
      it is not a name, and its first letter is visible. Target: wrong at most 1 in 100,000 checks. Bible text only; not general English.</p>
    {locked && <p>Locked verdicts: observed {rate(locked.K1_observed.value)} ({locked.K1_observed.met ? 'met' : 'not met'}),
      95% upper bound {rate(locked.K2_bound.value)} ({locked.K2_bound.met ? 'met' : 'not met'}).</p>}
    <table>
      <thead><tr><th>Run</th><th>Strict checks</th><th>Wrong</th><th>95% upper bound</th><th>Words with a strict skeleton</th></tr></thead>
      <tbody>{rows.map(([name, run]) => {
        const strict = run.by_support[String(report.support)].held_check.strict;
        return <tr key={name}><td>{name}</td><td>{strict.checks}</td><td>{strict.wrong}</td><td>{rate(strict.upper_95)}</td>
          <td>{strict.word_occurrences_with_a_strict_hidden1_skeleton} of {strict.word_occurrences}</td></tr>;
      })}</tbody>
    </table>
    <p>Examples: {examples.slice(0, 10).map(item => `${item.skeleton} → ${item.word}`).join(', ')}</p>
    <details><summary>Exact skeleton report</summary><pre>{JSON.stringify({ criteria: report.criteria, preregistration: report.preregistration, locked: report.locked && report.locked.verdicts }, null, 2)}</pre></details>
  </section>;
}

function IndexPipelineBenchmark({ report }) {
  const layered = report.method === 'candidate_layer_comparison';
  const arms = report.settings.arms || ['baseline', 'separated'];
  const armName = arm => ({ baseline: 'Baseline', separated: 'Remainder base + placement index',
    layers_0: 'L0 + base', layers_2: 'L2 + base (no L0/L1)', layers_01: 'L0 + L1 + base',
    layers_01_quarter: 'L0 + L1 top 25% + base', layers_01_half: 'L0 + L1 top 50% + base',
    layers_01_pretrained: 'L0 + L1 + pretrained base',
    layers_01_injected: 'L0 + L1 + protected injection',
    layers_01_atomic: 'L0 + L1 atomic contextual model',
    layers_01_selective: 'L0 + selective L1 + base', layers_012: 'L0 + L1 + L2 + base' }[arm] || arm);
  return <section className="layered-efficiency">
    <h3>{layered ? 'Candidate-layer cascade comparison' : 'Separate index and remainder-only training'}</h3>
    <p>{report.status}; {report.settings.seeds.length} paired seeds;
      {' '}{report.source.training_characters} training characters; {report.source.evaluation_characters} evaluation
      characters; {report.settings.probes} common next-word probes.
      {' '}{report.source.corpus ? report.source.corpus.title : 'Reused Acts development data'}.
      {' '}Development experiment, not a locked test.</p>
    {report.source.corpus && <p>{report.source.corpus.provenance} {report.source.corpus.limitations}</p>}
    {report.source.whole_acts && <p><strong>IN-SAMPLE: all 28 chapters are training data.</strong>
      {' '}{report.source.whole_acts.evaluation_scope}</p>}
    {report.opportunity_check && <><p>{report.opportunity_check.focused_probes} explicit L1 target opportunities
      across {report.opportunity_check.distinct_l1_words} distinct words. {report.opportunity_check.policy}</p>
      <table><caption>Four-pass L1-focused learning checks (training text, not generalization)</caption>
        <thead><tr><th>Model / seed</th><th>Correct / opportunities</th><th>Edit distance</th></tr></thead>
        <tbody>{report.runs.map(run => {
          const row = run.checkpoints.find(item => item.passes === 4).l1_focused;
          return <tr key={`${run.seed}-${run.arm}`}><td>{armName(run.arm)} / {run.seed}</td>
            <td>{row.correct} / {row.probes}</td><td>{row.normalized_edit_distance.toFixed(3)}</td></tr>;
        })}</tbody></table></>}
    {report.unique_l1 && <><p>Corrected L1: {Object.keys(report.unique_l1.forms).length} complete forms
      identified by shortest unique fragments against {report.unique_l1.inventory.entries.toLocaleString()}
      {' '}comparison entries plus training words. Whole-word candidates replaced; no fallback or Layer 2.
      {' '}{report.unique_l1.policy}</p>
      <p>{Object.entries(report.unique_l1.forms).map(([word, evidence]) =>
        `${evidence.key} → ${word}`).join('; ') || 'No forms qualified; corrected L01 equals L0.'}</p></>}
    {report.pretraining && <><p>L0/L1-only pretraining: {report.pretraining.passes} passes over
      {' '}{report.pretraining.patterns.length} isolated patterns, then restore the saved checkpoint.
      {' '}{report.pretraining.policy}</p>
      <table><caption>Pretraining and retention (teacher-forced spelling, not free recall)</caption>
        <thead><tr><th>Model / seed</th><th>Pretraining CPU</th><th>Before</th><th>After pretraining</th><th>After 4 passage passes</th></tr></thead>
        <tbody>{report.runs.filter(run => run.pretraining).map(run => <tr key={`${run.seed}-${run.arm}`}>
          <td>{armName(run.arm)} / {run.seed}</td><td>{run.pretraining.cpu_seconds.toFixed(3)}</td>
          <td>{(100 * run.pretraining.before.teacher_forced_character_accuracy).toFixed(1)}%</td>
          <td>{(100 * run.pretraining.after.teacher_forced_character_accuracy).toFixed(1)}%</td>
          <td>{(100 * run.checkpoints.find(row => row.passes === 4).pattern_recall.teacher_forced_character_accuracy).toFixed(1)}%</td>
        </tr>)}</tbody></table>
      <p>Pattern literals: {report.pretraining.patterns.map(pattern => JSON.stringify(pattern)).join(', ')}.
        Identical starting contexts can have different targets; 100% teacher-forced accuracy may be impossible.
        Base updates below count passage training only; total CPU includes pretraining.</p></>}
    {report.injection && <><p>Protected reference injection: {report.injection.policy}</p>
      <p>Active-model recall above is measured without injection, not frozen-bank recall.
        Identical per-seed pretraining checkpoints are reused across the two pretrained arms;
        their measured pretraining cost is charged to both.
        Injection feature construction is included in total training CPU.</p></>}
    {report.reinforcement && <><p>Matched pretraining and occurrence reinforcement:
      {' '}{report.reinforcement.policy} {report.reinforcement.occurrences_per_pass} refreshes /
      {' '}{report.reinforcement.updates_per_pass} extra character updates per pass in each arm.</p>
      <table><caption>Four-pass replay cost (already included in total training CPU)</caption>
        <thead><tr><th>Model / seed</th><th>Replay CPU</th><th>Replay updates</th><th>All updates including pretraining</th></tr></thead>
        <tbody>{report.runs.map(run => {
          const row = run.checkpoints.find(item => item.passes === 4);
          return <tr key={`${run.seed}-${run.arm}`}><td>{armName(run.arm)} / {run.seed}</td>
            <td>{row.reinforcement_cpu_seconds.toFixed(3)}</td><td>{row.reinforcement_updates}</td>
            <td>{row.total_updates_including_pretraining}</td></tr>;
        })}</tbody></table></>}
    {report.atomic && <><p>Atomic contextual comparison: {report.atomic.policy}</p>
      <table><caption>Four-pass reference usage and surrounding predictions (pooled probe counts)</caption>
        <thead><tr><th>Model</th><th>Outputs / parameters</th><th>Correct L1 selections / emissions</th>
          <th>L1 target probes</th><th>Recent-reference word accuracy</th></tr></thead>
        <tbody>{arms.map(arm => {
          const entries = report.runs.filter(run => run.arm === arm);
          const scores = entries.flatMap(run => run.checkpoints.filter(row => row.passes === 4))
            .map(row => row.selection_metrics);
          const sum = key => scores.reduce((total, score) => total + score[key], 0);
          const recent = sum('recent_reference_probes');
          const correct = scores.reduce((total, score) =>
            total + (score.recent_reference_word_accuracy || 0) * score.recent_reference_probes, 0);
          return <tr key={arm}><td>{armName(arm)}</td>
            <td>{entries[0] ? `${entries[0].vocabulary} / ${entries[0].parameters}` : 'Pending'}</td>
            <td>{sum('correct_l1_first_emissions')} / {sum('l1_first_emissions')}</td>
            <td>{sum('eligible_l1_target_probes')}</td>
            <td>{recent ? `${(100 * correct / recent).toFixed(2)}% (${recent} probes)` : 'No probes'}</td></tr>;
        })}</tbody></table></>}
    {report.selections && Object.entries(report.selections).map(([arm, selection]) =>
      <p key={arm}>{armName(arm)}: {Object.values(selection.selective_l1).filter(item => item.selected).length}
        {' '}selected of {Object.values(selection.selective_l1).filter(item => item.occurrences >= 2).length}
        {' '}repeated forms ({Object.keys(selection.selective_l1).length} bank candidates).
        {' '}{selection.selection_policy}</p>)}
    {report.selection && <p>Selective L1 accepted{' '}
      {Object.values(report.selection.selective_l1).filter(item => item.selected).length} of{' '}
      {Object.keys(report.selection.selective_l1).length} eligible forms.
      {' '}{report.selection.selection_policy}
      {!Object.values(report.selection.selective_l1).some(item => item.selected) &&
        ' No L1 forms qualified: this arm reduces to L0 alone, not an intermediate extraction level.'}</p>}
    {!layered && <p>{report.extraction.indexed_forms} bank-backed forms; {report.extraction.occurrences} extracted occurrences;
      {' '}{report.extraction.removed_characters} characters removed from base training;
      {' '}{report.extraction.remainder_characters} characters remain.
      No indexed outputs or embeddings are added to the base.</p>}
    {layered && <><p>{report.settings.l2_only ?
      'Standalone L2: repeated 3–4-character blocks and hidden-backed complete forms, discovered directly from training. L0 and L1 are disabled.' :
      report.settings.atomic_l01 ? 'Same corrected L0/L1 recognition in both layered arms. Atomic model learns references in context; remainder control removes them. No Layer 2.' :
      report.settings.unique_l1 ? 'Layer 0 unchanged; Layer 1 uses only complete forms identified by unique partial keys. Base learns none of an extracted form’s characters. No Layer 2.' :
      report.settings.savings_l1 ? 'Layer 0 unchanged; L1 selects the top 25% and 50% of repeated forms by training-only character savings, alongside L0 and full L1 controls. No Layer 2.' :
      report.settings.selective_l1 ?
      'Layer 0 unchanged; selective L1 uses a fixed training-only context reliability and savings gate, compared with full L1. No Layer 2 in this run.' :
      'Layer 0 unchanged; Layer 1 whole-form candidates; Layer 2 repeated 3–4-character blocks and hidden-backed complete matches.'}
      {' '}Base remains separate. No old generation exclusions enabled.
      Candidates are experimental, not foundational admissions.</p>
      <table><thead><tr><th>Arm</th><th>Removed characters</th><th>Remainder</th><th>Matched kinds</th></tr></thead>
        <tbody>{Object.entries(report.layer_extractions).map(([arm, row]) => <tr key={arm}>
          <td>{armName(arm)}</td><td>{row.removed_characters}</td><td>{row.remainder_characters}</td>
          <td>{JSON.stringify(row.by_kind)}</td></tr>)}</tbody></table></>}
    {report.summary && <table>
      <thead><tr><th>Passes</th><th>Model</th><th>Total training CPU</th><th>Base CPU</th><th>Index CPU</th>
        <th>Base updates</th><th>Word accuracy</th><th>Word edit distance (lower better)</th></tr></thead>
      <tbody>{Object.entries(report.summary).flatMap(([budget, arms]) => Object.entries(arms).map(([arm, row]) =>
        <tr key={`${budget}-${arm}`}><td>{budget}</td><td>{armName(arm)}</td>
          <td>{row.total_training_cpu_seconds.toFixed(3)}</td><td>{row.neural_cpu_seconds.toFixed(3)}</td>
          <td>{row.selector_cpu_seconds.toFixed(3)}</td><td>{row.updates}</td>
          <td>{(100 * row.exact_next_word_accuracy).toFixed(2)}%</td>
          <td>{row.normalized_edit_distance.toFixed(3)}</td></tr>))}</tbody>
    </table>}
    {report.status === 'completed' && <table>
      <caption>Generation after four passes (120 characters/sample; not a coherence test)</caption>
      <thead><tr><th>Model</th><th>Mean CPU seconds/sample</th><th>Neural calls</th><th>Index lookups</th>
        <th>Indexed emissions</th>{report.injection && <th>Conditioned calls</th>}</tr></thead>
      <tbody>{arms.map(arm => {
        const samples = report.runs.filter(run => run.arm === arm).flatMap(run => run.samples);
        const sum = key => samples.reduce((total, sample) => total + (sample[key] || 0), 0);
        return <tr key={arm}><td>{armName(arm)}</td><td>{(sum('cpu_seconds') / samples.length).toFixed(4)}</td>
          <td>{sum('neural_calls')}</td><td>{sum('index_lookups')}</td><td>{sum('indexed_emissions')}</td>
          {report.injection && <td>{sum('conditioned_neural_calls')}</td>}</tr>;
      })}</tbody>
    </table>}
    <p>Training CPU includes both base updates and placement-count learning; setup/extraction is separate
      ({report.setup_cpu_seconds.toFixed(3)} CPU seconds). Prior bank construction is excluded.
      {report.atomic ? ' The atomic arm learns both references and characters in context without a separate placement index; the remainder control retains separate placement.' :
        ' The base uses only remainder characters as context. The separate index uses preceding stream events, including references, to learn placement.'}
      Recognition runs ahead, not as concurrent workers.
      {report.injection && ' The protected-injection arm additionally conditions base hidden state on the latest reference vector.'}
      No comparable all-character loss or equal-update claim is made.</p>
    <details><summary>Protocol, placement policy and saved samples (not scripture)</summary>
      <pre>{JSON.stringify(report, null, 2)}</pre></details>
  </section>;
}

function UnitBenchmark({ report }) {
  if (['separate_index_remainder', 'candidate_layer_comparison'].includes(report.method))
    return <IndexPipelineBenchmark report={report} />;
  const corrected = report.method === 'character_recognition_onlies';
  const indexed = report.method === 'indexed_onlies';
  const title = report.source_run_id ? 'Baseline vs recognition vs combined' :
    indexed ? 'Atomic indexed-onlies comparison' :
      corrected ? 'Character-only recognition comparison' : 'Legacy expanded-output comparison';
  const armName = arm => ({ characters: 'Baseline', skeleton_units: indexed ? 'Base + indexed onlies' :
    corrected ? 'Recognition (skip spelling)' : 'Legacy units',
    baseline_plus_rules: 'Baseline + rules (all updates)' }[arm] || arm);
  const updateBudget = `updates-${report.settings?.matched_updates || 12000}`;
  const pct = value => `${(100 * value).toFixed(1)}%`;
  if (!report.summary) return <section className="layered-efficiency">
    <h3>{title}</h3>
    <p>Status: {report.status}; {report.runs.length} arms completed. Development data only.</p>
  </section>;
  return <section className="layered-efficiency">
    <h3>{title}</h3>
    <p>{report.settings.seeds.length} paired seeds; {report.source.training_characters.toLocaleString()} training characters;
      {' '}{report.source.evaluation_characters.toLocaleString()} evaluation characters;
      {' '}{report.settings.probes} shared next-word questions.
      Development only: the pattern bank already saw these Bible chapters.
      No character-versus-unit token-loss comparison is used.</p>
    <table>
      <thead><tr><th>Budget</th><th>Model</th><th>CPU seconds</th><th>Wall seconds</th>
        <th>Estimated update work</th><th>Next-word accuracy</th><th>Word edit distance (lower better)</th>
        <th>All-character probe loss</th></tr></thead>
      <tbody>{['passes-4', updateBudget].flatMap(budget =>
        Object.entries(report.summary[budget].arms).map(([arm, row]) =>
          <tr key={`${budget}-${arm}`}><td>{budget}</td><td>{armName(arm)}</td>
            <td>{row.training_cpu_seconds.toFixed(2)}</td><td>{row.training_wall_seconds.toFixed(2)}</td>
            <td>{(row.estimated_update_operations / 1e6).toFixed(1)}M</td>
            <td>{pct(row.exact_next_word_accuracy)}</td><td>{row.normalized_edit_distance.toFixed(3)}</td>
            <td>{row.all_character_probe_loss?.toFixed(3) ?? 'not measured'}</td></tr>))}</tbody>
    </table>
    <p>{corrected ? 'Recognition-arm input decisions saved' : 'Input decisions saved'}: {pct(report.reading.decisions_saved_fraction)}.
      {' '}{corrected ? 'Recognition vs baseline equal-pass CPU ratio' : 'Equal-pass CPU ratio'}: {report.summary['passes-4'].cpu_ratio.toFixed(2)}× baseline.
      {' '}Less-effort criterion: {report.summary.verdicts.less_effort ? 'met' : 'not met'}.
      {' '}Loose similar-quality criterion: {report.summary.verdicts.similar_quality ? 'met' : 'not met'},
      but this loose criterion alone is not evidence of useful quality or coherence.</p>
    {indexed ? <p>Each indexed pattern occupies one context position and is emitted with one neural selection.
      The base learns pattern placement, not its internal spelling. Vocabulary and parameter counts differ,
      so character-versus-unit loss is not reported as comparable.
      This bounded test uses up to 128 bank-backed patterns found in training, not arbitrary dictionary entries.
      {' '}{report.units} patterns indexed.
      {' '}{report.runs.filter(run => run.seed === report.settings.seeds[0]).map(run =>
        `${armName(run.arm)}: ${run.vocabulary} outputs, ${run.parameters} parameters`).join('; ')}.</p> :
      <p>All-character probe loss scores the same original positions, including skipped spelling, without
        generation rules. It tests neural character prediction, not the combined rule-assisted decoder.</p>}
    {report.combined_note && <p>{report.combined_note}</p>}
    {report.summary['passes-4'].arms.skeleton_units.all_character_probe_loss >
      report.summary['passes-4'].arms.characters.all_character_probe_loss &&
      <p>The recognition trainer has worse all-character prediction loss. Lower update cost does not
        establish equal overall prediction quality.</p>}
    <p>Training timings exclude evaluation. Setup is recorded separately; prior bank construction is excluded.
      {corrected ? ' All arms use identical character vocabularies, parameter counts and character context. The recognition arm skips known interior spelling updates; baseline plus rules retains every baseline update.'
        : indexed ? ' Atomic references add output classes and lengthen effective text context. Recognition runs ahead of training; concurrent-worker throughput is not measured.'
          : ' The legacy unit model has more parameters and a longer effective text context; it is not the intended recognition-only method.'}
      No ordinary-tokenizer control was run.</p>
    <details><summary>All seeds, checkpoints and generated samples (not scripture)</summary>
      <pre>{JSON.stringify(report.runs, null, 2)}</pre></details>
    {report.matched_generation && <details>
      <summary>Equal-update generated sample pairs (not scripture)</summary>
      <p>{report.sample_budget_note}</p>
      <pre>{JSON.stringify(report.matched_generation, null, 2)}</pre>
    </details>}
    {(report.source_run_id || indexed) && report.matched_generation && <table>
      <caption>Fresh equal-update generation samples (up to 120 characters each; not a coherence test)</caption>
      <thead><tr><th>Model</th><th>Mean CPU seconds/sample</th><th>{indexed ? 'Indexed emissions' : 'Rule matches'}</th>
        <th>{indexed ? 'Emitted characters' : 'Forced characters'}</th>{indexed && <th>Neural decisions</th>}</tr></thead>
      <tbody>{['characters', 'skeleton_units', 'baseline_plus_rules'].map(arm => {
        const samples = report.matched_generation.runs.filter(run => run.arm === arm).flatMap(run => run.samples);
        if (!samples.length) return null;
        return <tr key={arm}><td>{armName(arm)}</td>
          <td>{(samples.reduce((sum, sample) => sum + sample.cpu_seconds, 0) / samples.length).toFixed(4)}</td>
          <td>{samples.reduce((sum, sample) => sum + sample.unit_occurrences, 0)}</td>
          <td>{samples.reduce((sum, sample) => sum +
            (indexed ? sample.text.length : sample.forced_characters || 0), 0)}</td>
          {indexed && <td>{samples.reduce((sum, sample) => sum + sample.decisions, 0)}</td>}</tr>;
      })}</tbody>
    </table>}
  </section>;
}

function NeuralPanel() {
  const [report, setReport] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [rateComparison, setRateComparison] = useState(null);
  const [layered, setLayered] = useState(null);
  const [proof, setProof] = useState(null);
  const [onlies, setOnlies] = useState(null);
  const [wordOnlies, setWordOnlies] = useState(null);
  const [patterns, setPatterns] = useState(null);
  const [strict, setStrict] = useState(null);
  const [skeletonReport, setSkeletonReport] = useState(null);
  const [runtimeBank, setRuntimeBank] = useState(null);
  const [unitBenchmark, setUnitBenchmark] = useState(null);
  const [useUnits, setUseUnits] = useState(false);
  const [generationPrompt, setGenerationPrompt] = useState('');
  const [liveGeneration, setLiveGeneration] = useState(null);
  const [trainingText, setTrainingText] = useState('');
  const [heldOutText, setHeldOutText] = useState('');
  const [source, setSource] = useState('');
  const [approved, setApproved] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    fetch('/api/neural').then(async response => {
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      setReport(data.report);
      setComparison(data.context_comparison);
      setRateComparison(data.rate_comparison);
      setLayered(data.layered_comparison);
      setProof(data.proof_comparison);
      setOnlies(data.onlies_comparison);
      setWordOnlies(data.word_onlies_comparison);
      setPatterns(data.patterns_comparison);
      setStrict(data.strict_comparison);
      setSkeletonReport(data.skeletons_comparison);
      setRuntimeBank(data.skeleton_runtime);
      setUnitBenchmark(data.unit_benchmark);
    }).catch(err => setError(err.message));
  }, []);

  const train = async (synthetic, automated = false) => {
    setBusy(true);
    setError('');
    try {
      const response = await fetch('/api/neural/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          synthetic, automated_synthetic: automated, training_text: trainingText, held_out_text: heldOutText,
          source, acknowledge_review: approved, skeleton_units: !synthetic && useUnits
        })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      setReport(data);
      setLiveGeneration(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const generate = async () => {
    setBusy(true);
    setError('');
    try {
      const response = await fetch('/api/neural/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ run_id: report.run_id, prompt: generationPrompt })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      setLiveGeneration(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    if (![report, comparison, rateComparison, layered, proof, unitBenchmark].some(item => item?.status === 'running')) return;
    const timer = setInterval(() => {
      fetch('/api/neural').then(async response => {
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
        setReport(data.report);
        setComparison(data.context_comparison);
        setRateComparison(data.rate_comparison);
        setLayered(data.layered_comparison);
        setProof(data.proof_comparison);
        setOnlies(data.onlies_comparison);
        setWordOnlies(data.word_onlies_comparison);
        setPatterns(data.patterns_comparison);
        setStrict(data.strict_comparison);
        setSkeletonReport(data.skeletons_comparison);
        setUnitBenchmark(data.unit_benchmark);
      }).catch(err => setError(err.message));
    }, 3000);
    return () => clearInterval(timer);
  }, [report?.status, comparison?.status, rateComparison?.status, layered?.status, proof?.status, unitBenchmark?.status]);

  return (
    <section className="hero-card generator-panel neural-panel" style={{ marginTop: 24 }}>
      <h2>Neural weight lab</h2>
      <p>Tiny from-scratch character MLP with optional only-state recognition and spelling shortcuts.
        UI training starts fresh; saved checkpoints can generate without retraining.
        This does not replace chat, import memories, or update the old generator.</p>
      <button disabled={busy} onClick={() => train(true)}>Run synthetic learning check</button>
      <button disabled={busy} onClick={() => train(true, true)}>Run 50 synthetic rounds (5,000 updates)</button>
      <form className="generator-form" onSubmit={event => { event.preventDefault(); train(false); }}>
        <label>New training text (8–2000 characters)
          <textarea required minLength={8} maxLength={2000} value={trainingText} onChange={event => setTrainingText(event.target.value)} />
        </label>
        <label>Separate held-out text (8–1000 characters; never used for weight updates)
          <textarea required minLength={8} maxLength={1000} value={heldOutText} onChange={event => setHeldOutText(event.target.value)} />
        </label>
        <label>Provenance and rights note
          <input required maxLength={500} value={source} onChange={event => setSource(event.target.value)} />
        </label>
        <label><input type="checkbox" checked={useUnits} disabled={busy || !runtimeBank?.available}
          onChange={event => setUseUnits(event.target.checked)} />
          Use expanded only-state recognition (character vocabulary unchanged)
        </label>
        <p>{runtimeBank === null ? 'Loading skeleton bank status…' : runtimeBank.available
          ? 'Bank ready: absorb exact recognized forms, skip their interior spelling updates, and preserve original character context. Generation shortcuts require a closed-list unique prefix; ambiguous fragments stay character predictions.'
          : `Bank unavailable. Build locally with ${runtimeBank.build_command}, then reload.`}
          {' '}Both modes remain character-only. Unskipped-target loss is not directly comparable to all-character loss.</p>
        <label><input type="checkbox" required checked={approved} onChange={event => setApproved(event.target.checked)} />
          I have reviewed these texts and authorize their use in this neural experiment.
          This declaration is not legal clearance. Full texts and weights are saved locally.
        </label>
        <button disabled={busy || !approved} type="submit">{busy ? 'Training…' : 'Train separate neural experiment'}</button>
      </form>
      {error && <p className="banner banner-error">{error}</p>}
      {unitBenchmark && <UnitBenchmark report={unitBenchmark} />}
      {skeletonReport && <SkeletonLayer report={skeletonReport} />}
      {strict && <StrictLayers report={strict} />}
      {patterns && <FixedPatterns report={patterns} />}
      {wordOnlies && <WordOnlyStates report={wordOnlies} />}
      {onlies && <OnlyStates report={onlies} />}
      {proof && proof.status !== 'absent' && <LayerProof report={proof} />}
      {layered && <LayeredEfficiency report={layered} />}
      {rateComparison && <NeuralComparison comparison={rateComparison} rates />}
      {comparison && <NeuralComparison comparison={comparison} />}
      {report && <>
        <h3>{report.synthetic ? 'Synthetic mechanics check' : 'Approved-text experiment'} progress</h3>
        {(report.automated_rounds || report.automated_synthetic) && <p>
          Status: {report.status}. Rounds: {report.rounds_completed}/{report.rounds_planned}.
          {' '}Final test loss: {report.final_test_loss?.toFixed(4) ?? 'not evaluated in this run'}.
          {report.parent_run_id && ` Continued from ${report.parent_run_id}; prior test is not reused as a fresh evaluation.`}
          {' '}Layer 2 counts built once from training text; no generated text is used for training.
          {report.error && ` Run error: ${report.error}`}
        </p>}
        <p>{report.trainable_parameters.toLocaleString()} trainable parameters;
          {' '}{report.changed_parameters.toLocaleString()} changed;
          {' '}{report.steps} steps / {report.processed_target_tokens} target tokens processed.</p>
        <p>{report.training_characters} training characters; {report.held_out_characters} held-out;
          {' '}{report.held_out_unknown_characters} held-out characters unknown to the training vocabulary.</p>
        <table>
          <thead><tr><th>Step</th><th>Training loss</th><th>Held-out loss</th></tr></thead>
          <tbody>{report.history.map(row => <tr key={row.step}>
            <td>{row.step}</td><td>{row.training_loss.toFixed(4)}</td><td>{row.held_out_loss.toFixed(4)}</td>
          </tr>)}</tbody>
        </table>
        <p>{report.evaluation_note} Loss is mean negative log probability per {report.loss_unit || 'character'}; lower is better within the same tokenization.</p>
        {report.unit_runtime && <section>
          <h3>{report.model === 'original-character-onlies-mlp-v1' ? 'Character-only recognition use' : 'Legacy unit-token use'}</h3>
          <p>{report.unit_runtime.units} available recognized forms.
            {' '}{report.unit_runtime.training.decisions_saved} skipped internal spelling targets
            ({(100 * report.unit_runtime.training.decisions_saved_fraction).toFixed(1)}%).
            {' '}{report.unit_runtime.training_targets_per_pass} update targets per source pass;
            actual updates remain {report.steps}.</p>
          <p>Generated sample: {report.unit_runtime.generation.unit_occurrences} pattern shortcuts,
            {' '}{report.unit_runtime.generation.forced_characters ?? 0} rule-supplied characters,
            {' '}{report.unit_runtime.generation.character_steps} character emissions.
            Fewer decisions do not prove a wall-clock speedup, understanding, or correct responses.</p>
          <details><summary>Runtime provenance and counts</summary><pre>{JSON.stringify(report.unit_runtime, null, 2)}</pre></details>
        </section>}
        <p>Continuation after <code>{JSON.stringify(report.sample_prompt)}</code>:</p>
        <p>Before training:</p>
        <pre className="generator-sample">{report.initial_sample}</pre>
        <p>After training:</p>
        <pre className="generator-sample">{report.sample}</pre>
        {['original-character-mlp-v1', 'original-unit-character-mlp-v1', 'original-character-onlies-mlp-v1'].includes(report.model) && <form
          className="generator-form" onSubmit={event => { event.preventDefault(); generate(); }}>
          <label>Prompt for saved neural checkpoint
            <input maxLength={1000} value={generationPrompt} onChange={event => setGenerationPrompt(event.target.value)} />
          </label>
          <button type="submit" disabled={busy}>Generate from saved neural checkpoint</button>
          {liveGeneration && <div>
            <pre className="generator-sample">{liveGeneration.text || '(model chose end of output)'}</pre>
            <p>{liveGeneration.decisions} neural decisions; {liveGeneration.unit_occurrences} pattern shortcuts.
              {' '}{liveGeneration.note}</p>
          </div>}
        </form>}
        {report.fixed_prompt_samples && <details>
          <summary>Fixed-prompt samples every 10 rounds</summary>
          <pre>{JSON.stringify(report.fixed_prompt_samples, null, 2)}</pre>
        </details>}
        <details><summary>Exact progress report</summary><pre>{JSON.stringify(report, null, 2)}</pre></details>
      </>}
    </section>
  );
}

function GeneratorPanel() {
  const [status, setStatus] = useState(null);
  const [title, setTitle] = useState('');
  const [source, setSource] = useState('Me');
  const [rightsNote, setRightsNote] = useState('');
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [evaluation, setEvaluation] = useState({ prompts: [] });
  const [evaluationPrompt, setEvaluationPrompt] = useState('');
  const [evaluationLoading, setEvaluationLoading] = useState(false);
  const [evaluationError, setEvaluationError] = useState('');
  const [editingRun, setEditingRun] = useState(null);
  const [editText, setEditText] = useState('');

  const refreshStatus = async () => {
    const response = await fetch('/api/generator');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    setStatus(data);
  };

  const refreshEvaluations = async () => {
    const response = await fetch('/api/generator/evaluations');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
    setEvaluation(data);
  };

  useEffect(() => {
    refreshStatus().catch((err) => setError(err.message));
    refreshEvaluations().catch((err) => setEvaluationError(err.message));
  }, []);

  const addDocument = async (event) => {
    event.preventDefault();
    if (loading) return;
    setLoading(true);
    setError('');
    setNotice('');
    try {
      const response = await fetch('/api/generator/documents', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title: title.trim(),
          source: source.trim() || 'Me',
          rights_note: rightsNote,
          text,
          acknowledge_review: true
        })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      setStatus(data);
      await refreshEvaluations();
      setTitle('');
      setSource('Me');
      setRightsNote('');
      setText('');
      setNotice(`Local model updated to generation ${data.generation}.`);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const removeDocument = async (documentId) => {
    if (loading) return;
    setLoading(true);
    setError('');
    setNotice('');
    try {
      const response = await fetch(`/api/generator/documents/${encodeURIComponent(documentId)}`, { method: 'DELETE' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      setStatus(data);
      await refreshEvaluations();
      setNotice(`Local model rebuilt as generation ${data.generation}.`);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const trackEvaluationPrompt = async (event) => {
    event.preventDefault();
    if (evaluationLoading) return;
    setEvaluationLoading(true);
    setEvaluationError('');
    try {
      const response = await fetch('/api/generator/evaluations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: evaluationPrompt })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      setEvaluation(data);
      setEvaluationPrompt('');
    } catch (err) {
      setEvaluationError(err.message);
    } finally {
      setEvaluationLoading(false);
    }
  };

  const saveEvaluationRating = async (promptId, generation, rating, correction = '') => {
    if (evaluationLoading) return;
    setEvaluationLoading(true);
    setEvaluationError('');
    try {
      const response = await fetch(`/api/generator/evaluations/${encodeURIComponent(promptId)}/${generation}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rating, correction })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      setEvaluation(data);
      setEditingRun(null);
      setEditText('');
    } catch (err) {
      setEvaluationError(err.message);
    } finally {
      setEvaluationLoading(false);
    }
  };

  const beginEditingRun = (promptId, generation, text) => {
    setEditingRun({ promptId, generation });
    setEditText(text);
  };

  return (
    <section className="hero-card generator-panel" style={{ marginTop: 24 }}>
      <div>
        <p className="eyebrow">From-scratch language experiment</p>
        <h2>Generator corpus</h2>
      </div>

      <div className="generator-status" aria-live="polite">
        {status ? (
          <>
            <span>{status.ready ? 'Ready to sample' : 'No approved text yet'}</span>
            <span>{status.document_count} documents</span>
            <span>{status.training_characters.toLocaleString()} characters</span>
            <span>Generation {status.generation}</span>
            <span>{status.evaluation_prompt_count} fixed prompts</span>
            <span>{status.evaluation_sample_count} samples</span>
          </>
        ) : <span>Loading local model status...</span>}
      </div>

      {status?.seed_extraction && (
        <p className="message-meta">
          Seed extraction alongside training: {status.seed_extraction.counts.alive} alive,
          {' '}{status.seed_extraction.counts.dead} dead, {status.seed_extraction.counts.maybe} review occurrences.
          {' '}{status.seed_extraction.unprocessed_document_count} older sources have not been extracted.
          {' '}Patterns are not automatically approved; the generator still trains on your approved text unchanged.
        </p>
      )}
      <form className="generator-form" onSubmit={addDocument}>
        <label>
          Document title (optional)
          <input value={title} onChange={(event) => setTitle(event.target.value)} maxLength={160} />
        </label>
        <label>
          Source / provenance
          <input value={source} onChange={(event) => setSource(event.target.value)} maxLength={500} />
        </label>
        <label className="generator-full-width">
          Rights and provenance note (optional)
          <input value={rightsNote} onChange={(event) => setRightsNote(event.target.value)} maxLength={1000} placeholder="Additional source or rights details" />
        </label>
        <label className="generator-full-width">
          Text to add to local training
          <textarea value={text} onChange={(event) => setText(event.target.value)} maxLength={50000} rows={7} required />
        </label>
        <button className="primary generator-full-width" type="submit" disabled={loading || !text.trim()}>
          {loading ? 'Updating…' : 'Approve text and train'}
        </button>
      </form>

      {error && <p className="banner banner-error">{error}</p>}
      {notice && <p className="banner banner-success">{notice}</p>}

      <section className="generator-evaluations">
        <div className="generator-section-heading">
          <h3>Progress checks</h3>
          <span>Compare the same prompts across model generations.</span>
        </div>
        <form className="evaluation-prompt-form" onSubmit={trackEvaluationPrompt}>
          <label htmlFor="generator-evaluation-prompt">Fixed prompt</label>
          <textarea
            id="generator-evaluation-prompt"
            value={evaluationPrompt}
            onChange={(event) => setEvaluationPrompt(event.target.value)}
            maxLength={1000}
            rows={2}
            required
          />
          <button type="submit" disabled={evaluationLoading || !evaluationPrompt.trim()}>
            {evaluationLoading ? 'Saving…' : 'Track prompt'}
          </button>
        </form>
        <p className="evaluation-note">Samples are repeatable within a generation. Ratings and edits are saved for comparison, not used as training text.</p>
        {evaluationError && <p className="banner banner-error">{evaluationError}</p>}
        {evaluation.prompts?.length ? evaluation.prompts.map((prompt) => (
          <div className="evaluation-prompt" key={prompt.id}>
            <h4>{prompt.prompt}</h4>
            <div className="evaluation-runs">
              {prompt.runs.map((run) => {
                const isEditing = editingRun?.promptId === prompt.id && editingRun?.generation === run.generation;
                return (
                  <article className="evaluation-run" key={run.generation}>
                    <div className="evaluation-run-heading">
                      <strong>Generation {run.generation}</strong>
                      {run.rating && <span className={`evaluation-rating-label ${run.rating}`}>{run.rating === 'edit' ? 'Edited' : run.rating === 'keep' ? 'Kept' : 'Rejected'}</span>}
                    </div>
                    <p className="evaluation-sample">{run.text || 'No continuation was generated for this prompt.'}</p>
                    <small className="evaluation-sources">Sources: {run.training_sources.map((sourceItem) => sourceItem.title).join(', ') || 'none'}</small>
                    {run.rating === 'edit' && !isEditing && <div className="evaluation-correction"><strong>Your edit</strong><p>{run.correction}</p></div>}
                    {isEditing ? (
                      <div className="evaluation-edit">
                        <label htmlFor={`evaluation-edit-${prompt.id}-${run.generation}`}>Corrected version</label>
                        <textarea
                          id={`evaluation-edit-${prompt.id}-${run.generation}`}
                          value={editText}
                          onChange={(event) => setEditText(event.target.value)}
                          maxLength={2000}
                          rows={3}
                        />
                        <div className="evaluation-actions">
                          <button type="button" onClick={() => saveEvaluationRating(prompt.id, run.generation, 'edit', editText)} disabled={evaluationLoading || !editText.trim()}>Save edit</button>
                          <button type="button" onClick={() => { setEditingRun(null); setEditText(''); }}>Cancel</button>
                        </div>
                      </div>
                    ) : (
                      <div className="evaluation-actions">
                        <button type="button" onClick={() => saveEvaluationRating(prompt.id, run.generation, 'keep')} disabled={evaluationLoading}>Keep</button>
                        <button type="button" onClick={() => beginEditingRun(prompt.id, run.generation, run.correction || '')} disabled={evaluationLoading}>Edit</button>
                        <button type="button" onClick={() => saveEvaluationRating(prompt.id, run.generation, 'reject')} disabled={evaluationLoading}>Reject</button>
                      </div>
                    )}
                  </article>
                );
              })}
              {!prompt.runs.length && <p className="generator-empty">A sample will be captured after text is approved.</p>}
            </div>
          </div>
        )) : <p className="generator-empty">No fixed prompts tracked yet.</p>}
      </section>

      <div className="generator-documents">
        <h3>Approved sources</h3>
        {status?.documents?.length ? status.documents.map((document) => (
          <div className="generator-document" key={document.id}>
            <div>
              <strong>{document.title}</strong>
              <span>{document.source} · {document.characters.toLocaleString()} chars · SHA-256 {document.sha256.slice(0, 12)}</span>
              <small>{document.rights_note || 'No rights note recorded.'} Review recorded; legal clearance not asserted.</small>
            </div>
            <button type="button" className="remove-document" onClick={() => removeDocument(document.id)} disabled={loading} aria-label={`Remove ${document.title} from training corpus`} title="Remove and rebuild model">Remove</button>
          </div>
        )) : <p className="generator-empty">No text has been approved for this generator.</p>}
      </div>
    </section>
  );
}
