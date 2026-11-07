import { useEffect, useRef, useState } from 'react';

export default function App() {
  const [activeTab, setActiveTab] = useState(() => new URLSearchParams(window.location.search).get('view') === 'generation' ? 'generation' : 'chat');
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
        <button className={activeTab === 'graph' ? 'tab active' : 'tab'} onClick={() => setActiveTab('graph')}>Memory Graph</button>
      </nav>

      <main style={{ maxWidth: 800, margin: '0 auto', padding: '0 16px' }}>
        {activeTab === 'chat' && <ChatPanel sessionId={sessionId} />}
        {activeTab === 'generation' && <GeneratorPanel />}
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
  const [progressError, setProgressError] = useState('');
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const bottomRef = useRef(null);

  const refreshProgress = async () => {
    try {
      const response = await fetch('/api/generator');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setGeneratorStatus(await response.json());
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
    const text = input.trim();
    if (!text || loading) return;

    setMessages((prev) => [...prev, { role: 'user', text }]);
    setInput('');
    setLoading(true);
    setError('');

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, sessionId, include_in_generator: true })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      setMessages((prev) => [...prev, {
        role: 'assistant',
        text: data.reply,
        mode: data.mode,
        generation: data.generation,
        trainingSources: data.training_sources,
        generatorLearning: data.generator_learning,
        responseId: data.response_memory_id,
        selection: data.selection
      }]);
      void refreshProgress();
    } catch (err) {
      setError(err.message);
      setMessages((prev) => prev.slice(0, -1));
    } finally {
      setLoading(false);
    }
  };

  const rateResponse = async (responseId, rating, approveForScoring, correction = '') => {
    setMessages((previous) => previous.map((message) => message.responseId === responseId ? { ...message, ratingPending: true } : message));
    setError('');
    try {
      const response = await fetch('/api/response-feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ response_id: responseId, rating, correction, approve_for_scoring: approveForScoring, approve_correction: approveForScoring && Boolean(correction.trim()) })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
      setMessages((previous) => previous.map((message) => message.responseId === responseId ? { ...message, rating, scorer: data.scorer, correctionSaved: Boolean(correction.trim()), correctionTrained: data.scorer?.correction_training?.trained } : message));
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

      <div className="generator-status" aria-live="polite">
        {generatorStatus ? (
          <>
            <span>Generation {generatorStatus.generation}</span>
            <span>{generatorStatus.document_count} training sources</span>
            <span>{generatorStatus.training_characters.toLocaleString()} characters</span>
          </>
        ) : <span>Loading progress...</span>}
        <span>{messages.filter((message) => message.rating === 'useful').length} relevant this session</span>
        <span>{messages.filter((message) => message.rating === 'needs_correction').length} need correction this session</span>
        <a href="/?view=generation">Compare generations</a>
      </div>
      {progressError && <p className="banner banner-error">{progressError}</p>}
      {generatorStatus?.scorer && (
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
            {m.mode === 'local-character-ngram' && (
              <small className="message-meta">
                Experimental local character model, generation {m.generation}.
                {m.trainingSources?.length ? ` Sources: ${m.trainingSources.map((source) => source.title).join(', ')}.` : ''}
              </small>
            )}
            {m.generatorLearning?.added && <small className="message-meta">Your prompt was added for future generation {m.generatorLearning.generation}.</small>}
            {m.generatorLearning?.reason === 'already_present' && <small className="message-meta">Your prompt is already in the generator corpus.</small>}
            {m.generatorLearning?.reason === 'too_short' && <small className="message-meta">Prompt skipped: at least four characters are needed for generator training.</small>}
            {m.generatorLearning?.reason === 'error' && <small className="message-meta">Prompt was saved to memory but not generator training: {m.generatorLearning.message}</small>}
            {m.selection && <small className="message-meta">{Number.isFinite(m.selection.score) ? `Selected from ${m.selection.candidate_count} candidates. Scorer generation ${m.selection.generation}, selection score ${m.selection.score.toFixed(2)}.` : 'No generated candidate was available; showing the memory fallback.'}</small>}
            {m.responseId && (
              <div role="group" aria-label="Response relevance" style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 8 }}>
                <button type="button" title="Review this pair and train response selection" aria-pressed={m.rating === 'useful'} disabled={m.ratingPending} onClick={() => rateResponse(m.responseId, 'useful', m.mode === 'local-character-ngram')}>Relevant{m.mode === 'local-character-ngram' ? ' and train' : ''}</button>
                <button type="button" title="Review this pair and train response selection" aria-pressed={m.rating === 'needs_correction'} disabled={m.ratingPending} onClick={() => rateResponse(m.responseId, 'needs_correction', m.mode === 'local-character-ngram')}>Needs correction{m.mode === 'local-character-ngram' ? ' and train' : ''}</button>
                {m.rating && <small className="message-meta" role="status">{m.scorer ? `Local preference updated. Scorer generation ${m.scorer.generation}: ${m.scorer.training?.status}.` : 'Feedback saved for review.'}</small>}
              </div>
            )}
            {m.rating === 'needs_correction' && (
              <div style={{ marginTop: 8 }}>
                <label style={{ display: 'block', fontSize: 12 }}>Your corrected response
                  <textarea rows={3} maxLength={2000} value={m.correction || ''} style={{ width: '100%', marginTop: 6, resize: 'vertical' }} onChange={(event) => {
                    const correction = event.target.value;
                    setMessages((previous) => previous.map((message) => message.responseId === m.responseId ? { ...message, correction, correctionSaved: false } : message));
                  }} />
                </label>
                <button type="button" disabled={m.ratingPending || !m.correction?.trim() || (m.mode === 'local-character-ngram' && m.correction.trim().length < 4)} onClick={async () => {
                  await rateResponse(m.responseId, 'needs_correction', m.mode === 'local-character-ngram', m.correction);
                }}>{m.mode === 'local-character-ngram' ? 'Approve correction and train' : 'Save correction for review'}</button>
                {m.correctionSaved && <small className="message-meta" role="status">{m.correctionTrained ? 'Your correction trained the generator and positive response selection.' : 'Correction saved for review; not added to generator training text.'}</small>}
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
          disabled={loading}
        />
        <button onClick={send} disabled={loading} className="primary">
          {loading ? '…' : 'Send'}
        </button>
      </div>
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
