import { useEffect, useRef, useState } from 'react';

export default function App() {
  const [activeTab, setActiveTab] = useState('chat');
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
        <button className={activeTab === 'graph' ? 'tab active' : 'tab'} onClick={() => setActiveTab('graph')}>Memory Graph</button>
      </nav>

      <main style={{ maxWidth: 800, margin: '0 auto', padding: '0 16px' }}>
        {activeTab === 'chat' && <ChatPanel sessionId={sessionId} />}
        {activeTab === 'graph' && (
          <iframe src="/graph" style={{ width: '100%', height: '75vh', border: 'none', background: '#0f0f1a' }} title="Memory Graph" />
        )}
      </main>
    </div>
  );
}

function ChatPanel({ sessionId }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const bottomRef = useRef(null);

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
        body: JSON.stringify({ message: text, sessionId })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      setMessages((prev) => [...prev, { role: 'assistant', text: data.reply }]);
    } catch (err) {
      setError(err.message);
      setMessages((prev) => prev.slice(0, -1));
    } finally {
      setLoading(false);
    }
  };

  return (
    <section className="hero-card" style={{ marginTop: 24 }}>
      <div>
        <p className="eyebrow">AI Chat</p>
        <h2>Memory retrieval</h2>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 360, overflowY: 'auto', margin: '12px 0' }}>
        {messages.length === 0 && (
          <p style={{ color: '#aaa', textAlign: 'center', padding: 16 }}>No messages yet. Start a conversation!</p>
        )}
        {messages.map((m, i) => (
          <div key={i} style={{ alignSelf: m.role === 'user' ? 'flex-end' : 'flex-start', maxWidth: '80%', padding: '8px 12px', borderRadius: 12, background: m.role === 'user' ? '#3b82f6' : '#2a2a3a', color: '#eee', fontSize: 14 }}>
            {m.text}
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
          placeholder="Search memories..."
          disabled={loading}
        />
        <button onClick={send} disabled={loading} className="primary">
          {loading ? '…' : 'Send'}
        </button>
      </div>
    </section>
  );
}
