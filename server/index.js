import express from 'express';
import { execFile, spawn } from 'child_process';
import { randomUUID } from 'crypto';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import { promisify } from 'util';
import webpush from 'web-push';

const app = express();
const port = Number(process.env.PORT || 3001);
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');
const distDir = path.join(rootDir, 'dist');

const vapidKeys = process.env.VAPID_PUBLIC_KEY && process.env.VAPID_PRIVATE_KEY
  ? {
      publicKey: process.env.VAPID_PUBLIC_KEY,
      privateKey: process.env.VAPID_PRIVATE_KEY
    }
  : webpush.generateVAPIDKeys();

webpush.setVapidDetails(
  'mailto:dev@example.com',
  vapidKeys.publicKey,
  vapidKeys.privateKey
);

const graphStatePath = path.join(rootDir, 'memory_graph_state.json');
const updateScriptPath = path.join(rootDir, 'update_chat_memory.py');
const execFileAsync = promisify(execFile);
const learnerPython = process.env.MEMORY_PYTHON || 'python3';
let learnerQueue = Promise.resolve();

const state = {
  subscription: null,
  latestPush: null,
  latestResponse: null,
  events: []
};

app.use('/api/graph-state', express.json({ limit: '2mb' }));
app.use(express.json());
app.use((req, res, next) => {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') {
    res.sendStatus(204);
    return;
  }
  next();
});

function runLearner(command, payload = {}) {
  const result = learnerQueue.then(() => new Promise((resolve, reject) => {
    const child = spawn(learnerPython, [path.join(rootDir, 'memory_learner.py'), command], { cwd: rootDir });
    let output = '';
    let errors = '';
    const timer = setTimeout(() => child.kill(), 30000);
    child.stdout.on('data', chunk => { output += chunk; });
    child.stderr.on('data', chunk => { errors += chunk; });
    child.on('error', error => { clearTimeout(timer); reject(error); });
    child.on('close', code => {
      clearTimeout(timer);
      if (code !== 0) { reject(new Error(errors || 'Local learner failed')); return; }
      try { resolve(JSON.parse(output)); } catch (error) { reject(error); }
    });
    child.stdin.on('error', () => {});
    child.stdin.end(JSON.stringify(payload));
  }));
  learnerQueue = result.catch(() => {});
  return result;
}

async function readGraphState() {
  try {
    const [raw, stat] = await Promise.all([
      fs.readFile(graphStatePath, 'utf8'),
      fs.stat(graphStatePath)
    ]);
    return { snapshot: JSON.parse(raw), updatedAt: stat.mtimeMs };
  } catch (error) {
    if (error.code === 'ENOENT') {
      return { snapshot: null, updatedAt: null };
    }
    throw error;
  }
}

async function writeGraphState(snapshot) {
  const current = await readGraphState();
  let merged = snapshot;
  if (current.snapshot?.nodes) {
    merged = current.snapshot;
    const feedbackById = new Map((snapshot.nodes || []).map(node => [node.id, node]));
    for (const node of merged.nodes) {
      const feedback = feedbackById.get(node.id);
      if (!feedback) continue;
      for (const field of ['rawImportance', 'history', 'consolidated', 'feedback']) {
        if (Object.hasOwn(feedback, field)) node[field] = feedback[field];
      }
    }
    const strengths = new Map((snapshot.links || []).map(link => [`${link.source}|${link.target}|${link.kind || ''}`, link.strength]));
    for (const link of merged.links || []) {
      const strength = strengths.get(`${link.source}|${link.target}|${link.kind || ''}`);
      if (typeof strength === 'number' && Number.isFinite(strength)) link.strength = strength;
    }
    if (typeof snapshot.query === 'string') merged.query = snapshot.query;
    if (typeof snapshot.sidebarCollapsed === 'boolean') merged.sidebarCollapsed = snapshot.sidebarCollapsed;
  }
  await fs.writeFile(graphStatePath, `${JSON.stringify(merged, null, 2)}\n`, 'utf8');
  const stat = await fs.stat(graphStatePath);
  return stat.mtimeMs;
}

async function appendMemoryExchange({ user, assistant, userTopics = [], assistantTopics = [], sessionId, source = 'app' }) {
  const args = [
    updateScriptPath,
    '--user', user,
    '--assistant', assistant
  ];
  args.push('--source', source);
  if (typeof sessionId === 'string' && sessionId.trim()) {
    args.push('--session-id', sessionId);
  }
  userTopics.forEach((topic) => {
    args.push('--user-topic', topic);
  });
  assistantTopics.forEach((topic) => {
    args.push('--assistant-topic', topic);
  });
  await execFileAsync('python3', args, {
    cwd: rootDir,
    maxBuffer: 10 * 1024 * 1024
  });
  return readGraphState();
}

function inferMemoryTopics(text) {
  const normalized = String(text || '').toLowerCase();
  const topics = new Set();

  if (/(push|notification|phone|desktop|bridge)/.test(normalized)) {
    topics.add('push_bridge');
  }
  if (/(reply|response|respond|answer)/.test(normalized)) {
    topics.add('reply');
  }
  if (/(memory|graph|transcript|history)/.test(normalized)) {
    topics.add('memory');
  }

  return [...topics];
}


function appendEvent(type, payload) {
  state.events.unshift({
    id: randomUUID(),
    type,
    payload,
    createdAt: new Date().toISOString()
  });
  state.events = state.events.slice(0, 20);
}

app.get('/api/graph-state', async (req, res) => {
  try {
    const state = await readGraphState();
    res.json(state);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/graph-state', async (req, res) => {
  const snapshot = req.body;

  if (!snapshot || typeof snapshot !== 'object' || !Array.isArray(snapshot.nodes) || !Array.isArray(snapshot.links)) {
    res.status(400).json({ error: 'Snapshot payload is required.' });
    return;
  }

  try {
    const updatedAt = await writeGraphState(snapshot);
    res.json({ ok: true, updatedAt });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/append-exchange', async (req, res) => {
  const { user, assistant, userTopics, assistantTopics } = req.body || {};
  if (typeof user !== 'string' || !user.trim() || typeof assistant !== 'string' || !assistant.trim()) {
    res.status(400).json({ error: 'user and assistant text are required.' });
    return;
  }

  try {
    const state = await appendMemoryExchange({
      user: user.trim(),
      assistant: assistant.trim(),
      userTopics: Array.isArray(userTopics) ? userTopics.filter((topic) => typeof topic === 'string' && topic.trim()) : [],
      assistantTopics: Array.isArray(assistantTopics) ? assistantTopics.filter((topic) => typeof topic === 'string' && topic.trim()) : []
    });
    res.json({ ok: true, ...state });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/subscribe', (req, res) => {
  const subscription = req.body;

  if (!subscription || typeof subscription !== 'object') {
    res.status(400).json({ error: 'Missing subscription payload.' });
    return;
  }

  state.subscription = subscription;
  appendEvent('subscription', { endpoint: subscription.endpoint });
  res.json({ ok: true });
});

app.post('/api/send-push', async (req, res) => {
  if (!state.subscription) {
    res.status(409).json({ error: 'No phone subscription has been registered yet.' });
    return;
  }

  const payload = {
    title: 'Desktop ping',
    body: 'Open the app on your phone and send the response back.',
    timestamp: new Date().toISOString()
  };

  try {
    await webpush.sendNotification(state.subscription, JSON.stringify(payload));
    state.latestPush = payload;
    res.json({ ok: true, payload });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// ─── Chat API (mirrors /api/respond but for direct chat) ─────────────────────

app.post('/api/chat', async (req, res) => {
  const userMessage = String(req.body?.message || '').trim();
  if (!userMessage || userMessage.length > 2000) {
    res.status(400).json({ error: 'Missing message' });
    return;
  }
  try {
    const retrieval = await runLearner('search', { query: userMessage, limit: 3 });
    const best = retrieval.results.find(memory => memory.role === 'Assistant');
    const aiResponse = best
      ? `[${best.id}] ${best.text.length > 600 ? best.text.slice(0, 600).trimEnd() + '...' : best.text}`
      : 'No stored assistant reply matched. The current learner retrieves memories; it does not generate new language yet.';
    const assistantTopics = inferMemoryTopics(aiResponse);
    const saved = await appendMemoryExchange({ user: userMessage, assistant: aiResponse, userTopics: [], assistantTopics, sessionId: req.body?.sessionId, source: 'local-retrieval' });
    const recent = saved.snapshot.nodes.filter(node => node.kind === 'message' && node.created_cycle === saved.snapshot.cycles);

    res.json({ reply: aiResponse, model: retrieval.model, generation: retrieval.generation, mode: 'memory-excerpts', response_source_id: best?.id || null, memories: retrieval.results,
      prompt_memory_id: recent.find(node => node.role === 'You')?.id,
      response_memory_id: recent.find(node => node.role === 'Assistant')?.id });
  } catch (err) {
    console.error('Chat error:', err);
    res.status(500).json({ error: 'Failed to process chat' });
  }
});

app.post('/api/respond', async (req, res) => {
  const message = String(req.body?.message || '').trim();

  if (!message) {
    res.status(400).json({ error: 'Response message is required.' });
    return;
  }

  const timestamp = new Date().toISOString();
  state.latestResponse = { message, timestamp };

  try {
    const latestPushText = state.latestPush
      ? [state.latestPush.title, state.latestPush.body].filter(Boolean).join(' — ')
      : 'Phone response without a prior push';
    const graphState = await appendMemoryExchange({
      user: latestPushText,
      assistant: message,
      userTopics: inferMemoryTopics(latestPushText),
      assistantTopics: inferMemoryTopics(message)
    });
    res.json({ ok: true, response: state.latestResponse, memory: graphState });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/semantic-search', async (req, res) => {
  const query = String(req.query.query || req.query.q || '').trim();
  const limit = Math.max(1, Math.min(20, Number(req.query.limit || 6)));

  if (!query) {
    res.status(400).json({ error: 'Query is required.' });
    return;
  }

  try {
    res.json(await runLearner('search', { query, limit: Number.isFinite(limit) ? limit : 6, user_only: req.query.user_only === 'true', exclude_ids: typeof req.query.exclude_id === 'string' ? [req.query.exclude_id] : [] }));
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/learner-status', async (req, res) => {
  try { res.json(await runLearner('status')); }
  catch (error) { res.status(500).json({ error: error.message }); }
});

app.post('/api/learner-feedback', async (req, res) => {
  const { query, memory_id, label, approve_for_training = false } = req.body || {};
  if (typeof query !== 'string' || !query.trim() || query.length > 2000 || typeof memory_id !== 'string' || !['direct_answer', 'related_context', 'unrelated'].includes(label)) {
    res.status(400).json({ error: 'query, memory_id and label (direct_answer, related_context, unrelated) are required' });
    return;
  }
  if (typeof approve_for_training !== 'boolean') {
    res.status(400).json({ error: 'approve_for_training must be boolean' });
    return;
  }
  try {
    if (approve_for_training) {
      await runLearner('review', { memory_ids: [memory_id], queries: [query], acknowledge: true });
    }
    res.json(await runLearner('feedback', { query, memory_id, label }));
  }
  catch (error) { res.status(400).json({ error: error.message }); }
});

// ─── Graph HTML endpoint ─────────────────────────────────────────────────────
app.post('/api/response-feedback', async (req, res) => {
  const { response_id, rating, correction = '' } = req.body || {};
  if (typeof response_id !== 'string' || !['useful', 'needs_correction'].includes(rating) || typeof correction !== 'string' || correction.length > 2000) {
    res.status(400).json({ error: 'response_id, valid rating and correction up to 2000 characters are required' });
    return;
  }
  try { res.json(await runLearner('response-feedback', { response_id, rating, correction })); }
  catch (error) { res.status(400).json({ error: error.message }); }
});

const graphHtmlPath = path.join(rootDir, 'memory_graph.html');
app.get('/graph', (req, res) => {
  res.sendFile(graphHtmlPath);
});

if (process.env.NODE_ENV === 'production') {
  app.use(express.static(distDir));
  app.get('*', (req, res) => {
    res.sendFile(path.join(distDir, 'index.html'));
  });
}

app.listen(port, () => {
  console.log(`Push POC server running on http://localhost:${port}`);
  console.log(`VAPID public key: ${vapidKeys.publicKey}`);
});
