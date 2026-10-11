import express from 'express';
import { execFile } from 'child_process';
import { randomUUID } from 'crypto';
import { pipeline } from '@xenova/transformers';
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

const semanticModelName = process.env.SEMANTIC_MODEL || 'Xenova/all-MiniLM-L6-v2';
const memoryPath = path.join(rootDir, 'chat_memory.json');
const graphStatePath = path.join(rootDir, 'memory_graph_state.json');
const updateScriptPath = path.join(rootDir, 'update_chat_memory.py');
const execFileAsync = promisify(execFile);
let embedderPromise = null;
let semanticIndexCache = {
  mtimeMs: 0,
  messages: [],
  embeddings: []
};

const state = {
  subscription: null,
  latestPush: null,
  latestResponse: null,
  events: []
};

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

async function getEmbedder() {
  if (!embedderPromise) {
    embedderPromise = pipeline('feature-extraction', semanticModelName);
  }
  return embedderPromise;
}

function cosineScore(a, b) {
  let score = 0;
  for (let index = 0; index < a.length; index += 1) {
    score += a[index] * b[index];
  }
  return score;
}

async function loadSemanticIndex() {
  const stat = await fs.stat(memoryPath);
  if (semanticIndexCache.mtimeMs === stat.mtimeMs) {
    return semanticIndexCache;
  }

  const transcript = JSON.parse(await fs.readFile(memoryPath, 'utf8'));
  const messages = transcript.messages.map((message) => ({
    id: message.id,
    role: message.role,
    text: message.text,
    topics: Array.isArray(message.topics) ? message.topics : [],
    kind: 'message'
  }));
  const texts = messages.map((message) =>
    `${message.role}: ${message.text}\nTopics: ${message.topics.join(', ')}`
  );
  const embedder = await getEmbedder();
  const embeddingsTensor = await embedder(texts, { pooling: 'mean', normalize: true });
  const embeddings = embeddingsTensor.tolist();

  semanticIndexCache = {
    mtimeMs: stat.mtimeMs,
    messages,
    embeddings
  };
  return semanticIndexCache;
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
  await fs.writeFile(graphStatePath, `${JSON.stringify(snapshot, null, 2)}\n`, 'utf8');
  const stat = await fs.stat(graphStatePath);
  return stat.mtimeMs;
}

async function appendMemoryExchange({ user, assistant, userTopics = [], assistantTopics = [] }) {
  const args = [
    updateScriptPath,
    '--user', user,
    '--assistant', assistant
  ];
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

app.get('/api/config', (req, res) => {
  res.json({
    pushSupported: Boolean(vapidKeys.publicKey),
    vapidPublicKey: vapidKeys.publicKey,
    hasSubscription: Boolean(state.subscription)
  });
});

app.get('/api/status', (req, res) => {
  res.json({
    hasSubscription: Boolean(state.subscription),
    latestPush: state.latestPush,
    latestResponse: state.latestResponse,
    events: state.events
  });
});

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

  if (!snapshot || typeof snapshot !== 'object') {
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
    appendEvent('push', payload);
    res.json({ ok: true, payload });
  } catch (error) {
    res.status(500).json({ error: error.message });
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
    appendEvent('response', { ...state.latestResponse, savedToMemory: true, memoryCycles: graphState?.snapshot?.cycles ?? null });
    res.json({ ok: true, response: state.latestResponse, memory: graphState });
  } catch (error) {
    appendEvent('response', { ...state.latestResponse, savedToMemory: false, error: error.message });
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
    const index = await loadSemanticIndex();
    const embedder = await getEmbedder();
    const queryTensor = await embedder(query, { pooling: 'mean', normalize: true });
    const queryVector = queryTensor.tolist()[0] || queryTensor.tolist();

    const results = index.messages
      .map((message, indexPosition) => ({
        id: message.id,
        role: message.role,
        text: message.text,
        topics: message.topics,
        score: cosineScore(queryVector, index.embeddings[indexPosition])
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, limit)
      .map((item) => ({
        ...item,
        score: Number(item.score.toFixed(4))
      }));

    res.json({
      query,
      model: semanticModelName,
      results
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
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
