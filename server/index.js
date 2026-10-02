import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';
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

const state = {
  subscription: null,
  latestPush: null,
  latestResponse: null,
  events: []
};

app.use(express.json());

function appendEvent(type, payload) {
  state.events.unshift({
    id: crypto.randomUUID(),
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

app.post('/api/respond', (req, res) => {
  const message = String(req.body?.message || '').trim();

  if (!message) {
    res.status(400).json({ error: 'Response message is required.' });
    return;
  }

  state.latestResponse = {
    message,
    timestamp: new Date().toISOString()
  };
  appendEvent('response', state.latestResponse);
  res.json({ ok: true, response: state.latestResponse });
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
