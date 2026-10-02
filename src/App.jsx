import { useEffect, useMemo, useState } from 'react';

const base64ToUint8Array = (base64String) => {
  const padding = '='.repeat((4 - (base64String.length % 4)) % 4);
  const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');
  const raw = window.atob(base64);
  const output = new Uint8Array(raw.length);

  for (let index = 0; index < raw.length; index += 1) {
    output[index] = raw.charCodeAt(index);
  }

  return output;
};

const formatTime = (value) => {
  if (!value) {
    return '—';
  }

  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'medium'
  }).format(new Date(value));
};

const sectionCopy = {
  title: 'Push Bridge POC',
  subtitle: 'Desktop sends one push to a phone, and the phone can reply back into the same server state.'
};

export default function App() {
  const [config, setConfig] = useState(null);
  const [status, setStatus] = useState(null);
  const [message, setMessage] = useState('Response from phone');
  const [error, setError] = useState('');
  const [info, setInfo] = useState('');
  const [busy, setBusy] = useState(false);

  const isPhoneLike = useMemo(() => window.matchMedia('(max-width: 860px)').matches, []);

  const refreshStatus = async () => {
    const response = await fetch('/api/status');
    const data = await response.json();
    setStatus(data);
  };

  useEffect(() => {
    let isMounted = true;

    const load = async () => {
      try {
        const [configResponse, statusResponse] = await Promise.all([
          fetch('/api/config'),
          fetch('/api/status')
        ]);

        const nextConfig = await configResponse.json();
        const nextStatus = await statusResponse.json();

        if (isMounted) {
          setConfig(nextConfig);
          setStatus(nextStatus);
        }
      } catch (loadError) {
        if (isMounted) {
          setError(loadError.message);
        }
      }
    };

    load();
    const interval = window.setInterval(() => {
      refreshStatus().catch(() => {});
    }, 2000);

    return () => {
      isMounted = false;
      window.clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    if (!('serviceWorker' in navigator)) {
      return undefined;
    }

    navigator.serviceWorker.register('/sw.js').catch((registrationError) => {
      setError(registrationError.message);
    });

    return undefined;
  }, []);

  const subscribePhone = async () => {
    if (!config?.vapidPublicKey) {
      setError('Missing VAPID public key.');
      return;
    }

    if (!('serviceWorker' in navigator) || !('PushManager' in window)) {
      setError('Push notifications are not supported in this browser.');
      return;
    }

    setBusy(true);
    setError('');
    setInfo('');

    try {
      const permission = await Notification.requestPermission();
      if (permission !== 'granted') {
        throw new Error('Notification permission was not granted.');
      }

      const registration = await navigator.serviceWorker.ready;
      const subscription = await registration.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: base64ToUint8Array(config.vapidPublicKey)
      });

      const response = await fetch('/api/subscribe', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(subscription)
      });

      if (!response.ok) {
        throw new Error((await response.json()).error || 'Failed to subscribe this phone.');
      }

      setInfo('Phone subscription saved. Leave the app installed and ready for the push.');
      await refreshStatus();
      setConfig((current) => ({ ...current, hasSubscription: true }));
    } catch (subscribeError) {
      setError(subscribeError.message);
    } finally {
      setBusy(false);
    }
  };

  const sendPush = async () => {
    setBusy(true);
    setError('');
    setInfo('');

    try {
      const response = await fetch('/api/send-push', {
        method: 'POST'
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to send push.');
      }

      setInfo('Push sent to the registered phone.');
      await refreshStatus();
    } catch (sendError) {
      setError(sendError.message);
    } finally {
      setBusy(false);
    }
  };

  const sendResponse = async () => {
    setBusy(true);
    setError('');
    setInfo('');

    try {
      const response = await fetch('/api/respond', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ message })
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to send the response.');
      }

      setInfo('Response sent back from the phone.');
      await refreshStatus();
    } catch (responseError) {
      setError(responseError.message);
    } finally {
      setBusy(false);
    }
  };

  const subscriptionReady = Boolean(status?.hasSubscription);
  const latestPush = status?.latestPush;
  const latestResponse = status?.latestResponse;

  return (
    <main className="app-shell">
      <div className="backdrop backdrop-a" />
      <div className="backdrop backdrop-b" />
      <section className="hero-card">
        <div>
          <p className="eyebrow">{isPhoneLike ? 'Phone view' : 'Desktop view'}</p>
          <h1>{sectionCopy.title}</h1>
          <p className="subtitle">{sectionCopy.subtitle}</p>
        </div>

        <div className="status-row">
          <span className={subscriptionReady ? 'pill pill-ready' : 'pill pill-waiting'}>
            {subscriptionReady ? 'Phone subscribed' : 'Waiting for phone subscription'}
          </span>
          <span className="pill pill-muted">PWA + Web Push</span>
        </div>

        <div className="grid">
          <article className="panel">
            <h2>1. Subscribe the phone</h2>
            <p>
              Open this app on the phone, allow notifications, and register the subscription.
            </p>
            <button onClick={subscribePhone} disabled={busy} className="primary">
              {busy ? 'Working...' : 'Subscribe this phone'}
            </button>
            <p className="hint">
              The server keeps one active phone subscription for the demo.
            </p>
          </article>

          <article className="panel">
            <h2>2. Send the desktop push</h2>
            <p>
              Trigger a single push notification from the desktop, then open the app on the phone to reply.
            </p>
            <button onClick={sendPush} disabled={busy || !subscriptionReady} className="primary secondary">
              Send push to phone
            </button>
            <p className="hint">
              {subscriptionReady ? 'Ready to send.' : 'Subscribe the phone first.'}
            </p>
          </article>

          <article className="panel panel-wide">
            <h2>3. Response from the phone</h2>
            <p>
              Type a reply on the phone and send it back to the shared server state.
            </p>
            <div className="response-row">
              <input
                value={message}
                onChange={(event) => setMessage(event.target.value)}
                placeholder="Type a response"
              />
              <button onClick={sendResponse} disabled={busy} className="primary">
                Reply from phone
              </button>
            </div>
            <p className="hint">
              This part is the simplest proof that the device can talk back after the push lands.
            </p>
          </article>

          <article className="panel panel-wide">
            <h2>Live state</h2>
            <div className="state-grid">
              <div>
                <span className="label">Last push</span>
                <strong>{latestPush ? latestPush.title : 'None yet'}</strong>
                <p>{latestPush ? formatTime(latestPush.timestamp) : 'No push has been sent.'}</p>
              </div>
              <div>
                <span className="label">Last response</span>
                <strong>{latestResponse ? latestResponse.message : 'None yet'}</strong>
                <p>{latestResponse ? formatTime(latestResponse.timestamp) : 'No reply has been received.'}</p>
              </div>
            </div>
          </article>
        </div>

        {info ? <p className="banner banner-success">{info}</p> : null}
        {error ? <p className="banner banner-error">{error}</p> : null}
      </section>
    </main>
  );
}
