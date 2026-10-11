# Push Bridge POC

A proof-of-concept PWA that sends a web push notification from a desktop browser to a phone browser, then lets the phone reply back to the same server state.

## What it does

- Registers the phone as a push subscription.
- Sends one push from the desktop.
- Shows the notification on the phone via a service worker.
- Lets the phone send a response back to the backend.
- Polls the latest server state so the desktop can see the reply.

## Prerequisites

- Node.js 18 or newer.
- A browser with service worker and Push API support on the phone.
- An HTTPS URL for the phone to subscribe in a real test environment. Localhost works only on the same device; for cross-device testing, use a tunnel or deployment with HTTPS.

## Install

```bash
npm install
```

## Run locally

```bash
npm run dev
```

This starts the API server on `http://localhost:3001` and the Vite client on `http://localhost:5173`. The memory graph page can open directly from disk, but semantic retrieval requires the local API server so it can generate embeddings.

## Use it

1. Open the app on the phone and allow notifications.
2. Tap `Subscribe this phone`.
3. Open the app on the desktop and tap `Send push to phone`.
4. When the notification arrives on the phone, open the app and tap `Reply from phone`.

## Notes

- The server keeps one active subscription in memory for the demo.
- If you restart the server, the generated VAPID keys change unless you set `VAPID_PUBLIC_KEY` and `VAPID_PRIVATE_KEY`.
- For a more durable setup, store subscriptions and keys in a database or environment file.
- Each phone reply now appends a new memory exchange automatically, advances the cycle, regenerates `memory_graph.html`, and refreshes the backend graph-state snapshot.

## Local memory graph

The conversation memory prototype stores its transcript in `chat_memory.json` and renders it in `memory_graph.html`. The browser view now keeps its learning overlay in a backend graph-state file served from the local API, so helpful, correction, and weak-link feedback survives refreshes without depending on browser `localStorage`. The graph page also polls that backend snapshot, so it refreshes itself when the stored memory state changes. Use the search box to rank memories by query, click a node to open the selected-memory panel, hide that selected-memory section from the graph header if you want it out of the way, and hide the entire right menu with the button above the graph if you want the graph to use the full workspace. The feedback buttons teach the graph which memories should matter more. To add a conversation exchange and advance one persistent decay cycle:

```bash
python update_chat_memory.py \
  --user "What should be remembered?" \
  --assistant "A concise answer to remember." \
  --user-topic memory \
  --assistant-topic memory \
  --assistant-topic practical-use
```

This updates the local transcript and saved graph weights, then regenerates the HTML visualization and backend graph-state snapshot. Links combine turn order, manually supplied topic tags, sparse TF-IDF word similarity (up to two strongest text matches per message), and topic hub nodes that cluster related memories. The graph uses different line styles for those relationship types. The browser view also lets you reinforce or down-rank a memory interactively, and that feedback now persists through the backend graph-state file. Semantic search now uses local embeddings through the API server; if the server is unavailable, the page falls back to a lexical ranking.

To add newly inferred text-match links to existing messages without appending a conversation turn or advancing decay:

```bash
python update_chat_memory.py --reindex-links
```

## Memory architecture policy

This project is evolving from a chat-source memory into a queryable memory layer and eventually into a multi-agent system that can act from memory independently of the chat that created it. Keep these rules in mind:

- Preserve **provenance** for every memory so future agents can trace where it came from.
- Separate **domains** and **namespaces** so unrelated topics stay visible but not blended.
- Treat **evaluation/meta** as its own area rather than as core content memory.
- Allow **cluster promotion** only when a topic has enough weight, repetition, or importance.
- Prefer **stable references** and summaries for durable knowledge; keep transient turns lightweight.
- Support **retrieval-first behavior** so future agents can query memory instead of depending on chat context.
- Keep the graph and transcript compatible with later **agent-to-memory** workflows.
