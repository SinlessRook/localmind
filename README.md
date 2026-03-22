# 🧠 Local Mind

> Your browser has a memory now. Finally.

Local Mind is an AI-powered Chrome extension that transforms your browsing history into an 
intelligent, searchable knowledge base. Search everything you've ever read in plain English — 
the AI understands meaning, not just keywords.

## How it works

Local Mind runs a lightweight AI embedding model that converts every page you read into a 
semantic vector. When you search, it finds the most relevant pages using vector similarity — 
so searching "transformer attention mechanism" also surfaces pages about "self-attention in 
neural networks" even if those exact words never appear together.

When you search on Google, a mini panel automatically surfaces relevant pages from your own 
memory alongside the search results — turning your browsing history into a personal RAG system.

## Stack

| Layer | Technology |
|-------|-----------|
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector search | FAISS |
| Backend | Python + FastAPI + Uvicorn |
| Extension | Chrome MV3 — vanilla JS, no framework |
| Storage | chrome.storage.local + FAISS index |

## Structure

| Folder | What it does |
|--------|-------------|
| `extension/` | Chrome extension — sidebar UI, content script, background worker, injected panel |
| `backend/` | Embedding server — generates vectors and runs semantic search |
| `frontend/` | Public landing page |

## Features

- **Semantic search** — AI understands meaning, not just keywords
- **Auto-indexing** — passively captures pages after real dwell time (no manual saving)
- **Memory timeline** — chronological view of everything you've read
- **Search-aware panel** — mini panel on Google showing memory results for your query
- **Light and dark mode** — synced across sidebar, panel, and landing page
- **Per-entry controls** — copy URL, copy title, delete individual entries
- **Time-based clearing** — clear last hour, last 24h, or all memory

## Setup

### Backend
```bash
cd backend
python -m pip install -r requirements.txt
python server.py
```

### Extension
1. Go to `chrome://extensions`
2. Enable **Developer Mode**
3. Click **Load unpacked** → select the `extension/` folder
4. Pin Local Mind from the toolbar

## Team

| Area | Responsibility |
|------|---------------|
| Extension UI | Sidebar, content script, injected panel, landing page |
| Embedding engine | Local AI model, vector generation |
| Search backend | FAISS index, retrieval logic, FastAPI server |
