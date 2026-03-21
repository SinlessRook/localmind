# 🧠 Local Mind

### The Agentic Memory Layer for Your Browser

---

## 🚩 Problem

Modern users consume massive amounts of information while browsing, but existing tools like history and bookmarks are inefficient:

* Important content is often **not saved**
* Even saved content is **hard to rediscover**
* Users waste time trying to **recall where they saw something**
* Knowledge gained through browsing is **lost instead of reused**

---

## 💡 Solution

**Local Mind** transforms your browsing activity into a **searchable, intelligent personal memory system**.

Instead of manually saving pages, it:

* Automatically captures meaningful browsing activity
* Filters noise and irrelevant content
* Converts pages into **summaries + semantic embeddings**
* Allows you to **query your past browsing using natural language**

---

## 🔐 Privacy-First Design

* ✅ Runs entirely on the **user’s device**
* ✅ Stores data locally using browser storage
* ✅ Supports encryption for stored data
* ❌ No raw browsing data sent to external servers

---

## ⚙️ Core Features

### 🧩 Passive Memory Capture

* Extracts content from visited pages
* Uses dwell-time filtering to avoid noise
* Captures:

  * Title
  * URL
  * Page text

---

### 🧠 Intelligent Processing

* Generates:

  * Lightweight summaries
  * Semantic embeddings (vector representations)
* Enables **meaning-based search**, not just keyword matching

---

### 🔍 Natural Language Search

Ask questions like:

* *“Where did I read about microservices?”*
* *“Show articles about machine learning from last week”*

System performs:

* Semantic search
* Metadata filtering (time, relevance)

Returns:

* Direct answers
* Relevant pages with summaries

---

### 🕒 Memory Timeline

* Displays recently captured pages
* Helps users visually navigate browsing history
* Sorted by recency and relevance

---

### 🖥️ Split-Screen Experience

* Continue browsing on one side
* View AI-powered memory results on the other

---

## 🏗️ Architecture

### 📦 Chrome Extension (Frontend)

* Popup / Sidebar UI
* Content scripts for page extraction
* Background service worker for orchestration

### 🧠 AI Layer

* Embedding generation
* Semantic similarity search
* LLM-based response generation

### 💾 Storage

* IndexedDB / local storage
* Stores:

  * Page metadata
  * Summaries
  * Embeddings

---

## 🔄 Workflow

1. User visits a webpage
2. Content script extracts page data
3. Background script filters and stores it
4. Embeddings + summaries are generated
5. User enters a query
6. System performs semantic + metadata search
7. LLM generates a contextual answer
8. Results displayed in UI

---

## 🛠️ Tech Stack

**Frontend (Extension)**

* JavaScript (Vanilla / minimal framework)
* Chrome Extension Manifest V3

**Storage**

* IndexedDB / Chrome Storage API

**AI / Backend (Optional)**

* Python (FastAPI) / Node.js
* Embedding models (local or API)
* LLM APIs (for summarization + answers)

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/local-mind.git
cd local-mind
```

---

### 2. Load Extension in Browser

* Open Chrome → `chrome://extensions/`
* Enable **Developer Mode**
* Click **Load Unpacked**
* Select the `extension/` folder

---

### 3. Start Backend (Optional)

```bash
cd backend
pip install -r requirements.txt
python main.py
```

---

## 🎯 Demo Flow

1. Visit a few webpages
2. Open Local Mind extension
3. View **Memory Timeline**
4. Search:

   * *“microservices”*
5. Get:

   * Relevant pages
   * AI-generated summaries

---

## 🧪 Future Improvements

* Smarter noise filtering
* Better on-device embedding models
* Cross-device sync (privacy-preserving)
* Learning pattern insights
* Auto-tagging and categorization

---

## 🏆 Impact

* Reduces time spent rediscovering information
* Turns browsing into **active knowledge recall**
* Builds a **personal AI memory layer**
* Helps users organize and reuse knowledge effortlessly

---

## 👥 Team

* Extension & UI: *[Your Name]*
* AI / Backend: *[Teammate]*
* Data / Processing: *[Teammate]*

---

## 📌 Tagline

> *“Stop searching the web again. Start remembering it.”*

---
