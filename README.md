# Research Agent

An AI-powered research assistant built with **LangGraph**, **FastAPI**, and **Pinecone** that autonomously searches the web, fetches articles, stores knowledge, and writes structured research reports — with optional human-in-the-loop approval.

---

## ✨ Features

- **Smart Caching** — Checks Pinecone before searching; skips the full pipeline on cache hits
- **Human-in-the-Loop** — Pauses after search results so you can approve URLs before fetching
- **Autonomous Mode** — Runs the full pipeline end-to-end without interruption
- **Real-time Streaming** — SSE endpoint streams report tokens as they're generated
- **Pinecone Vector Memory** — Chunks and stores articles for future retrieval via MMR search
- **Structured Reports** — GPT-4o writes cited reports with Overview, Key Findings, Trends, and Sources
- **Fault-Tolerant Fetching** — Failed URLs are skipped gracefully; pipeline never crashes

---

## Architecture

```
User Request
     │
     ▼
┌─────────────┐
│ cache_check │ ──── Cache HIT ────────────────────────┐
└─────────────┘                                        │
     │ Cache MISS                                      │
     ▼                                                 │
┌────────┐    ┌───────┐    ┌────────┐    ┌──────────┐  │
│ search │───▶│ fetch │───▶│ ingest │───▶│ retrieve │◀─┘
└────────┘    └───────┘    └────────┘    └──────────┘
  Tavily       httpx +      Pinecone       MMR Search
  Search       BS4 parse    vectorstore    (k=6)
                                                │
                                                ▼
                                          ┌────────┐
                                          │ writer │
                                          └────────┘
                                           GPT-4o
                                           Report
```

**Human-in-the-loop** interrupts the graph between `search` and `fetch`, allowing URL review via the `/research/approve` endpoint.

---

## 📁 Project Structure

```
research-agent/
├── main.py                  # FastAPI app — all HTTP endpoints
├── requirements.txt         # Python dependencies
├── test.py                  # Test script for all 3 modes
├── .env                     # API keys (not committed)
├── graph/
│   ├── graph.py             # LangGraph graph definition & compilation
│   ├── nodes.py             # All 5 node functions + cache routing
│   └── state.py             # AgentState TypedDict
└── tools/
    ├── search.py            # Tavily web search wrapper
    ├── fetcher.py           # httpx + BeautifulSoup article fetcher
    ├── ingestor.py          # Pinecone chunk ingestion
    └── retriever.py         # Pinecone MMR retrieval
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- A [Pinecone](https://www.pinecone.io/) account with an index created
- An [OpenAI](https://platform.openai.com/) API key
- A [Tavily](https://tavily.com/) API key

### 1. Clone the repository

```bash
git clone https://github.com/ChandakChakma/research-agent.git
cd research-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
PINECONE_API_KEY=pcsk-...
PINECONE_INDEX=research-agent
```

> **Pinecone index settings:** dimension `1536`, metric `cosine` (matches `text-embedding-3-small`)

### 5. Run the server

```bash
uvicorn main:app --reload
```

The API will be live at `http://127.0.0.1:8000`.  
Interactive docs at `http://127.0.0.1:8000/docs`.

---

## 📡 API Reference

### `GET /health`
Health check.

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

### `POST /research/start`
Start a research task. Supports both autonomous and human-in-the-loop modes.

**Request body:**
```json
{
  "topic": "LangGraph multi-agent systems",
  "thread_id": "optional-uuid",
  "autonomous": false
}
```

**Autonomous response:**
```json
{
  "thread_id": "abc-123",
  "report": "## Overview\n...",
  "sources": ["https://..."],
  "chunks_ingested": 42
}
```

**Human-in-the-loop response** (pauses after search):
```json
{
  "thread_id": "abc-123",
  "status": "awaiting_approval",
  "search_results": [
    {"title": "...", "url": "https://...", "content": "..."}
  ],
  "message": "Review sources and call POST /research/approve to continue"
}
```

---

### `POST /research/approve`
Resume a paused human-in-the-loop session with approved URLs.

**Request body:**
```json
{
  "thread_id": "abc-123",
  "approved_urls": ["https://example.com/article1", "https://example.com/article2"]
}
```

**Response:** Same as autonomous (full `ResearchResponse`).

---

### `POST /research/stream`
Stream the report generation in real time using Server-Sent Events.

**Request body:**
```json
{
  "topic": "quantum computing breakthroughs 2024"
}
```

**SSE event types:**

| Type | Payload | Description |
|------|---------|-------------|
| `node_start` | `{"node": "search"}` | A pipeline node has started |
| `token` | `{"token": "The "}` | A report token streamed from GPT-4o |
| `done` | `{"report": "..."}` | Full report complete |
| `error` | `{"message": "..."}` | Something went wrong |

---

## 🧪 Testing

Make sure the server is running, then:

```bash
# Autonomous end-to-end test
python test.py autonomous

# Human-in-the-loop test (start → approve → report)
python test.py hitl

# Streaming test (watch tokens arrive in real time)
python test.py stream
```

---

## 🔄 Pipeline Flow

### Full Pipeline (Cache Miss)

```
POST /research/start
  → cache_check (similarity search in Pinecone)
  → search      (Tavily: 5 results, advanced depth)
  [PAUSE if human_approval=True]
  → fetch       (httpx: parallel download + BS4 parse)
  → ingest      (chunk 500 tokens, store to Pinecone with metadata)
  → retrieve    (MMR search: k=6 from 20 candidates)
  → writer      (GPT-4o: structured report with inline citations)
```

### Cached Pipeline (Cache Hit, score ≥ 0.50)

```
POST /research/start
  → cache_check (similarity score ≥ 0.50 → skip to retrieve)
  → retrieve    (MMR search from existing Pinecone data)
  → writer      (GPT-4o: report from cached context)
```

---

## ⚙️ Configuration

| Environment Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key | required |
| `TAVILY_API_KEY` | Tavily search API key | required |
| `PINECONE_API_KEY` | Pinecone API key | required |
| `PINECONE_INDEX` | Pinecone index name | `research-agent` |

Key constants in code:

| Setting | Value | Location |
|---|---|---|
| LLM model | `gpt-4o` | `nodes.py` |
| Embedding model | `text-embedding-3-small` | `ingestor.py`, `retriever.py` |
| Chunk size | `500` tokens, `50` overlap | `ingestor.py` |
| Retrieval | MMR, `k=6`, `fetch_k=20` | `retriever.py` |
| Cache threshold | `0.50` cosine similarity | `nodes.py` |
| Search results | `5` (advanced depth) | `search.py` |
| Fetch timeout | `15s` per URL | `fetcher.py` |
| Max article text | `8000` chars | `fetcher.py` |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | [LangGraph](https://github.com/langchain-ai/langgraph) 0.4 |
| LLM | OpenAI GPT-4o via [LangChain](https://github.com/langchain-ai/langchain) |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | [Pinecone](https://www.pinecone.io/) |
| Web search | [Tavily](https://tavily.com/) |
| HTTP client | [httpx](https://www.python-httpx.org/) |
| HTML parsing | [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) |
| API server | [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/) |
| Streaming | [sse-starlette](https://github.com/sysid/sse-starlette) |

