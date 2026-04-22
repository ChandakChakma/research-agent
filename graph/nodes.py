from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

from graph.state import AgentState
from tools.search import run_search
from tools.fetcher import fetch_articles
from tools.ingestor import ingest_articles
from tools.retriever import run_retrieval

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Lazy-init cache vectorstore — created once, reused every call
_cache_vectorstore = None


def _get_cache_vectorstore():
    global _cache_vectorstore
    if _cache_vectorstore is None:
        import os
        from langchain_openai import OpenAIEmbeddings
        from langchain_pinecone import PineconeVectorStore
        _cache_vectorstore = PineconeVectorStore(
            index_name=os.getenv("PINECONE_INDEX", "research-agent"),
            embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
            # No namespace
        )
    return _cache_vectorstore


# ─────────────────────────────────────────────────────
# CONDITIONAL — Cache Check
# ─────────────────────────────────────────────────────
def should_search(state: AgentState) -> str:
    topic = state["topic"]
    print(f"[cache_check] Checking if '{topic}' already in knowledge base...")

    try:
        results = _get_cache_vectorstore().similarity_search_with_score(topic, k=3)

        if not results:
            print(f"[cache_check] Cache MISS -- index empty")
            return "search"

        best_score = results[0][1]
        best_doc   = results[0][0].metadata.get("topic", "unknown")
        print(f"[cache_check] Best score: {best_score:.3f} (from topic: '{best_doc}')")

        if best_score >= 0.50:
            print(f"[cache_check] Cache HIT")
            return "cached"

        print(f"[cache_check] Cache MISS -- score too low")
        return "search"

    except Exception as e:
        print(f"[cache_check] Error: {e} -- defaulting to search")
        return "search"


# ─────────────────────────────────────────────────────
# NODE 1 — Search
# ─────────────────────────────────────────────────────
def search_node(state: AgentState) -> AgentState:
    topic = state["topic"]
    print(f"[search] Searching for: {topic}")
    results = run_search(topic)
    print(f"[search] Found {len(results)} results")
    return {
        "search_results": results,
        "messages": [HumanMessage(content=f"Found {len(results)} sources for: {topic}")],
    }


# ─────────────────────────────────────────────────────
# NODE 2 — Fetch
# ─────────────────────────────────────────────────────
def fetch_node(state: AgentState) -> AgentState:
    urls = state.get("approved_urls") or [r["url"] for r in state["search_results"]]
    print(f"[fetch] Fetching {len(urls)} articles in parallel...")
    articles = fetch_articles(urls)
    print(f"[fetch] Successfully fetched {len(articles)} articles")
    return {
        "fetched_articles": articles,
        "messages": [HumanMessage(content=f"Fetched {len(articles)} full articles")],
    }


# ─────────────────────────────────────────────────────
# NODE 3 — Ingest
# ─────────────────────────────────────────────────────
def ingest_node(state: AgentState) -> AgentState:
    topic    = state["topic"]
    articles = state.get("fetched_articles", [])
    print(f"[ingest] Ingesting {len(articles)} articles into Pinecone...")
    count = ingest_articles(articles, topic)
    return {
        "ingested_chunks": count,
        "messages": [HumanMessage(content=f"Ingested {count} chunks into knowledge base")],
    }


# ─────────────────────────────────────────────────────
# NODE 4 — Retrieve
# ─────────────────────────────────────────────────────
def retrieve_node(state: AgentState) -> AgentState:
    topic = state["topic"]
    print(f"[retrieve] Retrieving context for: {topic}")
    context = run_retrieval(topic)
    return {
        "retrieved_context": context,
        "messages": [HumanMessage(content="Retrieved relevant context from knowledge base")],
    }


# ─────────────────────────────────────────────────────
# NODE 5 — Writer
# ─────────────────────────────────────────────────────
def writer_node(state: AgentState) -> AgentState:
    topic    = state["topic"]
    context  = state.get("retrieved_context", "")
    articles = state.get("fetched_articles", [])

    # Fix — graceful fallback when cache hit (no fetched articles)
    if articles:
        sources = "\n".join(f"- {a['title']}: {a['url']}" for a in articles)
    else:
        sources = "Sources retrieved from Pinecone knowledge base."

    print(f"[writer] Writing research report on: {topic}")

    system = SystemMessage(content=(
        "You are an expert research analyst. "
        "Write clear, structured, well-cited research reports. "
        "Always ground your writing in the provided context."
    ))

    user_content = f"""
Write a comprehensive research report on: {topic}

Use this retrieved context as your primary source:
{context}

Sources available:
{sources}

Structure the report as:
## Overview
## Key Findings
## Important Details
## Current Trends
## Conclusion
## Sources

Be specific, cite sources inline, and avoid hallucination.
Write only what the context supports.
"""

    response = llm.invoke([system, HumanMessage(content=user_content)])
    return {
        "report":   response.content,
        "messages": [response],
    }
