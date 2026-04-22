from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # ── Input ──────────────────────────────────────────
    topic: str                  # user's research topic

    # ── Search phase ───────────────────────────────────
    search_results: list[dict]  # raw Tavily results [{title, url, content}]
    approved_urls: list[str]    # urls approved by human-in-the-loop

    # ── Fetch phase ─────────────────────────────────────
    fetched_articles: list[dict]  # [{url, title, full_text}]

    # ── Ingest phase ────────────────────────────────────
    ingested_chunks: int        # how many chunks stored in Pinecone

    # ── Retrieval phase ─────────────────────────────────
    retrieved_context: str      # formatted retrieved docs

    # ── Writing phase ───────────────────────────────────
    report: str                 # final structured research report

    # ── Message history ─────────────────────────────────
    messages: Annotated[list, add_messages]
