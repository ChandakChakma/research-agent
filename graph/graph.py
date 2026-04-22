from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import AgentState
from graph.nodes import (
    should_search,
    search_node,
    fetch_node,
    ingest_node,
    retrieve_node,
    writer_node,
)


def build_graph(human_approval: bool = True):
    """
    Build the research agent graph.

    Flow with cache hit:
        cache_check → retrieve → writer

    Flow with cache miss:
        cache_check → search → fetch → ingest → retrieve → writer

    Args:
        human_approval: If True, pauses before fetch_node for human approval.
    """
    g = StateGraph(AgentState)

    # Register all nodes
    g.add_node("cache_check", lambda s: s)  # passthrough — just used for routing
    g.add_node("search",      search_node)
    g.add_node("fetch",       fetch_node)
    g.add_node("ingest",      ingest_node)
    g.add_node("retrieve",    retrieve_node)
    g.add_node("writer",      writer_node)

    # Entry point
    g.set_entry_point("cache_check")

    # Conditional edge — cache hit or full pipeline
    g.add_conditional_edges(
        "cache_check",
        should_search,
        {
            "cached": "retrieve",   # skip search/fetch/ingest entirely
            "search": "search",     # full pipeline
        }
    )

    # Linear edges for full pipeline
    g.add_edge("search",   "fetch")
    g.add_edge("fetch",    "ingest")
    g.add_edge("ingest",   "retrieve")

    # Both paths converge at retrieve → writer → END
    g.add_edge("retrieve", "writer")
    g.add_edge("writer",   END)

    checkpointer = MemorySaver()

    if human_approval:
        # Pause AFTER search results are ready, BEFORE fetching full articles
        return g.compile(
            checkpointer=checkpointer,
            interrupt_before=["fetch"],
        )

    return g.compile(checkpointer=checkpointer)


# Two variants — imported by main.py
agent_with_approval = build_graph(human_approval=True)
agent_autonomous    = build_graph(human_approval=False)
