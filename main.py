import uuid
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse

from graph.graph import agent_with_approval, agent_autonomous

load_dotenv()

app = FastAPI(
    title="Research Agent",
    description="AI research assistant with streaming, human-in-the-loop, and Pinecone memory",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    topic: str
    thread_id: str | None = None
    autonomous: bool = False


class ApprovalRequest(BaseModel):
    thread_id: str
    approved_urls: list[str]


class ResearchResponse(BaseModel):
    thread_id: str
    report: str
    sources: list[str]
    chunks_ingested: int


def extract_sources(result: dict) -> list[str]:
    """
    On full pipeline  → sources from fetched_articles
    On cache hit      → sources parsed from retrieved_context
    """
    # Path 1 — full pipeline
    articles = result.get("fetched_articles", [])
    if articles:
        sources = [a["url"] for a in articles if a.get("url")]
        return list(dict.fromkeys(sources))

    # Path 2 — cache hit, parse from retrieved_context
    sources = []
    context = result.get("retrieved_context", "")
    for line in context.splitlines():
        if line.startswith("Source:"):
            url = line.replace("Source:", "").strip()
            if url and url != "unknown":
                sources.append(url)
    return list(dict.fromkeys(sources))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/research/start")
async def start_research(req: ResearchRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "topic":             req.topic,
        "search_results":    [],
        "approved_urls":     [],
        "fetched_articles":  [],
        "ingested_chunks":   0,
        "retrieved_context": "",
        "report":            "",
        "messages":          [],
    }

    try:
        if req.autonomous:
            result = agent_autonomous.invoke(initial_state, config=config)
            return ResearchResponse(
                thread_id=thread_id,
                report=result.get("report", ""),
                sources=extract_sources(result),
                chunks_ingested=result.get("ingested_chunks", 0),
            )
        else:
            result = agent_with_approval.invoke(initial_state, config=config)
            return {
                "thread_id":      thread_id,
                "status":         "awaiting_approval",
                "search_results": result.get("search_results", []),
                "message":        "Review sources and call POST /research/approve to continue",
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research/approve", response_model=ResearchResponse)
async def approve_and_continue(req: ApprovalRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    try:
        agent_with_approval.update_state(
            config,
            {"approved_urls": req.approved_urls}
        )
        result = agent_with_approval.invoke(None, config=config)
        return ResearchResponse(
            thread_id=req.thread_id,
            report=result.get("report", ""),
            sources=extract_sources(result),
            chunks_ingested=result.get("ingested_chunks", 0),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research/stream")
async def stream_research(req: ResearchRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    config    = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "topic":             req.topic,
        "search_results":    [],
        "approved_urls":     [],
        "fetched_articles":  [],
        "ingested_chunks":   0,
        "retrieved_context": "",
        "report":            "",
        "messages":          [],
    }

    async def event_generator():
        try:
            async for event in agent_autonomous.astream_events(
                initial_state, config=config, version="v2"
            ):
                kind = event.get("event", "")
                name = event.get("name", "")

                if kind == "on_chain_start" and name in (
                    "cache_check", "search", "fetch",
                    "ingest", "retrieve", "writer"
                ):
                    yield {
                        "data": json.dumps({
                            "type": "node_start",
                            "node": name,
                        })
                    }

                if kind == "on_chat_model_stream":
                    chunk = event["data"].get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        yield {
                            "data": json.dumps({
                                "type":  "token",
                                "token": chunk.content,
                            })
                        }

                if kind == "on_chain_end" and name == "writer":
                    output = event.get("data", {}).get("output", {})
                    yield {
                        "data": json.dumps({
                            "type":   "done",
                            "report": output.get("report", ""),
                        })
                    }
        except Exception as e:
            yield {"data": json.dumps({"type": "error", "message": str(e)})}

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
