"""
Test script for the Research Agent.
Run this after uvicorn is running.

Tests 3 flows:
1. Autonomous mode — runs end to end
2. Human-in-the-loop — start → approve → report
3. Streaming — see tokens arrive in real time
"""

import requests
import json

BASE = "http://127.0.0.1:8000"
TOPIC = "LangGraph multi-agent systems"


def test_autonomous():
    print("=" * 60)
    print("TEST 1 — Autonomous mode")
    print("=" * 60)

    resp = requests.post(f"{BASE}/research/start", json={
        "topic": TOPIC,
        "autonomous": True,
    })
    data = resp.json()

    print(f"Thread ID:       {data['thread_id']}")
    print(f"Chunks ingested: {data['chunks_ingested']}")
    print(f"Sources used:    {len(data['sources'])}")
    print()
    print("── REPORT (first 500 chars) ──")
    print(data["report"][:500])


def test_human_in_the_loop():
    print("=" * 60)
    print("TEST 2 — Human-in-the-loop")
    print("=" * 60)

    # Step 1 — Start (pauses after search)
    resp = requests.post(f"{BASE}/research/start", json={
        "topic": TOPIC,
        "autonomous": False,
    })
    data = resp.json()

    print(f"Status:    {data['status']}")
    print(f"Thread ID: {data['thread_id']}")
    print(f"Found {len(data['search_results'])} sources:")
    for r in data["search_results"]:
        print(f"  - {r['title']}: {r['url']}")

    # Step 2 — Approve first 3 URLs
    approved = [r["url"] for r in data["search_results"][:3]]
    print(f"\nApproving {len(approved)} URLs...")

    resp2 = requests.post(f"{BASE}/research/approve", json={
        "thread_id":     data["thread_id"],
        "approved_urls": approved,
    })
    result = resp2.json()

    print(f"Chunks ingested: {result['chunks_ingested']}")
    print()
    print("── REPORT (first 500 chars) ──")
    print(result["report"][:500])


def test_streaming():
    print("=" * 60)
    print("TEST 3 — Streaming mode")
    print("=" * 60)

    with requests.post(
        f"{BASE}/research/stream",
        json={"topic": TOPIC},
        stream=True
    ) as resp:
        for line in resp.iter_lines():
            if line and line.startswith(b"data:"):
                payload = json.loads(line[5:].strip())
                event_type = payload.get("type")

                if event_type == "node_start":
                    print(f"\n[{payload['node']}] starting...")
                elif event_type == "token":
                    print(payload["token"], end="", flush=True)
                elif event_type == "done":
                    print("\n\n── Stream complete ──")
                elif event_type == "error":
                    print(f"\n[ERROR] {payload['message']}")


if __name__ == "__main__":
    import sys
    test = sys.argv[1] if len(sys.argv) > 1 else "autonomous"

    if test == "autonomous":
        test_autonomous()
    elif test == "hitl":
        test_human_in_the_loop()
    elif test == "stream":
        test_streaming()
    else:
        print("Usage: python3 test.py [autonomous|hitl|stream]")
