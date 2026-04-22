from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

_search_tool = None


def get_search_tool():
    global _search_tool
    if _search_tool is None:
        _search_tool = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
        )
    return _search_tool


def run_search(query: str) -> list[dict]:
    """
    Search the web for a topic.
    Returns a list of {title, url, content} dicts.
    """
    try:
        results = get_search_tool().invoke(query)
        if not results:
            return []
        return [
            {
                "title":   r.get("title", "No title"),
                "url":     r.get("url", ""),
                "content": r.get("content", ""),
            }
            for r in results
            if r.get("url")
        ]
    except Exception as e:
        print(f"[search] Error: {e}")
        return []
