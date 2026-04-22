import httpx
from bs4 import BeautifulSoup


def _extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    lines = [line.strip() for line in soup.get_text(separator="\n").splitlines() if line.strip()]
    return "\n".join(lines)


def fetch_articles(urls: list[str]) -> list[dict]:
    """
    Fetch all URLs sequentially using httpx (sync).
    Never crashes — failed URLs are skipped gracefully.
    """
    if not urls:
        return []

    headers = {"User-Agent": "Mozilla/5.0 (research-agent/1.0)"}
    results = []

    with httpx.Client(
        headers=headers,
        timeout=15,
        follow_redirects=True,
    ) as client:
        for url in urls:
            if not url:
                continue
            try:
                response = client.get(url)
                response.raise_for_status()
                text = _extract_text(response.text)
                if not text:
                    print(f"[fetch] ⚠️  Empty content: {url[:60]}")
                    continue
                soup  = BeautifulSoup(response.text, "html.parser")
                title = soup.title.string.strip() if soup.title else url
                results.append({
                    "url":       url,
                    "title":     title,
                    "full_text": text[:8000],
                })
                print(f"[fetch] ✅ {title[:60]}")
            except httpx.HTTPStatusError as e:
                print(f"[fetch] ⚠️  HTTP {e.response.status_code}: {url[:60]}")
            except httpx.TimeoutException:
                print(f"[fetch] ⚠️  Timeout: {url[:60]}")
            except Exception as e:
                print(f"[fetch] ⚠️  Failed: {url[:60]} → {type(e).__name__}: {e}")

    print(f"[fetch] Successfully fetched {len(results)}/{len(urls)} articles")
    return results
