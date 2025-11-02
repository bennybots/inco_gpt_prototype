import trafilatura
from bs4 import BeautifulSoup

def extract_readable(html: str, url: str | None = None) -> str:
    txt = trafilatura.extract(html, url=url, include_comments=False) or ""
    if len((txt or "").strip()) > 300:
        return txt
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "footer", "header"]):
        tag.decompose()
    lines = [ch.strip() for ch in soup.get_text("\n").splitlines() if ch.strip()]
    return "\n".join(lines)
