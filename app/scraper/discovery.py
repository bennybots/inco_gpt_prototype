import requests
from urllib.parse import urljoin

UA = {"User-Agent": "Mozilla/5.0"}

def get_robots(base_url: str) -> str | None:
    try:
        r = requests.get(urljoin(base_url, "/robots.txt"), headers=UA, timeout=10)
        if r.status_code == 200:
            return r.text
    except Exception:
        pass
    return None
