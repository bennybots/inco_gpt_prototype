from playwright.sync_api import sync_playwright
import requests

DEFAULT_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36")

def fetch_static(url: str, timeout=20) -> str:
    r = requests.get(url, headers={"User-Agent": DEFAULT_UA}, timeout=timeout)
    r.raise_for_status()
    return r.text

def fetch_dynamic(url: str, wait_ms=1200) -> str:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(user_agent=DEFAULT_UA, java_script_enabled=True)
        page = ctx.new_page()
        page.goto(url, wait_until="networkidle")
        page.wait_for_timeout(wait_ms)
        html = page.content()
        browser.close()
        return html

def smart_fetch(url: str) -> tuple[str, str]:
    try:
        html = fetch_static(url)
        if html.count("<script") > 8 or len(html) < 1500:
            html = fetch_dynamic(url)
            return html, "dynamic"
        return html, "static"
    except Exception:
        html = fetch_dynamic(url)
        return html, "dynamic"
