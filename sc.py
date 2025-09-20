"""
Async Wikipedia scraper:
- Reads Social_Science.csv
- Normalizes 'Last, First' -> 'First Last'
- Attempts direct page; if disambiguation/404, falls back to Wikipedia Search API
- Saves HTML + meta.json under wikipedia_pages/<ID>/
- Concurrent with asyncio + aiohttp, polite retries/backoff, redirect-aware
"""

import asyncio
import aiohttp
from aiohttp import ClientResponse
import pandas as pd
from bs4 import BeautifulSoup
import os
import json
import random
import re
import logging
from urllib.parse import quote
from typing import Optional

# ---------------------------
# Config
# ---------------------------
WIKI_BASE = "https://en.wikipedia.org/wiki/"
WIKI_API  = "https://en.wikipedia.org/w/api.php"
OUT_ROOT  = "wikipedia_pages"
CSV_FILE  = "Social_Science.csv"

CONCURRENCY = 8           # tune politely (6–10 is reasonable)
TIMEOUT_SECS = 25
MAX_RETRIES = 3
BACKOFF_BASE = 1.6        # exponential backoff base for 429/5xx
HEADERS = {
    "User-Agent": (
        "HarjyotWikiScraper/1.0 (+https://example.local; contact: you@example.com) "
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
}

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(
    filename='scrape.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ---------------------------
# Utils
# ---------------------------
def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "", str(name))

def normalize_name(name: str) -> str:
    """Convert 'Last, First Middle' -> 'First Middle Last' if comma present."""
    if not isinstance(name, str):
        return ""
    name = name.strip()
    if "," in name:
        last, first = [p.strip() for p in name.split(",", 1)]
        if first and last:
            return f"{first} {last}".strip()
    return name

def guess_id(row) -> str:
    candidates = [
        "Profile.ID", "Profile.Id", "ProfileId", "ID", "Id", "id",
        "AstronautID", "Astronaut.Id", "PersonID"
    ]
    for c in candidates:
        if c in row and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c]).strip()
    nm = normalize_name(row.get("Profile.Name", ""))
    return sanitize_filename(nm) or "unknown"

def is_disambiguation_page(soup: BeautifulSoup) -> bool:
    hatnotes = soup.select(".hatnote, .hatnote.navigation-not-searchable")
    if any("disambiguation" in h.get_text(strip=True).lower() for h in hatnotes):
        return True
    cats = soup.select("#mw-normal-catlinks ul li a")
    if any("disambiguation" in c.get_text(strip=True).lower() for c in cats):
        return True
    return False

# ---------------------------
# HTTP helpers (async)
# ---------------------------
async def http_get(session: aiohttp.ClientSession, url: str) -> Optional[ClientResponse]:
    """GET with polite retries on 429/5xx."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = await session.get(url, allow_redirects=True, timeout=TIMEOUT_SECS)
            # Retry on 429/5xx
            if resp.status in (429, 500, 502, 503, 504):
                await resp.release()
                sleep_s = (BACKOFF_BASE ** attempt) + random.uniform(0, 0.5)
                await asyncio.sleep(sleep_s)
                continue
            return resp
        except (aiohttp.ClientError, asyncio.TimeoutError):
            sleep_s = (BACKOFF_BASE ** attempt) + random.uniform(0, 0.5)
            await asyncio.sleep(sleep_s)
    return None

async def wiki_search_best_title(session: aiohttp.ClientSession, query: str) -> Optional[str]:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": 1,
        "format": "json",
        "utf8": 1
    }
    try:
        async with session.get(WIKI_API, params=params, timeout=TIMEOUT_SECS) as r:
            if r.status == 200:
                data = await r.json()
                hits = data.get("query", {}).get("search", [])
                if hits:
                    title = hits[0].get("title", "").strip()
                    return title or None
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return None
    return None

# ---------------------------
# Core per-person scrape
# ---------------------------
async def scrape_one(row: dict, session: aiohttp.ClientSession, sem: asyncio.Semaphore):
    raw_name = row.get("Profile.Name", "") or row.get("Name", "") or row.get("FullName", "")
    norm_name = normalize_name(raw_name)
    person_id = guess_id(row)

    meta = {
        "raw_name": raw_name,
        "normalized_name": norm_name,
        "attempted_titles": [],
        "requested_url": None,
        "final_url": None,
        "status": None,
        "notes": []
    }

    if not isinstance(norm_name, str) or not norm_name.strip():
        logging.warning(f"Skipping row with empty name, id={person_id}")
        return

    # Ensure folder
    person_dir = os.path.join(OUT_ROOT, sanitize_filename(person_id))
    os.makedirs(person_dir, exist_ok=True)

    async with sem:
        # Small jitter between requests under concurrency to spread load
        await asyncio.sleep(random.uniform(0.05, 0.2))

        # 1) Try direct page
        title_enc = quote(norm_name.replace(" ", "_"))
        url = f"{WIKI_BASE}{title_enc}"
        meta["requested_url"] = url
        meta["attempted_titles"].append(norm_name)

        resp = await http_get(session, url)
        print(f"[{person_id}] GET {url} -> {resp.status if resp else 'no response'}")

        need_fallback = True
        soup = None

        if resp and resp.status == 200:
            html = await resp.text()
            soup = BeautifulSoup(html, "html.parser")
            if not is_disambiguation_page(soup):
                need_fallback = False
            else:
                meta["notes"].append("Disambiguation detected; using search fallback.")
        elif resp and resp.status == 404:
            meta["notes"].append("Direct page 404; using search fallback.")

        # 2) Search fallback
        if need_fallback:
            best_title = await wiki_search_best_title(session, norm_name)
            if best_title:
                meta["attempted_titles"].append(best_title)
                best_enc = quote(best_title.replace(" ", "_"))
                url = f"{WIKI_BASE}{best_enc}"
                resp = await http_get(session, url)
                print(f"[{person_id}] Fallback {url} -> {resp.status if resp else 'no response'}")
                if resp and resp.status == 200:
                    html = await resp.text()
                    soup = BeautifulSoup(html, "html.parser")
                else:
                    meta["status"] = f"http_{resp.status if resp else 'noresp'}"
                    with open(os.path.join(person_dir, "meta.json"), "w", encoding="utf-8") as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)
                    return
            else:
                meta["status"] = "search_failed"
                meta["notes"].append("Wikipedia search returned no results.")
                with open(os.path.join(person_dir, "meta.json"), "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                logging.warning(f"No search result for: {norm_name}")
                return

        if not resp or resp.status != 200 or soup is None:
            meta["status"] = f"http_{resp.status if resp else 'noresp'}"
            with open(os.path.join(person_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            logging.warning(f"Failed to retrieve page for {norm_name}.")
            return

        # Final URL after redirects
        meta["final_url"] = str(resp.url)
        meta["status"] = "ok"

        # Save HTML
        heading = soup.find("h1", id="firstHeading")
        title_text = heading.get_text(strip=True) if heading else norm_name
        titled_filename = sanitize_filename(f"{title_text}.html") or "page.html"

        idx_path = os.path.join(person_dir, "index.html")
        with open(idx_path, "w", encoding="utf-8") as f:
            f.write(soup.prettify())

        titled_path = os.path.join(person_dir, titled_filename)
        if titled_path != idx_path:
            with open(titled_path, "w", encoding="utf-8") as f:
                f.write(soup.prettify())

        # Save meta
        with open(os.path.join(person_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        logging.info(f"Saved {norm_name} -> {person_dir}")

# ---------------------------
# Orchestrator
# ---------------------------
async def main_async(concurrency: int = CONCURRENCY):
    df = pd.read_csv(CSV_FILE)

    # Choose name column (keep your original preference)
    name_col_candidates = ["Profile.Name", "Name", "FullName"]
    if not any(c in df.columns for c in name_col_candidates):
        raise ValueError(f"Expected one of {name_col_candidates} in CSV.")

    os.makedirs(OUT_ROOT, exist_ok=True)

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=TIMEOUT_SECS, sock_read=TIMEOUT_SECS)
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    sem = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession(headers=HEADERS, timeout=timeout, connector=connector, trust_env=True) as session:
        tasks = []
        for _, row in df.iterrows():
            tasks.append(asyncio.create_task(scrape_one(row, session, sem)))
        # Run with progress-friendly gather
        await asyncio.gather(*tasks)

def run():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("Cancelled by user.")

if __name__ == "__main__":
    run()
