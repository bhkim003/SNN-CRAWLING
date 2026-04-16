#!/usr/bin/env python3
"""
SNN Paper Tracker — build_site.py

Sources
-------
1. Semantic Scholar (primary) – covers arXiv, IEEE Xplore, ACM DL,
   NeurIPS/ICLR/ICML proceedings, Nature/Science, PubMed, and more.
   Google Scholar does not provide a public API; direct scraping is
   blocked in CI. Semantic Scholar indexes the same corpus and is free.
2. arXiv (supplementary) – catches the latest preprints before
   Semantic Scholar finishes indexing them.

Output
------
docs/papers.json   machine-readable dataset
docs/index.html    rendered GitHub Pages site
"""
from __future__ import annotations

import datetime as dt
import html
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"

ARXIV_API = "https://export.arxiv.org/api/query"
S2_API    = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "title,authors,year,venue,publicationDate,externalIds"

MAX_S2    = int(os.getenv("SNN_MAX_S2",    "10000"))
MAX_ARXIV = int(os.getenv("SNN_MAX_ARXIV", "500"))
BATCH     = 100
S2_API_KEY = os.getenv("S2_API_KEY", "")   # free key → higher rate limits


# ---------------------------------------------------------------------------
# Category rules  (first match wins; Etc is the fallback)
# ---------------------------------------------------------------------------
CATEGORY_RULES: dict[str, list[str]] = {
    "LLM": [
        r"\bllm\b", r"large language model", r"language model",
        r"\bgpt[\b\-]", r"\bbert\b", r"\btransformer\b",
        r"in-context learning", r"\bprompt\b", r"foundation model",
        r"instruction.tun", r"fine.tun",
    ],
    "Object Detection": [
        r"object detection", r"\byolo\b", r"\br-cnn\b", r"faster r-cnn",
        r"\bssd\b", r"anchor.free detection", r"bounding box",
    ],
    "Drone": [
        r"\bdrone\b", r"\buav\b", r"unmanned aerial", r"\bquadcopter\b",
        r"aerial robot", r"autonomous flight",
    ],
    "Event-based Vision": [
        r"event.based", r"event.driven vision", r"\bdvs\b",
        r"dynamic vision sensor", r"neuromorphic vision",
        r"silicon retina", r"event camera",
    ],
    "Neuromorphic Hardware": [
        r"neuromorphic hardware", r"neuromorphic chip", r"\bmemristor\b",
        r"\bfpga\b", r"\bvlsi\b", r"\basic\b",
        r"\bloihi\b", r"\btruenorth\b", r"\bspinnaker\b",
        r"hardware accelerat", r"on.chip learning",
    ],
    "ANN-to-SNN Conversion": [
        r"ann.to.snn", r"ann.snn conversion", r"convert.*ann.*snn",
        r"rate coding", r"firing rate conversion",
    ],
    "Learning Rules": [
        r"\bstdp\b", r"surrogate gradient", r"backpropagat.*snn",
        r"spike.timing.dependent", r"hebbian", r"synaptic plasticity",
        r"online learning.*spiking",
    ],
    "Reinforcement Learning": [
        r"reinforcement learning", r"policy gradient", r"\bppo\b",
        r"\bdqn\b", r"actor.critic", r"reward.based",
    ],
    "Robotics & Control": [
        r"\brobot", r"robotic", r"manipulation task",
        r"\bnavigation\b", r"\bslam\b", r"locomotion", r"actuator",
    ],
    "Segmentation": [
        r"segmentation", r"semantic segmentation",
        r"instance segmentation", r"\bpanoptic\b",
    ],
    "Medical & BCI": [
        r"\bmedical\b", r"\bhealthcare\b", r"\becg\b", r"\beeg\b",
        r"\bdiagnosis\b", r"\bbiomedical\b", r"brain.computer interface",
        r"\bseizure\b", r"\bbci\b", r"neural decod",
    ],
    "Speech & Audio": [
        r"\bspeech\b", r"\baudio\b", r"keyword spotting",
        r"\bvoice\b", r"sound classif", r"\basr\b", r"automatic speech",
    ],
    "Image Classification": [
        r"image classif", r"\bcifar\b", r"\bimagenet\b",
        r"\bmnist\b", r"\bn-mnist\b",
    ],
    "NLP": [
        r"\bnlp\b", r"natural language processing",
        r"text classif", r"sentiment analysis",
        r"named entity", r"machine translation",
    ],
    "Time Series": [
        r"time.series", r"\bforecast", r"anomaly detection",
        r"temporal data", r"sequential data",
    ],
}


# ---------------------------------------------------------------------------
# Semantic Scholar fetcher
# ---------------------------------------------------------------------------
def fetch_s2_entries() -> list[dict]:
    entries: list[dict] = []
    headers: dict[str, str] = {
        "User-Agent": "SNN-Paper-Tracker/2.0 (https://github.com/bhkim003/SNN-CRAWLING)"
    }
    if S2_API_KEY:
        headers["x-api-key"] = S2_API_KEY

    offset = 0
    print(f"[INFO] Fetching from Semantic Scholar (up to {MAX_S2} papers)…", file=sys.stderr)

    while len(entries) < MAX_S2:
        params = {
            "query": "spiking neural network",
            "fields": S2_FIELDS,
            "offset": str(offset),
            "limit": str(min(BATCH, MAX_S2 - len(entries))),
        }
        url = f"{S2_API}?{urllib.parse.urlencode(params)}"
        data: dict = {}

        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=40) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    wait = 30 * (attempt + 1)
                    print(f"[WARN] S2 rate limit hit, waiting {wait}s…", file=sys.stderr)
                    time.sleep(wait)
                elif attempt == 3:
                    print(f"[WARN] S2 failed after 4 attempts: {exc}", file=sys.stderr)
                    return entries
                else:
                    time.sleep(5)
            except Exception as exc:
                if attempt == 3:
                    print(f"[WARN] S2 fetch error: {exc}", file=sys.stderr)
                    return entries
                time.sleep(5)

        papers = data.get("data", [])
        if not papers:
            break

        total_available = data.get("total", 0)

        for p in papers:
            title = (p.get("title") or "").strip()
            if not title:
                continue
            authors = p.get("authors") or []
            first_author = authors[0].get("name", "Unknown") if authors else "Unknown"
            venue = (p.get("venue") or "").strip()
            pub_date = (p.get("publicationDate") or "").strip()
            year = p.get("year")

            ext = p.get("externalIds") or {}
            if ext.get("ArXiv"):
                link = f"https://arxiv.org/abs/{ext['ArXiv']}"
            elif ext.get("DOI"):
                link = f"https://doi.org/{ext['DOI']}"
            else:
                pid = p.get("paperId", "")
                link = f"https://www.semanticscholar.org/paper/{pid}"

            if not venue:
                venue = "arXiv" if ext.get("ArXiv") else (str(year) if year else "")
            if not pub_date and year:
                pub_date = f"{year}-01-01"

            entries.append({
                "first_author": first_author,
                "title": " ".join(title.split()),
                "venue": venue,
                "link": link,
                "published": pub_date,
            })

        offset += len(papers)
        print(f"[INFO] S2: {len(entries)}/{min(MAX_S2, total_available)} fetched", file=sys.stderr)

        if len(papers) < BATCH or offset >= total_available:
            break

        time.sleep(1.1 if not S2_API_KEY else 0.3)

    return entries


# ---------------------------------------------------------------------------
# arXiv supplementary fetcher
# ---------------------------------------------------------------------------
def fetch_arxiv_entries() -> list[dict]:
    entries: list[dict] = []
    print(f"[INFO] Fetching from arXiv (up to {MAX_ARXIV} recent preprints)…", file=sys.stderr)

    for start in range(0, MAX_ARXIV, BATCH):
        params = {
            "search_query": 'all:"spiking neural network"',
            "start": str(start),
            "max_results": str(min(BATCH, MAX_ARXIV - start)),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
        data = None

        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    data = resp.read()
                break
            except Exception as exc:
                if attempt == 2:
                    print(f"[WARN] arXiv error: {exc}", file=sys.stderr)
                    return entries
                time.sleep(3)

        if not data:
            break

        ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
        root = ET.fromstring(data)
        batch = root.findall("a:entry", ns)
        if not batch:
            break

        for e in batch:
            title = (e.findtext("a:title", default="", namespaces=ns) or "").strip()
            if not title:
                continue
            auths = e.findall("a:author", ns)
            first_author = (
                (auths[0].findtext("a:name", default="", namespaces=ns) or "Unknown").strip()
                if auths else "Unknown"
            )
            pub = (e.findtext("a:published", default="", namespaces=ns) or "").strip()
            journal_ref = (e.findtext("x:journal_ref", default="", namespaces=ns) or "").strip()
            venue = journal_ref if journal_ref else "arXiv"
            link = ""
            for lnk in e.findall("a:link", ns):
                if lnk.attrib.get("type") == "text/html":
                    link = lnk.attrib.get("href", "")
                    break
            if not link:
                link = (e.findtext("a:id", default="", namespaces=ns) or "").strip()
            entries.append({
                "first_author": first_author,
                "title": " ".join(title.split()),
                "venue": venue,
                "link": link,
                "published": pub,
            })

        time.sleep(3)  # arXiv rate limit

    return entries


# ---------------------------------------------------------------------------
# Classification + dedup + dataset builder
# ---------------------------------------------------------------------------
def classify_paper(title: str) -> str:
    lower = title.lower()
    for category, patterns in CATEGORY_RULES.items():
        for pat in patterns:
            if re.search(pat, lower):
                return category
    return "Etc"


def _parse_date(published: str) -> dt.datetime:
    s = published.strip()
    if not s:
        return dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return dt.datetime.fromisoformat(s)
    except ValueError:
        try:
            return dt.datetime.strptime(s[:10], "%Y-%m-%d").replace(
                tzinfo=dt.timezone.utc
            )
        except ValueError:
            return dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)


def build_dataset(s2_entries: list[dict], arxiv_entries: list[dict]) -> dict:
    categories: dict[str, list[dict]] = {name: [] for name in CATEGORY_RULES}
    categories["Etc"] = []

    seen_titles: set[str] = set()
    for paper in s2_entries + arxiv_entries:   # S2 takes dedup priority
        norm = re.sub(r"\W+", " ", paper["title"].lower()).strip()
        if not norm or norm in seen_titles:
            continue
        seen_titles.add(norm)
        cat = classify_paper(paper["title"])
        categories[cat].append({
            "first_author": paper["first_author"],
            "title": paper["title"],
            "venue": paper["venue"],
            "link": paper["link"],
            "published": paper["published"],
        })

    for cat in categories:
        categories[cat].sort(key=lambda p: _parse_date(p["published"]), reverse=True)

    total = sum(len(v) for v in categories.values())
    print(f"[INFO] Total unique papers: {total}", file=sys.stderr)
    return {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "sources": ["Semantic Scholar", "arXiv"],
        "total_papers": total,
        "categories": categories,
    }


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------
def render_html(dataset: dict) -> str:
    categories = dataset["categories"]
    generated = dataset["generated_at_utc"][:19].replace("T", " ") + " UTC"
    total = dataset["total_papers"]

    nav_items: list[str] = []
    sections: list[str] = []

    for cat, papers in categories.items():
        cid = re.sub(r"[^a-zA-Z0-9]+", "-", cat)
        count = len(papers)
        nav_items.append(
            f'<a href="#{cid}" class="chip">'
            f'{html.escape(cat)}'
            f'<span class="badge">{count}</span>'
            f'</a>'
        )

        rows: list[str] = []
        for p in papers:
            date_str = p["published"][:10] if p["published"] else "—"
            rows.append(
                f'<tr>'
                f'<td class="col-author">{html.escape(p["first_author"])}</td>'
                f'<td class="col-title">'
                f'<a href="{html.escape(p["link"])}" target="_blank" rel="noopener">'
                f'{html.escape(p["title"])}</a></td>'
                f'<td class="col-venue">{html.escape(p["venue"])}</td>'
                f'<td class="col-date">{html.escape(date_str)}</td>'
                f'</tr>'
            )
        if not rows:
            rows.append('<tr><td colspan="4" class="empty">No papers found yet.</td></tr>')

        sections.append(f"""
    <section id="{cid}">
      <h2>{html.escape(cat)} <span class="count">({count})</span></h2>
      <div class="table-wrap">
        <table>
          <thead><tr>
            <th class="col-author">1st Author</th>
            <th class="col-title">Title</th>
            <th class="col-venue">Journal / Conference</th>
            <th class="col-date">Date</th>
          </tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </section>""")

    nav_html = "\n    ".join(nav_items)
    sections_html = "\n".join(sections)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>SNN Paper Tracker</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      --bg: #f6f8fa;
      --surface: #ffffff;
      --border: #d0d7de;
      --accent: #2d6be4;
      --text: #24292f;
      --muted: #57606a;
      --row-hover: #f0f6ff;
      --header-bg: #0d1117;
    }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                   'Helvetica Neue', Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.6;
    }}

    .hero {{
      background: var(--header-bg);
      color: #e6edf3;
      padding: 36px 24px 28px;
      border-bottom: 1px solid #30363d;
    }}
    .hero h1 {{ font-size: 1.8rem; font-weight: 700; margin-bottom: 6px; }}
    .hero h1 span {{ color: #58a6ff; }}
    .hero .meta {{ font-size: 0.85rem; color: #8b949e; }}
    .hero .meta b {{ color: #c9d1d9; }}

    .search-bar {{
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 12px 24px;
      position: sticky;
      top: 0;
      z-index: 100;
      box-shadow: 0 1px 3px rgba(0,0,0,.06);
    }}
    .search-bar input {{
      width: 100%;
      max-width: 600px;
      padding: 8px 14px;
      border: 1px solid var(--border);
      border-radius: 6px;
      font-size: 0.9rem;
      outline: none;
    }}
    .search-bar input:focus {{
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(45,107,228,.15);
    }}

    .nav-section {{
      padding: 14px 24px;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
    }}
    .nav-chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 5px 12px;
      border: 1px solid var(--border);
      border-radius: 20px;
      text-decoration: none;
      color: var(--text);
      font-size: 0.82rem;
      font-weight: 500;
      background: var(--bg);
      transition: all .15s;
    }}
    .chip:hover {{
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }}
    .chip:hover .badge {{ background: rgba(255,255,255,.25); color: #fff; }}
    .badge {{
      background: #e8edf5;
      color: var(--muted);
      border-radius: 10px;
      padding: 1px 7px;
      font-size: 0.75rem;
      font-weight: 600;
    }}

    .content {{ padding: 24px; max-width: 1400px; margin: 0 auto; }}
    section {{ margin-bottom: 40px; scroll-margin-top: 56px; }}
    section h2 {{
      font-size: 1.1rem;
      font-weight: 700;
      margin-bottom: 10px;
      padding-bottom: 6px;
      border-bottom: 2px solid var(--accent);
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .count {{ font-size: 0.85rem; font-weight: 400; color: var(--muted); }}

    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--border);
      border-radius: 8px;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}
    thead {{ background: #f6f8fa; }}
    th {{
      padding: 10px 12px;
      text-align: left;
      font-weight: 600;
      color: var(--muted);
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
      font-size: 0.77rem;
      text-transform: uppercase;
      letter-spacing: .5px;
    }}
    td {{
      padding: 9px 12px;
      border-bottom: 1px solid #f0f0f0;
      vertical-align: top;
    }}
    tbody tr:last-child td {{ border-bottom: none; }}
    tbody tr:hover td {{ background: var(--row-hover); }}

    .col-author {{
      width: 145px;
      color: var(--muted);
      font-size: 0.82rem;
      white-space: nowrap;
    }}
    .col-title a {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 500;
    }}
    .col-title a:hover {{ text-decoration: underline; }}
    .col-venue {{
      width: 200px;
      color: var(--muted);
      font-size: 0.82rem;
    }}
    .col-date {{
      width: 95px;
      color: var(--muted);
      font-size: 0.8rem;
      white-space: nowrap;
    }}
    .empty {{ text-align: center; color: var(--muted); padding: 24px; }}
    .hidden {{ display: none !important; }}

    @media (max-width: 700px) {{
      .col-venue, .col-date {{ display: none; }}
      .hero h1 {{ font-size: 1.3rem; }}
    }}
  </style>
</head>
<body>

<header class="hero">
  <h1>⚡ <span>Spiking Neural Network</span> Papers</h1>
  <p class="meta">
    <b>{total:,}</b> papers &nbsp;·&nbsp; Updated <b>{generated}</b>
    &nbsp;·&nbsp; Sources: Semantic Scholar + arXiv
  </p>
</header>

<div class="search-bar">
  <input type="search" id="search"
         placeholder="Search by title, author, venue…" autocomplete="off">
</div>

<nav class="nav-section">
  <div class="nav-chips">
    {nav_html}
  </div>
</nav>

<main class="content">
  {sections_html}
</main>

<script>
  const input = document.getElementById('search');
  input.addEventListener('input', () => {{
    const q = input.value.toLowerCase().trim();
    document.querySelectorAll('tbody tr').forEach(tr => {{
      tr.classList.toggle('hidden', Boolean(q) && !tr.textContent.toLowerCase().includes(q));
    }});
    document.querySelectorAll('section').forEach(sec => {{
      const visible = sec.querySelectorAll('tbody tr:not(.hidden)').length;
      sec.classList.toggle('hidden', Boolean(q) && visible === 0);
    }});
  }});
</script>

</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    existing_json = DOCS_DIR / "papers.json"

    try:
        s2_entries    = fetch_s2_entries()
        arxiv_entries = fetch_arxiv_entries()
        dataset = build_dataset(s2_entries, arxiv_entries)
    except Exception as exc:
        print(f"[WARN] Fetch failed: {exc}. Falling back to cached data.", file=sys.stderr)
        if existing_json.exists():
            dataset = json.loads(existing_json.read_text(encoding="utf-8"))
            dataset["generated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        else:
            dataset = build_dataset([], [])

    existing_json.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DOCS_DIR / "index.html").write_text(render_html(dataset), encoding="utf-8")
    print(f"[INFO] Done. {dataset['total_papers']} papers written to docs/", file=sys.stderr)


if __name__ == "__main__":
    main()
