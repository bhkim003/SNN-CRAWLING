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
import subprocess
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
S2_FIELDS = "title,abstract,authors,year,venue,publicationDate,externalIds"

SEARCH_TERMS = [
    "spiking neural network",
    "SNN",
    "spike-based",
]
S2_QUERY = " OR ".join(SEARCH_TERMS)
ARXIV_QUERY = 'all:"spiking neural network" OR all:"spike-based" OR all:"snn"'

MAX_S2    = int(os.getenv("SNN_MAX_S2",    "10000"))
MAX_ARXIV = int(os.getenv("SNN_MAX_ARXIV", "500"))
BATCH     = 100
S2_API_KEY = os.getenv("S2_API_KEY", "")   # free key → higher rate limits
ARXIV_SLEEP_SECONDS = float(os.getenv("SNN_ARXIV_SLEEP", "0.1"))


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

GENERIC_TOKENS = {
    "spiking", "spike", "spikes", "neural", "network", "networks", "based",
    "using", "approach", "method", "methods", "learning", "model", "models",
    "analysis", "results", "paper", "snn", "system", "systems",
}

STOPWORDS = {
    "about", "above", "after", "again", "against", "among", "and", "are", "as", "at", "been",
    "before", "being", "between", "both", "but", "can", "could", "did", "does", "doing",
    "during", "each", "for", "from", "have", "in", "into", "is", "it", "its", "more",
    "most", "much", "of", "on", "or", "our", "out", "over", "per", "such", "than",
    "that", "the", "their", "there", "these", "they", "this", "those", "through", "to",
    "under", "using", "via", "very", "what", "when", "where", "which", "while", "with",
    "without", "would",
}

DYNAMIC_TOKEN_LABELS = {
    "gesture": "Gesture Recognition",
    "navigation": "Autonomous Navigation",
    "autonomous": "Autonomous Systems",
    "memristor": "Memristive Devices",
    "edge": "Edge AI",
    "dataset": "Datasets & Benchmarks",
    "federated": "Federated Learning",
    "quantization": "Quantization",
    "compression": "Model Compression",
    "optical": "Optical Computing",
    "security": "Security & Robustness",
    "adversarial": "Security & Robustness",
    "multimodal": "Multimodal Learning",
    "graph": "Graph Neural Models",
    "attention": "Attention Mechanisms",
}

DYNAMIC_TOPIC_BLACKLIST = {
    "based", "learning", "network", "networks", "spiking", "spike", "spikes",
    "neural", "model", "models", "method", "methods", "approach", "approaches",
    "paper", "system", "systems", "analysis", "study", "studies", "using",
    "toward", "towards", "with", "without", "from", "into", "over", "under",
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
            "query": S2_QUERY,
            "fields": S2_FIELDS,
            "offset": str(offset),
            "limit": str(min(BATCH, MAX_S2 - len(entries))),
        }
        url = f"{S2_API}?{urllib.parse.urlencode(params)}"
        data: dict = {}
        request_ok = False

        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=40) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                request_ok = True
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    retry_after = exc.headers.get("Retry-After")
                    wait = int(retry_after) if retry_after and retry_after.isdigit() else 0
                    if wait > 5:
                        print(f"[WARN] S2 rate limit hit, skipping remaining S2 fetches (Retry-After={wait}s).", file=sys.stderr)
                        return entries
                    if wait > 0:
                        print(f"[WARN] S2 rate limit hit, waiting {wait}s…", file=sys.stderr)
                        time.sleep(wait)
                    else:
                        print("[WARN] S2 rate limit hit, skipping remaining S2 fetches.", file=sys.stderr)
                        return entries
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

        if not request_ok:
            print("[WARN] S2 request failed repeatedly; keeping already-fetched S2 entries.", file=sys.stderr)
            return entries

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
            abstract = " ".join((p.get("abstract") or "").split())
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
                "abstract": abstract,
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
            "search_query": ARXIV_QUERY,
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
            abstract = " ".join((e.findtext("a:summary", default="", namespaces=ns) or "").split())
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
                "abstract": abstract,
                "venue": venue,
                "link": link,
                "published": pub,
            })

        if ARXIV_SLEEP_SECONDS > 0:
            time.sleep(ARXIV_SLEEP_SECONDS)  # arXiv rate limit

    return entries


# ---------------------------------------------------------------------------
# Classification + dedup + dataset builder
# ---------------------------------------------------------------------------
def infer_dynamic_category(title: str, abstract: str) -> str:
    text = f"{title} {abstract}".lower()
    for token, label in DYNAMIC_TOKEN_LABELS.items():
        if re.search(rf"\b{re.escape(token)}\b", text):
            return label

    phrase_sources = [
        (title, 3),
        (abstract[:600], 1),
    ]
    phrase_scores: dict[str, int] = {}

    for source_text, weight in phrase_sources:
        words = [w for w in re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", source_text.lower()) if w not in STOPWORDS]
        for size in (2, 3):
            for idx in range(len(words) - size + 1):
                phrase_words = words[idx:idx + size]
                if any(word in GENERIC_TOKENS or word in DYNAMIC_TOPIC_BLACKLIST for word in phrase_words):
                    continue
                if phrase_words[0] in STOPWORDS or phrase_words[-1] in STOPWORDS:
                    continue
                phrase = " ".join(phrase_words)
                phrase_scores[phrase] = phrase_scores.get(phrase, 0) + weight + size

    if not phrase_scores:
        return "General SNN"

    top_phrase = sorted(phrase_scores.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))[0][0]
    top_phrase = re.sub(r"\s+", " ", top_phrase).strip().title()
    return f"Topic: {top_phrase}"


def classify_paper(title: str, abstract: str = "") -> str:
    lower = f"{title} {abstract}".lower()
    for category, patterns in CATEGORY_RULES.items():
        for pat in patterns:
            if re.search(pat, lower):
                return category
    return infer_dynamic_category(title, abstract)


def _parse_date(published: str) -> dt.datetime:
    s = published.strip()
    if not s:
        return dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        parsed = dt.datetime.fromisoformat(s)
        # Normalize to timezone-aware UTC so sort comparisons never mix naive/aware datetimes.
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    except ValueError:
        try:
            return dt.datetime.strptime(s[:10], "%Y-%m-%d").replace(
                tzinfo=dt.timezone.utc
            )
        except ValueError:
            return dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)


def build_dataset(s2_entries: list[dict], arxiv_entries: list[dict]) -> dict:
    categories: dict[str, list[dict]] = {name: [] for name in CATEGORY_RULES}

    seen_titles: set[str] = set()
    for paper in s2_entries + arxiv_entries:   # S2 takes dedup priority
        norm = re.sub(r"\W+", " ", paper["title"].lower()).strip()
        if not norm or norm in seen_titles:
            continue
        seen_titles.add(norm)
        cat = classify_paper(paper["title"], paper.get("abstract", ""))
        if cat not in categories:
            categories[cat] = []
        categories[cat].append({
            "first_author": paper["first_author"],
            "title": paper["title"],
            "abstract": paper.get("abstract", ""),
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


def flatten_dataset_entries(dataset: dict) -> list[dict]:
    entries: list[dict] = []
    categories = dataset.get("categories", {}) or {}
    for papers in categories.values():
        for p in papers:
            entries.append({
                "first_author": p.get("first_author", "Unknown"),
                "title": p.get("title", "").strip(),
                "abstract": p.get("abstract", ""),
                "venue": p.get("venue", ""),
                "link": p.get("link", ""),
                "published": p.get("published", ""),
            })
    return entries


def load_committed_dataset() -> dict | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "show", "HEAD:docs/papers.json"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        return json.loads(result.stdout)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------
def render_html(dataset: dict) -> str:
    categories = dataset["categories"]
    generated = dataset["generated_at_utc"][:19].replace("T", " ") + " UTC"
    total = dataset["total_papers"]
    def make_rows(papers: list[dict], include_category: bool = False) -> str:
        rows: list[str] = []
        for p in papers:
            date_str = p["published"][:10] if p["published"] else "-"
            abstract = p.get("abstract", "").strip() or "Abstract unavailable."
            abstract_title = html.escape(abstract, quote=True)
            category_cell = (
                f'<td class="col-cat">{html.escape(p.get("category", "Etc"))}</td>'
                if include_category else ""
            )
            rows.append(
                f'<tr class="paper-row">'
                f'<td class="col-author">{html.escape(p["first_author"])}</td>'
                f'<td class="col-title">'
                f'<a href="{html.escape(p["link"])}" target="_blank" rel="noopener" title="{abstract_title}">'
                f'{html.escape(p["title"])}</a></td>'
                f'{category_cell}'
                f'<td class="col-venue">{html.escape(p["venue"])}</td>'
                f'<td class="col-date">{html.escape(date_str)}</td>'
                f'</tr>'
            )
        if not rows:
            colspan = "5" if include_category else "4"
            rows.append(f'<tr><td colspan="{colspan}" class="empty">No papers found.</td></tr>')
        return "".join(rows)

    total_papers: list[dict] = []
    for cat, papers in categories.items():
        for p in papers:
            total_papers.append({
                "first_author": p["first_author"],
                "title": p["title"],
                "venue": p["venue"],
                "link": p["link"],
                "published": p["published"],
                "category": cat,
            })
    total_papers.sort(key=lambda p: _parse_date(p["published"]), reverse=True)

    tab_buttons: list[str] = [
        f'<button class="tab-btn active" data-tab="tab-total">TOTAL <span class="badge">{total}</span></button>'
    ]
    tab_panels: list[str] = [f"""
    <section class="panel active" id="tab-total">
      <h2>Total Papers <span class="count">({total})</span></h2>
      <div class="table-wrap">
        <table>
          <thead><tr>
            <th class="col-author">1st Author</th>
            <th class="col-title">Title</th>
            <th class="col-cat">Category</th>
            <th class="col-venue">Journal / Conference</th>
            <th class="col-date">Date</th>
          </tr></thead>
          <tbody>{make_rows(total_papers, include_category=True)}</tbody>
        </table>
      </div>
    </section>"""]

    for cat, papers in categories.items():
        cid = re.sub(r"[^a-zA-Z0-9]+", "-", cat).strip("-").lower() or "etc"
        tab_id = f"tab-{cid}"
        count = len(papers)
        tab_buttons.append(
            f'<button class="tab-btn" data-tab="{tab_id}">{html.escape(cat)} <span class="badge">{count}</span></button>'
        )
        tab_panels.append(f"""
    <section class="panel" id="{tab_id}">
      <h2>{html.escape(cat)} <span class="count">({count})</span></h2>
      <div class="table-wrap">
        <table>
          <thead><tr>
            <th class="col-author">1st Author</th>
            <th class="col-title">Title</th>
            <th class="col-venue">Journal / Conference</th>
            <th class="col-date">Date</th>
          </tr></thead>
          <tbody>{make_rows(papers)}</tbody>
        </table>
      </div>
    </section>""")

    tabs_html = "\n    ".join(tab_buttons)
    panels_html = "\n".join(tab_panels)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>SNN Paper Tracker</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
            --bg: #05070b;
            --surface: #0d121b;
            --surface-2: #111827;
            --border: #2a364d;
            --accent: #65d3ff;
            --accent-2: #6ef7cf;
            --text: #f2f7ff;
            --muted: #b7c3d7;
            --row-hover: #152037;
            --chip-active: #183250;
            --shadow: rgba(0, 0, 0, 0.35);
    }}

    body {{
            font-family: 'Segoe UI', 'Noto Sans KR', 'Apple SD Gothic Neo', sans-serif;
            background:
                radial-gradient(1200px 600px at 5% -10%, #163457 0%, transparent 60%),
                radial-gradient(1000px 500px at 95% 0%, #143a2d 0%, transparent 60%),
                var(--bg);
      color: var(--text);
      line-height: 1.6;
            min-height: 100vh;
    }}

    .hero {{
            background: linear-gradient(120deg, #0f172a 0%, #111a31 55%, #0f1f2c 100%);
            color: var(--text);
            padding: 34px 24px 26px;
            border-bottom: 1px solid var(--border);
    }}
    .hero h1 {{ font-size: 1.8rem; font-weight: 700; margin-bottom: 6px; }}
        .hero h1 span {{ color: var(--accent); }}
        .hero .meta {{ font-size: 0.9rem; color: var(--muted); }}
        .hero .meta b {{ color: var(--accent-2); }}

        .toolbar {{
      background: var(--surface);
      border-bottom: 1px solid var(--border);
            padding: 14px 24px;
      position: sticky;
      top: 0;
      z-index: 100;
            box-shadow: 0 6px 20px var(--shadow);
    }}
        .search-wrap {{ max-width: 760px; }}
        .toolbar input {{
      width: 100%;
            padding: 10px 14px;
      border: 1px solid var(--border);
            border-radius: 10px;
            font-size: 0.92rem;
      outline: none;
            background: var(--surface-2);
            color: var(--text);
    }}
        .toolbar input::placeholder {{ color: #93a0b7; }}
        .toolbar input:focus {{
      border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(101,211,255,.2);
    }}

        .tab-section {{
      padding: 14px 24px;
      background: var(--surface);
      border-bottom: 1px solid var(--border);
    }}
        .tab-list {{
            display: flex;
            flex-wrap: nowrap;
            gap: 8px;
            overflow-x: auto;
            padding-bottom: 4px;
        }}
        .tab-list::-webkit-scrollbar {{ height: 8px; }}
        .tab-list::-webkit-scrollbar-thumb {{ background: #263750; border-radius: 999px; }}
        .tab-btn {{
      display: inline-flex;
      align-items: center;
            gap: 6px;
            padding: 6px 12px;
      border: 1px solid var(--border);
      border-radius: 20px;
      color: var(--text);
      font-size: 0.82rem;
            font-weight: 600;
            background: var(--surface-2);
            transition: all .18s;
            cursor: pointer;
            white-space: nowrap;
    }}
        .tab-btn:hover {{
            background: var(--chip-active);
            border-color: #406080;
        }}
        .tab-btn.active {{
            background: linear-gradient(120deg, #173454 0%, #1f3f68 100%);
            color: #eef9ff;
      border-color: var(--accent);
    }}
    .badge {{
            background: #1f2c43;
            color: #d3e2f8;
      border-radius: 10px;
      padding: 1px 7px;
      font-size: 0.75rem;
      font-weight: 600;
    }}
        .tab-btn.active .badge {{
            background: rgba(101, 211, 255, 0.22);
            color: #dff7ff;
        }}

        .content {{ padding: 24px; max-width: 1460px; margin: 0 auto; }}
        .panel {{ display: none; margin-bottom: 28px; }}
        .panel.active {{ display: block; }}
        .panel h2 {{
      font-size: 1.1rem;
      font-weight: 700;
      margin-bottom: 10px;
      padding-bottom: 6px;
            border-bottom: 2px solid #2b4668;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .count {{ font-size: 0.85rem; font-weight: 400; color: var(--muted); }}

    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--border);
            border-radius: 12px;
            background: var(--surface);
            box-shadow: 0 10px 24px var(--shadow);
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}
        thead {{ background: #101b2f; }}
    th {{
      padding: 10px 12px;
      text-align: left;
      font-weight: 600;
            color: #c5d5ef;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
      font-size: 0.77rem;
      text-transform: uppercase;
      letter-spacing: .5px;
    }}
    td {{
      padding: 9px 12px;
            border-bottom: 1px solid #1b2a3e;
      vertical-align: top;
            color: #e2ebf9;
    }}
    tbody tr:last-child td {{ border-bottom: none; }}
    tbody tr:hover td {{ background: var(--row-hover); }}

    .col-author {{
      width: 145px;
      color: var(--muted);
      font-size: 0.82rem;
      white-space: nowrap;
    }}
        .col-cat {{
            width: 180px;
            color: #c8d7ee;
            font-size: 0.82rem;
            white-space: nowrap;
        }}
    .col-title a {{
      color: var(--accent);
      text-decoration: none;
            font-weight: 600;
    }}
        .col-title a:hover {{ text-decoration: underline; color: #9ee6ff; }}
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
            .col-venue, .col-date, .col-cat {{ display: none; }}
      .hero h1 {{ font-size: 1.3rem; }}
            .toolbar, .tab-section, .content, .hero {{ padding-left: 14px; padding-right: 14px; }}
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

<div class="toolbar">
    <div class="search-wrap">
        <input type="search" id="search"
                     placeholder="Search within current tab (title, author, venue, category)" autocomplete="off">
    </div>
</div>

<nav class="tab-section">
    <div class="tab-list" id="tab-list">
        {tabs_html}
  </div>
</nav>

<main class="content">
    {panels_html}
</main>

<script>
    const tabButtons = Array.from(document.querySelectorAll('.tab-btn'));
    const panels = Array.from(document.querySelectorAll('.panel'));
  const input = document.getElementById('search');

    function setActiveTab(tabId) {{
        tabButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.tab === tabId));
        panels.forEach(panel => panel.classList.toggle('active', panel.id === tabId));
        applySearch();
    }}

    function applySearch() {{
    const q = input.value.toLowerCase().trim();
        const activePanel = document.querySelector('.panel.active');
        if (!activePanel) return;

        activePanel.querySelectorAll('tr.paper-row').forEach(row => {{
            const text = row.textContent.toLowerCase();
            row.classList.toggle('hidden', Boolean(q) && !text.includes(q));
    }});

        panels.filter(panel => panel !== activePanel).forEach(panel => {{
            panel.querySelectorAll('tr.paper-row').forEach(row => row.classList.remove('hidden'));
        }});
    }}

    tabButtons.forEach(btn => {{
        btn.addEventListener('click', () => setActiveTab(btn.dataset.tab));
    }});
    input.addEventListener('input', applySearch);
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
        if not s2_entries and existing_json.exists():
            print("[WARN] S2 returned 0 items. Reusing cached S2 data for this run.", file=sys.stderr)
            cached = json.loads(existing_json.read_text(encoding="utf-8"))
            if int(cached.get("total_papers", 0) or 0) < 50:
                committed = load_committed_dataset()
                if committed and int(committed.get("total_papers", 0) or 0) > int(cached.get("total_papers", 0) or 0):
                    print("[WARN] Local cache too small; recovered baseline from committed docs/papers.json.", file=sys.stderr)
                    cached = committed
            cached_entries = flatten_dataset_entries(cached)
            dataset = build_dataset(cached_entries, arxiv_entries)
            dataset["sources"] = ["Semantic Scholar (cached)", "arXiv"]
        else:
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
