#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
import html
import json
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path("/home/runner/work/SNN-CRAWLING/SNN-CRAWLING")
DOCS_DIR = ROOT / "docs"
ARXIV_API = "http://export.arxiv.org/api/query"
MAX_RESULTS = int(os.getenv("SNN_MAX_RESULTS", "1000"))
BATCH_SIZE = 100


CATEGORY_RULES = {
    "LLM": [r"\bllm\b", r"large language model", r"\blanguage model\b", r"\bgpt\b", r"\bbert\b", r"\btransformer\b", r"\bin-context\b", r"\bprompt\b"],
    "Object Detection": [r"object detection", r"\bdetection\b", r"\byolo\b", r"\br-cnn\b", r"\bfaster r-cnn\b", r"\bssd\b"],
    "Drone": [r"\bdrone\b", r"\buav\b", r"\bunmanned aerial\b", r"\bquadcopter\b", r"\baerial\b"],
    "Event-based Vision": [r"event[- ]based", r"\bdvs\b", r"\bneuromorphic vision\b", r"\bsilicon retina\b"],
    "Neuromorphic Hardware": [r"neuromorphic hardware", r"\bmemristor\b", r"\bfpga\b", r"\basic\b", r"\bloihi\b", r"\btruenorth\b", r"\bspinnaker\b"],
    "Learning Algorithms": [r"\bstdp\b", r"\bsurrogate gradient\b", r"\bbackprop", r"\bspike[- ]timing\b", r"\bplasticity\b"],
    "Reinforcement Learning": [r"reinforcement learning", r"\brl\b", r"\bpolicy gradient\b", r"\bq-learning\b"],
    "Robotics & Control": [r"\brobot", r"\bcontrol\b", r"\bmanipulation\b", r"\bnavigation\b", r"\bslam\b"],
    "Segmentation": [r"segmentation", r"\bmask\b", r"\bpanoptic\b", r"\bsemantic segmentation\b"],
    "Medical & Healthcare": [r"\bmedical\b", r"\bhealthcare\b", r"\becg\b", r"\beeg\b", r"\bdiagnosis\b", r"\bbiomedical\b"],
    "Speech & Audio": [r"\bspeech\b", r"\baudio\b", r"\bkeyword spotting\b", r"\bvoice\b"],
    "NLP": [r"\bnlp\b", r"\bnatural language\b", r"\btext classification\b", r"\bsentiment\b"],
    "Time Series": [r"\btime series\b", r"\bforecast", r"\banomaly detection\b", r"\bsensor data\b"],
}


def fetch_arxiv_entries() -> list[dict]:
    entries: list[dict] = []
    for start in range(0, MAX_RESULTS, BATCH_SIZE):
        params = {
            "search_query": 'all:"spiking neural network"',
            "start": str(start),
            "max_results": str(min(BATCH_SIZE, MAX_RESULTS - start)),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
        root = ET.fromstring(data)
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        batch = root.findall("atom:entry", ns)
        if not batch:
            break
        for entry in batch:
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
            summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
            link = ""
            for l in entry.findall("atom:link", ns):
                if l.attrib.get("type") == "text/html":
                    link = l.attrib.get("href", "")
                    break
            if not link:
                link = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
            published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
            authors = entry.findall("atom:author", ns)
            first_author = "Unknown"
            if authors:
                first_author = (authors[0].findtext("atom:name", default="", namespaces=ns) or "Unknown").strip()
            journal_ref = (entry.findtext("arxiv:journal_ref", default="", namespaces=ns) or "").strip()
            venue = journal_ref if journal_ref else "arXiv"

            entries.append(
                {
                    "first_author": first_author,
                    "title": " ".join(title.split()),
                    "summary": " ".join(summary.split()),
                    "venue": venue,
                    "link": link,
                    "published": published,
                }
            )
    return entries


def classify_paper(text: str) -> str:
    lower = text.lower()
    for category, patterns in CATEGORY_RULES.items():
        for pattern in patterns:
            if re.search(pattern, lower):
                return category
    return "Etc"


def sort_key(paper: dict) -> tuple:
    published = paper.get("published", "")
    try:
        ts = dt.datetime.fromisoformat(published.replace("Z", "+00:00"))
    except ValueError:
        ts = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    return (ts, paper.get("title", ""))


def build_dataset(entries: list[dict]) -> dict:
    categories: dict[str, list[dict]] = {name: [] for name in CATEGORY_RULES}
    categories["Etc"] = []
    seen = set()
    for paper in entries:
        dedup_key = (paper["title"].lower(), paper["first_author"].lower())
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        combined = f'{paper["title"]} {paper["summary"]}'
        category = classify_paper(combined)
        categories.setdefault(category, []).append(
            {
                "first_author": paper["first_author"],
                "title": paper["title"],
                "venue": paper["venue"],
                "link": paper["link"],
                "published": paper["published"],
            }
        )
    for cat in categories:
        categories[cat] = sorted(categories[cat], key=sort_key, reverse=True)

    return {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "query": 'all:"spiking neural network"',
        "total_papers": sum(len(v) for v in categories.values()),
        "categories": categories,
    }


def render_html(dataset: dict) -> str:
    categories = dataset["categories"]
    chips = []
    sections = []
    for cat, papers in categories.items():
        chips.append(f'<a href="#{html.escape(cat)}" class="chip">{html.escape(cat)} ({len(papers)})</a>')
        rows = []
        for p in papers:
            rows.append(
                "<tr>"
                f"<td>{html.escape(p['first_author'])}</td>"
                f"<td>{html.escape(p['title'])}</td>"
                f"<td>{html.escape(p['venue'])}</td>"
                f'<td><a href="{html.escape(p["link"])}" target="_blank" rel="noopener">Link</a></td>'
                "</tr>"
            )
        if not rows:
            rows.append('<tr><td colspan="4">No papers yet.</td></tr>')
        sections.append(
            f"""
            <section id="{html.escape(cat)}">
              <h2>{html.escape(cat)} <span class="count">({len(papers)})</span></h2>
              <table>
                <thead>
                  <tr><th>1st Author</th><th>Title</th><th>Journal/Conference</th><th>Link</th></tr>
                </thead>
                <tbody>
                  {''.join(rows)}
                </tbody>
              </table>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SNN Paper Tracker</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 24px; line-height: 1.5; }}
    h1 {{ margin-bottom: 8px; }}
    .meta {{ color: #555; margin-bottom: 12px; }}
    .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }}
    .chip {{ border: 1px solid #ddd; padding: 4px 10px; border-radius: 999px; text-decoration: none; color: #333; }}
    section {{ margin: 30px 0; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border: 1px solid #e5e5e5; padding: 8px; text-align: left; vertical-align: top; }}
    thead {{ background: #f7f7f7; }}
    .count {{ color: #777; font-size: 0.9em; }}
  </style>
</head>
<body>
  <h1>Spiking Neural Network Papers</h1>
  <p class="meta">Generated at: {html.escape(dataset['generated_at_utc'])} (UTC) · Total: {dataset['total_papers']}</p>
  <div class="chips">{''.join(chips)}</div>
  {''.join(sections)}
</body>
</html>
"""


def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    entries = fetch_arxiv_entries()
    dataset = build_dataset(entries)
    (DOCS_DIR / "papers.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    (DOCS_DIR / "index.html").write_text(render_html(dataset), encoding="utf-8")


if __name__ == "__main__":
    main()
