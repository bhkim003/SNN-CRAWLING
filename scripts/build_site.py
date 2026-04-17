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

import argparse
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

ARXIV_API    = "https://export.arxiv.org/api/query"
OPENALEX_API = "https://api.openalex.org/works"
CROSSREF_API = "https://api.crossref.org/works"

OPENALEX_CONTACT = os.getenv("SNN_OPENALEX_CONTACT", "bhkim003@snu.ac.kr")
OPENALEX_API_KEY = (
    os.getenv("OPENALEX_API_KEY")
    or os.getenv("SNN_OPENALEX_API_KEY")
    or ""
).strip()
OPENALEX_FIELDS  = "id,doi,title,authorships,primary_location,publication_date,abstract_inverted_index,cited_by_count"
CROSSREF_CONTACT = os.getenv("SNN_CROSSREF_CONTACT", "bhkim003@snu.ac.kr")

ARXIV_QUERY    = 'all:"spiking neural network" OR all:"spike-based" OR all:"snn"'
OPENALEX_QUERY = '"spiking neural network" OR "spike-based" OR snn'
CRAWL_START_DATE = "2017-01-01"
CRAWL_START_DT = dt.datetime(2017, 1, 1, tzinfo=dt.timezone.utc)
ARXIV_START_DATE = "2020-01-01"
ARXIV_START_DT = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)

MAX_OPENALEX = int(os.getenv("SNN_MAX_OPENALEX", "30000"))
MAX_ARXIV    = int(os.getenv("SNN_MAX_ARXIV",    "30000"))
BATCH        = 100
OPENALEX_LOG_EVERY_PAGES       = int(os.getenv("SNN_OPENALEX_LOG_EVERY_PAGES", "10"))
ARXIV_SLEEP_SECONDS = float(os.getenv("SNN_ARXIV_SLEEP",        "1.5"))
ARXIV_MAX_RETRIES   = int(os.getenv("SNN_ARXIV_MAX_RETRIES",    "8"))
ARXIV_BACKOFF_BASE  = float(os.getenv("SNN_ARXIV_BACKOFF_BASE", "15"))
ARXIV_BACKOFF_CAP   = float(os.getenv("SNN_ARXIV_BACKOFF_CAP",  "180"))

# Daily incremental mode settings
DAILY_DAYS         = int(os.getenv("SNN_DAILY_DAYS",         "30"))
MAX_DAILY_OPENALEX = int(os.getenv("SNN_MAX_DAILY_OPENALEX", "1000"))
MAX_DAILY_ARXIV    = int(os.getenv("SNN_MAX_DAILY_ARXIV",    "1000"))

MIN_REQUIRED_CITATIONS = int(os.getenv("SNN_MIN_REQUIRED_CITATIONS", "1"))
MIN_ARXIV_REQUIRED_CITATIONS = int(os.getenv("SNN_MIN_ARXIV_REQUIRED_CITATIONS", "0"))

# (Simplified) citation filter policy: keep papers cited at least once.

# ---------------------------------------------------------------------------
# Category rules  (first match wins; Etc is the fallback)
# ---------------------------------------------------------------------------
CATEGORY_RULES: dict[str, list[str]] = {
    # ── Top 3 (mandatory first) ──────────────────────────────────────────
    "LLM": [
        r"\bllm\b", r"\bllms\b", r"large language model", r"language model",
        r"\bgpt[\b\-]", r"\bbert\b", r"\btransformer\b",
        r"in.context learning", r"foundation model",
        r"instruction.tun", r"pretrain.*language", r"vision.language model",
    ],
    "Object Detection": [
        r"object detection", r"\byolo\b", r"\br-cnn\b", r"faster r-cnn",
        r"\bssd\b", r"anchor.free detection", r"bounding box",
        r"object recogni", r"pedestrian detection", r"face detection",
    ],
    "Drone": [
        r"\bdrone\b", r"\buav\b", r"unmanned aerial", r"\bquadcopter\b",
        r"aerial robot", r"autonomous flight", r"aerial vehicle",
        r"micro aerial", r"fixed.wing",
    ],
    # ── Neuromorphic & SNN core ──────────────────────────────────────────
    "Event-based Vision": [
        r"event.based", r"event.driven vision", r"\bdvs\b",
        r"dynamic vision sensor", r"neuromorphic vision",
        r"silicon retina", r"event camera", r"asynchronous vision",
        r"event.driven neural", r"spike.based vision",
    ],
    "Neuromorphic Hardware": [
        r"neuromorphic hardware", r"neuromorphic chip", r"neuromorphic comput",
        r"neuromorphic processor", r"\bmemristor\b", r"\bfpga\b",
        r"\bvlsi\b", r"\basic\b", r"\bloihi\b", r"\btruenorth\b",
        r"\bspinnaker\b", r"hardware accelerat", r"on.chip learning",
        r"resistive switch", r"in.memory comput",
    ],
    "ANN-to-SNN Conversion": [
        r"ann.to.snn", r"ann.snn conversion", r"convert.*ann.*snn",
        r"convert.*artificial.*spiking", r"rate coding",
        r"firing rate conversion", r"threshold balancing",
    ],
    "Surrogate Gradient": [
        r"surrogate gradient", r"backpropagat.*spiking",
        r"backpropagat.*snn", r"gradient.*spiking network",
        r"bptt.*spiking", r"temporal.*backprop",
        r"spike.*backprop", r"differentiable spiking",
    ],
    "Learning Rules & STDP": [
        r"\bstdp\b", r"spike.timing.dependent",
        r"\bhebbian\b", r"synaptic plasticity",
        r"online learning.*spiking", r"local learning rule",
        r"unsupervised.*spiking", r"biologically plausible",
    ],
    "Reservoir Computing": [
        r"reservoir computing", r"liquid state machine",
        r"echo state network", r"\blsm\b", r"\besn\b",
        r"recurrent.*reservoir", r"spiking.*reservoir",
    ],
    # ── Applications ────────────────────────────────────────────────────
    "Medical & BCI": [
        r"\bmedical imaging\b", r"\bhealthcare\b", r"\becg\b", r"\beeg\b",
        r"\bbiomedical\b", r"brain.computer interface",
        r"\bseizure\b", r"\bbci\b", r"neural decod",
        r"clinical diagnosis", r"\bemg\b", r"electromyograph",
        r"\bepilepsy\b", r"brain signal", r"neural signal",
    ],
    "Robotics & Control": [
        r"\brobotics\b", r"robotic manipul", r"robotic arm",
        r"robot locomotion", r"\bslam\b", r"\bactuator\b",
        r"robot control", r"legged robot", r"bio.inspired robot",
    ],
    "Autonomous Navigation": [
        r"autonomous navigat", r"autonomous driv", r"self.driving",
        r"autonomous vehicle", r"path planning", r"obstacle avoidance",
        r"\bnavigation\b", r"autonomous robot",
    ],
    "Gesture & Action Recognition": [
        r"gesture recogni", r"action recogni", r"activity recogni",
        r"pose estimat", r"\bskeleton\b", r"human action",
        r"hand gesture", r"body pose", r"sign language",
    ],
    "Speech & Audio": [
        r"\bspeech\b", r"\baudio\b", r"keyword spotting",
        r"\bvoice\b", r"sound classif", r"\basr\b", r"automatic speech",
        r"auditory cortex", r"\bspoken\b", r"audio event",
    ],
    "Medical Diagnosis": [
        r"\bmedical\b", r"\bdiagnosis\b", r"\bhealthcare\b",
        r"\bpathology\b", r"disease detect", r"tumor detect",
        r"cancer classif", r"retinal", r"chest x.ray",
        r"medical segmentation",
    ],
    # ── Vision & Image tasks ─────────────────────────────────────────────
    "Image Classification": [
        r"image classif", r"\bcifar\b", r"\bimagenet\b",
        r"\bmnist\b", r"\bn-mnist\b", r"\bn-caltech\b",
        r"visual recogni", r"image recogni",
    ],
    "Segmentation": [
        r"segmentation", r"semantic segmentation",
        r"instance segmentation", r"\bpanoptic\b",
        r"scene understanding", r"pixel.wise",
    ],
    "Point Cloud & 3D": [
        r"point cloud", r"3d object", r"depth estimation",
        r"\blidar\b", r"3d detection", r"3d classif",
        r"\bvoxel\b", r"stereo vision", r"depth.*sensor",
        r"3d scene", r"range image",
    ],
    # ── ML paradigms ────────────────────────────────────────────────────
    "Reinforcement Learning": [
        r"reinforcement learning", r"policy gradient", r"\bppo\b",
        r"\bdqn\b", r"actor.critic", r"reward.based",
        r"\bmarl\b", r"multi.agent.*reinforcement", r"q.learning",
    ],
    "Continual Learning": [
        r"continual learning", r"lifelong learning",
        r"catastrophic forgetting", r"incremental learning",
        r"class.incremental", r"task.incremental",
        r"few.shot.*continual",
    ],
    "Federated Learning": [
        r"federated learning", r"federated optim",
        r"privacy.preserv.*learn", r"distributed.*privacy",
        r"federated.*spiking",
    ],
    "Graph Neural Networks": [
        r"graph neural", r"\bgnn\b", r"graph convolution",
        r"graph.based learn", r"graph network", r"knowledge graph",
        r"graph spiking", r"graph attention",
    ],
    "Time Series": [
        r"time.series", r"\bforecast", r"anomaly detection",
        r"temporal data", r"sequential data", r"temporal pattern",
        r"time.series classif",
    ],
    "NLP": [
        r"\bnlp\b", r"natural language processing",
        r"text classif", r"sentiment analysis",
        r"named entity", r"machine translation", r"question answer",
    ],
    "Multimodal Learning": [
        r"multimodal", r"multi.modal", r"vision.language",
        r"image.text", r"cross.modal", r"audio.visual",
        r"multi.sensory",
    ],
    # ── Efficiency & Deployment ──────────────────────────────────────────
    "Model Compression": [
        r"model compression", r"\bpruning\b", r"knowledge distillation",
        r"\bquantization\b", r"\bquantized\b", r"network compression",
        r"lightweight.*network", r"compact.*network",
    ],
    "Edge AI & Embedded": [
        r"\bedge.ai\b", r"edge computing", r"edge inference",
        r"embedded system", r"\biot\b", r"internet of things",
        r"low.power.*neural", r"energy.efficient.*neural",
        r"on.device", r"microcontroller",
    ],
    # ── Security & Other ────────────────────────────────────────────────
    "Security & Adversarial": [
        r"adversarial attack", r"adversarial example",
        r"adversarial robustness", r"intrusion detection",
        r"backdoor attack", r"fault.*inject",
        r"\bcybersecur", r"adversarial.*spiking",
    ],
    "Optical & Photonic": [
        r"optical comput", r"photonic neural", r"photonic comput",
        r"optical neural", r"diffractive neural", r"\boadnn\b",
        r"all.optical",
    ],
    "Datasets & Benchmarks": [
        r"\bbenchmark\b.*spiking", r"\bbenchmark\b.*snn",
        r"evaluation.*spiking", r"neuromorphic dataset",
        r"\bncars\b", r"cifar10.dvs", r"n.mnist",
        r"dvs.{1,20}dataset", r"spiking.*dataset",
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
    "attention": "Attention Mechanisms",
    "recurrent": "Recurrent Networks",
    "generative": "Generative Models",
    "diffusion": "Diffusion Models",
    "contrastive": "Contrastive Learning",
    "transformer": "Transformer Models",
    "spikeformer": "Transformer Models",
    "pruning": "Model Compression",
    "distillation": "Model Compression",
    "autonomous": "Autonomous Systems",
    "locomotion": "Robotics & Control",
}

DYNAMIC_TOPIC_BLACKLIST = {
    "based", "learning", "network", "networks", "spiking", "spike", "spikes",
    "neural", "model", "models", "method", "methods", "approach", "approaches",
    "paper", "system", "systems", "analysis", "study", "studies", "using",
    "toward", "towards", "with", "without", "from", "into", "over", "under",
}


def _normalize_text_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


# ---------------------------------------------------------------------------
# OpenAlex helpers
# ---------------------------------------------------------------------------
def _reconstruct_abstract(inverted_index: dict | None) -> str:
    if not inverted_index:
        return ""
    words: dict[int, str] = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            words[pos] = word
    return " ".join(words[i] for i in sorted(words))


def _min_required_citations(published: str, source: str = "openalex") -> int:
    if source == "arxiv":
        return MIN_ARXIV_REQUIRED_CITATIONS
    return MIN_REQUIRED_CITATIONS


def _normalize_arxiv_abs_url(raw_url: str) -> str:
    s = (raw_url or "").strip()
    if not s:
        return ""
    if s.startswith("http://"):
        s = "https://" + s[len("http://"):]
    if "arxiv.org/pdf/" in s:
        s = s.replace("arxiv.org/pdf/", "arxiv.org/abs/")
        if s.endswith(".pdf"):
            s = s[:-4]
    return s


def _lookup_openalex_cited_by_for_arxiv(arxiv_url: str, headers: dict[str, str]) -> int:
    url = _normalize_arxiv_abs_url(arxiv_url)
    if not url:
        return 0

    candidates = [url]
    if url.startswith("https://"):
        candidates.append("http://" + url[len("https://"):])

    for candidate in candidates:
        params: dict[str, str] = {
            "filter": f"locations.landing_page_url:{candidate}",
            "per-page": "1",
            "select": "cited_by_count",
            "mailto": OPENALEX_CONTACT,
        }
        query = urllib.parse.urlencode(params)
        req = urllib.request.Request(f"{OPENALEX_API}?{query}", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            if results:
                return int(results[0].get("cited_by_count") or 0)
        except Exception:
            continue
    return 0


def _is_arxiv_venue(venue: str) -> bool:
    norm = _normalize_text_key(venue or "")
    return "arxiv" in norm


def _title_dedup_key(title: str) -> str:
    return re.sub(r"\W+", " ", title.lower()).strip()


def _parse_openalex_work(work: dict) -> dict | None:
    title = (work.get("title") or "").strip()
    if not title:
        return None
    pub_date = (work.get("publication_date") or "").strip()
    cited_by_count = int(work.get("cited_by_count") or 0)
    if cited_by_count < _min_required_citations(pub_date, source="openalex"):
        return None
    authors = work.get("authorships") or []
    first_author = "Unknown"
    if authors:
        author_obj = (authors[0].get("author") or {})
        first_author = (author_obj.get("display_name") or "Unknown").strip()
    doi_raw = work.get("doi") or ""
    if doi_raw.startswith("https://doi.org/"):
        doi = doi_raw[len("https://doi.org/"):]
    else:
        doi = doi_raw.lstrip("/")
    link = f"https://doi.org/{doi}" if doi else (work.get("id") or "")
    loc = work.get("primary_location") or {}
    source = loc.get("source") or {}
    venue = (source.get("display_name") or "").strip()
    abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
    is_arxiv = _is_arxiv_venue(venue)
    return {
        "first_author": first_author,
        "title": " ".join(title.split()),
        "abstract": abstract,
        "venue": venue,
        "link": link,
        "published": pub_date,
        "is_arxiv": is_arxiv,
    }


def _fetch_openalex_pass(
    search_query: str,
    from_date: str,
    limit: int,
    headers: dict[str, str],
) -> list[dict]:
    entries: list[dict] = []
    cursor = "*"
    page_count = 0

    while len(entries) < limit:
        min_cited_filter = ""
        if MIN_REQUIRED_CITATIONS > 0:
            min_cited_filter = f",cited_by_count:>{MIN_REQUIRED_CITATIONS - 1}"

        params: dict[str, str] = {
            "search": search_query,
            "per-page": str(min(200, limit - len(entries))),
            "sort": "publication_date:desc",
            "select": OPENALEX_FIELDS,
            "cursor": cursor,
            "mailto": OPENALEX_CONTACT,
            "filter": f"from_publication_date:{from_date}{min_cited_filter}",
        }
        url = f"{OPENALEX_API}?{urllib.parse.urlencode(params)}"
        data: dict = {}

        for attempt in range(6):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except Exception as exc:
                if attempt == 5:
                    print(f"[WARN] OpenAlex fetch error: {exc}", file=sys.stderr)
                    return entries
                time.sleep(5 * (attempt + 1))

        results = data.get("results", [])
        if not results:
            break

        for work in results:
            entry = _parse_openalex_work(work)
            if not entry:
                continue
            entries.append(entry)

        page_count += 1
        meta = data.get("meta", {})
        cursor = meta.get("next_cursor")
        if (
            page_count == 1
            or not cursor
            or (OPENALEX_LOG_EVERY_PAGES > 0 and page_count % OPENALEX_LOG_EVERY_PAGES == 0)
        ):
            print(f"[INFO] OpenAlex: {len(entries)} fetched (pages={page_count})", file=sys.stderr)
        if not cursor:
            break
        time.sleep(0.15)

    return entries


# ---------------------------------------------------------------------------
# OpenAlex fetcher (initial full mode — up to MAX_OPENALEX papers)
# ---------------------------------------------------------------------------
def fetch_openalex_entries(max_papers: int | None = None, from_date: str = CRAWL_START_DATE) -> list[dict]:
    limit = max_papers if max_papers is not None else MAX_OPENALEX
    entries: list[dict] = []
    seen_titles: set[str] = set()
    headers = {"User-Agent": f"SNN-Paper-Tracker/2.0 (mailto:{OPENALEX_CONTACT})"}
    if OPENALEX_API_KEY:
        headers["api-key"] = OPENALEX_API_KEY
    print(f"[INFO] Fetching from OpenAlex since {from_date} (cited>=1; up to {limit} papers)…", file=sys.stderr)

    def _append_unique(candidates: list[dict]) -> None:
        for entry in candidates:
            norm = _title_dedup_key(entry.get("title", ""))
            if not norm or norm in seen_titles:
                continue
            seen_titles.add(norm)
            entries.append(entry)
            if len(entries) >= limit:
                break

    remaining = limit - len(entries)
    candidates = _fetch_openalex_pass(
        search_query=OPENALEX_QUERY,
        from_date=from_date,
        limit=remaining,
        headers=headers,
    )
    _append_unique(candidates)

    return entries


# ---------------------------------------------------------------------------
# OpenAlex recent fetcher (daily mode — date-filtered)
# ---------------------------------------------------------------------------
def fetch_openalex_recent(from_date: str, max_papers: int = 200) -> list[dict]:
    entries: list[dict] = []
    seen_titles: set[str] = set()
    headers = {"User-Agent": f"SNN-Paper-Tracker/2.0 (mailto:{OPENALEX_CONTACT})"}
    if OPENALEX_API_KEY:
        headers["api-key"] = OPENALEX_API_KEY
    print(f"[INFO] Fetching recent OpenAlex papers since {from_date} (cited>=1; up to {max_papers})…", file=sys.stderr)

    def _append_unique(candidates: list[dict]) -> None:
        for entry in candidates:
            norm = _title_dedup_key(entry.get("title", ""))
            if not norm or norm in seen_titles:
                continue
            seen_titles.add(norm)
            entries.append(entry)
            if len(entries) >= max_papers:
                break

    remaining = max_papers - len(entries)
    candidates = _fetch_openalex_pass(
        search_query=OPENALEX_QUERY,
        from_date=from_date,
        limit=remaining,
        headers=headers,
    )
    _append_unique(candidates)

    return entries


# ---------------------------------------------------------------------------
# CrossRef fetcher (fast, for journal papers in daily mode)
# ---------------------------------------------------------------------------
def fetch_crossref_entries(from_date: str | None = None, max_papers: int = 200) -> list[dict]:
    entries: list[dict] = []
    headers = {
        "User-Agent": f"SNN-Paper-Tracker/2.0 (mailto:{CROSSREF_CONTACT})"
    }
    print(f"[INFO] Fetching from CrossRef (up to {max_papers} papers)…", file=sys.stderr)

    offset = 0
    while len(entries) < max_papers:
        params: dict[str, str] = {
            "query": "spiking neural network",
            "rows": str(min(100, max_papers - len(entries))),
            "offset": str(offset),
            "sort": "published",
            "order": "desc",
            "mailto": CROSSREF_CONTACT,
        }
        if from_date:
            params["filter"] = f"from-pub-date:{from_date}"
        url = f"{CROSSREF_API}?{urllib.parse.urlencode(params)}"
        data: dict = {}

        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except Exception as exc:
                if attempt == 3:
                    print(f"[WARN] CrossRef error: {exc}", file=sys.stderr)
                    return entries
                time.sleep(3 * (attempt + 1))

        items = data.get("message", {}).get("items", [])
        if not items:
            break

        for item in items:
            title_list = item.get("title", [])
            if not title_list:
                continue
            title = title_list[0].strip()

            authors = item.get("author", [])
            first_author = "Unknown"
            if authors:
                a = authors[0]
                first_author = f"{a.get('given', '')} {a.get('family', '')}".strip() or "Unknown"

            doi = item.get("DOI", "")
            link = f"https://doi.org/{doi}" if doi else ""
            if not link:
                continue

            pub_date = ""
            dp = (item.get("published") or item.get("published-print") or item.get("published-online") or {}).get("date-parts", [[]])
            if dp and dp[0]:
                parts = dp[0]
                if len(parts) >= 3:
                    pub_date = f"{parts[0]:04d}-{parts[1]:02d}-{parts[2]:02d}"
                elif len(parts) == 2:
                    pub_date = f"{parts[0]:04d}-{parts[1]:02d}-01"
                elif len(parts) == 1:
                    pub_date = f"{parts[0]:04d}-01-01"

            ct = item.get("container-title", [])
            venue = ct[0] if ct else ""

            abstract_raw = item.get("abstract", "")
            abstract = re.sub(r"<[^>]+>", " ", abstract_raw)
            abstract = " ".join(abstract.split())

            entries.append({
                "first_author": first_author,
                "title": " ".join(title.split()),
                "abstract": abstract,
                "venue": venue,
                "link": link,
                "published": pub_date,
            })

        offset += len(items)
        print(f"[INFO] CrossRef: {len(entries)} fetched", file=sys.stderr)
        if len(items) < 100:
            break
        time.sleep(1.0)

    return entries


# ---------------------------------------------------------------------------
# arXiv supplementary fetcher
# ---------------------------------------------------------------------------
def fetch_arxiv_entries(max_papers: int | None = None, from_date: str | None = ARXIV_START_DATE) -> list[dict]:
    limit = max_papers if max_papers is not None else MAX_ARXIV
    entries: list[dict] = []
    citation_cache: dict[str, int] = {}
    openalex_headers = {"User-Agent": f"SNN-Paper-Tracker/2.0 (mailto:{OPENALEX_CONTACT})"}
    if OPENALEX_API_KEY:
        openalex_headers["api-key"] = OPENALEX_API_KEY
    stop_crawl = False
    from_dt = ARXIV_START_DT
    arxiv_query = ARXIV_QUERY
    if from_date:
        try:
            from_dt = dt.datetime.strptime(from_date, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
            start_compact = from_dt.strftime("%Y%m%d")
            # arXiv date range queries are more reliable with an explicit upper bound.
            end_compact = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M")
            # Restrict to recent submissions directly in the arXiv query.
            arxiv_query = f"({ARXIV_QUERY}) AND submittedDate:[{start_compact}0000 TO {end_compact}]"
        except ValueError:
            print(f"[WARN] Invalid arXiv from_date '{from_date}', using default window.", file=sys.stderr)

    print(f"[INFO] Fetching from arXiv (up to {limit} recent preprints)…", file=sys.stderr)

    for start in range(0, limit, BATCH):
        if stop_crawl:
            break
        params = {
            "search_query": arxiv_query,
            "start": str(start),
            "max_results": str(min(BATCH, limit - start)),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        url = f"{ARXIV_API}?{urllib.parse.urlencode(params)}"
        data = None

        for attempt in range(ARXIV_MAX_RETRIES):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    data = resp.read()
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 429:
                    retry_after = exc.headers.get("Retry-After")
                    wait = (
                        float(retry_after)
                        if retry_after and retry_after.isdigit()
                        else min(ARXIV_BACKOFF_BASE * (2 ** attempt), ARXIV_BACKOFF_CAP)
                    )
                    print(
                        f"[WARN] arXiv rate limit hit (attempt {attempt + 1}/{ARXIV_MAX_RETRIES}), waiting {int(wait)}s…",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
                elif 500 <= exc.code < 600:
                    wait = min(3 * (attempt + 1), 30)
                    if attempt == ARXIV_MAX_RETRIES - 1:
                        print(f"[WARN] arXiv server error after {ARXIV_MAX_RETRIES} attempts: {exc}", file=sys.stderr)
                        return entries
                    print(
                        f"[WARN] arXiv server error {exc.code} (attempt {attempt + 1}/{ARXIV_MAX_RETRIES}), waiting {int(wait)}s…",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
                elif attempt == ARXIV_MAX_RETRIES - 1:
                    print(f"[WARN] arXiv error after {ARXIV_MAX_RETRIES} attempts: {exc}", file=sys.stderr)
                    return entries
                else:
                    time.sleep(3)
            except Exception as exc:
                if attempt == ARXIV_MAX_RETRIES - 1:
                    print(f"[WARN] arXiv error after {ARXIV_MAX_RETRIES} attempts: {exc}", file=sys.stderr)
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

            if _parse_date(pub) < from_dt:
                stop_crawl = True
                break

            required_citations = _min_required_citations(pub, source="arxiv")
            if required_citations > 0:
                arxiv_abs_url = _normalize_arxiv_abs_url(link)
                cited_by_count = citation_cache.get(arxiv_abs_url)
                if cited_by_count is None:
                    cited_by_count = _lookup_openalex_cited_by_for_arxiv(arxiv_abs_url, openalex_headers)
                    citation_cache[arxiv_abs_url] = cited_by_count
                if cited_by_count < required_citations:
                    continue

            entries.append({
                "first_author": first_author,
                "title": " ".join(title.split()),
                "abstract": abstract,
                "venue": venue,
                "link": link,
                "published": pub,
                "is_arxiv": True,
            })

        print(f"[INFO] arXiv: {len(entries)} fetched", file=sys.stderr)

        if ARXIV_SLEEP_SECONDS > 0:
            time.sleep(ARXIV_SLEEP_SECONDS)  # arXiv rate limit

        if stop_crawl:
            break

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
        return "Etc"

    top_phrase = sorted(phrase_scores.items(), key=lambda kv: (-kv[1], len(kv[0]), kv[0]))[0][0]
    if len(top_phrase.split()) < 2:
        return "Etc"
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
            "is_arxiv": bool(paper.get("is_arxiv", False)),
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
                "is_arxiv": bool(p.get("is_arxiv", False)),
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
# ---------------------------------------------------------------------------
# HTML renderer
# ---------------------------------------------------------------------------
def render_html(dataset: dict) -> str:
    categories = dataset["categories"]
    generated_raw = (dataset.get("generated_at_utc") or dt.datetime.now(dt.timezone.utc).isoformat()).strip()
    try:
        generated_dt = dt.datetime.fromisoformat(generated_raw.replace("Z", "+00:00"))
        if generated_dt.tzinfo is None:
            generated_dt = generated_dt.replace(tzinfo=dt.timezone.utc)
    except ValueError:
        generated_dt = dt.datetime.now(dt.timezone.utc)
    kst = dt.timezone(dt.timedelta(hours=9))
    generated = generated_dt.astimezone(kst).strftime("%Y-%m-%d %H:%M:%S KST")
    total = dataset["total_papers"]
    display_limit = 200

    def _tooltip_preview(abstract: str, max_chars: int = 400) -> str:
        text = abstract.strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rstrip() + "…"

    def make_rows(papers: list[dict], category_name: str, include_category: bool = False) -> str:
        rows: list[str] = []
        for p in papers:
            date_str = p["published"][:10] if p["published"] else "-"
            abstract = p.get("abstract", "").strip() or "Abstract unavailable."
            abstract_attr = html.escape(_tooltip_preview(abstract), quote=True)
            is_arxiv_attr = "1" if p.get("is_arxiv") else "0"
            title_key = f"{p['title']}_{date_str}"
            title_key_hash = str(hash(title_key) & 0x7FFFFFFF)
            category_cell = (
                f'<td class="col-cat">{html.escape(p.get("category", "Etc"))}</td>'
                if include_category else ""
            )
            rows.append(
                f'<tr class="paper-row" data-category="{html.escape(category_name, quote=True)}" data-abstract="{abstract_attr}" data-is-arxiv="{is_arxiv_attr}" data-title="{html.escape(p["title"], quote=True)}" data-date="{date_str}" data-author="{html.escape(p["first_author"], quote=True)}" data-id="{title_key_hash}">'
                f'<td class="col-actions"><button class="btn-delete" title="Delete" data-id="{title_key_hash}">✕</button><span class="rating" data-id="{title_key_hash}" data-rating="0">☆</span></td>'
                f'<td class="col-date">{html.escape(date_str)}</td>'
                f'<td class="col-venue">{html.escape(p["venue"])}</td>'
                f'<td class="col-title"><a class="paper-link" href="{html.escape(p["link"])}" target="_blank" rel="noopener">'
                f'{html.escape(p["title"])}</a></td>'
                f'{category_cell}'
                f'<td class="col-author">{html.escape(p["first_author"])}</td>'
                f'</tr>'
            )
        if not rows:
            colspan = "6" if include_category else "5"
            rows.append(f'<tr><td colspan="{colspan}" class="empty">No papers found.</td></tr>')
        return "".join(rows)

    # Collect total papers for TOTAL tab
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
                "abstract": p.get("abstract", ""),
                "is_arxiv": bool(p.get("is_arxiv", False)),
            })
    total_papers.sort(key=lambda p: _parse_date(p["published"]), reverse=True)

    # Build sidebar buttons
    buttons_html = f'<button class="category-button active" data-tab="tab-total">TOTAL <span class="badge">{total}</span></button>\n'
    for cat, papers in categories.items():
        cid = re.sub(r"[^a-zA-Z0-9]+", "-", cat).strip("-").lower() or "etc"
        tab_id = f"tab-{cid}"
        count = len(papers)
        buttons_html += f'<button class="category-button" data-tab="{tab_id}" data-category="{html.escape(cat, quote=True)}">{html.escape(cat)} <span class="badge">{count}</span></button>\n'

    # Build content panels
    panels_html = f"""<section class="panel active" id="tab-total">
      <h2>Total Papers <span class="count">({total})</span></h2>
      <div class="table-wrap">
        <table>
          <thead><tr>
                        <th class="col-actions">Act</th>
                        <th class="col-date">Date</th>
                        <th class="col-venue">Journal / Conference</th>
            <th class="col-title">Title</th>
            <th class="col-cat">Category</th>
                        <th class="col-author">1st Author</th>
          </tr></thead>
          <tbody>{make_rows(total_papers, category_name="TOTAL", include_category=True)}</tbody>
        </table>
      </div>
            <div class="pager" data-for="tab-total"></div>
    </section>
"""

    for cat, papers in categories.items():
        cid = re.sub(r"[^a-zA-Z0-9]+", "-", cat).strip("-").lower() or "etc"
        tab_id = f"tab-{cid}"
        count = len(papers)
        panels_html += f"""<section class="panel" id="{tab_id}">
      <h2>{html.escape(cat)} <span class="count">({count})</span></h2>
      <div class="table-wrap">
        <table>
          <thead><tr>
                        <th class="col-actions">Act</th>
                        <th class="col-date">Date</th>
                        <th class="col-venue">Journal / Conference</th>
            <th class="col-title">Title</th>
                        <th class="col-author">1st Author</th>
          </tr></thead>
          <tbody>{make_rows(papers, category_name=cat)}</tbody>
        </table>
      </div>
            <div class="pager" data-for="{tab_id}"></div>
    </section>
"""

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
    html[data-theme="light"] {{
      --bg: #f5f7fa;
      --surface: #ffffff;
      --surface-2: #f0f3f7;
      --border: #d4dce8;
      --accent: #0066cc;
      --accent-2: #00aa66;
      --text: #1a1a2e;
      --muted: #666666;
      --row-hover: #f0f3f7;
      --chip-active: #e8efff;
      --shadow: rgba(0, 0, 0, 0.1);
    }}
    body {{
      font-family: 'Segoe UI', 'Noto Sans KR', 'Apple SD Gothic Neo', sans-serif;
      background: radial-gradient(1200px 600px at 5% -10%, #163457 0%, transparent 60%), radial-gradient(1000px 500px at 95% 0%, #143a2d 0%, transparent 60%), var(--bg);
      color: var(--text);
      line-height: 1.6;
      min-height: 100vh;
      transition: background 0.3s;
    }}
    html[data-theme="light"] body {{
      background: linear-gradient(135deg, #f8fafc 0%, #f5f7fa 100%);
    }}
    .hero {{
      background: linear-gradient(120deg, #0f172a 0%, #111a31 55%, #0f1f2c 100%);
      color: var(--text);
      padding: 34px 24px 26px;
      border-bottom: 1px solid var(--border);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    html[data-theme="light"] .hero {{
      background: linear-gradient(120deg, #ffffff 0%, #f5f7fa 100%);
      border-bottom: 2px solid #d4dce8;
    }}
    .hero h1 {{ font-size: 1.8rem; font-weight: 700; margin-bottom: 6px; }}
    .hero h1 span {{ color: var(--accent); }}
    .hero .meta {{ font-size: 0.9rem; color: var(--muted); }}
    .hero .meta b {{ color: var(--accent-2); }}
    .hero-buttons {{
      display: flex;
      gap: 12px;
      align-items: center;
    }}
    .btn-login {{ 
      padding: 8px 16px; 
      border: 1px solid var(--accent); 
      background: var(--accent);
      color: var(--bg);
      border-radius: 8px;
      cursor: pointer;
      font-weight: 600;
      transition: all 0.2s;
    }}
    .btn-login:hover {{ opacity: 0.9; }}
    .btn-logout {{
      padding: 8px 16px;
      border: 1px solid var(--muted);
      background: transparent;
      color: var(--text);
      border-radius: 8px;
      cursor: pointer;
      font-size: 0.9rem;
      transition: all 0.2s;
    }}
    .btn-logout:hover {{ background: var(--surface-2); }}
    .user-info {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 0.9rem;
      color: var(--muted);
    }}
    .user-info strong {{ color: var(--text); }}
    .modal {{
      display: none;
      position: fixed;
      z-index: 3000;
      left: 0;
      top: 0;
      width: 100%;
      height: 100%;
      background: rgba(0,0,0,0.7);
      align-items: center;
      justify-content: center;
    }}
    .modal.show {{ display: flex; }}
    .modal-content {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 32px;
      max-width: 400px;
      width: 90%;
      box-shadow: 0 20px 60px rgba(0,0,0,0.6);
    }}
    .modal-content h2 {{
      margin-bottom: 20px;
      color: var(--accent);
      font-size: 1.4rem;
    }}
    .modal-form {{
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .modal-form label {{
      font-size: 0.9rem;
      font-weight: 600;
      color: var(--text);
    }}
    .modal-form input {{
      padding: 10px 12px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--surface-2);
      color: var(--text);
      font-size: 0.9rem;
      outline: none;
      transition: all 0.2s;
    }}
    .modal-form input:focus {{
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(101,211,255,0.2);
    }}
    .modal-buttons {{
      display: flex;
      gap: 10px;
      margin-top: 20px;
    }}
    .modal-buttons button {{
      flex: 1;
      padding: 10px 16px;
      border: none;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .modal-buttons button.btn-submit {{
      background: var(--accent);
      color: var(--bg);
    }}
    .modal-buttons button.btn-submit:hover {{ opacity: 0.9; }}
    .modal-buttons button.btn-cancel {{
      background: var(--surface-2);
      color: var(--text);
      border: 1px solid var(--border);
    }}
    .modal-buttons button.btn-cancel:hover {{ background: var(--chip-active); }}
    .toolbar {{
      background: var(--surface);
      border-bottom: 1px solid var(--border);
      padding: 14px 24px;
      position: sticky;
      top: 0;
      z-index: 100;
      box-shadow: 0 6px 20px var(--shadow);
    }}
    .toolbar-controls {{
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .search-wrap {{ max-width: 500px; flex: 1; min-width: 200px; }}
    .toolbar-left {{
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      flex: 1;
    }}
    .toolbar-right {{
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .filter-toggle {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-size: 0.85rem;
      color: var(--muted);
      user-select: none;
      white-space: nowrap;
    }}
    .filter-toggle input {{
      width: 16px;
      height: 16px;
      accent-color: var(--accent);
    }}
    .toolbar select, .toolbar .sort-btn, .toolbar .export-btn, .toolbar .theme-btn {{
      padding: 8px 12px;
      border: 1px solid var(--border);
      background: var(--surface-2);
      color: var(--text);
      border-radius: 8px;
      font-size: 0.85rem;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .toolbar select:hover, .toolbar .sort-btn:hover, .toolbar .export-btn:hover, .toolbar .theme-btn:hover {{
      border-color: var(--accent);
      background: var(--chip-active);
    }}
    .toolbar select:focus, .toolbar select:focus-visible {{
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(101,211,255,0.2);
    }}
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
    .layout {{
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
      min-height: calc(100vh - 128px);
      gap: 0;
    }}
    .sidebar {{
      background: var(--surface);
      border-right: 1px solid var(--border);
      max-height: calc(100vh - 128px);
      overflow-y: auto;
      padding: 12px;
      position: sticky;
      top: 128px;
      display: flex;
      flex-direction: column;
    }}
    .sidebar::-webkit-scrollbar {{ width: 10px; }}
    .sidebar::-webkit-scrollbar-thumb {{ background: #263750; border-radius: 999px; }}
    .sidebar::-webkit-scrollbar-track {{ background: transparent; }}
    .category-button {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 6px;
      padding: 9px 12px;
      border: 1px solid var(--border);
      border-radius: 12px;
      color: var(--text);
      font-size: 0.93rem;
      font-weight: 600;
      background: var(--surface-2);
      transition: all 0.18s;
      cursor: pointer;
      width: 100%;
      text-align: left;
      margin-bottom: 8px;
    }}
    .category-button:hover {{
      background: var(--chip-active);
      border-color: #406080;
    }}
    .category-button.active {{
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
    .category-button.active .badge {{
      background: rgba(101, 211, 255, 0.22);
      color: #dff7ff;
    }}
    .content {{
      padding: 24px;
      min-width: 0;
      overflow-y: auto;
      max-height: calc(100vh - 128px);
    }}
    .panel {{ display: none; margin-bottom: 28px; }}
    .panel.active {{ display: block; }}
    .panel h2 {{
      font-size: 1.02rem;
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
    .pager {{
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      padding: 12px 8px 0;
      flex-wrap: wrap;
    }}
    .pager-btn {{
      border: 1px solid var(--border);
      background: var(--surface-2);
      color: var(--text);
      border-radius: 999px;
      min-width: 36px;
      height: 36px;
      padding: 0 10px;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .pager-btn:hover {{
      background: var(--chip-active);
      border-color: #406080;
    }}
    .pager-btn.active {{
      border-color: var(--accent);
      color: var(--accent);
      background: #112035;
    }}
    .pager-btn:disabled {{
      opacity: 0.45;
      cursor: not-allowed;
    }}
    .pager-ellipsis {{
      color: var(--muted);
      font-size: 0.9rem;
      padding: 0 2px;
    }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.875rem; }}
    thead {{ background: #101b2f; }}
    html[data-theme="light"] thead {{ background: #f0f3f7; }}
    th {{
      padding: 10px 12px;
      text-align: left;
      font-weight: 600;
      color: #c5d5ef;
      border-bottom: 1px solid var(--border);
      white-space: nowrap;
      font-size: 0.77rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }}
    html[data-theme="light"] th {{
      color: #334155;
    }}
    td {{
      padding: 9px 12px;
      border-bottom: 1px solid #1b2a3e;
      vertical-align: middle;
      color: #e2ebf9;
    }}
    html[data-theme="light"] td {{
      border-bottom: 1px solid #e2e8f0;
      color: #1a1a2e;
    }}
    tbody tr:last-child td {{ border-bottom: none; }}
    tbody tr:hover td {{ background: var(--row-hover); }}
    .col-actions {{
      width: 60px;
      text-align: center;
      display: flex;
      gap: 4px;
      justify-content: center;
      align-items: center;
    }}
    .btn-delete {{
      border: none;
      background: transparent;
      color: #ff6b6b;
      cursor: pointer;
      font-size: 1.2rem;
      padding: 2px 6px;
      border-radius: 4px;
      transition: all 0.2s;
      display: none;
    }}
    tbody tr.logged-in .btn-delete {{
      display: inline-block;
    }}
    .btn-delete:hover {{
      background: rgba(255, 107, 107, 0.2);
    }}
    .rating {{
      font-size: 1.2rem;
      cursor: pointer;
      display: none;
      user-select: none;
    }}
    tbody tr.logged-in .rating {{
      display: inline;
    }}
    .rating:hover {{
      filter: brightness(1.2);
    }}
    .col-author {{
      width: 110px;
      max-width: 110px;
      color: var(--muted);
      font-size: 0.82rem;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
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
    .tooltip {{
      position: fixed;
      z-index: 2000;
      max-width: 700px;
      max-height: 500px;
      padding: 18px 20px;
      border: 1px solid #4a5f80;
      border-radius: 14px;
      background: rgba(10, 15, 28, 0.99);
      color: #f5faff;
      font-size: 0.72rem;
      line-height: 1.6;
      box-shadow: 0 20px 50px rgba(0,0,0,0.6);
      pointer-events: none;
      display: none;
      white-space: pre-wrap;
      word-wrap: break-word;
      overflow-y: auto;
      font-weight: 500;
      letter-spacing: 0.2px;
    }}
    .tooltip::-webkit-scrollbar {{ width: 8px; }}
    .tooltip::-webkit-scrollbar-thumb {{ background: #3a4f6f; border-radius: 999px; }}
    .tooltip::-webkit-scrollbar-track {{ background: transparent; }}
    .col-venue {{
      width: 150px;
      max-width: 150px;
      color: var(--muted);
      font-size: 0.82rem;
      word-break: break-word;
    }}
    .col-date {{
      width: 95px;
      color: var(--muted);
      font-size: 0.8rem;
      white-space: nowrap;
    }}
    th.col-date, td.col-date,
    th.col-venue, td.col-venue,
    th.col-title, td.col-title,
    th.col-cat, td.col-cat,
    th.col-author, td.col-author,
    th.col-actions, td.col-actions {{
      text-align: center;
    }}
    .empty {{ text-align: center; color: var(--muted); padding: 24px; }}
    .hidden {{ display: none !important; }}
    @media (max-width: 1100px) {{
      .layout {{ grid-template-columns: 220px minmax(0, 1fr); }}
      .sidebar {{ padding: 8px; }}
      .category-button {{ font-size: 0.88rem; padding: 8px 10px; }}
    }}
    @media (max-width: 900px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .sidebar {{
        position: static;
        max-height: 240px;
        border-right: none;
        border-bottom: 1px solid var(--border);
        flex-direction: row;
        flex-wrap: nowrap;
        overflow-x: auto;
        overflow-y: hidden;
        padding-bottom: 4px;
      }}
      .category-button {{
        width: auto;
        min-width: max-content;
        margin-bottom: 0;
        margin-right: 8px;
      }}
      .sidebar::-webkit-scrollbar {{ height: 8px; width: auto; }}
      .hero {{ flex-direction: column; gap: 12px; }}
      .toolbar-controls {{ flex-direction: column; }}
      .search-wrap {{ width: 100%; }}
      .toolbar-left {{ width: 100%; }}
    }}
    @media (max-width: 700px) {{
      .col-venue, .col-date, .col-cat {{ display: none; }}
      .hero h1 {{ font-size: 1.3rem; }}
      .toolbar, .sidebar, .content, .hero {{ padding-left: 14px; padding-right: 14px; }}
      .tooltip {{
        max-width: calc(100vw - 20px);
        font-size: 0.75rem;
        max-height: 400px;
      }}
      .category-button {{ font-size: 0.82rem; padding: 7px 9px; }}
      .pager-btn {{ min-width: 32px; height: 32px; font-size: 0.8rem; }}
      .col-actions {{ width: 50px; }}
    }}
  </style>
</head>
<body>
<div id="login-modal" class="modal">
  <div class="modal-content">
    <h2>Login</h2>
    <form class="modal-form" onsubmit="handleLogin(event)">
      <label for="user-id">User ID</label>
      <input type="text" id="user-id" placeholder="Enter your user ID" required>
      <label for="user-pass">Password</label>
      <input type="password" id="user-pass" placeholder="Enter your password" required>
      <div class="modal-buttons">
        <button type="submit" class="btn-submit">Login</button>
        <button type="button" class="btn-cancel" onclick="closeModal()">Cancel</button>
      </div>
    </form>
  </div>
</div>

<header class="hero">
  <div>
    <h1>⚡ <span>Spiking Neural Network</span> Papers</h1>
    <p class="meta"><b>{total:,}</b> papers · Updated <b>{generated}</b> · Sources: OpenAlex + arXiv</p>
  </div>
  <div class="hero-buttons">
    <div class="user-info" id="user-info" style="display:none;">
      <strong id="current-user"></strong>
    </div>
    <button class="btn-login" id="btn-login-header" onclick="openModal()">Login</button>
    <button class="btn-logout" id="btn-logout-header" onclick="handleLogout()" style="display:none;">Logout</button>
  </div>
</header>

<div class="toolbar">
  <div class="toolbar-controls">
    <div class="toolbar-left">
      <div class="search-wrap">
        <input type="search" id="search" placeholder="Search title, abstract, author, venue..." autocomplete="off">
      </div>
      <label class="filter-toggle" for="hide-arxiv">
        <input type="checkbox" id="hide-arxiv">
        <span>Exclude arXiv</span>
      </label>
      <label class="filter-toggle" for="date-from">
        From: <input type="date" id="date-from" style="max-width: 120px;">
      </label>
      <label class="filter-toggle" for="date-to">
        To: <input type="date" id="date-to" style="max-width: 120px;">
      </label>
      <select id="sort-by">
        <option value="date-desc">Latest First</option>
        <option value="date-asc">Oldest First</option>
        <option value="title-asc">Title (A-Z)</option>
        <option value="author-asc">Author (A-Z)</option>
      </select>
    </div>
    <div class="toolbar-right">
      <button class="export-btn" onclick="exportData('csv')">Export CSV</button>
      <button class="export-btn" onclick="exportData('json')">Export JSON</button>
      <button class="theme-btn" id="theme-toggle" onclick="toggleTheme()">🌙</button>
    </div>
  </div>
</div>

<div class="layout">
  <aside class="sidebar" id="sidebar">
{buttons_html}</aside>
  <main class="content" id="content">
{panels_html}</main>
</div>

<div id="abstract-tooltip" class="tooltip"></div>

<script>
    const PAGE_SIZE = {display_limit};
  const categoryButtons = Array.from(document.querySelectorAll('.category-button'));
  const panels = Array.from(document.querySelectorAll('.panel'));
  const input = document.getElementById('search');
  const hideArxiv = document.getElementById('hide-arxiv');
  const dateFrom = document.getElementById('date-from');
  const dateTo = document.getElementById('date-to');
  const sortBy = document.getElementById('sort-by');
  const tooltip = document.getElementById('abstract-tooltip');
  const loginModal = document.getElementById('login-modal');
  const userInfo = document.getElementById('user-info');
  const btnLoginHeader = document.getElementById('btn-login-header');
  const btnLogoutHeader = document.getElementById('btn-logout-header');
  const currentUserSpan = document.getElementById('current-user');

  // ===== Theme Management =====
  function loadTheme() {{
    const saved = localStorage.getItem('snn-theme') || 'dark';
    document.documentElement.setAttribute('data-theme', saved);
    updateThemeBtn(saved);
  }}
  function toggleTheme() {{
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('snn-theme', next);
    updateThemeBtn(next);
  }}
  function updateThemeBtn(theme) {{
    document.getElementById('theme-toggle').textContent = theme === 'dark' ? '☀️' : '🌙';
  }}
  loadTheme();

  // ===== User Management =====
  class UserSession {{
    constructor() {{
      this.data = JSON.parse(localStorage.getItem('snn-user-session') || '{{}}');
    }}
    isLoggedIn() {{ return !!this.data.userId; }}
    login(userId, password) {{
      this.data = {{ userId, password, loginTime: Date.now() }};
      localStorage.setItem('snn-user-session', JSON.stringify(this.data));
    }}
    logout() {{
      this.data = {{}};
      localStorage.removeItem('snn-user-session');
    }}
    get userId() {{ return this.data.userId || ''; }}
  }}

  const userSession = new UserSession();

  function updateUserUI() {{
    if (userSession.isLoggedIn()) {{
      userInfo.style.display = 'flex';
      currentUserSpan.textContent = userSession.userId;
      btnLoginHeader.style.display = 'none';
      btnLogoutHeader.style.display = 'block';
      document.querySelectorAll('tbody tr').forEach(row => {{
        row.classList.add('logged-in');
      }});
    }} else {{
      userInfo.style.display = 'none';
      btnLoginHeader.style.display = 'inline-block';
      btnLogoutHeader.style.display = 'none';
      document.querySelectorAll('tbody tr').forEach(row => {{
        row.classList.remove('logged-in');
      }});
    }}
  }}

  function openModal() {{ loginModal.classList.add('show'); }}
  function closeModal() {{ loginModal.classList.remove('show'); }}

  function handleLogin(event) {{
    event.preventDefault();
    const userId = document.getElementById('user-id').value.trim();
    const password = document.getElementById('user-pass').value;
    if (userId && password) {{
      userSession.login(userId, password);
      closeModal();
      updateUserUI();
      applySearch();
      document.getElementById('user-id').value = '';
      document.getElementById('user-pass').value = '';
    }}
  }}

  function handleLogout() {{
    if (confirm('Are you sure you want to logout?')) {{
      userSession.logout();
      updateUserUI();
      applySearch();
    }}
  }}

  updateUserUI();

  // ===== Paper Metadata (Delete/Rating) =====
  class PaperManager {{
    constructor() {{
      this.data = JSON.parse(localStorage.getItem('snn-papers') || '{{}}');
    }}
    save() {{
      localStorage.setItem('snn-papers', JSON.stringify(this.data));
    }}
    isDeleted(paperId) {{
      return this.data[paperId]?.deleted === true;
    }}
    delete(paperId) {{
      if (!this.data[paperId]) this.data[paperId] = {{}};
      this.data[paperId].deleted = true;
      this.save();
    }}
    restore(paperId) {{
      if (this.data[paperId]) {{
        delete this.data[paperId].deleted;
        this.save();
      }}
    }}
    setRating(paperId, rating) {{
      if (!this.data[paperId]) this.data[paperId] = {{}};
      this.data[paperId].rating = rating;
      this.save();
    }}
    getRating(paperId) {{
      return this.data[paperId]?.rating || 0;
    }}
  }}

  const paperManager = new PaperManager();

  // ===== Pagination =====
  function pageTokens(totalPages, currentPage) {{
    if (totalPages <= 7) return Array.from({{ length: totalPages }}, (_, i) => i + 1);
    if (currentPage <= 4) return [1, 2, 3, 4, 5, '...', totalPages];
    if (currentPage >= totalPages - 3) return [1, '...', totalPages - 4, totalPages - 3, totalPages - 2, totalPages - 1, totalPages];
    return [1, '...', currentPage - 1, currentPage, currentPage + 1, '...', totalPages];
  }}

  function renderPager(panel, totalPages, currentPage) {{
    const pager = panel.querySelector('.pager');
    if (!pager) return;
    pager.innerHTML = '';
    if (totalPages <= 1) return;

    const prev = document.createElement('button');
    prev.type = 'button';
    prev.className = 'pager-btn';
    prev.textContent = 'Prev';
    prev.dataset.page = String(currentPage - 1);
    prev.disabled = currentPage === 1;
    pager.appendChild(prev);

    pageTokens(totalPages, currentPage).forEach(token => {{
      if (token === '...') {{
        const span = document.createElement('span');
        span.className = 'pager-ellipsis';
        span.textContent = '...';
        pager.appendChild(span);
        return;
      }}
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = `pager-btn${{token === currentPage ? ' active' : ''}}`;
      btn.textContent = String(token);
      btn.dataset.page = String(token);
      pager.appendChild(btn);
    }});

    const next = document.createElement('button');
    next.type = 'button';
    next.className = 'pager-btn';
    next.textContent = 'Next';
    next.dataset.page = String(currentPage + 1);
    next.disabled = currentPage === totalPages;
    pager.appendChild(next);
  }}

  function paginatePanel(panel, resetPage = false) {{
    const rows = Array.from(panel.querySelectorAll('tr.paper-row'));
    const matched = rows.filter(row => !row.classList.contains('hidden'));
    const totalPages = Math.max(1, Math.ceil(matched.length / PAGE_SIZE));

    let page = parseInt(panel.dataset.page || '1', 10);
    if (!Number.isFinite(page) || page < 1) page = 1;
    if (resetPage) page = 1;
    if (page > totalPages) page = totalPages;
    panel.dataset.page = String(page);

    rows.forEach(row => {{
      row.style.display = row.classList.contains('hidden') ? 'none' : '';
    }});

    const start = (page - 1) * PAGE_SIZE;
    const end = start + PAGE_SIZE;
    matched.forEach((row, idx) => {{
      row.style.display = (idx >= start && idx < end) ? '' : 'none';
    }});

    renderPager(panel, totalPages, page);
  }}

  function setActiveTab(tabId) {{
    categoryButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.tab === tabId));
    panels.forEach(panel => panel.classList.toggle('active', panel.id === tabId));
    applySearch();
  }}

  // ===== Sorting & Filtering =====
  function applySearch() {{
    const q = input.value.toLowerCase().trim();
    const excludeArxiv = hideArxiv.checked;
    const fromDate = dateFrom.value ? new Date(dateFrom.value) : null;
    const toDate = dateTo.value ? new Date(dateTo.value) : null;
    const sortMode = sortBy.value;

    categoryButtons.forEach(btn => {{
      const label = (btn.textContent || '').toLowerCase();
      const catName = (btn.dataset.category || '').toLowerCase();
      btn.classList.toggle('hidden', Boolean(q) && !label.includes(q) && !catName.includes(q));
    }});

    const activePanel = document.querySelector('.panel.active');
    if (!activePanel) return;

    activePanel.querySelectorAll('tr.paper-row').forEach(row => {{
      const text = row.textContent.toLowerCase();
      const abs = (row.dataset.abstract || '').toLowerCase();
      const cat = (row.dataset.category || '').toLowerCase();
      const isArxiv = row.dataset.isArxiv === '1';
      const paperId = row.dataset.id;
      const isDeleted = paperManager.isDeleted(paperId);

      const matchesQuery = !q || text.includes(q) || abs.includes(q) || cat.includes(q);
      const hideForArxiv = Boolean(excludeArxiv && isArxiv);
      const rowDate = row.dataset.date ? new Date(row.dataset.date) : null;
      const hideForDate =  (fromDate && rowDate < fromDate) || (toDate && rowDate > toDate);
      const hidden = !matchesQuery || hideForArxiv || hideForDate || isDeleted;

      row.classList.toggle('hidden', hidden);
    }});

    // Sort
    const allRows = Array.from(activePanel.querySelectorAll('tr.paper-row')).filter(r => !r.classList.contains('hidden'));
    let sorted = allRows;
    if (sortMode === 'date-desc') {{
      sorted.sort((a, b) => (b.dataset.date || '').localeCompare(a.dataset.date || ''));
    }} else if (sortMode === 'date-asc') {{
      sorted.sort((a, b) => (a.dataset.date || '').localeCompare(b.dataset.date || ''));
    }} else if (sortMode === 'title-asc') {{
      sorted.sort((a, b) => (a.dataset.title || '').localeCompare(b.dataset.title || ''));
    }} else if (sortMode === 'author-asc') {{
      sorted.sort((a, b) => (a.dataset.author || '').localeCompare(b.dataset.author || ''));
    }}

    // Reorder in DOM
    const tbody = activePanel.querySelector('tbody');
    if (tbody && sorted.length > 0) {{
      sorted.forEach(row => {{ tbody.appendChild(row); }});
    }}

    paginatePanel(activePanel, true);

    panels.filter(panel => panel !== activePanel).forEach(panel => {{
      panel.querySelectorAll('tr.paper-row').forEach(row => {{
        row.classList.remove('hidden');
        row.style.display = '';
      }});
      panel.dataset.page = '1';
      renderPager(panel, 1, 1);
    }});
  }}

  categoryButtons.forEach(btn => {{
    btn.addEventListener('click', () => setActiveTab(btn.dataset.tab));
  }});

  document.addEventListener('click', evt => {{
    const button = evt.target.closest('.pager-btn');
    if (!button || button.disabled) return;
    const panel = button.closest('.panel');
    if (!panel) return;
    const nextPage = parseInt(button.dataset.page || '1', 10);
    if (!Number.isFinite(nextPage) || nextPage < 1) return;
    panel.dataset.page = String(nextPage);
    paginatePanel(panel, false);
  }});

  // ===== Delete & Rating Handlers =====
  document.addEventListener('click', evt => {{
    const deleteBtn = evt.target.closest('.btn-delete');
    if (deleteBtn && userSession.isLoggedIn()) {{
      evt.stopPropagation();
      const paperId = deleteBtn.dataset.id;
      paperManager.delete(paperId);
      const row = deleteBtn.closest('tr.paper-row');
      if (row) row.classList.add('hidden');
      applySearch();
      return;
    }}

    const rating = evt.target.closest('.rating');
    if (rating && userSession.isLoggedIn()) {{
      evt.stopPropagation();
      const paperId = rating.dataset.id;
      const currentRating = parseInt(rating.dataset.rating || '0', 10);
      const newRating = (currentRating + 1) % 6;
      paperManager.setRating(paperId, newRating);
      rating.dataset.rating = newRating;
      rating.textContent = ['☆', '★', '★★', '★★★', '★★★★', '★★★★★'][newRating];
    }}
  }});

  // ===== Tooltip =====
  function placeTooltip(evt) {{
    const margin = 16;
    let x = evt.clientX + 18;
    let y = evt.clientY + 18;
    const rect = tooltip.getBoundingClientRect();
    if (x + rect.width + margin > window.innerWidth) x = evt.clientX - rect.width - 18;
    if (y + rect.height + margin > window.innerHeight) y = evt.clientY - rect.height - 18;
    tooltip.style.left = `${{Math.max(margin, x)}}px`;
    tooltip.style.top = `${{Math.max(margin, y)}}px`;
  }}

  document.querySelectorAll('tbody tr.paper-row td.col-title').forEach(cell => {{
    const row = cell.closest('tr.paper-row');
    if (!row) return;

    cell.addEventListener('mouseenter', (evt) => {{
      const abs = row.dataset.abstract || 'Abstract unavailable.';
      tooltip.textContent = abs;
      tooltip.style.display = 'block';
      placeTooltip(evt);
    }});

    cell.addEventListener('mousemove', placeTooltip);
    cell.addEventListener('mouseleave', () => {{
      tooltip.style.display = 'none';
    }});
  }});

  // ===== Export Functions =====
  function getAllVisiblePapers() {{
    const papers = [];
    document.querySelectorAll('tbody tr.paper-row').forEach(row => {{
      if (!row.classList.contains('hidden')) {{
        const paperId = row.dataset.id;
        papers.push({{
          title: row.dataset.title,
          author: row.dataset.author,
          date: row.dataset.date,
          category: row.dataset.category,
          rating: paperManager.getRating(paperId),
        }});
      }}
    }});
    return papers;
  }}

  function exportData(format) {{
    const papers = getAllVisiblePapers();
    if (papers.length === 0) {{
      alert('No papers to export.');
      return;
    }}

    let content, filename, type;
    if (format === 'csv') {{
      const headers = ['Title', 'Author', 'Date', 'Category', 'Rating'].join(',');
      const rows = papers.map(p => [
        `"${{(p.title || '').replace(/"/g, '""')}}"`,
        `"${{(p.author || '').replace(/"/g, '""')}}"`,
        p.date,
        `"${{(p.category || '').replace(/"/g, '""')}}"`,
        p.rating,
      ].join(','));
      content = [headers, ...rows].join('\\n');
      filename = `snn-papers-${{new Date().toISOString().split('T')[0]}}.csv`;
      type = 'text/csv';
    }} else {{
      content = JSON.stringify(papers, null, 2);
      filename = `snn-papers-${{new Date().toISOString().split('T')[0]}}.json`;
      type = 'application/json';
    }}

    const blob = new Blob([content], {{ type }});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }}

  // ===== Event Listeners =====
  input.addEventListener('input', applySearch);
  hideArxiv.addEventListener('change', applySearch);
  dateFrom.addEventListener('change', applySearch);
  dateTo.addEventListener('change', applySearch);
  sortBy.addEventListener('change', applySearch);

  // Initialize
  updateUserUI();
  applySearch();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _run_initial(existing_json: Path) -> dict:
    """Full fetch: rebuild entire dataset from scratch (one-time or manual)."""
    print(f"[INFO] Initial mode: fetching OpenAlex since {CRAWL_START_DATE} and arXiv since {ARXIV_START_DATE}", file=sys.stderr)
    try:
        oa_entries    = fetch_openalex_entries(from_date=CRAWL_START_DATE)
        arxiv_entries = fetch_arxiv_entries(from_date=ARXIV_START_DATE)
        if not oa_entries and existing_json.exists():
            print("[WARN] OpenAlex returned 0 items. Reusing cached data for this run.", file=sys.stderr)
            cached = json.loads(existing_json.read_text(encoding="utf-8"))
            if int(cached.get("total_papers", 0) or 0) < 50:
                committed = load_committed_dataset()
                if committed and int(committed.get("total_papers", 0) or 0) > int(cached.get("total_papers", 0) or 0):
                    print("[WARN] Local cache too small; recovered baseline from committed docs/papers.json.", file=sys.stderr)
                    cached = committed
            cached_entries = flatten_dataset_entries(cached)
            dataset = build_dataset(cached_entries, arxiv_entries)
            dataset["sources"] = ["OpenAlex (cached)", "arXiv"]
        else:
            dataset = build_dataset(oa_entries, arxiv_entries)
            dataset["sources"] = ["OpenAlex", "arXiv"]
    except Exception as exc:
        print(f"[WARN] Fetch failed: {exc}. Falling back to cached data.", file=sys.stderr)
        if existing_json.exists():
            dataset = json.loads(existing_json.read_text(encoding="utf-8"))
            dataset["generated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        else:
            dataset = build_dataset([], [])
    return dataset


def _run_daily(existing_json: Path) -> dict:
    """Incremental fetch: only last DAILY_DAYS days, merge with existing dataset."""
    from_date = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=DAILY_DAYS)).strftime("%Y-%m-%d")
    print(f"[INFO] Daily mode: fetching papers since {from_date}", file=sys.stderr)

    try:
        oa_new    = fetch_openalex_recent(from_date, MAX_DAILY_OPENALEX)
        arxiv_new = fetch_arxiv_entries(max_papers=MAX_DAILY_ARXIV, from_date=from_date)

        # Load existing papers to merge with
        existing_entries: list[dict] = []
        if existing_json.exists():
            try:
                cached = json.loads(existing_json.read_text(encoding="utf-8"))
                existing_entries = flatten_dataset_entries(cached)
                print(f"[INFO] Loaded {len(existing_entries)} existing papers for merge.", file=sys.stderr)
            except Exception as exc:
                print(f"[WARN] Could not load existing papers.json: {exc}", file=sys.stderr)

        # OpenAlex takes dedup priority; then arXiv new; then existing
        dataset = build_dataset(oa_new, arxiv_new + existing_entries)
        dataset["sources"] = ["OpenAlex", "arXiv"]
    except Exception as exc:
        print(f"[WARN] Daily fetch failed: {exc}. Falling back to cached data.", file=sys.stderr)
        if existing_json.exists():
            dataset = json.loads(existing_json.read_text(encoding="utf-8"))
            dataset["generated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        else:
            dataset = build_dataset([], [])
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="SNN Paper Tracker")
    parser.add_argument("--daily", action="store_true",
                        help="Incremental daily update (fetch only recent papers, merge with existing)")
    args, _ = parser.parse_known_args()

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    existing_json = DOCS_DIR / "papers.json"

    if args.daily:
        dataset = _run_daily(existing_json)
    else:
        dataset = _run_initial(existing_json)

    existing_json.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DOCS_DIR / "index.html").write_text(render_html(dataset), encoding="utf-8")
    print(f"[INFO] Done. {dataset['total_papers']} papers written to docs/", file=sys.stderr)


if __name__ == "__main__":
    main()
