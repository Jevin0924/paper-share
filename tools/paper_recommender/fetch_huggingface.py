from __future__ import annotations

from datetime import datetime, timezone
from html import unescape
import logging
import re
from typing import Any

import requests

from .models import PaperCandidate
from .normalize import clean_arxiv_id, clean_text, deduplicate_papers


LOGGER = logging.getLogger("paper_recommender")
DEFAULT_URLS = ["https://huggingface.co/papers", "https://huggingface.co/papers/trending"]
ARTICLE_RE = re.compile(r"<article\b.*?</article>", flags=re.IGNORECASE | re.DOTALL)
PAPER_LINK_RE = re.compile(r'href="/papers/([^"#?]+)"', flags=re.IGNORECASE)
TITLE_LINK_RE = re.compile(
    r'<a\s+href="/papers/[^"#?]+"[^>]*>(.*?)</a>',
    flags=re.IGNORECASE | re.DOTALL,
)
ABSTRACT_RE = re.compile(r"<p\b[^>]*>(.*?)</p>", flags=re.IGNORECASE | re.DOTALL)
UPVOTE_RE = re.compile(r"Upvote\s*<div[^>]*>\s*([0-9][0-9,]*)\s*</div>", flags=re.IGNORECASE | re.DOTALL)
DATE_RE = re.compile(r"Published on\s+([A-Z][a-z]{2}\s+\d{1,2},\s+\d{4})")
ORG_RE = re.compile(
    r'<a\s+href="/[^"]+"[^>]*>\s*(?:(?!</a>).)*?<span[^>]*font-medium[^>]*>(.*?)</span>',
    flags=re.IGNORECASE | re.DOTALL,
)
GITHUB_LINK_RE = re.compile(r'<a\s+href="(https://github\.com/[^"]+)"[^>]*>(.*?)</a>', flags=re.IGNORECASE | re.DOTALL)
ARXIV_LINK_RE = re.compile(r'href="(https://arxiv\.org/(?:abs|pdf)/[^"]+)"', flags=re.IGNORECASE)
EXTERNAL_LINK_RE = re.compile(r'<a\s+href="(https?://[^"]+)"[^>]*>(.*?)</a>', flags=re.IGNORECASE | re.DOTALL)


def fetch_huggingface_papers(config: dict[str, Any]) -> list[PaperCandidate]:
    source_config = _source_config(config)
    if not source_config.get("enabled", False):
        return []
    urls = source_config.get("urls") or DEFAULT_URLS
    timeout = int(source_config.get("timeout_seconds", 20))
    max_results = int(source_config.get("max_results", 30))
    papers: list[PaperCandidate] = []
    headers = {"User-Agent": "paper-share-recommender/1.0"}
    for url in urls:
        try:
            response = requests.get(str(url), headers=headers, timeout=timeout)
            response.raise_for_status()
            papers.extend(parse_huggingface_papers(response.text, source_url=str(url)))
        except Exception as exc:  # noqa: BLE001 - hotness is an optional signal.
            LOGGER.warning("Hugging Face paper fetch failed for %s: %s", url, exc)
    return deduplicate_papers(papers)[:max_results]


def parse_huggingface_papers(html: str, source_url: str = "https://huggingface.co/papers") -> list[PaperCandidate]:
    papers: list[PaperCandidate] = []
    for article in ARTICLE_RE.findall(html):
        paper = _parse_article(article, source_url=source_url)
        if paper is not None:
            papers.append(paper)
    return deduplicate_papers(papers)


def parse_compact_number(value: str) -> int:
    text = value.strip().replace(",", "").lower()
    multiplier = 1
    if text.endswith("k"):
        multiplier = 1_000
        text = text[:-1]
    elif text.endswith("m"):
        multiplier = 1_000_000
        text = text[:-1]
    try:
        return int(float(text) * multiplier)
    except ValueError:
        return 0


def _parse_article(article: str, source_url: str) -> PaperCandidate | None:
    paper_match = PAPER_LINK_RE.search(article)
    if not paper_match:
        return None
    arxiv_id = clean_arxiv_id(paper_match.group(1))
    title = _extract_title(article)
    if not arxiv_id or not title:
        return None
    abstract = _extract_abstract(article)
    published = _extract_published(article)
    organization = _extract_organization(article)
    github_url, github_stars = _extract_github(article)
    arxiv_url = _extract_arxiv_url(article, arxiv_id)
    project_url = _extract_project_url(article, github_url, arxiv_url)
    hf_upvotes = _extract_upvotes(article)
    hotness_score = _hotness_score(hf_upvotes, github_stars)
    hotness_signals = _hotness_signals(hf_upvotes, github_stars, source_url)
    code_url = github_url
    return PaperCandidate(
        title=title,
        abstract=abstract,
        authors=[],
        published=published,
        updated=published,
        arxiv_id=arxiv_id,
        arxiv_url=arxiv_url,
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        venue=organization,
        institution_signals=[organization] if organization else [],
        code_url=code_url,
        project_url=project_url,
        hf_upvotes=hf_upvotes,
        github_stars=github_stars,
        source_names=["huggingface"],
        hotness_score=hotness_score,
        hotness_signals=hotness_signals,
        raw={
            "source": "huggingface",
            "huggingface_url": f"https://huggingface.co/papers/{arxiv_id}",
            "organization": organization,
        },
    )


def _extract_abstract(article: str) -> str:
    match = ABSTRACT_RE.search(article)
    if not match:
        return ""
    return _html_text(match.group(1))


def _extract_title(article: str) -> str:
    for match in TITLE_LINK_RE.finditer(article):
        title = _html_text(match.group(1))
        if title:
            return title
    return ""


def _extract_published(article: str) -> str:
    text = _html_text(article)
    match = DATE_RE.search(text)
    if match:
        try:
            parsed = datetime.strptime(match.group(1), "%b %d, %Y").replace(tzinfo=timezone.utc)
            return parsed.replace(microsecond=0).isoformat()
        except ValueError:
            pass
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _extract_organization(article: str) -> str:
    match = ORG_RE.search(article)
    if not match:
        return ""
    value = _html_text(match.group(1))
    blocked = {"upvote", "github", "arxiv page"}
    return "" if value.lower() in blocked else value


def _extract_github(article: str) -> tuple[str, int | None]:
    for match in GITHUB_LINK_RE.finditer(article):
        url = unescape(match.group(1)).strip()
        text = _html_text(match.group(2))
        stars_match = re.search(r"GitHub\s+([0-9][0-9.,]*\s*[kKmM]?)", text)
        stars = parse_compact_number(stars_match.group(1)) if stars_match else None
        return url, stars
    return "", None


def _extract_arxiv_url(article: str, arxiv_id: str) -> str:
    for match in ARXIV_LINK_RE.finditer(article):
        url = unescape(match.group(1)).strip()
        if clean_arxiv_id(url) == arxiv_id:
            return url
    return f"https://arxiv.org/abs/{arxiv_id}"


def _extract_project_url(article: str, github_url: str, arxiv_url: str) -> str:
    for match in EXTERNAL_LINK_RE.finditer(article):
        url = unescape(match.group(1)).strip()
        if url == github_url or url == arxiv_url or "arxiv.org/" in url:
            continue
        text = _html_text(match.group(2)).lower()
        if any(keyword in text for keyword in ["project", "demo", "page", "space", "website"]):
            return url
    return ""


def _extract_upvotes(article: str) -> int | None:
    match = UPVOTE_RE.search(article)
    return int(match.group(1).replace(",", "")) if match else None


def _hotness_score(hf_upvotes: int | None, github_stars: int | None) -> float:
    score = 0.0
    if hf_upvotes:
        score += min(4.0, hf_upvotes / 50.0)
    if github_stars:
        score += min(4.0, github_stars / 5000.0)
    return round(score, 2)


def _hotness_signals(hf_upvotes: int | None, github_stars: int | None, source_url: str) -> list[str]:
    signals: list[str] = []
    if hf_upvotes:
        signals.append(f"HF upvotes {hf_upvotes}")
    if github_stars:
        signals.append(f"GitHub stars {github_stars}")
    if signals:
        signals.append("Hugging Face daily/trending")
    elif source_url:
        signals.append("Hugging Face papers")
    return signals


def _html_text(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value)
    return clean_text(unescape(text))


def _source_config(config: dict[str, Any]) -> dict[str, Any]:
    sources = config.get("sources", {})
    if isinstance(sources, dict):
        hf = sources.get("huggingface_papers")
        if isinstance(hf, dict):
            return hf
    legacy = config.get("huggingface_papers")
    return legacy if isinstance(legacy, dict) else {}
