from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import re
import time
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.error import URLError
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

from .models import PaperCandidate
from .normalize import clean_arxiv_id, clean_text
from .venues import prestige_venue_bonus, prestige_venue_label_from_text


ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
RSS_NS = {"arxiv": "http://arxiv.org/schemas/atom", "dc": "http://purl.org/dc/elements/1.1/"}
ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_RSS_URL = "https://rss.arxiv.org/rss"


def fetch_arxiv(config: dict[str, Any]) -> list[PaperCandidate]:
    arxiv_config = config.get("arxiv", {})
    if arxiv_config.get("source", "api") == "rss":
        return fetch_arxiv_rss(config)
    categories = arxiv_config.get("categories", ["cs.CV"])
    max_results = int(arxiv_config.get("max_results", 100))
    days_back = int(arxiv_config.get("days_back", 7))
    query = " OR ".join(f"cat:{category}" for category in categories)
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    api_url = arxiv_config.get("api_url", ARXIV_API_URL)
    url = f"{api_url}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "paper-share-recommender/1.0"})
    timeout = int(arxiv_config.get("timeout_seconds", 30))
    retries = int(arxiv_config.get("retries", 3))
    retry_delay = int(arxiv_config.get("retry_delay_seconds", 10))
    try:
        body = _fetch_with_retries(request, timeout=timeout, retries=retries, retry_delay=retry_delay)
    except RuntimeError:
        if arxiv_config.get("rss_fallback", True):
            return fetch_arxiv_rss(config)
        raise
    papers = parse_arxiv_feed(body)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    return [paper for paper in papers if _parse_datetime(paper.published) >= cutoff]


def fetch_arxiv_rss(config: dict[str, Any]) -> list[PaperCandidate]:
    arxiv_config = config.get("arxiv", {})
    categories = arxiv_config.get("categories", ["cs.CV"])
    max_results = int(arxiv_config.get("max_results", 100))
    days_back = int(arxiv_config.get("days_back", 7))
    timeout = int(arxiv_config.get("timeout_seconds", 30))
    retries = int(arxiv_config.get("retries", 3))
    retry_delay = int(arxiv_config.get("retry_delay_seconds", 10))
    rss_base_url = arxiv_config.get("rss_url", ARXIV_RSS_URL)
    papers: list[PaperCandidate] = []
    for category in categories:
        url = f"{rss_base_url}/{category}"
        request = Request(url, headers={"User-Agent": "paper-share-recommender/1.0"})
        body = _fetch_with_retries(request, timeout=timeout, retries=retries, retry_delay=retry_delay)
        papers.extend(parse_arxiv_rss(body, category=category))
        if len(papers) >= max_results:
            break
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    filtered = [paper for paper in papers if _parse_datetime(paper.published) >= cutoff]
    return filtered[:max_results]


def _fetch_with_retries(request: Request, timeout: int, retries: int, retry_delay: int) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read()
        except HTTPError as exc:
            last_error = exc
            if exc.code not in (429, 500, 502, 503, 504) or attempt == retries:
                break
            retry_after = exc.headers.get("Retry-After")
            sleep_seconds = int(retry_after) if retry_after and retry_after.isdigit() else retry_delay * attempt
            time.sleep(sleep_seconds)
        except (TimeoutError, URLError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(retry_delay * attempt)
    raise RuntimeError(f"Failed to fetch arXiv API after {retries} attempts: {last_error}") from last_error


def parse_arxiv_feed(body: bytes) -> list[PaperCandidate]:
    root = ET.fromstring(body)
    papers: list[PaperCandidate] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        title = clean_text(_text(entry, "atom:title"))
        abstract = clean_text(_text(entry, "atom:summary"))
        published = _text(entry, "atom:published")
        updated = _text(entry, "atom:updated")
        arxiv_url = _text(entry, "atom:id")
        arxiv_id = clean_arxiv_id(arxiv_url)
        authors = [
            clean_text(name.text or "")
            for name in entry.findall("atom:author/atom:name", ATOM_NS)
            if clean_text(name.text or "")
        ]
        categories = [
            category.attrib.get("term", "")
            for category in entry.findall("atom:category", ATOM_NS)
            if category.attrib.get("term")
        ]
        primary_category = ""
        primary = entry.find("arxiv:primary_category", ATOM_NS)
        if primary is not None:
            primary_category = primary.attrib.get("term", "")
        doi = _text(entry, "arxiv:doi")
        comment = clean_text(_text(entry, "arxiv:comment"))
        journal_ref = clean_text(_text(entry, "arxiv:journal_ref"))
        venue = _extract_venue_label(comment=comment, journal_ref=journal_ref)
        pdf_url = ""
        for link in entry.findall("atom:link", ATOM_NS):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href", "")
                break
        paper = PaperCandidate(
                title=title,
                abstract=abstract,
                authors=authors,
                published=published,
                updated=updated,
                arxiv_id=arxiv_id,
                arxiv_url=arxiv_url,
                pdf_url=pdf_url,
                categories=categories,
                primary_category=primary_category,
                doi=doi,
                venue=venue,
                code_url=_extract_code_url(" ".join([comment, journal_ref])),
                raw={"source": "arxiv", "comment": comment, "journal_ref": journal_ref},
        )
        _apply_arxiv_venue_quality(paper)
        papers.append(paper)
    return papers


def parse_arxiv_rss(body: bytes, category: str) -> list[PaperCandidate]:
    root = ET.fromstring(body)
    papers: list[PaperCandidate] = []
    for item in root.findall("./channel/item"):
        title = clean_text(_rss_text(item, "title"))
        link = _rss_text(item, "link")
        description = clean_text(_rss_text(item, "description"))
        abstract = _extract_rss_abstract(description)
        published = _parse_rss_datetime(_rss_text(item, "pubDate"))
        arxiv_id = clean_arxiv_id(link)
        authors = [
            author.strip()
            for author in _rss_text(item, "dc:creator").split(",")
            if author.strip()
        ]
        categories = [_rss_text(item, "category") or category]
        papers.append(
            PaperCandidate(
                title=title,
                abstract=abstract,
                authors=authors,
                published=published,
                updated=published,
                arxiv_id=arxiv_id,
                arxiv_url=link,
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else "",
                categories=categories,
                primary_category=categories[0] if categories else category,
                code_url=_extract_code_url(description),
                raw={"source": "arxiv_rss"},
            )
        )
    return papers


def _text(entry: ET.Element, path: str) -> str:
    item = entry.find(path, ATOM_NS)
    if item is None or item.text is None:
        return ""
    return item.text.strip()


def _rss_text(item: ET.Element, path: str) -> str:
    found = item.find(path, RSS_NS)
    if found is None or found.text is None:
        return ""
    return found.text.strip()


def _parse_datetime(value: str) -> datetime:
    if not value:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _parse_rss_datetime(value: str) -> str:
    if not value:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    parsed = parsedate_to_datetime(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _extract_rss_abstract(description: str) -> str:
    marker = "Abstract:"
    if marker in description:
        return description.split(marker, 1)[1].strip()
    return description


def _extract_code_url(text: str) -> str:
    match = re.search(r"https?://github\.com/[^\s),.;]+", text)
    return match.group(0) if match else ""


def _extract_venue_label(comment: str, journal_ref: str) -> str:
    text = " ".join([journal_ref, comment]).strip()
    if not text:
        return ""
    return prestige_venue_label_from_text(text, {})


def _apply_arxiv_venue_quality(paper: PaperCandidate) -> None:
    if not paper.venue:
        return
    venue_name = prestige_venue_label_from_text(paper.venue, {})
    if not venue_name:
        return
    prestige_name = venue_name.split()[0]
    paper.quality_score = max(paper.quality_score, prestige_venue_bonus({}) + 1.0)
    paper.quality_signals = _dedupe_keep_order(
        [*paper.quality_signals, f"venue {paper.venue}", f"prestige venue {prestige_name}"]
    )


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value or "").strip()
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            result.append(item)
    return result
