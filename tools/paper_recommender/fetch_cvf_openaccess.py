from __future__ import annotations

from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from .models import PaperCandidate
from .normalize import clean_text
from .venues import match_prestige_venue, prestige_venue_bonus


DEFAULT_BASE_URL = "https://openaccess.thecvf.com"


def fetch_cvf_openaccess_papers(config: dict[str, Any]) -> list[PaperCandidate]:
    source_config = config.get("sources", {}).get("cvf_openaccess", {})
    if not source_config.get("enabled", False):
        return []
    timeout = int(source_config.get("timeout_seconds", 30))
    max_results = int(source_config.get("max_results", 80))
    detail_limit = int(source_config.get("detail_limit", min(max_results, 8)))
    fetch_details = bool(source_config.get("fetch_details", True))
    venues = source_config.get("venues") or []
    papers: list[PaperCandidate] = []
    for venue_config in venues:
        if not isinstance(venue_config, dict):
            continue
        url = str(venue_config.get("url") or "").strip()
        if not url:
            continue
        html = _fetch_html(url, timeout=timeout)
        parsed = parse_cvf_openaccess_papers(
            html,
            source_url=url,
            venue=str(venue_config.get("venue") or venue_config.get("name") or "").strip(),
            year=str(venue_config.get("year") or "").strip(),
            published=str(venue_config.get("published") or "").strip(),
            config=config,
        )
        if fetch_details and detail_limit > 0:
            _fill_detail_abstracts(parsed[:detail_limit], timeout=timeout)
        papers.extend(parsed)
        if len(papers) >= max_results:
            break
    return papers[:max_results]


def _fill_detail_abstracts(papers: list[PaperCandidate], timeout: int) -> None:
    for paper in papers:
        if not paper.arxiv_url:
            continue
        try:
            paper.abstract = parse_cvf_openaccess_abstract(_fetch_html(paper.arxiv_url, timeout=timeout))
        except Exception as exc:  # noqa: BLE001 - detail pages are best-effort.
            paper.raw["cvf_detail_error"] = str(exc)


def parse_cvf_openaccess_papers(
    html: str,
    source_url: str,
    venue: str,
    year: str,
    published: str = "",
    config: dict[str, Any] | None = None,
) -> list[PaperCandidate]:
    config = config or {}
    parser = _CVFOpenAccessParser(source_url)
    parser.feed(html)
    parser.close()
    venue_text = " ".join(item for item in [venue, year] if item).strip()
    published_at = _published_at(published, year)
    papers: list[PaperCandidate] = []
    for item in parser.items:
        title = clean_text(item.get("title", ""))
        if not title:
            continue
        paper = PaperCandidate(
            title=title,
            abstract="",
            authors=item.get("authors", []),
            published=published_at,
            updated=published_at,
            arxiv_id="",
            arxiv_url=item.get("detail_url", ""),
            pdf_url=item.get("pdf_url", ""),
            categories=[venue] if venue else [],
            primary_category=venue,
            venue=venue_text or venue,
            raw={"source": "cvf_openaccess", "source_url": source_url},
        )
        paper.source_names = ["cvf_openaccess"]
        prestige_venue = match_prestige_venue(paper, config)
        if prestige_venue:
            paper.quality_score = prestige_venue_bonus(config) + 1.0
            paper.quality_signals = [f"venue {paper.venue}", f"prestige venue {prestige_venue}"]
        elif paper.venue:
            paper.quality_score = 1.0
            paper.quality_signals = [f"venue {paper.venue}"]
        papers.append(paper)
    return papers


def _fetch_html(url: str, timeout: int) -> str:
    request = Request(url, headers={"User-Agent": "paper-share-recommender/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_cvf_openaccess_abstract(html: str) -> str:
    parser = _CVFAbstractParser()
    parser.feed(html)
    parser.close()
    return clean_text(" ".join(parser.parts))


def _published_at(published: str, year: str) -> str:
    if published:
        return published
    if year and year.isdigit():
        return f"{year}-01-01T00:00:00+00:00"
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class _CVFOpenAccessParser(HTMLParser):
    def __init__(self, source_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.source_url = source_url
        self.items: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = None
        self._in_title = False
        self._collect_title = False
        self._title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {name: value or "" for name, value in attrs}
        if tag == "dt" and "ptitle" in attr.get("class", "").split():
            self._finish_current()
            self._current = {"authors": []}
            self._in_title = True
            self._title_parts = []
            return

        if self._current is None:
            return

        if self._in_title and tag == "a":
            self._collect_title = True
            self._current["detail_url"] = urljoin(self.source_url, attr.get("href", ""))
            return

        if tag == "input" and attr.get("name") == "query_author":
            author = clean_text(attr.get("value", ""))
            if author and author not in self._current["authors"]:
                self._current["authors"].append(author)
            return

        if tag == "a":
            href = attr.get("href", "")
            if href.lower().endswith(".pdf") and "_paper" in href and not self._current.get("pdf_url"):
                self._current["pdf_url"] = urljoin(self.source_url, href)

    def handle_data(self, data: str) -> None:
        if self._collect_title:
            self._title_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._collect_title:
            self._collect_title = False
            return
        if tag == "dt" and self._in_title:
            self._in_title = False
            if self._current is not None:
                self._current["title"] = clean_text(" ".join(self._title_parts))

    def close(self) -> None:
        super().close()
        self._finish_current()

    def _finish_current(self) -> None:
        if self._current and self._current.get("title"):
            self.items.append(self._current)
        self._current = None
        self._in_title = False
        self._collect_title = False
        self._title_parts = []


class _CVFAbstractParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._in_abstract = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {name: value or "" for name, value in attrs}
        if tag == "div" and attr.get("id") == "abstract":
            self._in_abstract = True

    def handle_data(self, data: str) -> None:
        if self._in_abstract:
            self.parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "div" and self._in_abstract:
            self._in_abstract = False
