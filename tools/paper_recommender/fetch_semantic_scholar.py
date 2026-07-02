from __future__ import annotations

import json
import os
from typing import Any
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from .models import PaperCandidate
from .venues import match_prestige_venue, prestige_venue_bonus


BASE_URL = "https://api.semanticscholar.org/graph/v1"
FIELDS = ",".join(
    [
        "title",
        "abstract",
        "venue",
        "year",
        "citationCount",
        "influentialCitationCount",
        "externalIds",
        "openAccessPdf",
        "tldr",
        "authors",
    ]
)


def enrich_with_semantic_scholar(
    papers: list[PaperCandidate],
    config: dict[str, Any],
) -> list[PaperCandidate]:
    ss_config = config.get("semantic_scholar", {})
    if not ss_config.get("enabled", True):
        return papers
    timeout = int(ss_config.get("timeout_seconds", 20))
    for paper in papers:
        try:
            data = _fetch_by_arxiv_id(paper.arxiv_id, timeout=timeout)
            if not data:
                data = _fetch_by_title(paper.title, timeout=timeout)
            if data:
                apply_semantic_scholar_metadata(paper, data, config=config)
        except Exception as exc:  # noqa: BLE001 - enrichment is best-effort.
            paper.raw["semantic_scholar_error"] = str(exc)
    return papers


def apply_semantic_scholar_metadata(
    paper: PaperCandidate,
    data: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> None:
    config = config or {}
    paper.venue = data.get("venue") or paper.venue
    paper.citation_count = data.get("citationCount")
    paper.influential_citation_count = data.get("influentialCitationCount")
    paper.external_ids.update(data.get("externalIds") or {})
    paper.doi = paper.external_ids.get("DOI") or paper.doi
    tldr = data.get("tldr") or {}
    paper.tldr = tldr.get("text") or paper.tldr
    open_access_pdf = data.get("openAccessPdf") or {}
    paper.pdf_url = open_access_pdf.get("url") or paper.pdf_url
    paper.quality_score = max(paper.quality_score, _quality_score(paper, config))
    paper.quality_signals = _dedupe_keep_order([*paper.quality_signals, *_quality_signals(paper, config)])
    paper.raw["semantic_scholar"] = data


def _quality_score(paper: PaperCandidate, config: dict[str, Any]) -> float:
    score = 0.0
    if paper.citation_count:
        score += min(3.0, paper.citation_count / 25.0)
    if paper.influential_citation_count:
        score += min(4.0, paper.influential_citation_count / 5.0)
    if paper.venue and paper.venue.lower() not in {"arxiv", "corr"}:
        score += 1.0
    if match_prestige_venue(paper, config):
        score += prestige_venue_bonus(config)
    if paper.tldr:
        score += 0.5
    return round(score, 2)


def _quality_signals(paper: PaperCandidate, config: dict[str, Any]) -> list[str]:
    signals: list[str] = []
    if paper.citation_count is not None:
        signals.append(f"Semantic Scholar citations {paper.citation_count}")
    if paper.influential_citation_count is not None:
        signals.append(f"influential citations {paper.influential_citation_count}")
    if paper.venue:
        signals.append(f"venue {paper.venue}")
    prestige_venue = match_prestige_venue(paper, config)
    if prestige_venue:
        signals.append(f"prestige venue {prestige_venue}")
    if paper.tldr:
        signals.append("Semantic Scholar TLDR")
    return signals


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value or "").strip()
        if not item:
            continue
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _fetch_by_arxiv_id(arxiv_id: str, timeout: int) -> dict[str, Any] | None:
    if not arxiv_id:
        return None
    paper_id = quote(f"ARXIV:{arxiv_id}", safe="")
    url = f"{BASE_URL}/paper/{paper_id}?{urlencode({'fields': FIELDS})}"
    return _get_json(url, timeout=timeout)


def _fetch_by_title(title: str, timeout: int) -> dict[str, Any] | None:
    if not title:
        return None
    url = f"{BASE_URL}/paper/search/match?{urlencode({'query': title, 'fields': FIELDS})}"
    return _get_json(url, timeout=timeout)


def _get_json(url: str, timeout: int) -> dict[str, Any] | None:
    headers = {"User-Agent": "paper-share-recommender/1.0"}
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if api_key:
        headers["x-api-key"] = api_key
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=timeout) as response:
            if response.status == 404:
                return None
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise
