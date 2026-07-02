from __future__ import annotations

import re
from collections.abc import Iterable

from .models import PaperCandidate, normalize_title


ARXIV_VERSION_RE = re.compile(r"v\d+$", re.IGNORECASE)


def clean_arxiv_id(value: str) -> str:
    value = value.strip().split("/")[-1]
    return ARXIV_VERSION_RE.sub("", value)


def clean_text(value: str) -> str:
    return " ".join(value.replace("\n", " ").split())


def deduplicate_papers(papers: Iterable[PaperCandidate]) -> list[PaperCandidate]:
    seen: dict[str, PaperCandidate] = {}
    unique: list[PaperCandidate] = []
    for paper in papers:
        keys = {paper.key(), f"title:{normalize_title(paper.title)}"}
        if paper.doi:
            keys.add(f"doi:{paper.doi.lower()}")
        if paper.arxiv_id:
            keys.add(f"arxiv:{paper.arxiv_id.lower()}")
        match = next((seen[key] for key in keys if key in seen), None)
        if match is not None:
            merge_paper_metadata(match, paper)
            for key in keys:
                seen[key] = match
            continue
        merge_paper_metadata(paper, paper)
        for key in keys:
            seen[key] = paper
        unique.append(paper)
    return unique


def merge_paper_metadata(target: PaperCandidate, incoming: PaperCandidate) -> None:
    if not target.abstract and incoming.abstract:
        target.abstract = incoming.abstract
    if not target.tldr and incoming.tldr:
        target.tldr = incoming.tldr
    if not target.authors and incoming.authors:
        target.authors = incoming.authors
    if not target.published and incoming.published:
        target.published = incoming.published
    if not target.updated and incoming.updated:
        target.updated = incoming.updated
    if not target.arxiv_id and incoming.arxiv_id:
        target.arxiv_id = incoming.arxiv_id
    if not target.arxiv_url and incoming.arxiv_url:
        target.arxiv_url = incoming.arxiv_url
    if not target.pdf_url and incoming.pdf_url:
        target.pdf_url = incoming.pdf_url
    if not target.primary_category and incoming.primary_category:
        target.primary_category = incoming.primary_category
    if not target.doi and incoming.doi:
        target.doi = incoming.doi
    if not target.venue and incoming.venue:
        target.venue = incoming.venue
    target.institution_signals = _dedupe_keep_order([*target.institution_signals, *incoming.institution_signals])
    target.notable_author_signals = _dedupe_keep_order([*target.notable_author_signals, *incoming.notable_author_signals])
    if not target.code_url and incoming.code_url:
        target.code_url = incoming.code_url
    if not target.project_url and incoming.project_url:
        target.project_url = incoming.project_url
    if incoming.citation_count is not None:
        target.citation_count = max(target.citation_count or 0, incoming.citation_count)
    if incoming.influential_citation_count is not None:
        target.influential_citation_count = max(
            target.influential_citation_count or 0,
            incoming.influential_citation_count,
        )
    if incoming.hf_upvotes is not None:
        target.hf_upvotes = max(target.hf_upvotes or 0, incoming.hf_upvotes)
    if incoming.github_stars is not None:
        target.github_stars = max(target.github_stars or 0, incoming.github_stars)
    target.hotness_score = max(target.hotness_score, incoming.hotness_score)
    target.quality_score = max(target.quality_score, incoming.quality_score)
    target.categories = _dedupe_keep_order([*target.categories, *incoming.categories])
    target.source_names = _dedupe_keep_order([*target.source_names, *_source_names(incoming)])
    target.hotness_signals = _dedupe_keep_order([*target.hotness_signals, *incoming.hotness_signals])
    target.quality_signals = _dedupe_keep_order([*target.quality_signals, *incoming.quality_signals])
    target.external_ids.update(incoming.external_ids)
    _merge_raw(target, incoming)


def _source_names(paper: PaperCandidate) -> list[str]:
    names = list(paper.source_names)
    source = str(paper.raw.get("source") or "").strip()
    if source:
        names.append(source)
    return names


def _merge_raw(target: PaperCandidate, incoming: PaperCandidate) -> None:
    target.raw.setdefault("merged_sources", [])
    for source in _source_names(incoming):
        if source not in target.raw["merged_sources"]:
            target.raw["merged_sources"].append(source)
    if incoming.raw:
        for key, value in incoming.raw.items():
            if key not in target.raw:
                target.raw[key] = value


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
