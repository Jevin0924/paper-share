from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(slots=True)
class PaperCandidate:
    title: str
    abstract: str
    authors: list[str]
    published: str
    updated: str
    arxiv_id: str
    arxiv_url: str
    pdf_url: str
    categories: list[str] = field(default_factory=list)
    primary_category: str = ""
    doi: str = ""
    venue: str = ""
    institution_signals: list[str] = field(default_factory=list)
    notable_author_signals: list[str] = field(default_factory=list)
    tldr: str = ""
    code_url: str = ""
    project_url: str = ""
    citation_count: int | None = None
    influential_citation_count: int | None = None
    hf_upvotes: int | None = None
    github_stars: int | None = None
    external_ids: dict[str, Any] = field(default_factory=dict)
    source_names: list[str] = field(default_factory=list)
    hotness_score: float = 0.0
    quality_score: float = 0.0
    hotness_signals: list[str] = field(default_factory=list)
    quality_signals: list[str] = field(default_factory=list)
    score: float = 0.0
    score_reasons: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        if self.arxiv_id:
            return f"arxiv:{self.arxiv_id.lower()}"
        if self.doi:
            return f"doi:{self.doi.lower()}"
        return f"title:{self.normalized_title()}"

    def normalized_title(self) -> str:
        return normalize_title(self.title)

    def authors_text(self, limit: int = 8) -> str:
        if len(self.authors) <= limit:
            return ", ".join(self.authors)
        return ", ".join(self.authors[:limit]) + f", et al. (+{len(self.authors) - limit})"


@dataclass(slots=True)
class PaperSummary:
    display_title: str = ""
    hook: str = ""
    one_sentence: str = ""
    contribution: str = ""
    technical_points: list[str] = field(default_factory=list)
    results: str = ""
    why_now: str = ""
    quality_reason: str = ""
    hotness_reason: str = ""
    evidence_highlights: list[str] = field(default_factory=list)
    business_value: str = ""
    risks: str = ""
    action: str = "观察"
    recommendation_reason: str = ""
    recommendation_level: str = "可观察"
    primary_task: str = ""
    business_relevance: str = ""
    deployability: str = ""
    matched_keywords: list[str] = field(default_factory=list)
    codex_decision: str = ""


@dataclass(slots=True)
class PaperRecommendation:
    paper: PaperCandidate
    summary: PaperSummary
    recommended_on: str
    discovered_at: str = field(default_factory=now_iso)
    pushed: bool = False
    report_url: str = ""
    report_path: str = ""
    report_generated_at: str = ""
    report_status: str = ""
    report_basis: str = ""

    def to_bitable_fields(self, field_names: dict[str, str]) -> dict[str, object]:
        values = {
            "recommended_date": self.recommended_on,
            "first_seen_at": self.discovered_at,
            "updated_at": now_iso(),
            "arxiv_id": self.paper.arxiv_id,
            "doi": self.paper.doi,
            "title": self.paper.title,
            "display_title": self.summary.display_title,
            "authors": self.paper.authors_text(),
            "venue": self.paper.venue,
            "institutions": ", ".join(self.paper.institution_signals),
            "paper_url": self.paper.arxiv_url,
            "code_url": self.paper.code_url,
            "project_url": self.paper.project_url,
            "categories": ", ".join(self.paper.categories),
            "keywords": ", ".join(self.paper.score_reasons),
            "score": round(self.paper.score, 2),
            "hotness": "; ".join(self.paper.hotness_signals),
            "quality": "; ".join(self.paper.quality_signals),
            "recommendation_level": self.summary.recommendation_level,
            "hook": self.summary.hook,
            "one_sentence": self.summary.one_sentence,
            "contribution": self.summary.contribution,
            "technical_points": "\n".join(f"- {item}" for item in self.summary.technical_points),
            "results": self.summary.results,
            "why_now": self.summary.why_now,
            "quality_reason": self.summary.quality_reason,
            "hotness_reason": self.summary.hotness_reason,
            "evidence_highlights": "\n".join(f"- {item}" for item in self.summary.evidence_highlights),
            "business_value": self.summary.business_value,
            "risks": self.summary.risks,
            "action": self.summary.action,
            "recommendation_reason": self.summary.recommendation_reason,
            "primary_task": self.summary.primary_task,
            "business_relevance": self.summary.business_relevance,
            "deployability": self.summary.deployability,
            "matched_keywords": ", ".join(self.summary.matched_keywords),
            "codex_decision": self.summary.codex_decision,
            "report_url": self.report_url or self.report_path,
            "report_generated_at": self.report_generated_at,
            "report_status": self.report_status,
            "report_basis": self.report_basis,
            "pushed": "是" if self.pushed else "否",
        }
        return {
            field_names[key]: value
            for key, value in values.items()
            if key in field_names and field_names[key] and value not in (None, "")
        }


def normalize_title(title: str) -> str:
    chars: list[str] = []
    previous_space = False
    for char in title.strip().lower():
        if char.isalnum():
            chars.append(char)
            previous_space = False
        elif not previous_space:
            chars.append(" ")
            previous_space = True
    return " ".join("".join(chars).split())
