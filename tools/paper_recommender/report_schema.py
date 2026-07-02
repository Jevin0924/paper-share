from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


HEILMEIER_SECTION_ORDER = [
    ("what", "1. 这篇论文要做什么？"),
    ("current_limits", "2. 现有方法有什么限制？"),
    ("method", "3. 方法怎么做？"),
    ("method_details", "4. 关键机制与数学细节"),
    ("who_cares", "5. 谁会关心这项工作？"),
    ("experiments", "6. 实验是否支撑结论？"),
    ("risks", "7. 风险、成本与边界"),
]


@dataclass(slots=True)
class ReportSection:
    key: str
    title: str
    paper_claims: list[str] = field(default_factory=list)
    analysis: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ReportPayload:
    executive_summary: str = ""
    target_reader: str = ""
    business_action: str = ""
    sections: list[ReportSection] = field(default_factory=list)
    experiment_tables: list[dict[str, Any]] = field(default_factory=list)
    mermaid: str = ""
    figure_notes: list[dict[str, Any]] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    report_basis: str = ""


def report_payload_from_dict(report: dict[str, Any]) -> ReportPayload:
    heilmeier = report.get("heilmeier") if isinstance(report.get("heilmeier"), dict) else {}
    sections: list[ReportSection] = []
    for key, title in HEILMEIER_SECTION_ORDER:
        raw_section = heilmeier.get(key) if isinstance(heilmeier, dict) else None
        if isinstance(raw_section, dict):
            sections.append(
                ReportSection(
                    key=key,
                    title=str(raw_section.get("title") or title),
                    paper_claims=_string_list(raw_section.get("paper_claims") or raw_section.get("paper_claim")),
                    analysis=_string_list(raw_section.get("analysis")),
                    evidence=_string_list(raw_section.get("evidence")),
                )
            )
    if not sections:
        sections = _legacy_sections(report)
    return ReportPayload(
        executive_summary=str(report.get("executive_summary") or ""),
        target_reader=str(report.get("target_reader") or ""),
        business_action=str(report.get("business_action") or report.get("implementation_summary") or ""),
        sections=sections,
        experiment_tables=_dict_list(report.get("experiment_tables")),
        mermaid=str(report.get("method_mermaid") or report.get("mermaid") or ""),
        figure_notes=_dict_list(report.get("figure_notes")),
        open_questions=_string_list(report.get("open_questions")),
        evidence=_string_list(report.get("evidence")),
        report_basis=str(report.get("report_basis") or ""),
    )


def ensure_heilmeier_shape(report: dict[str, Any]) -> dict[str, Any]:
    """Populate Heilmeier fields from legacy report fields when the model returns the old schema."""
    if isinstance(report.get("heilmeier"), dict):
        return report
    legacy = {
        "what": {
            "paper_claims": _string_list(report.get("problem")),
            "analysis": [],
            "evidence": _page_evidence(report, ["problem"]),
        },
        "current_limits": {
            "paper_claims": _string_list(report.get("why_problem_matters")),
            "analysis": [],
            "evidence": _page_evidence(report, ["why_problem_matters"]),
        },
        "method": {
            "paper_claims": [str(report.get("method_overview") or "").strip()],
            "analysis": [],
            "evidence": _page_evidence(report, ["method_overview"]),
        },
        "method_details": {
            "paper_claims": _string_list(report.get("method_details")) + _string_list(report.get("method_deep_dive")),
            "analysis": [],
            "evidence": _page_evidence(report, ["method_details", "method_deep_dive"]),
        },
        "who_cares": {
            "paper_claims": [],
            "analysis": _string_list(report.get("business_value")),
            "evidence": _page_evidence(report, ["business_value"]),
        },
        "experiments": {
            "paper_claims": _string_list(report.get("experiments")),
            "analysis": [],
            "evidence": _page_evidence(report, ["experiments"]),
        },
        "risks": {
            "paper_claims": [],
            "analysis": _string_list(report.get("risks")) + _string_list(report.get("implementation_plan")),
            "evidence": _page_evidence(report, ["risks", "implementation_plan"]),
        },
    }
    merged = dict(report)
    merged["heilmeier"] = legacy
    if not merged.get("business_action"):
        action_items = _string_list(report.get("implementation_plan"))
        merged["business_action"] = "；".join(action_items[:2])
    return merged


def render_heilmeier_markdown(report: dict[str, Any]) -> str:
    payload = report_payload_from_dict(ensure_heilmeier_shape(report))
    lines: list[str] = []
    for section in payload.sections:
        lines.extend([f"## {section.title}", ""])
        if section.paper_claims:
            lines.extend(["**论文事实**", "", *_markdown_list(section.paper_claims), ""])
        if section.analysis:
            lines.extend(["**业务判断**", "", *_markdown_list(section.analysis), ""])
        if section.evidence:
            lines.extend(["**证据**", "", *_markdown_list(section.evidence), ""])
    return "\n".join(lines).strip()


def _legacy_sections(report: dict[str, Any]) -> list[ReportSection]:
    shaped = ensure_heilmeier_shape(report)
    sections: list[ReportSection] = []
    for key, title in HEILMEIER_SECTION_ORDER:
        raw_section = shaped.get("heilmeier", {}).get(key, {})
        sections.append(
            ReportSection(
                key=key,
                title=title,
                paper_claims=_string_list(raw_section.get("paper_claims")),
                analysis=_string_list(raw_section.get("analysis")),
                evidence=_string_list(raw_section.get("evidence")),
            )
        )
    return sections


def _page_evidence(report: dict[str, Any], keys: list[str]) -> list[str]:
    text = " ".join(_flatten_text(report.get(key)) for key in keys)
    if "PAGE " not in text.upper():
        return []
    return ["本节结论已在正文中标注 PAGE 页码；详见文末证据索引。"]


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return []


def _markdown_list(items: list[str]) -> list[str]:
    return [f"- {item}" for item in items if item]


def _flatten_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(_flatten_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(_flatten_text(item) for item in value.values())
    return str(value or "")
