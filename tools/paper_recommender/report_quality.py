from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .models import PaperRecommendation


INTERNAL_SIGNAL_RE = re.compile(
    r"\b(?:primary|method|engineering|category|freshness|penalty|code|citation|codex_judge|codex_keyword):",
    flags=re.IGNORECASE,
)

PAGE_RE = re.compile(r"\bPAGE\s+\d+\b", flags=re.IGNORECASE)


@dataclass(slots=True)
class ReportQualityResult:
    passed: bool
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        return "passed" if self.passed else "failed"


def validate_report_quality(
    report: dict[str, Any],
    recommendation: PaperRecommendation,
    require_pdf_evidence: bool = True,
) -> ReportQualityResult:
    issues: list[str] = []
    warnings: list[str] = []
    flattened = _flatten_text(report)
    if INTERNAL_SIGNAL_RE.search(flattened):
        issues.append("报告正文包含内部打分信号")
    if _looks_english_heavy(str(report.get("executive_summary") or "")):
        warnings.append("一页结论英文占比偏高")
    if require_pdf_evidence and _has_pdf_basis(report) and not PAGE_RE.search(flattened):
        issues.append("报告声称读取 PDF，但正文缺少 PAGE 页码证据")
    if recommendation.summary.primary_task in {"", "无关"} and recommendation.summary.recommendation_level == "高优先级":
        issues.append("无明确感知主任务，却被标为高优先级")
    for note in _dict_list(report.get("figure_notes")):
        path = str(note.get("path") or note.get("markdown_path") or "").strip()
        if not path:
            issues.append("图表说明缺少图片路径")
    return ReportQualityResult(passed=not issues, issues=issues, warnings=warnings)


def scrub_report_data(report: dict[str, Any]) -> dict[str, Any]:
    """Remove internal scoring traces from report data before rendering or publishing."""
    return _scrub_value(report)


def _scrub_value(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).startswith("_"):
                cleaned[key] = item
                continue
            scrubbed = _scrub_value(item)
            if scrubbed not in ("", [], {}, None):
                cleaned[key] = scrubbed
        return cleaned
    if isinstance(value, list):
        return [item for item in (_scrub_value(item) for item in value) if item not in ("", [], {}, None)]
    if isinstance(value, str):
        if INTERNAL_SIGNAL_RE.search(value):
            return ""
        return value.strip()
    return value


def _has_pdf_basis(report: dict[str, Any]) -> bool:
    basis = str(report.get("report_basis") or "")
    return "PDF全文" in basis


def _flatten_text(value: Any) -> str:
    if isinstance(value, dict):
        return "\n".join(_flatten_text(item) for key, item in value.items() if not str(key).startswith("_"))
    if isinstance(value, list):
        return "\n".join(_flatten_text(item) for item in value)
    return str(value or "")


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _looks_english_heavy(value: str) -> bool:
    text = re.sub(r"\s+", "", value)
    if len(text) < 80:
        return False
    ascii_letters = sum(1 for char in text if char.isascii() and char.isalpha())
    cjk_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    return ascii_letters > cjk_chars * 2
