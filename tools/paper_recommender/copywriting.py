from __future__ import annotations

import re

from .models import PaperCandidate, PaperSummary


HYPE_REPLACEMENTS = {
    "吊打": "优于部分基线",
    "暴涨": "提升",
    "碾压": "优于",
    "圣杯": "重要探索",
    "GPT时刻": "路线启发",
    "GPT 时刻": "路线启发",
    "刷新 SOTA": "报告较强结果",
    "刷新SOTA": "报告较强结果",
}

GENERIC_VENUES = {"", "arxiv", "corr"}
METRIC_RE = re.compile(
    r"(?:"
    r"(?:mAP|AP|PSNR|SSIM|FPS|fps|latency|recall|accuracy|F1|IoU|AUC)\s*(?:[=:]?\s*)?"
    r"[+\-]?\d+(?:\.\d+)?\s*(?:%|dB|ms|FPS|fps)?"
    r"|[+\-]?\d+(?:\.\d+)?\s*(?:%|dB|ms|FPS|fps)\s*"
    r"(?:mAP|AP|PSNR|SSIM|FPS|fps|latency|recall|accuracy|F1|IoU|AUC)?"
    r")",
    flags=re.IGNORECASE,
)


def sanitize_summary_copy(summary: PaperSummary) -> PaperSummary:
    summary.display_title = sanitize_copy_text(summary.display_title)
    summary.hook = sanitize_copy_text(summary.hook)
    summary.one_sentence = sanitize_copy_text(summary.one_sentence)
    summary.contribution = sanitize_copy_text(summary.contribution)
    summary.results = sanitize_copy_text(summary.results)
    summary.why_now = sanitize_copy_text(summary.why_now)
    summary.quality_reason = sanitize_copy_text(summary.quality_reason)
    summary.hotness_reason = sanitize_copy_text(summary.hotness_reason)
    summary.evidence_highlights = [sanitize_copy_text(item) for item in summary.evidence_highlights]
    summary.business_value = sanitize_copy_text(summary.business_value)
    summary.risks = sanitize_copy_text(summary.risks)
    summary.recommendation_reason = sanitize_copy_text(summary.recommendation_reason)
    return summary


def sanitize_copy_text(value: str) -> str:
    text = str(value or "")
    for source, replacement in HYPE_REPLACEMENTS.items():
        text = text.replace(source, replacement)
    return text


def annotate_copywriting_signals(papers: list[PaperCandidate], config: dict) -> list[PaperCandidate]:
    names = [
        str(name).strip()
        for name in (config.get("copywriting", {}).get("notable_authors") or [])
        if str(name).strip()
    ]
    lower_names = {name.lower(): name for name in names}
    for paper in papers:
        for author in paper.authors:
            canonical = lower_names.get(author.lower())
            if canonical and canonical not in paper.notable_author_signals:
                paper.notable_author_signals.append(canonical)
        if paper.notable_author_signals:
            paper.quality_signals = _dedupe_keep_order(
                [*paper.quality_signals, *[f"notable author {name}" for name in paper.notable_author_signals]]
            )
    return papers


def build_display_title(paper: PaperCandidate, primary_task: str = "") -> str:
    backing = _backing_signal(paper)
    field = primary_task or _field_from_text(paper)
    method = _method_phrase(paper)
    metric = extract_metric_highlight(" ".join([paper.title, paper.abstract, paper.tldr]))
    if backing and metric:
        title = f"{backing}：{field}{method}，{metric}"
    elif backing:
        title = f"{backing}：{field}{method}"
    elif metric:
        title = f"{field}{method}，{metric}"
    else:
        title = f"{field}{method}"
    return sanitize_copy_text(_clip_title(title))


def extract_metric_highlight(text: str) -> str:
    matches = [match.group(0).strip() for match in METRIC_RE.finditer(text or "")]
    for match in matches:
        if any(char.isdigit() for char in match):
            return re.sub(r"\s+", " ", match)
    return ""


def _backing_signal(paper: PaperCandidate) -> str:
    venue = str(paper.venue or "").strip()
    if venue.lower() not in GENERIC_VENUES:
        return venue
    if paper.institution_signals:
        return paper.institution_signals[0]
    if paper.notable_author_signals:
        return paper.notable_author_signals[0]
    if paper.code_url:
        owner = _github_owner(paper.code_url)
        if owner:
            return f"{owner}开源"
    return ""


def _github_owner(url: str) -> str:
    match = re.search(r"github\.com/([^/\s]+)", url or "", flags=re.IGNORECASE)
    return match.group(1) if match else ""


def _field_from_text(paper: PaperCandidate) -> str:
    text = " ".join([paper.title, paper.abstract, paper.tldr, " ".join(paper.score_reasons)]).lower()
    rules = [
        ("图像恢复", ["image restoration", "denoising", "deblurring", "super-resolution", "deraining"]),
        ("实时检测", ["real-time detection", "detection transformer", "object detection", "detector", "detr", "yolo"]),
        ("自动标注", ["annotate", "annotation", "label noise", "auto-label", "data engine"]),
        ("长视频理解", ["long video", "video understanding", "video mme", "lvbench"]),
        ("Deepfake检测", ["deepfake", "fake face", "forgery"]),
        ("姿态/关键点", ["pose", "keypoint", "hand"]),
        ("文档解析", ["document parsing", "ocr", "text-rich"]),
    ]
    for label, keywords in rules:
        if any(keyword in text for keyword in keywords):
            return label
    return "视觉感知"


def _method_phrase(paper: PaperCandidate) -> str:
    title = paper.title.strip()
    if not title:
        return "新论文"
    for separator in [":", "："]:
        if separator in title:
            head, tail = title.split(separator, 1)
            if 2 <= len(head.strip()) <= 24:
                return f"{head.strip()}，{_clip_fragment(tail.strip(), 20)}"
    return _clip_fragment(title, 26)


def _clip_fragment(value: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    return text if len(text) <= limit else text[:limit].rstrip()


def _clip_title(value: str, limit: int = 58) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    return text if len(text) <= limit else text[:limit].rstrip()


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
