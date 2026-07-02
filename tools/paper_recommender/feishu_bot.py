from __future__ import annotations

import base64
import hashlib
import hmac
import re
import time
from dataclasses import dataclass

import requests

from .models import PaperRecommendation


@dataclass
class FeishuBotClient:
    webhook_url: str
    sign_secret: str = ""

    def send_daily_recommendations(
        self,
        recommendations: list[PaperRecommendation],
        title: str,
        bitable_url: str = "",
        stats: dict[str, int] | None = None,
    ) -> None:
        payload = self._build_payload(recommendations, title=title, bitable_url=bitable_url, stats=stats or {})
        response = requests.post(self.webhook_url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data.get("StatusCode", 0) not in (0, None) or data.get("code", 0) not in (0, None):
            raise RuntimeError(f"Failed to send Feishu bot message: {data}")

    def _build_payload(
        self,
        recommendations: list[PaperRecommendation],
        title: str,
        bitable_url: str,
        stats: dict[str, int],
    ) -> dict[str, object]:
        timestamp = str(int(time.time()))
        elements = _build_card_elements(recommendations, bitable_url=bitable_url, stats=stats)
        payload: dict[str, object] = {
            "msg_type": "interactive",
            "card": {
                "config": {"wide_screen_mode": True},
                "header": {
                    "template": "blue",
                    "title": {"tag": "plain_text", "content": title},
                },
                "elements": elements,
            },
        }
        if self.sign_secret:
            payload["timestamp"] = timestamp
            payload["sign"] = _sign(timestamp, self.sign_secret)
        return payload


def _build_card_elements(
    recommendations: list[PaperRecommendation],
    bitable_url: str = "",
    stats: dict[str, int] | None = None,
) -> list[dict[str, object]]:
    stats = stats or {}
    direction_counts = _count_by_direction(recommendations)
    priority_counts = _count_by_priority(recommendations)
    elements: list[dict[str, object]] = [
        {
            "tag": "markdown",
            "content": "\n".join(
                [
                    _stats_line(len(recommendations), stats),
                    f"方向分布：{_format_distribution(direction_counts)}",
                    f"优先级：高 {priority_counts['高优先级']}｜中 {priority_counts['中优先级']}｜低 {priority_counts['低优先级']}｜归档 {priority_counts['仅归档']}",
                ]
            ),
        }
    ]
    if not recommendations:
        elements.append({"tag": "markdown", "content": "今日无强推荐论文，可稍后重试或降低筛选阈值。"})
    for index, item in enumerate(recommendations, start=1):
        paper = item.paper
        summary = item.summary
        direction = _infer_direction(item)
        score_text = _score_text(paper.score)
        business_relevance = summary.business_relevance or _business_relevance(direction, summary.recommendation_level)
        deployability = summary.deployability or _deployability(item)
        code_status = "已开源" if _is_http_url(paper.code_url) else "未知"
        keyword_hits = _keyword_hits(summary.matched_keywords or paper.score_reasons)
        display_title = summary.display_title or paper.title
        elements.append({"tag": "hr"})
        title_lines = [_paper_title_line(index, display_title, paper.arxiv_url, paper.arxiv_id)]
        if display_title != paper.title:
            title_lines.append(f"原题：{_escape_md(paper.title)}")
        elements.append(
            {
                "tag": "markdown",
                "content": "\n".join(
                    [
                        *title_lines,
                        (
                            f"{_priority_icon(summary.recommendation_level)} {summary.recommendation_level}｜"
                            f"{_action_icon(summary.action)} {summary.action}｜"
                            f"{direction}｜推荐分：{score_text}"
                        ),
                        f"业务相关度：{business_relevance}｜工程落地性：{deployability}｜代码：{code_status}",
                        "",
                        f"**推荐看点：**\n{_section_text(summary.hook or summary.one_sentence or _brief_about(item))}",
                        "",
                        f"**为什么现在值得看：**\n{_section_text(summary.why_now or _fallback_why_now(item))}",
                        "",
                        f"**论文硬信号：**\n{_hard_signals(item)}",
                        "",
                        f"**方法一句话：**\n{_method_sentence(item)}",
                        "",
                        f"**方法怎么做：**\n{_numbered_points(summary.technical_points)}",
                        "",
                        f"**关键创新点：**\n{_innovation_points(item)}",
                        "",
                        f"**为什么推荐：**\n{_section_text(summary.recommendation_reason)}",
                        "",
                        f"**业务价值：**\n{_section_text(summary.business_value)}",
                        "",
                        f"**风险：**\n{_section_text(summary.risks)}",
                        "",
                        f"**可验证依据：**\n{_bullet_points(summary.evidence_highlights[:4]) if summary.evidence_highlights else '暂无。'}",
                        "",
                        f"**命中关键词：** {keyword_hits}",
                    ]
                ),
            }
        )
        actions = _paper_actions(
            paper_url=paper.arxiv_url,
            arxiv_id=paper.arxiv_id,
            code_url=paper.code_url,
            project_url=paper.project_url,
            report_url=item.report_url,
            bitable_url=bitable_url,
        )
        if actions:
            elements.append({"tag": "action", "actions": actions})
    if bitable_url:
        elements.append({"tag": "hr"})
        elements.append(
            {
                "tag": "action",
                "actions": [
                    {
                        "tag": "button",
                        "text": {"tag": "plain_text", "content": "打开追溯表格"},
                        "url": bitable_url,
                        "type": "primary",
                    }
                ],
            }
        )
    return elements


def _count_by_direction(recommendations: list[PaperRecommendation]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in recommendations:
        direction = _infer_direction(item)
        counts[direction] = counts.get(direction, 0) + 1
    return counts


def _count_by_priority(recommendations: list[PaperRecommendation]) -> dict[str, int]:
    counts = {"高优先级": 0, "中优先级": 0, "低优先级": 0, "仅归档": 0}
    for item in recommendations:
        level = item.summary.recommendation_level
        counts[level if level in counts else "仅归档"] += 1
    return counts


def _format_distribution(counts: dict[str, int]) -> str:
    if not counts:
        return "无"
    return "｜".join(f"{name} {count}" for name, count in counts.items())


def _stats_line(recommendation_count: int, stats: dict[str, int]) -> str:
    parts = [
        f"**今日精选：{recommendation_count} 篇",
        f"候选：{stats.get('fetched', 0)}",
        f"去重后：{stats.get('deduped', 0)}",
    ]
    if stats.get("judged"):
        parts.append(f"Codex评审：{stats.get('judged', 0)}")
    return "｜".join(parts) + "**"


def _infer_direction(item: PaperRecommendation) -> str:
    if item.summary.primary_task and item.summary.primary_task != "无关":
        return item.summary.primary_task
    text = " ".join(
        [
            item.paper.title,
            item.paper.abstract,
            item.paper.tldr,
            " ".join(item.paper.score_reasons),
            item.summary.recommendation_reason,
            item.summary.business_value,
        ]
    ).lower()
    direction_rules = [
        ("检测", ["object detection", "detection", "detector", "yolo", "detr", "grounding dino"]),
        ("跟踪/ReID", ["tracking", "reid", "re-identification", "association", "tracklet"]),
        ("关键点/姿态", ["keypoint", "pose", "landmark", "skeleton", "gaze"]),
        ("VLM", ["vlm", "vision-language", "multimodal", "mllm", "visual grounding"]),
        ("自动标注/数据", ["auto-label", "annotation", "data cleaning", "dataset curation", "data engine"]),
        ("分类/人脸属性", ["classification", "scene recognition", "face attribute", "facial attribute"]),
        ("小模型/部署", ["compression", "quantization", "pruning", "distillation", "lightweight", "edge deployment"]),
        ("视频理解", ["video", "temporal", "motion", "action recognition"]),
    ]
    for name, keywords in direction_rules:
        if any(keyword in text for keyword in keywords):
            return name
    return "感知视觉"


def _score_text(score: float) -> str:
    normalized = max(0.0, min(score, 30.0))
    return f"{normalized:.1f} / 30"


def _business_relevance(direction: str, level: str) -> str:
    if direction in {"检测", "跟踪/ReID", "分类/人脸属性", "小模型/部署", "自动标注/数据"}:
        return "高" if level == "高优先级" else "中-高"
    if direction in {"VLM", "视频理解"}:
        return "中"
    return "低-中"


def _deployability(item: PaperRecommendation) -> str:
    text = " ".join([item.paper.title, item.paper.abstract, " ".join(item.paper.score_reasons)]).lower()
    if any(keyword in text for keyword in ["lightweight", "efficient", "real-time", "edge", "quantization", "pruning"]):
        return "中"
    if _infer_direction(item) in {"VLM", "视频理解"}:
        return "低-中"
    return "低"


def _brief_about(item: PaperRecommendation) -> str:
    summary = item.summary
    return _section_text(
        summary.one_sentence
        or summary.contribution
        or item.paper.tldr
        or item.paper.abstract
        or "暂无摘要信息，建议打开论文或阅读报告核对。"
    )


def _fallback_why_now(item: PaperRecommendation) -> str:
    if item.paper.hotness_signals:
        return "近期出现社区热度或开源信号，适合优先判断是否值得跟进。"
    return "论文发布时间较新，适合纳入近期方向观察。"


def _hard_signals(item: PaperRecommendation) -> str:
    signals: list[str] = []
    signals.extend(item.paper.hotness_signals[:3])
    signals.extend(item.paper.quality_signals[:3])
    if item.summary.hotness_reason:
        signals.append(item.summary.hotness_reason)
    if item.summary.quality_reason:
        signals.append(item.summary.quality_reason)
    if not signals:
        if item.paper.code_url:
            signals.append("代码链接已发现")
        signals.append(f"arXiv {item.paper.arxiv_id}")
    return _bullet_points(_dedupe_keep_order(signals), limit=5)


def _method_sentence(item: PaperRecommendation) -> str:
    summary = item.summary
    return _section_text(
        summary.contribution
        or summary.one_sentence
        or "摘要中未给出足够清晰的方法概述，建议阅读报告或论文正文。"
    )


def _innovation_points(item: PaperRecommendation) -> str:
    summary = item.summary
    candidates: list[str] = []
    candidates.extend(summary.technical_points[:3])
    if summary.results and summary.results.strip() not in {"未知", "不清楚", "证据不足"}:
        candidates.append(f"实验/证据：{summary.results}")
    if not candidates and summary.contribution:
        candidates.append(summary.contribution)
    return _bullet_points(candidates[:4]) if candidates else "暂无明确创新点，建议阅读报告核对。"


def _section_text(value: str, limit: int = 260) -> str:
    text = _escape_md(value)
    if not text:
        return "暂无。"
    return _clip_text(text, limit)


def _numbered_points(values: list[str], limit: int = 4) -> str:
    items = [_clip_text(_escape_md(item), 120) for item in values if _escape_md(item)]
    if not items:
        return "暂无明确步骤，建议阅读报告核对。"
    return "\n".join(f"{index}. {item}" for index, item in enumerate(items[:limit], start=1))


def _bullet_points(values: list[str], limit: int = 4) -> str:
    items = [_clip_text(_escape_md(item), 120) for item in values if _escape_md(item)]
    if not items:
        return "暂无。"
    return "\n".join(f"• {item}" for item in items[:limit])


def _clip_text(value: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", value).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rstrip()
    sentence_end = max(cut.rfind("。"), cut.rfind("."), cut.rfind("；"), cut.rfind(";"))
    if sentence_end > int(limit * 0.45):
        return cut[: sentence_end + 1]
    return cut + "..."


def _priority_icon(level: str) -> str:
    return {"高优先级": "🔴", "中优先级": "🟡", "低优先级": "🟢", "仅归档": "⚪"}.get(level, "⚪")


def _action_icon(action: str) -> str:
    return {"小试": "🧪", "观察": "👀", "归档": "📌", "关闭": "⏸"}.get(action, "👀")


def _keyword_hits(score_reasons: list[str]) -> str:
    keywords: list[str] = []
    for reason in score_reasons:
        if reason.startswith("codex_judge:"):
            continue
        keyword = _keyword_from_reason(reason)
        if keyword:
            keywords.append(keyword)
    keywords = _dedupe_keep_order([item for item in keywords if item and not item.startswith("cs.")])
    return " / ".join(keywords[:6]) if keywords else "规则筛选"


def _keyword_from_reason(reason: str) -> str:
    text = reason.strip()
    if not text:
        return ""
    if text.startswith("codex_keyword:"):
        return text.split(":", 1)[1].strip()
    if re.match(r"^(primary|method|engineering|category|freshness|penalty|code|citation):", text, flags=re.IGNORECASE):
        tail = text.rsplit(":", 1)[-1].strip()
        return tail if tail and "/" not in tail else ""
    if ":" in text:
        return text.split(":", 1)[1].strip()
    return text


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _paper_title_line(index: int, title: str, paper_url: str, arxiv_id: str) -> str:
    title_text = _escape_md(title)
    if _valid_paper_url(paper_url, arxiv_id):
        return f"**{index}.** [{title_text}]({paper_url})"
    return f"**{index}. {title_text}**"


def _paper_actions(
    paper_url: str,
    arxiv_id: str,
    code_url: str,
    project_url: str,
    report_url: str,
    bitable_url: str,
) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []
    if _valid_paper_url(paper_url, arxiv_id):
        actions.append(
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": "查看论文"},
                "url": paper_url,
                "type": "primary",
            }
        )
    if _is_http_url(code_url):
        actions.append(
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": "查看代码"},
                "url": code_url,
                "type": "default",
            }
        )
    if _is_http_url(project_url) and project_url != code_url:
        actions.append(
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": "项目页"},
                "url": project_url,
                "type": "default",
            }
        )
    if _is_http_url(report_url):
        actions.append(
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": "阅读报告"},
                "url": report_url,
                "type": "default",
            }
        )
    if _is_http_url(bitable_url):
        actions.append(
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": "追溯记录"},
                "url": bitable_url,
                "type": "default",
            }
        )
    return actions


def _build_markdown(
    recommendations: list[PaperRecommendation],
    bitable_url: str = "",
    stats: dict[str, int] | None = None,
) -> str:
    lines: list[str] = []
    if not recommendations:
        lines.append("今日无强推荐论文。")
    for index, item in enumerate(recommendations, start=1):
        paper = item.paper
        summary = item.summary
        lines.extend(
            [
                f"**{index}. [{_escape_md(summary.display_title or paper.title)}]({paper.arxiv_url})**",
                f"- 原题：{_escape_md(paper.title)}" if summary.display_title and summary.display_title != paper.title else "",
                f"- 推荐：{summary.recommendation_level} / {summary.action} / {paper.score:.1f}",
                f"- 看点：{_escape_md(summary.hook)}" if summary.hook else "",
                f"- 现在看：{_escape_md(summary.why_now)}" if summary.why_now else "",
                f"- 结论：{_escape_md(summary.one_sentence)}",
                f"- 理由：{_escape_md(summary.recommendation_reason)}",
                f"- 风险：{_escape_md(summary.risks)}",
            ]
        )
        if paper.code_url:
            lines.append(f"- 代码：[{paper.code_url}]({paper.code_url})")
        if item.report_url:
            lines.append(f"- 阅读报告：[{item.report_url}]({item.report_url})")
        lines.append("")
    if stats:
        lines.append(
            f"候选 {stats.get('fetched', 0)} 篇，去重后 {stats.get('deduped', 0)} 篇，入选 {len(recommendations)} 篇。"
        )
    if bitable_url:
        lines.append(f"[查看飞书多维表格]({bitable_url})")
    return "\n".join(lines).strip()


def _sign(timestamp: str, secret: str) -> str:
    string_to_sign = f"{timestamp}\n{secret}"
    digest = hmac.new(string_to_sign.encode("utf-8"), b"", hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def _escape_md(value: str) -> str:
    return value.replace("\n", " ").strip()


def _is_http_url(value: str) -> bool:
    return value.startswith("https://") or value.startswith("http://")


def _valid_paper_url(url: str, arxiv_id: str) -> bool:
    if not _is_http_url(url):
        return False
    if not arxiv_id:
        return True
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([^?#/]+)", url)
    if not match:
        return True
    linked_id = re.sub(r"v\d+$", "", match.group(1), flags=re.IGNORECASE)
    return linked_id.lower() == arxiv_id.lower()
