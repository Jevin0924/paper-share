from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .codex_summarizer import _run_codex
from .models import PaperRecommendation


DEFAULT_SKILL_PATH = Path("/home/wjw/.codex/skills/paper-analyzer")


def generate_paper_craft_html(
    recommendation: PaperRecommendation,
    paper_material: dict[str, Any],
    reports_config: dict[str, Any],
    codex_bin: str,
    timeout: int,
    repo_root: Path,
) -> str:
    style = str(reports_config.get("paper_craft_style") or reports_config.get("paper_analyzer_style") or "academic").strip() or "academic"
    prompt = build_paper_craft_html_prompt(
        recommendation=recommendation,
        paper_material=paper_material,
        reports_config=reports_config,
        style=style,
    )
    payload = _run_codex(codex_bin=codex_bin, prompt=prompt, timeout=timeout, repo_root=repo_root)
    return clean_paper_craft_html(payload)


def build_paper_craft_html_prompt(
    recommendation: PaperRecommendation,
    paper_material: dict[str, Any],
    reports_config: dict[str, Any],
    style: str,
) -> str:
    paper = recommendation.paper
    skill_path = Path(str(reports_config.get("paper_craft_skill_path") or DEFAULT_SKILL_PATH))
    skill_md = _read_text(skill_path / "SKILL.md", limit=18000)
    style_md = _read_text(skill_path / "styles" / f"{style}.md", limit=10000)
    figures = _figures_for_html(paper_material.get("figures"))
    return f"""你正在调用 paper-craft-skills / paper-analyzer 子技能，参数：--style {style}，输出格式：HTML。

下面是本地已安装的 paper-craft-skills/paper-analyzer 指令，请采用它的深度、结构、自检标准和 HTML 呈现方式。
本流水线是非交互式任务：不要询问用户，不要输出过程，不要写入本地文件，只在最终回复中输出一份完整 HTML。

<paper_craft_skill>
{skill_md}
</paper_craft_skill>

<style_{style}>
{style_md}
</style_{style}>

任务要求：
1. 只输出一篇完整 HTML 文档，不要输出 JSON、Markdown、解释文字或代码围栏包裹。
2. HTML 必须包含 `<!doctype html>` 或 `<html>`、`<head>`、`<body>` 和闭合 `</html>`。
3. 使用 paper-craft-skills / paper-analyzer 的 {style} 风格生成中文论文深度解析长文。
4. 优先依据 `paper_full_text_with_page_markers`，关键事实、实验、公式、图表都要标注页码证据，例如“见 PAGE 4”。
5. 不要编造论文没有的公式、实验、代码、源码路径或结论。证据不足时明确写“证据不足”。
6. 图片只能使用 figures 中提供的 `html_path` 作为 `<img src>`，不要输出不存在的图片路径。
7. 如果有 Mermaid 图，请在 HTML 中包含 Mermaid CDN 初始化；如果有公式，请包含 KaTeX 或 MathJax 渲染支持。
8. 如果无法确认公开代码实现，代码分析部分必须写“本文未提供可确认的公开代码”，不要伪造代码段。
9. 页面应可直接作为 GitLab Pages 静态页面打开。
10. 文末必须有“证据索引”，列出关键 PAGE 证据。

论文元信息：
{json.dumps(
    {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "authors": paper.authors[:12],
        "published": paper.published,
        "categories": paper.categories,
        "paper_url": paper.arxiv_url,
        "pdf_url": paper.pdf_url,
        "code_url": paper.code_url or "",
        "project_url": paper.project_url or "",
        "recommended_direction": recommendation.summary.primary_task,
        "recommendation_reason": recommendation.summary.recommendation_reason,
        "business_value": recommendation.summary.business_value,
        "risks": recommendation.summary.risks,
        "report_basis": paper_material.get("basis", "摘要与评审信息"),
        "text_status": paper_material.get("status", ""),
        "figures": figures,
    },
    ensure_ascii=False,
    indent=2,
)}

paper_abstract:
{paper.abstract}

paper_full_text_with_page_markers:
{paper_material.get("text", "")}
"""


def clean_paper_craft_html(payload: str) -> str:
    text = str(payload or "").strip()
    match = re.match(r"^```(?:html)?\s*(.*?)\s*```$", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1).strip()
    if not is_valid_html(text):
        extracted = _extract_html_document(text)
        if extracted:
            text = extracted
    if not is_valid_html(text):
        raise ValueError("paper-craft output is not a complete HTML document")
    return text.rstrip() + "\n"


def is_valid_html(value: str) -> bool:
    text = str(value or "").strip().lower()
    return "<html" in text and "</html>" in text


def _extract_html_document(value: str) -> str:
    text = str(value or "").strip()
    lower = text.lower()
    starts = [index for index in (lower.find("<!doctype"), lower.find("<html")) if index >= 0]
    if not starts:
        return ""
    start = min(starts)
    end = lower.rfind("</html>")
    if end < start:
        return ""
    return text[start : end + len("</html>")].strip()


def _figures_for_html(value: Any) -> list[dict[str, Any]]:
    figures = [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []
    result: list[dict[str, Any]] = []
    for figure in figures:
        copied = dict(figure)
        asset_path = str(copied.get("path") or "").strip().removeprefix("../").lstrip("/")
        if asset_path and not copied.get("html_path"):
            copied["html_path"] = "../../" + asset_path
        result.append(copied)
    return result


def _read_text(path: Path, limit: int) -> str:
    try:
        return path.read_text(encoding="utf-8")[:limit]
    except OSError:
        return f"{path} not found"
