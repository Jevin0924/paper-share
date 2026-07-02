from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .codex_summarizer import _run_codex
from .models import PaperRecommendation


DEFAULT_SKILL_PATH = Path("/home/wjw/.codex/skills/paper-analyzer")


def generate_paper_analyzer_markdown(
    recommendation: PaperRecommendation,
    paper_material: dict[str, Any],
    reports_config: dict[str, Any],
    codex_bin: str,
    timeout: int,
    repo_root: Path,
) -> str:
    style = str(reports_config.get("paper_analyzer_style") or "academic").strip() or "academic"
    prompt = build_paper_analyzer_markdown_prompt(
        recommendation=recommendation,
        paper_material=paper_material,
        reports_config=reports_config,
        style=style,
    )
    payload = _run_codex(codex_bin=codex_bin, prompt=prompt, timeout=timeout, repo_root=repo_root)
    return _clean_markdown_payload(payload)


def build_paper_analyzer_markdown_prompt(
    recommendation: PaperRecommendation,
    paper_material: dict[str, Any],
    reports_config: dict[str, Any],
    style: str,
) -> str:
    paper = recommendation.paper
    skill_path = Path(str(reports_config.get("paper_analyzer_skill_path") or DEFAULT_SKILL_PATH))
    skill_md = _read_text(skill_path / "SKILL.md", limit=16000)
    style_md = _read_text(skill_path / "styles" / f"{style}.md", limit=8000)
    figures = paper_material.get("figures") if isinstance(paper_material.get("figures"), list) else []
    return f"""你正在调用 Codex skill: paper-analyzer，参数：--style {style}，输出格式：Markdown。

下面是本地已安装 paper-analyzer/SKILL.md 的关键指令，请严格采用其 academic 风格的分析深度、章节组织和自检标准；但本流水线要求输出 `.md`，不是 HTML：

<paper_analyzer_skill>
{skill_md}
</paper_analyzer_skill>

<style_{style}>
{style_md}
</style_{style}>

任务要求：
1. 只输出一篇完整 Markdown 文档，不要输出 JSON、HTML、解释过程或代码围栏包裹。
2. 使用 paper-analyzer 的 academic 风格，即专业、严谨、术语准确，结构按“摘要 -> 背景与动机 -> 预备知识 -> 方法详解 -> 实验分析 -> 讨论 -> 局限分析 -> 结论”组织。
3. 优先依据 `paper_full_text_with_page_markers`，所有关键事实、实验、公式、图表都要标注页码证据，例如“见 PAGE 4”。不要编造论文没有的公式、实验、代码或结论。
4. 如果论文或全文材料中没有足够公式、图片、实验表、代码段，必须明确写“证据不足”，不要为了满足格式伪造内容。
5. 图片只能使用 `figures` 中提供的 `markdown_path`，并且每张图前后都要说明“用途 / 读图要点 / 支撑的判断”。不要输出不存在的图片路径。
6. 表格必须使用标准 Markdown 表格，且每张表后要写一段“表格解读”。
7. 代码分析：只有在论文或搜索结果中明确有公开代码时才写代码段；没有则写“本文未提供可确认的公开代码”。
8. 保留论文英文术语的中英对照，数学符号首次出现要解释含义。
9. Markdown 文档开头必须是 `# {paper.title}`。
10. 文末必须有“## 证据索引”，列出关键 PAGE 证据。

论文元信息：
- arXiv ID: {paper.arxiv_id}
- 标题: {paper.title}
- 作者: {", ".join(paper.authors[:12])}
- 发布时间: {paper.published}
- 类别: {", ".join(paper.categories)}
- 论文链接: {paper.arxiv_url}
- PDF 链接: {paper.pdf_url}
- 已知代码链接: {paper.code_url or "未知"}
- 推荐方向: {recommendation.summary.primary_task}
- 推荐理由: {recommendation.summary.recommendation_reason}
- 业务价值: {recommendation.summary.business_value}
- 风险: {recommendation.summary.risks}
- 报告依据: {paper_material.get("basis", "摘要与评审信息")}
- 文本抽取状态: {paper_material.get("status", "")}

figures:
{figures}

paper_abstract:
{paper.abstract}

paper_full_text_with_page_markers:
{paper_material.get("text", "")}
"""


def _read_text(path: Path, limit: int) -> str:
    try:
        return path.read_text(encoding="utf-8")[:limit]
    except OSError:
        return f"{path} not found"


def _clean_markdown_payload(payload: str) -> str:
    text = str(payload or "").strip()
    match = re.match(r"^```(?:markdown|md)?\s*(.*?)\s*```$", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        text = match.group(1).strip()
    return text.rstrip() + "\n"
