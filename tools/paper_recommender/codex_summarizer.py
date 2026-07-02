from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .copywriting import sanitize_summary_copy
from .models import PaperRecommendation, PaperSummary


def summarize_with_codex(
    recommendations: list[PaperRecommendation],
    config: dict[str, Any],
    repo_root: Path,
) -> list[PaperRecommendation]:
    codex_config = config.get("codex", {})
    if not codex_config.get("enabled", True):
        return recommendations
    if not recommendations:
        return recommendations
    codex_bin = codex_config.get("command", "codex")
    if not shutil.which(codex_bin):
        raise RuntimeError(f"Codex command not found: {codex_bin}")
    prompt = build_prompt(recommendations)
    timeout = int(codex_config.get("timeout_seconds", 300))
    payload = _run_codex(codex_bin=codex_bin, prompt=prompt, timeout=timeout, repo_root=repo_root)
    summaries = _parse_summary_payload(payload)
    by_key = {item.get("arxiv_id", ""): item for item in summaries}
    for recommendation in recommendations:
        data = by_key.get(recommendation.paper.arxiv_id)
        if data:
            recommendation.summary = summary_from_dict(data, recommendation.summary)
            code_url = str(data.get("code_url") or "").strip()
            if code_url.startswith("http"):
                recommendation.paper.code_url = code_url
    return recommendations


def build_prompt(recommendations: list[PaperRecommendation]) -> str:
    papers = [
        {
            "arxiv_id": item.paper.arxiv_id,
            "title": item.paper.title,
            "authors": item.paper.authors[:10],
            "published": item.paper.published,
            "venue": item.paper.venue,
            "institutions": item.paper.institution_signals,
            "notable_authors": item.paper.notable_author_signals,
            "categories": item.paper.categories,
            "abstract": item.paper.abstract,
            "tldr": item.paper.tldr,
            "score": item.paper.score,
            "score_reasons": item.paper.score_reasons,
            "source_names": item.paper.source_names,
            "hotness_signals": item.paper.hotness_signals,
            "quality_signals": item.paper.quality_signals,
            "hf_upvotes": item.paper.hf_upvotes,
            "github_stars": item.paper.github_stars,
            "citation_count": item.paper.citation_count,
            "influential_citation_count": item.paper.influential_citation_count,
            "project_url": item.paper.project_url,
            "code_url": item.paper.code_url,
            "url": item.paper.arxiv_url,
        }
        for item in recommendations
    ]
    return (
        "你是计算机视觉感知团队的论文推荐助手。请基于下面论文元信息和摘要，"
        "为以下方向生成中文精读推荐：检测/开放词汇检测、跟踪/ReID、VLM/多模态感知、"
        "自动标注/数据清洗/数据引擎、图像分类/场景识别/人脸属性分类、"
        "小模型/部署/剪枝/量化、视频理解/视频感知。\n"
        "要求：只输出 JSON，不输出 Markdown，不输出解释文字。JSON 顶层对象格式为：\n"
        '{"papers":[{"arxiv_id":"...","display_title":"中文展示标题，不超过42字","hook":"推荐钩子",'
        '"one_sentence":"...","contribution":"...",'
        '"technical_points":["..."],"results":"...","business_value":"...",'
        '"why_now":"为什么现在值得看","quality_reason":"质量信号判断","hotness_reason":"热点信号判断",'
        '"evidence_highlights":["可验证依据"],'
        '"risks":"...","action":"小试|观察|归档|关闭","recommendation_level":"高优先级|中优先级|低优先级|仅归档",'
        '"recommendation_reason":"...","code_url":"如果能确认官方代码或项目页则填写，否则空字符串"}]}\n'
        "每篇摘要应具体说明为什么值得看，避免空泛表述。display_title 必须优先包含期刊/会议/机构/知名作者等背书信号，"
        "并明确论文所属领域或任务，再总结方法抓手和可验证指标；推荐结构为 `背书：领域任务 + 方法/痛点，指标结果`。"
        "没有背书信号时不要编造；有明确指标时优先写入标题，例如 `召回提升80%`、`60 FPS`、`PSNR提升4dB`。"
        "标题和正文禁止无依据使用“吊打、暴涨、刷新 SOTA、碾压、圣杯”等夸张表述。"
        "若论文不适合落地，请明确风险。\n"
        f"论文列表：\n{json.dumps(papers, ensure_ascii=False)}"
    )


def summary_from_dict(data: dict[str, Any], fallback: PaperSummary) -> PaperSummary:
    points = data.get("technical_points") or fallback.technical_points
    if isinstance(points, str):
        points = [points]
    matched_keywords = data.get("matched_keywords") or fallback.matched_keywords
    if isinstance(matched_keywords, str):
        matched_keywords = [matched_keywords]
    evidence_highlights = data.get("evidence_highlights") or fallback.evidence_highlights
    if isinstance(evidence_highlights, str):
        evidence_highlights = [evidence_highlights]
    return sanitize_summary_copy(PaperSummary(
        display_title=str(data.get("display_title") or fallback.display_title),
        hook=str(data.get("hook") or fallback.hook),
        one_sentence=str(data.get("one_sentence") or fallback.one_sentence),
        contribution=str(data.get("contribution") or fallback.contribution),
        technical_points=[str(item) for item in points],
        results=str(data.get("results") or fallback.results),
        why_now=str(data.get("why_now") or fallback.why_now),
        quality_reason=str(data.get("quality_reason") or fallback.quality_reason),
        hotness_reason=str(data.get("hotness_reason") or fallback.hotness_reason),
        evidence_highlights=[str(item) for item in (evidence_highlights or [])],
        business_value=str(data.get("business_value") or fallback.business_value),
        risks=str(data.get("risks") or fallback.risks),
        action=str(data.get("action") or fallback.action),
        recommendation_reason=str(data.get("recommendation_reason") or fallback.recommendation_reason),
        recommendation_level=str(data.get("recommendation_level") or fallback.recommendation_level),
        primary_task=str(data.get("primary_task") or fallback.primary_task),
        business_relevance=str(data.get("business_relevance") or fallback.business_relevance),
        deployability=str(data.get("deployability") or fallback.deployability),
        matched_keywords=[str(item) for item in matched_keywords],
        codex_decision=str(data.get("codex_decision") or fallback.codex_decision),
    ))


def _run_codex(codex_bin: str, prompt: str, timeout: int, repo_root: Path) -> str:
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=True) as output_file:
        command = [
            codex_bin,
            "--search",
            "exec",
            "--sandbox",
            "read-only",
            "--cd",
            str(repo_root),
            "--ephemeral",
            "--color",
            "never",
            "--output-last-message",
            output_file.name,
            "-",
        ]
        completed = subprocess.run(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        output_file.seek(0)
        last_message = output_file.read().strip()
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"codex exec failed with exit code {completed.returncode}: {message[-1000:]}")
    return last_message or completed.stdout.strip()


def _parse_summary_payload(payload: str) -> list[dict[str, Any]]:
    data = json.loads(_extract_json(payload))
    if isinstance(data, dict):
        papers = data.get("papers", [])
        if isinstance(papers, list):
            return [item for item in papers if isinstance(item, dict)]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    raise ValueError("Codex summary JSON must contain a papers list")


def _extract_json(value: str) -> str:
    value = value.strip()
    if value.startswith("```"):
        lines = value.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        value = "\n".join(lines).strip()
    start_candidates = [index for index in [value.find("{"), value.find("[")] if index >= 0]
    if not start_candidates:
        raise ValueError("No JSON object found in Codex output")
    start = min(start_candidates)
    end = max(value.rfind("}"), value.rfind("]"))
    if end < start:
        raise ValueError("Incomplete JSON in Codex output")
    return value[start : end + 1]
