from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .codex_summarizer import _extract_json, _run_codex
from .copywriting import sanitize_summary_copy
from .models import PaperCandidate, PaperRecommendation, PaperSummary
from .venues import match_prestige_venue, prestige_venue_candidate_limit


def select_judge_candidates(papers: list[PaperCandidate], config: dict[str, Any]) -> list[PaperCandidate]:
    codex_config = config.get("codex", {})
    ranking_config = config.get("ranking", {})
    final_limit = int(ranking_config.get("final_limit", 5))
    candidate_limit = int(codex_config.get("judge_candidate_limit", max(final_limit * 3, final_limit)))
    min_rule_score = float(codex_config.get("judge_min_rule_score", ranking_config.get("min_score", 8.0)))
    hotness_min = float(codex_config.get("judge_min_hotness_score", ranking_config.get("judge_min_hotness_score", 3.0)))
    quality_min = float(codex_config.get("judge_min_quality_score", ranking_config.get("judge_min_quality_score", 3.0)))
    bucket_limit = int(codex_config.get("judge_bucket_limit", max(final_limit, 3)))
    venue_bucket_limit = prestige_venue_candidate_limit(config, default=max(final_limit, 5))
    selected: list[PaperCandidate] = []
    seen: set[str] = set()

    def add_bucket(candidates: list[PaperCandidate]) -> None:
        for paper in candidates:
            key = paper.key()
            if key in seen:
                continue
            selected.append(paper)
            seen.add(key)
            if len(selected) >= candidate_limit:
                return

    add_bucket([paper for paper in papers if match_prestige_venue(paper, config) and paper.score > 0][:venue_bucket_limit])
    add_bucket([paper for paper in papers if paper.score >= min_rule_score])
    if len(selected) < candidate_limit:
        add_bucket([paper for paper in papers if paper.hotness_score >= hotness_min and paper.score > 0][:bucket_limit])
    if len(selected) < candidate_limit:
        add_bucket([paper for paper in papers if paper.quality_score >= quality_min and paper.score > 0][:bucket_limit])
    return selected[:candidate_limit]


def judge_with_codex(
    papers: list[PaperCandidate],
    config: dict[str, Any],
    repo_root: Path,
    recommended_on: str,
) -> list[PaperRecommendation]:
    codex_config = config.get("codex", {})
    if not codex_config.get("enabled", True) or not codex_config.get("judge_enabled", True):
        return []
    if not papers:
        return []
    codex_bin = codex_config.get("command", "codex")
    if not shutil.which(codex_bin):
        raise RuntimeError(f"Codex command not found: {codex_bin}")
    prompt = build_judge_prompt(papers, config=config)
    timeout = int(codex_config.get("timeout_seconds", 300))
    payload = _run_codex(codex_bin=codex_bin, prompt=prompt, timeout=timeout, repo_root=repo_root)
    judgements = parse_judge_payload(payload)
    return apply_judgements(papers, judgements, recommended_on=recommended_on, config=config)


def build_judge_prompt(papers: list[PaperCandidate], config: dict[str, Any] | None = None) -> str:
    config = config or {}
    paper_payload = [
        {
            "arxiv_id": paper.arxiv_id,
            "title": paper.title,
            "authors": paper.authors[:10],
            "published": paper.published,
            "venue": paper.venue,
            "institutions": paper.institution_signals,
            "notable_authors": paper.notable_author_signals,
            "categories": paper.categories,
            "abstract": paper.abstract,
            "tldr": paper.tldr,
            "rule_score": paper.score,
            "rule_reasons": paper.score_reasons[:20],
            "source_names": paper.source_names,
            "hotness_score": paper.hotness_score,
            "quality_score": paper.quality_score,
            "hotness_signals": paper.hotness_signals,
            "quality_signals": paper.quality_signals,
            "prestige_venue": match_prestige_venue(paper, config),
            "hf_upvotes": paper.hf_upvotes,
            "github_stars": paper.github_stars,
            "citation_count": paper.citation_count,
            "influential_citation_count": paper.influential_citation_count,
            "project_url": paper.project_url,
            "code_url": paper.code_url,
            "url": paper.arxiv_url,
        }
        for paper in papers
    ]
    return (
        "你是计算机视觉感知业务的论文评审助手。你的任务不是简单摘要，而是判断论文是否值得推荐给"
        "检测、跟踪/ReID、关键点/姿态、属性/分类、视频理解、小模型部署、自动标注/数据闭环团队。\n"
        "评审规则：\n"
        "1. recommend=true 只能给业务主任务直接相关或强可迁移的论文。主任务包括：检测/开放词汇检测/定位、"
        "跟踪/ReID/轨迹关联、关键点/姿态/人体或人脸 landmark、自动标注/数据清洗/主动学习、分类/属性/人脸分析、"
        "小模型/端侧部署/量化/剪枝/蒸馏、视频感知/视频检测/视频跟踪。\n"
        "2. VLM、foundation model、benchmark、github、code、recent、lightweight 只是辅助信号，不能单独构成推荐理由。"
        "如果没有明确主任务，请 recommend=false。\n"
        "3. 纯 3D 生成、纯文生图/AIGC、纯 NLP/LLM 推理、AI skill/agent 构建、分子/蛋白/生物医学、显微镜、遥感专用论文，除非明确可迁移到上述主任务，"
        "否则 recommend=false。\n"
        "4. score 使用 0-30 分：主任务相关性 40%，业务价值 25%，工程落地性 20%，新颖性/实验可信度 15%。"
        "热点与质量信号只能帮助发现候选，不能替代业务相关性和论文事实。"
        "高优先级 >=22，中优先级 15-21.9，低优先级 10-14.9，低于 10 应 recommend=false。\n"
        "4a. 如果候选包含 prestige_venue 或 quality_signals 中有 prestige venue，说明它来自配置中的重点会议/期刊。"
        "未推荐过的重点会议/期刊论文默认值得进入每日推荐视野；只要不是第 3 条的强排除方向，"
        "即使业务主任务只是中等或间接相关，也应 recommend=true、score 至少 10、action=观察或归档。"
        "如果它还直接命中主任务，应按正常业务价值给中/高优先级。\n"
        "5. display_title 是决定点击率的核心字段，必须按以下优先级写："
        "优先使用 venue/期刊会议/机构/知名作者作为开头背书；然后明确领域或任务；再写方法抓手或痛点；最后尽量放可验证指标。"
        "推荐结构：`背书：领域任务 + 方法/痛点，指标结果`，例如 `北理工 TPAMI 2026：图像去噪用物理噪声建模，真实噪声更稳`。"
        "如果有明确开源信号，可写 `机构联合开源`；如果有明确数值，可写 `PSNR提升4dB`、`召回提升80%`、`60 FPS`、`AUC更稳`。"
        "标题必须让读者看出属于哪个领域，例如检测、图像恢复、视频理解、自动标注、Deepfake检测、姿态/关键点、文档解析。"
        "禁止编造机构、期刊、作者和指标；不要只把英文论文名翻译成中文。\n"
        "6. 推荐钩子 hook 可以更有吸引力，但仍必须有证据。避免无依据使用“吊打、暴涨、刷新 SOTA、碾压、GPT 时刻、圣杯”等表述；"
        "如果写 SOTA 或大幅提升，必须能从摘要、results 或 evidence_highlights 中找到依据。\n"
        "7. 请给出具体、克制的业务判断；不要因为摘要里出现泛化关键词或热点信号就推荐。\n"
        "只输出 JSON，不输出 Markdown，不输出解释文字。JSON 顶层对象格式为：\n"
        '{"papers":[{"arxiv_id":"...","recommend":true,"score":0,'
        '"recommendation_level":"高优先级|中优先级|低优先级|仅归档",'
        '"action":"小试|观察|归档|关闭","primary_task":"检测|跟踪/ReID|关键点/姿态|自动标注/数据|分类/属性|小模型/部署|视频理解|顶会/顶刊观察|无关",'
        '"business_relevance":"高|中-高|中|低-中|低","deployability":"高|中|低-中|低",'
        '"display_title":"中文展示标题，不超过42字","hook":"推荐钩子，说明最值得点开的看点",'
        '"one_sentence":"核心结论一句话","contribution":"核心贡献",'
        '"technical_points":["技术要点1","技术要点2"],"results":"实验或证据亮点，不清楚则写未知",'
        '"why_now":"为什么现在值得看","quality_reason":"质量信号判断","hotness_reason":"热点信号判断",'
        '"evidence_highlights":["可验证依据1","可验证依据2"],'
        '"recommendation_reason":"为什么值得推荐或不推荐","business_value":"对业务的具体价值",'
        '"risks":"风险和限制","matched_keywords":["..."],'
        '"code_url":"如果能确认官方代码或项目页则填写，否则空字符串"}]}\n'
        f"候选论文：\n{json.dumps(paper_payload, ensure_ascii=False)}"
    )


def parse_judge_payload(payload: str) -> list[dict[str, Any]]:
    data = json.loads(_extract_json(payload))
    if isinstance(data, dict):
        papers = data.get("papers", [])
        if isinstance(papers, list):
            return [item for item in papers if isinstance(item, dict)]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    raise ValueError("Codex judge JSON must contain a papers list")


def apply_judgements(
    papers: list[PaperCandidate],
    judgements: list[dict[str, Any]],
    recommended_on: str,
    config: dict[str, Any],
) -> list[PaperRecommendation]:
    min_codex_score = float(config.get("codex", {}).get("judge_min_codex_score", 10.0))
    by_id = {str(item.get("arxiv_id") or "").strip(): item for item in judgements}
    recommendations: list[PaperRecommendation] = []
    for paper in papers:
        data = by_id.get(paper.arxiv_id)
        if not data:
            continue
        recommend = _as_bool(data.get("recommend"))
        score = _clamp_score(_as_float(data.get("score"), paper.score))
        paper.raw["rule_score"] = paper.score
        paper.raw["codex_judge"] = data
        if not recommend or score < min_codex_score:
            continue
        paper.score = score
        code_url = str(data.get("code_url") or "").strip()
        if code_url.startswith(("http://", "https://")):
            paper.code_url = code_url
        matched_keywords = _string_list(data.get("matched_keywords"))
        codex_reason = str(data.get("recommendation_reason") or "").strip()
        paper.score_reasons = _dedupe_keep_order(
            [
                f"codex_judge:{codex_reason}" if codex_reason else "",
                *[f"codex_keyword:{keyword}" for keyword in matched_keywords],
                *paper.score_reasons,
            ]
        )
        summary = PaperSummary(
            display_title=str(data.get("display_title") or "").strip(),
            hook=str(data.get("hook") or "").strip(),
            one_sentence=str(data.get("one_sentence") or data.get("core_conclusion") or "").strip()
            or _fallback_one_sentence(paper),
            contribution=str(data.get("contribution") or "").strip() or paper.abstract[:240],
            technical_points=_string_list(data.get("technical_points")) or _fallback_points(paper),
            results=str(data.get("results") or "未知").strip(),
            why_now=str(data.get("why_now") or "").strip(),
            quality_reason=str(data.get("quality_reason") or "").strip(),
            hotness_reason=str(data.get("hotness_reason") or "").strip(),
            evidence_highlights=_string_list(data.get("evidence_highlights")),
            business_value=str(data.get("business_value") or "需结合内部数据和部署约束继续评估。").strip(),
            risks=str(data.get("risks") or data.get("risk") or "尚未完成内部复现，需验证业务数据分布下的效果。").strip(),
            action=_normalize_action(str(data.get("action") or ""), score),
            recommendation_reason=codex_reason or "Codex 评审认为与业务方向相关。",
            recommendation_level=_normalize_level(str(data.get("recommendation_level") or ""), score),
            primary_task=str(data.get("primary_task") or "感知视觉").strip(),
            business_relevance=str(data.get("business_relevance") or "").strip(),
            deployability=str(data.get("deployability") or "").strip(),
            matched_keywords=matched_keywords,
            codex_decision="推荐",
        )
        recommendations.append(PaperRecommendation(paper=paper, summary=sanitize_summary_copy(summary), recommended_on=recommended_on))
    return sorted(recommendations, key=lambda item: (item.paper.score, item.paper.published), reverse=True)


def _normalize_level(value: str, score: float) -> str:
    value = value.strip()
    aliases = {
        "high": "高优先级",
        "medium": "中优先级",
        "middle": "中优先级",
        "low": "低优先级",
        "archive": "仅归档",
        "not_recommended": "仅归档",
        "不推荐": "仅归档",
    }
    if value in {"高优先级", "中优先级", "低优先级", "仅归档"}:
        return value
    normalized = aliases.get(value.lower())
    if normalized:
        return normalized
    if score >= 22:
        return "高优先级"
    if score >= 15:
        return "中优先级"
    if score >= 10:
        return "低优先级"
    return "仅归档"


def _normalize_action(value: str, score: float) -> str:
    value = value.strip()
    if value in {"小试", "观察", "归档", "关闭"}:
        return value
    if score >= 22:
        return "小试"
    if score >= 15:
        return "观察"
    if score >= 10:
        return "归档"
    return "关闭"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1", "推荐", "值得推荐"}
    return False


def _as_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def _clamp_score(value: float) -> float:
    return max(0.0, min(float(value), 30.0))


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        value = value.strip()
        return [value] if value else []
    return []


def _fallback_one_sentence(paper: PaperCandidate) -> str:
    abstract = paper.tldr or paper.abstract
    if not abstract:
        return f"推荐关注 {paper.title}。"
    return abstract[:180].rstrip() + ("..." if len(abstract) > 180 else "")


def _fallback_points(paper: PaperCandidate) -> list[str]:
    points = []
    if paper.primary_category:
        points.append(f"主要类别：{paper.primary_category}")
    return points or ["等待精读后补充技术要点"]


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value:
            continue
        key = value.lower()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result
