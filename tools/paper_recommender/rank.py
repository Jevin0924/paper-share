from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any

from .copywriting import build_display_title
from .models import PaperCandidate, PaperRecommendation, PaperSummary


def rank_papers(papers: list[PaperCandidate], config: dict[str, Any]) -> list[PaperCandidate]:
    ranked: list[PaperCandidate] = []
    for paper in papers:
        score, reasons = score_paper(paper, config)
        paper.score = score
        paper.score_reasons = reasons
        ranked.append(paper)
    return sorted(ranked, key=lambda item: (item.score, item.published), reverse=True)


def score_paper(paper: PaperCandidate, config: dict[str, Any]) -> tuple[float, list[str]]:
    breakdown = score_paper_breakdown(paper, config)
    return float(breakdown["score"]), list(breakdown["reasons"])


def score_paper_breakdown(paper: PaperCandidate, config: dict[str, Any]) -> dict[str, Any]:
    ranking = config.get("ranking", {})
    title_text = paper.title.lower()
    abstract_text = " ".join([paper.abstract, paper.tldr]).lower()
    metadata_text = " ".join(
        [
            " ".join(paper.categories),
            paper.venue,
            " ".join(paper.institution_signals),
            " ".join(paper.notable_author_signals),
            paper.code_url,
            paper.project_url,
            " ".join(paper.source_names),
        ]
    ).lower()
    all_text = " ".join([title_text, abstract_text, metadata_text])
    reasons: list[str] = []
    components: dict[str, float] = {
        "primary_task": 0.0,
        "method": 0.0,
        "engineering": 0.0,
        "code": 0.0,
        "hotness": 0.0,
        "quality": 0.0,
        "freshness": 0.0,
        "category": 0.0,
        "penalty": 0.0,
    }

    primary_task_score, primary_task_name, primary_reasons = _score_primary_tasks(
        title_text=title_text,
        abstract_text=abstract_text,
        metadata_text=metadata_text,
        ranking=ranking,
    )
    components["primary_task"] = primary_task_score
    reasons.extend(primary_reasons)
    has_primary_tasks = bool(ranking.get("primary_tasks"))
    primary_threshold = float(ranking.get("primary_task_threshold", 5.0))
    passed_primary_gate = True if not has_primary_tasks else primary_task_score >= primary_threshold

    method_score, method_reasons = _score_method_keywords(
        title_text=title_text,
        abstract_text=abstract_text,
        metadata_text=metadata_text,
        ranking=ranking,
    )
    components["method"] = method_score if passed_primary_gate else min(method_score, 1.5)
    reasons.extend(method_reasons[:8])

    engineering_score, engineering_reasons = _score_weighted_keywords(
        all_text,
        ranking.get("engineering_keywords", {}),
        cap=float(ranking.get("engineering_cap", 5.0)),
        reason_prefix="engineering",
    )
    components["engineering"] = engineering_score if passed_primary_gate else min(engineering_score, 1.0)
    reasons.extend(engineering_reasons)

    code_score, code_reasons = _score_code(paper, all_text, ranking)
    components["code"] = code_score if passed_primary_gate else 0.0
    reasons.extend(code_reasons)

    hotness_score, hotness_reasons = _score_hotness(paper, ranking)
    components["hotness"] = hotness_score
    reasons.extend(hotness_reasons)

    quality_score, quality_reasons = _score_quality(paper, ranking)
    components["quality"] = quality_score
    reasons.extend(quality_reasons)

    age_days = _age_days(paper.published)
    if age_days <= int(ranking.get("freshness_days", 7)):
        components["freshness"] = float(ranking.get("freshness_weight", 2.0)) if passed_primary_gate else 0.0
        if components["freshness"]:
            reasons.append("freshness:recent")

    category_bonus = ranking.get("category_bonus") or {"cs.CV": 1.0, "cs.MM": 0.5, "cs.AI": 0.5, "cs.LG": 0.5}
    for category in [paper.primary_category, *paper.categories]:
        if category in category_bonus:
            components["category"] = float(category_bonus[category]) if passed_primary_gate else 0.0
            if components["category"]:
                reasons.append(f"category:{category}")
            break

    penalty, penalty_reasons = _score_negative_keywords(all_text, ranking)
    components["penalty"] = penalty
    reasons.extend(penalty_reasons)
    hard_negative_reasons = _hard_negative_reasons(all_text, ranking)
    if hard_negative_reasons:
        reasons[:0] = hard_negative_reasons
        return {
            "score": 0.0,
            "primary_task": primary_task_name,
            "primary_task_score": primary_task_score,
            "passed_primary_gate": False,
            "components": components,
            "reasons": _dedupe_keep_order(reasons),
        }

    if has_primary_tasks and not passed_primary_gate:
        reasons.insert(0, f"filtered:primary_task<{primary_threshold:g}")

    score = (
        components["primary_task"]
        + components["method"]
        + components["engineering"]
        + components["code"]
        + components["hotness"]
        + components["quality"]
        + components["freshness"]
        + components["category"]
        - components["penalty"]
    )
    max_score = float(ranking.get("max_score", 30.0))
    score = max(0.0, min(score, max_score))
    return {
        "score": score,
        "primary_task": primary_task_name,
        "primary_task_score": primary_task_score,
        "passed_primary_gate": passed_primary_gate,
        "components": components,
        "reasons": _dedupe_keep_order(reasons),
    }


def _score_primary_tasks(
    title_text: str,
    abstract_text: str,
    metadata_text: str,
    ranking: dict[str, Any],
) -> tuple[float, str, list[str]]:
    tasks = ranking.get("primary_tasks") or {}
    if not tasks:
        return 0.0, "legacy", []
    weights = ranking.get("primary_task_weights") or {"title": 6, "abstract": 3, "metadata": 2}
    cap = float(ranking.get("primary_task_cap", 15.0))
    best_score = 0.0
    best_task = ""
    best_reasons: list[str] = []
    for task_name, keywords in tasks.items():
        task_score = 0.0
        task_reasons: list[str] = []
        for keyword in keywords:
            if _keyword_in_text(keyword, title_text):
                task_score += float(weights.get("title", 6))
                task_reasons.append(f"primary:{task_name}/title:{keyword}")
            if _keyword_in_text(keyword, abstract_text):
                task_score += float(weights.get("abstract", 3))
                task_reasons.append(f"primary:{task_name}/abstract:{keyword}")
            if _keyword_in_text(keyword, metadata_text):
                task_score += float(weights.get("metadata", 2))
                task_reasons.append(f"primary:{task_name}/metadata:{keyword}")
        task_score = min(task_score, cap)
        if task_score > best_score:
            best_score = task_score
            best_task = task_name
            best_reasons = task_reasons
    return best_score, best_task or "未命中", best_reasons[:8]


def _score_method_keywords(
    title_text: str,
    abstract_text: str,
    metadata_text: str,
    ranking: dict[str, Any],
) -> tuple[float, list[str]]:
    keyword_tiers = ranking.get("keyword_tiers") or _legacy_keyword_tiers(ranking)
    tier_weights = ranking.get("method_tier_weights") or {
        "title": {"A": 2.0, "B": 1.5, "C": 1.0},
        "abstract": {"A": 1.5, "B": 1.0, "C": 0.5},
        "metadata": {"A": 1.0, "B": 0.5, "C": 0.25},
    }
    cap = float(ranking.get("method_cap", 6.0))
    score = 0.0
    reasons: list[str] = []
    for tier_name, keywords in keyword_tiers.items():
        tier = str(tier_name).upper()
        for keyword in keywords:
            if _keyword_in_text(keyword, title_text):
                score += float(tier_weights.get("title", {}).get(tier, 0))
                reasons.append(f"method:{tier}/title:{keyword}")
            if _keyword_in_text(keyword, abstract_text):
                score += float(tier_weights.get("abstract", {}).get(tier, 0))
                reasons.append(f"method:{tier}/abstract:{keyword}")
            if _keyword_in_text(keyword, metadata_text):
                score += float(tier_weights.get("metadata", {}).get(tier, 0))
                reasons.append(f"method:{tier}/metadata:{keyword}")
    return min(score, cap), _dedupe_keep_order(reasons)


def _score_weighted_keywords(
    text: str,
    keywords: dict[str, Any],
    cap: float,
    reason_prefix: str,
) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    for keyword, weight in keywords.items():
        if _keyword_in_text(keyword, text):
            score += float(weight)
            reasons.append(f"{reason_prefix}:{keyword}")
    return min(score, cap), reasons


def _score_code(paper: PaperCandidate, all_text: str, ranking: dict[str, Any]) -> tuple[float, list[str]]:
    reasons: list[str] = []
    if paper.code_url:
        reasons.append("code:paper_code_url")
        return float(ranking.get("code_weight", 2.0)), reasons
    for keyword in ranking.get("code_keywords", ["github", "code", "open-source", "open source"]):
        if _keyword_in_text(keyword, all_text):
            reasons.append(f"code:{keyword}")
            return float(ranking.get("code_weight", 2.0)), reasons
    return 0.0, reasons


def _score_hotness(paper: PaperCandidate, ranking: dict[str, Any]) -> tuple[float, list[str]]:
    cap = float(ranking.get("hotness_cap", 6.0))
    score = min(float(paper.hotness_score or 0.0), cap)
    reasons: list[str] = []
    if paper.hf_upvotes:
        reasons.append(f"hotness:hf_upvotes:{paper.hf_upvotes}")
    if paper.github_stars:
        reasons.append(f"hotness:github_stars:{paper.github_stars}")
    for signal in paper.hotness_signals[:4]:
        reasons.append(f"hotness_signal:{signal}")
    return score, reasons


def _score_quality(paper: PaperCandidate, ranking: dict[str, Any]) -> tuple[float, list[str]]:
    cap = float(ranking.get("quality_cap", 6.0))
    score = min(float(paper.quality_score or 0.0), cap)
    reasons: list[str] = []
    if paper.citation_count is not None:
        reasons.append(f"quality:citations:{paper.citation_count}")
    if paper.influential_citation_count is not None:
        reasons.append(f"quality:influential_citations:{paper.influential_citation_count}")
    if paper.venue:
        reasons.append(f"quality:venue:{paper.venue}")
    for signal in paper.quality_signals[:4]:
        reasons.append(f"quality_signal:{signal}")
    return score, reasons


def _score_negative_keywords(all_text: str, ranking: dict[str, Any]) -> tuple[float, list[str]]:
    negative_weights = ranking.get("negative_keyword_weights") or {
        keyword: ranking.get("negative_keyword_weight", 3.0)
        for keyword in ranking.get("negative_keywords", [])
    }
    penalty = 0.0
    reasons: list[str] = []
    for keyword, weight in negative_weights.items():
        if _keyword_in_text(keyword, all_text):
            penalty += float(weight)
            reasons.append(f"penalty:{keyword}")
    return penalty, reasons


def _hard_negative_reasons(all_text: str, ranking: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for keyword in ranking.get("hard_negative_keywords", []):
        if _keyword_in_text(keyword, all_text):
            reasons.append(f"filtered:hard_negative:{keyword}")
    return reasons


def pick_final_recommendations(
    papers: list[PaperCandidate],
    config: dict[str, Any],
    recommended_on: str,
) -> list[PaperRecommendation]:
    final_limit = int(config.get("ranking", {}).get("final_limit", 5))
    min_score = float(config.get("ranking", {}).get("min_score", 8.0))
    selected = [paper for paper in papers if paper.score >= min_score][:final_limit]
    return [
        PaperRecommendation(
            paper=paper,
            summary=PaperSummary(
                display_title=_fallback_display_title(paper),
                hook=_fallback_hook(paper),
                one_sentence=_fallback_one_sentence(paper),
                contribution=paper.tldr or paper.abstract[:240],
                technical_points=_fallback_points(paper),
                why_now=_fallback_why_now(paper),
                quality_reason="；".join(paper.quality_signals[:3]),
                hotness_reason="；".join(paper.hotness_signals[:3]),
                evidence_highlights=[*paper.hotness_signals[:2], *paper.quality_signals[:2]],
                business_value="需结合当前感知模型和部署约束进一步评估。",
                risks="尚未完成内部复现，论文实验设置可能与业务数据分布不同。",
                action=_action_from_score(paper.score),
                recommendation_level=_level_from_score(paper.score),
                recommendation_reason="该论文与当前感知业务方向存在相关性，建议结合全文方法、实验设置和内部数据进一步评估。",
            ),
            recommended_on=recommended_on,
        )
        for paper in selected
    ]


def _fallback_one_sentence(paper: PaperCandidate) -> str:
    abstract = paper.tldr or paper.abstract
    if not abstract:
        return f"推荐关注 {paper.title}。"
    return abstract[:180].rstrip() + ("..." if len(abstract) > 180 else "")


def _fallback_display_title(paper: PaperCandidate) -> str:
    return build_display_title(paper)


def _fallback_hook(paper: PaperCandidate) -> str:
    abstract = paper.tldr or paper.abstract
    if abstract:
        return abstract[:160].rstrip() + ("..." if len(abstract) > 160 else "")
    if paper.hotness_signals:
        return "该论文近期出现在热点论文源中，建议结合正文确认真实价值。"
    return "该论文与当前感知业务方向存在相关性，建议进一步确认。"


def _fallback_why_now(paper: PaperCandidate) -> str:
    if paper.hotness_signals:
        return "近期出现社区热度或开源信号，适合优先判断是否值得跟进。"
    return "论文发布时间较新，适合纳入近期方向观察。"


def _fallback_points(paper: PaperCandidate) -> list[str]:
    points = []
    if paper.primary_category:
        points.append(f"主要类别：{paper.primary_category}")
    if paper.code_url:
        points.append("存在代码链接，便于后续复现验证")
    return points or ["等待精读后补充技术要点"]


def _level_from_score(score: float) -> str:
    if score >= 18:
        return "高优先级"
    if score >= 12:
        return "中优先级"
    if score >= 8:
        return "低优先级"
    return "仅归档"


def _action_from_score(score: float) -> str:
    if score >= 18:
        return "小试"
    if score >= 12:
        return "观察"
    if score >= 8:
        return "归档"
    return "关闭"


def _legacy_keyword_tiers(ranking: dict[str, Any]) -> dict[str, list[str]]:
    return {
        "A": list(ranking.get("strong_keywords", [])),
        "B": list(ranking.get("positive_keywords", [])),
        "C": [],
    }


def _keyword_in_text(keyword: str, text: str) -> bool:
    keyword = keyword.strip().lower()
    if not keyword or not text:
        return False
    normalized_text = _normalize_for_match(text)
    normalized_keyword = _normalize_for_match(keyword)
    if re.fullmatch(r"[a-z0-9]{1,4}", normalized_keyword):
        return re.search(rf"(?<![a-z0-9]){re.escape(normalized_keyword)}(?![a-z0-9])", normalized_text) is not None
    return normalized_keyword in normalized_text


def _normalize_for_match(value: str) -> str:
    return re.sub(r"\s+", " ", value.lower().replace("_", "-")).strip()


def _age_days(value: str) -> int:
    if not value:
        return 999
    published = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return max((datetime.now(timezone.utc) - published).days, 0)


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        key = value.lower()
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result
