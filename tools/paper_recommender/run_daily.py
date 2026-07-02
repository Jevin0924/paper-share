from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.paper_recommender.codex_judge import judge_with_codex, select_judge_candidates
from tools.paper_recommender.codex_summarizer import summarize_with_codex
from tools.paper_recommender.config import env_optional, load_config, load_env_file, require_env
from tools.paper_recommender.copywriting import annotate_copywriting_signals
from tools.paper_recommender.feishu_auth import TenantTokenClient, resolve_wiki_bitable_app_token
from tools.paper_recommender.feishu_bitable import FeishuBitableClient
from tools.paper_recommender.feishu_bot import FeishuBotClient
from tools.paper_recommender.fetch_arxiv import fetch_arxiv
from tools.paper_recommender.fetch_cvf_openaccess import fetch_cvf_openaccess_papers
from tools.paper_recommender.fetch_huggingface import fetch_huggingface_papers
from tools.paper_recommender.fetch_semantic_scholar import enrich_with_semantic_scholar
from tools.paper_recommender.github_pages import publish_reports_to_github_pages
from tools.paper_recommender.models import PaperCandidate, PaperRecommendation
from tools.paper_recommender.normalize import deduplicate_papers
from tools.paper_recommender.rank import pick_final_recommendations, rank_papers
from tools.paper_recommender.report_generator import generate_reports


LOGGER = logging.getLogger("paper_recommender")
REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.debug)
    load_env_file(args.env)
    config = load_config(args.config)
    if args.no_codex:
        config.setdefault("codex", {})["enabled"] = False
    if args.no_semantic_scholar:
        config.setdefault("semantic_scholar", {})["enabled"] = False
    if args.no_huggingface:
        config.setdefault("sources", {}).setdefault("huggingface_papers", {})["enabled"] = False
    if args.no_cvf_openaccess:
        config.setdefault("sources", {}).setdefault("cvf_openaccess", {})["enabled"] = False
    if args.arxiv_max_results:
        config.setdefault("arxiv", {})["max_results"] = args.arxiv_max_results
    if args.arxiv_source:
        config.setdefault("arxiv", {})["source"] = args.arxiv_source
    if args.no_reports:
        config.setdefault("reports", {})["enabled"] = False
    if args.force_reports:
        config.setdefault("reports", {})["force_regenerate"] = True
    if args.allow_repeat:
        config.setdefault("history", {})["enabled"] = False
    if args.check_feishu:
        client = build_bitable_client(config)
        records = client.list_records()
        print(json.dumps({"feishu": "ok", "records": len(records)}, ensure_ascii=False, indent=2))
        return 0
    if args.ensure_feishu_fields:
        client = build_bitable_client(config)
        result = client.ensure_configured_fields(field_type_by_key=feishu_field_types())
        print(json.dumps({"feishu_fields": "ok", **result}, ensure_ascii=False, indent=2))
        return 0
    recommended_on = args.date or today_for_config(config)
    apply_report_publish_templates(config, recommended_on)
    if args.publish_existing_reports:
        publish_existing_reports(config, recommended_on=recommended_on, repo_root=REPO_ROOT)
        return 0
    history_state = load_recommendation_state(config, REPO_ROOT)

    LOGGER.info("Starting daily paper recommendation for %s", recommended_on)
    papers = sample_papers() if args.sample else fetch_arxiv(config)
    arxiv_count = len(papers)
    hf_count = 0
    cvf_count = 0
    LOGGER.info("Fetched %d arXiv papers", arxiv_count)
    if args.sample:
        LOGGER.info("Using built-in sample papers; skipping external hotness and Semantic Scholar enrichment")
    else:
        hf_papers = fetch_huggingface_papers(config)
        if hf_papers:
            papers.extend(hf_papers)
            hf_count = len(hf_papers)
            LOGGER.info("Fetched %d Hugging Face paper signals", hf_count)
        cvf_papers = fetch_cvf_openaccess_papers(config)
        if cvf_papers:
            papers.extend(cvf_papers)
            cvf_count = len(cvf_papers)
            LOGGER.info("Fetched %d CVF Open Access papers", cvf_count)
    deduped = deduplicate_papers(papers)
    deduped, history_filtered = filter_previously_pushed_papers(deduped, config, history_state)
    if history_filtered:
        LOGGER.info("Filtered %d previously pushed papers", history_filtered)
    if not args.sample:
        deduped = enrich_with_semantic_scholar(deduped, config)
    deduped = annotate_copywriting_signals(deduped, config)
    LOGGER.info("Deduplicated to %d papers", len(deduped))
    ranked = rank_papers(deduped, config)
    codex_enabled = bool(config.get("codex", {}).get("enabled", True))
    codex_review_attempted = False
    judge_candidates = select_judge_candidates(ranked, config) if codex_enabled else []
    recommendations = []
    if judge_candidates:
        try:
            codex_review_attempted = True
            recommendations = judge_with_codex(judge_candidates, config, repo_root=REPO_ROOT, recommended_on=recommended_on)
            LOGGER.info(
                "Codex judged %d candidates and recommended %d papers",
                len(judge_candidates),
                len(recommendations),
            )
        except Exception as exc:  # noqa: BLE001 - rule recommendations are the safe fallback.
            codex_review_attempted = False
            recommendations = []
            LOGGER.warning("Codex judge failed; falling back to rule ranking: %s", exc)
    if not codex_review_attempted:
        recommendations = pick_final_recommendations(ranked, config, recommended_on=recommended_on)
        if recommendations and codex_enabled:
            try:
                recommendations = summarize_with_codex(recommendations, config, repo_root=REPO_ROOT)
                LOGGER.info("Codex summaries generated")
            except Exception as exc:  # noqa: BLE001 - fallback summaries are acceptable.
                LOGGER.warning("Codex summary failed; using fallback summaries: %s", exc)
    final_limit = int(config.get("ranking", {}).get("final_limit", 5))
    recommendations = recommendations[: args.limit or final_limit]
    LOGGER.info("Selected %d recommendations", len(recommendations))

    stats = {
        "fetched": len(papers),
        "arxiv": arxiv_count,
        "huggingface": hf_count,
        "cvf_openaccess": cvf_count,
        "deduped": len(deduped),
        "history_filtered": history_filtered,
        "judged": len(judge_candidates),
        "selected": len(recommendations),
    }
    if args.dry_run:
        print(json.dumps(to_json_payload(recommendations, stats), ensure_ascii=False, indent=2))
        return 0

    if recommendations and config.get("reports", {}).get("enabled", True):
        try:
            recommendations = generate_reports(recommendations, config, repo_root=REPO_ROOT)
            generated_count = sum(1 for item in recommendations if item.report_status == "已生成")
            stats["reports"] = generated_count
            LOGGER.info("Generated %d paper reading reports", generated_count)
        except Exception as exc:  # noqa: BLE001 - reports should not block Feishu tracking.
            LOGGER.warning("Paper report generation failed; continuing without report links: %s", exc)

    if recommendations and _reports_publish_on_current_run(config, no_send=args.no_send):
        publish_reports(config, recommendations, recommended_on=recommended_on, repo_root=REPO_ROOT)

    bitable_client = build_bitable_client(config)
    for recommendation in recommendations:
        bitable_client.upsert_recommendation(recommendation)
    LOGGER.info("Wrote %d recommendations to Feishu bitable", len(recommendations))

    if args.no_send:
        LOGGER.info("Skipping Feishu bot send because --no-send is set")
        return 0

    title = f"每日论文推荐 | {recommended_on}"
    bot = FeishuBotClient(
        webhook_url=require_env("FEISHU_WEBHOOK_URL"),
        sign_secret=env_optional("FEISHU_SIGN_SECRET"),
    )
    bot.send_daily_recommendations(
        recommendations,
        title=title,
        bitable_url=env_optional("FEISHU_BITABLE_URL", config.get("feishu", {}).get("bitable_url", "")),
        stats=stats,
    )
    LOGGER.info("Sent Feishu bot message")
    for recommendation in recommendations:
        recommendation.pushed = True
        bitable_client.mark_pushed(recommendation)
    mark_recommendations_pushed_in_history(recommendations, config, REPO_ROOT, recommended_on=recommended_on)
    LOGGER.info("Marked recommendations as pushed")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Daily paper recommendation pipeline.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "tools" / "paper_recommender" / "config.yaml")
    parser.add_argument("--env", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--date", help="Recommendation date, e.g. 2026-05-29.")
    parser.add_argument("--limit", type=int, default=0, help="Override final recommendation count.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write Feishu or send messages.")
    parser.add_argument("--no-send", action="store_true", help="Write Feishu bitable but do not send group message.")
    parser.add_argument("--no-codex", action="store_true", help="Skip local Codex judging and summarization.")
    parser.add_argument("--no-reports", action="store_true", help="Skip per-paper reading report generation.")
    parser.add_argument("--no-huggingface", action="store_true", help="Skip Hugging Face Daily/Trending Papers hotness signals.")
    parser.add_argument("--no-cvf-openaccess", action="store_true", help="Skip CVF Open Access conference paper source.")
    parser.add_argument("--force-reports", action="store_true", help="Regenerate existing per-paper reading reports.")
    parser.add_argument("--allow-repeat", action="store_true", help="Allow papers that were already pushed in previous runs.")
    parser.add_argument("--no-semantic-scholar", action="store_true", help="Skip Semantic Scholar enrichment.")
    parser.add_argument("--arxiv-max-results", type=int, default=0, help="Override arXiv max_results.")
    parser.add_argument("--arxiv-source", choices=["api", "rss"], help="Choose arXiv API or RSS source.")
    parser.add_argument("--sample", action="store_true", help="Use built-in sample papers for offline validation.")
    parser.add_argument("--check-feishu", action="store_true", help="Validate Feishu auth and bitable read access only.")
    parser.add_argument("--ensure-feishu-fields", action="store_true", help="Create missing Feishu bitable fields from config.")
    parser.add_argument("--publish-existing-reports", action="store_true", help="Publish the current reports directory through the configured report target and exit.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging without printing secrets.")
    return parser.parse_args(argv)


def setup_logging(debug: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def today_for_config(config: dict[str, Any]) -> str:
    timezone_name = config.get("timezone", "Asia/Shanghai")
    return datetime.now(ZoneInfo(timezone_name)).strftime("%Y-%m-%d")


def load_recommendation_state(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    path = recommendation_state_path(config, repo_root)
    if not path.exists():
        return {"processed_arxiv_ids": [], "generated_reports": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def recommendation_state_path(config: dict[str, Any], repo_root: Path) -> Path:
    history_config = config.get("history", {})
    state_path = history_config.get("state_path") or config.get("reports", {}).get("state_path") or "reports/data/state.json"
    return _resolve_repo_path(str(state_path), repo_root)


def filter_previously_pushed_papers(
    papers: list[PaperCandidate],
    config: dict[str, Any],
    state: dict[str, Any],
) -> tuple[list[PaperCandidate], int]:
    if not _history_enabled(config):
        return papers, 0
    pushed_ids = historical_pushed_arxiv_ids(state)
    if not pushed_ids:
        return papers, 0
    filtered: list[PaperCandidate] = []
    skipped = 0
    for paper in papers:
        arxiv_id = _history_arxiv_id(paper.arxiv_id)
        if arxiv_id and arxiv_id in pushed_ids:
            skipped += 1
            continue
        filtered.append(paper)
    return filtered, skipped


def historical_pushed_arxiv_ids(state: dict[str, Any]) -> set[str]:
    history = state.get("recommendation_history")
    if isinstance(history, dict):
        return {
            _history_arxiv_id(arxiv_id)
            for arxiv_id, item in history.items()
            if _truthy((item or {}).get("pushed") if isinstance(item, dict) else False)
        } - {""}

    ids: set[str] = set()
    for arxiv_id in state.get("processed_arxiv_ids") or []:
        ids.add(_history_arxiv_id(str(arxiv_id)))
    generated_reports = state.get("generated_reports") or {}
    if isinstance(generated_reports, dict):
        for arxiv_id in generated_reports:
            ids.add(_history_arxiv_id(str(arxiv_id)))
    return ids - {""}


def mark_recommendations_pushed_in_history(
    recommendations: list[PaperRecommendation],
    config: dict[str, Any],
    repo_root: Path,
    recommended_on: str,
) -> None:
    if not recommendations or not _history_enabled(config):
        return

    path = recommendation_state_path(config, repo_root)
    state = load_recommendation_state(config, repo_root)
    history = _ensure_recommendation_history(state)
    updated_at = _now_iso()
    for recommendation in recommendations:
        arxiv_id = _history_arxiv_id(recommendation.paper.arxiv_id)
        if not arxiv_id:
            continue
        item = history.setdefault(arxiv_id, {})
        item.setdefault("first_recommended_on", item.get("last_recommended_on") or recommended_on)
        item["last_recommended_on"] = recommended_on
        item["last_pushed_on"] = recommended_on
        item["pushed"] = True
        item["title"] = recommendation.paper.title
        item["updated_at"] = updated_at
    state["schema_version"] = state.get("schema_version") or "1"
    state["last_history_update_at"] = updated_at
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _ensure_recommendation_history(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    history = state.get("recommendation_history")
    if isinstance(history, dict):
        return history

    migrated: dict[str, dict[str, Any]] = {}
    generated_reports = state.get("generated_reports") or {}
    if not isinstance(generated_reports, dict):
        generated_reports = {}
    fallback_ids = [str(item) for item in state.get("processed_arxiv_ids") or []]
    fallback_ids.extend(str(item) for item in generated_reports)
    for raw_arxiv_id in fallback_ids:
        arxiv_id = _history_arxiv_id(raw_arxiv_id)
        if not arxiv_id or arxiv_id in migrated:
            continue
        generated = generated_reports.get(raw_arxiv_id) or generated_reports.get(arxiv_id) or {}
        generated_at = str(generated.get("generated_at") or "") if isinstance(generated, dict) else ""
        recommended_on = generated_at[:10] if generated_at else ""
        migrated[arxiv_id] = {
            "first_recommended_on": recommended_on,
            "last_recommended_on": recommended_on,
            "last_pushed_on": recommended_on,
            "pushed": True,
            "migrated_from": "processed_arxiv_ids",
        }
    state["recommendation_history"] = migrated
    return migrated


def _history_enabled(config: dict[str, Any]) -> bool:
    history_config = config.get("history", {})
    if history_config.get("enabled", True) is False:
        return False
    mode = str(history_config.get("mode") or "pushed_forever").strip()
    return mode == "pushed_forever"


def _history_arxiv_id(value: str) -> str:
    return value.strip().removesuffix(".pdf").lower()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1", "是"}
    return bool(value)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _reports_publish_on_current_run(config: dict[str, Any], no_send: bool) -> bool:
    reports_config = config.get("reports", {})
    if not reports_config.get("enabled", True):
        return False
    if no_send:
        return bool(reports_config.get("publish_on_no_send", reports_config.get("git_push_on_no_send", True)))
    return bool(reports_config.get("publish_before_send", reports_config.get("git_push_before_send", True)))


def publish_reports(
    config: dict[str, Any],
    recommendations: list[PaperRecommendation],
    recommended_on: str,
    repo_root: Path,
) -> None:
    reports_config = config.get("reports", {})
    target = _report_publish_target(reports_config)
    if target in {"none", "off", "false"}:
        LOGGER.info("Skipping report publishing because reports.publish_target is %s", target)
        return
    if target in {"github", "github_pages", "pages"}:
        result = publish_reports_to_github_pages(
            config,
            repo_root=repo_root,
            recommended_on=recommended_on,
            current_arxiv_ids=[item.paper.arxiv_id for item in recommendations],
            report_urls=[item.report_url for item in recommendations],
        )
        LOGGER.info(
            "Published reports to GitHub Pages at %s; verified %d URLs",
            result.workflow_run_url or result.commit_sha,
            len(result.verified_urls),
        )
        return
    if target == "git":
        push_reports_to_git_remote(config, recommended_on=recommended_on, repo_root=repo_root)
        return
    raise RuntimeError(f"Unsupported reports.publish_target: {target}")


def publish_existing_reports(config: dict[str, Any], recommended_on: str, repo_root: Path) -> None:
    reports_config = config.get("reports", {})
    output_dir = str(reports_config.get("output_dir") or "reports").strip() or "reports"
    report_root = _resolve_repo_path(output_dir, repo_root)
    reports_json = report_root / "data" / "reports.json"
    if not reports_json.exists():
        raise RuntimeError(f"Cannot publish existing reports because {reports_json} does not exist")
    target = _report_publish_target(reports_config)
    if target in {"none", "off", "false"}:
        LOGGER.info("Skipping existing report publishing because reports.publish_target is %s", target)
        return
    if target == "git":
        push_reports_to_git_remote(config, recommended_on=recommended_on, repo_root=repo_root)
        return
    if target not in {"github", "github_pages", "pages"}:
        raise RuntimeError(f"Unsupported reports.publish_target: {target}")
    payload = json.loads(reports_json.read_text(encoding="utf-8"))
    records = payload.get("reports") or []
    base_url = (env_optional("PAPER_REPORT_BASE_URL") or str(reports_config.get("base_url") or "")).strip()
    arxiv_ids = [str(record.get("arxiv_id") or "").strip() for record in records if isinstance(record, dict)]
    report_urls = [_record_report_url(record, base_url) for record in records if isinstance(record, dict)]
    result = publish_reports_to_github_pages(
        config,
        repo_root=repo_root,
        recommended_on=recommended_on,
        current_arxiv_ids=arxiv_ids,
        report_urls=report_urls,
    )
    LOGGER.info(
        "Published existing reports to GitHub Pages at %s; verified %d URLs",
        result.workflow_run_url or result.commit_sha,
        len(result.verified_urls),
    )


def _report_publish_target(reports_config: dict[str, Any]) -> str:
    target = str(reports_config.get("publish_target") or "").strip().lower()
    if not target:
        return "git"
    if target in {"git", "gitlab", "gitlab_pages", "gitlab-pages"}:
        return "git"
    return target


def _record_report_url(record: dict[str, Any], base_url: str) -> str:
    url = str(record.get("report_url") or "").strip()
    if url.startswith(("http://", "https://")):
        return url
    if not base_url:
        return ""
    path = str(record.get("report_path") or "").strip()
    if path.startswith("reports/reports/"):
        path = path.removeprefix("reports/")
    return base_url.rstrip("/") + "/" + path.lstrip("/")


def push_reports_to_git_remote(config: dict[str, Any], recommended_on: str, repo_root: Path) -> None:
    reports_config = config.get("reports", {})
    if not reports_config.get("git_push_before_send", True):
        LOGGER.info("Skipping report git push because reports.git_push_before_send is disabled")
        return
    if not reports_config.get("enabled", True):
        LOGGER.info("Skipping report git push because reports are disabled")
        return

    output_dir = str(reports_config.get("output_dir") or "reports").strip() or "reports"
    report_root = _resolve_repo_path(output_dir, repo_root)
    if not report_root.exists():
        LOGGER.info("Skipping report git push because %s does not exist", output_dir)
        return

    _ensure_no_staged_changes_outside(output_dir, repo_root)
    _run_git(["add", "--", output_dir], repo_root)
    _ensure_no_staged_changes_outside(output_dir, repo_root)

    has_report_changes = _run_git(["diff", "--cached", "--quiet", "--", output_dir], repo_root, check=False).returncode != 0
    if has_report_changes:
        message = str(reports_config.get("git_commit_message") or f"chore: update paper reports {recommended_on}")
        _run_git(["commit", "-m", message], repo_root)
        LOGGER.info("Committed generated paper reports to Git")
    else:
        LOGGER.info("No generated paper report changes to commit")

    remote = str(reports_config.get("git_remote") or "origin")
    branch = str(reports_config.get("git_branch") or "").strip() or _current_git_branch(repo_root)
    _run_git(["push", remote, f"HEAD:{branch}"], repo_root)
    LOGGER.info("Pushed generated paper reports to git remote %s/%s", remote, branch)


def apply_report_publish_templates(config: dict[str, Any], recommended_on: str) -> None:
    reports_config = config.get("reports", {})
    values = {
        "date": recommended_on,
        "date_compact": recommended_on.replace("-", ""),
    }
    for key in ("base_url", "git_branch", "git_commit_message"):
        value = reports_config.get(key)
        if isinstance(value, str):
            reports_config[key] = value.format(**values)


def _resolve_repo_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _ensure_no_staged_changes_outside(allowed_path: str, repo_root: Path) -> None:
    completed = _run_git(["diff", "--cached", "--name-only", "--"], repo_root)
    staged_paths = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    prefix = allowed_path.rstrip("/") + "/"
    outside = [path for path in staged_paths if path != allowed_path.rstrip("/") and not path.startswith(prefix)]
    if outside:
        raise RuntimeError(
            "Refusing to auto-commit reports because unrelated staged files exist: "
            + ", ".join(outside[:10])
        )


def _current_git_branch(repo_root: Path) -> str:
    completed = _run_git(["branch", "--show-current"], repo_root)
    branch = completed.stdout.strip()
    if not branch:
        raise RuntimeError("Cannot push reports before Feishu send because current Git branch is detached")
    return branch


def _run_git(args: list[str], repo_root: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return completed


def build_bitable_client(config: dict[str, Any]) -> FeishuBitableClient:
    feishu_config = config.get("feishu", {})
    token_client = TenantTokenClient(
        app_id=require_env("FEISHU_APP_ID"),
        app_secret=require_env("FEISHU_APP_SECRET"),
    )
    app_token = env_optional("FEISHU_BITABLE_APP_TOKEN")
    wiki_token = env_optional("FEISHU_BITABLE_WIKI_TOKEN")
    if not app_token and wiki_token:
        app_token = resolve_wiki_bitable_app_token(token_client, wiki_token)
        LOGGER.info("Resolved Feishu wiki token to bitable app token")
    if not app_token:
        raise RuntimeError("Missing FEISHU_BITABLE_APP_TOKEN or FEISHU_BITABLE_WIKI_TOKEN")
    return FeishuBitableClient(
        token_client=token_client,
        app_token=app_token,
        table_id=require_env("FEISHU_BITABLE_TABLE_ID"),
        field_names=feishu_config.get("field_names", {}),
    )


def feishu_field_types() -> dict[str, int]:
    return {
        "score": 2,
    }


def to_json_payload(recommendations: list[Any], stats: dict[str, int]) -> dict[str, Any]:
    return {
        "stats": stats,
        "recommendations": [
            {
                "arxiv_id": item.paper.arxiv_id,
                "title": item.paper.title,
                "display_title": item.summary.display_title,
                "url": item.paper.arxiv_url,
                "project_url": item.paper.project_url,
                "code_url": item.paper.code_url,
                "published": item.paper.published,
                "authors": item.paper.authors,
                "venue": item.paper.venue,
                "institutions": item.paper.institution_signals,
                "notable_authors": item.paper.notable_author_signals,
                "categories": item.paper.categories,
                "sources": item.paper.source_names,
                "hotness_score": item.paper.hotness_score,
                "quality_score": item.paper.quality_score,
                "hotness_signals": item.paper.hotness_signals,
                "quality_signals": item.paper.quality_signals,
                "hf_upvotes": item.paper.hf_upvotes,
                "github_stars": item.paper.github_stars,
                "citation_count": item.paper.citation_count,
                "influential_citation_count": item.paper.influential_citation_count,
                "score": item.paper.score,
                "score_reasons": item.paper.score_reasons,
                "report_url": item.report_url,
                "report_path": item.report_path,
                "report_status": item.report_status,
                "report_basis": item.report_basis,
                "report_generated_at": item.report_generated_at,
                "summary": {
                    "display_title": item.summary.display_title,
                    "hook": item.summary.hook,
                    "one_sentence": item.summary.one_sentence,
                    "contribution": item.summary.contribution,
                    "technical_points": item.summary.technical_points,
                    "results": item.summary.results,
                    "why_now": item.summary.why_now,
                    "quality_reason": item.summary.quality_reason,
                    "hotness_reason": item.summary.hotness_reason,
                    "evidence_highlights": item.summary.evidence_highlights,
                    "business_value": item.summary.business_value,
                    "risks": item.summary.risks,
                    "action": item.summary.action,
                    "recommendation_level": item.summary.recommendation_level,
                    "recommendation_reason": item.summary.recommendation_reason,
                    "primary_task": item.summary.primary_task,
                    "business_relevance": item.summary.business_relevance,
                    "deployability": item.summary.deployability,
                    "matched_keywords": item.summary.matched_keywords,
                    "codex_decision": item.summary.codex_decision,
                },
            }
            for item in recommendations
        ],
    }


def sample_papers() -> list[PaperCandidate]:
    return [
        PaperCandidate(
            title="测试样例：端侧人脸检测量化蒸馏推荐卡片",
            abstract=(
                "This is a local sample paper used only to validate Feishu card rendering, "
                "bitable writes, and Codex summarization. It is not a real arXiv paper. "
                "The sample topic mimics lightweight, efficient, real-time face detection "
                "with edge deployment, quantization-aware training, and distillation."
            ),
            authors=["Sample Author"],
            published="2026-05-28T00:00:00+00:00",
            updated="2026-05-28T00:00:00+00:00",
            arxiv_id="sample-card-001",
            arxiv_url="",
            pdf_url="",
            categories=["cs.CV"],
            primary_category="cs.CV",
        ),
        PaperCandidate(
            title="A Survey of Diffusion Models for Artistic Image Generation",
            abstract="This survey reviews recent text-to-image diffusion models and aesthetic generation.",
            authors=["Sample Author"],
            published="2026-05-27T00:00:00+00:00",
            updated="2026-05-27T00:00:00+00:00",
            arxiv_id="2605.00002",
            arxiv_url="https://arxiv.org/abs/2605.00002",
            pdf_url="https://arxiv.org/pdf/2605.00002",
            categories=["cs.CV"],
            primary_category="cs.CV",
        ),
    ]


if __name__ == "__main__":
    raise SystemExit(main())
