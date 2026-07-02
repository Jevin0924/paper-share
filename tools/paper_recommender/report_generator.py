from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from io import BytesIO
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from .codex_summarizer import _extract_json, _run_codex
from .feishu_doc import build_feishu_doc_client
from .feishu_markdown_report import publish_feishu_markdown_report
from .feishu_native_report import publish_feishu_native_report
from .models import PaperRecommendation
from .paper_analyzer_report import generate_paper_analyzer_markdown
from .paper_craft_report import generate_paper_craft_html
from .report_quality import scrub_report_data, validate_report_quality
from .report_schema import ensure_heilmeier_shape, render_heilmeier_markdown


REPORT_SCHEMA_VERSION = "1"
LOGGER = logging.getLogger("paper_recommender")


def generate_reports(
    recommendations: list[PaperRecommendation],
    config: dict[str, Any],
    repo_root: Path,
) -> list[PaperRecommendation]:
    reports_config = config.get("reports", {})
    if not reports_config.get("enabled", True) or not recommendations:
        return recommendations

    output_dir = _resolve_path(str(reports_config.get("output_dir") or "web"), repo_root)
    data_dir = output_dir / "data"
    publisher = _report_publisher(reports_config)
    report_format = _archive_report_format(reports_config, publisher)
    data_dir.mkdir(parents=True, exist_ok=True)

    state_path = _resolve_path(str(reports_config.get("state_path") or "web/data/state.json"), repo_root)
    state = _load_json(state_path, fallback={"processed_arxiv_ids": [], "generated_reports": {}})
    base_url = _report_base_url(reports_config)
    force = bool(reports_config.get("force_regenerate", False))
    doc_client = build_feishu_doc_client(config) if publisher == "feishu_doc" else None

    report_targets: dict[str, dict[str, str]] = {}
    to_generate: list[PaperRecommendation] = []
    for recommendation in recommendations:
        slug = _paper_slug(recommendation.paper.arxiv_id or recommendation.paper.normalized_title())
        rel_path = _report_relative_path(report_format, slug)
        report_path = output_dir / rel_path
        report_url = _absolute_report_url(base_url, rel_path)
        recommendation.report_path = _repo_relative(report_path, repo_root)
        recommendation.report_url = report_url
        recommendation.report_status = "已生成" if report_path.exists() else "待生成"
        recommendation.report_basis = recommendation.report_basis or "摘要与评审信息"
        report_targets[recommendation.paper.arxiv_id] = {
            "slug": slug,
            "rel_path": rel_path,
            "path": str(report_path),
            "url": report_url,
            "format": report_format,
            "publisher": publisher,
        }
        stored = state.get("generated_reports", {}).get(recommendation.paper.arxiv_id, {})
        missing_doc_report = publisher == "feishu_doc" and not stored.get("doc_url")
        if force or not report_path.exists() or missing_doc_report:
            to_generate.append(recommendation)

    generated_at = _now_iso()
    report_payloads = (
        _build_report_payloads(to_generate, config, repo_root, report_targets, output_dir)
        if to_generate
        else {}
    )
    for recommendation in to_generate:
        paper_id = recommendation.paper.arxiv_id
        target = report_targets[paper_id]
        data = report_payloads.get(paper_id) or fallback_report_data(recommendation)
        data = _normalize_report_data(data, recommendation)
        data = ensure_heilmeier_shape(scrub_report_data(data))
        basis = str(data.get("report_basis") or recommendation.report_basis or "摘要与评审信息")
        quality = validate_report_quality(
            data,
            recommendation,
            require_pdf_evidence=bool(reports_config.get("quality_require_pdf_evidence", True)),
        )
        extra_quality_issues = _string_list(data.get("_quality_issues"))
        quality_issues = [*quality.issues, *extra_quality_issues]
        data["quality_status"] = "failed" if quality_issues else quality.status
        data["quality_issues"] = quality_issues
        data["quality_warnings"] = quality.warnings
        rendered_html = str(data.get("_rendered_html") or "").strip()
        rendered_markdown = str(data.get("_rendered_markdown") or "").strip()
        if rendered_html and str(target["format"]) == "html":
            content = rendered_html + "\n"
        elif rendered_markdown and str(target["format"]) == "markdown":
            content = rendered_markdown + "\n"
        else:
            content = render_report(
                recommendation,
                data,
                generated_at=generated_at,
                report_basis=basis,
                report_format=str(target["format"]),
            )
        report_path = Path(target["path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(content, encoding="utf-8")
        report_url = str(target["url"])
        stored_report = state.get("generated_reports", {}).get(paper_id, {})
        doc_id = str(stored_report.get("doc_id") or stored_report.get("doc_token") or "").strip()
        doc_url = str(stored_report.get("doc_url") or "").strip()
        wiki_node_token = str(stored_report.get("wiki_node_token") or "").strip()
        wiki_space_id = str(stored_report.get("wiki_space_id") or "").strip()
        if publisher != "feishu_doc":
            doc_id = ""
            doc_url = ""
            wiki_node_token = ""
            wiki_space_id = ""
        can_publish_doc = (
            publisher == "feishu_doc"
            and doc_client is not None
            and (quality.passed or not reports_config.get("quality_gate", True))
        )
        if can_publish_doc:
            try:
                title = f"{recommendation.paper.title} | 论文阅读报告"
                folder_token = str(reports_config.get("feishu_doc_folder_token") or os.getenv("FEISHU_REPORT_FOLDER_TOKEN") or "")
                wiki_parent_token = str(reports_config.get("feishu_wiki_parent_token") or os.getenv("FEISHU_REPORT_WIKI_TOKEN") or "")
                max_images = int(reports_config.get("feishu_doc_max_images", 4))
                if rendered_markdown and reports_config.get("feishu_native_blocks", True):
                    document = publish_feishu_markdown_report(
                        client=doc_client,
                        title=title,
                        markdown=content,
                        markdown_base_dir=report_path.parent,
                        folder_token=folder_token,
                        document_id=doc_id,
                        document_url=doc_url,
                        wiki_parent_token=wiki_parent_token,
                        wiki_node_token=wiki_node_token,
                        wiki_space_id=wiki_space_id,
                    )
                elif reports_config.get("feishu_native_blocks", True):
                    document = publish_feishu_native_report(
                        client=doc_client,
                        recommendation=recommendation,
                        report=data,
                        generated_at=generated_at,
                        report_basis=basis,
                        output_dir=output_dir,
                        title=title,
                        folder_token=folder_token,
                        document_id=doc_id,
                        document_url=doc_url,
                        wiki_parent_token=wiki_parent_token,
                        wiki_node_token=wiki_node_token,
                        wiki_space_id=wiki_space_id,
                        max_images=max_images,
                    )
                else:
                    document = doc_client.publish_markdown(
                        title=title,
                        content=render_report_feishu_markdown(
                            recommendation,
                            data,
                            generated_at=generated_at,
                            report_basis=basis,
                        ),
                        folder_token=folder_token,
                        image_paths=_feishu_image_paths(data, output_dir, max_images=max_images),
                        document_id=doc_id,
                        document_url=doc_url,
                        wiki_parent_token=wiki_parent_token,
                        wiki_node_token=wiki_node_token,
                        wiki_space_id=wiki_space_id,
                    )
                doc_id = document.document_id
                doc_url = document.url
                wiki_node_token = document.wiki_node_token or wiki_node_token
                wiki_space_id = document.wiki_space_id or wiki_space_id
                report_url = doc_url
            except Exception as exc:  # noqa: BLE001 - archive Markdown should remain available.
                LOGGER.warning("Feishu doc publish failed for %s; keeping Markdown archive: %s", paper_id, exc)
                data["quality_warnings"] = [*quality.warnings, "飞书文档发布失败，已保留 Markdown 归档。"]
        recommendation.report_url = report_url
        recommendation.report_status = "已生成"
        if data["quality_issues"]:
            recommendation.report_status = "需复核"
        recommendation.report_basis = basis
        recommendation.report_generated_at = generated_at
        state.setdefault("processed_arxiv_ids", [])
        if paper_id not in state["processed_arxiv_ids"]:
            state["processed_arxiv_ids"].append(paper_id)
        state.setdefault("generated_reports", {})[paper_id] = {
            "path": target["rel_path"],
            "url": report_url,
            "archive_url": target["url"],
            "format": target["format"],
            "publisher": publisher,
            "doc_id": doc_id,
            "doc_url": doc_url,
            "wiki_node_token": wiki_node_token,
            "wiki_space_id": wiki_space_id,
            "quality_status": data["quality_status"],
            "quality_issues": data["quality_issues"],
            "generated_at": generated_at,
            "basis": basis,
        }

    for recommendation in recommendations:
        stored = state.get("generated_reports", {}).get(recommendation.paper.arxiv_id, {})
        if stored:
            recommendation.report_url = str(stored.get("url") or recommendation.report_url)
            recommendation.report_generated_at = str(stored.get("generated_at") or recommendation.report_generated_at)
            recommendation.report_basis = str(stored.get("basis") or recommendation.report_basis)
            recommendation.report_status = "需复核" if stored.get("quality_status") == "failed" else "已生成"

    write_reports_index(recommendations, output_dir=output_dir, generated_at=generated_at, report_format=report_format)
    state["last_success_at"] = generated_at
    state["schema_version"] = REPORT_SCHEMA_VERSION
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return recommendations


def render_report(
    recommendation: PaperRecommendation,
    report: dict[str, Any],
    generated_at: str,
    report_basis: str,
    report_format: str,
) -> str:
    if report_format == "markdown":
        return render_report_markdown(recommendation, report, generated_at=generated_at, report_basis=report_basis)
    return render_report_html(recommendation, report, generated_at=generated_at, report_basis=report_basis)


def render_report_feishu_markdown(
    recommendation: PaperRecommendation,
    report: dict[str, Any],
    generated_at: str,
    report_basis: str,
) -> str:
    """Render Markdown that Feishu's docx converter can ingest cleanly."""
    paper = recommendation.paper
    summary = recommendation.summary
    meta_items = [
        ("arXiv ID", paper.arxiv_id),
        ("发布时间", paper.published[:10]),
        ("方向", summary.primary_task or "感知视觉"),
        ("推荐等级", summary.recommendation_level),
        ("推荐分", f"{paper.score:.1f} / 30"),
        ("报告依据", report_basis),
        ("生成时间", generated_at),
    ]
    lines = [
        f"# {_md_text(paper.title)}",
        "",
        "> 论文精读报告。报告采用 Heilmeier 七问结构，区分论文事实与业务判断。",
        "",
        "## 基本信息",
        "",
        _markdown_table(meta_items),
        "",
        "## 原始链接",
        "",
        _markdown_links([("查看论文", paper.arxiv_url), ("下载 PDF", paper.pdf_url), ("查看代码", paper.code_url)]),
        "",
        "## 一页结论",
        "",
        _md_text(str(report.get("executive_summary") or "")),
        "",
        f"**适合读者：** {_md_text(str(report.get('target_reader') or ''))}",
        "",
        f"**建议动作：** {_md_text(str(report.get('business_action') or '需结合业务场景、复现成本和内部数据进一步评估。'))}",
        "",
    ]
    mermaid = str(report.get("method_mermaid") or report.get("mermaid") or "").strip()
    if mermaid:
        lines.extend(["## 方法流程图", "", "```mermaid", mermaid, "```", ""])
    lines.extend([render_heilmeier_markdown(report), ""])
    experiment_tables = _experiment_tables_markdown(report.get("experiment_tables"))
    if experiment_tables:
        lines.extend(["## 结构化实验表", "", experiment_tables, ""])
    figure_manifest = _figure_manifest_markdown(report)
    if figure_manifest:
        lines.extend(["## 图表清单", "", figure_manifest, ""])
    lines.extend(
        [
            "## 待确认问题",
            "",
            _markdown_list(report.get("open_questions")),
            "",
            "## 证据索引",
            "",
            _markdown_list(report.get("evidence")),
            "",
        ]
    )
    return "\n".join(line for line in lines if line is not None)


def _build_report_payloads(
    recommendations: list[PaperRecommendation],
    config: dict[str, Any],
    repo_root: Path,
    report_targets: dict[str, dict[str, str]],
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    if not recommendations:
        return {}
    codex_config = config.get("codex", {})
    reports_config = config.get("reports", {})
    paper_texts = {
        item.paper.arxiv_id: collect_paper_text(
            item,
            reports_config,
            assets_dir=output_dir / "assets" / report_targets[item.paper.arxiv_id]["slug"],
            assets_rel_dir=f"assets/{report_targets[item.paper.arxiv_id]['slug']}",
        )
        for item in recommendations
    }
    if not codex_config.get("enabled", True) or not reports_config.get("codex_enabled", True):
        return {item.paper.arxiv_id: fallback_report_data(item, paper_texts.get(item.paper.arxiv_id)) for item in recommendations}
    codex_bin = codex_config.get("command", "codex")
    if not shutil.which(codex_bin):
        return {item.paper.arxiv_id: fallback_report_data(item, paper_texts.get(item.paper.arxiv_id)) for item in recommendations}
    timeout = int(reports_config.get("timeout_seconds", codex_config.get("timeout_seconds", 300)))
    generator = _report_content_generator(reports_config)
    if generator == "paper_craft":
        return _build_paper_craft_payloads(
            recommendations=recommendations,
            paper_texts=paper_texts,
            reports_config=reports_config,
            codex_bin=str(codex_bin),
            timeout=timeout,
            repo_root=repo_root,
        )
    if generator == "paper_analyzer":
        return _build_paper_analyzer_payloads(
            recommendations=recommendations,
            paper_texts=paper_texts,
            reports_config=reports_config,
            codex_bin=str(codex_bin),
            timeout=timeout,
            repo_root=repo_root,
        )
    report_payloads: dict[str, dict[str, Any]] = {}
    for item in recommendations:
        material = paper_texts.get(item.paper.arxiv_id) or {}
        try:
            prompt = build_report_prompt([item], {item.paper.arxiv_id: material})
            payload = _run_codex(codex_bin=codex_bin, prompt=prompt, timeout=timeout, repo_root=repo_root)
            reports = parse_report_payload(payload)
            by_id = {str(report.get("arxiv_id") or "").strip(): report for report in reports}
            report_payloads[item.paper.arxiv_id] = _merge_report_material(
                by_id.get(item.paper.arxiv_id) or fallback_report_data(item, material),
                material,
            )
        except Exception:  # noqa: BLE001 - fallback reports are better than blocking the daily flow.
            report_payloads[item.paper.arxiv_id] = fallback_report_data(item, material)
    return report_payloads


def _build_paper_craft_payloads(
    recommendations: list[PaperRecommendation],
    paper_texts: dict[str, dict[str, Any]],
    reports_config: dict[str, Any],
    codex_bin: str,
    timeout: int,
    repo_root: Path,
) -> dict[str, dict[str, Any]]:
    report_payloads: dict[str, dict[str, Any]] = {}
    style = str(reports_config.get("paper_craft_style") or reports_config.get("paper_analyzer_style") or "academic").strip() or "academic"
    for item in recommendations:
        material = paper_texts.get(item.paper.arxiv_id) or {}
        fallback = _merge_report_material(fallback_report_data(item, material), material)
        try:
            html = generate_paper_craft_html(
                recommendation=item,
                paper_material=material,
                reports_config=reports_config,
                codex_bin=codex_bin,
                timeout=timeout,
                repo_root=repo_root,
            )
            report_payloads[item.paper.arxiv_id] = {
                **fallback,
                "_rendered_html": html,
                "report_generator": "paper_craft",
                "paper_craft_style": style,
                "report_basis": material.get("basis") or fallback.get("report_basis") or "摘要与评审信息",
                "evidence": fallback.get("evidence") or _fallback_evidence(material),
            }
        except Exception as exc:  # noqa: BLE001 - fallback HTML should still be published for the daily flow.
            LOGGER.warning("paper-craft HTML generation failed for %s; using fallback HTML: %s", item.paper.arxiv_id, exc)
            report_payloads[item.paper.arxiv_id] = {
                **fallback,
                "_quality_issues": [f"paper-craft HTML generation failed: {type(exc).__name__}"],
                "report_generator": "paper_craft",
                "paper_craft_style": style,
            }
    return report_payloads


def _build_paper_analyzer_payloads(
    recommendations: list[PaperRecommendation],
    paper_texts: dict[str, dict[str, Any]],
    reports_config: dict[str, Any],
    codex_bin: str,
    timeout: int,
    repo_root: Path,
) -> dict[str, dict[str, Any]]:
    report_payloads: dict[str, dict[str, Any]] = {}
    style = str(reports_config.get("paper_analyzer_style") or "academic").strip() or "academic"
    for item in recommendations:
        material = paper_texts.get(item.paper.arxiv_id) or {}
        fallback = _merge_report_material(fallback_report_data(item, material), material)
        try:
            markdown = generate_paper_analyzer_markdown(
                recommendation=item,
                paper_material=material,
                reports_config=reports_config,
                codex_bin=codex_bin,
                timeout=timeout,
                repo_root=repo_root,
            )
            if not markdown.lstrip().startswith("#"):
                markdown = f"# {item.paper.title}\n\n{markdown.strip()}\n"
            report_payloads[item.paper.arxiv_id] = {
                **fallback,
                "_rendered_markdown": markdown,
                "report_generator": "paper_analyzer",
                "paper_analyzer_style": style,
                "report_basis": material.get("basis") or fallback.get("report_basis") or "摘要与评审信息",
                "evidence": _paper_analyzer_evidence(markdown, fallback.get("evidence")),
            }
        except Exception as exc:  # noqa: BLE001 - fallback reports are better than blocking the daily flow.
            LOGGER.warning("paper-analyzer report generation failed for %s; using fallback report: %s", item.paper.arxiv_id, exc)
            report_payloads[item.paper.arxiv_id] = fallback
    return report_payloads


def collect_paper_text(
    recommendation: PaperRecommendation,
    reports_config: dict[str, Any],
    assets_dir: Path | None = None,
    assets_rel_dir: str = "",
) -> dict[str, Any]:
    paper = recommendation.paper
    max_chars = int(reports_config.get("pdf_text_chars", 80000))
    if not reports_config.get("read_pdf", True):
        return {"basis": "摘要与评审信息", "text": paper.abstract[:max_chars], "status": "disabled"}
    if not paper.pdf_url.startswith(("http://", "https://")):
        return {"basis": "摘要与评审信息", "text": paper.abstract[:max_chars], "status": "no_pdf_url"}
    timeout = int(reports_config.get("pdf_timeout_seconds", 45))
    try:
        response = requests.get(paper.pdf_url, timeout=timeout, headers={"User-Agent": "paper-share-recommender/1.0"})
        response.raise_for_status()
        pdf_bytes = response.content
        max_pages = int(reports_config.get("pdf_max_pages", 0))
        pages = _extract_pages_with_pypdf(pdf_bytes, max_pages=max_pages)
        text = _pages_to_prompt_text(pages)
        status = "fulltext:pypdf" if text else "pdf_text_extractor_empty"
        if not text:
            text = _extract_text_with_pdftotext(pdf_bytes, timeout=timeout, reports_config=reports_config)
            status = "fulltext:pdftotext" if text else "pdf_text_extractor_empty"
        figures = (
            extract_pdf_figures(
                pdf_bytes,
                assets_dir=assets_dir,
                assets_rel_dir=assets_rel_dir,
                pages=pages,
                reports_config=reports_config,
            )
            if reports_config.get("extract_figures", True) and assets_dir is not None
            else []
        )
        if text:
            truncated = len(text) > max_chars
            return {
                "basis": "PDF全文与摘要" if not truncated else "PDF全文截断与摘要",
                "text": text[:max_chars],
                "status": f"{status}:truncated" if truncated else status,
                "pages_read": len(pages) if pages else 0,
                "text_chars": min(len(text), max_chars),
                "figures": figures,
            }
    except Exception as exc:  # noqa: BLE001 - report generation should degrade gracefully.
        return {"basis": "摘要与评审信息", "text": paper.abstract[:max_chars], "status": f"pdf_failed:{type(exc).__name__}"}
    return {"basis": "摘要与评审信息", "text": paper.abstract[:max_chars], "status": "extract_empty"}


def _merge_report_material(report: dict[str, Any], paper_material: dict[str, Any]) -> dict[str, Any]:
    merged = dict(report)
    if paper_material.get("basis") and not merged.get("report_basis"):
        merged["report_basis"] = paper_material["basis"]
    if paper_material.get("figures") and not merged.get("figures"):
        merged["figures"] = paper_material["figures"]
    merged.setdefault(
        "material_status",
        {
            "text_status": paper_material.get("status", ""),
            "pages_read": paper_material.get("pages_read", 0),
            "text_chars": paper_material.get("text_chars", 0),
            "figure_count": len(paper_material.get("figures") or []),
        },
    )
    return merged


def _extract_pages_with_pypdf(pdf_bytes: bytes, max_pages: int = 0) -> list[dict[str, Any]]:
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError:
        return []
    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        pages = []
        limit = len(reader.pages) if max_pages <= 0 else min(len(reader.pages), max_pages)
        for index, page in enumerate(reader.pages[:limit], start=1):
            text = _normalize_page_text(page.extract_text() or "")
            if text:
                pages.append({"page": index, "text": text, "captions": _find_caption_candidates(text)})
        return pages
    except Exception:  # noqa: BLE001 - pypdf extraction is best effort.
        return []


def _pages_to_prompt_text(pages: list[dict[str, Any]]) -> str:
    chunks = []
    for page in pages:
        chunks.append(f"\n\n[PAGE {page.get('page')}]\n{page.get('text', '')}")
    return "\n".join(chunks).strip()


def _extract_text_with_pdftotext(pdf_bytes: bytes, timeout: int, reports_config: dict[str, Any]) -> str:
    pdftotext_bin = shutil.which(str(reports_config.get("pdftotext_command") or "pdftotext"))
    if not pdftotext_bin:
        return ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as pdf_file:
            pdf_file.write(pdf_bytes)
            pdf_file.flush()
            completed = subprocess.run(
                [pdftotext_bin, "-layout", pdf_file.name, "-"],
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        text = _normalize_page_text(completed.stdout)
        if completed.returncode == 0 and text:
            return text
    except Exception:  # noqa: BLE001 - fallback only.
        return ""
    return ""


def extract_pdf_figures(
    pdf_bytes: bytes,
    assets_dir: Path | None,
    assets_rel_dir: str,
    pages: list[dict[str, Any]],
    reports_config: dict[str, Any],
) -> list[dict[str, Any]]:
    if assets_dir is None:
        return []
    max_figures = int(reports_config.get("max_figures", 4))
    if max_figures <= 0:
        return []
    figures = _extract_figures_with_pymupdf(
        pdf_bytes,
        assets_dir=assets_dir,
        assets_rel_dir=assets_rel_dir,
        pages=pages,
        reports_config=reports_config,
        max_figures=max_figures,
    )
    if figures:
        return figures
    return _extract_figures_with_pdfimages(
        pdf_bytes,
        assets_dir=assets_dir,
        assets_rel_dir=assets_rel_dir,
        max_figures=max_figures,
    )


def _extract_figures_with_pymupdf(
    pdf_bytes: bytes,
    assets_dir: Path,
    assets_rel_dir: str,
    pages: list[dict[str, Any]],
    reports_config: dict[str, Any],
    max_figures: int,
) -> list[dict[str, Any]]:
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError:
        return []
    min_width = int(reports_config.get("figure_min_width", 220))
    min_height = int(reports_config.get("figure_min_height", 160))
    figures: list[dict[str, Any]] = []
    seen_xrefs: set[int] = set()
    try:
        assets_dir.mkdir(parents=True, exist_ok=True)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        for page_index, page in enumerate(doc, start=1):
            page_captions = _captions_for_page(pages, page_index)
            for image in page.get_images(full=True):
                xref = int(image[0])
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)
                pix = fitz.Pixmap(doc, xref)
                try:
                    if pix.width < min_width or pix.height < min_height:
                        continue
                    if pix.n - pix.alpha > 3:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    filename = f"figure-{len(figures) + 1:02d}-p{page_index}.png"
                    output_path = assets_dir / filename
                    pix.save(str(output_path))
                    asset_path = f"{assets_rel_dir}/{filename}".strip("/")
                    figures.append(
                        {
                            "path": asset_path,
                            "markdown_path": _markdown_asset_path(asset_path),
                            "page": page_index,
                            "caption_candidates": page_captions,
                        }
                    )
                    if len(figures) >= max_figures:
                        return figures
                finally:
                    pix = None
    except Exception:  # noqa: BLE001 - images are optional.
        return figures
    return figures


def _extract_figures_with_pdfimages(
    pdf_bytes: bytes,
    assets_dir: Path,
    assets_rel_dir: str,
    max_figures: int,
) -> list[dict[str, Any]]:
    pdfimages_bin = shutil.which("pdfimages")
    if not pdfimages_bin:
        return []
    min_bytes = 20_000
    figures: list[dict[str, Any]] = []
    try:
        assets_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            pdf_path = tmp_dir / "paper.pdf"
            pdf_path.write_bytes(pdf_bytes)
            prefix = tmp_dir / "figure"
            completed = subprocess.run(
                [pdfimages_bin, "-png", str(pdf_path), str(prefix)],
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )
            if completed.returncode != 0:
                return []
            for image_path in sorted(tmp_dir.glob("figure-*.png")):
                if image_path.stat().st_size < min_bytes:
                    continue
                filename = f"figure-{len(figures) + 1:02d}.png"
                target = assets_dir / filename
                shutil.copyfile(image_path, target)
                asset_path = f"{assets_rel_dir}/{filename}".strip("/")
                figures.append(
                    {
                        "path": asset_path,
                        "markdown_path": _markdown_asset_path(asset_path),
                        "page": 0,
                        "caption_candidates": [],
                    }
                )
                if len(figures) >= max_figures:
                    return figures
    except Exception:  # noqa: BLE001 - images are optional.
        return figures
    return figures


def build_report_prompt(
    recommendations: list[PaperRecommendation],
    paper_texts: dict[str, dict[str, Any]],
) -> str:
    papers = []
    for item in recommendations:
        text_info = paper_texts.get(item.paper.arxiv_id) or {}
        papers.append(
            {
                "arxiv_id": item.paper.arxiv_id,
                "title": item.paper.title,
                "authors": item.paper.authors[:12],
                "published": item.paper.published,
                "categories": item.paper.categories,
                "abstract": item.paper.abstract,
                "paper_full_text_with_page_markers": text_info.get("text", ""),
                "text_basis": text_info.get("basis", "摘要与评审信息"),
                "text_status": text_info.get("status", ""),
                "pages_read": text_info.get("pages_read", 0),
                "text_chars": text_info.get("text_chars", 0),
                "figures": text_info.get("figures", []),
                "score": item.paper.score,
                "primary_task": item.summary.primary_task,
                "recommendation_level": item.summary.recommendation_level,
                "recommendation_reason": item.summary.recommendation_reason,
                "business_value": item.summary.business_value,
                "risks": item.summary.risks,
                "technical_points": item.summary.technical_points,
                "results": item.summary.results,
                "matched_keywords": item.summary.matched_keywords,
                "paper_url": item.paper.arxiv_url,
                "pdf_url": item.paper.pdf_url,
                "code_url": item.paper.code_url,
            }
        )
    return (
        "你是计算机视觉感知团队的论文精读报告作者。请为每篇已推荐论文生成单篇中文精读报告，"
        "目标读者是算法工程师和研发负责人。必须优先依据 paper_full_text_with_page_markers 中的 PDF 全文内容，"
        "引用页码证据，例如“见 PAGE 3”。报告必须克制、可追溯，不能编造论文中没有的实验、数据集、指标或结论。"
        "如果 text_basis 不是 PDF全文与摘要 或 PDF全文截断与摘要，或某个细节在全文中找不到，请明确写“证据不足”。"
        "figures 字段是从 PDF 抽取出的图片候选和同页 caption 候选；你不能直接看图，只能基于 caption 和正文判断是否值得放入报告。\n"
        "报告结构参考 Heilmeier 七问和 paper-craft 的图解原则：先讲事实，再讲判断；图表只服务方法流程、核心机制或关键结果，"
        "不要写空泛宣传，不要输出 primary:/method:/engineering: 等内部打分信号。\n"
        "输出要求：只输出 JSON，不输出 Markdown，不输出解释文字。JSON 顶层对象格式为：\n"
        '{"reports":[{"arxiv_id":"...",'
        '"executive_summary":"一页结论，说明是否值得读、适合谁读、建议动作，并给出页码依据",'
        '"target_reader":"适合阅读的人群",'
        '"business_action":"小试/观察/归档/关闭及原因",'
        '"heilmeier":{'
        '"what":{"paper_claims":["论文事实，只写论文声称要解决什么，含 PAGE 依据"],"analysis":[],"evidence":["PAGE 1: ..."]},'
        '"current_limits":{"paper_claims":["现有方法或任务限制，含 PAGE 依据"],"analysis":[],"evidence":["PAGE 2: ..."]},'
        '"method":{"paper_claims":["核心方法总览，含 PAGE 依据"],"analysis":[],"evidence":["PAGE 3: ..."]},'
        '"method_details":{"paper_claims":["关键模块、公式、训练策略或推理流程，含 PAGE 依据"],"analysis":[],"evidence":["PAGE 4: ..."]},'
        '"who_cares":{"paper_claims":[],"analysis":["对检测/跟踪/ReID/关键点/属性/部署/自动标注等业务的价值"],"evidence":["PAGE 5: ..."]},'
        '"experiments":{"paper_claims":["数据集、指标、关键结果；未知则写证据不足"],"analysis":["实验可信度判断"],"evidence":["PAGE 6: ..."]},'
        '"risks":{"paper_claims":["论文承认的限制，含 PAGE 依据"],"analysis":["复现成本、部署风险、业务边界"],"evidence":["PAGE 7: ..."]}'
        '},'
        '"experiment_tables":[{"title":"主结果表","columns":["模型","数据集","指标","结果","证据"],"rows":[["...","...","...","...","PAGE 6"]]}],'
        '"method_mermaid":"可选 Mermaid flowchart，必须只画方法流程；证据不足则空字符串",'
        '"problem":"论文解决的问题",'
        '"why_problem_matters":"为什么这个问题重要",'
        '"method_overview":"核心方法整体框架",'
        '"method_details":["关键模块或流程1，包含页码依据","关键模块或流程2，包含页码依据"],'
        '"method_deep_dive":["按论文结构拆解的技术细节，包含页码依据"],'
        '"innovation_points":["创新点1","创新点2"],'
        '"experiments":"数据集、指标、关键结果；未知则写证据不足",'
        '"business_value":"对检测/跟踪/ReID/关键点/属性/部署/自动标注等业务的价值",'
        '"implementation_plan":["1天内验证项","1周内小实验","是否进入技术储备"],'
        '"risks":["风险1","风险2"],'
        '"open_questions":["需要精读或复现确认的问题"],'
        '"figure_notes":[{"path":"figures 中的 markdown_path","caption":"基于 caption 候选整理的图注","explanation":"这张图在报告中的作用；证据不足则不要放"}],'
        '"evidence":["PAGE 1: 证据摘要","PAGE 4: 实验证据摘要"],'
        '"report_basis":"PDF全文与摘要|PDF全文截断与摘要|摘要与评审信息"}]}\n'
        f"论文列表：\n{json.dumps(papers, ensure_ascii=False)}"
    )


def parse_report_payload(payload: str) -> list[dict[str, Any]]:
    data = json.loads(_extract_json(payload))
    if isinstance(data, dict):
        reports = data.get("reports", [])
        if isinstance(reports, list):
            return [item for item in reports if isinstance(item, dict)]
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    raise ValueError("Codex report JSON must contain a reports list")


_INTERNAL_SIGNAL_RE = re.compile(
    r"\b(?:primary|method|engineering|category|freshness|penalty|code|citation|codex_judge|codex_keyword):",
    flags=re.IGNORECASE,
)


def _looks_like_internal_signal(value: str) -> bool:
    text = str(value or "")
    return bool(_INTERNAL_SIGNAL_RE.search(text))


def _public_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text or _looks_like_internal_signal(text):
        return ""
    return text


def _public_list(value: Any) -> list[str]:
    return [item for item in _string_list(value) if not _looks_like_internal_signal(item)]


def _trim_text(value: Any, limit: int = 420) -> str:
    text = _normalize_space(str(value or ""))
    if len(text) <= limit:
        return text
    cut = text[:limit].rstrip()
    sentence_end = max(cut.rfind("。"), cut.rfind("."), cut.rfind("；"), cut.rfind(";"))
    if sentence_end > int(limit * 0.45):
        return cut[: sentence_end + 1]
    return cut + "..."


def fallback_report_data(
    recommendation: PaperRecommendation,
    paper_text: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paper = recommendation.paper
    summary = recommendation.summary
    basis = (paper_text or {}).get("basis") or "摘要与评审信息"
    one_sentence = _public_text(summary.one_sentence)
    contribution = _public_text(summary.contribution)
    reason = _public_text(summary.recommendation_reason)
    business_value = _public_text(summary.business_value)
    results = _public_text(summary.results)
    technical_points = _public_list(summary.technical_points)
    abstract_summary = _trim_text(paper.abstract, 420)
    primary_task = summary.primary_task or "感知视觉"
    problem = contribution or abstract_summary or "需阅读论文正文确认问题定义。"
    why_problem_matters = (
        reason
        or f"该论文与{primary_task}方向存在业务相关性，建议结合全文方法、实验设置和内部数据进一步评估。"
    )
    return {
        "arxiv_id": paper.arxiv_id,
        "executive_summary": one_sentence or reason or _trim_text(paper.abstract, 260),
        "target_reader": f"{primary_task} 方向算法工程师",
        "problem": problem,
        "why_problem_matters": why_problem_matters,
        "method_overview": contribution or "需阅读论文正文确认方法细节。",
        "method_details": technical_points or ["基于当前摘要和推荐信息生成，方法细节需精读全文确认。"],
        "method_deep_dive": technical_points or ["证据不足：未能完成 PDF 全文结构化解析。"],
        "innovation_points": technical_points or ["等待精读全文后补充明确创新点。"],
        "experiments": results or "证据不足，需阅读正文确认实验设置和结果。",
        "business_value": business_value or "需结合内部数据和部署约束继续评估。",
        "implementation_plan": [
            "1天内：确认论文任务、数据集、指标和是否有官方代码。",
            "1周内：选择一个内部样本集做最小复现实验或离线评估。",
            f"后续：根据 {primary_task} 收益决定是否进入技术储备。",
        ],
        "risks": _public_list(summary.risks) or ["尚未完成内部复现，需验证业务数据分布下的效果。"],
        "open_questions": ["代码是否可用", "关键指标是否能迁移到内部数据", "推理成本是否满足部署约束"],
        "figure_notes": _figure_notes_from_material((paper_text or {}).get("figures") or []),
        "figures": (paper_text or {}).get("figures") or [],
        "evidence": _fallback_evidence(paper_text),
        "report_basis": basis,
    }


def _normalize_report_data(data: dict[str, Any], recommendation: PaperRecommendation) -> dict[str, Any]:
    fallback = fallback_report_data(recommendation)
    normalized = dict(fallback)
    normalized.update({key: value for key, value in data.items() if value not in (None, "")})
    for key in (
        "executive_summary",
        "target_reader",
        "problem",
        "why_problem_matters",
        "method_overview",
        "experiments",
        "business_value",
    ):
        if _looks_like_internal_signal(str(normalized.get(key) or "")):
            normalized[key] = fallback.get(key, "")
    for key in ("method_details", "method_deep_dive", "innovation_points", "implementation_plan", "risks", "open_questions", "evidence"):
        normalized[key] = _public_list(normalized.get(key)) or _string_list(fallback.get(key))
    normalized["figure_notes"] = _figure_note_list(normalized.get("figure_notes")) or _figure_note_list(fallback.get("figure_notes"))
    normalized["figures"] = _figure_note_list(normalized.get("figures")) or _figure_note_list(fallback.get("figures"))
    return normalized


def render_report_html(
    recommendation: PaperRecommendation,
    report: dict[str, Any],
    generated_at: str,
    report_basis: str,
) -> str:
    paper = recommendation.paper
    summary = recommendation.summary
    title = paper.title
    authors = ", ".join(paper.authors[:12])
    if len(paper.authors) > 12:
        authors += f", et al. (+{len(paper.authors) - 12})"
    meta_items = [
        ("arXiv ID", paper.arxiv_id),
        ("发布时间", paper.published[:10]),
        ("方向", summary.primary_task or "感知视觉"),
        ("推荐等级", summary.recommendation_level),
        ("推荐分", f"{paper.score:.1f} / 30"),
        ("报告依据", report_basis),
    ]
    link_buttons = [
        ("查看论文", paper.arxiv_url),
        ("下载 PDF", paper.pdf_url),
        ("查看代码", paper.code_url),
    ]
    link_html = "\n".join(
        f'<a class="button" href="{escape(url)}" target="_blank" rel="noreferrer">{escape(label)}</a>'
        for label, url in link_buttons
        if url and url.startswith(("http://", "https://"))
    )
    business_judgement = (
        _public_text(summary.recommendation_reason)
        or _public_text(report.get("business_value"))
        or "需结合业务场景、复现成本和内部数据进一步评估。"
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{escape(title)} | 论文阅读报告</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f8fb;
      --text: #172033;
      --muted: #667085;
      --line: #d8dee9;
      --panel: #ffffff;
      --panel-alt: #eef4f7;
      --accent: #0f766e;
      --accent-2: #b45309;
      --danger: #b42318;
      --shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); line-height: 1.62; }}
    a {{ color: inherit; }}
    .shell {{ width: min(1120px, 100%); margin: 0 auto; padding: 32px 20px 54px; }}
    .hero {{ border: 1px solid var(--line); border-radius: 8px; background: var(--panel); box-shadow: var(--shadow); padding: 28px; }}
    .eyebrow {{ margin: 0 0 10px; color: var(--accent); font-size: 13px; font-weight: 800; }}
    h1 {{ margin: 0; font-size: clamp(26px, 4vw, 42px); line-height: 1.16; letter-spacing: 0; }}
    .authors {{ margin: 14px 0 0; color: var(--muted); }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 22px; }}
    .meta {{ border: 1px solid var(--line); border-radius: 8px; background: var(--panel-alt); padding: 12px; min-width: 0; }}
    .meta span {{ display: block; color: var(--muted); font-size: 12px; }}
    .meta strong {{ display: block; margin-top: 4px; overflow-wrap: anywhere; }}
    .actions {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 22px; }}
    .button {{ display: inline-flex; align-items: center; min-height: 38px; border: 1px solid var(--accent); border-radius: 8px; color: var(--accent); text-decoration: none; font-weight: 760; padding: 7px 13px; }}
    .button:hover {{ background: rgba(15, 118, 110, 0.08); }}
    .section {{ margin-top: 18px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel); box-shadow: var(--shadow); padding: 22px; }}
    h2 {{ margin: 0 0 12px; font-size: 20px; letter-spacing: 0; }}
    .lead {{ font-size: 17px; }}
    .two-col {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .label {{ color: var(--accent-2); font-size: 13px; font-weight: 800; }}
    ul {{ margin: 8px 0 0; padding-left: 20px; }}
    li {{ margin: 6px 0; }}
    .risk li::marker {{ color: var(--danger); }}
    .footer {{ margin-top: 22px; color: var(--muted); font-size: 13px; }}
    @media (max-width: 760px) {{ .meta-grid, .two-col {{ grid-template-columns: 1fr; }} .hero, .section {{ padding: 18px; }} }}
  </style>
</head>
<body>
  <main class="shell">
    <header class="hero">
      <p class="eyebrow">论文阅读报告</p>
      <h1>{escape(title)}</h1>
      <p class="authors">{escape(authors or "作者未知")}</p>
      <div class="meta-grid">
        {_meta_grid(meta_items)}
      </div>
      <div class="actions">{link_html}</div>
    </header>

    <section class="section">
      <h2>一页结论</h2>
      <p class="lead">{escape(str(report.get("executive_summary") or ""))}</p>
      <p><span class="label">适合读者：</span>{escape(str(report.get("target_reader") or ""))}</p>
      <p><span class="label">业务判断：</span>{escape(business_judgement)}</p>
    </section>

    <section class="section two-col">
      <div>
        <h2>问题定义</h2>
        <p>{escape(str(report.get("problem") or ""))}</p>
      </div>
      <div>
        <h2>问题价值</h2>
        <p>{escape(str(report.get("why_problem_matters") or ""))}</p>
      </div>
    </section>

    <section class="section">
      <h2>核心方法拆解</h2>
      <p>{escape(str(report.get("method_overview") or ""))}</p>
      {_list_html(report.get("method_details"))}
    </section>

    <section class="section">
      <h2>创新点</h2>
      {_list_html(report.get("innovation_points"))}
    </section>

    <section class="section two-col">
      <div>
        <h2>实验与证据</h2>
        <p>{escape(str(report.get("experiments") or ""))}</p>
      </div>
      <div>
        <h2>业务价值</h2>
        <p>{escape(str(report.get("business_value") or ""))}</p>
      </div>
    </section>

    <section class="section">
      <h2>落地建议</h2>
      {_list_html(report.get("implementation_plan"))}
    </section>

    <section class="section risk">
      <h2>风险限制</h2>
      {_list_html(report.get("risks"))}
    </section>

    <section class="section">
      <h2>待确认问题</h2>
      {_list_html(report.get("open_questions"))}
    </section>

    <p class="footer">生成时间：{escape(generated_at)}。报告依据：{escape(report_basis)}。如报告依据不是 PDF 正文，结论应以论文全文复核为准。</p>
  </main>
</body>
</html>
"""


def render_report_markdown(
    recommendation: PaperRecommendation,
    report: dict[str, Any],
    generated_at: str,
    report_basis: str,
) -> str:
    paper = recommendation.paper
    summary = recommendation.summary
    authors = ", ".join(paper.authors[:12])
    if len(paper.authors) > 12:
        authors += f", et al. (+{len(paper.authors) - 12})"

    meta_items = [
        ("arXiv ID", paper.arxiv_id),
        ("发布时间", paper.published[:10]),
        ("作者", authors or "作者未知"),
        ("类别", ", ".join(paper.categories)),
        ("方向", summary.primary_task or "感知视觉"),
        ("推荐等级", summary.recommendation_level),
        ("推荐分", f"{paper.score:.1f} / 30"),
        ("业务相关度", summary.business_relevance),
        ("工程落地性", summary.deployability),
        ("代码", paper.code_url or "未知"),
        ("报告依据", report_basis),
        ("生成时间", generated_at),
    ]
    source_links = [
        ("查看论文", paper.arxiv_url),
        ("下载 PDF", paper.pdf_url),
        ("查看代码", paper.code_url),
    ]
    business_judgement = (
        _public_text(summary.recommendation_reason)
        or _public_text(report.get("business_value"))
        or "需结合业务场景、复现成本和内部数据进一步评估。"
    )
    mermaid = str(report.get("method_mermaid") or report.get("mermaid") or "").strip()
    experiment_tables = _experiment_tables_markdown(report.get("experiment_tables"))
    return "\n".join(
        [
            f"# {_md_text(paper.title)}",
            "",
            "> 论文阅读报告。若“报告依据”不是 PDF 全文，结论需以论文全文复核为准。",
            "",
            "## 基本信息",
            "",
            _markdown_table(meta_items),
            "",
            "## 原始链接",
            "",
            _markdown_links(source_links),
            "",
            "## 一页结论",
            "",
            _md_text(str(report.get("executive_summary") or "")),
            "",
            f"**适合读者：** {_md_text(str(report.get('target_reader') or ''))}",
            "",
            f"**业务判断：** {_md_text(business_judgement)}",
            "",
            "## 图解材料",
            "",
            _markdown_figures(report),
            "",
            "## 方法流程图",
            "",
            f"```mermaid\n{mermaid}\n```" if mermaid else "暂无可用方法流程图。",
            "",
            "## Heilmeier 七问精读",
            "",
            render_heilmeier_markdown(report),
            "",
            "## 创新点",
            "",
            _markdown_list(report.get("innovation_points")),
            "",
            "## 结构化实验表",
            "",
            experiment_tables or "暂无结构化实验表。",
            "",
            "## 业务价值",
            "",
            _md_text(str(report.get("business_value") or "")),
            "",
            "## 落地建议",
            "",
            _markdown_list(report.get("implementation_plan")),
            "",
            "## 风险限制",
            "",
            _markdown_list(report.get("risks")),
            "",
            "## 待确认问题",
            "",
            _markdown_list(report.get("open_questions")),
            "",
            "## 证据索引",
            "",
            _markdown_list(report.get("evidence")),
            "",
        ]
    )


def write_reports_index(
    recommendations: list[PaperRecommendation],
    output_dir: Path,
    generated_at: str,
    report_format: str = "html",
) -> None:
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "arxiv_id": item.paper.arxiv_id,
            "title": item.paper.title,
            "authors": item.paper.authors,
            "published": item.paper.published,
            "recommended_on": item.recommended_on,
            "score": round(item.paper.score, 2),
            "level": item.summary.recommendation_level,
            "primary_task": item.summary.primary_task,
            "one_sentence": item.summary.one_sentence,
            "report_path": item.report_path,
            "report_url": item.report_url,
            "paper_url": item.paper.arxiv_url,
            "pdf_url": item.paper.pdf_url,
            "code_url": item.paper.code_url,
            "report_basis": item.report_basis,
        }
        for item in recommendations
    ]
    payload = {"generated_at": generated_at, "reports": records}
    (data_dir / "reports.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if report_format == "markdown":
        (output_dir / "README.md").write_text(render_reports_index_markdown(records, generated_at), encoding="utf-8")
    else:
        (output_dir / "index.html").write_text(render_reports_index_html(records, generated_at), encoding="utf-8")


def render_reports_index_html(records: list[dict[str, Any]], generated_at: str) -> str:
    cards = "\n".join(_index_card(record) for record in records) or '<div class="empty">暂无论文阅读报告。</div>'
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>论文阅读报告</title>
  <style>
    :root {{ color-scheme: light; --bg: #f7f8fb; --panel: #fff; --text: #172033; --muted: #667085; --line: #d8dee9; --accent: #0f766e; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); }}
    .shell {{ width: min(1120px, 100%); margin: 0 auto; padding: 32px 20px 54px; }}
    header {{ margin-bottom: 18px; }}
    h1 {{ margin: 0; font-size: clamp(28px, 4vw, 44px); letter-spacing: 0; }}
    .muted {{ color: var(--muted); }}
    .grid {{ display: grid; gap: 12px; }}
    .card {{ border: 1px solid var(--line); border-radius: 8px; background: var(--panel); padding: 18px; }}
    .meta {{ color: var(--muted); font-size: 13px; }}
    h2 {{ margin: 8px 0; font-size: 20px; letter-spacing: 0; }}
    p {{ line-height: 1.58; }}
    .actions {{ display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }}
    a.button {{ display: inline-flex; min-height: 36px; align-items: center; border: 1px solid var(--accent); border-radius: 8px; color: var(--accent); text-decoration: none; font-weight: 760; padding: 7px 12px; }}
    .empty {{ border: 1px dashed var(--line); border-radius: 8px; padding: 28px; color: var(--muted); text-align: center; }}
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <h1>论文阅读报告</h1>
      <p class="muted">生成时间：{escape(generated_at)}</p>
    </header>
    <section class="grid">
      {cards}
    </section>
  </main>
</body>
</html>
"""


def render_reports_index_markdown(records: list[dict[str, Any]], generated_at: str) -> str:
    lines = [
        "# 论文阅读报告索引",
        "",
        f"生成时间：{_md_text(generated_at)}",
        "",
        "| 日期 | arXiv ID | 方向 | 等级 | 推荐分 | 标题 |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    if not records:
        lines.extend(["", "暂无论文阅读报告。", ""])
        return "\n".join(lines)
    for record in records:
        href = _markdown_report_href(record)
        arxiv_id = str(record.get("arxiv_id") or "")
        title = str(record.get("title") or "")
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_table_cell(str(record.get("recommended_on") or "")),
                    f"[{_md_link_label(arxiv_id)}](<{href}>)" if href else _md_table_cell(arxiv_id),
                    _md_table_cell(str(record.get("primary_task") or "感知视觉")),
                    _md_table_cell(str(record.get("level") or "")),
                    _md_table_cell(str(record.get("score") or "")),
                    f"[{_md_link_label(title)}](<{href}>)" if href else _md_table_cell(title),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _index_card(record: dict[str, Any]) -> str:
    report_href = _html_report_href(record)
    return f"""<article class="card">
  <div class="meta">{escape(str(record.get("recommended_on") or ""))} · {escape(str(record.get("primary_task") or "感知视觉"))} · {escape(str(record.get("level") or ""))} · {escape(str(record.get("score") or ""))}/30</div>
  <h2>{escape(str(record.get("title") or ""))}</h2>
  <p>{escape(str(record.get("one_sentence") or ""))}</p>
  <div class="actions">
    <a class="button" href="{escape(report_href)}">阅读报告</a>
    {_optional_button("查看论文", str(record.get("paper_url") or ""))}
    {_optional_button("PDF", str(record.get("pdf_url") or ""))}
  </div>
</article>"""


def _html_report_href(record: dict[str, Any]) -> str:
    path = str(record.get("report_path") or "")
    if path.startswith("web/"):
        return path.removeprefix("web/")
    if path.startswith("reports/reports/"):
        return path.removeprefix("reports/")
    if path.startswith("reports/"):
        return path
    return str(record.get("report_url") or "#")


def _markdown_report_href(record: dict[str, Any]) -> str:
    path = str(record.get("report_path") or "")
    if path.startswith("reports/"):
        return path.removeprefix("reports/")
    return path


def _optional_button(label: str, url: str) -> str:
    if not url.startswith(("http://", "https://")):
        return ""
    return f'<a class="button" href="{escape(url)}" target="_blank" rel="noreferrer">{escape(label)}</a>'


def _meta_grid(items: list[tuple[str, str]]) -> str:
    return "\n".join(
        f'<div class="meta"><span>{escape(label)}</span><strong>{escape(str(value or "-"))}</strong></div>'
        for label, value in items
    )


def _list_html(values: Any) -> str:
    items = _string_list(values)
    if not items:
        return "<p>暂无。</p>"
    return "<ul>" + "".join(f"<li>{escape(item)}</li>" for item in items) + "</ul>"


def _markdown_table(items: list[tuple[str, str]]) -> str:
    lines = ["| 字段 | 内容 |", "| --- | --- |"]
    lines.extend(f"| {_md_table_cell(label)} | {_md_table_cell(str(value or '-'))} |" for label, value in items)
    return "\n".join(lines)


def _markdown_links(items: list[tuple[str, str]]) -> str:
    links = [
        f"- [{_md_link_label(label)}](<{url}>)"
        for label, url in items
        if url and url.startswith(("http://", "https://"))
    ]
    return "\n".join(links) if links else "暂无。"


def _markdown_list(values: Any) -> str:
    items = _string_list(values)
    if not items:
        return "暂无。"
    return "\n".join(f"- {_md_text(item)}" for item in items)


def _experiment_tables_markdown(value: Any) -> str:
    tables = [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []
    blocks: list[str] = []
    for table in tables:
        title = str(table.get("title") or table.get("name") or "").strip()
        columns = _string_list(table.get("columns"))
        rows = table.get("rows") if isinstance(table.get("rows"), list) else []
        if title:
            blocks.extend([f"### {_md_text(title)}", ""])
        if not columns or not rows:
            note = str(table.get("note") or "").strip()
            if note:
                blocks.extend([_md_text(note), ""])
            continue
        blocks.append("| " + " | ".join(_md_table_cell(column) for column in columns) + " |")
        blocks.append("| " + " | ".join("---" for _ in columns) + " |")
        for row in rows:
            if isinstance(row, dict):
                cells = [str(row.get(column) or "") for column in columns]
            elif isinstance(row, list):
                cells = [str(item) for item in row[: len(columns)]]
                cells.extend([""] * (len(columns) - len(cells)))
            else:
                continue
            blocks.append("| " + " | ".join(_md_table_cell(cell) for cell in cells) + " |")
        blocks.append("")
    return "\n".join(blocks).strip()


def _figure_manifest_markdown(report: dict[str, Any]) -> str:
    figure_notes = _figure_note_list(report.get("figure_notes"))
    if not figure_notes:
        figure_notes = _figure_notes_from_material(_figure_note_list(report.get("figures")))
    lines: list[str] = []
    for index, figure in enumerate(figure_notes, start=1):
        caption = _md_text(str(figure.get("caption") or f"图 {index}"))
        explanation = _md_text(str(figure.get("explanation") or "该图来自论文 PDF，需结合正文核对。"))
        path = str(figure.get("path") or figure.get("markdown_path") or "").strip()
        lines.append(f"- 图 {index}：{caption}；用途：{explanation}；归档路径：{path or '无'}")
    return "\n".join(lines)


def _markdown_figures(report: dict[str, Any]) -> str:
    figure_notes = _figure_note_list(report.get("figure_notes"))
    if not figure_notes:
        figure_notes = _figure_notes_from_material(_figure_note_list(report.get("figures")))
    if not figure_notes:
        return "未抽取到可用图像，或图像候选缺少可解释 caption。"
    blocks = []
    for index, figure in enumerate(figure_notes, start=1):
        path = str(figure.get("path") or figure.get("markdown_path") or "").strip()
        caption = _md_text(str(figure.get("caption") or f"图 {index}"))
        explanation = _md_text(str(figure.get("explanation") or "该图来自论文 PDF，需结合正文核对。"))
        if not path:
            continue
        blocks.extend([f"![{_md_link_label(caption)}](<{path}>)", "", f"> {explanation}", ""])
    return "\n".join(blocks).strip() or "未抽取到可用图像，或图像候选缺少可解释 caption。"


def _figure_notes_from_material(figures: list[dict[str, Any]]) -> list[dict[str, Any]]:
    notes = []
    for index, figure in enumerate(figures, start=1):
        path = str(figure.get("markdown_path") or figure.get("path") or "").strip()
        captions = _string_list(figure.get("caption_candidates"))
        caption = captions[0] if captions else f"PDF 图像候选 {index}"
        page = figure.get("page") or "未知"
        notes.append(
            {
                "path": path,
                "caption": caption,
                "explanation": f"来自 PDF 第 {page} 页的图像候选；自动抽图无法保证与 caption 一一对应，建议人工核对。",
            }
        )
    return notes


def _figure_note_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    notes = []
    for item in value:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or item.get("markdown_path") or "").strip()
        if not path:
            continue
        notes.append({key: item.get(key) for key in item.keys()})
    return notes


def _fallback_evidence(paper_text: dict[str, Any] | None) -> list[str]:
    if not paper_text:
        return ["证据不足：未读取 PDF 全文。"]
    basis = str(paper_text.get("basis") or "摘要与评审信息")
    status = str(paper_text.get("status") or "")
    pages_read = int(paper_text.get("pages_read") or 0)
    if basis in {"PDF全文与摘要", "PDF全文截断与摘要"}:
        page_evidence = _evidence_from_page_text(str(paper_text.get("text") or ""))
        if page_evidence:
            return page_evidence
        return [f"PAGE 1: 已读取 PDF 全文材料：{pages_read} 页，抽取状态 {status}。"]
    return [f"证据不足：报告依据为 {basis}，抽取状态 {status}。"]


def _evidence_from_page_text(text: str, limit: int = 6) -> list[str]:
    evidence: list[str] = []
    for match in re.finditer(r"\[PAGE\s+(\d+)\]\s*(.*?)(?=\n\s*\[PAGE\s+\d+\]|\Z)", text, flags=re.IGNORECASE | re.DOTALL):
        page = match.group(1)
        page_text = _normalize_space(match.group(2))
        if not page_text:
            continue
        evidence.append(f"PAGE {page}: {_trim_text(page_text, 180)}")
        if len(evidence) >= limit:
            break
    return evidence


def _paper_analyzer_evidence(markdown: str, fallback: Any) -> list[str]:
    page_refs = []
    for match in re.finditer(r"PAGE\s+\d+[^。\n]*(?:。|$)", markdown, flags=re.IGNORECASE):
        text = match.group(0).strip()
        if text and text not in page_refs:
            page_refs.append(text)
        if len(page_refs) >= 8:
            break
    return page_refs or _string_list(fallback)


def _md_text(value: str) -> str:
    text = str(value or "").strip()
    return text or "暂无。"


def _md_link_label(value: str) -> str:
    return _md_text(value).replace("[", "\\[").replace("]", "\\]")


def _md_table_cell(value: str) -> str:
    return _md_text(value).replace("\n", "<br>").replace("|", "\\|")


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        parts = [item.strip(" -\t") for item in re.split(r"[\n；;]+", value) if item.strip(" -\t")]
        return parts if len(parts) > 1 else ([value.strip()] if value.strip() else [])
    return []


def _paper_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-")
    return slug or "paper"


def _report_publisher(reports_config: dict[str, Any]) -> str:
    value = str(reports_config.get("publisher") or "file").strip().lower()
    return "feishu_doc" if value in {"feishu", "feishu_doc", "doc", "docx"} else "file"


def _report_content_generator(reports_config: dict[str, Any]) -> str:
    value = str(reports_config.get("generator") or reports_config.get("content_generator") or "heilmeier").strip().lower()
    if value in {"paper_craft", "paper-craft", "paper_craft_skill", "paper-craft-skills"}:
        return "paper_craft"
    if value in {"paper_analyzer", "paper-analyzer", "paper_analyzer_skill"}:
        return "paper_analyzer"
    return "heilmeier"


def _archive_report_format(reports_config: dict[str, Any], publisher: str) -> str:
    if publisher == "feishu_doc" and reports_config.get("archive_markdown", True):
        return "markdown"
    return _report_format(reports_config)


def _report_format(reports_config: dict[str, Any]) -> str:
    value = str(reports_config.get("format") or "html").strip().lower()
    return "markdown" if value in {"md", "markdown"} else "html"


def _report_relative_path(report_format: str, slug: str) -> str:
    if report_format == "markdown":
        return f"papers/{slug}.md"
    return f"reports/{slug}/index.html"


def _markdown_asset_path(asset_path: str) -> str:
    return "../" + asset_path.strip("/")


def _feishu_image_paths(report: dict[str, Any], output_dir: Path, max_images: int = 4) -> list[Path]:
    if max_images <= 0:
        return []
    paths: list[Path] = []
    figure_notes = _figure_note_list(report.get("figure_notes")) or _figure_note_list(report.get("figures"))
    for figure in figure_notes:
        raw_path = str(figure.get("path") or figure.get("markdown_path") or "").strip()
        if not raw_path:
            continue
        normalized = raw_path.removeprefix("../").lstrip("/")
        image_path = output_dir / normalized
        if image_path.exists() and image_path.is_file():
            paths.append(image_path)
        if len(paths) >= max_images:
            break
    return paths


def _captions_for_page(pages: list[dict[str, Any]], page_number: int) -> list[str]:
    for page in pages:
        if int(page.get("page") or 0) == page_number:
            return _string_list(page.get("captions"))
    return []


def _find_caption_candidates(text: str) -> list[str]:
    normalized = _normalize_page_text(text)
    patterns = [
        r"(?:Fig\.|Figure)\s*\d+[a-zA-Z]?[.:]\s*.{20,500}",
        r"(?:Table)\s*\d+[a-zA-Z]?[.:]\s*.{20,500}",
    ]
    captions: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
            caption = match.group(0)
            caption = re.split(r"\s{2,}|\n", caption)[0].strip()
            if caption and caption not in captions:
                captions.append(caption[:500])
            if len(captions) >= 4:
                return captions
    return captions


def _absolute_report_url(base_url: str, rel_path: str) -> str:
    if not base_url:
        return ""
    return base_url.rstrip("/") + "/" + quote(rel_path, safe="/._-")


def _report_base_url(reports_config: dict[str, Any]) -> str:
    base_url = (os.getenv("PAPER_REPORT_BASE_URL") or str(reports_config.get("base_url") or "")).strip()
    if (
        _report_publisher(reports_config) == "file"
        and reports_config.get("publish_before_send", reports_config.get("git_push_before_send", True)) is False
        and not reports_config.get("expose_unpublished_report_urls", False)
    ):
        return ""
    return base_url


def _resolve_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _load_json(path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return dict(fallback)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else dict(fallback)
    except json.JSONDecodeError:
        return dict(fallback)


def _normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _normalize_page_text(value: str) -> str:
    text = value.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
