from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.paper_recommender.codex_summarizer import _parse_summary_payload
from tools.paper_recommender.codex_judge import apply_judgements, parse_judge_payload, select_judge_candidates
from tools.paper_recommender.config import load_config
from tools.paper_recommender.copywriting import annotate_copywriting_signals, build_display_title, sanitize_copy_text
from tools.paper_recommender.feishu_bitable import FeishuBitableClient
from tools.paper_recommender.feishu_bot import FeishuBotClient
from tools.paper_recommender.feishu_doc import FeishuDocClient, FeishuDocument
from tools.paper_recommender.fetch_arxiv import parse_arxiv_feed
from tools.paper_recommender.fetch_cvf_openaccess import parse_cvf_openaccess_abstract, parse_cvf_openaccess_papers
from tools.paper_recommender.fetch_huggingface import parse_compact_number, parse_huggingface_papers
from tools.paper_recommender.fetch_semantic_scholar import apply_semantic_scholar_metadata
from tools.paper_recommender.models import PaperCandidate, PaperRecommendation, PaperSummary
from tools.paper_recommender.normalize import deduplicate_papers
from tools.paper_recommender.paper_craft_report import build_paper_craft_html_prompt, clean_paper_craft_html
from tools.paper_recommender.rank import rank_papers, score_paper_breakdown
from tools.paper_recommender.report_generator import generate_reports, parse_report_payload, render_report_markdown
from tools.paper_recommender.report_quality import scrub_report_data, validate_report_quality
from tools.paper_recommender.run_daily import (
    filter_previously_pushed_papers,
    historical_pushed_arxiv_ids,
    mark_recommendations_pushed_in_history,
    publish_reports,
)


SAMPLE_ARXIV = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2605.00001v1</id>
    <updated>2026-05-28T00:00:00Z</updated>
    <published>2026-05-28T00:00:00Z</published>
    <title>Efficient Real-Time Face Detection on Edge Devices</title>
    <summary>We propose a lightweight detector with quantization-aware training.</summary>
    <author><name>Alice</name></author>
    <author><name>Bob</name></author>
    <arxiv:primary_category term="cs.CV" />
    <category term="cs.CV" />
    <link href="http://arxiv.org/pdf/2605.00001v1" title="pdf" type="application/pdf" />
  </entry>
</feed>
"""

SAMPLE_HF_HTML = """
<article class="relative overflow-hidden rounded-xl border">
  <h3><a href="/papers/2606.12345" class="line-clamp-2">Vision Parser: Efficient Document Understanding</a></h3>
  <p class="line-clamp-2">A compact vision-language model parses high-resolution documents efficiently.</p>
  <a href="/ExampleLab"><span class="block min-w-0 truncate font-medium">Example Lab</span></a>
  <span>Published on Jun 17, 2026</span>
  <div>Upvote <div class="font-semibold text-orange-500">167</div></div>
  <a href="https://github.com/example/vision-parser">GitHub <span>8.14k</span></a>
  <a href="https://arxiv.org/abs/2606.12345">arXiv Page</a>
</article>
"""

SAMPLE_CVF_HTML = """
<dl>
<dt class="ptitle"><br><a href="/content/CVPR2026/html/Liu_From_Pairs_to_Sequences_Track-Aware_Policy_Gradients_for_Keypoint_Detection_CVPR_2026_paper.html">From Pairs to Sequences: Track-Aware Policy Gradients for Keypoint Detection</a></dt>
<dd>
<form><input type="hidden" name="query_author" value="Yepeng Liu"><a>Yepeng Liu</a>,</form>
<form><input type="hidden" name="query_author" value="Hao Li"><a>Hao Li</a></form>
</dd>
<dd>
[<a href="/content/CVPR2026/papers/Liu_From_Pairs_to_Sequences_Track-Aware_Policy_Gradients_for_Keypoint_Detection_CVPR_2026_paper.pdf">pdf</a>]
<div class="bibref pre-white-space">@InProceedings{Liu_2026_CVPR}</div>
</dd>
</dl>
"""

SAMPLE_CVF_DETAIL_HTML = """
<html><body>
<div id="abstract">
3D single object tracking is essential for real-world perception.
The method improves Success by 4.37% and Precision by 5.16%.
</div>
</body></html>
"""


class PaperRecommenderTests(unittest.TestCase):
    def test_parse_arxiv_feed(self) -> None:
        papers = parse_arxiv_feed(SAMPLE_ARXIV)
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].arxiv_id, "2605.00001")
        self.assertEqual(papers[0].authors, ["Alice", "Bob"])
        self.assertEqual(papers[0].primary_category, "cs.CV")

    def test_parse_arxiv_feed_extracts_prestige_venue_from_comment(self) -> None:
        body = SAMPLE_ARXIV.replace(
            b"<arxiv:primary_category term=\"cs.CV\" />",
            b"<arxiv:primary_category term=\"cs.CV\" />\n    <arxiv:comment>Accepted by ECCV 2026. Project page: https://github.com/example/textds</arxiv:comment>",
        )
        paper = parse_arxiv_feed(body)[0]
        self.assertEqual(paper.venue, "ECCV 2026")
        self.assertEqual(paper.code_url, "https://github.com/example/textds")
        self.assertIn("prestige venue ECCV", paper.quality_signals)

    def test_deduplicate_by_arxiv_and_title(self) -> None:
        first = make_paper("2605.00001", "Efficient Detector")
        second = make_paper("2605.00001", "Efficient Detector v2")
        third = make_paper("2605.00002", "Efficient Detector")
        unique = deduplicate_papers([first, second, third])
        self.assertEqual(len(unique), 1)

    def test_huggingface_parser_extracts_hotness_signals(self) -> None:
        papers = parse_huggingface_papers(SAMPLE_HF_HTML)
        self.assertEqual(len(papers), 1)
        paper = papers[0]
        self.assertEqual(paper.arxiv_id, "2606.12345")
        self.assertEqual(paper.hf_upvotes, 167)
        self.assertEqual(paper.github_stars, 8140)
        self.assertEqual(paper.code_url, "https://github.com/example/vision-parser")
        self.assertEqual(paper.venue, "Example Lab")
        self.assertIn("Example Lab", paper.institution_signals)
        self.assertGreater(paper.hotness_score, 3)
        self.assertIn("Hugging Face daily/trending", paper.hotness_signals)
        self.assertEqual(parse_compact_number("72.3k"), 72300)

    def test_cvf_openaccess_parser_extracts_prestige_venue_papers(self) -> None:
        papers = parse_cvf_openaccess_papers(
            SAMPLE_CVF_HTML,
            source_url="https://openaccess.thecvf.com/CVPR2026?day=all",
            venue="CVPR",
            year="2026",
            published="2026-05-23T00:00:00+00:00",
            config={"prestige_venues": {"enabled": True, "quality_bonus": 5.0}},
        )
        self.assertEqual(len(papers), 1)
        paper = papers[0]
        self.assertEqual(paper.title, "From Pairs to Sequences: Track-Aware Policy Gradients for Keypoint Detection")
        self.assertEqual(paper.authors, ["Yepeng Liu", "Hao Li"])
        self.assertEqual(paper.venue, "CVPR 2026")
        self.assertIn("openaccess.thecvf.com/content/CVPR2026/html", paper.arxiv_url)
        self.assertTrue(paper.pdf_url.endswith("_CVPR_2026_paper.pdf"))
        self.assertIn("prestige venue CVPR", paper.quality_signals)

    def test_cvf_openaccess_abstract_parser_extracts_detail_abstract(self) -> None:
        abstract = parse_cvf_openaccess_abstract(SAMPLE_CVF_DETAIL_HTML)
        self.assertIn("3D single object tracking", abstract)
        self.assertIn("4.37%", abstract)

    def test_deduplicate_merges_huggingface_metadata(self) -> None:
        arxiv = make_paper("2606.12345", "Vision Parser: Efficient Document Understanding")
        hf = parse_huggingface_papers(SAMPLE_HF_HTML)[0]
        unique = deduplicate_papers([arxiv, hf])
        self.assertEqual(len(unique), 1)
        self.assertEqual(unique[0].hf_upvotes, 167)
        self.assertEqual(unique[0].github_stars, 8140)
        self.assertIn("huggingface", unique[0].source_names)
        self.assertIn("Example Lab", unique[0].institution_signals)

    def test_history_filters_legacy_processed_arxiv_ids(self) -> None:
        previous = make_paper("2605.00001", "Already Recommended")
        fresh = make_paper("2605.00002", "Fresh Candidate")
        filtered, skipped = filter_previously_pushed_papers(
            [previous, fresh],
            {"history": {"enabled": True, "mode": "pushed_forever"}},
            {"processed_arxiv_ids": ["2605.00001"], "generated_reports": {}},
        )
        self.assertEqual(skipped, 1)
        self.assertEqual([paper.arxiv_id for paper in filtered], ["2605.00002"])

    def test_history_prefers_explicit_pushed_records_when_present(self) -> None:
        papers = [
            make_paper("2605.00001", "Generated But Not Pushed"),
            make_paper("2605.00002", "Pushed"),
            make_paper("2605.00003", "Legacy Processed"),
        ]
        state = {
            "processed_arxiv_ids": ["2605.00003"],
            "recommendation_history": {
                "2605.00001": {"pushed": False},
                "2605.00002": {"pushed": True},
            },
        }
        filtered, skipped = filter_previously_pushed_papers(
            papers,
            {"history": {"enabled": True, "mode": "pushed_forever"}},
            state,
        )
        self.assertEqual(skipped, 1)
        self.assertEqual([paper.arxiv_id for paper in filtered], ["2605.00001", "2605.00003"])

    def test_history_filter_can_be_disabled(self) -> None:
        previous = make_paper("2605.00001", "Already Recommended")
        filtered, skipped = filter_previously_pushed_papers(
            [previous],
            {"history": {"enabled": False, "mode": "pushed_forever"}},
            {"processed_arxiv_ids": ["2605.00001"]},
        )
        self.assertEqual(skipped, 0)
        self.assertEqual(filtered, [previous])

    def test_mark_recommendations_pushed_migrates_and_updates_history(self) -> None:
        rec = PaperRecommendation(
            paper=make_paper("2605.00001", "Efficient Detector"),
            summary=PaperSummary(recommendation_reason="值得关注"),
            recommended_on="2026-06-21",
        )
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            state_path = repo_root / "reports/data/state.json"
            state_path.parent.mkdir(parents=True)
            state_path.write_text(
                json.dumps(
                    {
                        "processed_arxiv_ids": ["2605.00002"],
                        "generated_reports": {
                            "2605.00002": {"generated_at": "2026-06-20T01:00:00+00:00"}
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            mark_recommendations_pushed_in_history(
                [rec],
                {"history": {"enabled": True}, "reports": {"state_path": "reports/data/state.json"}},
                repo_root,
                recommended_on="2026-06-21",
            )
            state = json.loads(state_path.read_text(encoding="utf-8"))
            pushed_ids = historical_pushed_arxiv_ids(state)
            self.assertIn("2605.00001", pushed_ids)
            self.assertIn("2605.00002", pushed_ids)
            self.assertEqual(state["recommendation_history"]["2605.00002"]["last_pushed_on"], "2026-06-20")
            self.assertEqual(state["recommendation_history"]["2605.00001"]["first_recommended_on"], "2026-06-21")
            self.assertEqual(state["recommendation_history"]["2605.00001"]["last_pushed_on"], "2026-06-21")

    def test_display_title_prefers_backing_field_and_metric(self) -> None:
        paper = make_paper(
            "2606.12345",
            "RF-DETR: Neural Architecture Search for Real-Time Detection Transformers",
            abstract="RF-DETR reaches 60 FPS and improves AP 2.1 on real-time detection.",
        )
        paper.venue = "Roboflow"
        title = build_display_title(paper, primary_task="实时检测")
        self.assertIn("Roboflow", title)
        self.assertIn("实时检测", title)
        self.assertTrue("60 FPS" in title or "AP 2.1" in title)

    def test_copywriting_marks_configured_notable_authors(self) -> None:
        paper = make_paper("2606.00001", "Generalist Vision Learners")
        paper.authors = ["Kaiming He", "Alice"]
        annotate_copywriting_signals([paper], {"copywriting": {"notable_authors": ["Kaiming He"]}})
        self.assertIn("Kaiming He", paper.notable_author_signals)
        self.assertIn("notable author Kaiming He", paper.quality_signals)

    def test_semantic_scholar_marks_prestige_venue_as_quality_signal(self) -> None:
        paper = make_paper("2606.00001", "A Strong Vision Paper")
        apply_semantic_scholar_metadata(
            paper,
            {
                "venue": "IEEE/CVF Conference on Computer Vision and Pattern Recognition",
                "citationCount": 0,
                "influentialCitationCount": 0,
                "externalIds": {},
            },
            config={"prestige_venues": {"enabled": True, "quality_bonus": 5.0}},
        )
        self.assertGreaterEqual(paper.quality_score, 6.0)
        self.assertIn("prestige venue CVPR", paper.quality_signals)

    def test_rank_prefers_lightweight_detection(self) -> None:
        target = make_paper(
            "2605.00001",
            "Efficient real-time face detection with quantization",
            abstract="A lightweight detector for edge deployment with QAT.",
        )
        unrelated = make_paper(
            "2605.00002",
            "A theorem for molecular language models",
            abstract="A theoretical study for molecular generation.",
        )
        ranked = rank_papers(
            [unrelated, target],
            {
                "ranking": {
                    "strong_keywords": ["efficient", "real-time", "quantization", "detection", "face"],
                    "positive_keywords": ["edge", "lightweight", "qat"],
                    "negative_keywords": ["molecular", "theorem"],
                }
            },
        )
        self.assertEqual(ranked[0].arxiv_id, "2605.00001")
        self.assertGreater(ranked[0].score, ranked[1].score)

    def test_parse_codex_summary_payload_from_fenced_json(self) -> None:
        payload = """```json
        {"papers":[{"arxiv_id":"2605.00001","one_sentence":"值得关注"}]}
        ```"""
        parsed = _parse_summary_payload(payload)
        self.assertEqual(parsed[0]["arxiv_id"], "2605.00001")

    def test_parse_codex_judge_payload_from_fenced_json(self) -> None:
        payload = """```json
        {"papers":[{"arxiv_id":"2605.00001","recommend":true,"score":23.5}]}
        ```"""
        parsed = parse_judge_payload(payload)
        self.assertTrue(parsed[0]["recommend"])
        self.assertEqual(parsed[0]["score"], 23.5)

    def test_parse_report_payload_from_fenced_json(self) -> None:
        payload = """```json
        {"reports":[{"arxiv_id":"2605.00001","executive_summary":"值得读"}]}
        ```"""
        parsed = parse_report_payload(payload)
        self.assertEqual(parsed[0]["arxiv_id"], "2605.00001")
        self.assertEqual(parsed[0]["executive_summary"], "值得读")

    def test_apply_codex_judgement_filters_and_uses_business_fields(self) -> None:
        target = make_paper(
            "2605.00001",
            "Open Vocabulary Object Detection for Data Engines",
            abstract="We improve open-vocabulary detection for automatic annotation.",
        )
        unrelated = make_paper(
            "2605.00002",
            "A Vision-Language Foundation Model for 3D Generation",
            abstract="We generate 3D assets with a VLM.",
        )
        target.score = 12
        unrelated.score = 11
        recommendations = apply_judgements(
            [target, unrelated],
            [
                {
                    "arxiv_id": "2605.00001",
                    "recommend": True,
                    "score": 24,
                    "recommendation_level": "高优先级",
                    "action": "小试",
                    "primary_task": "检测",
                    "business_relevance": "高",
                    "deployability": "中",
                    "one_sentence": "开放词汇检测可服务自动标注。",
                    "recommendation_reason": "直接命中检测和数据闭环主任务。",
                    "business_value": "可用于开放词汇数据筛选和标注。",
                    "risks": "需要验证业务类别。",
                    "matched_keywords": ["open-vocabulary detection", "automatic annotation"],
                },
                {
                    "arxiv_id": "2605.00002",
                    "recommend": False,
                    "score": 6,
                    "recommendation_reason": "偏 3D 生成。",
                },
            ],
            recommended_on="2026-05-29",
            config={"codex": {"judge_min_codex_score": 10}},
        )
        self.assertEqual(len(recommendations), 1)
        self.assertEqual(recommendations[0].paper.arxiv_id, "2605.00001")
        self.assertEqual(recommendations[0].paper.score, 24)
        self.assertEqual(recommendations[0].summary.primary_task, "检测")
        self.assertEqual(recommendations[0].summary.business_relevance, "高")
        self.assertIn("automatic annotation", recommendations[0].summary.matched_keywords)

    def test_apply_codex_judgement_sanitizes_hype_copy(self) -> None:
        paper = make_paper("2605.00001", "Efficient Detector")
        paper.score = 12
        recommendations = apply_judgements(
            [paper],
            [
                {
                    "arxiv_id": "2605.00001",
                    "recommend": True,
                    "score": 24,
                    "display_title": "新方法吊打旧检测器",
                    "hook": "PSNR暴涨，堪称GPT时刻",
                    "recommendation_reason": "值得关注",
                }
            ],
            recommended_on="2026-05-29",
            config={"codex": {"judge_min_codex_score": 10}},
        )
        summary = recommendations[0].summary
        text = f"{summary.display_title} {summary.hook} {summary.recommendation_reason}"
        self.assertNotIn("吊打", text)
        self.assertNotIn("暴涨", text)
        self.assertNotIn("GPT时刻", text)
        self.assertEqual(sanitize_copy_text("刷新SOTA"), "报告较强结果")

    def test_feishu_payload_contains_card_markdown(self) -> None:
        rec = PaperRecommendation(
            paper=make_paper("2605.00001", "Efficient Detector"),
            summary=PaperSummary(
                one_sentence="复杂场景下的检测链路值得小试。",
                contribution="通过场景分析、增强、检测和校验模块动态编排提升鲁棒性。",
                technical_points=["分析图像退化类型", "动态选择增强和检测工具", "根据失败经验调整策略"],
                recommendation_reason="对低光、模糊和遮挡检测有参考价值。",
                business_value="可用于 hard case 归因和困难样本处理策略生成。",
                risks="需要验证推理成本和稳定性。",
                recommendation_level="高优先级",
                action="小试",
                primary_task="检测",
            ),
            recommended_on="2026-05-29",
        )
        payload = FeishuBotClient("https://example.com")._build_payload(
            [rec],
            title="每日论文推荐",
            bitable_url="https://example.com/base",
            stats={"fetched": 10, "deduped": 8},
        )
        self.assertEqual(payload["msg_type"], "interactive")
        text = json.dumps(payload, ensure_ascii=False)
        self.assertIn("Efficient Detector", text)
        self.assertIn("打开追溯表格", text)
        self.assertIn("查看论文", text)
        self.assertIn("**1.** [Efficient Detector](https://arxiv.org/abs/2605.00001)", text)
        self.assertIn("业务相关度", text)
        self.assertIn("工程落地性", text)
        self.assertIn("推荐看点", text)
        self.assertIn("为什么现在值得看", text)
        self.assertIn("论文硬信号", text)
        self.assertIn("方法一句话", text)
        self.assertIn("方法怎么做", text)
        self.assertIn("关键创新点", text)
        self.assertIn("为什么推荐", text)
        self.assertIn("1. 分析图像退化类型", text)
        self.assertIn("命中关键词", text)

    def test_judge_candidate_selection_includes_hot_and_quality_buckets(self) -> None:
        hot = make_paper(
            "2605.00005",
            "A Generalist Vision Model",
            abstract="A broadly useful vision model.",
        )
        hot.hotness_score = 4.0
        hot.hotness_signals = ["HF upvotes 200"]
        quality = make_paper(
            "2605.00006",
            "A Strong Visual Benchmark",
            abstract="A benchmark for visual perception.",
        )
        quality.quality_score = 4.0
        quality.quality_signals = ["influential citations 20"]
        blocked = make_paper(
            "2605.00007",
            "Automated AI Skill Generation with Massive GitHub Stars",
            abstract="An agent skill generation method.",
        )
        blocked.hotness_score = 6.0
        ranked = rank_papers(
            [hot, quality, blocked],
            {
                "ranking": {
                    "primary_task_threshold": 5,
                    "primary_tasks": {"检测": ["object detection"]},
                    "hard_negative_keywords": ["skill generation"],
                    "min_score": 8,
                }
            },
        )
        candidates = select_judge_candidates(
            ranked,
            {
                "ranking": {"min_score": 8, "final_limit": 5, "judge_min_hotness_score": 3, "judge_min_quality_score": 3},
                "codex": {"judge_candidate_limit": 5, "judge_min_rule_score": 8},
            },
        )
        candidate_ids = {paper.arxiv_id for paper in candidates}
        self.assertIn("2605.00005", candidate_ids)
        self.assertIn("2605.00006", candidate_ids)
        self.assertNotIn("2605.00007", candidate_ids)

    def test_judge_candidate_selection_prioritizes_prestige_venues(self) -> None:
        prestige = make_paper(
            "2605.00008",
            "A CVPR Paper Without Strong Business Keywords",
            abstract="A general computer vision method.",
        )
        prestige.venue = "CVPR 2026"
        prestige.quality_score = 6.0
        prestige.quality_signals = ["prestige venue CVPR"]
        prestige.score = 6.0
        ordinary = make_paper(
            "2605.00009",
            "Efficient Object Detection for Edge Devices",
            abstract="A lightweight object detection model.",
        )
        ordinary.score = 20.0
        candidates = select_judge_candidates(
            [ordinary, prestige],
            {
                "ranking": {"min_score": 8, "final_limit": 5},
                "codex": {"judge_candidate_limit": 5, "judge_min_rule_score": 8},
                "prestige_venues": {"enabled": True, "candidate_bucket_limit": 5},
            },
        )
        self.assertEqual(candidates[0].arxiv_id, "2605.00008")
        self.assertEqual(candidates[1].arxiv_id, "2605.00009")

    def test_feishu_payload_scrubs_internal_keyword_reasons(self) -> None:
        paper = make_paper("2605.00001", "Efficient Detector")
        paper.score_reasons = [
            "primary:检测/title:localization",
            "method:A/abstract:vision-language model",
            "category:cs.CV",
        ]
        rec = PaperRecommendation(
            paper=paper,
            summary=PaperSummary(one_sentence="一句话", recommendation_reason="值得观察", risks="需复现"),
            recommended_on="2026-05-29",
        )
        payload = FeishuBotClient("https://example.com")._build_payload(
            [rec],
            title="每日论文推荐",
            bitable_url="",
            stats={"fetched": 10, "deduped": 8},
        )
        text = json.dumps(payload, ensure_ascii=False)
        self.assertIn("localization", text)
        self.assertIn("vision-language model", text)
        self.assertNotIn("primary:", text)
        self.assertNotIn("method:A", text)
        self.assertNotIn("category:", text)

    def test_feishu_payload_contains_report_button_when_url_exists(self) -> None:
        rec = PaperRecommendation(
            paper=make_paper("2605.00001", "Efficient Detector"),
            summary=PaperSummary(one_sentence="一句话", recommendation_reason="轻量化", risks="需复现"),
            recommended_on="2026-05-29",
            report_url="https://example.com/reports/2605.00001/",
        )
        payload = FeishuBotClient("https://example.com")._build_payload(
            [rec],
            title="每日论文推荐",
            bitable_url="https://example.com/base",
            stats={"fetched": 10, "deduped": 8},
        )
        self.assertIn("阅读报告", json.dumps(payload, ensure_ascii=False))

    def test_feishu_bitable_filters_unknown_optional_fields(self) -> None:
        client = FeishuBitableClient(
            token_client=None,  # type: ignore[arg-type]
            app_token="app",
            table_id="tbl",
            field_names={},
        )
        client._known_field_names = {"标题", "推荐分"}
        fields = client.filter_existing_fields({"标题": "Paper", "展示标题": "Display", "推荐分": 20})
        self.assertEqual(fields, {"标题": "Paper", "推荐分": 20})

    def test_feishu_title_link_requires_matching_arxiv_id(self) -> None:
        paper = make_paper("2605.00001", "Efficient Detector")
        paper.arxiv_url = "https://arxiv.org/abs/2605.99999"
        rec = PaperRecommendation(
            paper=paper,
            summary=PaperSummary(one_sentence="一句话", recommendation_reason="轻量化", risks="需复现"),
            recommended_on="2026-05-29",
        )
        payload = FeishuBotClient("https://example.com")._build_payload(
            [rec],
            title="每日论文推荐",
            bitable_url="https://example.com/base",
            stats={"fetched": 10, "deduped": 8},
        )
        text = json.dumps(payload, ensure_ascii=False)
        self.assertIn("**1. Efficient Detector**", text)
        self.assertNotIn("[Efficient Detector](https://arxiv.org/abs/2605.99999)", text)

    def test_generic_vlm_terms_do_not_pass_without_primary_task(self) -> None:
        paper = make_paper(
            "2605.00003",
            "A Vision-Language Foundation Model for 3D Generation",
            abstract="A VLM foundation model with code and benchmark for 3D generation.",
        )
        config = {
            "ranking": {
                "primary_task_threshold": 5,
                "primary_tasks": {"检测": ["object detection"]},
                "keyword_tiers": {"A": ["VLM", "foundation model"], "B": [], "C": []},
                "engineering_keywords": {"lightweight": 2},
                "code_keywords": ["github"],
                "freshness_days": 7,
                "freshness_weight": 2,
                "category_bonus": {"cs.CV": 1.5},
                "min_score": 8,
            }
        }
        breakdown = score_paper_breakdown(paper, config)
        self.assertFalse(breakdown["passed_primary_gate"])
        self.assertLess(breakdown["score"], 8)

    def test_hard_negative_skill_generation_is_filtered(self) -> None:
        paper = make_paper(
            "2605.00004",
            "COLLEAGUE.SKILL: Automated AI Skill Generation via Expert Knowledge Distillation",
            abstract="We build LLM agents that distill human expertise into reusable skills.",
        )
        config = {
            "ranking": {
                "primary_task_threshold": 5,
                "primary_tasks": {"小模型/部署": ["knowledge distillation"]},
                "keyword_tiers": {"A": ["knowledge distillation"], "B": [], "C": []},
                "engineering_keywords": {"deployment": 1},
                "hard_negative_keywords": ["skill generation", "llm agents"],
                "min_score": 8,
            }
        }
        breakdown = score_paper_breakdown(paper, config)
        self.assertEqual(breakdown["score"], 0.0)
        self.assertFalse(breakdown["passed_primary_gate"])
        self.assertIn("filtered:hard_negative:skill generation", breakdown["reasons"])

    def test_report_quality_scrubs_internal_signals(self) -> None:
        rec = PaperRecommendation(
            paper=make_paper("2605.00001", "Efficient Detector"),
            summary=PaperSummary(primary_task="检测", recommendation_level="高优先级"),
            recommended_on="2026-05-29",
        )
        report = {
            "executive_summary": "PAGE 1: 值得读。",
            "business_value": "primary:检测/title:detection",
            "report_basis": "PDF全文与摘要",
        }
        self.assertFalse(validate_report_quality(report, rec).passed)
        cleaned = scrub_report_data(report)
        self.assertTrue(validate_report_quality(cleaned, rec).passed)
        self.assertNotIn("business_value", cleaned)

    def test_feishu_doc_publish_updates_existing_document(self) -> None:
        client = FakeFeishuDocClient()
        document = client.publish_markdown("报告", "# 内容", document_id="doc-old")
        self.assertEqual(document.document_id, "doc-old")
        self.assertEqual(client.calls, ["replace:doc-old"])

    def test_feishu_doc_publish_creates_when_no_existing_document(self) -> None:
        client = FakeFeishuDocClient()
        document = client.publish_markdown("报告", "# 内容")
        self.assertEqual(document.document_id, "doc-new")
        self.assertEqual(client.calls, ["create:报告", "append:doc-new"])

    def test_feishu_doc_publish_creates_wiki_document_when_parent_token_exists(self) -> None:
        client = FakeFeishuDocClient()
        document = client.publish_markdown("报告", "# 内容", wiki_parent_token="wiki-parent")
        self.assertEqual(document.document_id, "doc-wiki")
        self.assertEqual(document.url, "https://tenant.feishu.cn/wiki/wiki-child")
        self.assertEqual(document.wiki_node_token, "wiki-child")
        self.assertEqual(client.calls, ["create_wiki:wiki-parent:报告", "append:doc-wiki"])

    def test_markdown_report_uses_heilmeier_structure(self) -> None:
        rec = PaperRecommendation(
            paper=make_paper("2605.00001", "Efficient Detector"),
            summary=PaperSummary(primary_task="检测", recommendation_level="高优先级", recommendation_reason="值得小试"),
            recommended_on="2026-05-29",
        )
        content = render_report_markdown(
            rec,
            {
                "executive_summary": "PAGE 1: 这是一篇检测论文。",
                "target_reader": "检测算法工程师",
                "problem": "PAGE 1: 解决端侧检测问题。",
                "method_overview": "PAGE 2: 使用轻量检测器。",
                "experiments": "PAGE 3: COCO AP 提升。",
                "business_value": "可做端侧检测小试。",
                "risks": ["PAGE 4: 需验证延迟。"],
                "evidence": ["PAGE 1: 摘要。"],
                "report_basis": "PDF全文与摘要",
            },
            generated_at="2026-06-01T00:00:00+00:00",
            report_basis="PDF全文与摘要",
        )
        self.assertIn("## Heilmeier 七问精读", content)
        self.assertIn("**论文事实**", content)
        self.assertIn("## 图解材料", content)

    def test_paper_craft_prompt_requests_html_and_html_image_paths(self) -> None:
        rec = PaperRecommendation(
            paper=make_paper("2605.00001", "Efficient Detector"),
            summary=PaperSummary(primary_task="检测", recommendation_reason="值得小试"),
            recommended_on="2026-05-29",
        )
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp)
            (skill_dir / "styles").mkdir()
            (skill_dir / "SKILL.md").write_text("Paper Craft Skill", encoding="utf-8")
            (skill_dir / "styles" / "academic.md").write_text("Academic style", encoding="utf-8")
            prompt = build_paper_craft_html_prompt(
                recommendation=rec,
                paper_material={
                    "basis": "摘要与评审信息",
                    "text": "PAGE 1: abstract",
                    "figures": [{"path": "assets/2605.00001/figure-01.png"}],
                },
                reports_config={"paper_craft_skill_path": str(skill_dir), "paper_craft_style": "academic"},
                style="academic",
            )
        self.assertIn("paper-craft-skills / paper-analyzer", prompt)
        self.assertIn("完整 HTML", prompt)
        self.assertIn("不要输出 JSON", prompt)
        self.assertIn('"html_path": "../../assets/2605.00001/figure-01.png"', prompt)

    def test_clean_paper_craft_html_strips_fenced_html(self) -> None:
        content = clean_paper_craft_html(
            """```html
<!doctype html>
<html><head><title>报告</title></head><body>ok</body></html>
```"""
        )
        self.assertTrue(content.lstrip().lower().startswith("<!doctype html>"))
        self.assertNotIn("```", content)

    def test_generate_reports_writes_per_paper_html_and_index(self) -> None:
        rec = PaperRecommendation(
            paper=make_paper("2605.00001", "Efficient Detector"),
            summary=PaperSummary(
                one_sentence="一句话结论",
                recommendation_reason="直接命中检测主任务。",
                business_value="可用于端侧检测验证。",
                risks="需要复现。",
                primary_task="检测",
                recommendation_level="高优先级",
            ),
            recommended_on="2026-05-29",
        )
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            generate_reports(
                [rec],
                {
                    "codex": {"enabled": False},
                    "reports": {
                        "enabled": True,
                        "output_dir": "web",
                        "state_path": "web/data/state.json",
                        "base_url": "https://example.com/papers",
                        "read_pdf": False,
                    },
                },
                repo_root=repo_root,
            )
            report_file = repo_root / "web" / "reports" / "2605.00001" / "index.html"
            self.assertTrue(report_file.exists())
            self.assertIn("论文阅读报告", report_file.read_text(encoding="utf-8"))
            self.assertTrue((repo_root / "web" / "index.html").exists())
            self.assertEqual(rec.report_url, "https://example.com/papers/reports/2605.00001/index.html")
            self.assertEqual(rec.report_status, "已生成")

    def test_generate_reports_hides_remote_url_when_git_publish_disabled(self) -> None:
        rec = PaperRecommendation(
            paper=make_paper("2605.00001", "Efficient Detector"),
            summary=PaperSummary(one_sentence="一句话结论", recommendation_reason="检测", risks="需要复现。"),
            recommended_on="2026-05-29",
        )
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            generate_reports(
                [rec],
                {
                    "codex": {"enabled": False},
                    "reports": {
                        "enabled": True,
                        "output_dir": "reports",
                        "state_path": "reports/data/state.json",
                        "base_url": "https://pages.example.com/paper-share",
                        "git_push_before_send": False,
                        "read_pdf": False,
                    },
                },
                repo_root=repo_root,
            )
            self.assertTrue((repo_root / "reports" / "reports" / "2605.00001" / "index.html").exists())
            self.assertEqual(rec.report_url, "")

    def test_generate_reports_writes_paper_craft_html(self) -> None:
        rec = PaperRecommendation(
            paper=make_paper("2605.00001", "Efficient Detector"),
            summary=PaperSummary(
                one_sentence="一句话结论",
                recommendation_reason="直接命中检测主任务。",
                business_value="可用于端侧检测验证。",
                risks="需要复现。",
                primary_task="检测",
                recommendation_level="高优先级",
            ),
            recommended_on="2026-05-29",
        )
        html = """```html
<!doctype html>
<html><head><title>Paper Craft</title></head><body><h1>Paper Craft HTML</h1><p>PAGE 1</p></body></html>
```"""
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            with mock.patch("tools.paper_recommender.report_generator.shutil.which", return_value="/usr/bin/codex"), mock.patch(
                "tools.paper_recommender.paper_craft_report._run_codex",
                return_value=html,
            ):
                generate_reports(
                    [rec],
                    {
                        "codex": {"enabled": True, "command": "codex"},
                        "reports": {
                            "enabled": True,
                            "generator": "paper_craft",
                            "format": "html",
                            "publisher": "file",
                            "output_dir": "reports",
                            "state_path": "reports/data/state.json",
                            "base_url": "https://pages.example.com/paper-share",
                            "read_pdf": False,
                        },
                    },
                    repo_root=repo_root,
                )
            report_file = repo_root / "reports" / "reports" / "2605.00001" / "index.html"
            self.assertTrue(report_file.exists())
            content = report_file.read_text(encoding="utf-8")
            self.assertIn("Paper Craft HTML", content)
            self.assertNotIn("```", content)
            self.assertTrue((repo_root / "reports" / "index.html").exists())
            self.assertEqual(rec.report_url, "https://pages.example.com/paper-share/reports/2605.00001/index.html")
            self.assertEqual(rec.report_status, "已生成")
            state = json.loads((repo_root / "reports" / "data" / "state.json").read_text(encoding="utf-8"))
            stored = state["generated_reports"]["2605.00001"]
            self.assertEqual(stored["format"], "html")
            self.assertEqual(stored["doc_url"], "")

    def test_generate_reports_falls_back_when_paper_craft_output_is_not_html(self) -> None:
        rec = PaperRecommendation(
            paper=make_paper("2605.00001", "Efficient Detector"),
            summary=PaperSummary(
                one_sentence="一句话结论",
                recommendation_reason="直接命中检测主任务。",
                business_value="可用于端侧检测验证。",
                risks="需要复现。",
                primary_task="检测",
                recommendation_level="高优先级",
            ),
            recommended_on="2026-05-29",
        )
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            with mock.patch("tools.paper_recommender.report_generator.shutil.which", return_value="/usr/bin/codex"), mock.patch(
                "tools.paper_recommender.paper_craft_report._run_codex",
                return_value="not html",
            ):
                generate_reports(
                    [rec],
                    {
                        "codex": {"enabled": True, "command": "codex"},
                        "reports": {
                            "enabled": True,
                            "generator": "paper_craft",
                            "format": "html",
                            "publisher": "file",
                            "output_dir": "reports",
                            "state_path": "reports/data/state.json",
                            "base_url": "https://pages.example.com/paper-share",
                            "read_pdf": False,
                        },
                    },
                    repo_root=repo_root,
                )
            report_file = repo_root / "reports" / "reports" / "2605.00001" / "index.html"
            self.assertTrue(report_file.exists())
            self.assertIn("论文阅读报告", report_file.read_text(encoding="utf-8"))
            self.assertEqual(rec.report_status, "需复核")
            state = json.loads((repo_root / "reports" / "data" / "state.json").read_text(encoding="utf-8"))
            stored = state["generated_reports"]["2605.00001"]
            self.assertEqual(stored["quality_status"], "failed")
            self.assertIn("paper-craft HTML generation failed", "\n".join(stored["quality_issues"]))

    def test_file_publisher_does_not_build_feishu_doc_client(self) -> None:
        rec = PaperRecommendation(
            paper=make_paper("2605.00001", "Efficient Detector"),
            summary=PaperSummary(one_sentence="一句话结论", recommendation_reason="值得观察", risks="需复现"),
            recommended_on="2026-05-29",
        )
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            with mock.patch("tools.paper_recommender.report_generator.build_feishu_doc_client") as build_doc:
                generate_reports(
                    [rec],
                    {
                        "codex": {"enabled": False},
                        "reports": {
                            "enabled": True,
                            "publisher": "file",
                            "format": "html",
                            "output_dir": "reports",
                            "state_path": "reports/data/state.json",
                            "base_url": "https://pages.example.com/paper-share",
                            "read_pdf": False,
                        },
                    },
                    repo_root=repo_root,
                )
            build_doc.assert_not_called()

    def test_generate_reports_writes_markdown_archive(self) -> None:
        rec = PaperRecommendation(
            paper=make_paper("2605.00001", "Efficient Detector"),
            summary=PaperSummary(
                one_sentence="一句话结论",
                recommendation_reason="直接命中检测主任务。",
                business_value="可用于端侧检测验证。",
                risks="需要复现。",
                primary_task="检测",
                recommendation_level="高优先级",
            ),
            recommended_on="2026-05-29",
        )
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            generate_reports(
                [rec],
                {
                    "codex": {"enabled": False},
                    "reports": {
                        "enabled": True,
                        "format": "markdown",
                        "output_dir": "reports",
                        "state_path": "reports/data/state.json",
                        "base_url": "https://git.example.com/team/paper-share/-/blob/master/reports",
                        "read_pdf": False,
                    },
                },
                repo_root=repo_root,
            )
            report_file = repo_root / "reports" / "papers" / "2605.00001.md"
            self.assertTrue(report_file.exists())
            content = report_file.read_text(encoding="utf-8")
            self.assertIn("# Efficient Detector", content)
            self.assertIn("## 图解材料", content)
            self.assertTrue((repo_root / "reports" / "README.md").exists())
            self.assertEqual(
                rec.report_url,
                "https://git.example.com/team/paper-share/-/blob/master/reports/papers/2605.00001.md",
            )
            self.assertEqual(rec.report_status, "已生成")

    def test_generate_reports_preserves_existing_doc_id_when_doc_client_unavailable(self) -> None:
        rec = PaperRecommendation(
            paper=make_paper("2605.00001", "Efficient Detector"),
            summary=PaperSummary(
                one_sentence="一句话结论",
                recommendation_reason="直接命中检测主任务。",
                business_value="可用于端侧检测验证。",
                risks="需要复现。",
                primary_task="检测",
                recommendation_level="高优先级",
            ),
            recommended_on="2026-05-29",
        )
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            state_path = repo_root / "reports/data/state.json"
            state_path.parent.mkdir(parents=True)
            state_path.write_text(
                json.dumps(
                    {
                        "processed_arxiv_ids": ["2605.00001"],
                        "generated_reports": {
                            "2605.00001": {
                                "path": "papers/2605.00001.md",
                                "url": "https://tenant.feishu.cn/docx/doc-old",
                                "doc_id": "doc-old",
                                "doc_url": "https://tenant.feishu.cn/docx/doc-old",
                            }
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            generate_reports(
                [rec],
                {
                    "codex": {"enabled": False},
                    "reports": {
                        "enabled": True,
                        "publisher": "feishu_doc",
                        "archive_markdown": True,
                        "force_regenerate": True,
                        "output_dir": "reports",
                        "state_path": "reports/data/state.json",
                        "base_url": "https://git.example.com/team/paper-share/-/blob/master/reports",
                        "read_pdf": False,
                    },
                },
                repo_root=repo_root,
            )
            state = json.loads(state_path.read_text(encoding="utf-8"))
            stored = state["generated_reports"]["2605.00001"]
            self.assertEqual(stored["doc_id"], "doc-old")
            self.assertEqual(stored["doc_url"], "https://tenant.feishu.cn/docx/doc-old")

    def test_default_config_publishes_reports_to_github_pages_without_gitlab_push(self) -> None:
        config = load_config(ROOT / "tools" / "paper_recommender" / "config.yaml")
        reports = config["reports"]
        self.assertEqual(reports["generator"], "paper_craft")
        self.assertEqual(reports["publisher"], "file")
        self.assertEqual(reports["format"], "html")
        self.assertEqual(reports["paper_craft_style"], "academic")
        self.assertEqual(reports["paper_craft_skill_path"], "/home/wjw/.codex/skills/paper-analyzer")
        self.assertEqual(reports["base_url"], "https://jevin0924.github.io/paper-share")
        self.assertEqual(reports["publish_target"], "github_pages")
        self.assertIs(reports["publish_before_send"], True)
        self.assertIs(reports["publish_on_no_send"], True)
        self.assertEqual(reports["github_repository"], "Jevin0924/paper-share")
        self.assertEqual(reports["github_branch"], "main")
        self.assertEqual(reports["github_bundle_path"], "reports_bundle.tgz.b64")
        self.assertEqual(reports["github_pages_include_assets"], "none")
        self.assertNotIn("git_remote", reports)
        self.assertNotIn("git_branch", reports)

    def test_github_pages_workflow_restores_report_bundle(self) -> None:
        workflow = (ROOT / ".github" / "workflows" / "pages.yml").read_text(encoding="utf-8")
        self.assertIn("Deploy GitHub Pages", workflow)
        self.assertIn("reports_bundle.tgz.b64", workflow)
        self.assertIn("actions/deploy-pages", workflow)

    def test_publish_reports_uses_github_pages_by_default(self) -> None:
        rec = PaperRecommendation(
            paper=make_paper("2605.00001", "Efficient Detector"),
            summary=PaperSummary(one_sentence="一句话结论", recommendation_reason="检测", risks="需要复现。"),
            recommended_on="2026-05-29",
        )
        rec.report_url = "https://pages.example.com/reports/2605.00001/index.html"
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            with mock.patch("tools.paper_recommender.run_daily.publish_reports_to_github_pages") as publish_pages, mock.patch(
                "tools.paper_recommender.run_daily.push_reports_to_git_remote"
            ) as push_git:
                publish_pages.return_value = mock.Mock(commit_sha="abc123", workflow_run_url="https://github.com/run", verified_urls=[rec.report_url])
                publish_reports(
                    {
                        "reports": {
                            "enabled": True,
                            "github_repository": "Jevin0924/paper-share",
                            "github_branch": "main",
                        }
                    },
                    [rec],
                    recommended_on="2026-05-29",
                    repo_root=repo_root,
                )
        publish_pages.assert_called_once()
        push_git.assert_not_called()


def make_paper(arxiv_id: str, title: str, abstract: str = "abstract") -> PaperCandidate:
    return PaperCandidate(
        title=title,
        abstract=abstract,
        authors=["Alice"],
        published="2026-05-28T00:00:00+00:00",
        updated="2026-05-28T00:00:00+00:00",
        arxiv_id=arxiv_id,
        arxiv_url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        categories=["cs.CV"],
        primary_category="cs.CV",
    )


class FakeFeishuDocClient(FeishuDocClient):
    def __init__(self) -> None:
        super().__init__(token_client=None, doc_base_url="https://tenant.feishu.cn")  # type: ignore[arg-type]
        self.calls: list[str] = []

    def create_document(self, title: str, folder_token: str = "") -> FeishuDocument:
        self.calls.append(f"create:{title}")
        return FeishuDocument(document_id="doc-new", title=title, url="https://tenant.feishu.cn/docx/doc-new")

    def create_wiki_document(self, title: str, parent_wiki_token: str) -> FeishuDocument:
        self.calls.append(f"create_wiki:{parent_wiki_token}:{title}")
        return FeishuDocument(
            document_id="doc-wiki",
            title=title,
            url="https://tenant.feishu.cn/wiki/wiki-child",
            wiki_node_token="wiki-child",
            wiki_space_id="space-1",
        )

    def preflight_markdown_write(self) -> None:
        return None

    def replace_markdown(
        self,
        document_id: str,
        content: str,
        image_paths: list[Path] | None = None,
        document_url: str = "",
        wiki_node_token: str = "",
        wiki_space_id: str = "",
    ) -> FeishuDocument:
        self.calls.append(f"replace:{document_id}")
        return FeishuDocument(
            document_id=document_id,
            title="",
            url=document_url or f"https://tenant.feishu.cn/docx/{document_id}",
            wiki_node_token=wiki_node_token,
            wiki_space_id=wiki_space_id,
        )

    def append_converted_markdown(self, document_id: str, content: str) -> None:
        self.calls.append(f"append:{document_id}")


if __name__ == "__main__":
    unittest.main()
