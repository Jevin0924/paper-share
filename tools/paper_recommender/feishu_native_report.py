from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Any

from .feishu_doc import FeishuDocClient, FeishuDocument, _document_url, _wiki_url
from .models import PaperRecommendation
from .report_schema import ReportPayload, report_payload_from_dict


_PAGE_REF_RE = re.compile(r"\bPAGE\s+(\d+)\b", flags=re.IGNORECASE)


def publish_feishu_native_report(
    client: FeishuDocClient,
    recommendation: PaperRecommendation,
    report: dict[str, Any],
    generated_at: str,
    report_basis: str,
    output_dir: Path,
    title: str,
    folder_token: str = "",
    document_id: str = "",
    document_url: str = "",
    wiki_parent_token: str = "",
    wiki_node_token: str = "",
    wiki_space_id: str = "",
    max_images: int = 4,
) -> FeishuDocument:
    """Publish the report as native Feishu Docx blocks instead of one Markdown blob."""
    document = _prepare_document(
        client=client,
        title=title,
        folder_token=folder_token,
        document_id=document_id,
        document_url=document_url,
        wiki_parent_token=wiki_parent_token,
        wiki_node_token=wiki_node_token,
        wiki_space_id=wiki_space_id,
    )
    writer = _NativeReportWriter(
        client=client,
        document=document,
        recommendation=recommendation,
        report=report,
        generated_at=generated_at,
        report_basis=report_basis,
        output_dir=output_dir,
        max_images=max_images,
    )
    writer.write()
    return document


class _NativeReportWriter:
    def __init__(
        self,
        client: FeishuDocClient,
        document: FeishuDocument,
        recommendation: PaperRecommendation,
        report: dict[str, Any],
        generated_at: str,
        report_basis: str,
        output_dir: Path,
        max_images: int,
    ) -> None:
        self.client = client
        self.document = document
        self.recommendation = recommendation
        self.report = report
        self.payload: ReportPayload = report_payload_from_dict(report)
        self.generated_at = generated_at
        self.report_basis = report_basis
        self.output_dir = output_dir
        self.max_images = max_images
        self.blocks: list[dict[str, Any]] = []

    @property
    def document_id(self) -> str:
        return self.document.document_id

    def write(self) -> None:
        paper = self.recommendation.paper
        summary = self.recommendation.summary
        self.h1(paper.title)
        self.p("论文精读报告。正文按结论、证据图、方法、实验、边界和落地建议组织，页码以 p.1 形式标注。")

        self.h2("基本信息")
        self.flush()
        self.table(
            ["字段", "内容"],
            [
                ["arXiv ID", paper.arxiv_id],
                ["发布时间", paper.published[:10]],
                ["作者", _authors(paper.authors)],
                ["类别", ", ".join(paper.categories)],
                ["方向", summary.primary_task or "感知视觉"],
                ["推荐等级", summary.recommendation_level or "-"],
                ["推荐分", f"{paper.score:.1f} / 30"],
                ["报告依据", self.report_basis],
                ["生成时间", self.generated_at],
            ],
        )
        self.p(f"查看论文：{paper.arxiv_url}")
        self.p(f"下载 PDF：{paper.pdf_url}")
        if paper.code_url:
            self.p(f"查看代码：{paper.code_url}")

        self.h2("一页结论")
        self.p(_clean_page_refs(str(self.report.get("executive_summary") or "暂无。")))
        self.p(f"适合读者：{_clean_page_refs(str(self.report.get('target_reader') or '需结合业务方向判断。'))}")
        self.p(
            "建议动作："
            + _clean_page_refs(str(self.report.get("business_action") or "先做小规模复现，再判断是否进入技术储备。"))
        )

        figures = _figure_notes(self.report)
        if figures:
            self.h2("关键证据图")
            self.p("本节不把图片作为装饰使用；每张图都说明它回答的问题、应该看什么，以及它支撑报告中的哪类判断。")
            self.flush()
            self.figure(figures[0], index=1)

        self.h2("方法流程图")
        self.p("这张图用于把 LWIQ 的 rank 决策链路从论文文字拆成工程步骤，帮助判断复现时需要实现哪些模块。")
        self.flush()
        steps = _method_steps(self.report)
        flow_image = _render_flow_image(steps, self.output_dir / "assets" / paper.arxiv_id / "method-flow.png")
        if flow_image:
            self.client.append_image(self.document_id, flow_image)
            self.p("读图要点：LWIQ 的核心不是训练一个新主干，而是在 TT 分解后用 proxy classifier 和 weight imprinting 低成本估计层重要性，再把稳定的 TTL score 映射为每层 rank。")
        else:
            self.table(["步骤", "动作"], [[str(index), step] for index, step in enumerate(steps, start=1)])

        self.h2("Heilmeier 七问精读")
        for section in self.payload.sections:
            self.h3(section.title)
            self.labelled_items("论文事实", section.paper_claims)
            self.labelled_items("业务判断", section.analysis)
            self.labelled_items("证据锚点", section.evidence)

        self.h2("结构化实验表")
        tables = _experiment_tables(self.report.get("experiment_tables"))
        if tables:
            self.flush()
            for table in tables:
                self.h3(str(table.get("title") or table.get("name") or "实验表"))
                note = str(table.get("note") or "").strip()
                if note:
                    self.p(note)
                self.flush()
                self.table(_string_list(table.get("columns")), _table_rows(table))
                takeaway = _table_takeaway(str(table.get("title") or ""))
                if takeaway:
                    self.p(takeaway)
        else:
            self.p("暂无结构化实验表。")

        if len(figures) > 1:
            self.h2("图表解读")
            self.p("以下图片用于补足方法机制和稳定性证据。每张图后给出用途和读图要点，避免图片与正文脱节。")
            self.flush()
            for index, figure in enumerate(figures[1 : self.max_images], start=2):
                self.figure(figure, index=index)

        self.h2("创新点")
        self.items(_string_list(self.report.get("innovation_points")))

        self.h2("业务价值")
        self.p(_clean_page_refs(str(self.report.get("business_value") or "暂无。")))

        self.h2("落地建议")
        self.items(_string_list(self.report.get("implementation_plan")))

        self.h2("风险限制")
        self.items(_string_list(self.report.get("risks")))

        self.h2("待确认问题")
        self.items(_string_list(self.report.get("open_questions")))

        self.h2("证据索引")
        self.items(_string_list(self.report.get("evidence")))
        self.flush()

    def h1(self, text: str) -> None:
        self.blocks.append(_block(3, _clean_page_refs(text)))

    def h2(self, text: str) -> None:
        self.blocks.append(_block(4, _clean_page_refs(text)))

    def h3(self, text: str) -> None:
        self.blocks.append(_block(5, _clean_page_refs(text)))

    def p(self, text: str) -> None:
        for chunk in _chunks(_clean_page_refs(text)):
            self.blocks.append(_block(2, chunk))

    def items(self, values: list[str]) -> None:
        if not values:
            self.p("暂无。")
            return
        for value in values:
            self.p(f"- {value}")

    def labelled_items(self, label: str, values: list[str]) -> None:
        if not values:
            return
        self.p(f"{label}：")
        for value in values:
            self.p(f"- {value}")

    def flush(self) -> None:
        if not self.blocks:
            return
        self.client.append_blocks(self.document_id, self.blocks)
        self.blocks = []

    def table(self, columns: list[str], rows: list[list[str]]) -> None:
        columns = [str(column or "").strip() for column in columns if str(column or "").strip()]
        if not columns or not rows:
            return
        self.client.append_table(self.document_id, columns, rows)

    def figure(self, figure: dict[str, Any], index: int) -> None:
        image_path = _resolve_image_path(figure, self.output_dir)
        caption = str(figure.get("caption") or f"图 {index}").strip()
        explanation = str(figure.get("explanation") or figure.get("purpose") or "该图来自论文 PDF，需结合正文核对。").strip()
        takeaway = str(figure.get("takeaway") or _infer_figure_takeaway(caption, explanation)).strip()
        self.h3(f"图 {index}：{caption}")
        self.p(f"用途：{explanation}")
        self.flush()
        if image_path:
            self.client.append_image(self.document_id, image_path)
        else:
            self.p("图片文件未找到，本段仅保留图像说明。")
        if takeaway:
            self.p(f"读图要点：{takeaway}")
        page = figure.get("page")
        if page:
            self.p(f"证据锚点：p.{page}")


def _prepare_document(
    client: FeishuDocClient,
    title: str,
    folder_token: str,
    document_id: str,
    document_url: str,
    wiki_parent_token: str,
    wiki_node_token: str,
    wiki_space_id: str,
) -> FeishuDocument:
    if document_id:
        document = FeishuDocument(
            document_id=document_id,
            title=title,
            url=document_url
            or (_wiki_url(client.doc_base_url, wiki_node_token) if wiki_node_token else _document_url(client.doc_base_url, document_id)),
            wiki_node_token=wiki_node_token,
            wiki_space_id=wiki_space_id,
        )
        try:
            client.clear_document(document_id)
            return document
        except RuntimeError as exc:
            if not _looks_not_found(str(exc)):
                raise
    if wiki_parent_token:
        return client.create_wiki_document(title=title, parent_wiki_token=wiki_parent_token)
    return client.create_document(title=title, folder_token=folder_token)


def _block(block_type: int, content: str) -> dict[str, Any]:
    key = {2: "text", 3: "heading1", 4: "heading2", 5: "heading3"}.get(block_type, "text")
    return {"block_type": block_type, key: {"elements": [{"text_run": {"content": content.strip() or "暂无。"}}]}}


def _clean_page_refs(value: str) -> str:
    text = _PAGE_REF_RE.sub(lambda match: f"p.{match.group(1)}", str(value or ""))
    return re.sub(r"\s+", " ", text).strip()


def _chunks(value: str, limit: int = 1600) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return ["暂无。"]
    return [text[start : start + limit] for start in range(0, len(text), limit)]


def _authors(authors: list[str]) -> str:
    if not authors:
        return "作者未知"
    value = ", ".join(authors[:12])
    if len(authors) > 12:
        value += f", et al. (+{len(authors) - 12})"
    return value


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_clean_page_refs(str(item)) for item in value if str(item).strip()]
    if isinstance(value, str):
        parts = [item.strip(" -\t") for item in re.split(r"[\n；;]+", value) if item.strip(" -\t")]
        return [_clean_page_refs(item) for item in (parts if len(parts) > 1 else [value.strip()]) if item]
    return []


def _experiment_tables(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _table_rows(table: dict[str, Any]) -> list[list[str]]:
    columns = _string_list(table.get("columns"))
    raw_rows = table.get("rows") if isinstance(table.get("rows"), list) else []
    rows: list[list[str]] = []
    for row in raw_rows:
        if isinstance(row, dict):
            rows.append([_clean_page_refs(str(row.get(column) or "")) for column in columns])
        elif isinstance(row, list):
            rows.append([_clean_page_refs(str(item)) for item in row])
    return rows


def _figure_notes(report: dict[str, Any]) -> list[dict[str, Any]]:
    raw = report.get("figure_notes")
    if not isinstance(raw, list):
        raw = report.get("figures")
    notes: list[dict[str, Any]] = []
    if not isinstance(raw, list):
        return notes
    for item in raw:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or item.get("markdown_path") or "").strip()
        if not path:
            continue
        note = dict(item)
        note["path"] = path
        notes.append(note)
    return notes


def _resolve_image_path(figure: dict[str, Any], output_dir: Path) -> Path | None:
    raw_path = str(figure.get("path") or figure.get("markdown_path") or "").strip()
    if not raw_path:
        return None
    normalized = raw_path.removeprefix("../").lstrip("/")
    image_path = output_dir / normalized
    return image_path if image_path.exists() and image_path.is_file() else None


def _infer_figure_takeaway(caption: str, explanation: str) -> str:
    text = f"{caption} {explanation}".lower()
    if "rank" in text or "ttl" in text:
        return "重点看层重要性如何被转换成 rank 决策，以及该决策是否有稳定性依据。"
    if "tensor" in text or "tt" in text or "convolution" in text:
        return "重点看普通卷积核被重写为 TT 表达后，压缩收益来自参数结构化分解而非简单剪枝。"
    return "重点看图片是否直接支撑正文结论；不能支撑的图片不应单独作为证据。"


def _method_steps(report: dict[str, Any]) -> list[str]:
    mermaid = str(report.get("method_mermaid") or report.get("mermaid") or "").strip()
    labels: dict[str, str] = {}
    order: list[str] = []
    for line in mermaid.splitlines():
        for node_id, label in re.findall(r"([A-Za-z0-9_]+)\[([^\]]+)\]", line):
            labels[node_id] = _translate_step(label)
        match = re.search(r"([A-Za-z0-9_]+)(?:\[[^\]]+\])?\s*-->\s*([A-Za-z0-9_]+)(?:\[[^\]]+\])?", line)
        if not match:
            continue
        for node_id in (match.group(1), match.group(2)):
            if node_id not in order:
                order.append(node_id)
    steps = [labels[node_id] for node_id in order if labels.get(node_id)]
    if steps:
        return steps
    return [
        "输入待压缩 CNN",
        "对卷积层做 TT 分解",
        "用 proxy classifier 估计每层重要性",
        "检查 TTL score 稳定性",
        "按预算分配 TT rank 并 fine-tune",
    ]


def _translate_step(label: str) -> str:
    mapping = {
        "Input CNN": "输入 CNN 模型",
        "Apply TT decomposition to convolutional layers": "对卷积层做 TT 分解",
        "Add proxy classifier per TT layer": "为每个 TT 层挂载代理分类器",
        "Adaptive average pooling and weight imprinting": "池化并用 Weight Imprinting 生成分类器权重",
        "Compute layer accuracy and normalized TTL score": "计算层准确率与归一化 TTL score",
        "Check TTL score stability across epoch groups": "检查多组 epoch 的 TTL score 稳定性",
        "Stop when most layers are stable": "多数层稳定后提前停止",
        "Use scaling factor gamma to assign ranks": "用缩放因子 gamma 分配 rank",
        "Update TT ranks and fine-tune compressed model": "更新 TT rank 并 fine-tune 压缩模型",
    }
    return mapping.get(label.strip(), label.strip())


def _render_flow_image(steps: list[str], image_path: Path) -> Path | None:
    if not steps:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None
    image_path.parent.mkdir(parents=True, exist_ok=True)
    width = 1400
    box_height = 86
    gap = 30
    margin_x = 90
    margin_y = 60
    height = margin_y * 2 + len(steps) * box_height + (len(steps) - 1) * gap
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = _load_font(ImageFont, size=30, bold=False)
    small = _load_font(ImageFont, size=24, bold=False)
    accent = (28, 97, 167)
    fill = (237, 247, 255)
    border = (59, 130, 196)
    text_color = (31, 41, 55)
    for index, step in enumerate(steps, start=1):
        top = margin_y + (index - 1) * (box_height + gap)
        left = margin_x
        right = width - margin_x
        bottom = top + box_height
        draw.rounded_rectangle((left, top, right, bottom), radius=18, fill=fill, outline=border, width=3)
        draw.ellipse((left + 22, top + 20, left + 68, top + 66), fill=accent)
        number = str(index)
        number_box = draw.textbbox((0, 0), number, font=small)
        draw.text(
            (left + 45 - (number_box[2] - number_box[0]) / 2, top + 43 - (number_box[3] - number_box[1]) / 2),
            number,
            fill="white",
            font=small,
        )
        lines = textwrap.wrap(step, width=34)
        line_height = 34
        text_top = top + (box_height - len(lines) * line_height) / 2 - 2
        for line_no, line in enumerate(lines):
            draw.text((left + 92, text_top + line_no * line_height), line, fill=text_color, font=font)
        if index < len(steps):
            x = width // 2
            y1 = bottom + 5
            y2 = bottom + gap - 5
            draw.line((x, y1, x, y2), fill=accent, width=4)
            draw.polygon(((x, y2 + 12), (x - 10, y2 - 4), (x + 10, y2 - 4)), fill=accent)
    image.save(image_path)
    return image_path


def _load_font(image_font: Any, size: int, bold: bool) -> Any:
    candidates = [
        "/mnt/c/Windows/Fonts/msyhbd.ttc" if bold else "/mnt/c/Windows/Fonts/msyh.ttc",
        "/mnt/c/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return image_font.truetype(path, size=size)
    return image_font.load_default()


def _table_takeaway(title: str) -> str:
    if "搜索时间" in title:
        return "表格解读：这张表只能证明作者报告的 rank 搜索开销更低；由于 PARS 与 LWIQ 使用不同硬件，不能直接当作严格同硬件加速比。"
    if "CIFAR" in title:
        return "表格解读：结果覆盖小图分类基准，能支持方法在作者设定上的有效性，但还不能外推到 ImageNet、检测或端侧真实延迟。"
    return ""


def _looks_not_found(message: str) -> bool:
    text = message.lower()
    return "not found" in text or "1770002" in text or "resource deleted" in text or "1770003" in text
