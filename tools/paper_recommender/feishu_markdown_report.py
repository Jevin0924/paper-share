from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .feishu_doc import FeishuDocClient, FeishuDocument, _document_url, _wiki_url


def publish_feishu_markdown_report(
    client: FeishuDocClient,
    title: str,
    markdown: str,
    markdown_base_dir: Path,
    folder_token: str = "",
    document_id: str = "",
    document_url: str = "",
    wiki_parent_token: str = "",
    wiki_node_token: str = "",
    wiki_space_id: str = "",
) -> FeishuDocument:
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
    writer = _MarkdownDocWriter(client=client, document_id=document.document_id, base_dir=markdown_base_dir)
    writer.write(markdown)
    return document


class _MarkdownDocWriter:
    def __init__(self, client: FeishuDocClient, document_id: str, base_dir: Path) -> None:
        self.client = client
        self.document_id = document_id
        self.base_dir = base_dir
        self.blocks: list[dict[str, Any]] = []

    def write(self, markdown: str) -> None:
        lines = markdown.splitlines()
        index = 0
        paragraph: list[str] = []
        while index < len(lines):
            line = lines[index].rstrip()
            stripped = line.strip()
            if not stripped:
                self._flush_paragraph(paragraph)
                paragraph = []
                index += 1
                continue
            if stripped.startswith("```"):
                self._flush_paragraph(paragraph)
                paragraph = []
                code_lines, index = self._consume_code(lines, index)
                self._text("代码：\n" + "\n".join(code_lines))
                continue
            if _is_table_start(lines, index):
                self._flush_paragraph(paragraph)
                paragraph = []
                table_lines, index = _consume_table(lines, index)
                columns, rows = _parse_table(table_lines)
                self._flush_blocks()
                if columns and rows:
                    self.client.append_table(self.document_id, columns, rows)
                continue
            image = _parse_image(stripped)
            if image:
                self._flush_paragraph(paragraph)
                paragraph = []
                self._flush_blocks()
                alt, raw_path = image
                image_path = _resolve_image_path(raw_path, self.base_dir)
                if alt:
                    self._text(alt)
                    self._flush_blocks()
                if image_path:
                    self.client.append_image(self.document_id, image_path)
                index += 1
                continue
            heading_level = _heading_level(stripped)
            if heading_level:
                self._flush_paragraph(paragraph)
                paragraph = []
                self._heading(stripped.lstrip("#").strip(), heading_level)
                index += 1
                continue
            paragraph.append(_strip_inline_markdown(stripped))
            index += 1
        self._flush_paragraph(paragraph)
        self._flush_blocks()

    def _consume_code(self, lines: list[str], start: int) -> tuple[list[str], int]:
        code: list[str] = []
        index = start + 1
        while index < len(lines):
            if lines[index].strip().startswith("```"):
                return code, index + 1
            code.append(lines[index].rstrip())
            index += 1
        return code, index

    def _heading(self, text: str, level: int) -> None:
        block_type = 3 if level <= 1 else 4 if level == 2 else 5
        self.blocks.append(_block(block_type, _strip_inline_markdown(text)))

    def _text(self, text: str) -> None:
        chunks = _chunks(text)
        self.blocks.extend(_block(2, chunk) for chunk in chunks)

    def _flush_paragraph(self, paragraph: list[str]) -> None:
        text = " ".join(item for item in paragraph if item).strip()
        if text:
            self._text(text)

    def _flush_blocks(self) -> None:
        if not self.blocks:
            return
        self.client.append_blocks(self.document_id, self.blocks)
        self.blocks = []


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


def _heading_level(value: str) -> int:
    if not value.startswith("#"):
        return 0
    return min(len(value) - len(value.lstrip("#")), 6)


def _is_table_start(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    return _is_table_row(lines[index]) and _is_table_separator(lines[index + 1])


def _consume_table(lines: list[str], start: int) -> tuple[list[str], int]:
    table_lines: list[str] = []
    index = start
    while index < len(lines) and _is_table_row(lines[index]):
        table_lines.append(lines[index])
        index += 1
    return table_lines, index


def _parse_table(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    if len(lines) < 2:
        return [], []
    columns = _split_table_row(lines[0])
    rows = [_split_table_row(line) for line in lines[2:] if not _is_table_separator(line)]
    return columns, rows


def _split_table_row(line: str) -> list[str]:
    return [_strip_inline_markdown(cell.strip()) for cell in line.strip().strip("|").split("|")]


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def _is_table_separator(line: str) -> bool:
    text = line.strip().replace("|", "").replace(" ", "")
    return bool(text) and set(text) <= {"-", ":"}


def _parse_image(line: str) -> tuple[str, str] | None:
    match = re.match(r"!\[([^\]]*)\]\((?:<([^>]+)>|([^)]+))\)", line)
    if not match:
        return None
    return match.group(1).strip(), (match.group(2) or match.group(3) or "").strip()


def _resolve_image_path(raw_path: str, base_dir: Path) -> Path | None:
    if not raw_path or raw_path.startswith(("http://", "https://")):
        return None
    image_path = (base_dir / raw_path).resolve()
    return image_path if image_path.exists() and image_path.is_file() else None


def _strip_inline_markdown(value: str) -> str:
    text = str(value or "").replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"\[([^\]]+)\]\(<([^>]+)>\)", r"\1：\2", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1：\2", text)
    return text.strip()


def _chunks(value: str, limit: int = 1600) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [text[start : start + limit] for start in range(0, len(text), limit)]


def _looks_not_found(message: str) -> bool:
    text = message.lower()
    return "not found" in text or "1770002" in text or "resource deleted" in text or "1770003" in text
