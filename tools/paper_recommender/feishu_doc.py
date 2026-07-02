from __future__ import annotations

import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .feishu_auth import OPEN_FEISHU, TenantTokenClient


@dataclass(slots=True)
class FeishuDocument:
    document_id: str
    title: str
    url: str
    wiki_node_token: str = ""
    wiki_space_id: str = ""


@dataclass
class FeishuDocClient:
    token_client: TenantTokenClient
    base_url: str = OPEN_FEISHU
    doc_base_url: str = ""

    def create_document(self, title: str, folder_token: str = "") -> FeishuDocument:
        body: dict[str, Any] = {"title": title}
        if folder_token:
            body["folder_token"] = folder_token
        data = self._request("post", f"{self.base_url}/docx/v1/documents", json=body)
        document = (data.get("data") or {}).get("document") or {}
        document_id = str(document.get("document_id") or "")
        if not document_id:
            raise RuntimeError(f"Feishu document create response did not include document_id: {data}")
        return FeishuDocument(
            document_id=document_id,
            title=str(document.get("title") or title),
            url=_document_url(self.doc_base_url, document_id),
        )

    def get_wiki_node(self, token: str, obj_type: str = "") -> dict[str, Any]:
        params = {"token": token}
        if obj_type:
            params["obj_type"] = obj_type
        data = self._request("get", f"{self.base_url}/wiki/v2/spaces/get_node", params=params)
        node = (data.get("data") or {}).get("node") or {}
        if not node:
            raise RuntimeError(f"Feishu wiki get_node response did not include node: {data}")
        return node

    def create_wiki_document(self, title: str, parent_wiki_token: str) -> FeishuDocument:
        parent = self.get_wiki_node(parent_wiki_token)
        space_id = str(parent.get("space_id") or "")
        parent_node_token = str(parent.get("node_token") or parent_wiki_token)
        if not space_id or not parent_node_token:
            raise RuntimeError(f"Cannot resolve Feishu wiki parent node: {parent}")
        data = self._request(
            "post",
            f"{self.base_url}/wiki/v2/spaces/{space_id}/nodes",
            json={
                "obj_type": "docx",
                "parent_node_token": parent_node_token,
                "node_type": "origin",
                "title": title,
            },
        )
        node = (data.get("data") or {}).get("node") or {}
        document_id = str(node.get("obj_token") or "")
        node_token = str(node.get("node_token") or "")
        if not document_id or not node_token:
            raise RuntimeError(f"Feishu wiki create response did not include obj_token/node_token: {data}")
        return FeishuDocument(
            document_id=document_id,
            title=str(node.get("title") or title),
            url=_wiki_url(self.doc_base_url, node_token),
            wiki_node_token=node_token,
            wiki_space_id=space_id,
        )

    def convert_markdown(self, content: str, document_id: str = "") -> dict[str, Any]:
        data = self._request(
            "post",
            f"{self.base_url}/docx/v1/documents/blocks/convert",
            json={"content_type": "markdown", "content": content},
        )
        payload = data.get("data") or {}
        return {
            "children_id": list(payload.get("children_id") or payload.get("first_level_block_ids") or []),
            "descendants": _strip_table_merge_info(list(payload.get("descendants") or payload.get("blocks") or [])),
        }

    def append_converted_markdown(self, document_id: str, content: str) -> None:
        converted = self.convert_markdown(content, document_id=document_id)
        children_id = converted.get("children_id") or []
        descendants = converted.get("descendants") or []
        if not children_id or not descendants:
            return
        self._sleep_after_write()
        try:
            self._request(
                "post",
                f"{self.base_url}/docx/v1/documents/{document_id}/blocks/{document_id}/descendant",
                params={"document_revision_id": "-1"},
                json={"index": -1, "children_id": children_id, "descendants": descendants},
            )
        except Exception:
            self.append_basic_markdown(document_id, content)

    def append_basic_markdown(self, document_id: str, content: str) -> None:
        blocks = _basic_markdown_blocks(content)
        self.append_blocks(document_id, blocks)

    def append_blocks(self, document_id: str, blocks: list[dict[str, Any]]) -> None:
        for start in range(0, len(blocks), 40):
            batch = blocks[start : start + 40]
            if not batch:
                continue
            self._sleep_after_write()
            self._request(
                "post",
                f"{self.base_url}/docx/v1/documents/{document_id}/blocks/{document_id}/children",
                params={"document_revision_id": "-1"},
                json={"index": -1, "children": batch},
            )

    def append_text(self, document_id: str, content: str, block_type: int = 2) -> None:
        blocks: list[dict[str, Any]] = []
        _extend_text_blocks(blocks, content, block_type=block_type)
        self.append_blocks(document_id, blocks)

    def append_table(self, document_id: str, columns: list[str], rows: list[list[str]]) -> None:
        markdown = _markdown_table(columns, rows)
        try:
            converted = self.convert_markdown(markdown, document_id=document_id)
            children_id = converted.get("children_id") or []
            descendants = converted.get("descendants") or []
            if children_id and descendants:
                self._sleep_after_write()
                self._request(
                    "post",
                    f"{self.base_url}/docx/v1/documents/{document_id}/blocks/{document_id}/descendant",
                    params={"document_revision_id": "-1"},
                    json={"index": -1, "children_id": children_id, "descendants": descendants},
                )
                return
        except Exception:
            pass
        table = _table_descendants(columns, rows)
        if not table:
            return
        self._sleep_after_write()
        try:
            self._request(
                "post",
                f"{self.base_url}/docx/v1/documents/{document_id}/blocks/{document_id}/descendant",
                params={"document_revision_id": "-1"},
                json={"index": -1, "children_id": table["children_id"], "descendants": table["descendants"]},
            )
        except Exception:
            self.append_basic_markdown(document_id, markdown)

    def list_children(self, document_id: str, block_id: str = "") -> list[dict[str, Any]]:
        block_id = block_id or document_id
        children: list[dict[str, Any]] = []
        page_token = ""
        while True:
            params: dict[str, str] = {
                "document_revision_id": "-1",
                "page_size": "500",
            }
            if page_token:
                params["page_token"] = page_token
            data = self._request(
                "get",
                f"{self.base_url}/docx/v1/documents/{document_id}/blocks/{block_id}/children",
                params=params,
            )
            payload = data.get("data") or {}
            children.extend(item for item in payload.get("items", []) if isinstance(item, dict))
            if not payload.get("has_more"):
                return children
            page_token = str(payload.get("page_token") or "")
            if not page_token:
                return children

    def clear_document(self, document_id: str) -> None:
        children = self.list_children(document_id, document_id)
        if not children:
            return
        self._sleep_after_write()
        self._request(
            "delete",
            f"{self.base_url}/docx/v1/documents/{document_id}/blocks/{document_id}/children/batch_delete",
            params={"document_revision_id": "-1"},
            json={"start_index": 0, "end_index": len(children)},
        )

    def replace_markdown(
        self,
        document_id: str,
        content: str,
        image_paths: list[Path] | None = None,
        document_url: str = "",
        wiki_node_token: str = "",
        wiki_space_id: str = "",
    ) -> FeishuDocument:
        self.clear_document(document_id)
        self.append_converted_markdown(document_id, content)
        for image_path in image_paths or []:
            self._sleep_after_write()
            self.append_image(document_id, image_path)
        return FeishuDocument(
            document_id=document_id,
            title="",
            url=document_url or (_wiki_url(self.doc_base_url, wiki_node_token) if wiki_node_token else _document_url(self.doc_base_url, document_id)),
            wiki_node_token=wiki_node_token,
            wiki_space_id=wiki_space_id,
        )

    def append_image(self, document_id: str, image_path: Path) -> None:
        if not image_path.exists() or not image_path.is_file():
            return
        self._sleep_after_write()
        data = self._request(
            "post",
            f"{self.base_url}/docx/v1/documents/{document_id}/blocks/{document_id}/children",
            params={"document_revision_id": "-1"},
            json={"index": -1, "children": [{"block_type": 27, "image": {}}]},
        )
        children = ((data.get("data") or {}).get("children") or [])
        if not children:
            return
        image_block_id = str(children[0].get("block_id") or "")
        if not image_block_id:
            return
        token = self.upload_media(image_path, parent_type="docx_image", parent_node=image_block_id)
        if token:
            self._sleep_after_write()
            self._request(
                "patch",
                f"{self.base_url}/docx/v1/documents/{document_id}/blocks/{image_block_id}",
                params={"document_revision_id": "-1"},
                json={"replace_image": {"token": token}},
            )

    def upload_media(self, image_path: Path, parent_type: str, parent_node: str) -> str:
        with image_path.open("rb") as file_obj:
            files = {"file": (image_path.name, file_obj)}
            data = {
                "file_name": image_path.name,
                "parent_type": parent_type,
                "parent_node": parent_node,
                "size": str(image_path.stat().st_size),
            }
            response = requests.post(
                f"{self.base_url}/drive/v1/medias/upload_all",
                headers={"Authorization": f"Bearer {self.token_client.token()}"},
                files=files,
                data=data,
                timeout=60,
            )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError(f"Failed to upload Feishu media {image_path}: {payload.get('msg') or payload}")
        return str((payload.get("data") or {}).get("file_token") or "")

    def publish_markdown(
        self,
        title: str,
        content: str,
        folder_token: str = "",
        image_paths: list[Path] | None = None,
        document_id: str = "",
        document_url: str = "",
        wiki_parent_token: str = "",
        wiki_node_token: str = "",
        wiki_space_id: str = "",
    ) -> FeishuDocument:
        if document_id:
            try:
                document = self.replace_markdown(
                    document_id,
                    content,
                    image_paths=image_paths,
                    document_url=document_url,
                    wiki_node_token=wiki_node_token,
                    wiki_space_id=wiki_space_id,
                )
                document.title = title
                return document
            except RuntimeError as exc:
                if not _looks_not_found_error(str(exc)):
                    raise
        self.preflight_markdown_write()
        if wiki_parent_token:
            document = self.create_wiki_document(title=title, parent_wiki_token=wiki_parent_token)
        else:
            document = self.create_document(title=title, folder_token=folder_token)
        self.append_converted_markdown(document.document_id, content)
        for image_path in image_paths or []:
            self.append_image(document.document_id, image_path)
        return document

    def preflight_markdown_write(self) -> None:
        self.convert_markdown("# permission check")

    def _request(
        self,
        method: str,
        url: str,
        params: dict[str, str] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        last_data: dict[str, Any] | None = None
        for attempt in range(4):
            response = requests.request(
                method,
                url,
                headers={
                    "Authorization": f"Bearer {self.token_client.token()}",
                    "Content-Type": "application/json; charset=utf-8",
                },
                params=params,
                json=json,
                timeout=60,
            )
            try:
                data = response.json()
            except ValueError:
                data = {}
            last_data = data
            retry_after = _retry_after_seconds(response, data, attempt)
            if retry_after:
                time.sleep(retry_after)
                continue
            response.raise_for_status()
            if data.get("code") != 0:
                raise RuntimeError(f"Feishu doc API failed: {data.get('msg') or data}")
            return data
        raise RuntimeError(f"Feishu doc API failed after retries: {last_data or {}}")

    def _sleep_after_write(self) -> None:
        time.sleep(0.36)


def build_feishu_doc_client(config: dict[str, Any]) -> FeishuDocClient | None:
    reports_config = config.get("reports", {})
    feishu_config = config.get("feishu", {})
    app_id = os.getenv("FEISHU_APP_ID", "").strip()
    app_secret = os.getenv("FEISHU_APP_SECRET", "").strip()
    if not app_id or not app_secret:
        return None
    doc_base_url = (
        os.getenv("FEISHU_DOC_BASE_URL")
        or str(reports_config.get("doc_base_url") or "")
        or _origin_from_url(os.getenv("FEISHU_BITABLE_URL") or str(feishu_config.get("bitable_url") or ""))
    )
    return FeishuDocClient(
        token_client=TenantTokenClient(app_id=app_id, app_secret=app_secret),
        doc_base_url=doc_base_url,
    )


def _document_url(doc_base_url: str, document_id: str) -> str:
    base = doc_base_url.rstrip("/") if doc_base_url else "https://feishu.cn"
    return f"{base}/docx/{document_id}"


def _wiki_url(doc_base_url: str, node_token: str) -> str:
    base = doc_base_url.rstrip("/") if doc_base_url else "https://feishu.cn"
    return f"{base}/wiki/{node_token}"


def _origin_from_url(value: str) -> str:
    match = re.match(r"^(https?://[^/]+)", value.strip())
    return match.group(1) if match else ""


def _retry_after_seconds(response: requests.Response, data: dict[str, Any], attempt: int) -> float:
    if response.status_code == 429 or data.get("code") == 99991400:
        try:
            return float(response.headers.get("Retry-After") or 0) or min(2**attempt * 0.5, 4.0)
        except ValueError:
            return min(2**attempt * 0.5, 4.0)
    return 0.0


def _looks_not_found_error(message: str) -> bool:
    text = message.lower()
    return "not found" in text or "1770002" in text or "resource deleted" in text or "1770003" in text


def _strip_table_merge_info(blocks: list[Any]) -> list[Any]:
    cleaned: list[Any] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        item = dict(block)
        table = item.get("table")
        if isinstance(table, dict):
            table = dict(table)
            table.pop("merge_info", None)
            item["table"] = table
        cleaned.append(item)
    return cleaned


def _table_descendants(columns: list[str], rows: list[list[str]]) -> dict[str, Any]:
    headers = [str(column or "").strip() for column in columns if str(column or "").strip()]
    if not headers:
        return {}
    body_rows: list[list[str]] = []
    for raw_row in rows:
        cells = [str(item or "").strip() for item in raw_row[: len(headers)]]
        cells.extend([""] * (len(headers) - len(cells)))
        body_rows.append(cells)
    all_rows = [headers, *body_rows]
    table_id = _block_id("table")
    cell_ids: list[str] = []
    descendants: list[dict[str, Any]] = []
    for row_index, row in enumerate(all_rows):
        for column_index, cell in enumerate(row):
            cell_id = _block_id(f"cell{row_index}_{column_index}")
            text_id = _block_id(f"text{row_index}_{column_index}")
            cell_ids.append(cell_id)
            descendants.append(
                {
                    "block_id": cell_id,
                    "block_type": 32,
                    "table_cell": {},
                    "children": [text_id],
                }
            )
            descendants.append(
                {
                    "block_id": text_id,
                    "block_type": 2,
                    "text": {"elements": [{"text_run": {"content": cell or "-"}}]},
                    "children": [],
                }
            )
    descendants.insert(
        0,
        {
            "block_id": table_id,
            "block_type": 31,
            "table": {
                "cells": cell_ids,
                "property": {
                    "row_size": len(all_rows),
                    "column_size": len(headers),
                }
            },
            "children": cell_ids,
        },
    )
    return {"children_id": [table_id], "descendants": descendants}


def _block_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _escape_table_cell(value: str) -> str:
    return str(value or "-").replace("\n", " ").replace("|", "\\|")


def _markdown_table(columns: list[str], rows: list[list[str]]) -> str:
    headers = [str(column or "").strip() for column in columns if str(column or "").strip()]
    if not headers:
        return ""
    lines = ["| " + " | ".join(_escape_table_cell(cell) for cell in headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for raw_row in rows:
        cells = [str(item or "").strip() for item in raw_row[: len(headers)]]
        cells.extend([""] * (len(headers) - len(cells)))
        lines.append("| " + " | ".join(_escape_table_cell(cell) for cell in cells) + " |")
    return "\n".join(lines)


def _basic_markdown_blocks(content: str) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    in_code = False
    code_lines: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_code:
                _extend_text_blocks(blocks, "\n".join(code_lines), prefix="代码：")
                code_lines = []
                in_code = False
            else:
                in_code = True
                code_lines = []
            continue
        if in_code:
            code_lines.append(line)
            continue
        if not stripped:
            continue
        if stripped.startswith("!["):
            continue
        if _is_markdown_table_separator(stripped):
            continue
        if stripped.startswith("|") and stripped.endswith("|"):
            stripped = " | ".join(cell.strip() for cell in stripped.strip("|").split("|"))
        heading_level = _heading_level(stripped)
        if heading_level:
            text = stripped.lstrip("#").strip()
            blocks.append(_textual_block(min(heading_level + 2, 5), _strip_inline_markdown(text)))
            continue
        if stripped.startswith(">"):
            stripped = stripped.lstrip("> ").strip()
        _extend_text_blocks(blocks, _strip_inline_markdown(stripped))
    if code_lines:
        _extend_text_blocks(blocks, "\n".join(code_lines), prefix="代码：")
    return blocks or [_textual_block(2, "暂无内容。")]


def _extend_text_blocks(blocks: list[dict[str, Any]], text: str, prefix: str = "", block_type: int = 2) -> None:
    text = (prefix + text).strip()
    if not text:
        return
    max_len = 1600
    for start in range(0, len(text), max_len):
        blocks.append(_textual_block(block_type, text[start : start + max_len]))


def _textual_block(block_type: int, content: str) -> dict[str, Any]:
    key = {2: "text", 3: "heading1", 4: "heading2", 5: "heading3"}.get(block_type, "text")
    return {
        "block_type": block_type,
        key: {"elements": [{"text_run": {"content": content}}]},
    }


def _heading_level(value: str) -> int:
    if not value.startswith("#"):
        return 0
    return min(len(value) - len(value.lstrip("#")), 3)


def _is_markdown_table_separator(value: str) -> bool:
    text = value.strip().strip("|").replace(" ", "")
    return bool(text) and set(text) <= {"-", ":"}


def _strip_inline_markdown(value: str) -> str:
    text = value.replace("**", "").replace("__", "").replace("`", "")
    text = re.sub(r"\[([^\]]+)\]\(<([^>]+)>\)", r"\1：\2", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1：\2", text)
    return text.strip()
