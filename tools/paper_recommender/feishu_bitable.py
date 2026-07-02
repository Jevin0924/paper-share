from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import requests

from .feishu_auth import OPEN_FEISHU, TenantTokenClient
from .models import PaperRecommendation, normalize_title


LOGGER = logging.getLogger("paper_recommender")


@dataclass(slots=True)
class BitableRecord:
    record_id: str
    fields: dict[str, Any]


@dataclass(slots=True)
class BitableField:
    field_id: str
    field_name: str
    type: int


@dataclass
class FeishuBitableClient:
    token_client: TenantTokenClient
    app_token: str
    table_id: str
    field_names: dict[str, str]
    base_url: str = OPEN_FEISHU
    _known_field_names: set[str] | None = field(default=None, init=False, repr=False)

    def upsert_recommendation(self, recommendation: PaperRecommendation) -> BitableRecord:
        existing = self.find_existing(recommendation)
        fields = self.filter_existing_fields(recommendation.to_bitable_fields(self.field_names))
        if existing:
            first_seen_field = self.field_names.get("first_seen_at")
            if first_seen_field:
                fields.pop(first_seen_field, None)
            return self.update_record(existing.record_id, fields)
        return self.create_record(fields)

    def mark_pushed(self, recommendation: PaperRecommendation) -> BitableRecord | None:
        existing = self.find_existing(recommendation)
        if not existing:
            return None
        pushed_field = self.field_names.get("pushed")
        if not pushed_field:
            return existing
        fields = self.filter_existing_fields({pushed_field: "是"})
        if not fields:
            return existing
        return self.update_record(existing.record_id, fields)

    def find_existing(self, recommendation: PaperRecommendation) -> BitableRecord | None:
        records = self.list_records()
        paper = recommendation.paper
        arxiv_field = self.field_names.get("arxiv_id", "")
        doi_field = self.field_names.get("doi", "")
        title_field = self.field_names.get("title", "")
        normalized_title = normalize_title(paper.title)
        for record in records:
            fields = record.fields
            if arxiv_field and paper.arxiv_id and _text_value(fields.get(arxiv_field)).lower() == paper.arxiv_id.lower():
                return record
            if doi_field and paper.doi and _text_value(fields.get(doi_field)).lower() == paper.doi.lower():
                return record
            if title_field and normalized_title and normalize_title(_text_value(fields.get(title_field))) == normalized_title:
                return record
        return None

    def list_records(self) -> list[BitableRecord]:
        url = f"{self.base_url}/bitable/v1/apps/{self.app_token}/tables/{self.table_id}/records/search"
        records: list[BitableRecord] = []
        page_token = ""
        while True:
            params: dict[str, Any] = {"page_size": 500}
            if page_token:
                params["page_token"] = page_token
            response = requests.post(url, headers=self._headers(), params=params, json={}, timeout=30)
            response.raise_for_status()
            data = response.json()
            if data.get("code") != 0:
                raise RuntimeError(f"Failed to list Feishu bitable records: {data.get('msg') or data}")
            payload = data.get("data") or {}
            for item in payload.get("items", []):
                records.append(BitableRecord(record_id=item["record_id"], fields=item.get("fields") or {}))
            if not payload.get("has_more"):
                return records
            page_token = payload.get("page_token") or ""
            if not page_token:
                return records

    def list_fields(self) -> list[BitableField]:
        url = f"{self.base_url}/bitable/v1/apps/{self.app_token}/tables/{self.table_id}/fields"
        fields: list[BitableField] = []
        page_token = ""
        while True:
            params: dict[str, Any] = {"page_size": 100}
            if page_token:
                params["page_token"] = page_token
            response = requests.get(url, headers=self._headers(), params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if data.get("code") != 0:
                raise RuntimeError(f"Failed to list Feishu bitable fields: {data.get('msg') or data}")
            payload = data.get("data") or {}
            for item in payload.get("items", []):
                fields.append(
                    BitableField(
                        field_id=str(item.get("field_id") or ""),
                        field_name=str(item.get("field_name") or ""),
                        type=int(item.get("type") or 0),
                    )
                )
            if not payload.get("has_more"):
                return fields
            page_token = payload.get("page_token") or ""
            if not page_token:
                return fields

    def filter_existing_fields(self, fields: dict[str, Any]) -> dict[str, Any]:
        existing = self.known_field_names()
        filtered = {name: value for name, value in fields.items() if name in existing}
        missing = sorted(set(fields) - set(filtered))
        if missing:
            LOGGER.warning(
                "Skipping Feishu bitable fields not present in table: %s",
                ", ".join(missing),
            )
        return filtered

    def known_field_names(self) -> set[str]:
        if self._known_field_names is None:
            self._known_field_names = {field.field_name for field in self.list_fields()}
        return self._known_field_names

    def ensure_configured_fields(self, field_type_by_key: dict[str, int] | None = None) -> dict[str, Any]:
        field_type_by_key = field_type_by_key or {}
        existing = {field.field_name for field in self.list_fields()}
        created: list[dict[str, Any]] = []
        skipped: list[str] = []
        for key, field_name in self.field_names.items():
            if not field_name:
                continue
            if field_name in existing:
                skipped.append(field_name)
                continue
            field = self.create_field(field_name, field_type_by_key.get(key, 1))
            existing.add(field.field_name)
            created.append({"field_name": field.field_name, "type": field.type})
        return {"created": created, "existing": skipped}

    def create_field(self, field_name: str, field_type: int = 1) -> BitableField:
        url = f"{self.base_url}/bitable/v1/apps/{self.app_token}/tables/{self.table_id}/fields"
        response = requests.post(
            url,
            headers=self._headers(),
            json={"field_name": field_name, "type": field_type},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Failed to create Feishu bitable field {field_name}: {data.get('msg') or data}")
        field = data["data"]["field"]
        return BitableField(
            field_id=str(field.get("field_id") or ""),
            field_name=str(field.get("field_name") or field_name),
            type=int(field.get("type") or field_type),
        )

    def create_record(self, fields: dict[str, Any]) -> BitableRecord:
        url = f"{self.base_url}/bitable/v1/apps/{self.app_token}/tables/{self.table_id}/records"
        response = requests.post(url, headers=self._headers(), json={"fields": fields}, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Failed to create Feishu bitable record: {data.get('msg') or data}")
        record = data["data"]["record"]
        return BitableRecord(record_id=record["record_id"], fields=record.get("fields") or {})

    def update_record(self, record_id: str, fields: dict[str, Any]) -> BitableRecord:
        url = f"{self.base_url}/bitable/v1/apps/{self.app_token}/tables/{self.table_id}/records/{record_id}"
        response = requests.put(url, headers=self._headers(), json={"fields": fields}, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Failed to update Feishu bitable record: {data.get('msg') or data}")
        record = data["data"]["record"]
        return BitableRecord(record_id=record["record_id"], fields=record.get("fields") or {})

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token_client.token()}",
            "Content-Type": "application/json; charset=utf-8",
        }


def _text_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(_text_value(item) for item in value)
    if isinstance(value, dict):
        for key in ("text", "link", "value", "name"):
            if key in value:
                return _text_value(value[key])
    return str(value)
