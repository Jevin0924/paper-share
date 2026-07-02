from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests


OPEN_FEISHU = "https://open.feishu.cn/open-apis"


@dataclass
class TenantTokenClient:
    app_id: str
    app_secret: str
    base_url: str = OPEN_FEISHU
    _token: str = ""
    _expires_at: float = 0

    def token(self) -> str:
        if self._token and time.time() < self._expires_at - 120:
            return self._token
        url = f"{self.base_url}/auth/v3/tenant_access_token/internal"
        response = requests.post(
            url,
            json={"app_id": self.app_id, "app_secret": self.app_secret},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("code") != 0:
            raise RuntimeError(f"Failed to get Feishu tenant token: {data.get('msg') or data}")
        self._token = data["tenant_access_token"]
        self._expires_at = time.time() + int(data.get("expire", 7200))
        return self._token


def resolve_wiki_bitable_app_token(
    token_client: TenantTokenClient,
    wiki_token: str,
    base_url: str = OPEN_FEISHU,
) -> str:
    """Resolve a feishu.cn/wiki node token to the bitable app_token."""
    token = token_client.token()
    data = _get_wiki_node(base_url=base_url, access_token=token, params={"token": wiki_token})
    if data.get("code") != 0:
        data = _get_wiki_node(base_url=base_url, access_token=token, params={"token": wiki_token, "obj_type": "bitable"})
    if data.get("code") != 0:
        raise RuntimeError(f"Failed to resolve Feishu wiki bitable token: {data.get('msg') or data}")
    node = (data.get("data") or {}).get("node") or {}
    obj_token = node.get("obj_token") or ""
    obj_type = node.get("obj_type") or ""
    if not obj_token:
        raise RuntimeError("Feishu wiki node response did not contain obj_token")
    if obj_type and obj_type != "bitable":
        raise RuntimeError(f"Feishu wiki node is {obj_type}, expected bitable")
    return obj_token


def _get_wiki_node(base_url: str, access_token: str, params: dict[str, str]) -> dict[str, Any]:
    url = f"{base_url}/wiki/v2/spaces/get_node"
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        params=params,
        timeout=20,
    )
    data: dict[str, Any] = response.json()
    if response.status_code >= 400 and data.get("code") == 0:
        response.raise_for_status()
    return data
