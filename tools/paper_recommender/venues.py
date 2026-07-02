from __future__ import annotations

import re
from typing import Any

from .models import PaperCandidate


DEFAULT_PRESTIGE_VENUES: dict[str, list[str]] = {
    "CVPR": [
        "CVPR",
        "IEEE CVF Conference on Computer Vision and Pattern Recognition",
        "IEEE/CVF Conference on Computer Vision and Pattern Recognition",
        "Computer Vision and Pattern Recognition",
    ],
    "ICCV": [
        "ICCV",
        "IEEE International Conference on Computer Vision",
        "International Conference on Computer Vision",
    ],
    "ECCV": [
        "ECCV",
        "European Conference on Computer Vision",
    ],
    "NeurIPS": [
        "NeurIPS",
        "NIPS",
        "Conference on Neural Information Processing Systems",
        "Neural Information Processing Systems",
    ],
    "ICLR": [
        "ICLR",
        "International Conference on Learning Representations",
    ],
    "ICML": [
        "ICML",
        "International Conference on Machine Learning",
    ],
    "AAAI": [
        "AAAI",
        "AAAI Conference on Artificial Intelligence",
    ],
    "IJCAI": [
        "IJCAI",
        "International Joint Conference on Artificial Intelligence",
    ],
    "WACV": [
        "WACV",
        "Winter Conference on Applications of Computer Vision",
    ],
    "TPAMI": [
        "TPAMI",
        "IEEE Transactions on Pattern Analysis and Machine Intelligence",
        "Transactions on Pattern Analysis and Machine Intelligence",
    ],
    "IJCV": [
        "IJCV",
        "International Journal of Computer Vision",
    ],
    "TIP": [
        "TIP",
        "IEEE Transactions on Image Processing",
        "Transactions on Image Processing",
    ],
    "TMM": [
        "TMM",
        "IEEE Transactions on Multimedia",
        "Transactions on Multimedia",
    ],
    "TCSVT": [
        "TCSVT",
        "IEEE Transactions on Circuits and Systems for Video Technology",
    ],
}


def match_prestige_venue(paper: PaperCandidate, config: dict[str, Any]) -> str:
    if not config.get("prestige_venues", {}).get("enabled", True):
        return ""
    texts = [
        paper.venue,
        *(paper.quality_signals or []),
    ]
    return match_prestige_venue_text(" ".join(texts), config)


def match_prestige_venue_text(value: str, config: dict[str, Any]) -> str:
    if not value:
        return ""
    normalized_value = _normalize(value)
    for venue_name, aliases in _configured_venues(config).items():
        if any(_alias_matches(alias, normalized_value) for alias in aliases):
            return venue_name
    return ""


def prestige_venue_bonus(config: dict[str, Any]) -> float:
    venue_config = config.get("prestige_venues", {})
    return float(venue_config.get("quality_bonus", 5.0))


def prestige_venue_candidate_limit(config: dict[str, Any], default: int) -> int:
    venue_config = config.get("prestige_venues", {})
    return int(venue_config.get("candidate_bucket_limit", default))


def prestige_venue_label_from_text(value: str, config: dict[str, Any]) -> str:
    venue_name = match_prestige_venue_text(value, config)
    if not venue_name:
        return ""
    year = _year_near_venue(value, venue_name)
    return f"{venue_name} {year}" if year else venue_name


def _configured_venues(config: dict[str, Any]) -> dict[str, list[str]]:
    venue_config = config.get("prestige_venues", {})
    configured = venue_config.get("venues") or venue_config.get("names")
    if not configured:
        return DEFAULT_PRESTIGE_VENUES

    result: dict[str, list[str]] = {}
    if isinstance(configured, dict):
        for name, aliases in configured.items():
            result[str(name)] = _as_aliases(str(name), aliases)
        return result

    if isinstance(configured, list):
        for item in configured:
            if isinstance(item, str):
                result[item] = [item]
            elif isinstance(item, dict):
                name = str(item.get("name") or "").strip()
                if name:
                    result[name] = _as_aliases(name, item.get("aliases"))
        return result

    return DEFAULT_PRESTIGE_VENUES


def _as_aliases(name: str, aliases: Any) -> list[str]:
    values = [name]
    if isinstance(aliases, str):
        values.append(aliases)
    elif isinstance(aliases, list):
        values.extend(str(item) for item in aliases)
    return _dedupe_keep_order(values)


def _alias_matches(alias: str, normalized_value: str) -> bool:
    normalized_alias = _normalize(alias)
    if not normalized_alias:
        return False
    if " " in normalized_alias:
        return normalized_alias in normalized_value
    return re.search(rf"(^|\s){re.escape(normalized_alias)}(\s|$)", normalized_value) is not None


def _normalize(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.lower()).split())


def _year_near_venue(value: str, venue_name: str) -> str:
    match = re.search(rf"\b{re.escape(venue_name)}\b\D{{0,20}}((?:19|20)\d{{2}})", value, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    return ""


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = str(value or "").strip()
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            result.append(item)
    return result
