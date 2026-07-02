from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from urllib.request import Request, urlopen

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.paper_recommender.config import load_config
from tools.paper_recommender.models import PaperCandidate
from tools.paper_recommender.rank import score_paper_breakdown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score arXiv URLs with the paper recommender ranking rules.")
    parser.add_argument("urls", nargs="+")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "tools" / "paper_recommender" / "config.yaml")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    results = []
    for url in args.urls:
        paper = fetch_arxiv_abs(url)
        breakdown = score_paper_breakdown(paper, config)
        results.append(
            {
                "url": url,
                "arxiv_id": paper.arxiv_id,
                "title": paper.title,
                "categories": paper.categories,
                "score": round(float(breakdown["score"]), 2),
                "primary_task": breakdown["primary_task"],
                "primary_task_score": round(float(breakdown["primary_task_score"]), 2),
                "passed_primary_gate": bool(breakdown["passed_primary_gate"]),
                "components": {key: round(float(value), 2) for key, value in breakdown["components"].items()},
                "reasons": breakdown["reasons"][:16],
            }
        )
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        for result in results:
            print(f"{result['arxiv_id']} | {result['score']}/30 | {result['primary_task']} | {result['title']}")
            print(f"  primary={result['primary_task_score']} gate={result['passed_primary_gate']} components={result['components']}")
            print("  reasons=" + "; ".join(result["reasons"]))
    return 0


def fetch_arxiv_abs(url_or_id: str) -> PaperCandidate:
    arxiv_id = extract_arxiv_id(url_or_id)
    url = f"https://arxiv.org/abs/{arxiv_id}"
    request = Request(url, headers={"User-Agent": "paper-share-recommender/1.0"})
    with urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8", errors="replace")
    title = _extract_descriptor(html, "Title") or arxiv_id
    abstract = _extract_descriptor(html, "Abstract")
    authors = _extract_authors(html)
    categories = _extract_categories(html)
    published = _extract_published(html) or _date_from_arxiv_id(arxiv_id)
    return PaperCandidate(
        title=title,
        abstract=abstract,
        authors=authors,
        published=published,
        updated=published,
        arxiv_id=arxiv_id,
        arxiv_url=url,
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        categories=categories,
        primary_category=categories[0] if categories else "",
        raw={"source": "arxiv_abs"},
    )


def extract_arxiv_id(value: str) -> str:
    value = value.strip()
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([^?#/]+)", value, flags=re.IGNORECASE)
    if match:
        value = match.group(1)
    value = value.rsplit("/", 1)[-1]
    value = re.sub(r"\.pdf$", "", value, flags=re.IGNORECASE)
    value = re.sub(r"v\d+$", "", value, flags=re.IGNORECASE)
    if not re.fullmatch(r"\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7}", value, flags=re.IGNORECASE):
        raise ValueError(f"Cannot parse arXiv id from: {value}")
    return value


def _extract_descriptor(html: str, name: str) -> str:
    pattern = rf'<(?:h1|blockquote) class="(?:title|abstract) mathjax">.*?<span class="descriptor">{name}:</span>(.*?)</(?:h1|blockquote)>'
    match = re.search(pattern, html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return _clean_html(match.group(1))


def _extract_authors(html: str) -> list[str]:
    match = re.search(r'<div class="authors">(.*?)</div>', html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    authors = re.findall(r"<a [^>]*>(.*?)</a>", match.group(1), flags=re.IGNORECASE | re.DOTALL)
    return [_clean_html(author) for author in authors if _clean_html(author)]


def _extract_categories(html: str) -> list[str]:
    match = re.search(r"<td[^>]*class=\"tablecell subjects\"[^>]*>(.*?)</td>", html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return []
    return re.findall(r"\((cs\.[A-Z]+|eess\.[A-Z]+|stat\.[A-Z]+)\)", _clean_html(match.group(1)))


def _extract_published(html: str) -> str:
    match = re.search(r"Submitted on (\d{1,2}) ([A-Z][a-z]{2}) (\d{4})", html)
    if not match:
        return ""
    day, month_name, year = match.groups()
    month = datetime.strptime(month_name, "%b").month
    return datetime(int(year), month, int(day), tzinfo=timezone.utc).isoformat()


def _date_from_arxiv_id(arxiv_id: str) -> str:
    match = re.match(r"(\d{2})(\d{2})\.", arxiv_id)
    if not match:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    year = 2000 + int(match.group(1))
    month = int(match.group(2))
    return datetime(year, month, 1, tzinfo=timezone.utc).isoformat()


def _clean_html(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    return " ".join(unescape(value).split())


if __name__ == "__main__":
    raise SystemExit(main())
