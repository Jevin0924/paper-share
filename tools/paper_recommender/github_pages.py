from __future__ import annotations

import base64
import io
import json
import logging
import os
import shutil
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import requests


LOGGER = logging.getLogger("paper_recommender")
CONTENTS_API_SAFE_BYTES = 900_000


@dataclass
class GitHubPagesPublishResult:
    commit_sha: str
    workflow_run_url: str
    verified_urls: list[str]


def publish_reports_to_github_pages(
    config: dict[str, Any],
    repo_root: Path,
    recommended_on: str,
    current_arxiv_ids: Iterable[str],
    report_urls: Iterable[str],
) -> GitHubPagesPublishResult:
    reports_config = config.get("reports", {})
    repository = _github_repository(reports_config)
    branch = str(reports_config.get("github_branch") or "main").strip()
    bundle_path = str(reports_config.get("github_bundle_path") or "reports_bundle.tgz.b64").strip()
    token = _github_token()
    if not token:
        raise RuntimeError("Missing GITHUB_PAGES_TOKEN, GH_TOKEN, or GITHUB_TOKEN for GitHub Pages publishing")

    output_dir = str(reports_config.get("output_dir") or "reports").strip() or "reports"
    report_root = _resolve_repo_path(output_dir, repo_root)
    if not report_root.exists():
        raise RuntimeError(f"Cannot publish GitHub Pages because report output does not exist: {output_dir}")

    current_ids = sorted({value.strip() for value in current_arxiv_ids if value and value.strip()})
    bundle = build_reports_bundle(
        report_root=report_root,
        current_arxiv_ids=current_ids,
        include_assets=str(reports_config.get("github_pages_include_assets") or "current"),
    )
    max_mb = float(reports_config.get("github_bundle_max_mb") or 95)
    bundle_mb = len(bundle.encode("utf-8")) / 1024 / 1024
    if bundle_mb > max_mb:
        raise RuntimeError(f"GitHub Pages bundle is too large: {bundle_mb:.1f} MiB > {max_mb:.1f} MiB")

    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }
    )
    api_base = str(reports_config.get("github_api_base_url") or "https://api.github.com").rstrip("/")
    owner_repo = repository.strip("/")
    existing_sha = _fetch_content_sha(session, api_base, owner_repo, bundle_path, branch)
    commit_sha = _put_bundle(
        session=session,
        api_base=api_base,
        owner_repo=owner_repo,
        path=bundle_path,
        branch=branch,
        content=bundle,
        existing_sha=existing_sha,
        message=str(
            reports_config.get("github_commit_message")
            or f"chore: publish paper reports {recommended_on}"
        ),
    )
    LOGGER.info("Published report bundle to GitHub %s@%s (%s)", owner_repo, branch, commit_sha[:12])

    workflow_run_url = ""
    if reports_config.get("github_pages_wait_for_actions", True):
        workflow_run_url = _wait_for_pages_workflow(
            session=session,
            api_base=api_base,
            owner_repo=owner_repo,
            branch=branch,
            commit_sha=commit_sha,
            workflow_name=str(reports_config.get("github_workflow_name") or "Deploy GitHub Pages"),
            timeout_seconds=int(reports_config.get("github_pages_timeout_seconds") or 900),
            poll_seconds=int(reports_config.get("github_pages_poll_seconds") or 10),
        )

    verified: list[str] = []
    if reports_config.get("github_pages_verify_urls", True):
        base_url = (os.getenv("PAPER_REPORT_BASE_URL") or str(reports_config.get("base_url") or "")).strip()
        urls = [url for url in report_urls if str(url).startswith(("http://", "https://"))]
        if base_url:
            urls.insert(0, base_url.rstrip("/") + "/")
        verified = _wait_for_urls(
            urls=urls,
            timeout_seconds=int(reports_config.get("github_pages_verify_timeout_seconds") or 240),
            poll_seconds=int(reports_config.get("github_pages_verify_poll_seconds") or 8),
        )
    return GitHubPagesPublishResult(commit_sha=commit_sha, workflow_run_url=workflow_run_url, verified_urls=verified)


def build_reports_bundle(report_root: Path, current_arxiv_ids: Iterable[str], include_assets: str = "current") -> str:
    include_assets = include_assets.strip().lower()
    current_ids = sorted({value.strip() for value in current_arxiv_ids if value and value.strip()})
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        staged = tmp_root / "reports"
        staged.mkdir(parents=True, exist_ok=True)
        _copy_optional_file(report_root / "index.html", staged / "index.html")
        _copy_optional_file(report_root / "README.md", staged / "README.md")
        (staged / ".nojekyll").write_text("", encoding="utf-8")
        _copy_tree(report_root / "data", staged / "data")
        _copy_report_html(report_root / "reports", staged / "reports")
        if include_assets == "all":
            _copy_tree(report_root / "assets", staged / "assets")
        elif include_assets == "current":
            for arxiv_id in current_ids:
                _copy_tree(report_root / "assets" / arxiv_id, staged / "assets" / arxiv_id)
        elif include_assets not in {"", "none", "false", "0"}:
            raise ValueError(f"Unsupported github_pages_include_assets value: {include_assets}")
        return _tar_gzip_base64(staged)


def _copy_optional_file(source: Path, target: Path) -> None:
    if not source.exists() or not source.is_file():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _copy_tree(source: Path, target: Path) -> None:
    if not source.exists() or not source.is_dir():
        return
    for path in source.rglob("*"):
        if path.is_file():
            rel = path.relative_to(source)
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)


def _copy_report_html(source: Path, target: Path) -> None:
    if not source.exists() or not source.is_dir():
        return
    for html_path in sorted(source.glob("*/index.html")):
        dest = target / html_path.parent.name / "index.html"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(html_path, dest)


def _tar_gzip_base64(source: Path) -> str:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.add(source, arcname=source.name)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _github_repository(reports_config: dict[str, Any]) -> str:
    repository = (
        os.getenv("GITHUB_PAGES_REPOSITORY")
        or str(reports_config.get("github_repository") or "")
        or os.getenv("GITHUB_REPOSITORY")
    ).strip()
    if not repository or "/" not in repository:
        raise RuntimeError("Missing reports.github_repository or GITHUB_PAGES_REPOSITORY, expected owner/repo")
    return repository


def _github_token() -> str:
    return (os.getenv("GITHUB_PAGES_TOKEN") or os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN") or "").strip()


def _fetch_content_sha(
    session: requests.Session,
    api_base: str,
    owner_repo: str,
    path: str,
    branch: str,
) -> str:
    response = session.get(f"{api_base}/repos/{owner_repo}/contents/{path}", params={"ref": branch}, timeout=30)
    if response.status_code == 404:
        return ""
    _raise_for_github(response)
    data = response.json()
    return str(data.get("sha") or "")


def _put_bundle(
    session: requests.Session,
    api_base: str,
    owner_repo: str,
    path: str,
    branch: str,
    content: str,
    existing_sha: str,
    message: str,
) -> str:
    if len(content.encode("utf-8")) > CONTENTS_API_SAFE_BYTES:
        return _put_bundle_with_git_data_api(
            session=session,
            api_base=api_base,
            owner_repo=owner_repo,
            path=path,
            branch=branch,
            content=content,
            message=message,
        )
    payload: dict[str, Any] = {
        "message": message,
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        "branch": branch,
    }
    if existing_sha:
        payload["sha"] = existing_sha
    response = session.put(f"{api_base}/repos/{owner_repo}/contents/{path}", json=payload, timeout=60)
    if response.status_code == 422 and "too large" in response.text.lower():
        return _put_bundle_with_git_data_api(
            session=session,
            api_base=api_base,
            owner_repo=owner_repo,
            path=path,
            branch=branch,
            content=content,
            message=message,
        )
    _raise_for_github(response)
    data = response.json()
    return str((data.get("commit") or {}).get("sha") or "")


def _put_bundle_with_git_data_api(
    session: requests.Session,
    api_base: str,
    owner_repo: str,
    path: str,
    branch: str,
    content: str,
    message: str,
) -> str:
    ref_response = session.get(f"{api_base}/repos/{owner_repo}/git/ref/heads/{branch}", timeout=30)
    _raise_for_github(ref_response)
    head_sha = str(((ref_response.json().get("object") or {}).get("sha") or ""))
    if not head_sha:
        raise RuntimeError(f"Cannot resolve GitHub branch head for {owner_repo}@{branch}")

    commit_response = session.get(f"{api_base}/repos/{owner_repo}/git/commits/{head_sha}", timeout=30)
    _raise_for_github(commit_response)
    base_tree_sha = str(((commit_response.json().get("tree") or {}).get("sha") or ""))
    if not base_tree_sha:
        raise RuntimeError(f"Cannot resolve GitHub tree for {owner_repo}@{head_sha[:12]}")

    blob_response = session.post(
        f"{api_base}/repos/{owner_repo}/git/blobs",
        json={"content": content, "encoding": "utf-8"},
        timeout=180,
    )
    _raise_for_github(blob_response)
    blob_sha = str(blob_response.json().get("sha") or "")
    if not blob_sha:
        raise RuntimeError("GitHub did not return a blob SHA for report bundle")

    tree_response = session.post(
        f"{api_base}/repos/{owner_repo}/git/trees",
        json={
            "base_tree": base_tree_sha,
            "tree": [
                {
                    "path": path,
                    "mode": "100644",
                    "type": "blob",
                    "sha": blob_sha,
                }
            ],
        },
        timeout=60,
    )
    _raise_for_github(tree_response)
    tree_sha = str(tree_response.json().get("sha") or "")
    if not tree_sha:
        raise RuntimeError("GitHub did not return a tree SHA for report bundle")

    new_commit_response = session.post(
        f"{api_base}/repos/{owner_repo}/git/commits",
        json={"message": message, "tree": tree_sha, "parents": [head_sha]},
        timeout=60,
    )
    _raise_for_github(new_commit_response)
    new_commit_sha = str(new_commit_response.json().get("sha") or "")
    if not new_commit_sha:
        raise RuntimeError("GitHub did not return a commit SHA for report bundle")

    update_response = session.patch(
        f"{api_base}/repos/{owner_repo}/git/refs/heads/{branch}",
        json={"sha": new_commit_sha, "force": False},
        timeout=60,
    )
    _raise_for_github(update_response)
    return new_commit_sha


def _wait_for_pages_workflow(
    session: requests.Session,
    api_base: str,
    owner_repo: str,
    branch: str,
    commit_sha: str,
    workflow_name: str,
    timeout_seconds: int,
    poll_seconds: int,
) -> str:
    deadline = time.monotonic() + timeout_seconds
    last_state = "not found"
    while time.monotonic() < deadline:
        response = session.get(
            f"{api_base}/repos/{owner_repo}/actions/runs",
            params={"branch": branch, "per_page": 20},
            timeout=30,
        )
        _raise_for_github(response)
        runs = response.json().get("workflow_runs") or []
        for run in runs:
            if str(run.get("head_sha") or "") != commit_sha:
                continue
            if workflow_name and str(run.get("name") or "") != workflow_name:
                continue
            status = str(run.get("status") or "")
            conclusion = str(run.get("conclusion") or "")
            last_state = f"{status}/{conclusion or 'pending'}"
            if status == "completed":
                if conclusion == "success":
                    return str(run.get("html_url") or "")
                raise RuntimeError(f"GitHub Pages workflow failed: {run.get('html_url') or commit_sha} ({conclusion})")
        time.sleep(max(poll_seconds, 1))
    raise RuntimeError(f"Timed out waiting for GitHub Pages workflow for {commit_sha[:12]} ({last_state})")


def _wait_for_urls(urls: Iterable[str], timeout_seconds: int, poll_seconds: int) -> list[str]:
    unique_urls = list(dict.fromkeys(str(url).strip() for url in urls if str(url).strip()))
    pending = set(unique_urls)
    verified: list[str] = []
    deadline = time.monotonic() + timeout_seconds
    while pending and time.monotonic() < deadline:
        for url in list(pending):
            try:
                response = requests.get(url, timeout=20, headers={"User-Agent": "paper-share-pages-verify/1.0"})
                if response.status_code == 200:
                    pending.remove(url)
                    verified.append(url)
            except requests.RequestException:
                pass
        if pending:
            time.sleep(max(poll_seconds, 1))
    if pending:
        raise RuntimeError("GitHub Pages verification failed for URLs: " + ", ".join(sorted(pending)))
    return verified


def _raise_for_github(response: requests.Response) -> None:
    if response.ok:
        return
    try:
        data = response.json()
    except json.JSONDecodeError:
        data = response.text
    message = data.get("message") if isinstance(data, dict) else str(data)
    raise RuntimeError(f"GitHub API request failed ({response.status_code}): {message}")


def _resolve_repo_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path
