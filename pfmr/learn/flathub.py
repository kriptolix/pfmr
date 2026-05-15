"""
pfmr.learn.flathub
~~~~~~~~~~~~~~~~~~~
FlathubMiner — mines the Flathub GitHub repository for Flatpak manifests.

Mines ALL manifests regardless of Python content because any manifest can
contain native library modules whose recipes are useful.

Progress tracking:
  A progress file (flathub-progress.json) records every repo already processed.
  Re-running the command resumes from where it stopped, making it safe to
  interrupt and continue without re-downloading anything.

  Format:
    { "processed": ["org.app.One", ...], "last_run": "2025-05-14T..." }

Access strategy:
  - GitHub API (60 req/h unauthenticated, 5000 req/h with GITHUB_TOKEN)
  - Respects rate limits with exponential backoff
  - Raw manifest JSON/YAML cached locally per app-id
  - Set GITHUB_TOKEN env var or use --token for higher rate limits

Usage (standalone, no CI needed)::

    pfmr learn flathub --limit 500
    pfmr learn flathub --limit 500   # resumes, skipping already processed
    pfmr learn flathub --reset       # start over
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

from pfmr.learn.manifest import ManifestAnalyzer, ManifestAnalysis
from pfmr.utils.logging import get_logger

logger = get_logger(__name__)

_GITHUB_API = "https://api.github.com"
_FLATHUB_ORG = "flathub"
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "pfmr" / "flathub"
_PROGRESS_FILENAME = "flathub-progress.json"


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

def _gh_headers(token: Optional[str] = None) -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    tok = token or os.environ.get("GITHUB_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    return headers


def _gh_get(url: str, headers: dict, retries: int = 3) -> Optional[dict | list]:
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=headers, timeout=20)
        except requests.RequestException as exc:
            logger.warning("Request failed (%s): %s", url, exc)
            time.sleep(2 ** attempt)
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 403:
            reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(5, reset - int(time.time())) + 2
            logger.warning("GitHub rate limit; waiting %ds", wait)
            time.sleep(min(wait, 300))
            continue
        if resp.status_code == 404:
            return None
        logger.debug("GitHub API %s → %d", url, resp.status_code)
        time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

class MineProgress:
    """
    Persists the list of already-processed app-ids so runs can be
    interrupted and resumed without re-downloading.

    File: <cache_dir>/flathub-progress.json
    """

    def __init__(self, cache_dir: Path):
        self._path = cache_dir / _PROGRESS_FILENAME
        self._processed: set[str] = set()
        self._load()

    def already_done(self, app_id: str) -> bool:
        return app_id in self._processed

    def mark_done(self, app_id: str) -> None:
        self._processed.add(app_id)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "processed": sorted(self._processed),
            "count": len(self._processed),
            "last_run": datetime.now(timezone.utc).isoformat(),
        }
        self._path.write_text(json.dumps(data, indent=2))

    def reset(self) -> None:
        self._processed.clear()
        if self._path.exists():
            self._path.unlink()

    def count(self) -> int:
        return len(self._processed)

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            self._processed = set(data.get("processed", []))
            logger.debug(
                "Resumed Flathub progress: %d repos already processed",
                len(self._processed),
            )
        except Exception as exc:
            logger.warning("Could not load progress file: %s", exc)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: Path, app_id: str) -> Path:
    return cache_dir / f"{app_id}.json"


def _load_cached(cache_dir: Path, app_id: str) -> Optional[dict]:
    p = _cache_path(cache_dir, app_id)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return None


def _save_cached(cache_dir: Path, app_id: str, data: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    _cache_path(cache_dir, app_id).write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# MineResult
# ---------------------------------------------------------------------------

@dataclass
class MineResult:
    """Summary of a mining run."""
    total_repos: int = 0
    manifests_found: int = 0
    skipped_cached: int = 0
    analyses: list[ManifestAnalysis] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    # Kept for backward compat
    @property
    def python_apps(self) -> int:
        return sum(1 for a in self.analyses if a.python_packages)


# ---------------------------------------------------------------------------
# FlathubMiner
# ---------------------------------------------------------------------------

class FlathubMiner:
    """
    Mines the Flathub GitHub organisation for ALL Flatpak manifests
    (not just Python apps) to extract native library recipes and deps.

    Supports resuming interrupted runs via MineProgress.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        github_token: Optional[str] = None,
        app_id_prefixes: Optional[list[str]] = None,
        force_refresh: bool = False,
    ):
        self.cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self._token = github_token or os.environ.get("GITHUB_TOKEN")
        self._prefixes = app_id_prefixes or []
        self._force = force_refresh
        self._analyzer = ManifestAnalyzer()
        self._headers = _gh_headers(self._token)
        self._progress = MineProgress(self.cache_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mine(
        self,
        limit: int = 200,
        only_python: bool = False,   # default False — mine everything
        progress_save_every: int = 10,
    ) -> MineResult:
        """
        Mine up to `limit` NEW (not yet processed) Flathub repositories.

        Uses MineProgress to skip already-processed repos, so running
        with the same limit repeatedly processes new repos each time.
        Running without --reset lets you process the full Flathub org
        incrementally across many sessions.
        """
        result = MineResult()
        repos = self._list_repos(limit * 3)   # fetch more to account for skips
        result.total_repos = len(repos)

        processed_this_run = 0
        for repo in repos:
            if processed_this_run >= limit:
                break

            app_id = repo.get("name", "")
            if not app_id:
                continue
            if self._prefixes and not any(app_id.startswith(p) for p in self._prefixes):
                continue

            # Skip if already done in a previous run
            if self._progress.already_done(app_id) and not self._force:
                result.skipped_cached += 1
                continue

            analysis = self._mine_repo(app_id, only_python=only_python)
            self._progress.mark_done(app_id)
            processed_this_run += 1

            if analysis is not None:
                result.manifests_found += 1
                result.analyses.append(analysis)
            else:
                result.errors.append(app_id)

            # Persist progress periodically so interruptions don't lose work
            if processed_this_run % progress_save_every == 0:
                self._progress.save()
                logger.debug("Progress saved (%d processed this run)", processed_this_run)

        self._progress.save()
        logger.info(
            "Mining complete: %d new repos, %d manifests, %d skipped (already done)",
            processed_this_run, result.manifests_found, result.skipped_cached,
        )
        return result

    def mine_app(self, app_id: str) -> Optional[ManifestAnalysis]:
        return self._mine_repo(app_id, only_python=False)

    def mine_manifest_url(self, url: str) -> Optional[ManifestAnalysis]:
        resp = requests.get(url, timeout=20)
        if resp.status_code != 200:
            return None
        try:
            if url.endswith(".json"):
                data = resp.json()
            else:
                import yaml as _yaml
                data = _yaml.safe_load(resp.text) or {}
        except Exception as exc:
            logger.warning("Failed to parse manifest from %s: %s", url, exc)
            return None
        return self._analyzer.analyze_dict(data, source=url)

    def reset_progress(self) -> None:
        """Clear the progress file to start over from scratch."""
        self._progress.reset()
        logger.info("Flathub mining progress reset")

    def progress(self) -> MineProgress:
        return self._progress

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _list_repos(self, limit: int) -> list[dict]:
        repos: list[dict] = []
        page = 1
        per_page = min(100, limit)
        while len(repos) < limit:
            url = (
                f"{_GITHUB_API}/orgs/{_FLATHUB_ORG}/repos"
                f"?per_page={per_page}&page={page}&type=public&sort=updated"
            )
            batch = _gh_get(url, self._headers)
            if not batch or not isinstance(batch, list):
                break
            repos.extend(batch)
            if len(batch) < per_page:
                break
            page += 1
        return repos[:limit]

    def _mine_repo(
        self,
        app_id: str,
        only_python: bool = False,
    ) -> Optional[ManifestAnalysis]:
        # Use cached raw data if available and not forcing refresh
        if not self._force:
            cached = _load_cached(self.cache_dir, app_id)
            if cached:
                try:
                    analysis = self._analyzer.analyze_dict(cached, source=f"flathub:{app_id}")
                    if only_python and not analysis.python_packages:
                        return None
                    return analysis
                except Exception:
                    pass

        manifest_data = self._fetch_manifest(app_id)
        if manifest_data is None:
            return None

        if only_python and not self._is_relevant(manifest_data):
            _save_cached(self.cache_dir, app_id, manifest_data)
            return None

        analysis = self._analyzer.analyze_dict(manifest_data, source=f"flathub:{app_id}")
        _save_cached(self.cache_dir, app_id, manifest_data)
        return analysis

    def _fetch_manifest(self, app_id: str) -> Optional[dict]:
        base = f"{_GITHUB_API}/repos/{_FLATHUB_ORG}/{app_id}/contents"
        listing = _gh_get(base, self._headers)
        if not isinstance(listing, list):
            return None

        filenames = {
            f["name"]: f.get("download_url")
            for f in listing
            if isinstance(f, dict) and f.get("download_url")
        }
        for ext in (".json", ".yaml", ".yml"):
            candidate = f"{app_id}{ext}"
            if candidate in filenames:
                url = filenames[candidate]
                try:
                    resp = requests.get(url, timeout=20)
                    if resp.status_code != 200:
                        continue
                    if ext == ".json":
                        return resp.json()
                    import yaml as _yaml
                    return _yaml.safe_load(resp.text) or {}
                except Exception as exc:
                    logger.debug("Parse error for %s: %s", candidate, exc)
        return None

    @staticmethod
    def _is_relevant(manifest: dict) -> bool:
        """
        True if the manifest has native modules or Python content.
        Much broader than the old _has_python check — a manifest with
        any non-trivial module is worth analyzing for recipe extraction.
        """
        modules = manifest.get("modules", [])
        return bool(modules)