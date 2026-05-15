"""
pfmr.learn.flathub
~~~~~~~~~~~~~~~~~~~
FlathubMiner — mines the Flathub GitHub repository for Flatpak manifests
and extracts knowledge about Python packages and their native dependencies.

Access strategy:
  - GitHub API (no auth needed for public repos, 60 req/h unauthenticated)
  - With GITHUB_TOKEN: 5000 req/h
  - Respects rate limits with exponential backoff
  - Results cached locally so repeated runs don't re-download unchanged manifests

Flathub repository structure:
  github.com/flathub/<app-id>/  (one repo per app)
  or github.com/flathub/flathub/ (monorepo, recent)

The miner searches for:
  - Files matching *.json / *.yaml / *.yml in the repo root
  - Identifies manifests by presence of "app-id" / "modules" keys
  - Prioritises apps that have Python-related modules

Output:
  list[ManifestAnalysis] — one per successfully mined manifest

Usage (standalone)::

    miner = FlathubMiner(cache_dir=Path("~/.cache/pfmr/flathub"))
    analyses = miner.mine(limit=100)
    for analysis in analyses:
        print(analysis.app_id, analysis.python_packages)
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests

from pfmr.learn.manifest import ManifestAnalyzer, ManifestAnalysis
from pfmr.utils.logging import get_logger

logger = get_logger(__name__)

_GITHUB_API = "https://api.github.com"
_FLATHUB_ORG = "flathub"
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "pfmr" / "flathub"

# Python-related keywords that indicate a manifest is worth mining
_PYTHON_SIGNALS = frozenset({
    "python3", "python", "pip", "uv pip", "site-packages",
    "maturin", "setuptools", "scikit-build", "meson-python",
})


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
    """GET a GitHub API URL with retry on rate limit."""
    for attempt in range(retries):
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 403:
            reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(1, reset - int(time.time())) + 1
            logger.warning("GitHub rate limit hit; waiting %ds", wait)
            time.sleep(min(wait, 120))
            continue
        if resp.status_code == 404:
            return None
        logger.debug("GitHub API %s → %d", url, resp.status_code)
        time.sleep(2 ** attempt)
    return None


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
# FlathubMiner
# ---------------------------------------------------------------------------

@dataclass
class MineResult:
    """Summary of a mining run."""
    total_repos: int = 0
    manifests_found: int = 0
    python_apps: int = 0
    analyses: list[ManifestAnalysis] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class FlathubMiner:
    """
    Mines the Flathub GitHub organisation for Flatpak manifests that contain
    Python packages, and returns ManifestAnalysis objects.

    Completely standalone — no pfmr.pipeline dependency.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        github_token: Optional[str] = None,
        # Only process repos whose name matches any of these prefixes
        app_id_prefixes: Optional[list[str]] = None,
        # Force re-download even if cached
        force_refresh: bool = False,
    ):
        self.cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self._token = github_token or os.environ.get("GITHUB_TOKEN")
        self._prefixes = app_id_prefixes or []
        self._force = force_refresh
        self._analyzer = ManifestAnalyzer()
        self._headers = _gh_headers(self._token)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mine(
        self,
        limit: int = 200,
        only_python: bool = True,
    ) -> MineResult:
        """
        Mine up to `limit` Flathub repositories.

        Args:
            limit:       Maximum number of repos to inspect.
            only_python: If True, skip manifests that don't reference Python.

        Returns:
            MineResult with all discovered ManifestAnalysis objects.
        """
        result = MineResult()
        repos = self._list_repos(limit)
        result.total_repos = len(repos)
        logger.info("Flathub mining: %d repos to inspect", len(repos))

        for repo in repos:
            app_id = repo.get("name", "")
            if self._prefixes and not any(app_id.startswith(p) for p in self._prefixes):
                continue

            analysis = self._mine_repo(app_id, only_python=only_python)
            if analysis is not None:
                result.manifests_found += 1
                if analysis.python_packages:
                    result.python_apps += 1
                result.analyses.append(analysis)

        logger.info(
            "Mining complete: %d manifests, %d with Python packages",
            result.manifests_found, result.python_apps,
        )
        return result

    def mine_app(self, app_id: str) -> Optional[ManifestAnalysis]:
        """Mine a single Flathub app by its app-id."""
        return self._mine_repo(app_id, only_python=False)

    def mine_manifest_url(self, url: str, app_id: str = "") -> Optional[ManifestAnalysis]:
        """
        Download and analyze a manifest at an arbitrary GitHub raw URL.
        Useful for testing against specific known apps.
        """
        resp = requests.get(url, timeout=15)
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

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _list_repos(self, limit: int) -> list[dict]:
        """List repositories in the flathub GitHub organisation."""
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
        only_python: bool = True,
    ) -> Optional[ManifestAnalysis]:
        """Download and analyze the manifest from a single Flathub repo."""
        # Try cache first
        if not self._force:
            cached = _load_cached(self.cache_dir, app_id)
            if cached:
                try:
                    return self._analysis_from_cache(cached)
                except Exception:
                    pass

        # Get repo tree to find the manifest file
        manifest_data = self._fetch_manifest(app_id)
        if manifest_data is None:
            return None

        # Quick Python relevance check
        if only_python and not self._has_python(manifest_data):
            return None

        analysis = self._analyzer.analyze_dict(
            manifest_data,
            source=f"flathub:{app_id}",
        )

        # Cache the raw manifest data
        _save_cached(self.cache_dir, app_id, manifest_data)
        return analysis

    def _fetch_manifest(self, app_id: str) -> Optional[dict]:
        """
        Try to download the main manifest from the Flathub repo.
        Tries: <app-id>.json → <app-id>.yaml → <app-id>.yml
        """
        base = f"{_GITHUB_API}/repos/{_FLATHUB_ORG}/{app_id}/contents"
        listing = _gh_get(base, self._headers)
        if not isinstance(listing, list):
            return None

        filenames = {f["name"]: f["download_url"] for f in listing if isinstance(f, dict)}
        for ext in (".json", ".yaml", ".yml"):
            candidate = f"{app_id}{ext}"
            if candidate in filenames:
                url = filenames[candidate]
                resp = requests.get(url, timeout=15)
                if resp.status_code != 200:
                    continue
                try:
                    if ext == ".json":
                        return resp.json()
                    import yaml as _yaml
                    return _yaml.safe_load(resp.text) or {}
                except Exception as exc:
                    logger.debug("Parse error for %s: %s", candidate, exc)
        return None

    @staticmethod
    def _has_python(manifest: dict) -> bool:
        """Quick check: does the manifest reference Python anywhere?"""
        text = json.dumps(manifest).lower()
        return any(sig in text for sig in _PYTHON_SIGNALS)

    @staticmethod
    def _analysis_from_cache(data: dict) -> ManifestAnalysis:
        return ManifestAnalyzer().analyze_dict(data, source="cache")