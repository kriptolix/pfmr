"""
pfmr.resolvers.uv_resolver
~~~~~~~~~~~~~~~~~~~~~~~~~~
UV Resolver Engine — Phase 1 core component.

Responsibilities:
- Invoke `uv` to resolve dependencies from pyproject.toml / requirements.txt / package name
- Parse the resulting lockfile (uv.lock) into a ResolvedPackage list
- Detect build backends per package
- Identify whether a wheel is available on PyPI
- Produce a deterministic dependency graph
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from packaging.utils import canonicalize_name

from pfmr.models import BuildBackend, ResolvedPackage, ResolutionResult, SourceType
from pfmr.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Build-backend detection heuristics
# ---------------------------------------------------------------------------

_BACKEND_PATTERNS: list[tuple[re.Pattern, BuildBackend]] = [
    (re.compile(r"maturin"), BuildBackend.MATURIN),
    (re.compile(r"meson[_-]python|mesonpy"), BuildBackend.MESON_PYTHON),
    (re.compile(r"setuptools[_-]rust"), BuildBackend.SETUPTOOLS_RUST),
    (re.compile(r"scikit[_-]build[_-]core"), BuildBackend.SCIKIT_BUILD_CORE),
    (re.compile(r"scikit[_-]build"), BuildBackend.SCIKIT_BUILD),
    (re.compile(r"flit"), BuildBackend.FLIT),
    (re.compile(r"pdm"), BuildBackend.PDM),
    (re.compile(r"hatchling|hatch"), BuildBackend.HATCH),
    (re.compile(r"poetry"), BuildBackend.POETRY),
    (re.compile(r"setuptools|setup\.py"), BuildBackend.SETUPTOOLS),
]

_NATIVE_BACKENDS = {
    BuildBackend.MATURIN,
    BuildBackend.MESON_PYTHON,
    BuildBackend.SETUPTOOLS_RUST,
    BuildBackend.SCIKIT_BUILD,
    BuildBackend.SCIKIT_BUILD_CORE,
}

# Packages whose name implies they require native compilation even with
# generic build backends (heuristic, can be extended via recipes).
_KNOWN_NATIVE_PACKAGES = frozenset(
    {
        "cryptography",
        "cffi",
        "lxml",
        "Pillow",
        "pillow",
        "numpy",
        "scipy",
        "pandas",
        "psycopg2",
        "psycopg2-binary",
        "pycurl",
        "greenlet",
        "grpcio",
        "ujson",
        "orjson",
        "msgpack",
        "aiohttp",
        "yarl",
        "multidict",
        "frozenlist",
        "mypy-extensions",
        "pydantic-core",
        "rpds-py",
        "dulwich",
    }
)


def _detect_backend(backend_str: Optional[str]) -> BuildBackend:
    if not backend_str:
        return BuildBackend.SETUPTOOLS
    for pattern, backend in _BACKEND_PATTERNS:
        if pattern.search(backend_str):
            return backend
    return BuildBackend.UNKNOWN


def _requires_native(package_name: str, backend: BuildBackend) -> bool:
    if backend in _NATIVE_BACKENDS:
        return True
    if canonicalize_name(package_name) in {canonicalize_name(p) for p in _KNOWN_NATIVE_PACKAGES}:
        return True
    return False


# ---------------------------------------------------------------------------
# PyPI helpers
# ---------------------------------------------------------------------------

_PYPI_CACHE: dict[str, dict] = {}


def _fetch_pypi_info(name: str, version: str) -> dict:
    key = f"{canonicalize_name(name)}-{version}"
    if key in _PYPI_CACHE:
        return _PYPI_CACHE[key]
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        _PYPI_CACHE[key] = data
        return data
    except Exception as exc:
        logger.warning("Could not fetch PyPI info for %s==%s: %s", name, version, exc)
        return {}


def _get_wheel_info(name: str, version: str) -> tuple[bool, Optional[str], Optional[str]]:
    """
    Returns (wheel_available, url, sha256).
    Prefer cp3XX linux wheel, then manylinux, then pure py3/none-any.
    Falls back to sdist.
    """
    data = _fetch_pypi_info(name, version)
    if not data:
        return False, None, None

    urls = data.get("urls", [])

    # Priority: any linux wheel > pure-python wheel > sdist
    def _wheel_score(u: dict) -> int:
        fn = u.get("filename", "")
        if u.get("packagetype") != "bdist_wheel":
            return -1
        if "linux" in fn or "manylinux" in fn or "musllinux" in fn:
            return 2
        if "none-any" in fn or "py3-none" in fn:
            return 1
        return 0

    wheels = [u for u in urls if u.get("packagetype") == "bdist_wheel"]
    if wheels:
        best = max(wheels, key=_wheel_score)
        digests = best.get("digests", {})
        return True, best.get("url"), digests.get("sha256")

    # fallback to sdist
    sdists = [u for u in urls if u.get("packagetype") == "sdist"]
    if sdists:
        s = sdists[0]
        digests = s.get("digests", {})
        return False, s.get("url"), digests.get("sha256")

    return False, None, None


def _get_pypi_build_backend(name: str, version: str) -> Optional[str]:
    data = _fetch_pypi_info(name, version)
    if not data:
        return None
    info = data.get("info", {})
    # Sometimes PyPI exposes requires_dist which has build metadata
    # More reliable: check the project URLs or classifiers
    # We rely primarily on uv lock metadata; this is a fallback.
    return None


# ---------------------------------------------------------------------------
# uv.lock parser
# ---------------------------------------------------------------------------

def _parse_uv_lock(lock_path: Path) -> list[dict]:
    """
    Parse a uv.lock (TOML format) and return a list of package dicts with
    keys: name, version, build_backend, source.
    """
    with open(lock_path, "rb") as f:
        data = tomllib.load(f)

    packages = []
    for pkg in data.get("package", []):
        entry = {
            "name": pkg.get("name", ""),
            "version": pkg.get("version", ""),
            "build_backend": None,
            "sdist": pkg.get("sdist"),
            "wheels": pkg.get("wheels", []),
            "metadata": pkg.get("metadata", {}),
        }
        # uv.lock may store build-backend in metadata
        meta = pkg.get("metadata", {})
        entry["build_backend"] = meta.get("requires-build", None)
        packages.append(entry)
    return packages


# ---------------------------------------------------------------------------
# Main resolver
# ---------------------------------------------------------------------------

class UVResolver:
    """
    Wraps the `uv` CLI to resolve Python dependencies and produce a
    list of ResolvedPackage objects.
    """

    def __init__(
        self,
        python_version: str = "3.11",
        extra_index_urls: Optional[list[str]] = None,
        offline: bool = False,
    ):
        self.python_version = python_version
        self.extra_index_urls = extra_index_urls or []
        self.offline = offline
        self._uv_bin = self._find_uv()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_from_pyproject(self, pyproject_path: Path) -> ResolutionResult:
        """Resolve dependencies declared in a pyproject.toml file."""
        logger.info("Resolving from pyproject.toml: %s", pyproject_path)
        direct_deps = self._parse_direct_deps_pyproject(pyproject_path)
        return self._run_uv_lock(pyproject_path.parent, direct_deps)

    def resolve_from_requirements(self, requirements_path: Path) -> ResolutionResult:
        """Resolve dependencies from a requirements.txt file."""
        logger.info("Resolving from requirements.txt: %s", requirements_path)
        deps = [
            line.strip()
            for line in requirements_path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        with tempfile.TemporaryDirectory(prefix="pfmr-req-") as tmp:
            tmp_path = Path(tmp)
            # Create a minimal pyproject.toml wrapping the requirements
            pyproject = tmp_path / "pyproject.toml"
            pyproject.write_text(self._minimal_pyproject(deps))
            return self._run_uv_lock(tmp_path, set(canonicalize_name(d.split("==")[0].split(">=")[0].strip()) for d in deps))

    def resolve_package(self, package_spec: str) -> ResolutionResult:
        """
        Resolve a single package (and all its transitive deps) by name/spec.
        E.g.: "requests", "numpy==1.26.4", "django>=4.2"
        """
        logger.info("Resolving package: %s", package_spec)
        with tempfile.TemporaryDirectory(prefix="pfmr-pkg-") as tmp:
            tmp_path = Path(tmp)
            pyproject = tmp_path / "pyproject.toml"
            pyproject.write_text(self._minimal_pyproject([package_spec]))
            name = re.split(r"[>=<!@\[]", package_spec)[0].strip()
            return self._run_uv_lock(tmp_path, {canonicalize_name(name)})

    def resolve_from_lockfile(self, lock_path: Path) -> ResolutionResult:
        """
        Parse an existing uv.lock directly (no re-resolution needed).
        Useful when the project already has a lockfile.
        """
        logger.info("Parsing existing uv.lock: %s", lock_path)
        raw_pkgs = _parse_uv_lock(lock_path)
        return self._build_result(raw_pkgs, direct_names=set())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_uv(self) -> str:
        uv = shutil.which("uv")
        if uv:
            return uv
        # Try common install locations
        candidates = [
            Path.home() / ".cargo" / "bin" / "uv",
            Path("/usr/local/bin/uv"),
            Path("/usr/bin/uv"),
        ]
        for c in candidates:
            if c.exists():
                return str(c)
        raise RuntimeError(
            "uv not found. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
        )

    def _run_uv_lock(self, project_dir: Path, direct_names: set[str]) -> ResolutionResult:
        lock_path = project_dir / "uv.lock"

        cmd = [
            self._uv_bin,
            "lock",
            "--python",
            self.python_version,
        ]
        if self.offline:
            cmd.append("--offline")
        for url in self.extra_index_urls:
            cmd.extend(["--extra-index-url", url])

        logger.debug("Running: %s", " ".join(str(c) for c in cmd))
        result = subprocess.run(
            cmd,
            cwd=project_dir,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"uv lock failed:\n{result.stderr}"
            )

        raw_pkgs = _parse_uv_lock(lock_path)
        return self._build_result(raw_pkgs, direct_names)

    def _build_result(self, raw_pkgs: list[dict], direct_names: set[str]) -> ResolutionResult:
        packages: list[ResolvedPackage] = []
        for raw in raw_pkgs:
            name = raw["name"]
            version = raw["version"]

            # Determine build backend
            backend_str = raw.get("build_backend") or ""
            backend = _detect_backend(backend_str)

            # Check wheel availability from lock data first, then PyPI
            wheels = raw.get("wheels", [])
            wheel_available = bool(wheels)
            source_url: Optional[str] = None
            source_hash: Optional[str] = None
            source_type: Optional[SourceType] = None

            if wheel_available and wheels:
                w = wheels[0]
                source_url = w.get("url")
                source_hash = w.get("hash", "").replace("sha256:", "")
                source_type = SourceType.WHEEL
            else:
                sdist = raw.get("sdist")
                if sdist:
                    source_url = sdist.get("url")
                    source_hash = sdist.get("hash", "").replace("sha256:", "")
                    source_type = SourceType.SDIST

            # If we don't have source info from lock, fall back to PyPI API
            if not source_url:
                wheel_available, source_url, source_hash = _get_wheel_info(name, version)
                source_type = SourceType.WHEEL if wheel_available else SourceType.SDIST

            # If still no backend from lock, try to infer from PyPI classifiers
            if backend == BuildBackend.UNKNOWN:
                pypi_backend = _get_pypi_build_backend(name, version)
                if pypi_backend:
                    backend = _detect_backend(pypi_backend)

            pkg = ResolvedPackage(
                name=name,
                version=version,
                wheel_available=wheel_available,
                build_backend=backend,
                requires_native=_requires_native(name, backend),
                is_direct=canonicalize_name(name) in direct_names,
                source_url=source_url,
                source_hash=source_hash,
                source_type=source_type,
            )
            packages.append(pkg)
            logger.debug(
                "  %s==%s  wheel=%s  backend=%s  native=%s",
                name, version, wheel_available, backend.value, pkg.requires_native,
            )

        # Compute a lockfile hash for cache keying
        lock_hash = hashlib.sha256(
            json.dumps(
                [{"name": p.name, "version": p.version} for p in packages],
                sort_keys=True,
            ).encode()
        ).hexdigest()

        result = ResolutionResult(packages=packages, lockfile_hash=lock_hash)
        logger.info(
            "Resolved %d packages (%d need native compilation)",
            len(packages),
            sum(1 for p in packages if p.requires_native),
        )
        return result

    @staticmethod
    def _parse_direct_deps_pyproject(path: Path) -> set[str]:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        deps: set[str] = set()
        project_deps = data.get("project", {}).get("dependencies", [])
        for dep in project_deps:
            name = re.split(r"[>=<!@\[ ]", dep)[0].strip()
            deps.add(canonicalize_name(name))
        return deps

    @staticmethod
    def _minimal_pyproject(deps: list[str]) -> str:
        dep_list = "\n".join(f'  "{d}",' for d in deps)
        return f"""[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[project]
name = "pfmr-resolve-target"
version = "0.0.1"
dependencies = [
{dep_list}
]
"""
