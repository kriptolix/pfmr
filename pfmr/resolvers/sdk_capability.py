"""
pfmr.resolvers.sdk_capability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SDKCapabilityResolver — Phase 2 component.

Objetivo (spec §5.3):
  Entender se dependências nativas já existem no ambiente Flatpak,
  evitando recompilação desnecessária de libs que o SDK já provê.

Fontes de dados (em ordem de prioridade):

  Primárias (live — requerem flatpak instalado na máquina host):
    1. `flatpak info <sdk-id>//<version>` — metadados do runtime
    2. scan de /usr/lib/pkgconfig e /usr/lib/sdk/<ext>/lib/pkgconfig
       dentro do SDK via `flatpak run --command=...`
    3. `pkg-config --list-all` dentro do SDK

  Secundárias (estáticas — sempre disponíveis):
    4. Perfis TOML em data/sdk-profiles/<sdk-id>/<version>.toml
    5. Cache de perfis gerado por probe anterior (~/.cache/pfmr/sdk-profiles/)

Fluxo de resolução:
  SDKCapabilityResolver.resolve(sdk_id, sdk_version, queries)
      ↓
  _load_capability(sdk_id, sdk_version)
      ├── tenta probe live → SDKCapability(probed_live=True)
      └── fallback para perfil estático → SDKCapability(probed_live=False)
      ↓
  _check_query(capability, query) × N
      ↓
  SDKResolutionReport
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pfmr.models import (
    SDKCapability,
    SDKCheckResult,
    SDKResolutionReport,
    NativeRecipe,
)
from pfmr.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Built-in static profiles shipped with pfmr
_BUILTIN_PROFILES_DIR = Path(__file__).parent.parent / "data" / "sdk-profiles"

# User cache — populated after a successful live probe
_CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "pfmr" / "sdk-profiles"


# ---------------------------------------------------------------------------
# Query descriptor
# ---------------------------------------------------------------------------

@dataclass
class SDKQuery:
    """
    A single capability query: "does this SDK provide X?"

    query_type:
      "library"    — shared library by soname, e.g. "libssl.so.3"
      "pkgconfig"  — pkg-config module name, e.g. "openssl"
      "header"     — C header path, e.g. "openssl/ssl.h"
      "executable" — binary in PATH, e.g. "python3"
    """
    value: str
    query_type: str = "pkgconfig"    # most common query type from build systems
    # optional: which Python package triggered this query
    origin_package: Optional[str] = None


# ---------------------------------------------------------------------------
# Live prober
# ---------------------------------------------------------------------------

class _LiveProber:
    """
    Executes read-only commands inside a Flatpak SDK to enumerate its capabilities.
    Falls back gracefully when flatpak is not installed.
    """

    def __init__(self, sdk_id: str, sdk_version: str):
        self.sdk_id = sdk_id
        self.sdk_version = sdk_version
        self._flatpak = shutil.which("flatpak")
        self._available: Optional[bool] = None

    @property
    def is_available(self) -> bool:
        if self._available is None:
            self._available = self._check_sdk_installed()
        return self._available

    def probe(self) -> Optional[SDKCapability]:
        """
        Probe the SDK live. Returns None if flatpak is unavailable or
        the SDK is not installed.
        """
        if not self.is_available:
            logger.debug(
                "Live probe skipped: flatpak not available or SDK %s//%s not installed",
                self.sdk_id, self.sdk_version,
            )
            return None

        logger.info("Probing SDK live: %s//%s", self.sdk_id, self.sdk_version)
        cap = SDKCapability(
            sdk_id=self.sdk_id,
            sdk_version=self.sdk_version,
            probed_live=True,
        )

        cap.pkgconfig = self._probe_pkgconfig()
        cap.libraries = self._probe_libraries()
        cap.executables = self._probe_executables()

        logger.info(
            "Live probe result: %d pc, %d libs, %d exes",
            len(cap.pkgconfig), len(cap.libraries), len(cap.executables),
        )
        return cap

    # ------------------------------------------------------------------

    def _flatpak_run(self, command: str) -> Optional[str]:
        """Run a shell command inside the SDK and return stdout, or None on failure."""
        if not self._flatpak:
            return None
        cmd = [
            self._flatpak, "run",
            "--command=sh",
            f"--runtime={self.sdk_id}/{self._arch()}/{self.sdk_version}",
            "--share=none", "--socket=none", "--device=none",
            "--nofilesystem=host",
            self.sdk_id,   # app-id placeholder (ignored with --runtime)
            "-c", command,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                return result.stdout
        except (subprocess.TimeoutExpired, FileNotFoundError, PermissionError) as e:
            logger.debug("flatpak run failed: %s", e)
        return None

    def _flatpak_builder_run(self, command: str) -> Optional[str]:
        """
        Alternative: use flatpak-builder --run to enter a build environment.
        More reliable than flatpak run for SDK inspection.
        """
        fb = shutil.which("flatpak-builder")
        if not fb:
            return None
        import tempfile, json
        with tempfile.TemporaryDirectory(prefix="pfmr-probe-") as tmp:
            tmp_path = Path(tmp)
            build_dir = tmp_path / "build"
            build_dir.mkdir()
            manifest = {
                "app-id": "org.pfmr.Probe",
                "runtime": self.sdk_id.replace("Sdk", "Platform"),
                "runtime-version": self.sdk_version,
                "sdk": self.sdk_id,
                "modules": [],
            }
            manifest_path = tmp_path / "probe-manifest.json"
            import json as _json
            manifest_path.write_text(_json.dumps(manifest))
            cmd = [fb, "--run", str(build_dir), str(manifest_path), "sh", "-c", command]
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=60
                )
                if result.returncode == 0:
                    return result.stdout
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                logger.debug("flatpak-builder --run failed: %s", e)
        return None

    def _probe_pkgconfig(self) -> list[str]:
        """List all pkg-config modules available in the SDK."""
        output = self._run_in_sdk("pkg-config --list-all 2>/dev/null | awk '{print $1}'")
        if not output:
            return []
        return [line.strip() for line in output.splitlines() if line.strip()]

    def _probe_libraries(self) -> list[str]:
        """List .so files in the SDK's lib directories."""
        cmd = (
            "find /usr/lib /usr/lib64 /lib /lib64 "
            r"-name '*.so*' -type f 2>/dev/null | "
            r"sed 's|.*/||' | sort -u"
        )
        output = self._run_in_sdk(cmd)
        if not output:
            return []
        return [line.strip() for line in output.splitlines() if line.strip()]

    def _probe_executables(self) -> list[str]:
        """List executables in the SDK's bin directories."""
        cmd = "ls /usr/bin /usr/local/bin 2>/dev/null | sort -u"
        output = self._run_in_sdk(cmd)
        if not output:
            return []
        return [line.strip() for line in output.splitlines() if line.strip()]

    def _run_in_sdk(self, command: str) -> Optional[str]:
        """Try flatpak-builder first (more reliable), then flatpak run."""
        out = self._flatpak_builder_run(command)
        if out is not None:
            return out
        return self._flatpak_run(command)

    def _check_sdk_installed(self) -> bool:
        if not self._flatpak:
            return False
        try:
            result = subprocess.run(
                [self._flatpak, "info", f"{self.sdk_id}//{self.sdk_version}"],
                capture_output=True, text=True, timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    @staticmethod
    def _arch() -> str:
        import platform
        machine = platform.machine()
        return {"x86_64": "x86_64", "aarch64": "aarch64", "arm": "arm"}.get(machine, "x86_64")


# ---------------------------------------------------------------------------
# Profile loader (static + cache)
# ---------------------------------------------------------------------------

def _profile_path(sdk_id: str, sdk_version: str, base_dir: Path) -> Path:
    """
    Profiles are stored as:
      <base_dir>/<sdk_id>/<sdk_version>.toml
    e.g.:
      data/sdk-profiles/org.freedesktop.Sdk/24.08.toml
    """
    safe_id = sdk_id.replace("/", "_")
    safe_ver = sdk_version.replace("/", "_")
    return base_dir / safe_id / f"{safe_ver}.toml"


def _load_profile(sdk_id: str, sdk_version: str, base_dir: Path) -> Optional[SDKCapability]:
    path = _profile_path(sdk_id, sdk_version, base_dir)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
        cap = SDKCapability(
            sdk_id=data.get("sdk_id", sdk_id),
            sdk_version=data.get("sdk_version", sdk_version),
            libraries=data.get("libraries", []),
            pkgconfig=data.get("pkgconfig", []),
            headers=data.get("headers", []),
            executables=data.get("executables", []),
            python_modules=data.get("python_modules", []),
            probed_live=False,
        )
        logger.debug(
            "Loaded static profile %s//%s from %s (%d pc, %d libs)",
            sdk_id, sdk_version, path,
            len(cap.pkgconfig), len(cap.libraries),
        )
        return cap
    except Exception as exc:
        logger.warning("Failed to load SDK profile %s: %s", path, exc)
        return None


def _save_profile(cap: SDKCapability) -> None:
    """Persist a live-probed capability to the user cache."""
    path = _profile_path(cap.sdk_id, cap.sdk_version, _CACHE_DIR)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f'sdk_id = "{cap.sdk_id}"',
        f'sdk_version = "{cap.sdk_version}"',
        "",
        "libraries = [",
        *[f'  "{lib}",' for lib in sorted(cap.libraries)],
        "]",
        "",
        "pkgconfig = [",
        *[f'  "{pc}",' for pc in sorted(cap.pkgconfig)],
        "]",
        "",
        "headers = [",
        *[f'  "{h}",' for h in sorted(cap.headers)],
        "]",
        "",
        "executables = [",
        *[f'  "{e}",' for e in sorted(cap.executables)],
        "]",
        "",
        "python_modules = [",
        *[f'  "{m}",' for m in sorted(cap.python_modules)],
        "]",
    ]
    path.write_text("\n".join(lines) + "\n")
    logger.info("Saved live probe cache to %s", path)


# ---------------------------------------------------------------------------
# Public resolver
# ---------------------------------------------------------------------------

class SDKCapabilityResolver:
    """
    Resolves whether native dependencies are already satisfied by the
    target Flatpak SDK/runtime, avoiding unnecessary module rebuilds.

    Usage::

        resolver = SDKCapabilityResolver(
            sdk_id="org.freedesktop.Sdk",
            sdk_version="24.08",
        )
        report = resolver.resolve(queries)
        for miss in report.missing:
            print(f"  ✗ {miss.query}  ({miss.query_type})")
    """

    def __init__(
        self,
        sdk_id: str = "org.freedesktop.Sdk",
        sdk_version: str = "24.08",
        # Extra SDKs to check (extensions, runtime, etc.)
        extra_sdk_ids: Optional[list[str]] = None,
        # Force live probe even when a cache entry exists
        force_probe: bool = False,
        # Skip live probe entirely (offline / CI mode)
        offline: bool = False,
        # Additional profile directories to search
        extra_profile_dirs: Optional[list[Path]] = None,
        # Recipe DB for fallback candidates when a dep is missing
        recipe_db=None,
    ):
        self.sdk_id = sdk_id
        self.sdk_version = sdk_version
        self.extra_sdk_ids = extra_sdk_ids or []
        self.force_probe = force_probe
        self.offline = offline
        self.extra_profile_dirs = extra_profile_dirs or []
        self.recipe_db = recipe_db

        # Eagerly load capabilities for all SDKs
        self._capabilities: dict[str, SDKCapability] = {}
        self._load_all()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(self, queries: list[SDKQuery]) -> SDKResolutionReport:
        """
        Check each query against all loaded SDK capabilities.
        Returns a report with per-query results.
        """
        report = SDKResolutionReport(
            sdk_id=self.sdk_id,
            sdk_version=self.sdk_version,
        )
        for q in queries:
            check = self._check_query(q)
            report.checks.append(check)

        logger.info(
            "SDK check: %d satisfied, %d missing (against %s//%s)",
            len(report.satisfied), len(report.missing),
            self.sdk_id, self.sdk_version,
        )
        return report

    def resolve_native_deps(
        self,
        native_deps: list[str],
        query_type: str = "pkgconfig",
        origin_package: Optional[str] = None,
    ) -> SDKResolutionReport:
        """
        Convenience wrapper: takes a flat list of dep names and resolves them.
        """
        queries = [
            SDKQuery(value=dep, query_type=query_type, origin_package=origin_package)
            for dep in native_deps
        ]
        return self.resolve(queries)

    def capability(self, sdk_id: Optional[str] = None) -> Optional[SDKCapability]:
        """Return the loaded SDKCapability for a given sdk_id (default: primary)."""
        return self._capabilities.get(sdk_id or self.sdk_id)

    def all_capabilities(self) -> list[SDKCapability]:
        return list(self._capabilities.values())

    def refresh(self) -> None:
        """Force a fresh live probe and reload all capabilities."""
        self.force_probe = True
        self._capabilities.clear()
        self._load_all()

    # ------------------------------------------------------------------
    # Internal: loading
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        all_ids = [self.sdk_id] + self.extra_sdk_ids
        for sid in all_ids:
            cap = self._load_capability(sid, self.sdk_version)
            if cap:
                self._capabilities[sid] = cap

    def _load_capability(self, sdk_id: str, sdk_version: str) -> Optional[SDKCapability]:
        """
        Load SDK capability using the priority chain:
          1. Live probe (if not offline and not cached or force_probe)
          2. User cache (~/.cache/pfmr/sdk-profiles/)
          3. Built-in static profiles (data/sdk-profiles/)
          4. Extra profile dirs
        """
        # Step 1: live probe
        if not self.offline:
            cached = _load_profile(sdk_id, sdk_version, _CACHE_DIR)
            if cached and not self.force_probe:
                logger.debug("Using cached profile for %s//%s", sdk_id, sdk_version)
                return cached

            prober = _LiveProber(sdk_id, sdk_version)
            live = prober.probe()
            if live:
                _save_profile(live)
                return live

        # Step 2: user cache
        cached = _load_profile(sdk_id, sdk_version, _CACHE_DIR)
        if cached:
            return cached

        # Step 3: built-in static profiles
        builtin = _load_profile(sdk_id, sdk_version, _BUILTIN_PROFILES_DIR)
        if builtin:
            return builtin

        # Step 4: extra dirs
        for extra_dir in self.extra_profile_dirs:
            extra = _load_profile(sdk_id, sdk_version, extra_dir)
            if extra:
                return extra

        logger.warning(
            "No SDK profile found for %s//%s (no live flatpak, no static profile). "
            "Run `pfmr sdk probe` to generate one.",
            sdk_id, sdk_version,
        )
        return None

    # ------------------------------------------------------------------
    # Internal: checking
    # ------------------------------------------------------------------

    def _check_query(self, query: SDKQuery) -> SDKCheckResult:
        """Check a single query against all loaded capabilities."""
        for sdk_id, cap in self._capabilities.items():
            if self._matches(cap, query):
                return SDKCheckResult(
                    query=query.value,
                    query_type=query.query_type,
                    satisfied=True,
                    provided_by=sdk_id,
                )

        # Not satisfied — look up recipe candidates
        candidates = self._find_recipe_candidates(query.value)
        return SDKCheckResult(
            query=query.value,
            query_type=query.query_type,
            satisfied=False,
            provided_by=None,
            recipe_candidates=candidates,
        )

    @staticmethod
    def _matches(cap: SDKCapability, query: SDKQuery) -> bool:
        if query.query_type == "library":
            return cap.provides_library(query.value)
        if query.query_type == "pkgconfig":
            return cap.provides_pkgconfig(query.value)
        if query.query_type == "header":
            return cap.provides_header(query.value)
        if query.query_type == "executable":
            return cap.provides_executable(query.value)
        # fallback: try all
        return (
            cap.provides_pkgconfig(query.value)
            or cap.provides_library(query.value)
            or cap.provides_executable(query.value)
        )

    def _find_recipe_candidates(self, hint: str) -> list[str]:
        if not self.recipe_db:
            return []
        recipe = self.recipe_db.find(hint)
        return [recipe.id] if recipe else []