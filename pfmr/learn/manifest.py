"""
pfmr.learn.manifest
~~~~~~~~~~~~~~~~~~~~
ManifestAnalyzer — extracts knowledge from Flatpak manifests.

Understands JSON and YAML manifests (Flathub format). Identifies:

  - Python packages installed via pip/uv
  - Native library modules (autotools / cmake / meson buildsystems)
  - SDK extensions declared
  - Build environment (runtime, sdk, sdk-version)
  - Source URLs and checksums for native modules

App module detection
--------------------
Flatpak manifests conventionally place the application itself as the LAST
module. This module has a "dir" source (the local source tree) and is NOT
a reusable dependency. The analyzer detects and skips it to avoid polluting
the recipe database with app-specific modules.

Detection logic (in order):
  1. Module name is in mappings.toml [skip_module_names]         → skip
  2. Module is the last in the list AND has a "dir" source type  → skip (app)
  3. Module name is in mappings.toml [app_module_indicators.always_app_names] → skip
  4. Module name is in [app_module_indicators.never_app_names]   → keep

Name mappings
-------------
All name correspondence tables are loaded from pfmr/data/mappings.toml
via pfmr.data.mappings.MAPPINGS — no inline dicts in this file.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from pfmr.data.mappings import MAPPINGS
from pfmr.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class LearnedNativeModule:
    """A native library module extracted from a manifest."""
    module_name: str
    buildsystem: str
    source_url: Optional[str] = None
    source_sha256: Optional[str] = None
    pkgconfig_names: list[str] = field(default_factory=list)
    provides_sonames: list[str] = field(default_factory=list)
    config_opts: list[str] = field(default_factory=list)
    cleanup: list[str] = field(default_factory=list)


@dataclass
class ManifestAnalysis:
    """Result of analyzing a single Flatpak manifest."""
    app_id: str
    runtime: str
    sdk: str
    sdk_version: str
    sdk_extensions: list[str] = field(default_factory=list)
    python_packages: list[str] = field(default_factory=list)
    native_modules: list[LearnedNativeModule] = field(default_factory=list)
    source_path: str = ""


# ---------------------------------------------------------------------------
# pip/uv install command parser
# ---------------------------------------------------------------------------

_PIP_INSTALL_RE = re.compile(
    r"(?:pip(?:3)?|uv\s+pip)\s+install\s+(?:--[^\s]+\s+)*(?P<specs>[^&;|]+)",
    re.IGNORECASE,
)
_PKG_SPEC_RE = re.compile(r"(?P<name>[A-Za-z0-9_\-\.]+)(?:[>=<!][^\s,]+)?")

_PIP_NOISE = frozenset(
    "install pip uv python python3 wheel setuptools no index find links".split()
)


def _module_is_pip_only(mod: dict) -> bool:
    """
    Return True if a module's only purpose is to install Python packages
    (pip/uv install commands, buildsystem=simple, no archive sources).

    Such modules are dependencies, not the application itself — even when
    they are the sole module in a manifest.
    """
    if mod.get("buildsystem") != "simple":
        return False
    # Must have no archive source (those would be native deps)
    source_types = [
        s.get("type", "")
        for s in mod.get("sources", [])
        if isinstance(s, dict)
    ]
    if "archive" in source_types or "dir" in source_types:
        return False
    # Must have at least one pip/uv install command
    cmds = mod.get("build-commands", [])
    for cmd in cmds:
        cmd_str = _command_to_str(cmd)
        if _PIP_INSTALL_RE.search(cmd_str):
            return True
    return False


def _command_to_str(command) -> str:
    """
    Normalise a build-commands entry to a plain string.

    Flatpak manifests allow two forms:
      - "pip install foo"               (string)
      - {"pip install foo": null}       (dict — conditional command)
      - {"pip install foo": "cond"}     (dict — conditional command)

    In the dict form the command string is the single key.
    Any other type returns an empty string so callers can safely skip it.
    """
    if isinstance(command, str):
        return command
    if isinstance(command, dict) and command:
        return next(iter(command))
    return ""


def _parse_pip_packages(command) -> list[str]:
    cmd_str = _command_to_str(command)
    if not cmd_str:
        return []
    packages: list[str] = []
    for m in _PIP_INSTALL_RE.finditer(cmd_str):
        for spec_m in _PKG_SPEC_RE.finditer(m.group("specs").strip()):
            name = spec_m.group("name").strip()
            if not name.startswith("-") and name.lower() not in _PIP_NOISE and len(name) > 1:
                packages.append(name)
    return packages


# ---------------------------------------------------------------------------
# ManifestAnalyzer
# ---------------------------------------------------------------------------

class ManifestAnalyzer:
    """
    Analyzes Flatpak manifests and extracts reusable knowledge.

    Standalone — no pfmr.pipeline dependency.

    Usage::

        analyzer = ManifestAnalyzer()
        analysis = analyzer.analyze(Path("org.gnome.App.json"))
        for mod in analysis.native_modules:
            print(mod.module_name, mod.pkgconfig_names)
    """

    def analyze(self, manifest_path: Path) -> Optional[ManifestAnalysis]:
        """Analyze a manifest file. Returns None on parse failure."""
        try:
            data = self._load(manifest_path)
        except Exception as exc:
            logger.warning("Failed to load manifest %s: %s", manifest_path, exc)
            return None
        return self._analyze_dict(data, source=str(manifest_path))

    def analyze_directory(
        self,
        directory: Path,
        recursive: bool = True,
    ) -> list[ManifestAnalysis]:
        """
        Analyze all manifest files in a directory.

        Scans for *.json, *.yaml, *.yml and attempts to parse each as a
        Flatpak manifest. Files that don't look like manifests are silently
        skipped. Shared-modules files (individual modules without app-id)
        are also silently skipped — use SharedModulesImporter for those.
        """
        results: list[ManifestAnalysis] = []
        patterns = ["**/*.json", "**/*.yaml", "**/*.yml"] if recursive \
                   else ["*.json", "*.yaml", "*.yml"]
        seen: set[str] = set()

        for pattern in patterns:
            for p in sorted(directory.glob(pattern)):
                key = str(p.resolve())
                if key in seen:
                    continue
                seen.add(key)
                try:
                    data = self._load(p)
                except Exception:
                    continue
                if not isinstance(data, dict):
                    continue
                # Must look like a full manifest (not a bare module file)
                if not ("app-id" in data or "id" in data):
                    continue
                if "modules" not in data:
                    continue
                analysis = self._analyze_dict(data, source=str(p))
                if analysis:
                    results.append(analysis)

        logger.info("Analyzed %d manifests in %s", len(results), directory)
        return results

    def analyze_dict(self, data: dict, source: str = "") -> ManifestAnalysis:
        return self._analyze_dict(data, source=source)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _load(path: Path) -> dict:
        text = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            return yaml.safe_load(text) or {}
        return json.loads(text)

    def _analyze_dict(self, data: dict, source: str = "") -> ManifestAnalysis:
        analysis = ManifestAnalysis(
            app_id=data.get("app-id", data.get("id", "")),
            runtime=data.get("runtime", ""),
            sdk=data.get("sdk", ""),
            sdk_version=str(data.get("runtime-version", "")),
            sdk_extensions=data.get("sdk-extensions", []),
            source_path=source,
        )
        modules = data.get("modules", [])
        self._process_modules(modules, analysis)
        return analysis

    def _process_modules(
        self,
        modules: list,
        analysis: ManifestAnalysis,
    ) -> None:
        # Work on a flat list to have correct index / total context
        flat = [m for m in modules if isinstance(m, dict)]
        total = len(flat)

        for idx, mod in enumerate(flat):
            name = mod.get("name", "")
            source_types = [
                s.get("type", "")
                for s in mod.get("sources", [])
                if isinstance(s, dict)
            ]

            # Skip bootstrap/meta modules regardless of position
            if MAPPINGS.should_skip_module(name):
                logger.debug("Skipping meta module: %s", name)
                continue

            # Recurse into sub-modules BEFORE deciding about the parent.
            # Sub-modules are always dependency modules — even if the parent
            # module turns out to be the app itself, its children are deps.
            for sub in mod.get("modules", []):
                if isinstance(sub, dict):
                    self._process_modules([sub], analysis)

            # Decide whether this module is the application itself
            if self._is_app_module(
                mod=mod,
                name=name,
                idx=idx,
                total=total,
                source_types=source_types,
                app_id=analysis.app_id,
            ):
                logger.debug(
                    "Skipping app module: %s (idx=%d/%d sources=%s)",
                    name, idx + 1, total, source_types,
                )
                continue

            buildsystem = mod.get("buildsystem", "autotools")
            if buildsystem == "simple":
                self._process_simple_module(mod, analysis)
            else:
                self._process_native_module(mod, analysis)

    @staticmethod
    def _is_app_module(
        mod: dict,
        name: str,
        idx: int,
        total: int,
        source_types: list[str],
        app_id: str,
    ) -> bool:
        """
        Decide whether a module is the application being packaged (not a dep).

        Rules (checked in order, first match wins):

        NEVER-app overrides (always treat as dependency):
          1. Name is in MAPPINGS never_app_names (known dep libs)
          2. Name is in MAPPINGS always_app_names (explicit app markers)  ← skip
          (handled by MAPPINGS.is_app_module below for explicit lists)

        ALWAYS-app heuristics:
          3. Single module in manifest → must be the app itself.
          4. Last module AND has a "dir" source (local source tree).
          5. Last module AND module name matches the app-id suffix
             (e.g. app_id="org.gnome.Fractal", name="fractal").
          6. Last module AND no "archive" source anywhere (i.e. nothing to
             download — it is the code already on disk).
          7. MAPPINGS explicit override (always_app_names / never_app_names).

        Note: sub-modules (modules inside a module) are processed separately
        and are always treated as deps — this function is never called for them.
        """
        name_low = name.lower()

        # Rule 1 — explicit never-app names (known dep libraries)
        if MAPPINGS._never_app_names and name_low in MAPPINGS._never_app_names:
            return False

        # Rule 2 — explicit always-app names
        if MAPPINGS._always_app_names and name_low in MAPPINGS._always_app_names:
            return True

        is_last = (idx == total - 1)
        has_dir_source = "dir" in source_types
        has_archive_source = "archive" in source_types

        # Rule 3 — only module in the manifest
        # Exception: a single "simple" module whose commands only do pip/uv
        # installs is a Python-deps module, not the app.
        if total == 1:
            if _module_is_pip_only(mod):
                return False  # treat as a deps module
            return True

        # Rule 4 — last module with a "dir" source (canonical app pattern)
        if is_last and has_dir_source:
            return True

        # Rule 5 — last module whose name matches the app-id tail
        if is_last and app_id:
            app_tail = app_id.split(".")[-1].lower()
            if name_low == app_tail or name_low.replace("-", "") == app_tail.replace("-", ""):
                return True

        # Rule 6 — last module with no downloadable source at all
        # (pure local source, so definitely the app not a dep)
        if is_last and not has_archive_source and not source_types:
            return True

        # Rule 7 — delegate remaining cases to MAPPINGS generic check
        return MAPPINGS.is_app_module(name, is_last=is_last, source_types=source_types)

    def _process_simple_module(
        self, mod: dict, analysis: ManifestAnalysis
    ) -> None:
        from packaging.utils import canonicalize_name
        for cmd in mod.get("build-commands", []):
            for pkg in _parse_pip_packages(cmd):
                canonical = canonicalize_name(pkg)
                if canonical not in analysis.python_packages:
                    analysis.python_packages.append(canonical)

    def _process_native_module(
        self, mod: dict, analysis: ManifestAnalysis
    ) -> None:
        name = mod.get("name", "")
        buildsystem = mod.get("buildsystem", "autotools")
        config_opts = mod.get("config-opts", [])
        cleanup = mod.get("cleanup", [])

        source_url: Optional[str] = None
        source_sha256: Optional[str] = None
        for src in mod.get("sources", []):
            if isinstance(src, dict) and src.get("type") == "archive":
                source_url = src.get("url")
                source_sha256 = src.get("sha256")
                break

        pkgconfig_names = self._infer_pkgconfig(name, config_opts)
        provides_sonames = MAPPINGS.module_to_soname(name)

        analysis.native_modules.append(LearnedNativeModule(
            module_name=name,
            buildsystem=buildsystem,
            source_url=source_url,
            source_sha256=source_sha256,
            pkgconfig_names=pkgconfig_names,
            provides_sonames=provides_sonames,
            config_opts=config_opts,
            cleanup=cleanup,
        ))

    @staticmethod
    def _infer_pkgconfig(name: str, config_opts: list[str]) -> list[str]:
        names = list(MAPPINGS.module_to_pkgconfig(name))

        # Also extract from --with-<name> configure flags
        for opt in config_opts:
            m = re.search(r"--with-([a-z0-9_\-]+)", opt)
            if m and m.group(1) not in names:
                names.append(m.group(1))

        return list(dict.fromkeys(names))  # dedup, preserve order