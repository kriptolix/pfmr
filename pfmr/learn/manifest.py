"""
pfmr.learn.manifest
~~~~~~~~~~~~~~~~~~~~
ManifestAnalyzer — extracts knowledge from Flatpak manifests.

Understands JSON and YAML manifests (Flathub format).  Identifies:

  - Python packages being installed (pip install / uv pip install commands)
  - Native library modules (autotools / cmake / meson buildsystems)
  - SDK extensions declared (sdk-extensions field)
  - Build environment (runtime, sdk, sdk-version)
  - pkg-config names used in build-options or configure flags
  - Source URLs and checksums for native modules

The output is a list of LearnedFact objects that the caller can feed into
the KnowledgeGraph.  This module has zero dependency on the resolver
pipeline.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

import yaml

from pfmr.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class LearnedPackageDep:
    """A Python package that needs a native dep, as learned from a manifest."""
    python_package: str         # canonical name
    native_dep: str             # pkgconfig name or soname
    dep_type: str               # "pkgconfig" | "library" | "extension"
    confidence: float = 0.8
    source: str = ""


@dataclass
class LearnedNativeModule:
    """A native library module extracted from a manifest."""
    module_name: str
    buildsystem: str            # autotools | cmake | meson | simple
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
    package_deps: list[LearnedPackageDep] = field(default_factory=list)
    source_path: str = ""


# ---------------------------------------------------------------------------
# Patterns for pip/uv install commands
# ---------------------------------------------------------------------------

_PIP_INSTALL_RE = re.compile(
    r"(?:pip(?:3)?|uv\s+pip)\s+install\s+(?:--[^\s]+\s+)*(?P<specs>[^&;|]+)",
    re.IGNORECASE,
)

_PKG_SPEC_RE = re.compile(
    r"(?P<name>[A-Za-z0-9_\-\.]+)(?:[>=<!][^\s,]+)?"
)

# Detect --find-links / --no-index / file:// installs (offline bundles)
_OFFLINE_RE = re.compile(r"--(?:find-links|no-index|extra-index-url)\s", re.IGNORECASE)


def _parse_pip_packages(command: str) -> list[str]:
    """Extract package names from a pip/uv install command string."""
    packages: list[str] = []
    for m in _PIP_INSTALL_RE.finditer(command):
        specs_str = m.group("specs").strip()
        for spec_m in _PKG_SPEC_RE.finditer(specs_str):
            name = spec_m.group("name").strip()
            # Filter flags and common noise
            if name.startswith("-") or name.lower() in (
                "install", "pip", "uv", "python", "python3", "wheel",
                "setuptools", "no", "index", "find", "links",
            ):
                continue
            if len(name) > 1:
                packages.append(name)
    return packages


# ---------------------------------------------------------------------------
# ManifestAnalyzer
# ---------------------------------------------------------------------------

class ManifestAnalyzer:
    """
    Analyzes a Flatpak manifest and extracts knowledge about Python packages
    and their native dependencies.

    Completely standalone — does not import from pfmr.pipeline.

    Usage::

        analyzer = ManifestAnalyzer()
        analysis = analyzer.analyze(Path("org.gnome.App.json"))
        for mod in analysis.native_modules:
            print(mod.module_name, mod.pkgconfig_names)
    """

    def analyze(self, manifest_path: Path) -> Optional[ManifestAnalysis]:
        """Analyze a manifest file. Returns None if the file cannot be parsed."""
        try:
            data = self._load(manifest_path)
        except Exception as exc:
            logger.warning("Failed to load manifest %s: %s", manifest_path, exc)
            return None

        return self._analyze_dict(data, source=str(manifest_path))

    def analyze_dict(self, data: dict, source: str = "") -> ManifestAnalysis:
        """Analyze a manifest already loaded as a dict."""
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

    def analyze_directory(
        self,
        directory: Path,
        recursive: bool = True,
        glob: str = "**/*.{json,yaml,yml}",
    ) -> list[ManifestAnalysis]:
        """
        Analyze all manifest files found in a directory.

        Scans for *.json, *.yaml, *.yml files and attempts to parse each
        as a Flatpak manifest (identified by presence of "app-id" or "modules"
        keys). Non-manifest files are silently skipped.

        Useful for:
          - Local Flathub repo checkouts
          - shared-modules repository (https://github.com/flathub/shared-modules)
          - Any directory of collected manifests
        """
        results: list[ManifestAnalysis] = []
        patterns = ["**/*.json", "**/*.yaml", "**/*.yml"] if recursive else ["*.json", "*.yaml", "*.yml"]
        seen: set[str] = set()

        for pattern in patterns:
            for p in sorted(directory.glob(pattern)):
                if str(p) in seen:
                    continue
                seen.add(str(p))
                try:
                    data = self._load(p)
                except Exception:
                    continue
                # Must look like a Flatpak manifest
                if not isinstance(data, dict):
                    continue
                if not ("app-id" in data or "id" in data or "modules" in data):
                    continue
                analysis = self._analyze_dict(data, source=str(p))
                if analysis:
                    results.append(analysis)

        logger.info("Analyzed %d manifests in %s", len(results), directory)
        return results

    def analyze_dict(self, data: dict, source: str = "") -> ManifestAnalysis:
        """Analyze a manifest already loaded as a dict."""
        return self._analyze_dict(data, source=source)

    def _process_modules(self, modules: list, analysis: ManifestAnalysis) -> None:
        for mod in modules:
            if not isinstance(mod, dict):
                continue
            # Recurse into sub-modules first
            for sub in mod.get("modules", []):
                self._process_modules([sub], analysis)

            buildsystem = mod.get("buildsystem", "autotools")
            name = mod.get("name", "")

            if buildsystem == "simple":
                self._process_simple_module(mod, analysis)
            else:
                self._process_native_module(mod, analysis)

    def _process_simple_module(self, mod: dict, analysis: ManifestAnalysis) -> None:
        """Extract Python packages from simple buildsystem modules."""
        cmds = mod.get("build-commands", [])
        for cmd in cmds:
            pkgs = _parse_pip_packages(cmd)
            for pkg in pkgs:
                from packaging.utils import canonicalize_name
                canonical = canonicalize_name(pkg)
                if canonical not in analysis.python_packages:
                    analysis.python_packages.append(canonical)

    def _process_native_module(self, mod: dict, analysis: ManifestAnalysis) -> None:
        """Extract native library module information."""
        name = mod.get("name", "")
        buildsystem = mod.get("buildsystem", "autotools")
        sources = mod.get("sources", [])
        config_opts = mod.get("config-opts", [])
        cleanup = mod.get("cleanup", [])
        build_opts = mod.get("build-options", {})

        # Extract source URL + hash
        source_url: Optional[str] = None
        source_sha256: Optional[str] = None
        for src in sources:
            if isinstance(src, dict) and src.get("type") == "archive":
                source_url = src.get("url")
                source_sha256 = src.get("sha256")
                break

        # Infer pkgconfig names from module name
        pkgconfig_names = self._infer_pkgconfig(name, config_opts, build_opts)
        provides_sonames = self._infer_sonames(name)

        native_mod = LearnedNativeModule(
            module_name=name,
            buildsystem=buildsystem,
            source_url=source_url,
            source_sha256=source_sha256,
            pkgconfig_names=pkgconfig_names,
            provides_sonames=provides_sonames,
            config_opts=config_opts,
            cleanup=cleanup,
        )
        analysis.native_modules.append(native_mod)

    @staticmethod
    def _infer_pkgconfig(
        name: str,
        config_opts: list[str],
        build_opts: dict,
    ) -> list[str]:
        """Guess pkg-config names from module name and build flags."""
        names: list[str] = []

        # Common module-name → pkgconfig name mappings
        _MODULE_PC_MAP = {
            "openssl":    ["openssl"],
            "libssl":     ["openssl"],
            "zlib":       ["zlib"],
            "libffi":     ["libffi"],
            "libxml2":    ["libxml-2.0"],
            "libxslt":    ["libxslt"],
            "libjpeg":    ["libjpeg"],
            "libjpeg-turbo": ["libjpeg"],
            "libpng":     ["libpng"],
            "libtiff":    ["libtiff-4"],
            "libwebp":    ["libwebp"],
            "sqlite":     ["sqlite3"],
            "sqlite3":    ["sqlite3"],
            "curl":       ["libcurl"],
            "libcurl":    ["libcurl"],
            "libusb":     ["libusb-1.0"],
            "hidapi":     ["hidapi-libusb"],
            "openblas":   ["openblas"],
            "libvips":    ["vips"],
            "portaudio":  ["portaudio-2.0"],
            "libsndfile": ["sndfile"],
            "libzmq":     ["libzmq"],
            "zeromq":     ["libzmq"],
            "postgresql": ["libpq"],
            "libpq":      ["libpq"],
        }
        canonical = name.lower().strip()
        if canonical in _MODULE_PC_MAP:
            names.extend(_MODULE_PC_MAP[canonical])
        elif not names:
            # Fallback: use the module name itself as a pkgconfig candidate
            names.append(canonical)

        # Also scan --pkg-config-path and similar flags
        for opt in config_opts:
            m = re.search(r"--with-([a-z0-9_\-]+)", opt)
            if m and m.group(1) not in names:
                names.append(m.group(1))

        return list(dict.fromkeys(names))  # dedup, preserve order

    @staticmethod
    def _infer_sonames(name: str) -> list[str]:
        """Guess sonames from module name."""
        _SONAME_MAP = {
            "openssl":    ["libssl.so.3", "libcrypto.so.3"],
            "zlib":       ["libz.so.1"],
            "libffi":     ["libffi.so.8"],
            "libxml2":    ["libxml2.so.2"],
            "libxslt":    ["libxslt.so.1"],
            "libjpeg":    ["libjpeg.so.62"],
            "libjpeg-turbo": ["libjpeg.so.62"],
            "libpng":     ["libpng16.so.16"],
            "libwebp":    ["libwebp.so.7"],
            "sqlite":     ["libsqlite3.so.0"],
            "libusb":     ["libusb-1.0.so.0"],
            "openblas":   ["libopenblas.so.0"],
            "libvips":    ["libvips.so.42"],
            "portaudio":  ["libportaudio.so.2"],
            "libsndfile": ["libsndfile.so.1"],
        }
        return _SONAME_MAP.get(name.lower(), [])