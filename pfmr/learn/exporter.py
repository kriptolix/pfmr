"""
pfmr.learn.exporter
~~~~~~~~~~~~~~~~~~~~
Exporter — converts ManifestAnalysis objects directly into recipe files.

No knowledge graph intermediary. The pipeline is:

  ManifestAnalysis (from FlathubMiner / ManifestAnalyzer)
      ↓
  Exporter.export()
      ↓
  recipes/native/<lib-id>.yaml   — how to build the native lib
  recipes/python/<pkg-id>.yaml   — what libs a Python package needs

Two recipe types
----------------
native recipe (same format as existing curated recipes):
  id: libusb
  provides: [libusb-1.0.so.0]
  pkgconfig: [libusb-1.0]
  buildsystem: autotools
  source:
    type: archive
    url: https://...
    sha256: ...
  cleanup: [/include, /lib/pkgconfig]

python recipe (new — declares what a Python package needs):
  id: cryptography
  type: python
  pypi_name: cryptography
  requires:
    pkgconfig: [openssl, libffi]
    extensions: [org.freedesktop.Sdk.Extension.rust-stable]
  confidence: 0.6
  source: flathub:org.gnome.Crypto
  updated: 2025-05-14

Confidence
----------
Co-occurrence in a manifest (Python pkg next to native module) → 0.6.
This is intentionally low — enough to surface a candidate without
over-claiming. Higher confidence requires sandbox probing (Phase 3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import yaml
from packaging.utils import canonicalize_name

from pfmr.data.mappings import MAPPINGS
from pfmr.learn.manifest import ManifestAnalysis, LearnedNativeModule
from pfmr.utils.logging import get_logger

logger = get_logger(__name__)

# pkg-config name → known soname(s), for enriching recipe "provides"



@dataclass
class ExportChange:
    path: Path
    action: str       # "create" | "update" | "skip"
    reason: str = ""


@dataclass
class ExportReport:
    changes: list[ExportChange] = field(default_factory=list)
    dry_run: bool = False

    @property
    def created(self) -> list[ExportChange]:
        return [c for c in self.changes if c.action == "create"]

    @property
    def updated(self) -> list[ExportChange]:
        return [c for c in self.changes if c.action == "update"]

    @property
    def skipped(self) -> list[ExportChange]:
        return [c for c in self.changes if c.action == "skip"]


class Exporter:
    """
    Converts a list of ManifestAnalysis objects into recipe YAML files.

    Usage::

        exporter = Exporter(analyses, repo_root=Path("."))
        report = exporter.export()
        for c in report.created:
            print("New:", c.path)
    """

    def __init__(self, analyses: list[ManifestAnalysis], repo_root: Path):
        self.analyses = analyses
        self.repo_root = repo_root
        self._native_dir = repo_root / "recipes" / "native"
        self._python_dir = repo_root / "recipes" / "python"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(self, dry_run: bool = False) -> ExportReport:
        report = ExportReport(dry_run=dry_run)
        self._export_native(report, dry_run)
        self._export_python(report, dry_run)
        logger.info(
            "Export: %d created, %d updated, %d skipped",
            len(report.created), len(report.updated), len(report.skipped),
        )
        return report

    def export_native_recipes(self, dry_run: bool = False) -> ExportReport:
        report = ExportReport(dry_run=dry_run)
        self._export_native(report, dry_run)
        return report

    def export_python_recipes(self, dry_run: bool = False) -> ExportReport:
        report = ExportReport(dry_run=dry_run)
        self._export_python(report, dry_run)
        return report

    # ------------------------------------------------------------------
    # Native
    # ------------------------------------------------------------------

    def _export_native(self, report: ExportReport, dry_run: bool) -> None:
        existing = {p.stem for p in self._native_dir.glob("*.yaml")} if self._native_dir.exists() else set()
        # Collect all native modules across analyses, dedup by normalised id
        seen: dict[str, tuple[LearnedNativeModule, str]] = {}  # id → (mod, source)

        for analysis in self.analyses:
            for mod in analysis.native_modules:
                mod_id = _normalise_id(mod.module_name)
                if mod_id in existing or mod_id in seen:
                    continue
                if not mod.source_url:
                    continue
                seen[mod_id] = (mod, analysis.source_path or analysis.app_id)

        for mod_id, (mod, source) in sorted(seen.items()):
            recipe = _build_native_recipe(mod_id, mod)
            path = self._native_dir / f"{mod_id}.yaml"

            if not dry_run:
                self._native_dir.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    yaml.dump(recipe, default_flow_style=False, allow_unicode=True, sort_keys=False)
                )

            report.changes.append(ExportChange(
                path=path,
                action="create" if not dry_run else "skip",
                reason=f"from {source}",
            ))

    # ------------------------------------------------------------------
    # Python
    # ------------------------------------------------------------------

    def _export_python(self, report: ExportReport, dry_run: bool) -> None:
        existing = {p.stem for p in self._python_dir.glob("*.yaml")} if self._python_dir.exists() else set()

        # Collect package → {pkgconfig_names, extensions} across analyses
        # Key: canonical package name
        # Value: dict with "pkgconfig": set, "extensions": set, "sources": list
        pkg_data: dict[str, dict] = {}

        for analysis in self.analyses:
            if not analysis.python_packages or not analysis.native_modules:
                continue
            source = f"flathub:{analysis.app_id}" if analysis.app_id else analysis.source_path

            # All pkgconfig names provided by native modules in this manifest
            manifest_pc: set[str] = set()
            for mod in analysis.native_modules:
                manifest_pc.update(mod.pkgconfig_names)

            for pkg_name in analysis.python_packages:
                canonical = canonicalize_name(pkg_name)
                if canonical not in pkg_data:
                    pkg_data[canonical] = {
                        "pypi_name": pkg_name,
                        "pkgconfig": set(),
                        "extensions": set(),
                        "sources": [],
                    }
                pkg_data[canonical]["pkgconfig"].update(manifest_pc)
                pkg_data[canonical]["extensions"].update(analysis.sdk_extensions)
                if source not in pkg_data[canonical]["sources"]:
                    pkg_data[canonical]["sources"].append(source)

        for canonical, data in sorted(pkg_data.items()):
            if not data["pkgconfig"] and not data["extensions"]:
                continue
            if canonical in existing:
                # Update only if new deps found
                existing_recipe = _load_yaml(self._python_dir / f"{canonical}.yaml")
                if existing_recipe:
                    old_pc = set(existing_recipe.get("requires", {}).get("pkgconfig", []))
                    if not data["pkgconfig"] - old_pc:
                        report.changes.append(ExportChange(
                            self._python_dir / f"{canonical}.yaml", "skip", "unchanged"
                        ))
                        continue

            recipe = _build_python_recipe(canonical, data)
            path = self._python_dir / f"{canonical}.yaml"

            action = "update" if canonical in existing else "create"
            if not dry_run:
                self._python_dir.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    yaml.dump(recipe, default_flow_style=False, allow_unicode=True, sort_keys=False)
                )

            report.changes.append(ExportChange(
                path=path,
                action=action if not dry_run else "skip",
                reason=f"pc={sorted(data['pkgconfig'])[:3]} confidence=0.6",
            ))


# ---------------------------------------------------------------------------
# Recipe builders
# ---------------------------------------------------------------------------

def _build_native_recipe(recipe_id: str, mod: LearnedNativeModule) -> dict:
    pkgconfig = mod.pkgconfig_names or [recipe_id]
    provides: list[str] = []
    for pc in pkgconfig:
        provides.extend(MAPPINGS.pkgconfig_to_soname(pc))
    provides = list(dict.fromkeys(provides))  # dedup

    recipe: dict = {
        "id": recipe_id,
        "provides": provides,
        "pkgconfig": pkgconfig,
        "buildsystem": mod.buildsystem,
    }

    if mod.source_url:
        src: dict = {"type": "archive", "url": mod.source_url}
        if mod.source_sha256:
            src["sha256"] = mod.source_sha256
        recipe["source"] = src

    if mod.config_opts:
        recipe["config-opts"] = mod.config_opts

    recipe["cleanup"] = mod.cleanup or ["/include", "/lib/pkgconfig"]
    return recipe


def _build_python_recipe(canonical: str, data: dict) -> dict:
    requires: dict = {}
    if data["pkgconfig"]:
        requires["pkgconfig"] = sorted(data["pkgconfig"])
    if data["extensions"]:
        requires["extensions"] = sorted(data["extensions"])

    recipe: dict = {
        "id": canonical,
        "type": "python",
        "pypi_name": data["pypi_name"],
    }
    if requires:
        recipe["requires"] = requires
    recipe["confidence"] = 0.6
    if data["sources"]:
        recipe["source"] = data["sources"][0]
    recipe["updated"] = str(date.today())
    return recipe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_id(name: str) -> str:
    return name.lower().replace(" ", "-").replace("_", "-")


def _load_yaml(path: Path) -> Optional[dict]:
    try:
        return yaml.safe_load(path.read_text()) or {}
    except Exception:
        return None