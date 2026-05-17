"""
pfmr.learn.sandbox
~~~~~~~~~~~~~~~~~~~
SandboxLearner — ingests a SandboxProbeReport and writes the learned facts
directly into recipes/python/<pkg>.yaml files.

No KnowledgeGraph. Output is immediately usable by pfmr's resolver.

Confidence values:
  1.0 — successful install + import in the sandbox
  0.9 — successful install, import failed (runtime dep, not build dep)
  0.8 — failed install with a clear missing dep error
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import yaml
from packaging.utils import canonicalize_name

from pfmr.models import SandboxErrorType, SandboxProbeReport
from pfmr.utils.logging import get_logger

logger = get_logger(__name__)


class SandboxLearner:
    """
    Writes probe results directly into recipes/python/<pkg>.yaml.

    Usage::

        learner = SandboxLearner(repo_root=Path("."))
        learner.ingest(report, package_name="cryptography")
    """

    def __init__(self, repo_root: Path = Path(".")):
        self.recipes_dir = repo_root / "recipes" / "python"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(
        self,
        report: SandboxProbeReport,
        package_name: Optional[str] = None,
        sdk_id: str = "org.freedesktop.Sdk",
        sdk_version: str = "24.08",
    ) -> int:
        """
        Ingest a probe report and write/update recipe files.

        Returns the number of recipe files written.
        """
        if not report.ran:
            logger.debug("Skipping ingestion — probe did not run")
            return 0

        source = f"sandbox:{sdk_id}/{sdk_version}"
        pkg_facts: dict[str, dict] = {}

        for err in report.errors:
            if package_name:
                pkg = canonicalize_name(package_name)
            else:
                raw = err.context.split(" ")[0]
                pkg = canonicalize_name(raw) if raw else None
            if not pkg:
                continue

            if pkg not in pkg_facts:
                pkg_facts[pkg] = {
                    "pypi_name": package_name or pkg,
                    "pkgconfig": [], "libraries": [],
                    "headers": [], "extensions": [],
                    "confidence": 0.8, "source": source,
                }
            facts = pkg_facts[pkg]
            if err.error_type == SandboxErrorType.MISSING_PKGCONFIG:
                if err.missing not in facts["pkgconfig"]:
                    facts["pkgconfig"].append(err.missing)
            elif err.error_type == SandboxErrorType.MISSING_NATIVE_DEP:
                if err.missing not in facts["libraries"]:
                    facts["libraries"].append(err.missing)
            elif err.error_type == SandboxErrorType.MISSING_HEADER:
                if err.missing not in facts["headers"]:
                    facts["headers"].append(err.missing)

        written = sum(self._write_recipe(pkg, facts) for pkg, facts in pkg_facts.items())

        # Record clean builds too
        if report.ran and not report.errors and package_name:
            pkg = canonicalize_name(package_name)
            written += self._write_recipe(pkg, {
                "pypi_name": package_name,
                "pkgconfig": [], "libraries": [], "headers": [], "extensions": [],
                "confidence": 1.0, "source": source, "sdk_sufficient": True,
            })

        logger.info("SandboxLearner: %d recipe(s) written", written)
        return written

    def ingest_successful_build(
        self,
        package_name: str,
        native_deps: list[str],
        required_extensions: list[str],
        sdk_id: str = "org.freedesktop.Sdk",
        sdk_version: str = "24.08",
    ) -> int:
        """Record a confirmed successful build (confidence=1.0)."""
        return self._write_recipe(canonicalize_name(package_name), {
            "pypi_name": package_name,
            "pkgconfig": list(native_deps), "libraries": [], "headers": [],
            "extensions": list(required_extensions),
            "confidence": 1.0,
            "source": f"sandbox:{sdk_id}/{sdk_version}",
        })

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_recipe(self, pkg: str, facts: dict) -> int:
        recipe_path = self.recipes_dir / f"{pkg}.yaml"

        requires: dict = {}
        if facts.get("pkgconfig"):
            requires["pkgconfig"] = sorted(facts["pkgconfig"])
        if facts.get("libraries"):
            requires["libraries"] = sorted(facts["libraries"])
        if facts.get("extensions"):
            requires["extensions"] = sorted(facts["extensions"])

        # Merge with existing file if present
        if recipe_path.exists():
            try:
                existing = yaml.safe_load(recipe_path.read_text()) or {}
            except Exception:
                existing = {}
            existing_conf = existing.get("confidence", 0.0)
            new_conf = facts["confidence"]
            old_pc = set(existing.get("requires", {}).get("pkgconfig", []))
            new_pc = set(requires.get("pkgconfig", []))
            if new_conf <= existing_conf and not (new_pc - old_pc):
                return 0
            # Merge deps, take highest confidence
            merged = dict(existing.get("requires", {}))
            for key in requires:
                merged[key] = sorted(set(merged.get(key, [])) | set(requires[key]))
            requires = {k: v for k, v in merged.items() if v}
            facts["confidence"] = max(existing_conf, new_conf)

        recipe: dict = {
            "id": pkg, "type": "python",
            "pypi_name": facts.get("pypi_name", pkg),
        }
        if requires:
            recipe["requires"] = requires
        if facts.get("sdk_sufficient"):
            recipe["sdk_sufficient"] = True
        recipe["confidence"] = facts["confidence"]
        recipe["source"] = facts.get("source", "")
        recipe["updated"] = str(date.today())

        self.recipes_dir.mkdir(parents=True, exist_ok=True)
        recipe_path.write_text(
            yaml.dump(recipe, default_flow_style=False, allow_unicode=True, sort_keys=False)
        )
        logger.info("Written recipe: %s (confidence=%.1f)", recipe_path.name, facts["confidence"])
        return 1