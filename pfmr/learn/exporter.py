"""
pfmr.learn.exporter
~~~~~~~~~~~~~~~~~~~~
Exporter — writes knowledge graph facts directly into the recipes/
directory as YAML files that pfmr's resolver can consume immediately.

Structure evaluation
--------------------
The original design had 4 output layers:
  1. recipes/native/*.yaml       — how to build a lib
  2. data/native-hints/…         — what libs a Python package needs
  3. data/sdk-profiles/…         — what the SDK already provides
  4. data/extension-profiles/…  — what extensions provide

Layers 3 and 4 are distinct and necessary: they describe the *environment*,
not the packages.

Layers 1 and 2 are partially redundant. A recipe already contains:
  pkgconfig, provides (sonames), id — everything native-hints needs.
And native-hints needs:
  which packages require which pkgconfig names.

Resolution: merge into a single recipe YAML format that covers both:
  - recipes/native/<lib>.yaml         — builds the lib (existing format)
  - recipes/python/<pkg>.yaml         — NEW: which libs a Python package needs

This way there is exactly one place to look for any package dependency,
and contributions via git are straightforward: add a YAML file.

python package recipe format:
  id: cryptography
  type: python
  pypi_name: cryptography
  build_backend: maturin
  requires:
    pkgconfig: [openssl, libffi]
    extensions: [org.freedesktop.Sdk.Extension.rust-stable]
  confidence: 1.0
  source: sandbox:org.freedesktop.Sdk/24.08
  updated: 2025-05-14
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

import yaml

from pfmr.learn.graph import KGEdge, KGNode, KnowledgeGraph, Rel
from pfmr.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExportChange:
    path: Path
    action: str          # "create" | "update" | "skip"
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
    Exports KnowledgeGraph facts into the recipes/ directory.

    Two recipe types:
      recipes/native/<lib-id>.yaml    — how to build the native lib
      recipes/python/<pkg-id>.yaml    — what native libs a Python pkg needs

    Usage::

        kg = KnowledgeGraph(Path("knowledge/"))
        exporter = Exporter(kg, repo_root=Path("."))
        report = exporter.export()
        for c in report.created:
            print("New:", c.path)
    """

    def __init__(self, graph: KnowledgeGraph, repo_root: Path):
        self.graph = graph
        self.repo_root = repo_root

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(self, dry_run: bool = False) -> ExportReport:
        report = ExportReport(dry_run=dry_run)
        self._export_native_recipes(report, dry_run)
        self._export_python_recipes(report, dry_run)
        logger.info(
            "Export: %d created, %d updated, %d skipped",
            len(report.created), len(report.updated), len(report.skipped),
        )
        return report

    def export_native_recipes(self, dry_run: bool = False) -> ExportReport:
        report = ExportReport(dry_run=dry_run)
        self._export_native_recipes(report, dry_run)
        return report

    def export_python_recipes(self, dry_run: bool = False) -> ExportReport:
        report = ExportReport(dry_run=dry_run)
        self._export_python_recipes(report, dry_run)
        return report

    # ------------------------------------------------------------------
    # Native recipes (how to build a lib)
    # ------------------------------------------------------------------

    def _export_native_recipes(self, report: ExportReport, dry_run: bool) -> None:
        recipes_dir = self.repo_root / "recipes" / "native"
        existing_ids = {p.stem for p in recipes_dir.glob("*.yaml")} if recipes_dir.exists() else set()

        for lib_node in sorted(self.graph.nodes_of_type("library"), key=lambda n: n.id):
            if lib_node.id in existing_ids:
                continue  # never overwrite curated recipes

            source_url = lib_node.attrs.get("source_url")
            if not source_url:
                continue  # not enough info to build a recipe

            recipe = self._build_native_recipe(lib_node)
            recipe_path = recipes_dir / f"{lib_node.id}.yaml"

            if not dry_run:
                recipes_dir.mkdir(parents=True, exist_ok=True)
                recipe_path.write_text(
                    yaml.dump(recipe, default_flow_style=False, allow_unicode=True, sort_keys=False)
                )

            report.changes.append(ExportChange(
                path=recipe_path,
                action="create" if not dry_run else "skip",
                reason=f"learned from {lib_node.attrs.get('source', '?')}",
            ))

    # ------------------------------------------------------------------
    # Python recipes (what libs a Python pkg needs)
    # ------------------------------------------------------------------

    def _export_python_recipes(self, report: ExportReport, dry_run: bool) -> None:
        recipes_dir = self.repo_root / "recipes" / "python"
        existing_ids = {p.stem for p in recipes_dir.glob("*.yaml")} if recipes_dir.exists() else set()

        for pkg_node in sorted(self.graph.nodes_of_type("package"), key=lambda n: n.id):
            pc_edges = [e for e in self.graph.edges_from(pkg_node.id, Rel.REQUIRES_PKGCONFIG)
                        if e.confidence >= 0.7]
            lib_edges = [e for e in self.graph.edges_from(pkg_node.id, Rel.REQUIRES_LIBRARY)
                         if e.confidence >= 0.7]
            ext_edges = [e for e in self.graph.edges_from(pkg_node.id, Rel.REQUIRES_EXTENSION)
                         if e.confidence >= 0.7]

            if not (pc_edges or lib_edges or ext_edges):
                continue  # nothing to write

            recipe = self._build_python_recipe(pkg_node, pc_edges, lib_edges, ext_edges)
            recipe_path = recipes_dir / f"{pkg_node.id}.yaml"

            # Python recipes CAN be updated — new evidence may raise confidence
            # or add new deps. We update if the file already exists.
            action = "skip"
            if pkg_node.id in existing_ids:
                existing = self._load_yaml(recipe_path)
                if existing and self._recipe_changed(existing, recipe):
                    action = "update"
                else:
                    report.changes.append(ExportChange(recipe_path, "skip", "unchanged"))
                    continue
            else:
                action = "create"

            if not dry_run:
                recipes_dir.mkdir(parents=True, exist_ok=True)
                recipe_path.write_text(
                    yaml.dump(recipe, default_flow_style=False, allow_unicode=True, sort_keys=False)
                )

            report.changes.append(ExportChange(
                path=recipe_path,
                action=action if not dry_run else "skip",
                reason=f"confidence={max((e.confidence for e in pc_edges + lib_edges + ext_edges), default=0):.1f}",
            ))

    # ------------------------------------------------------------------
    # Recipe builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_native_recipe(node: KGNode) -> dict:
        attrs = node.attrs
        recipe: dict = {"id": node.id, "type": "native"}

        pc = attrs.get("pkgconfig", "")
        recipe["pkgconfig"] = [pc] if isinstance(pc, str) and pc else (pc if pc else [])

        if soname := attrs.get("soname"):
            recipe["provides"] = [soname]
        else:
            recipe["provides"] = []

        recipe["buildsystem"] = attrs.get("buildsystem", "autotools")

        source_entry: dict = {"type": "archive", "url": attrs["source_url"]}
        if sha256 := attrs.get("source_sha256"):
            source_entry["sha256"] = sha256
        recipe["source"] = source_entry
        recipe["cleanup"] = ["/include", "/lib/pkgconfig"]
        return recipe

    @staticmethod
    def _build_python_recipe(
        node: KGNode,
        pc_edges: list[KGEdge],
        lib_edges: list[KGEdge],
        ext_edges: list[KGEdge],
    ) -> dict:
        attrs = node.attrs
        # Best confidence across all edges
        max_conf = max(
            (e.confidence for e in pc_edges + lib_edges + ext_edges), default=0.0
        )
        sources = sorted({e.source for e in pc_edges + lib_edges + ext_edges if e.source})

        recipe: dict = {
            "id": node.id,
            "type": "python",
            "pypi_name": attrs.get("pypi_name", node.id),
        }
        if backend := attrs.get("build_backend"):
            recipe["build_backend"] = backend

        requires: dict = {}
        if pc_edges:
            requires["pkgconfig"] = sorted({e.to_id for e in pc_edges})
        if lib_edges:
            requires["libraries"] = sorted({e.to_id for e in lib_edges})
        if ext_edges:
            requires["extensions"] = sorted({e.to_id for e in ext_edges})
        if requires:
            recipe["requires"] = requires

        recipe["confidence"] = round(max_conf, 2)
        if sources:
            recipe["source"] = sources[0]
        recipe["updated"] = str(date.today())
        return recipe

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_yaml(path: Path) -> Optional[dict]:
        try:
            return yaml.safe_load(path.read_text()) or {}
        except Exception:
            return None

    @staticmethod
    def _recipe_changed(existing: dict, new: dict) -> bool:
        """True if the new recipe has more or higher-confidence deps."""
        old_reqs = existing.get("requires", {})
        new_reqs = new.get("requires", {})
        if set(new_reqs.keys()) != set(old_reqs.keys()):
            return True
        for key in new_reqs:
            if set(new_reqs[key]) != set(old_reqs.get(key, [])):
                return True
        old_conf = existing.get("confidence", 0.0)
        new_conf = new.get("confidence", 0.0)
        return new_conf > old_conf