"""
pfmr.learn.exporter
~~~~~~~~~~~~~~~~~~~~
Exporter — translates the KnowledgeGraph back into the repository artifacts
that pfmr consumes:

  recipes/native/<id>.yaml        — NativeRecipe YAML files
  data/native-hints/packages.toml — NativeDependencyAnalyzer hints
  data/extension-profiles/<id>.toml — SDKExtensionResolver profiles (partial)

This is the "write back to the repo" step of the learning loop:
  mine / probe → KnowledgeGraph.save() → Exporter.export() → git commit

Design goals:
  - Never overwrites an existing file without diffing first
  - Generates a human-readable change summary
  - Can run in --dry-run mode (returns planned changes without writing)
  - Each output is valid and usable by pfmr immediately after export

Completely standalone — no pfmr.pipeline dependency.
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

# ---------------------------------------------------------------------------
# Change record
# ---------------------------------------------------------------------------

@dataclass
class ExportChange:
    """A single file that would be created or updated."""
    path: Path
    action: str          # "create" | "update" | "skip"
    reason: str = ""


@dataclass
class ExportReport:
    """Summary of an export run."""
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


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------

class Exporter:
    """
    Generates repository files from the KnowledgeGraph.

    Usage::

        kg = KnowledgeGraph(Path("knowledge/"))
        exporter = Exporter(kg, repo_root=Path("."))
        report = exporter.export(dry_run=False)
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
        """
        Export all knowledge from the graph into repository files.
        Returns an ExportReport describing what changed.
        """
        report = ExportReport(dry_run=dry_run)

        self._export_native_recipes(report, dry_run)
        self._export_native_hints(report, dry_run)

        logger.info(
            "Export: %d created, %d updated, %d skipped (dry_run=%s)",
            len(report.created), len(report.updated), len(report.skipped), dry_run,
        )
        return report

    def export_native_hints(self, dry_run: bool = False) -> ExportReport:
        """Export only the native-hints/packages.toml file."""
        report = ExportReport(dry_run=dry_run)
        self._export_native_hints(report, dry_run)
        return report

    def export_native_recipes(self, dry_run: bool = False) -> ExportReport:
        """Export only recipes/native/*.yaml files."""
        report = ExportReport(dry_run=dry_run)
        self._export_native_recipes(report, dry_run)
        return report

    # ------------------------------------------------------------------
    # Hints exporter
    # ------------------------------------------------------------------

    def _export_native_hints(self, report: ExportReport, dry_run: bool) -> None:
        """
        Generate / update data/native-hints/packages.toml from graph edges.
        Only adds entries for packages that have REQUIRES_PKGCONFIG edges with
        confidence >= 0.7 and that are not already in the file.
        """
        hints_path = self.repo_root / "pfmr" / "data" / "native-hints" / "packages.toml"
        existing = self._load_toml_raw(hints_path)
        existing_keys = set(existing.keys()) if existing else set()

        new_entries: list[str] = []
        for pkg_node in sorted(
            self.graph.nodes_of_type("package"), key=lambda n: n.id
        ):
            if pkg_node.id in existing_keys:
                continue  # respect existing curated entries

            pc_edges = self.graph.edges_from(pkg_node.id, relation=Rel.REQUIRES_PKGCONFIG)
            lib_edges = self.graph.edges_from(pkg_node.id, relation=Rel.REQUIRES_LIBRARY)

            # Only export if we have at least one high-confidence dep
            high_conf = [
                e for e in pc_edges + lib_edges if e.confidence >= 0.7
            ]
            if not high_conf:
                continue

            pkgconfig = sorted({
                e.to_id for e in pc_edges if e.confidence >= 0.7
            })
            libraries = sorted({
                e.to_id for e in lib_edges if e.confidence >= 0.7
                and e.to_id.endswith(".so") or ".so." in e.to_id
            })

            entry_lines = [
                f"[{pkg_node.id}]",
                f"pkgconfig = [{', '.join(repr(p) for p in pkgconfig)}]",
                f"libraries = [{', '.join(repr(l) for l in libraries)}]",
                f"headers = []",
            ]
            # Add source comment
            sources = sorted({e.source for e in high_conf if e.source})
            if sources:
                entry_lines.insert(0, f"# learned from: {', '.join(sources[:2])}")
            new_entries.append("\n".join(entry_lines))

        if not new_entries:
            report.changes.append(ExportChange(
                path=hints_path, action="skip", reason="no new packages to add"
            ))
            return

        if not dry_run:
            hints_path.parent.mkdir(parents=True, exist_ok=True)
            with open(hints_path, "a") as f:
                f.write(f"\n# === learned entries ({date.today()}) ===\n")
                for entry in new_entries:
                    f.write("\n" + entry + "\n")

        action = "update" if hints_path.exists() else "create"
        report.changes.append(ExportChange(
            path=hints_path,
            action=action if not dry_run else "skip",
            reason=f"{len(new_entries)} new package entries",
        ))
        logger.info(
            "%s native hints: %d new entries (%s)",
            "Would add" if dry_run else "Added",
            len(new_entries), hints_path,
        )

    # ------------------------------------------------------------------
    # Recipe exporter
    # ------------------------------------------------------------------

    def _export_native_recipes(self, report: ExportReport, dry_run: bool) -> None:
        """
        Generate recipes/native/<id>.yaml for library nodes that have enough
        information (at least a source URL) and no existing recipe file.
        """
        recipes_dir = self.repo_root / "recipes" / "native"
        existing_ids = {p.stem for p in recipes_dir.glob("*.yaml")} if recipes_dir.exists() else set()

        for lib_node in sorted(
            self.graph.nodes_of_type("library"), key=lambda n: n.id
        ):
            if lib_node.id in existing_ids:
                continue  # never overwrite curated recipes

            # Only generate if we have a source URL
            source_url = lib_node.attrs.get("source_url")
            if not source_url:
                continue

            recipe = self._build_recipe_dict(lib_node)
            recipe_path = recipes_dir / f"{lib_node.id}.yaml"

            if not dry_run:
                recipes_dir.mkdir(parents=True, exist_ok=True)
                recipe_path.write_text(
                    yaml.dump(recipe, default_flow_style=False, allow_unicode=True, sort_keys=False)
                )

            action = "create"
            report.changes.append(ExportChange(
                path=recipe_path,
                action=action if not dry_run else "skip",
                reason=f"learned from {lib_node.attrs.get('source', 'unknown')}",
            ))
            logger.info(
                "%s recipe: %s",
                "Would create" if dry_run else "Created", recipe_path,
            )

    @staticmethod
    def _build_recipe_dict(node: KGNode) -> dict:
        """Build a recipe dict from a library KGNode."""
        attrs = node.attrs
        recipe: dict = {"id": node.id}

        if pkgconfig := attrs.get("pkgconfig"):
            recipe["pkgconfig"] = [pkgconfig] if isinstance(pkgconfig, str) else pkgconfig
        else:
            recipe["pkgconfig"] = []

        if soname := attrs.get("soname"):
            recipe["provides"] = [soname]
        else:
            recipe["provides"] = []

        recipe["buildsystem"] = attrs.get("buildsystem", "autotools")

        if source_url := attrs.get("source_url"):
            source_entry: dict = {"type": "archive", "url": source_url}
            if sha256 := attrs.get("source_sha256"):
                source_entry["sha256"] = sha256
            recipe["source"] = source_entry

        recipe["cleanup"] = ["/include", "/lib/pkgconfig"]
        return recipe

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_toml_raw(path: Path) -> Optional[dict]:
        if not path.exists():
            return None
        import tomllib
        try:
            with open(path, "rb") as f:
                return tomllib.load(f)
        except Exception:
            return None