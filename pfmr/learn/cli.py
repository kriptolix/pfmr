"""
pfmr.learn.cli
~~~~~~~~~~~~~~~
Standalone CLI commands for the learning subsystem.

These commands are fully independent of the resolver pipeline and are
designed to be run periodically (e.g. in a GitHub Actions workflow) to
keep the knowledge graph and recipe repository up to date.

Commands:
  pfmr learn flathub      — mine Flathub for manifests
  pfmr learn manifest     — analyze a local manifest file
  pfmr learn ingest       — ingest a probe report JSON
  pfmr learn export       — export graph knowledge to repo files
  pfmr learn stats        — show knowledge graph statistics
  pfmr learn graph show   — dump graph contents
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from pfmr.learn.graph import KnowledgeGraph, KGNode, KGEdge, Rel
from pfmr.learn.manifest import ManifestAnalyzer
from pfmr.learn.flathub import FlathubMiner
from pfmr.learn.sandbox import SandboxLearner
from pfmr.learn.exporter import Exporter
from pfmr.utils.logging import get_logger

logger = get_logger(__name__)

learn_app = typer.Typer(
    name="learn",
    help="Mine, learn and export knowledge about Flatpak packages.",
    rich_markup_mode="rich",
)
console = Console()

_DEFAULT_KG_DIR = Path("knowledge")
_DEFAULT_REPO_ROOT = Path(".")


def _kg(knowledge_dir: Path) -> KnowledgeGraph:
    return KnowledgeGraph(knowledge_dir)


# ---------------------------------------------------------------------------
# pfmr learn flathub
# ---------------------------------------------------------------------------

@learn_app.command("flathub")
def cmd_flathub(
    limit: int = typer.Option(100, "--limit", "-n", help="Max repos to inspect"),
    knowledge_dir: Path = typer.Option(_DEFAULT_KG_DIR, "--knowledge-dir", "-k"),
    repo_root: Path = typer.Option(_DEFAULT_REPO_ROOT, "--repo-root", "-r"),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir"),
    token: Optional[str] = typer.Option(None, "--token", envvar="GITHUB_TOKEN"),
    prefix: Optional[list[str]] = typer.Option(None, "--prefix", "-p",
                                                help="Filter by app-id prefix"),
    export: bool = typer.Option(True, "--export/--no-export",
                                help="Auto-export new knowledge to repo files"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """
    Mine Flathub GitHub repositories for Python package manifests.

    Discovers what native deps Python packages need by analyzing real
    Flathub manifests and adds the knowledge to the graph.
    """
    import os
    if verbose:
        os.environ["PFMR_LOG_LEVEL"] = "DEBUG"

    miner = FlathubMiner(
        cache_dir=cache_dir,
        github_token=token,
        app_id_prefixes=list(prefix or []),
    )

    with console.status(f"[bold green]Mining up to {limit} Flathub repos..."):
        result = miner.mine(limit=limit)

    rprint(f"\n[bold]Flathub mining complete[/bold]")
    rprint(f"  Repos inspected   : {result.total_repos}")
    rprint(f"  Manifests found   : {result.manifests_found}")
    rprint(f"  Python apps       : {result.python_apps}")

    if result.errors:
        rprint(f"  [red]Errors          : {len(result.errors)}[/red]")

    # Feed into KnowledgeGraph
    kg = _kg(knowledge_dir)
    added = 0
    for analysis in result.analyses:
        added += _ingest_analysis_into_graph(kg, analysis)

    if not dry_run:
        kg.save()
        rprint(f"\n[green]Added {added} new facts to knowledge graph[/green]")

    if export and not dry_run:
        exporter = Exporter(kg, repo_root)
        exp_report = exporter.export(dry_run=dry_run)
        for c in exp_report.created + exp_report.updated:
            rprint(f"  [cyan]{c.action}[/cyan] {c.path}")

    if dry_run:
        rprint(f"\n[yellow]Dry run — {added} facts would be added[/yellow]")


# ---------------------------------------------------------------------------
# pfmr learn manifest
# ---------------------------------------------------------------------------

@learn_app.command("manifest")
def cmd_manifest(
    manifest_path: Path = typer.Argument(..., help="Path to manifest JSON or YAML"),
    knowledge_dir: Path = typer.Option(_DEFAULT_KG_DIR, "--knowledge-dir", "-k"),
    repo_root: Path = typer.Option(_DEFAULT_REPO_ROOT, "--repo-root", "-r"),
    export: bool = typer.Option(False, "--export", help="Export new facts to repo files"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """
    Analyze a local Flatpak manifest and add knowledge to the graph.
    """
    analyzer = ManifestAnalyzer()
    analysis = analyzer.analyze(manifest_path)
    if analysis is None:
        rprint(f"[red]Could not parse manifest: {manifest_path}[/red]")
        raise typer.Exit(1)

    rprint(f"\n[bold cyan]{analysis.app_id}[/bold cyan]")
    rprint(f"  runtime    : {analysis.runtime}//{analysis.sdk_version}")
    rprint(f"  sdk        : {analysis.sdk}")
    if analysis.sdk_extensions:
        rprint(f"  extensions : {analysis.sdk_extensions}")

    if analysis.python_packages:
        rprint(f"\n[bold]Python packages[/bold] ({len(analysis.python_packages)}):")
        for pkg in analysis.python_packages:
            rprint(f"  {pkg}")

    if analysis.native_modules:
        rprint(f"\n[bold]Native modules[/bold] ({len(analysis.native_modules)}):")
        for mod in analysis.native_modules:
            rprint(f"  {mod.module_name}  [{mod.buildsystem}]  pc={mod.pkgconfig_names}")

    kg = _kg(knowledge_dir)
    added = _ingest_analysis_into_graph(kg, analysis)

    if not dry_run:
        kg.save()
        rprint(f"\n[green]Added {added} new facts[/green]")

    if export and not dry_run:
        exporter = Exporter(kg, repo_root)
        exp_report = exporter.export(dry_run=dry_run)
        for c in exp_report.created + exp_report.updated:
            rprint(f"  [cyan]{c.action}[/cyan] {c.path}")


# ---------------------------------------------------------------------------
# pfmr learn ingest
# ---------------------------------------------------------------------------

@learn_app.command("ingest")
def cmd_ingest(
    report_path: Path = typer.Argument(..., help="Path to SandboxProbeReport JSON"),
    package: Optional[str] = typer.Option(None, "--package", "-p",
                                          help="Package name this report is about"),
    sdk: str = typer.Option("org.freedesktop.Sdk", "--sdk"),
    sdk_version: str = typer.Option("24.08", "--sdk-version"),
    knowledge_dir: Path = typer.Option(_DEFAULT_KG_DIR, "--knowledge-dir", "-k"),
    repo_root: Path = typer.Option(_DEFAULT_REPO_ROOT, "--repo-root", "-r"),
    export: bool = typer.Option(False, "--export"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """
    Ingest a SandboxProbeReport JSON file into the knowledge graph.

    The report JSON must match the SandboxProbeReport dataclass structure.
    Generate one with: pfmr probe <target> --json-report report.json
    """
    try:
        raw = json.loads(report_path.read_text())
    except Exception as exc:
        rprint(f"[red]Could not read report: {exc}[/red]")
        raise typer.Exit(1)

    from pfmr.models import SandboxProbeReport, SandboxError, SandboxErrorType
    errors = []
    for e in raw.get("errors", []):
        errors.append(SandboxError(
            error_type=SandboxErrorType(e.get("error_type", "unknown")),
            missing=e.get("missing", ""),
            source=e.get("source", ""),
            context=e.get("context", ""),
            raw_line=e.get("raw_line", ""),
        ))
    report = SandboxProbeReport(
        probed_packages=raw.get("probed_packages", []),
        errors=errors,
        missing_python_packages=raw.get("missing_python_packages", []),
        missing_native_libs=raw.get("missing_native_libs", []),
        missing_headers=raw.get("missing_headers", []),
        missing_pkgconfig=raw.get("missing_pkgconfig", []),
        sdk_sufficient=raw.get("sdk_sufficient", True),
        ran=raw.get("ran", True),
    )

    kg = _kg(knowledge_dir)
    learner = SandboxLearner(kg)
    added = learner.ingest(report, package_name=package, sdk_id=sdk, sdk_version=sdk_version)

    if not dry_run:
        kg.save()

    rprint(f"[green]Ingested: {added} new facts[/green] (package={package})")

    if export and not dry_run:
        exporter = Exporter(kg, repo_root)
        exp_report = exporter.export(dry_run=False)
        for c in exp_report.created + exp_report.updated:
            rprint(f"  [cyan]{c.action}[/cyan] {c.path}")


# ---------------------------------------------------------------------------
# pfmr learn export
# ---------------------------------------------------------------------------

@learn_app.command("export")
def cmd_export(
    knowledge_dir: Path = typer.Option(_DEFAULT_KG_DIR, "--knowledge-dir", "-k"),
    repo_root: Path = typer.Option(_DEFAULT_REPO_ROOT, "--repo-root", "-r"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    hints_only: bool = typer.Option(False, "--hints-only"),
    recipes_only: bool = typer.Option(False, "--recipes-only"),
):
    """
    Export knowledge graph facts to repository files.

    Generates / updates:
      recipes/native/*.yaml
      pfmr/data/native-hints/packages.toml
    """
    kg = _kg(knowledge_dir)
    exporter = Exporter(kg, repo_root)

    if hints_only:
        exp_report = exporter.export_native_hints(dry_run=dry_run)
    elif recipes_only:
        exp_report = exporter.export_native_recipes(dry_run=dry_run)
    else:
        exp_report = exporter.export(dry_run=dry_run)

    if not exp_report.changes:
        rprint("[yellow]Nothing to export.[/yellow]")
        return

    table = Table(title=f"Export results ({'dry run' if dry_run else 'applied'})")
    table.add_column("Action", style="cyan")
    table.add_column("File")
    table.add_column("Reason", style="dim")
    for c in exp_report.changes:
        color = "green" if c.action == "create" else ("yellow" if c.action == "update" else "dim")
        table.add_row(f"[{color}]{c.action}[/{color}]", str(c.path), c.reason)
    console.print(table)


# ---------------------------------------------------------------------------
# pfmr learn stats
# ---------------------------------------------------------------------------

@learn_app.command("stats")
def cmd_stats(
    knowledge_dir: Path = typer.Option(_DEFAULT_KG_DIR, "--knowledge-dir", "-k"),
):
    """Show knowledge graph statistics."""
    kg = _kg(knowledge_dir)
    stats = kg.stats()

    rprint(f"\n[bold]Knowledge Graph[/bold] — {knowledge_dir}")
    rprint(f"  Total nodes  : {stats['total_nodes']}")
    rprint(f"  Total edges  : {stats['total_edges']}")

    if stats["nodes_by_type"]:
        rprint("\n  [bold]Nodes by type:[/bold]")
        for t, n in sorted(stats["nodes_by_type"].items()):
            rprint(f"    {t:<15} {n}")

    if stats["edges_by_relation"]:
        rprint("\n  [bold]Edges by relation:[/bold]")
        for r, n in sorted(stats["edges_by_relation"].items()):
            rprint(f"    {r:<30} {n}")


# ---------------------------------------------------------------------------
# pfmr learn graph
# ---------------------------------------------------------------------------

graph_app = typer.Typer(help="Inspect and query the knowledge graph.")
learn_app.add_typer(graph_app, name="graph")


@graph_app.command("show")
def graph_show(
    node_id: Optional[str] = typer.Argument(None, help="Node ID to show (default: all)"),
    knowledge_dir: Path = typer.Option(_DEFAULT_KG_DIR, "--knowledge-dir", "-k"),
    node_type: Optional[str] = typer.Option(None, "--type", "-t"),
    min_confidence: float = typer.Option(0.0, "--min-confidence"),
):
    """Show nodes and their edges in the knowledge graph."""
    kg = _kg(knowledge_dir)

    if node_id:
        node = kg.node(node_id)
        if not node:
            rprint(f"[red]Node '{node_id}' not found.[/red]")
            raise typer.Exit(1)
        _print_node(kg, node, min_confidence)
    else:
        filter_type = node_type
        nodes = kg.nodes_of_type(filter_type) if filter_type else list(kg._nodes.values())
        for node in sorted(nodes, key=lambda n: (n.node_type, n.id)):
            _print_node(kg, node, min_confidence)


@graph_app.command("deps")
def graph_deps(
    package: str = typer.Argument(..., help="Package name"),
    knowledge_dir: Path = typer.Option(_DEFAULT_KG_DIR, "--knowledge-dir", "-k"),
):
    """Show all known native deps for a Python package."""
    kg = _kg(knowledge_dir)
    from packaging.utils import canonicalize_name
    pkg_id = canonicalize_name(package)

    deps = kg.requires(pkg_id)
    if not deps:
        rprint(f"[yellow]No deps found for '{pkg_id}'.[/yellow]")
        return

    rprint(f"\n[bold cyan]{pkg_id}[/bold cyan] requires:")
    for dep in deps:
        edges = kg.edges_from(pkg_id)
        edge = next((e for e in edges if e.to_id == dep), None)
        conf = f"{edge.confidence:.1f}" if edge else "?"
        rprint(f"  [{conf}] {dep}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ingest_analysis_into_graph(kg: KnowledgeGraph, analysis) -> int:
    """Convert a ManifestAnalysis into KnowledgeGraph facts."""
    from packaging.utils import canonicalize_name
    today = __import__("datetime").date.today().isoformat()
    added = 0
    source = f"flathub:{analysis.app_id}" if analysis.app_id else analysis.source_path

    for pkg_name in analysis.python_packages:
        canonical = canonicalize_name(pkg_name)
        changed = kg.add_node(KGNode(
            id=canonical,
            node_type="package",
            attrs={"pypi_name": pkg_name},
        ))
        if changed:
            added += 1

    for native_mod in analysis.native_modules:
        lib_id = native_mod.module_name
        changed = kg.add_node(KGNode(
            id=lib_id,
            node_type="library",
            attrs={
                "buildsystem": native_mod.buildsystem,
                "source_url": native_mod.source_url or "",
                "source_sha256": native_mod.source_sha256 or "",
                "pkgconfig": ",".join(native_mod.pkgconfig_names),
                "source": source,
            },
        ))
        if changed:
            added += 1

        # Record that this sdk-extension is needed when this lib is present
        for ext in analysis.sdk_extensions:
            kg.add_node(KGNode(id=ext, node_type="extension", attrs={"extension_id": ext}))

    # Associate Python packages with native modules found in the same manifest
    for pkg_name in analysis.python_packages:
        canonical = canonicalize_name(pkg_name)
        for native_mod in analysis.native_modules:
            for pc in native_mod.pkgconfig_names:
                edge = KGEdge(
                    from_id=canonical,
                    to_id=pc,
                    relation=Rel.REQUIRES_PKGCONFIG,
                    confidence=0.6,   # manifest co-occurrence is suggestive, not definitive
                    source=source,
                    updated=today,
                )
                if kg.add_edge(edge):
                    added += 1

    return added


def _print_node(kg: KnowledgeGraph, node: KGNode, min_confidence: float) -> None:
    rprint(f"\n[bold cyan]{node.id}[/bold cyan]  [dim]({node.node_type})[/dim]")
    for k, v in node.attrs.items():
        rprint(f"  {k}: {v}")
    edges = [
        e for e in kg.edges_from(node.id) + kg.edges_to(node.id)
        if e.confidence >= min_confidence
    ]
    if edges:
        rprint(f"  [dim]edges ({len(edges)}):[/dim]")
        for e in edges[:10]:
            direction = "->" if e.from_id == node.id else "<-"
            other = e.to_id if e.from_id == node.id else e.from_id
            rprint(f"    {direction} {other}  [{e.relation}]  conf={e.confidence:.1f}")
        if len(edges) > 10:
            rprint(f"    ... and {len(edges)-10} more")