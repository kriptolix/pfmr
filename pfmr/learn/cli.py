"""
pfmr.learn.cli
~~~~~~~~~~~~~~~
Standalone CLI commands for the learning subsystem.

Designed to run locally without CI. All commands export by default.

Commands:
  pfmr learn flathub        — mine Flathub repos (resumable)
  pfmr learn manifest       — analyze a manifest file or directory
  pfmr learn ingest         — ingest a probe report JSON
  pfmr learn sdk probe      — download SDK, introspect, write profile, cleanup
  pfmr learn sdk probe-all  — probe all default SDKs and extensions
  pfmr learn export         — re-export all graph knowledge to recipes/
  pfmr learn stats          — show knowledge graph statistics
  pfmr learn graph show     — inspect graph nodes/edges
  pfmr learn graph deps     — show deps for a package
"""
from __future__ import annotations

import json
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


def _export_and_print(kg: KnowledgeGraph, repo_root: Path, dry_run: bool) -> None:
    exporter = Exporter(kg, repo_root)
    report = exporter.export(dry_run=dry_run)
    if not report.changes:
        rprint("[dim]No new files to export.[/dim]")
        return
    for c in report.created + report.updated:
        verb = "[green]create[/green]" if c.action == "create" else "[yellow]update[/yellow]"
        rprint(f"  {verb}  {c.path}  [dim]{c.reason}[/dim]")


# ---------------------------------------------------------------------------
# pfmr learn flathub
# ---------------------------------------------------------------------------

@learn_app.command("flathub")
def cmd_flathub(
    limit: int = typer.Option(
        100, "--limit", "-n",
        help="Max new repos to process this run (already-processed are skipped automatically)",
    ),
    knowledge_dir: Path = typer.Option(_DEFAULT_KG_DIR, "--knowledge-dir", "-k"),
    repo_root: Path = typer.Option(_DEFAULT_REPO_ROOT, "--repo-root", "-r"),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir"),
    token: Optional[str] = typer.Option(None, "--token", envvar="GITHUB_TOKEN"),
    prefix: Optional[list[str]] = typer.Option(None, "--prefix", "-p",
                                                help="Filter repos by app-id prefix"),
    no_export: bool = typer.Option(False, "--no-export", help="Skip writing recipe files"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would change, don't write"),
    reset: bool = typer.Option(False, "--reset", help="Reset progress and start over"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """
    Mine Flathub GitHub repositories for manifest knowledge.

    Mines ALL repos (not just Python apps) — any manifest can contain native
    library modules whose recipes are useful.

    Progress is tracked in <cache-dir>/flathub-progress.json so you can
    run the command repeatedly to process the full Flathub org incrementally:

      pfmr learn flathub --limit 200   # first 200 new repos
      pfmr learn flathub --limit 200   # next 200 new repos
      pfmr learn flathub --reset       # start over from scratch
    """
    import os
    if verbose:
        os.environ["PFMR_LOG_LEVEL"] = "DEBUG"

    effective_cache = cache_dir or (Path.home() / ".cache" / "pfmr" / "flathub")
    miner = FlathubMiner(
        cache_dir=effective_cache,
        github_token=token,
        app_id_prefixes=list(prefix or []),
    )

    if reset:
        miner.reset_progress()
        rprint("[yellow]Progress reset.[/yellow]")

    progress = miner.progress()
    rprint(
        f"\n[dim]Already processed: {progress.count()} repos. "
        f"Mining up to {limit} new repos...[/dim]"
    )

    with console.status("[bold green]Mining Flathub..."):
        result = miner.mine(limit=limit)

    rprint(f"\n[bold]Flathub mining complete[/bold]")
    rprint(f"  New repos processed : {result.manifests_found + len(result.errors)}")
    rprint(f"  Manifests extracted : {result.manifests_found}")
    rprint(f"  Skipped (cached)    : {result.skipped_cached}")
    rprint(f"  Python apps found   : {result.python_apps}")
    rprint(f"  Total processed     : {progress.count()}")
    if result.errors:
        rprint(f"  [dim]Failed fetches   : {len(result.errors)}[/dim]")

    kg = _kg(knowledge_dir)
    added = 0
    for analysis in result.analyses:
        added += _ingest_analysis_into_graph(kg, analysis)

    if not dry_run:
        kg.save()

    rprint(f"\n[green]Added {added} new facts[/green]" if not dry_run
           else f"\n[yellow]{added} facts would be added (dry run)[/yellow]")

    if not no_export and not dry_run:
        rprint("\n[bold]Exporting to recipes/...[/bold]")
        _export_and_print(kg, repo_root, dry_run=False)
    elif dry_run:
        rprint("\n[bold]Dry run — export preview:[/bold]")
        _export_and_print(kg, repo_root, dry_run=True)


# ---------------------------------------------------------------------------
# pfmr learn manifest
# ---------------------------------------------------------------------------

@learn_app.command("manifest")
def cmd_manifest(
    target: Path = typer.Argument(
        ...,
        help="Path to a manifest file (JSON/YAML) OR a directory to scan recursively",
    ),
    knowledge_dir: Path = typer.Option(_DEFAULT_KG_DIR, "--knowledge-dir", "-k"),
    repo_root: Path = typer.Option(_DEFAULT_REPO_ROOT, "--repo-root", "-r"),
    no_export: bool = typer.Option(False, "--no-export"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive",
                                   help="When target is a dir, recurse into subdirs"),
):
    """
    Analyze a manifest file or directory and add knowledge to the graph.

    Accepts:
      - A single manifest file (JSON or YAML)
      - A directory — scans recursively for all *.json/*.yaml/*.yml files
        that look like Flatpak manifests

    Especially useful for the shared-modules repository:
      pfmr learn manifest /path/to/shared-modules/

    Export to recipes/ is the default behavior.
    """
    analyzer = ManifestAnalyzer()

    if target.is_dir():
        analyses = analyzer.analyze_directory(target, recursive=recursive)
        rprint(f"\nFound [bold]{len(analyses)}[/bold] manifests in {target}")
    elif target.is_file():
        analysis = analyzer.analyze(target)
        analyses = [analysis] if analysis else []
    else:
        rprint(f"[red]Not found: {target}[/red]")
        raise typer.Exit(1)

    if not analyses:
        rprint("[yellow]No manifests found or parsed.[/yellow]")
        raise typer.Exit()

    # Summary table
    table = Table(title=f"Manifest analysis ({len(analyses)} files)")
    table.add_column("App ID", style="cyan")
    table.add_column("SDK", style="dim")
    table.add_column("Python pkgs", justify="right")
    table.add_column("Native modules", justify="right")
    table.add_column("Extensions")
    for a in analyses[:40]:  # cap display at 40
        table.add_row(
            a.app_id or a.source_path.split("/")[-1],
            f"{a.sdk}//{a.sdk_version}" if a.sdk else "-",
            str(len(a.python_packages)),
            str(len(a.native_modules)),
            ", ".join(e.split(".")[-1] for e in a.sdk_extensions[:2])
            + ("..." if len(a.sdk_extensions) > 2 else ""),
        )
    if len(analyses) > 40:
        table.add_row(f"... +{len(analyses)-40} more", "", "", "", "")
    console.print(table)

    kg = _kg(knowledge_dir)
    added = 0
    for analysis in analyses:
        added += _ingest_analysis_into_graph(kg, analysis)

    if not dry_run:
        kg.save()

    rprint(f"\n[green]Added {added} new facts[/green]" if not dry_run
           else f"\n[yellow]{added} facts would be added (dry run)[/yellow]")

    if not no_export and not dry_run:
        rprint("\n[bold]Exporting to recipes/...[/bold]")
        _export_and_print(kg, repo_root, dry_run=False)
    elif dry_run:
        _export_and_print(kg, repo_root, dry_run=True)


# ---------------------------------------------------------------------------
# pfmr learn ingest
# ---------------------------------------------------------------------------

@learn_app.command("ingest")
def cmd_ingest(
    report_path: Path = typer.Argument(..., help="Path to SandboxProbeReport JSON"),
    package: Optional[str] = typer.Option(None, "--package", "-p"),
    sdk: str = typer.Option("org.freedesktop.Sdk", "--sdk"),
    sdk_version: str = typer.Option("24.08", "--sdk-version"),
    knowledge_dir: Path = typer.Option(_DEFAULT_KG_DIR, "--knowledge-dir", "-k"),
    repo_root: Path = typer.Option(_DEFAULT_REPO_ROOT, "--repo-root", "-r"),
    no_export: bool = typer.Option(False, "--no-export"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """
    Ingest a SandboxProbeReport JSON into the knowledge graph.

    Export to recipes/ is the default behavior.
    """
    try:
        raw = json.loads(report_path.read_text())
    except Exception as exc:
        rprint(f"[red]Could not read report: {exc}[/red]")
        raise typer.Exit(1)

    from pfmr.models import SandboxProbeReport, SandboxError, SandboxErrorType
    errors = [
        SandboxError(
            error_type=SandboxErrorType(e.get("error_type", "unknown")),
            missing=e.get("missing", ""),
            source=e.get("source", ""),
            context=e.get("context", ""),
            raw_line=e.get("raw_line", ""),
        )
        for e in raw.get("errors", [])
    ]
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

    rprint(f"[green]Ingested: {added} new facts[/green]")

    if not no_export and not dry_run:
        _export_and_print(kg, repo_root, dry_run=False)
    elif dry_run:
        _export_and_print(kg, repo_root, dry_run=True)


# ---------------------------------------------------------------------------
# pfmr learn sdk
# ---------------------------------------------------------------------------

sdk_learn_app = typer.Typer(help="Download and introspect Flatpak SDKs and extensions.")
learn_app.add_typer(sdk_learn_app, name="sdk")


@sdk_learn_app.command("probe")
def cmd_sdk_probe(
    sdk: str = typer.Option(..., "--sdk", "-s"),
    sdk_version: str = typer.Option("25.08", "--sdk-version", "-V"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o",
                                              help="Where to write the profile TOML"),
    cleanup: bool = typer.Option(False, "--cleanup",
                                 help="Uninstall the SDK after probing to reclaim disk space"),
    no_install: bool = typer.Option(False, "--no-install",
                                    help="Skip flatpak install (SDK must already be present)"),
):
    """
    Download and introspect a Flatpak SDK, then write a static profile.

    Runs without CI — just needs flatpak on the host.

    The generated profile is written to:
      pfmr/data/sdk-profiles/<sdk-id>/<version>.toml

    Use --cleanup to remove the SDK after probing (saves ~1-2 GB per SDK).
    """
    from pfmr.learn.sdk_probe import SDKProber

    prober = SDKProber(
        output_dir=output_dir,
        auto_install=not no_install,
        cleanup_after=cleanup,
    )
    if not prober.is_available():
        rprint("[red]flatpak not found. Install flatpak and try again.[/red]")
        raise typer.Exit(1)

    with console.status(f"[bold green]Probing {sdk}//{sdk_version}..."):
        result = prober.probe_sdk(sdk, sdk_version)

    if result.success:
        rprint(f"[bold green]Success[/bold green] — {sdk}//{sdk_version}")
        rprint(f"  pkg-config modules : {len(result.pkgconfig)}")
        rprint(f"  shared libraries   : {len(result.libraries)}")
        rprint(f"  executables        : {len(result.executables)}")
    else:
        rprint(f"[red]Failed: {result.error}[/red]")
        raise typer.Exit(1)


@sdk_learn_app.command("probe-ext")
def cmd_sdk_probe_ext(
    ext_id: str = typer.Argument(..., help="Full extension ID, e.g. org.freedesktop.Sdk.Extension.rust-stable"),
    sdk_version: str = typer.Option("24.08", "--sdk-version", "-V"),
    base_sdk: Optional[str] = typer.Option(None, "--base-sdk"),
    cleanup: bool = typer.Option(False, "--cleanup"),
):
    """
    Probe a Flatpak SDK extension and update its extension profile.
    """
    from pfmr.learn.sdk_probe import SDKProber

    prober = SDKProber(cleanup_after=cleanup)
    if not prober.is_available():
        rprint("[red]flatpak not found.[/red]")
        raise typer.Exit(1)

    with console.status(f"[bold green]Probing extension {ext_id}..."):
        result = prober.probe_extension(ext_id, sdk_version, base_sdk=base_sdk)

    if result.success:
        rprint(f"[bold green]Success[/bold green] — {ext_id}")
        rprint(f"  executables  : {result.executables}")
        rprint(f"  pkg-config   : {len(result.pkgconfig)} entries")
        rprint(f"  libraries    : {len(result.libraries)} entries")
    else:
        rprint(f"[red]Failed: {result.error}[/red]")
        raise typer.Exit(1)


@sdk_learn_app.command("probe-all")
def cmd_sdk_probe_all(
    cleanup: bool = typer.Option(False, "--cleanup",
                                 help="Uninstall each SDK after probing"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir"),
    skip_extensions: bool = typer.Option(False, "--skip-extensions"),
):
    """
    Probe all default SDKs and extensions in sequence.

    Populates pfmr/data/sdk-profiles/ and updates extension profiles.
    Can be run locally; use --cleanup to save disk space.
    """
    from pfmr.learn.sdk_probe import SDKProber, DEFAULT_SDK_LIST, DEFAULT_EXTENSION_LIST

    prober = SDKProber(output_dir=output_dir, cleanup_after=cleanup)
    if not prober.is_available():
        rprint("[red]flatpak not found.[/red]")
        raise typer.Exit(1)

    results = prober.probe_all(
        sdk_list=DEFAULT_SDK_LIST,
        ext_list=[] if skip_extensions else DEFAULT_EXTENSION_LIST,
    )

    table = Table(title="SDK probe results")
    table.add_column("SDK / Extension", style="cyan")
    table.add_column("Version")
    table.add_column("Status", justify="center")
    table.add_column("pc", justify="right")
    table.add_column("libs", justify="right")
    for r in results:
        status = "[green]ok[/green]" if r.success else f"[red]{r.error[:30]}[/red]"
        table.add_row(r.sdk_id, r.sdk_version, status,
                      str(len(r.pkgconfig)), str(len(r.libraries)))
    console.print(table)


@sdk_learn_app.command("list")
def cmd_sdk_list():
    """List all available SDK profiles (built-in and cached)."""
    from pfmr.resolvers.sdk_capability import _BUILTIN_PROFILES_DIR
    profiles = sorted(_BUILTIN_PROFILES_DIR.glob("**/*.toml"))

    table = Table(title=f"Available SDK profiles ({len(profiles)})")
    table.add_column("SDK", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Path", style="dim")
    for p in profiles:
        table.add_row(p.parent.name, p.stem, str(p))
    console.print(table)


# ---------------------------------------------------------------------------
# pfmr learn export
# ---------------------------------------------------------------------------

@learn_app.command("export")
def cmd_export(
    knowledge_dir: Path = typer.Option(_DEFAULT_KG_DIR, "--knowledge-dir", "-k"),
    repo_root: Path = typer.Option(_DEFAULT_REPO_ROOT, "--repo-root", "-r"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    native_only: bool = typer.Option(False, "--native-only"),
    python_only: bool = typer.Option(False, "--python-only"),
):
    """
    Export knowledge graph facts to recipes/ directory.

    Generates:
      recipes/native/<lib>.yaml   — how to build a native lib
      recipes/python/<pkg>.yaml   — what libs a Python package needs
    """
    kg = _kg(knowledge_dir)
    exporter = Exporter(kg, repo_root)

    if native_only:
        report = exporter.export_native_recipes(dry_run=dry_run)
    elif python_only:
        report = exporter.export_python_recipes(dry_run=dry_run)
    else:
        report = exporter.export(dry_run=dry_run)

    if not report.changes:
        rprint("[yellow]Nothing to export.[/yellow]")
        return

    table = Table(title=f"Export ({'dry run' if dry_run else 'applied'})")
    table.add_column("Action", style="cyan")
    table.add_column("File")
    table.add_column("Reason", style="dim")
    for c in report.changes:
        color = {"create": "green", "update": "yellow", "skip": "dim"}.get(c.action, "white")
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
    node_id: Optional[str] = typer.Argument(None),
    knowledge_dir: Path = typer.Option(_DEFAULT_KG_DIR, "--knowledge-dir", "-k"),
    node_type: Optional[str] = typer.Option(None, "--type", "-t"),
    min_confidence: float = typer.Option(0.0, "--min-confidence"),
):
    """Show nodes and edges in the knowledge graph."""
    kg = _kg(knowledge_dir)
    if node_id:
        node = kg.node(node_id)
        if not node:
            rprint(f"[red]Node '{node_id}' not found.[/red]")
            raise typer.Exit(1)
        _print_node(kg, node, min_confidence)
    else:
        nodes = kg.nodes_of_type(node_type) if node_type else list(kg._nodes.values())
        for node in sorted(nodes, key=lambda n: (n.node_type, n.id)):
            _print_node(kg, node, min_confidence)


@graph_app.command("deps")
def graph_deps(
    package: str = typer.Argument(...),
    knowledge_dir: Path = typer.Option(_DEFAULT_KG_DIR, "--knowledge-dir", "-k"),
):
    """Show all known deps for a Python package."""
    from packaging.utils import canonicalize_name
    kg = _kg(knowledge_dir)
    pkg_id = canonicalize_name(package)
    deps = kg.requires(pkg_id)
    if not deps:
        rprint(f"[yellow]No deps found for '{pkg_id}'.[/yellow]")
        return
    rprint(f"\n[bold cyan]{pkg_id}[/bold cyan] requires:")
    for dep in deps:
        edge = next((e for e in kg.edges_from(pkg_id) if e.to_id == dep), None)
        conf = f"{edge.confidence:.1f}" if edge else "?"
        rprint(f"  [{conf}] {dep}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ingest_analysis_into_graph(kg: KnowledgeGraph, analysis) -> int:
    from packaging.utils import canonicalize_name
    import datetime
    today = datetime.date.today().isoformat()
    added = 0
    source = f"flathub:{analysis.app_id}" if analysis.app_id else analysis.source_path

    for pkg_name in analysis.python_packages:
        canonical = canonicalize_name(pkg_name)
        if kg.add_node(KGNode(id=canonical, node_type="package", attrs={"pypi_name": pkg_name})):
            added += 1

    for native_mod in analysis.native_modules:
        lib_id = native_mod.module_name
        if kg.add_node(KGNode(id=lib_id, node_type="library", attrs={
            "buildsystem": native_mod.buildsystem,
            "source_url": native_mod.source_url or "",
            "source_sha256": native_mod.source_sha256 or "",
            "pkgconfig": ",".join(native_mod.pkgconfig_names),
            "source": source,
        })):
            added += 1

        for ext in analysis.sdk_extensions:
            kg.add_node(KGNode(id=ext, node_type="extension", attrs={"extension_id": ext}))

    # co-occurrence edges: Python pkg + native module in same manifest
    for pkg_name in analysis.python_packages:
        canonical = canonicalize_name(pkg_name)
        for native_mod in analysis.native_modules:
            for pc in native_mod.pkgconfig_names:
                edge = KGEdge(
                    from_id=canonical, to_id=pc,
                    relation=Rel.REQUIRES_PKGCONFIG,
                    confidence=0.6,
                    source=source, updated=today,
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