"""
pfmr.learn.cli — standalone learning commands.

All commands write directly to recipes/ and data/ — no knowledge graph.
Export is the default behavior for every command.

Commands:
  pfmr learn flathub          — mine Flathub repos (resumable)
  pfmr learn manifest <path>  — analyze a manifest file or directory
  pfmr learn shared-modules   — import modules from a shared-modules clone
  pfmr learn sdk probe        — introspect an installed SDK → sdk-profile TOML
  pfmr learn sdk probe-all    — probe all default SDKs and extensions
  pfmr learn sdk list         — list available sdk-profile TOMLs
  pfmr learn stats            — show recipe/data counts
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from pfmr.learn.manifest import ManifestAnalyzer
from pfmr.learn.flathub import FlathubMiner
from pfmr.learn.shared_modules import SharedModulesImporter
from pfmr.learn.exporter import Exporter, ExportReport
from pfmr.utils.logging import get_logger

logger = get_logger(__name__)

learn_app = typer.Typer(
    name="learn",
    help="Mine manifests and import recipes without a CI environment.",
    rich_markup_mode="rich",
)
console = Console()

_DEFAULT_REPO_ROOT = Path(".")


# ---------------------------------------------------------------------------
# Shared helper: analysis → recipes
# ---------------------------------------------------------------------------

def _analyses_to_recipes(analyses, repo_root: Path, dry_run: bool) -> ExportReport:
    """Convert ManifestAnalysis objects directly to recipe files."""
    exporter = Exporter(analyses, repo_root)
    return exporter.export(dry_run=dry_run)


def _print_export_report(report: ExportReport, dry_run: bool) -> None:
    if not any(report.created + report.updated):
        rprint("[dim]No new recipe files.[/dim]")
        return
    label = "Would write" if dry_run else "Written"
    for c in report.created:
        rprint(f"  [green]create[/green]  {c.path}  [dim]{c.reason}[/dim]")
    for c in report.updated:
        rprint(f"  [yellow]update[/yellow]  {c.path}  [dim]{c.reason}[/dim]")


# ---------------------------------------------------------------------------
# pfmr learn flathub
# ---------------------------------------------------------------------------

@learn_app.command("flathub")
def cmd_flathub(
    limit: int = typer.Option(100, "--limit", "-n",
        help="Max new repos to process this run"),
    repo_root: Path = typer.Option(_DEFAULT_REPO_ROOT, "--repo-root", "-r"),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir"),
    token: Optional[str] = typer.Option(None, "--token", envvar="GITHUB_TOKEN"),
    prefix: Optional[list[str]] = typer.Option(None, "--prefix", "-p"),
    no_export: bool = typer.Option(False, "--no-export"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    reset: bool = typer.Option(False, "--reset", help="Reset progress, start over"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """
    Mine Flathub GitHub repos for native library module recipes.

    Mines ALL repos (not just Python apps) — any manifest can contain
    native modules worth importing as recipes.

    Progress is tracked automatically so runs can be interrupted and
    resumed:
      pfmr learn flathub --limit 200   # first 200 new repos
      pfmr learn flathub --limit 200   # next 200 new repos
      pfmr learn flathub --reset       # restart from scratch
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
    rprint(f"\n[dim]Already processed: {progress.count()} repos. Mining {limit} new...[/dim]")

    with console.status("[bold green]Mining Flathub..."):
        result = miner.mine(limit=limit)

    rprint(f"\n[bold]Flathub mining[/bold]")
    rprint(f"  New repos processed : {result.manifests_found + len(result.errors)}")
    rprint(f"  Manifests extracted : {result.manifests_found}")
    rprint(f"  Skipped (cached)    : {result.skipped_cached}")
    rprint(f"  Total done so far   : {progress.count()}")

    if not no_export:
        rprint("\n[bold]Exporting recipes...[/bold]")
        exporter = Exporter(result.analyses, repo_root)
        report = exporter.export(dry_run=dry_run)
        _print_export_report(report, dry_run)
        total = len(report.created) + len(report.updated)
        rprint(f"\n[green]{total} recipe file(s) {'would be ' if dry_run else ''}written[/green]")


# ---------------------------------------------------------------------------
# pfmr learn manifest
# ---------------------------------------------------------------------------

@learn_app.command("manifest")
def cmd_manifest(
    target: Path = typer.Argument(
        ...,
        help="Manifest file (JSON/YAML) or directory to scan recursively",
    ),
    repo_root: Path = typer.Option(_DEFAULT_REPO_ROOT, "--repo-root", "-r"),
    no_export: bool = typer.Option(False, "--no-export"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive"),
):
    """
    Analyze a Flatpak manifest file or directory and extract native recipes.

    Accepts a single JSON/YAML manifest or a directory (scanned recursively).
    Writes extracted native module recipes to recipes/native/.
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
        rprint("[yellow]No manifests found.[/yellow]")
        raise typer.Exit()

    # Quick summary
    total_native = sum(len(a.native_modules) for a in analyses)
    total_python = sum(len(a.python_packages) for a in analyses)
    rprint(f"  Native modules : {total_native}")
    rprint(f"  Python packages: {total_python}")

    if not no_export:
        rprint("\n[bold]Exporting recipes...[/bold]")
        exporter = Exporter(analyses, repo_root)
        report = exporter.export(dry_run=dry_run)
        _print_export_report(report, dry_run)


# ---------------------------------------------------------------------------
# pfmr learn shared-modules
# ---------------------------------------------------------------------------

@learn_app.command("shared-modules")
def cmd_shared_modules(
    modules_dir: Path = typer.Argument(
        ...,
        help="Path to a cloned shared-modules repo or any dir with module JSON files",
    ),
    repo_root: Path = typer.Option(_DEFAULT_REPO_ROOT, "--repo-root", "-r"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """
    Import native library recipes from a shared-modules directory.

    shared-modules (https://github.com/flathub/shared-modules) contains
    individual Flatpak module JSON files — not full manifests. This command
    scans the directory recursively and converts each module into a
    recipes/native/<id>.yaml file.

    Example:
      git clone https://github.com/flathub/shared-modules /tmp/shared-modules
      pfmr learn shared-modules /tmp/shared-modules
    """
    import os
    if verbose:
        os.environ["PFMR_LOG_LEVEL"] = "DEBUG"

    if not modules_dir.exists():
        rprint(f"[red]Directory not found: {modules_dir}[/red]")
        raise typer.Exit(1)

    importer = SharedModulesImporter(repo_root=repo_root)

    with console.status("[bold green]Scanning modules..."):
        report = importer.import_from(modules_dir, dry_run=dry_run)

    rprint(f"\n[bold]shared-modules import[/bold]")
    rprint(f"  Scanned           : {report.scanned}")
    rprint(f"  Imported          : {report.imported}")
    rprint(f"  Skipped (exists)  : {report.skipped_existing}")
    rprint(f"  Skipped (no src)  : {report.skipped_no_source}")

    if report.created:
        rprint(f"\n[bold]{'Would create' if dry_run else 'Created'} {len(report.created)} recipe(s):[/bold]")
        for p in report.created:
            rprint(f"  [green]{'(dry)' if dry_run else ''}[/green] {p}")

    if report.errors:
        rprint(f"\n[red]{len(report.errors)} error(s):[/red]")
        for e in report.errors[:5]:
            rprint(f"  {e}")


# ---------------------------------------------------------------------------
# pfmr learn sdk
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# pfmr learn ingest
# ---------------------------------------------------------------------------

@learn_app.command("ingest")
def cmd_ingest(
    report_path: Path = typer.Argument(..., help="SandboxProbeReport JSON file"),
    package: Optional[str] = typer.Option(None, "--package", "-p",
                                          help="Package name the report is about"),
    sdk: str = typer.Option("org.freedesktop.Sdk", "--sdk"),
    sdk_version: str = typer.Option("24.08", "--sdk-version"),
    repo_root: Path = typer.Option(_DEFAULT_REPO_ROOT, "--repo-root", "-r"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """
    Ingest a SandboxProbeReport JSON into recipes/python/.

    Writes or updates recipes/python/<package>.yaml with the deps found.
    Generate a report with: pfmr probe <target> --json-report report.json
    """
    from pfmr.learn.sandbox import SandboxLearner
    try:
        import json as _json
        raw = _json.loads(report_path.read_text())
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

    if dry_run:
        rprint(f"[dim]Dry run — would ingest {len(errors)} errors for package '{package}'[/dim]")
        return

    learner = SandboxLearner(repo_root=repo_root)
    written = learner.ingest(report, package_name=package, sdk_id=sdk, sdk_version=sdk_version)
    rprint(f"[green]Ingested: {written} recipe(s) written[/green]")

sdk_learn_app = typer.Typer(help="Download and introspect Flatpak SDKs.")
learn_app.add_typer(sdk_learn_app, name="sdk")


@sdk_learn_app.command("probe")
def cmd_sdk_probe(
    sdk: str = typer.Option("org.freedesktop.Sdk", "--sdk", "-s",
        help="SDK or Extension ID. Extensions (containing .Extension.) are detected automatically."),
    sdk_version: str = typer.Option("24.08", "--sdk-version", "-V"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o"),
    cleanup: bool = typer.Option(False, "--cleanup",
        help="Uninstall after probing to reclaim disk space"),
    no_install: bool = typer.Option(False, "--no-install",
        help="Skip flatpak install (must already be installed)"),
):
    """
    Introspect a Flatpak SDK or Extension and write a static profile TOML.

    Works with both base SDKs and extensions — auto-detected from the ID:

      # Base SDK:
      pfmr learn sdk probe -s org.freedesktop.Sdk -V 24.08

      # Extension (auto-routed to extension probe):
      pfmr learn sdk probe -s org.freedesktop.Sdk.Extension.node24 -V 25.08

    Requires flatpak to be installed. Uses flatpak build-init + flatpak build.
    For extensions the base SDK must also be installed.

    Written to:
      Base SDK:  pfmr/data/sdk-profiles/<sdk-id>/<version>.toml
      Extension: pfmr/data/extension-profiles/<shortname>.toml
    """
    from pfmr.learn.sdk_probe import SDKProber, _is_extension, _base_sdk_from_extension

    prober = SDKProber(
        output_dir=output_dir,
        auto_install=not no_install,
        cleanup_after=cleanup,
    )
    if not prober.is_available():
        rprint("[red]flatpak not found. Install with your package manager.[/red]")
        raise typer.Exit(1)

    # Show what we're about to do
    if _is_extension(sdk):
        base = _base_sdk_from_extension(sdk)
        rprint(f"Detected extension: [cyan]{sdk}[/cyan]")
        rprint(f"Base SDK          : [dim]{base}//{sdk_version}[/dim]")
        rprint(f"Mount path        : [dim]/usr/lib/sdk/{sdk.split('.')[-1]}[/dim]")
    else:
        rprint(f"Probing SDK: [cyan]{sdk}//{sdk_version}[/cyan]")

    with console.status(f"[bold green]Running probe..."):
        result = prober.probe_sdk(sdk, sdk_version)

    if result.success:
        rprint(f"[bold green]OK[/bold green] — {result.sdk_id}//{result.sdk_version}")
        rprint(f"  pkg-config : {len(result.pkgconfig)}")
        rprint(f"  libraries  : {len(result.libraries)}")
        rprint(f"  executables: {', '.join(result.executables[:8])}"
               + (f" +{len(result.executables)-8} more" if len(result.executables) > 8 else ""))
    else:
        rprint(f"[bold red]Failed[/bold red]: {result.error}")
        raise typer.Exit(1)


@sdk_learn_app.command("probe-ext")
def cmd_sdk_probe_ext(
    ext_id: str = typer.Argument(..., help="Full extension ID"),
    sdk_version: str = typer.Option("24.08", "--sdk-version", "-V"),
    base_sdk: Optional[str] = typer.Option(None, "--base-sdk"),
    cleanup: bool = typer.Option(False, "--cleanup"),
):
    """Probe a Flatpak SDK extension and update its extension profile."""
    from pfmr.learn.sdk_probe import SDKProber

    prober = SDKProber(cleanup_after=cleanup)
    if not prober.is_available():
        rprint("[red]flatpak not found.[/red]")
        raise typer.Exit(1)

    with console.status(f"[bold green]Probing {ext_id}..."):
        result = prober.probe_extension(ext_id, sdk_version, base_sdk=base_sdk)

    if result.success:
        rprint(f"[bold green]OK[/bold green] — {ext_id}")
        rprint(f"  executables: {result.executables}")
        rprint(f"  pkg-config : {len(result.pkgconfig)}")
    else:
        rprint(f"[red]Failed: {result.error}[/red]")
        raise typer.Exit(1)


@sdk_learn_app.command("probe-all")
def cmd_sdk_probe_all(
    cleanup: bool = typer.Option(False, "--cleanup"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir"),
    skip_extensions: bool = typer.Option(False, "--skip-extensions"),
):
    """Probe all default SDKs and extensions and write profile TOMLs."""
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
    """List available SDK profile TOMLs."""
    from pfmr.resolvers.sdk_capability import _BUILTIN_PROFILES_DIR
    profiles = sorted(_BUILTIN_PROFILES_DIR.glob("**/*.toml"))
    table = Table(title=f"SDK profiles ({len(profiles)})")
    table.add_column("SDK", style="cyan")
    table.add_column("Version")
    for p in profiles:
        table.add_row(p.parent.name, p.stem)
    console.print(table)


# ---------------------------------------------------------------------------
# pfmr learn stats
# ---------------------------------------------------------------------------

@learn_app.command("stats")
def cmd_stats(
    repo_root: Path = typer.Option(_DEFAULT_REPO_ROOT, "--repo-root", "-r"),
):
    """Show counts of recipes and data files."""
    def _count(directory: Path, glob: str) -> int:
        if not directory.exists():
            return 0
        return len(list(directory.glob(glob)))

    rprint(f"\n[bold]pfmr repository stats[/bold] ({repo_root.resolve()})")
    rprint(f"  recipes/native/   : {_count(repo_root/'recipes'/'native',  '*.yaml')}")
    rprint(f"  recipes/python/   : {_count(repo_root/'recipes'/'python',  '*.yaml')}")
    rprint(f"  sdk-profiles      : {_count(repo_root/'pfmr'/'data'/'sdk-profiles', '**/*.toml')}")
    rprint(f"  extension-profiles: {_count(repo_root/'pfmr'/'data'/'extension-profiles', '*.toml')}")
    rprint(f"  native-hints      : {_count(repo_root/'pfmr'/'data'/'native-hints', '*.toml')}")
    rprint("")
    rprint("  [dim]Note: extensions are data (extension-profiles/), not recipes.[/dim]")