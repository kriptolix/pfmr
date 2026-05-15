"""pfmr.cli - command-line interface (Phase 1 + Phase 2 SDKCapabilityResolver)."""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from pfmr import __version__
from pfmr.pipeline import Pipeline
from pfmr.recipes.db import RecipeDB
from pfmr.resolvers.sdk_capability import SDKCapabilityResolver, SDKQuery
from pfmr.resolvers.sdk_extension import SDKExtensionResolver
from pfmr.resolvers.native_dependency import NativeDependencyAnalyzer
from pfmr.sandbox.probe import BuildSandboxProber
from pfmr.learn.cli import learn_app
from pfmr.utils.logging import get_logger

logger = get_logger(__name__)

app = typer.Typer(
    name="pfmr",
    help="Python Flatpak Manifest Resolver - modern replacement for flatpak-pip-generator.",
    rich_markup_mode="rich",
)
console = Console()

recipes_app = typer.Typer(help="Manage local recipe database.")
app.add_typer(recipes_app, name="recipes")

sdk_app = typer.Typer(help="Inspect and probe Flatpak SDK capabilities.")
app.add_typer(sdk_app, name="sdk")

ext_app = typer.Typer(help="Inspect SDK Extension profiles and requirements.")
app.add_typer(ext_app, name="ext")

app.add_typer(learn_app, name="learn")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class OutputFormat(str, Enum):
    yaml = "yaml"
    json = "json"


def _make_pipeline(
    app_id: str,
    runtime: str,
    runtime_version: str,
    sdk: str,
    python_version: str,
    offline: bool = False,
) -> Pipeline:
    return Pipeline(
        app_id=app_id,
        runtime=runtime,
        runtime_version=runtime_version,
        sdk=sdk,
        python_version=python_version,
        offline=offline,
    )


# ---------------------------------------------------------------------------
# pfmr resolve
# ---------------------------------------------------------------------------

@app.command("resolve")
def resolve(
    target: str = typer.Argument(
        ...,
        help="Path to pyproject.toml / requirements.txt, or a package spec like 'requests'",
    ),
    python_version: str = typer.Option("3.11", "--python", "-p"),
    app_id: str = typer.Option("org.example.App", "--app-id"),
    runtime: str = typer.Option("org.freedesktop.Platform", "--runtime"),
    runtime_version: str = typer.Option("24.08", "--runtime-version"),
    sdk: str = typer.Option("org.freedesktop.Sdk", "--sdk"),
    offline: bool = typer.Option(False, "--offline"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Resolve Python dependencies and display a summary."""
    import os
    if verbose:
        os.environ["PFMR_LOG_LEVEL"] = "DEBUG"

    pipeline = _make_pipeline(app_id, runtime, runtime_version, sdk, python_version, offline)
    p = Path(target)

    with console.status("[bold green]Resolving dependencies..."):
        if p.exists() and p.name == "pyproject.toml":
            result = pipeline.resolve_pyproject(p)
        elif p.exists() and p.suffix in (".txt",):
            result = pipeline.resolve_requirements(p)
        else:
            result = pipeline.resolve_package(target)

    table = Table(title=f"Resolved packages ({len(result.packages)} total)")
    table.add_column("Package", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Wheel", justify="center")
    table.add_column("Backend", style="yellow")
    table.add_column("Native", justify="center")
    table.add_column("Direct", justify="center")

    for pkg in sorted(result.packages, key=lambda p: p.name.lower()):
        table.add_row(
            pkg.name,
            pkg.version,
            "[green]v[/green]" if pkg.wheel_available else "[red]x[/red]",
            pkg.build_backend.value,
            "[red]v[/red]" if pkg.requires_native else "-",
            "[bold]v[/bold]" if pkg.is_direct else "-",
        )
    console.print(table)

    if result.native_recipes:
        rprint(f"\n[bold]Native recipes:[/bold] {[r.id for r in result.native_recipes]}")
    if result.unresolved_natives:
        rprint(f"\n[bold red]Unresolved natives:[/bold red] {result.unresolved_natives}")
    rprint(f"\n[dim]Lockfile hash:[/dim] {result.lockfile_hash}")


# ---------------------------------------------------------------------------
# pfmr generate
# ---------------------------------------------------------------------------

@app.command("generate")
def generate(
    target: str = typer.Argument(...),
    output: Path = typer.Option(Path("flatpak-python.yaml"), "--output", "-o"),
    fmt: OutputFormat = typer.Option(OutputFormat.yaml, "--format", "-f"),
    python_version: str = typer.Option("3.11", "--python", "-p"),
    app_id: str = typer.Option("org.example.App", "--app-id"),
    runtime: str = typer.Option("org.freedesktop.Platform", "--runtime"),
    runtime_version: str = typer.Option("24.08", "--runtime-version"),
    sdk: str = typer.Option("org.freedesktop.Sdk", "--sdk"),
    offline: bool = typer.Option(False, "--offline"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    stdout: bool = typer.Option(False, "--stdout"),
):
    """Resolve dependencies and generate a Flatpak manifest module file."""
    import os
    if verbose:
        os.environ["PFMR_LOG_LEVEL"] = "DEBUG"

    pipeline = _make_pipeline(app_id, runtime, runtime_version, sdk, python_version, offline)
    p = Path(target)
    out_path = None if stdout else output

    with console.status("[bold green]Resolving and generating manifest..."):
        if p.exists() and p.name == "pyproject.toml":
            text = pipeline.run_from_pyproject(p, fmt.value, out_path)
        elif p.exists() and p.suffix in (".txt",):
            text = pipeline.run_from_requirements(p, fmt.value, out_path)
        else:
            text = pipeline.run_from_package(target, fmt.value, out_path)

    if stdout:
        print(text)
    else:
        rprint(f"[bold green]v[/bold green] Manifest written to [cyan]{output}[/cyan]")


# ---------------------------------------------------------------------------
# pfmr recipes
# ---------------------------------------------------------------------------

@recipes_app.command("list")
def recipes_list(
    recipe_dir: Optional[Path] = typer.Option(None, "--dir", "-d"),
):
    """List all recipes in the local database."""
    db = RecipeDB(recipe_dirs=[recipe_dir] if recipe_dir else None)
    if not db.all_recipes():
        rprint("[yellow]No recipes found.[/yellow]")
        raise typer.Exit()

    table = Table(title=f"Local recipes ({len(db)} total)")
    table.add_column("ID", style="cyan")
    table.add_column("Build system", style="yellow")
    table.add_column("Provides (.so)")
    table.add_column("pkg-config")
    for r in sorted(db.all_recipes(), key=lambda x: x.id):
        table.add_row(r.id, r.buildsystem, ", ".join(r.provides), ", ".join(r.pkgconfig))
    console.print(table)


@recipes_app.command("show")
def recipes_show(
    recipe_id: str = typer.Argument(...),
    recipe_dir: Optional[Path] = typer.Option(None, "--dir", "-d"),
):
    """Show full details of a specific recipe."""
    db = RecipeDB(recipe_dirs=[recipe_dir] if recipe_dir else None)
    recipe = db.find_by_id(recipe_id)
    if not recipe:
        rprint(f"[red]Recipe '{recipe_id}' not found.[/red]")
        raise typer.Exit(1)
    rprint(f"[bold cyan]{recipe.id}[/bold cyan]")
    rprint(f"  buildsystem : {recipe.buildsystem}")
    rprint(f"  provides    : {recipe.provides}")
    rprint(f"  pkgconfig   : {recipe.pkgconfig}")
    rprint(f"  headers     : {recipe.headers}")
    rprint(f"  aliases     : {recipe.aliases}")
    rprint(f"  cleanup     : {recipe.cleanup}")
    if recipe.config_opts:
        rprint(f"  config-opts : {recipe.config_opts}")
    if recipe.source:
        rprint(f"  source      : {recipe.source}")


# ---------------------------------------------------------------------------
# pfmr sdk
# ---------------------------------------------------------------------------

@sdk_app.command("list")
def sdk_list():
    """List all built-in SDK profiles available offline."""
    from pfmr.resolvers.sdk_capability import _BUILTIN_PROFILES_DIR
    profiles = sorted(_BUILTIN_PROFILES_DIR.glob("**/*.toml"))
    if not profiles:
        rprint("[yellow]No built-in profiles found.[/yellow]")
        raise typer.Exit()

    table = Table(title=f"Built-in SDK profiles ({len(profiles)} total)")
    table.add_column("SDK", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Source")
    for p in profiles:
        version = p.stem
        sdk_id = p.parent.name
        table.add_row(sdk_id, version, str(p.relative_to(_BUILTIN_PROFILES_DIR.parent.parent)))
    console.print(table)


@sdk_app.command("info")
def sdk_info(
    sdk: str = typer.Option("org.freedesktop.Sdk", "--sdk", "-s"),
    sdk_version: str = typer.Option("24.08", "--sdk-version", "-V"),
    offline: bool = typer.Option(True, "--offline/--live"),
    show_libs: bool = typer.Option(False, "--libs", help="Show library list"),
    show_pc: bool = typer.Option(False, "--pkgconfig", "--pc", help="Show pkg-config list"),
    show_exes: bool = typer.Option(False, "--exes", help="Show executables list"),
):
    """Show capabilities of a specific SDK profile."""
    resolver = SDKCapabilityResolver(
        sdk_id=sdk,
        sdk_version=sdk_version,
        offline=offline,
    )
    cap = resolver.capability()
    if not cap:
        rprint(f"[red]No profile found for {sdk}//{sdk_version}.[/red]")
        rprint("Use [bold]pfmr sdk probe[/bold] to generate one, or [bold]pfmr sdk list[/bold] to see available profiles.")
        raise typer.Exit(1)

    source = "[bold green]live probe[/bold green]" if cap.probed_live else "[dim]static profile[/dim]"
    rprint(f"\n[bold cyan]{cap.sdk_id}[/bold cyan] // [green]{cap.sdk_version}[/green]  ({source})")
    rprint(f"  pkg-config modules : {len(cap.pkgconfig)}")
    rprint(f"  shared libraries   : {len(cap.libraries)}")
    rprint(f"  executables        : {len(cap.executables)}")
    rprint(f"  headers            : {len(cap.headers)}")
    rprint(f"  python modules     : {len(cap.python_modules)}")

    if show_pc:
        rprint("\n[bold]pkg-config modules:[/bold]")
        for pc in sorted(cap.pkgconfig):
            rprint(f"  {pc}")
    if show_libs:
        rprint("\n[bold]Shared libraries:[/bold]")
        for lib in sorted(cap.libraries):
            rprint(f"  {lib}")
    if show_exes:
        rprint("\n[bold]Executables:[/bold]")
        for exe in sorted(cap.executables):
            rprint(f"  {exe}")


@sdk_app.command("check")
def sdk_check(
    deps: list[str] = typer.Argument(..., help="pkg-config names to check, e.g. openssl libffi"),
    sdk: str = typer.Option("org.freedesktop.Sdk", "--sdk", "-s"),
    sdk_version: str = typer.Option("24.08", "--sdk-version", "-V"),
    query_type: str = typer.Option("pkgconfig", "--type", "-t",
                                   help="Query type: pkgconfig | library | header | executable"),
    offline: bool = typer.Option(True, "--offline/--live"),
):
    """Check whether specific native deps are satisfied by the SDK."""
    resolver = SDKCapabilityResolver(
        sdk_id=sdk,
        sdk_version=sdk_version,
        offline=offline,
    )
    queries = [SDKQuery(value=d, query_type=query_type) for d in deps]
    report = resolver.resolve(queries)

    table = Table(title=f"SDK check: {sdk}//{sdk_version}")
    table.add_column("Dependency")
    table.add_column("Type", style="dim")
    table.add_column("Status", justify="center")
    table.add_column("Provided by", style="cyan")
    table.add_column("Recipe candidates", style="yellow")

    for check in report.checks:
        status = "[bold green]satisfied[/bold green]" if check.satisfied else "[bold red]MISSING[/bold red]"
        table.add_row(
            check.query,
            check.query_type,
            status,
            check.provided_by or "-",
            ", ".join(check.recipe_candidates) or "-",
        )
    console.print(table)

    if report.is_sufficient:
        rprint("\n[bold green]All dependencies satisfied by SDK.[/bold green]")
    else:
        rprint(f"\n[bold red]{len(report.missing)} dep(s) missing from SDK.[/bold red]")
        raise typer.Exit(1)


@sdk_app.command("probe")
def sdk_probe(
    sdk: str = typer.Option("org.freedesktop.Sdk", "--sdk", "-s"),
    sdk_version: str = typer.Option("24.08", "--sdk-version", "-V"),
):
    """
    Live-probe a locally installed Flatpak SDK and cache the result.
    Requires flatpak or flatpak-builder to be installed.
    """
    with console.status(f"[bold green]Probing {sdk}//{sdk_version}..."):
        resolver = SDKCapabilityResolver(
            sdk_id=sdk,
            sdk_version=sdk_version,
            offline=False,
            force_probe=True,
        )
    cap = resolver.capability()
    if not cap:
        rprint(f"[red]Probe failed. Is {sdk}//{sdk_version} installed?[/red]")
        rprint("Install with: [bold]flatpak install flathub " + sdk + "//" + sdk_version + "[/bold]")
        raise typer.Exit(1)

    if cap.probed_live:
        rprint(f"[bold green]v[/bold green] Live probe successful and cached.")
        rprint(f"  pkg-config: {len(cap.pkgconfig)}, libs: {len(cap.libraries)}, exes: {len(cap.executables)}")
    else:
        rprint("[yellow]Fell back to static profile (flatpak may not be available).[/yellow]")


# ---------------------------------------------------------------------------
# pfmr ext
# ---------------------------------------------------------------------------

@ext_app.command("list")
def ext_list():
    """List all available SDK Extension profiles."""
    from pfmr.resolvers.sdk_extension import _BUILTIN_EXTENSION_PROFILES_DIR
    resolver = SDKExtensionResolver()
    profiles = resolver.profiles()
    if not profiles:
        rprint("[yellow]No extension profiles found.[/yellow]")
        raise typer.Exit()

    table = Table(title=f"SDK Extension profiles ({len(profiles)} total)")
    table.add_column("Extension ID", style="cyan")
    table.add_column("Display name", style="green")
    table.add_column("Build backends", style="yellow")
    table.add_column("Package triggers")
    for p in sorted(profiles, key=lambda x: x.extension_id):
        table.add_row(
            p.extension_id,
            p.display_name,
            ", ".join(p.build_backends) or "-",
            ", ".join(p.package_triggers[:4]) + ("..." if len(p.package_triggers) > 4 else ""),
        )
    console.print(table)


@ext_app.command("show")
def ext_show(
    ext_id: str = typer.Argument(..., help="Extension ID (full or short name like rust-stable)"),
):
    """Show full details of a specific extension profile."""
    resolver = SDKExtensionResolver()
    profile = resolver.profile_by_id(ext_id)
    # Try short-name lookup
    if not profile:
        for p in resolver.profiles():
            if p.extension_id.endswith("." + ext_id) or p.extension_id.endswith("-" + ext_id):
                profile = p
                break
    if not profile:
        rprint(f"[red]Extension profile '{ext_id}' not found.[/red]")
        raise typer.Exit(1)

    rprint(f"[bold cyan]{profile.extension_id}[/bold cyan]")
    rprint(f"  display_name      : {profile.display_name}")
    rprint(f"  mount_path        : {profile.mount_path or '-'}")
    rprint(f"  build_backends    : {profile.build_backends or '-'}")
    rprint(f"  pkgconfig_triggers: {profile.pkgconfig_triggers or '-'}")
    rprint(f"  library_triggers  : {profile.library_triggers or '-'}")
    rprint(f"  package_triggers  : {profile.package_triggers}")
    rprint(f"  provides_exes     : {profile.provides_executables}")
    if profile.env:
        rprint("  env:")
        for k, v in profile.env.items():
            rprint(f"    {k} = {v}")
    if profile.compatible_sdks:
        rprint(f"  compatible_sdks   : {profile.compatible_sdks}")
    if profile.description:
        rprint(f"\\n  [dim]{profile.description.strip()}[/dim]")


@ext_app.command("check")
def ext_check(
    packages: list[str] = typer.Argument(
        ..., help="Python package names to check, e.g. cryptography orjson llvmlite"
    ),
    sdk: str = typer.Option("org.freedesktop.Sdk", "--sdk", "-s"),
    sdk_version: str = typer.Option("24.08", "--sdk-version", "-V"),
    forced: Optional[list[str]] = typer.Option(None, "--force", "-f",
                                                help="Force-include extension IDs"),
):
    """
    Determine which SDK Extensions are required for a set of packages.
    Packages can be bare names; native deps are inferred from the hints DB.
    """
    from pfmr.models import ResolvedPackage, BuildBackend
    from pfmr.models import SourceType

    analyzer = NativeDependencyAnalyzer()
    # Build minimal ResolvedPackage stubs
    pkgs: list[ResolvedPackage] = []
    for pkg_name in packages:
        pkg = ResolvedPackage(name=pkg_name, version="0.0.0", build_backend=BuildBackend.UNKNOWN)
        pkgs.append(pkg)

    # Run native analysis to fill native_deps
    analyzer.analyze(pkgs)

    ext_resolver = SDKExtensionResolver(forced_extensions=forced or [])
    report = ext_resolver.resolve(pkgs, sdk_id=sdk, sdk_version=sdk_version)

    if not report.required_extensions:
        rprint("[green]No SDK Extensions required.[/green]")
        return

    table = Table(title=f"Required extensions for: {', '.join(packages)}")
    table.add_column("Extension", style="cyan")
    table.add_column("Triggered by", style="yellow")
    table.add_column("Reason")
    table.add_column("env vars")
    for match in report.required_extensions:
        reason_strs = [f"{t}:{v}" for t, v in match.reasons[:3]]
        if len(match.reasons) > 3:
            reason_strs.append(f"+{len(match.reasons)-3} more")
        table.add_row(
            match.extension_id,
            ", ".join(match.triggered_by_packages[:3]),
            ", ".join(reason_strs),
            ", ".join(match.env.keys()) if match.env else "-",
        )
    console.print(table)

    rprint(f"\\n[bold]sdk-extensions to declare:[/bold]")
    for ext_id in report.extension_ids:
        rprint(f"  - {ext_id}")

# ---------------------------------------------------------------------------
# pfmr probe
# ---------------------------------------------------------------------------

@app.command("probe")
def probe(
    target: str = typer.Argument(
        ...,
        help="Path to pyproject.toml / requirements.txt, or a package spec like 'requests'",
    ),
    python_version: str = typer.Option("3.11", "--python", "-p"),
    runtime: str = typer.Option("org.freedesktop.Platform", "--runtime"),
    runtime_version: str = typer.Option("24.08", "--runtime-version"),
    sdk: str = typer.Option("org.freedesktop.Sdk", "--sdk"),
    ext: Optional[list[str]] = typer.Option(
        None, "--ext", "-e",
        help="SDK extensions to mount (e.g. org.freedesktop.Sdk.Extension.rust-stable)",
    ),
    work_dir: Optional[Path] = typer.Option(
        None, "--work-dir", "-w",
        help="Working directory for the sandbox (default: temp dir)",
    ),
    keep: bool = typer.Option(False, "--keep", help="Keep work-dir after probe for debugging"),
    timeout: int = typer.Option(120, "--timeout", "-t", help="Timeout per command (seconds)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """
    Probe packages inside a real Flatpak SDK sandbox.

    Builds a temporary org.pfmr.TestSandbox, sets up a Python venv,
    attempts to install each package, and reports what is missing.

    Requires flatpak-builder to be installed on the host.
    """
    import os
    if verbose:
        os.environ["PFMR_LOG_LEVEL"] = "DEBUG"

    pipeline = _make_pipeline(
        "org.pfmr.TestSandbox", runtime, runtime_version, sdk, python_version
    )
    p = Path(target)

    with console.status("[bold green]Resolving dependencies..."):
        if p.exists() and p.name == "pyproject.toml":
            result = pipeline.resolve_pyproject(p)
        elif p.exists() and p.suffix == ".txt":
            result = pipeline.resolve_requirements(p)
        else:
            result = pipeline.resolve_package(target)

    extensions = list(ext or []) or result.required_extensions

    prober = BuildSandboxProber(
        runtime=runtime,
        runtime_version=runtime_version,
        sdk=sdk,
        sdk_extensions=extensions,
        work_dir=work_dir,
        keep_work_dir=keep,
        command_timeout=timeout,
    )

    if not prober.is_available():
        rprint("[bold red]flatpak-builder not found.[/bold red]")
        rprint("Install with: [bold]flatpak install flathub org.flatpak.Builder[/bold]")
        raise typer.Exit(1)

    rprint(f"\\n[bold]Probing {len(result.packages)} packages[/bold] in {sdk}//{runtime_version}")
    if extensions:
        rprint(f"Extensions: {extensions}")

    with console.status("[bold green]Running sandbox probe..."):
        report = prober.probe(result.packages)

    # --- Summary ---
    _print_probe_report(report)

    if not report.build_possible:
        raise typer.Exit(1)


def _print_probe_report(report) -> None:
    """Print a formatted SandboxProbeReport."""
    if not report.ran:
        rprint(f"[yellow]Probe did not run: {report.skip_reason}[/yellow]")
        return

    rprint(f"\\n[bold]Probe results[/bold] (probed {len(report.probed_packages)} packages)")

    if not report.errors:
        rprint("[bold green]All packages installed and imported successfully.[/bold green]")
    else:
        table = Table(title=f"Probe errors ({len(report.errors)} total)")
        table.add_column("Type", style="red")
        table.add_column("Missing")
        table.add_column("Source", style="dim")
        table.add_column("Context", style="dim")
        for err in report.errors:
            table.add_row(
                err.error_type.value,
                err.missing,
                err.source,
                err.context,
            )
        console.print(table)

    # Verdicts
    rprint("")
    verdict_rows = [
        ("SDK sufficient",        report.sdk_sufficient,    True),
        ("Build possible",        report.build_possible,    True),
        ("Missing native libs",   not report.missing_native_libs,   False),
        ("Missing headers",       not report.missing_headers,       False),
        ("Missing pkg-config",    not report.missing_pkgconfig,     False),
        ("Missing Python pkgs",   not report.missing_python_packages, False),
    ]
    for label, ok, positive in verdict_rows:
        icon = "[green]v[/green]" if ok == positive or (positive and ok) else "[red]x[/red]"
        if not ok and label == "Missing native libs":
            detail = f": {report.missing_native_libs}"
        elif not ok and label == "Missing headers":
            detail = f": {report.missing_headers}"
        elif not ok and label == "Missing pkg-config":
            detail = f": {report.missing_pkgconfig}"
        elif not ok and label == "Missing Python pkgs":
            detail = f": {report.missing_python_packages}"
        else:
            detail = ""
        rprint(f"  {icon}  {label}{detail}")

    if report.suggested_extensions:
        rprint(f"\\n[bold]Suggested extensions:[/bold]")
        for ext in report.suggested_extensions:
            rprint(f"  - {ext}")

# ---------------------------------------------------------------------------
# pfmr version
# ---------------------------------------------------------------------------

@app.command("version")
def version():
    """Print pfmr version."""
    rprint(f"pfmr [bold]{__version__}[/bold]")


def main():
    app()


if __name__ == "__main__":
    main()