"""
pfmr.sandbox.probe
~~~~~~~~~~~~~~~~~~~
BuildSandboxProber — Phase 3 core component.

Orchestrates the full sandbox probe sequence for a set of Python packages:

  1. Write test manifest + infoscript.sh to a temp work directory
  2. Initialise the build directory (flatpak build-init)
  3. Set up a Python venv inside the sandbox
  4. For each package:
       a. Attempt `uv pip install <pkg>` (or pip)
       b. If install fails → parse errors, record missing deps
       c. If install succeeds → attempt `python -c "import <pkg>"`
       d. If import fails → record ImportError
       e. Run ldd on any .so files found in site-packages → record missing libs
       f. Run pkg-config checks for declared native_deps
  5. Collate all errors into a SandboxProbeReport with high-level verdicts

The prober skips gracefully when:
  - flatpak is not installed (ran=False, skip_reason set)
  - The sandbox build itself fails
  - A package is pure Python and its wheel installs without issues

Cache strategy (spec §18.4):
  - The build-dir IS cached between probe calls for the same work_dir
    (flatpak build-init creates a persistent build-dir on disk)
  - The Python venv is set up once per session and reused
  - Failed probe states are NOT cached: if errors were found, the
    work-dir is left intact for debugging but not reused as a "clean" base
"""
from __future__ import annotations

import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pfmr.models import (
    ResolvedPackage,
    ResolutionResult,
    SandboxError,
    SandboxErrorType,
    SandboxProbeReport,
)
from pfmr.data.mappings import MAPPINGS
from pfmr.sandbox.errors import parse_errors
from pfmr.sandbox.runner import SandboxRunner
from pfmr.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Python venv setup commands (run once after base build)
# ---------------------------------------------------------------------------

_VENV_SETUP_CMDS = [
    "python3 -m venv /app/venv",
    "/app/venv/bin/pip install --quiet --upgrade pip",
    "/app/venv/bin/pip install --quiet uv",
]

_VENV_SETUP_SH = " && ".join(_VENV_SETUP_CMDS)

# ldd output line parser
_LDD_LINE = re.compile(r"^\s*(?P<lib>\S+\.so\S*)\s*=>\s*(?P<path>\S+)", re.MULTILINE)


# ---------------------------------------------------------------------------
# Main prober
# ---------------------------------------------------------------------------

class BuildSandboxProber:
    """
    Probes a set of Python packages inside a real Flatpak build environment.

    Usage::

        prober = BuildSandboxProber(
            runtime="org.freedesktop.Platform",
            runtime_version="24.08",
            sdk="org.freedesktop.Sdk",
        )
        report = prober.probe(packages)
        if report.ran:
            for err in report.errors:
                print(err)
    """

    def __init__(
        self,
        runtime: str = "org.freedesktop.Platform",
        runtime_version: str = "24.08",
        sdk: str = "org.freedesktop.Sdk",
        sdk_extensions: Optional[list[str]] = None,
        # Working directory for the sandbox (auto-created if None)
        work_dir: Optional[Path] = None,
        # Keep work_dir after probe (useful for debugging)
        keep_work_dir: bool = False,
        # Timeout per individual sandbox command
        command_timeout: int = 120,
        # Timeout for the initial sandbox build step
        build_timeout: int = 600,
        # Use uv instead of pip for installation tests
        use_uv: bool = True,
    ):
        self.runtime = runtime
        self.runtime_version = runtime_version
        self.sdk = sdk
        self.sdk_extensions = sdk_extensions or []
        self._work_dir = work_dir
        self._keep_work_dir = keep_work_dir
        self._command_timeout = command_timeout
        self._build_timeout = build_timeout
        self._use_uv = use_uv
        self._owned_work_dir: Optional[Path] = None   # temp dir we created

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if flatpak is available on the host."""
        return shutil.which("flatpak") is not None

    def probe(
        self,
        packages: list[ResolvedPackage],
        work_dir: Optional[Path] = None,
    ) -> SandboxProbeReport:
        """
        Run the full probe sequence for the given packages.
        Returns a SandboxProbeReport regardless of outcome.
        """
        effective_work_dir = work_dir or self._work_dir or self._make_work_dir()
        try:
            return self._probe(packages, effective_work_dir)
        finally:
            if self._owned_work_dir and not self._keep_work_dir:
                shutil.rmtree(self._owned_work_dir, ignore_errors=True)
                self._owned_work_dir = None

    def probe_result(
        self,
        result: ResolutionResult,
        work_dir: Optional[Path] = None,
    ) -> SandboxProbeReport:
        """Convenience: probe all packages in a ResolutionResult."""
        return self.probe(result.packages, work_dir=work_dir)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _make_work_dir(self) -> Path:
        d = Path(tempfile.mkdtemp(prefix="pfmr-probe-"))
        self._owned_work_dir = d
        return d

    def _probe(
        self,
        packages: list[ResolvedPackage],
        work_dir: Path,
    ) -> SandboxProbeReport:

        report = SandboxProbeReport(
            probed_packages=[p.name for p in packages],
        )

        # --- preflight ---
        if not self.is_available():
            report.ran = False
            report.skip_reason = (
                "flatpak not found. "
                "Install it with your distribution package manager (e.g. apt install flatpak)"
            )
            logger.warning("Probe skipped: %s", report.skip_reason)
            return report

        if not packages:
            report.ran = True
            report.build_possible = True
            return report

        import tempfile
        build_dir = work_dir / "build"
        runner = SandboxRunner(
            build_dir=build_dir,
            sdk=self.sdk,
            runtime=self.runtime,
            runtime_version=self.runtime_version,
            sdk_extensions=self.sdk_extensions,
            timeout=self._command_timeout,
        )

        # --- initialise sandbox ---
        init_result = runner.init()
        report.stdout += init_result.stdout
        report.stderr += init_result.stderr

        if not init_result.succeeded:
            report.ran = True
            report.exit_code = init_result.exit_code
            report.build_possible = False
            errors = parse_errors(
                init_result.stderr, init_result.stdout, context="sandbox-init"
            )
            report.errors.extend(errors)
            _apply_errors_to_report(report, errors)
            logger.error("Sandbox init failed — probe aborted")
            return report

        # --- set up Python venv ---
        venv_result = runner.run(_VENV_SETUP_SH, timeout=180)
        report.stdout += venv_result.stdout
        report.stderr += venv_result.stderr
        if not venv_result.succeeded:
            logger.warning("venv setup failed:\n%s", venv_result.stderr[-500:])

        # --- probe each package ---
        for pkg in packages:
            self._probe_package(pkg, runner, report)

        # --- high-level verdicts ---
        report.ran = True
        report.exit_code = 0 if not report.errors else 1
        report.build_possible = not any(
            e.error_type in (
                SandboxErrorType.MISSING_NATIVE_DEP,
                SandboxErrorType.MISSING_HEADER,
                SandboxErrorType.MISSING_PKGCONFIG,
                SandboxErrorType.BUILD_FAILURE,
            )
            for e in report.errors
        )
        report.sdk_sufficient = len(report.missing_native_libs) == 0 and \
                                 len(report.missing_headers) == 0 and \
                                 len(report.missing_pkgconfig) == 0

        logger.info(
            "Probe complete: %d errors, sdk_sufficient=%s, build_possible=%s",
            len(report.errors), report.sdk_sufficient, report.build_possible,
        )
        return report

    def _probe_package(
        self,
        pkg: ResolvedPackage,
        runner: SandboxRunner,
        report: SandboxProbeReport,
    ) -> None:
        logger.info("Probing package: %s==%s", pkg.name, pkg.version)

        # --- 1. Installation attempt ---
        install_result = self._try_install(pkg, runner)
        report.stdout += install_result.stdout
        report.stderr += install_result.stderr

        if not install_result.succeeded:
            errors = parse_errors(
                install_result.stderr,
                install_result.stdout,
                context=f"{pkg.name} install",
            )
            if not errors:
                # Generic build failure
                errors = [SandboxError(
                    error_type=SandboxErrorType.BUILD_FAILURE,
                    missing=pkg.name,
                    source="stderr",
                    context=f"{pkg.name} install",
                    raw_line=install_result.stderr[-400:].strip(),
                )]
            report.errors.extend(errors)
            _apply_errors_to_report(report, errors)
            logger.warning(
                "Install failed for %s (exit %d)", pkg.name, install_result.exit_code
            )
            return

        # --- 2. Import test ---
        import_result = self._try_import(pkg, runner)
        report.stdout += import_result.stdout
        report.stderr += import_result.stderr

        if not import_result.succeeded:
            errors = parse_errors(
                import_result.stderr,
                import_result.stdout,
                context=f"{pkg.name} import",
            )
            if not errors:
                errors = [SandboxError(
                    error_type=SandboxErrorType.IMPORT_ERROR,
                    missing=pkg.name,
                    source="import",
                    context=f"{pkg.name} import",
                    raw_line=import_result.stderr[-200:].strip(),
                )]
            report.errors.extend(errors)
            _apply_errors_to_report(report, errors)

        # --- 3. ldd check on installed .so files ---
        ldd_result = self._run_ldd(pkg, runner)
        if ldd_result:
            report.stdout += ldd_result.stdout
            ldd_errors = parse_errors(
                ldd_result.stderr,
                ldd_output=ldd_result.stdout,
                context=f"{pkg.name} ldd",
            )
            report.errors.extend(ldd_errors)
            _apply_errors_to_report(report, ldd_errors)

        # --- 4. pkg-config checks for declared native deps ---
        for dep in pkg.native_deps:
            if not dep.endswith(".so") and not dep.endswith(".h"):
                pc_result = runner.run(f"pkg-config --exists {dep} && echo OK || echo MISSING")
                if "MISSING" in pc_result.stdout or pc_result.exit_code != 0:
                    err = SandboxError(
                        error_type=SandboxErrorType.MISSING_PKGCONFIG,
                        missing=dep,
                        source="pkg-config",
                        context=f"{pkg.name} dep-check",
                    )
                    if not any(
                        e.missing == dep and e.error_type == SandboxErrorType.MISSING_PKGCONFIG
                        for e in report.errors
                    ):
                        report.errors.append(err)
                        if dep not in report.missing_pkgconfig:
                            report.missing_pkgconfig.append(dep)

    def _try_install(self, pkg: ResolvedPackage, runner: SandboxRunner):
        spec = f"{pkg.name}=={pkg.version}"
        if self._use_uv:
            cmd = f"/app/venv/bin/uv pip install --no-cache {spec}"
        else:
            cmd = f"/app/venv/bin/pip install --no-cache-dir {spec}"
        return runner.run(cmd, timeout=self._command_timeout)

    def _try_import(self, pkg: ResolvedPackage, runner: SandboxRunner):
        # Use top-level import name (may differ from package name)
        import_name = _pkg_to_import_name(pkg.name)
        cmd = f"/app/venv/bin/python -c 'import {import_name}; print(\"OK\")'"
        return runner.run(cmd, timeout=30)

    def _run_ldd(self, pkg: ResolvedPackage, runner: SandboxRunner):
        """Find .so files installed by pkg and run ldd on them."""
        find_cmd = (
            f"find /app/venv/lib -name '*.so' -path '*{pkg.name.replace('-','_').lower()}*' "
            f"2>/dev/null | head -5"
        )
        find_result = runner.run(find_cmd, timeout=10)
        so_files = [f.strip() for f in find_result.stdout.splitlines() if f.strip()]
        if not so_files:
            return None
        ldd_cmd = "ldd " + " ".join(so_files) + " 2>&1"
        return runner.run(ldd_cmd, timeout=15)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_errors_to_report(report: SandboxProbeReport, errors: list[SandboxError]) -> None:
    """Distribute parsed errors into the typed lists on the report."""
    for err in errors:
        if err.error_type == SandboxErrorType.MISSING_NATIVE_DEP:
            if err.missing not in report.missing_native_libs:
                report.missing_native_libs.append(err.missing)
                report.sdk_sufficient = False
        elif err.error_type == SandboxErrorType.MISSING_HEADER:
            if err.missing not in report.missing_headers:
                report.missing_headers.append(err.missing)
                report.sdk_sufficient = False
        elif err.error_type == SandboxErrorType.MISSING_PKGCONFIG:
            if err.missing not in report.missing_pkgconfig:
                report.missing_pkgconfig.append(err.missing)
        elif err.error_type in (
            SandboxErrorType.MISSING_PYTHON_PKG, SandboxErrorType.IMPORT_ERROR
        ):
            if err.missing not in report.missing_python_packages:
                report.missing_python_packages.append(err.missing)


def _pkg_to_import_name(pkg_name: str) -> str:
    """Map PyPI package name to Python import name via mappings.toml."""
    return MAPPINGS.python_import_name(pkg_name)