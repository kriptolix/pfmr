"""
pfmr.sandbox.runner
~~~~~~~~~~~~~~~~~~~~
Executes arbitrary shell commands inside the Flatpak build sandbox using
`flatpak-builder --run`.

This is the chosen execution strategy (spec §18.3 option 2) because:
  - No installation required — the sandbox is ephemeral
  - The SDK and all declared extensions are mounted automatically
  - Incremental builds are possible via a persistent build-dir
  - Idempotent — can be re-run any number of times

Invocation pattern:
  flatpak-builder --run <build-dir> <manifest.json> sh -c "<command>"

The build-dir is created once per probe session and reused across
individual command executions to benefit from flatpak-builder's
incremental caching.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pfmr.utils.logging import get_logger

logger = get_logger(__name__)

# Timeout for individual sandbox commands (seconds)
_DEFAULT_TIMEOUT = 120


@dataclass
class RunResult:
    """Result of a single command execution inside the sandbox."""
    command: str
    stdout: str
    stderr: str
    exit_code: int

    @property
    def succeeded(self) -> bool:
        return self.exit_code == 0

    @property
    def combined(self) -> str:
        return self.stdout + "\n" + self.stderr


class SandboxRunner:
    """
    Manages a flatpak-builder sandbox session.

    A session corresponds to a single build-dir on disk.
    The first command triggers the initial `flatpak-builder` build
    (which installs the infoscript module); subsequent commands reuse
    the same build-dir.

    Usage::

        runner = SandboxRunner(manifest_path, work_dir)
        if not runner.is_available():
            # flatpak-builder not installed — skip probe
            ...
        runner.build()          # build the base sandbox once
        result = runner.run("python3 --version")
        result = runner.run("pkg-config --list-all")
    """

    def __init__(
        self,
        manifest_path: Path,
        work_dir: Path,
        build_dir: Optional[Path] = None,
        timeout: int = _DEFAULT_TIMEOUT,
        # Extra env vars to set inside the sandbox commands
        extra_env: Optional[dict[str, str]] = None,
    ):
        self.manifest_path = manifest_path
        self.work_dir = work_dir
        self.build_dir = build_dir or (work_dir / "build")
        self.timeout = timeout
        self.extra_env = extra_env or {}
        self._flatpak_builder = shutil.which("flatpak-builder")
        self._built = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if flatpak-builder is present on the system."""
        return self._flatpak_builder is not None

    def build(self, force: bool = False) -> RunResult:
        """
        Run `flatpak-builder <build-dir> <manifest>` to prepare the sandbox.
        Uses --disable-rofiles-fuse to avoid needing user namespaces in CI.
        Subsequent calls are no-ops unless force=True.
        """
        if self._built and not force:
            logger.debug("Sandbox already built; skipping rebuild")
            return RunResult(command="(cached)", stdout="", stderr="", exit_code=0)

        self.build_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self._flatpak_builder,
            "--disable-rofiles-fuse",
            "--force-clean",
            str(self.build_dir),
            str(self.manifest_path),
        ]
        logger.info("Building test sandbox: %s", " ".join(cmd))
        result = self._exec(cmd, cwd=self.work_dir)

        if result.succeeded:
            self._built = True
            logger.info("Test sandbox built successfully")
        else:
            logger.warning(
                "Sandbox build failed (exit %d):\n%s",
                result.exit_code, result.stderr[-2000:],
            )
        return result

    def run(self, shell_command: str, timeout: Optional[int] = None) -> RunResult:
        """
        Execute a shell command inside the sandbox via:
          flatpak-builder --run <build-dir> <manifest> sh -c "<command>"

        The sandbox must have been built first (call build()).
        """
        if not self._flatpak_builder:
            return RunResult(
                command=shell_command,
                stdout="",
                stderr="flatpak-builder not found",
                exit_code=127,
            )

        cmd = [
            self._flatpak_builder,
            "--disable-rofiles-fuse",
            "--run",
            str(self.build_dir),
            str(self.manifest_path),
            "sh", "-c", shell_command,
        ]
        logger.debug("Sandbox run: %s", shell_command[:120])
        return self._exec(cmd, cwd=self.work_dir, timeout=timeout)

    def run_python(self, python_command: str, timeout: Optional[int] = None) -> RunResult:
        """Convenience: run a Python one-liner inside the sandbox venv."""
        return self.run(
            f"/app/venv/bin/python -c {_sh_quote(python_command)}",
            timeout=timeout,
        )

    def run_pip(self, pip_args: str, timeout: Optional[int] = None) -> RunResult:
        """Convenience: run pip/uv inside the sandbox venv."""
        return self.run(
            f"/app/venv/bin/pip {pip_args}",
            timeout=timeout,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _exec(
        self,
        cmd: list[str],
        cwd: Optional[Path] = None,
        timeout: Optional[int] = None,
    ) -> RunResult:
        env = dict(os.environ)
        env.update(self.extra_env)
        # Prevent flatpak-builder from picking up the host DISPLAY which can
        # cause X11 connection errors during headless CI runs.
        env.pop("DISPLAY", None)

        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout or self.timeout,
            )
            return RunResult(
                command=" ".join(str(c) for c in cmd),
                stdout=proc.stdout,
                stderr=proc.stderr,
                exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            logger.warning("Sandbox command timed out after %ds: %s", timeout or self.timeout, cmd)
            return RunResult(
                command=" ".join(str(c) for c in cmd),
                stdout="",
                stderr=f"TIMEOUT after {timeout or self.timeout}s",
                exit_code=-1,
            )
        except FileNotFoundError:
            return RunResult(
                command=" ".join(str(c) for c in cmd),
                stdout="",
                stderr=f"flatpak-builder not found: {cmd[0]}",
                exit_code=127,
            )


def _sh_quote(s: str) -> str:
    """Minimal shell quoting — wrap in single quotes, escape existing single quotes."""
    return "'" + s.replace("'", "'\\''") + "'"