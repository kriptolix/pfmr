"""
pfmr.learn.sdk_probe
~~~~~~~~~~~~~~~~~~~~~
SDKProber — downloads a Flatpak SDK or extension locally, inspects its
contents (pkg-config, shared libraries, executables), generates a static
profile TOML, and then removes the downloaded SDK to free disk space.

No CI required — runs on any machine with flatpak installed.

Workflow:
  1. `flatpak install <sdk-id>//<version>` (if not already installed)
  2. Enter a shell via `flatpak run --command=sh --devel <sdk-id>`
     or via `flatpak-builder --run` with a minimal manifest
  3. Execute introspection commands:
       pkg-config --list-all
       find /usr/lib -name '*.so*' -type f
       ls /usr/bin /usr/lib/sdk/*/bin
  4. Parse output → SDKCapability
  5. Write to data/sdk-profiles/<sdk-id>/<version>.toml
  6. Optionally uninstall the SDK to reclaim disk space

Usage (no CI)::

    pfmr learn sdk probe --sdk org.freedesktop.Sdk --sdk-version 24.08
    pfmr learn sdk probe --sdk org.gnome.Sdk --sdk-version 48 --cleanup
    pfmr learn sdk list-available
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pfmr.utils.logging import get_logger

logger = get_logger(__name__)

_BUILTIN_PROFILES_DIR = Path(__file__).parent.parent / "data" / "sdk-profiles"
_EXT_PROFILES_DIR = Path(__file__).parent.parent / "data" / "extension-profiles"

# SDK/extension pairs to probe in bulk (used by `pfmr learn sdk probe-all`)
DEFAULT_SDK_LIST = [
    ("org.freedesktop.Sdk", "25.08"),
    ("org.freedesktop.Sdk", "24.08"),
    ("org.freedesktop.Sdk", "23.08"),
    ("org.gnome.Sdk",       "50"),
    ("org.gnome.Sdk",       "49"),
    ("org.kde.Sdk",         "6.10"),
    ("org.kde.Sdk",         "6.9"),
]

DEFAULT_EXTENSION_LIST = [
    ("org.freedesktop.Sdk.Extension.bazel", "24.08"),
    ("org.freedesktop.Sdk.Extension.dmd", "22.08"),
    ("org.freedesktop.Sdk.Extension.dotnet10", "25.08"),
    ("org.freedesktop.Sdk.Extension.dotnet9", "25.08"),
    ("org.freedesktop.Sdk.Extension.freepascal", "25.08"),
    ("org.freedesktop.Sdk.Extension.gcc7", "1.6"),
    ("org.freedesktop.Sdk.Extension.gcc8", "1.6"),
    ("org.freedesktop.Sdk.Extension.gfortran-62", "1.6"),
    ("org.freedesktop.Sdk.Extension.gnat14", "24.08"),
    ("org.freedesktop.Sdk.Extension.gnat15", "24.08"),
    ("org.freedesktop.Sdk.Extension.golang", "25.08"),
    ("org.freedesktop.Sdk.Extension.haskell", "21.08"),
    ("org.freedesktop.Sdk.Extension.ldc", "25.08"),
    ("org.freedesktop.Sdk.Extension.llvm21", "25.08"),
    ("org.freedesktop.Sdk.Extension.llvm22", "25.08"),
    ("org.freedesktop.Sdk.Extension.mingw-w64", "25.08"),
    ("org.freedesktop.Sdk.Extension.mono5", "19.08"),
    ("org.freedesktop.Sdk.Extension.mono6", "25.08"),
    ("org.freedesktop.Sdk.Extension.node22", "25.08"),
    ("org.freedesktop.Sdk.Extension.node24", "25.08"),
    ("org.freedesktop.Sdk.Extension.nvidia-base", "24.08"),
    ("org.freedesktop.Sdk.Extension.ocaml", "25.08"),
    ("org.freedesktop.Sdk.Extension.openjdk", "25.08"),
    ("org.freedesktop.Sdk.Extension.openjdk25", "25.08"),
    ("org.freedesktop.Sdk.Extension.php83", "24.08"),
    ("org.freedesktop.Sdk.Extension.php84", "25.08"),
    ("org.freedesktop.Sdk.Extension.rust-nightly", "25.08"),
    ("org.freedesktop.Sdk.Extension.rust-stable", "25.08"),
    ("org.freedesktop.Sdk.Extension.swift5", "23.08"),
    ("org.freedesktop.Sdk.Extension.swift6", "25.08"),
    ("org.freedesktop.Sdk.Extension.texlive", "25.08"),
    ("org.freedesktop.Sdk.Extension.toolchain-i386", "25.08"),
    ("org.freedesktop.Sdk.Extension.typescript", "25.08"),
    ("org.freedesktop.Sdk.Extension.vala", "25.08"),
    ("org.freedesktop.Sdk.Extension.ziglang", "25.08"),
    ("org.freedesktop.Sdk.PreBootstrap", "18.08"),
]


# ---------------------------------------------------------------------------
# Introspection script run inside the SDK
# ---------------------------------------------------------------------------

_INTROSPECT_SH = r"""
echo '=== PKGCONFIG ==='
pkg-config --list-all 2>/dev/null | awk '{print $1}' | sort -u
echo '=== LIBRARIES ==='
find /usr/lib /usr/lib64 /lib /lib64 -name '*.so*' -type f 2>/dev/null \
  | sed 's|.*/||' | sort -u
echo '=== EXECUTABLES ==='
ls /usr/bin /usr/local/bin 2>/dev/null | sort -u
echo '=== DONE ==='
"""

_EXT_INTROSPECT_SH_TEMPLATE = r"""
EXT_PATH={mount}
echo '=== EXT_EXECUTABLES ==='
ls "$EXT_PATH/bin" 2>/dev/null | sort -u
echo '=== EXT_PKGCONFIG ==='
pkg-config --with-path="$EXT_PATH/lib/pkgconfig" --list-all 2>/dev/null \
  | awk '{{print $1}}' | sort -u
echo '=== EXT_LIBRARIES ==='
find "$EXT_PATH/lib" "$EXT_PATH/lib64" -name '*.so*' -type f 2>/dev/null \
  | sed 's|.*/||' | sort -u
echo '=== DONE ==='
"""


# ---------------------------------------------------------------------------
# Probe result
# ---------------------------------------------------------------------------

@dataclass
class ProbeResult:
    sdk_id: str
    sdk_version: str
    pkgconfig: list[str] = field(default_factory=list)
    libraries: list[str] = field(default_factory=list)
    executables: list[str] = field(default_factory=list)
    success: bool = False
    error: str = ""


# ---------------------------------------------------------------------------
# SDKProber
# ---------------------------------------------------------------------------

class SDKProber:
    """
    Downloads a Flatpak SDK, introspects it, and writes a static profile.

    Standalone — no pfmr.pipeline dependency.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,       # default: built-in sdk-profiles dir
        ext_output_dir: Optional[Path] = None,    # default: built-in extension-profiles dir
        auto_install: bool = True,                # flatpak install if not present
        cleanup_after: bool = False,              # uninstall after probing
    ):
        self.output_dir = output_dir or _BUILTIN_PROFILES_DIR
        self.ext_output_dir = ext_output_dir or _EXT_PROFILES_DIR
        self.auto_install = auto_install
        self.cleanup_after = cleanup_after
        self._flatpak = shutil.which("flatpak")
        self._flatpak_builder = shutil.which("flatpak-builder")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return bool(self._flatpak)

    def probe_sdk(self, sdk_id: str, sdk_version: str) -> ProbeResult:
        """
        Probe a single SDK and write its profile.
        Returns a ProbeResult (success=True if the profile was written).
        """
        result = ProbeResult(sdk_id=sdk_id, sdk_version=sdk_version)

        if not self._flatpak:
            result.error = "flatpak not found"
            return result

        # Install if needed
        if self.auto_install and not self._is_installed(sdk_id, sdk_version):
            logger.info("Installing %s//%s ...", sdk_id, sdk_version)
            ok = self._install(sdk_id, sdk_version)
            if not ok:
                result.error = f"flatpak install failed for {sdk_id}//{sdk_version}"
                return result

        # Introspect
        output = self._run_in_sdk(sdk_id, sdk_version, _INTROSPECT_SH)
        if output is None:
            result.error = f"introspection failed for {sdk_id}//{sdk_version}"
            return result

        result = self._parse_sdk_output(output, sdk_id, sdk_version)
        self._write_sdk_profile(result)

        if self.cleanup_after:
            self._uninstall(sdk_id, sdk_version)

        return result

    def probe_extension(
        self,
        ext_id: str,
        sdk_version: str,
        base_sdk: Optional[str] = None,
    ) -> ProbeResult:
        """
        Probe a SDK extension and update its extension profile with what
        executables, pkg-config names, and libraries it actually provides.
        """
        result = ProbeResult(sdk_id=ext_id, sdk_version=sdk_version)

        if not self._flatpak:
            result.error = "flatpak not found"
            return result

        # Extensions are usually installed alongside their SDK
        sdk = base_sdk or ext_id.rsplit(".Extension.", 1)[0]

        if self.auto_install and not self._is_installed(ext_id, sdk_version):
            logger.info("Installing extension %s//%s ...", ext_id, sdk_version)
            self._install(ext_id, sdk_version)

        # Derive mount path from extension id
        short_name = ext_id.split(".")[-1]
        mount = f"/usr/lib/sdk/{short_name}"
        script = _EXT_INTROSPECT_SH_TEMPLATE.format(mount=mount)

        output = self._run_in_sdk(sdk, sdk_version, script, extra_extensions=[ext_id])
        if output is None:
            result.error = f"extension introspection failed"
            return result

        result = self._parse_extension_output(output, ext_id, sdk_version, mount)
        self._update_extension_profile(result, ext_id, mount)

        if self.cleanup_after:
            self._uninstall(ext_id, sdk_version)

        return result

    def probe_all(
        self,
        sdk_list: Optional[list[tuple[str, str]]] = None,
        ext_list: Optional[list[tuple[str, str]]] = None,
    ) -> list[ProbeResult]:
        """Probe all SDKs and extensions in the default lists."""
        results: list[ProbeResult] = []
        for sdk_id, version in (sdk_list or DEFAULT_SDK_LIST):
            logger.info("Probing SDK: %s//%s", sdk_id, version)
            results.append(self.probe_sdk(sdk_id, version))

        for ext_id, version in (ext_list or DEFAULT_EXTENSION_LIST):
            logger.info("Probing extension: %s//%s", ext_id, version)
            results.append(self.probe_extension(ext_id, version))

        return results

    # ------------------------------------------------------------------
    # Flatpak helpers
    # ------------------------------------------------------------------

    def _is_installed(self, ref_id: str, version: str) -> bool:
        result = subprocess.run(
            [self._flatpak, "info", f"{ref_id}//{version}"],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0

    def _install(self, ref_id: str, version: str) -> bool:
        result = subprocess.run(
            [self._flatpak, "install", "--noninteractive", "--assumeyes",
             "flathub", f"{ref_id}//{version}"],
            capture_output=True, timeout=600,
        )
        if result.returncode != 0:
            logger.warning("Install failed: %s", result.stderr.decode()[-500:])
        return result.returncode == 0

    def _uninstall(self, ref_id: str, version: str) -> None:
        subprocess.run(
            [self._flatpak, "uninstall", "--noninteractive", f"{ref_id}//{version}"],
            capture_output=True, timeout=60,
        )
        logger.info("Uninstalled %s//%s", ref_id, version)

    def _run_in_sdk(
        self,
        sdk_id: str,
        sdk_version: str,
        script: str,
        extra_extensions: Optional[list[str]] = None,
    ) -> Optional[str]:
        """
        Run a shell script inside the SDK using one of two strategies:
          1. flatpak-builder --run (preferred, no install needed for runtime)
          2. flatpak run --command=sh --devel (needs SDK installed)
        """
        if self._flatpak_builder:
            return self._run_via_builder(sdk_id, sdk_version, script, extra_extensions)
        return self._run_via_flatpak(sdk_id, sdk_version, script)

    def _run_via_builder(
        self,
        sdk_id: str,
        sdk_version: str,
        script: str,
        extra_extensions: Optional[list[str]] = None,
    ) -> Optional[str]:
        print("via builder")
        with tempfile.TemporaryDirectory(prefix="pfmr-sdk-probe-") as tmp:
            tmp_path = Path(tmp)
            build_dir = tmp_path / "build"
            build_dir.mkdir()

            # Derive platform from SDK id
            platform = sdk_id.replace("Sdk", "Platform")
            manifest: dict = {
                "app-id": "org.pfmr.SdkProbe",
                "runtime": platform,
                "runtime-version": sdk_version,
                "sdk": sdk_id,
                "modules": [],
            }
            if extra_extensions:
                manifest["sdk-extensions"] = extra_extensions

            manifest_path = tmp_path / "probe.json"
            manifest_path.write_text(json.dumps(manifest))

            cmd = [
                self._flatpak_builder,
                "--disable-rofiles-fuse",
                "--run",
                str(build_dir),
                str(manifest_path),
                "sh", "-c", script,
            ]
            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120
                )                
                if result.returncode == 0:                    
                    return result.stdout
                logger.warning(
                    "flatpak-builder --run failed: %s", result.stderr[-500:]
                )
            except subprocess.TimeoutExpired:
                logger.warning("SDK probe timed out")
        return None

    def _run_via_flatpak(
        self,
        sdk_id: str,
        sdk_version: str,
        script: str,
    ) -> Optional[str]:
        print("via flatpak")
        cmd = [
            self._flatpak, "run",
            "--command=sh",
            "--devel",
            "--share=none",
            "--socket=none",
            "--device=none",
            "--nofilesystem=host",
            f"{sdk_id}//{sdk_version}",
            "-c", script,
        ]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60
            )
            if result.returncode == 0:
                return result.stdout
        except subprocess.TimeoutExpired:
            pass
        return None

    # ------------------------------------------------------------------
    # Output parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_sdk_output(output: str, sdk_id: str, sdk_version: str) -> ProbeResult:
        result = ProbeResult(sdk_id=sdk_id, sdk_version=sdk_version, success=True)
        section = None
        for line in output.splitlines():
            line = line.strip()
            if line == "=== PKGCONFIG ===":
                section = "pc"
            elif line == "=== LIBRARIES ===":
                section = "libs"
            elif line == "=== EXECUTABLES ===":
                section = "exes"
            elif line == "=== DONE ===":
                break
            elif line and section == "pc":
                result.pkgconfig.append(line)
            elif line and section == "libs":
                result.libraries.append(line)
            elif line and section == "exes":
                result.executables.append(line)
        return result

    @staticmethod
    def _parse_extension_output(
        output: str, ext_id: str, sdk_version: str, mount: str
    ) -> ProbeResult:
        result = ProbeResult(sdk_id=ext_id, sdk_version=sdk_version, success=True)
        section = None
        for line in output.splitlines():
            line = line.strip()
            if line == "=== EXT_EXECUTABLES ===":
                section = "exes"
            elif line == "=== EXT_PKGCONFIG ===":
                section = "pc"
            elif line == "=== EXT_LIBRARIES ===":
                section = "libs"
            elif line == "=== DONE ===":
                break
            elif line and section == "exes":
                result.executables.append(line)
            elif line and section == "pc":
                result.pkgconfig.append(line)
            elif line and section == "libs":
                result.libraries.append(line)
        return result

    # ------------------------------------------------------------------
    # Profile writers
    # ------------------------------------------------------------------

    def _write_sdk_profile(self, result: ProbeResult) -> Path:
        safe_id = result.sdk_id.replace("/", "_")
        profile_dir = self.output_dir / safe_id
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_path = profile_dir / f"{result.sdk_version}.toml"

        lines = [
            f'sdk_id = "{result.sdk_id}"',
            f'sdk_version = "{result.sdk_version}"',
            "",
            "pkgconfig = [",
            *[f'  "{pc}",' for pc in sorted(result.pkgconfig)],
            "]",
            "",
            "libraries = [",
            *[f'  "{lib}",' for lib in sorted(result.libraries)],
            "]",
            "",
            "headers = []",
            "",
            "executables = [",
            *[f'  "{exe}",' for exe in sorted(result.executables)],
            "]",
            "",
            "python_modules = []",
        ]
        profile_path.write_text("\n".join(lines) + "\n")
        logger.info("Wrote SDK profile: %s", profile_path)
        return profile_path

    def _update_extension_profile(
        self,
        result: ProbeResult,
        ext_id: str,
        mount: str,
    ) -> None:
        """
        Find the matching extension profile TOML and update its
        provides_* fields with live-probed data.
        """
        safe_name = ext_id.split(".")[-1]
        candidates = list(self.ext_output_dir.glob(f"*{safe_name}*.toml"))
        if not candidates:
            logger.debug("No extension profile found for %s; skipping update", ext_id)
            return

        profile_path = candidates[0]
        content = profile_path.read_text()

        def _replace_list(key: str, values: list[str]) -> str:
            pattern = re.compile(rf"^{key}\s*=\s*\[.*?\]", re.MULTILINE | re.DOTALL)
            new_val = key + " = [\n" + "".join(f'  "{v}",\n' for v in sorted(values)) + "]"
            if pattern.search(content):
                return pattern.sub(new_val, content)
            return content + f"\n{new_val}\n"

        if result.executables:
            content = _replace_list("provides_executables", result.executables)
        if result.pkgconfig:
            content = _replace_list("provides_pkgconfig", result.pkgconfig)
        if result.libraries:
            content = _replace_list("provides_libraries", result.libraries)

        profile_path.write_text(content)
        logger.info("Updated extension profile: %s", profile_path)