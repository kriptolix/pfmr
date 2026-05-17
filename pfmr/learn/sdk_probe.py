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
     via `flatpak build-init` + `flatpak build`
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from pfmr.utils.logging import get_logger

logger = get_logger(__name__)

_BUILTIN_PROFILES_DIR = Path(__file__).parent.parent / "data" / "sdk-profiles"
_EXT_PROFILES_DIR = Path(__file__).parent.parent / "data" / "extension-profiles"

# SDK/extension pairs to probe in bulk (used by `pfmr learn sdk probe-all`)
DEFAULT_SDK_LIST = [
    ("org.freedesktop.Sdk", "24.08"),
    ("org.freedesktop.Sdk", "23.08"),
    ("org.gnome.Sdk",       "48"),
    ("org.gnome.Sdk",       "47"),
    ("org.kde.Sdk",         "6.8"),
]

DEFAULT_EXTENSION_LIST = [
    ("org.freedesktop.Sdk.Extension.rust-stable", "24.08"),
    ("org.freedesktop.Sdk.Extension.llvm18",      "24.08"),
    ("org.freedesktop.Sdk.Extension.openjdk21",   "24.08"),
    ("org.freedesktop.Sdk.Extension.node20",       "24.08"),
    ("org.freedesktop.Sdk.Extension.golang",       "24.08"),
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
# Module-level helpers
# ---------------------------------------------------------------------------

def _is_extension(ref_id: str) -> bool:
    """Return True if ref_id is a Flatpak SDK Extension (not a base SDK)."""
    return ".Extension." in ref_id


def _base_sdk_from_extension(ext_id: str) -> str:
    """
    Derive the base SDK id from an extension id.

    Examples:
      org.freedesktop.Sdk.Extension.node24  → org.freedesktop.Sdk
      org.gnome.Sdk.Extension.rust-stable   → org.gnome.Sdk
    """
    if ".Extension." in ext_id:
        return ext_id.rsplit(".Extension.", 1)[0]
    return ext_id


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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        return bool(self._flatpak)

    def probe_sdk(self, sdk_id: str, sdk_version: str) -> ProbeResult:
        """
        Probe a single SDK (or extension) and write its profile.

        If sdk_id contains ".Extension." it is automatically routed to
        probe_extension() — no need to use a separate command.
        """
        if _is_extension(sdk_id):
            logger.info("%s looks like an extension — routing to probe_extension", sdk_id)
            return self.probe_extension(sdk_id, sdk_version)

        result = ProbeResult(sdk_id=sdk_id, sdk_version=sdk_version)

        if not self._flatpak:
            result.error = "flatpak not found"
            return result

        if self.auto_install and not self._is_installed(sdk_id, sdk_version):
            logger.info("Installing %s//%s ...", sdk_id, sdk_version)
            ok = self._install(sdk_id, sdk_version)
            if not ok:
                result.error = f"flatpak install failed for {sdk_id}//{sdk_version}"
                return result

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
        Probe a Flatpak SDK extension and write/update its profile TOML.

        The extension must be installed on the host (or auto_install=True).
        The base SDK is derived from the extension id if not given:
          org.freedesktop.Sdk.Extension.node24 → org.freedesktop.Sdk
        """
        result = ProbeResult(sdk_id=ext_id, sdk_version=sdk_version)

        if not self._flatpak:
            result.error = "flatpak not found"
            return result

        # Derive base SDK from extension id
        sdk = base_sdk or _base_sdk_from_extension(ext_id)
        logger.info("Using base SDK: %s for extension %s", sdk, ext_id)

        # Install extension if needed
        if self.auto_install and not self._is_installed(ext_id, sdk_version):
            logger.info("Installing extension %s//%s ...", ext_id, sdk_version)
            ok = self._install(ext_id, sdk_version)
            if not ok:
                result.error = (
                    f"flatpak install failed for {ext_id}//{sdk_version}. "
                    f"Try: flatpak install flathub {ext_id}//{sdk_version}"
                )
                return result

        # Verify base SDK is available for build-init
        if not self._is_installed(sdk, sdk_version):
            result.error = (
                f"Base SDK {sdk}//{sdk_version} is not installed. "
                f"Install it with: flatpak install flathub {sdk}//{sdk_version}"
            )
            return result

        # Derive mount path: last segment of extension id
        # e.g. org.freedesktop.Sdk.Extension.node24 → node24 → /usr/lib/sdk/node24
        short_name = ext_id.split(".")[-1]
        mount = f"/usr/lib/sdk/{short_name}"
        script = _EXT_INTROSPECT_SH_TEMPLATE.format(mount=mount)

        output = self._run_in_sdk(sdk, sdk_version, script, extra_extensions=[ext_id])
        if output is None:
            result.error = (
                f"Extension introspection failed for {ext_id}. "
                f"Verify the extension is installed: "
                f"flatpak info {ext_id}//{sdk_version}"
            )
            return result

        result = self._parse_extension_output(output, ext_id, sdk_version, mount)

        # Write a new profile TOML (or update existing one)
        self._write_extension_profile(result, ext_id, mount)
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
        Run a shell script inside the SDK via flatpak build-init + flatpak build.

        Path problem and solution
        -------------------------
        `flatpak build <dir> /path/to/script.sh` fails because the build-dir
        is NOT mounted at its host path inside the bubblewrap sandbox —
        flatpak mounts it at /run/build/<app-id>/ internally.  Any host
        path written to disk is therefore invisible inside the sandbox.

        The correct approach is to pass the script entirely via stdin:
          flatpak build ... /usr/bin/sh < script_content

        /usr/bin/sh is resolved inside the SDK runtime (always present),
        and stdin is inherited through bubblewrap without path issues.

        Extensions are activated via --env=PATH so their binaries are found
        without needing further setup.
        """
        with tempfile.TemporaryDirectory(prefix="pfmr-sdk-probe-") as tmp:
            tmp_path = Path(tmp)
            build_dir = tmp_path / "build"
            build_dir.mkdir()

            # Derive Platform from SDK id (replace only the trailing "Sdk"):
            #   org.freedesktop.Sdk → org.freedesktop.Platform
            #   org.gnome.Sdk       → org.gnome.Platform
            parts = sdk_id.split(".")
            platform = ".".join(
                "Platform" if (p == "Sdk" and i == len(parts) - 1) else p
                for i, p in enumerate(parts)
            )
            if platform == sdk_id:
                platform = "org.freedesktop.Platform"
                logger.warning(
                    "Could not derive Platform from %s, falling back to %s",
                    sdk_id, platform,
                )

            # Step 1 — initialise the build directory
            init_cmd = [
                self._flatpak, "build-init",
                str(build_dir),
                "org.pfmr.SdkProbe",
                sdk_id,
                platform,
                sdk_version,
            ]
            logger.debug("build-init: %s", " ".join(init_cmd))
            try:
                init_result = subprocess.run(
                    init_cmd, capture_output=True, text=True, timeout=30
                )
                if init_result.returncode != 0:
                    logger.warning(
                        "build-init failed (exit %d): %s",
                        init_result.returncode, init_result.stderr[-400:],
                    )
                    return None
            except subprocess.TimeoutExpired:
                logger.warning("build-init timed out")
                return None

            # Step 2 — build PATH for extensions
            env_args: list[str] = []
            if extra_extensions:
                ext_bins = [
                    f"/usr/lib/sdk/{e.split('.')[-1]}/bin"
                    for e in extra_extensions
                ]
                env_args = [f"--env=PATH={':'.join(ext_bins)}:/usr/bin:/bin"]

            # Step 3 — run the script via stdin.
            # /usr/bin/sh is resolved inside the SDK runtime.
            # Passing the script via stdin avoids any host/sandbox path mismatch.
            run_cmd = [
                self._flatpak, "build",
                "--with-appdir",
                "--allow=devel",
            ] + env_args + [
                str(build_dir),
                "/usr/bin/sh",
            ]
            logger.debug("flatpak build: %s", " ".join(run_cmd[:8]) + " ...")
            try:
                result = subprocess.run(
                    run_cmd,
                    input=script,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode == 0:
                    return result.stdout
                logger.warning(
                    "flatpak build failed (exit %d): %s",
                    result.returncode, result.stderr[-600:],
                )
            except subprocess.TimeoutExpired:
                logger.warning("SDK introspection timed out (>120s)")
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

    def _write_extension_profile(
        self, result: ProbeResult, ext_id: str, mount: str
    ) -> Path:
        """
        Write a new extension profile TOML to the extension-profiles directory.
        Creates the file only if it does not exist yet (does not overwrite
        manually curated profiles).
        """
        safe_name = ext_id.split(".")[-1]
        # Check if an existing profile already covers this extension
        existing = list(self.ext_output_dir.glob(f"*{safe_name}*.toml"))
        if existing:
            logger.debug(
                "Extension profile already exists at %s — skipping create", existing[0]
            )
            return existing[0]

        self.ext_output_dir.mkdir(parents=True, exist_ok=True)
        profile_path = self.ext_output_dir / f"{safe_name}.toml"

        lines = [
            f'extension_id = "{ext_id}"',
            f'display_name = "{safe_name}"',
            f'description = "Auto-generated by pfmr learn sdk probe"',
            f'mount_path = "{mount}"',
            "",
            "build_backends = []",
            "pkgconfig_triggers = []",
            "library_triggers = []",
            "package_triggers = []",
            "",
            "provides_executables = [",
            *[f'  "{e}",' for e in sorted(result.executables)],
            "]",
            "provides_pkgconfig = [",
            *[f'  "{p}",' for p in sorted(result.pkgconfig)],
            "]",
            "provides_libraries = [",
            *[f'  "{l}",' for l in sorted(result.libraries)],
            "]",
            "",
            "[env]",
            f'PATH = "{mount}/bin:$PATH"',
            "",
            "compatible_sdks = []",
        ]
        profile_path.write_text("\n".join(lines) + "\n")
        logger.info("Wrote extension profile: %s", profile_path)
        return profile_path

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