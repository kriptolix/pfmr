"""
pfmr.models — shared data models for the entire pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class BuildBackend(str, Enum):
    SETUPTOOLS = "setuptools"
    MATURIN = "maturin"
    MESON_PYTHON = "meson-python"
    SETUPTOOLS_RUST = "setuptools-rust"
    SCIKIT_BUILD = "scikit-build"
    SCIKIT_BUILD_CORE = "scikit-build-core"
    FLIT = "flit"
    PDM = "pdm-backend"
    HATCH = "hatchling"
    POETRY = "poetry"
    UNKNOWN = "unknown"


class SourceType(str, Enum):
    WHEEL = "wheel"
    SDIST = "sdist"


@dataclass
class ResolvedPackage:
    """A single resolved Python package with all metadata needed for Flatpak manifest generation."""

    name: str
    version: str
    wheel_available: bool = False
    build_backend: BuildBackend = BuildBackend.UNKNOWN
    requires_native: bool = False
    # direct or transitive
    is_direct: bool = False
    # sha256 of the chosen source (wheel or sdist)
    source_hash: Optional[str] = None
    source_url: Optional[str] = None
    source_type: Optional[SourceType] = None
    # native libraries this package needs (populated by NativeDependencyAnalyzer in Phase 2)
    native_deps: list[str] = field(default_factory=list)
    # sdk extensions this package needs (populated by SDKExtensionResolver in Phase 2)
    required_extensions: list[str] = field(default_factory=list)
    # extras / env markers
    extras: list[str] = field(default_factory=list)


@dataclass
class FlatpakSource:
    type: str  # "archive", "file", "git", "patch"
    url: Optional[str] = None
    sha256: Optional[str] = None
    path: Optional[str] = None
    dest_filename: Optional[str] = None
    branch: Optional[str] = None
    commit: Optional[str] = None
    tag: Optional[str] = None


@dataclass
class FlatpakModule:
    name: str
    buildsystem: str = "simple"
    build_commands: list[str] = field(default_factory=list)
    sources: list[FlatpakSource] = field(default_factory=list)
    build_options: dict = field(default_factory=dict)
    modules: list["FlatpakModule"] = field(default_factory=list)  # sub-modules
    cleanup: list[str] = field(default_factory=list)
    config_opts: list[str] = field(default_factory=list)


@dataclass
class FlatpakManifest:
    app_id: str
    runtime: str
    runtime_version: str
    sdk: str
    sdk_extensions: list[str] = field(default_factory=list)
    modules: list[FlatpakModule] = field(default_factory=list)
    finish_args: list[str] = field(default_factory=list)


@dataclass
class NativeRecipe:
    """A recipe for building a native library not available in the Flatpak SDK."""

    id: str
    provides: list[str] = field(default_factory=list)   # .so names
    pkgconfig: list[str] = field(default_factory=list)  # .pc names
    headers: list[str] = field(default_factory=list)
    buildsystem: str = "autotools"
    source: Optional[FlatpakSource] = None
    build_commands: list[str] = field(default_factory=list)
    config_opts: list[str] = field(default_factory=list)
    cleanup: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)


@dataclass
class ResolutionResult:
    """Final output of the full resolver pipeline."""

    packages: list[ResolvedPackage] = field(default_factory=list)
    # packages that need native recipes
    unresolved_natives: list[str] = field(default_factory=list)
    # recipes found in local DB
    native_recipes: list[NativeRecipe] = field(default_factory=list)
    required_extensions: list[str] = field(default_factory=list)
    lockfile_hash: Optional[str] = None


# ---------------------------------------------------------------------------
# Phase 2 — SDKCapabilityResolver types
# ---------------------------------------------------------------------------

@dataclass
class SDKCapability:
    """
    Describes what a single Flatpak runtime/SDK/extension provides.

    Populated from:
      - static profile files (data/sdk-profiles/<id>/<version>.toml)
      - live probing via flatpak info + pkg-config (when available)
    """
    # identity
    sdk_id: str           # e.g. "org.freedesktop.Sdk"
    sdk_version: str      # e.g. "24.08"

    # what the SDK exposes
    libraries: list[str] = field(default_factory=list)      # sonames: "libssl.so.3"
    pkgconfig: list[str] = field(default_factory=list)      # pc names: "openssl"
    headers: list[str] = field(default_factory=list)        # "openssl/ssl.h"
    executables: list[str] = field(default_factory=list)    # "python3", "gcc"
    python_modules: list[str] = field(default_factory=list) # stdlib or pre-installed

    # source of truth
    probed_live: bool = False   # True if populated by live flatpak probe

    def provides_library(self, soname: str) -> bool:
        """Match exact soname or base name (strips version suffix)."""
        import re
        if soname in self.libraries:
            return True
        base = re.sub(r"\.so(\..+)?$", ".so", soname)
        return any(re.sub(r"\.so(\..+)?$", ".so", lib) == base for lib in self.libraries)

    def provides_pkgconfig(self, pc: str) -> bool:
        pc = pc.removesuffix(".pc")
        return pc in self.pkgconfig

    def provides_header(self, header: str) -> bool:
        return header in self.headers

    def provides_executable(self, exe: str) -> bool:
        return exe in self.executables


@dataclass
class SDKCheckResult:
    """Result of checking a single native dependency against the SDK."""
    query: str                  # what was queried (soname / pc name / header)
    query_type: str             # "library" | "pkgconfig" | "header" | "executable"
    satisfied: bool
    provided_by: Optional[str] = None   # sdk_id that satisfies it
    # If not satisfied, candidate recipes from local DB
    recipe_candidates: list[str] = field(default_factory=list)


@dataclass
class SDKResolutionReport:
    """Full report of SDK capability resolution for a ResolutionResult."""
    sdk_id: str
    sdk_version: str
    checks: list[SDKCheckResult] = field(default_factory=list)

    @property
    def satisfied(self) -> list[SDKCheckResult]:
        return [c for c in self.checks if c.satisfied]

    @property
    def missing(self) -> list[SDKCheckResult]:
        return [c for c in self.checks if not c.satisfied]

    @property
    def is_sufficient(self) -> bool:
        return len(self.missing) == 0


# ---------------------------------------------------------------------------
# Phase 2 — SDKExtensionResolver types
# ---------------------------------------------------------------------------

@dataclass
class ExtensionProfile:
    """
    Describes a single Flatpak SDK Extension and everything it provides.

    Profiles live at data/extension-profiles/<ext-id>.toml.
    The extension_id is the full Flatpak ref, e.g.:
      org.freedesktop.Sdk.Extension.rust-stable
    """
    extension_id: str          # full ref
    display_name: str          # human label, e.g. "Rust (stable)"

    # Build backends that unconditionally require this extension
    build_backends: list[str] = field(default_factory=list)

    # pkg-config names whose presence in native_deps triggers this extension
    pkgconfig_triggers: list[str] = field(default_factory=list)

    # sonames whose presence triggers this extension
    library_triggers: list[str] = field(default_factory=list)

    # Canonical Python package names that always need this extension
    package_triggers: list[str] = field(default_factory=list)

    # What the extension adds to the build environment
    provides_executables: list[str] = field(default_factory=list)
    provides_pkgconfig: list[str] = field(default_factory=list)
    provides_libraries: list[str] = field(default_factory=list)

    # Flatpak mount path inside the build sandbox
    mount_path: str = ""          # e.g. "/usr/lib/sdk/rust-stable"

    # Environment variables the extension injects (written to build-options)
    env: dict[str, str] = field(default_factory=dict)

    # Compatible SDK base IDs (empty list = works with any SDK)
    compatible_sdks: list[str] = field(default_factory=list)

    # Description / notes shown in CLI
    description: str = ""


@dataclass
class ExtensionMatch:
    """A single extension decided to be required, with full reasoning."""
    extension_id: str
    display_name: str
    # (reason_type, value): reason_type in "build_backend"|"pkgconfig"|"library"|"package"
    reasons: list[tuple[str, str]] = field(default_factory=list)
    triggered_by_packages: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    mount_path: str = ""


@dataclass
class ExtensionResolutionReport:
    """Full report produced by SDKExtensionResolver.resolve()."""
    required_extensions: list[ExtensionMatch] = field(default_factory=list)
    # ordered list of extension_ids for sdk-extensions: manifest field
    extension_ids: list[str] = field(default_factory=list)

    def has_extension(self, ext_id: str) -> bool:
        return ext_id in self.extension_ids