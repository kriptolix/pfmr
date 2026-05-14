"""
pfmr.resolvers.sdk_extension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SDKExtensionResolver — Phase 2 component.

Objetivo (spec §5.4):
  Detectar quais SDK Extensions do Flatpak precisam ser declaradas no
  manifesto para que os pacotes Python resolvidos consigam compilar.

Fontes de decisao (em ordem de prioridade):

  1. Build backend do pacote
       maturin / setuptools-rust  → rust-stable
       scikit-build / scikit-build-core → (sem extensao obrigatoria, mas
         pode precisar de llvm se o projeto linkar contra clang)

  2. Dependencias nativas explicitas (pkg-config / sonames)
       "llvm" / "clang"           → llvm18
       "libjvm.so"                → openjdk21
       "libgfortran.so"           → gfortran

  3. Nome canonico do pacote Python
       "cryptography", "orjson"   → rust-stable
       "llvmlite", "numba"        → llvm18
       "pyjnius", "jpype1"        → openjdk21
       "scipy", "numpy"           → gfortran (quando sem wheel)

  4. Overrides manuais (injetados pela Pipeline via `forced_extensions`)

Fluxo:
  SDKExtensionResolver
    .resolve(packages, sdk_id, sdk_version)
        ↓
    para cada ExtensionProfile carregado:
        _check_profile(profile, packages)  →  ExtensionMatch | None
        ↓
    deduplicar + ordenar por prioridade
        ↓
    ExtensionResolutionReport
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from packaging.utils import canonicalize_name

from pfmr.models import (
    BuildBackend,
    ExtensionMatch,
    ExtensionProfile,
    ExtensionResolutionReport,
    ResolvedPackage,
)
from pfmr.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_BUILTIN_EXTENSION_PROFILES_DIR = (
    Path(__file__).parent.parent / "data" / "extension-profiles"
)

# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------

def _parse_extension_profile(path: Path) -> ExtensionProfile:
    with open(path, "rb") as f:
        data = tomllib.load(f)

    env_raw = data.get("env", {})
    # tomllib parses [env] as a nested dict — flatten to str→str
    env: dict[str, str] = {k: str(v) for k, v in env_raw.items()}

    return ExtensionProfile(
        extension_id=data["extension_id"],
        display_name=data.get("display_name", data["extension_id"]),
        description=data.get("description", "").strip(),
        mount_path=data.get("mount_path", ""),
        build_backends=data.get("build_backends", []),
        pkgconfig_triggers=data.get("pkgconfig_triggers", []),
        library_triggers=data.get("library_triggers", []),
        package_triggers=[
            canonicalize_name(p) for p in data.get("package_triggers", [])
        ],
        provides_executables=data.get("provides_executables", []),
        provides_pkgconfig=data.get("provides_pkgconfig", []),
        provides_libraries=data.get("provides_libraries", []),
        env=env,
        compatible_sdks=data.get("compatible_sdks", []),
    )


def load_extension_profiles(
    extra_dirs: Optional[list[Path]] = None,
) -> list[ExtensionProfile]:
    """Load all TOML extension profiles from built-in dir + any extra dirs."""
    dirs = [_BUILTIN_EXTENSION_PROFILES_DIR] + (extra_dirs or [])
    profiles: list[ExtensionProfile] = []
    seen: set[str] = set()

    for d in dirs:
        if not d.exists():
            logger.debug("Extension profile dir not found (skipping): %s", d)
            continue
        for toml_path in sorted(d.glob("*.toml")):
            try:
                profile = _parse_extension_profile(toml_path)
                if profile.extension_id in seen:
                    logger.debug("Duplicate extension profile skipped: %s", profile.extension_id)
                    continue
                profiles.append(profile)
                seen.add(profile.extension_id)
                logger.debug("Loaded extension profile: %s", profile.extension_id)
            except Exception as exc:
                logger.warning("Failed to parse extension profile %s: %s", toml_path, exc)

    logger.info("Loaded %d extension profiles", len(profiles))
    return profiles


# ---------------------------------------------------------------------------
# Public resolver
# ---------------------------------------------------------------------------

class SDKExtensionResolver:
    """
    Determines which Flatpak SDK Extensions must be declared in the manifest
    for a given set of resolved Python packages.

    Usage::

        resolver = SDKExtensionResolver()
        report = resolver.resolve(packages, sdk_id="org.freedesktop.Sdk")
        for match in report.required_extensions:
            print(match.extension_id, "because:", match.reasons)
    """

    def __init__(
        self,
        extra_profile_dirs: Optional[list[Path]] = None,
        # Extensions forced by the user regardless of analysis
        forced_extensions: Optional[list[str]] = None,
        # Extensions explicitly excluded (overrides detection)
        excluded_extensions: Optional[list[str]] = None,
    ):
        self._profiles = load_extension_profiles(extra_profile_dirs)
        self._forced = list(forced_extensions or [])
        self._excluded = set(excluded_extensions or [])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        packages: list[ResolvedPackage],
        sdk_id: str = "org.freedesktop.Sdk",
        sdk_version: str = "24.08",
    ) -> ExtensionResolutionReport:
        """
        Analyse `packages` and return an ExtensionResolutionReport describing
        every SDK Extension that must be listed in sdk-extensions.
        """
        # Index packages by canonical name for O(1) lookup
        pkg_by_name: dict[str, ResolvedPackage] = {
            canonicalize_name(p.name): p for p in packages
        }

        matched: dict[str, ExtensionMatch] = {}

        # --- extension-profile-driven detection ---
        for profile in self._profiles:
            # Skip extensions incompatible with this SDK
            if profile.compatible_sdks and sdk_id not in profile.compatible_sdks:
                continue
            # Skip explicitly excluded
            if profile.extension_id in self._excluded:
                continue

            match = self._check_profile(profile, packages, pkg_by_name)
            if match:
                if profile.extension_id not in matched:
                    matched[profile.extension_id] = match
                else:
                    # Merge reasons
                    existing = matched[profile.extension_id]
                    for r in match.reasons:
                        if r not in existing.reasons:
                            existing.reasons.append(r)
                    for p in match.triggered_by_packages:
                        if p not in existing.triggered_by_packages:
                            existing.triggered_by_packages.append(p)

        # --- forced extensions ---
        for ext_id in self._forced:
            if ext_id not in matched and ext_id not in self._excluded:
                profile = self._profile_by_id(ext_id)
                matched[ext_id] = ExtensionMatch(
                    extension_id=ext_id,
                    display_name=profile.display_name if profile else ext_id,
                    reasons=[("forced", "manual override")],
                    env=profile.env if profile else {},
                    mount_path=profile.mount_path if profile else "",
                )

        # Build final ordered list (forced first, then by first-match order)
        ordered = list(matched.values())
        report = ExtensionResolutionReport(
            required_extensions=ordered,
            extension_ids=[m.extension_id for m in ordered],
        )
        logger.info(
            "Extension resolution: %d required — %s",
            len(ordered),
            report.extension_ids,
        )
        return report

    def profiles(self) -> list[ExtensionProfile]:
        """Return all loaded extension profiles."""
        return list(self._profiles)

    def profile_by_id(self, ext_id: str) -> Optional[ExtensionProfile]:
        return self._profile_by_id(ext_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_profile(
        self,
        profile: ExtensionProfile,
        packages: list[ResolvedPackage],
        pkg_by_name: dict[str, ResolvedPackage],
    ) -> Optional[ExtensionMatch]:
        """
        Return an ExtensionMatch if any package in the list triggers this
        profile, or None if the profile is not needed.
        """
        reasons: list[tuple[str, str]] = []
        triggered_by: list[str] = []

        for pkg in packages:
            pkg_canonical = canonicalize_name(pkg.name)

            # 1. Build-backend trigger
            if pkg.build_backend.value in profile.build_backends:
                reasons.append(("build_backend", pkg.build_backend.value))
                if pkg.name not in triggered_by:
                    triggered_by.append(pkg.name)

            # 2. pkg-config trigger (from package's recorded native_deps)
            for pc in profile.pkgconfig_triggers:
                if pc in (pkg.native_deps or []):
                    reasons.append(("pkgconfig", pc))
                    if pkg.name not in triggered_by:
                        triggered_by.append(pkg.name)

            # 3. Library trigger (from package's recorded native_deps)
            for soname in profile.library_triggers:
                soname_base = re.sub(r"\.so(\..+)?$", ".so", soname)
                for dep in pkg.native_deps or []:
                    dep_base = re.sub(r"\.so(\..+)?$", ".so", dep)
                    if dep == soname or dep_base == soname_base:
                        reasons.append(("library", soname))
                        if pkg.name not in triggered_by:
                            triggered_by.append(pkg.name)

            # 4. Package-name trigger
            if pkg_canonical in profile.package_triggers:
                reasons.append(("package", pkg.name))
                if pkg.name not in triggered_by:
                    triggered_by.append(pkg.name)

        if not reasons:
            return None

        # Deduplicate reasons (keep first occurrence of each unique (type, value))
        seen_reasons: set[tuple[str, str]] = set()
        deduped: list[tuple[str, str]] = []
        for r in reasons:
            if r not in seen_reasons:
                deduped.append(r)
                seen_reasons.add(r)

        return ExtensionMatch(
            extension_id=profile.extension_id,
            display_name=profile.display_name,
            reasons=deduped,
            triggered_by_packages=triggered_by,
            env=dict(profile.env),
            mount_path=profile.mount_path,
        )

    def _profile_by_id(self, ext_id: str) -> Optional[ExtensionProfile]:
        for p in self._profiles:
            if p.extension_id == ext_id:
                return p
        return None