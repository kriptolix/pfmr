"""
pfmr.resolvers.native_dependency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
NativeDependencyAnalyzer — Phase 2 component.

Objetivo (spec §5.2):
  Detectar as dependencias nativas reais de cada pacote Python resolvido,
  preenchendo ResolvedPackage.native_deps com nomes pkg-config / sonames
  que precisam existir no ambiente de build.

Fontes de analise (em ordem de aplicacao):

  1. Banco estatico de hints (data/native-hints/packages.toml)
       Conhecimento curado: mapeia nome canonico → {pkgconfig, libraries}
       Mais rapido e confiavel; cobre a maioria dos pacotes conhecidos.

  2. Build-backend heuristics
       Backend → implicacoes nativas conhecidas sem precisar inspecionar o wheel.
       Ex.: maturin → nenhuma lib nativa externa (so Rust toolchain);
            meson-python → pode precisar de qualquer coisa (inspecao necessaria);
            scikit-build / scikit-build-core → CMake build, verifica hints.

  3. Wheel tag analysis
       Analisa o filename do wheel para inferir se e nativo:
         *-cp3XX-cp3XX-linux_x86_64.whl   → nativo
         *-py3-none-any.whl               → puro Python
         *-cp3XX-abi3-manylinux*.whl      → nativo (abi3)
       Se nativo mas ausente do banco, marca requires_native=True sem deps
       especificos (aguarda fase 3 — Build Sandbox Inspector).

  4. Inspecao de ELF (opcional, requer pyelftools)
       Quando um wheel ja esta no disco local (cache do uv), executa ldd/
       readelf para extrair as bibliotecas linkadas diretamente.
       Usado apenas quando pyelftools esta disponivel.

  5. Manylinux tag inference
       Wheels manylinux excluem bibliotecas que fazem parte do runtime
       manylinux (glibc, libpthread, etc.); o analyzer subtrai essas libs
       do resultado para nao gerar modulos redundantes.

Saida:
  Para cada ResolvedPackage analizado, preenche:
    pkg.requires_native  — True se precisa de compilacao nativa
    pkg.native_deps      — lista de pkgconfig names das dependencias
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from packaging.utils import canonicalize_name

from pfmr.models import BuildBackend, ResolvedPackage, SourceType
from pfmr.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HINTS_FILE = Path(__file__).parent.parent / "data" / "native-hints" / "packages.toml"

# ---------------------------------------------------------------------------
# Manylinux / musllinux baseline — libs that are always present in the
# Flatpak runtime and must NOT be added as native_deps.
# ---------------------------------------------------------------------------

_RUNTIME_BASELINE_LIBS: frozenset[str] = frozenset(
    {
        "libpthread.so.0",
        "libm.so.6",
        "libc.so.6",
        "libdl.so.2",
        "librt.so.1",
        "libutil.so.1",
        "libgcc_s.so.1",
        "libstdc++.so.6",
        "ld-linux-x86-64.so.2",
        "ld-linux-aarch64.so.1",
        "linux-vdso.so.1",
    }
)

# Soname prefixes that are part of glibc / base runtime
_BASELINE_PREFIXES = ("linux-vdso", "ld-linux")

# ---------------------------------------------------------------------------
# Wheel filename parser
# ---------------------------------------------------------------------------

_WHEEL_RE = re.compile(
    r"^(?P<name>[^-]+)-(?P<version>[^-]+)"
    r"(-(?P<build>\d[^-]*))?-(?P<pyver>[^-]+)-(?P<abi>[^-]+)-(?P<plat>.+)\.whl$"
)


@dataclass
class WheelTag:
    name: str
    version: str
    pyver: str   # cp312, py3, cp3XX
    abi: str     # cp312, abi3, none
    plat: str    # linux_x86_64, manylinux_..., none


def _parse_wheel_tag(filename: str) -> Optional[WheelTag]:
    m = _WHEEL_RE.match(filename)
    if not m:
        return None
    return WheelTag(
        name=m.group("name"),
        version=m.group("version"),
        pyver=m.group("pyver"),
        abi=m.group("abi"),
        plat=m.group("plat"),
    )


def _wheel_is_native(tag: WheelTag) -> bool:
    """Return True if the wheel contains compiled C/C++/Rust code."""
    if tag.plat == "any" or tag.abi == "none":
        return False
    if "linux" in tag.plat or "macos" in tag.plat or "win" in tag.plat:
        return True
    if tag.abi.startswith("cp") or tag.abi == "abi3":
        return tag.plat != "any"
    return False


# ---------------------------------------------------------------------------
# ELF inspector (optional — requires pyelftools)
# ---------------------------------------------------------------------------

def _elf_needed_libs(so_path: Path) -> list[str]:
    """
    Return DT_NEEDED entries from an ELF shared object.
    Falls back to an empty list if pyelftools is not installed.
    """
    try:
        from elftools.elf.elffile import ELFFile
        from elftools.elf.dynamic import DynamicSection
    except ImportError:
        logger.debug("pyelftools not installed; skipping ELF analysis of %s", so_path)
        return []

    needed: list[str] = []
    try:
        with open(so_path, "rb") as f:
            elf = ELFFile(f)
            for section in elf.iter_sections():
                if isinstance(section, DynamicSection):
                    for tag in section.iter_tags():
                        if tag.entry.d_tag == "DT_NEEDED":
                            needed.append(tag.needed)
    except Exception as exc:
        logger.debug("ELF analysis failed for %s: %s", so_path, exc)
    return needed


def _filter_baseline(libs: list[str]) -> list[str]:
    """Remove baseline runtime libraries that are always present."""
    result = []
    for lib in libs:
        if lib in _RUNTIME_BASELINE_LIBS:
            continue
        if any(lib.startswith(p) for p in _BASELINE_PREFIXES):
            continue
        result.append(lib)
    return result


# ---------------------------------------------------------------------------
# Static hints database
# ---------------------------------------------------------------------------

@dataclass
class NativeHint:
    """Static knowledge about a single package's native dependencies."""
    pkgconfig: list[str] = field(default_factory=list)
    libraries: list[str] = field(default_factory=list)
    headers: list[str] = field(default_factory=list)
    extras: str = ""


def _load_hints(hints_file: Path) -> dict[str, NativeHint]:
    """
    Load the packages.toml hints file.
    Returns dict keyed by canonical package name.
    """
    if not hints_file.exists():
        logger.warning("Native hints file not found: %s", hints_file)
        return {}
    try:
        with open(hints_file, "rb") as f:
            raw = tomllib.load(f)
    except Exception as exc:
        logger.error("Failed to load native hints: %s", exc)
        return {}

    hints: dict[str, NativeHint] = {}
    for pkg_name, data in raw.items():
        if not isinstance(data, dict):
            continue
        hints[canonicalize_name(pkg_name)] = NativeHint(
            pkgconfig=data.get("pkgconfig", []),
            libraries=data.get("libraries", []),
            headers=data.get("headers", []),
            extras=data.get("extras", ""),
        )
    logger.debug("Loaded %d native hint entries", len(hints))
    return hints


# ---------------------------------------------------------------------------
# Backend implications
# ---------------------------------------------------------------------------

# Build backends that produce native extensions but have no *external* library
# requirements by themselves — they only need the toolchain extension.
_TOOLCHAIN_ONLY_BACKENDS: set[BuildBackend] = {
    BuildBackend.MATURIN,
    BuildBackend.SETUPTOOLS_RUST,
}

# Build backends that may link against external libs — we flag requires_native
# but can only know the specific libs from hints or ELF analysis.
_EXTERNAL_POSSIBLE_BACKENDS: set[BuildBackend] = {
    BuildBackend.MESON_PYTHON,
    BuildBackend.SCIKIT_BUILD,
    BuildBackend.SCIKIT_BUILD_CORE,
    BuildBackend.SETUPTOOLS,   # when the package has ext_modules
}


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    """Analysis result for a single package."""
    package_name: str
    requires_native: bool = False
    # Final merged native deps (pkgconfig names preferred over sonames)
    native_deps: list[str] = field(default_factory=list)
    # Source of analysis that produced the result
    source: str = "unknown"   # "hint" | "wheel_tag" | "elf" | "backend" | "none"


class NativeDependencyAnalyzer:
    """
    Determines the native library dependencies of each resolved Python package
    and populates ResolvedPackage.native_deps + ResolvedPackage.requires_native.

    Usage::

        analyzer = NativeDependencyAnalyzer()
        analyzer.analyze(packages)   # mutates packages in-place
    """

    def __init__(
        self,
        # Override hints file location (for testing)
        hints_file: Optional[Path] = None,
        # Additional hints dirs to merge
        extra_hints_files: Optional[list[Path]] = None,
        # Whether to attempt ELF inspection when wheels are on disk
        enable_elf: bool = True,
        # Local wheel cache directory (e.g. ~/.cache/uv/wheels/)
        wheel_cache_dir: Optional[Path] = None,
    ):
        self._enable_elf = enable_elf
        self._wheel_cache_dir = wheel_cache_dir
        self._hints = _load_hints(hints_file or _HINTS_FILE)

        # Merge extra hints files
        for extra in extra_hints_files or []:
            extra_hints = _load_hints(extra)
            # Extra files override built-in hints for the same package
            self._hints.update(extra_hints)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, packages: list[ResolvedPackage]) -> list[AnalysisResult]:
        """
        Analyze all packages and mutate them in-place.
        Returns the list of AnalysisResult for inspection / logging.
        """
        results: list[AnalysisResult] = []
        for pkg in packages:
            result = self._analyze_one(pkg)
            results.append(result)
            # Apply back to the package
            if result.requires_native:
                pkg.requires_native = True
            if result.native_deps:
                # Merge: keep existing deps, add new ones
                existing = set(pkg.native_deps)
                for dep in result.native_deps:
                    if dep not in existing:
                        pkg.native_deps.append(dep)
                        existing.add(dep)

        native_count = sum(1 for r in results if r.requires_native)
        logger.info(
            "Native analysis: %d/%d packages require native compilation",
            native_count, len(packages),
        )
        return results

    def analyze_one(self, pkg: ResolvedPackage) -> AnalysisResult:
        """Analyze a single package without mutating it."""
        return self._analyze_one(pkg)

    def hints_for(self, package_name: str) -> Optional[NativeHint]:
        """Return the static hint for a package, or None."""
        return self._hints.get(canonicalize_name(package_name))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _analyze_one(self, pkg: ResolvedPackage) -> AnalysisResult:
        canonical = canonicalize_name(pkg.name)

        # --- 1. Static hints (highest confidence) ---
        hint = self._hints.get(canonical)
        if hint:
            # A hint with pkgconfig entries means native
            requires_native = bool(hint.pkgconfig or hint.libraries)
            deps = list(hint.pkgconfig) if hint.pkgconfig else list(hint.libraries)
            if requires_native or pkg.requires_native:
                return AnalysisResult(
                    package_name=pkg.name,
                    requires_native=True,
                    native_deps=deps,
                    source="hint",
                )
            # Hint exists but says no external deps — still respect requires_native flag
            if pkg.requires_native:
                return AnalysisResult(
                    package_name=pkg.name,
                    requires_native=True,
                    native_deps=[],
                    source="hint",
                )
            # Pure package with hint entry (extras-only hint)
            return AnalysisResult(
                package_name=pkg.name,
                requires_native=False,
                native_deps=[],
                source="hint",
            )

        # --- 2. Build-backend heuristic ---
        if pkg.build_backend in _TOOLCHAIN_ONLY_BACKENDS:
            # Only needs toolchain extension — no external native libs
            return AnalysisResult(
                package_name=pkg.name,
                requires_native=True,
                native_deps=[],   # toolchain handled by SDKExtensionResolver
                source="backend",
            )

        # --- 3. ELF inspection — runs before wheel_tag when cache is available,
        #        because it gives richer information (actual linked libs).
        if self._enable_elf and pkg.source_type == SourceType.WHEEL:
            elf_result = self._analyze_elf(pkg)
            if elf_result is not None:
                return elf_result

        # --- 4. Wheel tag analysis (fallback when ELF not available) ---
        wheel_result = self._analyze_wheel_tag(pkg)
        if wheel_result is not None:
            return wheel_result

        # --- 5. Propagate existing requires_native flag ---
        if pkg.requires_native:
            return AnalysisResult(
                package_name=pkg.name,
                requires_native=True,
                native_deps=list(pkg.native_deps),
                source="propagated",
            )

        return AnalysisResult(
            package_name=pkg.name,
            requires_native=False,
            native_deps=[],
            source="none",
        )

    def _analyze_wheel_tag(self, pkg: ResolvedPackage) -> Optional[AnalysisResult]:
        """
        Infer native status from wheel filename without downloading anything.
        Returns None if no wheel URL is available.
        """
        if not pkg.source_url:
            return None
        filename = pkg.source_url.split("/")[-1].split("?")[0]
        tag = _parse_wheel_tag(filename)
        if tag is None:
            return None  # not a wheel URL

        is_native = _wheel_is_native(tag)
        return AnalysisResult(
            package_name=pkg.name,
            requires_native=is_native,
            native_deps=[],   # filename alone can't tell us which libs
            source="wheel_tag",
        )

    def _analyze_elf(self, pkg: ResolvedPackage) -> Optional[AnalysisResult]:
        """
        If the wheel is cached locally, extract its .so files and run ELF
        analysis to find DT_NEEDED entries.
        """
        if not self._wheel_cache_dir:
            return None

        # Find cached wheel file
        wheel_filename = (pkg.source_url or "").split("/")[-1].split("?")[0]
        if not wheel_filename.endswith(".whl"):
            return None

        candidates = list(self._wheel_cache_dir.glob(f"**/{wheel_filename}"))
        if not candidates:
            logger.debug("Wheel not in local cache: %s", wheel_filename)
            return None

        wheel_path = candidates[0]
        needed_libs = self._extract_elf_from_wheel(wheel_path)
        if needed_libs is None:
            return None

        filtered = _filter_baseline(needed_libs)
        return AnalysisResult(
            package_name=pkg.name,
            requires_native=True,
            native_deps=filtered,
            source="elf",
        )

    @staticmethod
    def _extract_elf_from_wheel(wheel_path: Path) -> Optional[list[str]]:
        """
        Unzip the wheel (it's a zip file) and run ELF analysis on every .so.
        Returns the merged list of DT_NEEDED sonames, or None on failure.
        """
        import zipfile
        import tempfile

        all_needed: list[str] = []
        try:
            with zipfile.ZipFile(wheel_path, "r") as zf:
                so_names = [n for n in zf.namelist() if n.endswith(".so") or ".so." in n]
                if not so_names:
                    return None
                with tempfile.TemporaryDirectory(prefix="pfmr-elf-") as tmp:
                    tmp_path = Path(tmp)
                    for so_name in so_names:
                        extracted = tmp_path / Path(so_name).name
                        extracted.write_bytes(zf.read(so_name))
                        needed = _elf_needed_libs(extracted)
                        all_needed.extend(needed)
        except Exception as exc:
            logger.debug("ELF wheel extraction failed for %s: %s", wheel_path, exc)
            return None

        # Deduplicate
        seen: set[str] = set()
        result: list[str] = []
        for lib in all_needed:
            if lib not in seen:
                result.append(lib)
                seen.add(lib)
        return result