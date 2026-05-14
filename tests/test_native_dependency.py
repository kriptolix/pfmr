"""
Tests for pfmr — NativeDependencyAnalyzer (Phase 2).

Covers:
- Static hints lookup
- Build-backend heuristic
- Wheel tag analysis (native vs pure)
- ELF inspection (mocked)
- analyze() mutation of packages
- Extra hints file merging
"""
from __future__ import annotations

import textwrap
import zipfile
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest

from pfmr.models import BuildBackend, ResolvedPackage, SourceType
from pfmr.resolvers.native_dependency import (
    AnalysisResult,
    NativeDependencyAnalyzer,
    NativeHint,
    WheelTag,
    _filter_baseline,
    _load_hints,
    _parse_wheel_tag,
    _wheel_is_native,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pkg(
    name: str,
    backend: BuildBackend = BuildBackend.UNKNOWN,
    source_url: Optional[str] = None,
    source_type: Optional[SourceType] = None,
    requires_native: bool = False,
    native_deps: Optional[list[str]] = None,
) -> ResolvedPackage:
    return ResolvedPackage(
        name=name,
        version="1.0",
        build_backend=backend,
        source_url=source_url,
        source_type=source_type,
        requires_native=requires_native,
        native_deps=native_deps or [],
    )


# ---------------------------------------------------------------------------
# Wheel tag parsing
# ---------------------------------------------------------------------------

class TestWheelTagParsing:
    def test_pure_python_wheel(self):
        tag = _parse_wheel_tag("requests-2.31.0-py3-none-any.whl")
        assert tag is not None
        assert tag.abi == "none"
        assert tag.plat == "any"
        assert not _wheel_is_native(tag)

    def test_native_linux_wheel(self):
        tag = _parse_wheel_tag("numpy-1.26.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl")
        assert tag is not None
        assert _wheel_is_native(tag)

    def test_abi3_wheel_is_native(self):
        tag = _parse_wheel_tag("cryptography-43.0.0-cp37-abi3-manylinux_2_28_x86_64.whl")
        assert tag is not None
        assert _wheel_is_native(tag)

    def test_none_any_not_native(self):
        tag = _parse_wheel_tag("six-1.16.0-py2.py3-none-any.whl")
        assert tag is not None
        assert not _wheel_is_native(tag)

    def test_invalid_filename_returns_none(self):
        assert _parse_wheel_tag("not-a-wheel.tar.gz") is None
        assert _parse_wheel_tag("just_a_name") is None

    def test_cp3xx_linux_is_native(self):
        tag = _parse_wheel_tag("lxml-5.1.0-cp312-cp312-linux_x86_64.whl")
        assert tag is not None
        assert _wheel_is_native(tag)


# ---------------------------------------------------------------------------
# Baseline filter
# ---------------------------------------------------------------------------

class TestBaselineFilter:
    def test_removes_glibc(self):
        libs = ["libc.so.6", "libssl.so.3", "libpthread.so.0"]
        filtered = _filter_baseline(libs)
        assert "libc.so.6" not in filtered
        assert "libpthread.so.0" not in filtered
        assert "libssl.so.3" in filtered

    def test_removes_vdso(self):
        filtered = _filter_baseline(["linux-vdso.so.1", "libfoo.so.1"])
        assert "linux-vdso.so.1" not in filtered
        assert "libfoo.so.1" in filtered

    def test_keeps_real_libs(self):
        libs = ["libssl.so.3", "libxml2.so.2", "libcurl.so.4"]
        assert _filter_baseline(libs) == libs


# ---------------------------------------------------------------------------
# Hints loading
# ---------------------------------------------------------------------------

class TestHintsLoading:
    def test_load_builtin_hints(self):
        from pfmr.resolvers.native_dependency import _HINTS_FILE
        hints = _load_hints(_HINTS_FILE)
        assert "cryptography" in hints
        assert "cffi" in hints
        assert "lxml" in hints

    def test_hints_have_pkgconfig(self):
        from pfmr.resolvers.native_dependency import _HINTS_FILE
        hints = _load_hints(_HINTS_FILE)
        crypto = hints["cryptography"]
        assert "openssl" in crypto.pkgconfig
        assert "libffi" in crypto.pkgconfig

    def test_custom_hints_file(self, tmp_path):
        f = tmp_path / "custom.toml"
        f.write_text(textwrap.dedent("""\
            [myspecialpackage]
            pkgconfig = ["mylib"]
            libraries = ["libmylib.so.1"]
        """))
        hints = _load_hints(f)
        assert "myspecialpackage" in hints
        assert hints["myspecialpackage"].pkgconfig == ["mylib"]

    def test_missing_file_returns_empty(self, tmp_path):
        hints = _load_hints(tmp_path / "nonexistent.toml")
        assert hints == {}

    def test_canonical_name_key(self, tmp_path):
        """Package names in hints are canonicalized (PyYAML → pyyaml)."""
        f = tmp_path / "h.toml"
        f.write_text('[PyYAML]\npkgconfig = ["yaml-0.1"]')
        hints = _load_hints(f)
        assert "pyyaml" in hints


# ---------------------------------------------------------------------------
# NativeDependencyAnalyzer.analyze_one()
# ---------------------------------------------------------------------------

class TestAnalyzeOne:
    @pytest.fixture(autouse=True)
    def analyzer(self):
        self.a = NativeDependencyAnalyzer(enable_elf=False)

    def test_cryptography_from_hints(self):
        pkg = _pkg("cryptography")
        result = self.a.analyze_one(pkg)
        assert result.requires_native
        assert "openssl" in result.native_deps
        assert result.source == "hint"

    def test_cffi_from_hints(self):
        pkg = _pkg("cffi")
        result = self.a.analyze_one(pkg)
        assert result.requires_native
        assert "libffi" in result.native_deps

    def test_requests_is_pure(self):
        pkg = _pkg("requests", source_url="https://example.com/requests-2.31.0-py3-none-any.whl",
                   source_type=SourceType.WHEEL)
        result = self.a.analyze_one(pkg)
        assert not result.requires_native

    def test_maturin_backend_is_native_no_external_deps(self):
        pkg = _pkg("mylib", BuildBackend.MATURIN)
        result = self.a.analyze_one(pkg)
        assert result.requires_native
        assert result.native_deps == []   # toolchain-only, no external libs
        assert result.source == "backend"

    def test_setuptools_rust_backend_is_native(self):
        pkg = _pkg("mylib", BuildBackend.SETUPTOOLS_RUST)
        result = self.a.analyze_one(pkg)
        assert result.requires_native
        assert result.source == "backend"

    def test_native_linux_wheel_tag(self):
        pkg = _pkg(
            "lxml",
            source_url="https://example.com/lxml-5.1.0-cp312-cp312-linux_x86_64.whl",
            source_type=SourceType.WHEEL,
        )
        # lxml is in hints, so hint takes priority; test wheel tag fallback with unknown package
        pkg2 = _pkg(
            "unknownpkg",
            source_url="https://example.com/unknownpkg-1.0-cp312-cp312-manylinux_2_17_x86_64.whl",
            source_type=SourceType.WHEEL,
        )
        result = self.a.analyze_one(pkg2)
        assert result.requires_native
        assert result.source == "wheel_tag"

    def test_pure_any_wheel_tag(self):
        pkg = _pkg(
            "unknownpkg",
            source_url="https://example.com/unknownpkg-1.0-py3-none-any.whl",
            source_type=SourceType.WHEEL,
        )
        result = self.a.analyze_one(pkg)
        assert not result.requires_native
        assert result.source == "wheel_tag"

    def test_propagates_existing_requires_native(self):
        pkg = _pkg("unknownpkg", requires_native=True)
        result = self.a.analyze_one(pkg)
        assert result.requires_native
        assert result.source == "propagated"

    def test_pure_package_no_hint_no_flag(self):
        pkg = _pkg("unknownpure")
        result = self.a.analyze_one(pkg)
        assert not result.requires_native
        assert result.source == "none"


# ---------------------------------------------------------------------------
# NativeDependencyAnalyzer.analyze() — mutation
# ---------------------------------------------------------------------------

class TestAnalyzeMutation:
    @pytest.fixture(autouse=True)
    def analyzer(self):
        self.a = NativeDependencyAnalyzer(enable_elf=False)

    def test_analyze_fills_native_deps(self):
        pkg = _pkg("cryptography")
        self.a.analyze([pkg])
        assert "openssl" in pkg.native_deps
        assert pkg.requires_native

    def test_analyze_does_not_duplicate_existing_deps(self):
        pkg = _pkg("cryptography", native_deps=["openssl"])
        self.a.analyze([pkg])
        assert pkg.native_deps.count("openssl") == 1

    def test_analyze_merges_new_deps(self):
        pkg = _pkg("cryptography", native_deps=["openssl"])
        self.a.analyze([pkg])
        # libffi should be added
        assert "libffi" in pkg.native_deps

    def test_analyze_multiple_packages(self):
        pkgs = [_pkg("cryptography"), _pkg("cffi"), _pkg("requests")]
        results = self.a.analyze(pkgs)
        assert sum(1 for r in results if r.requires_native) == 2
        assert not any(
            r.requires_native for r in results if r.package_name == "requests"
        )

    def test_returns_analysis_results(self):
        pkgs = [_pkg("cryptography"), _pkg("requests")]
        results = self.a.analyze(pkgs)
        assert len(results) == 2
        assert all(isinstance(r, AnalysisResult) for r in results)


# ---------------------------------------------------------------------------
# Extra hints file merging
# ---------------------------------------------------------------------------

class TestExtraHints:
    def test_extra_hints_file_merged(self, tmp_path):
        f = tmp_path / "extra.toml"
        f.write_text(textwrap.dedent("""\
            [superspecialpackage]
            pkgconfig = ["speciallib"]
            libraries = ["libspecial.so.1"]
        """))
        analyzer = NativeDependencyAnalyzer(extra_hints_files=[f], enable_elf=False)
        pkg = _pkg("superspecialpackage")
        result = analyzer.analyze_one(pkg)
        assert result.requires_native
        assert "speciallib" in result.native_deps

    def test_extra_hints_override_builtin(self, tmp_path):
        """Extra hints file can override built-in entries."""
        f = tmp_path / "override.toml"
        f.write_text(textwrap.dedent("""\
            [cryptography]
            pkgconfig = ["my-custom-ssl"]
            libraries = []
        """))
        analyzer = NativeDependencyAnalyzer(extra_hints_files=[f], enable_elf=False)
        pkg = _pkg("cryptography")
        result = analyzer.analyze_one(pkg)
        assert "my-custom-ssl" in result.native_deps
        assert "openssl" not in result.native_deps


# ---------------------------------------------------------------------------
# ELF inspection (mocked)
# ---------------------------------------------------------------------------

class TestELFInspection:
    def _make_fake_wheel(self, tmp_path: Path, name: str, needed: list[str]) -> Path:
        """
        Create a minimal fake .whl (zip) with a single .so whose DT_NEEDED
        will be mocked.
        """
        wheel_path = tmp_path / f"{name}-1.0-cp312-cp312-linux_x86_64.whl"
        with zipfile.ZipFile(wheel_path, "w") as zf:
            # Write a placeholder bytes — ELF parsing is mocked anyway
            zf.writestr(f"{name}/_lib.so", b"\x7fELF" + b"\x00" * 60)
        return wheel_path

    def test_elf_analysis_fills_native_deps(self, tmp_path):
        wheel_path = self._make_fake_wheel(tmp_path, "mylib", ["libssl.so.3", "libz.so.1"])

        with patch(
            "pfmr.resolvers.native_dependency._elf_needed_libs",
            return_value=["libssl.so.3", "libz.so.1", "libc.so.6"],
        ):
            analyzer = NativeDependencyAnalyzer(
                enable_elf=True,
                wheel_cache_dir=tmp_path,
            )
            pkg = _pkg(
                "mylib",
                source_url=f"https://example.com/{wheel_path.name}",
                source_type=SourceType.WHEEL,
            )
            result = analyzer.analyze_one(pkg)

        assert result.requires_native
        assert result.source == "elf"
        # libc.so.6 must be filtered out (baseline)
        assert "libc.so.6" not in result.native_deps
        assert "libssl.so.3" in result.native_deps

    def test_elf_skipped_when_disabled(self, tmp_path):
        wheel_path = self._make_fake_wheel(tmp_path, "mylib2", [])
        analyzer = NativeDependencyAnalyzer(enable_elf=False, wheel_cache_dir=tmp_path)
        pkg = _pkg(
            "mylib2",
            source_url=f"https://example.com/{wheel_path.name}",
            source_type=SourceType.WHEEL,
        )
        result = analyzer.analyze_one(pkg)
        # ELF disabled → falls back to wheel_tag (native linux wheel)
        assert result.source == "wheel_tag"

    def test_elf_skipped_when_no_cache_dir(self):
        analyzer = NativeDependencyAnalyzer(enable_elf=True, wheel_cache_dir=None)
        pkg = _pkg(
            "mylib",
            source_url="https://example.com/mylib-1.0-cp312-cp312-linux_x86_64.whl",
            source_type=SourceType.WHEEL,
        )
        result = analyzer.analyze_one(pkg)
        # No cache dir → falls to wheel_tag
        assert result.source == "wheel_tag"

    def test_elf_skipped_when_wheel_not_in_cache(self, tmp_path):
        analyzer = NativeDependencyAnalyzer(enable_elf=True, wheel_cache_dir=tmp_path)
        pkg = _pkg(
            "notcached",
            source_url="https://example.com/notcached-1.0-cp312-cp312-linux_x86_64.whl",
            source_type=SourceType.WHEEL,
        )
        result = analyzer.analyze_one(pkg)
        # Wheel not in cache → wheel_tag
        assert result.source == "wheel_tag"