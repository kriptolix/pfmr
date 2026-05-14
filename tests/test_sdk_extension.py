"""
Tests for pfmr — SDKExtensionResolver (Phase 2).

Covers:
- ExtensionProfile loading from TOML
- Resolution by build backend
- Resolution by package name
- Resolution by pkgconfig trigger
- Resolution by library trigger
- Forced / excluded extensions
- Multi-package deduplication
- ExtensionResolutionReport
- Pipeline integration
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from pfmr.models import (
    BuildBackend,
    ExtensionMatch,
    ExtensionProfile,
    ExtensionResolutionReport,
    ResolvedPackage,
)
from pfmr.resolvers.sdk_extension import (
    SDKExtensionResolver,
    _parse_extension_profile,
    load_extension_profiles,
    _BUILTIN_EXTENSION_PROFILES_DIR,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ext_dir(tmp_path: Path) -> Path:
    """Custom extension profiles directory with two profiles."""
    d = tmp_path / "ext-profiles"
    d.mkdir()

    (d / "rust-test.toml").write_text(textwrap.dedent("""\
        extension_id = "org.test.Sdk.Extension.rust"
        display_name = "Rust test"
        build_backends = ["maturin", "setuptools-rust"]
        package_triggers = ["orjson", "pydantic-core"]
        pkgconfig_triggers = []
        library_triggers = []
        provides_executables = ["rustc", "cargo"]
        provides_pkgconfig = []
        provides_libraries = []
        mount_path = "/usr/lib/sdk/rust"
        compatible_sdks = ["org.test.Sdk", "org.freedesktop.Sdk"]
        description = "Rust for tests"

        [env]
        PATH = "/usr/lib/sdk/rust/bin:$PATH"
        CARGO_HOME = "/run/build/cargo"
    """))

    (d / "llvm-test.toml").write_text(textwrap.dedent("""\
        extension_id = "org.test.Sdk.Extension.llvm"
        display_name = "LLVM test"
        build_backends = []
        pkgconfig_triggers = ["llvm"]
        library_triggers = ["libLLVM.so"]
        package_triggers = ["llvmlite"]
        provides_executables = ["clang"]
        provides_pkgconfig = ["llvm"]
        provides_libraries = ["libLLVM.so"]
        mount_path = "/usr/lib/sdk/llvm"
        compatible_sdks = ["org.test.Sdk", "org.freedesktop.Sdk"]
        description = "LLVM for tests"

        [env]
        LLVM_CONFIG = "/usr/lib/sdk/llvm/bin/llvm-config"
    """))

    return d


@pytest.fixture
def resolver(ext_dir: Path) -> SDKExtensionResolver:
    return SDKExtensionResolver(extra_profile_dirs=[ext_dir])


def _pkg(name: str, backend: BuildBackend = BuildBackend.UNKNOWN,
         native_deps: list[str] | None = None) -> ResolvedPackage:
    return ResolvedPackage(
        name=name,
        version="1.0",
        build_backend=backend,
        native_deps=native_deps or [],
    )


# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------

class TestProfileLoading:
    def test_parse_profile_fields(self, ext_dir):
        profile = _parse_extension_profile(ext_dir / "rust-test.toml")
        assert profile.extension_id == "org.test.Sdk.Extension.rust"
        assert profile.display_name == "Rust test"
        assert "maturin" in profile.build_backends
        assert "orjson" in profile.package_triggers
        assert profile.mount_path == "/usr/lib/sdk/rust"

    def test_parse_profile_env(self, ext_dir):
        profile = _parse_extension_profile(ext_dir / "rust-test.toml")
        assert "PATH" in profile.env
        assert "CARGO_HOME" in profile.env

    def test_parse_profile_compatible_sdks(self, ext_dir):
        profile = _parse_extension_profile(ext_dir / "rust-test.toml")
        assert "org.freedesktop.Sdk" in profile.compatible_sdks

    def test_load_extension_profiles_from_dir(self, ext_dir):
        profiles = load_extension_profiles(extra_dirs=[ext_dir])
        ids = [p.extension_id for p in profiles]
        assert "org.test.Sdk.Extension.rust" in ids
        assert "org.test.Sdk.Extension.llvm" in ids

    def test_load_deduplicates_same_id(self, ext_dir, tmp_path):
        """A second dir with the same extension_id should not produce duplicates."""
        dup_dir = tmp_path / "dup"
        dup_dir.mkdir()
        (dup_dir / "rust-test.toml").write_text(textwrap.dedent("""\
            extension_id = "org.test.Sdk.Extension.rust"
            display_name = "Rust duplicate"
            build_backends = ["maturin"]
            compatible_sdks = []
        """))
        profiles = load_extension_profiles(extra_dirs=[ext_dir, dup_dir])
        rust_profiles = [p for p in profiles if p.extension_id == "org.test.Sdk.Extension.rust"]
        assert len(rust_profiles) == 1

    def test_builtin_profiles_load(self):
        profiles = load_extension_profiles()
        ids = [p.extension_id for p in profiles]
        assert "org.freedesktop.Sdk.Extension.rust-stable" in ids
        assert "org.freedesktop.Sdk.Extension.llvm18" in ids
        assert "org.freedesktop.Sdk.Extension.openjdk21" in ids

    def test_missing_dir_skipped(self, tmp_path):
        profiles = load_extension_profiles(extra_dirs=[tmp_path / "nonexistent"])
        # Should not crash; built-in profiles still loaded
        assert len(profiles) >= 0

    def test_malformed_toml_skipped(self, tmp_path):
        d = tmp_path / "bad"
        d.mkdir()
        (d / "bad.toml").write_text("this is not valid toml {{{{")
        # Should not crash
        profiles = load_extension_profiles(extra_dirs=[d])
        assert isinstance(profiles, list)


# ---------------------------------------------------------------------------
# Resolution — build backend trigger
# ---------------------------------------------------------------------------

class TestResolutionByBackend:
    def test_maturin_triggers_rust(self, resolver):
        pkgs = [_pkg("mylib", BuildBackend.MATURIN)]
        report = resolver.resolve(pkgs, sdk_id="org.test.Sdk")
        assert report.has_extension("org.test.Sdk.Extension.rust")

    def test_setuptools_rust_triggers_rust(self, resolver):
        pkgs = [_pkg("mylib", BuildBackend.SETUPTOOLS_RUST)]
        report = resolver.resolve(pkgs, sdk_id="org.test.Sdk")
        assert report.has_extension("org.test.Sdk.Extension.rust")

    def test_pure_backend_no_extension(self, resolver):
        pkgs = [_pkg("requests", BuildBackend.SETUPTOOLS)]
        report = resolver.resolve(pkgs, sdk_id="org.test.Sdk")
        assert not report.has_extension("org.test.Sdk.Extension.rust")

    def test_unknown_backend_no_extension(self, resolver):
        pkgs = [_pkg("something", BuildBackend.UNKNOWN)]
        report = resolver.resolve(pkgs, sdk_id="org.test.Sdk")
        assert len(report.required_extensions) == 0

    def test_reason_type_is_build_backend(self, resolver):
        pkgs = [_pkg("mylib", BuildBackend.MATURIN)]
        report = resolver.resolve(pkgs, sdk_id="org.test.Sdk")
        match = report.required_extensions[0]
        reason_types = [r[0] for r in match.reasons]
        assert "build_backend" in reason_types

    def test_triggered_by_package_name_recorded(self, resolver):
        pkgs = [_pkg("mylib", BuildBackend.MATURIN)]
        report = resolver.resolve(pkgs, sdk_id="org.test.Sdk")
        match = report.required_extensions[0]
        assert "mylib" in match.triggered_by_packages


# ---------------------------------------------------------------------------
# Resolution — package name trigger
# ---------------------------------------------------------------------------

class TestResolutionByPackage:
    def test_orjson_triggers_rust(self, resolver):
        pkgs = [_pkg("orjson")]
        report = resolver.resolve(pkgs, sdk_id="org.test.Sdk")
        assert report.has_extension("org.test.Sdk.Extension.rust")

    def test_pydantic_core_triggers_rust(self, resolver):
        pkgs = [_pkg("pydantic-core")]
        report = resolver.resolve(pkgs, sdk_id="org.test.Sdk")
        assert report.has_extension("org.test.Sdk.Extension.rust")

    def test_llvmlite_triggers_llvm(self, resolver):
        pkgs = [_pkg("llvmlite")]
        report = resolver.resolve(pkgs, sdk_id="org.test.Sdk")
        assert report.has_extension("org.test.Sdk.Extension.llvm")

    def test_reason_type_is_package(self, resolver):
        pkgs = [_pkg("orjson")]
        report = resolver.resolve(pkgs, sdk_id="org.test.Sdk")
        match = report.required_extensions[0]
        reason_types = [r[0] for r in match.reasons]
        assert "package" in reason_types

    def test_canonical_name_matching(self, resolver):
        """pydantic_core (underscore) == pydantic-core (hyphen)."""
        pkgs = [_pkg("pydantic_core")]
        report = resolver.resolve(pkgs, sdk_id="org.test.Sdk")
        assert report.has_extension("org.test.Sdk.Extension.rust")


# ---------------------------------------------------------------------------
# Resolution — pkgconfig trigger
# ---------------------------------------------------------------------------

class TestResolutionByPkgconfig:
    def test_llvm_pkgconfig_triggers_llvm_ext(self, resolver):
        pkgs = [_pkg("some-pkg", native_deps=["llvm"])]
        report = resolver.resolve(pkgs, sdk_id="org.test.Sdk")
        assert report.has_extension("org.test.Sdk.Extension.llvm")

    def test_unrelated_pkgconfig_no_trigger(self, resolver):
        pkgs = [_pkg("some-pkg", native_deps=["openssl", "zlib"])]
        report = resolver.resolve(pkgs, sdk_id="org.test.Sdk")
        assert not report.has_extension("org.test.Sdk.Extension.rust")

    def test_reason_type_is_pkgconfig(self, resolver):
        pkgs = [_pkg("some-pkg", native_deps=["llvm"])]
        report = resolver.resolve(pkgs, sdk_id="org.test.Sdk")
        match = report.required_extensions[0]
        reason_types = [r[0] for r in match.reasons]
        assert "pkgconfig" in reason_types


# ---------------------------------------------------------------------------
# Resolution — library trigger
# ---------------------------------------------------------------------------

class TestResolutionByLibrary:
    def test_libllvm_so_triggers_llvm_ext(self, resolver):
        pkgs = [_pkg("numba", native_deps=["libLLVM.so"])]
        report = resolver.resolve(pkgs, sdk_id="org.test.Sdk")
        assert report.has_extension("org.test.Sdk.Extension.llvm")

    def test_versioned_soname_triggers(self, resolver):
        """libLLVM.so.18 should match trigger libLLVM.so via base name."""
        pkgs = [_pkg("numba", native_deps=["libLLVM.so.18"])]
        report = resolver.resolve(pkgs, sdk_id="org.test.Sdk")
        assert report.has_extension("org.test.Sdk.Extension.llvm")

    def test_reason_type_is_library(self, resolver):
        pkgs = [_pkg("numba", native_deps=["libLLVM.so"])]
        report = resolver.resolve(pkgs, sdk_id="org.test.Sdk")
        match = report.required_extensions[0]
        reason_types = [r[0] for r in match.reasons]
        assert "library" in reason_types


# ---------------------------------------------------------------------------
# Forced / excluded extensions
# ---------------------------------------------------------------------------

class TestForcedExcluded:
    def test_forced_extension_always_included(self, ext_dir):
        resolver = SDKExtensionResolver(
            extra_profile_dirs=[ext_dir],
            forced_extensions=["org.test.Sdk.Extension.rust"],
        )
        pkgs = [_pkg("requests")]  # would not normally trigger rust
        report = resolver.resolve(pkgs, sdk_id="org.test.Sdk")
        assert report.has_extension("org.test.Sdk.Extension.rust")

    def test_forced_reason_is_forced(self, ext_dir):
        resolver = SDKExtensionResolver(
            extra_profile_dirs=[ext_dir],
            forced_extensions=["org.test.Sdk.Extension.rust"],
        )
        pkgs = [_pkg("requests")]
        report = resolver.resolve(pkgs, sdk_id="org.test.Sdk")
        match = report.required_extensions[0]
        reason_types = [r[0] for r in match.reasons]
        assert "forced" in reason_types

    def test_excluded_extension_never_included(self, ext_dir):
        resolver = SDKExtensionResolver(
            extra_profile_dirs=[ext_dir],
            excluded_extensions=["org.test.Sdk.Extension.rust"],
        )
        pkgs = [_pkg("orjson")]  # would trigger rust
        report = resolver.resolve(pkgs, sdk_id="org.test.Sdk")
        assert not report.has_extension("org.test.Sdk.Extension.rust")

    def test_excluded_overrides_forced(self, ext_dir):
        """excluded takes priority over forced for the same extension."""
        resolver = SDKExtensionResolver(
            extra_profile_dirs=[ext_dir],
            forced_extensions=["org.test.Sdk.Extension.rust"],
            excluded_extensions=["org.test.Sdk.Extension.rust"],
        )
        pkgs = [_pkg("orjson")]
        report = resolver.resolve(pkgs, sdk_id="org.test.Sdk")
        assert not report.has_extension("org.test.Sdk.Extension.rust")


# ---------------------------------------------------------------------------
# SDK compatibility filter
# ---------------------------------------------------------------------------

class TestSDKCompatibility:
    def test_incompatible_sdk_skipped(self, ext_dir):
        """Profiles with compatible_sdks that doesn't include the target SDK are skipped."""
        resolver = SDKExtensionResolver(extra_profile_dirs=[ext_dir])
        pkgs = [_pkg("orjson")]
        # org.unrelated.Sdk is not in compatible_sdks
        report = resolver.resolve(pkgs, sdk_id="org.unrelated.Sdk")
        assert not report.has_extension("org.test.Sdk.Extension.rust")

    def test_empty_compatible_sdks_means_any(self, tmp_path):
        """A profile with no compatible_sdks list matches any SDK."""
        d = tmp_path / "ext"
        d.mkdir()
        (d / "any.toml").write_text(textwrap.dedent("""\
            extension_id = "org.any.Extension.foo"
            display_name = "Foo"
            package_triggers = ["foobar"]
            build_backends = []
            compatible_sdks = []
        """))
        resolver = SDKExtensionResolver(extra_profile_dirs=[d])
        pkgs = [_pkg("foobar")]
        report = resolver.resolve(pkgs, sdk_id="org.whatever.Sdk")
        assert report.has_extension("org.any.Extension.foo")


# ---------------------------------------------------------------------------
# Multi-package deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_same_extension_triggered_by_two_packages_once(self, resolver):
        pkgs = [_pkg("orjson"), _pkg("pydantic-core")]
        report = resolver.resolve(pkgs, sdk_id="org.test.Sdk")
        rust_matches = [m for m in report.required_extensions
                        if m.extension_id == "org.test.Sdk.Extension.rust"]
        assert len(rust_matches) == 1

    def test_merged_triggered_packages_list(self, resolver):
        pkgs = [_pkg("orjson"), _pkg("pydantic-core")]
        report = resolver.resolve(pkgs, sdk_id="org.test.Sdk")
        rust_match = next(m for m in report.required_extensions
                          if m.extension_id == "org.test.Sdk.Extension.rust")
        assert "orjson" in rust_match.triggered_by_packages
        assert "pydantic-core" in rust_match.triggered_by_packages

    def test_multiple_extensions_all_included(self, resolver):
        pkgs = [_pkg("orjson"), _pkg("llvmlite")]
        report = resolver.resolve(pkgs, sdk_id="org.test.Sdk")
        assert report.has_extension("org.test.Sdk.Extension.rust")
        assert report.has_extension("org.test.Sdk.Extension.llvm")

    def test_reason_deduplication(self, resolver):
        """Same reason (type, value) must appear only once per match."""
        pkgs = [_pkg("orjson", BuildBackend.MATURIN)]
        report = resolver.resolve(pkgs, sdk_id="org.test.Sdk")
        rust_match = next(m for m in report.required_extensions
                          if m.extension_id == "org.test.Sdk.Extension.rust")
        seen = set()
        for r in rust_match.reasons:
            assert r not in seen, f"Duplicate reason: {r}"
            seen.add(r)


# ---------------------------------------------------------------------------
# ExtensionResolutionReport
# ---------------------------------------------------------------------------

class TestExtensionResolutionReport:
    def test_extension_ids_list(self, resolver):
        pkgs = [_pkg("orjson"), _pkg("llvmlite")]
        report = resolver.resolve(pkgs, sdk_id="org.test.Sdk")
        assert isinstance(report.extension_ids, list)
        assert len(report.extension_ids) == len(report.required_extensions)

    def test_has_extension_true(self, resolver):
        pkgs = [_pkg("orjson")]
        report = resolver.resolve(pkgs, sdk_id="org.test.Sdk")
        assert report.has_extension("org.test.Sdk.Extension.rust")

    def test_has_extension_false(self, resolver):
        pkgs = [_pkg("requests")]
        report = resolver.resolve(pkgs, sdk_id="org.test.Sdk")
        assert not report.has_extension("org.test.Sdk.Extension.rust")

    def test_empty_packages(self, resolver):
        report = resolver.resolve([], sdk_id="org.test.Sdk")
        assert len(report.required_extensions) == 0

    def test_env_in_match(self, resolver):
        pkgs = [_pkg("orjson")]
        report = resolver.resolve(pkgs, sdk_id="org.test.Sdk")
        rust_match = next(m for m in report.required_extensions
                          if m.extension_id == "org.test.Sdk.Extension.rust")
        assert "PATH" in rust_match.env
        assert "CARGO_HOME" in rust_match.env


# ---------------------------------------------------------------------------
# Built-in profiles spot-checks
# ---------------------------------------------------------------------------

class TestBuiltinExtensionProfiles:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.resolver = SDKExtensionResolver()

    def test_cryptography_needs_rust(self):
        pkgs = [_pkg("cryptography")]
        report = self.resolver.resolve(pkgs, sdk_id="org.freedesktop.Sdk")
        assert report.has_extension("org.freedesktop.Sdk.Extension.rust-stable")

    def test_maturin_package_needs_rust(self):
        pkgs = [_pkg("mylib", BuildBackend.MATURIN)]
        report = self.resolver.resolve(pkgs, sdk_id="org.freedesktop.Sdk")
        assert report.has_extension("org.freedesktop.Sdk.Extension.rust-stable")

    def test_llvmlite_needs_llvm(self):
        pkgs = [_pkg("llvmlite", native_deps=["llvm"])]
        report = self.resolver.resolve(pkgs, sdk_id="org.freedesktop.Sdk")
        assert report.has_extension("org.freedesktop.Sdk.Extension.llvm18")

    def test_pyjnius_needs_openjdk(self):
        pkgs = [_pkg("pyjnius", native_deps=["libjvm.so"])]
        report = self.resolver.resolve(pkgs, sdk_id="org.freedesktop.Sdk")
        assert report.has_extension("org.freedesktop.Sdk.Extension.openjdk21")

    def test_scipy_needs_gfortran(self):
        pkgs = [_pkg("scipy", native_deps=["libgfortran.so"])]
        report = self.resolver.resolve(pkgs, sdk_id="org.freedesktop.Sdk")
        assert report.has_extension("org.freedesktop.Sdk.Extension.gfortran")

    def test_requests_needs_nothing(self):
        pkgs = [_pkg("requests")]
        report = self.resolver.resolve(pkgs, sdk_id="org.freedesktop.Sdk")
        assert len(report.required_extensions) == 0

    def test_rust_env_has_path(self):
        pkgs = [_pkg("orjson")]
        report = self.resolver.resolve(pkgs, sdk_id="org.freedesktop.Sdk")
        rust_match = next(
            (m for m in report.required_extensions
             if "rust" in m.extension_id),
            None,
        )
        assert rust_match is not None
        assert "PATH" in rust_match.env


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

class TestPipelineExtensionIntegration:
    def test_extensions_populated_in_result(self):
        from pfmr.pipeline import Pipeline
        from pfmr.models import ResolutionResult, ResolvedPackage

        pipeline = Pipeline.__new__(Pipeline)
        pipeline.sdk_id = "org.freedesktop.Sdk"
        pipeline.sdk_version = "24.08"
        pipeline.ext_resolver = SDKExtensionResolver()

        result = ResolutionResult(
            packages=[
                ResolvedPackage(
                    name="cryptography", version="43.0.0",
                    build_backend=BuildBackend.MATURIN,
                    requires_native=True,
                ),
            ]
        )
        result = pipeline._resolve_extensions(result)
        assert "org.freedesktop.Sdk.Extension.rust-stable" in result.required_extensions

    def test_extension_matches_populated(self):
        from pfmr.pipeline import Pipeline
        from pfmr.models import ResolutionResult, ResolvedPackage

        pipeline = Pipeline.__new__(Pipeline)
        pipeline.sdk_id = "org.freedesktop.Sdk"
        pipeline.sdk_version = "24.08"
        pipeline.ext_resolver = SDKExtensionResolver()

        result = ResolutionResult(
            packages=[
                ResolvedPackage(
                    name="orjson", version="3.9.0",
                    build_backend=BuildBackend.MATURIN,
                ),
            ]
        )
        result = pipeline._resolve_extensions(result)
        assert len(result.extension_matches) > 0
        assert result.extension_matches[0].env  # env must be populated

    def test_no_native_packages_no_extensions(self):
        from pfmr.pipeline import Pipeline
        from pfmr.models import ResolutionResult, ResolvedPackage

        pipeline = Pipeline.__new__(Pipeline)
        pipeline.sdk_id = "org.freedesktop.Sdk"
        pipeline.sdk_version = "24.08"
        pipeline.ext_resolver = SDKExtensionResolver()

        result = ResolutionResult(
            packages=[
                ResolvedPackage(name="requests", version="2.31.0",
                                build_backend=BuildBackend.SETUPTOOLS),
                ResolvedPackage(name="urllib3", version="2.0.7",
                                build_backend=BuildBackend.HATCH),
            ]
        )
        result = pipeline._resolve_extensions(result)
        assert len(result.required_extensions) == 0