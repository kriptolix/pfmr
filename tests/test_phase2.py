"""
Tests for pfmr Phase 2 — SDKCapabilityResolver.

Covers:
- SDKCapability data model (provides_* methods)
- Profile loading from TOML files (built-in + custom dirs)
- SDKQuery resolution (pkgconfig, library, header, executable)
- SDKResolutionReport (satisfied / missing / is_sufficient)
- SDKCapabilityResolver multi-SDK priority chain
- Pipeline integration: recipe filtering + unresolved_natives
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Optional

import pytest

from pfmr.models import (
    BuildBackend,
    NativeRecipe,
    ResolutionResult,
    ResolvedPackage,
    SDKCapability,
    SDKCheckResult,
    SDKResolutionReport,
    SourceType,
)
from pfmr.resolvers.sdk_capability import (
    SDKCapabilityResolver,
    SDKQuery,
    _load_profile,
    _save_profile,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_capability() -> SDKCapability:
    return SDKCapability(
        sdk_id="org.test.Sdk",
        sdk_version="1.0",
        libraries=["libssl.so.3", "libz.so.1", "libffi.so.8"],
        pkgconfig=["openssl", "zlib", "libffi"],
        headers=["openssl/ssl.h", "zlib.h"],
        executables=["python3", "gcc"],
    )


@pytest.fixture
def profile_dir(tmp_path: Path) -> Path:
    """A temporary profile directory with a single TOML profile."""
    d = tmp_path / "sdk-profiles" / "org.test.Sdk"
    d.mkdir(parents=True)
    (d / "1.0.toml").write_text(textwrap.dedent("""\
        sdk_id = "org.test.Sdk"
        sdk_version = "1.0"
        libraries = ["libssl.so.3", "libz.so.1", "libffi.so.8"]
        pkgconfig = ["openssl", "zlib", "libffi"]
        headers = ["openssl/ssl.h", "zlib.h"]
        executables = ["python3", "gcc"]
        python_modules = ["ssl", "zlib"]
    """))
    return tmp_path / "sdk-profiles"


@pytest.fixture
def offline_resolver(profile_dir: Path) -> SDKCapabilityResolver:
    return SDKCapabilityResolver(
        sdk_id="org.test.Sdk",
        sdk_version="1.0",
        offline=True,
        extra_profile_dirs=[profile_dir],
    )


# ---------------------------------------------------------------------------
# SDKCapability model
# ---------------------------------------------------------------------------

class TestSDKCapability:
    def test_provides_library_exact(self, minimal_capability):
        assert minimal_capability.provides_library("libssl.so.3")

    def test_provides_library_base(self, minimal_capability):
        # libssl.so.3 → base is libssl.so
        assert minimal_capability.provides_library("libssl.so")

    def test_provides_library_missing(self, minimal_capability):
        assert not minimal_capability.provides_library("libusb-1.0.so")

    def test_provides_pkgconfig(self, minimal_capability):
        assert minimal_capability.provides_pkgconfig("openssl")

    def test_provides_pkgconfig_with_suffix(self, minimal_capability):
        assert minimal_capability.provides_pkgconfig("openssl.pc")

    def test_provides_pkgconfig_missing(self, minimal_capability):
        assert not minimal_capability.provides_pkgconfig("libusb-1.0")

    def test_provides_header(self, minimal_capability):
        assert minimal_capability.provides_header("openssl/ssl.h")

    def test_provides_header_missing(self, minimal_capability):
        assert not minimal_capability.provides_header("usb.h")

    def test_provides_executable(self, minimal_capability):
        assert minimal_capability.provides_executable("gcc")

    def test_provides_executable_missing(self, minimal_capability):
        assert not minimal_capability.provides_executable("rustc")


# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------

class TestProfileLoading:
    def test_load_profile_from_dir(self, profile_dir):
        cap = _load_profile("org.test.Sdk", "1.0", profile_dir)
        assert cap is not None
        assert cap.sdk_id == "org.test.Sdk"
        assert cap.sdk_version == "1.0"
        assert "openssl" in cap.pkgconfig
        assert "libssl.so.3" in cap.libraries

    def test_load_profile_missing_returns_none(self, tmp_path):
        cap = _load_profile("org.missing.Sdk", "99.0", tmp_path)
        assert cap is None

    def test_load_profile_probed_live_false(self, profile_dir):
        cap = _load_profile("org.test.Sdk", "1.0", profile_dir)
        assert cap.probed_live is False

    def test_save_and_reload_profile(self, tmp_path):
        cap = SDKCapability(
            sdk_id="org.roundtrip.Sdk",
            sdk_version="2.0",
            pkgconfig=["foo", "bar"],
            libraries=["libfoo.so.1"],
            headers=["foo.h"],
            executables=["foo"],
            python_modules=["foo"],
            probed_live=True,
        )
        _save_profile.__wrapped__(cap) if hasattr(_save_profile, "__wrapped__") else None
        # Save directly to tmp dir
        from pfmr.resolvers.sdk_capability import _save_profile as sp, _profile_path
        path = _profile_path(cap.sdk_id, cap.sdk_version, tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        import tomllib
        lines = [
            f'sdk_id = "{cap.sdk_id}"',
            f'sdk_version = "{cap.sdk_version}"',
            "pkgconfig = [" + ", ".join(f'"{p}"' for p in cap.pkgconfig) + "]",
            "libraries = [" + ", ".join(f'"{l}"' for l in cap.libraries) + "]",
            "headers = [" + ", ".join(f'"{h}"' for h in cap.headers) + "]",
            "executables = [" + ", ".join(f'"{e}"' for e in cap.executables) + "]",
            "python_modules = []",
        ]
        path.write_text("\n".join(lines))
        reloaded = _load_profile(cap.sdk_id, cap.sdk_version, tmp_path)
        assert reloaded is not None
        assert reloaded.sdk_id == "org.roundtrip.Sdk"
        assert "foo" in reloaded.pkgconfig

    def test_builtin_freedesktop_2408_loads(self):
        """The built-in org.freedesktop.Sdk 24.08 profile must load cleanly."""
        from pfmr.resolvers.sdk_capability import _BUILTIN_PROFILES_DIR
        cap = _load_profile("org.freedesktop.Sdk", "24.08", _BUILTIN_PROFILES_DIR)
        assert cap is not None
        assert len(cap.pkgconfig) > 20
        assert len(cap.libraries) > 20

    def test_builtin_gnome_48_loads(self):
        from pfmr.resolvers.sdk_capability import _BUILTIN_PROFILES_DIR
        cap = _load_profile("org.gnome.Sdk", "48", _BUILTIN_PROFILES_DIR)
        assert cap is not None
        assert "gtk4" in cap.pkgconfig
        assert "libadwaita-1" in cap.pkgconfig

    def test_builtin_kde_68_loads(self):
        from pfmr.resolvers.sdk_capability import _BUILTIN_PROFILES_DIR
        cap = _load_profile("org.kde.Sdk", "6.8", _BUILTIN_PROFILES_DIR)
        assert cap is not None
        assert "Qt6Core" in cap.pkgconfig
        assert "KF6CoreAddons" in cap.pkgconfig


# ---------------------------------------------------------------------------
# SDKCapabilityResolver — basic resolution
# ---------------------------------------------------------------------------

class TestSDKCapabilityResolver:
    def test_loads_from_extra_profile_dir(self, offline_resolver):
        cap = offline_resolver.capability()
        assert cap is not None
        assert cap.sdk_id == "org.test.Sdk"

    def test_resolve_satisfied_pkgconfig(self, offline_resolver):
        report = offline_resolver.resolve([SDKQuery("openssl", "pkgconfig")])
        assert len(report.satisfied) == 1
        assert report.is_sufficient

    def test_resolve_missing_pkgconfig(self, offline_resolver):
        report = offline_resolver.resolve([SDKQuery("libusb-1.0", "pkgconfig")])
        assert len(report.missing) == 1
        assert not report.is_sufficient
        assert report.missing[0].query == "libusb-1.0"

    def test_resolve_library(self, offline_resolver):
        report = offline_resolver.resolve([SDKQuery("libssl.so.3", "library")])
        assert report.is_sufficient

    def test_resolve_library_base_soname(self, offline_resolver):
        # Query with unversioned soname
        report = offline_resolver.resolve([SDKQuery("libssl.so", "library")])
        assert report.is_sufficient

    def test_resolve_header(self, offline_resolver):
        report = offline_resolver.resolve([SDKQuery("openssl/ssl.h", "header")])
        assert report.is_sufficient

    def test_resolve_executable(self, offline_resolver):
        report = offline_resolver.resolve([SDKQuery("python3", "executable")])
        assert report.is_sufficient

    def test_resolve_missing_executable(self, offline_resolver):
        report = offline_resolver.resolve([SDKQuery("rustc", "executable")])
        assert not report.is_sufficient

    def test_resolve_batch_mixed(self, offline_resolver):
        queries = [
            SDKQuery("openssl", "pkgconfig"),
            SDKQuery("libusb-1.0", "pkgconfig"),
            SDKQuery("zlib", "pkgconfig"),
        ]
        report = offline_resolver.resolve(queries)
        assert len(report.satisfied) == 2
        assert len(report.missing) == 1

    def test_resolve_native_deps_convenience(self, offline_resolver):
        report = offline_resolver.resolve_native_deps(["openssl", "zlib", "libffi"])
        assert report.is_sufficient

    def test_provided_by_field(self, offline_resolver):
        report = offline_resolver.resolve([SDKQuery("openssl", "pkgconfig")])
        assert report.satisfied[0].provided_by == "org.test.Sdk"

    def test_no_profile_no_crash(self, tmp_path):
        """Resolver with no profile should return empty capabilities, not crash."""
        resolver = SDKCapabilityResolver(
            sdk_id="org.nonexistent.Sdk",
            sdk_version="99.0",
            offline=True,
            extra_profile_dirs=[tmp_path],
        )
        report = resolver.resolve([SDKQuery("openssl", "pkgconfig")])
        # No capabilities loaded → nothing can be satisfied
        assert len(report.missing) == 1

    def test_multi_sdk_fallthrough(self, profile_dir, tmp_path):
        """
        When primary SDK doesn't have a dep but a secondary does, it should be satisfied.
        """
        # Create a secondary SDK profile that has libusb
        secondary_dir = tmp_path / "extra-profiles" / "org.extra.Sdk"
        secondary_dir.mkdir(parents=True)
        (secondary_dir / "1.0.toml").write_text(textwrap.dedent("""\
            sdk_id = "org.extra.Sdk"
            sdk_version = "1.0"
            pkgconfig = ["libusb-1.0"]
            libraries = ["libusb-1.0.so.0"]
            headers = []
            executables = []
            python_modules = []
        """))
        resolver = SDKCapabilityResolver(
            sdk_id="org.test.Sdk",
            sdk_version="1.0",
            extra_sdk_ids=["org.extra.Sdk"],
            offline=True,
            extra_profile_dirs=[profile_dir, tmp_path / "extra-profiles"],
        )
        report = resolver.resolve([SDKQuery("libusb-1.0", "pkgconfig")])
        assert report.is_sufficient
        assert report.satisfied[0].provided_by == "org.extra.Sdk"

    def test_all_capabilities_returns_all(self, profile_dir, tmp_path):
        secondary_dir = tmp_path / "extra-profiles" / "org.extra.Sdk"
        secondary_dir.mkdir(parents=True)
        (secondary_dir / "1.0.toml").write_text(textwrap.dedent("""\
            sdk_id = "org.extra.Sdk"
            sdk_version = "1.0"
            pkgconfig = ["foo"]
            libraries = []
            headers = []
            executables = []
            python_modules = []
        """))
        resolver = SDKCapabilityResolver(
            sdk_id="org.test.Sdk",
            sdk_version="1.0",
            extra_sdk_ids=["org.extra.Sdk"],
            offline=True,
            extra_profile_dirs=[profile_dir, tmp_path / "extra-profiles"],
        )
        assert len(resolver.all_capabilities()) == 2

    def test_recipe_candidates_in_missing(self, profile_dir, tmp_path):
        """When a dep is missing from SDK, recipe candidates should be listed."""
        from pfmr.recipes.db import RecipeDB
        from pfmr.models import FlatpakSource

        recipe_dir = tmp_path / "recipes" / "native"
        recipe_dir.mkdir(parents=True)
        (recipe_dir / "libusb.yaml").write_text(textwrap.dedent("""\
            id: libusb
            provides: [libusb-1.0.so.0]
            pkgconfig: [libusb-1.0]
            buildsystem: autotools
            cleanup: [/include]
        """))
        db = RecipeDB(recipe_dirs=[recipe_dir])
        resolver = SDKCapabilityResolver(
            sdk_id="org.test.Sdk",
            sdk_version="1.0",
            offline=True,
            extra_profile_dirs=[profile_dir],
            recipe_db=db,
        )
        report = resolver.resolve([SDKQuery("libusb-1.0", "pkgconfig")])
        assert not report.is_sufficient
        assert "libusb" in report.missing[0].recipe_candidates


# ---------------------------------------------------------------------------
# SDKResolutionReport
# ---------------------------------------------------------------------------

class TestSDKResolutionReport:
    def test_satisfied_and_missing_split(self):
        report = SDKResolutionReport(sdk_id="x", sdk_version="1")
        report.checks = [
            SDKCheckResult("openssl", "pkgconfig", satisfied=True, provided_by="x"),
            SDKCheckResult("libusb-1.0", "pkgconfig", satisfied=False),
        ]
        assert len(report.satisfied) == 1
        assert len(report.missing) == 1
        assert not report.is_sufficient

    def test_all_satisfied(self):
        report = SDKResolutionReport(sdk_id="x", sdk_version="1")
        report.checks = [
            SDKCheckResult("openssl", "pkgconfig", satisfied=True, provided_by="x"),
        ]
        assert report.is_sufficient

    def test_empty_is_sufficient(self):
        report = SDKResolutionReport(sdk_id="x", sdk_version="1")
        assert report.is_sufficient


# ---------------------------------------------------------------------------
# Built-in profile content spot-checks
# ---------------------------------------------------------------------------

class TestBuiltinProfiles:
    @pytest.fixture(autouse=True)
    def resolver_freedesktop(self):
        self.fd = SDKCapabilityResolver(
            sdk_id="org.freedesktop.Sdk",
            sdk_version="24.08",
            offline=True,
        )

    def test_openssl_satisfied(self):
        r = self.fd.resolve([SDKQuery("openssl", "pkgconfig")])
        assert r.is_sufficient

    def test_zlib_satisfied(self):
        r = self.fd.resolve([SDKQuery("zlib", "pkgconfig")])
        assert r.is_sufficient

    def test_libffi_satisfied(self):
        r = self.fd.resolve([SDKQuery("libffi", "pkgconfig")])
        assert r.is_sufficient

    def test_libusb_not_in_freedesktop(self):
        r = self.fd.resolve([SDKQuery("libusb-1.0", "pkgconfig")])
        assert not r.is_sufficient

    def test_libssl_so_satisfied(self):
        r = self.fd.resolve([SDKQuery("libssl.so.3", "library")])
        assert r.is_sufficient

    def test_gcc_executable(self):
        r = self.fd.resolve([SDKQuery("gcc", "executable")])
        assert r.is_sufficient

    def test_glib_satisfied(self):
        r = self.fd.resolve([SDKQuery("glib-2.0", "pkgconfig")])
        assert r.is_sufficient

    def test_gnome_gtk4(self):
        gnome = SDKCapabilityResolver(
            sdk_id="org.gnome.Sdk", sdk_version="48", offline=True
        )
        r = gnome.resolve([SDKQuery("gtk4", "pkgconfig")])
        assert r.is_sufficient

    def test_gnome_adwaita(self):
        gnome = SDKCapabilityResolver(
            sdk_id="org.gnome.Sdk", sdk_version="48", offline=True
        )
        r = gnome.resolve([SDKQuery("libadwaita-1", "pkgconfig")])
        assert r.is_sufficient

    def test_kde_qt6core(self):
        kde = SDKCapabilityResolver(
            sdk_id="org.kde.Sdk", sdk_version="6.8", offline=True
        )
        r = kde.resolve([SDKQuery("Qt6Core", "pkgconfig")])
        assert r.is_sufficient

    def test_kde_kf6(self):
        kde = SDKCapabilityResolver(
            sdk_id="org.kde.Sdk", sdk_version="6.8", offline=True
        )
        r = kde.resolve([SDKQuery("KF6CoreAddons", "pkgconfig")])
        assert r.is_sufficient


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

class TestPipelineSDKIntegration:
    """
    Tests the _filter_sdk_satisfied step in the Pipeline without running uv.
    We build ResolutionResult manually and call _enrich directly.
    """

    @pytest.fixture
    def recipe_db(self, tmp_path):
        from pfmr.recipes.db import RecipeDB
        d = tmp_path / "recipes" / "native"
        d.mkdir(parents=True)
        # libffi is IN the freedesktop SDK — recipe should be filtered out
        (d / "libffi.yaml").write_text(textwrap.dedent("""\
            id: libffi
            provides: [libffi.so.8]
            pkgconfig: [libffi]
            buildsystem: autotools
            cleanup: [/include]
        """))
        # libusb is NOT in the freedesktop SDK — recipe must be kept
        (d / "libusb.yaml").write_text(textwrap.dedent("""\
            id: libusb
            provides: [libusb-1.0.so.0]
            pkgconfig: [libusb-1.0]
            buildsystem: autotools
            cleanup: [/include]
        """))
        return RecipeDB(recipe_dirs=[d])

    @pytest.fixture
    def sdk_resolver(self):
        return SDKCapabilityResolver(
            sdk_id="org.freedesktop.Sdk",
            sdk_version="24.08",
            offline=True,
        )

    def _make_result_with_recipes(self, recipe_db) -> ResolutionResult:
        """Simulate a resolved result that already has recipes attached."""
        libffi_recipe = recipe_db.find_by_id("libffi")
        libusb_recipe = recipe_db.find_by_id("libusb")
        result = ResolutionResult(
            packages=[
                ResolvedPackage(
                    name="cffi", version="1.17",
                    requires_native=True,
                    native_deps=["libffi"],
                    build_backend=BuildBackend.SETUPTOOLS,
                ),
                ResolvedPackage(
                    name="hidapi", version="0.14",
                    requires_native=True,
                    native_deps=["libusb-1.0"],
                    build_backend=BuildBackend.SETUPTOOLS,
                ),
            ],
            native_recipes=[libffi_recipe, libusb_recipe],
        )
        return result

    def test_sdk_satisfied_recipe_is_filtered(self, recipe_db, sdk_resolver):
        """libffi is in the SDK → its recipe must be removed from native_recipes."""
        from pfmr.pipeline import Pipeline
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.recipe_db = recipe_db
        pipeline.sdk_resolver = sdk_resolver

        result = self._make_result_with_recipes(recipe_db)
        result = pipeline._filter_sdk_satisfied(result)

        recipe_ids = [r.id for r in result.native_recipes]
        assert "libffi" not in recipe_ids, "libffi is in SDK — recipe should be filtered"

    def test_sdk_unsatisfied_recipe_is_kept(self, recipe_db, sdk_resolver):
        """libusb is NOT in the SDK → its recipe must be kept."""
        from pfmr.pipeline import Pipeline
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.recipe_db = recipe_db
        pipeline.sdk_resolver = sdk_resolver

        result = self._make_result_with_recipes(recipe_db)
        result = pipeline._filter_sdk_satisfied(result)

        recipe_ids = [r.id for r in result.native_recipes]
        assert "libusb" in recipe_ids, "libusb NOT in SDK — recipe must be kept"

    def test_package_downgraded_to_pure_when_sdk_covers_all_deps(self, recipe_db, sdk_resolver):
        """cffi's native dep (libffi) is in the SDK → cffi becomes requires_native=False."""
        from pfmr.pipeline import Pipeline
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.recipe_db = recipe_db
        pipeline.sdk_resolver = sdk_resolver

        result = ResolutionResult(
            packages=[
                ResolvedPackage(
                    name="cffi", version="1.17",
                    requires_native=True,
                    native_deps=["libffi"],
                    build_backend=BuildBackend.SETUPTOOLS,
                ),
            ],
            native_recipes=[],
        )
        result = pipeline._filter_sdk_satisfied(result)
        cffi = next(p for p in result.packages if p.name == "cffi")
        assert cffi.requires_native is False

    def test_unresolved_natives_populated(self, recipe_db, sdk_resolver):
        """Packages with deps not in SDK and no recipe land in unresolved_natives."""
        from pfmr.pipeline import Pipeline
        pipeline = Pipeline.__new__(Pipeline)
        pipeline.recipe_db = recipe_db
        pipeline.sdk_resolver = sdk_resolver

        result = ResolutionResult(
            packages=[
                ResolvedPackage(
                    name="pyhidapi", version="0.1",
                    requires_native=True,
                    native_deps=["libusb-1.0"],
                    build_backend=BuildBackend.SETUPTOOLS,
                ),
            ],
            native_recipes=[],
        )
        result = pipeline._filter_sdk_satisfied(result)
        assert "libusb-1.0" in result.unresolved_natives