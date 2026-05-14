"""Tests for pfmr Phase 1 — models, recipe DB, manifest generator."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from pfmr.models import (
    BuildBackend,
    FlatpakSource,
    NativeRecipe,
    ResolutionResult,
    ResolvedPackage,
    SourceType,
)
from pfmr.recipes.db import RecipeDB
from pfmr.generators.manifest import ManifestGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def recipe_dir(tmp_path: Path) -> Path:
    d = tmp_path / "recipes" / "native"
    d.mkdir(parents=True)
    (d / "libfoo.yaml").write_text("""
id: libfoo
provides:
  - libfoo.so
  - libfoo.so.1
pkgconfig:
  - foo
headers:
  - foo.h
aliases:
  - foo
buildsystem: autotools
source:
  type: archive
  url: https://example.com/libfoo-1.0.tar.gz
  sha256: aaaa
config-opts:
  - --disable-static
cleanup:
  - /include
  - /lib/pkgconfig
""")
    (d / "libbar.yaml").write_text("""
id: libbar
provides:
  - libbar.so.2
pkgconfig:
  - bar
buildsystem: cmake
source:
  type: archive
  url: https://example.com/libbar-2.0.tar.gz
  sha256: bbbb
cleanup:
  - /include
""")
    return d


@pytest.fixture
def sample_result() -> ResolutionResult:
    return ResolutionResult(
        packages=[
            ResolvedPackage(
                name="requests",
                version="2.31.0",
                wheel_available=True,
                build_backend=BuildBackend.SETUPTOOLS,
                requires_native=False,
                is_direct=True,
                source_url="https://files.pythonhosted.org/packages/requests-2.31.0-py3-none-any.whl",
                source_hash="abc123",
                source_type=SourceType.WHEEL,
            ),
            ResolvedPackage(
                name="urllib3",
                version="2.0.7",
                wheel_available=True,
                build_backend=BuildBackend.HATCH,
                requires_native=False,
                source_url="https://files.pythonhosted.org/packages/urllib3-2.0.7-py3-none-any.whl",
                source_hash="def456",
                source_type=SourceType.WHEEL,
            ),
            ResolvedPackage(
                name="cryptography",
                version="43.0.0",
                wheel_available=False,
                build_backend=BuildBackend.MATURIN,
                requires_native=True,
                is_direct=True,
                source_url="https://files.pythonhosted.org/packages/cryptography-43.0.0.tar.gz",
                source_hash="xyz789",
                source_type=SourceType.SDIST,
            ),
        ],
        lockfile_hash="deadbeef",
    )


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestResolvedPackage:
    def test_defaults(self):
        pkg = ResolvedPackage(name="foo", version="1.0")
        assert not pkg.wheel_available
        assert pkg.build_backend == BuildBackend.UNKNOWN
        assert not pkg.requires_native
        assert pkg.native_deps == []

    def test_native_flag(self):
        pkg = ResolvedPackage(name="numpy", version="1.26", requires_native=True)
        assert pkg.requires_native


# ---------------------------------------------------------------------------
# RecipeDB tests
# ---------------------------------------------------------------------------

class TestRecipeDB:
    def test_load_recipes(self, recipe_dir):
        db = RecipeDB(recipe_dirs=[recipe_dir])
        assert len(db) == 2

    def test_find_by_id(self, recipe_dir):
        db = RecipeDB(recipe_dirs=[recipe_dir])
        r = db.find_by_id("libfoo")
        assert r is not None
        assert r.id == "libfoo"

    def test_find_by_soname(self, recipe_dir):
        db = RecipeDB(recipe_dirs=[recipe_dir])
        r = db.find_by_soname("libfoo.so")
        assert r is not None
        assert r.id == "libfoo"

    def test_find_by_soname_versioned(self, recipe_dir):
        db = RecipeDB(recipe_dirs=[recipe_dir])
        r = db.find_by_soname("libbar.so.2")
        assert r is not None
        assert r.id == "libbar"

    def test_find_by_pkgconfig(self, recipe_dir):
        db = RecipeDB(recipe_dirs=[recipe_dir])
        r = db.find_by_pkgconfig("foo")
        assert r is not None
        assert r.id == "libfoo"

    def test_find_by_alias(self, recipe_dir):
        db = RecipeDB(recipe_dirs=[recipe_dir])
        r = db.find_by_alias("foo")
        assert r is not None
        assert r.id == "libfoo"

    def test_find_universal(self, recipe_dir):
        db = RecipeDB(recipe_dirs=[recipe_dir])
        assert db.find("libfoo.so") is not None
        assert db.find("foo") is not None
        assert db.find("bar") is not None
        assert db.find("nonexistent") is None

    def test_empty_dirs(self, tmp_path):
        db = RecipeDB(recipe_dirs=[tmp_path / "missing"])
        assert len(db) == 0

    def test_source_parsed(self, recipe_dir):
        db = RecipeDB(recipe_dirs=[recipe_dir])
        r = db.find_by_id("libfoo")
        assert r.source is not None
        assert r.source.url == "https://example.com/libfoo-1.0.tar.gz"
        assert r.source.sha256 == "aaaa"

    def test_config_opts_parsed(self, recipe_dir):
        db = RecipeDB(recipe_dirs=[recipe_dir])
        r = db.find_by_id("libfoo")
        assert "--disable-static" in r.config_opts


# ---------------------------------------------------------------------------
# ManifestGenerator tests
# ---------------------------------------------------------------------------

class TestManifestGenerator:
    def test_basic_generation(self, sample_result):
        gen = ManifestGenerator(app_id="org.test.App")
        manifest = gen.generate(sample_result)
        assert manifest.app_id == "org.test.App"
        assert len(manifest.modules) >= 2  # venv + at least one pip module

    def test_rust_extension_added(self, sample_result):
        # Extensions now come from SDKExtensionResolver via result.required_extensions.
        # Pre-populate it to simulate the pipeline having run the extension resolver.
        sample_result.required_extensions = ["org.freedesktop.Sdk.Extension.rust-stable"]
        gen = ManifestGenerator()
        manifest = gen.generate(sample_result)
        assert "org.freedesktop.Sdk.Extension.rust-stable" in manifest.sdk_extensions

    def test_yaml_serialisation(self, sample_result):
        gen = ManifestGenerator(app_id="org.test.App")
        manifest = gen.generate(sample_result)
        text = gen.to_yaml(manifest)
        data = yaml.safe_load(text)
        assert data["app-id"] == "org.test.App"
        assert "modules" in data
        assert isinstance(data["modules"], list)

    def test_json_serialisation(self, sample_result):
        gen = ManifestGenerator(app_id="org.test.App")
        manifest = gen.generate(sample_result)
        text = gen.to_json(manifest)
        data = json.loads(text)
        assert data["app-id"] == "org.test.App"

    def test_native_module_separate(self, sample_result):
        gen = ManifestGenerator()
        manifest = gen.generate(sample_result)
        names = [m.name for m in manifest.modules]
        # cryptography is native → gets its own module
        assert any("cryptography" in n for n in names)

    def test_pure_python_batched(self, sample_result):
        gen = ManifestGenerator()
        manifest = gen.generate(sample_result)
        names = [m.name for m in manifest.modules]
        assert "python-pure-deps" in names

    def test_venv_setup_module(self, sample_result):
        gen = ManifestGenerator()
        manifest = gen.generate(sample_result)
        names = [m.name for m in manifest.modules]
        assert "python-venv-setup" in names

    def test_write_yaml_to_file(self, sample_result, tmp_path):
        gen = ManifestGenerator()
        manifest = gen.generate(sample_result)
        out = tmp_path / "test.yaml"
        gen.to_yaml(manifest, out)
        assert out.exists()
        data = yaml.safe_load(out.read_text())
        assert "modules" in data

    def test_native_recipe_module_prepended(self, sample_result, recipe_dir):
        db = RecipeDB(recipe_dirs=[recipe_dir])
        # Inject a fake native recipe into result
        recipe = db.find_by_id("libfoo")
        sample_result.native_recipes.append(recipe)

        gen = ManifestGenerator()
        manifest = gen.generate(sample_result)
        # libfoo module must come before python modules
        names = [m.name for m in manifest.modules]
        assert names.index("libfoo") < names.index("python-venv-setup")

    def test_module_sources_have_sha256(self, sample_result):
        gen = ManifestGenerator()
        manifest = gen.generate(sample_result)
        for mod in manifest.modules:
            for src in mod.sources:
                if src.type == "file":
                    assert src.sha256, f"Missing sha256 on source in module {mod.name}"