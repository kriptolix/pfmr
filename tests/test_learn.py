"""
Tests for pfmr.learn — KnowledgeGraph, ManifestAnalyzer, SandboxLearner, Exporter.

The FlathubMiner is tested with mocked HTTP only (no live network calls).
All tests are fully standalone — no pfmr.pipeline dependency.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from pfmr.learn.graph import KGEdge, KGNode, KnowledgeGraph, Rel
from pfmr.learn.manifest import ManifestAnalyzer, _parse_pip_packages
from pfmr.learn.sandbox import SandboxLearner
from pfmr.learn.exporter import Exporter
from pfmr.models import SandboxError, SandboxErrorType, SandboxProbeReport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_kg(tmp_path: Path) -> KnowledgeGraph:
    return KnowledgeGraph(tmp_path / "knowledge")


def _make_report(
    ran: bool = True,
    errors: list | None = None,
    missing_pkgconfig: list | None = None,
    missing_native_libs: list | None = None,
) -> SandboxProbeReport:
    return SandboxProbeReport(
        ran=ran,
        errors=errors or [],
        missing_pkgconfig=missing_pkgconfig or [],
        missing_native_libs=missing_native_libs or [],
    )


# ===========================================================================
# KnowledgeGraph
# ===========================================================================

class TestKnowledgeGraph:
    def test_add_node_new(self, tmp_path):
        kg = _fresh_kg(tmp_path)
        changed = kg.add_node(KGNode("cryptography", "package", {"build_backend": "maturin"}))
        assert changed
        assert kg.node("cryptography") is not None

    def test_add_node_merge_attrs(self, tmp_path):
        kg = _fresh_kg(tmp_path)
        kg.add_node(KGNode("pkg", "package", {"a": 1}))
        kg.add_node(KGNode("pkg", "package", {"b": 2}))  # merge
        node = kg.node("pkg")
        assert node.attrs["a"] == 1
        assert node.attrs["b"] == 2

    def test_add_node_no_overwrite_existing(self, tmp_path):
        kg = _fresh_kg(tmp_path)
        kg.add_node(KGNode("pkg", "package", {"confirmed": True}))
        changed = kg.add_node(KGNode("pkg", "package", {"confirmed": True}))
        assert not changed

    def test_add_edge_new(self, tmp_path):
        kg = _fresh_kg(tmp_path)
        edge = KGEdge("cryptography", "openssl", Rel.REQUIRES_PKGCONFIG, confidence=0.8)
        assert kg.add_edge(edge)
        assert len(kg.edges_from("cryptography")) == 1

    def test_add_edge_dedup(self, tmp_path):
        kg = _fresh_kg(tmp_path)
        kg.add_edge(KGEdge("a", "b", "rel", 0.5))
        kg.add_edge(KGEdge("a", "b", "rel", 0.5))
        assert len(kg.edges_from("a")) == 1

    def test_add_edge_upgrades_confidence(self, tmp_path):
        kg = _fresh_kg(tmp_path)
        kg.add_edge(KGEdge("a", "b", "rel", confidence=0.5))
        kg.add_edge(KGEdge("a", "b", "rel", confidence=1.0))
        edge = kg.edges_from("a")[0]
        assert edge.confidence == 1.0

    def test_add_edge_does_not_downgrade(self, tmp_path):
        kg = _fresh_kg(tmp_path)
        kg.add_edge(KGEdge("a", "b", "rel", confidence=1.0))
        kg.add_edge(KGEdge("a", "b", "rel", confidence=0.3))
        edge = kg.edges_from("a")[0]
        assert edge.confidence == 1.0

    def test_edges_from_filter_relation(self, tmp_path):
        kg = _fresh_kg(tmp_path)
        kg.add_edge(KGEdge("pkg", "openssl", Rel.REQUIRES_PKGCONFIG))
        kg.add_edge(KGEdge("pkg", "rust-ext", Rel.REQUIRES_EXTENSION))
        pc_edges = kg.edges_from("pkg", relation=Rel.REQUIRES_PKGCONFIG)
        assert len(pc_edges) == 1
        assert pc_edges[0].to_id == "openssl"

    def test_edges_to(self, tmp_path):
        kg = _fresh_kg(tmp_path)
        kg.add_edge(KGEdge("pkg1", "openssl", Rel.REQUIRES_PKGCONFIG))
        kg.add_edge(KGEdge("pkg2", "openssl", Rel.REQUIRES_PKGCONFIG))
        edges = kg.edges_to("openssl")
        assert len(edges) == 2

    def test_requires_returns_all_dep_ids(self, tmp_path):
        kg = _fresh_kg(tmp_path)
        kg.add_edge(KGEdge("cryptography", "openssl", Rel.REQUIRES_PKGCONFIG))
        kg.add_edge(KGEdge("cryptography", "libffi", Rel.REQUIRES_PKGCONFIG))
        deps = kg.requires("cryptography")
        assert "openssl" in deps
        assert "libffi" in deps

    def test_what_provides(self, tmp_path):
        kg = _fresh_kg(tmp_path)
        kg.add_edge(KGEdge("org.freedesktop.Sdk//24.08", "openssl", Rel.PROVIDES_PKGCONFIG))
        providers = kg.what_provides("openssl")
        assert "org.freedesktop.Sdk//24.08" in providers

    def test_nodes_of_type(self, tmp_path):
        kg = _fresh_kg(tmp_path)
        kg.add_node(KGNode("pkg1", "package"))
        kg.add_node(KGNode("pkg2", "package"))
        kg.add_node(KGNode("lib1", "library"))
        assert len(kg.nodes_of_type("package")) == 2
        assert len(kg.nodes_of_type("library")) == 1

    def test_stats(self, tmp_path):
        kg = _fresh_kg(tmp_path)
        kg.add_node(KGNode("pkg", "package"))
        kg.add_node(KGNode("lib", "library"))
        kg.add_edge(KGEdge("pkg", "lib", Rel.REQUIRES_PKGCONFIG))
        stats = kg.stats()
        assert stats["total_nodes"] == 2
        assert stats["total_edges"] == 1
        assert stats["nodes_by_type"]["package"] == 1

    def test_save_and_reload(self, tmp_path):
        kg = _fresh_kg(tmp_path)
        kg.add_node(KGNode("cryptography", "package", {"build_backend": "maturin"}))
        kg.add_edge(KGEdge("cryptography", "openssl", Rel.REQUIRES_PKGCONFIG, confidence=1.0))
        kg.save()

        # Reload
        kg2 = KnowledgeGraph(tmp_path / "knowledge")
        assert kg2.node("cryptography") is not None
        assert kg2.node("cryptography").attrs["build_backend"] == "maturin"
        edges = kg2.edges_from("cryptography", Rel.REQUIRES_PKGCONFIG)
        assert len(edges) == 1
        assert edges[0].confidence == 1.0

    def test_save_creates_dirs(self, tmp_path):
        kg = KnowledgeGraph(tmp_path / "deep" / "nested" / "knowledge")
        kg.add_node(KGNode("x", "package"))
        kg.save()
        assert (tmp_path / "deep" / "nested" / "knowledge" / "nodes" / "packages.toml").exists()

    def test_reload_empty_dir(self, tmp_path):
        """Loading from a non-existent dir should not crash."""
        kg = KnowledgeGraph(tmp_path / "nonexistent")
        assert len(kg._nodes) == 0

    def test_edge_dedup_across_save_reload(self, tmp_path):
        kg = _fresh_kg(tmp_path)
        kg.add_edge(KGEdge("a", "b", "rel", 0.8))
        kg.save()
        kg2 = KnowledgeGraph(tmp_path / "knowledge")
        kg2.add_edge(KGEdge("a", "b", "rel", 0.8))  # same edge again
        assert len(kg2.edges_from("a")) == 1

    def test_len(self, tmp_path):
        kg = _fresh_kg(tmp_path)
        kg.add_node(KGNode("x", "package"))
        kg.add_edge(KGEdge("x", "y", "rel"))
        assert len(kg) == 2  # 1 node + 1 edge


# ===========================================================================
# ManifestAnalyzer
# ===========================================================================

class TestManifestAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return ManifestAnalyzer()

    def test_analyze_json_manifest(self, analyzer, tmp_path):
        manifest = {
            "app-id": "org.test.App",
            "runtime": "org.freedesktop.Platform",
            "runtime-version": "24.08",
            "sdk": "org.freedesktop.Sdk",
            "modules": [
                {
                    "name": "python-deps",
                    "buildsystem": "simple",
                    "build-commands": [
                        "pip install requests cryptography"
                    ],
                }
            ],
        }
        p = tmp_path / "org.test.App.json"
        p.write_text(json.dumps(manifest))
        analysis = analyzer.analyze(p)
        assert analysis is not None
        assert analysis.app_id == "org.test.App"
        assert "requests" in analysis.python_packages
        assert "cryptography" in analysis.python_packages

    def test_analyze_yaml_manifest(self, analyzer, tmp_path):
        manifest = {
            "app-id": "org.test.App",
            "runtime": "org.gnome.Platform",
            "runtime-version": "48",
            "sdk": "org.gnome.Sdk",
            "modules": [],
        }
        p = tmp_path / "org.test.App.yaml"
        p.write_text(yaml.dump(manifest))
        analysis = analyzer.analyze(p)
        assert analysis is not None
        assert analysis.sdk_version == "48"

    def test_sdk_extensions_captured(self, analyzer, tmp_path):
        manifest = {
            "app-id": "org.test.App",
            "runtime": "org.freedesktop.Platform",
            "runtime-version": "24.08",
            "sdk": "org.freedesktop.Sdk",
            "sdk-extensions": ["org.freedesktop.Sdk.Extension.rust-stable"],
            "modules": [],
        }
        p = tmp_path / "app.json"
        p.write_text(json.dumps(manifest))
        analysis = analyzer.analyze(p)
        assert "org.freedesktop.Sdk.Extension.rust-stable" in analysis.sdk_extensions

    def test_native_module_extracted(self, analyzer, tmp_path):
        manifest = {
            "app-id": "org.test.App",
            "runtime": "org.freedesktop.Platform",
            "runtime-version": "24.08",
            "sdk": "org.freedesktop.Sdk",
            "modules": [
                {
                    "name": "libusb",
                    "buildsystem": "autotools",
                    "sources": [{"type": "archive", "url": "https://example.com/libusb.tar.bz2", "sha256": "abc"}],
                }
            ],
        }
        p = tmp_path / "app.json"
        p.write_text(json.dumps(manifest))
        analysis = analyzer.analyze(p)
        assert len(analysis.native_modules) == 1
        assert analysis.native_modules[0].module_name == "libusb"
        assert analysis.native_modules[0].source_url == "https://example.com/libusb.tar.bz2"

    def test_invalid_manifest_returns_none(self, analyzer, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("this is not json {{{")
        analysis = analyzer.analyze(p)
        assert analysis is None

    def test_uv_pip_install_parsed(self, analyzer):
        cmds = ["uv pip install requests cryptography==43.0.0"]
        from pfmr.learn.manifest import _parse_pip_packages
        pkgs = _parse_pip_packages(cmds[0])
        assert "requests" in pkgs
        assert "cryptography" in pkgs

    def test_pip3_install_parsed(self):
        from pfmr.learn.manifest import _parse_pip_packages
        pkgs = _parse_pip_packages("pip3 install numpy scipy pandas")
        assert "numpy" in pkgs
        assert "scipy" in pkgs
        assert "pandas" in pkgs

    def test_flags_not_captured_as_packages(self):
        from pfmr.learn.manifest import _parse_pip_packages
        pkgs = _parse_pip_packages(
            "pip install --no-cache-dir --find-links /wheels requests"
        )
        for pkg in pkgs:
            assert not pkg.startswith("-")
        assert "requests" in pkgs

    def test_recursive_submodules(self, analyzer, tmp_path):
        manifest = {
            "app-id": "org.test.App",
            "runtime": "org.freedesktop.Platform",
            "runtime-version": "24.08",
            "sdk": "org.freedesktop.Sdk",
            "modules": [
                {
                    "name": "outer",
                    "buildsystem": "autotools",
                    "modules": [
                        {
                            "name": "python-inner",
                            "buildsystem": "simple",
                            "build-commands": ["pip install requests"],
                        }
                    ],
                }
            ],
        }
        p = tmp_path / "app.json"
        p.write_text(json.dumps(manifest))
        analysis = analyzer.analyze(p)
        assert "requests" in analysis.python_packages

    def test_dict_command_string_key_parsed(self):
        """build-commands entry as dict: {'pip install foo': None}"""
        from pfmr.learn.manifest import _parse_pip_packages, _command_to_str
        pkgs = _parse_pip_packages({"pip install requests": None})
        assert "requests" in pkgs

    def test_dict_command_with_condition_parsed(self):
        from pfmr.learn.manifest import _parse_pip_packages
        pkgs = _parse_pip_packages({"pip install numpy": "some-condition"})
        assert "numpy" in pkgs

    def test_none_command_returns_empty(self):
        from pfmr.learn.manifest import _parse_pip_packages
        assert _parse_pip_packages(None) == []

    def test_int_command_returns_empty(self):
        from pfmr.learn.manifest import _parse_pip_packages
        assert _parse_pip_packages(42) == []

    def test_empty_dict_command_returns_empty(self):
        from pfmr.learn.manifest import _parse_pip_packages
        assert _parse_pip_packages({}) == []

    def test_command_to_str_string(self):
        from pfmr.learn.manifest import _command_to_str
        assert _command_to_str("pip install foo") == "pip install foo"

    def test_command_to_str_dict(self):
        from pfmr.learn.manifest import _command_to_str
        assert _command_to_str({"fc-cache -fsv ||": None}) == "fc-cache -fsv ||"

    def test_manifest_with_dict_commands_no_crash(self, tmp_path):
        """Manifest whose build-commands mix strings and dicts must not crash."""
        import json
        manifest = {
            "app-id": "org.test.App",
            "runtime": "org.freedesktop.Platform",
            "runtime-version": "24.08",
            "sdk": "org.freedesktop.Sdk",
            "modules": [
                {
                    "name": "python-deps",
                    "buildsystem": "simple",
                    "build-commands": [
                        "pip install requests",
                        {"fc-cache -fsv || true": None},
                        "pip install cryptography",
                    ],
                }
            ],
        }
        p = tmp_path / "app.json"
        p.write_text(json.dumps(manifest))
        analysis = ManifestAnalyzer().analyze(p)
        assert analysis is not None
        assert "requests" in analysis.python_packages
        assert "cryptography" in analysis.python_packages

    def test_single_module_manifest_is_skipped(self, tmp_path):
        """A manifest with only one module — that module IS the app, skip it."""
        import json
        manifest = {
            "app-id": "org.gnome.Fractal",
            "runtime": "org.gnome.Platform",
            "runtime-version": "48",
            "sdk": "org.gnome.Sdk",
            "modules": [
                {
                    "name": "fractal",
                    "buildsystem": "meson",
                    "sources": [{"type": "dir", "path": "."}],
                }
            ],
        }
        p = tmp_path / "app.json"
        p.write_text(json.dumps(manifest))
        analysis = ManifestAnalyzer().analyze(p)
        assert analysis is not None
        assert len(analysis.native_modules) == 0

    def test_single_module_no_dir_source_still_skipped(self, tmp_path):
        """Single module even without dir source is still the app."""
        import json
        manifest = {
            "app-id": "org.test.App",
            "runtime": "org.freedesktop.Platform",
            "runtime-version": "24.08",
            "sdk": "org.freedesktop.Sdk",
            "modules": [
                {
                    "name": "myapp",
                    "buildsystem": "cmake",
                    "sources": [{"type": "archive", "url": "https://example.com/src.tar.gz"}],
                }
            ],
        }
        p = tmp_path / "app.json"
        p.write_text(json.dumps(manifest))
        analysis = ManifestAnalyzer().analyze(p)
        assert len(analysis.native_modules) == 0

    def test_last_module_matches_app_id_tail_is_skipped(self, tmp_path):
        """Last module whose name == app-id last segment is skipped."""
        import json
        manifest = {
            "app-id": "org.gnome.Fractal",
            "runtime": "org.gnome.Platform",
            "runtime-version": "48",
            "sdk": "org.gnome.Sdk",
            "modules": [
                {
                    "name": "libusb",
                    "buildsystem": "autotools",
                    "sources": [{"type": "archive", "url": "https://example.com/libusb.tar.bz2", "sha256": "abc"}],
                },
                {
                    "name": "fractal",
                    "buildsystem": "meson",
                    "sources": [{"type": "dir", "path": "."}],
                },
            ],
        }
        p = tmp_path / "app.json"
        p.write_text(json.dumps(manifest))
        analysis = ManifestAnalyzer().analyze(p)
        # libusb is a dep → kept; fractal is the app → skipped
        names = [m.module_name for m in analysis.native_modules]
        assert "libusb" in names
        assert "fractal" not in names

    def test_two_modules_dep_plus_app(self, tmp_path):
        """Two modules: first is a real dep, second is the app."""
        import json
        manifest = {
            "app-id": "org.test.App",
            "runtime": "org.freedesktop.Platform",
            "runtime-version": "24.08",
            "sdk": "org.freedesktop.Sdk",
            "modules": [
                {
                    "name": "libfoo",
                    "buildsystem": "cmake",
                    "sources": [{"type": "archive", "url": "https://example.com/foo.tar.gz", "sha256": "abc"}],
                },
                {
                    "name": "myapp",
                    "buildsystem": "meson",
                    "sources": [{"type": "dir", "path": "."}],
                },
            ],
        }
        p = tmp_path / "app.json"
        p.write_text(json.dumps(manifest))
        analysis = ManifestAnalyzer().analyze(p)
        names = [m.module_name for m in analysis.native_modules]
        assert "libfoo" in names
        assert "myapp" not in names

    def test_last_module_no_source_is_skipped(self, tmp_path):
        """Last module with no source entries at all is treated as app."""
        import json
        manifest = {
            "app-id": "org.test.App",
            "runtime": "org.freedesktop.Platform",
            "runtime-version": "24.08",
            "sdk": "org.freedesktop.Sdk",
            "modules": [
                {
                    "name": "libdep",
                    "buildsystem": "autotools",
                    "sources": [{"type": "archive", "url": "https://example.com/lib.tar.gz", "sha256": "xyz"}],
                },
                {
                    "name": "myapp",
                    "buildsystem": "simple",
                    "build-commands": ["true"],
                    "sources": [],
                },
            ],
        }
        p = tmp_path / "app.json"
        p.write_text(json.dumps(manifest))
        analysis = ManifestAnalyzer().analyze(p)
        names = [m.module_name for m in analysis.native_modules]
        assert "libdep" in names
        assert "myapp" not in names

    def test_never_app_name_always_kept(self, tmp_path):
        """A module whose name is in never_app_names is always treated as a dep."""
        import json
        # libusb alone — would normally be skipped as "single module"
        # but libusb is in never_app_names so it's kept
        manifest = {
            "app-id": "org.test.LibusbApp",
            "runtime": "org.freedesktop.Platform",
            "runtime-version": "24.08",
            "sdk": "org.freedesktop.Sdk",
            "modules": [
                {
                    "name": "libusb",
                    "buildsystem": "autotools",
                    "sources": [{"type": "archive", "url": "https://example.com/libusb.tar.bz2", "sha256": "aaa"}],
                }
            ],
        }
        p = tmp_path / "app.json"
        p.write_text(json.dumps(manifest))
        analysis = ManifestAnalyzer().analyze(p)
        # libusb is in never_app_names — must be kept even when it's the only module
        names = [m.module_name for m in analysis.native_modules]
        assert "libusb" in names


# ===========================================================================
# SandboxLearner
# ===========================================================================

class TestSandboxLearner:
    """SandboxLearner now writes to recipes/python/ directly — no KnowledgeGraph."""

    def test_ingest_skipped_when_not_ran(self, tmp_path):
        learner = SandboxLearner(repo_root=tmp_path)
        report = _make_report(ran=False)
        written = learner.ingest(report, package_name="cryptography")
        assert written == 0

    def test_ingest_pkgconfig_error_writes_recipe(self, tmp_path):
        learner = SandboxLearner(repo_root=tmp_path)
        report = _make_report(
            errors=[SandboxError(SandboxErrorType.MISSING_PKGCONFIG, "openssl", "stderr",
                                 context="cryptography install")]
        )
        written = learner.ingest(report, package_name="cryptography")
        assert written > 0
        recipe_path = tmp_path / "recipes" / "python" / "cryptography.yaml"
        assert recipe_path.exists()
        recipe = yaml.safe_load(recipe_path.read_text())
        assert "openssl" in recipe["requires"]["pkgconfig"]

    def test_ingest_library_error_writes_recipe(self, tmp_path):
        learner = SandboxLearner(repo_root=tmp_path)
        report = _make_report(
            errors=[SandboxError(SandboxErrorType.MISSING_NATIVE_DEP, "libusb-1.0.so.0", "ldd",
                                 context="hidapi install")]
        )
        learner.ingest(report, package_name="hidapi")
        recipe_path = tmp_path / "recipes" / "python" / "hidapi.yaml"
        assert recipe_path.exists()
        recipe = yaml.safe_load(recipe_path.read_text())
        assert "libusb-1.0.so.0" in recipe["requires"]["libraries"]

    def test_ingest_successful_build(self, tmp_path):
        learner = SandboxLearner(repo_root=tmp_path)
        written = learner.ingest_successful_build(
            "cryptography",
            native_deps=["openssl", "libffi"],
            required_extensions=["org.freedesktop.Sdk.Extension.rust-stable"],
        )
        assert written > 0
        recipe = yaml.safe_load(
            (tmp_path / "recipes" / "python" / "cryptography.yaml").read_text()
        )
        assert "openssl" in recipe["requires"]["pkgconfig"]
        assert "org.freedesktop.Sdk.Extension.rust-stable" in recipe["requires"]["extensions"]

    def test_confidence_is_1_for_successful_build(self, tmp_path):
        learner = SandboxLearner(repo_root=tmp_path)
        learner.ingest_successful_build("cffi", native_deps=["libffi"], required_extensions=[])
        recipe = yaml.safe_load(
            (tmp_path / "recipes" / "python" / "cffi.yaml").read_text()
        )
        assert recipe["confidence"] == 1.0

    def test_confidence_is_08_for_probe_error(self, tmp_path):
        learner = SandboxLearner(repo_root=tmp_path)
        report = _make_report(
            errors=[SandboxError(SandboxErrorType.MISSING_PKGCONFIG, "openssl", "stderr")]
        )
        learner.ingest(report, package_name="cryptography")
        recipe = yaml.safe_load(
            (tmp_path / "recipes" / "python" / "cryptography.yaml").read_text()
        )
        assert recipe["confidence"] == 0.8

    def test_successful_no_errors_writes_sdk_sufficient(self, tmp_path):
        learner = SandboxLearner(repo_root=tmp_path)
        report = _make_report(ran=True, errors=[])
        learner.ingest(report, package_name="requests")
        recipe_path = tmp_path / "recipes" / "python" / "requests.yaml"
        assert recipe_path.exists()
        recipe = yaml.safe_load(recipe_path.read_text())
        assert recipe.get("sdk_sufficient") is True

    def test_merge_with_existing_recipe(self, tmp_path):
        """Ingesting a second report merges deps with existing recipe."""
        learner = SandboxLearner(repo_root=tmp_path)
        # First report: openssl missing
        report1 = _make_report(
            errors=[SandboxError(SandboxErrorType.MISSING_PKGCONFIG, "openssl", "stderr",
                                 context="cryptography install")]
        )
        learner.ingest(report1, package_name="cryptography")
        # Second report: libffi also missing
        report2 = _make_report(
            errors=[SandboxError(SandboxErrorType.MISSING_PKGCONFIG, "libffi", "stderr",
                                 context="cryptography install")]
        )
        learner.ingest(report2, package_name="cryptography")
        recipe = yaml.safe_load(
            (tmp_path / "recipes" / "python" / "cryptography.yaml").read_text()
        )
        # Both deps should be present after merge
        assert "openssl" in recipe["requires"]["pkgconfig"]
        assert "libffi" in recipe["requires"]["pkgconfig"]

    def test_higher_confidence_updates_recipe(self, tmp_path):
        """A higher-confidence ingest updates the recipe."""
        learner = SandboxLearner(repo_root=tmp_path)
        # Low confidence first (probe error)
        r1 = _make_report(errors=[SandboxError(SandboxErrorType.MISSING_PKGCONFIG, "openssl", "stderr")])
        learner.ingest(r1, package_name="cryptography")
        # Higher confidence (successful build)
        learner.ingest_successful_build("cryptography", ["openssl"], [])
        recipe = yaml.safe_load(
            (tmp_path / "recipes" / "python" / "cryptography.yaml").read_text()
        )
        assert recipe["confidence"] == 1.0


# ===========================================================================
# Exporter
# ===========================================================================

class TestExporter:
    """Tests for Exporter — takes list[ManifestAnalysis], writes recipes/."""

    def _make_analysis(self, app_id="org.test.App", native_modules=None, python_packages=None):
        from pfmr.learn.manifest import ManifestAnalysis, LearnedNativeModule
        return ManifestAnalysis(
            app_id=app_id,
            runtime="org.freedesktop.Platform",
            sdk="org.freedesktop.Sdk",
            sdk_version="24.08",
            python_packages=python_packages or [],
            native_modules=native_modules or [],
            source_path=f"flathub:{app_id}",
        )

    def _make_native_mod(self, name, url="https://example.com/lib.tar.gz",
                         sha256="abc123", pkgconfig=None):
        from pfmr.learn.manifest import LearnedNativeModule
        return LearnedNativeModule(
            module_name=name,
            buildsystem="autotools",
            source_url=url,
            source_sha256=sha256,
            pkgconfig_names=pkgconfig or [name],
            cleanup=["/include", "/lib/pkgconfig"],
        )

    def test_export_native_recipe_created(self, tmp_path):
        mod = self._make_native_mod("libfoo", pkgconfig=["foo"])
        analysis = self._make_analysis(native_modules=[mod])
        exporter = Exporter([analysis], tmp_path)
        report = exporter.export_native_recipes(dry_run=False)
        recipe_path = tmp_path / "recipes" / "native" / "libfoo.yaml"
        assert recipe_path.exists()
        recipe = yaml.safe_load(recipe_path.read_text())
        assert recipe["id"] == "libfoo"
        assert recipe["source"]["url"] == "https://example.com/lib.tar.gz"
        assert len(report.created) == 1

    def test_export_recipe_skip_no_source_url(self, tmp_path):
        from pfmr.learn.manifest import LearnedNativeModule
        mod = LearnedNativeModule("nosource", "autotools")  # no source_url
        analysis = self._make_analysis(native_modules=[mod])
        exporter = Exporter([analysis], tmp_path)
        exporter.export_native_recipes(dry_run=False)
        assert not (tmp_path / "recipes" / "native" / "nosource.yaml").exists()

    def test_export_recipe_skip_existing(self, tmp_path):
        recipes_dir = tmp_path / "recipes" / "native"
        recipes_dir.mkdir(parents=True)
        (recipes_dir / "libfoo.yaml").write_text("id: libfoo\n")
        mod = self._make_native_mod("libfoo")
        analysis = self._make_analysis(native_modules=[mod])
        exporter = Exporter([analysis], tmp_path)
        exporter.export_native_recipes(dry_run=False)
        assert (recipes_dir / "libfoo.yaml").read_text() == "id: libfoo\n"

    def test_export_python_recipe_created(self, tmp_path):
        mod = self._make_native_mod("openssl", pkgconfig=["openssl"])
        analysis = self._make_analysis(
            native_modules=[mod],
            python_packages=["cryptography"],
        )
        exporter = Exporter([analysis], tmp_path)
        report = exporter.export_python_recipes(dry_run=False)
        recipe_path = tmp_path / "recipes" / "python" / "cryptography.yaml"
        assert recipe_path.exists()
        recipe = yaml.safe_load(recipe_path.read_text())
        assert recipe["id"] == "cryptography"
        assert "openssl" in recipe["requires"]["pkgconfig"]

    def test_export_dry_run_no_write(self, tmp_path):
        mod = self._make_native_mod("libbar", pkgconfig=["bar"])
        analysis = self._make_analysis(native_modules=[mod])
        exporter = Exporter([analysis], tmp_path)
        report = exporter.export(dry_run=True)
        assert not (tmp_path / "recipes" / "native" / "libbar.yaml").exists()

    def test_export_dedup_across_analyses(self, tmp_path):
        """Same native module from two analyses → created once."""
        mod = self._make_native_mod("libdedup", pkgconfig=["dedup"])
        a1 = self._make_analysis("org.app.One", native_modules=[mod])
        a2 = self._make_analysis("org.app.Two", native_modules=[mod])
        exporter = Exporter([a1, a2], tmp_path)
        report = exporter.export_native_recipes(dry_run=False)
        assert len(report.created) == 1

    def test_export_report_actions(self, tmp_path):
        mod = self._make_native_mod("libreport", pkgconfig=["report"])
        analysis = self._make_analysis(native_modules=[mod])
        exporter = Exporter([analysis], tmp_path)
        report = exporter.export(dry_run=False)
        actions = {c.action for c in report.changes}
        assert "create" in actions


# ===========================================================================
# FlathubMiner (mocked)
# ===========================================================================

class TestFlathubMiner:
    def test_mine_calls_github_api(self, tmp_path):
        from pfmr.learn.flathub import FlathubMiner

        repos_response = [{"name": "org.test.App"}, {"name": "org.test.App2"}]
        contents_response = [{"name": "org.test.App.json", "download_url": "https://raw/app.json"}]
        manifest_data = {
            "app-id": "org.test.App",
            "runtime": "org.freedesktop.Platform",
            "runtime-version": "24.08",
            "sdk": "org.freedesktop.Sdk",
            "modules": [
                {"name": "python-deps", "buildsystem": "simple",
                 "build-commands": ["pip install requests"]}
            ],
        }

        with patch("pfmr.learn.flathub._gh_get") as mock_gh, \
             patch("requests.get") as mock_req:

            mock_gh.side_effect = [repos_response, contents_response, contents_response]
            mock_req.return_value = MagicMock(
                status_code=200,
                json=lambda: manifest_data,
                text=json.dumps(manifest_data),
            )

            # Use tmp_path for cache so progress doesn't persist from other tests
            miner = FlathubMiner(cache_dir=tmp_path, force_refresh=True)
            result = miner.mine(limit=2, only_python=False)

        assert result.manifests_found >= 1
        assert any("requests" in a.python_packages for a in result.analyses)

    def test_mine_all_manifests_by_default(self, tmp_path):
        """Without only_python=True, all repos with any modules are mined."""
        from pfmr.learn.flathub import FlathubMiner

        non_python_manifest = {
            "app-id": "org.test.NoScript",
            "runtime": "org.freedesktop.Platform",
            "runtime-version": "24.08",
            "sdk": "org.freedesktop.Sdk",
            "modules": [{"name": "myapp", "buildsystem": "cmake"}],
        }
        repos_response = [{"name": "org.test.NoScript"}]
        contents_response = [{"name": "org.test.NoScript.json",
                               "download_url": "https://raw/app.json"}]
        with patch("pfmr.learn.flathub._gh_get") as mock_gh, \
             patch("requests.get") as mock_req:
            mock_gh.side_effect = [repos_response, contents_response]
            mock_req.return_value = MagicMock(
                status_code=200,
                json=lambda: non_python_manifest,
                text=json.dumps(non_python_manifest),
            )
            miner = FlathubMiner(cache_dir=tmp_path, force_refresh=True)
            result = miner.mine(limit=1, only_python=False)
        # Non-python manifests are now mined (only_python=False by default)
        assert result.manifests_found >= 1

    def test_mine_manifest_url(self):
        from pfmr.learn.flathub import FlathubMiner

        manifest_data = {
            "app-id": "org.test.App",
            "runtime": "org.freedesktop.Platform",
            "runtime-version": "24.08",
            "sdk": "org.freedesktop.Sdk",
            "modules": [
                {"name": "pymod", "buildsystem": "simple",
                 "build-commands": ["pip install numpy"]}
            ],
        }

        with patch("requests.get") as mock_req:
            mock_req.return_value = MagicMock(
                status_code=200,
                json=lambda: manifest_data,
            )
            miner = FlathubMiner()
            analysis = miner.mine_manifest_url(
                "https://raw.githubusercontent.com/flathub/app/main/app.json"
            )

        assert analysis is not None
        assert "numpy" in analysis.python_packages


# ===========================================================================
# CLI helpers (learn.cli)
# ===========================================================================

class TestLearnCliHelpers:
    def test_exporter_from_analyses(self, tmp_path):
        """CLI exporter writes recipes from ManifestAnalysis list."""
        from pfmr.learn.manifest import ManifestAnalysis, LearnedNativeModule
        from pfmr.learn.exporter import Exporter

        analysis = ManifestAnalysis(
            app_id="org.test.App",
            runtime="org.freedesktop.Platform",
            sdk="org.freedesktop.Sdk",
            sdk_version="24.08",
            python_packages=["cryptography"],
            native_modules=[
                LearnedNativeModule(
                    module_name="openssl",
                    buildsystem="autotools",
                    source_url="https://example.com/openssl.tar.gz",
                    source_sha256="abc",
                    pkgconfig_names=["openssl"],
                )
            ],
        )
        exporter = Exporter([analysis], tmp_path)
        report = exporter.export(dry_run=False)
        assert (tmp_path / "recipes" / "native" / "openssl.yaml").exists()
        assert (tmp_path / "recipes" / "python" / "cryptography.yaml").exists()

    def test_python_recipe_has_co_occurrence_pkgconfig(self, tmp_path):
        """Python recipe lists pkgconfig from co-occurring native modules."""
        from pfmr.learn.manifest import ManifestAnalysis, LearnedNativeModule
        from pfmr.learn.exporter import Exporter

        analysis = ManifestAnalysis(
            app_id="org.test.App",
            runtime="org.freedesktop.Platform",
            sdk="org.freedesktop.Sdk",
            sdk_version="24.08",
            python_packages=["cryptography"],
            native_modules=[
                LearnedNativeModule("openssl", "autotools",
                                    source_url="https://example.com/x.tar.gz",
                                    pkgconfig_names=["openssl"])
            ],
        )
        exporter = Exporter([analysis], tmp_path)
        exporter.export(dry_run=False)
        import yaml
        recipe = yaml.safe_load(
            (tmp_path / "recipes" / "python" / "cryptography.yaml").read_text()
        )
        assert "openssl" in recipe["requires"]["pkgconfig"]
        assert recipe["confidence"] == 0.6  # co-occurrence confidence