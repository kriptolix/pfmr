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
from pfmr.learn.sandbox import SandboxLearner, _normalise_soname
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
        assert analysis.runtime_version == "48"

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


# ===========================================================================
# SandboxLearner
# ===========================================================================

class TestSandboxLearner:
    def test_ingest_skipped_when_not_ran(self, tmp_path):
        kg = _fresh_kg(tmp_path)
        learner = SandboxLearner(kg)
        report = _make_report(ran=False)
        added = learner.ingest(report, package_name="cryptography")
        assert added == 0

    def test_ingest_pkgconfig_error_adds_edge(self, tmp_path):
        kg = _fresh_kg(tmp_path)
        learner = SandboxLearner(kg)
        report = _make_report(
            errors=[SandboxError(SandboxErrorType.MISSING_PKGCONFIG, "openssl", "stderr",
                                 context="cryptography install")]
        )
        added = learner.ingest(report, package_name="cryptography")
        assert added > 0
        edges = kg.edges_from("cryptography", Rel.REQUIRES_PKGCONFIG)
        assert any(e.to_id == "openssl" for e in edges)

    def test_ingest_library_error_adds_edge(self, tmp_path):
        kg = _fresh_kg(tmp_path)
        learner = SandboxLearner(kg)
        report = _make_report(
            errors=[SandboxError(SandboxErrorType.MISSING_NATIVE_DEP, "libusb-1.0.so.0", "ldd",
                                 context="hidapi install")]
        )
        learner.ingest(report, package_name="hidapi")
        edges = kg.edges_from("hidapi", Rel.REQUIRES_LIBRARY)
        assert len(edges) > 0

    def test_ingest_successful_build(self, tmp_path):
        kg = _fresh_kg(tmp_path)
        learner = SandboxLearner(kg)
        added = learner.ingest_successful_build(
            "cryptography",
            native_deps=["openssl", "libffi"],
            required_extensions=["org.freedesktop.Sdk.Extension.rust-stable"],
        )
        assert added > 0
        # Check edges
        pc_edges = kg.edges_from("cryptography", Rel.REQUIRES_PKGCONFIG)
        ext_edges = kg.edges_from("cryptography", Rel.REQUIRES_EXTENSION)
        assert any(e.to_id == "openssl" for e in pc_edges)
        assert any("rust" in e.to_id for e in ext_edges)

    def test_confidence_is_1_for_successful_build(self, tmp_path):
        kg = _fresh_kg(tmp_path)
        learner = SandboxLearner(kg)
        learner.ingest_successful_build("cffi", native_deps=["libffi"], required_extensions=[])
        edges = kg.edges_from("cffi", Rel.REQUIRES_PKGCONFIG)
        assert all(e.confidence == 1.0 for e in edges)

    def test_confidence_is_08_for_probe_error(self, tmp_path):
        kg = _fresh_kg(tmp_path)
        learner = SandboxLearner(kg)
        report = _make_report(
            errors=[SandboxError(SandboxErrorType.MISSING_PKGCONFIG, "openssl", "stderr")]
        )
        learner.ingest(report, package_name="cryptography")
        edges = kg.edges_from("cryptography", Rel.REQUIRES_PKGCONFIG)
        assert all(e.confidence == 0.8 for e in edges)

    def test_successful_no_errors_marks_sdk_sufficient(self, tmp_path):
        kg = _fresh_kg(tmp_path)
        learner = SandboxLearner(kg)
        report = _make_report(ran=True, errors=[])
        learner.ingest(report, package_name="requests")
        node = kg.node("requests")
        assert node is not None
        assert node.attrs.get("sdk_sufficient") is True

    def test_sdk_node_created(self, tmp_path):
        kg = _fresh_kg(tmp_path)
        learner = SandboxLearner(kg)
        report = _make_report()
        learner.ingest(report, sdk_id="org.gnome.Sdk", sdk_version="48")
        assert kg.node("org.gnome.Sdk//48") is not None

    def test_normalise_soname(self):
        assert _normalise_soname("libssl.so.3") == "ssl"
        assert _normalise_soname("libusb-1.0.so.0") == "usb-1.0"
        assert _normalise_soname("libz.so.1") == "z"


# ===========================================================================
# Exporter
# ===========================================================================

class TestExporter:
    def test_export_native_hints_new_package(self, tmp_path):
        kg = _fresh_kg(tmp_path)
        kg.add_node(KGNode("mylib", "package"))
        kg.add_node(KGNode("mypc", "library", {"pkgconfig": "mypc"}))
        kg.add_edge(KGEdge("mylib", "mypc", Rel.REQUIRES_PKGCONFIG, confidence=0.9))

        exporter = Exporter(kg, tmp_path)
        report = exporter.export_native_hints(dry_run=False)

        hints_path = tmp_path / "pfmr" / "data" / "native-hints" / "packages.toml"
        assert hints_path.exists()
        content = hints_path.read_text()
        assert "mylib" in content
        assert "mypc" in content

    def test_export_hints_skip_existing_entries(self, tmp_path):
        """Packages already in the hints file must not be duplicated."""
        hints_path = tmp_path / "pfmr" / "data" / "native-hints" / "packages.toml"
        hints_path.parent.mkdir(parents=True, exist_ok=True)
        hints_path.write_text('[cryptography]\npkgconfig = ["openssl"]\nlibraries = []\nheaders = []\n')

        kg = _fresh_kg(tmp_path)
        kg.add_node(KGNode("cryptography", "package"))
        kg.add_edge(KGEdge("cryptography", "openssl", Rel.REQUIRES_PKGCONFIG, confidence=1.0))

        exporter = Exporter(kg, tmp_path)
        exporter.export_native_hints(dry_run=False)

        content = hints_path.read_text()
        assert content.count("[cryptography]") == 1  # not duplicated

    def test_export_hints_dry_run_no_write(self, tmp_path):
        kg = _fresh_kg(tmp_path)
        kg.add_node(KGNode("newpkg", "package"))
        kg.add_edge(KGEdge("newpkg", "newlib", Rel.REQUIRES_PKGCONFIG, confidence=0.9))

        exporter = Exporter(kg, tmp_path)
        exporter.export_native_hints(dry_run=True)

        hints_path = tmp_path / "pfmr" / "data" / "native-hints" / "packages.toml"
        assert not hints_path.exists()  # dry run — should not write

    def test_export_recipe_new_library(self, tmp_path):
        kg = _fresh_kg(tmp_path)
        kg.add_node(KGNode("mylib", "library", {
            "pkgconfig": "mylib-1.0",
            "soname": "libmylib.so.1",
            "buildsystem": "autotools",
            "source_url": "https://example.com/mylib-1.0.tar.gz",
            "source_sha256": "abc123",
            "source": "flathub:org.test.App",
        }))

        exporter = Exporter(kg, tmp_path)
        exporter.export_native_recipes(dry_run=False)

        recipe_path = tmp_path / "recipes" / "native" / "mylib.yaml"
        assert recipe_path.exists()
        recipe = yaml.safe_load(recipe_path.read_text())
        assert recipe["id"] == "mylib"
        assert recipe["source"]["url"] == "https://example.com/mylib-1.0.tar.gz"

    def test_export_recipe_skip_no_source_url(self, tmp_path):
        kg = _fresh_kg(tmp_path)
        kg.add_node(KGNode("nousource", "library", {"pkgconfig": "nousource"}))

        exporter = Exporter(kg, tmp_path)
        exporter.export_native_recipes(dry_run=False)

        recipe_path = tmp_path / "recipes" / "native" / "nousource.yaml"
        assert not recipe_path.exists()

    def test_export_recipe_skip_existing(self, tmp_path):
        recipes_dir = tmp_path / "recipes" / "native"
        recipes_dir.mkdir(parents=True)
        existing = recipes_dir / "libusb.yaml"
        existing.write_text("id: libusb\n")  # already exists

        kg = _fresh_kg(tmp_path)
        kg.add_node(KGNode("libusb", "library", {
            "source_url": "https://example.com/libusb.tar.gz",
        }))

        exporter = Exporter(kg, tmp_path)
        report = exporter.export_native_recipes(dry_run=False)

        # Should not overwrite
        assert existing.read_text() == "id: libusb\n"

    def test_export_report_actions(self, tmp_path):
        kg = _fresh_kg(tmp_path)
        kg.add_node(KGNode("pkg1", "package"))
        kg.add_edge(KGEdge("pkg1", "dep1", Rel.REQUIRES_PKGCONFIG, confidence=0.9))

        exporter = Exporter(kg, tmp_path)
        report = exporter.export_native_hints(dry_run=False)
        actions = {c.action for c in report.changes}
        assert "create" in actions or "update" in actions

    def test_export_low_confidence_not_exported(self, tmp_path):
        """Edges with confidence < 0.7 should not produce hints entries."""
        kg = _fresh_kg(tmp_path)
        kg.add_node(KGNode("weakpkg", "package"))
        kg.add_edge(KGEdge("weakpkg", "something", Rel.REQUIRES_PKGCONFIG, confidence=0.4))

        exporter = Exporter(kg, tmp_path)
        exporter.export_native_hints(dry_run=False)

        hints_path = tmp_path / "pfmr" / "data" / "native-hints" / "packages.toml"
        if hints_path.exists():
            assert "weakpkg" not in hints_path.read_text()


# ===========================================================================
# FlathubMiner (mocked)
# ===========================================================================

class TestFlathubMiner:
    def test_mine_calls_github_api(self):
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

            miner = FlathubMiner(cache_dir=None)
            result = miner.mine(limit=2, only_python=True)

        assert result.manifests_found >= 1
        assert result.python_apps >= 1
        assert any("requests" in a.python_packages for a in result.analyses)

    def test_mine_skips_non_python(self):
        from pfmr.learn.flathub import FlathubMiner, _PYTHON_SIGNALS

        non_python_manifest = {
            "app-id": "org.test.NoScript",
            "runtime": "org.freedesktop.Platform",
            "runtime-version": "24.08",
            "sdk": "org.freedesktop.Sdk",
            "modules": [{"name": "myapp", "buildsystem": "cmake"}],
        }
        # Verify the has_python check correctly rejects this
        text = json.dumps(non_python_manifest).lower()
        assert not any(sig in text for sig in _PYTHON_SIGNALS)

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
    def test_ingest_analysis_into_graph(self, tmp_path):
        from pfmr.learn.cli import _ingest_analysis_into_graph
        from pfmr.learn.manifest import ManifestAnalysis, LearnedNativeModule

        kg = _fresh_kg(tmp_path)
        analysis = ManifestAnalysis(
            app_id="org.test.App",
            runtime="org.freedesktop.Platform",
            sdk="org.freedesktop.Sdk",
            sdk_version="24.08",
            python_packages=["requests", "cryptography"],
            native_modules=[
                LearnedNativeModule(
                    module_name="openssl",
                    buildsystem="autotools",
                    source_url="https://example.com/openssl.tar.gz",
                    pkgconfig_names=["openssl"],
                )
            ],
        )
        added = _ingest_analysis_into_graph(kg, analysis)
        assert added > 0
        assert kg.node("requests") is not None
        assert kg.node("cryptography") is not None
        assert kg.node("openssl") is not None

    def test_ingest_creates_edges_from_cooccurrence(self, tmp_path):
        from pfmr.learn.cli import _ingest_analysis_into_graph
        from pfmr.learn.manifest import ManifestAnalysis, LearnedNativeModule

        kg = _fresh_kg(tmp_path)
        analysis = ManifestAnalysis(
            app_id="org.test.App",
            runtime="org.freedesktop.Platform",
            sdk="org.freedesktop.Sdk",
            sdk_version="24.08",
            python_packages=["cryptography"],
            native_modules=[
                LearnedNativeModule("openssl", "autotools", pkgconfig_names=["openssl"])
            ],
        )
        _ingest_analysis_into_graph(kg, analysis)
        edges = kg.edges_from("cryptography", Rel.REQUIRES_PKGCONFIG)
        assert any(e.to_id == "openssl" for e in edges)
        # Co-occurrence has lower confidence than a probe
        assert all(e.confidence < 1.0 for e in edges)