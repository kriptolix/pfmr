"""
pfmr.generators.manifest
~~~~~~~~~~~~~~~~~~~~~~~~
Flatpak Manifest Generator — Phase 1 output component.

Takes a ResolutionResult and produces:
- A FlatpakManifest object
- JSON or YAML serialisation of the manifest

Phase 1 strategy:
  - Pure Python packages: bundled into a single "python-deps" pip install module
    using --find-links to point at locally cached wheels/sdists.
  - Packages requiring native compilation: each becomes its own module using
    pip install with appropriate env flags.
  - Native library recipes: inserted as prerequisite modules before the Python
    install steps.
  - Build-backend extensions (Rust, etc.): declared in sdk-extensions.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import yaml

from pfmr.models import (
    BuildBackend,
    FlatpakManifest,
    FlatpakModule,
    FlatpakSource,
    NativeRecipe,
    ResolutionResult,
    ResolvedPackage,
    SourceType,
)
from pfmr.utils.logging import get_logger

logger = get_logger(__name__)

# Extensions required by each build backend
_BACKEND_EXTENSIONS: dict[BuildBackend, str] = {
    BuildBackend.MATURIN: "org.freedesktop.Sdk.Extension.rust-stable",
    BuildBackend.SETUPTOOLS_RUST: "org.freedesktop.Sdk.Extension.rust-stable",
}

# Path inside the Flatpak where Python venv lives
_VENV_PATH = "/app/venv"
_PIP = f"{_VENV_PATH}/bin/pip"
_UV = f"{_VENV_PATH}/bin/uv"


class ManifestGenerator:
    """
    Converts a ResolutionResult into a FlatpakManifest.
    """

    def __init__(
        self,
        app_id: str = "org.example.App",
        runtime: str = "org.freedesktop.Platform",
        runtime_version: str = "24.08",
        sdk: str = "org.freedesktop.Sdk",
        python_version: str = "3.11",
        install_dir: str = "/app",
        use_uv: bool = True,
    ):
        self.app_id = app_id
        self.runtime = runtime
        self.runtime_version = runtime_version
        self.sdk = sdk
        self.python_version = python_version
        self.install_dir = install_dir
        self.use_uv = use_uv

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, result: ResolutionResult) -> FlatpakManifest:
        """Generate a complete FlatpakManifest from a ResolutionResult."""
        sdk_extensions = list(dict.fromkeys(result.required_extensions))  # dedup

        # Collect extensions needed by build backends
        for pkg in result.packages:
            ext = _BACKEND_EXTENSIONS.get(pkg.build_backend)
            if ext and ext not in sdk_extensions:
                sdk_extensions.append(ext)

        modules: list[FlatpakModule] = []

        # 1. Native library prerequisite modules
        for recipe in result.native_recipes:
            modules.append(self._recipe_to_module(recipe))

        # 2. Python venv setup module
        modules.append(self._venv_setup_module())

        # 3. Separate native packages into individual modules; batch pure ones
        pure_pkgs = [p for p in result.packages if not p.requires_native]
        native_pkgs = [p for p in result.packages if p.requires_native]

        if pure_pkgs:
            modules.append(self._pure_python_module(pure_pkgs))

        for pkg in native_pkgs:
            modules.append(self._native_python_module(pkg))

        manifest = FlatpakManifest(
            app_id=self.app_id,
            runtime=self.runtime,
            runtime_version=self.runtime_version,
            sdk=self.sdk,
            sdk_extensions=sdk_extensions,
            modules=modules,
        )
        logger.info(
            "Generated manifest: %d modules, %d sdk-extensions",
            len(modules),
            len(sdk_extensions),
        )
        return manifest

    def to_yaml(self, manifest: FlatpakManifest, path: Optional[Path] = None) -> str:
        data = self._manifest_to_dict(manifest)
        text = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
        if path:
            path.write_text(text)
            logger.info("Wrote YAML manifest to %s", path)
        return text

    def to_json(self, manifest: FlatpakManifest, path: Optional[Path] = None) -> str:
        data = self._manifest_to_dict(manifest)
        text = json.dumps(data, indent=2, ensure_ascii=False)
        if path:
            path.write_text(text)
            logger.info("Wrote JSON manifest to %s", path)
        return text

    # ------------------------------------------------------------------
    # Module builders
    # ------------------------------------------------------------------

    def _venv_setup_module(self) -> FlatpakModule:
        """Creates the Python venv and installs uv inside it."""
        cmds = [
            f"python{self.python_version} -m venv {_VENV_PATH}",
            f"{_VENV_PATH}/bin/pip install --upgrade pip",
        ]
        if self.use_uv:
            cmds.append(f"{_VENV_PATH}/bin/pip install uv")
        return FlatpakModule(
            name="python-venv-setup",
            buildsystem="simple",
            build_commands=cmds,
        )

    def _pure_python_module(self, packages: list[ResolvedPackage]) -> FlatpakModule:
        """All pure-python packages batched into a single pip install."""
        sources: list[FlatpakSource] = []
        wheel_args: list[str] = []

        for pkg in packages:
            if pkg.source_url and pkg.source_hash:
                fname = pkg.source_url.split("/")[-1]
                sources.append(
                    FlatpakSource(
                        type="file",
                        url=pkg.source_url,
                        sha256=pkg.source_hash,
                        dest_filename=fname,
                    )
                )
                wheel_args.append(fname)

        if self.use_uv:
            install_cmd = f"{_UV} pip install --no-index --find-links=. " + " ".join(
                f"{p.name}=={p.version}" for p in packages
            )
        else:
            install_cmd = f"{_PIP} install --no-index --find-links=. --no-build-isolation " + " ".join(
                f"{p.name}=={p.version}" for p in packages
            )

        return FlatpakModule(
            name="python-pure-deps",
            buildsystem="simple",
            build_commands=[install_cmd],
            sources=sources,
        )

    def _native_python_module(self, pkg: ResolvedPackage) -> FlatpakModule:
        """One module per native Python package."""
        sources: list[FlatpakSource] = []
        if pkg.source_url and pkg.source_hash:
            fname = pkg.source_url.split("/")[-1]
            sources.append(
                FlatpakSource(
                    type="file",
                    url=pkg.source_url,
                    sha256=pkg.source_hash,
                    dest_filename=fname,
                )
            )

        build_options: dict = {}
        build_cmds: list[str] = []

        if pkg.build_backend == BuildBackend.MATURIN:
            build_options["env"] = {
                "PATH": "/usr/lib/sdk/rust-stable/bin:$PATH",
                "CARGO_HOME": "/run/build/cargo-home",
            }

        if self.use_uv:
            install_cmd = (
                f"{_UV} pip install --no-build-isolation "
                f"{pkg.name}=={pkg.version}"
            )
        else:
            install_cmd = (
                f"{_PIP} install --no-build-isolation "
                f"--no-index --find-links=. "
                f"{pkg.name}=={pkg.version}"
            )

        build_cmds.append(install_cmd)

        return FlatpakModule(
            name=f"python-{pkg.name.lower().replace('_', '-')}",
            buildsystem="simple",
            build_commands=build_cmds,
            sources=sources,
            build_options=build_options,
        )

    def _recipe_to_module(self, recipe: NativeRecipe) -> FlatpakModule:
        """Convert a NativeRecipe into a FlatpakModule."""
        sources = []
        if recipe.source:
            sources.append(recipe.source)

        if recipe.build_commands:
            return FlatpakModule(
                name=recipe.id,
                buildsystem="simple",
                build_commands=recipe.build_commands,
                sources=sources,
                cleanup=recipe.cleanup,
            )

        return FlatpakModule(
            name=recipe.id,
            buildsystem=recipe.buildsystem,
            sources=sources,
            config_opts=recipe.config_opts,
            cleanup=recipe.cleanup,
        )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def _manifest_to_dict(self, manifest: FlatpakManifest) -> dict:
        d: dict = {
            "app-id": manifest.app_id,
            "runtime": manifest.runtime,
            "runtime-version": manifest.runtime_version,
            "sdk": manifest.sdk,
        }
        if manifest.sdk_extensions:
            d["sdk-extensions"] = manifest.sdk_extensions
        if manifest.finish_args:
            d["finish-args"] = manifest.finish_args
        d["modules"] = [self._module_to_dict(m) for m in manifest.modules]
        return d

    def _module_to_dict(self, module: FlatpakModule) -> dict:
        d: dict = {"name": module.name}

        if module.buildsystem != "simple":
            d["buildsystem"] = module.buildsystem
        else:
            d["buildsystem"] = "simple"

        if module.build_commands:
            d["build-commands"] = module.build_commands

        if module.config_opts:
            d["config-opts"] = module.config_opts

        if module.build_options:
            d["build-options"] = module.build_options

        if module.sources:
            d["sources"] = [self._source_to_dict(s) for s in module.sources]

        if module.cleanup:
            d["cleanup"] = module.cleanup

        if module.modules:
            d["modules"] = [self._module_to_dict(sub) for sub in module.modules]

        return d

    @staticmethod
    def _source_to_dict(src: FlatpakSource) -> dict:
        d: dict = {"type": src.type}
        if src.url:
            d["url"] = src.url
        if src.sha256:
            d["sha256"] = src.sha256
        if src.path:
            d["path"] = src.path
        if src.dest_filename:
            d["dest-filename"] = src.dest_filename
        if src.branch:
            d["branch"] = src.branch
        if src.commit:
            d["commit"] = src.commit
        if src.tag:
            d["tag"] = src.tag
        return d
