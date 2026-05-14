"""
pfmr.pipeline
~~~~~~~~~~~~~
Orchestrator that wires all Phase 1 + Phase 2 components together.

Flow:
  Input (pyproject.toml | requirements.txt | package name)
      ↓
  UVResolver             → ResolutionResult (packages + lockfile hash)
      ↓
  RecipeDB lookup        → attach NativeRecipes for known native deps
      ↓
  SDKCapabilityResolver  → filter recipes already satisfied by SDK,
                           mark unresolved natives
      ↓
  ManifestGenerator      → FlatpakManifest
      ↓
  Output (JSON | YAML)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pfmr.generators.manifest import ManifestGenerator
from pfmr.models import ResolutionResult, SDKResolutionReport, SandboxProbeReport
from pfmr.recipes.db import RecipeDB
from pfmr.resolvers.uv_resolver import UVResolver
from pfmr.resolvers.sdk_capability import SDKCapabilityResolver, SDKQuery
from pfmr.resolvers.sdk_extension import SDKExtensionResolver
from pfmr.resolvers.native_dependency import NativeDependencyAnalyzer
from pfmr.sandbox.probe import BuildSandboxProber
from pfmr.utils.logging import get_logger

logger = get_logger(__name__)


class Pipeline:
    """
    High-level entry point for pfmr (Phase 1 + Phase 2 SDKCapabilityResolver).

    Usage::

        pipeline = Pipeline(
            app_id="org.gnome.MyApp",
            runtime="org.gnome.Platform",
            runtime_version="48",
            sdk="org.gnome.Sdk",
        )
        manifest_text = pipeline.run_from_pyproject(
            Path("pyproject.toml"),
            output_format="yaml",
        )
    """

    def __init__(
        self,
        app_id: str = "org.example.App",
        runtime: str = "org.freedesktop.Platform",
        runtime_version: str = "24.08",
        sdk: str = "org.freedesktop.Sdk",
        python_version: str = "3.11",
        use_uv: bool = True,
        extra_index_urls: Optional[list[str]] = None,
        recipe_dirs: Optional[list[Path]] = None,
        offline: bool = False,
        # SDKCapabilityResolver options
        sdk_extra_ids: Optional[list[str]] = None,
        force_sdk_probe: bool = False,
        extra_profile_dirs: Optional[list[Path]] = None,
        # SDKExtensionResolver options
        forced_extensions: Optional[list[str]] = None,
        excluded_extensions: Optional[list[str]] = None,
        extra_extension_profile_dirs: Optional[list[Path]] = None,
        # NativeDependencyAnalyzer options
        extra_hints_files: Optional[list[Path]] = None,
        enable_elf: bool = True,
        wheel_cache_dir: Optional[Path] = None,
    ):
        self.sdk_id = sdk
        self.sdk_version = runtime_version
        self.recipe_db = RecipeDB(recipe_dirs=recipe_dirs)

        self.resolver = UVResolver(
            python_version=python_version,
            extra_index_urls=extra_index_urls,
            offline=offline,
        )
        self.sdk_resolver = SDKCapabilityResolver(
            sdk_id=sdk,
            sdk_version=runtime_version,
            extra_sdk_ids=sdk_extra_ids or [],
            force_probe=force_sdk_probe,
            offline=offline,
            extra_profile_dirs=extra_profile_dirs or [],
            recipe_db=self.recipe_db,
        )
        self.ext_resolver = SDKExtensionResolver(
            extra_profile_dirs=extra_extension_profile_dirs or [],
            forced_extensions=forced_extensions or [],
            excluded_extensions=excluded_extensions or [],
        )
        self.native_analyzer = NativeDependencyAnalyzer(
            extra_hints_files=extra_hints_files or [],
            enable_elf=enable_elf,
            wheel_cache_dir=wheel_cache_dir,
        )
        self.generator = ManifestGenerator(
            app_id=app_id,
            runtime=runtime,
            runtime_version=runtime_version,
            sdk=sdk,
            python_version=python_version,
            use_uv=use_uv,
        )    

    # ------------------------------------------------------------------
    # Resolution entry points
    # ------------------------------------------------------------------

    def resolve_pyproject(self, path: Path) -> ResolutionResult:
        result = self.resolver.resolve_from_pyproject(path)
        return self._enrich(result)

    def resolve_requirements(self, path: Path) -> ResolutionResult:
        result = self.resolver.resolve_from_requirements(path)
        return self._enrich(result)

    def resolve_package(self, spec: str) -> ResolutionResult:
        result = self.resolver.resolve_package(spec)
        return self._enrich(result)

    def resolve_lockfile(self, path: Path) -> ResolutionResult:
        result = self.resolver.resolve_from_lockfile(path)
        return self._enrich(result)

    # ------------------------------------------------------------------
    # Combined run helpers
    # ------------------------------------------------------------------

    def run_from_pyproject(
        self,
        path: Path,
        output_format: str = "yaml",
        output_path: Optional[Path] = None,
    ) -> str:
        result = self.resolve_pyproject(path)
        return self._generate(result, output_format, output_path)

    def run_from_requirements(
        self,
        path: Path,
        output_format: str = "yaml",
        output_path: Optional[Path] = None,
    ) -> str:
        result = self.resolve_requirements(path)
        return self._generate(result, output_format, output_path)

    def run_from_package(
        self,
        spec: str,
        output_format: str = "yaml",
        output_path: Optional[Path] = None,
    ) -> str:
        result = self.resolve_package(spec)
        return self._generate(result, output_format, output_path)

    # ------------------------------------------------------------------
    # Internal: enrichment pipeline
    # ------------------------------------------------------------------

    def _enrich(self, result: ResolutionResult) -> ResolutionResult:
        """
        Full enrichment pipeline:
          1. NativeDependencyAnalyzer  — fill pkg.native_deps
          2. Attach NativeRecipes       — match deps to local recipe DB
          3. SDKCapabilityResolver      — filter recipes already in SDK
          4. SDKExtensionResolver       — determine required sdk-extensions
        """
        result = self._analyze_native_deps(result)
        result = self._attach_recipes(result)
        result = self._filter_sdk_satisfied(result)
        result = self._resolve_extensions(result)
        return result

    def _analyze_native_deps(self, result: ResolutionResult) -> ResolutionResult:
        """Run NativeDependencyAnalyzer over all packages."""
        self.native_analyzer.analyze(result.packages)
        return result

    def _resolve_extensions(self, result: ResolutionResult) -> ResolutionResult:
        """Run SDKExtensionResolver and populate result.required_extensions."""
        report = self.ext_resolver.resolve(
            result.packages,
            sdk_id=self.sdk_id,
            sdk_version=self.sdk_version,
        )
        result.required_extensions = report.extension_ids
        result.extension_matches = report.required_extensions
        if report.extension_ids:
            logger.info(
                "Required SDK extensions: %s", report.extension_ids
            )
        return result

    def _attach_recipes(self, result: ResolutionResult) -> ResolutionResult:
        """Attach native recipes for packages that need them."""
        seen: set[str] = set()
        for pkg in result.packages:
            if not pkg.requires_native:
                continue
            for hint in pkg.native_deps or [pkg.name]:
                recipe = self.recipe_db.find(hint)
                if recipe and recipe.id not in seen:
                    result.native_recipes.append(recipe)
                    seen.add(recipe.id)
                    logger.debug(
                        "Attached recipe '%s' for package '%s'",
                        recipe.id, pkg.name,
                    )
        if result.native_recipes:
            logger.info(
                "Attached %d recipes: %s",
                len(result.native_recipes),
                [r.id for r in result.native_recipes],
            )
        return result

    def _filter_sdk_satisfied(self, result: ResolutionResult) -> ResolutionResult:
        """
        Check all native pkg-config / library requirements against the SDK.
        - Recipes whose pkgconfig names are already in the SDK are removed
          (no need to build that lib — it's already there).
        - Packages whose native deps are fully covered by the SDK have their
          requires_native flag cleared (they become pure installs).
        - Remaining unresolved natives are recorded in result.unresolved_natives.
        """
        if not self.sdk_resolver.all_capabilities():
            logger.debug("No SDK capabilities loaded; skipping SDK filter")
            return result

        # --- filter out recipes already provided by the SDK ---
        filtered_recipes = []
        for recipe in result.native_recipes:
            # A recipe is redundant if ALL its pkgconfig names are in the SDK
            if recipe.pkgconfig:
                queries = [SDKQuery(value=pc, query_type="pkgconfig") for pc in recipe.pkgconfig]
                report = self.sdk_resolver.resolve(queries)
                if report.is_sufficient:
                    logger.info(
                        "Recipe '%s' skipped — all pkg-config deps satisfied by SDK (%s)",
                        recipe.id,
                        report.satisfied[0].provided_by if report.satisfied else "?",
                    )
                    continue
            filtered_recipes.append(recipe)
        result.native_recipes = filtered_recipes

        # --- check per-package native deps ---
        unresolved: list[str] = []
        for pkg in result.packages:
            if not pkg.requires_native:
                continue
            # Collect all hints for this package
            hints = pkg.native_deps or []
            if not hints:
                # No explicit deps recorded yet — can't SDK-check, keep as-is
                continue
            queries = [SDKQuery(value=h, query_type="pkgconfig") for h in hints]
            report = self.sdk_resolver.resolve(queries)
            if report.is_sufficient:
                # All native deps covered — this package can be pip-installed normally
                pkg.requires_native = False
                logger.info(
                    "Package '%s' native deps fully covered by SDK; treating as pure",
                    pkg.name,
                )
            else:
                for miss in report.missing:
                    if miss.query not in unresolved:
                        unresolved.append(miss.query)

        result.unresolved_natives = unresolved
        if unresolved:
            logger.warning(
                "%d native deps not covered by SDK and have no recipe: %s",
                len(unresolved), unresolved,
            )
        return result

    def probe(
        self,
        packages_or_result,
        work_dir: Optional[Path] = None,
        sdk_extensions: Optional[list[str]] = None,
    ) -> "SandboxProbeReport":
        """
        Run the build sandbox probe for a list of packages or a ResolutionResult.
        Uses the runtime/SDK already configured on this Pipeline instance.
        """
        from pfmr.models import ResolvedPackage

        if isinstance(packages_or_result, ResolutionResult):
            packages = packages_or_result.packages
            exts = sdk_extensions or packages_or_result.required_extensions
        else:
            packages = packages_or_result
            exts = sdk_extensions or []

        prober = BuildSandboxProber(
            runtime=self.generator.runtime,
            runtime_version=self.generator.runtime_version,
            sdk=self.generator.sdk,
            sdk_extensions=exts,
        )
        return prober.probe(packages, work_dir=work_dir)

    def _generate(
        self,
        result: ResolutionResult,
        output_format: str,
        output_path: Optional[Path]) -> str:
        manifest = self.generator.generate(result)
        if output_format == "json":
            return self.generator.to_json(manifest, output_path)
        return self.generator.to_yaml(manifest, output_path)