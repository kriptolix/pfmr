"""
pfmr.recipes.db
~~~~~~~~~~~~~~~
Recipe database — loads and indexes all recipe YAML files.

Two recipe types are supported:

  Native recipe (recipes/native/*.yaml):
    id: libusb
    provides: [libusb-1.0.so.0]
    pkgconfig: [libusb-1.0]
    buildsystem: autotools
    source: {type: archive, url: ..., sha256: ...}
    config-opts: [--disable-static]
    cleanup: [/include, /lib/pkgconfig]

  Python recipe (recipes/python/*.yaml):
    id: cryptography
    type: python
    pypi_name: cryptography
    requires:
      pkgconfig: [openssl, libffi]
      extensions: [org.freedesktop.Sdk.Extension.rust-stable]
    confidence: 1.0
    source: sandbox:org.freedesktop.Sdk/24.08

The DB provides separate lookup methods for each type and silently skips
files that don't match either format.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import yaml

from pfmr.models import FlatpakSource, NativeRecipe, PythonRecipe
from pfmr.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_RECIPE_DIRS = [
    Path(__file__).parent.parent.parent / "recipes" / "native",
    Path(__file__).parent.parent.parent / "recipes" / "python",
    Path(__file__).parent.parent.parent / "recipes" / "sdk",
    # Note: extensions are data (data/extension-profiles/), not recipes
]


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_native_recipe(data: dict, path: Path) -> NativeRecipe:
    source_data = data.get("source")
    source = None
    if isinstance(source_data, dict):
        source = FlatpakSource(
            type=source_data.get("type", "archive"),
            url=source_data.get("url"),
            sha256=source_data.get("sha256"),
            path=source_data.get("path"),
            dest_filename=source_data.get("dest-filename"),
            branch=source_data.get("branch"),
            commit=source_data.get("commit"),
            tag=source_data.get("tag"),
        )
    return NativeRecipe(
        id=data["id"],
        provides=data.get("provides", []),
        pkgconfig=data.get("pkgconfig", []),
        headers=data.get("headers", []),
        buildsystem=data.get("buildsystem", "autotools"),
        source=source,
        build_commands=data.get("build-commands", []),
        config_opts=data.get("config-opts", []),
        cleanup=data.get("cleanup", ["/include", "/lib/pkgconfig"]),
        aliases=data.get("aliases", []),
    )


def _parse_python_recipe(data: dict, path: Path) -> PythonRecipe:
    requires = data.get("requires", {})
    # requires may be a dict or absent
    if not isinstance(requires, dict):
        requires = {}
    return PythonRecipe(
        id=data["id"],
        pypi_name=data.get("pypi_name", data["id"]),
        requires_pkgconfig=requires.get("pkgconfig", []),
        requires_libraries=requires.get("libraries", []),
        requires_extensions=requires.get("extensions", []),
        sdk_sufficient=bool(data.get("sdk_sufficient", False)),
        confidence=float(data.get("confidence", 0.0)),
        source=data.get("source", ""),
    )


def _parse_recipe_file(path: Path) -> tuple[Optional[NativeRecipe], Optional[PythonRecipe]]:
    """
    Parse a recipe YAML and return (native, python) — exactly one will be set.
    Returns (None, None) on parse failure.
    """
    try:
        data = yaml.safe_load(path.read_text())
    except Exception as exc:
        logger.warning("Failed to read recipe %s: %s", path, exc)
        return None, None

    if not isinstance(data, dict) or "id" not in data:
        logger.debug("Skipping non-recipe file: %s", path)
        return None, None

    recipe_type = data.get("type", "native")

    if recipe_type == "python":
        try:
            return None, _parse_python_recipe(data, path)
        except Exception as exc:
            logger.warning("Failed to parse python recipe %s: %s", path, exc)
            return None, None
    else:
        try:
            return _parse_native_recipe(data, path), None
        except Exception as exc:
            logger.warning("Failed to parse native recipe %s: %s", path, exc)
            return None, None


# ---------------------------------------------------------------------------
# RecipeDB
# ---------------------------------------------------------------------------

class RecipeDB:
    """
    Immutable snapshot of all locally available recipes.
    """

    def __init__(self, recipe_dirs: Optional[list[Path]] = None):
        dirs = recipe_dirs or _DEFAULT_RECIPE_DIRS
        self._native: dict[str, NativeRecipe] = {}
        self._python: dict[str, PythonRecipe] = {}
        # indexes for native recipes
        self._soname_index: dict[str, str] = {}
        self._pkgconfig_index: dict[str, str] = {}
        self._header_index: dict[str, str] = {}
        self._alias_index: dict[str, str] = {}
        self._load(dirs)

    # ------------------------------------------------------------------
    # Native recipe lookup
    # ------------------------------------------------------------------

    def find_by_soname(self, soname: str) -> Optional[NativeRecipe]:
        rid = self._soname_index.get(soname) or self._soname_index.get(
            re.sub(r"\.so\..+$", ".so", soname)
        )
        return self._native.get(rid) if rid else None

    def find_by_pkgconfig(self, pc_name: str) -> Optional[NativeRecipe]:
        pc = pc_name.removesuffix(".pc")
        rid = self._pkgconfig_index.get(pc)
        return self._native.get(rid) if rid else None

    def find_by_id(self, recipe_id: str) -> Optional[NativeRecipe]:
        return self._native.get(recipe_id)

    def find_by_alias(self, alias: str) -> Optional[NativeRecipe]:
        rid = self._alias_index.get(alias.lower())
        return self._native.get(rid) if rid else None

    def find(self, hint: str) -> Optional[NativeRecipe]:
        return (
            self.find_by_soname(hint)
            or self.find_by_pkgconfig(hint)
            or self.find_by_id(hint)
            or self.find_by_alias(hint)
        )

    def all_recipes(self) -> list[NativeRecipe]:
        return list(self._native.values())

    # ------------------------------------------------------------------
    # Python recipe lookup
    # ------------------------------------------------------------------

    def find_python(self, package_name: str) -> Optional[PythonRecipe]:
        """Look up a Python package recipe by canonical name."""
        from packaging.utils import canonicalize_name
        return self._python.get(canonicalize_name(package_name))

    def all_python_recipes(self) -> list[PythonRecipe]:
        return list(self._python.values())

    # ------------------------------------------------------------------
    # General
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._native) + len(self._python)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, dirs: list[Path]) -> None:
        native_count = python_count = 0
        for d in dirs:
            if not d.exists():
                logger.debug("Recipe dir not found (skipping): %s", d)
                continue
            for yml in sorted(d.glob("**/*.yaml")) + sorted(d.glob("**/*.yml")):
                native, python = _parse_recipe_file(yml)
                if native:
                    self._register_native(native)
                    native_count += 1
                elif python:
                    self._register_python(python)
                    python_count += 1
        logger.info(
            "Loaded %d native + %d python recipes from %d directories",
            native_count, python_count, len(dirs),
        )

    def _register_native(self, recipe: NativeRecipe) -> None:
        self._native[recipe.id] = recipe
        for soname in recipe.provides:
            self._soname_index[soname] = recipe.id
            base = re.sub(r"\.so\..+$", ".so", soname)
            self._soname_index.setdefault(base, recipe.id)
        for pc in recipe.pkgconfig:
            self._pkgconfig_index[pc.removesuffix(".pc")] = recipe.id
        for h in recipe.headers:
            self._header_index[h] = recipe.id
        for alias in recipe.aliases:
            self._alias_index[alias.lower()] = recipe.id
        self._alias_index[recipe.id.lower()] = recipe.id

    def _register_python(self, recipe: PythonRecipe) -> None:
        from packaging.utils import canonicalize_name
        self._python[canonicalize_name(recipe.id)] = recipe