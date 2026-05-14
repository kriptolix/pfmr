"""
pfmr.recipes.db
~~~~~~~~~~~~~~~
Local recipe database — Phase 1 component.

Loads YAML recipe files from the recipes/ directory tree and exposes a
simple matching API: given a library name / soname / pkg-config name,
return the matching NativeRecipe if one exists.

Recipe format (YAML):
  id: libusb
  provides:
    - libusb-1.0.so
    - libusb-1.0.so.0
  pkgconfig:
    - libusb-1.0
  headers:
    - libusb.h
  aliases:
    - usb
  buildsystem: autotools    # autotools | cmake | meson | simple
  source:
    type: archive
    url: https://...
    sha256: ...
  config_opts:
    - --disable-static
  cleanup:
    - /include
    - /lib/pkgconfig
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import yaml

from pfmr.models import FlatpakSource, NativeRecipe
from pfmr.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_RECIPE_DIRS = [
    Path(__file__).parent.parent.parent / "recipes" / "native",
    Path(__file__).parent.parent.parent / "recipes" / "python",
    Path(__file__).parent.parent.parent / "recipes" / "sdk",
    Path(__file__).parent.parent.parent / "recipes" / "extensions",
]


def _parse_recipe(path: Path) -> NativeRecipe:
    data = yaml.safe_load(path.read_text())
    source_data = data.get("source")
    source = None
    if source_data:
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


class RecipeDB:
    """
    Immutable snapshot of all locally available recipes, loaded at construction
    time from one or more recipe directories.
    """

    def __init__(self, recipe_dirs: Optional[list[Path]] = None):
        dirs = recipe_dirs or _DEFAULT_RECIPE_DIRS
        self._recipes: dict[str, NativeRecipe] = {}
        # index: soname → recipe_id, pkgconfig → recipe_id, header → recipe_id
        self._soname_index: dict[str, str] = {}
        self._pkgconfig_index: dict[str, str] = {}
        self._header_index: dict[str, str] = {}
        self._alias_index: dict[str, str] = {}
        self._load(dirs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def find_by_soname(self, soname: str) -> Optional[NativeRecipe]:
        """Look up a recipe by shared library name (e.g. 'libusb-1.0.so.0')."""
        rid = self._soname_index.get(soname) or self._soname_index.get(
            re.sub(r"\.so\..+$", ".so", soname)
        )
        return self._recipes.get(rid) if rid else None

    def find_by_pkgconfig(self, pc_name: str) -> Optional[NativeRecipe]:
        """Look up a recipe by pkg-config name (without .pc suffix)."""
        pc = pc_name.removesuffix(".pc")
        rid = self._pkgconfig_index.get(pc)
        return self._recipes.get(rid) if rid else None

    def find_by_id(self, recipe_id: str) -> Optional[NativeRecipe]:
        return self._recipes.get(recipe_id)

    def find_by_alias(self, alias: str) -> Optional[NativeRecipe]:
        rid = self._alias_index.get(alias.lower())
        return self._recipes.get(rid) if rid else None

    def find(self, hint: str) -> Optional[NativeRecipe]:
        """
        Universal lookup: tries soname, pkg-config name, recipe ID, and aliases.
        """
        return (
            self.find_by_soname(hint)
            or self.find_by_pkgconfig(hint)
            or self.find_by_id(hint)
            or self.find_by_alias(hint)
        )

    def all_recipes(self) -> list[NativeRecipe]:
        return list(self._recipes.values())

    def __len__(self) -> int:
        return len(self._recipes)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, dirs: list[Path]) -> None:
        count = 0
        for d in dirs:
            if not d.exists():
                logger.debug("Recipe dir not found (skipping): %s", d)
                continue
            for yml in sorted(d.glob("**/*.yaml")) + sorted(d.glob("**/*.yml")):
                try:
                    recipe = _parse_recipe(yml)
                    self._register(recipe)
                    count += 1
                except Exception as exc:
                    logger.warning("Failed to parse recipe %s: %s", yml, exc)
        logger.info("Loaded %d recipes from %d directories", count, len(dirs))

    def _register(self, recipe: NativeRecipe) -> None:
        self._recipes[recipe.id] = recipe
        for soname in recipe.provides:
            self._soname_index[soname] = recipe.id
            # also index without version suffix
            base = re.sub(r"\.so\..+$", ".so", soname)
            self._soname_index.setdefault(base, recipe.id)
        for pc in recipe.pkgconfig:
            self._pkgconfig_index[pc.removesuffix(".pc")] = recipe.id
        for h in recipe.headers:
            self._header_index[h] = recipe.id
        for alias in recipe.aliases:
            self._alias_index[alias.lower()] = recipe.id
        # always index by id as alias too
        self._alias_index[recipe.id.lower()] = recipe.id
