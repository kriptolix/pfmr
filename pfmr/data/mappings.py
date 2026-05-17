"""
pfmr.data.mappings
~~~~~~~~~~~~~~~~~~~
Loads pfmr/data/mappings.toml and exposes its tables as typed, cached
module-level objects. All subsystems that need name mappings import from
here instead of defining their own inline dicts.

Usage::

    from pfmr.data.mappings import MAPPINGS

    sonames = MAPPINGS.pkgconfig_to_soname("libusb-1.0")
    pcnames = MAPPINGS.module_to_pkgconfig("libusb")
    import_name = MAPPINGS.python_import_name("pillow")
    is_baseline = MAPPINGS.is_baseline_lib("libc.so.6")
    is_app = MAPPINGS.is_app_module(mod_name, is_last=True, source_types=["dir"])
"""
from __future__ import annotations

import tomllib
from functools import cached_property
from pathlib import Path
from typing import Optional

_MAPPINGS_FILE = Path(__file__).parent / "mappings.toml"


class _Mappings:
    """
    Lazy-loaded, cached wrapper around mappings.toml.
    The singleton is instantiated once at module import and shared everywhere.
    """

    def __init__(self, path: Path = _MAPPINGS_FILE):
        self._path = path
        self._data: Optional[dict] = None

    def _load(self) -> dict:
        if self._data is None:
            try:
                with open(self._path, "rb") as f:
                    self._data = tomllib.load(f)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).error(
                    "Failed to load mappings.toml: %s", exc
                )
                self._data = {}
        return self._data

    # ------------------------------------------------------------------
    # module name → pkgconfig
    # ------------------------------------------------------------------

    def module_to_pkgconfig(self, module_name: str) -> list[str]:
        """Return pkgconfig name(s) for a Flatpak module name, or [module_name]."""
        table = self._load().get("module_to_pkgconfig", {})
        return table.get(module_name.lower(), [module_name.lower()])

    # ------------------------------------------------------------------
    # module name → sonames
    # ------------------------------------------------------------------

    def module_to_soname(self, module_name: str) -> list[str]:
        """Return soname(s) for a Flatpak module name, or []."""
        table = self._load().get("module_to_soname", {})
        return table.get(module_name.lower(), [])

    # ------------------------------------------------------------------
    # pkgconfig → sonames
    # ------------------------------------------------------------------

    def pkgconfig_to_soname(self, pc_name: str) -> list[str]:
        """Return soname(s) for a pkg-config name, or []."""
        table = self._load().get("pkgconfig_to_soname", {})
        return table.get(pc_name, [])

    # ------------------------------------------------------------------
    # Python import names
    # ------------------------------------------------------------------

    def python_import_name(self, pypi_name: str) -> str:
        """
        Return the Python import name for a PyPI package.
        Falls back to pypi_name with hyphens replaced by underscores.
        """
        table = self._load().get("python_import_names", {})
        normalised = pypi_name.lower().replace("-", "_")
        return table.get(normalised, table.get(pypi_name.lower(), normalised))

    # ------------------------------------------------------------------
    # Runtime baseline libs
    # ------------------------------------------------------------------

    @cached_property
    def _baseline_libs(self) -> frozenset[str]:
        data = self._load().get("runtime_baseline_libs", {})
        return frozenset(data.get("libs", []))

    @cached_property
    def _baseline_prefixes(self) -> tuple[str, ...]:
        data = self._load().get("runtime_baseline_libs", {})
        return tuple(data.get("baseline_prefixes", []))

    def is_baseline_lib(self, soname: str) -> bool:
        """Return True if soname is part of the glibc/runtime baseline."""
        if soname in self._baseline_libs:
            return True
        return any(soname.startswith(p) for p in self._baseline_prefixes)

    def filter_baseline(self, sonames: list[str]) -> list[str]:
        """Remove baseline libs from a list of sonames."""
        return [s for s in sonames if not self.is_baseline_lib(s)]

    # ------------------------------------------------------------------
    # App module detection
    # ------------------------------------------------------------------

    @cached_property
    def _app_source_types(self) -> frozenset[str]:
        data = self._load().get("app_module_indicators", {})
        return frozenset(data.get("app_source_types", ["dir"]))

    @cached_property
    def _always_app_names(self) -> frozenset[str]:
        data = self._load().get("app_module_indicators", {})
        return frozenset(n.lower() for n in data.get("always_app_names", []))

    @cached_property
    def _never_app_names(self) -> frozenset[str]:
        data = self._load().get("app_module_indicators", {})
        return frozenset(n.lower() for n in data.get("never_app_names", []))

    def is_app_module(
        self,
        name: str,
        is_last: bool,
        source_types: Optional[list[str]] = None,
    ) -> bool:
        """
        Return True if this module is the application itself (not a dep).

        A module is identified as the app when:
          - Its name is in always_app_names, OR
          - It is the last module AND has a "dir" source type (local src tree)

        A module is never the app if its name is in never_app_names.
        """
        low = name.lower()
        if low in self._never_app_names:
            return False
        if low in self._always_app_names:
            return True
        if is_last and source_types:
            return bool(self._app_source_types & set(source_types))
        return False

    # ------------------------------------------------------------------
    # Skip module names
    # ------------------------------------------------------------------

    @cached_property
    def _skip_names(self) -> frozenset[str]:
        data = self._load().get("skip_module_names", {})
        return frozenset(n.lower() for n in data.get("names", []))

    def should_skip_module(self, name: str) -> bool:
        """Return True if this module should always be ignored."""
        return name.lower() in self._skip_names


# Module-level singleton — import this everywhere
MAPPINGS = _Mappings()